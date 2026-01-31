"""
M-Step Function for GPyTorch Variational GP

Handles optimization of kernel hyperparameters while holding variational
parameters (m, V) and firing rate parameters (A, λ₀) fixed.

Key functions:
- m_step(): Adam-based optimization of kernel hyperparameters (for GPyTorch mode)
- mstep_eigenspace_autograd(): LBFGS with PyTorch autograd (eigenspace mode)
- mstep_eigenspace_analytical(): LBFGS with analytical gradients (eigenspace mode)

DEPRECATED FUNCTIONS (deleted):
- m_step_lbfgs(): Did not support hyperparameter clamping
- m_step_lbfgs_grouped(): Did not support hyperparameter clamping
"""

import warnings
import torch
import gpytorch

# Eigenspace mode imports
from estep import STABILITY_THRESHOLD
# Gradient functions from direct_vargp (TEMPORARY - deferred reorganization)
from direct_vargp import (
    compute_C_and_gradients,
    compute_kernel_and_gradients,
    compute_lambda_moments_and_gradients,
    compute_loss_gradients,
)


def m_step(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    n_mstep: int,
    lr: float,  # Required - no default to prevent silent bugs
    verbose: bool = False
):
    """M-step: Optimize kernel hyperparameters with Adam.

    Structural change from baseline: Only kernel hyperparameters are optimized here,
    not A or lambda0 (those are handled in F-step).

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        r: Training spike counts, shape (N,)
        n_mstep: Number of Adam iterations
        lr: Learning rate for Adam
        verbose: Print debug info
    """
    if n_mstep == 0:
        return

    optimizer = torch.optim.Adam(model.covar_module.parameters(), lr=lr)

    for _ in range(n_mstep):
        optimizer.zero_grad()
        output = model(X)
        loss = -likelihood.expected_log_prob(r, output) + \
               model.variational_strategy.kl_divergence()
        loss.backward()
        optimizer.step()

        # Clamp hyperparameters to valid bounds (projected gradient descent)
        # Since kernel is now ArcCosineKernel directly (not ScaleKernel wrapper),
        # clamp_hyperparameters is called on model.covar_module directly
        kernel = model.covar_module
        if hasattr(kernel, 'clamp_hyperparameters'):
            kernel.clamp_hyperparameters()


# =============================================================================
# Eigenspace M-Step Functions
# =============================================================================

def mstep_eigenspace_autograd(model, r: torch.Tensor, n_mstep: int, lr: float):
    """M-step for eigenspace mode: Optimize kernel hyperparameters with LBFGS using autograd.

    Uses LBFGS with PyTorch autograd for gradients (not analytical gradients).
    Kernel matrices are recomputed with gradients enabled during closure.

    Note: This does NOT use eigenspace projection during M-step optimization.
    The eigenspace is fixed during M-step; reprojection happens after.

    Args:
        model: DirectVGPModel instance
        r: Spike counts, shape (N,)
        n_mstep: Number of LBFGS iterations
        lr: Learning rate for LBFGS
    """
    kernel = model.kernel
    likelihood = model.likelihood
    X = model.X_train
    X_tilde = model.X_tilde
    state = model.state
    if n_mstep == 0:
        return

    # Get kernel parameters
    kernel_params = list(kernel.parameters())

    optimizer = torch.optim.LBFGS(
        kernel_params,
        lr=lr,
        max_iter=n_mstep,
        tolerance_change=1e-9,
        tolerance_grad=1e-7,
        history_size=100,
        line_search_fn='strong_wolfe'
    )

    def closure():
        optimizer.zero_grad()

        # Compute kernels WITH gradients
        K_tilde = kernel(X_tilde, X_tilde).evaluate()
        K = kernel(X, X_tilde).evaluate()
        Kvec = kernel(X, diag=True)

        # Project into FIXED eigenspace (use state.B)
        # This is an approximation - we don't recompute eigenspace during M-step
        K_tilde_b = state.B.T @ K_tilde @ state.B  # (n_b, n_b)
        K_b = K @ state.B  # (N, n_b)

        # Check if K_tilde_b is still well-conditioned
        try:
            K_tilde_b_inv = torch.linalg.solve(K_tilde_b, torch.eye(K_tilde_b.shape[0],
                                               device=K_tilde_b.device, dtype=K_tilde_b.dtype))
        except RuntimeError:
            # Singular matrix - return inf to reject this step
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        # Compute moments with fixed m_b, V_b
        a = K_b @ K_tilde_b_inv  # (N, n_b)
        lambda_m = a @ state.m_b  # (N,)

        V_minus_K = state.V_b - K_tilde_b
        aV = a @ V_minus_K
        lambda_var = Kvec + (a * aV).sum(dim=1)
        lambda_var = torch.clamp(lambda_var, min=1e-6)

        # Compute log-likelihood
        A = likelihood.A.squeeze()
        lambda0 = likelihood.lambda0.squeeze()
        f_mean = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + lambda0)

        if f_mean.mean().item() > STABILITY_THRESHOLD or torch.any(torch.isnan(f_mean)):
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        log_lik = (r * (A * lambda_m + lambda0) - f_mean).sum()

        # Compute KL divergence
        n_b = len(state.eigvals_b)

        # KL trace term: tr(K_tilde_inv @ V) using full matrix
        # K_tilde_b_inv was already computed via solve() at line 139-143
        # This is correct even when K_tilde_b is non-diagonal (after hyperparam changes)
        trace_term = torch.trace(K_tilde_b_inv @ state.V_b)
        quad_term = (state.m_b @ K_tilde_b_inv @ state.m_b)

        sign_K, log_det_K = torch.linalg.slogdet(K_tilde_b)
        sign_V, log_det_V = torch.linalg.slogdet(state.V_b)

        if sign_K.item() <= 0 or sign_V.item() <= 0:
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        KL = 0.5 * (trace_term + quad_term - n_b + log_det_K - log_det_V)

        # Negative ELBO (for minimization)
        loss = -log_lik + KL

        # Use autograd.grad to compute gradients (works with LBFGS line search)
        if loss.requires_grad:
            grads = torch.autograd.grad(loss, kernel_params, create_graph=False)
            for param, grad in zip(kernel_params, grads):
                param.grad = grad

        return loss

    optimizer.step(closure)
    kernel.clamp_hyperparameters()


def mstep_eigenspace_analytical(model, r: torch.Tensor, n_mstep: int, lr: float):
    """M-step for eigenspace mode with analytical gradients (matching vargp_old).

    Computes dK/dtheta matrices ONCE and caches them for LBFGS closure.
    Uses @torch.no_grad() closure for speed (no autograd graph construction).

    Implements all 8 numerical guardrails from original varGP.

    Args:
        model: DirectVGPModel instance
        r: Spike counts, shape (N,)
        n_mstep: Number of LBFGS iterations
        lr: Learning rate for LBFGS

    Reference:
        utils.py M-step closure, lines 5859-5963
    """
    kernel = model.kernel
    likelihood = model.likelihood
    X = model.X_train
    X_tilde = model.X_tilde
    state = model.state
    if n_mstep == 0:
        return

    # Fixed variational params and eigenspace
    B = state.B
    m_b = state.m_b
    V_b = state.V_b
    eigvals_b = state.eigvals_b
    n_b = len(eigvals_b)

    # Get kernel parameter references
    kernel_params = list(kernel.parameters())

    optimizer = torch.optim.LBFGS(
        kernel_params,
        lr=lr,
        max_iter=n_mstep,
        tolerance_change=1e-9,
        tolerance_grad=1e-7,
        history_size=100,
        line_search_fn='strong_wolfe'
    )

    # Likelihood params (fixed during M-step)
    A = likelihood.A.squeeze().detach()
    lambda0 = likelihood.lambda0.squeeze().detach()

    @torch.no_grad()
    def closure():
        optimizer.zero_grad()

        # ===== Guardrail 1: Check hyperparameter bounds =====
        # Return inf to reject LBFGS step if out of bounds
        try:
            sigma_0_val = kernel.sigma_0.item()
            Amp_val = kernel.Amp.item()
            if sigma_0_val <= 0 or Amp_val <= 0 or Amp_val > kernel.AMP_MAX:
                return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

            if hasattr(kernel, 'raw_m2log2beta'):
                raw_beta = kernel.raw_m2log2beta.item()
                raw_rho = kernel.raw_mlog2rho2.item()
                if raw_beta < kernel.RAW_BETA_MIN or raw_beta > kernel.RAW_BETA_MAX:
                    return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)
                if raw_rho < kernel.RAW_RHO_MIN or raw_rho > kernel.RAW_RHO_MAX:
                    return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

            if hasattr(kernel, 'eps_0x'):
                eps_x = kernel.eps_0x.item()
                eps_y = kernel.eps_0y.item()
                if eps_x < kernel.EPS_MIN or eps_x > kernel.EPS_MAX:
                    return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)
                if eps_y < kernel.EPS_MIN or eps_y > kernel.EPS_MAX:
                    return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)
        except Exception:
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        # ===== 1. Compute C and dC (includes masking) =====
        C, mask, dC = compute_C_and_gradients(kernel)
        sigma_0 = kernel.sigma_0.squeeze()

        # Apply mask to inputs (mask from compute_C_and_gradients)
        X_masked = X[:, mask]
        X_tilde_masked = X_tilde[:, mask]

        # ===== 2. Compute K, K_tilde, Kvec and all gradients =====
        K_tilde, dK_tilde = compute_kernel_and_gradients(
            X_tilde_masked, X_tilde_masked, C, dC, sigma_0, diag=False
        )
        K, dK = compute_kernel_and_gradients(
            X_masked, X_tilde_masked, C, dC, sigma_0, diag=False
        )
        Kvec, dKvec = compute_kernel_and_gradients(
            X_masked, None, C, dC, sigma_0, diag=True
        )

        # ===== Guardrail 5: Symmetrize K_tilde =====
        K_tilde = (K_tilde + K_tilde.T) / 2

        # ===== 3. Project to eigenspace =====
        K_tilde_b = B.T @ K_tilde @ B
        K_b = K @ B

        # Symmetrize K_tilde_b
        K_tilde_b = (K_tilde_b + K_tilde_b.T) / 2

        # CRITICAL FIX: Compute K_tilde_inv_b using solve(), matching vargp_old
        # The original code (utils.py lines 5920-5921) uses:
        #   K_tilde_inv_b = torch.linalg.solve(K_tilde_b, eye)
        # NOT eigendecomposition. This is more numerically stable.
        eye_b = torch.eye(n_b, device=K_tilde_b.device, dtype=K_tilde_b.dtype)
        K_tilde_inv_b = torch.linalg.solve(K_tilde_b, eye_b)

        # Project gradient matrices
        dK_tilde_b = {k: B.T @ v @ B for k, v in dK_tilde.items()}
        dK_b = {k: v @ B for k, v in dK.items()}

        # ===== 4. Compute moments and gradients =====
        lambda_m, lambda_var, dlambda_m, dlambda_var = compute_lambda_moments_and_gradients(
            K_b, K_tilde_b, Kvec, m_b, V_b,
            dK_b, dK_tilde_b, dKvec, K_tilde_inv_b  # Use full matrix inverse
        )

        # Compute f_mean
        f_mean = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + lambda0)

        # ===== Guardrail 2: Firing rate check =====
        if f_mean.mean().item() > 100 or torch.any(torch.isnan(f_mean)):
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        # ===== 5. Compute loss =====
        # Log-likelihood
        log_lik = (r * (A * lambda_m + lambda0) - f_mean).sum()

        # KL divergence using full matrix K_tilde_inv_b (matching vargp_old)
        # KL = 0.5 * (tr(K_tilde_inv @ V) + m.T @ K_tilde_inv @ m - n_b + log|K_tilde| - log|V|)
        trace_term = torch.trace(K_tilde_inv_b @ V_b)
        quad_term = m_b @ K_tilde_inv_b @ m_b

        # log|K_tilde_b| via Cholesky (matches vargp_old's log_det function)
        try:
            L_K = torch.linalg.cholesky(K_tilde_b)
            log_det_K = 2 * torch.log(torch.diag(L_K)).sum()
        except RuntimeError:
            # Fallback if Cholesky fails - return inf to reject this step
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        # log|V_b|
        sign_V, log_det_V = torch.linalg.slogdet(V_b)
        if sign_V.item() <= 0:
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        KL = 0.5 * (trace_term + quad_term - n_b + log_det_K - log_det_V)

        loss = -log_lik + KL

        # ===== Guardrail 7: Loss NaN/Inf detection =====
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        # ===== 6. Compute analytical gradients =====
        dL = compute_loss_gradients(
            r, f_mean, A, m_b, V_b, K_tilde_b, K_tilde_inv_b,  # Use full matrix inverse
            dlambda_m, dlambda_var, dK_tilde_b
        )

        # ===== Guardrail 8: Gradient NaN/Inf tracking =====
        nan_grad_count = 0
        for key, grad_val in dL.items():
            if torch.isnan(grad_val) or torch.isinf(grad_val):
                nan_grad_count += 1
        if nan_grad_count > 0:
            warnings.warn(f"M-step: {nan_grad_count} gradients contain NaN/Inf")

        # ===== 7. Set parameter gradients =====
        # Apply softplus transform correction for sigma_0 and Amp
        # softplus'(x) = sigmoid(x)
        kernel.raw_sigma_0.grad = dL['sigma_0'] * torch.sigmoid(kernel.raw_sigma_0)
        kernel.raw_Amp.grad = dL['Amp'] * torch.sigmoid(kernel.raw_Amp)

        # Direct gradients for other parameters
        kernel.eps_0x.grad = dL['eps_0x'].unsqueeze(0) if dL['eps_0x'].dim() == 0 else dL['eps_0x']
        kernel.eps_0y.grad = dL['eps_0y'].unsqueeze(0) if dL['eps_0y'].dim() == 0 else dL['eps_0y']
        kernel.raw_m2log2beta.grad = dL['raw_m2log2beta'].unsqueeze(0) if dL['raw_m2log2beta'].dim() == 0 else dL['raw_m2log2beta']
        kernel.raw_mlog2rho2.grad = dL['raw_mlog2rho2'].unsqueeze(0) if dL['raw_mlog2rho2'].dim() == 0 else dL['raw_mlog2rho2']

        return loss

    optimizer.step(closure)
    kernel.clamp_hyperparameters()
