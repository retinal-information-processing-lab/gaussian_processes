"""
Custom E-Step Implementation for GPyTorch Variational GP

Implements closed-form Newton update for variational parameters (m, V).

CORRECT Formulas (from .claude/ESTEP_MATH_ANALYSIS.md):
    V_new = K̃(K̃ + G)⁻¹K̃
    m_new = m + K̃(K̃ + G)⁻¹(g - m)

where:
    g = A · Kᵀ @ (r - f̄)
    G = A² · Kᵀ @ diag(f̄) @ K
    f̄ᵢ = exp(A·μᵢ + ½A²σᵢ² + λ₀)
"""

import torch
import gpytorch
from typing import Tuple, Optional


def set_kernel_requires_grad(model: gpytorch.models.ApproximateGP, requires_grad: bool):
    """Toggle requires_grad for kernel parameters.

    This is used to skip gradient computation during E-step and F-step where
    kernel gradients are not needed. Disabling requires_grad allows the
    analytical gradients optimization to skip dK computation.

    Args:
        model: VariationalGPModel instance
        requires_grad: Whether to enable gradient computation for kernel params
    """
    for name, param in model.covar_module.named_parameters():
        param.requires_grad = requires_grad


def e_step(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    jitter: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform one E-step: closed-form Newton update of variational parameters.

    Uses CORRECT formulas (from .claude/ESTEP_MATH_ANALYSIS.md):
        g = A · Kᵀ @ (r - f̄)
        G = A² · Kᵀ @ diag(f̄) @ K

        V_new = K̃ @ solve(K̃ + G, K̃)  = K̃(K̃ + G)⁻¹K̃
        m_new = m + K̃ @ solve(K̃ + G, g - m)  = m + K̃(K̃ + G)⁻¹(g - m)

    Note: The original utils.py:Estep() uses a different m formula that has a
    discrepancy. This implementation uses the mathematically correct Newton step.

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instancenc
        X: Training inputs, shape (N, n_features)
        r: Training spike counts, shape (N,)
        jitter: Small value for numerical stability

    Returns:
        m_new: Updated variational mean, shape (M,)
        V_new: Updated variational covariance, shape (M, M)
    """
    # Get current variational mean
    var_params = model.variational_strategy._variational_distribution
    m = var_params.variational_mean  # (M,)

    # Get likelihood parameters
    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()

    # Use GPyTorch's native forward pass for posterior moments
    output = model(X)
    lambda_mean = output.mean      # (N,)
    lambda_var = output.variance   # (N,)

    # Expected firing rate: f̄ = exp(A·μ + ½A²σ² + λ₀)
    f_mean = torch.exp(A * lambda_mean + 0.5 * A**2 * lambda_var + lambda0)

    # Get kernel matrices
    inducing_points = model.variational_strategy.inducing_points
    kernel = model.covar_module

    K = kernel(X, inducing_points).evaluate()           # (N, M)
    K_tilde = kernel(inducing_points).evaluate()        # (M, M)

    # Add jitter for numerical stability
    M = K_tilde.shape[0]
    eye = torch.eye(M, dtype=K_tilde.dtype, device=K_tilde.device)
    K_tilde_j = K_tilde + jitter * eye

    # Standard (non-transformed) g and G
    g = A * K.T @ (r - f_mean)                              # (M,)
    G = A**2 * K.T @ (f_mean[:, None] * K)                  # (M, M)

    # V update: V_new = K̃ @ solve(K̃ + G, K̃) = K̃(K̃ + G)⁻¹K̃
    V_new = K_tilde_j @ torch.linalg.solve(K_tilde_j + G, K_tilde_j)

    # m update: m_new = m + K̃ @ solve(K̃ + G, g - m) = m + K̃(K̃ + G)⁻¹(g - m)
    m_new = m + K_tilde_j @ torch.linalg.solve(K_tilde_j + G, g - m)

    # Symmetrize V
    V_new = (V_new + V_new.T) / 2

    return m_new, V_new


def update_variational_parameters(
    model: gpytorch.models.ApproximateGP,
    m_new: torch.Tensor,
    V_new: torch.Tensor
):
    """Write updated (m, V) back to GPyTorch model.

    GPyTorch stores:
        - variational_mean: m directly
        - chol_variational_covar: L where V = LLᵀ
    """
    # Access the parameter storage object (not the distribution)
    var_params = model.variational_strategy._variational_distribution

    # Update mean (stored directly)
    var_params.variational_mean.data.copy_(m_new)

    # Compute Cholesky factor L where V = LLᵀ
    try:
        L_new = torch.linalg.cholesky(V_new)
    except RuntimeError:
        # Add jitter if Cholesky fails
        eye = torch.eye(V_new.shape[0], dtype=V_new.dtype, device=V_new.device)
        L_new = torch.linalg.cholesky(V_new + 1e-6 * eye)

    var_params.chol_variational_covar.data.copy_(L_new)


# =============================================================================
# varGP-style training functions (matching utils.py:varGP structure)
# =============================================================================

def get_variational_mean(model: gpytorch.models.ApproximateGP) -> torch.Tensor:
    """Get the variational mean m from the model."""
    return model.variational_strategy._variational_distribution.variational_mean


def get_variational_covar(model: gpytorch.models.ApproximateGP) -> torch.Tensor:
    """Get the variational covariance V from the model (V = LLᵀ)."""
    L = model.variational_strategy._variational_distribution.chol_variational_covar
    return L @ L.T


def compute_moments(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute lambda moments and f_mean from current variational params.

    This is the GPyTorch equivalent of the old code's:
        lambda_m, lambda_var = lambda_moments(...)
        f_mean = mean_f_given_lambda_moments(...)

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X: Input points, shape (N, n_features)

    Returns:
        lambda_m: Posterior mean of λ at X, shape (N,)
        lambda_var: Posterior variance of λ at X, shape (N,)
        f_mean: Expected firing rate exp(A·μ + ½A²σ² + λ₀), shape (N,)
    """
    output = model(X)
    lambda_m = output.mean
    lambda_var = output.variance

    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()
    f_mean = torch.exp(A * lambda_m + 0.5 * A**2 * lambda_var + lambda0)

    return lambda_m, lambda_var, f_mean


def lambda0_given_A(
    A: torch.Tensor,
    r: torch.Tensor,
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor
) -> torch.Tensor:
    """Closed-form optimal lambda0 given A.

    Derived from setting dL/d(lambda0) = 0 where L is the expected log-likelihood.
    This matches utils.py:lambda0_given_logA() but takes A directly (not logA).

    The expected log-likelihood contains:
        E[r*lambda0 - exp(A*lambda + lambda0)]
      = r*lambda0 - exp(lambda0)*E[exp(A*lambda)]
      = r*lambda0 - exp(lambda0)*exp(A*lambda_m + 0.5*A^2*lambda_var)

    Setting d/d(lambda0) = 0:
        sum(r) = exp(lambda0) * sum(exp(A*lambda_m + 0.5*A^2*lambda_var))

    Solution:
        lambda0 = log(sum(r)) - log(sum(exp(A*lambda_m + 0.5*A^2*lambda_var)))

    Args:
        A: Gain parameter (scalar tensor)
        r: Spike counts, shape (N,)
        lambda_m: GP posterior mean, shape (N,)
        lambda_var: GP posterior variance, shape (N,)

    Returns:
        Optimal lambda0 (scalar tensor)
    """
    sumr = r.sum()
    expexpr = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var)
    sumexpr = expexpr.sum()
    return torch.log(sumr) - torch.log(sumexpr)


def e_step_loop(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    n_estep: int,
    jitter: float = 1e-6,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run Newton loop with moment recomputation and stability checks.

    This matches the structure of the old varGP E-step loop (utils.py:5664-5712):
    - Save previous state before each Newton step
    - Recompute moments after each Newton step (CRITICAL)
    - Stability check: revert if f_mean.mean() > 1000
    - Early stopping: break if rel_change < 1e-5

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        r: Training spike counts, shape (N,)
        n_estep: Number of Newton iterations
        jitter: Small value for numerical stability
        verbose: Print debug info

    Returns:
        lambda_m: Final posterior mean of λ at X, shape (N,)
        lambda_var: Final posterior variance of λ at X, shape (N,)
    """
    # Initial moment computation
    lambda_m, lambda_var, f_mean = compute_moments(model, likelihood, X)

    for i in range(n_estep):
        # Save previous state
        m_prev = get_variational_mean(model).clone()
        V_prev = get_variational_covar(model).clone()
        f_mean_prev = f_mean.clone()

        # Newton update
        m_new, V_new = e_step(model, likelihood, X, r, jitter)
        update_variational_parameters(model, m_new, V_new)

        # Recompute moments (CRITICAL - old code does this after each Newton step)
        lambda_m, lambda_var, f_mean = compute_moments(model, likelihood, X)

        # Stability check: revert if f_mean is too large
        if f_mean.mean() > 1000:
            if verbose:
                print(f"f_mean.mean() = {f_mean.mean():.1f} > 1000, reverting to previous state")
            update_variational_parameters(model, m_prev, V_prev)
            lambda_m, lambda_var, f_mean = compute_moments(model, likelihood, X)
            break

        # Convergence check (early stopping)
        if i > 0:
            rel_change = (f_mean - f_mean_prev).norm() / (f_mean_prev.norm() + 1e-6)
            if rel_change < 1e-5:
                if verbose:
                    print(f"E-step converged after {i+1} iterations (rel_change={rel_change:.2e})")
                break

    return lambda_m, lambda_var


def f_step(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor,
    n_fstep: int,
    lr: float = 0.01,
    verbose: bool = False
):
    """F-step: Optimize A with Adam, lambda0 computed analytically.

    Structural change from baseline: lambda0 is set analytically (not optimized).
    Only A is optimized via gradient descent.

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        r: Training spike counts, shape (N,)
        lambda_m: GP posterior mean (held fixed), shape (N,)
        lambda_var: GP posterior variance (held fixed), shape (N,)
        n_fstep: Number of Adam iterations
        lr: Learning rate for Adam
        verbose: Print debug info
    """
    # First set analytical lambda0
    A = likelihood.A.squeeze()
    new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
    likelihood.lambda0.data.copy_(new_lambda0.unsqueeze(0))

    if n_fstep == 0:
        return

    optimizer = torch.optim.Adam([likelihood.raw_A], lr=lr)

    for _ in range(n_fstep):
        optimizer.zero_grad()

        # Update lambda0 analytically for current A
        A = likelihood.A.squeeze()
        with torch.no_grad():
            new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
            likelihood.lambda0.data.copy_(new_lambda0.unsqueeze(0))

        # Compute loss
        output = model(X)
        loss = -likelihood.expected_log_prob(r, output) + \
               model.variational_strategy.kl_divergence()

        loss.backward()
        optimizer.step()

    # Final lambda0 update
    A = likelihood.A.squeeze()
    with torch.no_grad():
        new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
        likelihood.lambda0.data.copy_(new_lambda0.unsqueeze(0))


def f_step_lbfgs(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor,
    n_fstep: int,
    lr: float = 0.1,
    verbose: bool = False
):
    """F-step using LBFGS optimizer - matches original varGP exactly.

    This replicates utils.py:varGP() F-step structure:
    - Uses LBFGS with strong_wolfe line search
    - Optimizes logA (raw_A = logA since A = exp(raw_A))
    - Computes gradients manually via analytical formula
    - lambda0 set analytically inside closure
    - Stability check: returns inf if f_mean.mean() > 100

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        r: Training spike counts, shape (N,)
        lambda_m: GP posterior mean (held fixed), shape (N,)
        lambda_var: GP posterior variance (held fixed), shape (N,)
        n_fstep: Number of LBFGS iterations (max_iter)
        lr: Learning rate for LBFGS (default 0.1 matches varGP)
        verbose: Print debug info
    """
    if n_fstep == 0:
        return

    # Get current A and convert to logA (varGP uses logA parameterization)
    A_current = likelihood.A.squeeze().detach()
    logA = torch.log(A_current).clone().requires_grad_(True)

    # Initial lambda0 update
    A = torch.exp(logA)
    new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
    likelihood.lambda0.data.copy_(new_lambda0.reshape(likelihood.lambda0.shape))

    # Track f_mean across closure calls (nonlocal update like original)
    f_mean_container = [None]

    # LBFGS optimizer matching original varGP settings
    optimizer = torch.optim.LBFGS(
        [logA],
        lr=lr,
        max_iter=n_fstep,
        tolerance_change=1e-9,
        tolerance_grad=1e-7,
        history_size=n_fstep,
        line_search_fn='strong_wolfe'
    )

    closure_counter = [0]

    def closure():
        closure_counter[0] += 1
        optimizer.zero_grad()

        # Get current A from logA
        A = torch.exp(logA)

        # Update lambda0 analytically (inside closure, like original)
        with torch.no_grad():
            lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
            likelihood.lambda0.data.copy_(lambda0.reshape(likelihood.lambda0.shape))

        # Compute f_mean = exp(A*lambda_m + 0.5*A^2*lambda_var + lambda0)
        f_mean = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + lambda0)
        f_mean_container[0] = f_mean

        # Stability check: return inf if f_mean is too large (like original)
        if f_mean.mean() > 100 or torch.any(torch.isnan(f_mean)):
            if verbose:
                print(f"f_mean.mean() = {f_mean.mean():.1f} at closure call {closure_counter[0]}, returning inf")
            return torch.tensor(float('inf'), device=logA.device, dtype=logA.dtype)

        # Compute loglikelihood: L = A*r@lambda_m + lambda0*sum(r) - sum(f_mean)
        rlambda_m = r @ lambda_m
        sum_r = r.sum()
        loglikelihood = A * rlambda_m + lambda0 * sum_r - f_mean.sum()

        # Compute gradient of loglikelihood w.r.t. logA (analytical, like original)
        # dL/dlogA = A * (r@lambda_m - (lambda_m + A*lambda_var) @ f_mean)
        dloglikelihood_dlogA = A * (rlambda_m - torch.dot(lambda_m + A * lambda_var, f_mean))

        # Set gradient manually (negative because LBFGS minimizes)
        logA.grad = -dloglikelihood_dlogA

        # Return negative loglikelihood (minimize)
        return -loglikelihood

    # Run LBFGS
    optimizer.step(closure)

    # Final updates after optimization
    with torch.no_grad():
        # Update likelihood's raw_A from optimized logA
        # raw_A = logA since A = exp(raw_A)
        likelihood.raw_A.data.copy_(logA.reshape(likelihood.raw_A.shape))

        # Final lambda0 update (like original: "the optimal logA value found by
        # the optimizer might not be the one used in the last closure call")
        A_check = likelihood.A.squeeze()
        new_lambda0 = lambda0_given_A(A_check, r, lambda_m, lambda_var)
        likelihood.lambda0.data.copy_(new_lambda0.reshape(likelihood.lambda0.shape))

    if verbose:
        print(f"F-step LBFGS: {closure_counter[0]} closure calls, A: {likelihood.A.item():.4f}")


def m_step(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    n_mstep: int,
    lr: float = 0.01,
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


def m_step_lbfgs(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    n_mstep: int,
    lr: float = 0.1,  # varGP default
    verbose: bool = False,
    debug: bool = False
):
    """M-step: Optimize kernel hyperparameters using LBFGS.

    Uses autograd for gradients (not analytical).
    Matches varGP LBFGS settings: lr=0.1, strong_wolfe line search.

    Key feature: Returns infinite loss when parameters exceed bounds (matching
    original varGP behavior). This forces LBFGS to try smaller steps.

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        r: Training spike counts, shape (N,)
        n_mstep: Number of LBFGS iterations (max_iter)
        lr: Learning rate for LBFGS (default 0.1 matches varGP)
        verbose: Print debug info
        debug: Print detailed debugging info
    """
    if n_mstep == 0:
        return

    # Get kernel parameters (from ScaleKernel wrapper)
    kernel_params = list(model.covar_module.parameters())

    # Debug: capture initial state
    if debug:
        with torch.no_grad():
            output_init = model(X)
            ell_init = likelihood.expected_log_prob(r, output_init)
            kl_init = model.variational_strategy.kl_divergence()
            loss_init = (-ell_init + kl_init).item()
        param_init = {name: p.clone().detach() for name, p in model.covar_module.named_parameters()}
        print(f"  M-step LBFGS DEBUG: initial loss = {loss_init:.2f}")
        for name, p in param_init.items():
            print(f"    {name}: {p.item():.6f}" if p.numel() == 1 else f"    {name}: shape {p.shape}")

    # LBFGS optimizer matching varGP settings
    optimizer = torch.optim.LBFGS(
        kernel_params,
        lr=lr,
        max_iter=n_mstep,
        tolerance_change=1e-9,
        tolerance_grad=1e-7,
        history_size=100,
        line_search_fn='strong_wolfe'
    )

    closure_counter = [0]
    bounds_violations = [0]
    grad_norms = []

    def closure():
        closure_counter[0] += 1
        optimizer.zero_grad()

        # Check parameter bounds (matching varGP behavior)
        # If bounds violated, set gradient to inf and return inf loss
        # This tells LBFGS to try a smaller step
        base_kernel = model.covar_module.base_kernel
        return_infinite_loss = False
        if hasattr(base_kernel, 'eps_0x') and hasattr(base_kernel, 'eps_0y'):
            eps_0x = base_kernel.eps_0x.item()
            eps_0y = base_kernel.eps_0y.item()
            # Bounds: eps_0 should be in [-0.99, 0.99] (slightly inside image boundary)
            # Using 0.99 instead of 1.0 to keep RF center well inside image
            if not (-0.99 <= eps_0x <= 0.99):
                return_infinite_loss = True
                if base_kernel.eps_0x.requires_grad:
                    base_kernel.eps_0x.grad = torch.full_like(base_kernel.eps_0x, float('inf'))
            if not (-0.99 <= eps_0y <= 0.99):
                return_infinite_loss = True
                if base_kernel.eps_0y.requires_grad:
                    base_kernel.eps_0y.grad = torch.full_like(base_kernel.eps_0y, float('inf'))

        if return_infinite_loss:
            bounds_violations[0] += 1
            if debug:
                print(f"  Bounds violation at closure {closure_counter[0]}: "
                      f"eps_0=({eps_0x:.3f}, {eps_0y:.3f})")
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        # Forward pass
        output = model(X)

        # Compute ELBO loss
        ell = likelihood.expected_log_prob(r, output)
        kl = model.variational_strategy.kl_divergence()
        loss = -ell + kl

        # Check for NaN/inf
        if torch.isnan(loss) or torch.isinf(loss):
            bounds_violations[0] += 1
            # Set all gradients to inf
            for p in kernel_params:
                if p.requires_grad:
                    p.grad = torch.full_like(p, float('inf'))
            return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

        # Backward pass (autograd)
        loss.backward()

        # Debug: track gradient norms
        if debug:
            total_grad_norm = 0.0
            for p in kernel_params:
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            grad_norms.append(total_grad_norm ** 0.5)

        return loss

    # Run LBFGS (single step, but closure called multiple times)
    optimizer.step(closure)

    # Debug: capture final state
    if debug:
        with torch.no_grad():
            output_final = model(X)
            ell_final = likelihood.expected_log_prob(r, output_final)
            kl_final = model.variational_strategy.kl_divergence()
            loss_final = (-ell_final + kl_final).item()
        print(f"  M-step LBFGS DEBUG: final loss = {loss_final:.2f} (delta = {loss_final - loss_init:.2f})")
        print(f"  M-step LBFGS DEBUG: {closure_counter[0]} closure calls, {bounds_violations[0]} bounds violations")
        print(f"  M-step LBFGS DEBUG: grad norms: {grad_norms[:5]}...")
        for name, p in model.covar_module.named_parameters():
            p_init = param_init[name]
            delta = (p - p_init).abs().max().item()
            print(f"    {name}: {p.item():.6f} (delta={delta:.6f})" if p.numel() == 1 else f"    {name}: max_delta={delta:.6f}")

    if verbose:
        print(f"M-step LBFGS: {closure_counter[0]} closure calls, {bounds_violations[0]} bounds violations")


def m_step_lbfgs_grouped(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    n_mstep: int,
    lr_center: float = 0.1,
    lr_sigma0: float = 1.0,  # 10x larger for sigma_0
    lr_other: float = 0.1,
    verbose: bool = False
):
    """M-step with grouped LBFGS: separate optimizers for different parameter groups.

    Splits kernel parameters into 3 groups with different learning rates:
    1. RF center (eps_0x, eps_0y): lr_center, with bounds checking
    2. sigma_0: lr_sigma0 (larger, since gradient is small)
    3. Other (outputscale, beta, rho): lr_other

    This is block coordinate descent - each group optimized while others held fixed.

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        r: Training spike counts, shape (N,)
        n_mstep: Number of LBFGS iterations per group
        lr_center: Learning rate for RF center
        lr_sigma0: Learning rate for sigma_0 (default 10x larger)
        lr_other: Learning rate for other parameters
        verbose: Print debug info
    """
    if n_mstep == 0:
        return

    base_kernel = model.covar_module.base_kernel

    # Identify parameter groups
    center_params = []
    sigma0_params = []
    other_params = []

    for name, p in model.covar_module.named_parameters():
        if 'eps_0x' in name or 'eps_0y' in name:
            center_params.append(p)
        elif 'sigma_0' in name:
            sigma0_params.append(p)
        else:
            other_params.append(p)

    def make_closure(params_to_optimize):
        """Create closure that only computes gradients for specified params."""
        def closure():
            # Zero all gradients
            for p in model.covar_module.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # Forward pass
            output = model(X)
            ell = likelihood.expected_log_prob(r, output)
            kl = model.variational_strategy.kl_divergence()
            loss = -ell + kl

            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

            loss.backward()
            return loss
        return closure

    def make_center_closure():
        """Closure for RF center with bounds checking."""
        def closure():
            for p in model.covar_module.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # Bounds check for eps_0
            if hasattr(base_kernel, 'eps_0x') and hasattr(base_kernel, 'eps_0y'):
                eps_0x = base_kernel.eps_0x.item()
                eps_0y = base_kernel.eps_0y.item()
                if not (-0.99 <= eps_0x <= 0.99):
                    if base_kernel.eps_0x.requires_grad:
                        base_kernel.eps_0x.grad = torch.full_like(base_kernel.eps_0x, float('inf'))
                    return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)
                if not (-0.99 <= eps_0y <= 0.99):
                    if base_kernel.eps_0y.requires_grad:
                        base_kernel.eps_0y.grad = torch.full_like(base_kernel.eps_0y, float('inf'))
                    return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

            output = model(X)
            ell = likelihood.expected_log_prob(r, output)
            kl = model.variational_strategy.kl_divergence()
            loss = -ell + kl

            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(float('inf'), device=X.device, dtype=X.dtype)

            loss.backward()
            return loss
        return closure

    total_closures = [0]

    # Group 1: sigma_0 - Use ADAM (handles small gradients well)
    # LBFGS overshoots for this parameter due to tiny gradient magnitude
    if sigma0_params:
        optimizer = torch.optim.Adam(sigma0_params, lr=lr_sigma0)
        for _ in range(n_mstep):
            optimizer.zero_grad()
            output = model(X)
            ell = likelihood.expected_log_prob(r, output)
            kl = model.variational_strategy.kl_divergence()
            loss = -ell + kl
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                optimizer.step()
        total_closures[0] += n_mstep

    # Group 2: RF center (eps_0x, eps_0y) with bounds
    if center_params:
        optimizer = torch.optim.LBFGS(
            center_params, lr=lr_center, max_iter=n_mstep,
            tolerance_change=1e-9, tolerance_grad=1e-7,
            history_size=100, line_search_fn='strong_wolfe'
        )
        closure = make_center_closure()
        closure_count = [0]
        def counted_closure():
            closure_count[0] += 1
            return closure()
        optimizer.step(counted_closure)
        total_closures[0] += closure_count[0]

    # Group 3: Other params (outputscale, beta, rho)
    if other_params:
        optimizer = torch.optim.LBFGS(
            other_params, lr=lr_other, max_iter=n_mstep,
            tolerance_change=1e-9, tolerance_grad=1e-7,
            history_size=100, line_search_fn='strong_wolfe'
        )
        closure = make_closure(other_params)
        closure_count = [0]
        def counted_closure():
            closure_count[0] += 1
            return closure()
        optimizer.step(counted_closure)
        total_closures[0] += closure_count[0]

    if verbose:
        print(f"M-step LBFGS grouped: {total_closures[0]} total closure calls")


def train_varGP_style(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_iterations: int = 50,
    n_estep: int = 10,
    n_fstep: int = 10,
    n_mstep: int = 10,  # WARNING: n_mstep=0 disables kernel learning - not recommended
    lr_f: float = 0.1,  # Match varGP default (lr_Fparamstep)
    lr_m: float = 0.1,  # Match varGP default (lr_Mstep)
    print_every: int = 10,
    verbose: bool = False,
    device: Optional[torch.device] = None
):
    """Train using varGP-style loop: E-step (with F-step inside), then M-step.

    This matches the structure of utils.py:varGP() main loop:

    for iteration in range(maxiter):
        # E-STEP BLOCK:
        for _ in range(nEstep):
            m, V = Estep(...)
            f_mean, lambda_m, lambda_var = recompute_moments()  # CRITICAL
            stability_check(); convergence_check()
        # F-step (inside E-step block):
        lambda0 = analytical(A); LBFGS([A])

        # M-STEP (separate, LBFGS on kernel params):
        if nMstep > 0 and iteration < maxiter-1:
            LBFGS(kernel_params)

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        train_x: Training inputs, shape (N, n_features)
        train_y: Training spike counts, shape (N,)
        n_iterations: Number of EM iterations
        n_estep: Number of Newton iterations per E-step
        n_fstep: Number of LBFGS iterations for A (F-step)
        n_mstep: Number of LBFGS iterations for kernel (M-step), 0 to disable
        lr_f: Learning rate for F-step LBFGS
        lr_m: Learning rate for M-step LBFGS
        print_every: Print progress every N iterations (0 to disable)
        verbose: Print debug info
        device: Device to use

    Returns:
        dict with keys:
            'losses': List of ELBO values
            'time_estep_total': Total time spent in E-step block (includes F-step)
            'time_mstep_total': Total time spent in M-step block
    """
    import time

    if device is None:
        device = train_x.device

    model = model.to(device)
    likelihood = likelihood.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    losses = []
    time_estep_total = 0.0
    time_mstep_total = 0.0

    for iteration in range(n_iterations):
        # ===== E-STEP BLOCK (includes F-step) =====
        start_time_estep = time.time()

        # Disable kernel gradients (not needed, speeds up analytical grad computation)
        set_kernel_requires_grad(model, False)
        model.eval()

        # Newton loop with moment recomputation
        with torch.no_grad():
            lambda_m, lambda_var = e_step_loop(
                model, likelihood, train_x, train_y, n_estep, verbose=verbose
            )

        # F-step (inside E-step block, matches old varGP structure)
        # Uses LBFGS with logA parameterization (like original varGP)
        # Kernel gradients still disabled (only optimizing A, lambda0)
        model.train()
        with torch.enable_grad():
            f_step_lbfgs(model, likelihood, train_x, train_y,
                         lambda_m, lambda_var, n_fstep, lr_f, verbose=verbose)

        time_estep = time.time() - start_time_estep
        time_estep_total += time_estep

        # ===== M-STEP =====
        start_time_mstep = time.time()

        # Skip M-step on last iteration (like old varGP: "to avoid generating a
        # new eigenspace that will not be used by V and m")
        if n_mstep > 0 and iteration < n_iterations - 1:
            # Re-enable kernel gradients for M-step
            set_kernel_requires_grad(model, True)
            with torch.enable_grad():
                m_step(model, likelihood, train_x, train_y, n_mstep, lr_m, verbose=verbose)
            # Disable kernel gradients after M-step (for loss recording)
            set_kernel_requires_grad(model, False)

        time_mstep = time.time() - start_time_mstep
        time_mstep_total += time_mstep

        # Record loss
        model.eval()
        with torch.no_grad():
            output = model(train_x)
            ell = likelihood.expected_log_prob(train_y, output)
            kl = model.variational_strategy.kl_divergence()
            current_loss = (-ell + kl).item()
        losses.append(current_loss)

        if print_every > 0 and (iteration + 1) % print_every == 0:
            A = likelihood.A.item()
            lambda0 = likelihood.lambda0.item()
            print(f"Iter {iteration+1}/{n_iterations}, Loss: {current_loss:.2f}, "
                  f"A: {A:.4f}, lambda0: {lambda0:.4f}")

    return {
        'losses': losses,
        'time_estep_total': time_estep_total,
        'time_mstep_total': time_mstep_total,
    }


def test_estep():
    """Test E-step implementation."""
    from kernels import ArcCosineKernel
    from likelihoods import PoissonLikelihood
    from model import VariationalGPModel

    torch.manual_seed(42)
    print("Testing E-step...")

    # Synthetic data
    n_train, n_inducing, n_features = 100, 20, 50
    X = torch.randn(n_train, n_features, dtype=torch.float64)
    y = torch.poisson(torch.exp(0.1 * X.sum(dim=1)))
    inducing = X[torch.randperm(n_train)[:n_inducing]].clone()

    # Model
    kernel = ArcCosineKernel(sigma_0=1.0)
    model = VariationalGPModel(inducing, kernel).double()
    likelihood = PoissonLikelihood(A_init=0.1, lambda0_init=1.0).double()

    # Test single E-step
    print("1. Single E-step...")
    m_new, V_new = e_step(model, likelihood, X, y)
    print(f"   m shape: {m_new.shape}, V shape: {V_new.shape}")
    print(f"   V symmetric: {(V_new - V_new.T).abs().max().item():.2e}")
    print(f"   V positive definite: {torch.linalg.eigvalsh(V_new).min().item():.2e}")

    # Test update
    print("2. Update variational params...")
    update_variational_parameters(model, m_new, V_new)
    m_check = model.variational_strategy.variational_distribution.mean
    print(f"   Mean updated: {(m_check - m_new).abs().max().item():.2e}")

    # Test training loop
    print("3. Training loop (5 iters)...")
    kernel = ArcCosineKernel(sigma_0=1.0)
    model = VariationalGPModel(inducing, kernel).double()
    likelihood = PoissonLikelihood(A_init=0.1, lambda0_init=1.0).double()

    losses = train_efm(model, likelihood, X, y,
                       n_iterations=5, n_fstep=2, n_mstep=10,
                       lr=0.1, print_every=1)
    print(f"   Loss decreased: {losses[-1] < losses[0]}")

    print("\nAll tests PASSED!")
    return True


# =============================================================================
# Simple E-F-M Training Loop (Clean Implementation)
# =============================================================================

def train_efm(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_iterations: int = 50,
    n_fstep: int = 10,
    n_mstep: int = 10,
    lr: float = 0.01,
    print_every: int = 10,
    device: Optional[torch.device] = None
):
    """Simple E-F-M training loop.

    Structure per iteration:
        1. E-step: One Newton update of (m, V)
        2. F-step: n_fstep Adam steps on (A, lambda0)
        3. M-step: n_mstep Adam steps on kernel hyperparameters

    This is a stripped-down, simple implementation without:
    - Moment recomputation between Newton steps
    - Analytical lambda0
    - Stability checks / early stopping
    - Multiple E-step iterations

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        train_x: Training inputs, shape (N, n_features)
        train_y: Training spike counts, shape (N,)
        n_iterations: Number of E-F-M iterations
        n_fstep: Number of Adam steps for F-step (A, lambda0)
        n_mstep: Number of Adam steps for M-step (kernel)
        lr: Learning rate for Adam
        print_every: Print progress every N iterations (0 to disable)
        device: Device to use

    Returns:
        losses: List of ELBO values per iteration
    """
    if device is None:
        device = train_x.device

    model = model.to(device)
    likelihood = likelihood.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # Separate optimizers for F-step and M-step
    optimizer_f = torch.optim.Adam(likelihood.parameters(), lr=lr)
    optimizer_m = torch.optim.Adam(model.covar_module.parameters(), lr=lr)

    losses = []

    for iteration in range(n_iterations):
        # ===== E-STEP: One Newton update =====
        model.eval()
        with torch.no_grad():
            m_new, V_new = e_step(model, likelihood, train_x, train_y)
            update_variational_parameters(model, m_new, V_new)

        # ===== F-STEP: Optimize A, lambda0 =====
        model.train()
        for _ in range(n_fstep):
            optimizer_f.zero_grad()
            output = model(train_x)
            loss = -likelihood.expected_log_prob(train_y, output) + \
                   model.variational_strategy.kl_divergence()
            loss.backward()
            optimizer_f.step()

        # ===== M-STEP: Optimize kernel hyperparameters =====
        for _ in range(n_mstep):
            optimizer_m.zero_grad()
            output = model(train_x)
            loss = -likelihood.expected_log_prob(train_y, output) + \
                   model.variational_strategy.kl_divergence()
            loss.backward()
            optimizer_m.step()

        # Record loss
        model.eval()
        with torch.no_grad():
            output = model(train_x)
            ell = likelihood.expected_log_prob(train_y, output)
            kl = model.variational_strategy.kl_divergence()
            current_loss = (-ell + kl).item()
        losses.append(current_loss)

        if print_every > 0 and (iteration + 1) % print_every == 0:
            A = likelihood.A.item()
            lambda0 = likelihood.lambda0.item()
            print(f"Iter {iteration+1}/{n_iterations}, Loss: {current_loss:.2f}, "
                  f"A: {A:.4f}, lambda0: {lambda0:.4f}")

    return losses


if __name__ == '__main__':
    test_estep()
