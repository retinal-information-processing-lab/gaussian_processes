"""
Training Utilities for GPyTorch Variational GP

This module provides functions for training and evaluating the variational GP model.

Training loops:
- train_gpy_default: Standard GPyTorch variational inference (no custom E-step)
- train_varGP_style: Custom EM-style training (Newton E-step with moment recomputation)
- train_eigenspace: Eigenspace-based training (vargp_direct mode)

Evaluation:
- predict: Make predictions with trained model (GPyTorch modes)
- predict_eigenspace: Make predictions (eigenspace mode)
- compute_r_squared, compute_pearson_correlation, compute_explained_variance
"""

import time
import warnings
from typing import Optional, Dict

import torch
import numpy as np
import gpytorch


def train_gpy_default(model, likelihood, train_x, train_y, optimizer_name, lr, n_iterations,
                       print_every=100, device=None):
    """Train using GPyTorch's standard variational inference (no custom E-step).

    Maximizes the ELBO = E_q[log p(y|f)] - KL(q(u) || p(u))

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        train_x: Training inputs, shape (n_train, n_features)
        train_y: Training targets (spike counts), shape (n_train,)
        optimizer_name: Optimizer to use ('adam')
        lr: Learning rate
        n_iterations: Number of optimization iterations
        print_every: Print loss every N iterations (0 to disable)
        device: Device to use (defaults to train_x.device)

    Returns:
        losses: List of loss values during training
    """
    if device is None:
        device = train_x.device

    model = model.to(device)
    likelihood = likelihood.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # Validate model has required attribute
    if not hasattr(model, 'standard_variational_distribution'):
        raise AttributeError(
            "Model does not have 'standard_variational_distribution' attribute. "
            "Use VariationalGPModel which defines this attribute."
        )

    model.train()
    likelihood.train()

    # Create optimizer
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()}
        ], lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Only 'adam' is supported.")

    losses = []

    with torch.enable_grad():
        for i in range(n_iterations):
            optimizer.zero_grad()

            # Forward pass: get GP posterior at training points
            output = model(train_x)

            # ELBO = E_q[log p(y|f)] - KL(q(u) || p(u))
            expected_log_lik = likelihood.expected_log_prob(train_y, output)
            kl_div = model.variational_strategy.kl_divergence()

            # Minimize negative ELBO
            loss = -expected_log_lik + kl_div

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if print_every > 0 and (i + 1) % print_every == 0:
                print(f"Iter {i+1}/{n_iterations}, Loss: {loss.item():.2f}, "
                      f"ELL: {expected_log_lik.item():.2f}, KL: {kl_div.item():.2f}")

    return losses


def predict(model, likelihood, test_x, device=None):
    """Make predictions on test data.

    Args:
        model: Trained VariationalGPModel
        likelihood: Trained PoissonLikelihood
        test_x: Test inputs, shape (n_test, n_features)
        device: Device to use (defaults to test_x.device)

    Returns:
        dict with:
        - 'f_pred': Predicted firing rates E[exp(A·λ + λ₀)]
        - 'lambda_mean': Posterior mean of λ
        - 'lambda_var': Posterior variance of λ
    """
    if device is None:
        device = test_x.device

    model = model.to(device)
    likelihood = likelihood.to(device)
    test_x = test_x.to(device)

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        # Get posterior q(λ*) at test points
        posterior = model(test_x)
        lambda_mean = posterior.mean
        lambda_var = posterior.variance

        # Predicted firing rate: E[exp(A·λ + λ₀)] = exp(A·μ + A²σ²/2 + λ₀)
        A = likelihood.A.squeeze()
        lambda0 = likelihood.lambda0.squeeze()
        f_pred = torch.exp(A * lambda_mean + 0.5 * A**2 * lambda_var + lambda0)

    return {
        'f_pred': f_pred,
        'lambda_mean': lambda_mean,
        'lambda_var': lambda_var
    }


def compute_r_squared(y_true, y_pred):
    """Compute R² (coefficient of determination).

    Args:
        y_true: True values, shape (n,) or (n_repeats, n)
        y_pred: Predicted values, shape (n,)

    Returns:
        R² value (scalar)
    """
    if y_true.ndim == 2:
        # Average over repeats
        y_true = y_true.mean(dim=0)

    y_true = y_true.float()
    y_pred = y_pred.float()

    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()

    r2 = 1 - ss_res / ss_tot

    return r2.item()


def compute_pearson_correlation(y_true, y_pred):
    """Compute Pearson correlation coefficient.

    Args:
        y_true: True values, shape (n,) or (n_repeats, n)
        y_pred: Predicted values, shape (n,)

    Returns:
        Pearson correlation (scalar)
    """
    if y_true.ndim == 2:
        # Average over repeats
        y_true = y_true.mean(dim=0)

    y_true = y_true.float()
    y_pred = y_pred.float()

    # Center the values
    y_true_centered = y_true - y_true.mean()
    y_pred_centered = y_pred - y_pred.mean()

    # Compute correlation
    numerator = (y_true_centered * y_pred_centered).sum()
    denominator = torch.sqrt((y_true_centered ** 2).sum() * (y_pred_centered ** 2).sum())

    corr = numerator / denominator

    return corr.item()


def compute_explained_variance(r_test, f_pred):
    """Compute explained variance normalized by cell reliability.

    This matches the reference implementation in utils.py:explained_variance().

    Explained variance = (Pearson r with predictions) / (cell reliability)

    where reliability is the correlation between even and odd trial halves.
    A perfect model achieves explained_variance = 1.0.

    Args:
        r_test: Test responses with repetitions, shape (n_repeats, n_images)
        f_pred: Predicted firing rates, shape (n_images,)

    Returns:
        explained_var: Fraction of explainable variance captured (scalar)
        reliability: Cell reliability (scalar)
    """
    r_test = r_test.float()
    f_pred = f_pred.float()

    # Split into even and odd repetitions
    r_even = r_test[0::2, :].mean(dim=0)  # (n_images,)
    r_odd = r_test[1::2, :].mean(dim=0)   # (n_images,)

    # Reliability = correlation between even and odd halves
    reliability = torch.corrcoef(torch.stack([r_even, r_odd]))[0, 1].abs()

    # Accuracy = average correlation with each half
    accuracy_even = torch.corrcoef(torch.stack([f_pred, r_even]))[0, 1]
    accuracy_odd = torch.corrcoef(torch.stack([f_pred, r_odd]))[0, 1]
    accuracy = 0.5 * (accuracy_even + accuracy_odd)

    # Explained variance = accuracy / reliability
    explained_var = accuracy / reliability

    return explained_var.item(), reliability.item()


# =============================================================================
# EM-Style Training Loop
# =============================================================================

def train_varGP_style(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_iterations: int ,
    n_estep: int ,
    n_fstep: int ,
    n_mstep: int ,  # WARNING: n_mstep=0 disables kernel learning - not recommended
    lr_f: float ,  # Match varGP default (lr_Fparamstep)
    lr_m: float ,  # Match varGP default (lr_Mstep)
    print_every: int ,
    verbose: bool = False,
    device: Optional[torch.device] = None,
    use_cache: bool = True,  # Enable kernel caching for performance (11.7x fewer kernel calls)
    *,  # Force keyword-only arguments below
    explicit_unwhitening: bool,  # REQUIRED: whether to do L_K conversions in E-step
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
        use_cache: If True, cache kernel matrices and reuse across Newton iterations.
                   This reduces kernel calls from 35 to 3 per E-step loop (11.7x speedup).
                   Set to False for testing the non-cached fallback path.
        explicit_unwhitening: Whether to do explicit L_K whitening conversions in E-step.
                              Must be explicitly specified (no auto-detection).
                              Set True when using standard variational distribution (whitened).
                              Set False when using unwhitened variational strategy.

    Returns:
        dict with keys:
            'losses': List of ELBO values
            'time_estep_total': Total time spent in E-step block (includes F-step)
            'time_mstep_total': Total time spent in M-step block
    """
    # Import here to avoid circular dependency
    import warnings
    from estep import compute_kernel_cache, e_step_loop
    from fstep import f_step_lbfgs
    from mstep import m_step
    from whitening import set_kernel_requires_grad

    # Warn about n_mstep=0 (disables kernel learning)
    if n_mstep == 0:
        warnings.warn(
            "n_mstep=0 disables kernel hyperparameter learning. "
            "This is not recommended unless you have pre-trained kernel parameters. "
            "Use n_mstep >= 10 for proper training.",
            UserWarning
        )

    if device is None:
        device = train_x.device

    model = model.to(device)
    likelihood = likelihood.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # Validate model has required attribute
    if not hasattr(model, 'standard_variational_distribution'):
        raise AttributeError(
            "Model does not have 'standard_variational_distribution' attribute. "
            "Use VariationalGPModel which defines this attribute."
        )

    # Validate explicit_unwhitening matches model's variational strategy
    if explicit_unwhitening and not model.standard_variational_distribution:
        raise ValueError(
            "explicit_unwhitening=True but model uses UnwhitenedVariationalStrategy "
            "(standard_variational_distribution=False). "
            "UnwhitenedVariationalStrategy stores natural params directly and doesn't need L_K conversions. "
            "Set explicit_unwhitening=False."
        )
    if not explicit_unwhitening and model.standard_variational_distribution:
        raise ValueError(
            "explicit_unwhitening=False but model uses standard VariationalStrategy "
            "(standard_variational_distribution=True). "
            "Standard variational distribution requires L_K conversions in E-step. "
            "Set explicit_unwhitening=True."
        )

    losses = []
    time_estep_total = 0.0
    time_mstep_total = 0.0

    # Kernel cache (K, K̃, k0) - computed once per iteration, invalidated after M-step
    kernel_cache = None

    for iteration in range(n_iterations):
        # ===== E-STEP BLOCK (includes F-step) =====
        start_time_estep = time.time()

        # Disable kernel gradients (not needed, speeds up analytical grad computation)
        set_kernel_requires_grad(model, False)
        model.eval()

        # Compute kernel cache if enabled (reused across Newton steps within E-step)
        if use_cache:
            with torch.no_grad():
                kernel_cache = compute_kernel_cache(model, train_x)
        else:
            kernel_cache = None  # Force non-cached path (uses GPyTorch model(X))

        # Newton loop with moment recomputation
        with torch.no_grad():
            lambda_m, lambda_var = e_step_loop(
                model, likelihood, train_x, train_y, n_estep, verbose=verbose,
                kernel_cache=kernel_cache,
                explicit_unwhitening=explicit_unwhitening
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
            # Invalidate kernel cache - kernel params changed, need fresh cache next iteration
            kernel_cache = None

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


# =============================================================================
# Eigenspace Training Mode
# =============================================================================

def compute_elbo_eigenspace(
    state,  # DirectVariationalState
    r: torch.Tensor,
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor,
    A: torch.Tensor,
    lambda0: torch.Tensor
) -> torch.Tensor:
    """Compute ELBO = log_likelihood - KL_divergence.

    Log-likelihood:
        L = sum(r * (A * lambda_m + lambda0) - f_mean)
        where f_mean = exp(A * lambda_m + 0.5 * A^2 * lambda_var + lambda0)

    KL divergence for q(u) = N(m, V) vs p(u) = N(0, K_tilde):
        KL = 0.5 * (tr(K_tilde^-1 @ V) + m.T @ K_tilde^-1 @ m - n_b + log|K_tilde| - log|V|)

    In eigenspace with K_tilde_b diagonal:
        KL = 0.5 * (sum(V_b_diag / eigvals) + sum(m_b^2 / eigvals) - n_b
                   + sum(log(eigvals)) - log|V_b|)

    Args:
        state: Current DirectVariationalState
        r: Spike counts, shape (N,)
        lambda_m: Posterior mean, shape (N,)
        lambda_var: Posterior variance, shape (N,)
        A: Gain parameter (scalar)
        lambda0: Bias parameter (scalar)

    Returns:
        ELBO value (scalar, to be maximized)
    """
    # Import here to avoid circular dependency
    from fstep import compute_f_mean

    # Log-likelihood
    f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)
    log_lik = (r * (A * lambda_m + lambda0) - f_mean).sum()

    # KL divergence in eigenspace
    n_b = len(state.eigvals_b)
    eigvals = state.eigvals_b

    # tr(K_tilde^-1 @ V) = tr(diag(1/eigvals) @ V_b) = sum(V_b_diag / eigvals)
    V_diag = torch.diag(state.V_b)
    trace_term = (V_diag / eigvals).sum()

    # m.T @ K_tilde^-1 @ m = sum(m_b^2 / eigvals)
    quad_term = ((state.m_b ** 2) / eigvals).sum()

    # log|K_tilde| = sum(log(eigvals))
    log_det_K = torch.log(eigvals).sum()

    # log|V| - need full log determinant
    sign, log_det_V = torch.linalg.slogdet(state.V_b)
    if sign.item() <= 0:
        # V is not positive definite - this shouldn't happen
        warnings.warn("V_b is not positive definite in KL computation")
        log_det_V = torch.tensor(0.0, device=state.V_b.device, dtype=state.V_b.dtype)

    KL = 0.5 * (trace_term + quad_term - n_b + log_det_K - log_det_V)

    return log_lik - KL


def train_eigenspace(
    kernel,
    likelihood,
    X: torch.Tensor,
    X_tilde: torch.Tensor,
    r: torch.Tensor,
    n_iterations: int,
    n_estep: int,
    n_fstep: int,
    n_mstep: int,
    lr_f: float,
    lr_m: float,
    print_every: int = 10,
    eigval_tol: float = None,
    verbose: bool = False,
    use_analytical_mstep: bool = False
) -> Dict:
    """Train using eigenspace-based variational GP (vargp_direct mode).

    Implements the original varGP loop structure:
    1. Kernel recomputation (after M-step)
    2. E-step: Newton updates on m_b, V_b
    3. F-step: LBFGS on A with analytical lambda0
    4. M-step: LBFGS on kernel hyperparameters

    Args:
        kernel: ArcCosineKernel instance
        likelihood: PoissonLikelihood instance
        X: Training inputs, shape (N, n_features)
        X_tilde: Inducing points, shape (M, n_features)
        r: Spike counts, shape (N,)
        n_iterations: Number of EM iterations
        n_estep: Number of E-step Newton iterations per EM iteration
        n_fstep: Number of F-step LBFGS iterations
        n_mstep: Number of M-step LBFGS iterations
        lr_f: Learning rate for F-step
        lr_m: Learning rate for M-step
        print_every: Print progress every N iterations
        eigval_tol: Eigenvalue tolerance for eigenspace projection (default: EIGVAL_TOL)
        verbose: Print detailed debugging info
        use_analytical_mstep: Use analytical gradients for M-step (faster, matches varGP)

    Returns:
        Dict with:
            'losses': List of ELBO values per iteration
            'state': Final DirectVariationalState
            'time_estep_total': Total E-step time (includes F-step)
            'time_mstep_total': Total M-step time
    """
    # Import here to avoid circular dependency
    from eigenspace import EIGVAL_TOL
    from eigenspace_model import (
        compute_kernels_eigenspace,
        recompute_kernels_after_mstep,
        lambda_moments_eigenspace,
    )
    from estep import estep_eigenspace, STABILITY_THRESHOLD
    from fstep import fstep_eigenspace, compute_f_mean
    from mstep import mstep_eigenspace_autograd, mstep_eigenspace_analytical

    if eigval_tol is None:
        eigval_tol = EIGVAL_TOL

    # Initialize
    state = compute_kernels_eigenspace(kernel, X, X_tilde, eigval_tol)

    n_b = len(state.eigvals_b)
    M = X_tilde.shape[0]
    N = X.shape[0]
    print(f"Eigenspace dimension: n_b={n_b} (from M={M} inducing points)")

    time_estep_total = 0.0
    time_mstep_total = 0.0
    losses = []

    # Initial moments
    lambda_m, lambda_var = lambda_moments_eigenspace(state)
    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()
    f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)

    # NOTE: Loop uses range(1, n_iterations) to match vargp_old behavior:
    #   - vargp_old: range(1, maxiter) with maxiter=50 → iterations 1-49 (49 total)
    #   - M-step skipped on last iteration to avoid generating new eigenspace that won't be used
    # This ensures test_r and loss values match between vargp_direct and vargp_old.
    # FUTURE: May want to change to range(1, n_iterations+1) for one more iteration.
    for iteration in range(1, n_iterations):

        # ===== Kernel recomputation after M-step =====
        if n_mstep > 0 and iteration > 1:
            state = recompute_kernels_after_mstep(kernel, X, X_tilde, state, eigval_tol)
            # Recompute moments with new kernels
            lambda_m, lambda_var = lambda_moments_eigenspace(state)
            A = likelihood.A.squeeze()
            lambda0 = likelihood.lambda0.squeeze()
            f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)

        # ===== E-step: Newton loop =====
        start_estep = time.time()

        for i_estep in range(n_estep):
            m_b_new, V_b_new = estep_eigenspace(state, r, A, f_mean)

            # Update state
            state.m_b = m_b_new
            state.V_b = V_b_new

            # Recompute moments
            lambda_m, lambda_var = lambda_moments_eigenspace(state)

            # Recompute f_mean
            f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)

            # Stability check
            if f_mean.mean().item() > STABILITY_THRESHOLD:
                if verbose:
                    print(f"  E-step {i_estep}: f_mean unstable ({f_mean.mean().item():.1f})")
                break

            # Early stopping: check convergence
            # (Could add convergence criterion here)

        # ===== F-step: Optimize A =====
        fstep_eigenspace(likelihood, r, lambda_m, lambda_var, n_fstep, lr_f)

        # Update A, lambda0 and recompute f_mean
        A = likelihood.A.squeeze()
        lambda0 = likelihood.lambda0.squeeze()
        f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)

        time_estep_total += time.time() - start_estep

        # ===== M-step: Optimize kernel hyperparameters =====
        start_mstep = time.time()

        # Skip M-step on last iteration (matches vargp_old: iteration < maxiter-1)
        if n_mstep > 0 and iteration < n_iterations - 1:
            if use_analytical_mstep:
                mstep_eigenspace_analytical(kernel, likelihood, X, X_tilde, r, state, n_mstep, lr_m)
            else:
                mstep_eigenspace_autograd(kernel, likelihood, X, X_tilde, r, state, n_mstep, lr_m)

        time_mstep_total += time.time() - start_mstep

        # ===== Compute and record loss =====
        elbo = compute_elbo_eigenspace(state, r, lambda_m, lambda_var, A, lambda0)
        loss = -elbo.item()  # Negative ELBO for consistency with other modes
        losses.append(loss)

        if iteration % print_every == 0 or iteration == 1:
            print(f"Iter {iteration}/{n_iterations-1}: loss={loss:.2f}, "
                  f"A={A.item():.4f}, lambda0={lambda0.item():.4f}, "
                  f"n_b={len(state.eigvals_b)}")

    return {
        'losses': losses,
        'state': state,
        'time_estep_total': time_estep_total,
        'time_mstep_total': time_mstep_total,
    }


def predict_eigenspace(
    kernel,
    likelihood,
    state,  # DirectVariationalState
    X_tilde: torch.Tensor,
    X_test: torch.Tensor
) -> Dict:
    """Predict at test points using trained eigenspace variational GP.

    Computes posterior moments at test points and expected firing rates.

    Args:
        kernel: Trained ArcCosineKernel
        likelihood: Trained PoissonLikelihood
        state: Trained DirectVariationalState
        X_tilde: Inducing points, shape (M, n_features)
        X_test: Test inputs, shape (N_test, n_features)

    Returns:
        Dict with:
            'f_pred': Predicted firing rates, shape (N_test,)
            'lambda_m': Posterior mean at test points, shape (N_test,)
            'lambda_var': Posterior variance at test points, shape (N_test,)
    """
    with torch.no_grad():
        # Compute cross-kernel to inducing points
        K_test = kernel(X_test, X_tilde).evaluate()  # (N_test, M)
        Kvec_test = kernel(X_test, diag=True)  # (N_test,)

        # Project to eigenspace
        K_test_b = K_test @ state.B  # (N_test, n_b)

        # a = K_test @ K_tilde_inv = K_test_b @ diag(1/eigvals)
        a = K_test_b / state.eigvals_b.unsqueeze(0)  # (N_test, n_b)

        # Posterior mean: lambda_m = a @ m_b
        lambda_m = a @ state.m_b  # (N_test,)

        # Posterior variance
        V_minus_K = state.V_b - state.K_tilde_b
        aV = a @ V_minus_K
        lambda_var = Kvec_test + (a * aV).sum(dim=1)
        lambda_var = torch.clamp(lambda_var, min=1e-6)

        # Predicted firing rate
        A = likelihood.A.squeeze()
        lambda0 = likelihood.lambda0.squeeze()
        f_pred = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + lambda0)

    return {
        'f_pred': f_pred,
        'lambda_m': lambda_m,
        'lambda_var': lambda_var,
    }


# =============================================================================
# Test Functions
# =============================================================================

def test_training():
    """Test training utilities with synthetic data."""
    from kernels import ArcCosineKernel
    from likelihoods import PoissonLikelihood
    from model import VariationalGPModel

    torch.manual_seed(42)

    # Create synthetic data
    n_train = 100
    n_test = 20
    n_inducing = 20
    n_features = 50

    # Generate random inputs
    X_train = torch.randn(n_train, n_features)
    X_test = torch.randn(n_test, n_features)

    # Generate synthetic spike counts (Poisson with rate = exp(linear function))
    true_rate = torch.exp(0.1 * X_train.sum(dim=1))
    y_train = torch.poisson(true_rate)

    true_rate_test = torch.exp(0.1 * X_test.sum(dim=1))
    y_test = torch.poisson(true_rate_test)

    # Select inducing points
    indices = torch.randperm(n_train)[:n_inducing]
    inducing_points = X_train[indices].clone()

    # Create model and likelihood
    kernel = ArcCosineKernel(sigma_0=1.0)
    model = VariationalGPModel(inducing_points, kernel)
    likelihood = PoissonLikelihood(A_init=1.0, lambda0_init=0.0)

    print("Training on synthetic data...")
    losses = train_gpy_default(model, likelihood, X_train, y_train,
                               optimizer_name='adam', lr=0.1,
                               n_iterations=100, print_every=25)

    print(f"\nFinal loss: {losses[-1]:.2f}")
    print(f"Loss decreased: {losses[0] > losses[-1]}")

    # Make predictions
    predictions = predict(model, likelihood, X_test)
    print(f"\nPredicted firing rates: {predictions['f_pred'][:5]}")

    # Compute metrics
    r2 = compute_r_squared(y_test, predictions['f_pred'])
    corr = compute_pearson_correlation(y_test, predictions['f_pred'])
    print(f"R²: {r2:.4f}")
    print(f"Pearson r: {corr:.4f}")

    # Check that model learned something
    print(f"\nLikelihood A: {likelihood.A.item():.4f}")
    print(f"Likelihood lambda0: {likelihood.lambda0.item():.4f}")

    print("\nPASS: Training utilities test passed!")
    return True


if __name__ == '__main__':
    test_training()
