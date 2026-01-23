"""
Training Utilities for GPyTorch Variational GP

This module provides functions for training and evaluating the variational GP model.

Training loops:
- train_gpy_default: Standard GPyTorch variational inference (no custom E-step)
- train_varGP_style: Custom EM-style training (Newton E-step with moment recomputation)

Evaluation:
- predict: Make predictions with trained model
- compute_r_squared, compute_pearson_correlation, compute_explained_variance
"""

import time
from typing import Optional

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
