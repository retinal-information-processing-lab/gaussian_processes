"""
DEPRECATED: Training function for vargp_style mode

This mode is no longer maintained. Use eigenspace or default_gpy instead.

This file is archived for reference only.
Last working commit: 23c3856
"""

import time
import warnings
from typing import Optional

import torch
import gpytorch


def train_varGP_style(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_iterations: int,
    n_estep: int,
    n_fstep: int,
    n_mstep: int,  # WARNING: n_mstep=0 disables kernel learning - not recommended
    lr_f: float,  # Match varGP default (lr_Fparamstep)
    lr_m: float,  # Match varGP default (lr_Mstep)
    print_every: int,
    verbose: bool = False,
    device: Optional[torch.device] = None,
    use_cache: bool = True,  # Enable kernel caching for performance (11.7x fewer kernel calls)
    *,  # Force keyword-only arguments below
    explicit_unwhitening: bool,  # REQUIRED: whether to do L_K conversions in E-step
):
    """Train using varGP-style loop: E-step (with F-step inside), then M-step.

    DEPRECATED: This mode is archived. Use eigenspace or default_gpy instead.

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
    # Import from deprecated modules
    from vargp_style_estep import compute_kernel_cache, e_step_loop
    from vargp_style_fstep import f_step_lbfgs
    from vargp_style_mstep import m_step
    from vargp_style_whitening import set_kernel_requires_grad

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
