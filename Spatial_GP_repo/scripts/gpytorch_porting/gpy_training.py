"""
Training Functions for Standard GPyTorch Variational GP

This module contains training and prediction functions for the default_gpy mode,
which uses standard GPyTorch variational inference (VariationalStrategy).

Key functions:
- train_gpy_default: Standard ELBO optimization with LBFGS or Adam
- predict: Prediction at test points using GPyTorch model

Extracted from train.py during codebase reorganization (2025-02).
"""

import warnings

import torch
from linear_operator import settings as lo_settings


def train_gpy_default(model, likelihood, train_x, train_y, optimizer_name, lr, n_iterations,
                       print_every=100, device=None,
                       early_stop=True, stop_window=20, stop_thresh=1e-3, min_iterations=10,
                       lbfgs_max_iter=20,
                       jitter=1e-4, cholesky_max_tries=3):
    """Train using GPyTorch's standard variational inference (no custom E-step).

    Maximizes the ELBO = E_q[log p(y|f)] - KL(q(u) || p(u))

    Args:
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        train_x: Training inputs, shape (n_train, n_features)
        train_y: Training targets (spike counts), shape (n_train,)
        optimizer_name: Optimizer to use ('adam' or 'lbfgs')
        lr: Learning rate
        n_iterations: Maximum number of optimization iterations
        print_every: Print loss every N iterations (0 to disable)
        device: Device to use (defaults to train_x.device)
        early_stop: Enable early stopping based on loss stability (default: True)
        stop_window: Number of iterations to look back for improvement (default: 20)
        stop_thresh: Minimum relative improvement over window to continue (default: 1e-3 = 0.1%)
        min_iterations: Minimum iterations before early stopping can trigger (default: 10)
        lbfgs_max_iter: Max inner iterations for LBFGS per outer step (default: 20)
        jitter: Jitter value for Cholesky retry schedule starting point (default: 1e-4)
        cholesky_max_tries: Number of Cholesky retry attempts (default: 3)

    Returns:
        dict: {'losses': list, 'stopped_early': bool, 'final_iteration': int}
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

    # Collect all parameters
    all_params = list(model.parameters()) + list(likelihood.parameters())

    # Create optimizer
    if optimizer_name == 'lbfgs':
        optimizer = torch.optim.LBFGS(
            all_params,
            lr=lr,
            max_iter=lbfgs_max_iter,
            line_search_fn='strong_wolfe'
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()}
        ], lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Supported: 'lbfgs', 'adam'.")

    losses = []

    # For LBFGS, we need a closure that computes loss and gradients
    # We store the last computed values for logging
    last_output = [None]
    last_loss = [None]
    last_ell = [None]
    last_kl = [None]

    def closure():
        optimizer.zero_grad()
        # Reject trial step if parameters are out of bounds
        kernel = model.covar_module
        if hasattr(kernel, 'params_in_bounds') and not kernel.params_in_bounds():
            return torch.tensor(float('inf'), device=train_x.device, dtype=train_x.dtype)
        if hasattr(likelihood, 'params_in_bounds') and not likelihood.params_in_bounds():
            return torch.tensor(float('inf'), device=train_x.device, dtype=train_x.dtype)
        # Guard: catch kernel NaN/errors during LBFGS line search.
        # Matches the pattern in eigenspace_mstep.py (Guardrails 7-8):
        # return inf so LBFGS rejects the trial step.
        try:
            output = model(train_x)
        except Exception:
            # Kernel produced NaN (e.g. all-NaN K_uu) — reject this step
            return torch.tensor(float('inf'), device=train_x.device, dtype=train_x.dtype)
        ell = likelihood.expected_log_prob(train_y, output)
        kl = model.variational_strategy.kl_divergence()
        loss = -ell + kl
        if torch.isnan(loss) or torch.isinf(loss):
            # NaN/Inf loss (e.g. from overflow in kernel or likelihood) — reject
            return torch.tensor(float('inf'), device=train_x.device, dtype=train_x.dtype)
        loss.backward()
        # Store for logging
        last_output[0] = output
        last_loss[0] = loss
        last_ell[0] = ell
        last_kl[0] = kl
        return loss

    # Early stopping state
    stopped_early = False
    final_iteration = 0

    if train_x.dtype == torch.float64 and jitter >= 1e-4:
        warnings.warn(
            f"jitter={jitter} is high for float64 (GPyTorch default is 1e-6). "
            f"Consider reducing jitter for float64 training."
        )

    # Cholesky stability: GPyTorch adds jitter_val (our 1e-4) to K_uu before
    # Cholesky, then promotes to float64. If Cholesky still fails,
    # psd_safe_cholesky retries with escalating jitter. We override:
    #   - cholesky_jitter: retry starting jitter = our jitter value (not 1e-8)
    #   - cholesky_max_tries: number of retries (each adds 10x more jitter)
    # With jitter=1e-4, max_tries=3: retries at 1e-4, 1e-3, 1e-2.
    with torch.enable_grad(), \
         lo_settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
         lo_settings.cholesky_max_tries(cholesky_max_tries):
        for i in range(n_iterations):
            if optimizer_name == 'lbfgs':
                optimizer.step(closure)
                loss = last_loss[0]
                expected_log_lik = last_ell[0]
                kl_div = last_kl[0]
            else:
                optimizer.zero_grad()
                output = model(train_x)
                expected_log_lik = likelihood.expected_log_prob(train_y, output)
                kl_div = model.variational_strategy.kl_divergence()
                loss = -expected_log_lik + kl_div
                loss.backward()
                optimizer.step()

            # Clamp parameters to physical bounds after each step
            # (projected gradient descent — matches eigenspace_mstep.py pattern)
            kernel = model.covar_module
            if hasattr(kernel, 'clamp_hyperparameters'):
                kernel.clamp_hyperparameters()
            if hasattr(likelihood, 'clamp_params'):
                likelihood.clamp_params()

            current_loss = loss.item()
            losses.append(current_loss)
            final_iteration = i + 1

            if print_every > 0 and (i + 1) % print_every == 0:
                print(f"Iter {i+1}/{n_iterations}, Loss: {current_loss:.2f}, "
                      f"ELL: {expected_log_lik.item():.2f}, KL: {kl_div.item():.2f}")

            # Early stopping: check if loss improved enough over last stop_window iterations
            if early_stop and len(losses) >= stop_window + min_iterations:
                old_loss = losses[-(stop_window + 1)]
                rel_improvement = (old_loss - current_loss) / abs(old_loss)
                if rel_improvement < stop_thresh:
                    stopped_early = True
                    if print_every > 0:
                        print(f"Early stopping at iteration {i+1}: "
                              f"loss improved only {rel_improvement*100:.3f}% over last {stop_window} iterations")
                    break

    return {
        'losses': losses,
        'stopped_early': stopped_early,
        'final_iteration': final_iteration
    }


def predict(model, likelihood, test_x, device=None,
            jitter=1e-4, cholesky_max_tries=3):
    """Make predictions on test data.

    Args:
        model: Trained VariationalGPModel
        likelihood: Trained PoissonLikelihood
        test_x: Test inputs, shape (n_test, n_features)
        device: Device to use (defaults to test_x.device)
        jitter: Jitter value for Cholesky retry schedule starting point (default: 1e-4)
        cholesky_max_tries: Number of Cholesky retry attempts (default: 3)

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

    with torch.no_grad(), \
         lo_settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
         lo_settings.cholesky_max_tries(cholesky_max_tries):
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
