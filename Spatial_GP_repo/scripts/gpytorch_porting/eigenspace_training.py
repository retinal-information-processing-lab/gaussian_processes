"""
Eigenspace Training Functions for DirectVGPModel

This module contains training, evaluation, and prediction functions for the
eigenspace-based variational GP implementation (DirectVGPModel).

Key functions:
- train_eigenspace: Main EM-style training loop
- compute_elbo_eigenspace: ELBO computation in eigenspace
- predict_eigenspace: Prediction at test points

Extracted from train.py during codebase reorganization (2025-02).
"""

import time
import warnings
from typing import Dict

import torch


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
    from utils import compute_f_mean

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
    model,  # DirectVGPModel
    r: torch.Tensor,
    n_iterations: int,
    n_estep: int,
    n_fstep: int,
    n_mstep: int,
    lr_f: float,
    lr_m: float,
    print_every: int = 10,
    verbose: bool = False,
    use_analytical_mstep: bool = False,
    capture_checkpoints: bool = False,
    early_stop: bool = True,
    stop_window: int = 20,
    stop_thresh: float = 5e-3,
    min_iterations: int = 10,
    stability_threshold: float = 1000
) -> Dict:
    """Train using eigenspace-based variational GP - model-based API.

    Uses DirectVGPModel which owns kernel, likelihood, and state.

    Args:
        model: DirectVGPModel instance (owns kernel, likelihood, X, X_tilde, state)
        r: Spike counts, shape (N,)
        n_iterations: Maximum number of EM iterations
        n_estep: Number of E-step Newton iterations per EM iteration
        n_fstep: Number of F-step LBFGS iterations
        n_mstep: Number of M-step LBFGS iterations
        lr_f: Learning rate for F-step
        lr_m: Learning rate for M-step
        print_every: Print progress every N iterations
        verbose: Print detailed debugging info
        use_analytical_mstep: Use analytical gradients for M-step
        capture_checkpoints: If True, capture state at each checkpoint for validation
        early_stop: Enable early stopping based on loss stability (default: True)
        stop_window: Number of iterations to look back for improvement (default: 20)
        stop_thresh: Minimum relative improvement over window to continue (default: 5e-3 = 0.5%)
        min_iterations: Minimum iterations before early stopping can trigger (default: 10)
        stability_threshold: Max mean firing rate before step rejection (default: 1000)

    Returns:
        Dict with:
            'losses': List of ELBO values per iteration
            'model': Trained DirectVGPModel
            'time_estep_total': Total E-step time (includes F-step)
            'time_mstep_total': Total M-step time
            'stopped_early': Whether training stopped early
            'final_iteration': Final iteration number
            'checkpoints': List of checkpoint dicts (only if capture_checkpoints=True)
    """
    from eigenspace_estep import estep_eigenspace
    from eigenspace_fstep import fstep_eigenspace
    from eigenspace_mstep import mstep_eigenspace_autograd, mstep_eigenspace_analytical
    from utils import compute_f_mean

    if capture_checkpoints:
        from investigations.test_restructuring_invariance.checkpoint_utils import capture_checkpoint

    state = model.state
    n_b = len(state.eigvals_b)
    M = model.X_tilde.shape[0]
    print(f"Eigenspace dimension: n_b={n_b} (from M={M} inducing points)")

    time_estep_total = 0.0
    time_mstep_total = 0.0
    losses = []
    checkpoints = [] if capture_checkpoints else None

    # Early stopping state
    stopped_early = False
    final_iteration = 0

    # Initial moments (GPyTorch-like: call model to get posterior)
    posterior = model(model.X_train)
    lambda_m, lambda_var = posterior.mean, posterior.variance
    A = model.likelihood.A.squeeze()
    lambda0 = model.likelihood.lambda0.squeeze()
    f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)

    for iteration in range(1, n_iterations):

        # ===== Kernel recomputation after M-step =====
        if n_mstep > 0 and iteration > 1:
            model.recompute_eigenspace()
            posterior = model(model.X_train)
            lambda_m, lambda_var = posterior.mean, posterior.variance
            A = model.likelihood.A.squeeze()
            lambda0 = model.likelihood.lambda0.squeeze()
            f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)

            if capture_checkpoints:
                checkpoints.append(capture_checkpoint(
                    'C1_eigenspace', iteration, model.state,
                    lambda_m=lambda_m, lambda_var=lambda_var, f_mean=f_mean
                ))

        # ===== E-step: Newton loop =====
        start_estep = time.time()

        for i_estep in range(n_estep):
            estep_eigenspace(model, r, f_mean)

            posterior = model(model.X_train)
            lambda_m, lambda_var = posterior.mean, posterior.variance
            f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)

            if capture_checkpoints:
                checkpoints.append(capture_checkpoint(
                    'C2_estep', iteration, model.state,
                    lambda_m=lambda_m, lambda_var=lambda_var, f_mean=f_mean,
                    estep_idx=i_estep
                ))

            if f_mean.mean().item() > stability_threshold:
                if verbose:
                    print(f"  E-step {i_estep}: f_mean unstable ({f_mean.mean().item():.1f})")
                break

        # ===== F-step: Optimize A =====
        fstep_eigenspace(model, r, lambda_m, lambda_var, n_fstep, lr_f,
                         stability_threshold=stability_threshold)

        A = model.likelihood.A.squeeze()
        lambda0 = model.likelihood.lambda0.squeeze()
        f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)

        if capture_checkpoints:
            checkpoints.append(capture_checkpoint(
                'C3_fstep', iteration, model.state,
                likelihood=model.likelihood,
                lambda_m=lambda_m, lambda_var=lambda_var, f_mean=f_mean
            ))

        time_estep_total += time.time() - start_estep

        # ===== M-step: Optimize kernel hyperparameters =====
        start_mstep = time.time()

        if n_mstep > 0 and iteration < n_iterations - 1:
            if use_analytical_mstep:
                mstep_eigenspace_analytical(model, r, n_mstep, lr_m,
                                            stability_threshold=stability_threshold,
                                            lambda_var_clamp=model.lambda_var_clamp)
            else:
                mstep_eigenspace_autograd(model, r, n_mstep, lr_m,
                                          stability_threshold=stability_threshold,
                                          lambda_var_clamp=model.lambda_var_clamp)

            if capture_checkpoints:
                checkpoints.append(capture_checkpoint(
                    'C4_mstep', iteration, model.state,
                    kernel=model.kernel
                ))

        time_mstep_total += time.time() - start_mstep

        # ===== Compute and record loss =====
        elbo = compute_elbo_eigenspace(model.state, r, lambda_m, lambda_var, A, lambda0)
        loss = -elbo.item()
        losses.append(loss)
        final_iteration = iteration

        if capture_checkpoints:
            checkpoints.append(capture_checkpoint(
                'C5_end', iteration, model.state,
                loss=loss
            ))

        if iteration % print_every == 0 or iteration == 1:
            print(f"Iter {iteration}/{n_iterations-1}: loss={loss:.2f}, "
                  f"A={A.item():.4f}, lambda0={lambda0.item():.4f}, "
                  f"n_b={len(model.state.eigvals_b)}")

        # Early stopping: check if loss improved enough over last stop_window iterations
        if early_stop and len(losses) >= stop_window + min_iterations:
            old_loss = losses[-(stop_window + 1)]
            rel_improvement = (old_loss - loss) / abs(old_loss)
            if rel_improvement < stop_thresh:
                stopped_early = True
                print(f"Early stopping at iteration {iteration}: "
                      f"loss improved only {rel_improvement*100:.3f}% over last {stop_window} iterations")
                break

    result = {
        'losses': losses,
        'model': model,
        'time_estep_total': time_estep_total,
        'time_mstep_total': time_mstep_total,
        'stopped_early': stopped_early,
        'final_iteration': final_iteration,
    }
    if capture_checkpoints:
        result['checkpoints'] = checkpoints
    return result


def predict_eigenspace(model, X_test: torch.Tensor) -> Dict:
    """Predict at test points using trained eigenspace variational GP.

    Computes posterior moments at test points and expected firing rates.

    Args:
        model: Trained DirectVGPModel
        X_test: Test inputs, shape (N_test, n_features)

    Returns:
        Dict with:
            'f_pred': Predicted firing rates, shape (N_test,)
            'lambda_m': Posterior mean at test points, shape (N_test,)
            'lambda_var': Posterior variance at test points, shape (N_test,)
    """
    kernel = model.kernel
    likelihood = model.likelihood
    state = model.state
    X_tilde = model.X_tilde

    with torch.no_grad():
        # Compute cross-kernel to inducing points
        K_test = kernel(X_test, X_tilde).to_dense()  # (N_test, M)
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
        lambda_var = torch.clamp(lambda_var, min=model.lambda_var_clamp)

        # Predicted firing rate
        A = likelihood.A.squeeze()
        lambda0 = likelihood.lambda0.squeeze()
        f_pred = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + lambda0)

    return {
        'f_pred': f_pred,
        'lambda_m': lambda_m,
        'lambda_var': lambda_var,
    }
