"""
F-Step for Eigenspace Variational GP

Handles optimization of firing rate parameter A while holding variational
parameters (m_b, V_b) and kernel hyperparameters fixed. lambda0 is computed
analytically given A.

Key functions:
- fstep_eigenspace: LBFGS optimization of A with analytical lambda0 (once per EM cycle)
- damped_newton_update_A_lambda0: Joint damped Newton on (A, lambda0), called per E-step
  iteration when interleave_fstep=True (matches paper's updateA)

Extracted from fstep.py during codebase reorganization (2025-02).
"""

import torch

# Import shared utilities
from utils import lambda0_given_A, compute_f_mean, log_normal_prior_A_terms


def fstep_eigenspace(
    model,
    r: torch.Tensor,
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor,
    n_fstep: int,
    lr: float,
    f_mean_max_threshold: float = 500,
    f_mean_mean_threshold: float = 100,
    A_prior_enabled: bool = False,
    A_prior_mu: float = -3.0,
    A_prior_sigma: float = 1.0,
):
    """F-step for eigenspace mode: Optimize A with LBFGS, lambda0 computed analytically.

    This is a simplified version that directly uses LBFGS on raw_A.

    When A_prior_enabled is True, maximizes the log-posterior
    log_lik(A) - 0.5 * (log A - A_prior_mu)^2 / A_prior_sigma^2 instead of
    pure log-likelihood. The prior gradient is added to the analytical
    gradient assigned to raw_A.grad. lambda0 update is unaffected — the
    prior is on A only.

    Args:
        model: DirectVGPModel instance
        r: Spike counts, shape (N,)
        lambda_m: Posterior mean (held fixed), shape (N,)
        lambda_var: Posterior variance (held fixed), shape (N,)
        n_fstep: Number of LBFGS iterations
        lr: Learning rate for LBFGS
        A_prior_enabled: If True, add log-normal prior penalty on A.
        A_prior_mu, A_prior_sigma: log-normal prior hyperparameters.
    """
    likelihood = model.likelihood

    if n_fstep == 0:
        # Still update lambda0 analytically
        A = likelihood.A.squeeze()
        with torch.no_grad():
            new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
            likelihood.lambda0.copy_(new_lambda0.reshape(likelihood.lambda0.shape))
        return

    # Initial lambda0 update
    A = likelihood.A.squeeze()
    with torch.no_grad():
        new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
        likelihood.lambda0.copy_(new_lambda0.reshape(likelihood.lambda0.shape))

    optimizer = torch.optim.LBFGS(
        [likelihood.raw_A],
        lr=lr,
        max_iter=n_fstep,
        tolerance_change=1e-9,
        tolerance_grad=1e-7,
        history_size=n_fstep,
        line_search_fn='strong_wolfe'
    )

    def closure():
        optimizer.zero_grad()

        # Reject trial step if likelihood parameters are out of bounds
        if not likelihood.params_in_bounds():
            return torch.tensor(float('inf'), device=likelihood.raw_A.device,
                                dtype=likelihood.raw_A.dtype)

        A = likelihood.A.squeeze()

        # Update lambda0 analytically
        with torch.no_grad():
            lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
            likelihood.lambda0.copy_(lambda0.reshape(likelihood.lambda0.shape))

        # Reject if lambda0 overflowed (A too large for current lambda_m/lambda_var)
        if torch.isinf(lambda0) or torch.isnan(lambda0):
            return torch.tensor(float('inf'), device=A.device, dtype=A.dtype)

        # Compute f_mean
        f_mean = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + lambda0)

        # Stability check: mean threshold catches global blowup,
        # max threshold catches localized blowup, NaN catches overflow
        if (f_mean.mean().item() > f_mean_mean_threshold
                or f_mean.max().item() > f_mean_max_threshold
                or torch.any(torch.isnan(f_mean))):
            return torch.tensor(float('inf'), device=A.device, dtype=A.dtype)

        # Log-likelihood (negative for minimization)
        log_lik = (r * (A * lambda_m + lambda0) - f_mean).sum()

        # Compute gradient analytically for efficiency
        # dL/dA = r @ lambda_m - (lambda_m + A * lambda_var) @ f_mean
        # dL/d(logA) = A * dL_dA
        dL_dA = r @ lambda_m - torch.dot(lambda_m + A * lambda_var, f_mean)
        dL_dlogA = A * dL_dA

        # MAP loss = -log_lik + prior_penalty.
        # Gradient w.r.t. raw_A (= log A):
        #   d(-log_lik)/d(raw_A)        = -dL_dlogA
        #   d(prior_penalty)/d(log A)   = (log A - mu) / sigma^2   (by chain rule on the
        #                                A-space gradient: A * (log A - mu)/(A * sigma^2))
        loss = -log_lik
        if A_prior_enabled:
            penalty, _, _ = log_normal_prior_A_terms(
                A, A_prior_mu, A_prior_sigma
            )
            loss = loss + penalty
            d_penalty_d_logA = (torch.log(A) - A_prior_mu) / (A_prior_sigma ** 2)
            likelihood.raw_A.grad = (-dL_dlogA + d_penalty_d_logA).reshape(
                likelihood.raw_A.shape
            )
        else:
            likelihood.raw_A.grad = -dL_dlogA.reshape(likelihood.raw_A.shape)

        return loss

    # Save pre-step state for revert on divergence
    raw_A_prev = likelihood.raw_A.detach().clone()
    lambda0_prev = likelihood.lambda0.detach().clone()

    try:
        optimizer.step(closure)
    except (IndexError, RuntimeError) as e:
        import warnings
        warnings.warn(f"F-step LBFGS crashed: {e}. Keeping pre-step parameters.")

    # Clamp likelihood parameters to valid bounds after step
    likelihood.clamp_params()

    # Final lambda0 update — check for overflow and revert if needed
    with torch.no_grad():
        A = likelihood.A.squeeze()
        new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
        if torch.isinf(new_lambda0) or torch.isnan(new_lambda0):
            # A is too large for the current posterior — revert to pre-step values
            likelihood.raw_A.copy_(raw_A_prev)
            likelihood.lambda0.copy_(lambda0_prev)
            print(f"  F-step: lambda0 overflowed at A={A.item():.4f}, reverted")
        else:
            likelihood.lambda0.copy_(new_lambda0.reshape(likelihood.lambda0.shape))


def damped_newton_update_A_lambda0(
    model,
    r: torch.Tensor,
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor,
    alpha: float = 0.25,
    max_iter: int = 100,
    tol: float = 1e-6,
    f_mean_max_threshold: float = 500,
    f_mean_mean_threshold: float = 100,
    A_prior_enabled: bool = False,
    A_prior_mu: float = -3.0,
    A_prior_sigma: float = 1.0,
) -> torch.Tensor:
    """Damped Newton update for A and lambda0 (matches paper's updateA).

    Jointly optimizes (A, lambda0) by iterating a damped Newton step:
        psi_new = psi - alpha * solve(H, g)
    where g is the gradient and H the Hessian of the expected log-likelihood
    (or log-posterior, when A_prior_enabled) w.r.t. psi = [A, lambda0].

    Called at every E-step Newton iteration when interleave_fstep=True,
    keeping A/lambda0 in sync with variational parameters (m, V).

    Math:
        f_i = exp(A * mu_i + 0.5 * A^2 * var_i + lambda0)
        d_i = mu_i + A * var_i

        Gradient (log-lik): R = [r @ mu - d @ f,  sum(r) - sum(f)]
        Hessian  (log-lik): H = -[[var @ f + d^2 @ f, d @ f], [d @ f, sum(f)]]
        Update: psi -= alpha * solve(H, R)

    When A_prior_enabled, augment R[0] and H[0,0] with the log-normal
    prior contribution (Fisher-information curvature for the Hessian to
    stay PSD-safe; see utils.log_normal_prior_A_terms).

    Args:
        model: DirectVGPModel instance
        r: Spike counts, shape (N,)
        lambda_m: Posterior mean, shape (N,)
        lambda_var: Posterior variance, shape (N,)
        alpha: Damping factor (default 0.25, matching paper)
        max_iter: Maximum Newton iterations (default 100)
        tol: Convergence tolerance on sum(|gradient|) (default 1e-6)
        f_mean_max_threshold: Max f_mean.max() before rejecting step
        f_mean_mean_threshold: Max f_mean.mean() before rejecting step
        A_prior_enabled: If True, add log-normal prior on A.
        A_prior_mu, A_prior_sigma: log-normal prior hyperparameters.

    Returns:
        f_mean: Updated expected firing rate, shape (N,)
    """
    likelihood = model.likelihood

    with torch.no_grad():
        A_val = likelihood.A.squeeze().clone()
        lambda0_val = likelihood.lambda0.squeeze().clone()

        sum_r = r.sum()
        r_dot_mu = r @ lambda_m

        for it in range(max_iter):
            # Predicted firing rate (clamp exponent for float32 safety)
            linear = A_val * lambda_m + 0.5 * A_val * A_val * lambda_var + lambda0_val
            f_mean = torch.exp(linear.clamp(max=80.0))

            # Derivative factor: d(exponent)/dA = mu + A*var
            d_exp = lambda_m + A_val * lambda_var
            f_star = d_exp * f_mean

            # Gradient of log-likelihood w.r.t. [A, lambda0]
            R0 = r_dot_mu - f_star.sum()
            R1 = sum_r - f_mean.sum()

            # Hessian (negative definite)
            H00 = -(lambda_var @ f_mean + d_exp @ f_star)
            H01 = -f_star.sum()
            H11 = -f_mean.sum()

            # Add prior penalty contribution (subtract from objective being
            # maximized: log_post = log_lik - penalty). The Fisher curvature
            # 1/(A^2 * sigma^2) is positive, so subtracting it from H00 (which
            # is already <= 0) keeps H negative-definite. The exact 2nd
            # derivative in A-space is sign-indefinite, hence the Fisher
            # approximation — see utils.log_normal_prior_A_terms.
            if A_prior_enabled:
                _, d_penalty_dA, h_fisher = log_normal_prior_A_terms(
                    A_val, A_prior_mu, A_prior_sigma
                )
                R0 = R0 - d_penalty_dA
                H00 = H00 - h_fisher

            R = torch.stack([R0, R1])

            # Convergence check (gradient of log-posterior, not log-lik)
            if R.abs().sum().item() < tol:
                break

            H = torch.stack([
                torch.stack([H00, H01]),
                torch.stack([H01, H11]),
            ])

            # Newton direction
            try:
                delta = torch.linalg.solve(H, R)
            except RuntimeError:
                break  # singular Hessian

            # Damped step
            A_new = A_val - alpha * delta[0]
            lambda0_new = lambda0_val - alpha * delta[1]

            # Clamp to valid range
            A_new = A_new.clamp(min=1e-10, max=likelihood.A_MAX)
            lambda0_new = lambda0_new.clamp(min=likelihood.LAMBDA0_MIN,
                                            max=likelihood.LAMBDA0_MAX)

            # Stability check on trial step
            trial_linear = A_new * lambda_m + 0.5 * A_new * A_new * lambda_var + lambda0_new
            if trial_linear.max().item() > 80.0:
                break
            trial_f = torch.exp(trial_linear)
            if (trial_f.mean().item() > f_mean_mean_threshold
                    or trial_f.max().item() > f_mean_max_threshold
                    or torch.any(torch.isnan(trial_f))):
                break

            # Accept step
            A_val = A_new
            lambda0_val = lambda0_new

        # Write back to model
        likelihood.raw_A.copy_(
            torch.log(A_val.clamp(min=1e-10)).reshape(likelihood.raw_A.shape)
        )
        likelihood.lambda0.copy_(
            lambda0_val.reshape(likelihood.lambda0.shape)
        )

    return compute_f_mean(lambda_m, lambda_var, A_val, lambda0_val)
