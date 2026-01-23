"""
F-Step Functions for GPyTorch Variational GP

Handles optimization of firing rate parameters (A, λ₀) while holding
variational parameters (m, V) and kernel hyperparameters fixed.

Key functions:
- f_step(): Adam-based optimization of A with analytical λ₀
- f_step_lbfgs(): LBFGS-based optimization matching original varGP
- lambda0_given_A(): Closed-form optimal λ₀ given A
"""

import torch
import gpytorch
from typing import Tuple


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


def f_step(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor,
    n_fstep: int,
    lr: float,  # Required - no default to prevent silent bugs
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
    with torch.no_grad():
        likelihood.lambda0.copy_(new_lambda0.unsqueeze(0))

    if n_fstep == 0:
        return

    optimizer = torch.optim.Adam([likelihood.raw_A], lr=lr)

    for _ in range(n_fstep):
        optimizer.zero_grad()

        # Update lambda0 analytically for current A
        A = likelihood.A.squeeze()
        with torch.no_grad():
            new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
            likelihood.lambda0.copy_(new_lambda0.unsqueeze(0))

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
        likelihood.lambda0.copy_(new_lambda0.unsqueeze(0))


def f_step_lbfgs(
    model: gpytorch.models.ApproximateGP,
    likelihood,
    X: torch.Tensor,
    r: torch.Tensor,
    lambda_m: torch.Tensor,
    lambda_var: torch.Tensor,
    n_fstep: int,
    lr: float,  # Required - no default to prevent silent bugs
    verbose: bool = False
):
    """F-step using LBFGS optimizer.

    Uses LBFGS with strong_wolfe line search to optimize A.
    - Optimizes likelihood.raw_A directly (raw_A = logA, A = exp(raw_A))
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
        lr: Learning rate for LBFGS
        verbose: Print debug info
    """
    if n_fstep == 0:
        return

    # Initial lambda0 update
    A = likelihood.A.squeeze()
    with torch.no_grad():
        new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
        likelihood.lambda0.copy_(new_lambda0.reshape(likelihood.lambda0.shape))

    # LBFGS optimizer - directly optimizes raw_A (which is logA)
    optimizer = torch.optim.LBFGS(
        [likelihood.raw_A],
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

        # Get A from likelihood (applies exp to raw_A)
        A = likelihood.A.squeeze()

        # Update lambda0 analytically
        with torch.no_grad():
            lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
            likelihood.lambda0.copy_(lambda0.reshape(likelihood.lambda0.shape))

        # Compute f_mean = exp(A*lambda_m + 0.5*A^2*lambda_var + lambda0)
        f_mean = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + lambda0)

        # Stability check
        if f_mean.mean() > 100 or torch.any(torch.isnan(f_mean)):
            if verbose:
                print(f"f_mean.mean() = {f_mean.mean():.1f} at closure call {closure_counter[0]}, returning inf")
            return torch.tensor(float('inf'), device=A.device, dtype=A.dtype)

        # Compute loglikelihood: L = A*r@lambda_m + lambda0*sum(r) - sum(f_mean)
        rlambda_m = r @ lambda_m
        sum_r = r.sum()
        loglikelihood = A * rlambda_m + lambda0 * sum_r - f_mean.sum()

        # Compute gradient w.r.t. raw_A (= logA)
        # dL/d(logA) = dL/dA * dA/d(logA) = dL/dA * A
        # dL/dA = r@lambda_m - (lambda_m + A*lambda_var) @ f_mean
        dL_dA = rlambda_m - torch.dot(lambda_m + A * lambda_var, f_mean)
        dL_dlogA = A * dL_dA

        # Set gradient (negative because LBFGS minimizes)
        likelihood.raw_A.grad = -dL_dlogA.reshape(likelihood.raw_A.shape)

        return -loglikelihood

    # Run LBFGS
    optimizer.step(closure)

    # Final lambda0 update
    with torch.no_grad():
        A = likelihood.A.squeeze()
        new_lambda0 = lambda0_given_A(A, r, lambda_m, lambda_var)
        likelihood.lambda0.copy_(new_lambda0.reshape(likelihood.lambda0.shape))

    if verbose:
        print(f"F-step LBFGS: {closure_counter[0]} closure calls, A: {likelihood.A.item():.4f}")
