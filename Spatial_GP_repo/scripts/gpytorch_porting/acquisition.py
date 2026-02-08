"""
Acquisition functions for active learning with variational GPs.

Implements:
- standard_utility: H_marg - H_noise (no conditioning, fast)
- distribution_aware_utility: H_marg - E[H_cond] (MC over p(x), accounts for
  cross-covariance between query and natural stimuli)

Currently supports default_gpy mode only (requires model(X).covariance_matrix).
vargp_direct support deferred (needs augmented matrix approach).

Fully gradient-compatible: All functions support gradient flow from x_candidates
through the kernel into the utility values. The caller controls whether gradients
are tracked (pass requires_grad=True tensors) or suppressed (wrap in
torch.no_grad()).

All dependencies are local (utils.py). No playground imports.
"""

import importlib.util
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Import from local utils.py via importlib to avoid sys.path/sys.modules
# shadowing by the repo-root utils.py (which gets cached when other modules
# import from the old codebase).
# ---------------------------------------------------------------------------
_local_utils_path = Path(__file__).resolve().parent / 'utils.py'
_spec = importlib.util.spec_from_file_location("gpytorch_porting_utils", str(_local_utils_path))
_local_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_utils)

get_gp_marginal_moments = _local_utils.get_gp_marginal_moments
get_gp_conditional_moments = _local_utils.get_gp_conditional_moments
compute_H = _local_utils.compute_H
nd_utility_new = _local_utils.nd_utility_new
compute_adaptive_rmax = _local_utils.compute_adaptive_rmax


def standard_utility(model, likelihood, x_candidates, r_max=100,
                     adaptive_r_max=False):
    """Compute standard (non-distribution-aware) utility at candidate points.

    U(x*) = H_marg(x*) - H_noise(x*)

    This measures the information gain from observing R at x*, without
    accounting for the natural stimulus distribution p(x).

    Works with any model that returns .mean and .variance (both default_gpy
    and vargp_direct).

    Args:
        model: Trained GP model. Must be in eval mode.
        likelihood: PoissonLikelihood with .A and .lambda0 attributes.
        x_candidates: (N,) or (N, d) query points to evaluate utility at.
        r_max: Max spike count for Laplace approximation truncation.
            Ignored when adaptive_r_max=True.
        adaptive_r_max: If True, compute r_max adaptively from GP moments
            to prevent entropy collapse at high mu_g (non-stationary kernels).

    Returns:
        dict with:
            'utility': (N,) utility values U(x*) = H_marg - H_noise
    """
    lambda_mean, lambda_var = get_gp_marginal_moments(model, x_candidates)

    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()

    # nd_utility_new expects log-firing rate moments g = A*lambda + lambda0
    mu_g = A * lambda_mean + lambda0
    sigma2_g = A ** 2 * lambda_var

    if adaptive_r_max:
        r_max = compute_adaptive_rmax(mu_g, sigma2_g)

    utility = nd_utility_new(mu_g, sigma2_g, r_max=r_max)

    return {'utility': utility}


def distribution_aware_utility(model, likelihood, x_candidates, x_samples,
                               r_max=100, sample_lambda=True,
                               adaptive_r_max=False):
    """Compute distribution-aware utility at candidate points.

    U(x*) = H_marg(x*) - E_{x~p(x), lambda~q}[H(R | x*, lambda(x), D)]

    Measures how much observing R at x* reduces uncertainty about the neuron's
    responses to natural stimuli x ~ p(x). The expectation is estimated by
    Monte Carlo over the pre-drawn x_samples.

    Currently requires default_gpy model (needs .covariance_matrix for
    Gaussian conditioning).

    Note: This function is fully differentiable w.r.t. x_candidates. The
    caller should wrap in torch.no_grad() for evaluation-only use, or pass
    x_candidates with requires_grad=True for gradient-based optimization.

    Args:
        model: Trained GP model in eval mode. Must support
            model(X).covariance_matrix (GPyTorch VariationalGP).
        likelihood: PoissonLikelihood with .A and .lambda0 attributes.
        x_candidates: (N_candidates,) or (N_candidates, d) query points.
        x_samples: (N_mc,) or (N_mc, d) pre-drawn samples from p(x).
            Caller decides how to sample (image pool subset, Gaussian, etc.).
        r_max: Max spike count for Laplace approximation truncation.
            Ignored when adaptive_r_max=True.
        sample_lambda: If True (default), sample lambda_i ~ N(mu_i, sigma2_i)
            at each x_i. If False, use posterior mean mu_i (deterministic,
            useful for sanity checks).
        adaptive_r_max: If True, compute r_max adaptively from GP moments.
            Separate adaptive r_max is computed for marginal entropy and
            for each MC conditional entropy step (different mu_g/sigma2_g).

    Returns:
        dict with:
            'utility': (N_candidates,) utility values H_marg - H_cond
            'H_marg': (N_candidates,) marginal entropy at each candidate
            'H_cond': (N_candidates,) average conditional entropy
    """
    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()

    # Step 1: Marginal entropy at all candidates
    mu_marg, sigma2_marg = get_gp_marginal_moments(model, x_candidates)

    if adaptive_r_max:
        mu_g_marg = A * mu_marg + lambda0
        sigma2_g_marg = A ** 2 * sigma2_marg
        r_max_marg = compute_adaptive_rmax(mu_g_marg, sigma2_g_marg)
    else:
        r_max_marg = r_max

    H_marg = compute_H(mu_marg, sigma2_marg, r_max=r_max_marg, a=A, lambda0=lambda0)

    # Step 2: Monte Carlo estimate of conditional entropy
    n_mc = x_samples.shape[0]
    H_cond_sum = torch.zeros_like(H_marg)

    for i in range(n_mc):
        x_i = x_samples[i]

        # GP posterior at x_i
        post_i = model(x_i.unsqueeze(0))
        mu_i = post_i.mean[0]
        sigma2_i = post_i.variance[0]

        # Lambda value: sample or use mean (keep as tensor for gradient flow)
        if sample_lambda:
            lambda_i = mu_i + sigma2_i.sqrt() * torch.randn(1, dtype=mu_i.dtype, device=mu_i.device)
        else:
            lambda_i = mu_i

        # Conditional moments at all candidates given lambda(x_i)
        mu_cond, sigma2_cond = get_gp_conditional_moments(
            model, x_candidates, x_i, lambda_i
        )

        # Adaptive r_max for this conditional step
        if adaptive_r_max:
            mu_g_cond = A * mu_cond + lambda0
            sigma2_g_cond = A ** 2 * sigma2_cond
            r_max_cond = compute_adaptive_rmax(mu_g_cond, sigma2_g_cond)
        else:
            r_max_cond = r_max

        # Conditional entropy
        H_cond_i = compute_H(mu_cond, sigma2_cond, r_max=r_max_cond, a=A, lambda0=lambda0)
        H_cond_sum += H_cond_i

        if (i + 1) % 100 == 0:
            print(f"  MC sample {i + 1}/{n_mc}")

    H_cond = H_cond_sum / n_mc

    return {
        'utility': H_marg - H_cond,
        'H_marg': H_marg,
        'H_cond': H_cond,
    }
