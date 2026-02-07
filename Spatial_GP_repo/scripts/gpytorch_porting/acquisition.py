"""
Acquisition functions for active learning with variational GPs.

Implements:
- standard_utility: H_marg - H_noise (no conditioning, fast)
- distribution_aware_utility: H_marg - E[H_cond] (MC over p(x), accounts for
  cross-covariance between query and natural stimuli)

Currently supports default_gpy mode only (requires model(X).covariance_matrix).
vargp_direct support deferred (needs augmented matrix approach).

Dependencies (imported, not modified):
- 1D_playground/gp_utility_playground.py: compute_H, get_marginal_moments
- 2D_playground/utility_2d_rbf_base.py: get_conditional_moments_nd
- Spatial_GP_repo/utility.py: nd_utility_new (via 1D playground sys.path)
"""

import sys
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup for imports from existing codebase
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent.parent
# => .../Spatial_GP_repo
_scripts_dir = _repo_root / "scripts"

# Guard against side effects from playground imports:
# - gp_utility_playground sets torch.set_default_dtype(float64)
# - gp_utility_playground inserts Spatial_GP_repo root into sys.path[0],
#   which shadows local utils.py with the repo-root utils.py
_prev_path = sys.path.copy()
_prev_dtype = torch.get_default_dtype()

sys.path.append(str(_repo_root))
sys.path.append(str(_scripts_dir / "1D_playground"))
sys.path.append(str(_scripts_dir / "2D_playground"))

from gp_utility_playground import compute_H, get_marginal_moments
from utility_2d_rbf_base import get_conditional_moments_nd
from utility import nd_utility_new

sys.path = _prev_path
torch.set_default_dtype(_prev_dtype)


def standard_utility(model, likelihood, x_candidates, r_max=100):
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

    Returns:
        dict with:
            'utility': (N,) utility values U(x*) = H_marg - H_noise
    """
    lambda_mean, lambda_var = get_marginal_moments(model, x_candidates)

    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()

    # nd_utility_new expects log-firing rate moments g = A*lambda + lambda0
    mu_g = A * lambda_mean + lambda0
    sigma2_g = A ** 2 * lambda_var

    utility = nd_utility_new(mu_g, sigma2_g, r_max=r_max)

    return {'utility': utility}


def distribution_aware_utility(model, likelihood, x_candidates, x_samples,
                               r_max=100, sample_lambda=True):
    """Compute distribution-aware utility at candidate points.

    U(x*) = H_marg(x*) - E_{x~p(x), lambda~q}[H(R | x*, lambda(x), D)]

    Measures how much observing R at x* reduces uncertainty about the neuron's
    responses to natural stimuli x ~ p(x). The expectation is estimated by
    Monte Carlo over the pre-drawn x_samples.

    Currently requires default_gpy model (needs .covariance_matrix for
    Gaussian conditioning).

    Note: This function does NOT wrap its body in torch.no_grad(). The caller
    should wrap externally for evaluation. This keeps the door open for
    gradient-based x* optimization in future sessions.

    Args:
        model: Trained GP model in eval mode. Must support
            model(X).covariance_matrix (GPyTorch VariationalGP).
        likelihood: PoissonLikelihood with .A and .lambda0 attributes.
        x_candidates: (N_candidates,) or (N_candidates, d) query points.
        x_samples: (N_mc,) or (N_mc, d) pre-drawn samples from p(x).
            Caller decides how to sample (image pool subset, Gaussian, etc.).
        r_max: Max spike count for Laplace approximation truncation.
        sample_lambda: If True (default), sample lambda_i ~ N(mu_i, sigma2_i)
            at each x_i. If False, use posterior mean mu_i (deterministic,
            useful for sanity checks).

    Returns:
        dict with:
            'utility': (N_candidates,) utility values H_marg - H_cond
            'H_marg': (N_candidates,) marginal entropy at each candidate
            'H_cond': (N_candidates,) average conditional entropy
    """
    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()

    # Step 1: Marginal entropy at all candidates
    mu_marg, sigma2_marg = get_marginal_moments(model, x_candidates)
    H_marg = compute_H(mu_marg, sigma2_marg, r_max=r_max, a=A, lambda0=lambda0)

    # Step 2: Monte Carlo estimate of conditional entropy
    n_mc = x_samples.shape[0]
    H_cond_sum = torch.zeros_like(H_marg)

    for i in range(n_mc):
        x_i = x_samples[i]

        # GP posterior at x_i
        post_i = model(x_i.unsqueeze(0))
        mu_i = post_i.mean[0]
        sigma2_i = post_i.variance[0]

        # Lambda value: sample or use mean
        if sample_lambda:
            lambda_i = (mu_i + sigma2_i.sqrt() * torch.randn(1, dtype=mu_i.dtype, device=mu_i.device)).item()
        else:
            lambda_i = mu_i.item()

        # Conditional moments at all candidates given lambda(x_i)
        mu_cond, sigma2_cond = get_conditional_moments_nd(
            model, x_candidates, x_i, lambda_i
        )

        # Conditional entropy
        H_cond_i = compute_H(mu_cond, sigma2_cond, r_max=r_max, a=A, lambda0=lambda0)
        H_cond_sum += H_cond_i

        if (i + 1) % 100 == 0:
            print(f"  MC sample {i + 1}/{n_mc}")

    H_cond = H_cond_sum / n_mc

    return {
        'utility': H_marg - H_cond,
        'H_marg': H_marg,
        'H_cond': H_cond,
    }
