"""
Tests for acquisition.py — utility functions for active learning.

Validates:
- standard_utility matches direct nd_utility_new call
- distribution_aware_utility produces correct results (compared against
  manual loop using the same imported building blocks)
- Determinism with sample_lambda=False
- Non-negativity of utility (mutual information >= 0)

Uses a fresh 1D GP (from the 1D playground) trained on synthetic Poisson
data as the test model. This is intentional: the imported functions
were developed and tested with playground-style GPyTorch models.

Run: python -m pytest tests/test_acquisition.py -v
"""

import importlib.util
import sys
import torch
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — save/restore sys.path around playground imports.
# (gp_utility_playground inserts repo root at sys.path[0], shadowing local utils.py)
# NOTE: We do NOT restore dtype here. The playground sets float64, and this test
# trains a playground model that needs float64. Restoring to float32 would cause
# dtype mismatches. The dtype restore is only needed in acquisition.py (production).
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
_scripts_dir = _repo_root / "scripts"
_gpytorch_dir = Path(__file__).resolve().parent.parent

_prev_path = sys.path.copy()

sys.path.append(str(_repo_root))
sys.path.append(str(_scripts_dir / "1D_playground"))
sys.path.append(str(_gpytorch_dir))

from gp_utility_playground import (
    VariationalGP, PoissonLikelihood as PlaygroundPoissonLikelihood,
    generate_poisson_data, train_gp, lambda_true,
    DEVICE, DTYPE,
)
from acquisition import standard_utility, distribution_aware_utility

sys.path = _prev_path

# ---------------------------------------------------------------------------
# Import gpytorch_porting's local utils via importlib (same pattern as
# acquisition.py) to avoid sys.modules collision with repo-root utils.py.
# ---------------------------------------------------------------------------
_local_utils_path = _gpytorch_dir / 'utils.py'
_spec = importlib.util.spec_from_file_location("gpytorch_porting_utils_test", str(_local_utils_path))
_local_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_utils)

get_gp_marginal_moments = _local_utils.get_gp_marginal_moments
get_gp_conditional_moments = _local_utils.get_gp_conditional_moments
compute_H = _local_utils.compute_H
nd_utility_new = _local_utils.nd_utility_new


# ---------------------------------------------------------------------------
# Mock likelihood for tests
# ---------------------------------------------------------------------------
class MockLikelihood:
    """Minimal likelihood with A and lambda0 for testing.

    The 1D playground's PoissonLikelihood has no A/lambda0 (it models
    f = exp(lambda) directly). Our acquisition functions expect a likelihood
    with .A and .lambda0. This mock bridges the gap for testing with A=1, lambda0=0.
    """

    def __init__(self, A=1.0, lambda0=0.0, device=DEVICE, dtype=DTYPE):
        self._A = torch.tensor([A], device=device, dtype=dtype)
        self.lambda0 = torch.nn.Parameter(
            torch.tensor([lambda0], device=device, dtype=dtype)
        )

    @property
    def A(self):
        return self._A


# ---------------------------------------------------------------------------
# Shared fixture: train a 1D GP once
# ---------------------------------------------------------------------------
_cached_model = None
_cached_likelihood = None


def get_trained_1d_model():
    """Train a 1D GP on synthetic Poisson data. Cached across tests."""
    global _cached_model, _cached_likelihood

    if _cached_model is not None:
        return _cached_model, _cached_likelihood

    torch.manual_seed(42)
    np.random.seed(42)

    # Training data
    train_x = torch.linspace(-2, 2, 50, dtype=DTYPE, device=DEVICE)
    train_y = generate_poisson_data(train_x, lambda_true)

    # Model with 20 inducing points
    inducing_points = torch.linspace(-2, 2, 20, dtype=DTYPE, device=DEVICE)
    model = VariationalGP(inducing_points).to(DEVICE)
    playground_likelihood = PlaygroundPoissonLikelihood().to(DEVICE)

    # Train
    train_gp(model, playground_likelihood, train_x, train_y,
             n_iterations=200, lr=0.1)
    model.eval()

    # Mock likelihood with A=1, lambda0=0 (matches playground's implicit params)
    mock_likelihood = MockLikelihood(A=1.0, lambda0=0.0)

    _cached_model = model
    _cached_likelihood = mock_likelihood
    return model, mock_likelihood


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_standard_utility():
    """standard_utility matches direct nd_utility_new call.

    With A=1, lambda0=0, the transform is identity:
      mu_g = 1.0 * lambda_mean + 0.0 = lambda_mean
      sigma2_g = 1.0^2 * lambda_var = lambda_var

    So standard_utility should give identical results to calling
    nd_utility_new(posterior.mean, posterior.variance) directly.
    """
    model, likelihood = get_trained_1d_model()
    x_candidates = torch.linspace(-2, 2, 30, dtype=DTYPE, device=DEVICE)

    # Our wrapper
    with torch.no_grad():
        result = standard_utility(model, likelihood, x_candidates, r_max=100)

    # Direct call
    with torch.no_grad():
        posterior = model(x_candidates)
        expected = nd_utility_new(posterior.mean, posterior.variance, r_max=100)

    # Tolerance: acquisition.py now uses a local Laplace implementation
    # (torch.log1p path) instead of utility.py:nd_utility_new (GP_utils.safe_log
    # path). Both compute the same math but via different float operations.
    # Max observed diff: 4.4e-16 (double-precision machine epsilon).
    torch.testing.assert_close(result['utility'], expected, atol=1e-14, rtol=1e-13)
    print("PASSED: test_standard_utility — matches direct nd_utility_new within 1e-14")


def test_distribution_aware_utility_matches_manual():
    """distribution_aware_utility matches a manual loop using the same functions.

    This is the most important test. We manually replicate the MC loop
    using the same building blocks from gpytorch_porting/utils.py
    (compute_H, get_gp_marginal_moments, get_gp_conditional_moments)
    and verify exact match.

    Uses sample_lambda=False for determinism.
    """
    model, likelihood = get_trained_1d_model()
    x_candidates = torch.linspace(-2, 2, 20, dtype=DTYPE, device=DEVICE)

    # Pre-draw x_samples (deterministic)
    torch.manual_seed(123)
    x_samples = 0.0 + 0.5 * torch.randn(50, dtype=DTYPE, device=DEVICE)

    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()

    # --- Our function ---
    with torch.no_grad():
        result = distribution_aware_utility(
            model, likelihood, x_candidates, x_samples,
            r_max=75, sample_lambda=False,
        )

    # --- Manual computation (using same functions as acquisition.py) ---
    with torch.no_grad():
        # H_marg
        mu_marg, sigma2_marg = get_gp_marginal_moments(model, x_candidates)
        H_marg_manual = compute_H(mu_marg, sigma2_marg, r_max=75, a=A, lambda0=lambda0)

        # MC loop
        H_cond_sum = torch.zeros_like(H_marg_manual)
        for i in range(len(x_samples)):
            x_i = x_samples[i]
            post_i = model(x_i.unsqueeze(0))
            mu_i = post_i.mean[0]
            lambda_i = mu_i  # sample_lambda=False -> use mean (keep as tensor)

            mu_cond, sigma2_cond = get_gp_conditional_moments(
                model, x_candidates, x_i, lambda_i
            )
            H_cond_i = compute_H(mu_cond, sigma2_cond, r_max=75, a=A, lambda0=lambda0)
            H_cond_sum += H_cond_i

        H_cond_manual = H_cond_sum / len(x_samples)
        utility_manual = H_marg_manual - H_cond_manual

    # Exact match expected (same functions, same data, deterministic)
    torch.testing.assert_close(result['utility'], utility_manual, atol=0, rtol=0)
    torch.testing.assert_close(result['H_marg'], H_marg_manual, atol=0, rtol=0)
    torch.testing.assert_close(result['H_cond'], H_cond_manual, atol=0, rtol=0)
    print("PASSED: test_distribution_aware_utility_matches_manual — exact match")


def test_sample_lambda_deterministic():
    """With sample_lambda=False, repeated calls give identical results."""
    model, likelihood = get_trained_1d_model()
    x_candidates = torch.linspace(-1, 1, 10, dtype=DTYPE, device=DEVICE)
    x_samples = torch.linspace(-0.5, 0.5, 20, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        result1 = distribution_aware_utility(
            model, likelihood, x_candidates, x_samples,
            r_max=50, sample_lambda=False,
        )
        result2 = distribution_aware_utility(
            model, likelihood, x_candidates, x_samples,
            r_max=50, sample_lambda=False,
        )

    torch.testing.assert_close(result1['utility'], result2['utility'], atol=0, rtol=0)
    print("PASSED: test_sample_lambda_deterministic — repeated calls match exactly")


def test_sample_lambda_stochastic():
    """With sample_lambda=True, different seeds give different results."""
    model, likelihood = get_trained_1d_model()
    x_candidates = torch.linspace(-1, 1, 10, dtype=DTYPE, device=DEVICE)
    x_samples = torch.linspace(-0.5, 0.5, 30, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        torch.manual_seed(100)
        result1 = distribution_aware_utility(
            model, likelihood, x_candidates, x_samples,
            r_max=50, sample_lambda=True,
        )
        torch.manual_seed(200)
        result2 = distribution_aware_utility(
            model, likelihood, x_candidates, x_samples,
            r_max=50, sample_lambda=True,
        )

    # Results should differ (different lambda samples)
    assert not torch.allclose(result1['utility'], result2['utility']), \
        "Stochastic results should differ with different seeds"
    print("PASSED: test_sample_lambda_stochastic — different seeds give different results")


def test_utility_non_negative():
    """Utility (mutual information) must be non-negative.

    Also checks: H_cond <= H_marg, no NaN/Inf.

    Note: MC estimation with finite samples can produce slightly negative
    utility values. We allow a small tolerance (-0.01) to account for this.
    Theoretical utility (mutual information) is always >= 0.
    """
    model, likelihood = get_trained_1d_model()
    x_candidates = torch.linspace(-2, 2, 30, dtype=DTYPE, device=DEVICE)

    torch.manual_seed(42)
    x_samples = 0.0 + 0.5 * torch.randn(100, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        result = distribution_aware_utility(
            model, likelihood, x_candidates, x_samples,
            r_max=75, sample_lambda=True,
        )

    utility = result['utility']
    H_marg = result['H_marg']
    H_cond = result['H_cond']

    # No NaN or Inf
    assert not torch.any(torch.isnan(utility)), f"NaN in utility"
    assert not torch.any(torch.isinf(utility)), f"Inf in utility"
    assert not torch.any(torch.isnan(H_marg)), f"NaN in H_marg"
    assert not torch.any(torch.isnan(H_cond)), f"NaN in H_cond"

    # Non-negative (MC tolerance: finite samples can give slightly negative values)
    min_utility = utility.min().item()
    MC_TOL = -0.01  # MC noise tolerance
    assert min_utility >= MC_TOL, f"Utility should be ~non-negative, got min={min_utility}"

    # H_cond <= H_marg (with same MC tolerance)
    diff = H_cond - H_marg
    max_excess = diff.max().item()
    assert max_excess <= abs(MC_TOL), f"H_cond should be <= H_marg, got max excess={max_excess}"

    print(f"PASSED: test_utility_non_negative")
    print(f"  Utility range: [{utility.min().item():.4f}, {utility.max().item():.4f}]")
    print(f"  H_marg range: [{H_marg.min().item():.4f}, {H_marg.max().item():.4f}]")
    print(f"  H_cond range: [{H_cond.min().item():.4f}, {H_cond.max().item():.4f}]")


def test_standard_utility_non_negative():
    """Standard utility should also be non-negative."""
    model, likelihood = get_trained_1d_model()
    x_candidates = torch.linspace(-2, 2, 30, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        result = standard_utility(model, likelihood, x_candidates, r_max=100)

    utility = result['utility']
    assert not torch.any(torch.isnan(utility)), "NaN in standard utility"
    assert not torch.any(torch.isinf(utility)), "Inf in standard utility"

    min_utility = utility.min().item()
    assert min_utility >= -1e-6, f"Standard utility should be non-negative, got min={min_utility}"

    print(f"PASSED: test_standard_utility_non_negative")
    print(f"  Utility range: [{utility.min().item():.4f}, {utility.max().item():.4f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Acquisition function tests")
    print("=" * 70)

    tests = [
        test_standard_utility,
        test_distribution_aware_utility_matches_manual,
        test_sample_lambda_deterministic,
        test_sample_lambda_stochastic,
        test_utility_non_negative,
        test_standard_utility_non_negative,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        print(f"\n--- {test_fn.__name__} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAILED: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'=' * 70}")
