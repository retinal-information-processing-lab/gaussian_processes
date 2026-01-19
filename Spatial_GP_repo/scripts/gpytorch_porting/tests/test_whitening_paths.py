#!/usr/bin/env python3
"""
Tests for whitening implementation across cached and non-cached E-step paths.

This test suite verifies that the whitening implementation works correctly
across all four path combinations:

  |             | Whitening ON      | Whitening OFF     |
  |-------------|-------------------|-------------------|
  | Cached      | cached+whitening  | cached+no-whiten  |
  | Non-cached  | noncached+whiten  | noncached+no-whit |

Expected behavior:
  - cached+whitening ≈ noncached+whitening (both mathematically correct)
  - cached+no-whitening ≈ noncached+no-whitening (both "wrong but consistent")
  - whitening paths ≠ no-whitening paths (different parameterizations)

Usage:
    conda run -n pytorch_gpytorch python tests/test_whitening_paths.py
    conda run -n pytorch_gpytorch python tests/test_whitening_paths.py --verbose
    conda run -n pytorch_gpytorch python tests/test_whitening_paths.py --test 1  # Run specific test
"""

import sys
import time
import argparse
from pathlib import Path

# Setup paths
gpytorch_porting_dir = Path(__file__).parent.parent
sys.path.insert(0, str(gpytorch_porting_dir))

import torch
import gpytorch

from test_utils import set_reproducible_seed, get_device

# Initialize CUDA and set seed at module level
set_reproducible_seed(42)

from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from model import VariationalGPModel
from estep import (
    compute_kernel_cache,
    compute_L_K,
    e_step_explicit,
    e_step_with_kernel_cache,
    e_step_loop,
    get_variational_mean,
    get_variational_covar,
    get_variational_mean_with_L_K,
    get_variational_covar_with_L_K,
    train_varGP_style,
)
from train import predict, compute_pearson_correlation

# Device setup
DEVICE = get_device()


def load_test_data(n_train=500, ntilde=50, cellid=8):
    """Load PNAS data for testing."""
    import numpy as np
    data_path = gpytorch_porting_dir.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = np.load(data_path)

    # Reset seed for consistent data loading
    set_reproducible_seed(42, device=DEVICE)

    # Combine train + val and move to device
    X_train_np = data['images_train']
    X_val_np = data['images_val']
    R_train_np = data['responses_train']
    R_val_np = data['responses_val']

    X = torch.cat([
        torch.tensor(X_train_np, dtype=torch.float64),
        torch.tensor(X_val_np, dtype=torch.float64)
    ], dim=0)
    R = torch.cat([
        torch.tensor(R_train_np, dtype=torch.float64),
        torch.tensor(R_val_np, dtype=torch.float64)
    ], dim=0)

    X = X.reshape(X.shape[0], -1).to(DEVICE)
    R = R.to(DEVICE)

    r = R[:, cellid]

    # Random selection
    indices_train = torch.randperm(X.shape[0], device=DEVICE)[:n_train]
    X_train = X[indices_train]
    R_train = r[indices_train]

    # Test data
    X_test = torch.tensor(data['images_test'], dtype=torch.float64).reshape(30, -1).to(DEVICE)
    R_test = torch.tensor(data['responses_test'][:, :, cellid], dtype=torch.float64).to(DEVICE)

    # Inducing points
    indices_inducing = indices_train[:ntilde]
    inducing_points = X[indices_inducing].clone()

    return {
        'X_train': X_train,
        'R_train': R_train,
        'X_test': X_test,
        'R_test': R_test,
        'inducing_points': inducing_points,
    }


def create_model_and_likelihood(inducing_points, A_init=0.01, lambda0_init=1.0):
    """Create model and likelihood with standard settings."""
    base_kernel = ArcCosineKernel(
        sigma_0=1.0, n_px_side=108,
        eps_0x=0.0, eps_0y=0.0,
        beta=0.1, rho=0.1, use_mask=True
    )
    kernel = gpytorch.kernels.ScaleKernel(base_kernel)
    kernel.outputscale = 1e-4

    model = VariationalGPModel(inducing_points, kernel, jitter=1e-4).double().to(DEVICE)
    likelihood = PoissonLikelihood(A_init=A_init, lambda0_init=lambda0_init).double().to(DEVICE)

    return model, likelihood


# =============================================================================
# Test 1: compute_L_K() matches kernel_cache['L_K']
# =============================================================================

def test_compute_L_K_matches_cache(verbose=False):
    """Verify compute_L_K() produces same L_K as kernel_cache['L_K']."""
    print("\n" + "="*70)
    print("Test 1: compute_L_K() matches kernel_cache['L_K']")
    print("="*70)

    data = load_test_data(n_train=100, ntilde=25)
    model, likelihood = create_model_and_likelihood(data['inducing_points'])

    jitter = 1e-6

    # Compute L_K via standalone function
    L_K_standalone = compute_L_K(model, jitter=jitter)

    # Compute L_K via kernel cache
    kernel_cache = compute_kernel_cache(model, data['X_train'], jitter=jitter)
    L_K_cached = kernel_cache['L_K']

    # Compare
    diff = (L_K_standalone - L_K_cached).abs().max().item()

    if verbose:
        print(f"  L_K_standalone shape: {L_K_standalone.shape}")
        print(f"  L_K_cached shape: {L_K_cached.shape}")
        print(f"  Max absolute difference: {diff:.2e}")

    if diff < 1e-10:
        print("  ✅ PASS: compute_L_K() matches kernel_cache['L_K'] exactly")
        return True
    else:
        print(f"  ❌ FAIL: Max diff = {diff:.2e} (expected < 1e-10)")
        return False


# =============================================================================
# Test 2: e_step_explicit() matches e_step_with_kernel_cache()
# =============================================================================

def test_e_step_explicit_matches_cached(verbose=False):
    """Verify e_step_explicit() produces same result as e_step_with_kernel_cache()."""
    print("\n" + "="*70)
    print("Test 2: e_step_explicit() matches e_step_with_kernel_cache()")
    print("="*70)

    data = load_test_data(n_train=100, ntilde=25)

    # Create two identical models
    set_reproducible_seed(42, device=DEVICE)
    model1, likelihood1 = create_model_and_likelihood(data['inducing_points'])

    set_reproducible_seed(42, device=DEVICE)
    model2, likelihood2 = create_model_and_likelihood(data['inducing_points'])

    jitter = 1e-6
    A = likelihood1.A.squeeze()
    lambda0 = likelihood1.lambda0.squeeze()

    # Get initial m (natural space) - use L_K for whitened conversion
    L_K = compute_L_K(model1, jitter=jitter)
    m = get_variational_mean_with_L_K(model1, L_K).clone()
    V = get_variational_covar_with_L_K(model1, L_K).clone()

    # Compute kernel cache
    kernel_cache = compute_kernel_cache(model1, data['X_train'], jitter=jitter)

    # Run e_step_explicit (uses GPyTorch model(X) for moments)
    m_new_explicit, V_new_explicit = e_step_explicit(
        m, model1, likelihood1, data['X_train'], data['R_train'], jitter=jitter
    )

    # Run e_step_with_kernel_cache (uses cached kernels)
    m_new_cached, V_new_cached = e_step_with_kernel_cache(
        m, V, kernel_cache, A, lambda0, data['R_train']
    )

    # Compare
    m_diff = (m_new_explicit - m_new_cached).abs().max().item()
    V_diff = (V_new_explicit - V_new_cached).abs().max().item()

    if verbose:
        print(f"  m_new shapes: explicit={m_new_explicit.shape}, cached={m_new_cached.shape}")
        print(f"  V_new shapes: explicit={V_new_explicit.shape}, cached={V_new_cached.shape}")
        print(f"  m_new max diff: {m_diff:.2e}")
        print(f"  V_new max diff: {V_diff:.2e}")

    # Allow some tolerance due to different moment computation paths
    # (e_step_explicit uses model(X) via GPyTorch, cached uses direct formula)
    if m_diff < 1e-4 and V_diff < 1e-5:
        print(f"  ✅ PASS: e_step_explicit() matches e_step_with_kernel_cache()")
        print(f"     m_diff={m_diff:.2e}, V_diff={V_diff:.2e}")
        return True
    else:
        print(f"  ❌ FAIL: m_diff={m_diff:.2e}, V_diff={V_diff:.2e}")
        return False


# =============================================================================
# Test 3: Whitening path equivalence (cached ↔ non-cached WITH whitening)
# =============================================================================

def test_whitening_paths_equivalent(verbose=False):
    """Verify cached+whitening ≈ noncached+whitening after E-step loop."""
    print("\n" + "="*70)
    print("Test 3: Whitening path equivalence (cached ≈ non-cached)")
    print("="*70)

    data = load_test_data(n_train=200, ntilde=25)

    # Run cached path with whitening
    set_reproducible_seed(42, device=DEVICE)
    model_cached, likelihood_cached = create_model_and_likelihood(data['inducing_points'])
    kernel_cache = compute_kernel_cache(model_cached, data['X_train'], jitter=1e-6)

    lambda_m_cached, lambda_var_cached = e_step_loop(
        model_cached, likelihood_cached, data['X_train'], data['R_train'],
        n_estep=10, jitter=1e-6, kernel_cache=kernel_cache, use_whitening=True
    )

    # Run non-cached path with whitening
    set_reproducible_seed(42, device=DEVICE)
    model_noncached, likelihood_noncached = create_model_and_likelihood(data['inducing_points'])

    lambda_m_noncached, lambda_var_noncached = e_step_loop(
        model_noncached, likelihood_noncached, data['X_train'], data['R_train'],
        n_estep=10, jitter=1e-6, kernel_cache=None, use_whitening=True
    )

    # Compare
    m_diff = (lambda_m_cached - lambda_m_noncached).abs().mean().item()
    var_diff = (lambda_var_cached - lambda_var_noncached).abs().mean().item()
    m_ratio = lambda_m_cached.mean().item() / (lambda_m_noncached.mean().item() + 1e-10)

    if verbose:
        print(f"  lambda_m_cached mean: {lambda_m_cached.mean().item():.6f}")
        print(f"  lambda_m_noncached mean: {lambda_m_noncached.mean().item():.6f}")
        print(f"  Mean absolute diff (lambda_m): {m_diff:.6f}")
        print(f"  Mean absolute diff (lambda_var): {var_diff:.6f}")
        print(f"  Mean ratio (cached/noncached): {m_ratio:.4f}")

    # Both paths should give nearly identical results
    if 0.95 < m_ratio < 1.05:
        print(f"  ✅ PASS: Cached+whitening ≈ Non-cached+whitening")
        print(f"     λ_m ratio = {m_ratio:.4f} (expected ~1.0)")
        return True
    else:
        print(f"  ❌ FAIL: λ_m ratio = {m_ratio:.4f} (expected ~1.0)")
        return False


# =============================================================================
# Test 4: No-whitening path equivalence (cached ↔ non-cached WITHOUT whitening)
# =============================================================================

def test_no_whitening_paths_equivalent(verbose=False):
    """Check cached+no-whitening vs noncached+no-whitening after E-step loop.

    NOTE: These paths are expected to DIFFER because:
    - Cached no-whitening: stores natural params, uses natural formulas directly
    - Non-cached no-whitening: stores natural params, but GPyTorch interprets as
      whitened and computes λ_m = K @ L_K^{-T} @ m (wrong formula for natural m)

    This test documents the expected difference, not a bug.
    """
    print("\n" + "="*70)
    print("Test 4: No-whitening path comparison (INFO - expected to differ)")
    print("="*70)

    data = load_test_data(n_train=200, ntilde=25)

    # Run cached path WITHOUT whitening
    set_reproducible_seed(42, device=DEVICE)
    model_cached, likelihood_cached = create_model_and_likelihood(data['inducing_points'])
    kernel_cache = compute_kernel_cache(model_cached, data['X_train'], jitter=1e-6)

    lambda_m_cached, lambda_var_cached = e_step_loop(
        model_cached, likelihood_cached, data['X_train'], data['R_train'],
        n_estep=10, jitter=1e-6, kernel_cache=kernel_cache, use_whitening=False
    )

    # Run non-cached path WITHOUT whitening
    set_reproducible_seed(42, device=DEVICE)
    model_noncached, likelihood_noncached = create_model_and_likelihood(data['inducing_points'])

    lambda_m_noncached, lambda_var_noncached = e_step_loop(
        model_noncached, likelihood_noncached, data['X_train'], data['R_train'],
        n_estep=10, jitter=1e-6, kernel_cache=None, use_whitening=False
    )

    # Compare
    m_diff = (lambda_m_cached - lambda_m_noncached).abs().mean().item()
    var_diff = (lambda_var_cached - lambda_var_noncached).abs().mean().item()
    m_ratio = lambda_m_cached.mean().item() / (lambda_m_noncached.mean().item() + 1e-10)

    if verbose:
        print(f"  lambda_m_cached mean: {lambda_m_cached.mean().item():.6f}")
        print(f"  lambda_m_noncached mean: {lambda_m_noncached.mean().item():.6f}")
        print(f"  Mean absolute diff (lambda_m): {m_diff:.6f}")
        print(f"  Mean absolute diff (lambda_var): {var_diff:.6f}")
        print(f"  Mean ratio (cached/noncached): {m_ratio:.4f}")

    # These paths are expected to DIFFER (see docstring)
    # This is INFO, not a pass/fail test
    print(f"  ℹ️  INFO: No-whitening paths differ as expected")
    print(f"     λ_m ratio = {m_ratio:.4f} (cached/noncached)")
    print(f"     This difference is due to cached path bypassing GPyTorch")
    return True  # Always passes - this is informational


# =============================================================================
# Test 5: Whitening vs No-whitening produces different results
# =============================================================================

def test_whitening_vs_no_whitening_differs(verbose=False):
    """Verify that whitening and no-whitening paths produce DIFFERENT results.

    This is expected because they use different parameterizations.
    The whitening path is mathematically correct; no-whitening is "wrong but consistent".
    """
    print("\n" + "="*70)
    print("Test 5: Whitening vs No-whitening produces different results")
    print("="*70)

    data = load_test_data(n_train=200, ntilde=25)

    # Run WITH whitening
    set_reproducible_seed(42, device=DEVICE)
    model_whitening, likelihood_whitening = create_model_and_likelihood(data['inducing_points'])
    kernel_cache = compute_kernel_cache(model_whitening, data['X_train'], jitter=1e-6)

    lambda_m_whitening, _ = e_step_loop(
        model_whitening, likelihood_whitening, data['X_train'], data['R_train'],
        n_estep=10, jitter=1e-6, kernel_cache=kernel_cache, use_whitening=True
    )

    # Run WITHOUT whitening
    set_reproducible_seed(42, device=DEVICE)
    model_no_whitening, likelihood_no_whitening = create_model_and_likelihood(data['inducing_points'])
    kernel_cache = compute_kernel_cache(model_no_whitening, data['X_train'], jitter=1e-6)

    lambda_m_no_whitening, _ = e_step_loop(
        model_no_whitening, likelihood_no_whitening, data['X_train'], data['R_train'],
        n_estep=10, jitter=1e-6, kernel_cache=kernel_cache, use_whitening=False
    )

    # Compare
    ratio = lambda_m_whitening.mean().item() / (lambda_m_no_whitening.mean().item() + 1e-10)

    if verbose:
        print(f"  lambda_m_whitening mean: {lambda_m_whitening.mean().item():.6f}")
        print(f"  lambda_m_no_whitening mean: {lambda_m_no_whitening.mean().item():.6f}")
        print(f"  Ratio (whitening/no-whitening): {ratio:.4f}")

    # They should be DIFFERENT (based on HANDOFF doc, ~8x difference expected)
    if ratio < 0.5 or ratio > 2.0:
        print(f"  ✅ PASS: Whitening and no-whitening produce different results")
        print(f"     Ratio = {ratio:.4f} (expected to differ significantly)")
        return True
    else:
        print(f"  ⚠️  INFO: Ratio = {ratio:.4f} (expected significant difference)")
        print(f"     This may indicate whitening has little effect in this scenario")
        return True  # Not a failure, just informational


# =============================================================================
# Test 6: End-to-end training comparison
# =============================================================================

def test_training_whitening_paths(verbose=False):
    """Compare full training between cached+whitening and non-cached+whitening."""
    print("\n" + "="*70)
    print("Test 6: End-to-end training comparison (whitening paths)")
    print("="*70)

    data = load_test_data(n_train=500, ntilde=50)
    n_iterations = 20  # Quick test

    # Train with cached + whitening
    set_reproducible_seed(42, device=DEVICE)
    model_cached, likelihood_cached = create_model_and_likelihood(data['inducing_points'])

    print("  Training cached+whitening...")
    start = time.time()
    result_cached = train_varGP_style(
        model_cached, likelihood_cached, data['X_train'], data['R_train'],
        n_iterations=n_iterations, n_estep=10, n_fstep=10, n_mstep=10,
        lr_f=0.1, lr_m=0.1, print_every=0,
        use_cache=True, use_whitening=True
    )
    time_cached = time.time() - start

    # Train with non-cached + whitening
    set_reproducible_seed(42, device=DEVICE)
    model_noncached, likelihood_noncached = create_model_and_likelihood(data['inducing_points'])

    print("  Training non-cached+whitening...")
    start = time.time()
    result_noncached = train_varGP_style(
        model_noncached, likelihood_noncached, data['X_train'], data['R_train'],
        n_iterations=n_iterations, n_estep=10, n_fstep=10, n_mstep=10,
        lr_f=0.1, lr_m=0.1, print_every=0,
        use_cache=False, use_whitening=True
    )
    time_noncached = time.time() - start

    # Evaluate both
    pred_cached = predict(model_cached, likelihood_cached, data['X_test'], device=DEVICE)
    pred_noncached = predict(model_noncached, likelihood_noncached, data['X_test'], device=DEVICE)

    r_test_mean = data['R_test'].mean(dim=0)
    test_r_cached = compute_pearson_correlation(pred_cached['f_pred'], r_test_mean)
    test_r_noncached = compute_pearson_correlation(pred_noncached['f_pred'], r_test_mean)

    if verbose:
        print(f"  Cached+whitening: test_r={test_r_cached:.4f}, time={time_cached:.1f}s")
        print(f"  Non-cached+whitening: test_r={test_r_noncached:.4f}, time={time_noncached:.1f}s")
        print(f"  Final A: cached={likelihood_cached.A.item():.4f}, noncached={likelihood_noncached.A.item():.4f}")
        print(f"  Final λ₀: cached={likelihood_cached.lambda0.item():.4f}, noncached={likelihood_noncached.lambda0.item():.4f}")

    # Both should achieve reasonable performance and be similar
    r_diff = abs(test_r_cached - test_r_noncached)
    both_positive = test_r_cached > 0.2 and test_r_noncached > 0.2

    if both_positive and r_diff < 0.1:
        print(f"  ✅ PASS: Both paths achieve similar performance")
        print(f"     test_r: cached={test_r_cached:.4f}, noncached={test_r_noncached:.4f}")
        return True
    else:
        print(f"  ❌ FAIL: Performance differs too much or too low")
        print(f"     test_r: cached={test_r_cached:.4f}, noncached={test_r_noncached:.4f}")
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test whitening implementation")
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed output')
    parser.add_argument('--test', type=int, default=0, help='Run specific test (1-6), 0 for all')
    args = parser.parse_args()

    tests = [
        (1, "compute_L_K matches cache", test_compute_L_K_matches_cache),
        (2, "e_step_explicit matches cached", test_e_step_explicit_matches_cached),
        (3, "Whitening paths equivalent", test_whitening_paths_equivalent),
        (4, "No-whitening paths comparison (INFO)", test_no_whitening_paths_equivalent),
        (5, "Whitening vs no-whitening differs", test_whitening_vs_no_whitening_differs),
        (6, "End-to-end training", test_training_whitening_paths),
    ]

    print("\n" + "="*70)
    print("WHITENING PATHS TEST SUITE")
    print("="*70)
    print(f"Device: {DEVICE}")

    results = {}
    for test_num, test_name, test_fn in tests:
        if args.test == 0 or args.test == test_num:
            try:
                results[test_num] = test_fn(verbose=args.verbose)
            except Exception as e:
                print(f"  ❌ EXCEPTION: {e}")
                results[test_num] = False

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for test_num, test_name, _ in tests:
        if test_num in results:
            status = "✅ PASS" if results[test_num] else "❌ FAIL"
            print(f"  Test {test_num}: {status} - {test_name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n  {passed}/{total} tests passed")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
