#!/usr/bin/env python3
"""
Hypothesis Tests for GPyTorch Whitening Understanding

These tests verify our understanding of the L_K mismatch problem in whitened
parameterization. Each test has a clear hypothesis and expected outcome - if
the test passes, our understanding is confirmed.

Context: When EM optimization changes kernel params (M-step), L_K changes but
the stored m was whitened with the OLD L_K. This creates a mismatch.

Tests:
  - Test A: Explicit corruption formula matches (math verification)
  - Test B: Frozen kernel training works (isolation test)
  - Test C: Single gradient step causes corruption (immediate effect)
  - Test D: Corruption accumulates over M-step (severity)

Created: 2026-01-19 by Claude
Run: conda run -n pytorch_gpytorch python tests/test_whitening_hypothesis.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

from test_utils import set_reproducible_seed, get_device
from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from model import VariationalGPModel
from estep import (
    compute_kernel_cache,
    compute_moments_from_kernel_cache,
    get_variational_mean_with_L_K,
    update_variational_mean_with_L_K,
    get_variational_covar_with_L_K,
    update_variational_covar_with_L_K,
    clear_variational_cache,
    train_varGP_style,
)
import gpytorch

# Device setup
DEVICE = get_device()
DATA_PATH = Path(__file__).parent.parent.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'


def load_pnas_data(n_train=200, cellid=8):
    """Load PNAS data for testing."""
    data = np.load(DATA_PATH)

    set_reproducible_seed(42, device=DEVICE)

    X_train = torch.tensor(data['images_train'][:n_train], dtype=torch.float64, device=DEVICE)
    X_train = X_train.reshape(X_train.shape[0], -1)
    r_train = torch.tensor(data['responses_train'][:n_train, cellid], dtype=torch.float64, device=DEVICE)

    return X_train, r_train


def create_model_and_likelihood(X_train, ntilde=50):
    """Create model with standard settings."""
    set_reproducible_seed(42, device=DEVICE)
    indices = torch.randperm(X_train.shape[0], device=DEVICE)[:ntilde]
    inducing_points = X_train[indices].clone()

    # ArcCosineKernel now has internal Amp parameter (matches legacy varGP)
    # No need for ScaleKernel wrapper
    kernel = ArcCosineKernel(
        sigma_0=1.0, Amp=1e-4, n_px_side=108,
        eps_0x=0.0, eps_0y=0.0,
        beta=0.1, rho=0.1, use_mask=True
    )

    model = VariationalGPModel(inducing_points, kernel, jitter=1e-6).double().to(DEVICE)
    likelihood = PoissonLikelihood(A_init=0.01, lambda0_init=1.0).double().to(DEVICE)

    return model, likelihood


# =============================================================================
# TEST A: Explicit Corruption Formula
# =============================================================================

def test_A_explicit_corruption_formula():
    """
    HYPOTHESIS: When reading m with a different L_K, the corruption follows
    the exact formula: m_corrupted = L_K_new @ L_K_old^{-1} @ m_natural

    EXPECTED: m_corrupted == m_expected (exact match within numerical precision)
    """
    print("\n" + "="*70)
    print("TEST A: Explicit Corruption Formula")
    print("="*70)
    print("HYPOTHESIS: Corruption follows m_corrupted = L_K_new @ L_K_old^{-1} @ m_natural")

    X_train, r_train = load_pnas_data(n_train=200)
    model, likelihood = create_model_and_likelihood(X_train, ntilde=50)
    M = 50

    # 1. Get initial L_K
    kernel_cache_old = compute_kernel_cache(model, X_train)
    L_K_old = kernel_cache_old['L_K'].clone()

    # 2. Store a known m_natural using L_K_old whitening
    set_reproducible_seed(42, device=DEVICE)
    m_natural = torch.randn(M, dtype=torch.float64, device=DEVICE) * 0.01
    update_variational_mean_with_L_K(model, m_natural, L_K_old)

    # Verify storage was correct (round-trip)
    m_recovered = get_variational_mean_with_L_K(model, L_K_old)
    roundtrip_ok = torch.allclose(m_natural, m_recovered, rtol=1e-6)
    print(f"  Round-trip (sanity check): {'OK' if roundtrip_ok else 'FAILED'}")

    # 3. Get what was actually stored (whitened m)
    m_stored = model.variational_strategy._variational_distribution.variational_mean.clone()

    # 4. Manually perturb kernel params (simulating M-step)
    # Change sigma_0 by adding to raw_sigma_0
    old_sigma_0 = model.covar_module.base_kernel.sigma_0.item()
    model.covar_module.base_kernel.raw_sigma_0.data += 0.5
    new_sigma_0 = model.covar_module.base_kernel.sigma_0.item()
    print(f"  Kernel perturbed: sigma_0 {old_sigma_0:.4f} -> {new_sigma_0:.4f}")

    # 5. Compute new L_K
    kernel_cache_new = compute_kernel_cache(model, X_train)
    L_K_new = kernel_cache_new['L_K']

    # Measure how much L_K changed
    L_K_change = (L_K_new - L_K_old).norm() / L_K_old.norm()
    print(f"  L_K relative change: {L_K_change:.4f}")

    # 6. Read m using L_K_new (what our code does after M-step)
    m_corrupted = get_variational_mean_with_L_K(model, L_K_new)

    # 7. Compute expected corruption using the formula
    # m_corrupted_expected = L_K_new @ m_stored = L_K_new @ L_K_old^{-1} @ m_natural
    m_expected_corrupted = L_K_new @ m_stored

    # 8. Verify match
    diff = (m_corrupted - m_expected_corrupted).abs().max().item()
    rel_diff = diff / m_expected_corrupted.abs().max().item()

    # Also verify this is NOT equal to original m_natural
    diff_from_original = (m_corrupted - m_natural).abs().max().item()
    rel_diff_from_original = diff_from_original / m_natural.abs().max().item()

    print(f"\n  Results:")
    print(f"    m_corrupted matches expected formula: diff={diff:.2e}, rel={rel_diff:.2e}")
    print(f"    m_corrupted differs from m_natural: diff={diff_from_original:.2e}, rel={rel_diff_from_original:.2e}")

    # PASS criteria: formula matches AND m is corrupted
    formula_matches = diff < 1e-10
    m_is_corrupted = rel_diff_from_original > 0.01  # At least 1% different

    passed = formula_matches and m_is_corrupted

    print(f"\n  HYPOTHESIS {'CONFIRMED' if passed else 'REJECTED'}:")
    print(f"    - Formula matches: {formula_matches}")
    print(f"    - m is corrupted: {m_is_corrupted}")

    return passed


# =============================================================================
# TEST B: Frozen Kernel Training Works
# =============================================================================

def test_B_frozen_kernel_training():
    """
    HYPOTHESIS: With n_mstep=0 (kernel frozen), whitening works correctly
    because L_K never changes throughout training.

    EXPECTED: Cached path λ_m matches GPyTorch model(X) λ_m within 1%
    """
    print("\n" + "="*70)
    print("TEST B: Frozen Kernel Training Works")
    print("="*70)
    print("HYPOTHESIS: With frozen kernel (n_mstep=0), whitening is consistent")

    X_train, r_train = load_pnas_data(n_train=300)
    model, likelihood = create_model_and_likelihood(X_train, ntilde=50)

    # Train with kernel frozen
    print("  Training with n_mstep=0 (kernel frozen)...")
    result = train_varGP_style(
        model, likelihood, X_train, r_train,
        n_iterations=20,  # Enough to converge
        n_estep=10, n_fstep=10,
        n_mstep=0,  # KEY: No kernel updates
        lr_f=0.1, lr_m=0.1,
        print_every=5,
        device=DEVICE,
        use_cache=True
    )
    print(f"  Final loss: {result['losses'][-1]:.2f}")

    # Get L_K (unchanged throughout training)
    kernel_cache = compute_kernel_cache(model, X_train)
    L_K = kernel_cache['L_K']

    # Read m and V with whitening conversions
    m = get_variational_mean_with_L_K(model, L_K)
    V = get_variational_covar_with_L_K(model, L_K)

    # Compute λ_m via cached path
    lambda_m_cached, lambda_var_cached = compute_moments_from_kernel_cache(kernel_cache, m, V)

    # Get λ_m via GPyTorch model(X)
    model.eval()
    with torch.no_grad():
        output = model(X_train)
        lambda_m_gpytorch = output.mean
        lambda_var_gpytorch = output.variance

    # Compare
    mean_ratio = (lambda_m_gpytorch.abs().mean() / lambda_m_cached.abs().mean()).item()
    var_ratio = (lambda_var_gpytorch.abs().mean() / lambda_var_cached.abs().mean()).item()

    print(f"\n  Results:")
    print(f"    λ_m ratio (GPyTorch/cached): {mean_ratio:.4f} (should be ~1.0)")
    print(f"    λ_var ratio (GPyTorch/cached): {var_ratio:.4f} (should be ~1.0)")

    # PASS criteria: ratios within 1% of 1.0
    mean_ok = 0.99 < mean_ratio < 1.01
    var_ok = 0.95 < var_ratio < 1.05  # Variance has more numerical error

    passed = mean_ok and var_ok

    print(f"\n  HYPOTHESIS {'CONFIRMED' if passed else 'REJECTED'}:")
    print(f"    - Mean consistent: {mean_ok}")
    print(f"    - Variance consistent: {var_ok}")

    return passed


# =============================================================================
# TEST C: Single Gradient Step Causes Corruption
# =============================================================================

def test_C_single_gradient_step_corruption():
    """
    HYPOTHESIS: A single gradient step on kernel params causes immediate
    λ_m corruption because GPyTorch uses new L_K with old m_stored.

    EXPECTED: λ_m from GPyTorch differs from correct λ_m after ONE gradient step.
    """
    print("\n" + "="*70)
    print("TEST C: Single Gradient Step Causes Corruption")
    print("="*70)
    print("HYPOTHESIS: ONE gradient step on kernel causes λ_m corruption")

    X_train, r_train = load_pnas_data(n_train=200)
    model, likelihood = create_model_and_likelihood(X_train, ntilde=50)

    # First, do some E-steps to get non-trivial m
    print("  Running initial E-step...")
    kernel_cache_old = compute_kernel_cache(model, X_train)
    L_K_old = kernel_cache_old['L_K']

    # Initialize variational params at prior (whitened form)
    # At init, stored m=0 means natural m=0 (both are zero)

    # Do one E-step to get non-zero m
    from estep import e_step_loop
    with torch.no_grad():
        e_step_loop(model, likelihood, X_train, r_train, n_estep=5, kernel_cache=kernel_cache_old)

    # Now get the natural m that we have
    m_natural = get_variational_mean_with_L_K(model, L_K_old).clone()
    print(f"  m_natural norm after E-step: {m_natural.norm():.4f}")

    # Compute CORRECT λ_m (what it should be with current kernel and m_natural)
    K = kernel_cache_old['K']
    K_tilde_j = kernel_cache_old['K_tilde_j']
    lambda_m_correct = K @ torch.linalg.solve(K_tilde_j, m_natural)

    # Also verify GPyTorch matches before kernel change
    model.eval()
    with torch.no_grad():
        output_before = model(X_train)
        lambda_m_gpytorch_before = output_before.mean

    ratio_before = (lambda_m_gpytorch_before.abs().mean() / lambda_m_correct.abs().mean()).item()
    print(f"  Ratio before kernel change: {ratio_before:.4f} (should be ~1.0)")

    # Take ONE gradient step on kernel
    print("  Taking ONE gradient step on kernel params...")
    model.train()
    optimizer = torch.optim.Adam(model.covar_module.parameters(), lr=0.1)

    with torch.enable_grad():
        optimizer.zero_grad()
        output = model(X_train)
        loss = -likelihood.expected_log_prob(r_train, output) + model.variational_strategy.kl_divergence()
        loss.backward()
        optimizer.step()

    # Now compute λ_m via GPyTorch (uses NEW L_K with OLD m_stored)
    model.eval()
    with torch.no_grad():
        output_after = model(X_train)
        lambda_m_gpytorch_after = output_after.mean

    # Compute what the correct λ_m should be now (with new kernel but SAME m_natural)
    kernel_cache_new = compute_kernel_cache(model, X_train)
    K_new = kernel_cache_new['K']
    K_tilde_j_new = kernel_cache_new['K_tilde_j']
    lambda_m_correct_after = K_new @ torch.linalg.solve(K_tilde_j_new, m_natural)

    # Measure the error
    error = (lambda_m_gpytorch_after - lambda_m_correct_after).abs().mean().item()
    correct_scale = lambda_m_correct_after.abs().mean().item()
    rel_error = error / (correct_scale + 1e-10)

    ratio_after = (lambda_m_gpytorch_after.abs().mean() / lambda_m_correct_after.abs().mean()).item()

    print(f"\n  Results:")
    print(f"    λ_m ratio after kernel change: {ratio_after:.4f} (deviation from 1.0 = corruption)")
    print(f"    Absolute error: {error:.4e}")
    print(f"    Relative error: {rel_error:.2%}")

    # PASS criteria: There should be noticeable error (corruption occurred)
    has_corruption = abs(ratio_after - 1.0) > 0.01  # More than 1% off

    passed = has_corruption

    print(f"\n  HYPOTHESIS {'CONFIRMED' if passed else 'REJECTED'}:")
    print(f"    - Corruption detected: {has_corruption}")

    return passed


# =============================================================================
# TEST D: Corruption Accumulates Over M-Step
# =============================================================================

def test_D_corruption_accumulates():
    """
    HYPOTHESIS: Multiple gradient steps within M-step accumulate corruption.

    EXPECTED: Error increases (or at least doesn't decrease) with more gradient steps.
    """
    print("\n" + "="*70)
    print("TEST D: Corruption Accumulates Over M-Step")
    print("="*70)
    print("HYPOTHESIS: Multiple gradient steps accumulate λ_m corruption")

    X_train, r_train = load_pnas_data(n_train=200)
    model, likelihood = create_model_and_likelihood(X_train, ntilde=50)

    # Initial E-step to get non-trivial m
    print("  Running initial E-step...")
    kernel_cache_init = compute_kernel_cache(model, X_train)
    L_K_init = kernel_cache_init['L_K']

    from estep import e_step_loop
    with torch.no_grad():
        e_step_loop(model, likelihood, X_train, r_train, n_estep=5, kernel_cache=kernel_cache_init)

    # Get the natural m (this is what we want to preserve)
    m_natural = get_variational_mean_with_L_K(model, L_K_init).clone()
    print(f"  m_natural norm: {m_natural.norm():.4f}")

    # Set up optimizer for kernel params
    optimizer = torch.optim.Adam(model.covar_module.parameters(), lr=0.1)

    errors = []
    ratios = []
    jitter = 1e-6

    # Take 10 gradient steps, measuring error after each
    print("\n  Taking gradient steps and measuring corruption...")
    for step in range(10):
        # Get current kernel matrices
        inducing_points = model.variational_strategy.inducing_points
        K_curr = model.covar_module(X_train, inducing_points).evaluate()
        K_tilde_curr = model.covar_module(inducing_points).evaluate()
        M = K_tilde_curr.shape[0]
        K_tilde_j_curr = K_tilde_curr + jitter * torch.eye(M, dtype=K_tilde_curr.dtype, device=DEVICE)

        # Compute correct λ_m (using current kernel but original m_natural)
        lambda_m_correct = K_curr @ torch.linalg.solve(K_tilde_j_curr, m_natural)

        # Compute GPyTorch λ_m (uses current L_K with m_stored from initial L_K)
        model.eval()
        with torch.no_grad():
            output = model(X_train)
            lambda_m_gpytorch = output.mean

        # Measure error
        error = (lambda_m_gpytorch - lambda_m_correct).abs().mean().item()
        ratio = (lambda_m_gpytorch.abs().mean() / lambda_m_correct.abs().mean()).item()
        errors.append(error)
        ratios.append(ratio)

        if step % 3 == 0:
            print(f"    Step {step}: ratio={ratio:.4f}, error={error:.4e}")

        # Take gradient step
        model.train()
        with torch.enable_grad():
            optimizer.zero_grad()
            output = model(X_train)
            loss = -likelihood.expected_log_prob(r_train, output) + model.variational_strategy.kl_divergence()
            loss.backward()
            optimizer.step()

    print(f"\n  Results:")
    print(f"    Initial ratio: {ratios[0]:.4f}")
    print(f"    Final ratio: {ratios[-1]:.4f}")
    print(f"    Initial error: {errors[0]:.4e}")
    print(f"    Final error: {errors[-1]:.4e}")

    # Check if error increased
    error_increased = errors[-1] > errors[0] * 0.8  # Allow some fluctuation
    ratio_deviated = abs(ratios[-1] - 1.0) > abs(ratios[0] - 1.0) * 0.8

    passed = error_increased or ratio_deviated

    print(f"\n  HYPOTHESIS {'CONFIRMED' if passed else 'REJECTED'}:")
    print(f"    - Error increased: {error_increased}")
    print(f"    - Ratio deviation increased: {ratio_deviated}")

    return passed


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all hypothesis tests."""
    print("\n" + "="*70)
    print("WHITENING HYPOTHESIS TESTS")
    print("="*70)
    print(f"Device: {DEVICE}")

    results = {}

    # Test A: Explicit formula
    results['A_explicit_formula'] = test_A_explicit_corruption_formula()

    # Test B: Frozen kernel
    results['B_frozen_kernel'] = test_B_frozen_kernel_training()

    # Test C: Single gradient step
    results['C_single_step'] = test_C_single_gradient_step_corruption()

    # Test D: Accumulation
    results['D_accumulation'] = test_D_corruption_accumulates()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_confirmed = all(results.values())
    for name, passed in results.items():
        status = "CONFIRMED" if passed else "REJECTED"
        print(f"  {name}: {status}")

    print(f"\nOverall: {'ALL HYPOTHESES CONFIRMED' if all_confirmed else 'SOME HYPOTHESES REJECTED'}")

    if all_confirmed:
        print("\nOur understanding of the L_K mismatch problem is validated!")
        print("Next step: Design and implement a fix.")
    else:
        print("\nSome hypotheses were rejected - need to investigate further.")

    return 0 if all_confirmed else 1


if __name__ == '__main__':
    sys.exit(main())
