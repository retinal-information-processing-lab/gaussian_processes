#!/usr/bin/env python3
"""
Validation tests for pixel masking implementation.

Tests compare GPyTorch masking against reference implementation in kernels/kernels.py.

Test 1: Mask equivalence - same theta produces same mask
Test 2: C matrix equivalence - C values match on masked coordinates
Test 3: Kernel equivalence - K(X_masked) matches reference
Test 4: End-to-end fit - Pearson r within tolerance of non-masked version
"""

import sys
from pathlib import Path
import importlib.util

# Paths
gpytorch_porting_dir = Path(__file__).parent.parent
spatial_gp_dir = gpytorch_porting_dir.parent.parent

# Import gpytorch_porting/kernels.py directly (avoid name collision)
spec = importlib.util.spec_from_file_location("gpy_kernels", gpytorch_porting_dir / "kernels.py")
gpy_kernels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpy_kernels)
ArcCosineKernel = gpy_kernels.ArcCosineKernel

# Import reference implementation
sys.path.insert(0, str(spatial_gp_dir))
from kernels.kernels import localker_clean, acosker_clean

import numpy as np
import torch


def test_mask_equivalence():
    """Test 1: Same theta produces same mask as reference."""
    print("\n" + "="*60)
    print("TEST 1: Mask Equivalence")
    print("="*60)

    n_px_side = 108
    beta = 0.1
    rho = 0.1
    eps_0x = 0.0
    eps_0y = 0.0

    # Reference implementation
    theta_ref = {
        'Amp': torch.tensor(1.0, dtype=torch.float64),
        '-2log2beta': torch.tensor(-2 * np.log(2 * beta), dtype=torch.float64),
        '-log2rho2': torch.tensor(-np.log(2 * rho**2), dtype=torch.float64),
        'eps_0x': torch.tensor(eps_0x, dtype=torch.float64),
        'eps_0y': torch.tensor(eps_0y, dtype=torch.float64),
    }
    C_ref, mask_ref = localker_clean(theta_ref, n_px_side)

    # GPyTorch implementation
    kernel = ArcCosineKernel(
        sigma_0=1.0,
        n_px_side=n_px_side,
        eps_0x=eps_0x,
        eps_0y=eps_0y,
        beta=beta,
        rho=rho,
        use_mask=True
    )
    mask_gpy = kernel.compute_mask()

    # Compare
    n_pixels_ref = mask_ref.sum().item()
    n_pixels_gpy = mask_gpy.sum().item()
    masks_match = torch.all(mask_ref == mask_gpy).item()

    print(f"Reference mask: {n_pixels_ref} pixels")
    print(f"GPyTorch mask:  {n_pixels_gpy} pixels")
    print(f"Masks identical: {masks_match}")

    if masks_match:
        print("PASS: Masks match exactly!")
        return True
    else:
        diff = (mask_ref != mask_gpy).sum().item()
        print(f"FAIL: {diff} pixels differ")
        return False


def test_C_matrix_equivalence():
    """Test 2: C matrix values match on masked coordinates."""
    print("\n" + "="*60)
    print("TEST 2: C Matrix Equivalence")
    print("="*60)

    n_px_side = 108
    beta = 0.1
    rho = 0.1
    eps_0x = 0.0
    eps_0y = 0.0

    # Reference implementation
    theta_ref = {
        'Amp': torch.tensor(1.0, dtype=torch.float64),
        '-2log2beta': torch.tensor(-2 * np.log(2 * beta), dtype=torch.float64),
        '-log2rho2': torch.tensor(-np.log(2 * rho**2), dtype=torch.float64),
        'eps_0x': torch.tensor(eps_0x, dtype=torch.float64),
        'eps_0y': torch.tensor(eps_0y, dtype=torch.float64),
    }
    C_ref, mask_ref = localker_clean(theta_ref, n_px_side)

    # GPyTorch implementation
    kernel = ArcCosineKernel(
        sigma_0=1.0,
        n_px_side=n_px_side,
        eps_0x=eps_0x,
        eps_0y=eps_0y,
        beta=beta,
        rho=rho,
        use_mask=True
    )
    C_gpy, mask_gpy = kernel._compute_C_matrix(apply_mask=True)

    # Compare (note: reference has Amp factor, GPyTorch doesn't include it in C)
    # Reference: C = Amp * alpha[:, None] * C_smooth * alpha[None, :]
    # GPyTorch: C = alpha[:, None] * C_smooth * alpha[None, :] (no Amp)
    # So we compare C_ref / Amp with C_gpy

    C_ref_normalized = C_ref / theta_ref['Amp']

    max_diff = (C_ref_normalized - C_gpy).abs().max().item()
    mean_diff = (C_ref_normalized - C_gpy).abs().mean().item()

    print(f"C matrix shape: {C_gpy.shape}")
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")

    if max_diff < 1e-6:
        print("PASS: C matrices match!")
        return True
    else:
        print(f"FAIL: Max diff {max_diff:.2e} > 1e-6")
        return False


def test_kernel_equivalence():
    """Test 3: K(X_masked) matches reference acosker_clean()."""
    print("\n" + "="*60)
    print("TEST 3: Kernel Equivalence")
    print("="*60)

    torch.manual_seed(42)
    n_px_side = 108
    n_px = n_px_side ** 2
    beta = 0.1
    rho = 0.1
    sigma_0 = 1.0
    eps_0x = 0.0
    eps_0y = 0.0

    # Create test data
    n_samples = 10
    X_full = torch.randn(n_samples, n_px, dtype=torch.float64)

    # Get mask and C from reference
    theta_C = {
        'Amp': torch.tensor(1.0, dtype=torch.float64),
        '-2log2beta': torch.tensor(-2 * np.log(2 * beta), dtype=torch.float64),
        '-log2rho2': torch.tensor(-np.log(2 * rho**2), dtype=torch.float64),
        'eps_0x': torch.tensor(eps_0x, dtype=torch.float64),
        'eps_0y': torch.tensor(eps_0y, dtype=torch.float64),
    }
    C_ref, mask_ref = localker_clean(theta_C, n_px_side)

    # Reference kernel computation (manually apply mask)
    X_masked_ref = X_full[:, mask_ref]
    theta_K = {'sigma_0': torch.tensor(sigma_0, dtype=torch.float64)}
    # Note: reference acosker_clean expects C without Amp factor
    C_ref_no_amp = C_ref / theta_C['Amp']
    K_ref = acosker_clean(theta_K, X_masked_ref, C=C_ref_no_amp, diag=False)

    # GPyTorch kernel computation
    kernel = ArcCosineKernel(
        sigma_0=sigma_0,
        n_px_side=n_px_side,
        eps_0x=eps_0x,
        eps_0y=eps_0y,
        beta=beta,
        rho=rho,
        use_mask=True
    )
    K_gpy = kernel(X_full, X_full).evaluate()

    # Compare
    max_diff = (K_ref - K_gpy).abs().max().item()
    mean_diff = (K_ref - K_gpy).abs().mean().item()

    print(f"Kernel shape: {K_gpy.shape}")
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")

    # Relative tolerance: diff/mean_value should be small
    mean_value = K_ref.abs().mean().item()
    rel_diff = max_diff / mean_value
    print(f"Relative difference: {rel_diff:.2e}")

    if rel_diff < 1e-5:
        print("PASS: Kernel matrices match!")
        return True
    else:
        print(f"FAIL: Relative diff {rel_diff:.2e} > 1e-5")
        return False


def test_end_to_end_fit(quick=True):
    """Test 4: End-to-end fit with masking achieves similar Pearson r.

    Args:
        quick: If True, use fewer iterations for quick validation.
    """
    print("\n" + "="*60)
    print("TEST 4: End-to-End Fit Quality")
    print("="*60)

    import gpytorch

    # Import from gpytorch_porting using importlib
    spec_like = importlib.util.spec_from_file_location("likelihoods", gpytorch_porting_dir / "likelihoods.py")
    likelihoods = importlib.util.module_from_spec(spec_like)
    spec_like.loader.exec_module(likelihoods)
    PoissonLikelihood = likelihoods.PoissonLikelihood

    spec_model = importlib.util.spec_from_file_location("model", gpytorch_porting_dir / "model.py")
    model_mod = importlib.util.module_from_spec(spec_model)
    spec_model.loader.exec_module(model_mod)
    VariationalGPModel = model_mod.VariationalGPModel

    spec_train = importlib.util.spec_from_file_location("train", gpytorch_porting_dir / "train.py")
    train_mod = importlib.util.module_from_spec(spec_train)
    spec_train.loader.exec_module(train_mod)
    train_model = train_mod.train_model
    predict = train_mod.predict
    compute_pearson_correlation = train_mod.compute_pearson_correlation

    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = np.load(data_path)

    X_train = torch.tensor(data['images_train'][:500], dtype=torch.float64).reshape(500, -1)
    R_train = torch.tensor(data['responses_train'][:500, 8], dtype=torch.float64)
    X_test = torch.tensor(data['images_test'], dtype=torch.float64).reshape(30, -1)
    R_test = torch.tensor(data['responses_test'][:, :, 8], dtype=torch.float64)

    # Settings
    torch.manual_seed(42)
    ntilde = 100
    n_iterations = 100 if quick else 300

    indices = torch.randperm(X_train.shape[0])[:ntilde]
    inducing_points = X_train[indices].clone()

    results = {}

    for use_mask in [True, False]:
        print(f"\nTraining with use_mask={use_mask}...")

        base_kernel = ArcCosineKernel(
            sigma_0=1.0,
            n_px_side=108,
            eps_0x=0.0,
            eps_0y=0.0,
            beta=0.1,
            rho=0.1,
            use_mask=use_mask
        )
        kernel = gpytorch.kernels.ScaleKernel(base_kernel)
        kernel.outputscale = 1e-4

        model = VariationalGPModel(inducing_points, kernel, jitter=1e-4).double()
        likelihood = PoissonLikelihood(A_init=1.0, lambda0_init=0.0).double()

        losses = train_model(model, likelihood, X_train, R_train,
                            n_iterations=n_iterations, lr=0.01, print_every=0)

        predictions = predict(model, likelihood, X_test)
        r_test_mean = R_test.mean(dim=0)
        corr = compute_pearson_correlation(r_test_mean, predictions['f_pred'])

        if use_mask:
            mask_size = base_kernel._cached_mask.sum().item()
            print(f"  Mask size: {mask_size} pixels")

        print(f"  Final loss: {losses[-1]:.2f}")
        print(f"  Pearson r: {corr:.4f}")

        results[use_mask] = corr

    # Compare
    diff = abs(results[True] - results[False])
    print(f"\nPearson r difference: {diff:.4f}")
    print(f"Masked:   r = {results[True]:.4f}")
    print(f"Full:     r = {results[False]:.4f}")

    # Success criterion: within 0.05 of each other
    if diff < 0.05:
        print("PASS: Masked fit quality within tolerance!")
        return True
    else:
        print(f"FAIL: Difference {diff:.4f} > 0.05 tolerance")
        return False


def run_all_tests(quick=True):
    """Run all validation tests."""
    print("\n" + "="*60)
    print("PIXEL MASKING VALIDATION TESTS")
    print("="*60)

    results = {
        'mask_equivalence': test_mask_equivalence(),
        'C_matrix_equivalence': test_C_matrix_equivalence(),
        'kernel_equivalence': test_kernel_equivalence(),
        'end_to_end_fit': test_end_to_end_fit(quick=quick),
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED'}")

    return all_passed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run mask validation tests')
    parser.add_argument('--full', action='store_true',
                        help='Run full end-to-end test (more iterations)')
    parser.add_argument('--test', type=int, choices=[1, 2, 3, 4],
                        help='Run specific test only')
    args = parser.parse_args()

    if args.test == 1:
        test_mask_equivalence()
    elif args.test == 2:
        test_C_matrix_equivalence()
    elif args.test == 3:
        test_kernel_equivalence()
    elif args.test == 4:
        test_end_to_end_fit(quick=not args.full)
    else:
        run_all_tests(quick=not args.full)
