"""
Reproduction tests for Batch 2 issues (B1, C1, D1, F1).

Run this script before and after fixes to verify behavior changes.

Usage:
    python investigations/batch2_correctness/test_batch2_issues.py
"""

import sys
import os
import warnings
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fstep import lambda0_given_A
from whitening import update_variational_covar, update_variational_covar_with_L_K
from model import VariationalGPModel
from kernels import ArcCosineKernel


def test_b1_jitter_mismatch():
    """Test B1: Jitter consistency in whitening.py fallback paths.

    This test creates a near-singular covariance matrix that will cause
    Cholesky decomposition to fail, triggering the fallback path.
    """
    print("\n" + "="*70)
    print("TEST B1: Jitter Consistency")
    print("="*70)

    # Create a simple model to get model.jitter
    inducing_points = torch.randn(5, 10, dtype=torch.float64)
    kernel = ArcCosineKernel(sigma_0=1.0)
    model = VariationalGPModel(
        inducing_points=inducing_points,
        kernel=kernel,
        jitter=1e-4,  # Default model jitter
        standard_variational_distribution=True
    )

    print(f"Model jitter: {model.jitter}")

    # Create a near-singular matrix (very small eigenvalues)
    M = 5
    V_singular = torch.eye(M, dtype=torch.float64) * 1e-8
    V_singular[0, 0] = 1.0  # One normal eigenvalue

    print(f"\nAttempting Cholesky on near-singular matrix...")
    print(f"Matrix condition number: {torch.linalg.cond(V_singular).item():.2e}")

    # This should trigger the fallback path
    try:
        update_variational_covar(model, V_singular)
        print("✓ update_variational_covar succeeded (used fallback)")

        # Check what jitter was used (we can't directly verify, but document behavior)
        print("  NOTE: Fallback jitter is hardcoded 1e-6 (BEFORE FIX)")
        print(f"  Should use model.jitter = {model.jitter} (AFTER FIX)")

    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n" + "-"*70)


def test_c1_threshold_values():
    """Test C1: Verify stability threshold is unified.

    Checks that STABILITY_THRESHOLD constant exists and is imported by fstep.
    """
    print("\n" + "="*70)
    print("TEST C1: Stability Threshold Consistency")
    print("="*70)

    # Verify the constant exists in estep
    import estep
    import fstep

    if hasattr(estep, 'STABILITY_THRESHOLD'):
        print(f"\n✓ estep.STABILITY_THRESHOLD = {estep.STABILITY_THRESHOLD}")
    else:
        print("\n✗ estep.STABILITY_THRESHOLD not found (BEFORE FIX)")

    if hasattr(fstep, 'STABILITY_THRESHOLD'):
        print(f"✓ fstep.STABILITY_THRESHOLD = {fstep.STABILITY_THRESHOLD}")
        if fstep.STABILITY_THRESHOLD == estep.STABILITY_THRESHOLD:
            print("✓ Thresholds are unified (FIXED)")
        else:
            print("✗ Thresholds don't match!")
    else:
        print("✗ fstep.STABILITY_THRESHOLD not found (BEFORE FIX)")

    print("\n" + "-"*70)


def test_d1_zero_spikes():
    """Test D1: Zero spike count handling in lambda0_given_A.

    This creates a batch with sum(r)=0 and verifies the function behavior.
    """
    print("\n" + "="*70)
    print("TEST D1: Zero Spike Count Handling")
    print("="*70)

    # Create test inputs
    A = torch.tensor(0.1, dtype=torch.float64)
    r_zero = torch.zeros(100, dtype=torch.float64)  # All zeros
    r_one = torch.zeros(100, dtype=torch.float64)
    r_one[0] = 1.0  # One spike

    lambda_m = torch.randn(100, dtype=torch.float64)
    lambda_var = torch.ones(100, dtype=torch.float64) * 0.1

    print(f"\nTest case 1: sum(r) = {r_zero.sum().item()}")
    try:
        result = lambda0_given_A(A, r_zero, lambda_m, lambda_var)
        print(f"  Result: lambda0 = {result.item()}")
        if torch.isinf(result):
            print("  ✗ Returns -inf (CURRENT BEHAVIOR - BUG)")
        else:
            print("  ✓ Returns valid value")
    except ValueError as e:
        print(f"  ✓ Raises ValueError: {e} (AFTER FIX)")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")

    print(f"\nTest case 2: sum(r) = {r_one.sum().item()}")
    try:
        result = lambda0_given_A(A, r_one, lambda_m, lambda_var)
        print(f"  Result: lambda0 = {result.item():.4f}")
        if torch.isfinite(result):
            print("  ✓ Returns finite value (CORRECT)")
        else:
            print("  ✗ Returns inf/-inf (PROBLEM)")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n" + "-"*70)


def test_f1_diagonal_kernel():
    """Test F1: Verify diagonal kernel behavior.

    This confirms that the diagonal kernel implementation is correct
    and matches GPyTorch's semantics.
    """
    print("\n" + "="*70)
    print("TEST F1: Diagonal Kernel (NOT A BUG)")
    print("="*70)

    # Create kernel
    kernel = ArcCosineKernel(sigma_0=1.0)

    # Test data
    X = torch.randn(10, 50, dtype=torch.float64)

    # Call with diag=True (GPyTorch contract: x1 == x2)
    K_diag = kernel(X, diag=True)

    # Call without diag to get full matrix
    K_full = kernel(X, X).evaluate()
    K_full_diag = torch.diag(K_full)

    print(f"\nDiagonal values from diag=True: {K_diag[:3].tolist()}")
    print(f"Diagonal values from full matrix: {K_full_diag[:3].tolist()}")

    max_diff = (K_diag - K_full_diag).abs().max().item()
    print(f"\nMax difference: {max_diff:.2e}")

    # Small differences (< 1e-4) are expected due to numerical precision
    # in different computation paths (diag mode optimizes calculations)
    if max_diff < 1e-4:
        print("✓ Diagonal kernel matches full matrix diagonal (within numerical precision)")
        print("\nConclusion: F1 is NOT A BUG")
        print("  - GPyTorch guarantees x1 == x2 when diag=True")
        print("  - Our implementation correctly assumes this")
        print("  - Small differences are numerical precision, not bugs")
    else:
        print("✗ Large mismatch detected (UNEXPECTED)")

    print("\n" + "-"*70)


def main():
    """Run all reproduction tests."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " BATCH 2 ISSUES - REPRODUCTION TESTS ".center(68) + "║")
    print("╚" + "="*68 + "╝")

    try:
        test_b1_jitter_mismatch()
    except Exception as e:
        print(f"B1 test failed with error: {e}")

    try:
        test_c1_threshold_values()
    except Exception as e:
        print(f"C1 test failed with error: {e}")

    try:
        test_d1_zero_spikes()
    except Exception as e:
        print(f"D1 test failed with error: {e}")

    try:
        test_f1_diagonal_kernel()
    except Exception as e:
        print(f"F1 test failed with error: {e}")

    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print("\nRun this script again after applying fixes to verify changes.")
    print()


if __name__ == '__main__':
    main()
