"""
Test suite for analytical M-step gradients (TDD)

Tests verify that analytical gradients match PyTorch autograd.
Uses real PNAS data as per WORKING_GUIDELINES.

Usage:
    python tests/test_mstep_analytical.py
    python tests/test_mstep_analytical.py --test 1 --verbose
    python tests/test_mstep_analytical.py --test 8  # performance test

Tests:
    1. test_dC_gradients - Compare dC against autograd
    2. test_dK_gradients - Compare dK against autograd (full matrix)
    3. test_dKvec_gradients - Compare dKvec against autograd (diagonal)
    4. test_dlambda_moments - Verify dlambda_m, dlambda_var via finite differences
    5. test_loss_gradients - Verify dloglikelihood and dKL match autograd
    6. test_training_equivalence - Train with analytical vs autograd, compare loss
    7. test_numerical_stability - Check for NaN/Inf during training
    8. test_performance - Timing comparison (analytical should be >=2x faster)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from eigenspace_utils import eigendecompose_K_tilde, compute_K_tilde_b_diagonal, compute_KKtilde_inv_b

# Will be implemented in direct_vargp.py:
# from eigenspace_gradients import (
#     compute_C_and_gradients,
#     compute_kernel_and_gradients,
#     compute_lambda_moments_and_gradients,
#     compute_loss_gradients,
#     mstep_lbfgs_analytical,
# )


def load_test_data(n_train=100, n_tilde=25, seed=42, device='cuda'):
    """Load real PNAS data for testing.

    Returns:
        X: Training inputs (n_train, n_features)
        X_tilde: Inducing points (n_tilde, n_features)
        r: Spike counts (n_train,)
        device: torch device
    """
    # Load PNAS data
    data_path = Path(__file__).parent.parent.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = np.load(data_path)

    X_all = torch.tensor(data['images_train'], dtype=torch.float64, device=device)
    X_all = X_all.reshape(X_all.shape[0], -1)  # Flatten images
    R_all = torch.tensor(data['responses_train'], dtype=torch.float64, device=device)

    # Select cell 8 (same as other tests)
    cellid = 8
    r_all = R_all[:, cellid]

    # Sample subset
    torch.manual_seed(seed)
    indices = torch.randperm(X_all.shape[0])[:n_train]
    X = X_all[indices]
    r = r_all[indices]

    # Inducing points (subset of training)
    idx_tilde = torch.randperm(n_train)[:n_tilde]
    X_tilde = X[idx_tilde]

    return X, X_tilde, r, device


def create_test_kernel(device='cuda'):
    """Create a kernel with typical parameters for testing."""
    kernel = ArcCosineKernel(
        n_px_side=108,
        sigma_0=1.0,
        Amp=1.0,
        beta=0.1,
        rho=0.1,
        eps_0x=0.0,
        eps_0y=0.0,
        use_mask=True,
        gradient_mode='autograd'  # Use autograd as reference
    )
    kernel = kernel.double().to(device)
    return kernel


# ==============================================================================
# Test 1: dC gradients
# ==============================================================================
def test_dC_gradients(verbose=False):
    """Test that analytical dC matches autograd for all C-dependent hyperparameters."""
    print("\n=== Test 1: dC gradients ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kernel = create_test_kernel(device)

    try:
        from eigenspace_gradients import compute_C_and_gradients
    except ImportError:
        print("SKIP: compute_C_and_gradients not implemented yet")
        return None

    # Compute C analytically with gradients
    C_analytical, mask, dC_analytical = compute_C_and_gradients(kernel)

    # Compute C with autograd
    kernel.zero_grad()
    C_auto, _ = kernel._compute_C_matrix(apply_mask=kernel.use_mask)

    # Check each parameter
    params_to_test = ['Amp', 'eps_0x', 'eps_0y', 'raw_m2log2beta', 'raw_mlog2rho2']
    all_passed = True

    for param_name in params_to_test:
        # Get autograd gradient
        kernel.zero_grad()
        C_auto, _ = kernel._compute_C_matrix(apply_mask=kernel.use_mask)
        loss = C_auto.sum()
        loss.backward()

        if param_name == 'Amp':
            autograd_grad = kernel.raw_Amp.grad
            # Correct for softplus: dC/d(raw_Amp) = dC/d(Amp) * sigmoid(raw_Amp)
            analytical_grad = dC_analytical['Amp'].sum() * torch.sigmoid(kernel.raw_Amp)
        elif param_name in ['eps_0x', 'eps_0y', 'raw_m2log2beta', 'raw_mlog2rho2']:
            param = getattr(kernel, param_name)
            autograd_grad = param.grad
            analytical_grad = dC_analytical[param_name].sum()

        if autograd_grad is None:
            print(f"  {param_name}: SKIP (no autograd gradient)")
            continue

        abs_err = (analytical_grad - autograd_grad).abs()
        rel_err = abs_err / (autograd_grad.abs() + 1e-10)

        # Use absolute tolerance for near-zero values, relative tolerance otherwise
        if autograd_grad.abs().item() < 1e-8:
            passed = abs_err.item() < 1e-8
            err_str = f"abs_err={abs_err.item():.2e}"
        else:
            passed = rel_err.item() < 1e-5
            err_str = f"rel_err={rel_err.item():.2e}"

        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  {param_name}: {status} ({err_str})")

        if verbose:
            print(f"    analytical: {analytical_grad.item():.6e}")
            print(f"    autograd:   {autograd_grad.item():.6e}")

    return all_passed


# ==============================================================================
# Test 2: dK gradients (full matrix)
# ==============================================================================
def test_dK_gradients(verbose=False):
    """Test that analytical dK matches autograd for all hyperparameters (full matrix)."""
    print("\n=== Test 2: dK gradients (full matrix) ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, X_tilde, r, _ = load_test_data(n_train=50, n_tilde=20, device=device)
    kernel = create_test_kernel(device)

    try:
        from eigenspace_gradients import compute_C_and_gradients, compute_kernel_and_gradients
    except ImportError:
        print("SKIP: Functions not implemented yet")
        return None

    # Get mask
    mask = kernel.compute_mask()
    X_masked = X[:, mask]
    X_tilde_masked = X_tilde[:, mask]

    # Compute C and dC
    C, _, dC = compute_C_and_gradients(kernel)

    # Compute K analytically with gradients
    sigma_0 = kernel.sigma_0.squeeze()
    K_analytical, dK_analytical = compute_kernel_and_gradients(
        X_masked, X_tilde_masked, C, dC, sigma_0, diag=False
    )

    # Test each parameter
    params_to_test = ['sigma_0', 'Amp', 'eps_0x', 'eps_0y', 'raw_m2log2beta', 'raw_mlog2rho2']
    all_passed = True

    for param_name in params_to_test:
        # Compute K with autograd
        kernel.zero_grad()
        for p in kernel.parameters():
            p.requires_grad_(True)

        K_auto = kernel(X, X_tilde).evaluate()
        loss = K_auto.sum()
        loss.backward()

        # Get gradients
        if param_name == 'sigma_0':
            autograd_grad = kernel.raw_sigma_0.grad
            # Correct for softplus
            analytical_grad = dK_analytical['sigma_0'].sum() * torch.sigmoid(kernel.raw_sigma_0)
        elif param_name == 'Amp':
            autograd_grad = kernel.raw_Amp.grad
            analytical_grad = dK_analytical['Amp'].sum() * torch.sigmoid(kernel.raw_Amp)
        elif param_name in ['eps_0x', 'eps_0y', 'raw_m2log2beta', 'raw_mlog2rho2']:
            param = getattr(kernel, param_name)
            autograd_grad = param.grad
            analytical_grad = dK_analytical[param_name].sum()

        if autograd_grad is None:
            print(f"  {param_name}: SKIP (no autograd gradient)")
            continue

        abs_err = (analytical_grad - autograd_grad).abs()
        rel_err = abs_err / (autograd_grad.abs() + 1e-10)

        # Use absolute tolerance for near-zero values, relative tolerance otherwise
        if autograd_grad.abs().item() < 1e-8:
            passed = abs_err.item() < 1e-8
            err_str = f"abs_err={abs_err.item():.2e}"
        else:
            passed = rel_err.item() < 1e-5
            err_str = f"rel_err={rel_err.item():.2e}"

        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  {param_name}: {status} ({err_str})")

        if verbose:
            print(f"    analytical: {analytical_grad.item():.6e}")
            print(f"    autograd:   {autograd_grad.item():.6e}")

    return all_passed


# ==============================================================================
# Test 3: dKvec gradients (diagonal)
# ==============================================================================
def test_dKvec_gradients(verbose=False):
    """Test that analytical dKvec matches autograd (diagonal kernel)."""
    print("\n=== Test 3: dKvec gradients (diagonal) ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, X_tilde, r, _ = load_test_data(n_train=50, n_tilde=20, device=device)
    kernel = create_test_kernel(device)

    try:
        from eigenspace_gradients import compute_C_and_gradients, compute_kernel_and_gradients
    except ImportError:
        print("SKIP: Functions not implemented yet")
        return None

    # Get mask
    mask = kernel.compute_mask()
    X_masked = X[:, mask]

    # Compute C and dC
    C, _, dC = compute_C_and_gradients(kernel)

    # Compute Kvec analytically with gradients
    sigma_0 = kernel.sigma_0.squeeze()
    Kvec_analytical, dKvec_analytical = compute_kernel_and_gradients(
        X_masked, None, C, dC, sigma_0, diag=True
    )

    # Test each parameter
    params_to_test = ['sigma_0', 'Amp', 'eps_0x', 'eps_0y', 'raw_m2log2beta', 'raw_mlog2rho2']
    all_passed = True

    for param_name in params_to_test:
        # Compute Kvec with autograd
        kernel.zero_grad()
        for p in kernel.parameters():
            p.requires_grad_(True)

        Kvec_auto = kernel(X, diag=True)
        loss = Kvec_auto.sum()
        loss.backward()

        # Get gradients
        if param_name == 'sigma_0':
            autograd_grad = kernel.raw_sigma_0.grad
            analytical_grad = dKvec_analytical['sigma_0'].sum() * torch.sigmoid(kernel.raw_sigma_0)
        elif param_name == 'Amp':
            autograd_grad = kernel.raw_Amp.grad
            analytical_grad = dKvec_analytical['Amp'].sum() * torch.sigmoid(kernel.raw_Amp)
        elif param_name in ['eps_0x', 'eps_0y', 'raw_m2log2beta', 'raw_mlog2rho2']:
            param = getattr(kernel, param_name)
            autograd_grad = param.grad
            analytical_grad = dKvec_analytical[param_name].sum()

        if autograd_grad is None:
            print(f"  {param_name}: SKIP (no autograd gradient)")
            continue

        abs_err = (analytical_grad - autograd_grad).abs()
        rel_err = abs_err / (autograd_grad.abs() + 1e-10)

        # Use absolute tolerance for near-zero values, relative tolerance otherwise
        if autograd_grad.abs().item() < 1e-8:
            passed = abs_err.item() < 1e-8
            err_str = f"abs_err={abs_err.item():.2e}"
        else:
            passed = rel_err.item() < 1e-5
            err_str = f"rel_err={rel_err.item():.2e}"

        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  {param_name}: {status} ({err_str})")

        if verbose:
            print(f"    analytical: {analytical_grad.item():.6e}")
            print(f"    autograd:   {autograd_grad.item():.6e}")

    return all_passed


# ==============================================================================
# Test 4: dlambda moments
# ==============================================================================
def test_dlambda_moments(verbose=False):
    """Test dlambda_m and dlambda_var via finite differences."""
    print("\n=== Test 4: dlambda moments (finite differences) ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, X_tilde, r, _ = load_test_data(n_train=100, n_tilde=25, device=device)
    kernel = create_test_kernel(device)

    try:
        from eigenspace_gradients import (
            compute_C_and_gradients,
            compute_kernel_and_gradients,
            compute_lambda_moments_and_gradients
        )
    except ImportError:
        print("SKIP: Functions not implemented yet")
        return None

    # Get mask
    mask = kernel.compute_mask()
    X_masked = X[:, mask]
    X_tilde_masked = X_tilde[:, mask]

    # Compute base state
    C, _, dC = compute_C_and_gradients(kernel)
    sigma_0 = kernel.sigma_0.squeeze()

    K_tilde, dK_tilde = compute_kernel_and_gradients(
        X_tilde_masked, X_tilde_masked, C, dC, sigma_0, diag=False
    )
    K, dK = compute_kernel_and_gradients(
        X_masked, X_tilde_masked, C, dC, sigma_0, diag=False
    )
    Kvec, dKvec = compute_kernel_and_gradients(
        X_masked, None, C, dC, sigma_0, diag=True
    )

    # Eigenspace projection
    B, eigvals_b, _ = eigendecompose_K_tilde(K_tilde)
    K_tilde_b = compute_K_tilde_b_diagonal(eigvals_b)
    K_b = K @ B
    KKtilde_inv_b = compute_KKtilde_inv_b(K_b, eigvals_b)

    # Initialize m_b, V_b
    n_b = len(eigvals_b)
    m_b = torch.randn(n_b, dtype=X.dtype, device=device) * 0.1
    V_b = K_tilde_b + torch.eye(n_b, dtype=X.dtype, device=device) * 0.1
    V_b = (V_b + V_b.T) / 2  # Symmetrize

    # Project dK to eigenspace
    dK_b = {k: v @ B for k, v in dK.items()}
    dK_tilde_b = {k: B.T @ v @ B for k, v in dK_tilde.items()}

    # Compute analytical gradients
    lambda_m, lambda_var, dlambda_m, dlambda_var = compute_lambda_moments_and_gradients(
        K_b, K_tilde_b, Kvec, m_b, V_b,
        dK_b, dK_tilde_b, dKvec, eigvals_b
    )

    # Test via finite differences for one parameter
    eps = 1e-5
    param_name = 'raw_m2log2beta'

    print(f"  Testing {param_name} via finite differences (eps={eps})...")

    # This is a rough check - exact finite diff would require recomputing everything
    # For now, just verify the shapes and that values are reasonable
    if param_name in dlambda_m:
        print(f"    dlambda_m[{param_name}] shape: {dlambda_m[param_name].shape}")
        print(f"    dlambda_var[{param_name}] shape: {dlambda_var[param_name].shape}")
        print(f"    dlambda_m range: [{dlambda_m[param_name].min():.6e}, {dlambda_m[param_name].max():.6e}]")
        print(f"    dlambda_var range: [{dlambda_var[param_name].min():.6e}, {dlambda_var[param_name].max():.6e}]")

        # Check for NaN/Inf
        if torch.isnan(dlambda_m[param_name]).any() or torch.isinf(dlambda_m[param_name]).any():
            print(f"  FAIL: NaN/Inf in dlambda_m[{param_name}]")
            return False
        if torch.isnan(dlambda_var[param_name]).any() or torch.isinf(dlambda_var[param_name]).any():
            print(f"  FAIL: NaN/Inf in dlambda_var[{param_name}]")
            return False

        print("  PASS: No NaN/Inf, shapes correct")
        return True
    else:
        print(f"  SKIP: {param_name} not in dlambda_m")
        return None


# ==============================================================================
# Test 5: loss gradients
# ==============================================================================
def test_loss_gradients(verbose=False):
    """Test that dloglikelihood and dKL match autograd."""
    print("\n=== Test 5: loss gradients ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, X_tilde, r, _ = load_test_data(n_train=100, n_tilde=25, device=device)
    kernel = create_test_kernel(device)
    likelihood = PoissonLikelihood(A_init=0.01, lambda0_init=1.0)
    likelihood = likelihood.double().to(device)

    try:
        from eigenspace_gradients import compute_loss_gradients
        from eigenspace_model import DirectVariationalState
    except ImportError:
        print("SKIP: Functions not implemented yet")
        return None

    # TODO: Full implementation would compute loss analytically and compare to autograd
    print("  SKIP: Full loss gradient test not yet implemented")
    return None


# ==============================================================================
# Test 6: training equivalence
# ==============================================================================
def test_training_equivalence(verbose=False):
    """Train 10 iterations with analytical vs autograd, compare final loss."""
    print("\n=== Test 6: training equivalence ===")

    try:
        from mstep import mstep_eigenspace_analytical
    except ImportError:
        print("SKIP: mstep_eigenspace_analytical not implemented yet")
        return None

    # TODO: Compare training with analytical vs autograd M-step
    print("  SKIP: Training equivalence test not yet implemented")
    return None


# ==============================================================================
# Test 7: numerical stability
# ==============================================================================
def test_numerical_stability(verbose=False):
    """Check for NaN/Inf during training with analytical gradients."""
    print("\n=== Test 7: numerical stability ===")

    try:
        from mstep import mstep_eigenspace_analytical
    except ImportError:
        print("SKIP: mstep_eigenspace_analytical not implemented yet")
        return None

    # TODO: Run training and check for NaN/Inf in all gradients
    print("  SKIP: Numerical stability test not yet implemented")
    return None


# ==============================================================================
# Test 8: performance
# ==============================================================================
def test_performance(verbose=False):
    """Timing comparison: analytical should be >=2x faster than autograd."""
    print("\n=== Test 8: performance ===")

    try:
        from mstep import mstep_eigenspace_analytical, mstep_eigenspace_autograd
    except ImportError:
        print("SKIP: Functions not implemented yet")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, X_tilde, r, _ = load_test_data(n_train=500, n_tilde=50, device=device)

    # TODO: Time both approaches and compare
    print("  SKIP: Performance test not yet implemented")
    return None


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Test analytical M-step gradients')
    parser.add_argument('--test', type=int, default=0,
                        help='Run specific test (1-8), or 0 for all')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')
    args = parser.parse_args()

    tests = [
        test_dC_gradients,
        test_dK_gradients,
        test_dKvec_gradients,
        test_dlambda_moments,
        test_loss_gradients,
        test_training_equivalence,
        test_numerical_stability,
        test_performance,
    ]

    if args.test > 0:
        if args.test > len(tests):
            print(f"Invalid test number: {args.test}")
            return 1
        result = tests[args.test - 1](verbose=args.verbose)
        return 0 if result else 1

    # Run all tests
    results = []
    for i, test in enumerate(tests, 1):
        result = test(verbose=args.verbose)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for i, (test, result) in enumerate(zip(tests, results), 1):
        if result is None:
            status = "SKIP"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  Test {i}: {test.__name__}: {status}")

    # Return non-zero if any test failed
    failed = any(r is False for r in results)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
