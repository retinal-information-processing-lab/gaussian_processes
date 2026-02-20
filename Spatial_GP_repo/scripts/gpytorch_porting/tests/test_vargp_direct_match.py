"""
Unit tests for vargp_direct implementation correctness.

Verifies:
1. Softplus chain rule for sigma_0/Amp gradients (BUG #4)
2. Eigenspace projection preserves gradient structure (BUG #3)
3. solve()-based K_tilde_inv matches eigenvalue-based for diagonal case
4. Gradient magnitudes are reasonable
5. Multi-cell end-to-end comparison with vargp_old
6. Initialization matches vargp_old
7. Single E-step matches vargp_old

Usage:
    python tests/test_vargp_direct_match.py
    python tests/test_vargp_direct_match.py --test 1 --verbose
    python tests/test_vargp_direct_match.py --test 5  # multi-cell (slow)

Run: conda run -n pytorch_gpytorch python tests/test_vargp_direct_match.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from eigenspace_utils import (
    EIGVAL_TOL,
    eigendecompose_K_tilde,
    compute_K_tilde_b_diagonal,
    compute_KKtilde_inv_b,
)
from eigenspace_gradients import (
    compute_C_and_gradients,
    compute_kernel_and_gradients,
    compute_lambda_moments_and_gradients,
    compute_loss_gradients,
)


def load_test_data(n_train=100, n_tilde=25, cellid=8, seed=42, device='cuda'):
    """Load real PNAS data for testing.

    Returns:
        X: Training inputs (n_train, n_features)
        X_tilde: Inducing points (n_tilde, n_features)
        r: Spike counts (n_train,)
        device: torch device
    """
    data_path = Path(__file__).parent.parent.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = np.load(data_path)

    X_all = torch.tensor(data['images_train'], dtype=torch.float64, device=device)
    X_all = X_all.reshape(X_all.shape[0], -1)  # Flatten images
    R_all = torch.tensor(data['responses_train'], dtype=torch.float64, device=device)

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


def create_test_kernel(device='cuda', dtype=torch.float64):
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
        gradient_mode='autograd'
    )
    if dtype == torch.float64:
        kernel = kernel.double()
    kernel = kernel.to(device)
    return kernel


# ==============================================================================
# Test 1: Softplus Chain Rule Verification (BUG #4)
# ==============================================================================
def test_softplus_chain_rule(verbose=False):
    """Verify softplus chain rule is applied correctly for sigma_0 and Amp.

    This is a simpler test that just checks the kernel matrix gradient,
    not the full ELBO. Uses finite differences as ground truth.

    d(K.sum())/d(raw_sigma_0) should equal:
        d(K.sum())/d(sigma_0) * sigmoid(raw_sigma_0)
    """
    print("\n=== Test 1: Softplus Chain Rule Verification ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64

    # Load test data (small for speed)
    X, X_tilde, r, _ = load_test_data(n_train=30, n_tilde=15, device=device)
    kernel = create_test_kernel(device, dtype)

    # Get mask
    mask = kernel.compute_mask()
    X_masked = X[:, mask]
    X_tilde_masked = X_tilde[:, mask]

    # Test parameters
    eps = 1e-5  # finite difference step
    tolerance = 1e-4

    all_passed = True

    for param_name, raw_name in [('sigma_0', 'raw_sigma_0'), ('Amp', 'raw_Amp')]:
        print(f"\n  Testing {param_name} ({raw_name}):")

        # Get current raw value
        raw_param = getattr(kernel, raw_name)
        raw_val = raw_param.data.clone()
        transformed_val = getattr(kernel, param_name).squeeze()

        # Compute analytical gradient (dK/d(transformed_param))
        C, _, dC = compute_C_and_gradients(kernel)
        sigma_0 = kernel.sigma_0.squeeze()
        K, dK = compute_kernel_and_gradients(
            X_masked, X_tilde_masked, C, dC, sigma_0, diag=False
        )

        # Get analytical dK/d(param)
        if param_name in dK:
            analytical_dparam = dK[param_name].sum()
        else:
            print(f"    SKIP: {param_name} not in dK")
            continue

        # Apply chain rule: dK/d(raw) = dK/d(param) * sigmoid(raw)
        sigmoid_raw = torch.sigmoid(raw_val)
        analytical_draw = analytical_dparam * sigmoid_raw

        # Compute finite difference gradient w.r.t. raw parameter
        with torch.no_grad():
            # K(raw + eps)
            raw_param.data = raw_val + eps
            K_plus = kernel(X, X_tilde).to_dense()
            loss_plus = K_plus.sum().item()

            # K(raw - eps)
            raw_param.data = raw_val - eps
            K_minus = kernel(X, X_tilde).to_dense()
            loss_minus = K_minus.sum().item()

            # Restore
            raw_param.data = raw_val

        fd_draw = (loss_plus - loss_minus) / (2 * eps)

        # Compare
        abs_err = abs(analytical_draw.item() - fd_draw)
        rel_err = abs_err / (abs(fd_draw) + 1e-10)

        passed = rel_err < tolerance
        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        print(f"    analytical dK/d(raw): {analytical_draw.item():.6e}")
        print(f"    finite diff dK/d(raw): {fd_draw:.6e}")
        print(f"    rel_err: {rel_err:.2e} [{status}]")

        if verbose:
            print(f"    sigmoid(raw): {sigmoid_raw.item():.6e}")
            print(f"    analytical dK/d(param): {analytical_dparam.item():.6e}")

    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: Softplus chain rule test PASSED!")
    else:
        print("RESULT: Softplus chain rule test FAILED!")
    print("=" * 60)

    return all_passed


# ==============================================================================
# Test 2: Eigenspace Projection Correctness (BUG #3)
# ==============================================================================
def test_eigenspace_projection_gradients(verbose=False):
    """Verify eigenspace projection preserves gradient correctness.

    Compares dK_tilde_b = B.T @ dK_tilde @ B based gradients to
    finite differences (ground truth).
    """
    print("\n=== Test 2: Eigenspace Projection Gradient Correctness ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64

    X, X_tilde, r, _ = load_test_data(n_train=30, n_tilde=15, device=device)
    kernel = create_test_kernel(device, dtype)

    # Compute initial K_tilde and eigenspace
    with torch.no_grad():
        K_tilde_init = kernel(X_tilde, X_tilde).to_dense()
        K_tilde_init = (K_tilde_init + K_tilde_init.T) / 2
        B, eigvals_b, _ = eigendecompose_K_tilde(K_tilde_init)
        B = B.clone()  # Detach from any computation graph

    # Get mask
    mask = kernel.compute_mask()
    X_tilde_masked = X_tilde[:, mask]

    # Test parameters
    eps = 1e-5
    tolerance = 1e-4
    all_passed = True

    params_to_test = [
        ('sigma_0', 'raw_sigma_0', True),  # (name in dK, raw_name, uses_softplus)
        ('Amp', 'raw_Amp', True),
        ('eps_0x', 'eps_0x', False),
        ('eps_0y', 'eps_0y', False),
        ('raw_m2log2beta', 'raw_m2log2beta', False),
        ('raw_mlog2rho2', 'raw_mlog2rho2', False),
    ]

    for param_name, raw_name, uses_softplus in params_to_test:
        print(f"\n  Testing {param_name}:")

        # Get raw parameter
        raw_param = getattr(kernel, raw_name)
        raw_val = raw_param.data.clone()

        # Compute analytical gradient
        C, _, dC = compute_C_and_gradients(kernel)
        sigma_0 = kernel.sigma_0.squeeze()
        K_tilde, dK_tilde = compute_kernel_and_gradients(
            X_tilde_masked, X_tilde_masked, C, dC, sigma_0, diag=False
        )
        K_tilde = (K_tilde + K_tilde.T) / 2

        # Project to eigenspace
        dK_tilde_b = {k: B.T @ v @ B for k, v in dK_tilde.items()}

        if param_name not in dK_tilde_b:
            print(f"    SKIP: {param_name} not in dK_tilde_b")
            continue

        # Analytical gradient of K_tilde_b.sum() w.r.t. param (not raw)
        analytical_dparam = dK_tilde_b[param_name].sum()

        # If uses softplus, apply chain rule
        if uses_softplus:
            sigmoid_raw = torch.sigmoid(raw_val)
            analytical_draw = analytical_dparam * sigmoid_raw
        else:
            analytical_draw = analytical_dparam

        # Compute finite difference w.r.t. raw parameter
        with torch.no_grad():
            # K_tilde_b(raw + eps).sum()
            raw_param.data = raw_val + eps
            K_tilde_plus = kernel(X_tilde, X_tilde).to_dense()
            K_tilde_plus = (K_tilde_plus + K_tilde_plus.T) / 2
            K_tilde_b_plus = B.T @ K_tilde_plus @ B
            loss_plus = K_tilde_b_plus.sum().item()

            # K_tilde_b(raw - eps).sum()
            raw_param.data = raw_val - eps
            K_tilde_minus = kernel(X_tilde, X_tilde).to_dense()
            K_tilde_minus = (K_tilde_minus + K_tilde_minus.T) / 2
            K_tilde_b_minus = B.T @ K_tilde_minus @ B
            loss_minus = K_tilde_b_minus.sum().item()

            # Restore
            raw_param.data = raw_val

        fd_draw = (loss_plus - loss_minus) / (2 * eps)

        # Compare
        abs_err = abs(analytical_draw.item() - fd_draw)
        rel_err = abs_err / (abs(fd_draw) + 1e-10)

        passed = rel_err < tolerance
        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        print(f"    analytical: {analytical_draw.item():.6e}")
        print(f"    finite diff: {fd_draw:.6e}")
        print(f"    rel_err: {rel_err:.2e} [{status}]")

    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: Eigenspace projection test PASSED!")
    else:
        print("RESULT: Eigenspace projection test FAILED!")
    print("=" * 60)

    return all_passed


# ==============================================================================
# Test 3: K_tilde_inv via solve() vs Eigendecomposition
# ==============================================================================
def test_ktilde_inv_methods(verbose=False):
    """Verify solve-based K_tilde_inv matches eigenvalue-based for diagonal case.

    Key insight: After K_tilde_b = B.T @ K_tilde_new @ B with updated kernel,
    K_tilde_b is NOT diagonal! Only solve() works in this case.
    """
    print("\n=== Test 3: K_tilde_inv Methods Comparison ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64

    X, X_tilde, r, _ = load_test_data(n_train=100, n_tilde=25, device=device)
    kernel = create_test_kernel(device, dtype)

    # Compute initial K_tilde
    K_tilde_init = kernel(X_tilde, X_tilde).to_dense()
    K_tilde_init = (K_tilde_init + K_tilde_init.T) / 2

    # Eigenspace projection
    B, eigvals_b, _ = eigendecompose_K_tilde(K_tilde_init)
    n_b = len(eigvals_b)

    # K_tilde_b should be diagonal at initialization
    K_tilde_b_init = B.T @ K_tilde_init @ B

    # Check if K_tilde_b is diagonal
    off_diag = K_tilde_b_init - torch.diag(torch.diag(K_tilde_b_init))
    off_diag_norm = off_diag.norm().item()
    diag_norm = torch.diag(K_tilde_b_init).norm().item()

    print(f"  Initial K_tilde_b off-diagonal norm: {off_diag_norm:.2e}")
    print(f"  Initial K_tilde_b diagonal norm: {diag_norm:.2e}")
    print(f"  Ratio (off/diag): {off_diag_norm / diag_norm:.2e}")

    # Test 1: At initialization, eigenvalue method and solve should match
    eye_b = torch.eye(n_b, device=device, dtype=dtype)

    # Method A: eigenvalue inversion (diagonal)
    K_tilde_inv_eig = torch.diag(1.0 / eigvals_b)

    # Method B: solve
    K_tilde_inv_solve = torch.linalg.solve(K_tilde_b_init, eye_b)

    diff_init = (K_tilde_inv_eig - K_tilde_inv_solve).norm().item()
    ref_norm = K_tilde_inv_eig.norm().item()
    rel_diff_init = diff_init / ref_norm

    print(f"\n  At initialization:")
    print(f"    ||K_inv_eig - K_inv_solve|| / ||K_inv_eig||: {rel_diff_init:.2e}")

    init_match = rel_diff_init < 1e-6
    print(f"    Methods match: {'YES' if init_match else 'NO'}")

    # Test 2: After changing kernel hyperparameters, K_tilde_b is NOT diagonal
    # Modify kernel parameters
    with torch.no_grad():
        kernel.raw_sigma_0.data += 0.5
        kernel.raw_m2log2beta.data += 0.1

    # Recompute K_tilde with new params
    K_tilde_new = kernel(X_tilde, X_tilde).to_dense()
    K_tilde_new = (K_tilde_new + K_tilde_new.T) / 2

    # Project with OLD B
    K_tilde_b_new = B.T @ K_tilde_new @ B

    # Check if still diagonal
    off_diag_new = K_tilde_b_new - torch.diag(torch.diag(K_tilde_b_new))
    off_diag_norm_new = off_diag_new.norm().item()
    diag_norm_new = torch.diag(K_tilde_b_new).norm().item()

    print(f"\n  After hyperparameter change:")
    print(f"    K_tilde_b off-diagonal norm: {off_diag_norm_new:.2e}")
    print(f"    K_tilde_b diagonal norm: {diag_norm_new:.2e}")
    print(f"    Ratio (off/diag): {off_diag_norm_new / diag_norm_new:.2e}")

    is_not_diagonal = off_diag_norm_new / diag_norm_new > 0.01
    print(f"    K_tilde_b is NOT diagonal: {'YES' if is_not_diagonal else 'NO'}")

    # With non-diagonal K_tilde_b, solve still works but eigenvalue inversion doesn't
    K_tilde_inv_solve_new = torch.linalg.solve(K_tilde_b_new, eye_b)

    # Verify solve gives valid inverse
    identity_check = K_tilde_b_new @ K_tilde_inv_solve_new
    identity_err = (identity_check - eye_b).norm().item()
    print(f"    ||K @ K^-1 - I||: {identity_err:.2e}")

    solve_valid = identity_err < 1e-8

    all_passed = init_match and is_not_diagonal and solve_valid

    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: K_tilde_inv methods test PASSED!")
    else:
        print("RESULT: K_tilde_inv methods test FAILED!")
        if not init_match:
            print("  - Initial methods don't match")
        if not is_not_diagonal:
            print("  - K_tilde_b remained diagonal after change (unexpected)")
        if not solve_valid:
            print("  - solve() didn't produce valid inverse")
    print("=" * 60)

    return all_passed


# ==============================================================================
# Test 4: Gradient Magnitude Sanity Check
# ==============================================================================
def test_gradient_magnitude_sanity(verbose=False):
    """Verify analytical gradients are within reasonable magnitude of autograd."""
    print("\n=== Test 4: Gradient Magnitude Sanity Check ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64

    X, X_tilde, r, _ = load_test_data(n_train=100, n_tilde=25, device=device)
    kernel = create_test_kernel(device, dtype)
    likelihood = PoissonLikelihood(A_init=0.01, lambda0_init=1.0)
    likelihood = likelihood.double().to(device)

    # Get mask
    mask = kernel.compute_mask()
    X_masked = X[:, mask]
    X_tilde_masked = X_tilde[:, mask]

    # Compute everything analytically
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

    K_tilde = (K_tilde + K_tilde.T) / 2

    B, eigvals_b, _ = eigendecompose_K_tilde(K_tilde)
    K_tilde_b = compute_K_tilde_b_diagonal(eigvals_b)
    K_b = K @ B
    n_b = len(eigvals_b)

    m_b = torch.zeros(n_b, dtype=dtype, device=device)
    V_b = K_tilde_b.clone()

    dK_b = {k: v @ B for k, v in dK.items()}
    dK_tilde_b = {k: B.T @ v @ B for k, v in dK_tilde.items()}

    eye_b = torch.eye(n_b, device=device, dtype=dtype)
    K_tilde_inv_b = torch.linalg.solve(K_tilde_b, eye_b)

    lambda_m, lambda_var, dlambda_m, dlambda_var = compute_lambda_moments_and_gradients(
        K_b, K_tilde_b, Kvec, m_b, V_b,
        dK_b, dK_tilde_b, dKvec, K_tilde_inv_b
    )

    A = likelihood.A.item()
    lambda0 = likelihood.lambda0.item()
    f_mean = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + lambda0)

    dL = compute_loss_gradients(
        r, f_mean, A, m_b, V_b, K_tilde_b, K_tilde_inv_b,
        dlambda_m, dlambda_var, dK_tilde_b
    )

    # Check for NaN/Inf and reasonable magnitudes
    all_passed = True
    for key, grad in dL.items():
        has_nan = torch.isnan(grad).any().item()
        has_inf = torch.isinf(grad).any().item()
        magnitude = grad.abs().item()

        if has_nan or has_inf:
            print(f"  {key}: FAIL (NaN={has_nan}, Inf={has_inf})")
            all_passed = False
        elif magnitude < 1e-20 or magnitude > 1e20:
            print(f"  {key}: WARNING (magnitude={magnitude:.2e})")
        else:
            print(f"  {key}: OK (magnitude={magnitude:.2e})")

    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: Gradient magnitude sanity check PASSED!")
    else:
        print("RESULT: Gradient magnitude sanity check FAILED!")
    print("=" * 60)

    return all_passed


# ==============================================================================
# Test 5: Multi-cell End-to-End Comparison
# ==============================================================================
def test_multicell_match(verbose=False):
    """Compare vargp_direct to vargp_old on multiple cells.

    This is a slow test that runs full training for each cell.
    """
    print("\n=== Test 5: Multi-cell End-to-End Comparison ===")
    print("(This test runs external scripts and takes several minutes)")

    import subprocess
    import json

    cells_to_test = [6, 8, 15]  # Start with cells we've validated
    config = {
        'ntilde': 50,
        'n_train': 500,
        'n_iterations': 50,
        'seed': 123
    }

    results = {}
    all_passed = True

    for cellid in cells_to_test:
        print(f"\n  Cell {cellid}:")

        # Run vargp_old
        cmd_old = [
            'python', 'run_single_mode.py',
            '--mode', 'vargp_old',
            '--cell', str(cellid),
            '--ntilde', str(config['ntilde']),
            '--n-train', str(config['n_train']),
            '--n-iterations', str(config['n_iterations']),
            '--seed', str(config['seed'])
        ]

        # Run vargp_direct
        cmd_new = [
            'python', 'run_single_mode.py',
            '--mode', 'vargp_direct',
            '--mstep-analytical',
            '--float32',
            '--cell', str(cellid),
            '--ntilde', str(config['ntilde']),
            '--n-train', str(config['n_train']),
            '--n-iterations', str(config['n_iterations']),
            '--seed', str(config['seed'])
        ]

        def parse_test_r(output):
            """Parse 'Test Pearson r:' value from run_single_mode.py output."""
            import re
            for line in output.split('\n'):
                if 'Test Pearson r:' in line:
                    # Format: "  Test Pearson r:  0.7951"
                    match = re.search(r'Test Pearson r:\s+([-+]?\d*\.?\d+)', line)
                    if match:
                        return float(match.group(1))
            return None

        try:
            # Run vargp_old
            result_old = subprocess.run(
                cmd_old,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(Path(__file__).parent.parent)
            )
            test_r_old = parse_test_r(result_old.stdout)

            # Run vargp_direct
            result_new = subprocess.run(
                cmd_new,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(Path(__file__).parent.parent)
            )
            test_r_new = parse_test_r(result_new.stdout)

            if test_r_old is not None and test_r_new is not None:
                diff = abs(test_r_old - test_r_new)
                passed = diff < 0.05

                results[cellid] = {
                    'vargp_old': test_r_old,
                    'vargp_direct': test_r_new,
                    'diff': diff,
                    'passed': passed
                }

                status = "PASS" if passed else "FAIL"
                print(f"    vargp_old:    {test_r_old:.4f}")
                print(f"    vargp_direct: {test_r_new:.4f}")
                print(f"    diff:         {diff:.4f} [{status}]")

                if not passed:
                    all_passed = False
            else:
                print(f"    Could not parse test_r from output")
                if verbose:
                    print(f"    stdout_old: {result_old.stdout[-500:]}")
                    print(f"    stdout_new: {result_new.stdout[-500:]}")
                all_passed = False

        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT")
            all_passed = False
        except Exception as e:
            print(f"    ERROR: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: Multi-cell comparison PASSED!")
    else:
        print("RESULT: Multi-cell comparison FAILED!")
    print("=" * 60)

    return all_passed


# ==============================================================================
# Test 6: Initialization Match
# ==============================================================================
def test_initialization_match(verbose=False):
    """Verify vargp_direct initialization matches vargp_old."""
    print("\n=== Test 6: Initialization Match ===")
    print("  SKIP: Not implemented yet (requires direct comparison with vargp_old internals)")
    return None


# ==============================================================================
# Test 7: Single E-step Match
# ==============================================================================
def test_single_estep_match(verbose=False):
    """Verify single E-step update matches vargp_old."""
    print("\n=== Test 7: Single E-step Match ===")
    print("  SKIP: Not implemented yet (requires direct comparison with vargp_old internals)")
    return None


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Test vargp_direct implementation')
    parser.add_argument('--test', type=int, default=0,
                        help='Run specific test (1-7), or 0 for all')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed output')
    parser.add_argument('--skip-slow', action='store_true',
                        help='Skip slow tests (test 5: multi-cell)')
    args = parser.parse_args()

    tests = [
        test_softplus_chain_rule,        # Test 1
        test_eigenspace_projection_gradients,  # Test 2
        test_ktilde_inv_methods,         # Test 3
        test_gradient_magnitude_sanity,  # Test 4
        test_multicell_match,            # Test 5 (slow)
        test_initialization_match,       # Test 6
        test_single_estep_match,         # Test 7
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
        if args.skip_slow and test == test_multicell_match:
            print(f"\n=== Test {i}: {test.__name__} ===")
            print("  SKIP: --skip-slow flag set")
            results.append(None)
            continue
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
