"""
Unit tests for DirectVGPModel wrapper.

Tests that verify actual implementation correctness (not just delegation):
1. expected_firing_rate formula is correct
2. Prediction at test points matches predict_eigenspace() (separate code paths)
3. Variational distribution properties work correctly

Run: conda run -n pytorch_gpytorch python tests/test_direct_vgp_model.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from train import predict_eigenspace
from fstep import compute_f_mean
from eigenspace_model import DirectVGPModel
from eigenspace import EIGVAL_TOL


def load_test_data(n_train=100, n_tilde=25, cellid=8, seed=42, dtype=torch.float32, device='cuda'):
    """Load real PNAS data for testing."""
    data_path = Path(__file__).parent.parent.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = np.load(data_path)

    X_all = torch.tensor(data['images_train'], dtype=dtype, device=device)
    X_all = X_all.reshape(X_all.shape[0], -1)

    X_test_all = torch.tensor(data['images_test'], dtype=dtype, device=device)
    X_test_all = X_test_all.reshape(X_test_all.shape[0], -1)

    torch.manual_seed(seed)
    indices = torch.randperm(X_all.shape[0])[:n_train]
    X = X_all[indices]

    idx_tilde = torch.randperm(n_train)[:n_tilde]
    X_tilde = X[idx_tilde]

    n_test = min(100, X_test_all.shape[0])
    X_test = X_test_all[:n_test]

    return X, X_tilde, X_test, device


def create_test_kernel(device='cuda', dtype=torch.float32):
    """Create a kernel with typical parameters for testing."""
    kernel = ArcCosineKernel(
        n_px_side=108, sigma_0=1.0, Amp=1.0, beta=0.1, rho=0.1,
        eps_0x=0.0, eps_0y=0.0, use_mask=True, gradient_mode='autograd'
    )
    if dtype == torch.float64:
        kernel = kernel.double()
    else:
        kernel = kernel.float()
    return kernel.to(device)


def create_test_likelihood(device='cuda', dtype=torch.float32):
    """Create a likelihood with typical parameters for testing."""
    likelihood = PoissonLikelihood(A_init=0.01, lambda0_init=1.0)
    if dtype == torch.float64:
        likelihood = likelihood.double()
    else:
        likelihood = likelihood.float()
    return likelihood.to(device)


# ==============================================================================
# Test 1: Expected Firing Rate Formula
# ==============================================================================
def test_expected_firing_rate(verbose=False):
    """Verify expected_firing_rate implements the correct formula."""
    print("\n=== Test 1: Expected Firing Rate Formula ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    X, X_tilde, _, _ = load_test_data(n_train=100, n_tilde=25, device=device, dtype=dtype)
    kernel = create_test_kernel(device, dtype)
    likelihood = create_test_likelihood(device, dtype)

    model = DirectVGPModel(kernel, likelihood, X, X_tilde, EIGVAL_TOL)
    posterior = model(X)

    # Method 1: Using wrapper method
    f_mean_wrapper = model.likelihood.expected_firing_rate(posterior)

    # Method 2: Manual formula
    A = model.likelihood.A.squeeze()
    lambda0 = model.likelihood.lambda0.squeeze()
    f_mean_manual = torch.exp(A * posterior.mean + 0.5 * A * A * posterior.variance + lambda0)

    # Method 3: Using compute_f_mean function
    f_mean_func = compute_f_mean(posterior.mean, posterior.variance, A, lambda0)

    wrapper_vs_manual = torch.equal(f_mean_wrapper, f_mean_manual)
    wrapper_vs_func = torch.equal(f_mean_wrapper, f_mean_func)

    print(f"  wrapper == manual formula: {'PASS' if wrapper_vs_manual else 'FAIL'}")
    print(f"  wrapper == compute_f_mean: {'PASS' if wrapper_vs_func else 'FAIL'}")

    passed = wrapper_vs_manual and wrapper_vs_func
    print(f"\n{'PASS' if passed else 'FAIL'}: Expected firing rate test")
    return passed


# ==============================================================================
# Test 2: Prediction at Test Points (Different Code Paths)
# ==============================================================================
def test_prediction_test_points(verbose=False):
    """Verify wrapper prediction matches predict_eigenspace() - SEPARATE implementations."""
    print("\n=== Test 2: Prediction at Test Points ===")
    print("  (Compares EigenspacePosterior._compute_moments vs predict_eigenspace)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    X, X_tilde, X_test, _ = load_test_data(n_train=100, n_tilde=25, device=device, dtype=dtype)
    kernel = create_test_kernel(device, dtype)
    likelihood = create_test_likelihood(device, dtype)

    model = DirectVGPModel(kernel, likelihood, X, X_tilde, EIGVAL_TOL)

    # Method 1: Wrapper (EigenspacePosterior._compute_moments - my new code)
    posterior_test = model(X_test)

    # Method 2: predict_eigenspace (existing code in direct_vargp.py)
    pred_direct = predict_eigenspace(kernel, likelihood, model.state, X_tilde, X_test)

    mean_exact = torch.equal(posterior_test.mean, pred_direct['lambda_m'])
    var_exact = torch.equal(posterior_test.variance, pred_direct['lambda_var'])

    f_mean_wrapper = model.likelihood.expected_firing_rate(posterior_test)
    f_exact = torch.equal(f_mean_wrapper, pred_direct['f_pred'])

    print(f"  lambda_m exact match: {'PASS' if mean_exact else 'FAIL'}")
    print(f"  lambda_var exact match: {'PASS' if var_exact else 'FAIL'}")
    print(f"  f_pred exact match: {'PASS' if f_exact else 'FAIL'}")

    if not mean_exact:
        diff = (posterior_test.mean - pred_direct['lambda_m']).abs().max().item()
        print(f"    lambda_m max diff: {diff:.2e}")
    if not var_exact:
        diff = (posterior_test.variance - pred_direct['lambda_var']).abs().max().item()
        print(f"    lambda_var max diff: {diff:.2e}")

    passed = mean_exact and var_exact and f_exact
    print(f"\n{'PASS' if passed else 'FAIL'}: Prediction test points")
    return passed


# ==============================================================================
# Test 3: Variational Distribution Interface
# ==============================================================================
def test_variational_distribution(verbose=False):
    """Verify variational_distribution properties return correct values."""
    print("\n=== Test 3: Variational Distribution Interface ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    X, X_tilde, _, _ = load_test_data(n_train=100, n_tilde=25, device=device, dtype=dtype)
    kernel = create_test_kernel(device, dtype)
    likelihood = create_test_likelihood(device, dtype)

    model = DirectVGPModel(kernel, likelihood, X, X_tilde, EIGVAL_TOL)

    # Set non-zero values
    torch.manual_seed(42)
    with torch.no_grad():
        model.state.m_b.copy_(torch.randn_like(model.state.m_b))
        model.state.V_b.copy_(torch.eye(model.state.V_b.shape[0], device=device, dtype=dtype) * 0.5)

    var_dist = model.variational_distribution
    all_passed = True

    # mean_eigenspace is state.m_b (same object)
    check1 = var_dist.mean_eigenspace is model.state.m_b
    print(f"  mean_eigenspace is state.m_b: {'PASS' if check1 else 'FAIL'}")
    all_passed = all_passed and check1

    # mean == B @ m_b
    mean_expected = model.state.B @ model.state.m_b
    check2 = torch.equal(var_dist.mean, mean_expected)
    print(f"  mean == B @ m_b: {'PASS' if check2 else 'FAIL'}")
    all_passed = all_passed and check2

    # covariance_eigenspace is state.V_b (same object)
    check3 = var_dist.covariance_eigenspace is model.state.V_b
    print(f"  covariance_eigenspace is state.V_b: {'PASS' if check3 else 'FAIL'}")
    all_passed = all_passed and check3

    # covariance == B @ V_b @ B.T
    cov_expected = model.state.B @ model.state.V_b @ model.state.B.T
    check4 = torch.equal(var_dist.covariance, cov_expected)
    print(f"  covariance == B @ V_b @ B.T: {'PASS' if check4 else 'FAIL'}")
    all_passed = all_passed and check4

    print(f"\n{'PASS' if all_passed else 'FAIL'}: Variational distribution test")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Test DirectVGPModel wrapper')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    tests = [
        test_expected_firing_rate,
        test_prediction_test_points,
        test_variational_distribution,
    ]

    results = [test(verbose=args.verbose) for test in tests]

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for test, result in zip(tests, results):
        print(f"  {test.__name__}: {'PASS' if result else 'FAIL'}")

    passed = sum(results)
    failed = len(results) - passed
    print(f"\n  {passed} passed, {failed} failed")

    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
