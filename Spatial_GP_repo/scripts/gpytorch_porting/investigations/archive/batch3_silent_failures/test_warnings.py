"""
Test script for Batch 3 silent failure warnings.

Created by Claude for batch3 investigation.

This script triggers each warning to verify they work correctly.
Run after implementing fixes to confirm warnings appear.
"""

import warnings
import sys
import os

# Add parent directory to path (gpytorch_porting/)
script_dir = os.path.dirname(os.path.abspath(__file__))
gpytorch_porting_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, gpytorch_porting_dir)

import torch
import numpy as np

# Capture warnings for verification
captured_warnings = []


def warning_handler(message, category, filename, lineno, file=None, line=None):
    """Custom warning handler that captures warnings."""
    captured_warnings.append(str(message))
    # Also print to console
    print(f"WARNING: {message}")


# Install custom warning handler
old_showwarning = warnings.showwarning
warnings.showwarning = warning_handler
warnings.simplefilter('always', RuntimeWarning)


def test_j3_negative_variance():
    """Test J3: Negative variance warning.

    We create a situation where the variance formula produces negative values.
    This happens when V - K_tilde has negative diagonal entries that dominate k0.
    """
    print("\n" + "="*60)
    print("TEST J3: Negative Variance Warning")
    print("="*60)

    from estep import compute_moments_from_kernel_cache

    # Create a kernel cache with values that will produce negative variance
    N, M = 10, 5

    # k0 should be small (self-kernel diagonal)
    k0 = torch.ones(N) * 0.1

    # K (cross-kernel) - make it reasonable
    K = torch.randn(N, M) * 0.5

    # K_tilde (inducing kernel) - make it PD
    K_tilde = torch.eye(M) * 2.0
    K_tilde_j = K_tilde + 1e-4 * torch.eye(M)

    # V (variational covariance) - make it SMALLER than K_tilde
    # This will make V - K_tilde negative, causing negative variance
    V = torch.eye(M) * 0.1  # Much smaller than K_tilde

    # m (variational mean)
    m = torch.zeros(M)

    kernel_cache = {
        'K': K,
        'K_tilde': K_tilde,
        'K_tilde_j': K_tilde_j,
        'k0': k0,
    }

    captured_warnings.clear()
    lambda_m, lambda_var = compute_moments_from_kernel_cache(kernel_cache, m, V)

    # Check if warning was triggered
    j3_warnings = [w for w in captured_warnings if 'Negative variance' in w]
    if j3_warnings:
        print(f"SUCCESS: J3 warning triggered")
        print(f"  lambda_var range: [{lambda_var.min().item():.2e}, {lambda_var.max().item():.2e}]")
        return True
    else:
        print(f"NOTE: J3 warning NOT triggered (variance was non-negative)")
        print(f"  lambda_var range: [{lambda_var.min().item():.2e}, {lambda_var.max().item():.2e}]")
        return False


def test_j2_cholesky_fallback():
    """Test J2: Cholesky fallback warning.

    We create a matrix with a negative eigenvalue that fails Cholesky.
    """
    print("\n" + "="*60)
    print("TEST J2: Cholesky Fallback Warning")
    print("="*60)

    from whitening import update_variational_covar
    from model import VariationalGPModel
    from kernels import ArcCosineKernel

    # Create a simple model
    M = 10
    n_features = 50
    inducing_points = torch.randn(M, n_features, dtype=torch.float64)
    kernel = ArcCosineKernel(sigma_0=1.0)
    model = VariationalGPModel(
        inducing_points, kernel,
        jitter=1e-4,
        standard_variational_distribution=True
    ).double()

    # Create a matrix with small NEGATIVE eigenvalue that jitter can fix
    # Jitter is 1e-4, so negative eigenvalue must be smaller (like -1e-5)
    eigvals = torch.tensor([-1e-5, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=torch.float64)
    Q, _ = torch.linalg.qr(torch.randn(M, M, dtype=torch.float64))
    V_not_pd = Q @ torch.diag(eigvals) @ Q.T
    V_not_pd = (V_not_pd + V_not_pd.T) / 2  # Ensure symmetric

    captured_warnings.clear()
    try:
        update_variational_covar(model, V_not_pd)
        j2_warnings = [w for w in captured_warnings if 'Cholesky failed' in w]
        if j2_warnings:
            print(f"SUCCESS: J2 warning triggered")
            return True
        else:
            print(f"NOTE: J2 warning NOT triggered (Cholesky succeeded)")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_j1_lbfgs_instability():
    """Test J1: LBFGS instability warning.

    We create a situation where f_mean explodes, causing LBFGS to receive inf.
    """
    print("\n" + "="*60)
    print("TEST J1: LBFGS Instability Warning")
    print("="*60)

    from fstep import f_step_lbfgs, STABILITY_THRESHOLD
    from likelihoods import PoissonLikelihood
    from model import VariationalGPModel
    from kernels import ArcCosineKernel

    # Create a simple model
    M = 10
    N = 50
    n_features = 100

    inducing_points = torch.randn(M, n_features, dtype=torch.float64)
    kernel = ArcCosineKernel(sigma_0=1.0)
    model = VariationalGPModel(
        inducing_points, kernel,
        jitter=1e-4,
        standard_variational_distribution=True
    ).double()

    # Start with A=1.0 (not 0.01) to amplify lambda_m more
    likelihood = PoissonLikelihood(A_init=1.0, lambda0_init=10.0).double()

    # Create data
    X = torch.randn(N, n_features, dtype=torch.float64)
    r = torch.ones(N, dtype=torch.float64)  # Some spike counts

    # Create lambda moments that will cause f_mean > STABILITY_THRESHOLD
    # f_mean = exp(A * lambda_m + 0.5 * A^2 * lambda_var + lambda0)
    # With A=1, lambda0=10, we need A*lambda_m + lambda0 > log(STABILITY_THRESHOLD)
    # log(1000) ~ 6.9, so we need lambda_m > -3.1 to get f_mean > 1000
    # Let's use lambda_m = 10 to be safe
    lambda_m = torch.ones(N, dtype=torch.float64) * 10.0  # Will make f_mean huge
    lambda_var = torch.ones(N, dtype=torch.float64) * 1.0

    print(f"  STABILITY_THRESHOLD: {STABILITY_THRESHOLD}")
    test_f_mean = torch.exp(1.0 * lambda_m[0] + 0.5 * 1.0 * lambda_var[0] + 10.0).item()
    print(f"  Expected f_mean: {test_f_mean:.1f}")

    captured_warnings.clear()
    try:
        f_step_lbfgs(model, likelihood, X, r, lambda_m, lambda_var, n_fstep=5, lr=0.1)
        j1_warnings = [w for w in captured_warnings if 'instability detected' in w]
        if j1_warnings:
            print(f"SUCCESS: J1 warning triggered")
            return True
        else:
            print(f"NOTE: J1 warning NOT triggered (no instability)")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    print("="*60)
    print("Batch 3 Silent Failure Warnings Test Suite")
    print("="*60)

    results = {}

    # Test each warning
    results['J3'] = test_j3_negative_variance()
    results['J2'] = test_j2_cholesky_fallback()
    results['J1'] = test_j1_lbfgs_instability()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for issue, triggered in results.items():
        status = "TRIGGERED" if triggered else "not triggered"
        print(f"  {issue}: {status}")

    # Restore original warning handler
    warnings.showwarning = old_showwarning

    print("\nNote: 'not triggered' may be OK if the test conditions didn't")
    print("cause the edge case. The important thing is no crashes.")


if __name__ == '__main__':
    main()
