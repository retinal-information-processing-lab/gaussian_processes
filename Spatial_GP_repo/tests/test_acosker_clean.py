"""
Test suite for acosker_clean() kernel function.

Tests:
1. Component tests (quadratic form, diagonal kernel)
2. Comparison tests against original acosker()
3. Gradient correctness via autograd
4. Edge cases and shape tests
"""

import torch
import pytest
import numpy as np

# Import the functions to test
import sys
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject')
from gaussian_processes.Spatial_GP_repo.utils import acosker, acosker_clean

# Test configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64  # Use float64 for numerical gradient tests
ATOL = 1e-5
RTOL = 1e-4


def make_random_C(nx, device=DEVICE, dtype=DTYPE):
    """Create a random positive definite symmetric matrix C."""
    A = torch.randn(nx, nx, device=device, dtype=dtype)
    C = A @ A.T + 0.1 * torch.eye(nx, device=device, dtype=dtype)
    return C


def make_test_inputs(n1, n2, nx, device=DEVICE, dtype=DTYPE):
    """Create random test inputs."""
    X1 = torch.randn(n1, nx, device=device, dtype=dtype)
    X2 = torch.randn(n2, nx, device=device, dtype=dtype)
    C = make_random_C(nx, device, dtype)
    theta = {'sigma_0': 1.0}
    return X1, X2, C, theta


# =============================================================================
# Test 1: Diagonal kernel values
# =============================================================================
class TestDiagonalKernel:
    """Test diagonal case K(x, x) = v_x = x^T C x + σ₀²"""

    def test_diagonal_values(self):
        """K_diag[i] should equal X[i] @ C @ X[i].T + sigma_0^2"""
        n, nx = 10, 5
        X1, _, C, theta = make_test_inputs(n, n, nx)
        sigma_0_sq = theta['sigma_0'] ** 2

        K_diag = acosker_clean(theta, X1, C=C, diag=True)

        # Manual computation
        expected = torch.zeros(n, device=DEVICE, dtype=DTYPE)
        for i in range(n):
            expected[i] = X1[i] @ C @ X1[i] + sigma_0_sq

        assert torch.allclose(K_diag, expected, atol=ATOL, rtol=RTOL), \
            f"Diagonal mismatch: max diff = {(K_diag - expected).abs().max()}"

    def test_diagonal_gradient(self):
        """∇_x K(x, x) = 2 C x"""
        n, nx = 10, 5
        X1, _, C, theta = make_test_inputs(n, n, nx)

        K_diag, dK = acosker_clean(theta, X1, C=C, diag=True, compute_grad=True)

        # Manual computation: dK[i, :] = 2 * C @ X[i]
        expected = 2 * (X1 @ C)

        assert K_diag.shape == (n,)
        assert dK.shape == (n, nx)
        assert torch.allclose(dK, expected, atol=ATOL, rtol=RTOL), \
            f"Diagonal gradient mismatch: max diff = {(dK - expected).abs().max()}"


# =============================================================================
# Test 2: Full matrix kernel symmetry
# =============================================================================
class TestKernelSymmetry:
    """Test that K(x, x') = K(x', x)"""

    def test_self_covariance_symmetric(self):
        """K(X, X) should be symmetric"""
        n, nx = 8, 4
        X1, _, C, theta = make_test_inputs(n, n, nx)

        K = acosker_clean(theta, X1, X2=X1, C=C, diag=False)

        assert torch.allclose(K, K.T, atol=ATOL, rtol=RTOL), \
            f"K(X, X) not symmetric: max diff = {(K - K.T).abs().max()}"

    def test_cross_covariance_transpose(self):
        """K(X1, X2) should equal K(X2, X1).T"""
        n1, n2, nx = 5, 7, 4
        X1, X2, C, theta = make_test_inputs(n1, n2, nx)

        K12 = acosker_clean(theta, X1, X2=X2, C=C, diag=False)
        K21 = acosker_clean(theta, X2, X2=X1, C=C, diag=False)

        assert torch.allclose(K12, K21.T, atol=ATOL, rtol=RTOL), \
            f"K(X1,X2) != K(X2,X1).T: max diff = {(K12 - K21.T).abs().max()}"


# =============================================================================
# Test 3: Comparison with original acosker()
# =============================================================================
class TestMatchesOriginal:
    """Compare acosker_clean outputs to original acosker"""

    def test_full_matrix_matches(self):
        """K values should match original implementation"""
        n1, n2, nx = 6, 8, 5
        X1, X2, C, theta = make_test_inputs(n1, n2, nx)

        # New implementation
        K_new = acosker_clean(theta, X1, X2=X2, C=C, diag=False)

        # Original (note: original transposes internally)
        K_orig = acosker(theta, X1, x2=X2, C=C, dC=None, diag=False)

        assert torch.allclose(K_new, K_orig, atol=ATOL, rtol=RTOL), \
            f"Full matrix mismatch: max diff = {(K_new - K_orig).abs().max()}"

    def test_diagonal_matches(self):
        """Diagonal values should match original implementation"""
        n, nx = 10, 5
        X1, _, C, theta = make_test_inputs(n, n, nx)

        K_new = acosker_clean(theta, X1, C=C, diag=True)
        K_orig = acosker(theta, X1, x2=None, C=C, dC=None, diag=True)

        assert torch.allclose(K_new, K_orig, atol=ATOL, rtol=RTOL), \
            f"Diagonal mismatch: max diff = {(K_new - K_orig).abs().max()}"

    def test_gradient_matches_original_full(self):
        """
        Compare gradient w.r.t. X1 between acosker_clean and original acosker.

        Strategy:
        - acosker_clean(X1, X2, compute_grad=True) gives dK/dX1
        - acosker(X2, X1, get_dK_x=True) with SWAPPED args gives dK/dX1 (as x2)

        Relationship:
        - K_orig has shape (n2, n1), K_new has shape (n1, n2)
        - K_orig.T == K_new (by kernel symmetry)
        - dK_orig has shape (n2, n1, nx), dK_new has shape (n1, n2, nx)
        - dK_orig.permute(1, 0, 2) == dK_new
        """
        n1, n2, nx = 5, 7, 4
        X1, X2, C, theta = make_test_inputs(n1, n2, nx)

        # New implementation: gradient w.r.t. X1 (first argument)
        K_new, dK_new = acosker_clean(theta, X1, X2=X2, C=C, diag=False, compute_grad=True)

        # Original with SWAPPED arguments: gradient w.r.t. X1 (now second argument)
        # acosker(theta, x1=X2, x2=X1, ...) returns gradient w.r.t. X1
        K_orig, dK_orig = acosker(theta, X2, x2=X1, C=C, dC=None, diag=False, get_dK_x=True)

        # Verify K values match (with transpose due to swapped args)
        assert torch.allclose(K_orig.T, K_new, atol=ATOL, rtol=RTOL), \
            f"K mismatch in gradient test: max diff = {(K_orig.T - K_new).abs().max()}"

        # Verify gradient values match (with permutation due to swapped args)
        # dK_orig[j,i,k] corresponds to dK_new[i,j,k]
        dK_orig_permuted = dK_orig.permute(1, 0, 2)  # (n2, n1, nx) -> (n1, n2, nx)

        assert torch.allclose(dK_orig_permuted, dK_new, atol=ATOL, rtol=RTOL), \
            f"Gradient mismatch with original acosker:\n" \
            f"  max diff = {(dK_orig_permuted - dK_new).abs().max()}\n" \
            f"  dK_orig_permuted shape: {dK_orig_permuted.shape}\n" \
            f"  dK_new shape: {dK_new.shape}"

    def test_gradient_matches_original_diag(self):
        """
        Compare diagonal gradient between acosker_clean and original acosker.

        Both compute ∂K(x,x)/∂x = 2Cx, so direct comparison.
        """
        n, nx = 8, 5
        X1, _, C, theta = make_test_inputs(n, n, nx)

        # New implementation
        K_new, dK_new = acosker_clean(theta, X1, C=C, diag=True, compute_grad=True)

        # Original implementation
        K_orig, dK_orig = acosker(theta, X1, x2=None, C=C, dC=None, diag=True, get_dK_x=True)

        # Verify K values match
        assert torch.allclose(K_new, K_orig, atol=ATOL, rtol=RTOL), \
            f"Diagonal K mismatch: max diff = {(K_new - K_orig).abs().max()}"

        # Verify gradient values match directly
        assert torch.allclose(dK_new, dK_orig, atol=ATOL, rtol=RTOL), \
            f"Diagonal gradient mismatch with original:\n" \
            f"  max diff = {(dK_new - dK_orig).abs().max()}\n" \
            f"  dK_new shape: {dK_new.shape}\n" \
            f"  dK_orig shape: {dK_orig.shape}"


# =============================================================================
# Test 4: Gradient correctness via autograd
# =============================================================================
class TestGradientAutograd:
    """Compare analytical gradient to torch.autograd"""

    def test_full_matrix_gradient_vs_autograd(self):
        """dK_X1 should match autograd gradient"""
        n1, n2, nx = 4, 5, 3
        X1, X2, C, theta = make_test_inputs(n1, n2, nx)

        # Enable gradients for this test (utils.py disables them globally)
        with torch.enable_grad():
            X1.requires_grad_(True)

            # Compute with analytical gradient
            K_analytic, dK_analytic = acosker_clean(theta, X1, X2=X2, C=C,
                                                     diag=False, compute_grad=True)

            # Compute with autograd for each K[i,j]
            # dK_analytic[i,j,k] = ∂K[i,j]/∂X1[i,k]
            for i in range(n1):
                for j in range(n2):
                    # Recompute K with fresh graph
                    X1_fresh = X1.detach().clone().requires_grad_(True)
                    K = acosker_clean(theta, X1_fresh, X2=X2, C=C, diag=False)

                    # Get gradient of K[i,j] w.r.t. X1
                    grad_output = torch.zeros_like(K)
                    grad_output[i, j] = 1.0
                    K.backward(grad_output)

                    # The gradient should only be non-zero for X1[i, :]
                    autograd_result = X1_fresh.grad[i, :]
                    analytic_result = dK_analytic[i, j, :]

                    assert torch.allclose(analytic_result, autograd_result, atol=ATOL, rtol=RTOL), \
                        f"Gradient mismatch at [{i},{j}]: " \
                        f"analytic={analytic_result}, autograd={autograd_result}"

    def test_diagonal_gradient_vs_autograd(self):
        """Diagonal dK should match autograd gradient"""
        n, nx = 5, 3
        X1, _, C, theta = make_test_inputs(n, n, nx)

        # Enable gradients for this test (utils.py disables them globally)
        with torch.enable_grad():
            # Analytical
            K_analytic, dK_analytic = acosker_clean(theta, X1, C=C, diag=True, compute_grad=True)

            # Autograd for each K_diag[i]
            for i in range(n):
                X1_fresh = X1.detach().clone().requires_grad_(True)
                K = acosker_clean(theta, X1_fresh, C=C, diag=True)
                K[i].backward()

                autograd_result = X1_fresh.grad[i, :]
                analytic_result = dK_analytic[i, :]

                assert torch.allclose(analytic_result, autograd_result, atol=ATOL, rtol=RTOL), \
                    f"Diagonal gradient mismatch at [{i}]: max diff = {(analytic_result - autograd_result).abs().max()}"


# =============================================================================
# Test 5: Numerical gradient (finite differences)
# =============================================================================
class TestNumericalGradient:
    """Verify gradient using finite differences"""

    def test_finite_difference_full(self):
        """Check gradient against finite differences for full matrix"""
        n1, n2, nx = 3, 4, 3
        X1, X2, C, theta = make_test_inputs(n1, n2, nx)
        eps = 1e-6

        K, dK = acosker_clean(theta, X1, X2=X2, C=C, diag=False, compute_grad=True)

        # Check gradient numerically for a few elements
        for i in range(min(2, n1)):
            for j in range(min(2, n2)):
                for k in range(nx):
                    X1_plus = X1.clone()
                    X1_plus[i, k] += eps
                    K_plus = acosker_clean(theta, X1_plus, X2=X2, C=C, diag=False)

                    X1_minus = X1.clone()
                    X1_minus[i, k] -= eps
                    K_minus = acosker_clean(theta, X1_minus, X2=X2, C=C, diag=False)

                    numerical_grad = (K_plus[i, j] - K_minus[i, j]) / (2 * eps)
                    analytic_grad = dK[i, j, k]

                    assert torch.allclose(analytic_grad, numerical_grad, atol=1e-4, rtol=1e-3), \
                        f"Numerical gradient mismatch at [{i},{j},{k}]: " \
                        f"analytic={analytic_grad.item():.6f}, numerical={numerical_grad.item():.6f}"

    def test_finite_difference_diagonal(self):
        """Check diagonal gradient against finite differences"""
        n, nx = 4, 3
        X1, _, C, theta = make_test_inputs(n, n, nx)
        eps = 1e-6

        K, dK = acosker_clean(theta, X1, C=C, diag=True, compute_grad=True)

        for i in range(min(2, n)):
            for k in range(nx):
                X1_plus = X1.clone()
                X1_plus[i, k] += eps
                K_plus = acosker_clean(theta, X1_plus, C=C, diag=True)

                X1_minus = X1.clone()
                X1_minus[i, k] -= eps
                K_minus = acosker_clean(theta, X1_minus, C=C, diag=True)

                numerical_grad = (K_plus[i] - K_minus[i]) / (2 * eps)
                analytic_grad = dK[i, k]

                assert torch.allclose(analytic_grad, numerical_grad, atol=1e-4, rtol=1e-3), \
                    f"Diagonal numerical gradient mismatch at [{i},{k}]"


# =============================================================================
# Test 6: Edge cases
# =============================================================================
class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_point_n1(self):
        """Test with n1=1"""
        n1, n2, nx = 1, 5, 4
        X1, X2, C, theta = make_test_inputs(n1, n2, nx)

        K = acosker_clean(theta, X1, X2=X2, C=C, diag=False)
        assert K.shape == (1, 5)

        K, dK = acosker_clean(theta, X1, X2=X2, C=C, diag=False, compute_grad=True)
        assert dK.shape == (1, 5, 4)

    def test_single_point_n2(self):
        """Test with n2=1"""
        n1, n2, nx = 5, 1, 4
        X1, X2, C, theta = make_test_inputs(n1, n2, nx)

        K = acosker_clean(theta, X1, X2=X2, C=C, diag=False)
        assert K.shape == (5, 1)

        K, dK = acosker_clean(theta, X1, X2=X2, C=C, diag=False, compute_grad=True)
        assert dK.shape == (5, 1, 4)

    def test_identical_points(self):
        """When X1 = X2, diagonal of K should equal K_diag"""
        n, nx = 6, 4
        X1, _, C, theta = make_test_inputs(n, n, nx)

        K_full = acosker_clean(theta, X1, X2=X1, C=C, diag=False)
        K_diag = acosker_clean(theta, X1, C=C, diag=True)

        assert torch.allclose(K_full.diag(), K_diag, atol=ATOL, rtol=RTOL), \
            f"Diagonal extraction mismatch"

    def test_identity_C(self):
        """Test with C = I (simplest case)"""
        n1, n2, nx = 4, 5, 3
        X1, X2, _, theta = make_test_inputs(n1, n2, nx)
        C = torch.eye(nx, device=DEVICE, dtype=DTYPE)

        K = acosker_clean(theta, X1, X2=X2, C=C, diag=False)
        K_no_C = acosker_clean(theta, X1, X2=X2, C=None, diag=False)

        assert torch.allclose(K, K_no_C, atol=ATOL, rtol=RTOL)

    def test_no_C_provided(self):
        """Test that C=None defaults to identity"""
        n, nx = 5, 3
        X1, _, _, theta = make_test_inputs(n, n, nx)

        K = acosker_clean(theta, X1, C=None, diag=True)
        assert K.shape == (n,)


# =============================================================================
# Test 7: Output shapes
# =============================================================================
class TestOutputShapes:
    """Verify all output tensor shapes are correct"""

    def test_full_kernel_shape(self):
        """K should have shape (n1, n2)"""
        n1, n2, nx = 7, 11, 5
        X1, X2, C, theta = make_test_inputs(n1, n2, nx)

        K = acosker_clean(theta, X1, X2=X2, C=C, diag=False)
        assert K.shape == (n1, n2), f"Expected {(n1, n2)}, got {K.shape}"

    def test_diagonal_kernel_shape(self):
        """K_diag should have shape (n,)"""
        n, nx = 9, 4
        X1, _, C, theta = make_test_inputs(n, n, nx)

        K = acosker_clean(theta, X1, C=C, diag=True)
        assert K.shape == (n,), f"Expected {(n,)}, got {K.shape}"

    def test_full_gradient_shape(self):
        """dK should have shape (n1, n2, nx)"""
        n1, n2, nx = 7, 11, 5
        X1, X2, C, theta = make_test_inputs(n1, n2, nx)

        K, dK = acosker_clean(theta, X1, X2=X2, C=C, diag=False, compute_grad=True)
        assert dK.shape == (n1, n2, nx), f"Expected {(n1, n2, nx)}, got {dK.shape}"

    def test_diagonal_gradient_shape(self):
        """dK_diag should have shape (n, nx)"""
        n, nx = 9, 4
        X1, _, C, theta = make_test_inputs(n, n, nx)

        K, dK = acosker_clean(theta, X1, C=C, diag=True, compute_grad=True)
        assert dK.shape == (n, nx), f"Expected {(n, nx)}, got {dK.shape}"


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    print(f"Running tests on device: {DEVICE}")
    print(f"Using dtype: {DTYPE}")
    print()

    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])
