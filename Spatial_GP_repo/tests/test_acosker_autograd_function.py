"""
Tests for AcoskerCleanFunction (torch.autograd.Function wrapper).
"""
import torch
import pytest
import sys
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject')
from gaussian_processes.Spatial_GP_repo.utils import acosker_clean, acosker_with_grad

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
ATOL = 1e-5


def make_random_C(nx):
    A = torch.randn(nx, nx, device=DEVICE, dtype=DTYPE)
    return A @ A.T + 0.1 * torch.eye(nx, device=DEVICE, dtype=DTYPE)


class TestAcoskerWithGrad:
    """Test acosker_with_grad matches autograd."""

    def test_full_matrix_gradient(self):
        """Custom backward matches autograd for full matrix."""
        with torch.enable_grad():
            n1, n2, nx = 4, 5, 3
            X1 = torch.randn(n1, nx, device=DEVICE, dtype=DTYPE, requires_grad=True)
            X2 = torch.randn(n2, nx, device=DEVICE, dtype=DTYPE)
            C = make_random_C(nx)
            theta = {'sigma_0': 1.0}

            K = acosker_with_grad(theta, X1, X2, C=C, diag=False)
            K.sum().backward()
            grad_custom = X1.grad.clone()

            X1_auto = X1.detach().clone().requires_grad_(True)
            K_auto = acosker_clean(theta, X1_auto, X2, C=C, diag=False)
            K_auto.sum().backward()

            assert torch.allclose(grad_custom, X1_auto.grad, atol=ATOL)

    def test_diagonal_gradient(self):
        """Custom backward matches autograd for diagonal."""
        with torch.enable_grad():
            n, nx = 5, 3
            X1 = torch.randn(n, nx, device=DEVICE, dtype=DTYPE, requires_grad=True)
            C = make_random_C(nx)
            theta = {'sigma_0': 1.0}

            K = acosker_with_grad(theta, X1, None, C=C, diag=True)
            K.sum().backward()
            grad_custom = X1.grad.clone()

            X1_auto = X1.detach().clone().requires_grad_(True)
            K_auto = acosker_clean(theta, X1_auto, None, C=C, diag=True)
            K_auto.sum().backward()

            assert torch.allclose(grad_custom, X1_auto.grad, atol=ATOL)

    def test_X2_none_diag_true(self):
        """Pattern from conditioned_utility_clean line 1143."""
        with torch.enable_grad():
            n, nx = 3, 4
            X1 = torch.randn(n, nx, device=DEVICE, dtype=DTYPE, requires_grad=True)
            C = make_random_C(nx)
            theta = {'sigma_0': 1.0}

            K = acosker_with_grad(theta, X1, None, C=C, diag=True)
            assert K.shape == (n,)
            K.sum().backward()
            assert X1.grad is not None
            assert X1.grad.shape == (n, nx)

    def test_full_matrix_with_xtilde(self):
        """Pattern from conditioned_utility_clean line 1145."""
        with torch.enable_grad():
            n1, n_tilde, nx = 1, 10, 5
            X1 = torch.randn(n1, nx, device=DEVICE, dtype=DTYPE, requires_grad=True)
            xtilde = torch.randn(n_tilde, nx, device=DEVICE, dtype=DTYPE)
            C = make_random_C(nx)
            theta = {'sigma_0': 1.0}

            K = acosker_with_grad(theta, X1, xtilde, C=C, diag=False)
            assert K.shape == (n1, n_tilde)
            K.sum().backward()
            assert X1.grad.shape == (n1, nx)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
