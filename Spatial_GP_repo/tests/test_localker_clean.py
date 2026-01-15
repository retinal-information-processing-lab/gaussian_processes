"""
Test suite for localker_clean() and LocalkerCleanFunction.

Tests:
1. Forward pass equivalence: localker_clean vs localker
2. Gradient equivalence: analytical dC vs autograd gradients
3. Mask consistency: same mask produced by both functions
4. Edge cases: different n_px_side values
"""

import torch
import pytest
import numpy as np

# Import the functions to test
import sys
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject')
from gaussian_processes.Spatial_GP_repo.utils import localker, localker_clean
from config.config import TORCH_DTYPE, DEVICE as CONFIG_DEVICE

# Test configuration - use same dtype as utils.py to avoid mismatch
DEVICE = CONFIG_DEVICE
DTYPE = TORCH_DTYPE
ATOL = 1e-5
RTOL = 1e-4


def make_test_theta(device=DEVICE, dtype=DTYPE, requires_grad=False):
    """Create test hyperparameters dictionary."""
    theta = {
        'Amp': torch.tensor(1.0, device=device, dtype=dtype, requires_grad=requires_grad),
        '-2log2beta': torch.tensor(-2.0, device=device, dtype=dtype, requires_grad=requires_grad),
        '-log2rho2': torch.tensor(-1.0, device=device, dtype=dtype, requires_grad=requires_grad),
        'eps_0x': torch.tensor(0.1, device=device, dtype=dtype, requires_grad=requires_grad),
        'eps_0y': torch.tensor(-0.1, device=device, dtype=dtype, requires_grad=requires_grad),
        'sigma_0': torch.tensor(0.5, device=device, dtype=dtype, requires_grad=requires_grad),
    }
    return theta


def make_theta_limits():
    """Create theta limits for localker."""
    theta_higher_lims = {
        'Amp': 10.0, '-2log2beta': 5.0, '-log2rho2': 5.0,
        'eps_0x': 1.0, 'eps_0y': 1.0, 'sigma_0': 5.0
    }
    theta_lower_lims = {
        'Amp': 0.01, '-2log2beta': -5.0, '-log2rho2': -5.0,
        'eps_0x': -1.0, 'eps_0y': -1.0, 'sigma_0': 0.01
    }
    return theta_higher_lims, theta_lower_lims


# =============================================================================
# Test 1: Forward Pass Equivalence
# =============================================================================
class TestForwardEquivalence:
    """Test that localker_clean produces same C and mask as localker."""

    def test_C_equivalence_small(self):
        """Test C matrix equivalence with small image (10x10)."""
        n_px_side = 10
        theta = make_test_theta(requires_grad=False)
        theta_higher_lims, theta_lower_lims = make_theta_limits()

        # Original localker
        C_orig, mask_orig = localker(theta, theta_higher_lims, theta_lower_lims, n_px_side, grad=False)

        # New localker_clean
        C_new, mask_new = localker_clean(theta, n_px_side)

        assert torch.allclose(C_orig, C_new, atol=ATOL, rtol=RTOL), \
            f"C matrix mismatch: max diff = {(C_orig - C_new).abs().max()}"

    def test_C_equivalence_medium(self):
        """Test C matrix equivalence with medium image (50x50)."""
        n_px_side = 50
        theta = make_test_theta(requires_grad=False)
        theta_higher_lims, theta_lower_lims = make_theta_limits()

        C_orig, mask_orig = localker(theta, theta_higher_lims, theta_lower_lims, n_px_side, grad=False)
        C_new, mask_new = localker_clean(theta, n_px_side)

        assert torch.allclose(C_orig, C_new, atol=ATOL, rtol=RTOL), \
            f"C matrix mismatch: max diff = {(C_orig - C_new).abs().max()}"

    def test_mask_equivalence(self):
        """Test that masks are identical."""
        n_px_side = 30
        theta = make_test_theta(requires_grad=False)
        theta_higher_lims, theta_lower_lims = make_theta_limits()

        C_orig, mask_orig = localker(theta, theta_higher_lims, theta_lower_lims, n_px_side, grad=False)
        C_new, mask_new = localker_clean(theta, n_px_side)

        assert torch.all(mask_orig == mask_new), \
            f"Mask mismatch: {mask_orig.sum()} vs {mask_new.sum()} active pixels"

    def test_C_shape(self):
        """Test that C has correct shape (n_masked x n_masked)."""
        n_px_side = 20
        theta = make_test_theta(requires_grad=False)

        C, mask = localker_clean(theta, n_px_side)
        n_masked = mask.sum().item()

        assert C.shape == (n_masked, n_masked), \
            f"C shape mismatch: expected ({n_masked}, {n_masked}), got {C.shape}"

    def test_C_symmetry(self):
        """Test that C is symmetric."""
        n_px_side = 25
        theta = make_test_theta(requires_grad=False)

        C, mask = localker_clean(theta, n_px_side)

        assert torch.allclose(C, C.T, atol=1e-10), \
            f"C is not symmetric: max diff = {(C - C.T).abs().max()}"


# =============================================================================
# Test 2: Gradient Equivalence
# =============================================================================
class TestGradientEquivalence:
    """Test that autograd gradients match analytical dC from localker."""

    def test_gradient_Amp(self):
        """Test gradient w.r.t. Amp."""
        with torch.enable_grad():
            n_px_side = 20
            theta = make_test_theta(requires_grad=True)
            theta_higher_lims, theta_lower_lims = make_theta_limits()

            # Get analytical gradient from localker
            C_orig, mask_orig, dC_orig = localker(theta, theta_higher_lims, theta_lower_lims, n_px_side, grad=True)

            # Reset gradients and compute autograd gradient
            for key in theta:
                if theta[key].grad is not None:
                    theta[key].grad.zero_()

            C_new, mask_new = localker_clean(theta, n_px_side)
            loss = C_new.sum()  # Simple loss: sum of all elements
            loss.backward()

            # Compare: analytical dC['Amp'].sum() should equal theta['Amp'].grad
            analytical_grad = dC_orig['Amp'].sum()
            autograd_grad = theta['Amp'].grad

            assert torch.allclose(analytical_grad, autograd_grad, atol=ATOL, rtol=RTOL), \
                f"Amp gradient mismatch: analytical={analytical_grad.item():.6f}, autograd={autograd_grad.item():.6f}"

    def test_gradient_2log2beta(self):
        """Test gradient w.r.t. -2log2beta."""
        with torch.enable_grad():
            n_px_side = 20
            theta = make_test_theta(requires_grad=True)
            theta_higher_lims, theta_lower_lims = make_theta_limits()

            C_orig, mask_orig, dC_orig = localker(theta, theta_higher_lims, theta_lower_lims, n_px_side, grad=True)

            for key in theta:
                if theta[key].grad is not None:
                    theta[key].grad.zero_()

            C_new, mask_new = localker_clean(theta, n_px_side)
            loss = C_new.sum()
            loss.backward()

            analytical_grad = dC_orig['-2log2beta'].sum()
            autograd_grad = theta['-2log2beta'].grad

            assert torch.allclose(analytical_grad, autograd_grad, atol=ATOL, rtol=RTOL), \
                f"-2log2beta gradient mismatch: analytical={analytical_grad.item():.6f}, autograd={autograd_grad.item():.6f}"

    def test_gradient_log2rho2(self):
        """Test gradient w.r.t. -log2rho2."""
        with torch.enable_grad():
            n_px_side = 20
            theta = make_test_theta(requires_grad=True)
            theta_higher_lims, theta_lower_lims = make_theta_limits()

            C_orig, mask_orig, dC_orig = localker(theta, theta_higher_lims, theta_lower_lims, n_px_side, grad=True)

            for key in theta:
                if theta[key].grad is not None:
                    theta[key].grad.zero_()

            C_new, mask_new = localker_clean(theta, n_px_side)
            loss = C_new.sum()
            loss.backward()

            analytical_grad = dC_orig['-log2rho2'].sum()
            autograd_grad = theta['-log2rho2'].grad

            assert torch.allclose(analytical_grad, autograd_grad, atol=ATOL, rtol=RTOL), \
                f"-log2rho2 gradient mismatch: analytical={analytical_grad.item():.6f}, autograd={autograd_grad.item():.6f}"

    def test_gradient_eps0x(self):
        """Test gradient w.r.t. eps_0x."""
        with torch.enable_grad():
            n_px_side = 20
            theta = make_test_theta(requires_grad=True)
            theta_higher_lims, theta_lower_lims = make_theta_limits()

            C_orig, mask_orig, dC_orig = localker(theta, theta_higher_lims, theta_lower_lims, n_px_side, grad=True)

            for key in theta:
                if theta[key].grad is not None:
                    theta[key].grad.zero_()

            C_new, mask_new = localker_clean(theta, n_px_side)
            loss = C_new.sum()
            loss.backward()

            analytical_grad = dC_orig['eps_0x'].sum()
            autograd_grad = theta['eps_0x'].grad

            assert torch.allclose(analytical_grad, autograd_grad, atol=ATOL, rtol=RTOL), \
                f"eps_0x gradient mismatch: analytical={analytical_grad.item():.6f}, autograd={autograd_grad.item():.6f}"

    def test_gradient_eps0y(self):
        """Test gradient w.r.t. eps_0y."""
        with torch.enable_grad():
            n_px_side = 20
            theta = make_test_theta(requires_grad=True)
            theta_higher_lims, theta_lower_lims = make_theta_limits()

            C_orig, mask_orig, dC_orig = localker(theta, theta_higher_lims, theta_lower_lims, n_px_side, grad=True)

            for key in theta:
                if theta[key].grad is not None:
                    theta[key].grad.zero_()

            C_new, mask_new = localker_clean(theta, n_px_side)
            loss = C_new.sum()
            loss.backward()

            analytical_grad = dC_orig['eps_0y'].sum()
            autograd_grad = theta['eps_0y'].grad

            assert torch.allclose(analytical_grad, autograd_grad, atol=ATOL, rtol=RTOL), \
                f"eps_0y gradient mismatch: analytical={analytical_grad.item():.6f}, autograd={autograd_grad.item():.6f}"

    def test_all_gradients_at_once(self):
        """Test all gradients in a single test with different loss function."""
        with torch.enable_grad():
            n_px_side = 25
            theta = make_test_theta(requires_grad=True)
            theta_higher_lims, theta_lower_lims = make_theta_limits()

            # Get analytical gradients
            C_orig, mask_orig, dC_orig = localker(theta, theta_higher_lims, theta_lower_lims, n_px_side, grad=True)

            # Reset and compute autograd gradients with a weighted loss
            for key in theta:
                if theta[key].grad is not None:
                    theta[key].grad.zero_()

            C_new, mask_new = localker_clean(theta, n_px_side)

            # Use a more complex loss: weighted sum
            weights = torch.randn_like(C_new)
            loss = (C_new * weights).sum()
            loss.backward()

            # For weighted loss, analytical gradient is: sum(weights * dC)
            params_to_test = ['Amp', '-2log2beta', '-log2rho2', 'eps_0x', 'eps_0y']
            for param in params_to_test:
                analytical_grad = (weights * dC_orig[param]).sum()
                autograd_grad = theta[param].grad

                assert torch.allclose(analytical_grad, autograd_grad, atol=ATOL, rtol=RTOL), \
                    f"{param} gradient mismatch with weighted loss: analytical={analytical_grad.item():.6f}, autograd={autograd_grad.item():.6f}"


# =============================================================================
# Test 3: Finite Difference Verification
# =============================================================================
class TestFiniteDifference:
    """Verify autograd gradients against finite differences."""

    def finite_diff_gradient(self, theta, param_key, n_px_side, eps=1e-5):
        """Compute finite difference gradient for a parameter."""
        # Store original value
        original_val = theta[param_key].item()

        # Forward pass at theta + eps
        theta[param_key] = torch.tensor(original_val + eps, device=DEVICE, dtype=DTYPE)
        C_plus, _ = localker_clean(theta, n_px_side)
        loss_plus = C_plus.sum().item()

        # Forward pass at theta - eps
        theta[param_key] = torch.tensor(original_val - eps, device=DEVICE, dtype=DTYPE)
        C_minus, _ = localker_clean(theta, n_px_side)
        loss_minus = C_minus.sum().item()

        # Restore original value
        theta[param_key] = torch.tensor(original_val, device=DEVICE, dtype=DTYPE, requires_grad=True)

        # Central difference
        return (loss_plus - loss_minus) / (2 * eps)

    def test_finite_diff_Amp(self):
        """Verify Amp gradient with finite differences."""
        with torch.enable_grad():
            n_px_side = 15
            theta = make_test_theta(requires_grad=True)

            # Autograd gradient
            C, _ = localker_clean(theta, n_px_side)
            loss = C.sum()
            loss.backward()
            autograd_grad = theta['Amp'].grad.item()

            # Finite difference gradient (need fresh theta without grad)
            theta_fd = make_test_theta(requires_grad=False)
            fd_grad = self.finite_diff_gradient(theta_fd, 'Amp', n_px_side)

            # Use relative tolerance for large gradient values
            rel_error = abs(autograd_grad - fd_grad) / (abs(fd_grad) + 1e-8)
            assert rel_error < 0.01, \
                f"Amp: autograd={autograd_grad:.6f}, finite_diff={fd_grad:.6f}, rel_error={rel_error:.4f}"

    def test_finite_diff_2log2beta(self):
        """Verify -2log2beta gradient with finite differences."""
        with torch.enable_grad():
            n_px_side = 15
            theta = make_test_theta(requires_grad=True)

            C, _ = localker_clean(theta, n_px_side)
            loss = C.sum()
            loss.backward()
            autograd_grad = theta['-2log2beta'].grad.item()

            theta_fd = make_test_theta(requires_grad=False)
            fd_grad = self.finite_diff_gradient(theta_fd, '-2log2beta', n_px_side)

            # Use relative tolerance for large gradient values
            rel_error = abs(autograd_grad - fd_grad) / (abs(fd_grad) + 1e-8)
            assert rel_error < 0.05, \
                f"-2log2beta: autograd={autograd_grad:.6f}, finite_diff={fd_grad:.6f}, rel_error={rel_error:.4f}"

    def test_finite_diff_eps0x(self):
        """Verify eps_0x gradient with finite differences."""
        with torch.enable_grad():
            n_px_side = 15
            theta = make_test_theta(requires_grad=True)

            C, _ = localker_clean(theta, n_px_side)
            loss = C.sum()
            loss.backward()
            autograd_grad = theta['eps_0x'].grad.item()

            theta_fd = make_test_theta(requires_grad=False)
            fd_grad = self.finite_diff_gradient(theta_fd, 'eps_0x', n_px_side)

            # Use relative tolerance (eps0x has some sensitivity due to mask boundary effects)
            rel_error = abs(autograd_grad - fd_grad) / (abs(fd_grad) + 1e-8)
            assert rel_error < 0.10, \
                f"eps_0x: autograd={autograd_grad:.6f}, finite_diff={fd_grad:.6f}, rel_error={rel_error:.4f}"


# =============================================================================
# Test 4: Edge Cases
# =============================================================================
class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_centered_rf(self):
        """Test with RF centered at origin."""
        n_px_side = 20
        theta = make_test_theta(requires_grad=False)
        theta['eps_0x'] = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
        theta['eps_0y'] = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
        theta_higher_lims, theta_lower_lims = make_theta_limits()

        C_orig, mask_orig = localker(theta, theta_higher_lims, theta_lower_lims, n_px_side, grad=False)
        C_new, mask_new = localker_clean(theta, n_px_side)

        assert torch.allclose(C_orig, C_new, atol=ATOL, rtol=RTOL)

    def test_corner_rf(self):
        """Test with RF at corner of image."""
        n_px_side = 20
        theta = make_test_theta(requires_grad=False)
        theta['eps_0x'] = torch.tensor(0.8, device=DEVICE, dtype=DTYPE)
        theta['eps_0y'] = torch.tensor(0.8, device=DEVICE, dtype=DTYPE)
        theta_higher_lims, theta_lower_lims = make_theta_limits()

        C_orig, mask_orig = localker(theta, theta_higher_lims, theta_lower_lims, n_px_side, grad=False)
        C_new, mask_new = localker_clean(theta, n_px_side)

        assert torch.allclose(C_orig, C_new, atol=ATOL, rtol=RTOL)
        assert torch.all(mask_orig == mask_new)

    def test_large_beta(self):
        """Test with large beta (narrow RF)."""
        n_px_side = 30
        theta = make_test_theta(requires_grad=False)
        theta['-2log2beta'] = torch.tensor(2.0, device=DEVICE, dtype=DTYPE)  # Large -> narrow RF
        theta_higher_lims, theta_lower_lims = make_theta_limits()

        C_orig, mask_orig = localker(theta, theta_higher_lims, theta_lower_lims, n_px_side, grad=False)
        C_new, mask_new = localker_clean(theta, n_px_side)

        assert torch.allclose(C_orig, C_new, atol=ATOL, rtol=RTOL)

    def test_small_beta(self):
        """Test with small beta (wide RF)."""
        n_px_side = 30
        theta = make_test_theta(requires_grad=False)
        theta['-2log2beta'] = torch.tensor(-4.0, device=DEVICE, dtype=DTYPE)  # Small -> wide RF
        theta_higher_lims, theta_lower_lims = make_theta_limits()

        C_orig, mask_orig = localker(theta, theta_higher_lims, theta_lower_lims, n_px_side, grad=False)
        C_new, mask_new = localker_clean(theta, n_px_side)

        assert torch.allclose(C_orig, C_new, atol=ATOL, rtol=RTOL)


# =============================================================================
# Run tests if executed directly
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
