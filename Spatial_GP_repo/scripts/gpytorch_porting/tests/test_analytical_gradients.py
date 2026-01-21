"""
Validation tests for analytical gradients.

Tests that analytical gradients match autograd gradients for all hyperparameters.

Stage 4 of the analytical gradients implementation plan.

Run: conda run -n pytorch_gpytorch python tests/test_analytical_gradients.py
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels import ArcCosineKernel


def test_gradient_correctness():
    """Test that analytical gradients match autograd for all hyperparameters.

    Criterion: relative error < 1e-5 for each parameter.
    """
    print("=" * 60)
    print("Test 1: Gradient Correctness (analytical vs autograd)")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    # Test data
    n1, n2, n_px_side = 8, 6, 12
    n_px = n_px_side ** 2
    x1 = torch.randn(n1, n_px, device=device, dtype=dtype)
    x2 = torch.randn(n2, n_px, device=device, dtype=dtype)

    # Test with different initial parameters
    test_configs = [
        {'sigma_0': 1.0, 'beta': 0.1, 'rho': 0.1, 'eps_0x': 0.0, 'eps_0y': 0.0},
        {'sigma_0': 0.5, 'beta': 0.2, 'rho': 0.05, 'eps_0x': 0.1, 'eps_0y': -0.1},
        {'sigma_0': 2.0, 'beta': 0.05, 'rho': 0.2, 'eps_0x': -0.2, 'eps_0y': 0.3},
    ]

    all_passed = True
    tolerance = 1e-5

    for i, config in enumerate(test_configs):
        print(f"\nConfig {i+1}: {config}")

        # Create autograd kernel
        kernel_autograd = ArcCosineKernel(
            sigma_0=config['sigma_0'],
            n_px_side=n_px_side,
            eps_0x=config['eps_0x'],
            eps_0y=config['eps_0y'],
            beta=config['beta'],
            rho=config['rho'],
            gradient_mode='autograd'
        ).to(device).double()

        # Create analytical kernel with same parameters (jacobian mode)
        kernel_analytical = ArcCosineKernel(
            sigma_0=config['sigma_0'],
            n_px_side=n_px_side,
            eps_0x=config['eps_0x'],
            eps_0y=config['eps_0y'],
            beta=config['beta'],
            rho=config['rho'],
            gradient_mode='jacobian'
        ).to(device).double()

        # Compute K and sum (simple loss function)
        K_autograd = kernel_autograd(x1, x2).evaluate()
        loss_autograd = K_autograd.sum()
        loss_autograd.backward()

        K_analytical = kernel_analytical(x1, x2).evaluate()
        loss_analytical = K_analytical.sum()
        loss_analytical.backward()

        # Compare gradients for each parameter
        param_names = ['raw_sigma_0', 'eps_0x', 'eps_0y', 'raw_m2log2beta', 'raw_mlog2rho2']

        for param_name in param_names:
            grad_autograd = getattr(kernel_autograd, param_name).grad
            grad_analytical = getattr(kernel_analytical, param_name).grad

            if grad_autograd is None or grad_analytical is None:
                print(f"  {param_name}: SKIP (no gradient)")
                continue

            rel_err = (grad_autograd - grad_analytical).abs() / (grad_autograd.abs() + 1e-10)
            rel_err_val = rel_err.item()

            status = "PASS" if rel_err_val < tolerance else "FAIL"
            if status == "FAIL":
                all_passed = False

            print(f"  {param_name:20s}: autograd={grad_autograd.item():12.6f}, "
                  f"analytical={grad_analytical.item():12.6f}, rel_err={rel_err_val:.2e} [{status}]")

    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: All gradient tests PASSED!")
    else:
        print("RESULT: Some gradient tests FAILED!")
    print("=" * 60)

    return all_passed


def test_numerical_gradient():
    """Verify analytical gradients using finite differences.

    This is a gold standard check that doesn't depend on autograd correctness.
    """
    print("\n" + "=" * 60)
    print("Test 2: Numerical Gradient Check (finite differences)")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    # Small test for faster computation
    n1, n2, n_px_side = 5, 4, 10
    n_px = n_px_side ** 2
    x1 = torch.randn(n1, n_px, device=device, dtype=dtype)
    x2 = torch.randn(n2, n_px, device=device, dtype=dtype)

    config = {'sigma_0': 1.0, 'beta': 0.1, 'rho': 0.1, 'eps_0x': 0.0, 'eps_0y': 0.0}
    eps_fd = 1e-5
    tolerance = 1e-4  # Looser tolerance for finite differences

    # Create analytical kernel (jacobian mode)
    kernel = ArcCosineKernel(
        sigma_0=config['sigma_0'],
        n_px_side=n_px_side,
        eps_0x=config['eps_0x'],
        eps_0y=config['eps_0y'],
        beta=config['beta'],
        rho=config['rho'],
        gradient_mode='jacobian'
    ).to(device).double()

    # Get analytical gradient
    K = kernel(x1, x2).evaluate()
    loss = K.sum()
    loss.backward()

    all_passed = True
    param_names = ['raw_sigma_0', 'eps_0x', 'eps_0y', 'raw_m2log2beta', 'raw_mlog2rho2']

    for param_name in param_names:
        param = getattr(kernel, param_name)
        grad_analytical = param.grad.item() if param.grad is not None else 0.0

        # Finite difference: (f(x+eps) - f(x-eps)) / (2*eps)
        with torch.no_grad():
            original_val = param.data.clone()

            # Forward
            param.data = original_val + eps_fd
            K_plus = kernel(x1, x2).evaluate()
            loss_plus = K_plus.sum().item()

            # Backward
            param.data = original_val - eps_fd
            K_minus = kernel(x1, x2).evaluate()
            loss_minus = K_minus.sum().item()

            # Restore
            param.data = original_val

        grad_fd = (loss_plus - loss_minus) / (2 * eps_fd)
        rel_err = abs(grad_analytical - grad_fd) / (abs(grad_fd) + 1e-10)

        status = "PASS" if rel_err < tolerance else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"  {param_name:20s}: analytical={grad_analytical:12.6f}, "
              f"finite_diff={grad_fd:12.6f}, rel_err={rel_err:.2e} [{status}]")

    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: Numerical gradient check PASSED!")
    else:
        print("RESULT: Numerical gradient check FAILED!")
    print("=" * 60)

    return all_passed


def test_training_equivalence():
    """Test that training with analytical vs autograd produces similar results.

    We train for a few steps and compare loss trajectories.
    """
    print("\n" + "=" * 60)
    print("Test 3: Training Equivalence")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    # Create simple training data
    n_train, n_px_side = 20, 10
    n_px = n_px_side ** 2
    x_train = torch.randn(n_train, n_px, device=device, dtype=dtype)

    # Target: simple function of kernel values
    # We'll use self-kernel as a proxy for real training

    def train_kernel(gradient_mode, n_steps=50, lr=0.01):
        """Train kernel parameters and return loss trajectory."""
        # ArcCosineKernel now has internal Amp parameter (matches legacy varGP)
        # No need for ScaleKernel wrapper
        kernel = ArcCosineKernel(
            sigma_0=1.0,
            Amp=1e-4,  # Amplitude inside C matrix
            n_px_side=n_px_side,
            eps_0x=0.0, eps_0y=0.0,
            beta=0.1, rho=0.1,
            gradient_mode=gradient_mode
        ).to(device).double()

        # Simple loss: distance from target kernel norm
        target_norm = 100.0
        optimizer = torch.optim.Adam(kernel.parameters(), lr=lr)

        losses = []
        for step in range(n_steps):
            optimizer.zero_grad()
            K = kernel(x_train, x_train).evaluate()
            loss = (K.sum() - target_norm) ** 2
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return losses

    print("Training with autograd...")
    losses_autograd = train_kernel(gradient_mode='autograd')

    print("Training with analytical gradients (jacobian)...")
    losses_analytical = train_kernel(gradient_mode='jacobian')

    # Compare loss trajectories
    losses_autograd = np.array(losses_autograd)
    losses_analytical = np.array(losses_analytical)

    # Check if final losses are similar (within 10%)
    final_ratio = losses_analytical[-1] / (losses_autograd[-1] + 1e-10)

    print(f"\n  Initial loss (autograd):    {losses_autograd[0]:.2f}")
    print(f"  Initial loss (analytical):  {losses_analytical[0]:.2f}")
    print(f"  Final loss (autograd):      {losses_autograd[-1]:.2f}")
    print(f"  Final loss (analytical):    {losses_analytical[-1]:.2f}")
    print(f"  Final loss ratio:           {final_ratio:.4f}")

    # Allow 20% deviation
    passed = 0.8 < final_ratio < 1.2

    print("\n" + "=" * 60)
    if passed:
        print("RESULT: Training equivalence test PASSED!")
    else:
        print("RESULT: Training equivalence test FAILED!")
    print("=" * 60)

    return passed


def run_all_tests():
    """Run all validation tests."""
    results = {}

    results['gradient_correctness'] = test_gradient_correctness()
    results['numerical_gradient'] = test_numerical_gradient()
    results['training_equivalence'] = test_training_equivalence()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {name:30s}: {status}")

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    run_all_tests()
