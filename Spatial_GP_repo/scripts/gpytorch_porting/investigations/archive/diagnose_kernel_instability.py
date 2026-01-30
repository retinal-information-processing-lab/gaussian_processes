"""
Diagnostic script for kernel hyperparameter instability during training.

This script tracks kernel hyperparameters (raw and natural space), gradients,
and matrix condition numbers during M-step iterations to diagnose instability.

Created by Claude for hypothesis testing (H3: unbounded kernel params cause collapse).

Context:
- Original varGP enforces bounds by returning inf loss in LBFGS closure
- GPyTorch implementation has NO bounds enforcement
- This may cause hyperparameters to explode during bad seeds
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import VariationalGPModel
from likelihoods import PoissonLikelihood
from kernels import ArcCosineKernel
from train import train_varGP_style
import csv


class MonitoringCallback:
    """Callback to record hyperparameters during training."""

    def __init__(self):
        self.history = {
            'iteration': [],
            # Raw parameters
            'raw_sigma_0': [],
            'raw_m2log2beta': [],
            'raw_mlog2rho2': [],
            'eps_0x': [],
            'eps_0y': [],
            # Natural parameters
            'sigma_0': [],
            'beta': [],
            'rho': [],
            # Matrix condition numbers
            'cond_C': [],
            'cond_K_inducing': [],
            # Performance
            'test_r': [],
            'loss': [],
        }

    def __call__(self, model, likelihood, train_x, train_y, test_x, test_y, iteration, loss):
        """Called after each EM iteration."""

        # Get the kernel
        kernel = model.covar_module
        if hasattr(kernel, 'base_kernel'):
            kernel = kernel.base_kernel

        # Extract raw parameters
        raw_sigma_0 = kernel.raw_sigma_0.item()
        raw_m2log2beta = kernel.raw_m2log2beta.item()
        raw_mlog2rho2 = kernel.raw_mlog2rho2.item()
        eps_0x = kernel.eps_0x.item()
        eps_0y = kernel.eps_0y.item()

        # Transform to natural space
        sigma_0 = kernel.sigma_0.item()
        beta = np.exp(raw_m2log2beta)
        rho = np.sqrt(np.exp(raw_mlog2rho2))

        # Compute C matrix condition number
        with torch.no_grad():
            C, _ = kernel._compute_C_matrix(apply_mask=True)
            cond_C = torch.linalg.cond(C).item()

            # Compute K at inducing points
            inducing_points = model.variational_strategy.inducing_points
            K_inducing = kernel(inducing_points, inducing_points).evaluate()
            cond_K = torch.linalg.cond(K_inducing).item()

            # Test correlation
            test_output = model(test_x)
            test_mu = test_output.mean
            test_var = test_output.variance
            A = likelihood.A
            lambda0 = likelihood.lambda0
            test_pred = torch.exp(A * test_mu + 0.5 * A**2 * test_var + lambda0)
            test_r = torch.corrcoef(torch.stack([test_y, test_pred]))[0, 1].item()

        # Store
        self.history['iteration'].append(iteration)
        self.history['raw_sigma_0'].append(raw_sigma_0)
        self.history['raw_m2log2beta'].append(raw_m2log2beta)
        self.history['raw_mlog2rho2'].append(raw_mlog2rho2)
        self.history['eps_0x'].append(eps_0x)
        self.history['eps_0y'].append(eps_0y)
        self.history['sigma_0'].append(sigma_0)
        self.history['beta'].append(beta)
        self.history['rho'].append(rho)
        self.history['cond_C'].append(cond_C)
        self.history['cond_K_inducing'].append(cond_K)
        self.history['test_r'].append(test_r)
        self.history['loss'].append(loss)

        # Print
        print(f"  Iter {iteration}: loss={loss:.2f}, test_r={test_r:.3f}, "
              f"beta={beta:.4f}, rho={rho:.4f}, sigma_0={sigma_0:.4f}, "
              f"eps_0=({eps_0x:.3f},{eps_0y:.3f}), cond(C)={cond_C:.2e}")

        # Check for extreme values
        check_extreme_values(kernel, iteration)


def track_mstep_hyperparams(model, likelihood, train_x, train_y, test_x, test_y,
                           n_iterations=50, seed=123, output_file=None):
    """
    Track kernel hyperparameters, gradients, and condition numbers during training.

    Manual training loop with monitoring after each iteration.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Import training functions
    from estep import e_step_loop, compute_kernel_cache, compute_moments_from_kernel_cache
    from fstep import f_step_lbfgs
    from mstep import m_step

    # Create callback
    callback = MonitoringCallback()

    # Whitening detection
    use_whitening = getattr(model, 'whitening', True)

    # Standard parameters matching varGP_style
    n_estep = 10
    n_fstep = 10
    n_mstep = 10
    lr_f = 0.1
    lr_m = 0.1

    model.train()
    likelihood.train()

    for iteration in range(n_iterations):
        # E-step
        e_step_loop(model, likelihood, train_x, train_y, n_estep=n_estep, use_whitening=use_whitening)

        # Compute moments for F-step
        with torch.no_grad():
            output = model(train_x)
            lambda_m = output.mean
            lambda_var = output.variance

        # F-step
        f_step_lbfgs(model, likelihood, train_x, train_y,
                    lambda_m=lambda_m, lambda_var=lambda_var,
                    n_fstep=n_fstep, lr=lr_f)

        # M-step
        m_step(model, likelihood, train_x, train_y, n_mstep=n_mstep, lr=lr_m)

        # Compute loss
        model.eval()
        with torch.no_grad():
            output = model(train_x)
            ell = likelihood.expected_log_prob(train_y, output)
            kl = model.variational_strategy.kl_divergence()
            loss = (-ell + kl).item()

        # Call callback
        callback(model, likelihood, train_x, train_y, test_x, test_y, iteration, loss)

        model.train()

    # Save to CSV
    if output_file:
        save_to_csv(callback.history, output_file)
        print(f"\nSaved tracking data to {output_file}")

    return callback.history


def check_extreme_values(kernel, iteration):
    """Check if any hyperparameters have extreme values."""

    sigma_0 = kernel.sigma_0.item()
    beta = np.exp(kernel.raw_m2log2beta.item())
    rho = np.sqrt(np.exp(kernel.raw_mlog2rho2.item()))
    eps_0x = kernel.eps_0x.item()
    eps_0y = kernel.eps_0y.item()

    warnings = []

    if sigma_0 < 1e-3 or sigma_0 > 1e3:
        warnings.append(f"sigma_0={sigma_0:.2e} (extreme)")
    if beta < 1e-3 or beta > 10:
        warnings.append(f"beta={beta:.2e} (extreme)")
    if rho < 1e-3 or rho > 10:
        warnings.append(f"rho={rho:.2e} (extreme)")
    if abs(eps_0x) > 0.9 or abs(eps_0y) > 0.9:
        warnings.append(f"eps_0=({eps_0x:.3f},{eps_0y:.3f}) (near edge)")

    if warnings:
        print(f"  ⚠️  WARNING at iteration {iteration}: {', '.join(warnings)}")


def OLD_record_state_REMOVED(model, likelihood, train_x, train_y, test_x, test_y,
                history, iteration, step_type, compute_gradients=False):
    """Record current state of kernel hyperparameters and metrics."""

    # Get the kernel (now directly ArcCosineKernel, not wrapped in ScaleKernel)
    kernel = model.covar_module

    # Compute loss and test_r
    with torch.no_grad():
        output = model(train_x)
        expected_log_lik = likelihood.expected_log_prob(train_y, output)
        kl_div = model.variational_strategy.kl_divergence()
        loss = (-expected_log_lik + kl_div).item()

        # Test performance
        test_output = model(test_x)
        test_mu = test_output.mean
        test_var = test_output.variance
        A = likelihood.A
        lambda0 = likelihood.lambda0
        test_pred = torch.exp(A * test_mu + 0.5 * A**2 * test_var + lambda0)
        test_r = torch.corrcoef(torch.stack([test_y, test_pred]))[0, 1].item()

    # Compute gradients if M-step
    if compute_gradients:
        # Backward pass to populate gradients
        model.train()
        likelihood.train()
        model.zero_grad()
        likelihood.zero_grad()

        output = model(train_x)
        expected_log_lik = likelihood.expected_log_prob(train_y, output)
        kl_div = model.variational_strategy.kl_divergence()
        loss_tensor = -expected_log_lik + kl_div
        loss_tensor.backward()

    # Extract raw parameters
    raw_sigma_0 = kernel.raw_sigma_0.item()
    raw_m2log2beta = kernel.raw_m2log2beta.item()
    raw_mlog2rho2 = kernel.raw_mlog2rho2.item()
    eps_0x = kernel.eps_0x.item()
    eps_0y = kernel.eps_0y.item()

    # Transform to natural space
    sigma_0 = kernel.sigma_0.item()
    beta = np.exp(raw_m2log2beta)
    rho = np.sqrt(np.exp(raw_mlog2rho2))

    # Extract gradients (only available after backward)
    if compute_gradients and kernel.raw_sigma_0.grad is not None:
        grad_raw_sigma_0 = kernel.raw_sigma_0.grad.item()
        grad_raw_m2log2beta = kernel.raw_m2log2beta.grad.item()
        grad_raw_mlog2rho2 = kernel.raw_mlog2rho2.grad.item()
        grad_eps_0x = kernel.eps_0x.grad.item()
        grad_eps_0y = kernel.eps_0y.grad.item()

        # Transform gradients to natural space using chain rule
        # sigma_0 = softplus(raw_sigma_0), dsigma_0/draw_sigma_0 = sigmoid(raw_sigma_0)
        # beta = exp(raw_m2log2beta), dbeta/draw = beta
        # rho = sqrt(exp(raw_mlog2rho2)), drho/draw = 0.5 * rho
        grad_sigma_0 = grad_raw_sigma_0 * torch.sigmoid(kernel.raw_sigma_0).item()
        grad_beta = grad_raw_m2log2beta * beta
        grad_rho = grad_raw_mlog2rho2 * 0.5 * rho

        grad_norm_sigma_0 = abs(grad_sigma_0)
        grad_norm_beta = abs(grad_beta)
        grad_norm_rho = abs(grad_rho)
        grad_norm_eps_0x = abs(grad_eps_0x)
        grad_norm_eps_0y = abs(grad_eps_0y)
    else:
        grad_sigma_0 = grad_beta = grad_rho = grad_eps_0x = grad_eps_0y = None
        grad_norm_sigma_0 = grad_norm_beta = grad_norm_rho = None
        grad_norm_eps_0x = grad_norm_eps_0y = None

    # Compute C matrix condition number
    with torch.no_grad():
        C, _ = kernel._compute_C_matrix(apply_mask=True)
        cond_C = torch.linalg.cond(C).item()

        # Compute K at inducing points
        inducing_points = model.variational_strategy.inducing_points
        K_inducing = kernel(inducing_points, inducing_points).evaluate()
        cond_K = torch.linalg.cond(K_inducing).item()

    # Store everything
    history['iteration'].append(iteration)
    history['step_type'].append(step_type)
    history['raw_sigma_0'].append(raw_sigma_0)
    history['raw_m2log2beta'].append(raw_m2log2beta)
    history['raw_mlog2rho2'].append(raw_mlog2rho2)
    history['eps_0x'].append(eps_0x)
    history['eps_0y'].append(eps_0y)
    history['sigma_0'].append(sigma_0)
    history['beta'].append(beta)
    history['rho'].append(rho)
    history['grad_sigma_0'].append(grad_sigma_0)
    history['grad_beta'].append(grad_beta)
    history['grad_rho'].append(grad_rho)
    history['grad_eps_0x'].append(grad_eps_0x)
    history['grad_eps_0y'].append(grad_eps_0y)
    history['grad_norm_sigma_0'].append(grad_norm_sigma_0)
    history['grad_norm_beta'].append(grad_norm_beta)
    history['grad_norm_rho'].append(grad_norm_rho)
    history['grad_norm_eps_0x'].append(grad_norm_eps_0x)
    history['grad_norm_eps_0y'].append(grad_norm_eps_0y)
    history['cond_C'].append(cond_C)
    history['cond_K_inducing'].append(cond_K)
    history['test_r'].append(test_r)
    history['loss'].append(loss)

    # Print current values
    print(f"  After {step_type}-step: loss={loss:.2f}, test_r={test_r:.3f}, "
          f"beta={beta:.4f}, rho={rho:.4f}, sigma_0={sigma_0:.4f}, "
          f"eps_0=({eps_0x:.3f},{eps_0y:.3f})")
    if compute_gradients and grad_beta is not None:
        print(f"    Gradients: beta={grad_beta:.2e}, rho={grad_rho:.2e}, "
              f"sigma_0={grad_sigma_0:.2e}, eps_0x={grad_eps_0x:.2e}")


def save_to_csv(history, output_file):
    """Save tracking history to CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(history.keys())

        # Rows
        n_rows = len(history['iteration'])
        for i in range(n_rows):
            row = [history[key][i] for key in history.keys()]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Diagnose kernel hyperparameter instability')
    parser.add_argument('--ntilde', type=int, default=50, help='Number of inducing points')
    parser.add_argument('--n-train', type=int, default=500, help='Number of training points')
    parser.add_argument('--n-iterations', type=int, default=50, help='Number of EM iterations')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--cell', type=int, default=8, help='Cell ID')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    print(f"Running with seed={args.seed}, ntilde={args.ntilde}")

    # Load data (matching run_single_mode.py)
    data_path = Path(__file__).parent.parent.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = np.load(data_path)

    X_train = torch.tensor(data['images_train'], dtype=dtype, device=device)
    R_train = torch.tensor(data['responses_train'], dtype=dtype, device=device)
    X_test = torch.tensor(data['images_test'], dtype=dtype, device=device)
    R_test = torch.tensor(data['responses_test'], dtype=dtype, device=device)

    # Flatten images
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Select cell
    r_train = R_train[:, args.cell]
    r_test = R_test[:, :, args.cell].mean(dim=0)  # Average over repeats

    # Subsample training
    torch.manual_seed(args.seed)
    n_available = X_train.shape[0]
    perm = torch.randperm(n_available)
    train_idx = perm[:args.n_train]

    X_train = X_train[train_idx]
    r_train = r_train[train_idx]

    # Initialize inducing points
    inducing_idx = torch.randperm(args.n_train)[:args.ntilde]
    inducing_points = X_train[inducing_idx]

    # Initialize model
    kernel = ArcCosineKernel(
        sigma_0=1.0,
        n_px_side=108,
        eps_0x=0.0,
        eps_0y=0.0,
        beta=0.1,
        rho=0.1,
        use_mask=True,
        gradient_mode='autograd'
    )
    kernel = kernel.to(device, dtype)

    model = VariationalGPModel(inducing_points, kernel).to(device, dtype)
    likelihood = PoissonLikelihood(A_init=0.01, lambda0_init=1.0).to(device, dtype)

    # Track hyperparameters through training
    output_file = Path(__file__).parent / f'kernel_stability_seed{args.seed}_M{args.ntilde}.csv'

    history = track_mstep_hyperparams(
        model, likelihood,
        X_train, r_train,
        X_test, r_test,
        n_iterations=args.n_iterations,
        seed=args.seed,
        output_file=output_file
    )

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Seed: {args.seed}")
    print(f"Final test_r: {history['test_r'][-1]:.3f}")
    print(f"Final loss: {history['loss'][-1]:.2f}")

    # Check for instability indicators
    beta_vals = history['beta']
    rho_vals = history['rho']

    if beta_vals:
        beta_range = max(beta_vals) - min(beta_vals)
        print(f"\nbeta range: {min(beta_vals):.4f} to {max(beta_vals):.4f} (Δ={beta_range:.4f})")
    if rho_vals:
        rho_range = max(rho_vals) - min(rho_vals)
        print(f"rho range: {min(rho_vals):.4f} to {max(rho_vals):.4f} (Δ={rho_range:.4f})")

    # Check condition numbers
    max_cond_C = max(history['cond_C'])
    max_cond_K = max(history['cond_K_inducing'])
    print(f"\nMax condition numbers:")
    print(f"  C matrix: {max_cond_C:.2e}")
    print(f"  K (inducing): {max_cond_K:.2e}")


if __name__ == '__main__':
    main()
