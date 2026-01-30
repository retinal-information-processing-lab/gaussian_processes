#!/usr/bin/env python3
"""
diagnose_firing_rate_instability.py - Track firing rate parameter dynamics during training

Created by Claude to investigate HYPOTHESIS 1 from whitening collapse investigation.

HYPOTHESIS: Firing rate parameters (A, lambda0) cause instability in whitened mode.

This script wraps the existing train_varGP_style function and records detailed diagnostics
EVERY iteration to identify when A starts diverging for bad seeds.

Usage:
    python diagnose_firing_rate_instability.py --ntilde 50 --seed 123 --output good_seed.csv
    python diagnose_firing_rate_instability.py --ntilde 50 --seed 456 --output bad_seed.csv
    python diagnose_firing_rate_instability.py --ntilde 75 --seed 42 --output m75_seed42.csv
"""

import sys
import os
import argparse
import csv
from pathlib import Path

# Add parent directory (gpytorch_porting) to path so we can import kernels, model, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add paths for imports
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject')
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/torchlambertw')

import numpy as np
import torch
import gpytorch

# Import our GPyTorch components
from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from model import VariationalGPModel
from tests.test_utils import set_reproducible_seed


def load_pnas_data(data_path, dtype=torch.float64):
    """Load PNAS dataset."""
    data = np.load(data_path)
    return {
        'X_train': torch.tensor(data['images_train'], dtype=dtype),
        'R_train': torch.tensor(data['responses_train'], dtype=dtype),
    }


def record_diagnostics(model, likelihood, train_x, train_y, iteration):
    """
    Record detailed diagnostics for current iteration.

    Returns dict with:
    - iteration
    - A, lambda0
    - rate_min, rate_max, rate_mean, rate_std
    - expected_log_lik, kl_div, loss
    - grad_norm_A, grad_norm_lambda0
    - has_nan_rate, has_inf_rate
    """
    with torch.no_grad():
        # Get current parameters
        A = likelihood.A.item()
        lambda0 = likelihood.lambda0.item()

        # Compute predicted firing rates
        output = model(train_x)
        lambda_mean = output.mean
        lambda_var = output.variance
        rate = torch.exp(A * lambda_mean + 0.5 * A**2 * lambda_var + lambda0)

        rate_min = rate.min().item()
        rate_max = rate.max().item()
        rate_mean = rate.mean().item()
        rate_std = rate.std().item()

        has_nan = torch.isnan(rate).any().item()
        has_inf = torch.isinf(rate).any().item()

        # Compute loss components
        expected_log_lik = likelihood.expected_log_prob(train_y, output).item()
        kl_div = model.variational_strategy.kl_divergence().item()
        loss = -expected_log_lik + kl_div

    # Compute gradient norms (temporarily enable grads)
    with torch.enable_grad():
        # Zero gradients
        if likelihood.raw_A.grad is not None:
            likelihood.raw_A.grad.zero_()
        if likelihood.lambda0.grad is not None:
            likelihood.lambda0.grad.zero_()

        # Forward pass
        output_grad = model(train_x)
        expected_log_lik_grad = likelihood.expected_log_prob(train_y, output_grad)
        kl_div_grad = model.variational_strategy.kl_divergence()
        loss_grad = -expected_log_lik_grad + kl_div_grad

        # Backward
        loss_grad.backward()

        # Get gradient norms
        grad_norm_A = likelihood.raw_A.grad.norm().item() if likelihood.raw_A.grad is not None else 0.0
        grad_norm_lambda0 = likelihood.lambda0.grad.norm().item() if likelihood.lambda0.grad is not None else 0.0

    return {
        'iteration': iteration,
        'A': A,
        'lambda0': lambda0,
        'rate_min': rate_min,
        'rate_max': rate_max,
        'rate_mean': rate_mean,
        'rate_std': rate_std,
        'expected_log_lik': expected_log_lik,
        'kl_div': kl_div,
        'loss': loss,
        'grad_norm_A': grad_norm_A,
        'grad_norm_lambda0': grad_norm_lambda0,
        'has_nan_rate': has_nan,
        'has_inf_rate': has_inf,
    }


def train_with_diagnostics(model, likelihood, train_x, train_y,
                           n_iterations=50, n_estep=10, n_fstep=10, n_mstep=10,
                           output_csv=None, device='cuda'):
    """
    Train using train_varGP_style but record diagnostics every iteration.

    This is a modified version of train_varGP_style that adds per-iteration
    diagnostic recording for investigating firing rate instability.
    """
    import time
    from estep import compute_kernel_cache, e_step_loop
    from fstep import f_step_lbfgs
    from mstep import m_step
    from whitening import set_kernel_requires_grad

    model = model.to(device)
    likelihood = likelihood.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    model.train()
    likelihood.train()

    use_whitening = getattr(model, 'whitening', True)
    use_cache = True
    lr_f = 0.1
    lr_m = 0.1

    diagnostics = []

    # CSV writer setup
    if output_csv:
        csv_file = open(output_csv, 'w', newline='')
        fieldnames = [
            'iteration',
            'A', 'lambda0',
            'rate_min', 'rate_max', 'rate_mean', 'rate_std',
            'expected_log_lik', 'kl_div', 'loss',
            'grad_norm_A', 'grad_norm_lambda0',
            'has_nan_rate', 'has_inf_rate'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
    else:
        csv_file = None
        writer = None

    print("\n" + "="*80)
    print("FIRING RATE PARAMETER DIAGNOSTICS")
    print("="*80)
    print(f"{'Iter':>5} | {'A':>8} | {'λ₀':>8} | {'Rate':>20} | {'Loss':>12} | {'Grads':>20}")
    print(f"{'':>5} | {'':>8} | {'':>8} | {'min/max/mean/std':>20} | {'ELL/KL/Tot':>12} | {'A/λ₀':>20}")
    print("-"*80)

    kernel_cache = None

    for iteration in range(n_iterations):
        # ===== E-STEP BLOCK =====
        set_kernel_requires_grad(model, False)
        model.eval()

        if use_cache:
            with torch.no_grad():
                kernel_cache = compute_kernel_cache(model, train_x)
        else:
            kernel_cache = None

        with torch.no_grad():
            lambda_m, lambda_var = e_step_loop(
                model, likelihood, train_x, train_y, n_estep, verbose=False,
                kernel_cache=kernel_cache,
                use_whitening=use_whitening
            )

        # ===== F-STEP =====
        model.train()
        with torch.enable_grad():
            f_step_lbfgs(model, likelihood, train_x, train_y,
                         lambda_m, lambda_var, n_fstep, lr_f, verbose=False)

        # ===== M-STEP =====
        if n_mstep > 0 and iteration < n_iterations - 1:
            set_kernel_requires_grad(model, True)
            with torch.enable_grad():
                m_step(model, likelihood, train_x, train_y, n_mstep, lr_m, verbose=False)
            set_kernel_requires_grad(model, False)
            kernel_cache = None

        # ===== RECORD DIAGNOSTICS =====
        model.eval()
        diag = record_diagnostics(model, likelihood, train_x, train_y, iteration + 1)
        diagnostics.append(diag)

        # Write to CSV
        if writer:
            writer.writerow(diag)

        # Print summary
        rate_str = f"{diag['rate_min']:.1f}/{diag['rate_max']:.1f}/{diag['rate_mean']:.1f}/{diag['rate_std']:.1f}"
        loss_str = f"{diag['expected_log_lik']:.0f}/{diag['kl_div']:.0f}/{diag['loss']:.0f}"
        grad_str = f"{diag['grad_norm_A']:.2e}/{diag['grad_norm_lambda0']:.2e}"

        flag = ""
        if diag['has_nan_rate']:
            flag = " [NaN!]"
        elif diag['has_inf_rate']:
            flag = " [INF!]"
        elif diag['A'] > 3.0:
            flag = " [A>3!]"

        print(f"{diag['iteration']:>5} | {diag['A']:>8.4f} | {diag['lambda0']:>8.4f} | {rate_str:>20} | "
              f"{loss_str:>12} | {grad_str:>20}{flag}")

    print("-"*80)

    if csv_file:
        csv_file.close()

    return diagnostics


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose firing rate parameter instability during training'
    )
    parser.add_argument('--ntilde', type=int, default=50,
                       help='Number of inducing points M (default: 50)')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed (default: 123)')
    parser.add_argument('--n-train', type=int, default=500,
                       help='Number of training samples (default: 500)')
    parser.add_argument('--n-iterations', type=int, default=50,
                       help='Number of EM iterations (default: 50)')
    parser.add_argument('--n-estep', type=int, default=10,
                       help='E-steps per iteration (default: 10)')
    parser.add_argument('--n-fstep', type=int, default=10,
                       help='F-steps per iteration (default: 10)')
    parser.add_argument('--n-mstep', type=int, default=10,
                       help='M-steps per iteration (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (default: auto-generate)')
    parser.add_argument('--cell', type=int, default=8,
                       help='Cell ID (default: 8)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Auto-generate output filename if not provided
    if args.output is None:
        args.output = f"diagnostics_M{args.ntilde}_seed{args.seed}.csv"

    print("\n" + "="*80)
    print("FIRING RATE INSTABILITY DIAGNOSTIC")
    print("="*80)
    print(f"M = {args.ntilde}")
    print(f"Seed = {args.seed}")
    print(f"n_train = {args.n_train}")
    print(f"n_iterations = {args.n_iterations}")
    print(f"Output = {args.output}")
    print("="*80)

    # Set seed
    set_reproducible_seed(args.seed)

    # Load data
    data_path = '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/notebooks/PNAS_paper_sorted_data.npz'
    data = load_pnas_data(data_path, dtype=torch.float64)

    # Prepare training data
    X_train = data['X_train'][:args.n_train].reshape(args.n_train, -1)
    R_train = data['R_train'][:args.n_train, args.cell]

    device = torch.device(args.device)
    X_train = X_train.to(device)
    R_train = R_train.to(device)

    # Select inducing points
    indices = torch.randperm(X_train.shape[0], device=device)[:args.ntilde]
    inducing_points = X_train[indices]

    # Initialize kernel (with masking)
    kernel = ArcCosineKernel(
        n_px_side=108,
        beta_init=0.1,
        rho_init=0.1,
        eps_0_init=(0.0, 0.0),
        use_masking=True,
        gradient_mode='autograd'
    )
    kernel = kernel.double().to(device)

    # Initialize model
    model = VariationalGPModel(
        inducing_points=inducing_points,
        kernel=kernel,
        whitening=True  # Use whitened mode (the problematic one)
    )
    model = model.double().to(device)

    # Initialize likelihood with VARGP-STYLE parameters
    likelihood = PoissonLikelihood(A_init=0.01, lambda0_init=1.0)
    likelihood = likelihood.double().to(device)

    print(f"\nInitial parameters:")
    print(f"  A = {likelihood.A.item():.4f}")
    print(f"  lambda0 = {likelihood.lambda0.item():.4f}")

    # Check raw_A constraints
    print(f"\nraw_A constraint: {likelihood.raw_A_constraint}")

    # Run diagnostic training
    diagnostics = train_with_diagnostics(
        model, likelihood, X_train, R_train,
        n_iterations=args.n_iterations,
        n_estep=args.n_estep,
        n_fstep=args.n_fstep,
        n_mstep=args.n_mstep,
        output_csv=args.output,
        device=device
    )

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Find when A starts diverging (A > 2.5 is suspicious)
    divergence_iter = None
    for diag in diagnostics:
        if diag['A'] > 2.5:
            divergence_iter = diag['iteration']
            break

    if divergence_iter:
        print(f"A diverged at iteration {divergence_iter}")
        print(f"  A value: {diagnostics[divergence_iter-1]['A']:.4f}")
        print(f"  Loss: {diagnostics[divergence_iter-1]['loss']:.2f}")
    else:
        print("A remained stable (< 2.5) throughout training")

    # Check for NaN/Inf
    has_nan_any = any(diag['has_nan_rate'] for diag in diagnostics)
    has_inf_any = any(diag['has_inf_rate'] for diag in diagnostics)

    if has_nan_any:
        print("\nWARNING: Firing rates became NaN during training!")
    if has_inf_any:
        print("\nWARNING: Firing rates became Inf during training!")

    # Final parameters
    final = diagnostics[-1]
    print(f"\nFinal parameters:")
    print(f"  A = {final['A']:.4f}")
    print(f"  lambda0 = {final['lambda0']:.4f}")
    print(f"  Loss = {final['loss']:.2f}")
    print(f"  Max firing rate = {final['rate_max']:.2f}")

    print(f"\nDiagnostics saved to: {args.output}")
    print("="*80)


if __name__ == '__main__':
    main()
