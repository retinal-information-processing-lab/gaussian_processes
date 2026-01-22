#!/usr/bin/env python3
"""
diagnose_whitening_collapse.py - Debug seed-specific whitening collapse

Created by Claude for debugging. TEMPORARY - ask before removing.

Tracks per-iteration diagnostics to identify why seed 456 with whitening
collapses to r=0.11 while seed 123 works (r=0.77) and seed 456 unwhitened works (r=0.59).

Diagnostics tracked per step:
- Kernel matrix: cond(K_tilde), eigenvalue range
- Kernel params: sigma_0, Amp, beta, rho, eps_0x, eps_0y
- F-params: A, lambda0
- Firing rates: min, max, mean, std
- Variational params: m_norm, V_trace, V_cond
- Loss: ELL, KL, total

Usage:
    python diagnose_whitening_collapse.py --seed 456              # Collapse case (whitened)
    python diagnose_whitening_collapse.py --seed 123              # Working whitened
    python diagnose_whitening_collapse.py --seed 456 --unwhitened # Working unwhitened
"""

import sys
import os
import argparse
import csv
from pathlib import Path

# Add parent directory (gpytorch_porting) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add paths for imports
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject')
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/torchlambertw')

import numpy as np
import torch

from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from model import VariationalGPModel
from tests.test_utils import set_reproducible_seed
from estep import (
    compute_kernel_cache,
    e_step_loop,
    f_step_lbfgs,
    m_step,
    set_kernel_requires_grad,
    get_variational_mean,
    get_variational_covar,
    get_variational_mean_with_L_K,
    get_variational_covar_with_L_K,
)


def compute_diagnostics(model, likelihood, train_x, train_y, kernel_cache,
                        iteration, step_type, use_whitening):
    """Compute comprehensive diagnostics at current state."""

    diag = {'iteration': iteration, 'step_type': step_type}

    kernel = model.covar_module
    K_tilde = kernel_cache['K_tilde']
    L_K = kernel_cache.get('L_K')

    with torch.no_grad():
        # === Kernel matrix diagnostics ===
        try:
            diag['cond_K_tilde'] = torch.linalg.cond(K_tilde).item()
        except:
            diag['cond_K_tilde'] = float('inf')

        eigvals = torch.linalg.eigvalsh(K_tilde)
        diag['eigval_min_K'] = eigvals.min().item()
        diag['eigval_max_K'] = eigvals.max().item()
        diag['cholesky_ok'] = L_K is not None

        # === Kernel hyperparameters ===
        diag['sigma_0'] = kernel.sigma_0.item()
        diag['Amp'] = kernel.Amp.item()
        diag['beta'] = kernel.beta.item()
        diag['rho'] = kernel.rho.item()
        diag['eps_0x'] = kernel.eps_0x.item()
        diag['eps_0y'] = kernel.eps_0y.item()

        # === F-parameters ===
        diag['A'] = likelihood.A.item()
        diag['lambda0'] = likelihood.lambda0.item()

        # === Variational parameters ===
        if use_whitening and L_K is not None:
            m = get_variational_mean_with_L_K(model, L_K)
            V = get_variational_covar_with_L_K(model, L_K)
        else:
            m = get_variational_mean(model)
            V = get_variational_covar(model)

        diag['m_norm'] = torch.norm(m).item()
        diag['m_min'] = m.min().item()
        diag['m_max'] = m.max().item()
        diag['V_trace'] = torch.trace(V).item()
        try:
            diag['V_cond'] = torch.linalg.cond(V).item()
        except:
            diag['V_cond'] = float('inf')

        # === Firing rate statistics ===
        output = model(train_x)
        lambda_m = output.mean
        lambda_var = output.variance
        A = likelihood.A.squeeze()
        lambda0 = likelihood.lambda0.squeeze()
        f_mean = torch.exp(A * lambda_m + 0.5 * A**2 * lambda_var + lambda0)

        diag['lambda_m_mean'] = lambda_m.mean().item()
        diag['lambda_m_std'] = lambda_m.std().item()
        diag['lambda_var_mean'] = lambda_var.mean().item()

        diag['f_mean_min'] = f_mean.min().item()
        diag['f_mean_max'] = f_mean.max().item()
        diag['f_mean_mean'] = f_mean.mean().item()
        diag['f_mean_std'] = f_mean.std().item()
        diag['has_nan'] = int(torch.isnan(f_mean).any().item())
        diag['has_inf'] = int(torch.isinf(f_mean).any().item())

        # === Loss components ===
        ell = likelihood.expected_log_prob(train_y, output)
        kl = model.variational_strategy.kl_divergence()
        diag['ell'] = ell.item()
        diag['kl'] = kl.item()
        diag['loss'] = (-ell + kl).item()

    return diag


def print_diagnostic_row(diag, verbose=True):
    """Print a summary diagnostic row."""
    step = diag['step_type']
    iter_num = diag['iteration']

    # Flag problematic values
    flags = []
    if diag['cond_K_tilde'] > 1e10:
        flags.append('K_cond!')
    if diag['A'] > 2.0:
        flags.append('A>2!')
    if diag['f_mean_max'] > 100:
        flags.append('f>100!')
    if diag['f_mean_std'] < 0.1:
        flags.append('f_collapsed!')
    if diag['has_nan']:
        flags.append('NaN!')
    if diag['V_cond'] > 1e10:
        flags.append('V_cond!')

    flag_str = ' '.join(flags) if flags else ''

    if verbose or flags:
        print(f"[{iter_num:2d}:{step:12s}] cond(K)={diag['cond_K_tilde']:.1e} "
              f"A={diag['A']:.4f} lam0={diag['lambda0']:.2f} "
              f"f_mean={diag['f_mean_mean']:.2f} f_std={diag['f_mean_std']:.3f} "
              f"loss={diag['loss']:.1f} {flag_str}")


def train_with_diagnostics(model, likelihood, train_x, train_y,
                           n_iterations=50, n_estep=10, n_fstep=10, n_mstep=10,
                           lr_f=0.1, lr_m=0.1,
                           use_whitening=True, output_csv=None, verbose=False):
    """Training loop with comprehensive per-step diagnostics."""

    device = train_x.device

    # CSV setup
    fieldnames = [
        'iteration', 'step_type',
        # Kernel matrix
        'cond_K_tilde', 'eigval_min_K', 'eigval_max_K', 'cholesky_ok',
        # Hyperparams
        'sigma_0', 'Amp', 'beta', 'rho', 'eps_0x', 'eps_0y',
        # F-params
        'A', 'lambda0',
        # Variational
        'm_norm', 'm_min', 'm_max', 'V_trace', 'V_cond',
        # Lambda moments
        'lambda_m_mean', 'lambda_m_std', 'lambda_var_mean',
        # Firing rates
        'f_mean_min', 'f_mean_max', 'f_mean_mean', 'f_mean_std', 'has_nan', 'has_inf',
        # Loss
        'ell', 'kl', 'loss',
    ]

    csv_file = open(output_csv, 'w', newline='') if output_csv else None
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames) if csv_file else None
    if writer:
        writer.writeheader()

    all_diagnostics = []

    for iteration in range(n_iterations):
        # ===== E-STEP BLOCK =====
        set_kernel_requires_grad(model, False)
        model.eval()

        # Compute kernel cache
        with torch.no_grad():
            kernel_cache = compute_kernel_cache(model, train_x)

        # Record diagnostics BEFORE E-step
        diag = compute_diagnostics(model, likelihood, train_x, train_y,
                                   kernel_cache, iteration, 'pre_estep', use_whitening)
        all_diagnostics.append(diag)
        if writer:
            writer.writerow(diag)
        print_diagnostic_row(diag, verbose)

        # E-step Newton updates
        with torch.no_grad():
            lambda_m, lambda_var = e_step_loop(
                model, likelihood, train_x, train_y, n_estep,
                kernel_cache=kernel_cache, use_whitening=use_whitening
            )

        # Record diagnostics AFTER E-step
        diag = compute_diagnostics(model, likelihood, train_x, train_y,
                                   kernel_cache, iteration, 'post_estep', use_whitening)
        all_diagnostics.append(diag)
        if writer:
            writer.writerow(diag)
        print_diagnostic_row(diag, verbose)

        # ===== F-STEP =====
        model.train()
        with torch.enable_grad():
            f_step_lbfgs(model, likelihood, train_x, train_y,
                        lambda_m, lambda_var, n_fstep, lr_f, verbose=False)

        # Record diagnostics AFTER F-step
        diag = compute_diagnostics(model, likelihood, train_x, train_y,
                                   kernel_cache, iteration, 'post_fstep', use_whitening)
        all_diagnostics.append(diag)
        if writer:
            writer.writerow(diag)
        print_diagnostic_row(diag, verbose)

        # ===== M-STEP =====
        if n_mstep > 0 and iteration < n_iterations - 1:
            set_kernel_requires_grad(model, True)
            with torch.enable_grad():
                m_step(model, likelihood, train_x, train_y, n_mstep, lr_m)
            set_kernel_requires_grad(model, False)

            # Recompute kernel cache after M-step
            with torch.no_grad():
                kernel_cache = compute_kernel_cache(model, train_x)

            # Record diagnostics AFTER M-step
            diag = compute_diagnostics(model, likelihood, train_x, train_y,
                                       kernel_cache, iteration, 'post_mstep', use_whitening)
            all_diagnostics.append(diag)
            if writer:
                writer.writerow(diag)
            print_diagnostic_row(diag, verbose)

    if csv_file:
        csv_file.close()
        print(f"\nDiagnostics saved to: {output_csv}")

    return all_diagnostics


def main():
    parser = argparse.ArgumentParser(description="Debug whitening collapse")
    parser.add_argument('--seed', type=int, default=456, help='Random seed')
    parser.add_argument('--ntilde', type=int, default=50, help='Number of inducing points')
    parser.add_argument('--n-train', type=int, default=500, help='Number of training samples')
    parser.add_argument('--n-iterations', type=int, default=50, help='Number of EM iterations')
    parser.add_argument('--n-mstep', type=int, default=10, help='M-step iterations (0 to disable)')

    # Three modes for whitening:
    # 1. --whitened (default): VariationalStrategy + whitening conversions
    # 2. --no-whitening-conversions: VariationalStrategy but use_whitening=False in E-step
    # 3. --unwhitened-strategy: UnwhitenedVariationalStrategy
    parser.add_argument('--no-whitening-conversions', action='store_true',
                        help='Use VariationalStrategy but skip whitening conversions in E-step')
    parser.add_argument('--unwhitened-strategy', action='store_true',
                        help='Use UnwhitenedVariationalStrategy (stores natural params)')

    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    parser.add_argument('--cell', type=int, default=8, help='Cell index')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--verbose', action='store_true', help='Print all rows')
    args = parser.parse_args()

    # Determine model whitening (which strategy to use) and E-step whitening (conversions)
    if args.unwhitened_strategy:
        model_whitening = False  # UnwhitenedVariationalStrategy
        use_whitening = False    # No conversions needed
        mode_str = 'unwhitened_strategy'
    elif args.no_whitening_conversions:
        model_whitening = True   # VariationalStrategy (whitened storage)
        use_whitening = False    # But skip conversions in E-step
        mode_str = 'no_conversions'
    else:
        model_whitening = True   # VariationalStrategy
        use_whitening = True     # With whitening conversions
        mode_str = 'whitened'

    if args.output is None:
        mstep_str = f'_nMstep{args.n_mstep}' if args.n_mstep != 10 else ''
        args.output = f'collapse_diag_seed{args.seed}_{mode_str}_M{args.ntilde}{mstep_str}.csv'

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_reproducible_seed(args.seed, device=device)

    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    print(f"Loading data from: {data_path}")
    data = np.load(data_path)

    X_train = torch.tensor(data['images_train'][:args.n_train], dtype=torch.float64, device=device)
    R_train = torch.tensor(data['responses_train'][:args.n_train, args.cell], dtype=torch.float64, device=device)
    X_train = X_train.reshape(args.n_train, -1)

    # Inducing points
    indices = torch.randperm(args.n_train, device=device)[:args.ntilde]
    inducing_points = X_train[indices]

    # Model setup
    kernel = ArcCosineKernel(
        sigma_0=1.0, n_px_side=108, eps_0x=0.0, eps_0y=0.0,
        beta=0.1, rho=0.1, use_mask=True, gradient_mode='autograd'
    ).double().to(device)
    kernel.Amp = 1e-4  # Match varGP init

    model = VariationalGPModel(inducing_points, kernel, jitter=1e-4, whitening=model_whitening)
    model = model.double().to(device)

    likelihood = PoissonLikelihood(A_init=0.01, lambda0_init=1.0)
    likelihood = likelihood.double().to(device)

    print(f"\n{'='*70}")
    print(f"WHITENING COLLAPSE DIAGNOSTIC")
    print(f"{'='*70}")
    print(f"Seed: {args.seed}")
    print(f"Mode: {mode_str}")
    print(f"  Model whitening (strategy): {model_whitening}")
    print(f"  E-step whitening (conversions): {use_whitening}")
    print(f"M: {args.ntilde}, N: {args.n_train}")
    print(f"nMstep: {args.n_mstep}")
    print(f"Output: {args.output}")
    print(f"{'='*70}\n")

    # Run training with diagnostics
    diagnostics = train_with_diagnostics(
        model, likelihood, X_train, R_train,
        n_iterations=args.n_iterations,
        n_mstep=args.n_mstep,
        use_whitening=use_whitening,
        output_csv=args.output,
        verbose=args.verbose
    )

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Find first iteration where problems appear
    for diag in diagnostics:
        if diag['cond_K_tilde'] > 1e10:
            print(f"WARNING: cond(K_tilde) exceeded 1e10 at iteration {diag['iteration']} ({diag['step_type']})")
            break

    for diag in diagnostics:
        if diag['A'] > 2.0:
            print(f"WARNING: A parameter exceeded 2.0 at iteration {diag['iteration']} ({diag['step_type']})")
            break

    for diag in diagnostics:
        if diag['f_mean_std'] < 0.1 and diag['iteration'] > 5:
            print(f"WARNING: f_mean collapsed (std < 0.1) at iteration {diag['iteration']} ({diag['step_type']})")
            break

    final = [d for d in diagnostics if d['step_type'] == 'post_fstep'][-1]
    print(f"\nFinal state (post F-step):")
    print(f"  Loss: {final['loss']:.2f}")
    print(f"  A: {final['A']:.4f}")
    print(f"  lambda0: {final['lambda0']:.2f}")
    print(f"  cond(K_tilde): {final['cond_K_tilde']:.2e}")
    print(f"  f_mean: mean={final['f_mean_mean']:.2f}, std={final['f_mean_std']:.3f}")
    print(f"  m_norm: {final['m_norm']:.4f}")
    print(f"  V_trace: {final['V_trace']:.4f}")


if __name__ == '__main__':
    main()
