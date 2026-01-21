#!/usr/bin/env python3
"""
Benchmark script for comparing training modes and whitening × caching combinations.

Compares:
- REF: Original varGP from utils.py (reference baseline)
- ADAM: Pure Adam optimization (no E-step)
- vargp_style with whitening/caching variations (A, B, C, D)

Test Matrix:
| Config | Mode | Cache | Whitening | Description |
|--------|------|-------|-----------|-------------|
| REF | vargp_old | - | - | Original varGP (reference) |
| ADAM | adam | - | - | Pure Adam optimization |
| A | vargp_style | ON | ON | Default (should match REF) |
| B | vargp_style | ON | OFF | Cached, no whitening |
| C | vargp_style | OFF | ON | Non-cached, whitening |
| D | vargp_style | OFF | OFF | Non-cached, no whitening |

Key comparisons:
- A vs REF: Does GPyTorch implementation match original varGP?
- A vs C: Are whitening paths equivalent?
- ADAM vs A: How does E-step compare to pure optimization?

Usage:
    conda run -n pytorch_gpytorch python benchmark_whitening_modes.py
    conda run -n pytorch_gpytorch python benchmark_whitening_modes.py --m-values 25 50
    conda run -n pytorch_gpytorch python benchmark_whitening_modes.py --no-baselines
    conda run -n pytorch_gpytorch python benchmark_whitening_modes.py --save results/benchmark.md
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Setup paths
gpytorch_porting_dir = Path(__file__).parent
sys.path.insert(0, str(gpytorch_porting_dir))

import torch
import gpytorch
import numpy as np

# Import GP_utils FIRST (before set_reproducible_seed) - it does CUDA init
from gaussian_processes.Spatial_GP_repo import utils as GP_utils

from tests.test_utils import set_reproducible_seed, get_device

# Initialize CUDA and set seed AFTER GP_utils import
set_reproducible_seed(42)

from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from model import VariationalGPModel
from estep import train_varGP_style
from train import train_adam, predict, compute_pearson_correlation

DEVICE = get_device()
N_PX_SIDE = 108


def load_data(n_train=500, cellid=8):
    """Load PNAS data."""
    data_path = gpytorch_porting_dir.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = np.load(data_path)

    X_train_np = data['images_train']
    X_val_np = data['images_val']
    R_train_np = data['responses_train']
    R_val_np = data['responses_val']

    X = torch.cat([
        torch.tensor(X_train_np, dtype=torch.float64),
        torch.tensor(X_val_np, dtype=torch.float64)
    ], dim=0)
    R = torch.cat([
        torch.tensor(R_train_np, dtype=torch.float64),
        torch.tensor(R_val_np, dtype=torch.float64)
    ], dim=0)

    X = X.reshape(X.shape[0], -1).to(DEVICE)
    R = R.to(DEVICE)
    r = R[:, cellid]

    # Test data
    X_test = torch.tensor(data['images_test'], dtype=torch.float64).reshape(30, -1).to(DEVICE)
    R_test = torch.tensor(data['responses_test'][:, :, cellid], dtype=torch.float64).to(DEVICE)

    return {
        'X_all': X,
        'r_all': r,
        'X_test': X_test,
        'R_test': R_test,
        'n_train': n_train,
        'cellid': cellid,
    }


def create_model_and_likelihood(inducing_points, A_init=0.01, lambda0_init=1.0):
    """Create model and likelihood for GPyTorch modes."""
    # ArcCosineKernel now has internal Amp parameter (matches legacy varGP)
    # No need for ScaleKernel wrapper
    kernel = ArcCosineKernel(
        sigma_0=1.0, Amp=1e-4, n_px_side=N_PX_SIDE,
        eps_0x=0.0, eps_0y=0.0,
        beta=0.1, rho=0.1, use_mask=True
    )

    model = VariationalGPModel(inducing_points, kernel, jitter=1e-4).double().to(DEVICE)
    likelihood = PoissonLikelihood(A_init=A_init, lambda0_init=lambda0_init).double().to(DEVICE)

    return model, likelihood


def run_vargp_old(data, M, n_iterations, n_estep, n_fstep, n_mstep, beta=0.1, rho=0.1):
    """Run original varGP implementation (reference baseline)."""
    set_reproducible_seed(42, device=DEVICE)

    # Select training data and inducing points
    indices_train = torch.randperm(data['X_all'].shape[0], device=DEVICE)[:data['n_train']]
    X_train = data['X_all'][indices_train]
    r_train = data['r_all'][indices_train]
    inducing_points = data['X_all'][indices_train[:M]].clone()

    # varGP requires float32
    X_train_f32 = X_train.float()
    r_train_f32 = r_train.float()

    # Initialize hyperparameters
    beta_t = torch.tensor(beta, device=DEVICE)
    rho_t = torch.tensor(rho, device=DEVICE)

    theta = {
        'sigma_0': torch.tensor(1.0, device=DEVICE).requires_grad_(),
        'Amp': torch.tensor(1.0, device=DEVICE).requires_grad_(),
        'eps_0x': torch.tensor(0.0, device=DEVICE).requires_grad_(),
        'eps_0y': torch.tensor(0.0, device=DEVICE).requires_grad_(),
        '-2log2beta': (-2 * torch.log(2 * beta_t)).requires_grad_(),
        '-log2rho2': (-torch.log(2 * rho_t * rho_t)).requires_grad_(),
    }

    hyperparams_tuple = GP_utils.generate_theta(
        x=X_train_f32, r=r_train_f32, n_px_side=N_PX_SIDE, display_hyper=False, **theta
    )

    # Link function parameters
    A_init = 0.01
    lambda0_init = 1.0
    f_params = {
        'logA': torch.log(torch.tensor(A_init, device=DEVICE)).requires_grad_(),
        'lambda0': torch.tensor(lambda0_init, device=DEVICE),
    }

    fit_parameters = {
        'ntilde': M,
        'maxiter': n_iterations,
        'nMstep': n_mstep,
        'nEstep': n_estep,
        'nFparamstep': n_fstep,
        'kernfun': GP_utils.acosker,
        'cellid': data['cellid'],
        'n_px_side': N_PX_SIDE,
    }

    vargp_args = {
        'fit_parameters': fit_parameters,
        'xtilde': inducing_points.float(),
        'hyperparams_tuple': hyperparams_tuple,
        'f_params': f_params,
        'm': torch.zeros(M, device=DEVICE),
    }

    # Run varGP
    start_time = time.time()
    fit_model, err_dict = GP_utils.varGP(X_train_f32, r_train_f32, **vargp_args)
    train_time = time.time() - start_time

    if err_dict['is_error']:
        return {
            'test_r': float('nan'),
            'train_time': train_time,
            'final_A': float('nan'),
            'final_lambda0': float('nan'),
            'final_loss': float('nan'),
            'error': err_dict['error'],
        }

    # Evaluate
    _, f_pred, r2, _ = GP_utils.test(
        data['X_test'].reshape(-1, N_PX_SIDE, N_PX_SIDE, 1).float(),
        data['R_test'].float(),
        X_train=data['X_all'].reshape(-1, N_PX_SIDE, N_PX_SIDE, 1).float(),
        at_iteration=None,
        **fit_model
    )

    if not isinstance(f_pred, torch.Tensor):
        f_pred = torch.tensor(f_pred, device=DEVICE)
    else:
        f_pred = f_pred.to(DEVICE)

    r_test_mean = data['R_test'].mean(dim=0)
    test_r = compute_pearson_correlation(f_pred.double(), r_test_mean)

    return {
        'test_r': test_r,
        'train_time': train_time,
        'final_A': torch.exp(fit_model['f_params']['logA']).item(),
        'final_lambda0': fit_model['f_params']['lambda0'].item(),
        'final_loss': float('nan'),  # varGP doesn't return loss directly
    }


def run_adam(data, M, n_iterations, lr=0.01):
    """Run pure Adam optimization."""
    set_reproducible_seed(42, device=DEVICE)

    # Select training data and inducing points
    indices_train = torch.randperm(data['X_all'].shape[0], device=DEVICE)[:data['n_train']]
    X_train = data['X_all'][indices_train]
    r_train = data['r_all'][indices_train]
    inducing_points = data['X_all'][indices_train[:M]].clone()

    # Create model (Adam uses different init: A=1.0, lambda0=0.0)
    model, likelihood = create_model_and_likelihood(inducing_points, A_init=1.0, lambda0_init=0.0)

    # Train
    start_time = time.time()
    losses = train_adam(
        model, likelihood, X_train, r_train,
        n_iterations=n_iterations,
        lr=lr,
        print_every=0,
        device=DEVICE
    )
    train_time = time.time() - start_time

    # Evaluate
    pred = predict(model, likelihood, data['X_test'], device=DEVICE)
    r_test_mean = data['R_test'].mean(dim=0)
    test_r = compute_pearson_correlation(pred['f_pred'], r_test_mean)

    return {
        'test_r': test_r,
        'train_time': train_time,
        'final_A': likelihood.A.item(),
        'final_lambda0': likelihood.lambda0.item(),
        'final_loss': losses[-1] if losses else float('nan'),
    }


def run_vargp_style(data, M, use_cache, use_whitening, n_iterations, n_estep, n_fstep, n_mstep):
    """Run vargp_style with specified cache/whitening settings."""
    set_reproducible_seed(42, device=DEVICE)

    # Select training data and inducing points
    indices_train = torch.randperm(data['X_all'].shape[0], device=DEVICE)[:data['n_train']]
    X_train = data['X_all'][indices_train]
    r_train = data['r_all'][indices_train]
    inducing_points = data['X_all'][indices_train[:M]].clone()

    # Create model (varGP defaults: A=0.01, lambda0=1.0)
    model, likelihood = create_model_and_likelihood(inducing_points, A_init=0.01, lambda0_init=1.0)

    # Train
    start_time = time.time()
    result = train_varGP_style(
        model, likelihood, X_train, r_train,
        n_iterations=n_iterations,
        n_estep=n_estep,
        n_fstep=n_fstep,
        n_mstep=n_mstep,
        lr_f=0.1,
        lr_m=0.1,
        print_every=0,
        device=DEVICE,
        use_cache=use_cache,
        use_whitening=use_whitening,
    )
    train_time = time.time() - start_time

    # Evaluate
    pred = predict(model, likelihood, data['X_test'], device=DEVICE)
    r_test_mean = data['R_test'].mean(dim=0)
    test_r = compute_pearson_correlation(pred['f_pred'], r_test_mean)

    return {
        'test_r': test_r,
        'train_time': train_time,
        'final_A': likelihood.A.item(),
        'final_lambda0': likelihood.lambda0.item(),
        'final_loss': result['losses'][-1] if result['losses'] else float('nan'),
    }


def run_benchmark(m_values, n_iterations, n_estep, n_fstep, n_mstep, n_train, include_baselines=True):
    """Run full benchmark across all configurations."""

    # Configuration definitions
    configs = []

    if include_baselines:
        configs.extend([
            {'name': 'REF', 'mode': 'vargp_old', 'desc': 'Original varGP (reference)'},
            {'name': 'ADAM', 'mode': 'adam', 'desc': 'Pure Adam optimization'},
        ])

    configs.extend([
        {'name': 'A', 'mode': 'vargp_style', 'cache': True,  'whitening': True,  'desc': 'cached+whitening (default)'},
        {'name': 'B', 'mode': 'vargp_style', 'cache': True,  'whitening': False, 'desc': 'cached+no-whitening'},
        {'name': 'C', 'mode': 'vargp_style', 'cache': False, 'whitening': True,  'desc': 'non-cached+whitening'},
        {'name': 'D', 'mode': 'vargp_style', 'cache': False, 'whitening': False, 'desc': 'non-cached+no-whitening'},
    ])

    # Load data once
    print("Loading data...")
    data = load_data(n_train=n_train)

    results = []
    total_runs = len(configs) * len(m_values)
    run_count = 0

    print(f"\nRunning {total_runs} configurations...")
    print(f"  M values: {m_values}")
    print(f"  n_iterations: {n_iterations}")
    print(f"  n_train: {n_train}")
    print(f"  Baselines: {'included' if include_baselines else 'excluded'}")
    print()

    for M in m_values:
        print(f"--- M = {M} ---")
        for cfg in configs:
            run_count += 1
            print(f"  [{run_count}/{total_runs}] Config {cfg['name']}: {cfg['desc']}...", end=' ', flush=True)

            try:
                if cfg['mode'] == 'vargp_old':
                    result = run_vargp_old(
                        data, M,
                        n_iterations=n_iterations,
                        n_estep=n_estep,
                        n_fstep=n_fstep,
                        n_mstep=n_mstep,
                    )
                elif cfg['mode'] == 'adam':
                    result = run_adam(data, M, n_iterations=n_iterations, lr=0.01)
                else:  # vargp_style
                    result = run_vargp_style(
                        data, M,
                        use_cache=cfg['cache'],
                        use_whitening=cfg['whitening'],
                        n_iterations=n_iterations,
                        n_estep=n_estep,
                        n_fstep=n_fstep,
                        n_mstep=n_mstep,
                    )

                print(f"r={result['test_r']:.4f}, t={result['train_time']:.1f}s")

            except Exception as e:
                print(f"ERROR: {e}")
                result = {
                    'test_r': float('nan'),
                    'train_time': float('nan'),
                    'final_A': float('nan'),
                    'final_lambda0': float('nan'),
                    'final_loss': float('nan'),
                    'error': str(e),
                }

            results.append({
                'M': M,
                'config': cfg['name'],
                'mode': cfg['mode'],
                'desc': cfg['desc'],
                **result,
            })

    return results, configs


def format_results_table(results, m_values, configs):
    """Format results as a markdown table."""
    lines = []
    config_names = [c['name'] for c in configs]

    # Header
    lines.append("## Benchmark Results: Training Modes Comparison")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Configuration legend
    lines.append("### Configuration Legend")
    lines.append("")
    lines.append("| Config | Mode | Cache | Whitening | Description |")
    lines.append("|--------|------|-------|-----------|-------------|")
    for cfg in configs:
        cache = cfg.get('cache', '-')
        whitening = cfg.get('whitening', '-')
        cache_str = '✓' if cache is True else ('✗' if cache is False else '-')
        whiten_str = '✓' if whitening is True else ('✗' if whitening is False else '-')
        lines.append(f"| {cfg['name']} | {cfg['mode']} | {cache_str} | {whiten_str} | {cfg['desc']} |")
    lines.append("")

    # Results by M - Test Correlation
    lines.append("### Test Correlation (Pearson r)")
    lines.append("")

    header = "| M |"
    separator = "|---|"
    for name in config_names:
        header += f" {name} |"
        separator += "------|"
    lines.append(header)
    lines.append(separator)

    for M in m_values:
        row = f"| {M} |"
        for name in config_names:
            r = next((x for x in results if x['M'] == M and x['config'] == name), None)
            if r and not np.isnan(r['test_r']):
                row += f" {r['test_r']:.4f} |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    # Timing table
    lines.append("### Training Time (seconds)")
    lines.append("")

    header = "| M |"
    separator = "|---|"
    for name in config_names:
        header += f" {name} |"
        separator += "------|"
    lines.append(header)
    lines.append(separator)

    for M in m_values:
        row = f"| {M} |"
        for name in config_names:
            r = next((x for x in results if x['M'] == M and x['config'] == name), None)
            if r and not np.isnan(r['train_time']):
                row += f" {r['train_time']:.1f} |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    # Key comparisons
    lines.append("### Key Comparisons")
    lines.append("")

    has_ref = 'REF' in config_names
    comparison_header = "| M |"
    comparison_sep = "|---|"

    if has_ref:
        comparison_header += " A vs REF |"
        comparison_sep += "----------|"

    comparison_header += " A vs C |"
    comparison_sep += "--------|"

    if has_ref:
        comparison_header += " ADAM vs REF |"
        comparison_sep += "-------------|"

    lines.append(comparison_header)
    lines.append(comparison_sep)

    for M in m_values:
        row = f"| {M} |"

        A = next((x for x in results if x['M'] == M and x['config'] == 'A'), None)
        C = next((x for x in results if x['M'] == M and x['config'] == 'C'), None)

        if has_ref:
            REF = next((x for x in results if x['M'] == M and x['config'] == 'REF'), None)
            ADAM = next((x for x in results if x['M'] == M and x['config'] == 'ADAM'), None)

            # A vs REF
            if A and REF and not np.isnan(A['test_r']) and not np.isnan(REF['test_r']):
                diff = A['test_r'] - REF['test_r']
                row += f" {diff:+.4f} |"
            else:
                row += " - |"

        # A vs C
        if A and C and not np.isnan(A['test_r']) and not np.isnan(C['test_r']):
            diff = abs(A['test_r'] - C['test_r'])
            status = "~" if diff < 0.02 else f"{diff:.4f}"
            row += f" {status} |"
        else:
            row += " - |"

        if has_ref:
            # ADAM vs REF
            if ADAM and REF and not np.isnan(ADAM['test_r']) and not np.isnan(REF['test_r']):
                diff = ADAM['test_r'] - REF['test_r']
                row += f" {diff:+.4f} |"
            else:
                row += " - |"

        lines.append(row)

    lines.append("")
    lines.append("**Interpretation:**")
    if has_ref:
        lines.append("- A vs REF: Positive = GPyTorch better, Negative = varGP better")
    lines.append("- A vs C: Should be ~ (equivalent whitening paths)")
    if has_ref:
        lines.append("- ADAM vs REF: Compares pure optimization to E-step approach")
    lines.append("")

    # Full details table
    lines.append("### Full Results")
    lines.append("")
    lines.append("| M | Config | test_r | time (s) | A | λ₀ |")
    lines.append("|---|--------|--------|----------|---|----|")

    for r in results:
        test_r = f"{r['test_r']:.4f}" if not np.isnan(r['test_r']) else "-"
        train_time = f"{r['train_time']:.1f}" if not np.isnan(r['train_time']) else "-"
        final_A = f"{r['final_A']:.4f}" if not np.isnan(r['final_A']) else "-"
        final_l0 = f"{r['final_lambda0']:.4f}" if not np.isnan(r['final_lambda0']) else "-"
        lines.append(f"| {r['M']} | {r['config']} | {test_r} | {train_time} | {final_A} | {final_l0} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark training modes and whitening configurations")
    parser.add_argument('--m-values', type=int, nargs='+', default=[25, 50, 75, 100],
                        help='M values to test (default: 25 50 75 100)')
    parser.add_argument('--n-iterations', type=int, default=50,
                        help='Number of training iterations (default: 50)')
    parser.add_argument('--n-estep', type=int, default=10, help='E-step iterations')
    parser.add_argument('--n-fstep', type=int, default=10, help='F-step iterations')
    parser.add_argument('--n-mstep', type=int, default=10, help='M-step iterations')
    parser.add_argument('--n-train', type=int, default=500, help='Training samples')
    parser.add_argument('--no-baselines', action='store_true',
                        help='Exclude REF and ADAM baselines (only run vargp_style configs)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to file (e.g., results/benchmark.md)')
    args = parser.parse_args()

    print("="*70)
    print("TRAINING MODES BENCHMARK")
    print("="*70)
    print(f"Device: {DEVICE}")
    print()

    results, configs = run_benchmark(
        m_values=args.m_values,
        n_iterations=args.n_iterations,
        n_estep=args.n_estep,
        n_fstep=args.n_fstep,
        n_mstep=args.n_mstep,
        n_train=args.n_train,
        include_baselines=not args.no_baselines,
    )

    # Format and print results
    table = format_results_table(results, args.m_values, configs)
    print("\n" + "="*70)
    print(table)

    # Save if requested
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(table)
        print(f"\nResults saved to: {save_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
