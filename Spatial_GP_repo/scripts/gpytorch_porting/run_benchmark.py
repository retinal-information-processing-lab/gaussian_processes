#!/usr/bin/env python3
"""
run_benchmark.py - Canonical benchmark comparing all training modes.

This is THE benchmark script for comparing varGP vs GPyTorch implementations.
Runs all 4 modes and produces a comparison table. Parameters are frozen.
For single-mode experimentation, use run_single_mode.py instead.

Usage:
    python run_benchmark.py
    python run_benchmark.py --ntilde 75  # Test different M values
    python run_benchmark.py --no-whitening --no-cache  # Test legacy mode

Whitening/Caching options (vargp_style only):
    --use-whitening / --no-whitening  (default: whitening ON)
    --use-cache / --no-cache          (default: caching ON)

    Default (whitening ON, caching ON): Mathematically correct, fast (8.8x E-step speedup)
    Legacy  (whitening OFF, caching ON): Pre-whitening implementation for comparison

    When running with default config, the script ALSO runs the legacy config
    (whitening OFF, cache ON) for side-by-side comparison.

Note on varGP timing:
    Original varGP prints E-step/M-step timing during training but does NOT
    return it in results. Check console output for varGP timing breakdown.

FROZEN PARAMETERS (from one_cell_fit.py):
    cellid          = 8
    ntilde          = 50   (canonical default)
    n_train         = 500
    maxiter         = 50
    nEstep          = 10
    nFparamstep     = 10
    nMstep          = 10
    beta            = 0.1
    rho             = 0.1
    eps_0x, eps_0y  = 0.0, 0.0
    sigma_0         = 1.0

IMPORTANT - Link Function Initialization:
    ┌─────────────┬─────────┬────────────────┬────────────┐
    │ Parameter   │ varGP   │ vargp_style    │ efm/adam   │
    ├─────────────┼─────────┼────────────────┼────────────┤
    │ A_init      │ 0.01    │ 0.01           │ 1.0        │
    │ lambda0_init│ 1.0     │ 1.0            │ 0.0        │
    └─────────────┴─────────┴────────────────┴────────────┘

    vargp_style uses same initialization as varGP for fair comparison.
    Per Q19 in CLAUDE.md, the model is robust to A initialization.

Reproducing results:
    Results in BENCHMARK_LOG.md include the exact command and git commit hash.
    To reproduce: checkout the commit, run the command shown in the log entry.

Created by Claude as the single source of truth for E-step testing.
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

import torch
torch.set_grad_enabled(False)

# Add project paths
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject')
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/torchlambertw')
sys.path.insert(0, str(Path(__file__).parent.parent))  # For GPyTorch imports

from gaussian_processes.Spatial_GP_repo import utils as GP_utils

# GPyTorch imports
import gpytorch
from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from model import VariationalGPModel #( /ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting/model.py)
from train import train_efm, train_varGP_style
from train import train_adam, predict, compute_pearson_correlation, compute_explained_variance
from tests.test_utils import set_reproducible_seed


# =============================================================================
# FROZEN PARAMETERS - Canonical defaults from one_cell_fit.py
# =============================================================================
PARAMS = {
    # Data
    'cellid': 8,
    'n_train': 500,
    'n_px_side': 108,

    # Model
    'ntilde': 50,  # Canonical default (GPyTorch efm works well here)

    # Training iterations
    'maxiter': 50,           # varGP outer iterations
    'nEstep': 10,            # E-steps per iteration
    'nFparamstep': 10,       # F-steps per iteration
    'nMstep': 10,            # M-steps per iteration

    # GPyTorch equivalent
    'gpytorch_iterations': 50,  # Match maxiter for fair comparison
    'gpytorch_n_fstep': 10,
    'gpytorch_n_mstep': 10,
    'lr': 0.01,

    # Kernel hyperparameters (same for both)
    'beta': 0.1,
    'rho': 0.1,
    'sigma_0': 1.0,
    'eps_0x': 0.0,
    'eps_0y': 0.0,

    # Link function - DIFFERENT between implementations (see docstring)
    'vargp_A_init': 0.01,
    'vargp_lambda0_init': 1.0,
    'gpytorch_A_init': 1.0,        # For efm/adam modes
    'gpytorch_lambda0_init': 0.0,  # For efm/adam modes

    # vargp_style uses varGP initialization for fair comparison
    'vargp_style_A_init': 0.01,
    'vargp_style_lambda0_init': 1.0,
    'vargp_style_lr_f': 0.1,   # varGP default (LBFGS)
    'vargp_style_lr_m': 0.1,   # varGP default
    'vargp_style_n_estep': 10,
}


def load_data(device, dtype=torch.float32):
    """Load PNAS data with consistent preprocessing."""
    data_path = Path(__file__).parent.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = np.load(data_path)

    # Combine train + val
    X = np.concatenate([data['images_train'], data['images_val']], axis=0)
    R = np.concatenate([data['responses_train'], data['responses_val']], axis=0)

    # Flatten images
    X = X.reshape(X.shape[0], -1)
    X_test = data['images_test'].reshape(data['images_test'].shape[0], -1)
    R_test = data['responses_test']

    return {
        'X': torch.tensor(X, dtype=dtype, device=device),
        'R': torch.tensor(R, dtype=dtype, device=device),
        'X_test': torch.tensor(X_test, dtype=dtype, device=device),
        'R_test': torch.tensor(R_test, dtype=dtype, device=device),
    }


def run_vargp(X, R, X_test, R_test, params, device):
    """Run reference varGP implementation."""
    print("\n" + "="*60)
    print("Running varGP (reference implementation)")
    print("="*60)

    cellid = params['cellid']
    ntilde = params['ntilde']
    n_train = params['n_train']
    n_px_side = params['n_px_side']

    # Select cell
    r = R[:, cellid]
    r_test = R_test[:, :, cellid]

    # Select training subset with fixed seed
    torch.manual_seed(42)
    indices = torch.randperm(X.shape[0], device=device)[:n_train]
    X_train = X[indices]
    r_train = r[indices]

    # Inducing points (first ntilde training points)
    xtilde_idx = indices[:ntilde]
    xtilde = X[xtilde_idx]

    # Initialize hyperparameters
    beta = torch.tensor(params['beta'], device=device)
    rho = torch.tensor(params['rho'], device=device)

    theta = {
        'sigma_0': torch.tensor(params['sigma_0'], device=device).requires_grad_(),
        'Amp': torch.tensor(1.0, device=device).requires_grad_(),
        'eps_0x': torch.tensor(params['eps_0x'], device=device).requires_grad_(),
        'eps_0y': torch.tensor(params['eps_0y'], device=device).requires_grad_(),
        '-2log2beta': (-2 * torch.log(2 * beta)).requires_grad_(),
        '-log2rho2': (-torch.log(2 * rho * rho)).requires_grad_(),
    }

    hyperparams_tuple = GP_utils.generate_theta(
        x=X_train, r=r_train, n_px_side=n_px_side, display_hyper=False, **theta
    )

    # Link function parameters (varGP defaults)
    A = torch.tensor(params['vargp_A_init'], device=device)
    f_params = {
        'logA': torch.log(A).requires_grad_(),
        'lambda0': torch.tensor(params['vargp_lambda0_init'], device=device),
    }

    fit_parameters = {
        'ntilde': ntilde,
        'maxiter': params['maxiter'],
        'nMstep': params['nMstep'],
        'nEstep': params['nEstep'],
        'nFparamstep': params['nFparamstep'],
        'kernfun': GP_utils.acosker,
        'cellid': cellid,
        'n_px_side': n_px_side,
    }

    args = {
        'fit_parameters': fit_parameters,
        'xtilde': xtilde,
        'hyperparams_tuple': hyperparams_tuple,
        'f_params': f_params,
        'm': torch.zeros(ntilde, device=device),
    }

    print(f"  A_init: {params['vargp_A_init']}, lambda0_init: {params['vargp_lambda0_init']}")
    print(f"  ntilde: {ntilde}, n_train: {n_train}")
    print(f"  maxiter: {params['maxiter']}, nEstep: {params['nEstep']}, nMstep: {params['nMstep']}")

    # Run varGP
    start_time = time.time()
    fit_model, err_dict = GP_utils.varGP(X_train, r_train, **args)
    elapsed = time.time() - start_time

    if err_dict['is_error']:
        print(f"  ERROR: {err_dict['error']}")
        return None

    # Evaluate
    _, _, r2, sigma_r2 = GP_utils.test(
        X_test.reshape(-1, n_px_side, n_px_side, 1),
        r_test,
        X_train=X,
        at_iteration=None,
        **fit_model
    )

    # Compute explained variance (same formula as GPyTorch)
    # r2 from GP_utils.test is actually explained variance
    explained_var = r2.item() if hasattr(r2, 'item') else r2

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Explained variance: {explained_var:.4f}")

    return {
        'implementation': 'varGP',
        'time': elapsed,
        'explained_var': explained_var,
        'ntilde': ntilde,
    }


def run_gpytorch_efm(X, R, X_test, R_test, params, device, gradient_mode='autograd'):
    """Run GPyTorch with E-F-M training loop."""
    print("\n" + "="*60)
    print("Running GPyTorch (efm mode)")
    print("="*60)

    cellid = params['cellid']
    ntilde = params['ntilde']
    n_train = params['n_train']
    n_px_side = params['n_px_side']

    # Select cell
    r = R[:, cellid]
    r_test = R_test[:, :, cellid]

    # Select training subset with SAME seed as varGP
    torch.manual_seed(42)
    indices = torch.randperm(X.shape[0], device=device)[:n_train]
    X_train = X[indices].double()
    r_train = r[indices].double()

    # Inducing points (first ntilde training points) - SAME as varGP
    inducing_points = X_train[:ntilde].clone()

    # Create kernel
    base_kernel = ArcCosineKernel(
        sigma_0=params['sigma_0'],
        n_px_side=n_px_side,
        eps_0x=params['eps_0x'],
        eps_0y=params['eps_0y'],
        beta=params['beta'],
        rho=params['rho'],
        use_mask=True,
        gradient_mode=gradient_mode,
    )
    kernel = gpytorch.kernels.ScaleKernel(base_kernel)
    kernel.outputscale = 1e-4

    # Create model and likelihood (GPyTorch defaults)
    model = VariationalGPModel(inducing_points, kernel, jitter=1e-4)
    likelihood = PoissonLikelihood(
        A_init=params['gpytorch_A_init'],
        lambda0_init=params['gpytorch_lambda0_init']
    )

    model = model.double().to(device)
    likelihood = likelihood.double().to(device)

    print(f"  A_init: {params['gpytorch_A_init']}, lambda0_init: {params['gpytorch_lambda0_init']}")
    print(f"  ntilde: {ntilde}, n_train: {n_train}")
    print(f"  n_iterations: {params['gpytorch_iterations']}, n_fstep: {params['gpytorch_n_fstep']}, n_mstep: {params['gpytorch_n_mstep']}")

    # Train (need gradients enabled)
    start_time = time.time()
    with torch.enable_grad():
        losses = train_efm(
            model, likelihood, X_train, r_train,
            n_iterations=params['gpytorch_iterations'],
            n_fstep=params['gpytorch_n_fstep'],
            n_mstep=params['gpytorch_n_mstep'],
            lr=params['lr'],
            print_every=params['gpytorch_iterations'] // 5,
            device=device,
        )
    elapsed = time.time() - start_time

    # Evaluate
    X_test_double = X_test.double()
    predictions = predict(model, likelihood, X_test_double, device=device)
    r_test_mean = r_test.mean(dim=0).double()
    explained_var, reliability = compute_explained_variance(r_test.double(), predictions['f_pred'])

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Explained variance: {explained_var:.4f}")
    print(f"  Reliability: {reliability:.4f}")

    return {
        'implementation': 'GPyTorch (efm)',
        'time': elapsed,
        'explained_var': explained_var,
        'reliability': reliability,
        'ntilde': ntilde,
    }


def run_gpytorch_vargp_style(X, R, X_test, R_test, params, device,
                              use_whitening=True, use_cache=True, whitening=True,
                              gradient_mode='autograd'):
    """Run GPyTorch with vargp_style training (matches varGP structure).

    Args:
        use_whitening: If True, use whitening conversions for correct math.
                       (Only relevant when whitening=True)
        use_cache: If True, cache kernel matrices for 8.8x E-step speedup.
        whitening: If True (default), use VariationalStrategy.
                   If False, use UnwhitenedVariationalStrategy (no L_K dependency).
    """
    strategy_str = "unwhitened" if not whitening else f"whitening={'ON' if use_whitening else 'OFF'}"
    mode_str = f"{strategy_str}, cache={'ON' if use_cache else 'OFF'}"
    print("\n" + "="*60)
    print(f"Running GPyTorch (vargp_style mode, {mode_str})")
    print("="*60)

    cellid = params['cellid']
    ntilde = params['ntilde']
    n_train = params['n_train']
    n_px_side = params['n_px_side']

    # Select cell
    r = R[:, cellid]
    r_test = R_test[:, :, cellid]

    # Select training subset with SAME seed as varGP
    torch.manual_seed(42)
    indices = torch.randperm(X.shape[0], device=device)[:n_train]
    X_train = X[indices].double()
    r_train = r[indices].double()

    # Inducing points (first ntilde training points) - SAME as varGP
    inducing_points = X_train[:ntilde].clone()

    # Create kernel
    base_kernel = ArcCosineKernel(
        sigma_0=params['sigma_0'],
        n_px_side=n_px_side,
        eps_0x=params['eps_0x'],
        eps_0y=params['eps_0y'],
        beta=params['beta'],
        rho=params['rho'],
        use_mask=True,
        gradient_mode=gradient_mode,
    )
    kernel = gpytorch.kernels.ScaleKernel(base_kernel)
    kernel.outputscale = 1e-4

    # Create model and likelihood (varGP-style initialization)
    model = VariationalGPModel(inducing_points, kernel, jitter=1e-4, whitening=whitening)
    likelihood = PoissonLikelihood(
        A_init=params['vargp_style_A_init'],
        lambda0_init=params['vargp_style_lambda0_init']
    )

    model = model.double().to(device)
    likelihood = likelihood.double().to(device)

    print(f"  A_init: {params['vargp_style_A_init']}, lambda0_init: {params['vargp_style_lambda0_init']}")
    print(f"  ntilde: {ntilde}, n_train: {n_train}")
    print(f"  n_iterations: {params['gpytorch_iterations']}, n_estep: {params['vargp_style_n_estep']}")
    print(f"  lr_f: {params['vargp_style_lr_f']}, lr_m: {params['vargp_style_lr_m']}")

    # Train with vargp_style (LBFGS F-step, analytical lambda0)
    # If using unwhitened strategy, let auto-detect handle use_whitening
    effective_use_whitening = None if not whitening else use_whitening
    start_time = time.time()
    with torch.enable_grad():
        train_result = train_varGP_style(
            model, likelihood, X_train, r_train,
            n_iterations=params['gpytorch_iterations'],
            n_estep=params['vargp_style_n_estep'],
            n_fstep=params['gpytorch_n_fstep'],
            n_mstep=params['gpytorch_n_mstep'],
            lr_f=params['vargp_style_lr_f'],
            lr_m=params['vargp_style_lr_m'],
            print_every=params['gpytorch_iterations'] // 5,
            device=device,
            use_cache=use_cache,
            use_whitening=effective_use_whitening,
        )
    elapsed = time.time() - start_time

    # Extract timing breakdown from train result
    time_estep = train_result.get('time_estep_total', 0.0)
    time_mstep = train_result.get('time_mstep_total', 0.0)

    # Evaluate
    X_test_double = X_test.double()
    predictions = predict(model, likelihood, X_test_double, device=device)
    explained_var, reliability = compute_explained_variance(r_test.double(), predictions['f_pred'])

    print(f"\n  Time: {elapsed:.1f}s (E-step: {time_estep:.1f}s, M-step: {time_mstep:.1f}s)")
    print(f"  Explained variance: {explained_var:.4f}")
    print(f"  Reliability: {reliability:.4f}")

    # Build implementation name with mode info
    impl_name = 'GPyTorch (vargp_style)'
    if not use_whitening or not use_cache:
        flags = []
        if not use_whitening:
            flags.append('no-whiten')
        if not use_cache:
            flags.append('no-cache')
        impl_name = f"GPyTorch (vargp_style, {'+'.join(flags)})"

    return {
        'implementation': impl_name,
        'time': elapsed,
        'time_estep': time_estep,
        'time_mstep': time_mstep,
        'explained_var': explained_var,
        'reliability': reliability,
        'ntilde': ntilde,
    }


def run_gpytorch_adam(X, R, X_test, R_test, params, device, gradient_mode='autograd'):
    """Run GPyTorch with pure Adam training (no E-step)."""
    print("\n" + "="*60)
    print("Running GPyTorch (adam mode - no E-step)")
    print("="*60)

    cellid = params['cellid']
    ntilde = params['ntilde']
    n_train = params['n_train']
    n_px_side = params['n_px_side']

    # Select cell
    r = R[:, cellid]
    r_test = R_test[:, :, cellid]

    # Select training subset with SAME seed as varGP
    torch.manual_seed(42)
    indices = torch.randperm(X.shape[0], device=device)[:n_train]
    X_train = X[indices].double()
    r_train = r[indices].double()

    # Inducing points (first ntilde training points) - SAME as varGP
    inducing_points = X_train[:ntilde].clone()

    # Create kernel
    base_kernel = ArcCosineKernel(
        sigma_0=params['sigma_0'],
        n_px_side=n_px_side,
        eps_0x=params['eps_0x'],
        eps_0y=params['eps_0y'],
        beta=params['beta'],
        rho=params['rho'],
        use_mask=True,
        gradient_mode=gradient_mode,
    )
    kernel = gpytorch.kernels.ScaleKernel(base_kernel)
    kernel.outputscale = 1e-4

    # Create model and likelihood (GPyTorch defaults)
    model = VariationalGPModel(inducing_points, kernel, jitter=1e-4)
    likelihood = PoissonLikelihood(
        A_init=params['gpytorch_A_init'],
        lambda0_init=params['gpytorch_lambda0_init']
    )

    model = model.double().to(device)
    likelihood = likelihood.double().to(device)

    print(f"  A_init: {params['gpytorch_A_init']}, lambda0_init: {params['gpytorch_lambda0_init']}")
    print(f"  ntilde: {ntilde}, n_train: {n_train}")
    print(f"  n_iterations: {params['gpytorch_iterations']}, lr: {params['lr']}")

    # Train with pure Adam (no E-step) - need gradients enabled
    start_time = time.time()
    with torch.enable_grad():
        losses = train_adam(
            model, likelihood, X_train, r_train,
            n_iterations=params['gpytorch_iterations'],
            lr=params['lr'],
            print_every=params['gpytorch_iterations'] // 5,
            device=device,
        )
    elapsed = time.time() - start_time

    # Evaluate
    X_test_double = X_test.double()
    predictions = predict(model, likelihood, X_test_double, device=device)
    explained_var, reliability = compute_explained_variance(r_test.double(), predictions['f_pred'])

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Explained variance: {explained_var:.4f}")
    print(f"  Reliability: {reliability:.4f}")

    return {
        'implementation': 'GPyTorch (adam)',
        'time': elapsed,
        'explained_var': explained_var,
        'reliability': reliability,
        'ntilde': ntilde,
    }


def print_comparison_table(results, use_whitening=True, use_cache=True, n_train=500):
    """Print comparison table with timing breakdown."""
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    ntilde = results[0]['ntilde'] if results[0] else 'N/A'
    print(f"\nParameters: M={ntilde}, cell=8, n_train={n_train}, iter=50")
    print("\nInit: varGP & vargp_style use A=0.01, lambda0=1.0")
    print("      efm & adam use A=1.0, lambda0=0.0")
    print(f"\nvargp_style config: whitening={'ON' if use_whitening else 'OFF'}, cache={'ON' if use_cache else 'OFF'}")
    print()

    # Check if any result has timing breakdown
    has_timing_breakdown = any(r and 'time_estep' in r for r in results)

    if has_timing_breakdown:
        print("┌──────────────────────────────────┬───────────┬──────────┬──────────┬──────────┐")
        print("│ Implementation                   │ Expl. Var │ Total(s) │ E-step   │ M-step   │")
        print("├──────────────────────────────────┼───────────┼──────────┼──────────┼──────────┤")

        for r in results:
            if r is not None:
                impl = r['implementation'][:32].ljust(32)
                ev = f"{r['explained_var']:.4f}".ljust(9)
                t = f"{r['time']:.1f}".ljust(8)
                t_e = f"{r.get('time_estep', 0.0):.1f}".ljust(8) if 'time_estep' in r else "N/A     "
                t_m = f"{r.get('time_mstep', 0.0):.1f}".ljust(8) if 'time_mstep' in r else "N/A     "
                print(f"| {impl} | {ev} | {t} | {t_e} | {t_m} |")
            else:
                print(f"| {'FAILED'.ljust(32)} | {'N/A'.ljust(9)} | {'N/A'.ljust(8)} | {'N/A'.ljust(8)} | {'N/A'.ljust(8)} |")

        print("└──────────────────────────────────┴───────────┴──────────┴──────────┴──────────┘")
    else:
        print("┌──────────────────────────────────┬───────────┬──────────┐")
        print("│ Implementation                   │ Expl. Var │ Time (s) │")
        print("├──────────────────────────────────┼───────────┼──────────┤")

        for r in results:
            if r is not None:
                impl = r['implementation'][:32].ljust(32)
                ev = f"{r['explained_var']:.4f}".ljust(9)
                t = f"{r['time']:.1f}".ljust(8)
                print(f"| {impl} | {ev} | {t} |")
            else:
                print(f"| {'FAILED'.ljust(32)} | {'N/A'.ljust(9)} | {'N/A'.ljust(8)} |")

        print("└──────────────────────────────────┴───────────┴──────────┘")

    # Print differences
    valid_results = [r for r in results if r is not None]
    if len(valid_results) >= 2:
        vargp = next((r for r in valid_results if r['implementation'] == 'varGP'), None)
        vargp_style = next((r for r in valid_results if 'vargp_style' in r['implementation']), None)
        efm = next((r for r in valid_results if 'efm' in r['implementation']), None)
        adam = next((r for r in valid_results if 'adam' in r['implementation']), None)

        print("\nDifferences (vs varGP reference):")
        if vargp and vargp_style:
            diff = vargp['explained_var'] - vargp_style['explained_var']
            print(f"  varGP - vargp_style: {diff:+.4f}")
            if abs(diff) > 0.05:
                print("    ^ Gap > 0.05 - check vargp_style implementation")
        if vargp and efm:
            diff = vargp['explained_var'] - efm['explained_var']
            print(f"  varGP - efm:         {diff:+.4f}")
        if vargp and adam:
            diff = vargp['explained_var'] - adam['explained_var']
            print(f"  varGP - adam:        {diff:+.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='E-step comparison: varGP vs GPyTorch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default (whitening ON, caching ON):
    python tests/test_estep_comparison.py

    # Test different M values:
    python tests/test_estep_comparison.py --ntilde 75

    # Legacy mode (for comparison with old results):
    python tests/test_estep_comparison.py --no-whitening --no-cache
        """
    )
    parser.add_argument('--ntilde', type=int, default=PARAMS['ntilde'],
                        help=f"Number of inducing points M (default: {PARAMS['ntilde']})")
    parser.add_argument('--n-train', type=int, default=PARAMS['n_train'],
                        help=f"Number of training samples (default: {PARAMS['n_train']})")
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')

    # Whitening and caching options (vargp_style only)
    parser.add_argument('--use-whitening', action='store_true', default=True,
                        help='Use whitening conversions (default: ON)')
    parser.add_argument('--no-whitening', action='store_false', dest='use_whitening',
                        help='Disable whitening (legacy mode)')
    parser.add_argument('--use-cache', action='store_true', default=True,
                        help='Use kernel caching for E-step (default: ON)')
    parser.add_argument('--no-cache', action='store_false', dest='use_cache',
                        help='Disable kernel caching (legacy mode)')
    parser.add_argument('--unwhitened', action='store_true',
                        help='Use UnwhitenedVariationalStrategy (stores natural params directly, no L_K dependency)')
    parser.add_argument('--gradient-mode', type=str, default='autograd',
                        choices=['autograd', 'vjp', 'jacobian'],
                        help='Gradient computation mode (default: autograd)')

    args = parser.parse_args()

    # Update params from CLI args
    params = PARAMS.copy()
    params['ntilde'] = args.ntilde
    params['n_train'] = args.n_train

    device = torch.device(args.device)

    # Set reproducible seed (see tests/test_utils.py and HANDOFF_2026-01-18.md Section 17)
    set_reproducible_seed(42, device=args.device)

    print(f"Device: {device}")
    print(f"Testing with M={params['ntilde']} inducing points, n_train={params['n_train']}")
    print(f"vargp_style: whitening={'OFF' if args.unwhitened else ('ON' if args.use_whitening else 'conversions OFF')}, cache={'ON' if args.use_cache else 'OFF'}")
    if args.unwhitened:
        print("Using UnwhitenedVariationalStrategy (no L_K dependency)")
    if args.gradient_mode != 'autograd':
        print(f"Gradient mode: {args.gradient_mode}")

    # Load data (float32 for varGP compatibility)
    print("\nLoading data...")
    data = load_data(device, dtype=torch.float32)

    # Run all implementations
    results = []

    # 1. Reference: original varGP
    # Note: varGP prints E-step/M-step timing during run but doesn't return it
    result_vargp = run_vargp(
        data['X'], data['R'], data['X_test'], data['R_test'],
        params, device
    )
    results.append(result_vargp)

    # 2. vargp_style with current config (from CLI args)
    result_gpytorch_vargp_style = run_gpytorch_vargp_style(
        data['X'], data['R'], data['X_test'], data['R_test'],
        params, device,
        use_whitening=args.use_whitening,
        use_cache=args.use_cache,
        whitening=not args.unwhitened,
        gradient_mode=args.gradient_mode
    )
    results.append(result_gpytorch_vargp_style)

    # 3. vargp_style with legacy config (cache ON, whitening OFF) for comparison
    # Only run if current config is different from legacy
    if args.use_whitening or not args.use_cache:
        # Legacy = cache ON, whitening OFF (pre-whitening implementation)
        result_gpytorch_vargp_style_legacy = run_gpytorch_vargp_style(
            data['X'], data['R'], data['X_test'], data['R_test'],
            params, device,
            use_whitening=False,
            use_cache=True,
            gradient_mode=args.gradient_mode
        )
        results.append(result_gpytorch_vargp_style_legacy)

    # 4. efm mode
    result_gpytorch_efm = run_gpytorch_efm(
        data['X'], data['R'], data['X_test'], data['R_test'],
        params, device,
        gradient_mode=args.gradient_mode
    )
    results.append(result_gpytorch_efm)

    # 5. adam mode
    result_gpytorch_adam = run_gpytorch_adam(
        data['X'], data['R'], data['X_test'], data['R_test'],
        params, device,
        gradient_mode=args.gradient_mode
    )
    results.append(result_gpytorch_adam)

    # Print comparison
    print_comparison_table(results, use_whitening=args.use_whitening, use_cache=args.use_cache, n_train=params['n_train'])

    return results


if __name__ == '__main__':
    main()
