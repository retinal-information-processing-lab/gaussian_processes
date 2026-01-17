#!/usr/bin/env python3
"""
E-step Implementation Comparison: varGP vs GPyTorch

This is the CANONICAL test script for comparing the reference varGP implementation
with the GPyTorch port. All parameters are frozen to match one_cell_fit.py defaults.

Usage:
    python tests/test_estep_comparison.py
    python tests/test_estep_comparison.py --ntilde 75  # Test different M values

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

IMPORTANT - Link Function Initialization Differs:
    ┌─────────────┬─────────┬────────────┐
    │ Parameter   │ varGP   │ GPyTorch   │
    ├─────────────┼─────────┼────────────┤
    │ A_init      │ 0.01    │ 1.0        │
    │ lambda0_init│ 1.0     │ 0.0        │
    └─────────────┴─────────┴────────────┘

    Per Q19 in CLAUDE.md, the model is robust to A initialization - both converge
    to similar Pearson r. However, this difference should be noted when interpreting
    results, especially if comparing intermediate training states or final parameter
    values (not just accuracy metrics).

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
from model import VariationalGPModel
from estep import train_efm
from train import train_adam, predict, compute_pearson_correlation, compute_explained_variance


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
    'gpytorch_A_init': 1.0,
    'gpytorch_lambda0_init': 0.0,
}


def load_data(device, dtype=torch.float32):
    """Load PNAS data with consistent preprocessing."""
    data_path = Path(__file__).parent.parent.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
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


def run_gpytorch_efm(X, R, X_test, R_test, params, device):
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


def run_gpytorch_adam(X, R, X_test, R_test, params, device):
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


def print_comparison_table(results):
    """Print comparison table."""
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    ntilde = results[0]['ntilde'] if results[0] else 'N/A'
    print(f"\nParameters: M={ntilde}, cell=8, n_train=500, iter=50")
    print("\nIMPORTANT: A_init differs (varGP=0.01, GPyTorch=1.0)")
    print("           lambda0_init differs (varGP=1.0, GPyTorch=0.0)")
    print()

    print("┌────────────────────┬───────────────┬──────────┐")
    print("│ Implementation     │ Expl. Var     │ Time (s) │")
    print("├────────────────────┼───────────────┼──────────┤")

    for r in results:
        if r is not None:
            impl = r['implementation'][:18].ljust(18)
            ev = f"{r['explained_var']:.4f}".ljust(13)
            t = f"{r['time']:.1f}".ljust(8)
            print(f"│ {impl} │ {ev} │ {t} │")
        else:
            print(f"│ {'FAILED'.ljust(18)} │ {'N/A'.ljust(13)} │ {'N/A'.ljust(8)} │")

    print("└────────────────────┴───────────────┴──────────┘")

    # Print differences
    valid_results = [r for r in results if r is not None]
    if len(valid_results) >= 2:
        vargp = next((r for r in valid_results if 'varGP' in r['implementation']), None)
        efm = next((r for r in valid_results if 'efm' in r['implementation']), None)
        adam = next((r for r in valid_results if 'adam' in r['implementation']), None)

        print("\nDifferences:")
        if vargp and efm:
            diff = vargp['explained_var'] - efm['explained_var']
            print(f"  varGP - GPyTorch(efm):  {diff:+.4f}")
            if abs(diff) > 0.1:
                print("    ^ WARNING: Large gap - investigate E-step implementation")
        if vargp and adam:
            diff = vargp['explained_var'] - adam['explained_var']
            print(f"  varGP - GPyTorch(adam): {diff:+.4f}")
        if efm and adam:
            diff = efm['explained_var'] - adam['explained_var']
            print(f"  GPyTorch(efm) - GPyTorch(adam): {diff:+.4f}")
            if diff > 0.05:
                print("    ^ E-step provides benefit over pure Adam")


def main():
    parser = argparse.ArgumentParser(description='E-step comparison: varGP vs GPyTorch')
    parser.add_argument('--ntilde', type=int, default=PARAMS['ntilde'],
                        help=f"Number of inducing points M (default: {PARAMS['ntilde']})")
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    args = parser.parse_args()

    # Update ntilde if specified
    params = PARAMS.copy()
    params['ntilde'] = args.ntilde

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Testing with M={params['ntilde']} inducing points")

    # Load data (float32 for varGP compatibility)
    print("\nLoading data...")
    data = load_data(device, dtype=torch.float32)

    # Run all three implementations
    results = []

    result_vargp = run_vargp(
        data['X'], data['R'], data['X_test'], data['R_test'],
        params, device
    )
    results.append(result_vargp)

    result_gpytorch_efm = run_gpytorch_efm(
        data['X'], data['R'], data['X_test'], data['R_test'],
        params, device
    )
    results.append(result_gpytorch_efm)

    result_gpytorch_adam = run_gpytorch_adam(
        data['X'], data['R'], data['X_test'], data['R_test'],
        params, device
    )
    results.append(result_gpytorch_adam)

    # Print comparison
    print_comparison_table(results)

    return results


if __name__ == '__main__':
    main()
