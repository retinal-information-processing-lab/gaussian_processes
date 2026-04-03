#!/usr/bin/env python3
"""
Test A: Reference Implementation Comparison (Apples-to-Apples)

Runs both reference varGP() and GPyTorch on same data with MATCHED initial parameters.
Compares Pearson r and final parameters after ensuring convergence.

Created by Claude for validation testing.

MATCHED PARAMETERS:
- A_init = 0.01 (both)
- lambda0_init = 1.0 (both)
- Amp = 1.0 (both) - GPyTorch now uses Amp directly inside C matrix (matching legacy)
- beta = 0.1, rho = 0.1, sigma_0 = 1.0, eps_0 = (0, 0)

CONVERGENCE SETTINGS:
- Reference: nEstep=30, nMstep=30, maxiter=150
- GPyTorch: 1500 iterations (equivalent work)
"""

import sys
import time
import numpy as np
from pathlib import Path

import torch
torch.set_grad_enabled(False)

# Add repo parent to sys.path for `from gaussian_processes.Spatial_GP_repo import ...`
_repo_root = next(p for p in Path(__file__).resolve().parents if (p / 'Spatial_GP_repo').is_dir())
sys.path.insert(0, str(_repo_root.parent))
from gaussian_processes.Spatial_GP_repo import utils as GP_utils

# Import GPyTorch components
import gpytorch
from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from model import VariationalGPModel
from train import train_gpy_default, predict, compute_pearson_correlation


# ============================================================================
# SHARED CONFIGURATION - Same for both implementations
# ============================================================================
CONFIG = {
    # Data settings
    'cellid': 15,
    'ntilde': 50,
    'n_train': 2000,
    'n_px_side': 108,

    # Initial hyperparameters (MATCHED)
    'beta_init': 0.1,
    'rho_init': 0.1,
    'sigma_0_init': 1.0,
    'eps_0x_init': 0.0,
    'eps_0y_init': 0.0,
    'Amp_init': 1.0,       # Both use Amp (GPyTorch now matches legacy)

    # Initial link function parameters (MATCHED)
    'A_init': 0.01,
    'lambda0_init': 1.0,

    # Convergence settings - Reference
    'ref_maxiter': 400,
    'ref_nEstep': 30,
    'ref_nMstep': 30,
    'ref_nFparamstep': 10,

    # Convergence settings - GPyTorch
    # 400 iterations * 60 inner steps ≈ 24000 gradient updates
    # Using 4000 Adam iterations as equivalent
    'gpytorch_iterations': 4000,
    'gpytorch_lr': 0.01,
}


def run_reference_varGP(X, R, X_test, R_test, device):
    """Run reference varGP implementation with matched parameters."""
    print("\n--- Running Reference varGP ---")

    cfg = CONFIG

    # Set seeds
    torch.manual_seed(42)

    # Select training subset and inducing points
    perm = torch.randperm(X.shape[0], device=device)
    train_idx = perm[:cfg['n_train']]
    xtilde_idx = perm[:cfg['ntilde']]

    X_train_ref = X[train_idx]
    r_train = R[train_idx, cfg['cellid']]
    xtilde = X[xtilde_idx]

    print(f"  Training samples: {X_train_ref.shape[0]}")
    print(f"  Inducing points: {xtilde.shape[0]}")

    # Initialize hyperparameters - MATCHED to GPyTorch
    dtype = torch.float32
    beta = torch.tensor(cfg['beta_init'], dtype=dtype, device=device)
    rho = torch.tensor(cfg['rho_init'], dtype=dtype, device=device)
    logbetaexpr = -2 * torch.log(2 * beta)
    logrhoexpr = -torch.log(2 * rho * rho)

    theta = {
        'sigma_0': torch.tensor(cfg['sigma_0_init'], dtype=dtype, device=device),
        'Amp': torch.tensor(cfg['Amp_init'], dtype=dtype, device=device),
        'eps_0x': torch.tensor(cfg['eps_0x_init'], dtype=dtype, device=device),
        'eps_0y': torch.tensor(cfg['eps_0y_init'], dtype=dtype, device=device),
        '-2log2beta': logbetaexpr,
        '-log2rho2': logrhoexpr
    }

    for key in theta:
        theta[key] = theta[key].requires_grad_()

    hyperparams_tuple = GP_utils.generate_theta(
        x=X_train_ref, r=r_train, n_px_side=cfg['n_px_side'], display_hyper=False, **theta
    )

    # Link function parameters - MATCHED to GPyTorch
    A = torch.tensor(cfg['A_init'], dtype=dtype, device=device)
    logA = torch.log(A)
    lambda0 = torch.tensor(cfg['lambda0_init'], dtype=dtype, device=device)
    f_params = {'logA': logA.requires_grad_(), 'lambda0': lambda0}

    # Fit parameters - convergence settings
    fit_parameters = {
        'ntilde': cfg['ntilde'],
        'maxiter': cfg['ref_maxiter'],
        'nMstep': cfg['ref_nMstep'],
        'nEstep': cfg['ref_nEstep'],
        'nFparamstep': cfg['ref_nFparamstep'],
        'kernfun': GP_utils.acosker,
        'cellid': cfg['cellid'],
        'n_px_side': cfg['n_px_side'],
    }

    args = {
        'fit_parameters': fit_parameters,
        'xtilde': xtilde,
        'hyperparams_tuple': hyperparams_tuple,
        'f_params': f_params,
        'm': torch.zeros(cfg['ntilde'], dtype=dtype, device=device)
    }

    print(f"  Settings: maxiter={cfg['ref_maxiter']}, nEstep={cfg['ref_nEstep']}, nMstep={cfg['ref_nMstep']}")
    print(f"  Initial: A={cfg['A_init']}, lambda0={cfg['lambda0_init']}, Amp={cfg['Amp_init']}")
    print("  Starting varGP fit...")
    start_time = time.time()

    try:
        fit_model, err_dict = GP_utils.varGP(X_train_ref, r_train, **args)
        elapsed = time.time() - start_time
        print(f"  Fitting completed in {elapsed:.1f}s")

        if err_dict['is_error']:
            print(f"  Error during fitting: {err_dict['error']}")
            return None

        # Test - compute predictions using eigenspace projection
        print("  Testing...")

        X_test_flat = X_test.reshape(-1, cfg['n_px_side'] * cfg['n_px_side'])
        R_test_cell = R_test[:, :, cfg['cellid']]

        # Get model quantities
        theta_model = fit_model['hyperparams_tuple'][0]
        C = fit_model['C']
        m_b = fit_model['m_b']
        V_b = fit_model['V_b']
        B = fit_model['B']
        K_tilde_b = fit_model['K_tilde_b']
        K_tilde_inv_b = fit_model['K_tilde_inv_b']
        xtilde = fit_model['xtilde']
        mask = fit_model['mask']
        kernfun = fit_model['fit_parameters']['kernfun']

        # Mask the test data
        X_test_masked = X_test_flat[:, mask]
        xtilde_masked = xtilde[:, mask]

        # Compute kernel and project to eigenspace
        K_test = kernfun(theta_model, X_test_masked, xtilde_masked, C=C, diag=False)
        K_test_b = K_test @ B
        u_b = K_test_b @ K_tilde_inv_b

        # Mean and variance
        lambda_mean = u_b @ m_b
        k0 = kernfun(theta_model, X_test_masked, X_test_masked, C=C, diag=True)
        lambda_var = k0 + torch.sum(u_b * (V_b @ u_b.T).T - u_b * (K_tilde_b @ u_b.T).T, dim=1)
        lambda_var = torch.clamp(lambda_var, min=1e-6)

        # Convert to firing rate
        A_val = torch.exp(fit_model['f_params']['logA'])
        lambda0_val = fit_model['f_params']['lambda0']
        f_pred = torch.exp(A_val * lambda_mean + 0.5 * A_val**2 * lambda_var + lambda0_val)

        # Compute Pearson r
        spk_test_mean = R_test_cell.mean(dim=0)
        pearson_r = compute_pearson_correlation(spk_test_mean, f_pred)

        # Extract final parameters
        theta_final = fit_model['hyperparams_tuple'][0]
        f_params_final = fit_model['f_params']

        raw_beta = theta_final['-2log2beta'].item()
        raw_rho = theta_final['-log2rho2'].item()
        beta_final = np.exp(-raw_beta / 2) / 2
        rho_final = np.sqrt(np.exp(-raw_rho) / 2)
        A_final = np.exp(f_params_final['logA'].item())
        Amp_final = theta_final['Amp'].item()

        print(f"\n  Reference Results:")
        print(f"    Pearson r: {pearson_r:.4f}")
        print(f"    beta: {cfg['beta_init']:.4f} -> {beta_final:.4f}")
        print(f"    rho: {cfg['rho_init']:.4f} -> {rho_final:.4f}")
        print(f"    eps_0: ({cfg['eps_0x_init']:.4f}, {cfg['eps_0y_init']:.4f}) -> ({theta_final['eps_0x'].item():.4f}, {theta_final['eps_0y'].item():.4f})")
        print(f"    A: {cfg['A_init']:.4f} -> {A_final:.4f}")
        print(f"    lambda0: {cfg['lambda0_init']:.4f} -> {f_params_final['lambda0'].item():.4f}")
        print(f"    Amp: {cfg['Amp_init']:.4f} -> {Amp_final:.4f}")

        return {
            'pearson_r': pearson_r,
            'beta': beta_final,
            'rho': rho_final,
            'eps_0x': theta_final['eps_0x'].item(),
            'eps_0y': theta_final['eps_0y'].item(),
            'A': A_final,
            'lambda0': f_params_final['lambda0'].item(),
            'Amp': Amp_final,
            'time': elapsed,
        }

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_gpytorch(X, R, X_test, R_test, device):
    """Run GPyTorch implementation with matched parameters."""
    print("\n--- Running GPyTorch ---")

    cfg = CONFIG

    # Convert to float64 for numerical stability
    X = X.double()
    R = R.double()
    X_test = X_test.double()
    R_test = R_test.double()

    # Set seeds
    torch.manual_seed(42)

    r = R[:, cfg['cellid']]
    r_test = R_test[:, :, cfg['cellid']]

    # Select inducing points and training data (same permutation as reference)
    perm = torch.randperm(X.shape[0], device=device)
    indices_inducing = perm[:cfg['ntilde']]
    indices_train = perm[:cfg['n_train']]

    inducing_points = X[indices_inducing].clone()
    X_train = X[indices_train]
    r_train = r[indices_train]

    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Inducing points: {inducing_points.shape[0]}")

    # Create model with RF structure - MATCHED to reference
    # ArcCosineKernel now has internal Amp parameter (matches legacy varGP)
    # No need for ScaleKernel wrapper
    kernel = ArcCosineKernel(
        sigma_0=cfg['sigma_0_init'],
        Amp=cfg['Amp_init'],  # MATCHED to legacy
        n_px_side=cfg['n_px_side'],
        eps_0x=cfg['eps_0x_init'],
        eps_0y=cfg['eps_0y_init'],
        beta=cfg['beta_init'],
        rho=cfg['rho_init']
    )

    model = VariationalGPModel(inducing_points, kernel, jitter=1e-4)
    likelihood = PoissonLikelihood(
        A_init=cfg['A_init'],          # MATCHED
        lambda0_init=cfg['lambda0_init']  # MATCHED
    )

    model = model.double()
    likelihood = likelihood.double().to(device)

    print(f"  Settings: iterations={cfg['gpytorch_iterations']}, lr={cfg['gpytorch_lr']}")
    print(f"  Initial: A={cfg['A_init']}, lambda0={cfg['lambda0_init']}, Amp={cfg['Amp_init']}")
    print("  Training...")
    start_time = time.time()

    with torch.enable_grad():
        losses = train_gpy_default(
            model, likelihood, X_train, r_train,
            optimizer_name='adam',
            lr=cfg['gpytorch_lr'],
            n_iterations=cfg['gpytorch_iterations'],
            print_every=cfg['gpytorch_iterations'] // 5
        )
    elapsed = time.time() - start_time
    print(f"  Training completed in {elapsed:.1f}s")

    # Get final parameters
    raw_beta = kernel.raw_m2log2beta.item()
    raw_rho = kernel.raw_mlog2rho2.item()
    beta_final = np.exp(-raw_beta / 2) / 2
    rho_final = np.sqrt(np.exp(-raw_rho) / 2)
    Amp_final = kernel.Amp.item()

    # Evaluate
    print("  Testing...")
    predictions = predict(model, likelihood, X_test.reshape(-1, cfg['n_px_side'] * cfg['n_px_side']))
    r_test_mean = r_test.mean(dim=0)
    pearson_r = compute_pearson_correlation(r_test_mean, predictions['f_pred'])

    print(f"\n  GPyTorch Results:")
    print(f"    Pearson r: {pearson_r:.4f}")
    print(f"    beta: {cfg['beta_init']:.4f} -> {beta_final:.4f}")
    print(f"    rho: {cfg['rho_init']:.4f} -> {rho_final:.4f}")
    print(f"    eps_0: ({cfg['eps_0x_init']:.4f}, {cfg['eps_0y_init']:.4f}) -> ({kernel.eps_0x.item():.4f}, {kernel.eps_0y.item():.4f})")
    print(f"    A: {cfg['A_init']:.4f} -> {likelihood.A.item():.4f}")
    print(f"    lambda0: {cfg['lambda0_init']:.4f} -> {likelihood.lambda0.item():.4f}")
    print(f"    Amp: {cfg['Amp_init']:.4f} -> {Amp_final:.4f}")

    return {
        'pearson_r': pearson_r,
        'beta': beta_final,
        'rho': rho_final,
        'eps_0x': kernel.eps_0x.item(),
        'eps_0y': kernel.eps_0y.item(),
        'A': likelihood.A.item(),
        'lambda0': likelihood.lambda0.item(),
        'Amp': Amp_final,
        'time': elapsed,
    }


def main():
    print("=" * 70)
    print("APPLES-TO-APPLES COMPARISON: Reference varGP vs GPyTorch")
    print("=" * 70)

    cfg = CONFIG

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Use float32 initially (reference uses float32)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)

    print(f"\n--- MATCHED INITIAL PARAMETERS ---")
    print(f"  cellid={cfg['cellid']}, ntilde={cfg['ntilde']}, n_train={cfg['n_train']}")
    print(f"  beta={cfg['beta_init']}, rho={cfg['rho_init']}, sigma_0={cfg['sigma_0_init']}")
    print(f"  eps_0=({cfg['eps_0x_init']}, {cfg['eps_0y_init']})")
    print(f"  A={cfg['A_init']}, lambda0={cfg['lambda0_init']}")
    print(f"  Amp={cfg['Amp_init']} (MATCHED - both use Amp in C matrix)")

    print(f"\n--- CONVERGENCE SETTINGS ---")
    print(f"  Reference: maxiter={cfg['ref_maxiter']}, nEstep={cfg['ref_nEstep']}, nMstep={cfg['ref_nMstep']}")
    print(f"  GPyTorch: iterations={cfg['gpytorch_iterations']}, lr={cfg['gpytorch_lr']}")

    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    print(f"\nLoading data from: {data_path}")

    data = np.load(data_path)
    X_train = torch.tensor(data['images_train'], dtype=torch.float32, device=device)
    X_val = torch.tensor(data['images_val'], dtype=torch.float32, device=device)
    X_test = torch.tensor(data['images_test'], dtype=torch.float32, device=device)
    R_train = torch.tensor(data['responses_train'], dtype=torch.float32, device=device)
    R_val = torch.tensor(data['responses_val'], dtype=torch.float32, device=device)
    R_test = torch.tensor(data['responses_test'], dtype=torch.float32, device=device)

    # Combine train + val
    X = torch.cat([X_train, X_val], dim=0)
    R = torch.cat([R_train, R_val], dim=0)

    # Flatten images
    X_flat = X.reshape(X.shape[0], -1)

    print(f"X shape: {X_flat.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Run reference
    ref_results = run_reference_varGP(X_flat, R, X_test, R_test, device)

    # Run GPyTorch
    gpytorch_results = run_gpytorch(X_flat, R, X_test, R_test, device)

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY (Apples-to-Apples)")
    print("=" * 70)

    print("\n| Parameter | Initial | Ref Final | GPyTorch Final | Diff |")
    print("|-----------|---------|-----------|----------------|------|")

    if ref_results and gpytorch_results:
        # Pearson r
        diff_r = abs(gpytorch_results['pearson_r'] - ref_results['pearson_r'])
        print(f"| Pearson r | - | {ref_results['pearson_r']:.4f} | {gpytorch_results['pearson_r']:.4f} | {diff_r:.4f} |")

        # Hyperparameters
        diff_beta = abs(gpytorch_results['beta'] - ref_results['beta'])
        print(f"| beta | {cfg['beta_init']:.4f} | {ref_results['beta']:.4f} | {gpytorch_results['beta']:.4f} | {diff_beta:.4f} |")

        diff_rho = abs(gpytorch_results['rho'] - ref_results['rho'])
        print(f"| rho | {cfg['rho_init']:.4f} | {ref_results['rho']:.4f} | {gpytorch_results['rho']:.4f} | {diff_rho:.4f} |")

        diff_eps0x = abs(gpytorch_results['eps_0x'] - ref_results['eps_0x'])
        print(f"| eps_0x | {cfg['eps_0x_init']:.4f} | {ref_results['eps_0x']:.4f} | {gpytorch_results['eps_0x']:.4f} | {diff_eps0x:.4f} |")

        diff_eps0y = abs(gpytorch_results['eps_0y'] - ref_results['eps_0y'])
        print(f"| eps_0y | {cfg['eps_0y_init']:.4f} | {ref_results['eps_0y']:.4f} | {gpytorch_results['eps_0y']:.4f} | {diff_eps0y:.4f} |")

        # Link function params
        diff_A = abs(gpytorch_results['A'] - ref_results['A'])
        print(f"| A | {cfg['A_init']:.4f} | {ref_results['A']:.4f} | {gpytorch_results['A']:.4f} | {diff_A:.4f} |")

        diff_lam = abs(gpytorch_results['lambda0'] - ref_results['lambda0'])
        print(f"| lambda0 | {cfg['lambda0_init']:.4f} | {ref_results['lambda0']:.4f} | {gpytorch_results['lambda0']:.4f} | {diff_lam:.4f} |")

        # Kernel scale (both now use Amp)
        diff_scale = abs(gpytorch_results['Amp'] - ref_results['Amp'])
        print(f"| Amp | {cfg['Amp_init']:.4f} | {ref_results['Amp']:.4f} | {gpytorch_results['Amp']:.4f} | {diff_scale:.4f} |")

        # Time
        print(f"| Time (s) | - | {ref_results['time']:.1f} | {gpytorch_results['time']:.1f} | - |")

        # Success criteria
        print("\n--- ASSESSMENT ---")
        if diff_r < 0.05:
            print(f"EXCELLENT: Pearson r difference = {diff_r:.4f} (< 0.05)")
        elif diff_r < 0.1:
            print(f"GOOD: Pearson r difference = {diff_r:.4f} (< 0.1)")
        else:
            print(f"WARNING: Pearson r difference = {diff_r:.4f} (>= 0.1)")

        # Check parameter convergence
        param_diffs = [diff_beta, diff_rho, diff_A]
        if all(d < 0.05 for d in param_diffs):
            print("Parameters converged to similar values (diffs < 0.05)")
        else:
            print(f"Parameter differences: beta={diff_beta:.4f}, rho={diff_rho:.4f}, A={diff_A:.4f}")

    else:
        print("One or both implementations failed - cannot compare")

    return ref_results, gpytorch_results


if __name__ == '__main__':
    main()
