#!/usr/bin/env python3
"""
run_single_mode.py - Run a single training mode on PNAS neural data.

For experimentation and development. Run ONE training mode with full CLI control.
For canonical benchmarks comparing all modes, use run_benchmark.py instead.

Training modes:
  - vargp_old: Original varGP implementation (reference)
  - adam: Pure Adam optimization (no E-step)
  - efm: E-F-M loop (1 E-step, n F-steps, n M-steps per iteration)
  - vargp_style: GPyTorch matching original varGP training structure

Usage:
    python run_single_mode.py --ntilde 50 --mode vargp_old       # Reference implementation
    python run_single_mode.py --ntilde 50 --mode adam
    python run_single_mode.py --ntilde 50 --mode efm
    python run_single_mode.py --ntilde 50 --mode vargp_style
    python run_single_mode.py  # Uses defaults: M=50, mode=efm

Gradient modes (for GPyTorch modes only):
    --gradient-mode autograd   # PyTorch autograd (default)
    --gradient-mode vjp        # VJP analytical - same speed as autograd
    --gradient-mode jacobian   # Old Jacobian materialization - slow but matches varGP

IMPORTANT - Link Function Initialization:
    ┌─────────────┬─────────────────┬────────────────────┐
    │ Parameter   │ adam/efm        │ vargp_old/vargp_style  │
    ├─────────────┼─────────────────┼────────────────────┤
    │ A_init      │ 1.0             │ 0.01               │
    │ lambda0_init│ 0.0             │ 1.0                │
    └─────────────┴─────────────────┴────────────────────┘
"""

import sys
import time
import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add paths for imports
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject')
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/torchlambertw')

import torch
import gpytorch
import matplotlib.pyplot as plt

# NOTE: GP_utils is imported lazily inside vargp_old mode to avoid
# side effects (torch.set_grad_enabled(False) at utils.py line 2)

# Import our GPyTorch components
from kernels import ArcCosineKernel, GRADIENT_MODES
from likelihoods import PoissonLikelihood
from model import VariationalGPModel
from train import train_efm, train_varGP_style
from train import train_adam, predict, compute_pearson_correlation, compute_explained_variance
from tests.test_utils import set_reproducible_seed


def get_git_commit():
    """Get current git commit hash (short form)."""
    import subprocess
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).parent
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_pnas_data(data_path, dtype=torch.float64):
    """Load PNAS dataset."""
    data = np.load(data_path)
    return {
        'X_train': torch.tensor(data['images_train'], dtype=dtype),
        'X_val': torch.tensor(data['images_val'], dtype=dtype),
        'X_test': torch.tensor(data['images_test'], dtype=dtype),
        'R_train': torch.tensor(data['responses_train'], dtype=dtype),
        'R_val': torch.tensor(data['responses_val'], dtype=dtype),
        'R_test': torch.tensor(data['responses_test'], dtype=dtype),
    }


def plot_fit(r_test_mean, f_pred, cellid, ntilde, test_corr, explained_var, reliability, output_path=None):
    """Plot actual vs predicted firing rates.

    Args:
        r_test_mean: Mean actual firing rates, shape (n_images,)
        f_pred: Predicted firing rates, shape (n_images,)
        cellid: Cell ID for title
        ntilde: Number of inducing points M
        test_corr: Pearson correlation on test set
        explained_var: Explained variance value
        reliability: Cell reliability
        output_path: If provided, save figure to this path instead of showing
    """
    r_actual = r_test_mean.cpu().numpy()
    f_predicted = f_pred.cpu().numpy()

    # Sort by actual firing rate for second subplot
    sort_idx = np.argsort(r_actual)
    r_sorted = r_actual[sort_idx]
    f_sorted = f_predicted[sort_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n_images = len(r_actual)
    x = np.arange(n_images)

    # Left subplot: original order
    for xi in x:
        ax1.axvline(xi, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax1.plot(x, r_actual, 'k-', linewidth=1.5, label='Actual (mean of 30 reps)')
    ax1.plot(x, f_predicted, 'r-', linewidth=1.5, label='Predicted')
    ax1.set_xlabel('Test image index')
    ax1.set_ylabel('Firing rate (spikes)')
    ax1.set_title('Original order')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Right subplot: sorted by actual firing rate
    for xi in x:
        ax2.axvline(xi, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax2.plot(x, r_sorted, 'k-', linewidth=1.5, label='Actual (mean of 30 reps)')
    ax2.plot(x, f_sorted, 'r-', linewidth=1.5, label='Predicted')
    ax2.set_xlabel('Test images (sorted by actual firing rate)')
    ax2.set_ylabel('Firing rate (spikes)')
    ax2.set_title('Sorted by actual firing rate')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Add metrics text box to right subplot
    metrics_text = (f'M = {ntilde}\n'
                    f'Pearson r = {test_corr:.3f}\n'
                    f'Reliability = {reliability:.3f}\n'
                    f'Expl. var = {explained_var:.3f}')
    ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'Cell {cellid} - M={ntilde}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description='Test E-step on PNAS data')
    parser.add_argument('--cell', type=int, default=8, help='Cell ID (default: 8)')
    parser.add_argument('--ntilde', type=int, default=50, help='Number of inducing points M (default: 50)')
    parser.add_argument('--n-train', type=int, default=500, help='Number of training samples (default: 500)')
    parser.add_argument('--n-iterations', type=int, default=50, help='Number of EM iterations (default: 50)')
    parser.add_argument('--n-estep', type=int, default=10, help='E-steps per iteration (default: 10)')
    parser.add_argument('--n-fstep', type=int, default=10, help='F-steps per iteration (efm mode, default: 10)')
    parser.add_argument('--n-mstep', type=int, default=10, help='M-steps per iteration (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (default: cuda)')
    parser.add_argument('--mode', type=str, default='efm',
                        choices=['vargp_old', 'adam', 'efm', 'vargp_style'],
                        help='Training mode: vargp_old (reference), adam (no E-step), efm (E-F-M loop), vargp_style (GPyTorch matching varGP)')

    # RF parameters - use defaults that work
    parser.add_argument('--beta', type=float, default=0.1, help='RF size (default: 0.1)')
    parser.add_argument('--rho', type=float, default=0.1, help='Smoothness (default: 0.1)')
    parser.add_argument('--use-mask', action='store_true', default=True, help='Use pixel masking')
    parser.add_argument('--no-mask', action='store_false', dest='use_mask')
    parser.add_argument('--gradient-mode', type=str, default='autograd',
                        choices=list(GRADIENT_MODES),
                        help='Gradient computation mode: autograd (default), vjp (fast analytical), jacobian (slow, matches varGP)')

    # Performance options
    parser.add_argument('--use-cache', action='store_true', default=True,
                        help='Use kernel caching in E-step (default: True, 11.7x fewer kernel calls)')
    parser.add_argument('--no-cache', action='store_false', dest='use_cache',
                        help='Disable kernel caching (for testing fallback path)')
    parser.add_argument('--no-whitening', action='store_true',
                        help='Disable whitening conversions (for debugging/comparison)')
    parser.add_argument('--unwhitened', action='store_true',
                        help='Use UnwhitenedVariationalStrategy (stores natural params directly, no L_K dependency)')

    # Plotting options
    parser.add_argument('--plot', action='store_true',
                        help='Show plot of actual vs predicted firing rates')
    parser.add_argument('--save-plot', type=str, default='auto',
                        help='Save plot path. "auto" saves to imgs/{mode}_M{ntilde}.png, "none" to disable')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    # JSON output for benchmark tracking
    parser.add_argument('--json-append', type=str, default=None,
                        help='Append results as JSON line to specified file (for benchmark tracking)')

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"M={args.ntilde} inducing points")
    if args.gradient_mode != 'autograd':
        print(f"Gradient mode: {args.gradient_mode}")
    if args.unwhitened:
        print("Using UnwhitenedVariationalStrategy (no L_K dependency)")

    # Set seed with explicit CUDA init for reproducibility
    # See tests/test_utils.py and HANDOFF_2026-01-18.md Section 20 for details
    set_reproducible_seed(args.seed, device=device)
    print(f"Seed: {args.seed}")

    # Load data
    data_path = Path(__file__).parent.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    print(f"Loading data from: {data_path}")
    data = load_pnas_data(data_path)

    # Combine train + val, flatten
    X = torch.cat([data['X_train'], data['X_val']], dim=0)
    R = torch.cat([data['R_train'], data['R_val']], dim=0)
    X = X.reshape(X.shape[0], -1).to(device)  # (N, 11664)
    R = R.to(device)

    X_test = data['X_test'].reshape(data['X_test'].shape[0], -1).to(device)
    R_test = data['R_test'].to(device)

    # Select cell
    r = R[:, args.cell]
    r_test = R_test[:, :, args.cell]  # (30 repeats, 30 images)

    # Select training subset
    n_train = min(args.n_train, X.shape[0])
    indices_train = torch.randperm(X.shape[0], device=device)[:n_train]
    X_train = X[indices_train]
    r_train = r[indices_train]

    # Select inducing points
    ntilde = min(args.ntilde, n_train)
    indices_inducing = indices_train[:ntilde]  # Use first ntilde training points
    inducing_points = X[indices_inducing].clone()

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  r_train: {r_train.shape}")
    print(f"  inducing_points: {inducing_points.shape}")
    print(f"  X_test: {X_test.shape}")

    n_px_side = 108

    # =========================================================================
    # VARGP MODE: Use original varGP implementation
    # =========================================================================
    if args.mode == 'vargp_old':
        # Lazy import to avoid side effects (utils.py disables gradients at module level)
        from gaussian_processes.Spatial_GP_repo import utils as GP_utils

        print(f"\nRunning original varGP (reference implementation)")
        print(f"  maxiter={args.n_iterations}, nEstep={args.n_estep}, nMstep={args.n_mstep}, nFparamstep={args.n_fstep}")

        # Initialize hyperparameters (matching test_estep_comparison.py exactly)
        beta = torch.tensor(args.beta, device=device)
        rho = torch.tensor(args.rho, device=device)

        theta = {
            'sigma_0': torch.tensor(1.0, device=device).requires_grad_(),
            'Amp': torch.tensor(1.0, device=device).requires_grad_(),
            'eps_0x': torch.tensor(0.0, device=device).requires_grad_(),
            'eps_0y': torch.tensor(0.0, device=device).requires_grad_(),
            '-2log2beta': (-2 * torch.log(2 * beta)).requires_grad_(),
            '-log2rho2': (-torch.log(2 * rho * rho)).requires_grad_(),
        }

        # Need float32 for varGP
        X_train_f32 = X_train.float()
        r_train_f32 = r_train.float()

        hyperparams_tuple = GP_utils.generate_theta(
            x=X_train_f32, r=r_train_f32, n_px_side=n_px_side, display_hyper=False, **theta
        )

        # Link function parameters (varGP defaults)
        A_init = 0.01
        lambda0_init = 1.0
        A = torch.tensor(A_init, device=device)
        f_params = {
            'logA': torch.log(A).requires_grad_(),
            'lambda0': torch.tensor(lambda0_init, device=device),
        }

        fit_parameters = {
            'ntilde': ntilde,
            'maxiter': args.n_iterations,
            'nMstep': args.n_mstep,
            'nEstep': args.n_estep,
            'nFparamstep': args.n_fstep,
            'kernfun': GP_utils.acosker,
            'cellid': args.cell,
            'n_px_side': n_px_side,
        }

        vargp_old_args = {
            'fit_parameters': fit_parameters,
            'xtilde': inducing_points.float(),
            'hyperparams_tuple': hyperparams_tuple,
            'f_params': f_params,
            'm': torch.zeros(ntilde, device=device),
        }

        print(f"  A_init: {A_init}, lambda0_init: {lambda0_init}")

        # Run varGP
        start_time = time.time()
        fit_model, err_dict = GP_utils.varGP(X_train_f32, r_train_f32, **vargp_old_args)
        train_time = time.time() - start_time

        if err_dict['is_error']:
            print(f"  ERROR: {err_dict['error']}")
            return None

        # Evaluate using GP_utils.test
        # test() returns (R_test_cell, R_pred_cell, expl_var, sigma_expl_var)
        # R_pred_cell is already the predicted firing rate (not lambda)
        _, f_pred, r2, sigma_r2 = GP_utils.test(
            X_test.reshape(-1, n_px_side, n_px_side, 1).float(),
            r_test.float(),
            X_train=X.float(),
            at_iteration=None,
            **fit_model
        )

        # Ensure f_pred is a tensor on the correct device
        if not isinstance(f_pred, torch.Tensor):
            f_pred = torch.tensor(f_pred, device=device)
        else:
            f_pred = f_pred.to(device)

        # Compute metrics consistently with GPyTorch modes
        r_test_mean = r_test.mean(dim=0)
        test_corr = compute_pearson_correlation(r_test_mean.float(), f_pred.float())
        explained_var, reliability = compute_explained_variance(r_test.float(), f_pred.float())

        # Train correlation not available for vargp_old (would need extra prediction pass)
        train_corr = float('nan')

        final_loss = fit_model.get('loss', 0.0)
        pred_std = f_pred.std().item()
        pred_mean = f_pred.mean().item()
        pred_min = f_pred.min().item()
        pred_max = f_pred.max().item()

        # Create predictions dict to match GPyTorch branch structure
        predictions = {'f_pred': f_pred}
        losses = [final_loss]

    # =========================================================================
    # GPYTORCH MODES: adam, efm, vargp_style
    # =========================================================================
    else:
        # Create kernel with RF structure
        base_kernel = ArcCosineKernel(
            sigma_0=1.0,
            n_px_side=n_px_side,
            eps_0x=0.0,
            eps_0y=0.0,
            beta=args.beta,
            rho=args.rho,
            use_mask=args.use_mask,
            gradient_mode=args.gradient_mode
        )
        # Use base_kernel directly with internal Amp parameter
        # Amp is multiplied into C (non-linear effect through sqrt/arccos)
        # This is different from ScaleKernel which scales output linearly
        kernel = base_kernel
        kernel.Amp = 1e-4  # Match legacy varGP initialization

        # Create model and likelihood
        # vargp_style uses varGP's init values for fair comparison
        if args.mode == 'vargp_style':
            A_init, lambda0_init = 0.01, 1.0  # Match varGP defaults
        else:
            A_init, lambda0_init = 1.0, 0.0   # GPyTorch defaults

        model = VariationalGPModel(inducing_points, kernel, jitter=1e-4, whitening=not args.unwhitened)
        likelihood = PoissonLikelihood(A_init=A_init, lambda0_init=lambda0_init)

        model = model.double().to(device)
        likelihood = likelihood.double().to(device)

        print(f"\nInitial parameters:")
        print(f"  A_init: {A_init}, lambda0_init: {lambda0_init}")
        print(f"  A: {likelihood.A.item():.4f}")
        print(f"  lambda0: {likelihood.lambda0.item():.4f}")
        print(f"  Amp: {kernel.Amp.item():.6f}")

        # Train with selected mode
        # Note: GP_utils import disables gradients globally (utils.py line 2).
        # Must use torch.enable_grad() to re-enable for training.
        print(f"\nTraining with mode='{args.mode}':")
        print_every = max(1, args.n_iterations // 5)
        start_time = time.time()

        with torch.enable_grad():
            if args.mode == 'adam':
                print(f"  n_iterations={args.n_iterations}, lr={args.lr}")
                losses = train_adam(
                    model, likelihood, X_train, r_train,
                    n_iterations=args.n_iterations,
                    lr=args.lr,
                    print_every=print_every,
                    device=device
                )
            elif args.mode == 'efm':
                print(f"  n_iterations={args.n_iterations}, n_fstep={args.n_fstep}, n_mstep={args.n_mstep}, lr={args.lr}")
                losses = train_efm(
                    model, likelihood, X_train, r_train,
                    n_iterations=args.n_iterations,
                    n_fstep=args.n_fstep,
                    n_mstep=args.n_mstep,
                    lr=args.lr,
                    print_every=print_every,
                    device=device
                )
            else:  # vargp_style
                # Uses varGP defaults: lr_f=0.1, lr_m=0.1 (from utils.py)
                print(f"  n_iterations={args.n_iterations}, n_estep={args.n_estep}, n_fstep={args.n_fstep}, n_mstep={args.n_mstep}")
                print(f"  lr_f=0.1, lr_m=0.1 (varGP defaults)")
                print(f"  kernel_cache: {'enabled' if args.use_cache else 'DISABLED (fallback path)'}")
                # Whitening: OFF if unwhitened strategy, otherwise depends on --no-whitening flag
                whitening_status = 'OFF (unwhitened strategy)' if args.unwhitened else ('DISABLED' if args.no_whitening else 'enabled')
                print(f"  whitening: {whitening_status}")
                result = train_varGP_style(
                    model, likelihood, X_train, r_train,
                    n_iterations=args.n_iterations,
                    n_estep=args.n_estep,
                    n_fstep=args.n_fstep,
                    n_mstep=args.n_mstep,
                    lr_f=0.1,  # varGP default
                    lr_m=0.1,  # varGP default
                    print_every=print_every,
                    device=device,
                    use_cache=args.use_cache,  # Kernel caching for E-step performance
                    # If unwhitened strategy, let auto-detect handle it (will be False)
                    # Otherwise use --no-whitening flag for whitening conversions
                    use_whitening=None if args.unwhitened else (not args.no_whitening),
                )
                losses = result['losses']
                time_estep_total = result['time_estep_total']
                time_mstep_total = result['time_mstep_total']

        train_time = time.time() - start_time
        print(f"\nTraining time: {train_time:.1f}s")

        # Print E-step/M-step timing breakdown for vargp_style
        if args.mode == 'vargp_style':
            print(f"  E-step (+ F-step): {time_estep_total:.1f}s")
            print(f"  M-step:            {time_mstep_total:.1f}s")

        print(f"\nFinal parameters:")
        print(f"  A: {likelihood.A.item():.4f}")
        print(f"  lambda0: {likelihood.lambda0.item():.4f}")

        # Evaluate on test data
        print("\nEvaluating on test data...")
        predictions = predict(model, likelihood, X_test, device=device)
        f_pred = predictions['f_pred']

        r_test_mean = r_test.mean(dim=0)
        test_corr = compute_pearson_correlation(r_test_mean, f_pred)
        explained_var, reliability = compute_explained_variance(r_test, f_pred)

        # Also check train correlation
        train_preds = predict(model, likelihood, X_train, device=device)
        train_corr = compute_pearson_correlation(r_train, train_preds['f_pred'])

        # Check prediction statistics
        pred_mean = f_pred.mean().item()
        pred_std = f_pred.std().item()
        pred_min = f_pred.min().item()
        pred_max = f_pred.max().item()

    print(f"\n" + "="*50)
    print(f"RESULTS: mode={args.mode}, M={args.ntilde}")
    print(f"="*50)
    print(f"  Training time:   {train_time:.1f}s")
    print(f"  Train Pearson r: {train_corr:.4f}")
    print(f"  Test Pearson r:  {test_corr:.4f}")
    print(f"  Reliability:     {reliability:.4f}")
    print(f"  Explained var:   {explained_var:.4f}")
    print(f"  Final loss:      {losses[-1]:.2f}")
    print(f"  Prediction stats: mean={pred_mean:.3f}, std={pred_std:.3f}, range=[{pred_min:.3f}, {pred_max:.3f}]")

    # Print kernel call stats if analytical gradients were used
    if args.gradient_mode != 'autograd':
        # Check which implementation was used
        if args.gradient_mode == 'vjp':
            from analytical_gradients_vjp import ArcCosineVJPGradients as GradImpl
        else:
            from analytical_gradients import ArcCosineJacobianGradients as GradImpl
        if hasattr(GradImpl, '_call_count'):
            cc = GradImpl._call_count
            total = cc['grad'] + cc['no_grad']
            print(f"  Kernel calls:     {total} total ({cc['grad']} with grads, {cc['no_grad']} without)")
            # Reset for next run
            GradImpl._call_count = {'grad': 0, 'no_grad': 0}

    # Check for collapsed predictions
    if pred_std < 0.1:
        print(f"  WARNING: Predictions appear collapsed (std={pred_std:.4f})")

    # Plot handling
    save_path = None
    if args.save_plot == 'auto':
        # Auto-generate path: imgs/{mode}_M{ntilde}.png
        imgs_dir = Path(__file__).parent / 'imgs'
        imgs_dir.mkdir(exist_ok=True)
        save_path = imgs_dir / f"{args.mode}_M{args.ntilde}.png"
    elif args.save_plot and args.save_plot.lower() != 'none':
        save_path = Path(args.save_plot)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.plot or save_path:
        plot_fit(r_test_mean, predictions['f_pred'], args.cell, args.ntilde,
                 test_corr, explained_var, reliability,
                 output_path=save_path)

    # Extract final hyperparameters based on mode
    if args.mode == 'vargp_old':
        # Extract from fit_model
        theta = fit_model['hyperparams_tuple'][0]
        f_p = fit_model['f_params']
        # Convert from stored representation
        raw_beta = theta['-2log2beta'].item() if hasattr(theta['-2log2beta'], 'item') else float(theta['-2log2beta'])
        raw_rho = theta['-log2rho2'].item() if hasattr(theta['-log2rho2'], 'item') else float(theta['-log2rho2'])
        final_beta = np.exp(-raw_beta / 2) / 2
        final_rho = np.sqrt(np.exp(-raw_rho) / 2)
        logA_val = f_p['logA'].item() if hasattr(f_p['logA'], 'item') else float(f_p['logA'])
        final_A = np.exp(logA_val)
        final_lambda0 = f_p['lambda0'].item() if hasattr(f_p['lambda0'], 'item') else float(f_p['lambda0'])
        final_sigma_0 = theta['sigma_0'].item() if hasattr(theta['sigma_0'], 'item') else float(theta['sigma_0'])
        final_Amp = theta['Amp'].item() if hasattr(theta['Amp'], 'item') else float(theta['Amp'])
        final_eps_0x = theta['eps_0x'].item() if hasattr(theta['eps_0x'], 'item') else float(theta['eps_0x'])
        final_eps_0y = theta['eps_0y'].item() if hasattr(theta['eps_0y'], 'item') else float(theta['eps_0y'])
        # vargp_old doesn't expose E-step/M-step timing in returned dict
        time_estep = None
        time_mstep = None
    else:
        # GPyTorch modes: extract from model/likelihood
        # kernel is now directly ArcCosineKernel (not wrapped in ScaleKernel)
        final_A = likelihood.A.item()
        final_lambda0 = likelihood.lambda0.item()
        final_sigma_0 = kernel.sigma_0.item()
        final_Amp = kernel.Amp.item()
        final_beta = kernel.beta.item()
        final_rho = kernel.rho.item()
        final_eps_0x = kernel.eps_0x.item()
        final_eps_0y = kernel.eps_0y.item()
        # Timing breakdown only available for vargp_style
        if args.mode == 'vargp_style':
            time_estep = time_estep_total
            time_mstep = time_mstep_total
        else:
            time_estep = None
            time_mstep = None

    # Build result dict
    result = {
        'mode': args.mode,
        'M': args.ntilde,
        'train_time': train_time,
        'train_r': float(train_corr) if not np.isnan(train_corr) else None,
        'test_r': float(test_corr) if not np.isnan(test_corr) else None,
        'explained_var': float(explained_var) if not np.isnan(explained_var) else None,
        'reliability': float(reliability) if not np.isnan(reliability) else None,
        'final_loss': float(losses[-1]) if not np.isnan(losses[-1]) else None,
        'pred_std': pred_std,
        'time_estep_s': time_estep,
        'time_mstep_s': time_mstep,
        'final_A': final_A,
        'final_lambda0': final_lambda0,
        'final_Amp': final_Amp,
        'final_beta': final_beta,
        'final_rho': final_rho,
        'final_eps_0x': final_eps_0x,
        'final_eps_0y': final_eps_0y,
        'final_sigma_0': final_sigma_0,
    }

    # Write JSON if requested
    if args.json_append:
        json_record = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'commit': get_git_commit(),
            'seed': args.seed,
            'mode': args.mode,
            'M': args.ntilde,
            'ntrain': n_train,
            'niter': args.n_iterations,
            'nestep': args.n_estep,
            'nmstep': args.n_mstep,
            'nfstep': args.n_fstep,
            'cell_id': args.cell,
            'gradient_mode': args.gradient_mode if args.mode != 'vargp_old' else None,
            'test_r': result['test_r'],
            'explained_var': result['explained_var'],
            'final_loss': result['final_loss'],
            'time_total_s': round(train_time, 2),
            'time_estep_s': round(time_estep, 2) if time_estep is not None else None,
            'time_mstep_s': round(time_mstep, 2) if time_mstep is not None else None,
            'final_A': round(final_A, 6),
            'final_lambda0': round(final_lambda0, 6),
            'final_Amp': round(final_Amp, 6),
            'final_beta': round(final_beta, 6),
            'final_rho': round(final_rho, 6),
            'final_eps_0x': round(final_eps_0x, 6),
            'final_eps_0y': round(final_eps_0y, 6),
            'final_sigma_0': round(final_sigma_0, 6),
        }
        # Round floats for cleaner output
        for key in ['test_r', 'explained_var', 'final_loss']:
            if json_record[key] is not None:
                json_record[key] = round(json_record[key], 4)

        json_path = Path(args.json_append)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'a') as f:
            f.write(json.dumps(json_record) + '\n')
        print(f"\nJSON result appended to: {json_path}")

    return result


if __name__ == '__main__':
    main()
