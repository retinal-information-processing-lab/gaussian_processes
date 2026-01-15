#!/usr/bin/env python3
"""
GPyTorch Porting Test - Fit Single Cell with Arc-Cosine Kernel

This script tests the GPyTorch implementation on real PNAS data.
Loads images and neural responses, trains a variational GP with
arc-cosine kernel, and evaluates on test data.

Stage 1 (default): Uses C=I (identity covariance)
Stage 2 (--use-rf): Uses structured C with RF parameters (beta, rho, eps_0)

Usage:
    python test_fit.py [--cell CELL_ID] [--ntilde N_INDUCING] [--iterations N]
    python test_fit.py --use-rf [--beta 0.1] [--rho 0.1]  # Stage 2 with RF
"""

import sys
import argparse
import numpy as np
from pathlib import Path

import torch
import gpytorch
import matplotlib.pyplot as plt

# Import our GPyTorch components
from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from model import VariationalGPModel
from train import (train_model, predict, compute_r_squared,
                   compute_pearson_correlation, compute_explained_variance)


def load_pnas_data(data_path, dtype=torch.float64):
    """Load PNAS dataset.

    Args:
        data_path: Path to PNAS_paper_sorted_data.npz
        dtype: Torch dtype for tensors (default: float64 for numerical stability)

    Returns:
        dict with tensors: X_train, X_val, X_test, R_train, R_val, R_test
    """
    data = np.load(data_path)

    return {
        'X_train': torch.tensor(data['images_train'], dtype=dtype),
        'X_val': torch.tensor(data['images_val'], dtype=dtype),
        'X_test': torch.tensor(data['images_test'], dtype=dtype),
        'R_train': torch.tensor(data['responses_train'], dtype=dtype),
        'R_val': torch.tensor(data['responses_val'], dtype=dtype),
        'R_test': torch.tensor(data['responses_test'], dtype=dtype),
    }


def preprocess_data(data, cellid):
    """Preprocess data for single-cell fitting.

    Args:
        data: dict from load_pnas_data()
        cellid: Index of neuron to fit

    Returns:
        dict with preprocessed tensors
    """
    # Combine train and val
    X = torch.cat([data['X_train'], data['X_val']], dim=0)
    R = torch.cat([data['R_train'], data['R_val']], dim=0)

    # Flatten images: (n, 108, 108, 1) -> (n, 11664)
    X = X.reshape(X.shape[0], -1)
    X_test = data['X_test'].reshape(data['X_test'].shape[0], -1)

    # Select single neuron
    r = R[:, cellid]
    r_test = data['R_test'][:, :, cellid]  # (30 repeats, 30 images)

    return {
        'X': X,
        'r': r,
        'X_test': X_test,
        'r_test': r_test,
    }


def plot_fit(r_test_mean, f_pred, cellid, r2, corr, reliability, explained_var, output_path=None):
    """Plot actual vs predicted firing rates.

    Args:
        r_test_mean: Mean actual firing rates, shape (n_images,)
        f_pred: Predicted firing rates, shape (n_images,)
        cellid: Cell ID for title
        r2: R² value
        corr: Pearson correlation
        reliability: Cell reliability
        explained_var: Explained variance value
        output_path: If provided, save figure to this path instead of showing
    """
    r_test_mean = r_test_mean.cpu().numpy()
    f_pred = f_pred.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))

    n_images = len(r_test_mean)
    x = np.arange(n_images)

    ax.plot(x, r_test_mean, 'k-', linewidth=1.5, label='Actual (mean of 30 reps)')
    ax.plot(x, f_pred, 'r-', linewidth=1.5, label='Predicted')

    ax.set_xlabel('Test image index')
    ax.set_ylabel('Firing rate (spikes)')
    ax.set_title(f'Cell {cellid}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add metrics text box
    metrics_text = (f'R² = {r2:.3f}\n'
                    f'Pearson r = {corr:.3f}\n'
                    f'Reliability = {reliability:.3f}\n'
                    f'Expl. var = {explained_var:.3f}')
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description='Test GPyTorch fit on PNAS data')
    parser.add_argument('--cell', type=int, default=8, help='Cell ID to fit (default: 8)')
    parser.add_argument('--ntilde', type=int, default=200, help='Number of inducing points (default: 200)')
    parser.add_argument('--n-train', type=int, default=500,
                        help='Number of training samples (default: 500, use 0 for all)')
    parser.add_argument('--iterations', type=int, default=200, help='Training iterations (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--device', type=str, default='auto', help='Device: cuda, cpu, or auto')

    # Stage 2: RF parameters
    parser.add_argument('--use-rf', action='store_true', default=True,
                        help='Use structured C matrix with RF parameters (default: True)')
    parser.add_argument('--no-rf', action='store_false', dest='use_rf',
                        help='Disable RF structure (use C=I, Stage 1)')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='RF size parameter (default: 0.1, smaller = more localized)')
    parser.add_argument('--rho', type=float, default=0.1,
                        help='Smoothness parameter (default: 0.1)')
    parser.add_argument('--eps-0x', type=float, default=0.0,
                        help='RF center x-coordinate on [-1,1] (default: 0.0 = center)')
    parser.add_argument('--eps-0y', type=float, default=0.0,
                        help='RF center y-coordinate on [-1,1] (default: 0.0 = center)')
    parser.add_argument('--A-init', type=float, default=1.0,
                        help='Initial value for A parameter (default: 1.0, reference uses 0.01)')
    parser.add_argument('--lambda0-init', type=float, default=0.0,
                        help='Initial value for lambda0 parameter (default: 0.0)')
    parser.add_argument('--use-mask', action='store_true', default=True,
                        help='Use pixel masking to reduce C matrix size (default: True)')
    parser.add_argument('--no-mask', action='store_false', dest='use_mask',
                        help='Disable pixel masking (use full C matrix)')
    parser.add_argument('--plot', action='store_true',
                        help='Show plot of actual vs predicted firing rates')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Save plot to this path (e.g., fit_cell8.png)')
    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Load data
    data_path = Path(__file__).parent.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    print(f"Loading data from: {data_path}")
    data = load_pnas_data(data_path)

    # Preprocess
    print(f"Fitting cell: {args.cell}")
    processed = preprocess_data(data, args.cell)

    X = processed['X'].to(device)
    r = processed['r'].to(device)
    X_test = processed['X_test'].to(device)
    r_test = processed['r_test'].to(device)

    print(f"Training data shape: X={X.shape}, r={r.shape}")
    print(f"Test data shape: X_test={X_test.shape}, r_test={r_test.shape}")

    # Select inducing points (random subset)
    ntilde = min(args.ntilde, X.shape[0])
    indices_inducing = torch.randperm(X.shape[0], device=device)[:ntilde]
    inducing_points = X[indices_inducing].clone()
    print(f"Using {ntilde} inducing points")

    # Select training data (separate from inducing points)
    n_train = args.n_train if args.n_train > 0 else X.shape[0]
    n_train = min(n_train, X.shape[0])
    indices_train = torch.randperm(X.shape[0], device=device)[:n_train]
    X_train = X[indices_train]
    r_train = r[indices_train]
    print(f"Training on {X_train.shape[0]} samples")

    # Create kernel and model - inducing_points already float64 on device
    print("\nCreating model...")
    # Wrap arc-cosine kernel with ScaleKernel to add amplitude parameter

    if args.use_rf:
        # Stage 2: Use structured C matrix with RF parameters
        print(f"Using Stage 2: Structured C matrix with RF parameters (mask={args.use_mask})")
        n_px_side = 108  # PNAS image dimension
        base_kernel = ArcCosineKernel(
            sigma_0=1.0,
            n_px_side=n_px_side,
            eps_0x=args.eps_0x,
            eps_0y=args.eps_0y,
            beta=args.beta,
            rho=args.rho,
            use_mask=args.use_mask
        )
        # Kernel values are still large with RF structure, need to scale down
        # to prevent exp() overflow in likelihood
        kernel = gpytorch.kernels.ScaleKernel(base_kernel)
        kernel.outputscale = 1e-4  # Same as Stage 1 to prevent overflow
    else:
        # Stage 1: Use C=I (identity)
        print("Using Stage 1: C=I (identity covariance)")
        base_kernel = ArcCosineKernel(sigma_0=1.0)
        kernel = gpytorch.kernels.ScaleKernel(base_kernel)
        # The raw kernel has values ~10000, so we scale down to get variance ~1
        kernel.outputscale = 1e-4  # Scale factor to bring kernel values to reasonable range

    model = VariationalGPModel(inducing_points, kernel, jitter=1e-4)
    likelihood = PoissonLikelihood(A_init=args.A_init, lambda0_init=args.lambda0_init)

    # Model inherits dtype from inducing_points, just need to move likelihood
    model = model.double()  # Ensure all parameters are float64
    likelihood = likelihood.double().to(device)

    # Print initial parameters
    print(f"\nInitial parameters:")
    print(f"  outputscale: {kernel.outputscale.item():.6f}")
    print(f"  sigma_0: {base_kernel.sigma_0.item():.4f}")
    if args.use_rf:
        print(f"  beta: {args.beta:.4f}")
        print(f"  rho: {args.rho:.4f}")
        print(f"  eps_0x: {base_kernel.eps_0x.item():.4f}")
        print(f"  eps_0y: {base_kernel.eps_0y.item():.4f}")
    print(f"  A: {likelihood.A.item():.4f}")
    print(f"  lambda0: {likelihood.lambda0.item():.4f}")

    # Train
    print(f"\nTraining for {args.iterations} iterations (lr={args.lr})...")
    losses = train_model(model, likelihood, X_train, r_train,
                         n_iterations=args.iterations, lr=args.lr,
                         print_every=max(1, args.iterations // 5))

    # Print final parameters
    print(f"\nFinal parameters:")
    print(f"  outputscale: {kernel.outputscale.item():.6f}")
    print(f"  sigma_0: {base_kernel.sigma_0.item():.4f}")
    if args.use_rf:
        # Get actual beta and rho from log-space parameters
        # raw = -2*log(2*beta) → beta = exp(-raw/2) / 2
        # raw = -log(2*rho²) → rho = sqrt(exp(-raw) / 2)
        raw_beta = base_kernel.raw_m2log2beta.item()
        raw_rho = base_kernel.raw_mlog2rho2.item()
        beta_final = np.exp(-raw_beta / 2) / 2
        rho_final = np.sqrt(np.exp(-raw_rho) / 2)
        print(f"  eps_0x: {base_kernel.eps_0x.item():.4f}")
        print(f"  eps_0y: {base_kernel.eps_0y.item():.4f}")
        print(f"  beta: {beta_final:.4f}")
        print(f"  rho: {rho_final:.4f}")
    print(f"  A: {likelihood.A.item():.4f}")
    print(f"  lambda0: {likelihood.lambda0.item():.4f}")

    # Evaluate on test data
    print("\nEvaluating on test data...")
    predictions = predict(model, likelihood, X_test)

    # r_test has shape (30 repeats, 30 images) - average over repeats
    r_test_mean = r_test.mean(dim=0)

    r2 = compute_r_squared(r_test_mean, predictions['f_pred'])
    corr = compute_pearson_correlation(r_test_mean, predictions['f_pred'])
    explained_var, reliability = compute_explained_variance(r_test, predictions['f_pred'])

    print(f"\nResults:")
    print(f"  Test R²: {r2:.4f}")
    print(f"  Test Pearson r: {corr:.4f}")
    print(f"  Reliability: {reliability:.4f}")
    print(f"  Explained variance: {explained_var:.4f}")

    # Print some predictions vs actual
    print(f"\nSample predictions (first 5 test images):")
    print(f"  Actual (avg): {r_test_mean[:5].tolist()}")
    print(f"  Predicted:    {predictions['f_pred'][:5].tolist()}")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Cell: {args.cell}")
    print(f"Mode: {'Stage 2 (RF structure)' if args.use_rf else 'Stage 1 (C=I)'}")
    if args.use_rf:
        print(f"RF params (initial): beta={args.beta}, rho={args.rho}, eps_0=({args.eps_0x}, {args.eps_0y})")
        print(f"RF params (final):   beta={beta_final:.4f}, rho={rho_final:.4f}, eps_0=({base_kernel.eps_0x.item():.4f}, {base_kernel.eps_0y.item():.4f})")
    print(f"Inducing points: {ntilde}")
    print(f"Training samples: {n_train}")
    print(f"Training iterations: {args.iterations}")
    print(f"Final loss: {losses[-1]:.2f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Test Pearson r: {corr:.4f}")
    print(f"Reliability: {reliability:.4f}")
    print(f"Explained variance: {explained_var:.4f}")

    # Plot if requested
    if args.plot or args.save_plot:
        plot_fit(r_test_mean, predictions['f_pred'], args.cell,
                 r2, corr, reliability, explained_var,
                 output_path=args.save_plot)

    return r2, corr, reliability, explained_var


if __name__ == '__main__':
    main()
