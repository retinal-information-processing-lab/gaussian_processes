#!/usr/bin/env python3
"""
Test training loops on real PNAS neural data.

Created by Claude to compare training modes:
  - adam: Pure Adam optimization (no E-step)
  - efm: E-F-M loop (1 E-step, n F-steps, n M-steps per iteration)

Usage:
    python test_estep_pnas.py --ntilde 50 --mode adam --save-plot imgs/adam_M50.png
    python test_estep_pnas.py --ntilde 50 --mode efm --save-plot imgs/efm_M50.png
    python test_estep_pnas.py  # Uses defaults: M=50, mode=efm, saves to imgs/
"""

import sys
import time
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
from estep import train_efm
from train import train_adam, predict, compute_pearson_correlation, compute_explained_variance


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
                        choices=['adam', 'efm'],
                        help='Training mode: adam (no E-step), efm (E-F-M loop)')

    # RF parameters - use defaults that work
    parser.add_argument('--beta', type=float, default=0.1, help='RF size (default: 0.1)')
    parser.add_argument('--rho', type=float, default=0.1, help='Smoothness (default: 0.1)')
    parser.add_argument('--use-mask', action='store_true', default=True, help='Use pixel masking')
    parser.add_argument('--no-mask', action='store_false', dest='use_mask')

    # Plotting options
    parser.add_argument('--plot', action='store_true',
                        help='Show plot of actual vs predicted firing rates')
    parser.add_argument('--save-plot', type=str, default='auto',
                        help='Save plot path. "auto" saves to imgs/{mode}_M{ntilde}.png, "none" to disable')

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"M={args.ntilde} inducing points")

    # Set seed
    torch.manual_seed(42)

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

    # Create kernel with RF structure
    n_px_side = 108
    base_kernel = ArcCosineKernel(
        sigma_0=1.0,
        n_px_side=n_px_side,
        eps_0x=0.0,
        eps_0y=0.0,
        beta=args.beta,
        rho=args.rho,
        use_mask=args.use_mask
    )
    kernel = gpytorch.kernels.ScaleKernel(base_kernel)
    kernel.outputscale = 1e-4  # Prevent overflow

    # Create model and likelihood
    model = VariationalGPModel(inducing_points, kernel, jitter=1e-4)
    likelihood = PoissonLikelihood(A_init=1.0, lambda0_init=0.0)

    model = model.double().to(device)
    likelihood = likelihood.double().to(device)

    print(f"\nInitial parameters:")
    print(f"  A: {likelihood.A.item():.4f}")
    print(f"  lambda0: {likelihood.lambda0.item():.4f}")
    print(f"  outputscale: {kernel.outputscale.item():.6f}")

    # Train with selected mode
    print(f"\nTraining with mode='{args.mode}':")
    print_every = max(1, args.n_iterations // 5)
    start_time = time.time()

    if args.mode == 'adam':
        print(f"  n_iterations={args.n_iterations}, lr={args.lr}")
        losses = train_adam(
            model, likelihood, X_train, r_train,
            n_iterations=args.n_iterations,
            lr=args.lr,
            print_every=print_every,
            device=device
        )
    else:  # efm
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

    train_time = time.time() - start_time
    print(f"\nTraining time: {train_time:.1f}s")

    print(f"\nFinal parameters:")
    print(f"  A: {likelihood.A.item():.4f}")
    print(f"  lambda0: {likelihood.lambda0.item():.4f}")

    # Evaluate on test data
    print("\nEvaluating on test data...")
    predictions = predict(model, likelihood, X_test, device=device)

    r_test_mean = r_test.mean(dim=0)
    test_corr = compute_pearson_correlation(r_test_mean, predictions['f_pred'])
    explained_var, reliability = compute_explained_variance(r_test, predictions['f_pred'])

    # Also check train correlation
    train_preds = predict(model, likelihood, X_train, device=device)
    train_corr = compute_pearson_correlation(r_train, train_preds['f_pred'])

    # Check prediction statistics
    pred_mean = predictions['f_pred'].mean().item()
    pred_std = predictions['f_pred'].std().item()
    pred_min = predictions['f_pred'].min().item()
    pred_max = predictions['f_pred'].max().item()

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

    return {
        'mode': args.mode,
        'M': args.ntilde,
        'train_time': train_time,
        'train_r': train_corr,
        'test_r': test_corr,
        'explained_var': explained_var,
        'reliability': reliability,
        'final_loss': losses[-1],
        'pred_std': pred_std,
    }


if __name__ == '__main__':
    main()
