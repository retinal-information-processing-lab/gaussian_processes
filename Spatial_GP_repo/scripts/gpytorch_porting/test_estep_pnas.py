#!/usr/bin/env python3
"""
Test E-step implementation on real PNAS neural data.

Created by Claude to validate E-step behavior with different numbers of inducing points (M).
This script tests the hypothesis that E-step fails for M >= 50.

Usage:
    python test_estep_pnas.py --ntilde 25
    python test_estep_pnas.py --ntilde 50
    python test_estep_pnas.py --ntilde 100
"""

import sys
import argparse
import numpy as np
from pathlib import Path

import torch
import gpytorch

# Import our GPyTorch components
from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from model import VariationalGPModel
from estep import train_with_estep
from train import predict, compute_pearson_correlation, compute_explained_variance


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


def main():
    parser = argparse.ArgumentParser(description='Test E-step on PNAS data')
    parser.add_argument('--cell', type=int, default=8, help='Cell ID (default: 8)')
    parser.add_argument('--ntilde', type=int, default=25, help='Number of inducing points M (default: 25)')
    parser.add_argument('--n-train', type=int, default=500, help='Number of training samples (default: 500)')
    parser.add_argument('--n-iterations', type=int, default=50, help='Number of EM iterations (default: 50)')
    parser.add_argument('--n-estep', type=int, default=3, help='E-steps per iteration (default: 3)')
    parser.add_argument('--n-mstep', type=int, default=10, help='M-steps per iteration (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for M-step (default: 0.01)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (default: cpu)')

    # RF parameters - use defaults that work
    parser.add_argument('--beta', type=float, default=0.1, help='RF size (default: 0.1)')
    parser.add_argument('--rho', type=float, default=0.1, help='Smoothness (default: 0.1)')
    parser.add_argument('--use-mask', action='store_true', default=True, help='Use pixel masking')
    parser.add_argument('--no-mask', action='store_false', dest='use_mask')

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Testing E-step with M={args.ntilde} inducing points")

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

    # Train with E-step
    print(f"\nTraining with E-step:")
    print(f"  n_iterations={args.n_iterations}, n_estep={args.n_estep}, n_mstep={args.n_mstep}, lr={args.lr}")

    losses = train_with_estep(
        model, likelihood, X_train, r_train,
        n_iterations=args.n_iterations,
        n_estep=args.n_estep,
        n_mstep=args.n_mstep,
        lr=args.lr,
        print_every=max(1, args.n_iterations // 5),
        device=device
    )

    print(f"\nFinal parameters:")
    print(f"  A: {likelihood.A.item():.4f}")
    print(f"  lambda0: {likelihood.lambda0.item():.4f}")

    # Evaluate on test data
    print("\nEvaluating on test data...")
    predictions = predict(model, likelihood, X_test, device=device)

    r_test_mean = r_test.mean(dim=0)
    test_corr = compute_pearson_correlation(r_test_mean, predictions['f_pred'])

    # Also check train correlation
    train_preds = predict(model, likelihood, X_train, device=device)
    train_corr = compute_pearson_correlation(r_train, train_preds['f_pred'])

    # Check prediction statistics
    pred_mean = predictions['f_pred'].mean().item()
    pred_std = predictions['f_pred'].std().item()
    pred_min = predictions['f_pred'].min().item()
    pred_max = predictions['f_pred'].max().item()

    print(f"\n" + "="*50)
    print(f"RESULTS: M={args.ntilde} inducing points")
    print(f"="*50)
    print(f"  Train Pearson r: {train_corr:.4f}")
    print(f"  Test Pearson r:  {test_corr:.4f}")
    print(f"  Final loss:      {losses[-1]:.2f}")
    print(f"  Prediction stats: mean={pred_mean:.3f}, std={pred_std:.3f}, range=[{pred_min:.3f}, {pred_max:.3f}]")

    # Check for collapsed predictions
    if pred_std < 0.1:
        print(f"  WARNING: Predictions appear collapsed (std={pred_std:.4f})")

    return {
        'M': args.ntilde,
        'train_r': train_corr,
        'test_r': test_corr,
        'final_loss': losses[-1],
        'pred_std': pred_std,
    }


if __name__ == '__main__':
    main()
