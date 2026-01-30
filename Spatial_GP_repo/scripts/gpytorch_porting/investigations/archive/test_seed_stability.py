#!/usr/bin/env python3
"""
test_seed_stability.py - Quick diagnostic comparing good vs problematic seeds

Created by Claude to test the effect of lambda0 log parameterization on training stability.

This script runs shortened training (30 iterations) on multiple seeds and reports:
- Whether training collapsed (firing rates became constant)
- Final A, lambda0, test correlation
- A trajectory summary

Usage:
    python investigations/test_seed_stability.py
    python investigations/test_seed_stability.py --seeds 123 456 42
    python investigations/test_seed_stability.py --ntilde 75
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject')
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/torchlambertw')

import numpy as np
import torch

from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from model import VariationalGPModel
from train import predict, compute_explained_variance
from tests.test_utils import set_reproducible_seed


def load_pnas_data(data_path, dtype=torch.float64):
    """Load PNAS dataset."""
    data = np.load(data_path)
    return {
        'X_train': torch.tensor(data['images_train'], dtype=dtype),
        'R_train': torch.tensor(data['responses_train'], dtype=dtype),
        'X_test': torch.tensor(data['images_test'], dtype=dtype),
        'R_test': torch.tensor(data['responses_test'], dtype=dtype),
    }


def train_with_tracking(model, likelihood, train_x, train_y,
                        n_iterations=30, n_estep=10, n_fstep=10, n_mstep=10,
                        device='cuda'):
    """
    Train and track key metrics per iteration.

    Returns dict with:
    - collapsed: bool - whether firing rates became constant
    - collapse_iter: int or None - iteration when collapse occurred
    - A_trajectory: list of A values
    - lambda0_trajectory: list of lambda0 values
    - rate_std_trajectory: list of firing rate std
    - final_loss: float
    """
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

    A_trajectory = []
    lambda0_trajectory = []
    rate_std_trajectory = []
    loss_trajectory = []
    collapse_iter = None

    kernel_cache = None

    for iteration in range(n_iterations):
        # E-STEP
        set_kernel_requires_grad(model, False)
        model.eval()

        with torch.no_grad():
            kernel_cache = compute_kernel_cache(model, train_x)

        with torch.no_grad():
            lambda_m, lambda_var = e_step_loop(
                model, likelihood, train_x, train_y, n_estep, verbose=False,
                kernel_cache=kernel_cache,
                use_whitening=use_whitening
            )

        # F-STEP
        model.train()
        with torch.enable_grad():
            f_step_lbfgs(model, likelihood, train_x, train_y,
                         lambda_m, lambda_var, n_fstep, 0.1, verbose=False)

        # M-STEP
        if n_mstep > 0 and iteration < n_iterations - 1:
            set_kernel_requires_grad(model, True)
            with torch.enable_grad():
                m_step(model, likelihood, train_x, train_y, n_mstep, 0.1, verbose=False)
            set_kernel_requires_grad(model, False)
            kernel_cache = None

        # Record metrics
        model.eval()
        with torch.no_grad():
            A = likelihood.A.item()
            lambda0 = likelihood.lambda0.item()

            output = model(train_x)
            lm = output.mean
            lv = output.variance
            rate = torch.exp(A * lm + 0.5 * A**2 * lv + lambda0)
            rate_std = rate.std().item()

            ell = likelihood.expected_log_prob(train_y, output)
            kl = model.variational_strategy.kl_divergence()
            loss = (-ell + kl).item()

        A_trajectory.append(A)
        lambda0_trajectory.append(lambda0)
        rate_std_trajectory.append(rate_std)
        loss_trajectory.append(loss)

        # Check for collapse (rate_std < 0.01 means nearly constant)
        if rate_std < 0.01 and collapse_iter is None:
            collapse_iter = iteration + 1

    collapsed = collapse_iter is not None

    return {
        'collapsed': collapsed,
        'collapse_iter': collapse_iter,
        'A_trajectory': A_trajectory,
        'lambda0_trajectory': lambda0_trajectory,
        'rate_std_trajectory': rate_std_trajectory,
        'loss_trajectory': loss_trajectory,
        'final_A': A_trajectory[-1],
        'final_lambda0': lambda0_trajectory[-1],
        'final_loss': loss_trajectory[-1],
    }


def run_seed_test(seed, ntilde, n_train, n_iterations, cell, device):
    """Run training for a single seed and return results."""
    set_reproducible_seed(seed)

    # Load data
    data_path = '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/notebooks/PNAS_paper_sorted_data.npz'
    data = load_pnas_data(data_path, dtype=torch.float64)

    X_train = data['X_train'][:n_train].reshape(n_train, -1).to(device)
    R_train = data['R_train'][:n_train, cell].to(device)
    X_test = data['X_test'].reshape(data['X_test'].shape[0], -1).to(device)
    R_test = data['R_test'][:, :, cell].to(device)

    # Select inducing points
    indices = torch.randperm(X_train.shape[0], device=device)[:ntilde]
    inducing_points = X_train[indices]

    # Initialize kernel
    kernel = ArcCosineKernel(
        n_px_side=108,
        beta_init=0.1,
        rho_init=0.1,
        eps_0_init=(0.0, 0.0),
        use_masking=True,
        gradient_mode='autograd'
    )
    kernel = kernel.double().to(device)

    # Initialize model (whitened mode)
    model = VariationalGPModel(
        inducing_points=inducing_points,
        kernel=kernel,
        whitening=True
    )
    model = model.double().to(device)

    # Initialize likelihood (varGP-style parameters)
    likelihood = PoissonLikelihood(A_init=0.01, lambda0_init=1.0)
    likelihood = likelihood.double().to(device)

    # Train with tracking
    results = train_with_tracking(
        model, likelihood, X_train, R_train,
        n_iterations=n_iterations,
        device=device
    )

    # Compute test correlation if not collapsed
    if not results['collapsed']:
        model.eval()
        predictions = predict(model, likelihood, X_test, device=device)
        explained_var, reliability = compute_explained_variance(R_test, predictions['f_pred'])
        results['test_explained_var'] = explained_var
    else:
        results['test_explained_var'] = float('nan')

    return results


def main():
    parser = argparse.ArgumentParser(description='Test seed stability')
    parser.add_argument('--seeds', type=int, nargs='+', default=[123, 456, 42],
                       help='Seeds to test (default: 123 456 42)')
    parser.add_argument('--ntilde', type=int, default=50,
                       help='Number of inducing points (default: 50)')
    parser.add_argument('--n-train', type=int, default=500,
                       help='Number of training samples (default: 500)')
    parser.add_argument('--n-iterations', type=int, default=30,
                       help='Number of EM iterations (default: 30)')
    parser.add_argument('--cell', type=int, default=8,
                       help='Cell ID (default: 8)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("SEED STABILITY TEST")
    print("="*80)
    print(f"M = {args.ntilde}, n_train = {args.n_train}, n_iterations = {args.n_iterations}")
    print(f"Seeds: {args.seeds}")
    print("="*80 + "\n")

    results_all = {}

    for seed in args.seeds:
        print(f"Running seed {seed}...")
        results = run_seed_test(
            seed=seed,
            ntilde=args.ntilde,
            n_train=args.n_train,
            n_iterations=args.n_iterations,
            cell=args.cell,
            device=args.device
        )
        results_all[seed] = results

        status = "COLLAPSED" if results['collapsed'] else "OK"
        if results['collapsed']:
            status += f" (iter {results['collapse_iter']})"

        print(f"  Seed {seed}: {status}")
        print(f"    Final A: {results['final_A']:.6f}")
        print(f"    Final lambda0: {results['final_lambda0']:.4f}")
        print(f"    Test explained var: {results['test_explained_var']:.4f}")
        print()

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Seed':>6} | {'Status':>12} | {'Final A':>10} | {'Final l0':>10} | {'Test EV':>10}")
    print("-"*60)

    for seed, res in results_all.items():
        status = f"COLLAPSE@{res['collapse_iter']}" if res['collapsed'] else "OK"
        ev_str = f"{res['test_explained_var']:.4f}" if not np.isnan(res['test_explained_var']) else "nan"
        print(f"{seed:>6} | {status:>12} | {res['final_A']:>10.6f} | {res['final_lambda0']:>10.4f} | {ev_str:>10}")

    print("="*80)

    # A trajectory for problematic seeds
    print("\nA TRAJECTORY FOR SEEDS THAT COLLAPSED:")
    for seed, res in results_all.items():
        if res['collapsed']:
            A_traj = res['A_trajectory']
            # Show key points: start, peak, end
            peak_idx = np.argmax(A_traj)
            print(f"\nSeed {seed}:")
            print(f"  Start (iter 1):     A = {A_traj[0]:.6f}")
            print(f"  Peak (iter {peak_idx+1}):      A = {A_traj[peak_idx]:.6f}")
            print(f"  End (iter {len(A_traj)}):      A = {A_traj[-1]:.6f}")
            print(f"  Collapse at iter:   {res['collapse_iter']}")

    return results_all


if __name__ == '__main__':
    main()
