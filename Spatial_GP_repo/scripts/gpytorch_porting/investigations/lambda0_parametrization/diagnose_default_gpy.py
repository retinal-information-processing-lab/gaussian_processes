#!/usr/bin/env python3
"""
Diagnostic script for default_gpy lambda0 optimization issue.
Created by Claude for investigating why lambda0 collapses in default_gpy mode.

Tracks A, lambda0, loss, and other key metrics at every iteration.
"""

import sys
import time
from pathlib import Path

# Add paths for imports
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject')
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting')

import torch
import numpy as np

from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from model import VariationalGPModel
from tests.test_utils import set_reproducible_seed
from fstep import lambda0_given_A
from utils_gpy import compute_rf_center_from_sta


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


def train_with_tracking(model, likelihood, train_x, train_y, lr, n_iterations, device,
                        use_analytical_lambda0=False, fix_likelihood=False, optimizer_type='adam'):
    """Train default_gpy style with full parameter tracking at each iteration.

    Args:
        use_analytical_lambda0: If True, remove lambda0 from optimizer and set it
                                analytically after each step using lambda0_given_A().
        fix_likelihood: If True, don't optimize A or lambda0 at all - only optimize
                        variational parameters (m, S) and kernel hyperparameters.
        optimizer_type: 'adam' or 'lbfgs'
    """

    model = model.to(device)
    likelihood = likelihood.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    model.train()
    likelihood.train()

    # Choose which parameters to optimize
    if fix_likelihood:
        # Don't optimize any likelihood parameters
        likelihood_params = []
    elif use_analytical_lambda0:
        # Exclude lambda0 from optimization - will be set analytically
        likelihood_params = [likelihood.raw_A]  # Only optimize A
    else:
        likelihood_params = list(likelihood.parameters())

    # Build optimizer param groups
    all_params = list(model.parameters()) + likelihood_params

    if optimizer_type == 'lbfgs':
        optimizer = torch.optim.LBFGS(all_params, lr=lr, max_iter=20, line_search_fn='strong_wolfe')
    else:
        param_groups = [{'params': model.parameters()}]
        if likelihood_params:
            param_groups.append({'params': likelihood_params})
        optimizer = torch.optim.Adam(param_groups, lr=lr)

    # Track history
    history = {
        'iteration': [],
        'loss': [],
        'ell': [],
        'kl': [],
        'A': [],
        'lambda0': [],
        'raw_A': [],  # logA
        'Amp': [],
        'pred_mean': [],
        'pred_std': [],
        'lambda_m_std': [],  # GP posterior mean std
        'lambda_var_mean': [],  # GP posterior variance mean
        'var_mean_norm': [],  # Variational mean norm (should change if learning)
    }

    # For LBFGS, we need a closure
    current_output = [None]
    current_loss = [None]
    current_ell = [None]
    current_kl = [None]

    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        ell = likelihood.expected_log_prob(train_y, output)
        kl = model.variational_strategy.kl_divergence()
        loss = -ell + kl
        loss.backward()
        current_output[0] = output
        current_loss[0] = loss
        current_ell[0] = ell
        current_kl[0] = kl
        return loss

    with torch.enable_grad():
        for i in range(n_iterations):
            if optimizer_type == 'lbfgs':
                optimizer.step(closure)
                output = current_output[0]
                loss = current_loss[0]
                ell = current_ell[0]
                kl = current_kl[0]
            else:
                optimizer.zero_grad()
                output = model(train_x)
                ell = likelihood.expected_log_prob(train_y, output)
                kl = model.variational_strategy.kl_divergence()
                loss = -ell + kl
                loss.backward()
                optimizer.step()

            # Compute predictions for monitoring
            with torch.no_grad():
                A = likelihood.A.squeeze()
                lambda0 = likelihood.lambda0.squeeze()
                mu = output.mean
                var = output.variance
                f_pred = torch.exp(A * mu + 0.5 * A**2 * var + lambda0)

            # Record after step
            history['iteration'].append(i)
            history['loss'].append(loss.item())
            history['ell'].append(ell.item())
            history['kl'].append(kl.item())
            history['A'].append(likelihood.A.item())
            history['lambda0'].append(likelihood.lambda0.item())
            history['raw_A'].append(likelihood.raw_A.item())
            history['Amp'].append(model.covar_module.Amp.item())
            history['pred_mean'].append(f_pred.mean().item())
            history['pred_std'].append(f_pred.std().item())
            history['lambda_m_std'].append(mu.std().item())
            history['lambda_var_mean'].append(var.mean().item())
            # Track variational mean norm (whitened m_w)
            var_mean = model.variational_strategy._variational_distribution.variational_mean
            history['var_mean_norm'].append(var_mean.norm().item())

            # Set lambda0 analytically after optimizer step
            if use_analytical_lambda0:
                with torch.no_grad():
                    A_current = likelihood.A.squeeze()
                    new_lambda0 = lambda0_given_A(A_current, train_y, mu, var)
                    likelihood.lambda0.copy_(new_lambda0.reshape(likelihood.lambda0.shape))

    return history


def main(cell=10, use_analytical_lambda0=False, A_init=0.01, lambda0_init=1.0,
         use_mask=True, fix_likelihood=False, optimizer_type='adam'):
    # Configuration
    n_train = 500
    ntilde = 50
    n_iterations = 100
    lr = 0.1
    seed = 123
    device = torch.device('cuda')
    dtype = torch.float32

    print(f"Diagnosing default_gpy on Cell {cell}")
    print(f"  ntrain={n_train}, M={ntilde}, iterations={n_iterations}, lr={lr}")
    print(f"  seed={seed}, dtype={dtype}, optimizer={optimizer_type}")
    print(f"  analytical_lambda0={use_analytical_lambda0}, A_init={A_init}, lambda0_init={lambda0_init}")
    print(f"  use_mask={use_mask}, fix_likelihood={fix_likelihood}")
    print()

    # Set seed
    set_reproducible_seed(seed, device=device)

    # Load data
    data_path = Path(__file__).parent.parent.parent.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = load_pnas_data(data_path, dtype=dtype)

    # Prepare data
    X = torch.cat([data['X_train'], data['X_val']], dim=0)
    R = torch.cat([data['R_train'], data['R_val']], dim=0)
    X = X.reshape(X.shape[0], -1).to(device)
    R = R.to(device)

    r = R[:, cell]

    # Training subset
    indices_train = torch.randperm(X.shape[0], device=device)[:n_train]
    X_train = X[indices_train]
    r_train = r[indices_train]

    # Inducing points
    inducing_points = X[indices_train[:ntilde]].clone()

    print(f"Data: X_train={X_train.shape}, r_train={r_train.shape}")
    print(f"Spike stats: mean={r_train.mean():.2f}, sum={r_train.sum():.0f}")
    print()

    # Compute RF center from STA
    n_px_side = 108
    eps_0x, eps_0y = compute_rf_center_from_sta(X_train, r_train, n_px_side, zscore=True)
    print(f"STA RF center: ({eps_0x:.4f}, {eps_0y:.4f})")

    # Create model
    kernel = ArcCosineKernel(
        sigma_0=1.0,
        Amp=1.0,
        n_px_side=n_px_side,
        eps_0x=eps_0x,
        eps_0y=eps_0y,
        beta=0.1,
        rho=0.1,
        use_mask=use_mask,
    )

    model = VariationalGPModel(inducing_points, kernel, jitter=1e-4, standard_variational_distribution=True)
    likelihood = PoissonLikelihood(A_init=A_init, lambda0_init=lambda0_init)

    model = model.float().to(device)
    likelihood = likelihood.float().to(device)

    print(f"Initial: A={likelihood.A.item():.4f}, lambda0={likelihood.lambda0.item():.4f}")

    # DEBUG: Check kernel structure
    print()
    print("Checking kernel structure...")
    model.eval()
    with torch.no_grad():
        # Get cross-covariance k(X_train, X_tilde)
        K_cross = model.covar_module(X_train, inducing_points).evaluate()
        print(f"  K_cross shape: {K_cross.shape}")
        print(f"  K_cross[0,0:5]: {K_cross[0, :5].tolist()}")
        print(f"  K_cross row std (should vary): {K_cross.std(dim=1).mean().item():.6f}")
        print(f"  K_cross col std (should vary): {K_cross.std(dim=0).mean().item():.6f}")

        # Check if all rows are the same
        row_diffs = (K_cross - K_cross[0:1, :]).abs().max(dim=1)[0]
        print(f"  Max diff from first row: {row_diffs.max().item():.8f}")
        if row_diffs.max().item() < 1e-6:
            print("  >>> KERNEL ISSUE: All rows of K_cross are identical!")

        # Check kernel params
        print(f"\n  Kernel params:")
        print(f"    sigma_0={kernel.sigma_0.item():.4f}")
        print(f"    Amp={kernel.Amp.item():.4f}")
        print(f"    beta={kernel.beta.item():.4f}")
        print(f"    rho={kernel.rho.item():.4f}")
        print(f"    eps_0x={kernel.eps_0x.item():.4f}, eps_0y={kernel.eps_0y.item():.4f}")
        print(f"    use_mask={kernel.use_mask}")

        # Check input data variance
        print(f"\n  Input stats:")
        print(f"    X_train std: {X_train.std().item():.4f}")
        print(f"    X_train[0] vs X_train[1] diff: {(X_train[0] - X_train[1]).abs().max().item():.4f}")
    print()

    # Train with tracking
    print("Training...")
    start = time.time()
    history = train_with_tracking(model, likelihood, X_train, r_train, lr, n_iterations, device,
                                   use_analytical_lambda0=use_analytical_lambda0,
                                   fix_likelihood=fix_likelihood,
                                   optimizer_type=optimizer_type)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")
    print()

    # Print trajectory
    print("=" * 115)
    print(f"{'Iter':>5} {'Loss':>10} {'ELL':>10} {'KL':>8} {'A':>8} {'lambda0':>10} {'lam_m_std':>12} {'m_w_norm':>10}")
    print("=" * 115)

    # Print every 10 iterations + first and last few
    for i in range(len(history['iteration'])):
        if i < 5 or i >= n_iterations - 5 or i % 10 == 0:
            print(f"{history['iteration'][i]:>5} "
                  f"{history['loss'][i]:>10.2f} "
                  f"{history['ell'][i]:>10.2f} "
                  f"{history['kl'][i]:>8.2f} "
                  f"{history['A'][i]:>8.4f} "
                  f"{history['lambda0'][i]:>10.4f} "
                  f"{history['lambda_m_std'][i]:>12.2e} "
                  f"{history['var_mean_norm'][i]:>10.4f}")

    print("=" * 80)
    print()

    # Summary
    print("Parameter trajectories:")
    print(f"  A:       {history['A'][0]:.4f} -> {history['A'][-1]:.4f}")
    print(f"  lambda0: {history['lambda0'][0]:.4f} -> {history['lambda0'][-1]:.4f}")
    print(f"  loss:    {history['loss'][0]:.2f} -> {history['loss'][-1]:.2f}")
    print(f"  pred_std: {history['pred_std'][0]:.4f} -> {history['pred_std'][-1]:.4f}")

    # Check what the analytical lambda0 should be
    with torch.no_grad():
        output = model(X_train)
        A = likelihood.A.squeeze()
        lambda_m = output.mean
        lambda_var = output.variance

        # Analytical lambda0 formula
        sumr = r_train.sum()
        expexpr = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var)
        analytical_lambda0 = torch.log(sumr) - torch.log(expexpr.sum())

        print()
        print(f"Analytical lambda0 (given current A): {analytical_lambda0.item():.4f}")
        print(f"Actual lambda0:                       {likelihood.lambda0.item():.4f}")
        print(f"Difference:                           {(likelihood.lambda0.item() - analytical_lambda0.item()):.4f}")

        # Check if posterior is effectively constant
        print()
        print("=" * 60)
        print("DIAGNOSTIC: Is posterior constant?")
        print("=" * 60)
        print(f"  lambda_m: mean={lambda_m.mean().item():.6f}, std={lambda_m.std().item():.8f}, min={lambda_m.min().item():.6f}, max={lambda_m.max().item():.6f}")
        print(f"  lambda_var: mean={lambda_var.mean().item():.4f}")

        # Compute firing rate predictions
        f_pred = torch.exp(A * lambda_m + 0.5 * A * A * lambda_var + likelihood.lambda0)
        print(f"  f_pred: mean={f_pred.mean().item():.4f}, std={f_pred.std().item():.4f}")

        # The key diagnostic: is A * lambda_m_std small?
        effective_signal = A.item() * lambda_m.std().item()
        print()
        print(f"  Effective signal (A * lambda_m_std): {effective_signal:.6f}")
        if effective_signal < 0.01:
            print("  >>> COLLAPSED: A * lambda_m_std < 0.01 means predictions are nearly constant!")

        # Compare with what vargp_direct achieves
        print()
        print("  Reference (vargp_direct on same cell):")
        print("    A=0.029, lambda0=-0.98, test_r=0.75")

    # Evaluate on test data
    print()
    print("=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    # X_test: (30 images, 108, 108, 1) -> (30, 11664)
    # R_test: (30 repeats, 30 images, 41 cells) -> average over repeats
    X_test = data['X_test'].reshape(data['X_test'].shape[0], -1).to(device)
    r_test = data['R_test'][:, :, cell].mean(dim=0).to(device)  # Average over 30 repeats

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        output_test = model(X_test)
        A = likelihood.A.squeeze()
        lambda0 = likelihood.lambda0.squeeze()
        lambda_m_test = output_test.mean
        lambda_var_test = output_test.variance
        f_pred_test = torch.exp(A * lambda_m_test + 0.5 * A**2 * lambda_var_test + lambda0)

        # Pearson correlation
        r_test_np = r_test.cpu().numpy()
        f_pred_np = f_pred_test.cpu().numpy()

        # Handle constant predictions, NaN, or inf
        if f_pred_np.std() < 1e-8 or np.isnan(f_pred_np).any() or np.isinf(f_pred_np).any():
            test_r = 0.0
            print(f"  Test Pearson r: {test_r:.4f} (predictions unstable)")
        else:
            test_r = np.corrcoef(r_test_np.flatten(), f_pred_np.flatten())[0, 1]
            print(f"  Test Pearson r: {test_r:.4f}")
        print(f"  Predictions: mean={f_pred_np.mean():.4f}, std={f_pred_np.std():.4f}")
        print(f"  Actual: mean={r_test_np.mean():.4f}, std={r_test_np.std():.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', type=int, default=10)
    parser.add_argument('--analytical-lambda0', action='store_true',
                        help='Use analytical lambda0 instead of optimizing it')
    parser.add_argument('--A-init', type=float, default=0.01,
                        help='Initial value for A (default: 0.01)')
    parser.add_argument('--lambda0-init', type=float, default=1.0,
                        help='Initial value for lambda0 (default: 1.0)')
    parser.add_argument('--no-mask', action='store_true',
                        help='Disable pixel masking')
    parser.add_argument('--fix-likelihood', action='store_true',
                        help='Fix A and lambda0 (do not optimize them)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'lbfgs'],
                        help='Optimizer to use (default: adam)')
    args = parser.parse_args()
    main(cell=args.cell, use_analytical_lambda0=args.analytical_lambda0,
         A_init=args.A_init, lambda0_init=args.lambda0_init, use_mask=not args.no_mask,
         fix_likelihood=args.fix_likelihood, optimizer_type=args.optimizer)
