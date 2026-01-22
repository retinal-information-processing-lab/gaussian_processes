"""
Compare whitened vs unwhitened vargp_style training to identify divergence point.

Created by Claude to investigate whitening collapse hypothesis.

HYPOTHESIS:
- Legacy (unwhitened) mode NEVER collapses regardless of seed
- Whitened mode collapses with certain seeds (456 at M=50, 42 at M=75)
- Need to understand WHY whitening fails where unwhitened doesn't

STRATEGY:
1. Initialize BOTH modes with IDENTICAL parameters (same seed, inducing points, params)
2. Track every iteration:
   - Variational parameters (m, V) in both natural and whitened space
   - L_K condition number
   - KL divergence
   - Expected log-likelihood
   - A, lambda0 values
3. Identify where the trajectories diverge
4. Check if whitening transformation causes numerical issues

OUTPUT:
- CSV files with parallel trajectories
- Plots showing divergence points
"""

import sys
import os
import argparse
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import VariationalGPModel
from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from estep import (
    e_step_loop,
    compute_kernel_cache,
    compute_L_K,
)
from fstep import f_step_lbfgs
from mstep import m_step
from whitening import (
    get_variational_mean,
    get_variational_covar,
    get_variational_mean_with_L_K,
    get_variational_covar_with_L_K,
    set_kernel_requires_grad,
)


def load_data(device, n_train=2000, cellid=8):
    """Load PNAS data (simplified version)."""
    data_path = '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/notebooks/PNAS_paper_sorted_data.npz'
    data = np.load(data_path)

    # Load training data
    X_train = torch.tensor(data['images_train'], dtype=torch.float64, device=device)
    X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten
    X_train = X_train[:n_train]

    R_train = torch.tensor(data['responses_train'], dtype=torch.float64, device=device)
    r_train = R_train[:n_train, cellid]

    return {'X_train': X_train, 'r_train': r_train}


def initialize_models(X_train, r_train, ntilde, seed, device, jitter=1e-4):
    """Initialize two IDENTICAL models (whitened and unwhitened) with same parameters.

    Returns:
        dict: {
            'whitened': {'model': ..., 'likelihood': ...},
            'unwhitened': {'model': ..., 'likelihood': ...}
        }
    """
    torch.manual_seed(seed)

    # Select inducing points (same for both)
    idx = torch.randperm(X_train.shape[0], device=device)[:ntilde]
    inducing_points = X_train[idx].clone()

    # Initialize kernel (same hyperparams for both)
    n_px_side = int(np.sqrt(X_train.shape[1]))
    kernel_params = {
        'beta': 0.1,
        'rho': 0.1,
        'eps_0': (0.0, 0.0),
        'sigma_0': 1.0,
    }

    models = {}

    for mode in ['whitened', 'unwhitened']:
        use_whitening = (mode == 'whitened')

        # Create kernel (SHARED parameters)
        # ArcCosineKernel now has internal Amp parameter (matches legacy varGP)
        # No need for ScaleKernel wrapper
        kernel = ArcCosineKernel(
            n_px_side=n_px_side,
            sigma_0=kernel_params['sigma_0'],
            Amp=1e-4,  # Amplitude inside C matrix
            beta=kernel_params['beta'],
            rho=kernel_params['rho'],
            eps_0x=kernel_params['eps_0'][0] if isinstance(kernel_params['eps_0'], (list, tuple)) else kernel_params['eps_0'],
            eps_0y=kernel_params['eps_0'][1] if isinstance(kernel_params['eps_0'], (list, tuple)) else kernel_params['eps_0'],
            use_mask=True,
            gradient_mode='autograd',
        )

        # Create model
        model = VariationalGPModel(
            inducing_points=inducing_points.clone(),
            kernel=kernel,
            jitter=jitter,
            whitening=use_whitening
        )
        model = model.to(device).double()

        # Create likelihood (SAME initial params)
        likelihood = PoissonLikelihood(A_init=0.01, lambda0_init=1.0)
        likelihood = likelihood.to(device).double()

        # Set variational parameters to SAME initial values
        # Both start with prior: m=0, V=K̃
        with torch.no_grad():
            model.variational_strategy._variational_distribution.variational_mean.zero_()
            # V = I (GPyTorch interprets as K̃ for whitened, I for unwhitened)
            M = inducing_points.shape[0]
            eye_chol = torch.eye(M, dtype=torch.float64, device=device)
            model.variational_strategy._variational_distribution.chol_variational_covar.copy_(eye_chol)

        models[mode] = {'model': model, 'likelihood': likelihood}

    return models, inducing_points


def track_iteration(mode, iteration, model, likelihood, X_train, r_train, kernel_cache,
                    use_whitening, jitter, records):
    """Track all relevant quantities for one mode at one iteration.

    Args:
        mode: 'whitened' or 'unwhitened'
        iteration: Iteration number
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X_train: Training data
        r_train: Training responses
        kernel_cache: Precomputed kernel cache
        use_whitening: Whether whitening is enabled
        jitter: Jitter value
        records: List to append record to
    """
    with torch.no_grad():
        # Get variational parameters in NATURAL space
        if use_whitening:
            L_K = kernel_cache['L_K']
            m_natural = get_variational_mean_with_L_K(model, L_K)
            V_natural = get_variational_covar_with_L_K(model, L_K)

            # Also get stored (whitened) params
            m_stored = get_variational_mean(model)
            V_stored = get_variational_covar(model)
        else:
            m_natural = get_variational_mean(model)
            V_natural = get_variational_covar(model)
            m_stored = m_natural.clone()
            V_stored = V_natural.clone()

        # Compute L_K condition number
        L_K = compute_L_K(model, jitter)
        K_tilde = L_K @ L_K.T
        eigvals = torch.linalg.eigvalsh(K_tilde)
        cond_L_K = eigvals.max() / eigvals.min()

        # Compute KL divergence
        kl = model.variational_strategy.kl_divergence().item()

        # Compute expected log-likelihood
        output = model(X_train)
        lambda_m = output.mean
        lambda_var = output.variance
        ell = likelihood.expected_log_prob(r_train, output).item()

        # Get firing rate params
        A = likelihood.A.squeeze().item()
        lambda0 = likelihood.lambda0.squeeze().item()

        # Norms and condition numbers
        m_norm = m_natural.norm().item()
        m_stored_norm = m_stored.norm().item()
        V_norm = V_natural.norm().item()
        V_stored_norm = V_stored.norm().item()
        V_cond = torch.linalg.cond(V_natural).item()
        V_stored_cond = torch.linalg.cond(V_stored).item()

        # Mean firing rate
        f_mean = torch.exp(A * lambda_m + 0.5 * A**2 * lambda_var + lambda0)
        f_mean_avg = f_mean.mean().item()

        record = {
            'mode': mode,
            'iteration': iteration,
            'kl': kl,
            'ell': ell,
            'elbo': ell - kl,
            'A': A,
            'lambda0': lambda0,
            'm_norm': m_norm,
            'm_stored_norm': m_stored_norm,
            'V_norm': V_norm,
            'V_stored_norm': V_stored_norm,
            'V_cond': V_cond,
            'V_stored_cond': V_stored_cond,
            'L_K_cond': cond_L_K.item(),
            'f_mean_avg': f_mean_avg,
        }
        records.append(record)

        return record


def run_training(mode, model, likelihood, X_train, r_train, n_iter, n_estep, n_fstep,
                n_mstep, jitter, records):
    """Run vargp_style training for one mode with detailed tracking.

    Args:
        mode: 'whitened' or 'unwhitened'
        model: VariationalGPModel instance
        likelihood: PoissonLikelihood instance
        X_train: Training data
        r_train: Training responses
        n_iter: Number of EM iterations
        n_estep: Newton iterations per E-step
        n_fstep: LBFGS iterations per F-step
        n_mstep: Adam iterations per M-step
        jitter: Jitter value
        records: List to append tracking records to
    """
    use_whitening = (mode == 'whitened')

    # Pre-compute kernel cache (used throughout training)
    kernel_cache = compute_kernel_cache(model, X_train, jitter)
    kernel_cache['L_K'] = compute_L_K(model, jitter)

    # Track initial state (iteration 0)
    print(f"\n{mode.upper()} MODE - Iteration 0:")
    rec = track_iteration(mode, 0, model, likelihood, X_train, r_train,
                         kernel_cache, use_whitening, jitter, records)
    print(f"  KL: {rec['kl']:.2f}, ELL: {rec['ell']:.2f}, ELBO: {rec['elbo']:.2f}")
    print(f"  A: {rec['A']:.4f}, lambda0: {rec['lambda0']:.4f}")
    print(f"  m_norm: {rec['m_norm']:.4f}, V_cond: {rec['V_cond']:.2e}")
    print(f"  L_K_cond: {rec['L_K_cond']:.2e}")

    for it in range(n_iter):
        print(f"\n{mode.upper()} MODE - Iteration {it+1}/{n_iter}")

        # E-step: update (m, V)
        set_kernel_requires_grad(model, False)
        lambda_m, lambda_var = e_step_loop(
            model, likelihood, X_train, r_train,
            n_estep=n_estep,
            jitter=jitter,
            kernel_cache=kernel_cache,
            use_whitening=use_whitening,
            verbose=False
        )

        # F-step: update A, lambda0
        f_step_lbfgs(
            model, likelihood, X_train, r_train, lambda_m, lambda_var,
            n_fstep=n_fstep,
            lr=0.1
        )

        # M-step: update kernel hyperparameters
        set_kernel_requires_grad(model, True)

        # Recompute kernel cache after M-step (kernel changed)
        # But do M-step first
        m_step(
            model, likelihood, X_train, r_train,
            n_mstep=n_mstep,
            lr=0.1
        )

        # Recompute kernel cache (kernel hyperparams changed in M-step)
        kernel_cache = compute_kernel_cache(model, X_train, jitter)
        kernel_cache['L_K'] = compute_L_K(model, jitter)

        # Track state after full iteration
        rec = track_iteration(mode, it+1, model, likelihood, X_train, r_train,
                            kernel_cache, use_whitening, jitter, records)
        print(f"  KL: {rec['kl']:.2f}, ELL: {rec['ell']:.2f}, ELBO: {rec['elbo']:.2f}")
        print(f"  A: {rec['A']:.4f}, lambda0: {rec['lambda0']:.4f}")
        print(f"  m_norm: {rec['m_norm']:.4f}, V_cond: {rec['V_cond']:.2e}")
        print(f"  L_K_cond: {rec['L_K_cond']:.2e}")

        # Check for collapse
        if rec['elbo'] < -1e6 or np.isnan(rec['elbo']):
            print(f"  WARNING: {mode} mode collapsed!")
            break


def plot_comparison(records, output_dir):
    """Create comparison plots showing where trajectories diverge."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle('Whitened vs Unwhitened Training Comparison', fontsize=14, y=0.995)

    metrics = [
        ('elbo', 'ELBO'),
        ('kl', 'KL Divergence'),
        ('ell', 'Expected Log-Likelihood'),
        ('A', 'Gain Parameter A'),
        ('lambda0', 'Bias Parameter λ₀'),
        ('m_norm', '||m|| (natural)'),
        ('m_stored_norm', '||m|| (stored)'),
        ('V_norm', '||V|| (natural)'),
        ('V_stored_norm', '||V|| (stored)'),
        ('V_cond', 'cond(V) natural'),
        ('V_stored_cond', 'cond(V) stored'),
        ('L_K_cond', 'cond(L_K)'),
    ]

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        for mode in ['whitened', 'unwhitened']:
            # Filter records by mode
            mode_records = [r for r in records if r['mode'] == mode]
            iterations = [r['iteration'] for r in mode_records]
            values = [r[metric] for r in mode_records]

            ax.plot(iterations, values, marker='o', label=mode, alpha=0.7)

        ax.set_xlabel('Iteration')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Use log scale for certain metrics
        if metric in ['V_cond', 'V_stored_cond', 'L_K_cond', 'kl']:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_plot.png', dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_dir / 'comparison_plot.png'}")
    plt.close()


def plot_divergence_metrics(records, output_dir):
    """Plot metrics that highlight divergence between modes."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Divergence Metrics: Whitened vs Unwhitened', fontsize=14)

    # Separate by mode
    whitened_records = sorted([r for r in records if r['mode'] == 'whitened'],
                             key=lambda x: x['iteration'])
    unwhitened_records = sorted([r for r in records if r['mode'] == 'unwhitened'],
                               key=lambda x: x['iteration'])

    # Ensure same iterations
    w_iters = [r['iteration'] for r in whitened_records]
    u_iters = [r['iteration'] for r in unwhitened_records]
    common_iters = sorted(set(w_iters) & set(u_iters))

    whitened_records = [r for r in whitened_records if r['iteration'] in common_iters]
    unwhitened_records = [r for r in unwhitened_records if r['iteration'] in common_iters]

    metrics_to_compare = [
        ('m_norm', '||m|| Difference'),
        ('V_norm', '||V|| Difference'),
        ('elbo', 'ELBO Difference'),
        ('A', 'A Difference'),
    ]

    for idx, (metric, label) in enumerate(metrics_to_compare):
        ax = axes[idx // 2, idx % 2]

        # Compute differences
        iterations = [r['iteration'] for r in whitened_records]
        w_values = [r[metric] for r in whitened_records]
        u_values = [r[metric] for r in unwhitened_records]
        diff = np.array(w_values) - np.array(u_values)

        ax.plot(iterations, diff, marker='o', color='red', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        ax.set_xlabel('Iteration')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{label} (whitened - unwhitened)')

    plt.tight_layout()
    plt.savefig(output_dir / 'divergence_metrics.png', dpi=150, bbox_inches='tight')
    print(f"Saved divergence plot to {output_dir / 'divergence_metrics.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare whitened vs unwhitened training')
    parser.add_argument('--ntilde', type=int, default=50, help='Number of inducing points')
    parser.add_argument('--seed', type=int, default=456, help='Random seed (456 = known collapse)')
    parser.add_argument('--n-train', type=int, default=2000, help='Training samples')
    parser.add_argument('--n-iter', type=int, default=10, help='EM iterations')
    parser.add_argument('--n-estep', type=int, default=10, help='E-step Newton iterations')
    parser.add_argument('--n-fstep', type=int, default=10, help='F-step LBFGS iterations')
    parser.add_argument('--n-mstep', type=int, default=10, help='M-step Adam iterations')
    parser.add_argument('--cellid', type=int, default=8, help='Neuron ID')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--jitter', type=float, default=1e-4, help='Jitter value')
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    output_dir = Path(__file__).parent / f'whitening_comparison_M{args.ntilde}_seed{args.seed}'
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print(f"WHITENING COMPARISON: M={args.ntilde}, seed={args.seed}")
    print("="*80)
    print(f"Training: {args.n_train} samples, {args.n_iter} iterations")
    print(f"E-step: {args.n_estep} Newton, F-step: {args.n_fstep} LBFGS, M-step: {args.n_mstep} Adam")
    print(f"Device: {device}, Jitter: {args.jitter}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    print("Loading data...")
    data_dict = load_data(device, n_train=args.n_train, cellid=args.cellid)
    X_train = data_dict['X_train']
    r_train = data_dict['r_train']
    print(f"X_train: {X_train.shape}, r_train: {r_train.shape}")

    # Initialize models with IDENTICAL parameters
    print("\nInitializing models with IDENTICAL parameters...")
    models, inducing_points = initialize_models(
        X_train, r_train, args.ntilde, args.seed, device, jitter=args.jitter
    )
    print(f"Inducing points: {inducing_points.shape}")
    print("Both models initialized with:")
    print("  - Same inducing points")
    print("  - Same kernel hyperparameters")
    print("  - Same firing rate parameters (A=0.01, λ₀=1.0)")
    print("  - Same variational parameters (m=0, V=I)")

    # Track both training runs
    records = []

    # Run whitened training
    print("\n" + "="*80)
    print("TRAINING WHITENED MODE")
    print("="*80)
    run_training(
        'whitened',
        models['whitened']['model'],
        models['whitened']['likelihood'],
        X_train, r_train,
        n_iter=args.n_iter,
        n_estep=args.n_estep,
        n_fstep=args.n_fstep,
        n_mstep=args.n_mstep,
        jitter=args.jitter,
        records=records
    )

    # Run unwhitened training
    print("\n" + "="*80)
    print("TRAINING UNWHITENED MODE")
    print("="*80)
    run_training(
        'unwhitened',
        models['unwhitened']['model'],
        models['unwhitened']['likelihood'],
        X_train, r_train,
        n_iter=args.n_iter,
        n_estep=args.n_estep,
        n_fstep=args.n_fstep,
        n_mstep=args.n_mstep,
        jitter=args.jitter,
        records=records
    )

    # Save results
    csv_path = output_dir / 'training_comparison.csv'
    with open(csv_path, 'w', newline='') as f:
        if records:
            fieldnames = list(records[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    print(f"\nSaved training data to {csv_path}")

    # Create plots
    print("\nGenerating plots...")
    plot_comparison(records, output_dir)
    plot_divergence_metrics(records, output_dir)

    # Summary analysis
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for mode in ['whitened', 'unwhitened']:
        mode_records = [r for r in records if r['mode'] == mode]
        if mode_records:
            final = mode_records[-1]
            print(f"\n{mode.upper()} - Final state (iteration {int(final['iteration'])}):")
            print(f"  ELBO: {final['elbo']:.2f}")
            print(f"  KL: {final['kl']:.2f}")
            print(f"  ELL: {final['ell']:.2f}")
            print(f"  A: {final['A']:.4f}, λ₀: {final['lambda0']:.4f}")
            print(f"  ||m||: {final['m_norm']:.4f}")
            print(f"  cond(V): {final['V_cond']:.2e}")
            print(f"  cond(L_K): {final['L_K_cond']:.2e}")

    print(f"\nAll results saved to {output_dir}/")


if __name__ == '__main__':
    main()
