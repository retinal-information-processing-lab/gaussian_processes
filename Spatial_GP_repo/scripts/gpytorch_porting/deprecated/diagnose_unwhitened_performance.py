#!/usr/bin/env python3
"""
Diagnostic Script: Unwhitened vs Whitened Performance Investigation

Created by Claude to empirically verify why UnwhitenedVariationalStrategy
achieves worse accuracy (0.6878 vs 0.8447 explained variance) despite
correct mathematical formulation.

HYPOTHESIS: The performance gap is due to gradient magnitude imbalance:
- Whitened: KL gradient w.r.t. m scales O(1)
- Unwhitened: KL gradient w.r.t. m scales O(cond(K_tilde)) ~ 10^4-10^5

This script measures:
1. Condition number of K_tilde (inducing kernel)
2. KL divergence values for equivalent natural parameters
3. Gradient magnitudes for variational parameters during training
4. Parameter evolution across iterations

Usage:
    conda run -n pytorch_gpytorch python tests/diagnose_unwhitened_performance.py
    conda run -n pytorch_gpytorch python tests/diagnose_unwhitened_performance.py --ntilde 75
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

import torch
torch.set_grad_enabled(True)  # Need gradients for diagnosis

# Add repo parent to sys.path for `from gaussian_processes.Spatial_GP_repo import ...`
_repo_root = next(p for p in Path(__file__).resolve().parents if (p / 'Spatial_GP_repo').is_dir())
sys.path.insert(0, str(_repo_root.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import gpytorch
from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from model import VariationalGPModel
from estep import compute_kernel_cache
from tests.test_utils import set_reproducible_seed


def load_data(device, dtype=torch.float64):
    """Load PNAS data."""
    data_path = Path(__file__).parent.parent.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = np.load(data_path)

    X = np.concatenate([data['images_train'], data['images_val']], axis=0)
    R = np.concatenate([data['responses_train'], data['responses_val']], axis=0)
    X = X.reshape(X.shape[0], -1)

    return {
        'X': torch.tensor(X, dtype=dtype, device=device),
        'R': torch.tensor(R, dtype=dtype, device=device),
    }


def create_model(inducing_points, params, device, whitening=True):
    """Create model with specified whitening setting."""
    # ArcCosineKernel now has internal Amp parameter (matches legacy varGP)
    # No need for ScaleKernel wrapper
    kernel = ArcCosineKernel(
        sigma_0=params['sigma_0'],
        Amp=1e-4,  # Amplitude inside C matrix
        n_px_side=params['n_px_side'],
        eps_0x=params['eps_0x'],
        eps_0y=params['eps_0y'],
        beta=params['beta'],
        rho=params['rho'],
        use_mask=True,
    )

    model = VariationalGPModel(inducing_points.clone(), kernel, jitter=1e-4, whitening=whitening)
    likelihood = PoissonLikelihood(A_init=0.01, lambda0_init=1.0)

    model = model.double().to(device)
    likelihood = likelihood.double().to(device)

    return model, likelihood


def get_natural_params(model):
    """Get variational parameters in natural space."""
    vs = model.variational_strategy
    vd = vs.variational_distribution

    m_stored = vd.mean.detach()
    L_stored = vd.chol_variational_covar.detach()
    V_stored = L_stored @ L_stored.T

    if model.whitening:
        # Whitened: m_stored = L_K^{-1} @ m_natural
        # Need to compute L_K
        inducing_points = vs.inducing_points
        K_tilde = model.covar_module(inducing_points).evaluate()
        M = K_tilde.shape[0]
        jitter = model.jitter
        K_tilde_j = K_tilde + jitter * torch.eye(M, dtype=K_tilde.dtype, device=K_tilde.device)
        L_K = torch.linalg.cholesky(K_tilde_j)

        # Convert to natural
        m_natural = L_K @ m_stored
        V_natural = L_K @ V_stored @ L_K.T
    else:
        # Unwhitened: stored directly in natural space
        m_natural = m_stored
        V_natural = V_stored

    return m_natural, V_natural


def compute_kl_divergence(model, X_train, r_train, likelihood):
    """Compute KL divergence component of ELBO."""
    model.train()
    output = model(X_train)
    kl = model.variational_strategy.kl_divergence()
    return kl.detach()


def compute_gradient_norms(model, X_train, r_train, likelihood):
    """Compute gradient norms for variational and kernel parameters."""
    model.train()

    # Zero grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    for p in likelihood.parameters():
        if p.grad is not None:
            p.grad.zero_()

    # Forward pass
    output = model(X_train)
    expected_log_lik = likelihood.expected_log_prob(r_train, output)
    kl = model.variational_strategy.kl_divergence()
    loss = -expected_log_lik + kl

    # Backward
    loss.backward()

    # Collect gradient norms
    grads = {}

    # Variational parameters - access via variational_strategy._variational_distribution
    vs = model.variational_strategy
    # In GPyTorch, variational parameters are stored in _variational_distribution
    for name, param in vs.named_parameters():
        if param.grad is not None:
            grads[f'grad_{name}'] = param.grad.norm().item()
            # Variational mean
            if 'variational_mean' in name:
                grads['grad_m'] = param.grad.norm().item()
                grads['grad_m_max'] = param.grad.abs().max().item()
                grads['grad_m_min'] = param.grad.abs().min().item()
            # Variational covariance (cholesky)
            if 'chol' in name.lower():
                grads['grad_L'] = param.grad.norm().item()

    # Kernel parameter gradients
    for name, param in model.covar_module.named_parameters():
        if param.grad is not None:
            grads[f'grad_{name}'] = param.grad.norm().item()

    return grads, loss.item()


def compute_ktilde_conditioning(model):
    """Compute condition number of K_tilde."""
    inducing_points = model.variational_strategy.inducing_points
    K_tilde = model.covar_module(inducing_points).evaluate()

    # Eigenvalues
    eigvals = torch.linalg.eigvalsh(K_tilde)
    cond_num = (eigvals.max() / eigvals.min()).item()

    return {
        'cond_num': cond_num,
        'eigval_max': eigvals.max().item(),
        'eigval_min': eigvals.min().item(),
        'eigval_ratio': (eigvals.max() / eigvals.min()).item(),
    }


def run_diagnostic(params, device, n_iterations=10):
    """Run diagnostic comparison between whitened and unwhitened."""

    print("="*70)
    print("DIAGNOSTIC: Whitened vs Unwhitened Performance Investigation")
    print("="*70)
    print(f"\nParameters: M={params['ntilde']}, n_train={params['n_train']}")

    # Load data
    data = load_data(device)
    X = data['X']
    R = data['R']

    cellid = params['cellid']
    r = R[:, cellid]

    # Same random subset as test_estep_comparison
    set_reproducible_seed(42)
    indices = torch.randperm(X.shape[0], device=device)[:params['n_train']]
    X_train = X[indices]
    r_train = r[indices]

    inducing_points = X_train[:params['ntilde']].clone()

    # Create both models
    model_w, lik_w = create_model(inducing_points, params, device, whitening=True)
    model_u, lik_u = create_model(inducing_points, params, device, whitening=False)

    # Initial K_tilde conditioning
    print("\n" + "-"*50)
    print("K_tilde Conditioning:")
    cond_info = compute_ktilde_conditioning(model_w)
    print(f"  Condition number: {cond_info['cond_num']:.2e}")
    print(f"  Eigenvalue range: [{cond_info['eigval_min']:.2e}, {cond_info['eigval_max']:.2e}]")
    print("-"*50)

    # Track metrics across iterations
    metrics_whitened = []
    metrics_unwhitened = []

    print(f"\nRunning {n_iterations} training iterations for each strategy...")
    print("\n" + "-"*70)
    print(f"{'Iter':>4} | {'Strategy':>10} | {'Loss':>10} | {'KL':>10} | {'grad_m':>12} | {'grad_m ratio':>12}")
    print("-"*70)

    for i in range(n_iterations):
        # Get gradients and loss for whitened
        grads_w, loss_w = compute_gradient_norms(model_w, X_train, r_train, lik_w)
        kl_w = compute_kl_divergence(model_w, X_train, r_train, lik_w)

        # Get gradients and loss for unwhitened
        grads_u, loss_u = compute_gradient_norms(model_u, X_train, r_train, lik_u)
        kl_u = compute_kl_divergence(model_u, X_train, r_train, lik_u)

        # Store
        metrics_whitened.append({
            'loss': loss_w, 'kl': kl_w.item(), **grads_w
        })
        metrics_unwhitened.append({
            'loss': loss_u, 'kl': kl_u.item(), **grads_u
        })

        # Compute gradient ratio
        grad_m_ratio = grads_u.get('grad_m', 0) / max(grads_w.get('grad_m', 1e-10), 1e-10)

        print(f"{i:>4} | {'whitened':>10} | {loss_w:>10.2f} | {kl_w.item():>10.2f} | {grads_w.get('grad_m', 0):>12.4e} | -")
        print(f"{i:>4} | {'unwhitened':>10} | {loss_u:>10.2f} | {kl_u.item():>10.2f} | {grads_u.get('grad_m', 0):>12.4e} | {grad_m_ratio:>12.2f}x")

        # Take one optimization step for each
        with torch.no_grad():
            # Simple SGD step for illustration
            lr = 0.01
            vs_w = model_w.variational_strategy
            vs_u = model_u.variational_strategy

            for name, param in vs_w.named_parameters():
                if param.grad is not None and 'variational_mean' in name:
                    param.copy_(param - lr * param.grad)
            for name, param in vs_u.named_parameters():
                if param.grad is not None and 'variational_mean' in name:
                    param.copy_(param - lr * param.grad)

    print("-"*70)

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    avg_grad_m_w = np.mean([m['grad_m'] for m in metrics_whitened])
    avg_grad_m_u = np.mean([m['grad_m'] for m in metrics_unwhitened])

    print(f"\nMean gradient norm (grad_m):")
    print(f"  Whitened:   {avg_grad_m_w:.4e}")
    print(f"  Unwhitened: {avg_grad_m_u:.4e}")
    print(f"  Ratio:      {avg_grad_m_u/max(avg_grad_m_w, 1e-10):.2f}x")

    print(f"\nGradient range (grad_m_min to grad_m_max):")
    print(f"  Whitened:   [{metrics_whitened[0].get('grad_m_min', 0):.2e}, {metrics_whitened[0].get('grad_m_max', 0):.2e}]")
    print(f"  Unwhitened: [{metrics_unwhitened[0].get('grad_m_min', 0):.2e}, {metrics_unwhitened[0].get('grad_m_max', 0):.2e}]")

    print(f"\nK_tilde condition number: {cond_info['cond_num']:.2e}")
    print(f"\nExpected gradient imbalance from theory: O(cond) = O({cond_info['cond_num']:.0e})")

    # Check if empirical matches theory
    empirical_ratio = avg_grad_m_u / max(avg_grad_m_w, 1e-10)
    theory_ratio = cond_info['cond_num']

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print(f"\nEmpirical gradient ratio:  {empirical_ratio:.2f}x")
    print(f"Theoretical (cond number): {theory_ratio:.2e}x")

    if empirical_ratio > 10:
        print("\n** CONFIRMED: Significant gradient imbalance observed **")
        print("   Unwhitened gradients are much larger than whitened.")
        print("   This can cause optimization instability and worse convergence.")
    else:
        print("\n** Note: Gradient imbalance less severe than expected **")
        print("   The performance gap may have other causes.")

    print("\n")
    return metrics_whitened, metrics_unwhitened, cond_info


def main():
    parser = argparse.ArgumentParser(description='Diagnose unwhitened performance gap')
    parser.add_argument('--ntilde', type=int, default=50, help='Number of inducing points')
    parser.add_argument('--n-train', type=int, default=500, help='Training samples')
    parser.add_argument('--n-iterations', type=int, default=10, help='Training iterations to analyze')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    params = {
        'cellid': 8,
        'n_train': args.n_train,
        'n_px_side': 108,
        'ntilde': args.ntilde,
        'beta': 0.1,
        'rho': 0.1,
        'sigma_0': 1.0,
        'eps_0x': 0.0,
        'eps_0y': 0.0,
    }

    run_diagnostic(params, device, n_iterations=args.n_iterations)


if __name__ == '__main__':
    main()
