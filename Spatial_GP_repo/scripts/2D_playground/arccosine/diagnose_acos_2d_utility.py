"""Diagnose Arc-Cosine utility values."""
import sys
import torch
import numpy as np
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / '1D_playground'))

from gp_utility_playground import (
    VariationalGP, compute_elbo, generate_poisson_data,
)
from utility_2d_rbf_base import (
    lambda_true_2d, create_2d_grid,
    PoissonLikelihood, standard_utility,
    DEVICE, DTYPE,
    ADAPTIVE_SAFETY_K, ADAPTIVE_MAX_RMAX, ADAPTIVE_MIN_RMAX,
)

from utility_2d_rbf_base import SimpleArcCosineKernel


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate training data
    train_x = create_2d_grid(5, 5, x_range=(-5.0, 5.0), y_range=(-5.0, 5.0))
    train_y = generate_poisson_data(train_x, lambda_true_2d)

    print(f'Training data range: x in [{train_x.min():.2f}, {train_x.max():.2f}]')
    print(f'Spike counts: min={train_y.min():.0f}, max={train_y.max():.0f}, mean={train_y.mean():.2f}')
    print(f'||x||^2 range: [{(train_x**2).sum(-1).min():.2f}, {(train_x**2).sum(-1).max():.2f}]')

    # Create Arc-Cosine model
    n_inducing = 15
    indices = torch.randperm(train_x.shape[0])[:n_inducing]
    inducing_points = train_x[indices]

    likelihood = PoissonLikelihood(A_init=0.01, lambda0_init=1.0).to(DEVICE)
    model = VariationalGP(inducing_points, jitter=1e-4).to(DEVICE)
    model.covar_module = SimpleArcCosineKernel(sigma_0=1.0).to(DEVICE)

    # Check kernel values BEFORE training
    model.eval()
    with torch.no_grad():
        K = model.covar_module(train_x).evaluate()
        print(f'\n=== BEFORE TRAINING ===')
        print(f'Kernel matrix K:')
        print(f'  Shape: {K.shape}')
        print(f'  Min: {K.min():.4f}, Max: {K.max():.4f}')
        print(f'  Diagonal (k(x,x)): min={K.diag().min():.4f}, max={K.diag().max():.4f}')

        # Check GP posterior
        posterior = model(train_x)
        print(f'\nGP posterior at training points:')
        print(f'  Mean: min={posterior.mean.min():.4f}, max={posterior.mean.max():.4f}')
        print(f'  Variance: min={posterior.variance.min():.4f}, max={posterior.variance.max():.4f}')

    # Train briefly
    print('\n=== TRAINING ===')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    with torch.enable_grad():
        for i in range(500):
            optimizer.zero_grad()
            elbo, ell, kl = compute_elbo(model, likelihood, train_x, train_y)
            (-elbo).backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'  Iter {i+1}: ELBO = {elbo.item():.2f}')

    # Check kernel values AFTER training
    model.eval()
    with torch.no_grad():
        K = model.covar_module(train_x).evaluate()
        print(f'\n=== AFTER TRAINING ===')
        print(f'Kernel matrix K:')
        print(f'  Min: {K.min():.4f}, Max: {K.max():.4f}')
        print(f'  Diagonal: min={K.diag().min():.4f}, max={K.diag().max():.4f}')

        posterior = model(train_x)
        print(f'\nGP posterior:')
        print(f'  Mean: min={posterior.mean.min():.4f}, max={posterior.mean.max():.4f}')
        print(f'  Variance: min={posterior.variance.min():.4f}, max={posterior.variance.max():.4f}')

        # Try utility
        print(f'\n=== UTILITY TEST ===')
        test_points = torch.tensor([[0.0, 0.0], [5.0, 5.0], [-5.0, -5.0]], dtype=DTYPE, device=DEVICE)
        utility = standard_utility(model, likelihood, test_points, r_max=None, adaptive_r_max=True, adaptive_safety_k=ADAPTIVE_SAFETY_K, adaptive_max_rmax=ADAPTIVE_MAX_RMAX, adaptive_min_rmax=ADAPTIVE_MIN_RMAX)['utility']
        print(f'Utility at (0,0): {utility[0].item():.10f}')
        print(f'Utility at (5,5): {utility[1].item():.10f}')
        print(f'Utility at (-5,-5): {utility[2].item():.10f}')
        print(f'Any NaN? {torch.isnan(utility).any()}')
        print(f'Any Inf? {torch.isinf(utility).any()}')

        # Check the full utility grid
        print(f'\n=== FULL UTILITY GRID ===')
        x_eval = torch.linspace(-5, 5, 10, dtype=DTYPE, device=DEVICE)
        y_eval = torch.linspace(-5, 5, 10, dtype=DTYPE, device=DEVICE)
        eval_grid_x, eval_grid_y = torch.meshgrid(x_eval, y_eval, indexing='xy')
        eval_flat = torch.stack([eval_grid_x.flatten(), eval_grid_y.flatten()], dim=-1)

        full_utility = standard_utility(model, likelihood, eval_flat, r_max=None, adaptive_r_max=True, adaptive_safety_k=ADAPTIVE_SAFETY_K, adaptive_max_rmax=ADAPTIVE_MAX_RMAX, adaptive_min_rmax=ADAPTIVE_MIN_RMAX)['utility']
        print(f'Utility stats:')
        print(f'  Min: {full_utility.min():.10f}')
        print(f'  Max: {full_utility.max():.10f}')
        print(f'  Mean: {full_utility.mean():.10f}')
        print(f'  Any NaN? {torch.isnan(full_utility).any()}')
        print(f'  Any Inf? {torch.isinf(full_utility).any()}')
        print(f'  How many zeros? {(full_utility == 0).sum()}')
        print(f'  How many near-zero (< 1e-10)? {(full_utility.abs() < 1e-10).sum()}')
