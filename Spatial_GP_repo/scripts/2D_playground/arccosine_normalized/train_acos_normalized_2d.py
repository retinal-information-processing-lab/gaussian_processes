"""
Train Normalized Arc-Cosine GP on 2D Grid - Created by Claude
==============================================================
Training-only script that generates a checkpoint for utility_acos_normalized_2d_base.py.

This script:
1. Generates 5x5 training grid in [-5, 5]^2 (same as RBF and unnormalized arc-cosine)
2. Trains Normalized Arc-Cosine GP with Poisson likelihood (gpytorch_porting version)
3. Saves checkpoint with kernel params (sigma_0) and likelihood params (A, lambda0)

The normalized kernel has K_bar(x,x) = 1 for all x (constant prior variance),
eliminating the norm-scaling incentive that causes utility divergence at domain corners.

Related files:
- utility_acos_normalized_2d_base.py (utility visualization using this checkpoint)
- ../arccosine/train_acos_2d.py (unnormalized version for comparison)
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent))  # 2D_playground
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "1D_playground"))

# From 2D playground (config + data generation helpers)
from utility_2d_rbf_base import (
    N_TRAIN_X_DEFAULT, N_TRAIN_Y_DEFAULT,
    TRAIN_X_RANGE_DEFAULT, TRAIN_Y_RANGE_DEFAULT,
    DEVICE, DTYPE,
    lambda_true_2d,
    create_2d_grid,
    # From gpytorch_porting (re-exported)
    SimpleArcCosineNormalizedKernel,
    PoissonLikelihood,
)

# From 1D playground (model class + data generation + compute_elbo)
from gp_utility_playground import (
    VariationalGP,
    generate_poisson_data,
    compute_elbo,
)


def train_arccosine_normalized_gp(model, likelihood, train_x, train_y, n_iterations=500, lr=0.1):
    """Train GP with SimpleArcCosineNormalizedKernel."""
    model.train()
    likelihood.train()

    params = list(model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    with torch.enable_grad():
        for i in range(n_iterations):
            optimizer.zero_grad()
            elbo, ell, kl = compute_elbo(model, likelihood, train_x, train_y)
            (-elbo).backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                sigma_0 = model.covar_module.sigma_0.item()
                A = likelihood.A.item()
                lambda0 = likelihood.lambda0.item()
                print(f"  Iter {i+1}/{n_iterations}, ELBO: {elbo.item():.2f} "
                      f"| sigma_0={sigma_0:.4f}, A={A:.4f}, lambda_0={lambda0:.3f}")

    model.eval()
    likelihood.eval()
    return model, likelihood


def save_checkpoint(model, likelihood, inducing_points, train_x, train_y, filepath, **kwargs):
    """Save normalized arc-cosine checkpoint with kernel and likelihood parameters."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'inducing_points': inducing_points,
        'train_x': train_x,
        'train_y': train_y,
        'kernel_params': {
            'sigma_0': model.covar_module.sigma_0.item(),
        },
        'likelihood_params': {
            'A': likelihood.A.item(),
            'lambda_0': likelihood.lambda0.item(),
        },
        'training_config': kwargs
    }
    torch.save(checkpoint, filepath)
    print(f"Saved normalized arc-cosine checkpoint to {filepath}")


def main():
    print("=" * 70)
    print("Train Normalized Arc-Cosine GP - 2D Playground")
    print("=" * 70)

    # Fixed seed for reproducibility (same as unnormalized version)
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Training configuration (same as unnormalized for fair comparison)
    n_train_x = N_TRAIN_X_DEFAULT
    n_train_y = N_TRAIN_Y_DEFAULT
    train_x_range = TRAIN_X_RANGE_DEFAULT
    train_y_range = TRAIN_Y_RANGE_DEFAULT

    print(f"\nTraining Configuration:")
    print(f"  Grid size: {n_train_x} x {n_train_y} = {n_train_x * n_train_y} points")
    print(f"  X range: {train_x_range}")
    print(f"  Y range: {train_y_range}")

    # Generate training data (same as unnormalized)
    print(f"\nGenerating training data...")
    train_x = create_2d_grid(n_train_x, n_train_y,
                             x_range=train_x_range,
                             y_range=train_y_range)
    train_y = generate_poisson_data(train_x, lambda_true_2d)

    print(f"  Training points: {train_x.shape}")
    print(f"  Spike count range: [{train_y.min().item():.0f}, {train_y.max().item():.0f}]")

    # Use 15 random inducing points (subset of training) - same as unnormalized
    n_inducing = 15
    indices = torch.randperm(len(train_x))[:n_inducing]
    inducing_points = train_x[indices].clone()

    # Initialize model with RBF kernel (default)
    model = VariationalGP(inducing_points, jitter=1e-4).to(DEVICE)

    # Swap to SimpleArcCosineNormalizedKernel
    model.covar_module = SimpleArcCosineNormalizedKernel(sigma_0=1.0).to(DEVICE)

    # Use gpytorch_porting PoissonLikelihood (with A, lambda0)
    likelihood = PoissonLikelihood(A_init=0.01, lambda0_init=1.0).to(DEVICE)

    print(f"\nModel initialized:")
    print(f"  Inducing points: {inducing_points.shape[0]} (random subset)")
    print(f"  Kernel: SimpleArcCosineNormalizedKernel (K_bar(x,x) = 1)")
    print(f"  Initial kernel: sigma_0={model.covar_module.sigma_0.item():.3f}")
    print(f"  Initial likelihood: A={likelihood.A.item():.4f}, lambda_0={likelihood.lambda0.item():.3f}")

    # Training settings (same as unnormalized)
    n_iterations = 500
    learning_rate = 0.1

    print(f"\nTraining Normalized Arc-Cosine GP...")
    print(f"  Iterations: {n_iterations}")
    print(f"  Learning rate: {learning_rate}")

    model, likelihood = train_arccosine_normalized_gp(model, likelihood, train_x, train_y,
                                                       n_iterations=n_iterations,
                                                       lr=learning_rate)

    # Compute final ELBO
    print(f"\nComputing final ELBO...")
    with torch.no_grad():
        final_elbo, ell, kl = compute_elbo(model, likelihood, train_x, train_y)
    print(f"  Final ELBO: {final_elbo.item():.2f} (ELL: {ell.item():.2f}, KL: {kl.item():.2f})")

    # Print final hyperparameters
    print(f"\nFinal Hyperparameters:")
    print(f"  Kernel: sigma_0={model.covar_module.sigma_0.item():.4f}")
    print(f"  Likelihood: A={likelihood.A.item():.4f}, lambda_0={likelihood.lambda0.item():.3f}")

    # Save checkpoint
    checkpoint_path = Path(__file__).parent / 'trained_arccosine_normalized_2d_checkpoint.pt'
    print(f"\nSaving checkpoint...")

    save_checkpoint(
        model, likelihood, inducing_points, train_x, train_y,
        filepath=checkpoint_path,
        n_train_x=n_train_x,
        n_train_y=n_train_y,
        train_x_range=train_x_range,
        train_y_range=train_y_range,
        n_inducing=n_inducing,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        seed=SEED
    )

    print(f"\n{'=' * 70}")
    print("Training complete!")
    print("NOTE: K_bar(x,x) = 1 for all x (constant prior variance)")
    print("ELBO comparison with unnormalized kernel is not meaningful")
    print(f"{'=' * 70}")

    print(f"\nNext step: Run utility_acos_normalized_2d_base.py to visualize utilities")


if __name__ == "__main__":
    main()
