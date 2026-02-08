"""
Train RBF GP on 2D Grid - Created by Claude
============================================
Training-only script that generates a checkpoint for utility_2d_base.py.

This script:
1. Generates 5x5 training grid in [-5, 5]^2
2. Trains RBF GP with Poisson likelihood
3. Saves checkpoint with full model state

Run this script once to create trained_rbf_2d_checkpoint.pt, then use
utility_2d_base.py to load and visualize.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "1D_playground"))

# Import from 2D playground (config + functions)
from utility_2d_rbf_base import (
    # Config
    N_TRAIN_X_DEFAULT, N_TRAIN_Y_DEFAULT,
    TRAIN_X_RANGE_DEFAULT, TRAIN_Y_RANGE_DEFAULT,
    DEVICE, DTYPE,
    # Functions
    lambda_true_2d,
    create_2d_grid,
    save_rbf_2d_checkpoint,
    # Likelihood from gpytorch_porting (re-exported)
    PoissonLikelihood,
)

# Import from 1D playground (core components)
from gp_utility_playground import (
    VariationalGP,
    train_gp,
    generate_poisson_data,
)


def main():
    print("=" * 70)
    print("Train RBF GP - 2D Playground")
    print("=" * 70)

    # Fixed seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Training configuration
    n_train_x = N_TRAIN_X_DEFAULT
    n_train_y = N_TRAIN_Y_DEFAULT
    train_x_range = TRAIN_X_RANGE_DEFAULT
    train_y_range = TRAIN_Y_RANGE_DEFAULT

    print(f"\nTraining Configuration:")
    print(f"  Grid size: {n_train_x} x {n_train_y} = {n_train_x * n_train_y} points")
    print(f"  X range: {train_x_range}")
    print(f"  Y range: {train_y_range}")

    # Generate training data
    print(f"\nGenerating training data...")
    train_x = create_2d_grid(n_train_x, n_train_y,
                             x_range=train_x_range,
                             y_range=train_y_range)
    train_y = generate_poisson_data(train_x, lambda_true_2d)

    print(f"  Training points: {train_x.shape}")
    print(f"  Spike count range: [{train_y.min().item():.0f}, {train_y.max().item():.0f}]")

    # Initialize model with inducing points at training locations
    inducing_points = train_x.clone()
    model = VariationalGP(inducing_points, jitter=1e-4).to(DEVICE)
    # RBF synthetic data: A=1, lambda0=0 (defaults match)
    likelihood = PoissonLikelihood(A_init=1.0, lambda0_init=0.0).to(DEVICE)

    print(f"\nModel initialized:")
    print(f"  Inducing points: {inducing_points.shape}")
    print(f"  Initial lengthscale: {model.covar_module.base_kernel.lengthscale.item():.3f}")
    print(f"  Initial outputscale: {model.covar_module.outputscale.item():.3f}")

    # Training settings
    n_iterations = 500
    learning_rate = 0.1

    print(f"\nTraining GP...")
    print(f"  Iterations: {n_iterations}")
    print(f"  Learning rate: {learning_rate}")

    # Note: train_gp's compute_elbo computes ELL inline with A=1, lambda0=0.
    # This is numerically identical to gpytorch_porting PoissonLikelihood with
    # A=1, lambda0=0 defaults. A/lambda0 are NOT used during training.
    model, likelihood = train_gp(model, likelihood, train_x, train_y,
                                 n_iterations=n_iterations,
                                 lr=learning_rate)

    # Print final hyperparameters
    print(f"\nFinal Hyperparameters:")
    print(f"  Lengthscale: {model.covar_module.base_kernel.lengthscale.item():.3f}")
    print(f"  Outputscale: {model.covar_module.outputscale.item():.3f}")

    # Save checkpoint
    checkpoint_path = Path(__file__).parent / 'trained_rbf_2d_checkpoint.pt'
    print(f"\nSaving checkpoint...")

    save_rbf_2d_checkpoint(
        model, likelihood, inducing_points, train_x, train_y,
        filepath=checkpoint_path,
        n_train_x=n_train_x,
        n_train_y=n_train_y,
        train_x_range=train_x_range,
        train_y_range=train_y_range,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        seed=SEED
    )

    print(f"\n{'=' * 70}")
    print("Training complete!")
    print(f"{'=' * 70}")
    print(f"\nNext step: Run utility_2d_base.py to load and visualize")


if __name__ == "__main__":
    main()
