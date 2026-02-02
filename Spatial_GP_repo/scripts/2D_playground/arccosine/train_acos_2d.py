"""
Train Arc-Cosine GP on 2D Grid - Created by Claude
===================================================
Training-only script that generates a checkpoint for utility_acos_2d_base.py.

This script:
1. Generates 5×5 training grid in [-5, 5]² (same as RBF for fair comparison)
2. Trains Arc-Cosine GP with Poisson likelihood (gpytorch_porting version)
3. Saves checkpoint with kernel params (sigma_0, Amp) and likelihood params (A, lambda0)

IMPORTANT:
- Uses PoissonLikelihood from gpytorch_porting (NOT from 1D playground!)
- Uses compute_elbo from gpytorch_porting/model.py (matches the likelihood)
- Arc-Cosine needs clamp_hyperparameters() after each optimizer step

Run this script once to create trained_arccosine_2d_checkpoint.pt, then use
utility_acos_2d_base.py to load and compare with RBF.
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
)

# From 1D playground (model class + data generation + compute_elbo)
from gp_utility_playground import (
    VariationalGP,
    generate_poisson_data,
    compute_elbo,  # Works with gpytorch_porting likelihood too
)

# From gpytorch_porting (Arc-Cosine specific - kernel, likelihood)
GPYTORCH_PORTING_PATH = Path(__file__).parent.parent.parent / 'gpytorch_porting'
sys.path.insert(0, str(GPYTORCH_PORTING_PATH))
from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood  # FULL version with A, lambda0


def train_arccosine_gp(model, likelihood, train_x, train_y, n_iterations=500, lr=0.1):
    """
    Train GP with Arc-Cosine kernel.

    CRITICAL: Must call model.covar_module.clamp_hyperparameters() after each step!
    """
    model.train()
    likelihood.train()

    # Combine model and likelihood parameters
    params = list(model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    with torch.enable_grad():
        for i in range(n_iterations):
            optimizer.zero_grad()
            elbo, ell, kl = compute_elbo(model, likelihood, train_x, train_y)
            (-elbo).backward()
            optimizer.step()

            # CRITICAL for Arc-Cosine: clamp hyperparameters to valid range
            model.covar_module.clamp_hyperparameters()

            if (i + 1) % 100 == 0:
                sigma_0 = model.covar_module.sigma_0.item()
                Amp = model.covar_module.Amp.item()
                A = likelihood.A.item()
                lambda0 = likelihood.lambda0.item()
                print(f"  Iter {i+1}/{n_iterations}, ELBO: {elbo.item():.2f} "
                      f"| σ₀={sigma_0:.4f}, Amp={Amp:.4f}, A={A:.4f}, λ₀={lambda0:.3f}")

    model.eval()
    likelihood.eval()
    return model, likelihood


def save_arccosine_2d_checkpoint(model, likelihood, inducing_points, train_x, train_y, filepath, **kwargs):
    """
    Save Arc-Cosine checkpoint with kernel and likelihood parameters.

    Key difference from RBF:
    - Stores sigma_0, Amp (Arc-Cosine kernel params)
    - Stores A, lambda_0 (Poisson likelihood params from gpytorch_porting)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'inducing_points': inducing_points,
        'train_x': train_x,
        'train_y': train_y,
        'kernel_params': {
            'sigma_0': model.covar_module.sigma_0.item(),
            'Amp': model.covar_module.Amp.item(),
        },
        'likelihood_params': {
            'A': likelihood.A.item(),
            'lambda_0': likelihood.lambda0.item(),  # Note: attribute is 'lambda0'
        },
        'training_config': kwargs
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Saved Arc-Cosine checkpoint to {filepath}")


def main():
    print("=" * 70)
    print("Train Arc-Cosine GP - 2D Playground")
    print("=" * 70)

    # Fixed seed for reproducibility (same as RBF)
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Training configuration (same as RBF for fair comparison)
    n_train_x = N_TRAIN_X_DEFAULT
    n_train_y = N_TRAIN_Y_DEFAULT
    train_x_range = TRAIN_X_RANGE_DEFAULT
    train_y_range = TRAIN_Y_RANGE_DEFAULT

    print(f"\nTraining Configuration:")
    print(f"  Grid size: {n_train_x} × {n_train_y} = {n_train_x * n_train_y} points")
    print(f"  X range: {train_x_range}")
    print(f"  Y range: {train_y_range}")

    # Generate training data (same as RBF)
    print(f"\nGenerating training data...")
    train_x = create_2d_grid(n_train_x, n_train_y,
                             x_range=train_x_range,
                             y_range=train_y_range)
    train_y = generate_poisson_data(train_x, lambda_true_2d)

    print(f"  Training points: {train_x.shape}")
    print(f"  Spike count range: [{train_y.min().item():.0f}, {train_y.max().item():.0f}]")

    # Use 15 random inducing points (subset of training) - same as current Arc-Cosine script
    n_inducing = 15
    indices = torch.randperm(len(train_x))[:n_inducing]
    inducing_points = train_x[indices].clone()

    # Initialize model with RBF kernel (default)
    model = VariationalGP(inducing_points, jitter=1e-4).to(DEVICE)

    # Swap to Arc-Cosine kernel
    model.covar_module = ArcCosineKernel(sigma_0=1.0, Amp=1.0, C=None).to(DEVICE)

    # Use gpytorch_porting PoissonLikelihood (with A, lambda0)
    likelihood = PoissonLikelihood(A_init=0.01, lambda0_init=1.0).to(DEVICE)

    print(f"\nModel initialized:")
    print(f"  Inducing points: {inducing_points.shape[0]} (random subset)")
    print(f"  Initial kernel: σ₀={model.covar_module.sigma_0.item():.3f}, Amp={model.covar_module.Amp.item():.3f}")
    print(f"  Initial likelihood: A={likelihood.A.item():.4f}, λ₀={likelihood.lambda0.item():.3f}")

    # Training settings
    n_iterations = 500
    learning_rate = 0.1

    print(f"\nTraining Arc-Cosine GP...")
    print(f"  Iterations: {n_iterations}")
    print(f"  Learning rate: {learning_rate}")

    model, likelihood = train_arccosine_gp(model, likelihood, train_x, train_y,
                                           n_iterations=n_iterations,
                                           lr=learning_rate)

    # Compute final ELBO
    print(f"\nComputing final ELBO...")
    with torch.no_grad():
        final_elbo, ell, kl = compute_elbo(model, likelihood, train_x, train_y)
    print(f"  Final ELBO: {final_elbo.item():.2f} (ELL: {ell.item():.2f}, KL: {kl.item():.2f})")

    # Print final hyperparameters
    print(f"\nFinal Hyperparameters:")
    print(f"  Kernel: σ₀={model.covar_module.sigma_0.item():.4f}, Amp={model.covar_module.Amp.item():.4f}")
    print(f"  Likelihood: A={likelihood.A.item():.4f}, λ₀={likelihood.lambda0.item():.3f}")

    # Save checkpoint
    checkpoint_path = Path(__file__).parent / 'trained_arccosine_2d_checkpoint.pt'
    print(f"\nSaving checkpoint...")

    save_arccosine_2d_checkpoint(
        model, likelihood, inducing_points, train_x, train_y,
        filepath=checkpoint_path,
        # Store training config
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
    print(f"{'=' * 70}")

    # Warning about expected behavior
    print("\n" + "⚠️ " * 10)
    print("NOTE: Arc-Cosine kernel is NON-STATIONARY (k(x,x) = ||x||² + σ₀²)")
    print("This causes poor fit on synthetic stationary bump data.")
    print("Negative or low ELBO is EXPECTED for this combination.")
    print("⚠️ " * 10)

    print(f"\nNext step: Run utility_acos_2d_base.py to compare with RBF")


if __name__ == "__main__":
    main()
