"""
Arc-Cosine Kernel Training Script - Created by Claude
======================================================
Trains an Arc-Cosine GP on the same data as the RBF model for fair comparison.

Training parameters match RBF exactly:
- seed=42, domain=[-2, 2], n_train=20, n_iterations=500

Saves checkpoint with:
- Full model state (hyperparameters + variational params m, V)
- Training data (train_x, train_y) for plotting
- Configuration metadata

Usage:
    python acosker_training.py

Output:
    trained_acos_checkpoint.pt
"""

import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gp_utility_playground import (
    DEVICE, DTYPE,
    lambda_true, generate_poisson_data,
    VariationalGP, PoissonLikelihood, compute_elbo
)

# Import ArcCosineKernel
GPYTORCH_PORTING_PATH = Path(__file__).parent.parent.parent / 'gpytorch_porting'
sys.path.insert(0, str(GPYTORCH_PORTING_PATH))
from kernels import ArcCosineKernel

# -----------------------------------------------------------------------------
# Training Configuration (SAME AS RBF)
# -----------------------------------------------------------------------------
SEED = 42
X_MIN, X_MAX = -2.0, 2.0  # Same domain as RBF
N_TRAIN = 20              # Same number of points
N_ITERATIONS = 500        # Same iterations
LR = 0.1                  # Same learning rate

# Arc-Cosine specific
ACOS_SIGMA_0_INIT = 1.0   # Initial σ₀
ACOS_AMP_INIT = 1.0       # Initial Amp (will be clamped during training)


def train_arccosine_gp(model, likelihood, train_x, train_y, n_iterations=500, lr=0.1):
    """Train GP with Arc-Cosine kernel.

    Same training procedure as RBF, but with hyperparameter clamping for Arc-Cosine.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training Arc-Cosine GP for {n_iterations} iterations...")
    print(f"Initial: σ₀={model.covar_module.sigma_0.item():.4f}, "
          f"Amp={model.covar_module.Amp.item():.4f}, "
          f"mean={model.mean_module.constant.item():.4f}")

    with torch.enable_grad():
        for i in range(n_iterations):
            optimizer.zero_grad()
            elbo, ell, kl = compute_elbo(model, likelihood, train_x, train_y)
            (-elbo).backward()
            optimizer.step()

            # Clamp Arc-Cosine hyperparameters after each step
            model.covar_module.clamp_hyperparameters()

            if (i + 1) % 100 == 0:
                sigma_0 = model.covar_module.sigma_0.item()
                Amp = model.covar_module.Amp.item()
                mean = model.mean_module.constant.item()
                print(f"  Iter {i+1}/{n_iterations}, ELBO: {elbo.item():.2f} "
                      f"(ELL: {ell.item():.2f}, KL: {kl.item():.2f}) | "
                      f"σ₀={sigma_0:.4f}, Amp={Amp:.4f}, mean={mean:.4f}")

    # Final parameters
    sigma_0 = model.covar_module.sigma_0.item()
    Amp = model.covar_module.Amp.item()
    mean = model.mean_module.constant.item()

    print(f"\nLearned hyperparameters:")
    print(f"  σ₀ = {sigma_0:.4f}")
    print(f"  Amp = {Amp:.4f}")
    print(f"  mean = {mean:.4f}")

    return model, likelihood


def main():
    # Reproducibility (same seed as RBF)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("Arc-Cosine Kernel Training")
    print("=" * 60)
    print(f"Config: seed={SEED}, domain=[{X_MIN}, {X_MAX}], "
          f"n_train={N_TRAIN}, iterations={N_ITERATIONS}")

    # Generate training data (SAME as RBF due to same seed)
    train_x = torch.linspace(X_MIN, X_MAX, N_TRAIN, dtype=DTYPE, device=DEVICE)
    train_y = generate_poisson_data(train_x, lambda_true)
    inducing_points = train_x.clone()

    print(f"\nTraining data:")
    print(f"  train_x: {N_TRAIN} points in [{train_x.min():.2f}, {train_x.max():.2f}]")
    print(f"  train_y: counts in [{train_y.min():.0f}, {train_y.max():.0f}]")

    # Create model with Arc-Cosine kernel
    model = VariationalGP(inducing_points, jitter=0).to(DEVICE)
    model.covar_module = ArcCosineKernel(
        sigma_0=ACOS_SIGMA_0_INIT,
        Amp=ACOS_AMP_INIT,
        C=None  # C=I for 1D
    )
    likelihood = PoissonLikelihood().to(DEVICE)

    # Check initial kernel condition
    with torch.no_grad():
        K = model.covar_module(train_x).evaluate()
        cond = torch.linalg.cond(K).item()
    print(f"\nInitial kernel condition: {cond:.2e}")

    # Train
    print()
    model, likelihood = train_arccosine_gp(
        model, likelihood, train_x, train_y,
        n_iterations=N_ITERATIONS, lr=LR
    )

    # Check final kernel condition
    with torch.no_grad():
        K = model.covar_module(train_x).evaluate()
        cond = torch.linalg.cond(K).item()
    print(f"\nFinal kernel condition: {cond:.2e}")

    # Create checkpoint with full metadata
    checkpoint = {
        # Model state (hyperparams + variational params)
        'model_state_dict': model.state_dict(),

        # Training configuration
        'config': {
            'seed': SEED,
            'n_train': N_TRAIN,
            'n_iterations': N_ITERATIONS,
            'lr': LR,
            'x_min': X_MIN,
            'x_max': X_MAX,
            'dtype': str(DTYPE),
            'ground_truth': 'lambda_true_asymmetric',
            'kernel': 'ArcCosineKernel',
        },

        # Inducing points (same as train_x for this case)
        'inducing_points': inducing_points.cpu(),

        # Training data (for plotting)
        'train_x': train_x.cpu(),
        'train_y': train_y.cpu(),

        # Learned hyperparameters (for quick reference)
        'hyperparameters': {
            'sigma_0': model.covar_module.sigma_0.item(),
            'Amp': model.covar_module.Amp.item(),
            'mean_constant': model.mean_module.constant.item(),
        },

        # Metadata
        'created': datetime.now().isoformat(),
        'description': 'Arc-Cosine GP trained on 1D asymmetric bump, Poisson likelihood',
        'condition_number': cond,
    }

    # Save checkpoint
    save_path = Path(__file__).parent / 'trained_acos_checkpoint.pt'
    torch.save(checkpoint, save_path)

    print(f"\n{'=' * 60}")
    print(f"Saved checkpoint to: {save_path}")
    print(f"{'=' * 60}")
    print(f"\nCheckpoint contents:")
    print(f"  config: {checkpoint['config']}")
    print(f"  hyperparameters: {checkpoint['hyperparameters']}")
    print(f"  inducing_points: {len(checkpoint['inducing_points'])} points")
    print(f"  train_x: {len(checkpoint['train_x'])} points")
    print(f"  train_y: {len(checkpoint['train_y'])} counts")
    print(f"  condition_number: {checkpoint['condition_number']:.2e}")


if __name__ == "__main__":
    main()
