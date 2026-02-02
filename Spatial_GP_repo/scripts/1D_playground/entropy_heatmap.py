"""
Entropy Heatmap - Created by Claude
====================================
Visualizes entropy H(R | mu, sigma2) as a function of latent GP parameters.

Creates a heatmap showing how Poisson response entropy varies with:
- mu (mean): latent log-firing rate parameter, range [-50, 50]
- sigma2 (variance): latent uncertainty, range [0, 100]

Uses the compute_H() function from gp_utility_playground.py which applies
Laplace approximation to compute the entropy of the marginal Poisson distribution.
"""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Import compute_H from gp_utility_playground.py
sys.path.insert(0, str(Path(__file__).parent))
from gp_utility_playground import compute_H, MAX_R, DEVICE, DTYPE

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MU_MIN, MU_MAX = -30.0, 50.0      # Range for latent mean (lambda)
SIGMA2_MIN, SIGMA2_MAX = 0.0, 10  # Range for latent variance
N_MU = 400                         # Grid resolution for mu
N_SIGMA2 = 400                     # Grid resolution for sigma2 (200×200 = 40k points)
R_MAX = MAX_R                      # Use default from gp_utility_playground

# Performance note: 200×200 grid with r_max=500 requires ~20M evaluations
# Takes ~3-5 seconds on GPU. For faster testing, reduce N_MU/N_SIGMA2 to 100.


def validate_sigma2(sigma2):
    """Validate that sigma2 (variance) is non-negative.

    Args:
        sigma2: Variance parameter(s) - can be scalar or tensor

    Raises:
        ValueError: If any sigma2 value is negative
    """
    if isinstance(sigma2, torch.Tensor):
        if torch.any(sigma2 < 0):
            raise ValueError(f"sigma2 must be non-negative, got min value: {sigma2.min().item()}")
    else:
        if sigma2 < 0:
            raise ValueError(f"sigma2 must be non-negative, got: {sigma2}")


def compute_entropy_grid(mu_range, sigma2_range, r_max=MAX_R):
    """Compute entropy H for all combinations of mu and sigma2.

    Args:
        mu_range: (n_mu,) tensor of mu values
        sigma2_range: (n_sigma2,) tensor of sigma2 values
        r_max: Maximum spike count for Laplace approximation

    Returns:
        H_grid: (n_mu, n_sigma2) array of entropy values
    """
    import time

    # Validate inputs
    validate_sigma2(sigma2_range)

    n_mu = len(mu_range)
    n_sigma2 = len(sigma2_range)

    # Create meshgrid
    mu_grid, sigma2_grid = torch.meshgrid(mu_range, sigma2_range, indexing='ij')

    # Flatten for batch computation
    mu_flat = mu_grid.flatten()
    sigma2_flat = sigma2_grid.flatten()

    print(f"Computing entropy for {len(mu_flat)} grid points...")
    print(f"  mu range: [{mu_range.min():.1f}, {mu_range.max():.1f}]")
    print(f"  sigma2 range: [{sigma2_range.min():.1f}, {sigma2_range.max():.1f}]")
    print(f"  r_max: {r_max}")
    print(f"  Total evaluations: {len(mu_flat)} × {r_max} = {len(mu_flat) * r_max:,}")

    # Compute entropy using compute_H (handles batches)
    start_time = time.time()
    H_flat = compute_H(mu_flat, sigma2_flat, r_max=r_max)
    elapsed = time.time() - start_time
    print(f"  Computation time: {elapsed:.2f}s")



 Two different PoissonLikelihood classes - 1D (simple) vs gpytorch_porting (full with A, lambda0). Arc-Cosine must use gpytorch_porting version.  



    # Reshape back to grid
    H_grid = H_flat.reshape(n_mu, n_sigma2)

    return H_grid


def plot_entropy_heatmap(mu_range, sigma2_range, H_grid, save_path=None):
    """Create heatmap visualization of entropy.

    Args:
        mu_range: (n_mu,) array of mu values
        sigma2_range: (n_sigma2,) array of sigma2 values
        H_grid: (n_mu, n_sigma2) array of entropy values
        save_path: Optional path to save figure
    """
    # Convert to numpy for plotting
    mu_np = mu_range.cpu().numpy()
    sigma2_np = sigma2_range.cpu().numpy()
    H_np = H_grid.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(H_np,
                   origin='lower',
                   aspect='auto',
                   extent=[sigma2_np.min(), sigma2_np.max(),
                          mu_np.min(), mu_np.max()],
                   cmap='viridis')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Entropy H(R | μ, σ²)', fontsize=12)

    # Labels and title
    ax.set_xlabel('Variance (σ²)', fontsize=12)
    ax.set_ylabel('Mean (μ)', fontsize=12)
    ax.set_title('Poisson Response Entropy as Function of GP Posterior Parameters calculated via Laplace Approximation',
                 fontsize=14, pad=15)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add statistics text box
    stats_text = f'Min H: {H_np.min():.3f}\nMax H: {H_np.max():.3f}\nMean H: {H_np.mean():.3f}'
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top',
            fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    plt.close(fig)


def main():
    """Main execution: compute and visualize entropy heatmap."""
    print("="*70)
    print("Entropy Heatmap Generator")
    print("="*70)

    # Create parameter ranges
    mu_range = torch.linspace(MU_MIN, MU_MAX, N_MU, dtype=DTYPE, device=DEVICE)
    sigma2_range = torch.linspace(SIGMA2_MIN, SIGMA2_MAX, N_SIGMA2, dtype=DTYPE, device=DEVICE)

    # Compute entropy grid
    H_grid = compute_entropy_grid(mu_range, sigma2_range, r_max=R_MAX)

    print(f"\nEntropy statistics:")
    print(f"  Min:  {H_grid.min().item():.4f}")
    print(f"  Max:  {H_grid.max().item():.4f}")
    print(f"  Mean: {H_grid.mean().item():.4f}")
    print(f"  Std:  {H_grid.std().item():.4f}")

    # Plot
    save_path = Path(__file__).parent / "entropy_heatmap_result.png"
    plot_entropy_heatmap(mu_range, sigma2_range, H_grid, save_path=save_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
