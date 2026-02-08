"""
Diagnostic: RBF 2D Posterior Before/After Conditioning Analysis
===============================================================

Created by Claude - RBF version of 2D conditioning diagnostic.
See arccosine/arccosine_conditioning_2d.py for Arc-Cosine version.

This script loads a pre-trained 2D RBF GP and visualizes how conditioning on
observations (sampled from p(x)) affects predictions across the domain.

Produces 3 heatmaps showing:
1. Δμ = E[μ_cond] - μ_marg       (Mean shift after conditioning)
2. Δσ² = σ²_marg - E[σ²_cond]   (Variance reduction after conditioning)
3. Utility U = H_marg - E[H_cond] (Information gain)

The key insight: In 2D we can't superimpose curves like in 1D, so we show
the DIFFERENCES directly as heatmaps.

NUMERICAL INSTABILITY NOTE:
---------------------------
In regions FAR from p(x), the conditioning effect is nearly zero because:
- Samples from p(x) are distant from these query points
- The GP correlation decays with distance (RBF kernel)

This means Δσ² ≈ 0 and Utility ≈ 0 in these regions. Due to MC noise, these
near-zero values can flip slightly negative. This is expected behavior, not a bug.

The effect is most pronounced when:
- p(x) is far from the training data (high GP uncertainty at p(x))
- The evaluation domain is much larger than p(x)'s support
- N_MC_SAMPLES is low (more variance in the MC estimate)

Increasing N_MC_SAMPLES reduces the noise but doesn't eliminate it entirely.
The script uses a threshold (eps=1e-3) to avoid flagging these small negatives.

Related files:
- 1D_playground/1d_utility_w_fixed_ntrain.py (1D version)
- utility_2d_base.py (config + checkpoint loading)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "1D_playground"))

from utility_2d_rbf_base import (
    # Checkpoint loading
    load_rbf_2d_checkpoint,
    RBF_2D_CHECKPOINT_PATH,
    # Functions (from gpytorch_porting, re-exported)
    get_gp_marginal_moments,
    get_gp_conditional_moments,
    compute_H,
    compute_mc_diagnostics_2d,
    lambda_true_2d,
    # Config
    DEVICE, DTYPE,
    X_MIN, X_MAX, Y_MIN, Y_MAX,
    DEFAULT_P_X_MEAN_2D, DEFAULT_P_X_STD_2D,
)


# =============================================================================
# Configuration
# =============================================================================
N_EVAL_X = 40  # Resolution of evaluation grid
N_EVAL_Y = 40
N_MC_SAMPLES = 500  # Number of MC samples for averaging (higher = less noise)


# =============================================================================
# Visualization
# =============================================================================
def plot_conditioning_diagnostics_2d(
    eval_grid_x, eval_grid_y,
    mu_marg, mu_cond_avg,
    sigma2_marg, sigma2_cond_avg,
    H_marg, H_cond,
    train_x, p_x_mean, p_x_std,
    save_path=None,
):
    """
    Create 2D diagnostic visualization: 3-panel layout showing conditioning effects.

    Layout:
        [Δμ (mean shift)] | [Δσ² (variance reduction)] | [Utility]

    Args:
        eval_grid_x, eval_grid_y: (n_x, n_y) meshgrids
        mu_marg: (K,) marginal means
        mu_cond_avg: (K,) MC-averaged conditional means
        sigma2_marg: (K,) marginal variances
        sigma2_cond_avg: (K,) MC-averaged conditional variances
        H_marg: (K,) marginal entropies
        H_cond: (K,) MC-averaged conditional entropies
        train_x: (N, 2) training points
        p_x_mean: (2,) mean of p(x)
        p_x_std: (2,) std of p(x)
        save_path: Where to save figure
    """
    n_x, n_y = eval_grid_x.shape

    # Compute differences
    delta_mu = mu_cond_avg - mu_marg  # Mean shift (can be positive or negative)
    delta_sigma2 = sigma2_marg - sigma2_cond_avg  # Variance reduction (should be >= 0)
    utility = H_marg - H_cond  # Information gain (should be >= 0)

    # Reshape for plotting
    delta_mu_2d = delta_mu.reshape(n_x, n_y).cpu().numpy()
    delta_sigma2_2d = delta_sigma2.reshape(n_x, n_y).cpu().numpy()
    utility_2d = utility.reshape(n_x, n_y).cpu().numpy()

    # Convert grids to numpy
    x_np = eval_grid_x.cpu().numpy()
    y_np = eval_grid_y.cpu().numpy()
    train_x_np = train_x.cpu().numpy()
    p_x_mean_np = p_x_mean.cpu().numpy()
    p_x_std_np = p_x_std.cpu().numpy()

    # Compute p(x) contours
    dx = x_np - p_x_mean_np[0]
    dy = y_np - p_x_mean_np[1]
    p_x_2d = np.exp(-0.5 * ((dx / p_x_std_np[0])**2 + (dy / p_x_std_np[1])**2))

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # =========================================================================
    # Plot 1: Δμ = E[μ_cond] - μ_marg (Mean shift after conditioning)
    # =========================================================================
    ax1 = axes[0]

    # Symmetric colormap centered at 0
    vmax_mu = max(abs(delta_mu_2d.min()), abs(delta_mu_2d.max()))
    im1 = ax1.contourf(x_np, y_np, delta_mu_2d, levels=20, cmap='RdBu_r',
                       vmin=-vmax_mu, vmax=vmax_mu)

    # Add p(x) contours - only 1σ and 2σ
    ax1.contour(x_np, y_np, p_x_2d, levels=[np.exp(-2), np.exp(-0.5)], colors='orange',
                linewidths=2, alpha=0.8, linestyles='--')

    # Training points
    ax1.scatter(train_x_np[:, 0], train_x_np[:, 1], c='black', s=40,
                edgecolors='white', linewidths=1.5, marker='o', zorder=5)

    # p(x) center
    ax1.scatter([p_x_mean_np[0]], [p_x_mean_np[1]], c='orange', s=150,
                marker='o', edgecolors='black', linewidths=2, zorder=6,
                label=f'p(x) center')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Δμ = E[μ_cond] - μ_marg\n(Mean shift after conditioning)')
    ax1.legend(loc='upper right', fontsize=8)
    plt.colorbar(im1, ax=ax1, label='Δμ')

    # =========================================================================
    # Plot 2: Δσ² = σ²_marg - E[σ²_cond] (Variance reduction)
    # =========================================================================
    ax2 = axes[1]

    # Variance reduction should be >= 0 (use single-sided colormap)
    im2 = ax2.contourf(x_np, y_np, delta_sigma2_2d, levels=20, cmap='Greens')

    # Add p(x) contours - only 1σ and 2σ
    ax2.contour(x_np, y_np, p_x_2d, levels=[np.exp(-2), np.exp(-0.5)], colors='orange',
                linewidths=2, alpha=0.8, linestyles='--')

    # Training points
    ax2.scatter(train_x_np[:, 0], train_x_np[:, 1], c='black', s=40,
                edgecolors='white', linewidths=1.5, marker='o', zorder=5)

    # Mark maximum variance reduction with contrasting dot
    max_idx = np.argmax(delta_sigma2_2d)
    max_y, max_x = np.unravel_index(max_idx, delta_sigma2_2d.shape)
    ax2.scatter([x_np[max_y, max_x]], [y_np[max_y, max_x]], c='yellow',
                s=80, marker='o', edgecolors='black', linewidths=1.5, zorder=6,
                label='Max Δσ²')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Δσ² = σ²_marg - E[σ²_cond]\n(Variance reduction after conditioning)')
    ax2.legend(loc='upper right', fontsize=8)
    plt.colorbar(im2, ax=ax2, label='Δσ²')

    # =========================================================================
    # Plot 3: Utility = H_marg - E[H_cond] (Information gain)
    # =========================================================================
    ax3 = axes[2]

    # Utility should be >= 0 (use single-sided colormap)
    im3 = ax3.contourf(x_np, y_np, utility_2d, levels=20, cmap='Blues')

    # Add p(x) contours - only 1σ and 2σ
    ax3.contour(x_np, y_np, p_x_2d, levels=[np.exp(-2), np.exp(-0.5)], colors='orange',
                linewidths=2, alpha=0.8, linestyles='--')

    # Training points
    ax3.scatter(train_x_np[:, 0], train_x_np[:, 1], c='black', s=40,
                edgecolors='white', linewidths=1.5, marker='o', zorder=5)

    # Mark maximum utility with contrasting dot
    max_idx_u = np.argmax(utility_2d)
    max_y_u, max_x_u = np.unravel_index(max_idx_u, utility_2d.shape)
    ax3.scatter([x_np[max_y_u, max_x_u]], [y_np[max_y_u, max_x_u]], c='yellow',
                s=80, marker='o', edgecolors='black', linewidths=1.5, zorder=6,
                label='Max Utility')

    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Utility U = H_marg - E[H_cond]\n(Information gain)')
    ax3.legend(loc='upper right', fontsize=8)
    plt.colorbar(im3, ax=ax3, label='U (nats)')

    # Supertitle
    fig.suptitle('2D GP: Before vs After Conditioning Analysis\n'
                 f'(MC: {N_MC_SAMPLES} samples from p(x) ~ N({p_x_mean_np}, diag({p_x_std_np}²)))',
                 fontsize=12, y=1.02)

    plt.tight_layout()

    if save_path is None:
        save_path = Path(__file__).parent / 'rbf_2d_conditioning.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {save_path}")

    return fig


# =============================================================================
# Main
# =============================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("Diagnostic: 2D Posterior Before/After Conditioning")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load pre-trained RBF model
    # -------------------------------------------------------------------------
    print("\nLoading trained RBF model...")
    model, likelihood, metadata = load_rbf_2d_checkpoint(RBF_2D_CHECKPOINT_PATH)

    train_x = metadata['train_x']
    train_y = metadata['train_y']

    print(f"\nConfiguration:")
    print(f"  Evaluation grid: {N_EVAL_X} × {N_EVAL_Y}")
    print(f"  MC samples: {N_MC_SAMPLES}")
    print(f"  p(x) mean: [{DEFAULT_P_X_MEAN_2D[0]:.1f}, {DEFAULT_P_X_MEAN_2D[1]:.1f}]")
    print(f"  p(x) std: [{DEFAULT_P_X_STD_2D[0]:.1f}, {DEFAULT_P_X_STD_2D[1]:.1f}]")

    # -------------------------------------------------------------------------
    # Create evaluation grid
    # -------------------------------------------------------------------------
    print(f"\nCreating evaluation grid...")
    x_eval = torch.linspace(X_MIN, X_MAX, N_EVAL_X, dtype=DTYPE, device=DEVICE)
    y_eval = torch.linspace(Y_MIN, Y_MAX, N_EVAL_Y, dtype=DTYPE, device=DEVICE)
    eval_grid_x, eval_grid_y = torch.meshgrid(x_eval, y_eval, indexing='xy')
    eval_points = torch.stack([eval_grid_x.flatten(), eval_grid_y.flatten()], dim=-1)

    print(f"  Grid shape: {eval_grid_x.shape}")
    print(f"  Total query points: {eval_points.shape[0]}")

    # -------------------------------------------------------------------------
    # Compute MC diagnostics
    # -------------------------------------------------------------------------
    print(f"\nComputing MC diagnostics...")
    H_marg, H_cond, mu_marg, mu_cond_avg, sigma2_marg, sigma2_cond_avg = compute_mc_diagnostics_2d(
        model, eval_points, N_MC_SAMPLES,
        DEFAULT_P_X_MEAN_2D, DEFAULT_P_X_STD_2D
    )

    # Print summary statistics
    delta_mu = mu_cond_avg - mu_marg
    delta_sigma2 = sigma2_marg - sigma2_cond_avg
    utility = H_marg - H_cond

    print(f"\nResults:")
    print(f"  Δμ range: [{delta_mu.min():.4f}, {delta_mu.max():.4f}]")
    print(f"  Δσ² range: [{delta_sigma2.min():.4f}, {delta_sigma2.max():.4f}]")
    print(f"  Utility range: [{utility.min():.4f}, {utility.max():.4f}]")

    # Sanity checks (only warn for significantly negative values, not numerical noise)
    # Small negatives near 0 are expected due to MC noise in low-effect regions
    eps = 1e-3
    if (delta_sigma2 < -eps).any():
        neg_pct = (delta_sigma2 < -eps).float().mean() * 100
        neg_min = delta_sigma2.min().item()
        print(f"  ⚠ Warning: {neg_pct:.1f}% of points have Δσ² < -{eps} (min={neg_min:.4f})")

    if (utility < -eps).any():
        neg_pct = (utility < -eps).float().mean() * 100
        neg_min = utility.min().item()
        print(f"  ⚠ Warning: {neg_pct:.1f}% of points have utility < -{eps} (min={neg_min:.4f})")

    # -------------------------------------------------------------------------
    # Find key locations
    # -------------------------------------------------------------------------
    max_utility_idx = torch.argmax(utility)
    max_varsred_idx = torch.argmax(delta_sigma2)

    max_utility_loc = eval_points[max_utility_idx].cpu().numpy()
    max_varsred_loc = eval_points[max_varsred_idx].cpu().numpy()
    p_x_center = DEFAULT_P_X_MEAN_2D.cpu().numpy()

    print(f"\nKey locations:")
    print(f"  Max utility at: ({max_utility_loc[0]:.2f}, {max_utility_loc[1]:.2f})")
    print(f"  Max Δσ² at: ({max_varsred_loc[0]:.2f}, {max_varsred_loc[1]:.2f})")
    print(f"  p(x) center: ({p_x_center[0]:.2f}, {p_x_center[1]:.2f})")

    # Distances from p(x) center
    dist_utility = np.sqrt(np.sum((max_utility_loc - p_x_center)**2))
    dist_varsred = np.sqrt(np.sum((max_varsred_loc - p_x_center)**2))
    print(f"\nDistances from p(x) center:")
    print(f"  Max utility: {dist_utility:.2f}")
    print(f"  Max Δσ²: {dist_varsred:.2f}")

    # -------------------------------------------------------------------------
    # Create visualization
    # -------------------------------------------------------------------------
    print("\nGenerating visualization...")
    plot_conditioning_diagnostics_2d(
        eval_grid_x, eval_grid_y,
        mu_marg, mu_cond_avg,
        sigma2_marg, sigma2_cond_avg,
        H_marg, H_cond,
        train_x,
        DEFAULT_P_X_MEAN_2D, DEFAULT_P_X_STD_2D
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
