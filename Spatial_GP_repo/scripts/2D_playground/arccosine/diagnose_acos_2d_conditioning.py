"""
Diagnostic: Arc-Cosine 2D Posterior Before/After Conditioning Analysis
======================================================================

Created by Claude - Arc-Cosine version of diagnose_rbf_2d_conditioning.py

This script loads a pre-trained 2D Arc-Cosine GP and visualizes how conditioning
on observations (sampled from p(x)) affects predictions across the domain.

Produces 3 heatmaps showing:
1. Δμ = E[μ_cond] - μ_marg       (Mean shift after conditioning)
2. Δσ² = σ²_marg - E[σ²_cond]   (Variance reduction after conditioning)
3. Utility U = H_marg - E[H_cond] (Information gain)

NON-STATIONARITY WARNING:
-------------------------
Arc-Cosine kernel has k(x,x) = ||x||² + σ₀² (varies with input magnitude!):
- At (0,0): k ≈ σ₀²
- At (30,30): k ≈ 1800 + σ₀²

This causes EXTREME utility values at domain corners (far from origin).
**Utility is clipped to 99th percentile for visualization** - this does NOT
affect the actual computation, only the display.

The conditioning math (Gaussian conditioning) is kernel-agnostic and works
identically for Arc-Cosine and RBF kernels.

NUMERICAL INSTABILITY NOTE (same as RBF):
-----------------------------------------
In regions FAR from p(x), the conditioning effect is nearly zero. Due to MC
noise, near-zero values can flip slightly negative. This is expected behavior.

Related files:
- diagnose_rbf_2d_conditioning.py (RBF version in parent folder)
- utility_acos_2d_base.py (checkpoint loading pattern)
- train_acos_2d.py (checkpoint creation)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))  # 2D_playground
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "1D_playground"))

from utility_2d_rbf_base import (
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

from utility_acos_2d_base import load_arccosine_2d_checkpoint, ARCCOSINE_CHECKPOINT_PATH


# =============================================================================
# Configuration
# =============================================================================
N_EVAL_X = 40  # Resolution of evaluation grid
N_EVAL_Y = 40
N_MC_SAMPLES = 500  # Number of MC samples for averaging (higher = less noise)

# Checkpoint path
ARCCOSINE_CHECKPOINT_PATH = Path(__file__).parent / 'trained_arccosine_2d_checkpoint.pt'


# =============================================================================
# Visualization (with Arc-Cosine clipping)
# =============================================================================
def plot_conditioning_diagnostics_2d(
    eval_grid_x, eval_grid_y,
    mu_marg, mu_cond_avg,
    sigma2_marg, sigma2_cond_avg,
    H_marg, H_cond,
    train_x, p_x_mean, p_x_std,
    clip_percentile=99,
    save_path=None,
):
    """
    Create 2D diagnostic visualization: 3-panel layout showing conditioning effects.

    Layout:
        [Δμ (mean shift)] | [Δσ² (variance reduction)] | [Utility (clipped)]

    Arc-Cosine specific: Utility is clipped to clip_percentile to handle
    extreme values caused by kernel non-stationarity.
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

    # ARC-COSINE SPECIFIC: Clip utility to handle extreme values
    utility_clip_val = np.percentile(utility_2d, clip_percentile)
    utility_2d_clipped = np.clip(utility_2d, None, utility_clip_val)

    # Also clip variance reduction for cleaner visualization
    varsred_clip_val = np.percentile(delta_sigma2_2d, clip_percentile)
    delta_sigma2_2d_clipped = np.clip(delta_sigma2_2d, None, varsred_clip_val)

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

    # Add p(x) contours
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
    # Plot 2: Δσ² = σ²_marg - E[σ²_cond] (Variance reduction, clipped)
    # =========================================================================
    ax2 = axes[1]

    # Use clipped variance reduction
    im2 = ax2.contourf(x_np, y_np, delta_sigma2_2d_clipped, levels=20, cmap='Greens')

    # Add p(x) contours
    ax2.contour(x_np, y_np, p_x_2d, levels=[np.exp(-2), np.exp(-0.5)], colors='orange',
                linewidths=2, alpha=0.8, linestyles='--')

    # Training points
    ax2.scatter(train_x_np[:, 0], train_x_np[:, 1], c='black', s=40,
                edgecolors='white', linewidths=1.5, marker='o', zorder=5)

    # Mark maximum variance reduction (on clipped data)
    max_idx = np.argmax(delta_sigma2_2d_clipped)
    max_y, max_x = np.unravel_index(max_idx, delta_sigma2_2d_clipped.shape)
    ax2.scatter([x_np[max_y, max_x]], [y_np[max_y, max_x]], c='yellow',
                s=80, marker='o', edgecolors='black', linewidths=1.5, zorder=6,
                label=f'Max Δσ²')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'Δσ² = σ²_marg - E[σ²_cond]\n(Variance reduction, clipped to {clip_percentile}th pctl)')
    ax2.legend(loc='upper right', fontsize=8)
    plt.colorbar(im2, ax=ax2, label='Δσ²')

    # =========================================================================
    # Plot 3: Utility = H_marg - E[H_cond] (Information gain, clipped)
    # =========================================================================
    ax3 = axes[2]

    # Use clipped utility
    im3 = ax3.contourf(x_np, y_np, utility_2d_clipped, levels=20, cmap='Blues')

    # Add p(x) contours
    ax3.contour(x_np, y_np, p_x_2d, levels=[np.exp(-2), np.exp(-0.5)], colors='orange',
                linewidths=2, alpha=0.8, linestyles='--')

    # Training points
    ax3.scatter(train_x_np[:, 0], train_x_np[:, 1], c='black', s=40,
                edgecolors='white', linewidths=1.5, marker='o', zorder=5)

    # Mark maximum utility (on clipped data)
    max_idx_u = np.argmax(utility_2d_clipped)
    max_y_u, max_x_u = np.unravel_index(max_idx_u, utility_2d_clipped.shape)
    ax3.scatter([x_np[max_y_u, max_x_u]], [y_np[max_y_u, max_x_u]], c='yellow',
                s=80, marker='o', edgecolors='black', linewidths=1.5, zorder=6,
                label=f'Max Utility')

    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title(f'Utility U = H_marg - E[H_cond]\n(Information gain, clipped to {clip_percentile}th pctl)')
    ax3.legend(loc='upper right', fontsize=8)
    plt.colorbar(im3, ax=ax3, label='U (nats)')

    # Supertitle
    fig.suptitle('Arc-Cosine GP: Before vs After Conditioning Analysis\n'
                 f'(MC: {N_MC_SAMPLES} samples from p(x) ~ N({p_x_mean_np}, diag({p_x_std_np}²)))',
                 fontsize=12, y=1.02)

    plt.tight_layout()

    if save_path is None:
        save_path = Path(__file__).parent / 'diagnose_acos_2d_conditioning.png'
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
    print("Diagnostic: Arc-Cosine 2D Posterior Before/After Conditioning")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load pre-trained Arc-Cosine model
    # -------------------------------------------------------------------------
    print("\nLoading trained Arc-Cosine model...")
    model, likelihood, metadata = load_arccosine_2d_checkpoint(ARCCOSINE_CHECKPOINT_PATH)

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

    print(f"\nResults (raw, before clipping):")
    print(f"  Δμ range: [{delta_mu.min():.4f}, {delta_mu.max():.4f}]")
    print(f"  Δσ² range: [{delta_sigma2.min():.4f}, {delta_sigma2.max():.4f}]")
    print(f"  Utility range: [{utility.min():.4f}, {utility.max():.4f}]")

    # Print clipping info
    utility_np = utility.cpu().numpy()
    utility_99 = np.percentile(utility_np, 99)
    print(f"\n  (Utility 99th percentile: {utility_99:.4f} - values above this will be clipped)")

    # Sanity checks (only warn for significantly negative values, not numerical noise)
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
    print("\nGenerating visualization (with 99th percentile clipping)...")
    plot_conditioning_diagnostics_2d(
        eval_grid_x, eval_grid_y,
        mu_marg, mu_cond_avg,
        sigma2_marg, sigma2_cond_avg,
        H_marg, H_cond,
        train_x,
        DEFAULT_P_X_MEAN_2D, DEFAULT_P_X_STD_2D,
        clip_percentile=99
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
