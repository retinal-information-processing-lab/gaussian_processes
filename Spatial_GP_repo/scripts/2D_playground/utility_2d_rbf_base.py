"""
GP Utility Playground 2D - Created by Claude
=============================================
A simple 2D Gaussian Process playground with Poisson likelihood.

Extends the 1D playground to 2D inputs to test the hypothesis that
distribution-aware utility peaks at the center of p(x) and decays toward
borders, while standard utility saturates above zero.

This script reuses all core components from the 1D version:
- PoissonLikelihood, VariationalGP classes
- get_marginal_moments(), get_conditional_moments()
- compute_H(), evaluate_nd_utility_new()
- Training functions

Only modified: ground truth function, data generation, p(x) sampling, visualization

NOTE: Includes get_conditional_moments_nd() wrapper to handle multi-dimensional
inputs. The 1D version expects scalar x_sample, but 2D passes (2,) tensors.
The conditioning math is dimension-agnostic; only input wrapping differs.
"""

import sys
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------------------------------------------------
# Import universal constants from 1D playground (authoritative source)
# -----------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "1D_playground"))

from gp_utility_playground import (
    # Core classes
    PoissonLikelihood,
    VariationalGP,
    # Utility functions
    get_marginal_moments,
    get_conditional_moments,
    compute_H,
    evaluate_nd_utility_new,
    # Training functions
    train_gp,
    train_gp_fixed_lengthscale,
    # Data generation
    generate_poisson_data,
    # Universal constants
    DEVICE,
    DTYPE,
    MAX_R,
    X_MIN, X_MAX,
)

import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.*DtypeTensor.*")
warnings.filterwarnings("ignore", message=".*torch.sparse.SparseTensor.*")

# -----------------------------------------------------------------------------
# 2D-specific configuration (EXPORTED for other scripts)
# -----------------------------------------------------------------------------
# Domain bounds (reuse X bounds for Y to create square domain)
# Y_MIN, Y_MAX = X_MIN, X_MAX

Y_MIN, Y_MAX = -30, 30  # Use wider Y range for better visualization
X_MIN, X_MAX = -30, 30


# Default p(x) distribution parameters (2D Gaussian, diagonal covariance)
DEFAULT_P_X_MEAN_2D = torch.tensor([10.0, 10.0], dtype=DTYPE, device=DEVICE)
DEFAULT_P_X_STD_2D = torch.tensor([2.0, 2.0], dtype=DTYPE, device=DEVICE)

# Training configuration (used by train_rbf_2d.py)
N_TRAIN_X_DEFAULT = 5
N_TRAIN_Y_DEFAULT = 5
TRAIN_X_RANGE_DEFAULT = (-5.0, 5.0)
TRAIN_Y_RANGE_DEFAULT = (-5.0, 5.0)

# Evaluation configuration
N_EVAL_X_DEFAULT = 40
N_EVAL_Y_DEFAULT = 40
N_MC_SAMPLES_DEFAULT = 500

# Checkpoint path
RBF_2D_CHECKPOINT_PATH = Path(__file__).parent / 'trained_rbf_2d_checkpoint.pt'

# -----------------------------------------------------------------------------
# Dimension-Agnostic Wrapper for Conditional Moments
# -----------------------------------------------------------------------------
def get_conditional_moments_nd(model, x_star, x_sample, lambda_sample):
    """Dimension-agnostic wrapper for get_conditional_moments.

    Handles both 1D (scalar x_sample) and multi-D (tensor x_sample) inputs.
    The Gaussian conditioning math is the same for any dimension - only the
    input wrapping differs.

    Args:
        model: Trained GP model
        x_star: Query points (K, d) where d is input dimension
        x_sample: Single observation point (d,) tensor or scalar
        lambda_sample: Observed lambda value at x_sample (scalar)

    Returns:
        mu_cond: (K,) conditional posterior means
        sigma2_cond: (K,) conditional posterior variances
    """
    model.eval()

    # Handle both scalar (1D) and tensor (multi-D) x_sample
    if isinstance(x_sample, torch.Tensor):
        # x_sample is already a tensor (e.g., shape (2,) for 2D)
        x_sample_tensor = x_sample.unsqueeze(0)  # Shape: (1, d)
    else:
        # x_sample is a Python scalar (1D case)
        x_sample_tensor = torch.tensor([x_sample],
                                       dtype=x_star.dtype,
                                       device=x_star.device)  # Shape: (1,)

    with torch.no_grad():
        # Concatenate sample point with query points
        all_x = torch.cat([x_sample_tensor, x_star])

        # Get joint posterior
        posterior = model(all_x)
        full_covar = posterior.covariance_matrix

        # Extract components (same as 1D - works for any dimension!)
        mu_sample = posterior.mean[0]
        var_sample = full_covar[0, 0]
        mu_star = posterior.mean[1:]
        var_star = full_covar.diag()[1:]
        cross_cov = full_covar[0, 1:]

        # Gaussian conditioning formulas (dimension-agnostic)
        innovation = lambda_sample - mu_sample
        mu_cond = mu_star + cross_cov * (innovation / var_sample)
        sigma2_cond = var_star - (cross_cov ** 2) / var_sample
        sigma2_cond = torch.clamp(sigma2_cond, min=1e-8)

    return mu_cond, sigma2_cond


# -----------------------------------------------------------------------------
# Checkpoint Functions
# -----------------------------------------------------------------------------
def save_rbf_2d_checkpoint(model, likelihood, inducing_points, train_x, train_y, filepath, **kwargs):
    """
    Save RBF GP checkpoint with full state.

    Checkpoint format:
        - model_state_dict: Full model state
        - likelihood_state_dict: Likelihood state
        - inducing_points: Inducing point locations
        - train_x, train_y: Training data
        - hyperparameters: Lengthscale, outputscale, likelihood variance
        - training_config: Grid size, ranges, etc.

    Args:
        model: Trained VariationalGP model
        likelihood: Trained PoissonLikelihood
        inducing_points: (M, 2) inducing point tensor
        train_x: (N, 2) training inputs
        train_y: (N,) training observations
        filepath: Path to save checkpoint
        **kwargs: Additional config to store (n_train_x, n_train_y, ranges, etc.)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'inducing_points': inducing_points,
        'train_x': train_x,
        'train_y': train_y,
        'hyperparameters': {
            'outputscale': model.covar_module.outputscale.item(),
            'lengthscale': model.covar_module.base_kernel.lengthscale.item(),
            # Note: PoissonLikelihood has no variance parameter
        },
        'training_config': kwargs
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Saved checkpoint to {filepath}")


def load_rbf_2d_checkpoint(filepath):
    """
    Load RBF GP checkpoint, return model + likelihood + metadata.

    Returns:
        model: VariationalGP with loaded weights (in eval mode)
        likelihood: PoissonLikelihood with loaded weights (in eval mode)
        metadata: dict with:
            - inducing_points: (M, 2) tensor
            - train_x: (N, 2) training inputs
            - train_y: (N,) training observations
            - hyperparameters: dict with lengthscale, outputscale, variance
            - training_config: dict with grid size, ranges
    """
    checkpoint = torch.load(filepath, map_location=DEVICE)

    # Reconstruct model
    inducing_points = checkpoint['inducing_points']
    model = VariationalGP(inducing_points, jitter=1e-4).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Reconstruct likelihood
    likelihood = PoissonLikelihood().to(DEVICE)
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    likelihood.eval()

    # Extract metadata
    metadata = {
        'inducing_points': inducing_points,
        'train_x': checkpoint['train_x'],
        'train_y': checkpoint['train_y'],
        'hyperparameters': checkpoint['hyperparameters'],
        'training_config': checkpoint.get('training_config', {})
    }

    print(f"✓ Loaded checkpoint from {filepath}")
    print(f"  Hyperparameters:")
    print(f"    Lengthscale: {metadata['hyperparameters']['lengthscale']:.3f}")
    print(f"    Outputscale: {metadata['hyperparameters']['outputscale']:.3f}")

    return model, likelihood, metadata


# -----------------------------------------------------------------------------
# Ground Truth Function (2D)
# -----------------------------------------------------------------------------
def lambda_true_2d(x):
    """2D latent function - asymmetric bump.

    Args:
        x: Input points, shape (N, 2) or (N_x, N_y, 2)

    Returns:
        lambda values, same shape as x[..., 0]
    """
    x_coord = x[..., 0]
    y_coord = x[..., 1]

    # Asymmetric 2D bump centered at (0.0, 0.0)
    # Wider to be captured by coarse 5x5 training grid (spacing ~2.5)
    peak_x, peak_y = 0.0, 0.0
    sigma_x = 2.0  # Moderate x spread
    sigma_y = 3.5  # Wider y spread (asymmetric)
    amplitude = 3.0
    baseline = 0.5

    r_squared = ((x_coord - peak_x) / sigma_x) ** 2 + ((y_coord - peak_y) / sigma_y) ** 2
    lam = baseline + (amplitude - baseline) * torch.exp(-0.5 * r_squared)

    return lam


# -----------------------------------------------------------------------------
# 2D Data Generation
# -----------------------------------------------------------------------------
def create_2d_grid(n_x, n_y, x_range=(-2, 2), y_range=(-2, 2)):
    """Create 2D grid of points.

    Args:
        n_x: Number of points in x direction
        n_y: Number of points in y direction
        x_range: (min, max) for x
        y_range: (min, max) for y

    Returns:
        grid_points: (n_x * n_y, 2) tensor of points
    """
    x = torch.linspace(x_range[0], x_range[1], n_x, dtype=DTYPE, device=DEVICE)
    y = torch.linspace(y_range[0], y_range[1], n_y, dtype=DTYPE, device=DEVICE)

    # Create meshgrid and stack into (N, 2) format
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    return grid_points


# -----------------------------------------------------------------------------
# Distribution-Aware Utility (2D version)
# -----------------------------------------------------------------------------
def evaluate_distribution_aware_utility_2d(model, x_candidates, n_mc_samples=500,
                                           p_x_mean=None, p_x_std=None, r_max=MAX_R):
    """Evaluate distribution-aware utility at 2D candidate points.

    U(x*) = H_marg(x*) - E[H_cond(x* | x, λ)]

    where expectation is over x ~ N(p_x_mean, diag(p_x_std²)), λ ~ q(λ|x).

    Args:
        model: Trained GP model
        x_candidates: (n_candidates, 2) query points
        n_mc_samples: Number of MC samples for expectation
        p_x_mean: (2,) mean of 2D Gaussian p(x) distribution
        p_x_std: (2,) std of 2D Gaussian p(x) distribution (diagonal covariance)
        r_max: Max spike count for entropy computation

    Returns:
        utility: (n_candidates,) utility at each candidate
    """
    model.eval()
    device = x_candidates.device
    dtype = x_candidates.dtype

    if p_x_mean is None:
        p_x_mean = DEFAULT_P_X_MEAN_2D
    if p_x_std is None:
        p_x_std = DEFAULT_P_X_STD_2D

    # Marginal entropy
    mu_marg, sigma2_marg = get_marginal_moments(model, x_candidates)
    H_marg = compute_H(mu_marg, sigma2_marg, r_max=r_max)

    # Average conditional entropy over MC samples
    H_cond_sum = torch.zeros_like(H_marg)

    with torch.no_grad():
        for i in range(n_mc_samples):
            # Sample x from 2D Gaussian p(x) with diagonal covariance
            x_i = p_x_mean + p_x_std * torch.randn(2, dtype=dtype, device=device)
            x_i_tensor = x_i.unsqueeze(0)  # (1, 2)

            # Get GP posterior at x_i
            post_i = model(x_i_tensor)
            mu_i = post_i.mean[0]
            std_i = post_i.variance[0].sqrt()

            # Sample lambda from GP posterior at x_i
            lambda_i = (mu_i + std_i * torch.randn(1, dtype=dtype, device=device)).item()

            # Conditional entropy at all candidate points
            mu_cond_i, sigma2_cond_i = get_conditional_moments_nd(
                model, x_candidates, x_i, lambda_i
            )
            H_cond_i = compute_H(mu_cond_i, sigma2_cond_i, r_max=r_max)
            H_cond_sum += H_cond_i

            if (i + 1) % 100 == 0:
                print(f"  MC sample {i+1}/{n_mc_samples}")

    H_cond = H_cond_sum / n_mc_samples
    utility = H_marg - H_cond

    return utility


# -----------------------------------------------------------------------------
# 2D Visualization
# -----------------------------------------------------------------------------
def plot_results_2d(model, train_x, train_y, lambda_fn,
                    eval_grid_x, eval_grid_y,
                    utility_standard, utility_distr_aware,
                    p_x_mean=None, p_x_std=None, save_path=None):
    """Create 2D visualization of GP fit and utility landscape.

    Args:
        model: Trained GP model
        train_x: Training inputs (N_train, 2)
        train_y: Training outputs (N_train,)
        lambda_fn: Ground truth function
        eval_grid_x: (n_x, n_y) meshgrid of x coordinates
        eval_grid_y: (n_x, n_y) meshgrid of y coordinates
        utility_standard: (n_x * n_y,) standard utility values
        utility_distr_aware: (n_x * n_y,) distribution-aware utility values
        p_x_mean: (2,) mean of p(x) distribution
        p_x_std: (2,) std of p(x) distribution
        save_path: Optional path to save figure
    """
    model.eval()
    n_x, n_y = eval_grid_x.shape

    # Flatten grid for GP evaluation
    grid_flat = torch.stack([eval_grid_x.flatten(), eval_grid_y.flatten()], dim=-1)

    with torch.no_grad():
        posterior = model(grid_flat)
        mean = posterior.mean
        std = posterior.variance.sqrt()
        true_lambda = lambda_fn(grid_flat)

    # Reshape for plotting
    mean_2d = mean.reshape(n_x, n_y).cpu().numpy()
    std_2d = std.reshape(n_x, n_y).cpu().numpy()
    true_2d = true_lambda.reshape(n_x, n_y).cpu().numpy()
    utility_std_2d = utility_standard.reshape(n_x, n_y).cpu().numpy()
    utility_da_2d = utility_distr_aware.reshape(n_x, n_y).cpu().numpy()

    x_np = eval_grid_x.cpu().numpy()
    y_np = eval_grid_y.cpu().numpy()
    train_x_np = train_x.cpu().numpy()

    # Clip utilities for visualization
    # DA utility is well-behaved; standard utility can overflow for non-stationary kernels
    def get_clip_value(arr, pctl=99):
        finite_vals = arr[np.isfinite(arr)]
        if len(finite_vals) == 0:
            return 1.0
        return np.percentile(finite_vals, pctl)

    da_clip = get_clip_value(utility_da_2d, 99)
    # For standard utility: use max of (its own 99th pctl, 5x DA clip) but cap at 5x DA
    # This handles non-stationary kernels where std utility explodes
    std_clip_raw = get_clip_value(utility_std_2d, 99)
    std_clip = min(std_clip_raw, da_clip * 5) if da_clip > 0 else std_clip_raw

    def clip_array(arr, clip_val):
        result = np.clip(arr, None, clip_val)
        result[~np.isfinite(result)] = clip_val
        return result

    utility_std_clipped = clip_array(utility_std_2d, std_clip)
    utility_da_clipped = clip_array(utility_da_2d, da_clip)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: True λ, GP mean, GP std
    ax1 = axes[0, 0]
    im1 = ax1.contourf(x_np, y_np, true_2d, levels=20, cmap='viridis')
    ax1.scatter(train_x_np[:, 0], train_x_np[:, 1], c='red', s=30,
                edgecolors='white', linewidths=1, marker='o', label='Training')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('True λ(x,y)')
    ax1.legend(loc='upper right', fontsize=8)
    plt.colorbar(im1, ax=ax1)

    ax2 = axes[0, 1]
    im2 = ax2.contourf(x_np, y_np, mean_2d, levels=20, cmap='viridis')
    ax2.scatter(train_x_np[:, 0], train_x_np[:, 1], c='red', s=30,
                edgecolors='white', linewidths=1, marker='o')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('GP Mean')
    plt.colorbar(im2, ax=ax2)

    ax3 = axes[0, 2]
    im3 = ax3.contourf(x_np, y_np, std_2d, levels=20, cmap='Reds')
    ax3.scatter(train_x_np[:, 0], train_x_np[:, 1], c='blue', s=30,
                edgecolors='white', linewidths=1, marker='o')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('GP Std')
    plt.colorbar(im3, ax=ax3)

    # Row 2: Standard utility, Distribution-aware utility, Difference
    ax4 = axes[1, 0]
    im4 = ax4.contourf(x_np, y_np, utility_std_clipped, levels=20, cmap='Greens')
    ax4.scatter(train_x_np[:, 0], train_x_np[:, 1], c='red', s=30, marker='x')
    max_idx = np.argmax(utility_std_clipped)
    max_y, max_x = np.unravel_index(max_idx, utility_std_clipped.shape)
    ax4.scatter([x_np[max_y, max_x]], [y_np[max_y, max_x]], c='yellow',
                s=80, marker='o', edgecolors='black', linewidths=1.5, zorder=10, label='Max')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Standard Utility (clipped)')
    ax4.legend(loc='upper right', fontsize=8)
    plt.colorbar(im4, ax=ax4)

    ax5 = axes[1, 1]
    im5 = ax5.contourf(x_np, y_np, utility_da_clipped, levels=20, cmap='Blues')
    ax5.scatter(train_x_np[:, 0], train_x_np[:, 1], c='red', s=30, marker='x')

    # p(x) contours (1σ and 2σ)
    if p_x_mean is not None and p_x_std is not None:
        p_x_mean_np = p_x_mean.cpu().numpy()
        p_x_std_np = p_x_std.cpu().numpy()
        dx = x_np - p_x_mean_np[0]
        dy = y_np - p_x_mean_np[1]
        p_x_2d = np.exp(-0.5 * ((dx / p_x_std_np[0])**2 + (dy / p_x_std_np[1])**2))
        ax5.contour(x_np, y_np, p_x_2d, levels=[np.exp(-2), np.exp(-0.5)],
                    colors='orange', linewidths=2, linestyles='--')

    max_idx_da = np.argmax(utility_da_clipped)
    max_y_da, max_x_da = np.unravel_index(max_idx_da, utility_da_clipped.shape)
    ax5.scatter([x_np[max_y_da, max_x_da]], [y_np[max_y_da, max_x_da]],
                c='yellow', s=80, marker='o', edgecolors='black', linewidths=1.5, zorder=10, label='Max')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title('Distr-Aware Utility\n(orange = p(x) 1σ, 2σ)')
    ax5.legend(loc='upper right', fontsize=8)
    plt.colorbar(im5, ax=ax5)

    ax6 = axes[1, 2]
    diff_2d = utility_da_clipped - utility_std_clipped
    diff_max = max(np.abs(diff_2d).max(), 1e-6)
    im6 = ax6.contourf(x_np, y_np, diff_2d, levels=20, cmap='RdBu_r',
                       vmin=-diff_max, vmax=diff_max)
    ax6.scatter(train_x_np[:, 0], train_x_np[:, 1], c='black', s=30, marker='x')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title('Difference (DA - Std)')
    plt.colorbar(im6, ax=ax6)

    plt.tight_layout()

    if save_path is None:
        save_path = Path(__file__).parent / 'utility_2d_rbf_base.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {save_path}")

    return fig


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Minimal randomness - fixed seed
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print("GP Utility Playground 2D")
    print("=" * 70)

    # Load checkpoint (NO TRAINING)
    print(f"\nLoading checkpoint...")
    model, likelihood, metadata = load_rbf_2d_checkpoint(RBF_2D_CHECKPOINT_PATH)

    # Extract training data from checkpoint
    train_x = metadata['train_x']
    train_y = metadata['train_y']

    print(f"\nTraining data from checkpoint:")
    print(f"  Training points: {train_x.shape}")
    print(f"  Spike count range: [{train_y.min().item():.0f}, {train_y.max().item():.0f}]")

    # Create evaluation grid (coarser for speed)
    n_eval_x, n_eval_y = 40, 40
    print(f"\nCreating {n_eval_x}×{n_eval_y} evaluation grid...")

    x_eval = torch.linspace(X_MIN, X_MAX, n_eval_x, dtype=DTYPE, device=DEVICE)
    y_eval = torch.linspace(Y_MIN, Y_MAX, n_eval_y, dtype=DTYPE, device=DEVICE)
    eval_grid_x, eval_grid_y = torch.meshgrid(x_eval, y_eval, indexing='xy')
    eval_points = torch.stack([eval_grid_x.flatten(), eval_grid_y.flatten()], dim=-1)

    print(f"Evaluation points: {eval_points.shape}")

    # Evaluate standard utility
    print("\nEvaluating standard utility...")
    utility_standard = evaluate_nd_utility_new(model, eval_points)

    # Evaluate distribution-aware utility
    print("\nEvaluating distribution-aware utility (this may take a minute)...")
    utility_distr_aware = evaluate_distribution_aware_utility_2d(
        model, eval_points,
        n_mc_samples=500,  # Reduced from 1000 for speed
        p_x_mean=DEFAULT_P_X_MEAN_2D,
        p_x_std=DEFAULT_P_X_STD_2D
    )

    # Find max utility locations
    max_idx_std = torch.argmax(utility_standard)
    max_idx_da = torch.argmax(utility_distr_aware)

    max_std_loc = eval_points[max_idx_std].cpu().numpy()
    max_da_loc = eval_points[max_idx_da].cpu().numpy()

    print(f"\nStandard utility max at: ({max_std_loc[0]:.3f}, {max_std_loc[1]:.3f})")
    print(f"Distribution-aware utility max at: ({max_da_loc[0]:.3f}, {max_da_loc[1]:.3f})")
    print(f"p(x) center: ({DEFAULT_P_X_MEAN_2D[0]:.3f}, {DEFAULT_P_X_MEAN_2D[1]:.3f})")

    # Distance from p(x) center
    dist_std = np.sqrt(np.sum((max_std_loc - DEFAULT_P_X_MEAN_2D.cpu().numpy())**2))
    dist_da = np.sqrt(np.sum((max_da_loc - DEFAULT_P_X_MEAN_2D.cpu().numpy())**2))
    print(f"\nDistance from p(x) center:")
    print(f"  Standard utility: {dist_std:.3f}")
    print(f"  Distribution-aware utility: {dist_da:.3f}")

    # Plot results
    print("\nGenerating visualization...")
    plot_results_2d(
        model, train_x, train_y, lambda_true_2d,
        eval_grid_x, eval_grid_y,
        utility_standard, utility_distr_aware,
        p_x_mean=DEFAULT_P_X_MEAN_2D,
        p_x_std=DEFAULT_P_X_STD_2D
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
