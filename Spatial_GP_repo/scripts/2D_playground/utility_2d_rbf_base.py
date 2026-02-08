"""
GP Utility Playground 2D
========================
A simple 2D Gaussian Process playground with Poisson likelihood.

Extends the 1D playground to 2D inputs to test the hypothesis that
distribution-aware utility peaks at the center of p(x) and decays toward
borders, while standard utility saturates above zero.

Math/utility functions come from gpytorch_porting (single source of truth).
Model class (VariationalGP) and training functions come from the 1D playground.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util
import warnings

warnings.filterwarnings("ignore", message=".*torch.cuda.*DtypeTensor.*")
warnings.filterwarnings("ignore", message=".*torch.sparse.SparseTensor.*")

# -----------------------------------------------------------------------------
# Import model, training, and data generation from 1D playground
# (VariationalGP, train_gp, generate_poisson_data are kept in 1D)
# Note: compute_elbo in 1D computes ELL inline with hardcoded A=1, lambda0=0.
# This is numerically identical to gpytorch_porting PoissonLikelihood with
# A=1, lambda0=0, so training behavior is unchanged.
# -----------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "1D_playground"))

from gp_utility_playground import (
    VariationalGP,
    train_gp,
    train_gp_fixed_lengthscale,
    generate_poisson_data,
    DEVICE,
    DTYPE,
)

# -----------------------------------------------------------------------------
# Import from gpytorch_porting (single source of truth for math/utility)
# Uses importlib.util to avoid sys.modules collision with repo-root utils.py
# (which gets cached when 1D playground adds Spatial_GP_repo/ to sys.path).
# -----------------------------------------------------------------------------
GPYTORCH_PATH = Path(__file__).parent.parent / 'gpytorch_porting'

def _load_gpytorch_module(filename, module_name):
    """Load a gpytorch_porting module via importlib.util."""
    path = GPYTORCH_PATH / filename
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_gpy_utils = _load_gpytorch_module('utils.py', 'gpytorch_porting_utils')
_gpy_likelihoods = _load_gpytorch_module('likelihoods.py', 'gpytorch_porting_likelihoods')
_gpy_kernels = _load_gpytorch_module('kernels.py', 'gpytorch_porting_kernels')
_gpy_acquisition = _load_gpytorch_module('acquisition.py', 'gpytorch_porting_acquisition')

# Re-export for use by other 2D scripts
PoissonLikelihood = _gpy_likelihoods.PoissonLikelihood
SimpleArcCosineKernel = _gpy_kernels.SimpleArcCosineKernel
standard_utility = _gpy_acquisition.standard_utility
distribution_aware_utility = _gpy_acquisition.distribution_aware_utility
# Low-level functions needed by conditioning diagnostic scripts
get_gp_marginal_moments = _gpy_utils.get_gp_marginal_moments
get_gp_conditional_moments = _gpy_utils.get_gp_conditional_moments
compute_H = _gpy_utils.compute_H
compute_adaptive_rmax = _gpy_utils.compute_adaptive_rmax


# -----------------------------------------------------------------------------
# 2D-specific configuration (EXPORTED for other scripts)
# -----------------------------------------------------------------------------
Y_MIN, Y_MAX = -30, 30
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
# Checkpoint Functions
# -----------------------------------------------------------------------------
def save_rbf_2d_checkpoint(model, likelihood, inducing_points, train_x, train_y, filepath, **kwargs):
    """Save RBF GP checkpoint with full state.

    Args:
        model: Trained VariationalGP model
        likelihood: Trained PoissonLikelihood (gpytorch_porting version)
        inducing_points: (M, 2) inducing point tensor
        train_x: (N, 2) training inputs
        train_y: (N,) training observations
        filepath: Path to save checkpoint
        **kwargs: Additional config to store
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
        },
        'training_config': kwargs
    }
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")


def load_rbf_2d_checkpoint(filepath):
    """Load RBF GP checkpoint, return model + likelihood + metadata.

    Returns:
        model: VariationalGP with loaded weights (in eval mode)
        likelihood: PoissonLikelihood with loaded weights (in eval mode)
        metadata: dict with inducing_points, train_x, train_y, hyperparameters
    """
    checkpoint = torch.load(filepath, map_location=DEVICE, weights_only=False)

    # Reconstruct model
    inducing_points = checkpoint['inducing_points']
    model = VariationalGP(inducing_points, jitter=1e-4).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Reconstruct likelihood (gpytorch_porting PoissonLikelihood)
    likelihood = PoissonLikelihood().to(DEVICE)
    if 'likelihood_state_dict' in checkpoint:
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

    print(f"Loaded checkpoint from {filepath}")
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
    peak_x, peak_y = 0.0, 0.0
    sigma_x = 2.0
    sigma_y = 3.5
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

    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    return grid_points


# -----------------------------------------------------------------------------
# MC Diagnostics (kernel-agnostic)
# -----------------------------------------------------------------------------
def compute_mc_diagnostics_2d(
    model,
    likelihood,
    candidates,
    n_mc_samples,
    p_x_mean,
    p_x_std,
    r_max=None,
    adaptive_r_max=False,
):
    """Compute MC statistics for conditioning analysis in 2D.

    Kernel-agnostic — works identically for RBF and Arc-Cosine. The conditioning
    math (Gaussian conditioning) only depends on the GP posterior covariance.

    Args:
        model: Trained 2D GP model
        likelihood: PoissonLikelihood with .A and .lambda0 attributes.
        candidates: (K, 2) query points
        n_mc_samples: Number of MC samples from p(x)
        p_x_mean: (2,) mean of 2D Gaussian p(x)
        p_x_std: (2,) std of 2D Gaussian p(x)
        r_max: Max spike count for entropy computation.
            Required unless adaptive_r_max=True.
        adaptive_r_max: If True, compute r_max adaptively from GP moments.

    Returns:
        H_marg, H_cond, mu_marg, mu_cond_avg, sigma2_marg, sigma2_cond_avg
    """
    if r_max is None and not adaptive_r_max:
        raise ValueError("Must specify either r_max=<int> or adaptive_r_max=True")
    if r_max is not None and adaptive_r_max:
        raise ValueError("Cannot specify both r_max and adaptive_r_max=True")

    model.eval()
    device = candidates.device
    dtype = candidates.dtype

    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()

    mu_marg, sigma2_marg = get_gp_marginal_moments(model, candidates)

    if adaptive_r_max:
        mu_g_marg = A * mu_marg + lambda0
        sigma2_g_marg = A ** 2 * sigma2_marg
        r_max_marg = compute_adaptive_rmax(mu_g_marg, sigma2_g_marg)
    else:
        r_max_marg = r_max

    H_marg = compute_H(mu_marg, sigma2_marg, r_max=r_max_marg, a=A, lambda0=lambda0)

    H_cond_sum = torch.zeros_like(H_marg)
    mu_cond_sum = torch.zeros_like(mu_marg)
    sigma2_cond_sum = torch.zeros_like(sigma2_marg)

    with torch.no_grad():
        for i in range(n_mc_samples):
            x_i = p_x_mean + p_x_std * torch.randn(2, dtype=dtype, device=device)

            post_i = model(x_i.unsqueeze(0))
            mu_i = post_i.mean[0]
            std_i = post_i.variance[0].sqrt()

            lambda_i = mu_i + std_i * torch.randn(1, dtype=dtype, device=device)

            mu_cond_i, sigma2_cond_i = get_gp_conditional_moments(
                model, candidates, x_i, lambda_i
            )

            if adaptive_r_max:
                mu_g_cond = A * mu_cond_i + lambda0
                sigma2_g_cond = A ** 2 * sigma2_cond_i
                r_max_cond = compute_adaptive_rmax(mu_g_cond, sigma2_g_cond)
            else:
                r_max_cond = r_max

            H_cond_i = compute_H(mu_cond_i, sigma2_cond_i, r_max=r_max_cond, a=A, lambda0=lambda0)
            H_cond_sum += H_cond_i
            mu_cond_sum += mu_cond_i
            sigma2_cond_sum += sigma2_cond_i

            if (i + 1) % 50 == 0:
                print(f"  MC sample {i+1}/{n_mc_samples}")

    H_cond = H_cond_sum / n_mc_samples
    mu_cond_avg = mu_cond_sum / n_mc_samples
    sigma2_cond_avg = sigma2_cond_sum / n_mc_samples

    return H_marg, H_cond, mu_marg, mu_cond_avg, sigma2_marg, sigma2_cond_avg


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

    grid_flat = torch.stack([eval_grid_x.flatten(), eval_grid_y.flatten()], dim=-1)

    with torch.no_grad():
        posterior = model(grid_flat)
        mean = posterior.mean
        std = posterior.variance.sqrt()
        true_lambda = lambda_fn(grid_flat)

    mean_2d = mean.reshape(n_x, n_y).cpu().numpy()
    std_2d = std.reshape(n_x, n_y).cpu().numpy()
    true_2d = true_lambda.reshape(n_x, n_y).cpu().numpy()
    utility_std_2d = utility_standard.reshape(n_x, n_y).cpu().numpy()
    utility_da_2d = utility_distr_aware.reshape(n_x, n_y).cpu().numpy()

    x_np = eval_grid_x.cpu().numpy()
    y_np = eval_grid_y.cpu().numpy()
    train_x_np = train_x.cpu().numpy()

    # Clip utilities for visualization
    def get_clip_value(arr, pctl=99):
        finite_vals = arr[np.isfinite(arr)]
        if len(finite_vals) == 0:
            return 1.0
        return np.percentile(finite_vals, pctl)

    da_clip = get_clip_value(utility_da_2d, 99)
    std_clip_raw = get_clip_value(utility_std_2d, 99)
    std_clip = min(std_clip_raw, da_clip * 5) if da_clip > 0 else std_clip_raw

    def clip_array(arr, clip_val):
        result = np.clip(arr, None, clip_val)
        result[~np.isfinite(result)] = clip_val
        return result

    utility_std_clipped = clip_array(utility_std_2d, std_clip)
    utility_da_clipped = clip_array(utility_da_2d, da_clip)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: True lambda, GP mean, GP std
    ax1 = axes[0, 0]
    im1 = ax1.contourf(x_np, y_np, true_2d, levels=20, cmap='viridis')
    ax1.scatter(train_x_np[:, 0], train_x_np[:, 1], c='red', s=30,
                edgecolors='white', linewidths=1, marker='o', label='Training')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('True lambda(x,y)')
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
    ax5.set_title('Distr-Aware Utility\n(orange = p(x) 1s, 2s)')
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
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print("GP Utility Playground 2D")
    print("=" * 70)

    # Load checkpoint
    print(f"\nLoading checkpoint...")
    model, likelihood, metadata = load_rbf_2d_checkpoint(RBF_2D_CHECKPOINT_PATH)

    train_x = metadata['train_x']
    train_y = metadata['train_y']

    print(f"\nTraining data from checkpoint:")
    print(f"  Training points: {train_x.shape}")
    print(f"  Spike count range: [{train_y.min().item():.0f}, {train_y.max().item():.0f}]")

    # Create evaluation grid
    n_eval_x, n_eval_y = 40, 40
    print(f"\nCreating {n_eval_x}x{n_eval_y} evaluation grid...")

    x_eval = torch.linspace(X_MIN, X_MAX, n_eval_x, dtype=DTYPE, device=DEVICE)
    y_eval = torch.linspace(Y_MIN, Y_MAX, n_eval_y, dtype=DTYPE, device=DEVICE)
    eval_grid_x, eval_grid_y = torch.meshgrid(x_eval, y_eval, indexing='xy')
    eval_points = torch.stack([eval_grid_x.flatten(), eval_grid_y.flatten()], dim=-1)

    print(f"Evaluation points: {eval_points.shape}")

    # Evaluate standard utility (adaptive r_max for non-stationary kernels)
    print("\nEvaluating standard utility...")
    with torch.no_grad():
        result_std = standard_utility(model, likelihood, eval_points, adaptive_r_max=True)
    utility_standard = result_std['utility']

    # Evaluate distribution-aware utility
    print("\nEvaluating distribution-aware utility (this may take a minute)...")
    # Draw MC samples from p(x) = N(mean, diag(std^2))
    n_mc = 500
    x_samples = DEFAULT_P_X_MEAN_2D + DEFAULT_P_X_STD_2D * torch.randn(
        n_mc, 2, dtype=DTYPE, device=DEVICE
    )
    with torch.no_grad():
        result_da = distribution_aware_utility(
            model, likelihood, eval_points, x_samples,
            adaptive_r_max=True
        )
    utility_distr_aware = result_da['utility']

    # Find max utility locations
    max_idx_std = torch.argmax(utility_standard)
    max_idx_da = torch.argmax(utility_distr_aware)

    max_std_loc = eval_points[max_idx_std].cpu().numpy()
    max_da_loc = eval_points[max_idx_da].cpu().numpy()

    print(f"\nStandard utility max at: ({max_std_loc[0]:.3f}, {max_std_loc[1]:.3f})")
    print(f"Distribution-aware utility max at: ({max_da_loc[0]:.3f}, {max_da_loc[1]:.3f})")
    print(f"p(x) center: ({DEFAULT_P_X_MEAN_2D[0]:.3f}, {DEFAULT_P_X_MEAN_2D[1]:.3f})")

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
