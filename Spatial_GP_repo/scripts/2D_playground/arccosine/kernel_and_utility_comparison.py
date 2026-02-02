"""
Arc-Cosine vs RBF Kernel Comparison (2D) - Created by Claude
============================================================
Compares Arc-Cosine kernel (C=I, Stage 1) with RBF kernel on 2D GP playground.

REFACTORED: Now loads pre-trained models from checkpoints instead of training.
- RBF checkpoint: ../trained_rbf_2d_checkpoint.pt
- Arc-Cosine checkpoint: ./trained_arccosine_2d_checkpoint.pt

Arc-Cosine kernel is NON-STATIONARY: k(x,x) = ||x||² + σ₀² (varies with x!)
RBF kernel is STATIONARY: k(x,x) = outputscale (constant)

Run train_rbf_2d.py and train_acos_2d.py first to create checkpoints.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# -----------------------------------------------------------------------------
# Import path setup
# -----------------------------------------------------------------------------

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent))  # 2D_playground
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "1D_playground"))  # 1D_playground

# Import from 1D playground (core components)
from gp_utility_playground import (
    VariationalGP,
    evaluate_nd_utility_new,
    DEVICE,
    DTYPE,
)

# Import from 2D playground (2D-specific components + checkpoint loader)
from utility_2d_rbf_base import (
    # Checkpoint loader
    load_rbf_2d_checkpoint,
    # 2D-specific functions
    lambda_true_2d,
    create_2d_grid,
    evaluate_distribution_aware_utility_2d,
    # Configuration - CORRECTED: use _2D suffix!
    DEFAULT_P_X_MEAN_2D,
    DEFAULT_P_X_STD_2D,
)

# Domain bounds (local override for tighter visualization)
X_MIN, X_MAX = -20, 20
Y_MIN, Y_MAX = -20, 20

# Import ArcCosineKernel and PoissonLikelihood from gpytorch_porting
GPYTORCH_PORTING_PATH = Path(__file__).parent.parent.parent / 'gpytorch_porting'
EXPECTED_KERNEL_PATH = '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting'

if not GPYTORCH_PORTING_PATH.exists():
    raise ImportError(f"gpytorch_porting not found at {GPYTORCH_PORTING_PATH}")
if str(GPYTORCH_PORTING_PATH.resolve()) != EXPECTED_KERNEL_PATH:
    raise ImportError(f"Wrong kernel path!\n  Expected: {EXPECTED_KERNEL_PATH}\n  Got: {GPYTORCH_PORTING_PATH.resolve()}")

sys.path.insert(0, str(GPYTORCH_PORTING_PATH))
from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood

warnings.filterwarnings("ignore", message=".*torch.cuda.*DtypeTensor.*")
warnings.filterwarnings("ignore", message=".*torch.sparse.SparseTensor.*")

# Override p(x) distribution with wider std for better visualization
P_X_MEAN = torch.tensor([0.0, 0.0], dtype=DTYPE, device=DEVICE)
P_X_STD = torch.tensor([2.0, 2.0], dtype=DTYPE, device=DEVICE)


# -----------------------------------------------------------------------------
# Checkpoint Loading
# -----------------------------------------------------------------------------
def load_arccosine_2d_checkpoint(filepath):
    """
    Load Arc-Cosine checkpoint with kernel swap.

    IMPORTANT: Kernel must be swapped BEFORE loading state_dict!
    The state dict contains covar_module.* params that must match kernel type.

    Returns:
        model: VariationalGP with ArcCosineKernel
        likelihood: PoissonLikelihood (gpytorch_porting version)
        metadata: dict with training info
    """
    checkpoint = torch.load(filepath, map_location=DEVICE, weights_only=False)

    # Reconstruct model with RBF kernel first (VariationalGP default)
    inducing_points = checkpoint['inducing_points']
    model = VariationalGP(inducing_points, jitter=1e-4).to(DEVICE)

    # CRITICAL: Swap to Arc-Cosine BEFORE loading state dict
    kernel_params = checkpoint['kernel_params']
    model.covar_module = ArcCosineKernel(
        sigma_0=kernel_params['sigma_0'],
        Amp=kernel_params['Amp'],
        C=None  # Stage 1 identity
    ).to(DEVICE)

    # NOW load model state (covar_module params will match)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Reconstruct likelihood (gpytorch_porting version)
    lik_params = checkpoint['likelihood_params']
    likelihood = PoissonLikelihood(
        A_init=lik_params['A'],
        lambda0_init=lik_params['lambda_0']
    ).to(DEVICE)
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    likelihood.eval()

    metadata = {
        'inducing_points': inducing_points,
        'train_x': checkpoint['train_x'],
        'train_y': checkpoint['train_y'],
        'kernel_params': kernel_params,
        'likelihood_params': lik_params,
        'training_config': checkpoint.get('training_config', {})
    }

    print(f"✓ Loaded Arc-Cosine checkpoint from {filepath}")
    print(f"  Kernel: σ₀={kernel_params['sigma_0']:.4f}, Amp={kernel_params['Amp']:.4f}")
    print(f"  Likelihood: A={lik_params['A']:.4f}, λ₀={lik_params['lambda_0']:.3f}")

    return model, likelihood, metadata


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def plot_comparison(model_acos, model_rbf, train_x, train_y, lambda_fn,
                    eval_grid_x, eval_grid_y,
                    utility_standard_acos, utility_distr_acos,
                    utility_standard_rbf, utility_distr_rbf,
                    p_x_mean=None, p_x_std=None, save_path=None):
    """Create 2x3 comparison: Arc-Cosine (top) vs RBF (bottom), with λ(x,y), standard utility, and dist-aware utility."""
    model_acos.eval()
    model_rbf.eval()

    n_x, n_y = eval_grid_x.shape
    grid_flat = torch.stack([eval_grid_x.flatten(), eval_grid_y.flatten()], dim=-1)

    with torch.no_grad():
        # GP posteriors
        posterior_acos = model_acos(grid_flat)
        mean_acos = posterior_acos.mean

        posterior_rbf = model_rbf(grid_flat)
        mean_rbf = posterior_rbf.mean

        # True function
        true_lambda = lambda_fn(grid_flat)

    # Reshape for plotting
    mean_acos_2d = mean_acos.reshape(n_x, n_y).cpu().numpy()
    mean_rbf_2d = mean_rbf.reshape(n_x, n_y).cpu().numpy()
    true_lambda_2d = true_lambda.reshape(n_x, n_y).cpu().numpy()

    utility_standard_acos_2d = utility_standard_acos.reshape(n_x, n_y).cpu().numpy()
    utility_distr_acos_2d = utility_distr_acos.reshape(n_x, n_y).cpu().numpy()
    utility_standard_rbf_2d = utility_standard_rbf.reshape(n_x, n_y).cpu().numpy()
    utility_distr_rbf_2d = utility_distr_rbf.reshape(n_x, n_y).cpu().numpy()

    # Clip Arc-Cosine utilities to reasonable range for visualization
    # (Arc-Cosine can have extreme values at corners due to k(x,x) = ||x||^2 + sigma_0^2)
    acos_clip_max = np.percentile(utility_standard_acos_2d, 99)
    utility_standard_acos_2d_clipped = np.clip(utility_standard_acos_2d, None, acos_clip_max)
    acos_distr_clip_max = np.percentile(utility_distr_acos_2d, 99)
    utility_distr_acos_2d_clipped = np.clip(utility_distr_acos_2d, None, acos_distr_clip_max)

    print(f"\n[Arc-Cosine Utility Clipping]")
    print(f"  Standard: max={utility_standard_acos_2d.max():.2e}, 99th percentile={acos_clip_max:.4f}")
    print(f"  Dist-aware: max={utility_distr_acos_2d.max():.2e}, 99th percentile={acos_distr_clip_max:.4f}")

    # Prepare arrays for plotting
    x_np = eval_grid_x.cpu().numpy()
    y_np = eval_grid_y.cpu().numpy()
    train_x_np = train_x.cpu().numpy()

    # Prepare p(x) contours
    if p_x_mean is not None and p_x_std is not None:
        px_mean_np = p_x_mean.cpu().numpy()
        px_std_np = p_x_std.cpu().numpy()
        theta = np.linspace(0, 2*np.pi, 100)

    # Create figure (2x3 layout)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Arc-Cosine vs RBF Kernel Comparison (2D) - Models loaded from checkpoints',
                 fontsize=14, fontweight='bold')

    # Condition numbers for titles
    with torch.no_grad():
        K_acos = model_acos.covar_module(train_x).evaluate()
        cond_acos = torch.linalg.cond(K_acos).item()
        K_rbf = model_rbf.covar_module(train_x).evaluate()
        cond_rbf = torch.linalg.cond(K_rbf).item()

    # Row 1: Arc-Cosine
    ax1, ax2, ax3 = axes[0]

    # Subplot 1: Arc-Cosine λ(x,y)
    im1 = ax1.contourf(x_np, y_np, mean_acos_2d, levels=20, cmap='viridis')
    ax1.contour(x_np, y_np, true_lambda_2d, levels=5, colors='white', linewidths=0.5, alpha=0.5)
    ax1.scatter(train_x_np[:, 0], train_x_np[:, 1], c='red', s=20, marker='x', alpha=0.7)
    ax1.set_title(f'Arc-Cosine: λ(x,y) | cond={cond_acos:.1e}', fontsize=10)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, label='λ')

    # Subplot 2: Arc-Cosine Standard Utility (CLIPPED for visualization)
    im2 = ax2.contourf(x_np, y_np, utility_standard_acos_2d_clipped, levels=20, cmap='coolwarm')
    ax2.scatter(train_x_np[:, 0], train_x_np[:, 1], c='black', s=20, marker='x', alpha=0.5)

    # Mark standard utility max (on clipped data)
    idx_max_std = np.argmax(utility_standard_acos_2d_clipped)
    max_y_std_idx, max_x_std_idx = np.unravel_index(idx_max_std, utility_standard_acos_2d_clipped.shape)
    ax2.scatter([x_np[max_y_std_idx, max_x_std_idx]], [y_np[max_y_std_idx, max_x_std_idx]],
                c='yellow', s=80, marker='o', edgecolors='black', linewidths=1.5, label='Max', zorder=10)

    # Add p(x) contours
    if p_x_mean is not None and p_x_std is not None:
        for n_sigma in [1, 2]:
            ellipse_x = px_mean_np[0] + n_sigma * px_std_np[0] * np.cos(theta)
            ellipse_y = px_mean_np[1] + n_sigma * px_std_np[1] * np.sin(theta)
            ax2.plot(ellipse_x, ellipse_y, 'lime', linewidth=2.5, alpha=0.9, label=f'{n_sigma}σ p(x)' if n_sigma == 1 else '')

    ax2.set_title(f'Arc-Cosine: Standard Utility\n(CLIPPED to 99th pctl: {acos_clip_max:.2f})', fontsize=10)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend(loc='upper right', fontsize=8)
    plt.colorbar(im2, ax=ax2, label='Utility')

    # Subplot 3: Arc-Cosine Distribution-Aware Utility (CLIPPED for visualization)
    im3 = ax3.contourf(x_np, y_np, utility_distr_acos_2d_clipped, levels=20, cmap='coolwarm')
    ax3.scatter(train_x_np[:, 0], train_x_np[:, 1], c='black', s=20, marker='x', alpha=0.5)

    # Mark dist-aware utility max (on clipped data)
    idx_max_dist = np.argmax(utility_distr_acos_2d_clipped)
    max_y_dist_idx, max_x_dist_idx = np.unravel_index(idx_max_dist, utility_distr_acos_2d_clipped.shape)
    ax3.scatter([x_np[max_y_dist_idx, max_x_dist_idx]], [y_np[max_y_dist_idx, max_x_dist_idx]],
                c='yellow', s=80, marker='o', edgecolors='black', linewidths=1.5, label='Max', zorder=10)

    # Add p(x) contours
    if p_x_mean is not None and p_x_std is not None:
        for n_sigma in [1, 2]:
            ellipse_x = px_mean_np[0] + n_sigma * px_std_np[0] * np.cos(theta)
            ellipse_y = px_mean_np[1] + n_sigma * px_std_np[1] * np.sin(theta)
            ax3.plot(ellipse_x, ellipse_y, 'lime', linewidth=2.5, alpha=0.9, label=f'{n_sigma}σ p(x)' if n_sigma == 1 else '')

    ax3.set_title(f'Arc-Cosine: Distribution-Aware Utility\n(CLIPPED to 99th pctl: {acos_distr_clip_max:.4f})', fontsize=10)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.legend(loc='upper right', fontsize=8)
    plt.colorbar(im3, ax=ax3, label='Utility')

    # Row 2: RBF
    ax4, ax5, ax6 = axes[1]

    # Subplot 4: RBF λ(x,y)
    im4 = ax4.contourf(x_np, y_np, mean_rbf_2d, levels=20, cmap='viridis')
    ax4.contour(x_np, y_np, true_lambda_2d, levels=5, colors='white', linewidths=0.5, alpha=0.5)
    ax4.scatter(train_x_np[:, 0], train_x_np[:, 1], c='red', s=20, marker='x', alpha=0.7)
    ax4.set_title(f'RBF: λ(x,y) | cond={cond_rbf:.1e}', fontsize=10)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plt.colorbar(im4, ax=ax4, label='λ')

    # Subplot 5: RBF Standard Utility
    im5 = ax5.contourf(x_np, y_np, utility_standard_rbf_2d, levels=20, cmap='coolwarm')
    ax5.scatter(train_x_np[:, 0], train_x_np[:, 1], c='black', s=20, marker='x', alpha=0.5)

    # Mark standard utility max
    idx_max_std = utility_standard_rbf.argmax()
    max_x_std, max_y_std = grid_flat[idx_max_std].cpu().numpy()
    ax5.scatter([max_x_std], [max_y_std], c='yellow', s=80, marker='o', edgecolors='black', linewidths=1.5, label='Max', zorder=10)

    # Add p(x) contours
    if p_x_mean is not None and p_x_std is not None:
        for n_sigma in [1, 2]:
            ellipse_x = px_mean_np[0] + n_sigma * px_std_np[0] * np.cos(theta)
            ellipse_y = px_mean_np[1] + n_sigma * px_std_np[1] * np.sin(theta)
            ax5.plot(ellipse_x, ellipse_y, 'lime', linewidth=2.5, alpha=0.9, label=f'{n_sigma}σ p(x)' if n_sigma == 1 else '')

    ax5.set_title('RBF: Standard Utility', fontsize=10)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.legend(loc='upper right', fontsize=8)
    plt.colorbar(im5, ax=ax5, label='Utility')

    # Subplot 6: RBF Distribution-Aware Utility
    im6 = ax6.contourf(x_np, y_np, utility_distr_rbf_2d, levels=20, cmap='coolwarm')
    ax6.scatter(train_x_np[:, 0], train_x_np[:, 1], c='black', s=20, marker='x', alpha=0.5)

    # Mark dist-aware utility max
    idx_max_dist = utility_distr_rbf.argmax()
    max_x_dist, max_y_dist = grid_flat[idx_max_dist].cpu().numpy()
    ax6.scatter([max_x_dist], [max_y_dist], c='yellow', s=80, marker='o', edgecolors='black', linewidths=1.5, label='Max', zorder=10)

    # Add p(x) contours
    if p_x_mean is not None and p_x_std is not None:
        for n_sigma in [1, 2]:
            ellipse_x = px_mean_np[0] + n_sigma * px_std_np[0] * np.cos(theta)
            ellipse_y = px_mean_np[1] + n_sigma * px_std_np[1] * np.sin(theta)
            ax6.plot(ellipse_x, ellipse_y, 'lime', linewidth=2.5, alpha=0.9, label=f'{n_sigma}σ p(x)' if n_sigma == 1 else '')

    ax6.set_title('RBF: Distribution-Aware Utility', fontsize=10)
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.legend(loc='upper right', fontsize=8)
    plt.colorbar(im6, ax=ax6, label='Utility')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")

    return fig


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("GP 2D Arc-Cosine vs RBF Kernel Comparison")
    print("(Loading pre-trained models from checkpoints)")
    print("=" * 70)

    # Checkpoint paths
    rbf_checkpoint_path = Path(__file__).parent.parent / 'trained_rbf_2d_checkpoint.pt'
    acos_checkpoint_path = Path(__file__).parent / 'trained_arccosine_2d_checkpoint.pt'

    # Check checkpoints exist
    if not rbf_checkpoint_path.exists():
        print(f"\n[ERROR] RBF checkpoint not found: {rbf_checkpoint_path}")
        print("Run: python ../train_rbf_2d.py first")
        sys.exit(1)
    if not acos_checkpoint_path.exists():
        print(f"\n[ERROR] Arc-Cosine checkpoint not found: {acos_checkpoint_path}")
        print("Run: python train_acos_2d.py first")
        sys.exit(1)

    # Load checkpoints
    print("\n" + "-" * 50)
    print("Loading RBF checkpoint...")
    model_rbf, likelihood_rbf, rbf_metadata = load_rbf_2d_checkpoint(rbf_checkpoint_path)

    print("\n" + "-" * 50)
    print("Loading Arc-Cosine checkpoint...")
    model_acos, likelihood_acos, acos_metadata = load_arccosine_2d_checkpoint(acos_checkpoint_path)

    # Get training data from checkpoints
    train_x = rbf_metadata['train_x']
    train_y = rbf_metadata['train_y']

    print(f"\nTraining data from checkpoint:")
    print(f"  Training points: {train_x.shape}")
    print(f"  Spike count range: [{train_y.min().item():.0f}, {train_y.max().item():.0f}]")

    # ------------------------------
    # Evaluate Utilities
    # ------------------------------
    print("\n" + "=" * 70)
    print("Evaluating utilities...")
    print("=" * 70)

    # Create evaluation grid
    n_eval = 40
    x_eval = torch.linspace(X_MIN, X_MAX, n_eval, dtype=DTYPE, device=DEVICE)
    y_eval = torch.linspace(Y_MIN, Y_MAX, n_eval, dtype=DTYPE, device=DEVICE)
    eval_grid_x, eval_grid_y = torch.meshgrid(x_eval, y_eval, indexing='xy')
    eval_grid_flat = torch.stack([eval_grid_x.flatten(), eval_grid_y.flatten()], dim=-1)

    # Arc-Cosine utilities
    print("\nArc-Cosine - Standard utility...")
    utility_standard_acos = evaluate_nd_utility_new(model_acos, eval_grid_flat, max_r=100)

    print("\nArc-Cosine - Distribution-aware utility...")
    utility_distr_acos = evaluate_distribution_aware_utility_2d(
        model_acos, eval_grid_flat, n_mc_samples=500, r_max=100,
        p_x_mean=P_X_MEAN, p_x_std=P_X_STD
    )

    # RBF utilities
    print("\nRBF - Standard utility...")
    utility_standard_rbf = evaluate_nd_utility_new(model_rbf, eval_grid_flat, max_r=100)

    print("\nRBF - Distribution-aware utility...")
    utility_distr_rbf = evaluate_distribution_aware_utility_2d(
        model_rbf, eval_grid_flat, n_mc_samples=500, r_max=100,
        p_x_mean=P_X_MEAN, p_x_std=P_X_STD
    )

    # ------------------------------
    # Visualization
    # ------------------------------
    print("\nGenerating visualization...")
    save_path = Path(__file__).parent / "kernel_and_utility_comparison.png"

    plot_comparison(
        model_acos, model_rbf,
        train_x, train_y, lambda_true_2d,
        eval_grid_x, eval_grid_y,
        utility_standard_acos, utility_distr_acos,
        utility_standard_rbf, utility_distr_rbf,
        p_x_mean=P_X_MEAN,
        p_x_std=P_X_STD,
        save_path=save_path
    )

    print("\nDone!")
