"""
Arc-Cosine vs RBF Kernel Comparison - Created by Claude
========================================================
Compares Arc-Cosine kernel (C=I, Stage 1) with RBF kernel on 1D GP playground.

Arc-Cosine kernel is NON-STATIONARY: k(x,x) = x² + σ₀² (varies with x!)
RBF kernel is STATIONARY: k(x,x) = outputscale (constant)

NO TRAINING: Hyperparameters are fixed for reproducibility.
Evaluation points use arange (not linspace) so widening domain adds points, not changes them.

Reuses code from gp_utility_playground.py and gpytorch_porting/.
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

# Add parent directory to import from gp_utility_playground
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from gp_utility_playground (only what's actually used)
from gp_utility_playground import (
    # Configuration
    DEVICE, DTYPE,
    X_MAX, X_MIN,
    # p(x) parameters (used for distribution-aware utility)
    DEFAULT_P_X_MEAN, DEFAULT_P_X_STD, X_SAMPLE, SAMPLE_X,
    # Conditioning seed (used for reproducible lambda_obs sampling)
    LAMBDA_SEED,
    # Data generation
    lambda_true,
    # Model components
    PoissonLikelihood,
    VariationalGP,
    # Utility evaluation
    evaluate_distribution_aware_utility,
    evaluate_nd_utility_new,
    # Conditioning (for before/after plots)
    get_conditional_moments,
)


# Domain configuration (OVERRIDE: wider than default for Arc-Cosine visualization)
X_MIN, X_MAX = X_MIN, X_MAX  # Default is [-5, 5], but Arc-Cosine needs wider range
X_STEP = 0.05  # Fixed step for arange (widening domain adds points, doesn't change them)

# Conditioning configuration (OVERRIDE: different X_SAMPLE for Arc-Cosine demo)
X_SAMPLE = X_SAMPLE  # Default is 2.0, but we use 13.5 to show non-stationary behavior far from origin

# Import ArcCosineKernel from gpytorch_porting
GPYTORCH_PORTING_PATH = Path(__file__).parent.parent.parent / 'gpytorch_porting'
EXPECTED_KERNEL_PATH = '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting'

if not GPYTORCH_PORTING_PATH.exists():
    raise ImportError(f"gpytorch_porting not found at {GPYTORCH_PORTING_PATH}")
if str(GPYTORCH_PORTING_PATH.resolve()) != EXPECTED_KERNEL_PATH:
    raise ImportError(f"Wrong kernel path!\n  Expected: {EXPECTED_KERNEL_PATH}\n  Got: {GPYTORCH_PORTING_PATH.resolve()}")

sys.path.insert(0, str(GPYTORCH_PORTING_PATH))
from kernels import ArcCosineKernel

import kernels
actual_kernel_file = Path(kernels.__file__).resolve()
if EXPECTED_KERNEL_PATH not in str(actual_kernel_file):
    raise ImportError(f"ArcCosineKernel imported from wrong location: {actual_kernel_file}")

print(f"[OK] ArcCosineKernel imported from: {actual_kernel_file}")

warnings.filterwarnings("ignore", message=".*torch.cuda.*DtypeTensor.*")
warnings.filterwarnings("ignore", message=".*torch.sparse.SparseTensor.*")


# -----------------------------------------------------------------------------
# Kernel Health Monitoring
# -----------------------------------------------------------------------------
def check_kernel_health(model, X, context=""):
    """Log kernel matrix condition number and warn if ill-conditioned."""
    with torch.no_grad():
        K = model.covar_module(X).evaluate()
        cond = torch.linalg.cond(K).item()
    print(f"[Kernel Health] {context}: cond(K) = {cond:.2e}")
    if cond > 1e12:
        warnings.warn(f"Ill-conditioned kernel matrix: {cond:.2e}")
    return cond


# -----------------------------------------------------------------------------
# Model Setup (no training - fixed hyperparameters)
# -----------------------------------------------------------------------------

def load_rbf_checkpoint(checkpoint_path):
    """Load trained RBF GP from checkpoint with full metadata.

    Checkpoint contains:
    - model_state_dict: Full model state (hyperparams + variational params m, V)
    - config: Training configuration (seed, n_train, x_min, x_max, etc.)
    - inducing_points: Exact inducing points used during training
    - hyperparameters: Quick reference to learned values
    - description: Human-readable description
    """
    print(f"  Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Print checkpoint info
    config = checkpoint['config']
    hyp = checkpoint['hyperparameters']
    print(f"  Description: {checkpoint.get('description', 'N/A')}")
    print(f"  Created: {checkpoint.get('created', 'N/A')}")
    print(f"  Training config: seed={config['seed']}, n_train={config['n_train']}, "
          f"iterations={config['n_iterations']}, domain=[{config['x_min']}, {config['x_max']}]")
    print(f"  Ground truth: {config['ground_truth']}")
    print(f"  Hyperparameters: lengthscale={hyp['lengthscale']:.4f}, "
          f"outputscale={hyp['outputscale']:.4f}, mean={hyp['mean_constant']:.4f}")

    # Reconstruct model with SAME inducing points
    inducing_points = checkpoint['inducing_points'].to(dtype=DTYPE, device=DEVICE)
    print(f"  Inducing points: {len(inducing_points)} points in [{inducing_points.min():.2f}, {inducing_points.max():.2f}]")

    model = VariationalGP(inducing_points, jitter=0).to(DEVICE)
    likelihood = PoissonLikelihood().to(DEVICE)

    # Load full state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    check_kernel_health(model, inducing_points, context="RBF")

    # Extract training data if available
    train_x = checkpoint.get('train_x', None)
    train_y = checkpoint.get('train_y', None)
    if train_x is not None:
        train_x = train_x.to(dtype=DTYPE, device=DEVICE)
        train_y = train_y.to(dtype=DTYPE, device=DEVICE)
        print(f"  Training data: {len(train_x)} points, y in [{train_y.min():.0f}, {train_y.max():.0f}]")

    return model, likelihood, inducing_points, config, train_x, train_y


def load_acos_checkpoint(checkpoint_path):
    """Load trained Arc-Cosine GP from checkpoint with full metadata.

    Checkpoint contains:
    - model_state_dict: Full model state (hyperparams + variational params m, V)
    - config: Training configuration (seed, n_train, x_min, x_max, etc.)
    - inducing_points: Exact inducing points used during training
    - train_x, train_y: Training data for plotting
    - hyperparameters: Quick reference to learned values
    """
    print(f"  Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Print checkpoint info
    config = checkpoint['config']
    hyp = checkpoint['hyperparameters']
    print(f"  Description: {checkpoint.get('description', 'N/A')}")
    print(f"  Created: {checkpoint.get('created', 'N/A')}")
    print(f"  Training config: seed={config['seed']}, n_train={config['n_train']}, "
          f"iterations={config['n_iterations']}, domain=[{config['x_min']}, {config['x_max']}]")
    print(f"  Ground truth: {config['ground_truth']}")
    print(f"  Hyperparameters: σ₀={hyp['sigma_0']:.4f}, "
          f"Amp={hyp['Amp']:.4f}, mean={hyp['mean_constant']:.4f}")
    print(f"  Condition number: {checkpoint.get('condition_number', 'N/A'):.2e}")

    # Reconstruct model with SAME inducing points
    inducing_points = checkpoint['inducing_points'].to(dtype=DTYPE, device=DEVICE)
    print(f"  Inducing points: {len(inducing_points)} points in [{inducing_points.min():.2f}, {inducing_points.max():.2f}]")

    model = VariationalGP(inducing_points, jitter=0).to(DEVICE)
    # Replace kernel with Arc-Cosine
    model.covar_module = ArcCosineKernel(
        sigma_0=hyp['sigma_0'],
        Amp=hyp['Amp'],
        C=None
    )
    likelihood = PoissonLikelihood().to(DEVICE)

    # Load full state (this loads trained variational params + hyperparams)
    model.load_state_dict(checkpoint['model_state_dict'])

    # DEBUG: Perturb hyperparameters AFTER loading (so they don't get overwritten)
    # DEBUG_PERTURB = True  # Set to True to perturb params
    DEBUG_PERTURB = False  # Set to True to perturb params
    if DEBUG_PERTURB:
        sigma_0_perturbed = hyp['sigma_0'] * 5.5  
        mean_perturbed = hyp['mean_constant'] * 5.8 
        print(f"  [DEBUG] Perturbing: σ₀ {hyp['sigma_0']:.4f}→{sigma_0_perturbed:.4f}, mean {hyp['mean_constant']:.4f}→{mean_perturbed:.4f}")
        # Apply perturbations
        model.covar_module.sigma_0.data.fill_(sigma_0_perturbed)
        model.mean_module.constant.data.fill_(mean_perturbed)

    model.eval()

    check_kernel_health(model, inducing_points, context="Arc-Cosine")

    # Extract training data
    train_x = checkpoint.get('train_x', None)
    train_y = checkpoint.get('train_y', None)
    if train_x is not None:
        train_x = train_x.to(dtype=DTYPE, device=DEVICE)
        train_y = train_y.to(dtype=DTYPE, device=DEVICE)
        print(f"  Training data: {len(train_x)} points, y in [{train_y.min():.0f}, {train_y.max():.0f}]")

    return model, likelihood, inducing_points, config, train_x, train_y


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def plot_comparison(model_acos, model_rbf, inducing_points, lambda_fn,
                    save_path=None, p_x_mean=None, p_x_std=None,
                    train_x=None, train_y=None):
    """Create 3x2 comparison: Arc-Cosine vs RBF with before/after conditioning."""
    model_acos.eval()
    model_rbf.eval()

    # Use arange for fixed points (widening domain adds points, doesn't change them)
    x_plot = torch.arange(X_MIN, X_MAX + X_STEP/2, X_STEP, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        # Arc-Cosine
        post_acos = model_acos(x_plot)
        mean_acos = post_acos.mean
        std_acos = post_acos.variance.sqrt()
        var_acos = post_acos.variance
        util_std_acos = evaluate_nd_utility_new(model_acos, x_plot)
        util_da_acos = evaluate_distribution_aware_utility(model_acos, x_plot, p_x_mean=p_x_mean, p_x_std=p_x_std) if p_x_mean is not None else None

        # RBF
        post_rbf = model_rbf(x_plot)
        mean_rbf = post_rbf.mean
        std_rbf = post_rbf.variance.sqrt()
        var_rbf = post_rbf.variance
        util_std_rbf = evaluate_nd_utility_new(model_rbf, x_plot)
        util_da_rbf = evaluate_distribution_aware_utility(model_rbf, x_plot, p_x_mean=p_x_mean, p_x_std=p_x_std) if p_x_mean is not None else None

        true_lambda = lambda_fn(x_plot)

        # =====================================================================
        # BEFORE/AFTER CONDITIONING: Sample λ_obs from RBF posterior at X_SAMPLE
        # =====================================================================
        x_sample_tensor = torch.tensor([X_SAMPLE], dtype=DTYPE, device=DEVICE)
        post_sample = model_rbf(x_sample_tensor)
        mu_at_sample = post_sample.mean[0]
        std_at_sample = post_sample.variance[0].sqrt()

        # Sample λ_obs from GP posterior (reproducible)
        torch.manual_seed(LAMBDA_SEED)
        eps = torch.randn(1, dtype=DTYPE, device=DEVICE)
        lambda_obs = (mu_at_sample + std_at_sample * eps).item()

        print(f"\n[Conditioning] Observation at x={X_SAMPLE}:")
        print(f"  RBF posterior mean: {mu_at_sample.item():.4f}")
        print(f"  RBF posterior std: {std_at_sample.item():.4f}")
        print(f"  Sampled λ_obs = {lambda_obs:.4f} (seed={LAMBDA_SEED})")

        # Compute conditional moments for both models (SAME λ_obs for fair comparison)
        mu_after_rbf, var_after_rbf = get_conditional_moments(model_rbf, x_plot, X_SAMPLE, lambda_obs)
        std_after_rbf = var_after_rbf.sqrt()

        mu_after_acos, var_after_acos = get_conditional_moments(model_acos, x_plot, X_SAMPLE, lambda_obs)
        std_after_acos = var_after_acos.sqrt()

    # Convert to numpy
    x_np = x_plot.cpu().numpy()
    true_np = true_lambda.cpu().numpy()

    # Training data for plotting
    train_x_np = train_x.cpu().numpy() if train_x is not None else None
    train_y_np = train_y.cpu().numpy() if train_y is not None else None

    # Before conditioning
    mean_acos_np = mean_acos.cpu().numpy()
    std_acos_np = std_acos.cpu().numpy()
    util_std_acos_np = util_std_acos.cpu().numpy()
    util_da_acos_np = util_da_acos.cpu().numpy() if util_da_acos is not None else None

    mean_rbf_np = mean_rbf.cpu().numpy()
    std_rbf_np = std_rbf.cpu().numpy()
    util_std_rbf_np = util_std_rbf.cpu().numpy()
    util_da_rbf_np = util_da_rbf.cpu().numpy() if util_da_rbf is not None else None

    # After conditioning
    mu_after_rbf_np = mu_after_rbf.cpu().numpy()
    std_after_rbf_np = std_after_rbf.cpu().numpy()
    mu_after_acos_np = mu_after_acos.cpu().numpy()
    std_after_acos_np = std_after_acos.cpu().numpy()

    # Kernel parameters
    sigma_0 = model_acos.covar_module.sigma_0.item()
    Amp_acos = model_acos.covar_module.Amp.item()
    k_xx_acos = x_np**2 + sigma_0**2  # Arc-Cosine: k(x,x) = x² + σ₀²

    ls_rbf = model_rbf.covar_module.base_kernel.lengthscale.item()
    os_rbf = model_rbf.covar_module.outputscale.item()
    k_xx_rbf = np.ones_like(x_np) * os_rbf  # RBF: k(x,x) = outputscale (constant)

    # Create 2x3 figure (added column for before/after conditioning)
    fig, axes = plt.subplots(2, 3, figsize=(23, 10))

    # Condition numbers
    cond_acos = check_kernel_health(model_acos, inducing_points, context="Arc-Cosine (plot)")
    cond_rbf = check_kernel_health(model_rbf, inducing_points, context="RBF (plot)")

    # =========================================================================
    # Row 1: Arc-Cosine Kernel
    # =========================================================================

    # Subplot 1: Arc-Cosine Latent Function
    ax1 = axes[0, 0]
    ax1.plot(x_np, true_np, 'k--', label='True λ(x)', linewidth=2)
    ax1.plot(x_np, mean_acos_np, 'b-', label='GP mean', linewidth=2)
    ax1.fill_between(x_np, mean_acos_np - 2*std_acos_np, mean_acos_np + 2*std_acos_np,
                     alpha=0.3, color='blue', label='±2σ')
    # Plot training points as vertical lines
    if train_x_np is not None:
        for i, tx in enumerate(train_x_np):
            ax1.axvline(x=tx, color='red', alpha=0.3, linewidth=1,
                       label='Training' if i == 0 else None)
    mean_acos = model_acos.mean_module.constant.item()
    ax1.set_ylabel('λ(x) = log(f)', fontsize=11)
    ax1.set_title(f'Arc-Cosine (C=I)\nσ₀={sigma_0:.3f}, Amp={Amp_acos:.3f}, mean={mean_acos:.3f}, cond={cond_acos:.1e}',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # k(x,x) on secondary axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_np, k_xx_acos, 'purple', linestyle=':', linewidth=2, alpha=0.7, label='k(x,x)')
    ax1_twin.set_ylabel('k(x,x) = x² + σ₀²', color='purple', fontsize=9)
    ax1_twin.tick_params(axis='y', labelcolor='purple')
    ax1_twin.legend(loc='upper right', fontsize=8)

    # Subplot 2: Arc-Cosine Utility
    ax2 = axes[0, 1]
    if p_x_mean is not None and p_x_std is not None:
        p_x = np.exp(-0.5 * ((x_np - p_x_mean) / p_x_std) ** 2)
        p_x_scaled = p_x / p_x.max() * util_std_acos_np.max() * 0.5
        ax2.fill_between(x_np, 0, p_x_scaled, alpha=0.15, color='orange', label='p(x)')
    ax2.plot(x_np, util_std_acos_np, 'g-', linewidth=2, label='Standard utility')

    ax2.fill_between(x_np, 0, util_std_acos_np, alpha=0.2, color='green')
    ax2.set_ylabel('Standard Utility', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    if util_da_acos_np is not None:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x_np, util_da_acos_np, 'b-', linewidth=2, label='Distr-aware')
        ax2_twin.fill_between(x_np, 0, util_da_acos_np, alpha=0.2, color='blue')
        ax2_twin.set_ylabel('Distr-Aware Utility', color='blue')
        ax2_twin.tick_params(axis='y', labelcolor='blue')

        max_idx = np.argmax(util_da_acos_np)
        ax2_twin.scatter([x_np[max_idx]], [util_da_acos_np[max_idx]], c='darkblue', s=100, marker='*', zorder=5)

    # Plot training points as vertical lines
    if train_x_np is not None:
        for i, tx in enumerate(train_x_np):
            ax2.axvline(x=tx, color='red', alpha=0.3, linewidth=1,
                       label='Training' if i == 0 else None)
    max_idx_std = np.argmax(util_std_acos_np)
    ax2.scatter([x_np[max_idx_std]], [util_std_acos_np[max_idx_std]], c='darkgreen', s=100, marker='*', zorder=5)
    ax2.set_title('Arc-Cosine Utility\nU(x*) = H_marg - H_cond', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=9)

    # =========================================================================
    # Row 2: RBF Kernel
    # =========================================================================

    # Subplot 3: RBF Latent Function
    ax3 = axes[1, 0]
    ax3.plot(x_np, true_np, 'k--', label='True λ(x)', linewidth=2)
    ax3.plot(x_np, mean_rbf_np, 'b-', label='GP mean', linewidth=2)
    ax3.fill_between(x_np, mean_rbf_np - 2*std_rbf_np, mean_rbf_np + 2*std_rbf_np,
                     alpha=0.3, color='blue', label='±2σ')
    # Plot training points as vertical lines
    if train_x_np is not None:
        for i, tx in enumerate(train_x_np):
            ax3.axvline(x=tx, color='red', alpha=0.3, linewidth=1,
                       label='Training' if i == 0 else None)
    mean_rbf = model_rbf.mean_module.constant.item()
    ax3.set_ylabel('λ(x) = log(f)', fontsize=11)
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_title(f'RBF\nlengthscale={ls_rbf:.3f}, outputscale={os_rbf:.3f}, mean={mean_rbf:.3f}, cond={cond_rbf:.1e}',
                  fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # k(x,x) on secondary axis (constant for RBF)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x_np, k_xx_rbf, 'purple', linestyle=':', linewidth=2, alpha=0.7, label='k(x,x)')
    ax3_twin.set_ylabel('k(x,x) = σ² (constant)', color='purple', fontsize=9)
    ax3_twin.tick_params(axis='y', labelcolor='purple')
    ax3_twin.legend(loc='upper right', fontsize=8)

    # Subplot 4: RBF Utility
    ax4 = axes[1, 1]
    if p_x_mean is not None and p_x_std is not None:
        p_x_scaled = p_x / p_x.max() * util_std_rbf_np.max() * 0.5
        ax4.fill_between(x_np, 0, p_x_scaled, alpha=0.15, color='orange', label='p(x)')
    ax4.plot(x_np, util_std_rbf_np, 'g-', linewidth=2, label='Standard utility')
    ax4.fill_between(x_np, 0, util_std_rbf_np, alpha=0.2, color='green')
    ax4.set_ylabel('Standard Utility', color='green', fontsize=10)
    ax4.tick_params(axis='y', labelcolor='green')

    if util_da_rbf_np is not None:
        ax4_twin = ax4.twinx()
        ax4_twin.plot(x_np, util_da_rbf_np, 'b-', linewidth=2, label='Distr-aware')
        ax4_twin.fill_between(x_np, 0, util_da_rbf_np, alpha=0.2, color='blue')
        ax4_twin.set_ylabel('Distr-Aware Utility', color='blue')
        ax4_twin.tick_params(axis='y', labelcolor='blue')

        max_idx = np.argmax(util_da_rbf_np)
        ax4_twin.scatter([x_np[max_idx]], [util_da_rbf_np[max_idx]], c='darkblue', s=100, marker='*', zorder=5)

    # Plot training points as vertical lines
    if train_x_np is not None:
        for i, tx in enumerate(train_x_np):
            ax4.axvline(x=tx, color='red', alpha=0.3, linewidth=1,
                       label='Training' if i == 0 else None)
    max_idx_std = np.argmax(util_std_rbf_np)
    ax4.scatter([x_np[max_idx_std]], [util_std_rbf_np[max_idx_std]], c='darkgreen', s=100, marker='*', zorder=5)
    ax4.set_title('RBF Utility\nU(x*) = H_marg - H_cond', fontsize=12, fontweight='bold')
    ax4.set_xlabel('x', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left', fontsize=9)

    # =========================================================================
    # Column 3: Before/After Conditioning Comparison
    # =========================================================================

    # Subplot 5: Arc-Cosine Before vs After Conditioning
    ax5 = axes[0, 2]

    # True function
    ax5.plot(x_np, true_np, 'k--', linewidth=2, label='True λ(x)', zorder=1)

    # BEFORE conditioning (blue)
    ax5.plot(x_np, mean_acos_np, 'b-', linewidth=2, label='Before: GP mean', zorder=2)
    ax5.fill_between(x_np,
                     mean_acos_np - 2*std_acos_np,
                     mean_acos_np + 2*std_acos_np,
                     alpha=0.25, color='blue', label='Before: ±2σ', zorder=1)

    # AFTER conditioning (red)
    ax5.plot(x_np, mu_after_acos_np, 'r-', linewidth=2, label='After: GP mean', zorder=3)
    ax5.fill_between(x_np,
                     mu_after_acos_np - 2*std_after_acos_np,
                     mu_after_acos_np + 2*std_after_acos_np,
                     alpha=0.25, color='red', label='After: ±2σ', zorder=1)

    # Mark observation point
    ax5.axvline(x=X_SAMPLE, color='green', linewidth=2, linestyle='--', zorder=4)
    ax5.scatter([X_SAMPLE], [lambda_obs], c='green', s=150, marker='*',
                zorder=5, edgecolors='darkgreen', linewidths=2,
                label=f'λ_obs = {lambda_obs:.2f}')

    # Training points as vertical lines
    if train_x_np is not None:
        for i, tx in enumerate(train_x_np):
            ax5.axvline(x=tx, color='gray', alpha=0.3, linewidth=1, linestyle=':',
                       label='Training' if i == 0 else None)

    ax5.set_ylabel('λ(x)', fontsize=11)
    ax5.set_title(f'Arc-Cosine: Before vs After Conditioning\non λ({X_SAMPLE}) = {lambda_obs:.2f}',
                  fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Subplot 6: RBF Before vs After Conditioning
    ax6 = axes[1, 2]

    # True function
    ax6.plot(x_np, true_np, 'k--', linewidth=2, label='True λ(x)', zorder=1)

    # BEFORE conditioning (blue)
    ax6.plot(x_np, mean_rbf_np, 'b-', linewidth=2, label='Before: GP mean', zorder=2)
    ax6.fill_between(x_np,
                     mean_rbf_np - 2*std_rbf_np,
                     mean_rbf_np + 2*std_rbf_np,
                     alpha=0.25, color='blue', label='Before: ±2σ', zorder=1)

    # AFTER conditioning (red)
    ax6.plot(x_np, mu_after_rbf_np, 'r-', linewidth=2, label='After: GP mean', zorder=3)
    ax6.fill_between(x_np,
                     mu_after_rbf_np - 2*std_after_rbf_np,
                     mu_after_rbf_np + 2*std_after_rbf_np,
                     alpha=0.25, color='red', label='After: ±2σ', zorder=1)

    # Mark observation point
    ax6.axvline(x=X_SAMPLE, color='green', linewidth=2, linestyle='--', zorder=4)
    ax6.scatter([X_SAMPLE], [lambda_obs], c='green', s=150, marker='*',
                zorder=5, edgecolors='darkgreen', linewidths=2,
                label=f'λ_obs = {lambda_obs:.2f}')

    # Training points as vertical lines
    if train_x_np is not None:
        for i, tx in enumerate(train_x_np):
            ax6.axvline(x=tx, color='gray', alpha=0.3, linewidth=1, linestyle=':',
                       label='Training' if i == 0 else None)

    ax6.set_ylabel('λ(x)', fontsize=11)
    ax6.set_xlabel('x', fontsize=11)
    ax6.set_title(f'RBF: Before vs After Conditioning\non λ({X_SAMPLE}) = {lambda_obs:.2f}',
                  fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)

    # Main title
    fig.suptitle('Arc-Cosine (NON-STATIONARY) vs RBF (STATIONARY) Kernel Comparison\n'
                 f'Likelihood: Poisson(exp(f))  |  Domain: [{X_MIN}, {X_MAX}]',
                 fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()
    if save_path is None:
        save_path = Path(__file__).parent / 'kernel_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")
    return fig


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
# Paths to saved checkpoints
RBF_CHECKPOINT_PATH = Path(__file__).parent.parent / 'trained_rbf_checkpoint.pt'
ACOS_CHECKPOINT_PATH = Path(__file__).parent / 'trained_acos_checkpoint.pt'


def main():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("Arc-Cosine vs RBF Kernel Comparison (Both Trained)")
    print("=" * 60)

    # =========================================================================
    # Model 1: RBF Kernel (load trained model)
    # =========================================================================
    print("\n[1] Loading trained RBF model...")
    model_rbf, _, inducing_points, _, train_x, train_y = load_rbf_checkpoint(RBF_CHECKPOINT_PATH)

    # =========================================================================
    # Model 2: Arc-Cosine Kernel (load trained model)
    # =========================================================================
    print("\n[2] Loading trained Arc-Cosine model...")
    model_acos, _, _, _, _, _ = load_acos_checkpoint(ACOS_CHECKPOINT_PATH)

    # =========================================================================
    # Plot comparison
    # =========================================================================
    print("\n[3] Plotting comparison...")
    plot_comparison(model_acos, model_rbf, inducing_points, lambda_true,
                    p_x_mean=DEFAULT_P_X_MEAN, p_x_std=DEFAULT_P_X_STD,
                    train_x=train_x, train_y=train_y)

    print("\nDone!")


if __name__ == "__main__":
    main()
