"""
Diagnostic: Posterior Before/After Conditioning on a Single Observation
========================================================================

This script loads a pre-trained RBF GP and analyzes how conditioning on
observations affects predictions.

Produces 4 vertically stacked plots:
1. GP Posterior Before vs After Conditioning (single sample)
2. GP Mean Before vs E[After] (MC averaged) + |Δμ|
3. GP Variance Before vs E[After] (MC averaged) + |Δσ²|
4. Entropy Components (H_marg, H_cond, Utility)

The `compute_mc_diagnostics()` function is defined here and can be imported
by other scripts (e.g., diagnose_arccosine_utility.py).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from datetime import datetime

# Add Spatial_GP_repo to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gp_utility_playground import (
    VariationalGP, PoissonLikelihood,
    lambda_true,
    get_marginal_moments, get_conditional_moments, compute_H,
    DEVICE, DTYPE, X_MIN, X_MAX,
    DEFAULT_P_X_MEAN, DEFAULT_P_X_STD,
    # Conditioning/MC parameters (can be overridden locally if needed)
    X_SAMPLE, LAMBDA_SEED, N_LAMBDA_SAMPLES, SAMPLE_X
)


# =============================================================================
# Paths
# =============================================================================
RBF_CHECKPOINT_PATH = Path(__file__).parent / 'trained_rbf_checkpoint.pt'


# =============================================================================
# Script-specific Configuration (aliases for clarity)
# =============================================================================
# Extended domain for visualization
X_PLOT_MIN = X_MIN
X_PLOT_MAX = X_MAX

# Gaussian p(x) distribution parameters
GAUSSIAN_MEAN = DEFAULT_P_X_MEAN
GAUSSIAN_STD = DEFAULT_P_X_STD


# =============================================================================
# Checkpoint Loading
# =============================================================================
def load_rbf_checkpoint(checkpoint_path):
    """Load trained RBF GP from checkpoint with full metadata.

    Checkpoint contains:
    - model_state_dict: Full model state (hyperparams + variational params m, V)
    - config: Training configuration (seed, n_train, x_min, x_max, etc.)
    - inducing_points: Exact inducing points used during training
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

    # Extract training data if available
    train_x = checkpoint.get('train_x', None)
    train_y = checkpoint.get('train_y', None)
    if train_x is not None:
        train_x = train_x.to(dtype=DTYPE, device=DEVICE)
        train_y = train_y.to(dtype=DTYPE, device=DEVICE)
        print(f"  Training data: {len(train_x)} points, y in [{train_y.min():.0f}, {train_y.max():.0f}]")

    return model, likelihood, inducing_points, config, train_x, train_y


# =============================================================================
# Helper Functions (can be imported by other scripts)
# =============================================================================
def get_posterior_at_points(model, x_points):
    """Get posterior mean and variance at given points."""
    model.eval()
    with torch.no_grad():
        posterior = model(x_points)
        return posterior.mean.clone(), posterior.variance.clone()


def compute_mc_diagnostics(
    model,
    candidates,
    mu_at_sample_scalar,
    std_at_sample,
    n_mc_samples,
    sample_x,
    x_sample,
    gaussian_mean,
    gaussian_std,
):
    """
    Compute all MC statistics in a single loop.

    Computes utility components (H_marg, H_cond) and averaged conditional moments
    (mu_cond_avg, sigma2_cond_avg) for diagnostic plots.

    Args:
        model: Trained GP model
        candidates: Query points (K,)
        mu_at_sample_scalar: GP mean at x_sample (scalar, for sample_x=False case)
        std_at_sample: GP std at x_sample (scalar, for sample_x=False case)
        n_mc_samples: Number of MC samples
        sample_x: If True, sample x from Gaussian p(x); if False, use fixed x_sample
        x_sample: Fixed observation point (used when sample_x=False)
        gaussian_mean: Mean of Gaussian p(x)
        gaussian_std: Std of Gaussian p(x)

    Returns:
        H_marg: Marginal entropy H(R | x*, D)
        H_cond: Expected conditional entropy E[H(R | x*, λ(x), D)]
        mu_cond_avg: Average conditional mean E[μ(x* | λ(x))]
        sigma2_cond_avg: Average conditional variance E[σ²(x* | λ(x))]
    """
    # Import dependencies (allows this function to be imported elsewhere)
    from gp_utility_playground import get_marginal_moments, get_conditional_moments, compute_H

    model.eval()
    device = candidates.device
    dtype = candidates.dtype

    # Compute H_marg once (no loop needed)
    mu_marg, sigma2_marg = get_marginal_moments(model, candidates)
    H_marg = compute_H(mu_marg, sigma2_marg)

    # Initialize accumulators
    H_cond_sum = torch.zeros_like(H_marg)
    mu_cond_sum = torch.zeros_like(mu_marg)
    sigma2_cond_sum = torch.zeros_like(sigma2_marg)

    # Single MC loop - accumulates both entropy AND moments
    with torch.no_grad():
        for _ in range(n_mc_samples):
            # Step 1: Get posterior predictive moments at x_i
            if sample_x:
                # Sample x from Gaussian p(x)
                x_i = gaussian_mean + gaussian_std * torch.randn(1, dtype=dtype, device=device).item()
                x_i_tensor = torch.tensor([x_i], dtype=dtype, device=device)
                post_i = model(x_i_tensor)
                mu_i = post_i.mean[0]
                std_i = post_i.variance[0].sqrt()
            else:
                # Use fixed x_sample - moments already computed
                x_i = x_sample
                mu_i = mu_at_sample_scalar
                std_i = std_at_sample

            # Step 2: Sample λ_i from posterior predictive at x_i
            lambda_i = (mu_i + std_i * torch.randn(1, dtype=dtype, device=device)).item()

            # Step 3: Update posterior at ALL query points x* after "observing" λ_i
            mu_cond_i, sigma2_cond_i = get_conditional_moments(model, candidates, x_i, lambda_i)

            # Step 4: Accumulate statistics for averaging
            H_cond_i = compute_H(mu_cond_i, sigma2_cond_i)
            H_cond_sum += H_cond_i
            mu_cond_sum += mu_cond_i
            sigma2_cond_sum += sigma2_cond_i

    # Average all accumulated quantities
    H_cond = H_cond_sum / n_mc_samples
    mu_cond_avg = mu_cond_sum / n_mc_samples
    sigma2_cond_avg = sigma2_cond_sum / n_mc_samples

    return H_marg, H_cond, mu_cond_avg, sigma2_cond_avg


# =============================================================================
# Main
# =============================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("Diagnostic: Posterior Before/After Conditioning (RBF)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load pre-trained RBF model
    # -------------------------------------------------------------------------
    print("\nLoading trained RBF model...")
    model, likelihood, inducing_points, config, train_x, train_y = load_rbf_checkpoint(RBF_CHECKPOINT_PATH)

    # Get hyperparameters for display
    lengthscale = model.covar_module.base_kernel.lengthscale.item()
    outputscale = model.covar_module.outputscale.item()
    mean_const = model.mean_module.constant.item()

    print(f"\nObservation point x_sample: {X_SAMPLE}")
    print(f"Plot domain: [{X_PLOT_MIN}, {X_PLOT_MAX}]")

    # -------------------------------------------------------------------------
    # Get posterior BEFORE conditioning
    # -------------------------------------------------------------------------
    x_plot = torch.linspace(X_PLOT_MIN, X_PLOT_MAX, 500, dtype=DTYPE, device=DEVICE)
    mu_before, var_before = get_posterior_at_points(model, x_plot)
    std_before = var_before.sqrt()

    # -------------------------------------------------------------------------
    # Sample observation at X_SAMPLE from GP posterior
    # -------------------------------------------------------------------------
    mu_at_sample, var_at_sample = get_posterior_at_points(
        model, torch.tensor([X_SAMPLE], dtype=DTYPE, device=DEVICE)
    )

    torch.manual_seed(LAMBDA_SEED)
    mu_at_sample_scalar = mu_at_sample[0]
    std_at_sample = var_at_sample[0].sqrt()
    eps = torch.randn(1, dtype=DTYPE, device=DEVICE)
    lambda_obs = (mu_at_sample_scalar + std_at_sample * eps).item()

    print(f"\nObservation at x={X_SAMPLE}:")
    print(f"  Posterior mean: {mu_at_sample_scalar.item():.4f}")
    print(f"  Posterior std: {std_at_sample.item():.4f}")
    print(f"  Sampled λ_obs = {lambda_obs:.4f} (seed={LAMBDA_SEED})")

    # -------------------------------------------------------------------------
    # Compute all MC statistics in a single loop
    # -------------------------------------------------------------------------
    if SAMPLE_X:
        print(f"\nComputing utility (MC: {N_LAMBDA_SAMPLES} samples, x ~ N({GAUSSIAN_MEAN}, {GAUSSIAN_STD}²))...")
    else:
        print(f"\nComputing utility (MC: {N_LAMBDA_SAMPLES} λ samples at fixed x={X_SAMPLE})...")

    H_marg, H_cond, mu_after_avg, sigma2_after_avg = compute_mc_diagnostics(
        model, x_plot, mu_at_sample_scalar, std_at_sample, N_LAMBDA_SAMPLES,
        SAMPLE_X, X_SAMPLE, GAUSSIAN_MEAN, GAUSSIAN_STD
    )
    utility = H_marg - H_cond

    # For Plot 1: single sample (lambda_obs at X_SAMPLE)
    mu_after_single, sigma2_after_single = get_conditional_moments(model, x_plot, X_SAMPLE, lambda_obs)
    std_after_single = sigma2_after_single.sqrt()

    print(f"  Utility range: [{utility.min():.3f}, {utility.max():.3f}]")

    # -------------------------------------------------------------------------
    # Convert to numpy for plotting
    # -------------------------------------------------------------------------
    x_np = x_plot.cpu().numpy()
    H_marg_np = H_marg.cpu().numpy()
    H_cond_np = H_cond.cpu().numpy()
    utility_np = utility.cpu().numpy()

    mu_before_np = mu_before.cpu().numpy()
    std_before_np = std_before.cpu().numpy()
    var_before_np = var_before.cpu().numpy()
    mu_after_single_np = mu_after_single.cpu().numpy()
    std_after_single_np = std_after_single.cpu().numpy()
    mu_after_avg_np = mu_after_avg.cpu().numpy()
    sigma2_after_avg_np = sigma2_after_avg.cpu().numpy()
    train_x_np = train_x.cpu().numpy() if train_x is not None else None

    true_lambda_np = lambda_true(x_plot).cpu().numpy()

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)

    # =========================================================================
    # Plot 1: GP Posterior Mean and Variance (Before vs After) - Single sample
    # =========================================================================
    ax1 = axes[0]

    ax1.plot(x_np, true_lambda_np, 'k--', linewidth=2, label='True λ(x)', zorder=1)
    ax1.plot(x_np, mu_before_np, 'b-', linewidth=2, label='Before: GP mean', zorder=2)
    ax1.fill_between(x_np, mu_before_np - 2*std_before_np, mu_before_np + 2*std_before_np,
                     alpha=0.25, color='blue', label='Before: ±2σ', zorder=1)
    ax1.plot(x_np, mu_after_single_np, 'r-', linewidth=2, label='After: GP mean', zorder=3)
    ax1.fill_between(x_np, mu_after_single_np - 2*std_after_single_np, mu_after_single_np + 2*std_after_single_np,
                     alpha=0.25, color='red', label='After: ±2σ', zorder=1)

    ax1.axvline(x=X_SAMPLE, color='green', linewidth=2, linestyle='--', label=f'x_sample = {X_SAMPLE}', zorder=4)
    ax1.scatter([X_SAMPLE], [lambda_obs], c='green', s=150, marker='*',
                zorder=5, edgecolors='darkgreen', linewidths=2, label=f'Observed λ = {lambda_obs:.2f}')

    if train_x_np is not None:
        for i, tx in enumerate(train_x_np):
            ax1.axvline(x=tx, color='gray', alpha=0.3, linewidth=1, linestyle=':', label='Training data' if i == 0 else None)

    ax1.set_ylabel('λ(x)')
    ax1.set_title(f'RBF GP: Before vs After Conditioning on λ({X_SAMPLE}) = {lambda_obs:.2f}\n'
                  f'(lengthscale={lengthscale:.3f}, outputscale={outputscale:.3f}, mean={mean_const:.3f})')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 2: GP Mean Before/After with Absolute Difference (MC averaged)
    # =========================================================================
    ax2 = axes[1]

    ax2.plot(x_np, mu_before_np, 'b-', linewidth=2, label='Before conditioning')
    ax2.plot(x_np, mu_after_avg_np, 'r-', linewidth=2, label=f'After (avg over {N_LAMBDA_SAMPLES} samples)')

    mu_diff = np.abs(mu_after_avg_np - mu_before_np)
    ax2_twin = ax2.twinx()
    ax2_twin.fill_between(x_np, 0, mu_diff, alpha=0.3, color='green')
    ax2_twin.plot(x_np, mu_diff, 'g-', linewidth=1.5, alpha=0.8, label='|Δμ|')
    ax2_twin.set_ylabel('|Δμ|', color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')

    ax2.axvline(x=X_SAMPLE, color='green', linewidth=2, linestyle='--')
    ax2.set_ylabel('GP Mean μ(x)')
    ax2.set_title('GP Predicted Mean: Before vs E[After] (MC averaged)')
    ax2.legend(loc='upper left', fontsize=8)
    ax2_twin.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 3: GP Variance Before/After with Absolute Difference (MC averaged)
    # =========================================================================
    ax3 = axes[2]

    ax3.plot(x_np, var_before_np, 'b-', linewidth=2, label='Before conditioning')
    ax3.plot(x_np, sigma2_after_avg_np, 'r-', linewidth=2, label=f'After (avg over {N_LAMBDA_SAMPLES} samples)')

    var_diff = np.abs(sigma2_after_avg_np - var_before_np)
    ax3_twin = ax3.twinx()
    ax3_twin.fill_between(x_np, 0, var_diff, alpha=0.3, color='green')
    ax3_twin.plot(x_np, var_diff, 'g-', linewidth=1.5, alpha=0.8, label='|Δσ²|')
    ax3_twin.set_ylabel('|Δσ²|', color='green')
    ax3_twin.tick_params(axis='y', labelcolor='green')

    ax3.axvline(x=X_SAMPLE, color='green', linewidth=2, linestyle='--')
    ax3.set_ylabel('GP Variance σ²(x)')
    ax3.set_title('GP Predicted Variance: Before vs E[After] (MC averaged)')
    ax3.legend(loc='upper left', fontsize=8)
    ax3_twin.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 4: Entropy Components (H_marginal, H_conditional, and Utility)
    # =========================================================================
    ax4 = axes[3]

    ax4.plot(x_np, H_marg_np, 'b-', linewidth=2, label='H_marg(x*)')
    ax4.plot(x_np, H_cond_np, 'r-', linewidth=2, label=f'E[H_cond(x* | λ)] ({N_LAMBDA_SAMPLES} samples)')

    ax4_twin = ax4.twinx()
    ax4_twin.plot(x_np, utility_np, 'g-', linewidth=2, alpha=0.8, label='Utility = H_marg - H_cond')
    ax4_twin.fill_between(x_np, 0, utility_np, alpha=0.15, color='green')
    ax4_twin.set_ylabel('Utility', color='green')
    ax4_twin.tick_params(axis='y', labelcolor='green')

    ax4.axvline(x=X_SAMPLE, color='green', linewidth=2, linestyle='--')

    # Show p(x) for reference (scaled)
    if SAMPLE_X:
        # True distribution: Gaussian p(x)
        gaussian_pdf = np.exp(-0.5 * ((x_np - GAUSSIAN_MEAN) / GAUSSIAN_STD)**2)
        H_max = max(H_marg_np.max(), H_cond_np.max())
        gaussian_pdf_scaled = gaussian_pdf / gaussian_pdf.max() * H_max * 0.3
        ax4.fill_between(x_np, 0, gaussian_pdf_scaled, alpha=0.2, color='orange',
                         label=f'p(x) ~ N({GAUSSIAN_MEAN}, {GAUSSIAN_STD}²)')
    else:
        # Effective distribution: Dirac delta at X_SAMPLE (shown as very narrow Gaussian)
        narrow_std = 0.05  # Very narrow to approximate delta function
        delta_pdf = np.exp(-0.5 * ((x_np - X_SAMPLE) / narrow_std)**2)
        H_max = max(H_marg_np.max(), H_cond_np.max())
        delta_pdf_scaled = delta_pdf / delta_pdf.max() * H_max * 0.3
        ax4.fill_between(x_np, 0, delta_pdf_scaled, alpha=0.3, color='orange',
                         label=f'p(x) ~ δ(x - {X_SAMPLE})')

    ax4.set_xlabel('x')
    ax4.set_ylabel('Entropy (nats)')
    if SAMPLE_X:
        title_suffix = f'MC: {N_LAMBDA_SAMPLES} samples, x ~ N({GAUSSIAN_MEAN}, {GAUSSIAN_STD}²)'
    else:
        title_suffix = f'MC: {N_LAMBDA_SAMPLES} λ samples at fixed x={X_SAMPLE}'
    ax4.set_title(f'Utility: U(x*) = H_marg(x*) - E[H_cond(x* | x, λ)]\n{title_suffix}')
    ax4.legend(loc='upper left', fontsize=8)
    ax4_twin.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    save_path = Path(__file__).parent / 'distribution_aware_1d_fixed_result.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved plot to: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
