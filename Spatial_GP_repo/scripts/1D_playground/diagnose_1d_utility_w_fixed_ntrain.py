"""
Diagnostic: Posterior Before/After Conditioning on a Single Observation
========================================================================

USE:
How much does conditioning on a sample x influence the mean and variance of the predictive?
in the frist plot we show this for one only x sample
in the second and third we shouw the average cange in mean and variance averaged over multiple samples
are these samples actually drawn the distribution p(x) ? we can set that with the SAMPLE_X flag
in the last plot we show the utility, averaged over samples.

Note:
Utility can go negative if we use low lambda samples number

Bug Resolved:
The utility exploded outside of the borders of the distribution from which we sample the monte carlo estimate of H_cond.

Turns out that using GPytorch estimates for m and V with out formulas for the predictive distribution was incorrect,
because GPytorch uses a whitened representation internally. ( m_gpy = K^{1/2} m_manual, V_gpy = K^{1/2} V_manual K^{1/2} ).
This is to save memory.

This script:
1. Trains GP with FIXED lengthscale (not learned)
2. Shows posterior mean and variance across extended domain - both before and after conditioning. 
   Note the conditioning is shown for a SINGLE sampled observation from the GP posterior at x_sample.
3. Conditions on a single new observation at x_sample
4. Shows H_marginal and H_conditional components of the utility
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add Spatial_GP_repo to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gp_utility_playground import (
    VariationalGP, PoissonLikelihood,
    generate_poisson_data, lambda_true,
    evaluate_nd_utility_new, 
    get_marginal_moments, get_conditional_moments, compute_H,
    remove_near_duplicates, train_gp_fixed_lengthscale,
    DEVICE, DTYPE, X_MIN, X_MAX, MAX_R,
    DEFAULT_P_X_MEAN, DEFAULT_P_X_STD
)


# =============================================================================
# Script-specific Configuration
# =============================================================================
# GP hyperparameters (fixed for diagnostic purposes)
FIXED_LENGTHSCALE = 0.2
FIXED_OUTPUTSCALE = 1.0

# Where to sample the new observation (inside Gaussian p(x) peak)
X_SAMPLE = 0.0

# Extended domain for visualization (uses X_MIN, X_MAX from gp_utility_playground)
X_PLOT_MIN = X_MIN
X_PLOT_MAX = X_MAX

# Gaussian p(x) distribution parameters
GAUSSIAN_MEAN = DEFAULT_P_X_MEAN
GAUSSIAN_STD = DEFAULT_P_X_STD

# Entropy computation parameters (MAX_R imported from gp_utility_playground)
A = 1.0       # Firing rate scaling (f = exp(A*lambda + lambda0))
LAMBDA0 = 0.0 # Firing rate offset
N_LAMBDA_SAMPLES = 100  # Number of MC samples for averaging

# Sampling flags
SAMPLE_X = True        # If True, sample x from Gaussian p(x); if False, use fixed X_SAMPLE
SAMPLE_TRAIN_X = True  # If True, sample training x from Gaussian p(x); if False, use uniform grid

# Utility function selection
USE_DISTRIBUTION_AWARE = True  # If True, use distribution-aware utility; if False, use standard nd_utility


# =============================================================================
# Script-specific utility wrapper (uses SAMPLE_X flag)
# =============================================================================
def _compute_diagnostic_utility(model, candidates, n_mc_samples=N_LAMBDA_SAMPLES):
    """
    Compute distribution-aware utility at each candidate point.

    U(x*) = H_marg(x*) - E[H_cond(x* | x, λ)]

    Where expectation is over x ~ p(x), λ ~ q(λ|x).

    Args:
        model: Trained GP model
        candidates: (n_candidates,) query points
        n_mc_samples: Number of MC samples for expectation

    Returns:
        utility: (n_candidates,) utility at each candidate
    """
    model.eval()
    device = candidates.device
    dtype = candidates.dtype

    # Marginal entropy
    mu_marg, sigma2_marg = get_marginal_moments(model, candidates)
    H_marg = compute_H(mu_marg, sigma2_marg)

    # Average conditional entropy over MC samples
    H_cond_sum = torch.zeros_like(H_marg)

    with torch.no_grad():
        for _ in range(n_mc_samples):
            if SAMPLE_X:
                # Sample x from Gaussian p(x)
                x_i = GAUSSIAN_MEAN + GAUSSIAN_STD * torch.randn(1, dtype=dtype, device=device).item()
                x_i_tensor = torch.tensor([x_i], dtype=dtype, device=device)
                post_i = model(x_i_tensor)
                mu_i = post_i.mean[0]
                std_i = post_i.variance[0].sqrt()
            else:
                # Use fixed X_SAMPLE
                x_i = X_SAMPLE
                x_i_tensor = torch.tensor([x_i], dtype=dtype, device=device)
                post_i = model(x_i_tensor)
                mu_i = post_i.mean[0]
                std_i = post_i.variance[0].sqrt()

            # Sample lambda from GP posterior at x_i
            lambda_i = (mu_i + std_i * torch.randn(1, dtype=dtype, device=device)).item()

            # Conditional entropy
            mu_cond_i, sigma2_cond_i = get_conditional_moments(model, candidates, x_i, lambda_i)
            H_cond_i = compute_H(mu_cond_i, sigma2_cond_i)
            H_cond_sum += H_cond_i

    H_cond = H_cond_sum / n_mc_samples
    utility = H_marg - H_cond

    return utility


def get_posterior_at_points(model, x_points):
    """Get posterior mean and variance at given points."""
    model.eval()
    with torch.no_grad():
        posterior = model(x_points)
        return posterior.mean.clone(), posterior.variance.clone()


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("Diagnostic: Posterior Before/After Conditioning")
    print("=" * 70)
    print(f"\nFixed lengthscale: {FIXED_LENGTHSCALE}")
    print(f"Observation point x_sample: {X_SAMPLE}")
    print(f"Training domain: [{X_MIN}, {X_MAX}]")
    print(f"Plot domain: [{X_PLOT_MIN}, {X_PLOT_MAX}]")

    # -------------------------------------------------------------------------
    # Create training data
    # -------------------------------------------------------------------------
    n_train = 100
    if SAMPLE_TRAIN_X:
        # Sample training x from Gaussian p(x)
        train_x = GAUSSIAN_MEAN + GAUSSIAN_STD * torch.randn(n_train, dtype=DTYPE, device=DEVICE)
        train_x = train_x.sort()[0]  # Sort for nicer visualization
        print(f"\nTraining points: n={n_train} sampled from N({GAUSSIAN_MEAN}, {GAUSSIAN_STD}²)")
    else:
        train_x = torch.linspace(X_MIN, X_MAX, n_train, dtype=DTYPE, device=DEVICE)
        print(f"\nTraining points: n={n_train} uniform in [{X_MIN}, {X_MAX}]")
    train_y = generate_poisson_data(train_x, lambda_true)

    # Remove near-duplicate points to keep kernel well-conditioned
    if SAMPLE_TRAIN_X:
        n_before = len(train_x)
        train_x, train_y = remove_near_duplicates(train_x, train_y, min_dist=1e-4)
        n_removed = n_before - len(train_x)
        if n_removed > 0:
            print(f"  Removed {n_removed} near-duplicate points (min_dist=1e-4)")

    # -------------------------------------------------------------------------
    # Train GP with FIXED lengthscale
    # -------------------------------------------------------------------------
    inducing_points = train_x.clone()
    model = VariationalGP(inducing_points, jitter=1e-4).to(DEVICE)
    likelihood = PoissonLikelihood().to(DEVICE)

    print("\nTraining GP with fixed lengthscale...")
    model, likelihood = train_gp_fixed_lengthscale(
        model, likelihood, train_x, train_y, n_iterations=500
    )

    # -------------------------------------------------------------------------
    # Get posterior BEFORE conditioning
    # -------------------------------------------------------------------------
    x_plot = torch.linspace(X_PLOT_MIN, X_PLOT_MAX, 500, dtype=DTYPE, device=DEVICE)
    mu_before, var_before = get_posterior_at_points(model, x_plot)
    std_before = var_before.sqrt()

    # -------------------------------------------------------------------------
    # Sample observation at x_sample from GP posterior (Gaussian, not Poisson!)
    # λ(x) ~ N(μ(x), σ²(x)) - this is the latent function, not the response R
    # -------------------------------------------------------------------------
    mu_at_sample, var_at_sample = get_posterior_at_points(
        model, torch.tensor([X_SAMPLE], dtype=DTYPE, device=DEVICE)
    )

    # Set seed for reproducibility - same seed will be used for verification
    LAMBDA_SEED = 12
    torch.manual_seed(LAMBDA_SEED)

    # Sample λ(x_sample) from Gaussian GP posterior: λ ~ N(μ, σ²)
    std_at_sample = var_at_sample[0].sqrt()
    eps = torch.randn(1, dtype=DTYPE, device=DEVICE)
    lambda_obs = (mu_at_sample[0] + std_at_sample * eps).item()

    print(f"\nObservation at x={X_SAMPLE}:")
    print(f"  Posterior mean: {mu_at_sample[0].item():.4f}")
    print(f"  Posterior std: {std_at_sample.item():.4f}")
    print(f"  Sampled λ_obs = {lambda_obs:.4f} (from GP posterior, seed={LAMBDA_SEED})")

    # -------------------------------------------------------------------------
    # Compute entropy components and conditional moments (MC)
    # -------------------------------------------------------------------------
    if SAMPLE_X:
        print(f"\nComputing utility (MC: {N_LAMBDA_SAMPLES} samples, x ~ N({GAUSSIAN_MEAN}, {GAUSSIAN_STD}²))...")
    else:
        print(f"\nComputing utility (MC: {N_LAMBDA_SAMPLES} λ samples at fixed x={X_SAMPLE})...")

    # Compute utility
    if USE_DISTRIBUTION_AWARE:
        utility = _compute_diagnostic_utility(model, x_plot, n_mc_samples=N_LAMBDA_SAMPLES)
    else:
        utility = evaluate_nd_utility_new(model, x_plot)

    # Also compute H_marg for plotting
    mu_marg, sigma2_marg = get_marginal_moments(model, x_plot)
    H_marg = compute_H(mu_marg, sigma2_marg)
    H_cond = H_marg - utility  # Recover H_cond from utility for plotting

    # For Plots 2 and 3: compute averaged conditional moments separately
    # (This is additional computation just for visualization)
    mu_cond_sum = torch.zeros_like(H_marg)
    sigma2_cond_sum = torch.zeros_like(H_marg)
    torch.manual_seed(99)  # Different seed for this averaging
    for _ in range(N_LAMBDA_SAMPLES):
        if SAMPLE_X:
            x_i = GAUSSIAN_MEAN + GAUSSIAN_STD * torch.randn(1, dtype=DTYPE, device=DEVICE).item()
            x_i_tensor = torch.tensor([x_i], dtype=DTYPE, device=DEVICE)
            with torch.no_grad():
                post_i = model(x_i_tensor)
                mu_i = post_i.mean[0]
                std_i = post_i.variance[0].sqrt()
        else:
            x_i = X_SAMPLE
            mu_i = mu_at_sample[0]
            std_i = std_at_sample
        lambda_i = (mu_i + std_i * torch.randn(1, dtype=DTYPE, device=DEVICE)).item()
        mu_cond_i, sigma2_cond_i = get_conditional_moments(model, x_plot, x_i, lambda_i)
        mu_cond_sum += mu_cond_i
        sigma2_cond_sum += sigma2_cond_i
    mu_after_avg = mu_cond_sum / N_LAMBDA_SAMPLES
    sigma2_after_avg = sigma2_cond_sum / N_LAMBDA_SAMPLES

    # For Plot 1: single sample (lambda_obs at X_SAMPLE) to show clear variance reduction
    mu_after_single, sigma2_after_single = get_conditional_moments(model, x_plot, X_SAMPLE, lambda_obs)
    std_after_single = sigma2_after_single.sqrt()

    print(f"  Utility range: [{utility.min():.3f}, {utility.max():.3f}]")

    # Convert to numpy for plotting
    x_np = x_plot.cpu().numpy()
    H_marg_np = H_marg.cpu().numpy()
    H_cond_np = H_cond.cpu().numpy()
    utility_np = utility.cpu().numpy()

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)

    # Convert to numpy
    mu_before_np = mu_before.cpu().numpy()
    std_before_np = std_before.cpu().numpy()
    var_before_np = var_before.cpu().numpy()
    mu_after_single_np = mu_after_single.cpu().numpy()  # Single sample for Plot 1
    std_after_single_np = std_after_single.cpu().numpy()
    mu_after_avg_np = mu_after_avg.cpu().numpy()  # MC average for Plot 2
    sigma2_after_avg_np = sigma2_after_avg.cpu().numpy()  # MC average for Plot 3
    train_x_np = train_x.cpu().numpy()

    true_lambda_np = lambda_true(x_plot).cpu().numpy()

    # =========================================================================
    # Plot 1: GP Posterior Mean and Variance (Before vs After) - Single sample
    # =========================================================================
    ax1 = axes[0]

    # True function
    ax1.plot(x_np, true_lambda_np, 'k--', linewidth=2, label='True λ(x)', zorder=1)

    # BEFORE conditioning (blue)
    ax1.plot(x_np, mu_before_np, 'b-', linewidth=2, label='Before: GP mean', zorder=2)
    ax1.fill_between(x_np,
                     mu_before_np - 2*std_before_np,
                     mu_before_np + 2*std_before_np,
                     alpha=0.25, color='blue', label='Before: ±2σ', zorder=1)

    # AFTER conditioning (red/orange) - single sample
    ax1.plot(x_np, mu_after_single_np, 'r-', linewidth=2, label='After: GP mean', zorder=3)
    ax1.fill_between(x_np,
                     mu_after_single_np - 2*std_after_single_np,
                     mu_after_single_np + 2*std_after_single_np,
                     alpha=0.25, color='red', label='After: ±2σ', zorder=1)

    # Mark observation point
    ax1.axvline(x=X_SAMPLE, color='green', linewidth=2, linestyle='--',
                label=f'x_sample = {X_SAMPLE}', zorder=4)
    ax1.scatter([X_SAMPLE], [lambda_obs], c='green', s=150, marker='*',
                zorder=5, edgecolors='darkgreen', linewidths=2,
                label=f'Observed λ = {lambda_obs:.2f}')

    # Training points
    for i, tx in enumerate(train_x_np):
        label = 'Training data' if i == 0 else None
        ax1.axvline(x=tx, color='gray', alpha=0.3, linewidth=1, linestyle=':', label=label)

    ax1.set_ylabel('λ(x)')
    ax1.set_title(f'GP Posterior: Before vs After Conditioning on λ({X_SAMPLE}) = {lambda_obs:.2f}\n'
                  f'(Fixed lengthscale = {FIXED_LENGTHSCALE})')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 2: GP Mean Before/After with Absolute Difference (MC averaged)
    # =========================================================================
    ax2 = axes[1]

    # Plot means (MC averaged for "after")
    ax2.plot(x_np, mu_before_np, 'b-', linewidth=2, label='Before conditioning')
    ax2.plot(x_np, mu_after_avg_np, 'r-', linewidth=2, label=f'After (avg over {N_LAMBDA_SAMPLES} samples)')

    # Absolute difference on secondary axis
    mu_diff = np.abs(mu_after_avg_np - mu_before_np)
    ax2_twin = ax2.twinx()
    ax2_twin.fill_between(x_np, 0, mu_diff, alpha=0.3, color='green')
    ax2_twin.plot(x_np, mu_diff, 'g-', linewidth=1.5, alpha=0.8, label='|Δμ|')
    ax2_twin.set_ylabel('|Δμ|', color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')

    # Mark observation point
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

    # Plot variances (MC averaged for "after")
    ax3.plot(x_np, var_before_np, 'b-', linewidth=2, label='Before conditioning')
    ax3.plot(x_np, sigma2_after_avg_np, 'r-', linewidth=2, label=f'After (avg over {N_LAMBDA_SAMPLES} samples)')

    # Absolute difference on secondary axis
    var_diff = np.abs(sigma2_after_avg_np - var_before_np)
    ax3_twin = ax3.twinx()
    ax3_twin.fill_between(x_np, 0, var_diff, alpha=0.3, color='green')
    ax3_twin.plot(x_np, var_diff, 'g-', linewidth=1.5, alpha=0.8, label='|Δσ²|')
    ax3_twin.set_ylabel('|Δσ²|', color='green')
    ax3_twin.tick_params(axis='y', labelcolor='green')

    # Mark observation point
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

    # Plot H_marginal and H_conditional
    ax4.plot(x_np, H_marg_np, 'b-', linewidth=2, label='H_marg(x*)')
    ax4.plot(x_np, H_cond_np, 'r-', linewidth=2,
             label=f'E[H_cond(x* | λ)] ({N_LAMBDA_SAMPLES} samples)')

    # Plot utility on secondary y-axis
    ax4_twin = ax4.twinx()
    ax4_twin.plot(x_np, utility_np, 'g-', linewidth=2, alpha=0.8,
                  label='Utility = H_marg - H_cond')
    ax4_twin.fill_between(x_np, 0, utility_np, alpha=0.15, color='green')
    ax4_twin.set_ylabel('Utility', color='green')
    ax4_twin.tick_params(axis='y', labelcolor='green')

    # Mark observation point
    ax4.axvline(x=X_SAMPLE, color='green', linewidth=2, linestyle='--')

    # Show Gaussian p(x) for reference (scaled)
    gaussian_pdf = np.exp(-0.5 * ((x_np - GAUSSIAN_MEAN) / GAUSSIAN_STD)**2)
    H_max = max(H_marg_np.max(), H_cond_np.max())
    gaussian_pdf_scaled = gaussian_pdf / gaussian_pdf.max() * H_max * 0.3
    ax4.fill_between(x_np, 0, gaussian_pdf_scaled, alpha=0.2, color='orange',
                     label=f'p(x) Gaussian (scaled)')

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
