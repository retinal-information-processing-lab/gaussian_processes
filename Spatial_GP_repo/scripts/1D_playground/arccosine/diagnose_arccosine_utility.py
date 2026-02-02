"""
Diagnostic: Arc-Cosine Kernel - Before/After Conditioning Analysis
===================================================================
Created by Claude

Equivalent to diagnose_1d_utility_w_fixed_ntrain.py but:
- Uses Arc-Cosine kernel (non-stationary)
- Loads pre-trained model from checkpoint (no training)
- Imports shared code from gp_arccosine_playground.py

Produces 4 vertically stacked plots:
1. GP Posterior Before vs After Conditioning (single sample)
2. GP Mean Before vs E[After] (MC averaged) + |Δμ|
3. GP Variance Before vs E[After] (MC averaged) + |Δσ²|
4. Entropy Components (H_marg, H_cond, Utility)

===============================================================================
PEDAGOGICAL EXPLANATION: What are we computing and why?
===============================================================================

THE GOAL:
---------
Understand how observing a neural response at some stimulus x affects our
predictions at all other stimuli x*. This tells us which stimuli are most
informative for learning about the neuron.

THE SETUP:
----------
- λ(x) is the LATENT FUNCTION (log-firing rate) we're learning
- At each point x, the GP gives us a posterior distribution: λ(x) ~ N(μ(x), σ²(x))
- This means: "We believe λ(x) is around μ(x), with uncertainty σ(x)"

KEY VARIABLES (in compute_mc_diagnostics):
------------------------------------------
- **x_i**: The stimulus where we imagine observing a response

- **μ_i, σ_i**: POSTERIOR PREDICTIVE moments at x_i
    * After training on data D, we have a posterior distribution over λ(x)
    * At x_i specifically: p(λ(x_i) | D) = N(μ_i, σ²_i)
    * μ_i = E[λ(x_i) | D] - our best guess for λ at x_i given training data
    * σ_i = √Var[λ(x_i) | D] - our uncertainty about λ at x_i
    * Code: post_i = model(x_i_tensor) → μ_i = post_i.mean, σ_i = post_i.std

- **λ_i**: SAMPLED value from posterior predictive
    * λ_i ~ N(μ_i, σ²_i) - a plausible value of λ at x_i
    * Simulates "what if the true λ(x_i) was this value?"
    * Code: λ_i = μ_i + σ_i · ε, where ε ~ N(0,1)

- **μ_cond_i, σ²_cond_i**: UPDATED posterior moments at x* after conditioning
    * After "observing" λ_i at x_i, we update beliefs everywhere
    * At query point x*: p(λ(x*) | D, λ(x_i)=λ_i) = N(μ_cond_i, σ²_cond_i)
    * Uses Gaussian conditioning formula (see get_conditional_moments)

WHY SAMPLE λ_i WITH GAUSSIAN NOISE?
------------------------------------
Because λ(x) is a Gaussian Process! After training on data D, the POSTERIOR
PREDICTIVE distribution at any point x is Gaussian:

    p(λ(x) | D) = N(μ(x), σ²(x))

So μ_i and σ_i are the mean and std of this posterior predictive at x_i.

To simulate "what if we observed λ(x_i) = λ_i?", we sample from this distribution:
    λ_i ~ N(μ_i, σ²_i)

This gives us different plausible values that the true (unknown) latent function
might have at x_i, weighted by our current posterior beliefs.

TWO MODES: SAMPLE_X = True vs False
------------------------------------

**SAMPLE_X = False** (current setting):
    - x_i = X_SAMPLE = -1.0 (ALWAYS the same location)
    - We sample N_LAMBDA_SAMPLES different λ values at this fixed point
    - Question: "If I repeatedly observe the neuron at x=-1.0, how much do I
                 learn on average about responses everywhere?"
    - λ_i ~ N(μ(-1.0), σ²(-1.0)) - same distribution every time

**SAMPLE_X = True**:
    - x_i ~ N(mean, std²) - sample DIFFERENT stimulus locations
    - For each x_i, sample λ_i ~ N(μ(x_i), σ²(x_i))
    - Question: "If I show a random natural image and observe the response,
                 how much do I learn on average?"
    - Different distribution for each sample (depends on x_i)

WHAT'S THE DIFFERENCE BETWEEN μ_i AND μ_cond_i?
------------------------------------------------
- **μ_i**: What we predict for λ(x_i) BEFORE observing anything new
- **μ_cond_i**: What we predict for λ(x*) AFTER observing λ_i at x_i

They're predictions at DIFFERENT locations:
- μ_i is at the observation point (x_i)
- μ_cond_i is at the query point (x*)

Example:
    Before: "I think λ(-1.0) ≈ 0.59 and λ(2.0) ≈ 1.2"
    Sample: λ_i = 0.85 at x_i=-1.0 (higher than expected!)
    After: "Oh, it was higher! So λ(2.0) is probably also higher: μ_cond ≈ 1.4"

THE FOUR PLOTS:
---------------
1. **Single sample**: Shows one specific observation (green star) and how it
   updates the posterior (blue → red)

2-3. **MC averaged μ and σ²**: Average change across many possible observations
   Shows: "On average, how much does the mean/variance change?"

4. **Utility**: Combines entropy reduction across all x*
   High utility = observing here tells us a lot about the neuron overall
===============================================================================
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared MC diagnostics function (authoritative source)
from diagnose_1d_utility_w_fixed_ntrain import (
    compute_mc_diagnostics,
    get_posterior_at_points,
)

# Import defaults from gp_utility_playground (only what's used)
from gp_utility_playground import (
    DEVICE, DTYPE,
    DEFAULT_P_X_MEAN, DEFAULT_P_X_STD,
    LAMBDA_SEED, X_SAMPLE, SAMPLE_X, N_LAMBDA_SAMPLES,
    X_MIN, X_MAX,  
    lambda_true,
    get_conditional_moments,
)

# Import Arc-Cosine specific overrides
from gp_arccosine_playground import (
    load_acos_checkpoint,
    check_kernel_health,
    ACOS_CHECKPOINT_PATH,
)

# =============================================================================
# Script-specific Configuration (local overrides)
# =============================================================================
# MC sampling (override: more samples for better accuracy)
N_LAMBDA_SAMPLES = 1  # Default is 100

# Sampling mode (override: fixed point instead of sampling from p(x))
SAMPLE_X = SAMPLE_X  # Use same sample x RBF diagnostic


# p(x) distribution (use defaults from gp_utility_playground)
GAUSSIAN_MEAN = DEFAULT_P_X_MEAN  # 0.0
GAUSSIAN_STD = DEFAULT_P_X_STD    # 1.5


X_SAMPLE = X_SAMPLE  # Use Arc-Cosine default observation point


# =============================================================================
# Main
# =============================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("Diagnostic: Arc-Cosine Kernel - Before/After Conditioning")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load pre-trained Arc-Cosine model
    # -------------------------------------------------------------------------
    print("\nLoading trained Arc-Cosine model...")
    model, likelihood, inducing_points, config, train_x, train_y = load_acos_checkpoint(ACOS_CHECKPOINT_PATH)

    # Get hyperparameters for display
    sigma_0 = model.covar_module.sigma_0.item()
    Amp = model.covar_module.Amp.item()
    mean_const = model.mean_module.constant.item()

    print(f"\nObservation point x_sample: {X_SAMPLE}")
    print(f"Plot domain: [{X_MIN}, {X_MAX}]")

    # -------------------------------------------------------------------------
    # Get posterior BEFORE conditioning
    # -------------------------------------------------------------------------
    x_plot = torch.linspace(X_MIN, X_MAX, 500, dtype=DTYPE, device=DEVICE)
    mu_before, var_before = get_posterior_at_points(model, x_plot)
    std_before = var_before.sqrt()

    # -------------------------------------------------------------------------
    # Sample observation at X_SAMPLE from GP posterior
    # -------------------------------------------------------------------------
    mu_at_sample, var_at_sample = get_posterior_at_points(
        model, torch.tensor([X_SAMPLE], dtype=DTYPE, device=DEVICE)
    )

    # Use same seed as gp_arccosine_playground for reproducibility
    torch.manual_seed(LAMBDA_SEED)
    mu_at_sample_scalar = mu_at_sample[0]  # Extract scalar from (1,) tensor
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
    train_x_np = train_x.cpu().numpy()

    true_lambda_np = lambda_true(x_plot).cpu().numpy()

    # -------------------------------------------------------------------------
    # Plotting (4 vertically stacked subplots)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)

    # =========================================================================
    # Plot 1: GP Posterior Before vs After Conditioning (Single sample)
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

    # AFTER conditioning (red) - single sample
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
    ax1.set_title(f'Arc-Cosine GP: Before vs After Conditioning on λ({X_SAMPLE}) = {lambda_obs:.2f}\n'
                  f'(σ₀={sigma_0:.3f}, Amp={Amp:.3f}, mean={mean_const:.3f})')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Description box for Plot 1
    desc1 = ("SINGLE OBSERVATION\n"
             f"Condition on one λ sampled at x={X_SAMPLE}\n"
             "Shows exact posterior update")
    ax1.text(0.02, 0.98, desc1, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # =========================================================================
    # Plot 2: GP Mean Before/After with Absolute Difference (MC averaged)
    # =========================================================================
    ax2 = axes[1]

    ax2.plot(x_np, mu_before_np, 'b-', linewidth=2, label='Before conditioning')
    ax2.plot(x_np, mu_after_avg_np, 'r-', linewidth=2, label=f'After (avg over {N_LAMBDA_SAMPLES} samples)')

    # Absolute difference on secondary axis
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

    # Description box for Plot 2
    if SAMPLE_X:
        desc2 = (f"MC AVERAGE ({N_LAMBDA_SAMPLES} samples)\n"
                 f"x ~ N({GAUSSIAN_MEAN}, {GAUSSIAN_STD}²)\n"
                 "λ ~ GP posterior at x")
    else:
        desc2 = (f"MC AVERAGE ({N_LAMBDA_SAMPLES} samples)\n"
                 f"x = {X_SAMPLE} (fixed)\n"
                 f"λ ~ N({mu_at_sample_scalar.item():.2f}, {std_at_sample.item():.2f}²)")
    ax2.text(0.52, 0.98, desc2, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # =========================================================================
    # Plot 3: GP Variance Before/After with Absolute Difference (MC averaged)
    # =========================================================================
    ax3 = axes[2]

    ax3.plot(x_np, var_before_np, 'b-', linewidth=2, label='Before conditioning')
    ax3.plot(x_np, sigma2_after_avg_np, 'r-', linewidth=2, label=f'After (avg over {N_LAMBDA_SAMPLES} samples)')

    # Absolute difference on secondary axis
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

    # Description box for Plot 3
    if SAMPLE_X:
        desc3 = (f"MC AVERAGE ({N_LAMBDA_SAMPLES} samples)\n"
                 f"x ~ N({GAUSSIAN_MEAN}, {GAUSSIAN_STD}²)\n"
                 "λ ~ GP posterior at x")
    else:
        desc3 = (f"MC AVERAGE ({N_LAMBDA_SAMPLES} samples)\n"
                 f"x = {X_SAMPLE} (fixed)\n"
                 f"λ ~ N({mu_at_sample_scalar.item():.2f}, {std_at_sample.item():.2f}²)")
    ax3.text(0.52, 0.98, desc3, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # =========================================================================
    # Plot 4: Entropy Components (H_marginal, H_conditional, and Utility)
    # =========================================================================
    ax4 = axes[3]

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

    # Description box for Plot 4
    if SAMPLE_X:
        desc4 = (f"MC AVERAGE ({N_LAMBDA_SAMPLES} samples)\n"
                 "Utility = information gain from\n"
                 f"observing λ at x ~ N({GAUSSIAN_MEAN}, {GAUSSIAN_STD}²)")
    else:
        desc4 = (f"MC AVERAGE ({N_LAMBDA_SAMPLES} samples)\n"
                 "Utility = information gain from\n"
                 f"observing λ at x = {X_SAMPLE} (fixed)")
    ax4.text(0.52, 0.98, desc4, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()

    # Save
    save_path = Path(__file__).parent / 'diagnose_arccosine_result.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved plot to: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
