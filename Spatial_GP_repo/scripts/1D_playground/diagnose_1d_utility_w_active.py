"""
1D Distribution-Aware Active Learning
=====================================
Active learning loop using distribution-aware utility for point selection.

Iteratively:
1. Train GP on current data
2. Compute utility at candidate points
3. Select max-utility point
4. Observe Poisson response
5. Add to training set
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
    evaluate_nd_utility_new, evaluate_distribution_aware_utility,
    get_marginal_moments, remove_near_duplicates, train_gp_fixed_lengthscale,
    DEVICE, DTYPE, X_MIN, X_MAX
)

from diagnose_1d_utility_w_fixed_ntrain import (
    FIXED_LENGTHSCALE, FIXED_OUTPUTSCALE,
    GAUSSIAN_MEAN, GAUSSIAN_STD, N_LAMBDA_SAMPLES
)

X_PLOT_MIN = X_MIN - 0.1 * (X_MAX - X_MIN) 
X_PLOT_MAX = X_MAX + 0.1 * (X_MAX - X_MIN)


# =============================================================================
# Configuration
# =============================================================================
N_INITIAL = 5       # Initial training points
N_ITERATIONS = 300    # Points to add via active learning
N_CANDIDATES = 400   # Grid resolution for utility evaluation
N_TRAIN_ITERS = 300  # Training iterations per GP fit

# Training data sampling (same flag as 1d_distribution_aware_fixed)
SAMPLE_TRAIN_X = True  # If True, initial x from Gaussian; if False, uniform

# Utility function selection
USE_DISTRIBUTION_AWARE = True  # If True, use distribution-aware utility; if False, use standard nd_utility

SEED = 42


# =============================================================================
# Main
# =============================================================================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print("Distribution-Aware Active Learning")
    print("=" * 70)
    print(f"Initial points: {N_INITIAL}")
    print(f"Iterations: {N_ITERATIONS}")
    print(f"Total points after loop: {N_INITIAL + N_ITERATIONS}")
    print(f"SAMPLE_TRAIN_X: {SAMPLE_TRAIN_X}")
    print(f"Gaussian p(x): N({GAUSSIAN_MEAN}, {GAUSSIAN_STD}²)")

    # -------------------------------------------------------------------------
    # Generate initial training data
    # -------------------------------------------------------------------------
    if SAMPLE_TRAIN_X:
        train_x = GAUSSIAN_MEAN + GAUSSIAN_STD * torch.randn(N_INITIAL, dtype=DTYPE, device=DEVICE)
        train_x = train_x.sort()[0]
        print(f"\nInitial x sampled from N({GAUSSIAN_MEAN}, {GAUSSIAN_STD}²)")
    else:
        train_x = torch.linspace(X_MIN, X_MAX, N_INITIAL, dtype=DTYPE, device=DEVICE)
        print(f"\nInitial x uniform in [{X_MIN}, {X_MAX}]")

    train_y = generate_poisson_data(train_x, lambda_true)

    # Remove near-duplicates
    if SAMPLE_TRAIN_X:
        n_before = len(train_x)
        train_x, train_y = remove_near_duplicates(train_x, train_y, min_dist=1e-4)
        n_removed = n_before - len(train_x)
        if n_removed > 0:
            print(f"  Removed {n_removed} near-duplicate points")

    print(f"Initial training: {len(train_x)} points")

    # -------------------------------------------------------------------------
    # Candidate grid for utility evaluation
    # -------------------------------------------------------------------------
    candidates = torch.linspace(X_MIN, X_MAX, N_CANDIDATES, dtype=DTYPE, device=DEVICE)

    # -------------------------------------------------------------------------
    # Active learning loop
    # -------------------------------------------------------------------------
    selected_x_history = []
    utility_snapshots = {}  # Store utility at 6 points throughout the loop
    # 6 evenly spaced snapshots
    snapshot_iters = [1,
                      N_ITERATIONS // 5,
                      2 * N_ITERATIONS // 5,
                      3 * N_ITERATIONS // 5,
                      4 * N_ITERATIONS // 5,
                      N_ITERATIONS]

    for i in range(N_ITERATIONS):
        print(f"\n--- Iteration {i+1}/{N_ITERATIONS} ---")

        # Create fresh model
        inducing_points = train_x.clone()
        model = VariationalGP(inducing_points, jitter=1e-4).to(DEVICE)
        likelihood = PoissonLikelihood().to(DEVICE)

        # Train GP
        model, likelihood = train_gp_fixed_lengthscale(
            model, likelihood, train_x, train_y,
            n_iterations=N_TRAIN_ITERS,
            fixed_lengthscale=FIXED_LENGTHSCALE,
            fixed_outputscale=FIXED_OUTPUTSCALE
        )

        # Compute utility
        if USE_DISTRIBUTION_AWARE:
            utility = evaluate_distribution_aware_utility(
                model, candidates, n_mc_samples=N_LAMBDA_SAMPLES,
                p_x_mean=GAUSSIAN_MEAN, p_x_std=GAUSSIAN_STD
            )
        else:
            utility = evaluate_nd_utility_new(model, candidates)

        # Store utility snapshot at key iterations
        iter_num = i + 1
        if iter_num in snapshot_iters:
            utility_snapshots[iter_num] = utility.cpu().numpy().copy()

        # Select max utility point
        best_idx = torch.argmax(utility)
        x_new = candidates[best_idx:best_idx+1]
        selected_x_history.append(x_new.item())
        print(f"Selected x = {x_new.item():.4f} (utility = {utility[best_idx].item():.4f})")

        # Observe response
        y_new = generate_poisson_data(x_new, lambda_true)
        print(f"Observed count = {y_new.item():.0f}")

        # Add to training set
        train_x = torch.cat([train_x, x_new])
        train_y = torch.cat([train_y, y_new])

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Active Learning Complete!")
    print("=" * 70)
    print(f"Final training set: {len(train_x)} points")
    print(f"Final x (sorted): {torch.sort(train_x).values.cpu().numpy()}")

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1])

    # Row 1: GP fit (full row)
    ax1 = fig.add_subplot(gs[0, :])

    # Row 2: Selection history (full row)
    ax2 = fig.add_subplot(gs[1, :])

    # Row 3: First 3 utility plots
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    ax5 = fig.add_subplot(gs[2, 2])

    # Row 4: Last 3 utility plots
    ax6 = fig.add_subplot(gs[3, 0])
    ax7 = fig.add_subplot(gs[3, 1])
    ax8 = fig.add_subplot(gs[3, 2])

    # --- Subplot 1: Final GP Fit ---
    x_plot = torch.linspace(X_PLOT_MIN, X_PLOT_MAX, 200, dtype=DTYPE, device=DEVICE)

    # Get final model predictions (model is still the last trained one)
    mu, sigma2 = get_marginal_moments(model, x_plot)
    std = sigma2.sqrt()

    # Convert to numpy
    x_plot_np = x_plot.cpu().numpy()
    mu_np = mu.cpu().numpy()
    std_np = std.cpu().numpy()
    true_lambda_np = lambda_true(x_plot).cpu().numpy()
    candidates_np = candidates.cpu().numpy()

    # True function
    ax1.plot(x_plot_np, true_lambda_np, 'k--', linewidth=2, label='True λ(x)')

    # GP posterior
    ax1.plot(x_plot_np, mu_np, 'b-', linewidth=2, label='GP mean')
    ax1.fill_between(x_plot_np, mu_np - 2*std_np, mu_np + 2*std_np,
                     alpha=0.25, color='blue', label='±2σ')

    # Selected x as gray dashed vertical lines
    for j, x_sel in enumerate(selected_x_history):
        label = 'Selected x' if j == 0 else None
        ax1.axvline(x=x_sel, color='gray', linestyle='--', alpha=0.5, label=label)

    ax1.set_xlabel('x')
    ax1.set_ylabel('λ(x)')
    ax1.set_title('Final GP Fit with Selected Points')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Selection History with Gaussian p(x) ---
    x_dense = np.linspace(X_PLOT_MIN, X_PLOT_MAX, 200)
    gaussian = np.exp(-0.5 * ((x_dense - GAUSSIAN_MEAN) / GAUSSIAN_STD)**2)
    gaussian_scaled = gaussian / gaussian.max() * N_ITERATIONS * 0.4

    ax2.fill_between(x_dense, 0, gaussian_scaled, alpha=0.3, color='orange', label='p(x)')

    # Scatter: (x_selected, iteration)
    ax2.scatter(selected_x_history, range(1, N_ITERATIONS + 1), c='blue', s=50, zorder=3)

    ax2.set_xlabel('x')
    ax2.set_ylabel('Iteration')
    ax2.set_title('Selection History')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Subplots 3-8: Utility snapshots ---
    utility_axes = [ax3, ax4, ax5, ax6, ax7, ax8]

    for ax, iter_num in zip(utility_axes, snapshot_iters):
        # Gaussian p(x) as background
        utility_max = max(u.max() for u in utility_snapshots.values())
        gaussian_for_utility = gaussian / gaussian.max() * utility_max * 0.5
        ax.fill_between(x_dense, 0, gaussian_for_utility, alpha=0.3, color='orange', label='p(x)')

        # Utility as green line
        ax.plot(candidates_np, utility_snapshots[iter_num], 'g-', linewidth=2, label='Utility')

        ax.set_xlabel('x')
        ax.set_ylabel('Utility')
        ax.set_title(f'Utility at Iter {iter_num}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(__file__).parent / 'distribution_aware_1d_active_result.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved plot to: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
