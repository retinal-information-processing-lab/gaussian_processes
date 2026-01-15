"""
Test: Distribution-Aware Utility with Dirac Delta p(x) = δ(x - x*)
==================================================================
Created by Claude to verify that the distribution-aware utility collapses
to nd_utility when p(x) is a Dirac delta centered at the query point x*.

When p(x) = δ(x - x*), the conditional distribution of λ(x*) | λ(x) has:
    μ_cond = λ (the sampled value)
    σ²_cond = 0 (conditioning on itself)

This test uses distribution_aware_utility_gpytorch with x_samples=x_star
to verify that it matches nd_utility_new and nd_utility_NUMERICAL.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add Spatial_GP_repo to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gp_utility_playground import (
    VariationalGP, PoissonLikelihood, train_gp,
    generate_poisson_data, lambda_true,
    evaluate_utility,
    DEVICE, DTYPE, X_MIN, X_MAX, MAX_R
)

# Import utility functions
from utility import distribution_aware_utility_gpytorch, nd_utility_NUMERICAL, nd_utility_hybrid


def evaluate_nd_utility_NUMERICAL(model, x_candidates, max_r=MAX_R, n_quadrature=100):
    """Evaluate nd_utility_NUMERICAL (Gauss-Hermite quadrature) at candidate points."""
    model.eval()

    with torch.no_grad():
        posterior = model(x_candidates)
        mu = posterior.mean
        sigma2 = posterior.variance

        utility = nd_utility_NUMERICAL(mu, sigma2, r_max=max_r, n_quadrature=n_quadrature)

    return utility


def evaluate_nd_utility_hybrid(model, x_candidates, max_r=MAX_R, n_samples=10000, batch_size=500):
    """Evaluate nd_utility_hybrid (Laplace H_marg + MC H_cond) at candidate points."""
    model.eval()

    with torch.no_grad():
        posterior = model(x_candidates)
        mu = posterior.mean
        sigma2 = posterior.variance

        utility = nd_utility_hybrid(mu, sigma2, r_max=max_r, n_samples=n_samples, batch_size=batch_size)

    return utility


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("Test: Dirac Delta Collapse")
    print("Verifying: distribution_aware_utility(x_samples=x_star) ≡ nd_utility")
    print("=" * 70)

    # =========================================================================
    # Setup: Create and train GP
    # =========================================================================
    n_train = 50
    train_x = torch.linspace(X_MIN, X_MAX, n_train, dtype=DTYPE, device=DEVICE)
    train_y = generate_poisson_data(train_x, lambda_true)

    inducing_points = train_x.clone()
    model = VariationalGP(inducing_points).to(DEVICE)
    likelihood = PoissonLikelihood().to(DEVICE)

    print("\nTraining GP...")
    model, likelihood = train_gp(model, likelihood, train_x, train_y, n_iterations=300)

    # =========================================================================
    # Evaluate utilities on candidate grid
    # =========================================================================
    x_candidates = torch.linspace(X_MIN, X_MAX, 200, dtype=DTYPE, device=DEVICE)

    r_max = MAX_R
    n_quadrature = 200

    print("\nEvaluating nd_utility_new (Laplace approximation)...")
    u_nd = evaluate_utility(model, x_candidates)

    print(f"Evaluating nd_utility_NUMERICAL (Gauss-Hermite, {n_quadrature} points)...")
    u_numerical = evaluate_nd_utility_NUMERICAL(model, x_candidates, max_r=r_max, n_quadrature=n_quadrature)

    n_lambda_samples = 100_000

    print(f"Evaluating nd_utility_hybrid (Laplace H_marg + MC H_cond, {n_lambda_samples:,} samples)...")
    u_hybrid = evaluate_nd_utility_hybrid(model, x_candidates, max_r=r_max, n_samples=n_lambda_samples, batch_size=500)

    print(f"Evaluating distribution_aware_utility with Dirac delta ({n_lambda_samples:,} lambda samples)...")

    # Call distribution_aware_utility_gpytorch with x_samples = x_star (Dirac delta case)
    model.eval()
    with torch.no_grad():
        u_dirac = distribution_aware_utility_gpytorch(
            model=model,
            x_star=x_candidates,
            r_max=r_max,
            n_lambda_samples=n_lambda_samples,
            batch_size=5000,
            x_samples=x_candidates,  # Dirac delta: x_samples = x_star
            A=1.0,
            lambda0=0.0
        )

    # =========================================================================
    # Get GP posterior for λ(x) and f(x) plots
    # =========================================================================
    model.eval()
    with torch.no_grad():
        posterior = model(x_candidates)
        mu = posterior.mean
        sigma = posterior.variance.sqrt()

        # True lambda (for comparison)
        true_lambda = lambda_true(x_candidates)

    # =========================================================================
    # Compute statistics
    # =========================================================================
    # Dirac vs nd_utility (Laplace)
    diff_laplace = (u_dirac - u_nd).abs()
    rel_diff_laplace = diff_laplace / (u_nd.abs() + 1e-10)

    # Dirac vs nd_utility_NUMERICAL
    diff_numerical = (u_dirac - u_numerical).abs()
    rel_diff_numerical = diff_numerical / (u_numerical.abs() + 1e-10)

    # Dirac vs nd_utility_hybrid
    diff_hybrid = (u_dirac - u_hybrid).abs()
    rel_diff_hybrid = diff_hybrid / (u_hybrid.abs() + 1e-10)

    # Laplace vs NUMERICAL (baseline comparison)
    diff_lap_num = (u_nd - u_numerical).abs()
    rel_diff_lap_num = diff_lap_num / (u_numerical.abs() + 1e-10)

    # Laplace vs hybrid
    diff_lap_hyb = (u_nd - u_hybrid).abs()
    rel_diff_lap_hyb = diff_lap_hyb / (u_hybrid.abs() + 1e-10)

    # hybrid vs NUMERICAL
    diff_hyb_num = (u_hybrid - u_numerical).abs()
    rel_diff_hyb_num = diff_hyb_num / (u_numerical.abs() + 1e-10)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nnd_utility (Laplace) range:   [{u_nd.min():.6f}, {u_nd.max():.6f}]")
    print(f"nd_utility (NUMERICAL) range: [{u_numerical.min():.6f}, {u_numerical.max():.6f}]")
    print(f"nd_utility (hybrid) range:    [{u_hybrid.min():.6f}, {u_hybrid.max():.6f}]")
    print(f"Dirac delta range:            [{u_dirac.min():.6f}, {u_dirac.max():.6f}]")

    print(f"\n--- nd_utility baselines comparison: ---")
    print(f"  Laplace vs NUMERICAL:  Mean rel error = {rel_diff_lap_num.mean():.6%}")
    print(f"  Laplace vs hybrid:     Mean rel error = {rel_diff_lap_hyb.mean():.6%}")
    print(f"  hybrid vs NUMERICAL:   Mean rel error = {rel_diff_hyb_num.mean():.6%}")

    print(f"\n--- Dirac Delta vs nd_utility (Laplace): ---")
    print(f"  Mean rel error: {rel_diff_laplace.mean():.6%}")
    print(f"  Max rel error:  {rel_diff_laplace.max():.6%}")

    print(f"\n--- Dirac Delta vs nd_utility (NUMERICAL): ---")
    print(f"  Mean rel error: {rel_diff_numerical.mean():.6%}")
    print(f"  Max rel error:  {rel_diff_numerical.max():.6%}")

    print(f"\n--- Dirac Delta vs nd_utility (hybrid): ---")
    print(f"  Mean rel error: {rel_diff_hybrid.mean():.6%}")
    print(f"  Max rel error:  {rel_diff_hybrid.max():.6%}")

    # Use 5% threshold for MC-based comparison
    passes_laplace = rel_diff_laplace.mean() < 0.05
    passes_numerical = rel_diff_numerical.mean() < 0.05
    passes_hybrid = rel_diff_hybrid.mean() < 0.05
    print(f"\n{'✓' if passes_laplace else '✗'} Dirac vs Laplace: Mean rel error = {rel_diff_laplace.mean():.4%} (threshold: 5%)")
    print(f"{'✓' if passes_numerical else '✗'} Dirac vs NUMERICAL: Mean rel error = {rel_diff_numerical.mean():.4%} (threshold: 5%)")
    print(f"{'✓' if passes_hybrid else '✗'} Dirac vs hybrid: Mean rel error = {rel_diff_hybrid.mean():.4%} (threshold: 5%)")

    # =========================================================================
    # Compute firing rate statistics for analysis
    # =========================================================================
    f_mean = torch.exp(mu)
    f_mean_corrected = torch.exp(mu + 0.5 * sigma**2)

    print(f"\nFiring rate f = exp(λ) statistics:")
    print(f"  Max E[f]:    {f_mean_corrected.max():.1f}")
    print(f"  Max μ:       {mu.max():.2f}")
    print(f"  Max σ:       {sigma.max():.2f}")
    print(f"  r_max used:  {MAX_R}")

    # =========================================================================
    # Visualization
    # =========================================================================
    x_np = x_candidates.cpu().numpy()
    train_x_np = train_x.cpu().numpy()
    train_y_np = train_y.cpu().numpy()

    u_nd_np = u_nd.cpu().numpy()
    u_numerical_np = u_numerical.cpu().numpy()
    u_hybrid_np = u_hybrid.cpu().numpy()
    u_dirac_np = u_dirac.cpu().numpy()

    mu_np = mu.cpu().numpy()
    sigma_np = sigma.cpu().numpy()
    true_lambda_np = true_lambda.cpu().numpy()

    f_mean_np = f_mean.cpu().numpy()
    f_mean_corrected_np = f_mean_corrected.cpu().numpy()
    true_f_np = np.exp(true_lambda_np)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # -------------------------------------------------------------------------
    # Subplot 1: Utility comparison
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(x_np, u_nd_np, 'b-', linewidth=2.5, label='nd_utility (Laplace)')
    ax1.plot(x_np, u_numerical_np, 'g--', linewidth=2, alpha=0.9,
             label='nd_utility (NUMERICAL)')
    ax1.plot(x_np, u_hybrid_np, 'm-.', linewidth=2, alpha=0.9,
             label='nd_utility (hybrid)')
    ax1.plot(x_np, u_dirac_np, 'r:', linewidth=2.5, alpha=0.8,
             label='Dirac delta MC')

    for i, tx in enumerate(train_x_np):
        ax1.axvline(x=tx, color='orange', alpha=0.3, linewidth=1, linestyle=':',
                    label='Training x' if i == 0 else None)

    ax1.set_ylabel('Utility U(x*)')
    ax1.set_title('Utility Comparison: 3 nd_utility Methods vs Dirac Delta MC')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Subplot 2: λ(x) space - GP fit quality
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(x_np, true_lambda_np, 'k--', linewidth=2, label='True λ(x)')
    ax2.plot(x_np, mu_np, 'b-', linewidth=2, label='GP mean μ(x)')
    ax2.fill_between(x_np, mu_np - 2*sigma_np, mu_np + 2*sigma_np,
                     alpha=0.3, color='blue', label='±2σ')

    for i, tx in enumerate(train_x_np):
        ax2.axvline(x=tx, color='red', alpha=0.5, linewidth=1.5,
                    label='Training x' if i == 0 else None)

    ax2.set_ylabel('λ(x) = log(f)')
    ax2.set_title('Latent Function (log firing rate) - GP Posterior')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Subplot 3: f(x) = exp(λ) space - Firing rate with observations
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    ax3.plot(x_np, true_f_np, 'k--', linewidth=2, label='True f(x) = exp(λ)')
    ax3.plot(x_np, f_mean_np, 'b-', linewidth=2, label='Predicted exp(μ)')

    # Approximate uncertainty in rate space (delta method)
    f_std_approx = f_mean_np * sigma_np
    ax3.fill_between(x_np,
                     np.maximum(0, f_mean_np - 2*f_std_approx),
                     f_mean_np + 2*f_std_approx,
                     alpha=0.3, color='blue', label='±2σ (approx)')

    # Horizontal line at r_max
    ax3.axhline(y=MAX_R, color='red', linestyle='--', linewidth=2,
                label=f'r_max = {MAX_R}')

    # Plot observed spike counts
    ax3.scatter(train_x_np, train_y_np, c='red', s=80, zorder=5,
                edgecolors='darkred', linewidths=1.5,
                label='Observed counts')

    ax3.set_ylabel('Firing rate / Spike count')
    ax3.set_title('Firing Rate Space')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    # -------------------------------------------------------------------------
    # Subplot 4: Error analysis
    # -------------------------------------------------------------------------
    ax4 = axes[3]

    # Plot relative errors: Dirac vs each nd_utility method
    ax4.semilogy(x_np, rel_diff_laplace.cpu().numpy() + 1e-10, 'b-', linewidth=2,
                 label=f'Dirac vs Laplace (mean: {rel_diff_laplace.mean():.4%})')
    ax4.semilogy(x_np, rel_diff_numerical.cpu().numpy() + 1e-10, 'g--', linewidth=2,
                 label=f'Dirac vs NUMERICAL (mean: {rel_diff_numerical.mean():.4%})')
    ax4.semilogy(x_np, rel_diff_hybrid.cpu().numpy() + 1e-10, 'm-.', linewidth=2,
                 label=f'Dirac vs hybrid (mean: {rel_diff_hybrid.mean():.4%})')

    # Secondary axis for firing rate
    ax4_twin = ax4.twinx()
    ax4_twin.plot(x_np, f_mean_corrected_np, 'r--', linewidth=1.5, alpha=0.5,
                  label='E[f] = exp(μ+σ²/2)')
    ax4_twin.set_ylabel('Expected firing rate E[f]', color='red')
    ax4_twin.tick_params(axis='y', labelcolor='red')

    ax4.set_xlabel('x')
    ax4.set_ylabel('Relative Error')
    ax4.set_title('Error Analysis: Dirac Delta vs All nd_utility Methods')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(__file__).parent / 'test_dirac_delta_collapse_result.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved plot to: {save_path}")
    plt.close()

    print("\nTest completed!")

    return passes_laplace and passes_numerical and passes_hybrid


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
