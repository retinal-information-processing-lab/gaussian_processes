"""
Test: Utility Function Comparison
=================================
Compares three utility computation methods:
    - nd_utility_new: Laplace approximation (log-space, numerically stable)
    - nd_utility_MC: Monte Carlo sampling
    - nd_utility_NUMERICAL: Gauss-Hermite quadrature

This test validates that all three methods agree, and analyzes
the Monte Carlo implementation in detail.
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
    evaluate_nd_utility_NUMERICAL,
    DEVICE, DTYPE, X_MIN, X_MAX, MAX_R
)

from utility import nd_utility_MC, nd_utility_MC_batched, nd_utility_new, nd_utility_hybrid


def diagnose_mc_sampling(model, x_candidates, r_max=500, n_samples=8000, r_threshold=100, batch_size=1000):
    """
    Diagnostic function to understand MC sampling behavior.

    Reports:
    1. How many lambda samples produce f = exp(lambda) > r_threshold
    2. How much of H_marg comes from r > r_threshold vs r <= r_threshold
    3. E[log(r!)] contribution analysis

    Uses batched computation to avoid OOM errors with large n_samples and r_max.
    """
    model.eval()

    with torch.no_grad():
        posterior = model(x_candidates)
        mu = posterior.mean        # (N,)
        sigma2 = posterior.variance  # (N,)
        sigma = sigma2.sqrt()

    N = mu.shape[0]
    device = mu.device
    dtype = mu.dtype

    r_values = torch.arange(0, r_max, dtype=dtype, device=device)
    log_r_fact = torch.lgamma(r_values + 1)

    # =========================================================================
    # MC Sampling Analysis (Batched)
    # =========================================================================
    n_batches = (n_samples + batch_size - 1) // batch_size  # ceiling division

    # Accumulators
    n_above_threshold = torch.zeros(N, dtype=dtype, device=device)
    p_true_sum = torch.zeros(N, r_max, dtype=dtype, device=device)
    E_logr_low_sum = torch.zeros(N, dtype=dtype, device=device)
    E_logr_high_sum = torch.zeros(N, dtype=dtype, device=device)

    total_samples = 0

    for batch_idx in range(n_batches):
        # Determine actual batch size (last batch may be smaller)
        current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
        total_samples += current_batch_size

        # Sample lambda values for this batch
        eps = torch.randn(N, current_batch_size, dtype=dtype, device=device)
        lambda_batch = mu[:, None] + sigma[:, None] * eps  # (N, batch_size)
        f_batch = torch.exp(lambda_batch)  # (N, batch_size)

        # Count samples where f > r_threshold
        n_above_threshold += (f_batch > r_threshold).sum(dim=1).float()

        # Compute Poisson probs for this batch: (N, batch_size, R)
        log_poisson = (lambda_batch[:, :, None] * r_values[None, None, :]
                       - f_batch[:, :, None]
                       - log_r_fact[None, None, :])
        poisson_probs = torch.exp(log_poisson)  # (N, batch_size, R)

        # Accumulate p_true (sum over samples, will divide later)
        p_true_sum += poisson_probs.sum(dim=1)  # (N, R)

        # E[log(r!)] contribution: weight log_r_fact by Poisson probs
        # Sum over r for low and high ranges, sum over batch samples
        weighted_logr = poisson_probs * log_r_fact[None, None, :]  # (N, batch_size, R)
        E_logr_low_sum += weighted_logr[:, :, :r_threshold].sum(dim=2).sum(dim=1)  # (N,)
        E_logr_high_sum += weighted_logr[:, :, r_threshold:].sum(dim=2).sum(dim=1)  # (N,)

        # Free memory
        del eps, lambda_batch, f_batch, log_poisson, poisson_probs, weighted_logr

    # Compute final averages
    pct_above_threshold = 100 * n_above_threshold / total_samples
    p_true = p_true_sum / total_samples  # (N, R)

    # Compute H_marg contribution from r <= threshold vs r > threshold
    p_log_p = torch.xlogy(p_true, p_true)  # (N, R)
    H_contribution = -p_log_p  # contribution to entropy from each r

    H_low_r = H_contribution[:, :r_threshold].sum(dim=1)   # r = 0 to threshold-1
    H_high_r = H_contribution[:, r_threshold:].sum(dim=1)  # r = threshold to r_max-1
    H_total = H_low_r + H_high_r

    pct_H_from_high_r = 100 * H_high_r / (H_total + 1e-10)

    # =========================================================================
    # H_cond Analysis: E[log(r!)] term contribution from high r
    # =========================================================================
    E_logr_low_mc = E_logr_low_sum / total_samples  # (N,)
    E_logr_high_mc = E_logr_high_sum / total_samples  # (N,)
    E_logr_total_mc = E_logr_low_mc + E_logr_high_mc
    pct_E_logr_high_mc = 100 * E_logr_high_mc / (E_logr_total_mc + 1e-10)

    # =========================================================================
    # Print Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"MC SAMPLING DIAGNOSTICS (r_threshold = {r_threshold}, batch_size = {batch_size})")
    print("=" * 70)

    print(f"\n--- Lambda Sampling Statistics ---")
    print(f"  Total samples per query point: {total_samples}")
    print(f"  Samples with f=exp(λ) > {r_threshold}:")
    print(f"    Mean across x:  {n_above_threshold.mean():.1f} ({pct_above_threshold.mean():.2f}%)")
    print(f"    Max across x:   {n_above_threshold.max():.0f} ({pct_above_threshold.max():.2f}%)")

    # Find the query point with max uncertainty
    max_sigma_idx = sigma.argmax()
    print(f"\n  At max uncertainty point (x={x_candidates[max_sigma_idx].item():.3f}, σ={sigma[max_sigma_idx].item():.3f}):")
    print(f"    μ = {mu[max_sigma_idx].item():.3f}")
    print(f"    Samples with f > {r_threshold}: {n_above_threshold[max_sigma_idx]:.0f} ({pct_above_threshold[max_sigma_idx]:.2f}%)")
    print(f"    λ needed for f>{r_threshold}: {np.log(r_threshold):.2f}")
    print(f"    That's {(np.log(r_threshold) - mu[max_sigma_idx].item()) / sigma[max_sigma_idx].item():.2f} sigma above mean")

    print(f"\n--- H_marg Contribution Analysis (MC) ---")
    print(f"  From r <= {r_threshold-1}:  mean = {H_low_r.mean():.4f}")
    print(f"  From r >= {r_threshold}:    mean = {H_high_r.mean():.4f}")
    print(f"  % from high r: mean = {pct_H_from_high_r.mean():.2f}%, max = {pct_H_from_high_r.max():.2f}%")

    print(f"\n--- H_cond E[log(r!)] Contribution Analysis (MC) ---")
    print(f"  From r <= {r_threshold-1}:  mean = {E_logr_low_mc.mean():.4f}")
    print(f"  From r >= {r_threshold}:    mean = {E_logr_high_mc.mean():.4f}")
    print(f"  % from high r: mean = {pct_E_logr_high_mc.mean():.2f}%, max = {pct_E_logr_high_mc.max():.2f}%")

    print(f"\n--- At max uncertainty point ---")
    i = max_sigma_idx
    print(f"  H_marg: H_low={H_low_r[i]:.4f}, H_high={H_high_r[i]:.4f}, %high={pct_H_from_high_r[i]:.2f}%")
    print(f"  E[log(r!)]: low={E_logr_low_mc[i]:.4f}, high={E_logr_high_mc[i]:.4f}, %high={pct_E_logr_high_mc[i]:.2f}%")

    # Check MC normalization
    p_true_sum = p_true.sum(dim=1)
    print(f"\nMC normalization: min={p_true_sum.min():.4f}, max={p_true_sum.max():.4f}")

    return {
        'pct_above_threshold': pct_above_threshold,
        'pct_H_from_high_r_mc': pct_H_from_high_r,
        'pct_E_logr_high_mc': pct_E_logr_high_mc,
    }


def evaluate_nd_utility_MC(model, x_candidates, max_r=MAX_R, n_samples=5000):
    """Evaluate nd_utility_MC (Monte Carlo) at candidate points."""
    model.eval()

    with torch.no_grad():
        posterior = model(x_candidates)
        mu = posterior.mean
        sigma2 = posterior.variance

        utility = nd_utility_MC(mu, sigma2, r_max=max_r, n_samples=n_samples)

    return utility


def evaluate_nd_utility_MC_batched(model, x_candidates, max_r=MAX_R, n_samples=100000, batch_size=10000):
    """
    Evaluate nd_utility_MC_batched (batched Monte Carlo) at candidate points.

    This allows testing with very high sample counts (100k-1M+) without
    exhausting GPU memory.
    """
    model.eval()

    with torch.no_grad():
        posterior = model(x_candidates)
        mu = posterior.mean
        sigma2 = posterior.variance

        utility = nd_utility_MC_batched(
            mu, sigma2, r_max=max_r, n_samples=n_samples, batch_size=batch_size
        )

    return utility





def evaluate_nd_utility_new(model, x_candidates, max_r=MAX_R):
    """
    Evaluate nd_utility_new (Laplace approximation) at candidate points.

    This uses the log-space Laplace approximation which is numerically stable
    and correctly normalized (Σp_Lap(r) ≈ 1.0) even for high uncertainty.
    """
    model.eval()

    with torch.no_grad():
        posterior = model(x_candidates)
        mu = posterior.mean
        sigma2 = posterior.variance

        utility = nd_utility_new(mu, sigma2, r_max=max_r)

    return utility


def evaluate_nd_utility_hybrid(model, x_candidates, max_r=MAX_R, n_samples=10000, batch_size=500):
    """
    Evaluate nd_utility_hybrid at candidate points.

    Uses Laplace approximation for H_marg and MC sampling for H_cond.
    """
    model.eval()

    with torch.no_grad():
        posterior = model(x_candidates)
        mu = posterior.mean
        sigma2 = posterior.variance

        utility = nd_utility_hybrid(
            mu, sigma2, r_max=max_r, n_samples=n_samples, batch_size=batch_size
        )

    return utility


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("Test: Utility Function Comparison")
    print("=" * 70)

    # =========================================================================
    # Setup: Create and train GP (same as test_dirac_delta_collapse.py)
    # =========================================================================
    n_train = 100
    train_x = torch.linspace(X_MIN, X_MAX, n_train, dtype=DTYPE, device=DEVICE)
    train_y = generate_poisson_data(train_x, lambda_true)

    # print(f"\nTraining points: {train_x.cpu().numpy()}")
    # print(f"Observed counts: {train_y.cpu().numpy()}")

    # inducing_points = train_x[::2].clone()
    inducing_points = train_x.clone()



    model = VariationalGP(inducing_points).to(DEVICE)
    likelihood = PoissonLikelihood().to(DEVICE)

    print("\nTraining GP...")

    model, likelihood = train_gp(model, likelihood, train_x, train_y, n_iterations=300)

    # =========================================================================
    # Evaluate utilities on candidate grid
    # =========================================================================
    x_candidates = torch.linspace(X_MIN, X_MAX, 200, dtype=DTYPE, device=DEVICE)

    n_mc_samples = 100_000
    batch_size = 3000

    n_quadrature = 200

    print(f"\nEvaluating nd_utility_new (Laplace, r_max={MAX_R})...")
    u_laplace = evaluate_nd_utility_new(model, x_candidates, max_r=MAX_R)

    print(f"Evaluating nd_utility_NUMERICAL (Gauss-Hermite, {n_quadrature} points, MAX_R={MAX_R})...")
    u_numerical = evaluate_nd_utility_NUMERICAL(model, x_candidates, max_r=MAX_R, n_quadrature=n_quadrature)

    # Batched MC with higher sample counts
    print(f"Evaluating nd_utility_MC_batched (100k samples, MAX_R={MAX_R})...")
    u_mc = evaluate_nd_utility_MC_batched(model, x_candidates, max_r=MAX_R, n_samples=n_mc_samples, batch_size=batch_size)

    # Hybrid: Laplace for H_marg, MC for H_cond
    print(f"Evaluating nd_utility_hybrid (Laplace H_marg + MC H_cond, {n_mc_samples} samples)...")
    u_hybrid = evaluate_nd_utility_hybrid(model, x_candidates, max_r=MAX_R, n_samples=n_mc_samples, batch_size=batch_size)

    # Run diagnostics to understand MC sampling behavior
    diagnostics = diagnose_mc_sampling(
        model, x_candidates, r_max=MAX_R, n_samples=n_mc_samples, r_threshold=100
    )

    # =========================================================================
    # Get GP posterior for visualization
    # =========================================================================
    model.eval()
    with torch.no_grad():
        posterior = model(x_candidates)
        mu = posterior.mean
        sigma = posterior.variance.sqrt()
        true_lambda = lambda_true(x_candidates)

    # =========================================================================
    # Compute statistics
    # =========================================================================
    # Laplace vs MC
    diff_lap_mc = (u_mc - u_laplace).abs()
    rel_diff_lap_mc = diff_lap_mc / (u_mc.abs() + 1e-10)

    # Laplace vs NUMERICAL
    diff_lap_num = (u_numerical - u_laplace).abs()
    rel_diff_lap_num = diff_lap_num / (u_numerical.abs() + 1e-10)

    # Numerical vs MC
    diff_num_mc = (u_numerical - u_mc).abs()
    rel_diff_num_mc = diff_num_mc / (u_mc.abs() + 1e-10)

    # Hybrid vs Numerical (ground truth comparison)
    diff_hybrid_num = (u_hybrid - u_numerical).abs()
    rel_diff_hybrid_num = diff_hybrid_num / (u_numerical.abs() + 1e-10)

    # Hybrid vs Laplace
    diff_hybrid_lap = (u_hybrid - u_laplace).abs()
    rel_diff_hybrid_lap = diff_hybrid_lap / (u_laplace.abs() + 1e-10)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nnd_utility_new (Laplace) range:  [{u_laplace.min():.6f}, {u_laplace.max():.6f}]")
    print(f"nd_utility_MC range:             [{u_mc.min():.6f}, {u_mc.max():.6f}]")
    print(f"nd_utility_NUMERICAL range:      [{u_numerical.min():.6f}, {u_numerical.max():.6f}]")
    print(f"nd_utility_hybrid range:         [{u_hybrid.min():.6f}, {u_hybrid.max():.6f}]")

    print(f"\n--- Laplace vs MC ---")
    print(f"  Mean rel error: {rel_diff_lap_mc.mean():.4%}")
    print(f"  Max rel error:  {rel_diff_lap_mc.max():.4%}")
    print(f"  Mean abs error: {diff_lap_mc.mean():.6e}")
    print(f"  Max abs error:  {diff_lap_mc.max():.6e}")

    print(f"\n--- Laplace vs NUMERICAL ---")
    print(f"  Mean rel error: {rel_diff_lap_num.mean():.4%}")
    print(f"  Max rel error:  {rel_diff_lap_num.max():.4%}")
    print(f"  Mean abs error: {diff_lap_num.mean():.6e}")
    print(f"  Max abs error:  {diff_lap_num.max():.6e}")

    print(f"\n--- Numerical vs MC ({n_mc_samples} samples) ---")
    print(f"  Mean rel error: {rel_diff_num_mc.mean():.4%}")
    print(f"  Max rel error:  {rel_diff_num_mc.max():.4%}")
    print(f"  Mean abs error: {diff_num_mc.mean():.6e}")
    print(f"  Max abs error:  {diff_num_mc.max():.6e}")

    print(f"\n--- Hybrid vs NUMERICAL ---")
    print(f"  Mean rel error: {rel_diff_hybrid_num.mean():.4%}")
    print(f"  Max rel error:  {rel_diff_hybrid_num.max():.4%}")
    print(f"  Mean abs error: {diff_hybrid_num.mean():.6e}")
    print(f"  Max abs error:  {diff_hybrid_num.max():.6e}")

    print(f"\n--- Hybrid vs Laplace ---")
    print(f"  Mean rel error: {rel_diff_hybrid_lap.mean():.4%}")
    print(f"  Max rel error:  {rel_diff_hybrid_lap.max():.4%}")
    print(f"  Mean abs error: {diff_hybrid_lap.mean():.6e}")
    print(f"  Max abs error:  {diff_hybrid_lap.max():.6e}")

    # =========================================================================
    # Compute firing rate statistics
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

    u_laplace_np = u_laplace.cpu().numpy()
    u_mc_np = u_mc.cpu().numpy()
    u_numerical_np = u_numerical.cpu().numpy()
    u_hybrid_np = u_hybrid.cpu().numpy()

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
    ax1.plot(x_np, u_laplace_np, 'b-', linewidth=2.5, label='nd_utility_new (Laplace)')
    ax1.plot(x_np, u_mc_np, 'r--', linewidth=2, alpha=0.8,
             label=f'nd_utility_MC')
    ax1.plot(x_np, u_numerical_np, 'g:', linewidth=2.5, alpha=0.9,
             label=f'nd_utility_NUMERICAL')
    ax1.plot(x_np, u_hybrid_np, 'm-.', linewidth=2, alpha=0.9,
             label=f'nd_utility_hybrid')

    for i, tx in enumerate(train_x_np):
        ax1.axvline(x=tx, color='orange', alpha=0.3, linewidth=1, linestyle=':',
                    label='Training x' if i == 0 else None)

    ax1.set_ylabel('Utility U(x*)')
    ax1.set_title('Utility Comparison: Laplace vs MC vs Numerical Integration')
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

    f_std_approx = f_mean_np * sigma_np
    ax3.fill_between(x_np,
                     np.maximum(0, f_mean_np - 2*f_std_approx),
                     f_mean_np + 2*f_std_approx,
                     alpha=0.3, color='blue', label='±2σ (approx)')

    ax3.axhline(y=MAX_R, color='red', linestyle='--', linewidth=2,
                label=f'r_max = {MAX_R}')

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

    ax4.semilogy(x_np, rel_diff_lap_mc.cpu().numpy() + 1e-10, 'b-', linewidth=2.5,
                 label=f'Laplace vs MC (mean: {rel_diff_lap_mc.mean():.2%})')
    ax4.semilogy(x_np, rel_diff_num_mc.cpu().numpy() + 1e-10, 'g--', linewidth=2,
                 label=f'Numerical vs MC (mean: {rel_diff_num_mc.mean():.2%})')

    ax4_twin = ax4.twinx()
    ax4_twin.plot(x_np, f_mean_corrected_np, 'k--', linewidth=1.5, alpha=0.5,
                  label='E[f] = exp(μ+σ²/2)')
    ax4_twin.set_ylabel('Expected firing rate E[f]', color='black')
    ax4_twin.tick_params(axis='y', labelcolor='black')

    ax4.set_xlabel('x')
    ax4.set_ylabel('Relative Error')
    ax4.set_title('Error Analysis: Laplace and Numerical vs MC (Ground Truth)')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(__file__).parent / 'test_nd_utility_MC_result.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved plot to: {save_path}")
    plt.close()

    print("\nTest completed!")

    return {
        'laplace_vs_mc_error': rel_diff_lap_mc.mean().item(),
        'laplace_vs_num_error': rel_diff_lap_num.mean().item(),
        'numerical_vs_mc_error': rel_diff_num_mc.mean().item(),
        'hybrid_vs_num_error': rel_diff_hybrid_num.mean().item(),
        'hybrid_vs_laplace_error': rel_diff_hybrid_lap.mean().item(),
    }


if __name__ == "__main__":
    results = main()
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Laplace vs MC error:        {results['laplace_vs_mc_error']:.4%}")
    print(f"Laplace vs Numerical error: {results['laplace_vs_num_error']:.4%}")
    print(f"Numerical vs MC:            {results['numerical_vs_mc_error']:.4%}")
    print(f"Hybrid vs Numerical:        {results['hybrid_vs_num_error']:.4%}")
    print(f"Hybrid vs Laplace:          {results['hybrid_vs_laplace_error']:.4%}")
    print("=" * 70)
