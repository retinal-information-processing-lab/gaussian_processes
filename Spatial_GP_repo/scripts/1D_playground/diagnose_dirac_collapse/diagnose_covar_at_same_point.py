"""
Diagnose covariance structure when querying same point twice.

When x_sample = x_star, we expect:
  var_star = var_sample = cross_cov
  sigma2_cond = var_star - cross_cov^2 / var_sample = 0

Check if this holds outside the training domain.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from gp_utility_playground import (
    VariationalGP, PoissonLikelihood, train_gp,
    generate_poisson_data, lambda_true,
    DEVICE, DTYPE, X_MIN, X_MAX
)


def get_covar_at_same_point(model, x_point):
    """Get covariance components when querying same point as observation and query."""
    model.eval()
    device = x_point.device
    dtype = x_point.dtype

    # Create tensor for the point (query it twice to get joint covariance)
    x_tensor = torch.tensor([x_point.item(), x_point.item()], dtype=dtype, device=device)

    with torch.no_grad():
        posterior = model(x_tensor)
        full_covar = posterior.covariance_matrix

        var_sample = full_covar[0, 0].item()  # Variance at "observation" point
        var_star = full_covar[1, 1].item()    # Variance at "query" point
        cross_cov = full_covar[0, 1].item()   # Cross-covariance

        # Compute sigma2_cond using the formula
        sigma2_cond = var_star - (cross_cov ** 2) / var_sample

    return var_sample, var_star, cross_cov, sigma2_cond


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Diagnose: Covariance at Same Point")
    print("=" * 60)

    # Train GP
    n_train = 30
    train_x = torch.linspace(X_MIN, X_MAX, n_train, dtype=DTYPE, device=DEVICE)
    train_y = generate_poisson_data(train_x, lambda_true)

    model = VariationalGP(train_x.clone()).to(DEVICE)
    likelihood = PoissonLikelihood().to(DEVICE)

    print("\nTraining GP...")
    model, likelihood = train_gp(model, likelihood, train_x, train_y, n_iterations=300)

    # Evaluate across extended domain
    x_plot_min = X_MIN - 3
    x_plot_max = X_MAX + 3
    n_points = 100
    x_candidates = torch.linspace(x_plot_min, x_plot_max, n_points, dtype=DTYPE, device=DEVICE)

    # Collect covariance data
    var_samples = []
    var_stars = []
    cross_covs = []
    sigma2_conds = []

    print("\nComputing covariance at same points...")
    for i in tqdm(range(n_points)):
        vs, vst, cc, s2c = get_covar_at_same_point(model, x_candidates[i:i+1])
        var_samples.append(vs)
        var_stars.append(vst)
        cross_covs.append(cc)
        sigma2_conds.append(s2c)

    var_samples = np.array(var_samples)
    var_stars = np.array(var_stars)
    cross_covs = np.array(cross_covs)
    sigma2_conds = np.array(sigma2_conds)
    x_np = x_candidates.cpu().numpy()

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Check equality
    diff_var = np.abs(var_samples - var_stars)
    diff_cov = np.abs(var_stars - cross_covs)

    print(f"\nExpected: var_sample = var_star = cross_cov")
    print(f"  |var_sample - var_star| max: {diff_var.max():.2e}")
    print(f"  |var_star - cross_cov| max: {diff_cov.max():.2e}")
    print(f"\nExpected: sigma2_cond = 0")
    print(f"  sigma2_cond range: [{sigma2_conds.min():.2e}, {sigma2_conds.max():.2e}]")

    # Find where discrepancies are largest
    idx_max_diff = np.argmax(diff_cov)
    print(f"\nMax discrepancy at x = {x_np[idx_max_diff]:.2f}:")
    print(f"  var_star = {var_stars[idx_max_diff]:.6f}")
    print(f"  cross_cov = {cross_covs[idx_max_diff]:.6f}")
    print(f"  sigma2_cond = {sigma2_conds[idx_max_diff]:.2e}")

    # Sample specific points inside and outside
    inside_idx = n_points // 2  # Middle of training domain
    outside_idx_left = 5  # Far left
    outside_idx_right = n_points - 5  # Far right

    print(f"\nInside training domain (x = {x_np[inside_idx]:.2f}):")
    print(f"  var_star = {var_stars[inside_idx]:.6f}, cross_cov = {cross_covs[inside_idx]:.6f}")
    print(f"  sigma2_cond = {sigma2_conds[inside_idx]:.2e}")

    print(f"\nOutside left (x = {x_np[outside_idx_left]:.2f}):")
    print(f"  var_star = {var_stars[outside_idx_left]:.6f}, cross_cov = {cross_covs[outside_idx_left]:.6f}")
    print(f"  sigma2_cond = {sigma2_conds[outside_idx_left]:.2e}")

    print(f"\nOutside right (x = {x_np[outside_idx_right]:.2f}):")
    print(f"  var_star = {var_stars[outside_idx_right]:.6f}, cross_cov = {cross_covs[outside_idx_right]:.6f}")
    print(f"  sigma2_cond = {sigma2_conds[outside_idx_right]:.2e}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Variances and cross-covariance
    ax1 = axes[0]
    ax1.plot(x_np, var_stars, 'b-', linewidth=2, label='var_star')
    ax1.plot(x_np, var_samples, 'g--', linewidth=2, label='var_sample')
    ax1.plot(x_np, cross_covs, 'r:', linewidth=2, label='cross_cov')
    ax1.axvline(x=X_MIN, color='gray', linestyle='--', alpha=0.5, label='Training region')
    ax1.axvline(x=X_MAX, color='gray', linestyle='--', alpha=0.5)
    ax1.axvspan(X_MIN, X_MAX, alpha=0.1, color='green')
    ax1.set_ylabel('Covariance')
    ax1.set_title('Covariance components (should all overlap)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Differences
    ax2 = axes[1]
    ax2.semilogy(x_np, diff_var + 1e-16, 'g-', linewidth=2, label='|var_sample - var_star|')
    ax2.semilogy(x_np, diff_cov + 1e-16, 'r-', linewidth=2, label='|var_star - cross_cov|')
    ax2.axvline(x=X_MIN, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=X_MAX, color='gray', linestyle='--', alpha=0.5)
    ax2.axvspan(X_MIN, X_MAX, alpha=0.1, color='green')
    ax2.set_ylabel('Absolute Difference')
    ax2.set_title('Covariance discrepancies (should be ~0)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: sigma2_cond
    ax3 = axes[2]
    ax3.semilogy(x_np, np.abs(sigma2_conds) + 1e-16, 'b-', linewidth=2)
    ax3.axhline(y=1e-8, color='red', linestyle='--', label='Clamp threshold (1e-8)')
    ax3.axvline(x=X_MIN, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=X_MAX, color='gray', linestyle='--', alpha=0.5)
    ax3.axvspan(X_MIN, X_MAX, alpha=0.1, color='green')
    ax3.set_xlabel('x')
    ax3.set_ylabel('|sigma2_cond|')
    ax3.set_title('Conditional variance (should be ~0 everywhere)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(__file__).parent / 'diagnose_covar_at_same_point.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
