"""
Diagnostic Script: Full Utility Decomposition
==============================================
Created by Claude to visualize H_marg, H_cond, variance reduction ratio,
and utility separately to understand distribution-aware utility behavior.

Key insight from single sample analysis: Negative cross-covariance can cause
high utility OUTSIDE the p(x) Gaussian because:
1. σ²_cond = Σ_** - Σ_x*²/Σ_xx uses SQUARED cross-covariance
2. Negative correlations also reduce variance

Related: See /home/idv-eqs8-pza/.claude/plans/encapsulated-shimmying-key.md
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
from utility import laplace_approximations_new


def compute_decomposition_for_gaussian_px(model, x_candidates, gaussian_mean, gaussian_std,
                                           n_samples=100, n_lambda_samples=10, r_max=MAX_R):
    """
    Compute H_marg, H_cond, and variance reduction ratio for Gaussian p(x).
    Returns detailed decomposition for analysis.
    """
    model.eval()
    K = len(x_candidates)
    device = x_candidates.device
    dtype = x_candidates.dtype

    r_values = torch.arange(0, r_max, dtype=dtype, device=device)

    # Get variational parameters
    z = model.variational_strategy.inducing_points
    if z.ndim == 2 and z.shape[1] == 1:
        z = z.squeeze(-1)
    M = z.shape[0]

    V = model.variational_strategy.variational_distribution.covariance_matrix
    K_zz = model.covar_module(z, z).evaluate()
    K_zz_inv = torch.linalg.inv(K_zz + 1e-6 * torch.eye(M, device=device, dtype=dtype))

    # Get posterior at all query points
    with torch.no_grad():
        posterior_star = model(x_candidates)
        mu_star = posterior_star.mean  # (K,)
        Sigma_star_star = posterior_star.variance  # (K,)

    # Compute H_marg for all query points
    p_r_marg, log_p_r_marg = laplace_approximations_new(mu_star, Sigma_star_star, r_values)
    H_marg = -torch.sum(p_r_marg * log_p_r_marg, dim=1)  # (K,)

    # Sample x from truncated Gaussian p(x)
    samples = []
    while len(samples) < n_samples:
        s = torch.randn(n_samples * 2, dtype=dtype, device=device) * gaussian_std + gaussian_mean
        valid = (s >= 2*X_MIN) & (s <= 2*X_MAX)
        samples.extend(s[valid].tolist())
    x_samples = torch.tensor(samples[:n_samples], dtype=dtype, device=device)
    N = n_samples

    # Get posterior at sample points
    with torch.no_grad():
        posterior_i = model(x_samples)
        mu_i = posterior_i.mean  # (N,)
        Sigma_ii = posterior_i.variance  # (N,)

    # Sample lambda(x_i) from posterior
    L = n_lambda_samples
    eps = torch.randn(N, L, dtype=dtype, device=device)
    lambda_samples = mu_i[:, None] + torch.sqrt(Sigma_ii[:, None]) * eps  # (N, L)

    # Compute kernel matrices for cross-covariance
    k_star_z = model.covar_module(x_candidates, z).evaluate()  # (K, M)
    k_i_z = model.covar_module(x_samples, z).evaluate()  # (N, M)
    k_i_star = model.covar_module(x_samples, x_candidates).evaluate()  # (N, K)

    u_star = K_zz_inv @ k_star_z.T  # (M, K)
    u_i = K_zz_inv @ k_i_z.T  # (M, N)

    # Cross-covariance
    epistemic_cov = u_i.T @ V @ u_star  # (N, K)
    prior_cov_via_inducing = u_i.T @ K_zz @ u_star  # (N, K)
    residual_cov = k_i_star - prior_cov_via_inducing  # (N, K)
    Sigma_i_star = residual_cov + epistemic_cov  # (N, K)

    # Clamp cross-covariance for stability
    max_abs_cov = torch.sqrt(Sigma_ii[:, None] * Sigma_star_star[None, :]) * 0.999
    Sigma_i_star = torch.clamp(Sigma_i_star, min=-max_abs_cov, max=max_abs_cov)

    # Conditional variance: sigma2_cond = Sigma_** - Sigma_x*^2 / Sigma_xx
    sigma2_cond = Sigma_star_star[None, :] - Sigma_i_star**2 / Sigma_ii[:, None]  # (N, K)
    sigma2_cond = torch.clamp(sigma2_cond, min=1e-8)

    # Variance reduction ratio: avg over samples
    var_reduction_ratio = (sigma2_cond / Sigma_star_star[None, :]).mean(dim=0)  # (K,)

    # Average cross-covariance (to show correlation structure)
    avg_cross_cov = Sigma_i_star.mean(dim=0)  # (K,)

    # Conditional mean
    innovation = lambda_samples - mu_i[:, None]  # (N, L)
    regression_coef = Sigma_i_star / Sigma_ii[:, None]  # (N, K)
    mu_cond = mu_star[None, None, :] + regression_coef[:, None, :] * innovation[:, :, None]  # (N, L, K)

    # Compute H_cond for each (sample, lambda) pair
    H_cond_samples = torch.zeros(N, L, K, dtype=dtype, device=device)

    for n in range(N):
        for l in range(L):
            p_r_cond, log_p_r_cond = laplace_approximations_new(
                mu=mu_cond[n, l],  # (K,)
                sigma2=sigma2_cond[n],  # (K,)
                r=r_values
            )
            H_cond_samples[n, l] = -torch.sum(p_r_cond * log_p_r_cond, dim=1)  # (K,)

    H_cond = H_cond_samples.mean(dim=(0, 1))  # (K,)

    # Utility
    utility = H_marg - H_cond

    return {
        'H_marg': H_marg.cpu().numpy(),
        'H_cond': H_cond.cpu().numpy(),
        'utility': utility.cpu().numpy(),
        'var_reduction_ratio': var_reduction_ratio.cpu().numpy(),
        'avg_cross_cov': avg_cross_cov.cpu().numpy(),
        'Sigma_star_star': Sigma_star_star.cpu().numpy(),
        'mu_star': mu_star.cpu().numpy()
    }


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("Diagnostic: Full Utility Decomposition")
    print("=" * 70)

    # Setup identical to test_distribution_aware_utility.py
    n_train = 5
    train_x = torch.linspace(0.3, 0.9, n_train, dtype=DTYPE, device=DEVICE)
    train_y = generate_poisson_data(train_x, lambda_true)

    inducing_points = train_x.clone()
    model = VariationalGP(inducing_points).to(DEVICE)
    likelihood = PoissonLikelihood().to(DEVICE)

    print("\nTraining GP...")
    model, likelihood = train_gp(model, likelihood, train_x, train_y, n_iterations=300)
    model.eval()

    # Candidate grid
    x_candidates = torch.linspace(2*X_MIN, 2*X_MAX, 500, dtype=DTYPE, device=DEVICE)

    # Gaussian p(x) parameters
    gaussian_mean = 0.7
    gaussian_std = 0.15

    print(f"\nComputing decomposition for Gaussian p(x) centered at {gaussian_mean}...")
    decomp = compute_decomposition_for_gaussian_px(
        model, x_candidates, gaussian_mean, gaussian_std,
        n_samples=200, n_lambda_samples=20
    )

    # Also compute original nd_utility
    print("Computing original nd_utility...")
    u_original = evaluate_utility(model, x_candidates).cpu().numpy()

    # Get GP posterior for plotting
    with torch.no_grad():
        posterior = model(x_candidates)
        gp_mean = posterior.mean.cpu().numpy()
        gp_std = posterior.variance.sqrt().cpu().numpy()
        true_lambda = lambda_true(x_candidates).cpu().numpy()

    # Convert to numpy
    x_np = x_candidates.cpu().numpy()
    train_x_np = train_x.cpu().numpy()
    train_y_np = train_y.cpu().numpy()

    # Create figure with 6 subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Gaussian p(x) for overlay
    gaussian_pdf = np.exp(-0.5 * ((x_np - gaussian_mean) / gaussian_std)**2)
    gaussian_pdf_norm = gaussian_pdf / gaussian_pdf.max()

    # -------------------------------------------------------------------------
    # Subplot 1: GP Posterior
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(x_np, true_lambda, 'k--', linewidth=2, label='True λ(x)')
    ax1.plot(x_np, gp_mean, 'b-', linewidth=2, label='GP mean')
    ax1.fill_between(x_np, gp_mean - 2*gp_std, gp_mean + 2*gp_std,
                     alpha=0.25, color='blue', label='±2σ')
    for i, tx in enumerate(train_x_np):
        ax1.axvline(x=tx, color='red', alpha=0.3, linewidth=1, linestyle=':',
                    label='Training x' if i == 0 else None)
    ax1.fill_between(x_np, ax1.get_ylim()[0], ax1.get_ylim()[0] + (ax1.get_ylim()[1]-ax1.get_ylim()[0])*0.15*gaussian_pdf_norm,
                     alpha=0.2, color='orange', label='p(x) Gaussian')
    ax1.set_ylabel('λ(x)')
    ax1.set_title('GP Posterior')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Subplot 2: GP Variance (Σ_**)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.plot(x_np, decomp['Sigma_star_star'], 'b-', linewidth=2, label='Σ_** (GP variance)')
    ax2.fill_between(x_np, 0, gaussian_pdf_norm * decomp['Sigma_star_star'].max() * 0.3,
                     alpha=0.2, color='orange', label='p(x) Gaussian')
    for tx in train_x_np:
        ax2.axvline(x=tx, color='red', alpha=0.3, linewidth=1, linestyle=':')
    ax2.set_ylabel('Variance')
    ax2.set_title('GP Posterior Variance Σ_**')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Subplot 3: Average Cross-Covariance
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.plot(x_np, decomp['avg_cross_cov'], 'purple', linewidth=2, label='E[Σ_x*] (avg cross-cov)')
    ax3.axhline(y=0, color='gray', linewidth=1, linestyle='-')
    ax3.fill_between(x_np, ax3.get_ylim()[0], ax3.get_ylim()[0] + (ax3.get_ylim()[1]-ax3.get_ylim()[0])*0.15*gaussian_pdf_norm,
                     alpha=0.2, color='orange', label='p(x) Gaussian')
    for tx in train_x_np:
        ax3.axvline(x=tx, color='red', alpha=0.3, linewidth=1, linestyle=':')
    ax3.set_ylabel('Cross-Covariance')
    ax3.set_title('Average Cross-Covariance E[Σ_x*] (can be NEGATIVE!)')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Subplot 4: Variance Reduction Ratio
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.plot(x_np, decomp['var_reduction_ratio'], 'green', linewidth=2, label='E[σ²_cond / Σ_**]')
    ax4.axhline(y=1, color='gray', linewidth=1, linestyle='--', label='No reduction')
    ax4.fill_between(x_np, 0, gaussian_pdf_norm * 0.3, alpha=0.2, color='orange', label='p(x) Gaussian')
    for tx in train_x_np:
        ax4.axvline(x=tx, color='red', alpha=0.3, linewidth=1, linestyle=':')
    ax4.set_ylabel('Ratio')
    ax4.set_ylim(0, 1.1)
    ax4.set_title('Variance Reduction Ratio (lower = more reduction)')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Subplot 5: H_marg and H_cond
    # -------------------------------------------------------------------------
    ax5 = axes[2, 0]
    ax5.plot(x_np, decomp['H_marg'], 'b-', linewidth=2, label='H_marg')
    ax5.plot(x_np, decomp['H_cond'], 'r--', linewidth=2, label='H_cond')
    ax5.fill_between(x_np, 0, gaussian_pdf_norm * decomp['H_marg'].max() * 0.2,
                     alpha=0.2, color='orange', label='p(x) Gaussian')
    for tx in train_x_np:
        ax5.axvline(x=tx, color='red', alpha=0.3, linewidth=1, linestyle=':')
    ax5.set_xlabel('x')
    ax5.set_ylabel('Entropy')
    ax5.set_title('Marginal vs Conditional Entropy')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Subplot 6: Utility Comparison
    # -------------------------------------------------------------------------
    ax6 = axes[2, 1]
    ax6.plot(x_np, decomp['utility'], 'm-', linewidth=2, label='Dist-aware (Gaussian)')
    ax6.plot(x_np, u_original, 'b--', linewidth=1.5, alpha=0.7, label='Original nd_utility (scaled)')

    # Scale original to match range
    scale = decomp['utility'].max() / u_original.max() if u_original.max() > 0 else 1
    ax6.plot(x_np, u_original * scale, 'b:', linewidth=1, alpha=0.5)

    ax6.fill_between(x_np, 0, gaussian_pdf_norm * decomp['utility'].max() * 0.3,
                     alpha=0.2, color='orange', label='p(x) Gaussian')
    ax6.axhline(y=0, color='gray', linewidth=1, linestyle='-')
    for tx in train_x_np:
        ax6.axvline(x=tx, color='red', alpha=0.3, linewidth=1, linestyle=':')
    ax6.set_xlabel('x')
    ax6.set_ylabel('Utility')
    ax6.set_title('Utility = H_marg - H_cond')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(__file__).parent / 'diagnose_utility_decomposition_result.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved plot to: {save_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Find key locations
    inside_mask = np.abs(x_np - gaussian_mean) < gaussian_std
    outside_mask = ~inside_mask

    print(f"\nInside Gaussian (|x - {gaussian_mean}| < {gaussian_std}):")
    print(f"  Avg H_marg: {decomp['H_marg'][inside_mask].mean():.4f}")
    print(f"  Avg H_cond: {decomp['H_cond'][inside_mask].mean():.4f}")
    print(f"  Avg Utility: {decomp['utility'][inside_mask].mean():.4f}")
    print(f"  Avg Var Reduction: {decomp['var_reduction_ratio'][inside_mask].mean():.4f}")
    print(f"  Avg Cross-Cov: {decomp['avg_cross_cov'][inside_mask].mean():.4f}")

    print(f"\nOutside Gaussian:")
    print(f"  Avg H_marg: {decomp['H_marg'][outside_mask].mean():.4f}")
    print(f"  Avg H_cond: {decomp['H_cond'][outside_mask].mean():.4f}")
    print(f"  Avg Utility: {decomp['utility'][outside_mask].mean():.4f}")
    print(f"  Avg Var Reduction: {decomp['var_reduction_ratio'][outside_mask].mean():.4f}")
    print(f"  Avg Cross-Cov: {decomp['avg_cross_cov'][outside_mask].mean():.4f}")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
The cross-covariance E[Σ_x*] can be NEGATIVE at certain x* locations.
Since σ²_cond = Σ_** - Σ_x*²/Σ_xx uses the SQUARED cross-covariance,
both positive AND negative correlations reduce conditional variance!

This explains why utility peaks OUTSIDE the Gaussian p(x):
- Points outside (e.g., x*=1.3) can have NEGATIVE correlation with
  points inside (x ~ Gaussian centered at 0.7)
- The negative correlation still reduces variance
- Combined with high H_marg outside (high GP uncertainty),
  this gives high utility outside the Gaussian
""")


if __name__ == "__main__":
    main()
