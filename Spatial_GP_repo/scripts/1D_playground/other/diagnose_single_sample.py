"""
Diagnostic Script: Single Sample Analysis for Distribution-Aware Utility
=========================================================================
Created by Claude to investigate why distribution-aware utility peaks OUTSIDE
the Gaussian p(x) distribution rather than inside.

This script traces through the utility computation step-by-step with single
x_samples to understand the behavior of cross-covariance, conditional variance,
and entropy terms.

Related: See /home/idv-eqs8-pza/.claude/plans/encapsulated-shimmying-key.md
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add Spatial_GP_repo to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gp_utility_playground import (
    VariationalGP, PoissonLikelihood, train_gp,
    generate_poisson_data, lambda_true,
    DEVICE, DTYPE, X_MIN, X_MAX, MAX_R
)
from utility import laplace_approximations_new


def compute_cross_covariance_manual(model, x_sample, x_star):
    """
    Manually compute cross-covariance between x_sample and x_star.
    Returns Sigma_xx, Sigma_star_star, Sigma_x_star for analysis.
    """
    # Get inducing points and variational parameters
    z = model.variational_strategy.inducing_points
    if z.ndim == 2 and z.shape[1] == 1:
        z = z.squeeze(-1)
    M = z.shape[0]

    V = model.variational_strategy.variational_distribution.covariance_matrix
    K_zz = model.covar_module(z, z).evaluate()
    K_zz_inv = torch.linalg.inv(K_zz + 1e-6 * torch.eye(M, device=DEVICE, dtype=DTYPE))

    # Get posterior at x_sample and x_star
    posterior_sample = model(x_sample.unsqueeze(0))
    posterior_star = model(x_star.unsqueeze(0))

    Sigma_xx = posterior_sample.variance.item()
    Sigma_star_star = posterior_star.variance.item()
    mu_x = posterior_sample.mean.item()
    mu_star = posterior_star.mean.item()

    # Compute cross-covariance
    k_sample_z = model.covar_module(x_sample.unsqueeze(0), z).evaluate()  # (1, M)
    k_star_z = model.covar_module(x_star.unsqueeze(0), z).evaluate()  # (1, M)
    k_sample_star = model.covar_module(x_sample.unsqueeze(0), x_star.unsqueeze(0)).evaluate()  # (1, 1)

    u_sample = K_zz_inv @ k_sample_z.T  # (M, 1)
    u_star = K_zz_inv @ k_star_z.T  # (M, 1)

    # Cross-covariance: Sigma_x* = k(x,x*) + u^T (V - K) u_*
    epistemic_cov = (u_sample.T @ V @ u_star).item()
    prior_cov_via_inducing = (u_sample.T @ K_zz @ u_star).item()
    residual_cov = k_sample_star.item() - prior_cov_via_inducing
    Sigma_x_star = residual_cov + epistemic_cov

    return {
        'Sigma_xx': Sigma_xx,
        'Sigma_star_star': Sigma_star_star,
        'Sigma_x_star': Sigma_x_star,
        'mu_x': mu_x,
        'mu_star': mu_star,
        'k_sample_star': k_sample_star.item(),
        'epistemic_cov': epistemic_cov,
        'residual_cov': residual_cov
    }


def compute_conditional_moments(cov_dict, lambda_sample):
    """
    Compute conditional moments of lambda(x*) given lambda(x) = lambda_sample.

    mu_cond = mu_star + (Sigma_x_star / Sigma_xx) * (lambda_sample - mu_x)
    sigma2_cond = Sigma_star_star - Sigma_x_star^2 / Sigma_xx
    """
    regression_coef = cov_dict['Sigma_x_star'] / cov_dict['Sigma_xx']
    innovation = lambda_sample - cov_dict['mu_x']

    mu_cond = cov_dict['mu_star'] + regression_coef * innovation
    sigma2_cond = cov_dict['Sigma_star_star'] - cov_dict['Sigma_x_star']**2 / cov_dict['Sigma_xx']
    sigma2_cond = max(sigma2_cond, 1e-8)  # Clamp for numerical stability

    return {
        'mu_cond': mu_cond,
        'sigma2_cond': sigma2_cond,
        'regression_coef': regression_coef,
        'innovation': innovation,
        'variance_reduction': cov_dict['Sigma_star_star'] - sigma2_cond
    }


def compute_entropy(mu, sigma2, r_max=MAX_R):
    """Compute entropy H(R | mu, sigma2) using Laplace approximation."""
    mu_t = torch.tensor([mu], dtype=DTYPE, device=DEVICE)
    sigma2_t = torch.tensor([sigma2], dtype=DTYPE, device=DEVICE)
    r_values = torch.arange(0, r_max, dtype=DTYPE, device=DEVICE)

    p_r, log_p_r = laplace_approximations_new(mu_t, sigma2_t, r_values)
    H = -torch.sum(p_r * log_p_r, dim=1).item()

    return H


def analyze_single_sample_case(model, x_sample, x_star, label=""):
    """
    Analyze utility computation for a single (x_sample, x_star) pair.
    """
    print(f"\n{'='*70}")
    print(f"Case: {label}")
    print(f"  x_sample = {x_sample:.4f}")
    print(f"  x_star   = {x_star:.4f}")
    print(f"  distance = {abs(x_star - x_sample):.4f}")
    print(f"{'='*70}")

    x_sample_t = torch.tensor(x_sample, dtype=DTYPE, device=DEVICE)
    x_star_t = torch.tensor(x_star, dtype=DTYPE, device=DEVICE)

    # Step 1: Compute covariances
    cov = compute_cross_covariance_manual(model, x_sample_t, x_star_t)

    print(f"\n--- Covariance Structure ---")
    print(f"  Sigma_xx (variance at x_sample):     {cov['Sigma_xx']:.6f}")
    print(f"  Sigma_** (variance at x_star):       {cov['Sigma_star_star']:.6f}")
    print(f"  Sigma_x* (cross-covariance):         {cov['Sigma_x_star']:.6f}")
    print(f"    - k(x,x*) direct kernel:           {cov['k_sample_star']:.6f}")
    print(f"    - residual_cov:                    {cov['residual_cov']:.6f}")
    print(f"    - epistemic_cov:                   {cov['epistemic_cov']:.6f}")
    print(f"  Correlation: Sigma_x* / sqrt(Sigma_xx * Sigma_**) = {cov['Sigma_x_star'] / np.sqrt(cov['Sigma_xx'] * cov['Sigma_star_star']):.4f}")

    # Step 2: Compute H_marg (marginal entropy at x_star)
    H_marg = compute_entropy(cov['mu_star'], cov['Sigma_star_star'])

    print(f"\n--- Marginal Entropy at x_star ---")
    print(f"  mu_star:     {cov['mu_star']:.4f}")
    print(f"  Sigma_**:    {cov['Sigma_star_star']:.6f}")
    print(f"  H_marg:      {H_marg:.6f}")

    # Step 3: Sample lambda(x) and compute conditional moments + entropy
    # Use multiple lambda samples to show the effect
    print(f"\n--- Conditional Analysis (sampling lambda at x_sample) ---")
    print(f"  mu_x:        {cov['mu_x']:.4f}")
    print(f"  sqrt(Sigma_xx): {np.sqrt(cov['Sigma_xx']):.4f}")

    # Sample 5 lambda values
    np.random.seed(42)
    lambda_samples = cov['mu_x'] + np.sqrt(cov['Sigma_xx']) * np.random.randn(5)

    H_cond_list = []
    print(f"\n  Sample | lambda(x) | mu_cond | sigma2_cond | H_cond | H_marg-H_cond")
    print(f"  " + "-"*75)

    for i, lam in enumerate(lambda_samples):
        cond = compute_conditional_moments(cov, lam)
        H_cond = compute_entropy(cond['mu_cond'], cond['sigma2_cond'])
        H_cond_list.append(H_cond)
        utility = H_marg - H_cond
        print(f"  {i+1:6d} | {lam:9.4f} | {cond['mu_cond']:7.4f} | {cond['sigma2_cond']:11.6f} | {H_cond:6.4f} | {utility:+.4f}")

    avg_H_cond = np.mean(H_cond_list)
    avg_utility = H_marg - avg_H_cond

    # Also compute variance reduction ratio
    sigma2_cond_fixed = cov['Sigma_star_star'] - cov['Sigma_x_star']**2 / cov['Sigma_xx']
    variance_reduction_ratio = sigma2_cond_fixed / cov['Sigma_star_star']

    print(f"\n--- Summary ---")
    print(f"  Variance reduction ratio (sigma2_cond / Sigma_**): {variance_reduction_ratio:.4f}")
    print(f"  Average H_cond:     {avg_H_cond:.6f}")
    print(f"  H_marg:             {H_marg:.6f}")
    print(f"  Ratio H_cond/H_marg: {avg_H_cond/H_marg:.4f}")
    print(f"  Average Utility:    {avg_utility:.6f}")

    return {
        'x_sample': x_sample,
        'x_star': x_star,
        'cov': cov,
        'H_marg': H_marg,
        'avg_H_cond': avg_H_cond,
        'avg_utility': avg_utility,
        'variance_reduction_ratio': variance_reduction_ratio
    }


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("Diagnostic: Single Sample Analysis for Distribution-Aware Utility")
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

    # Define test cases
    # Gaussian p(x) is centered at 0.7 with std 0.15
    gaussian_mean = 0.7

    print(f"\n{'#'*70}")
    print(f"GAUSSIAN p(x) is centered at {gaussian_mean}")
    print(f"Training points at x ∈ [{train_x.min().item():.2f}, {train_x.max().item():.2f}]")
    print(f"{'#'*70}")

    results = []

    # Case 1: x_sample at Gaussian center, x_star at Gaussian center
    results.append(analyze_single_sample_case(
        model, x_sample=0.7, x_star=0.7,
        label="Dirac case: x_sample = x_star = 0.7 (inside Gaussian)"
    ))

    # Case 2: x_sample at Gaussian center, x_star outside (at -0.1)
    results.append(analyze_single_sample_case(
        model, x_sample=0.7, x_star=-0.1,
        label="x_sample=0.7 (Gaussian center), x_star=-0.1 (outside)"
    ))

    # Case 3: x_sample at Gaussian center, x_star outside (at 1.3)
    results.append(analyze_single_sample_case(
        model, x_sample=0.7, x_star=1.3,
        label="x_sample=0.7 (Gaussian center), x_star=1.3 (outside)"
    ))

    # Case 4: x_sample far from query, x_star at query
    results.append(analyze_single_sample_case(
        model, x_sample=-1.0, x_star=0.7,
        label="x_sample=-1.0 (far), x_star=0.7 (inside Gaussian)"
    ))

    # Case 5: x_sample far from query, x_star also far
    results.append(analyze_single_sample_case(
        model, x_sample=-1.0, x_star=-0.1,
        label="x_sample=-1.0 (far), x_star=-0.1 (outside Gaussian)"
    ))

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Case':<55} | {'VarRed':<7} | {'H_cond/H_marg':<13} | {'Utility':<8}")
    print("-" * 90)

    for r in results:
        case_label = f"x_s={r['x_sample']:.1f}, x*={r['x_star']:.1f}"
        print(f"{case_label:<55} | {r['variance_reduction_ratio']:.4f}  | {r['avg_H_cond']/r['H_marg']:.4f}        | {r['avg_utility']:.4f}")

    print(f"\n{'='*70}")
    print("KEY INSIGHT:")
    print("If variance_reduction_ratio ≈ 1.0, conditioning doesn't help → utility ≈ 0")
    print("If variance_reduction_ratio << 1.0, conditioning helps a lot → utility > 0")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
