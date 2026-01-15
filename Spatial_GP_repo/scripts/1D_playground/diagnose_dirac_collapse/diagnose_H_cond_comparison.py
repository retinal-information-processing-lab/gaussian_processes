"""
Diagnose H_cond computation methods.

Compare:
1. nd_utility's analytical H_cond formula
2. MC H_cond using compute_H with clamped variance
3. MC H_cond using exact Poisson entropy

Check where these diverge.
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
    get_marginal_moments, compute_H,
    DEVICE, DTYPE, X_MIN, X_MAX, MAX_R
)

from utility import nd_mean_noise_entropy, laplace_approximations_new


def exact_poisson_entropy(lam):
    """Compute exact entropy of Poisson(lam) for scalar lam."""
    if lam < 1e-10:
        return 0.0

    # H = -sum_r p(r) log p(r) where p(r) = lam^r * exp(-lam) / r!
    # Use log-space for numerical stability
    max_r = max(int(lam + 10 * np.sqrt(lam)), 50)
    r_vals = np.arange(max_r)

    # log p(r) = r*log(lam) - lam - log(r!)
    # Compute log(r!) incrementally
    log_r_fact = np.zeros(max_r)
    for r in range(1, max_r):
        log_r_fact[r] = log_r_fact[r - 1] + np.log(r)

    log_p = r_vals * np.log(lam + 1e-30) - lam - log_r_fact

    p = np.exp(log_p)
    p = p / p.sum()  # Normalize

    # H = -sum p * log p
    H = -np.sum(p * np.log(p + 1e-30))
    return H


def compute_H_cond_analytical(mu, sigma2, r_max=MAX_R):
    """Compute nd_utility's analytical H_cond = E[H(Poisson(exp(λ)))]."""
    device = mu.device
    dtype = mu.dtype
    r_values = torch.arange(0, r_max, dtype=dtype, device=device)

    # Get p(r) from Laplace approximation
    p_r, log_p_r = laplace_approximations_new(mu=mu, sigma2=sigma2, r=r_values)

    # Compute log(r!) for each r
    log_r_fact = torch.zeros(r_max, dtype=dtype, device=device)
    for r in range(1, r_max):
        log_r_fact[r] = log_r_fact[r - 1] + np.log(r)

    # E[H(Poisson(exp(λ)))] using analytical formula from nd_utility
    # H_cond = exp(μ+σ²/2)(1-μ-σ²) + Σ_r p_Lap(r) log(r!)
    E_f = torch.exp(mu + sigma2 / 2)
    term1 = E_f * (1 - mu - sigma2)
    term2 = torch.sum(p_r * log_r_fact.unsqueeze(0), dim=1)
    H_cond = term1 + term2

    return H_cond


def compute_H_cond_mc_laplace(model, x_point, n_samples=1000):
    """MC H_cond using compute_H with the clamped variance from conditioning."""
    device = x_point.device
    dtype = x_point.dtype

    # Get marginal moments
    mu_marg, sigma2_marg = get_marginal_moments(model, x_point)
    mu = mu_marg[0]
    std = sigma2_marg[0].sqrt()

    # Sample lambda values
    H_sum = 0.0
    for _ in range(n_samples):
        lam_sample = (mu + std * torch.randn(1, dtype=dtype, device=device)).item()
        # compute_H with very small variance (mimicking sigma2_cond ~ 0)
        mu_tensor = torch.tensor([lam_sample], dtype=dtype, device=device)
        sigma2_tensor = torch.tensor([1e-8], dtype=dtype, device=device)  # Clamped value
        H = compute_H(mu_tensor, sigma2_tensor).item()
        H_sum += H

    return H_sum / n_samples


def compute_H_cond_mc_exact(model, x_point, n_samples=1000):
    """MC H_cond using exact Poisson entropy."""
    device = x_point.device
    dtype = x_point.dtype

    # Get marginal moments
    mu_marg, sigma2_marg = get_marginal_moments(model, x_point)
    mu = mu_marg[0].item()
    std = sigma2_marg[0].sqrt().item()

    # Sample lambda values and compute exact Poisson entropy
    H_sum = 0.0
    for _ in range(n_samples):
        lam_sample = mu + std * np.random.randn()
        f_sample = np.exp(lam_sample)  # Firing rate
        H = exact_poisson_entropy(f_sample)
        H_sum += H

    return H_sum / n_samples


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Diagnose: H_cond Comparison")
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
    n_points = 50
    n_mc_samples = 500
    x_candidates = torch.linspace(x_plot_min, x_plot_max, n_points, dtype=DTYPE, device=DEVICE)

    # Get marginal moments for analytical computation
    mu_marg, sigma2_marg = get_marginal_moments(model, x_candidates)

    # Compute H_cond using different methods
    print("\nComputing analytical H_cond...")
    H_cond_analytical = compute_H_cond_analytical(mu_marg, sigma2_marg).cpu().numpy()

    print(f"Computing MC H_cond (Laplace, {n_mc_samples} samples)...")
    H_cond_mc_laplace = []
    for i in tqdm(range(n_points)):
        H = compute_H_cond_mc_laplace(model, x_candidates[i:i+1], n_samples=n_mc_samples)
        H_cond_mc_laplace.append(H)
    H_cond_mc_laplace = np.array(H_cond_mc_laplace)

    print(f"Computing MC H_cond (exact Poisson, {n_mc_samples} samples)...")
    H_cond_mc_exact = []
    for i in tqdm(range(n_points)):
        H = compute_H_cond_mc_exact(model, x_candidates[i:i+1], n_samples=n_mc_samples)
        H_cond_mc_exact.append(H)
    H_cond_mc_exact = np.array(H_cond_mc_exact)

    x_np = x_candidates.cpu().numpy()

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Compare methods
    diff_laplace_analytical = np.abs(H_cond_mc_laplace - H_cond_analytical)
    diff_exact_analytical = np.abs(H_cond_mc_exact - H_cond_analytical)
    diff_laplace_exact = np.abs(H_cond_mc_laplace - H_cond_mc_exact)

    print(f"\n|MC_Laplace - Analytical| mean: {diff_laplace_analytical.mean():.4f}, max: {diff_laplace_analytical.max():.4f}")
    print(f"|MC_Exact - Analytical| mean: {diff_exact_analytical.mean():.4f}, max: {diff_exact_analytical.max():.4f}")
    print(f"|MC_Laplace - MC_Exact| mean: {diff_laplace_exact.mean():.4f}, max: {diff_laplace_exact.max():.4f}")

    # Sample specific points
    inside_idx = n_points // 2
    outside_idx = 2

    print(f"\nInside training domain (x = {x_np[inside_idx]:.2f}):")
    print(f"  Analytical:  {H_cond_analytical[inside_idx]:.4f}")
    print(f"  MC Laplace:  {H_cond_mc_laplace[inside_idx]:.4f}")
    print(f"  MC Exact:    {H_cond_mc_exact[inside_idx]:.4f}")

    print(f"\nOutside training domain (x = {x_np[outside_idx]:.2f}):")
    print(f"  Analytical:  {H_cond_analytical[outside_idx]:.4f}")
    print(f"  MC Laplace:  {H_cond_mc_laplace[outside_idx]:.4f}")
    print(f"  MC Exact:    {H_cond_mc_exact[outside_idx]:.4f}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot 1: H_cond values
    ax1 = axes[0]
    ax1.plot(x_np, H_cond_analytical, 'g-', linewidth=2, label='Analytical (nd_utility)')
    ax1.plot(x_np, H_cond_mc_laplace, 'b--', linewidth=2, label='MC Laplace')
    ax1.plot(x_np, H_cond_mc_exact, 'r:', linewidth=2, label='MC Exact Poisson')
    ax1.axvline(x=X_MIN, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=X_MAX, color='gray', linestyle='--', alpha=0.5)
    ax1.axvspan(X_MIN, X_MAX, alpha=0.1, color='green')
    ax1.set_ylabel('H_cond')
    ax1.set_title('Conditional Entropy E[H(Poisson(exp(λ)))]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Differences from analytical
    ax2 = axes[1]
    ax2.plot(x_np, diff_laplace_analytical, 'b-', linewidth=2, label='|MC_Laplace - Analytical|')
    ax2.plot(x_np, diff_exact_analytical, 'r-', linewidth=2, label='|MC_Exact - Analytical|')
    ax2.axvline(x=X_MIN, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=X_MAX, color='gray', linestyle='--', alpha=0.5)
    ax2.axvspan(X_MIN, X_MAX, alpha=0.1, color='green')
    ax2.set_ylabel('Absolute Difference')
    ax2.set_title('Deviation from Analytical H_cond')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Marginal variance (for context)
    ax3 = axes[2]
    ax3.plot(x_np, sigma2_marg.cpu().numpy(), 'k-', linewidth=2)
    ax3.axvline(x=X_MIN, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=X_MAX, color='gray', linestyle='--', alpha=0.5)
    ax3.axvspan(X_MIN, X_MAX, alpha=0.1, color='green')
    ax3.set_xlabel('x')
    ax3.set_ylabel('σ²_marginal')
    ax3.set_title('GP Marginal Variance (context)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(__file__).parent / 'diagnose_H_cond_comparison.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
