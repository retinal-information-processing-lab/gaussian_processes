"""
Diagnose utility decomposition: U = H_marg - H_cond

Compare nd_utility vs Dirac MC by decomposing into H_marg and H_cond.
Identify whether the discrepancy is in H_marg or H_cond.
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
    get_marginal_moments, get_conditional_moments, compute_H,
    evaluate_nd_utility_new,
    DEVICE, DTYPE, X_MIN, X_MAX, MAX_R
)

from utility import laplace_approximations_new, nd_mean_noise_entropy


def compute_nd_utility_decomposed(mu, sigma2, r_max=MAX_R):
    """Compute nd_utility with H_marg and H_cond returned separately."""
    device = mu.device
    dtype = mu.dtype
    r_values = torch.arange(0, r_max, dtype=dtype, device=device)

    # H_marg
    p_r, log_p_r = laplace_approximations_new(mu=mu, sigma2=sigma2, r=r_values)
    H_marg = -torch.sum(p_r * log_p_r, dim=1)

    # H_cond (analytical formula from nd_utility_new)
    log_r_fact = torch.zeros(r_max, dtype=dtype, device=device)
    for r in range(1, r_max):
        log_r_fact[r] = log_r_fact[r - 1] + np.log(r)

    p_r_T = p_r.T  # (R, N)
    log_r_fact_2d = log_r_fact[:, None].expand(r_max, len(mu))

    H_cond = nd_mean_noise_entropy(p_r_T, log_r_fact_2d, sigma2, mu)

    U = H_marg - H_cond

    return U, H_marg, H_cond


def compute_dirac_utility_decomposed(model, x_candidates, n_lambda_samples=500):
    """Compute Dirac utility with H_marg and H_cond returned separately."""
    model.eval()
    device = x_candidates.device
    dtype = x_candidates.dtype
    n_points = len(x_candidates)

    # H_marg (same as nd_utility)
    mu_marg, sigma2_marg = get_marginal_moments(model, x_candidates)
    H_marg = compute_H(mu_marg, sigma2_marg)

    # H_cond via MC
    H_cond = torch.zeros(n_points, dtype=dtype, device=device)

    with torch.no_grad():
        for i in tqdm(range(n_points), desc="Dirac H_cond"):
            x_i = x_candidates[i].item()
            mu_i = mu_marg[i]
            std_i = sigma2_marg[i].sqrt()

            H_cond_sum = 0.0
            for _ in range(n_lambda_samples):
                lam_sample = (mu_i + std_i * torch.randn(1, dtype=dtype, device=device)).item()
                mu_cond, sigma2_cond = get_conditional_moments(model, x_candidates[i:i+1], x_i, lam_sample)
                H_cond_sum += compute_H(mu_cond, sigma2_cond).item()

            H_cond[i] = H_cond_sum / n_lambda_samples

    U = H_marg - H_cond

    return U, H_marg, H_cond


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Diagnose: Utility Components (U = H_marg - H_cond)")
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

    # Get marginal moments for nd_utility
    mu_marg, sigma2_marg = get_marginal_moments(model, x_candidates)

    # Compute decomposed utilities
    print("\nComputing nd_utility decomposition...")
    U_nd, H_marg_nd, H_cond_nd = compute_nd_utility_decomposed(mu_marg, sigma2_marg)
    U_nd = U_nd.cpu().numpy()
    H_marg_nd = H_marg_nd.cpu().numpy()
    H_cond_nd = H_cond_nd.cpu().numpy()

    print(f"\nComputing Dirac utility decomposition ({n_mc_samples} samples)...")
    U_dirac, H_marg_dirac, H_cond_dirac = compute_dirac_utility_decomposed(
        model, x_candidates, n_lambda_samples=n_mc_samples
    )
    U_dirac = U_dirac.cpu().numpy()
    H_marg_dirac = H_marg_dirac.cpu().numpy()
    H_cond_dirac = H_cond_dirac.cpu().numpy()

    x_np = x_candidates.cpu().numpy()
    sigma2_np = sigma2_marg.cpu().numpy()

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nH_marg comparison (should be identical):")
    H_marg_diff = np.abs(H_marg_nd - H_marg_dirac)
    print(f"  |H_marg_nd - H_marg_dirac| max: {H_marg_diff.max():.6f}")

    print(f"\nH_cond comparison:")
    H_cond_diff = np.abs(H_cond_nd - H_cond_dirac)
    print(f"  |H_cond_nd - H_cond_dirac| mean: {H_cond_diff.mean():.4f}, max: {H_cond_diff.max():.4f}")

    print(f"\nUtility comparison:")
    U_diff = np.abs(U_nd - U_dirac)
    print(f"  |U_nd - U_dirac| mean: {U_diff.mean():.4f}, max: {U_diff.max():.4f}")

    # Check for negative H_cond (should never happen for entropy)
    if np.any(H_cond_nd < 0):
        neg_idx = np.where(H_cond_nd < 0)[0]
        print(f"\n*** WARNING: H_cond_nd is NEGATIVE at {len(neg_idx)} points! ***")
        print(f"  Negative H_cond_nd range: [{H_cond_nd[neg_idx].min():.4f}, {H_cond_nd[neg_idx].max():.4f}]")
        print(f"  At these points, σ² range: [{sigma2_np[neg_idx].min():.4f}, {sigma2_np[neg_idx].max():.4f}]")

    # Sample specific points
    inside_idx = n_points // 2
    outside_idx = 2

    print(f"\nInside training domain (x = {x_np[inside_idx]:.2f}, σ² = {sigma2_np[inside_idx]:.4f}):")
    print(f"  H_marg:  nd={H_marg_nd[inside_idx]:.4f}, dirac={H_marg_dirac[inside_idx]:.4f}")
    print(f"  H_cond:  nd={H_cond_nd[inside_idx]:.4f}, dirac={H_cond_dirac[inside_idx]:.4f}")
    print(f"  U:       nd={U_nd[inside_idx]:.4f}, dirac={U_dirac[inside_idx]:.4f}")

    print(f"\nOutside training domain (x = {x_np[outside_idx]:.2f}, σ² = {sigma2_np[outside_idx]:.4f}):")
    print(f"  H_marg:  nd={H_marg_nd[outside_idx]:.4f}, dirac={H_marg_dirac[outside_idx]:.4f}")
    print(f"  H_cond:  nd={H_cond_nd[outside_idx]:.4f}, dirac={H_cond_dirac[outside_idx]:.4f}")
    print(f"  U:       nd={U_nd[outside_idx]:.4f}, dirac={U_dirac[outside_idx]:.4f}")

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # Plot 1: H_marg
    ax1 = axes[0]
    ax1.plot(x_np, H_marg_nd, 'g-', linewidth=2, label='H_marg (nd_utility)')
    ax1.plot(x_np, H_marg_dirac, 'b--', linewidth=2, label='H_marg (Dirac)')
    ax1.axvline(x=X_MIN, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=X_MAX, color='gray', linestyle='--', alpha=0.5)
    ax1.axvspan(X_MIN, X_MAX, alpha=0.1, color='green')
    ax1.set_ylabel('H_marg')
    ax1.set_title('Marginal Entropy H(R|D) - should be identical')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: H_cond
    ax2 = axes[1]
    ax2.plot(x_np, H_cond_nd, 'g-', linewidth=2, label='H_cond (nd_utility analytical)')
    ax2.plot(x_np, H_cond_dirac, 'b--', linewidth=2, label='H_cond (Dirac MC)')
    ax2.axhline(y=0, color='red', linestyle='-', alpha=0.5, label='Zero line')
    ax2.axvline(x=X_MIN, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=X_MAX, color='gray', linestyle='--', alpha=0.5)
    ax2.axvspan(X_MIN, X_MAX, alpha=0.1, color='green')
    ax2.set_ylabel('H_cond')
    ax2.set_title('Conditional Entropy E[H(Poisson(exp(λ)))] - KEY DISCREPANCY')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Utility
    ax3 = axes[2]
    ax3.plot(x_np, U_nd, 'g-', linewidth=2, label='U (nd_utility)')
    ax3.plot(x_np, U_dirac, 'b--', linewidth=2, label='U (Dirac)')
    ax3.axvline(x=X_MIN, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=X_MAX, color='gray', linestyle='--', alpha=0.5)
    ax3.axvspan(X_MIN, X_MAX, alpha=0.1, color='green')
    ax3.set_ylabel('Utility')
    ax3.set_title('Utility = H_marg - H_cond')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Variance (context)
    ax4 = axes[3]
    ax4.plot(x_np, sigma2_np, 'k-', linewidth=2)
    ax4.axvline(x=X_MIN, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=X_MAX, color='gray', linestyle='--', alpha=0.5)
    ax4.axvspan(X_MIN, X_MAX, alpha=0.1, color='green')
    ax4.set_xlabel('x')
    ax4.set_ylabel('σ²_marginal')
    ax4.set_title('GP Marginal Variance (context)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(__file__).parent / 'diagnose_utility_components.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
