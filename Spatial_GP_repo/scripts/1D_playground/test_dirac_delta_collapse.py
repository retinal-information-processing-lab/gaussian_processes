"""
Test: Dirac delta collapse verification

Mathematical identity: When p(x) = δ(x - x*), distribution-aware utility collapses to:
    U(x*) = H_marg(x*) - E_λ[H(Poisson(exp(λ)))]
which is exactly nd_utility (mutual information I(R; λ)).

This test:
1. Verifies nd_utility_new (Laplace) matches nd_utility_NUMERICAL (ground truth)
2. Verifies distribution-aware utility with Dirac delta matches nd_utility_NUMERICAL
   by explicitly computing U(x*) with x_sample = x_star using Gauss-Hermite quadrature

Note: Direct MC verification fails due to heavy tails in E[f(1-log f)].
Using Gauss-Hermite quadrature avoids this issue. See diagnose_dirac_collapse/ for analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gp_utility_playground import (
    VariationalGP, PoissonLikelihood, train_gp,
    generate_poisson_data, lambda_true,
    evaluate_nd_utility_new, evaluate_nd_utility_NUMERICAL,
    get_marginal_moments,
    DEVICE, DTYPE, X_MIN, X_MAX, MAX_R
)

# Extended domain for visualization
X_PLOT_MIN = X_MIN - 3
X_PLOT_MAX = X_MAX + 3

N_CANDIDATES = 50


def evaluate_dirac_utility(model, x_candidates, r_max=MAX_R, method='gauss_hermite',
                           n_quadrature=100, n_mc_samples=1000):
    """Compute distribution-aware utility with Dirac delta p(x) = δ(x - x*).

    When x_sample = x_star, conditioning on (x*, λ) gives:
        μ_cond(x*) = λ  (perfect knowledge at x*)
        σ²_cond(x*) = 0

    So H_cond(x* | x*, λ) = H(Poisson(exp(λ))).

    The Dirac utility becomes:
        U(x*) = H_marg(x*) - E_λ[H(Poisson(exp(λ)))]

    This is exactly nd_utility.

    H_cond decomposes as:
        term1 = E[exp(λ)(1-λ)]  - always computed analytically via MGF
        term2 = E[Σ_r Poisson(r|exp(λ)) log(r!)]  - method selectable

    Args:
        model: Trained GP model
        x_candidates: (N,) query points
        r_max: Maximum spike count (truncation)
        method: 'gauss_hermite' or 'mc' for term2 computation
        n_quadrature: Number of Gauss-Hermite quadrature points (if method='gauss_hermite')
        n_mc_samples: Number of MC samples (if method='mc')

    Returns:
        utility: (N,) utility values at each query point
    """
    model.eval()

    # Get GP posterior moments
    mu, sigma2 = get_marginal_moments(model, x_candidates)

    N = mu.shape[0]
    device = mu.device
    dtype = mu.dtype
    sigma = torch.sqrt(sigma2)

    r_values = torch.arange(0, r_max, dtype=dtype, device=device)
    log_r_fact = torch.lgamma(r_values + 1)

    # Term 1: E[exp(λ)(1-λ)] - EXACT via moment generating function
    # This avoids heavy-tail MC issues (always use analytical formula)
    term1 = torch.exp(mu + 0.5 * sigma2) * (1 - mu - sigma2)

    if method == 'gauss_hermite':
        # Gauss-Hermite quadrature for integrating over λ ~ N(μ, σ²)
        z_nodes, weights = np.polynomial.hermite.hermgauss(n_quadrature)
        z_nodes = torch.tensor(z_nodes, dtype=dtype, device=device)
        weights = torch.tensor(weights, dtype=dtype, device=device)
        weights_normalized = weights / np.sqrt(np.pi)

        # g = μ + σ√2·z  (quadrature transform)
        g_values = mu[:, None] + sigma[:, None] * np.sqrt(2) * z_nodes[None, :]  # (N, Q)
        f_values = torch.exp(g_values)  # (N, Q)

        # Poisson(r | exp(g)) for all (query point, quadrature node, r)
        log_poisson = (g_values[:, :, None] * r_values[None, None, :]
                       - f_values[:, :, None]
                       - log_r_fact[None, None, :])
        poisson_probs = torch.exp(log_poisson)  # (N, Q, R)

        # H_marg: p_true(r) = ∫ Poisson(r|exp(g)) N(g|μ,σ²) dg
        p_true = torch.einsum('q,nqr->nr', weights_normalized, poisson_probs)
        H_marg = -torch.sum(torch.xlogy(p_true, p_true), dim=1)

        # Term 2: E[Σ_r Poisson(r|exp(λ)) log(r!)] - Gauss-Hermite
        weighted_log_r_fact = torch.sum(poisson_probs * log_r_fact[None, None, :], dim=2)
        term2 = torch.einsum('q,nq->n', weights_normalized, weighted_log_r_fact)

    elif method == 'mc':
        # Monte Carlo sampling for term2
        # Sample λ ~ N(μ, σ²) for each candidate
        # λ[i, s] = μ[i] + σ[i] * z[s] where z ~ N(0, 1)
        z_samples = torch.randn(n_mc_samples, dtype=dtype, device=device)
        lambda_samples = mu[:, None] + sigma[:, None] * z_samples[None, :]  # (N, S)
        f_samples = torch.exp(lambda_samples)  # (N, S)

        # Poisson(r | exp(λ)) for all (candidate, sample, r)
        log_poisson = (lambda_samples[:, :, None] * r_values[None, None, :]
                       - f_samples[:, :, None]
                       - log_r_fact[None, None, :])
        poisson_probs = torch.exp(log_poisson)  # (N, S, R)

        # H_marg: p_true(r) = E_λ[Poisson(r|exp(λ))] via MC
        p_true = poisson_probs.mean(dim=1)  # (N, R)
        H_marg = -torch.sum(torch.xlogy(p_true, p_true), dim=1)

        # Term 2: E[Σ_r Poisson(r|exp(λ)) log(r!)] - MC average
        weighted_log_r_fact = torch.sum(poisson_probs * log_r_fact[None, None, :], dim=2)  # (N, S)
        term2 = weighted_log_r_fact.mean(dim=1)  # (N,)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'gauss_hermite' or 'mc'.")

    H_cond = term1 + term2

    # Dirac utility = nd_utility
    utility = H_marg - H_cond

    return utility


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Test: Dirac Delta Collapse")
    print("=" * 60)
    print("\nMathematical identity: When p(x) = δ(x - x*),")
    print("distribution-aware utility = nd_utility.")
    print("\nVerifying nd_utility computations are correct...")

    # Train GP
    n_train = 30
    train_x = torch.linspace(X_MIN, X_MAX, n_train, dtype=DTYPE, device=DEVICE)
    train_y = generate_poisson_data(train_x, lambda_true)

    model = VariationalGP(train_x.clone()).to(DEVICE)
    likelihood = PoissonLikelihood().to(DEVICE)

    print("\nTraining GP...")
    model, likelihood = train_gp(model, likelihood, train_x, train_y, n_iterations=300)

    # Evaluate utilities
    x_candidates = torch.linspace(X_PLOT_MIN, X_PLOT_MAX, N_CANDIDATES, dtype=DTYPE, device=DEVICE)

    print("\nEvaluating nd_utility (Laplace)...")
    u_laplace = evaluate_nd_utility_new(model, x_candidates)

    print("Evaluating nd_utility_NUMERICAL (ground truth)...")
    u_numerical = evaluate_nd_utility_NUMERICAL(model, x_candidates)

    print("Evaluating Dirac utility (Gauss-Hermite for term2)...")
    u_dirac_gh = evaluate_dirac_utility(model, x_candidates, method='gauss_hermite')

    print("Evaluating Dirac utility (MC for term2)...")
    u_dirac_mc = evaluate_dirac_utility(model, x_candidates, method='mc', n_mc_samples=10000)

    # Statistics
    inside_mask = (x_candidates >= X_MIN) & (x_candidates <= X_MAX)

    # Error: Laplace vs NUMERICAL
    err_laplace = (u_laplace - u_numerical).abs() / (u_numerical.abs() + 1e-10)

    # Error: Dirac (Gauss-Hermite) vs NUMERICAL
    err_dirac_gh = (u_dirac_gh - u_numerical).abs() / (u_numerical.abs() + 1e-10)

    # Error: Dirac (MC) vs NUMERICAL
    err_dirac_mc = (u_dirac_mc - u_numerical).abs() / (u_numerical.abs() + 1e-10)

    print(f"\nResults:")
    print(f"  NUMERICAL range:   [{u_numerical.min():.4f}, {u_numerical.max():.4f}]")
    print(f"  Laplace range:     [{u_laplace.min():.4f}, {u_laplace.max():.4f}]")
    print(f"  Dirac (GH) range:  [{u_dirac_gh.min():.4f}, {u_dirac_gh.max():.4f}]")
    print(f"  Dirac (MC) range:  [{u_dirac_mc.min():.4f}, {u_dirac_mc.max():.4f}]")

    print(f"\n  Laplace vs NUMERICAL (full domain):")
    print(f"    Mean rel error: {err_laplace.mean().item():.2%}")
    print(f"    Max rel error:  {err_laplace.max().item():.2%}")

    print(f"\n  Dirac (Gauss-Hermite) vs NUMERICAL (full domain):")
    print(f"    Mean rel error: {err_dirac_gh.mean().item():.2%}")
    print(f"    Max rel error:  {err_dirac_gh.max().item():.2%}")

    print(f"\n  Dirac (MC) vs NUMERICAL (full domain):")
    print(f"    Mean rel error: {err_dirac_mc.mean().item():.2%}")
    print(f"    Max rel error:  {err_dirac_mc.max().item():.2%}")

    print(f"\n  Inside training domain [{X_MIN}, {X_MAX}]:")
    print(f"    Laplace error:     {err_laplace[inside_mask].mean().item():.2%}")
    print(f"    Dirac (GH) error:  {err_dirac_gh[inside_mask].mean().item():.2%}")
    print(f"    Dirac (MC) error:  {err_dirac_mc[inside_mask].mean().item():.2%}")

    # Pass/fail
    laplace_err = err_laplace.mean().item()
    dirac_gh_err = err_dirac_gh.mean().item()
    dirac_mc_err = err_dirac_mc.mean().item()

    laplace_passes = laplace_err < 0.10
    dirac_gh_passes = dirac_gh_err < 0.01  # Should be essentially 0 since same formula!
    # MC is for comparison only - not required to pass
    passes = laplace_passes and dirac_gh_passes

    print(f"\n{'PASS' if laplace_passes else 'FAIL'}: Laplace vs NUMERICAL mean error {laplace_err:.2%} {'<' if laplace_passes else '>='} 10%")
    print(f"{'PASS' if dirac_gh_passes else 'FAIL'}: Dirac (GH) vs NUMERICAL mean error {dirac_gh_err:.2%} {'<' if dirac_gh_passes else '>='} 1%")
    print(f"INFO: Dirac (MC) vs NUMERICAL mean error {dirac_mc_err:.2%} (for comparison, MC has variance)")

    print("\n" + "-" * 60)
    print("Dirac collapse verification:")
    print("  When p(x) = δ(x-x*), the distribution-aware utility formula")
    print("  mathematically reduces to nd_utility.")
    print("  ")
    print("  H_cond = term1 + term2 where:")
    print("    term1 = E[exp(λ)(1-λ)] - always analytical (MGF)")
    print("    term2 = E[Σ_r Poisson(r|exp(λ)) log(r!)] - GH or MC")
    print("-" * 60)

    # Get GP posterior for plotting
    model.eval()
    with torch.no_grad():
        posterior = model(x_candidates)
        mu = posterior.mean
        sigma = posterior.variance.sqrt()
        true_lambda = lambda_true(x_candidates)

    # Plot
    x_np = x_candidates.cpu().numpy()

    _, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Plot 1: Utility comparison
    ax1 = axes[0]
    ax1.axvspan(X_MIN, X_MAX, alpha=0.1, color='green', label='Training domain')
    ax1.plot(x_np, u_numerical.cpu().numpy(), 'r-', linewidth=2.5, label='NUMERICAL (ground truth)')
    ax1.plot(x_np, u_laplace.cpu().numpy(), 'g--', linewidth=2, label='Laplace')
    ax1.plot(x_np, u_dirac_gh.cpu().numpy(), 'b:', linewidth=2, label='Dirac (GH)')
    ax1.plot(x_np, u_dirac_mc.cpu().numpy(), 'm-.', linewidth=1.5, alpha=0.7, label='Dirac (MC)')
    ax1.axvline(x=X_MIN, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=X_MAX, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Utility')
    ax1.set_title('Dirac collapse: distribution-aware utility with δ(x-x*) = nd_utility')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: GP fit
    ax2 = axes[1]
    ax2.axvspan(X_MIN, X_MAX, alpha=0.1, color='green')
    ax2.plot(x_np, true_lambda.cpu().numpy(), 'k--', linewidth=2, label='True λ(x)')
    ax2.plot(x_np, mu.cpu().numpy(), 'b-', linewidth=2, label='GP mean')
    ax2.fill_between(x_np, (mu - 2*sigma).cpu().numpy(), (mu + 2*sigma).cpu().numpy(),
                     alpha=0.3, color='blue', label='±2σ')
    ax2.axvline(x=X_MIN, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=X_MAX, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('λ(x)')
    ax2.set_title('GP Posterior')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Relative error
    ax3 = axes[2]
    ax3.axvspan(X_MIN, X_MAX, alpha=0.1, color='green')
    ax3.semilogy(x_np, err_laplace.cpu().numpy() + 1e-10, 'g-', linewidth=2, label=f'Laplace (mean: {laplace_err:.2%})')
    ax3.semilogy(x_np, err_dirac_gh.cpu().numpy() + 1e-10, 'b--', linewidth=2, label=f'Dirac GH (mean: {dirac_gh_err:.2%})')
    ax3.semilogy(x_np, err_dirac_mc.cpu().numpy() + 1e-10, 'm:', linewidth=2, label=f'Dirac MC (mean: {dirac_mc_err:.2%})')
    ax3.axhline(y=0.10, color='k', linestyle='--', alpha=0.7, label='10% threshold')
    ax3.axhline(y=0.01, color='gray', linestyle=':', alpha=0.7, label='1% threshold')
    ax3.axvline(x=X_MIN, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=X_MAX, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('x')
    ax3.set_ylabel('Relative Error')
    ax3.set_title('Approximation errors vs NUMERICAL ground truth')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(__file__).parent / 'test_dirac_delta_collapse_result.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()

    return passes


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
