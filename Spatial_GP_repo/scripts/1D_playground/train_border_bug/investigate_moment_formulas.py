"""
Investigate numerical stability of GP moment formulas.

Created by Claude to compare manual LaTeX formulas with GPyTorch computations.

This script tests the formulas from:
- predictive_distribution_derivation_temp.tex (marginal and conditional moments)
- distribution_aware_utility_pietro.tex (conditional moments)

Key finding: K^{-1} causes numerical instability when the kernel matrix is ill-conditioned.
"""

import sys
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/1D_playground')
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo')

import torch
import numpy as np
import matplotlib.pyplot as plt

from gp_utility_playground import (
    VariationalGP, PoissonLikelihood,
    generate_poisson_data, lambda_true,
    DEVICE, DTYPE, X_MIN, X_MAX
)

# =============================================================================
# Configuration (SAME as simple_utility_gpytorch.py)
# =============================================================================
FIXED_LENGTHSCALE = 0.2
FIXED_OUTPUTSCALE = 1.0
N_TRAIN = 100
N_ITERATIONS = 500
X_SAMPLE = 0.0
X_PLOT_MIN = -2.0
X_PLOT_MAX = 2.0


def compute_marginal_manual(model, x_star, stable=False, unwhiten=False):
    """
    Compute marginal moments using manual formulas from LaTeX.

    Formulas (from predictive_distribution_derivation_temp.tex):
        μ* = u*^T m
        σ²* = s* + u*^T V u*

    where:
        u* = K^{-1} k(z, x*)       [projection vector]
        s* = k(x*,x*) - k(x*)^T K^{-1} k(x*)  [Schur complement]
        m = variational mean
        V = variational covariance

    Args:
        stable: If True, use torch.linalg.solve() instead of explicit K^{-1}
        unwhiten: If True, unwhiten m and V before computing (GPyTorch stores whitened)
    """
    model.eval()
    device = x_star.device
    dtype = x_star.dtype

    # 1. Get variational parameters
    z = model.variational_strategy.inducing_points
    if z.ndim == 2 and z.shape[1] == 1:
        z = z.squeeze(-1)
    M = z.shape[0]

    m = model.variational_strategy.variational_distribution.mean  # (M,)
    V = model.variational_strategy.variational_distribution.covariance_matrix  # (M, M)

    # 2. Compute kernel matrices
    K = model.covar_module(z, z).evaluate()  # (M, M)
    K_reg = K + 1e-6 * torch.eye(M, device=device, dtype=dtype)

    k_star_z = model.covar_module(x_star, z).evaluate()  # (N, M)
    k_star_star_diag = model.covar_module(x_star, x_star).evaluate().diag()  # (N,)

    # 3. If unwhiten=True, convert from whitened to actual space
    # GPyTorch stores: m_whitened = L^{-1} m_actual, V_whitened = L^{-1} V_actual L^{-T}
    # So: m_actual = L @ m_whitened, V_actual = L @ V_whitened @ L^T
    if unwhiten:
        L = torch.linalg.cholesky(K_reg)  # K = L @ L^T
        m = L @ m  # Unwhiten mean
        V = L @ V @ L.T  # Unwhiten covariance

    # 4. Compute projection vector: u* = K^{-1} k(z, x*)
    if stable:
        u_star = torch.linalg.solve(K_reg, k_star_z.T)  # (M, N)
    else:
        K_inv = torch.linalg.inv(K_reg)
        u_star = K_inv @ k_star_z.T  # (M, N)

    # 5. Compute Schur complement: s* = k(x*,x*) - k(x*)^T K^{-1} k(x*)
    s_star = k_star_star_diag - torch.sum(k_star_z.T * u_star, dim=0)  # (N,)

    # 6. Apply formulas
    # μ* = u*^T m
    mu_star = u_star.T @ m  # (N,)

    # σ²* = s* + u*^T V u*
    sigma2_star = s_star + torch.sum(u_star * (V @ u_star), dim=0)  # (N,)

    return mu_star, sigma2_star


def compute_conditional_manual(model, x_sample, x_star, lambda_sample, stable=False, unwhiten=False):
    """
    Compute conditional moments using manual formulas from LaTeX.

    From predictive_distribution_derivation_temp.tex Section 3:

    Step 1: Update inducing point posterior (condition on λ(x))
        m' = m + (V u / denom) * (λ(x) - u^T m)
        V' = V - (V u u^T V) / denom

        where:
            denom = s + u^T V u
            s = k(x,x) - u^T K u
            u = K^{-1} k(z, x)

    Step 2: Compute predictive moments using updated posterior
        μ_cond = u*^T m'           [Eq. 113]
        σ²_cond = s* + u*^T V' u*  [Eq. 130]

    Args:
        stable: If True, use torch.linalg.solve() instead of explicit K^{-1}
        unwhiten: If True, unwhiten m and V before computing (GPyTorch stores whitened)
    """
    model.eval()
    device = x_star.device
    dtype = x_star.dtype

    # 1. Get variational parameters
    z = model.variational_strategy.inducing_points
    if z.ndim == 2 and z.shape[1] == 1:
        z = z.squeeze(-1)
    M = z.shape[0]

    m = model.variational_strategy.variational_distribution.mean  # (M,)
    V = model.variational_strategy.variational_distribution.covariance_matrix  # (M, M)

    # 2. Compute kernel matrices
    K = model.covar_module(z, z).evaluate()  # (M, M)
    K_reg = K + 1e-6 * torch.eye(M, device=device, dtype=dtype)

    # 3. If unwhiten=True, convert from whitened to actual space
    if unwhiten:
        L = torch.linalg.cholesky(K_reg)  # K = L @ L^T
        m = L @ m  # Unwhiten mean
        V = L @ V @ L.T  # Unwhiten covariance

    # Kernels for sample point x
    k_sample_z = model.covar_module(x_sample, z).evaluate()  # (1, M)
    k_sample_sample = model.covar_module(x_sample, x_sample).evaluate()  # (1, 1)

    # Kernels for query points x*
    k_star_z = model.covar_module(x_star, z).evaluate()  # (N, M)
    k_star_star_diag = model.covar_module(x_star, x_star).evaluate().diag()  # (N,)

    # 4. Compute projection vectors
    if stable:
        u = torch.linalg.solve(K_reg, k_sample_z.T)  # (M, 1)
        u_star = torch.linalg.solve(K_reg, k_star_z.T)  # (M, N)
    else:
        K_inv = torch.linalg.inv(K_reg)
        u = K_inv @ k_sample_z.T  # (M, 1)
        u_star = K_inv @ k_star_z.T  # (M, N)

    # 5. Compute Schur complements
    # s = k(x,x) - u^T K u (at sample point)
    s = k_sample_sample - k_sample_z @ u  # (1, 1)
    s = s.squeeze()  # scalar

    # s* = k(x*,x*) - k(x*)^T K^{-1} k(x*) (at query points)
    s_star = k_star_star_diag - torch.sum(k_star_z.T * u_star, dim=0)  # (N,)

    # 6. Compute updated posterior EXPLICITLY
    # denom = s + u^T V u
    denom = s + (u.T @ V @ u).squeeze()  # scalar

    # m' = m + (V u / denom) * (λ(x) - u^T m)
    Vu = V @ u  # (M, 1)
    innovation = lambda_sample - (u.T @ m).squeeze()  # scalar
    m_prime = m + (Vu.squeeze() / denom) * innovation  # (M,)

    # V' = V - (V u u^T V) / denom
    V_prime = V - (Vu @ Vu.T) / denom  # (M, M)

    # 7. Apply formulas using m' and V' DIRECTLY
    # μ_cond = u*^T m'
    mu_cond = u_star.T @ m_prime  # (N,)

    # σ²_cond = s* + u*^T V' u*
    sigma2_cond = s_star + torch.sum(u_star * (V_prime @ u_star), dim=0)  # (N,)

    return mu_cond, sigma2_cond


def train_gp_fixed_lengthscale(model, likelihood, train_x, train_y,
                                n_iterations=N_ITERATIONS, lr=0.1):
    """Train GP but keep lengthscale and outputscale fixed."""
    model.covar_module.base_kernel.lengthscale = FIXED_LENGTHSCALE
    model.covar_module.base_kernel.raw_lengthscale.requires_grad = False

    model.covar_module.outputscale = FIXED_OUTPUTSCALE
    model.covar_module.raw_outputscale.requires_grad = False

    model.train()

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    with torch.enable_grad():
        for i in range(n_iterations):
            optimizer.zero_grad()

            output = model(train_x)
            mean = output.mean
            var = output.variance

            expected_log_lik = (train_y * mean - torch.exp(mean + var / 2)).sum()
            kl_div = model.variational_strategy.kl_divergence().sum()
            elbo = expected_log_lik - kl_div

            loss = -elbo
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Iter {i+1}/{n_iterations}, ELBO: {elbo.item():.2f}")

    print(f"\nHyperparameters (FIXED):")
    print(f"  lengthscale = {model.covar_module.base_kernel.lengthscale.item():.4f}")
    print(f"  outputscale = {model.covar_module.outputscale.item():.4f}")
    print(f"  mean = {model.mean_module.constant.item():.4f}")

    return model, likelihood


def main():
    SEED = 10
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Training data ---
    train_x = torch.linspace(X_MIN, X_MAX, N_TRAIN, dtype=DTYPE, device=DEVICE)
    train_y = generate_poisson_data(train_x, lambda_true)

    inducing_points = train_x.clone()
    model = VariationalGP(inducing_points, jitter=1e-4).to(DEVICE)
    likelihood = PoissonLikelihood().to(DEVICE)

    print("Training GP with fixed lengthscale...")
    model, likelihood = train_gp_fixed_lengthscale(model, likelihood, train_x, train_y)
    model.eval()

    # Plot grid
    x_plot = torch.linspace(X_PLOT_MIN, X_PLOT_MAX, 200, dtype=DTYPE, device=DEVICE)

    # --- Sample point ---
    x_sample = torch.tensor([X_SAMPLE], dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        sample_posterior = model(x_sample)
        mu_sample = sample_posterior.mean.item()
        std_sample = sample_posterior.variance.sqrt().item()

    torch.manual_seed(123)
    lambda_sample = mu_sample + std_sample * torch.randn(1).item()

    # --- Kernel matrix analysis ---
    z = model.variational_strategy.inducing_points
    if z.ndim == 2 and z.shape[1] == 1:
        z = z.squeeze(-1)
    K = model.covar_module(z, z).evaluate()
    K_eig = torch.linalg.eigvalsh(K)
    cond_number = K_eig.max().item() / K_eig.min().item()

    print(f"\n=== Kernel Matrix Analysis ===")
    print(f"  Condition number: {cond_number:.2e}")
    print(f"  Eigenvalue range: [{K_eig.min().item():.6f}, {K_eig.max().item():.4f}]")

    # --- Compute GPyTorch quantities ---
    with torch.no_grad():
        # Marginal moments (GPyTorch)
        posterior = model(x_plot)
        mu_marg_gpytorch = posterior.mean
        sigma2_marg_gpytorch = posterior.variance

        # For conditional moments, use joint covariance
        all_x = torch.cat([x_sample, x_plot])
        full_posterior = model(all_x)
        full_covar = full_posterior.covariance_matrix
        full_mean = full_posterior.mean

        Sigma_xx = full_covar[0, 0]
        Sigma_star_star = full_covar[1:, 1:].diag()
        Sigma_x_star = full_covar[0, 1:]
        mu_x = full_mean[0]
        mu_star = full_mean[1:]

        # Conditional moments (GPyTorch via standard Gaussian conditioning)
        innovation_gpytorch = lambda_sample - mu_x
        mu_cond_gpytorch = mu_star + (Sigma_x_star / Sigma_xx) * innovation_gpytorch
        sigma2_cond_gpytorch = Sigma_star_star - (Sigma_x_star ** 2) / Sigma_xx

    # --- Compute diagnostic quantities: u* and mean corrections ---
    with torch.no_grad():
        M = z.shape[0]
        m = model.variational_strategy.variational_distribution.mean  # (M,)

        # Get mean function values
        mean_at_z = model.mean_module(z).squeeze()  # (M,) or scalar
        mean_at_star = model.mean_module(x_plot).squeeze()  # (N,)

        # If mean_module returns a scalar (ConstantMean), broadcast it
        if mean_at_z.dim() == 0:
            mean_at_z = mean_at_z.expand(M)

        # Compute projection vector u* = K^{-1} k(z, x*)
        K_reg = K + 1e-6 * torch.eye(M, device=DEVICE, dtype=DTYPE)
        k_star_z = model.covar_module(x_plot, z).evaluate()  # (N, M)
        u_star = torch.linalg.solve(K_reg, k_star_z.T)  # (M, N)

        # Diagnostic 1: u*^T @ 1 (should be ~1 inside, ~0 outside)
        ones = torch.ones(M, device=DEVICE, dtype=DTYPE)
        u_star_sum = u_star.T @ ones  # (N,)

        # Diagnostic 2: Compare mean formulas (using whitened m directly - WRONG)
        mu_wrong = u_star.T @ m  # WRONG: m is in whitened space!

        # --- Diagnostic 3: Unwhiten m and compute mean correctly ---
        # GPyTorch's VariationalStrategy stores m in WHITENED space:
        #   m_stored = L^{-1} m_actual   where K = L L^T (Cholesky)
        # To get actual m: m_actual = L @ m_stored

        L = torch.linalg.cholesky(K_reg)  # K = L L^T, so L = K^{1/2}
        m_unwhitened = L @ m  # m_actual = K^{1/2} @ m_whitened

        # Now compute mean with unwhitened m: μ* = k(x*, z) K^{-1} m_unwhitened
        # Note: k(x*, z) K^{-1} = u*^T
        mu_unwhitened = u_star.T @ m_unwhitened  # Using standard formula with actual m

        # --- Diagnostic 4: Add mean function to complete the formula ---
        # Full formula: μ* = mean(x*) + k(x*, z) K^{-1} (m_unwhitened - mean(z))
        prior_mean = model.mean_module.constant.item()
        mu_full = prior_mean + u_star.T @ (m_unwhitened - prior_mean)  # Complete formula

    # --- Compute manual quantities (unstable: explicit K^{-1}) ---
    with torch.no_grad():
        mu_marg_manual, sigma2_marg_manual = compute_marginal_manual(model, x_plot, stable=False)
        mu_cond_manual, sigma2_cond_manual = compute_conditional_manual(
            model, x_sample, x_plot, lambda_sample, stable=False
        )

    # --- Compute manual quantities (stable: torch.linalg.solve) ---
    with torch.no_grad():
        mu_marg_stable, sigma2_marg_stable = compute_marginal_manual(model, x_plot, stable=True)
        mu_cond_stable, sigma2_cond_stable = compute_conditional_manual(
            model, x_sample, x_plot, lambda_sample, stable=True
        )

    # --- Compute CORRECTED manual quantities (unwhitened m and V) ---
    with torch.no_grad():
        mu_marg_corrected, sigma2_marg_corrected = compute_marginal_manual(
            model, x_plot, stable=True, unwhiten=True
        )
        mu_cond_corrected, sigma2_cond_corrected = compute_conditional_manual(
            model, x_sample, x_plot, lambda_sample, stable=True, unwhiten=True
        )

    # --- Convert to numpy ---
    x_np = x_plot.cpu().numpy()
    train_x_np = train_x.cpu().numpy()

    mu_marg_gpytorch_np = mu_marg_gpytorch.cpu().numpy()
    sigma2_marg_gpytorch_np = sigma2_marg_gpytorch.cpu().numpy()
    mu_marg_manual_np = mu_marg_manual.cpu().numpy()
    sigma2_marg_manual_np = sigma2_marg_manual.cpu().numpy()
    mu_marg_stable_np = mu_marg_stable.cpu().numpy()
    sigma2_marg_stable_np = sigma2_marg_stable.cpu().numpy()

    mu_cond_gpytorch_np = mu_cond_gpytorch.cpu().numpy()
    sigma2_cond_gpytorch_np = sigma2_cond_gpytorch.cpu().numpy()
    mu_cond_manual_np = mu_cond_manual.cpu().numpy()
    sigma2_cond_manual_np = sigma2_cond_manual.cpu().numpy()
    mu_cond_stable_np = mu_cond_stable.cpu().numpy()
    sigma2_cond_stable_np = sigma2_cond_stable.cpu().numpy()

    # Corrected (unwhitened) quantities
    mu_marg_corrected_np = mu_marg_corrected.cpu().numpy()
    sigma2_marg_corrected_np = sigma2_marg_corrected.cpu().numpy()
    mu_cond_corrected_np = mu_cond_corrected.cpu().numpy()
    sigma2_cond_corrected_np = sigma2_cond_corrected.cpu().numpy()

    # Diagnostic quantities
    u_star_sum_np = u_star_sum.cpu().numpy()
    mu_wrong_np = mu_wrong.cpu().numpy()
    mu_unwhitened_np = mu_unwhitened.cpu().numpy()
    mu_full_np = mu_full.cpu().numpy()

    # --- Print diagnostics ---
    print(f"\n=== Marginal Moments ===")
    print(f"  GPyTorch μ:       [{mu_marg_gpytorch_np.min():.4f}, {mu_marg_gpytorch_np.max():.4f}]")
    print(f"  Manual μ:         [{mu_marg_manual_np.min():.4f}, {mu_marg_manual_np.max():.4f}]")
    print(f"  Stable μ:         [{mu_marg_stable_np.min():.4f}, {mu_marg_stable_np.max():.4f}]")
    print(f"")
    print(f"  GPyTorch σ²:      [{sigma2_marg_gpytorch_np.min():.6f}, {sigma2_marg_gpytorch_np.max():.6f}]")
    print(f"  Manual σ²:        [{sigma2_marg_manual_np.min():.6f}, {sigma2_marg_manual_np.max():.6f}]")
    print(f"  Stable σ²:        [{sigma2_marg_stable_np.min():.6f}, {sigma2_marg_stable_np.max():.6f}]")

    print(f"\n=== Conditional Moments (λ = {lambda_sample:.4f}) ===")
    print(f"  GPyTorch μ_cond:  [{mu_cond_gpytorch_np.min():.4f}, {mu_cond_gpytorch_np.max():.4f}]")
    print(f"  Manual μ_cond:    [{mu_cond_manual_np.min():.4f}, {mu_cond_manual_np.max():.4f}]")
    print(f"  Stable μ_cond:    [{mu_cond_stable_np.min():.4f}, {mu_cond_stable_np.max():.4f}]")
    print(f"")
    print(f"  GPyTorch σ²_cond: [{sigma2_cond_gpytorch_np.min():.6f}, {sigma2_cond_gpytorch_np.max():.6f}]")
    print(f"  Manual σ²_cond:   [{sigma2_cond_manual_np.min():.6f}, {sigma2_cond_manual_np.max():.6f}]")
    print(f"  Stable σ²_cond:   [{sigma2_cond_stable_np.min():.6f}, {sigma2_cond_stable_np.max():.6f}]")

    print(f"\n=== Diagnostic: Projection Sanity ===")
    print(f"  u*^T @ 1 range: [{u_star_sum_np.min():.4f}, {u_star_sum_np.max():.4f}]")
    print(f"  (Should be ~1 inside training region, ~0 outside)")

    print(f"\n=== Diagnostic: Mean Formula Comparison ===")
    print(f"  GPyTorch mean:          [{mu_marg_gpytorch_np.min():.4f}, {mu_marg_gpytorch_np.max():.4f}]")
    print(f"  Wrong (u*^T m_whiten):  [{mu_wrong_np.min():.4f}, {mu_wrong_np.max():.4f}]")
    print(f"  Unwhitened (u*^T L@m):  [{mu_unwhitened_np.min():.4f}, {mu_unwhitened_np.max():.4f}]")
    print(f"  Full (+ mean func):     [{mu_full_np.min():.4f}, {mu_full_np.max():.4f}]")

    # --- Plot ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    # Helper for common plot elements
    def add_common_elements(ax):
        ax.axvspan(X_MIN, X_MAX, alpha=0.1, color='gray', label='Training region')
        for i, tx in enumerate(train_x_np):
            ax.axvline(x=tx, color='red', alpha=0.3, linewidth=1, linestyle=':',
                       label='Training x' if i == 0 else None)
        ax.axvline(x=X_SAMPLE, color='green', linestyle='--', alpha=0.7, label='x_sample')
        ax.grid(True, alpha=0.3)

    # Subplot 1: Marginal Mean
    ax1.plot(x_np, mu_marg_gpytorch_np, 'b-', linewidth=2, label='GPyTorch')
    ax1.plot(x_np, mu_marg_manual_np, 'r--', linewidth=2, label='Manual (K⁻¹)')
    ax1.plot(x_np, mu_marg_stable_np, 'g:', linewidth=2, label='Stable (solve)')
    add_common_elements(ax1)
    ax1.set_ylabel('μ*')
    ax1.set_title('Marginal Mean')
    ax1.legend(loc='upper right')

    # Subplot 2: Marginal Variance
    ax2.plot(x_np, sigma2_marg_gpytorch_np, 'b-', linewidth=2, label='GPyTorch')
    ax2.plot(x_np, sigma2_marg_manual_np, 'r--', linewidth=2, label='Manual (K⁻¹)')
    ax2.plot(x_np, sigma2_marg_stable_np, 'g:', linewidth=2, label='Stable (solve)')
    add_common_elements(ax2)
    ax2.set_ylabel('σ²*')
    ax2.set_title('Marginal Variance')
    ax2.legend(loc='upper right')

    # Subplot 3: Conditional Mean
    ax3.plot(x_np, mu_cond_gpytorch_np, 'b-', linewidth=2, label='GPyTorch')
    ax3.plot(x_np, mu_cond_manual_np, 'r--', linewidth=2, label='Manual (K⁻¹)')
    ax3.plot(x_np, mu_cond_stable_np, 'g:', linewidth=2, label='Stable (solve)')
    add_common_elements(ax3)
    ax3.set_xlabel('x')
    ax3.set_ylabel('μ_cond')
    ax3.set_title(f'Conditional Mean (given λ({X_SAMPLE}) = {lambda_sample:.2f})')
    ax3.legend(loc='upper right')

    # Subplot 4: Conditional Variance
    ax4.plot(x_np, sigma2_cond_gpytorch_np, 'b-', linewidth=2, label='GPyTorch')
    ax4.plot(x_np, sigma2_cond_manual_np, 'r--', linewidth=2, label='Manual (K⁻¹)')
    ax4.plot(x_np, sigma2_cond_stable_np, 'g:', linewidth=2, label='Stable (solve)')
    add_common_elements(ax4)
    ax4.set_xlabel('x')
    ax4.set_ylabel('σ²_cond')
    ax4.set_title(f'Conditional Variance (given λ({X_SAMPLE}) = {lambda_sample:.2f})')
    ax4.legend(loc='upper right')

    plt.suptitle(f'GP Moment Formulas: GPyTorch vs Manual (K cond = {cond_number:.2e})', fontsize=14)
    plt.tight_layout()

    output_path = '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/1D_playground/train_border_bug/investigate_moment_formulas_result.png'
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nPlot saved to: {output_path}")

    # --- Diagnostic Figure: 2 subplots ---
    fig_diag, (ax_diag1, ax_diag2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: u*^T @ 1 (projection sanity check)
    ax_diag1.plot(x_np, u_star_sum_np, 'b-', linewidth=2)
    ax_diag1.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Expected inside')
    ax_diag1.axhline(y=0.0, color='orange', linestyle='--', alpha=0.7, label='Expected outside')
    ax_diag1.axvspan(X_MIN, X_MAX, alpha=0.1, color='gray', label='Training region')
    for i, tx in enumerate(train_x_np):
        ax_diag1.axvline(x=tx, color='red', alpha=0.3, linewidth=1, linestyle=':',
                         label='Training x' if i == 0 else None)
    ax_diag1.set_xlabel('x*')
    ax_diag1.set_ylabel('u*ᵀ @ 1')
    ax_diag1.set_title('Projection Sanity: Sum of u* components')
    ax_diag1.legend(loc='upper right')
    ax_diag1.grid(True, alpha=0.3)

    # Subplot 2: Mean formula comparison
    ax_diag2.plot(x_np, mu_marg_gpytorch_np, 'b-', linewidth=2, label='GPyTorch')
    ax_diag2.plot(x_np, mu_wrong_np, 'r--', linewidth=2, alpha=0.5, label='Wrong: u*ᵀ m_whitened')
    ax_diag2.plot(x_np, mu_full_np, 'g-', linewidth=2, label='Full: μ₀ + u*ᵀ(Lm - μ₀)')
    ax_diag2.axvspan(X_MIN, X_MAX, alpha=0.1, color='gray', label='Training region')
    for i, tx in enumerate(train_x_np):
        ax_diag2.axvline(x=tx, color='red', alpha=0.3, linewidth=1, linestyle=':',
                         label='Training x' if i == 0 else None)
    ax_diag2.set_xlabel('x*')
    ax_diag2.set_ylabel('μ*')
    ax_diag2.set_title('Mean Formula Comparison')
    ax_diag2.legend(loc='upper right')
    ax_diag2.grid(True, alpha=0.3)

    plt.suptitle('Diagnostic: Isolating the Formula Bug', fontsize=14)
    plt.tight_layout()

    diag_output_path = '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/1D_playground/train_border_bug/investigate_moment_formulas_diagnostic.png'
    plt.savefig(diag_output_path, dpi=150)
    plt.close()

    print(f"Diagnostic plot saved to: {diag_output_path}")

    # --- NEW FIGURE: GPyTorch vs Corrected Manual (2x2) ---
    fig_corrected, ((ax_c1, ax_c2), (ax_c3, ax_c4)) = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    def add_common_elements_simple(ax):
        ax.axvspan(X_MIN, X_MAX, alpha=0.1, color='gray')
        ax.axvline(x=X_SAMPLE, color='orange', linestyle='--', alpha=0.7, label='x_sample')
        ax.grid(True, alpha=0.3)

    # Subplot 1: Marginal Mean
    ax_c1.plot(x_np, mu_marg_gpytorch_np, 'b-', linewidth=2, label='GPyTorch')
    ax_c1.plot(x_np, mu_marg_corrected_np, 'g--', linewidth=2, label='Corrected Manual')
    add_common_elements_simple(ax_c1)
    ax_c1.set_ylabel('μ*')
    ax_c1.set_title('Marginal Mean')
    ax_c1.legend(loc='upper right')

    # Subplot 2: Marginal Variance
    ax_c2.plot(x_np, sigma2_marg_gpytorch_np, 'b-', linewidth=2, label='GPyTorch')
    ax_c2.plot(x_np, sigma2_marg_corrected_np, 'g--', linewidth=2, label='Corrected Manual')
    add_common_elements_simple(ax_c2)
    ax_c2.set_ylabel('σ²*')
    ax_c2.set_title('Marginal Variance')
    ax_c2.legend(loc='upper right')

    # Subplot 3: Conditional Mean
    ax_c3.plot(x_np, mu_cond_gpytorch_np, 'b-', linewidth=2, label='GPyTorch')
    ax_c3.plot(x_np, mu_cond_corrected_np, 'g--', linewidth=2, label='Corrected Manual')
    add_common_elements_simple(ax_c3)
    ax_c3.set_xlabel('x')
    ax_c3.set_ylabel('μ_cond')
    ax_c3.set_title(f'Conditional Mean (given λ({X_SAMPLE}) = {lambda_sample:.2f})')
    ax_c3.legend(loc='upper right')

    # Subplot 4: Conditional Variance
    ax_c4.plot(x_np, sigma2_cond_gpytorch_np, 'b-', linewidth=2, label='GPyTorch')
    ax_c4.plot(x_np, sigma2_cond_corrected_np, 'g--', linewidth=2, label='Corrected Manual')
    add_common_elements_simple(ax_c4)
    ax_c4.set_xlabel('x')
    ax_c4.set_ylabel('σ²_cond')
    ax_c4.set_title(f'Conditional Variance (given λ({X_SAMPLE}) = {lambda_sample:.2f})')
    ax_c4.legend(loc='upper right')

    plt.suptitle('GPyTorch vs Corrected Manual (with unwhitened m, V)', fontsize=14)
    plt.tight_layout()

    corrected_output_path = '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/1D_playground/train_border_bug/investigate_moment_formulas_corrected.png'
    plt.savefig(corrected_output_path, dpi=150)
    plt.close()

    print(f"Corrected comparison plot saved to: {corrected_output_path}")


if __name__ == '__main__':
    main()
