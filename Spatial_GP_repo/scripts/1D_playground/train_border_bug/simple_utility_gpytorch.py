"""
Simple utility calculation using ONLY GPyTorch quantities.
Created by Claude to diagnose distribution-aware utility behavior.
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
from utility import laplace_approximations_new

# =============================================================================
# Configuration (SAME as diagnose_posterior_conditioning.py)
# =============================================================================
FIXED_LENGTHSCALE = 0.2
FIXED_OUTPUTSCALE = 1.0
N_TRAIN = 20
N_ITERATIONS = 500
X_SAMPLE = 1.0
X_PLOT_MIN = -2.0
X_PLOT_MAX = 2.0
MAX_R = 500


def compute_H_marginal(model, x_star, r_max=MAX_R):
    """Compute marginal entropy H_marg(x*) at each query point using GPyTorch."""
    model.eval()
    device = x_star.device
    dtype = x_star.dtype
    r_values = torch.arange(0, r_max, dtype=dtype, device=device)

    with torch.no_grad():
        posterior = model(x_star)
        mu_star = posterior.mean
        sigma2_star = posterior.variance

        p_r, log_p_r = laplace_approximations_new(mu=mu_star, sigma2=sigma2_star, r=r_values)
        H_marg = -torch.sum(p_r * log_p_r, dim=1)

    return H_marg


def compute_marginal_moments_manual(model, x_star, debug=False, unwhiten=False):
    """
    Compute marginal moments using the MANUAL formula from the LaTeX derivation.

    From predictive_distribution_derivation_temp.tex:
        μ* = u*^T m
        σ²* = s* + u*^T V u*

    where:
        u* = K^{-1} k(z, x*)
        s* = k(x*, x*) - k(x*)^T K^{-1} k(x*)
        m = variational mean
        V = variational covariance

    If unwhiten=True, converts m and V from whitened space to actual space.
    GPyTorch stores: m_whitened = L^{-1} @ m_actual, V_whitened = L^{-1} @ V_actual @ L^{-T}
    where K = L @ L^T (Cholesky).
    """
    model.eval()
    device = x_star.device
    dtype = x_star.dtype

    # Get inducing points and variational parameters
    z = model.variational_strategy.inducing_points
    if z.ndim == 2 and z.shape[1] == 1:
        z = z.squeeze(-1)
    M = z.shape[0]

    m = model.variational_strategy.variational_distribution.mean  # (M,) - whitened!
    V = model.variational_strategy.variational_distribution.covariance_matrix  # (M, M) - whitened!

    K_zz = model.covar_module(z, z).evaluate()
    K_zz_reg = K_zz + 1e-6 * torch.eye(M, device=device, dtype=dtype)

    # Unwhiten m and V if requested
    if unwhiten:
        L = torch.linalg.cholesky(K_zz_reg)  # K = L @ L^T
        m = L @ m  # m_actual = L @ m_whitened
        V = L @ V @ L.T  # V_actual = L @ V_whitened @ L^T

    K_zz_inv = torch.linalg.inv(K_zz_reg)

    # Kernel evaluations
    k_star_z = model.covar_module(x_star, z).evaluate()      # (K, M)
    k_star_star_diag = model.covar_module(x_star, x_star).evaluate().diag()  # (K,)

    # Projection vector: u* = K^{-1} k(z, x*)
    u_star = K_zz_inv @ k_star_z.T    # (M, K)

    # Schur complement: s* = k(x*, x*) - k(x*)^T K^{-1} k(x*)
    s_star = k_star_star_diag - torch.sum(k_star_z.T * u_star, dim=0)  # (K,)

    # Marginal mean: μ* = mean(x*) + u*^T m
    # Note: when unwhitened, m represents deviation from mean function
    mean_x_star = model.mean_module(x_star).squeeze()  # (K,)
    mu_star = mean_x_star + u_star.T @ m  # (K,)

    # Marginal variance: σ²* = s* + u*^T V u*
    ustar_T_V_ustar = torch.sum(u_star * (V @ u_star), dim=0)  # (K,)
    sigma2_star = s_star + ustar_T_V_ustar  # (K,)

    if debug:
        print(f"\nDEBUG marginal moments (manual formula):")
        print(f"  s* range: [{s_star.min().item():.6f}, {s_star.max().item():.6f}]")
        print(f"  u*^T V u* range: [{ustar_T_V_ustar.min().item():.4f}, {ustar_T_V_ustar.max().item():.4f}]")
        print(f"  σ²* (s* + u*^T V u*): [{sigma2_star.min().item():.4f}, {sigma2_star.max().item():.4f}]")
        print(f"  μ* range: [{mu_star.min().item():.4f}, {mu_star.max().item():.4f}]")

    return mu_star, sigma2_star


def compute_H_marginal_manual(model, x_star, r_max=MAX_R, debug=False, unwhiten=False):
    """Compute marginal entropy H_marg(x*) using manual formulas."""
    device = x_star.device
    dtype = x_star.dtype
    r_values = torch.arange(0, r_max, dtype=dtype, device=device)

    with torch.no_grad():
        mu_star, sigma2_star = compute_marginal_moments_manual(model, x_star, debug=debug, unwhiten=unwhiten)
        # Clamp variance for numerical stability
        sigma2_star = torch.clamp(sigma2_star, min=1e-8)

        p_r, log_p_r = laplace_approximations_new(mu=mu_star, sigma2=sigma2_star, r=r_values)
        H_marg = -torch.sum(p_r * log_p_r, dim=1)

    return H_marg, mu_star, sigma2_star


def compute_cross_cov_manual(model, x_sample, x_star, unwhiten=False):
    """
    Compute cross-covariance using the MANUAL variational formula.
    This is the WRONG formula from active_learning_pietro_corrected.tex:

        Sigma_{x,x*} = k(x, x*) + u^T (V - K) u_*

    This formula fails outside the inducing region!

    If unwhiten=True, converts V from whitened space to actual space.
    """
    model.eval()
    device = x_star.device
    dtype = x_star.dtype

    z = model.variational_strategy.inducing_points
    if z.ndim == 2 and z.shape[1] == 1:
        z = z.squeeze(-1)
    M = z.shape[0]

    V = model.variational_strategy.variational_distribution.covariance_matrix  # whitened!
    K_zz = model.covar_module(z, z).evaluate()
    K_zz_reg = K_zz + 1e-6 * torch.eye(M, device=device, dtype=dtype)

    # Unwhiten V if requested
    if unwhiten:
        L = torch.linalg.cholesky(K_zz_reg)  # K = L @ L^T
        V = L @ V @ L.T  # V_actual = L @ V_whitened @ L^T

    K_zz_inv = torch.linalg.inv(K_zz_reg)

    k_sample_z = model.covar_module(x_sample, z).evaluate()
    k_star_z = model.covar_module(x_star, z).evaluate()
    k_sample_star = model.covar_module(x_sample, x_star).evaluate()

    u_sample = K_zz_inv @ k_sample_z.T
    u_star = K_zz_inv @ k_star_z.T

    epistemic = u_sample.T @ V @ u_star
    prior_via_ind = u_sample.T @ K_zz @ u_star
    residual = k_sample_star - prior_via_ind
    Sigma_manual = residual + epistemic

    return Sigma_manual.squeeze(0)


def compute_conditional_moments_correct(model, x_sample, x_star, lambda_sample, debug=False, unwhiten=False):
    """
    Compute conditional moments using the CORRECT formula from
    distribution_aware_utility_pietro.tex (Eq. 216-219).

    Variance formula (Eq. 218):
        σ² = k(x*,x*) - u_*ᵀ ( K - [ V - (VuuᵀV)/(s + uᵀVu) ] ) u_*

    Expanding:
        σ² = k(x*,x*) - u_*ᵀ K u_* + u_*ᵀ V u_* - (u_*ᵀ V u)² / (s + uᵀVu)
           = s_* + u_*ᵀ V u_* - (u_*ᵀ V u)² / (s + uᵀVu)

    where:
        s = k(x,x) - k(x)^T K^{-1} k(x)     (Schur complement for x)
        s_* = k(x*,x*) - k(x*)^T K^{-1} k(x*)  (Schur complement for x*)

    If unwhiten=True, converts m and V from whitened space to actual space.
    """
    model.eval()
    device = x_star.device
    dtype = x_star.dtype

    # Get inducing points and variational parameters
    z = model.variational_strategy.inducing_points
    if z.ndim == 2 and z.shape[1] == 1:
        z = z.squeeze(-1)
    M = z.shape[0]

    m = model.variational_strategy.variational_distribution.mean  # (M,) - whitened!
    V = model.variational_strategy.variational_distribution.covariance_matrix  # (M, M) - whitened!

    K_zz = model.covar_module(z, z).evaluate()
    K_zz_reg = K_zz + 1e-6 * torch.eye(M, device=device, dtype=dtype)

    # Unwhiten m and V if requested
    if unwhiten:
        L = torch.linalg.cholesky(K_zz_reg)  # K = L @ L^T
        m = L @ m  # m_actual = L @ m_whitened
        V = L @ V @ L.T  # V_actual = L @ V_whitened @ L^T

    K_zz_inv = torch.linalg.inv(K_zz_reg)

    # Kernel evaluations
    k_sample_z = model.covar_module(x_sample, z).evaluate()  # (1, M)
    k_star_z = model.covar_module(x_star, z).evaluate()      # (K, M)
    k_sample_sample = model.covar_module(x_sample, x_sample).evaluate()  # (1, 1)
    k_star_star_diag = model.covar_module(x_star, x_star).evaluate().diag()  # (K,)

    # Projection vectors: u = K^{-1} k(z, x)
    u = K_zz_inv @ k_sample_z.T       # (M, 1)
    u_star = K_zz_inv @ k_star_z.T    # (M, K)

    # Schur complements (prior conditional variances)
    # s = k(x,x) - k(x)^T K^{-1} k(x)
    s = k_sample_sample - k_sample_z @ u  # scalar (1,1)
    s = s.squeeze()

    # s_* = k(x*,x*) - k(x*)^T K^{-1} k(x*) for each x*
    s_star = k_star_star_diag - torch.sum(k_star_z.T * u_star, dim=0)  # (K,)

    # Denominator: s + u^T V u
    uTVu = (u.T @ V @ u).squeeze()  # scalar
    denom = s + uTVu

    # V u product (used multiple times)
    Vu = V @ u  # (M, 1)

    # Conditional mean: μ = mean(x*) + u_*^T (m + (V u / denom) · (λ - [mean(x) + u^T m]))
    # Note: when unwhitened, m represents deviation from mean function
    mean_x_sample = model.mean_module(x_sample).squeeze()  # scalar
    mean_x_star = model.mean_module(x_star).squeeze()  # (K,)
    uTm = (u.T @ m).squeeze()  # scalar
    mu_at_sample = mean_x_sample + uTm  # full posterior mean at x_sample
    innovation = lambda_sample - mu_at_sample
    m_updated = m + (Vu.squeeze() / denom) * innovation  # (M,)
    mu_cond = mean_x_star + u_star.T @ m_updated  # (K,)

    # Conditional variance: σ² = s_* + u_*^T V u_* - (u_*^T V u)² / denom
    ustar_T_V_ustar = torch.sum(u_star * (V @ u_star), dim=0)  # (K,)
    ustar_T_Vu = u_star.T @ Vu  # (K, 1)
    correction = (ustar_T_Vu.squeeze() ** 2) / denom  # (K,)

    sigma2_cond = s_star + ustar_T_V_ustar - correction  # (K,)

    # Check: marginal variance should be s_* + u_*^T V u_*
    sigma2_marginal_formula = s_star + ustar_T_V_ustar

    if debug:
        print(f"\nDEBUG conditional moments (Schur complement):")
        print(f"  s (Schur at x_sample): {s.item():.6f}")
        print(f"  s_* range: [{s_star.min().item():.6f}, {s_star.max().item():.6f}]")
        print(f"  u_*^T V u_* range: [{ustar_T_V_ustar.min().item():.4f}, {ustar_T_V_ustar.max().item():.4f}]")
        print(f"  Marginal var formula (s_* + u_*^T V u_*): [{sigma2_marginal_formula.min().item():.4f}, {sigma2_marginal_formula.max().item():.4f}]")
        print(f"  denom (s + u^T V u): {denom.item():.6f}")
        print(f"  correction range: [{correction.min().item():.4f}, {correction.max().item():.4f}]")
        print(f"  sigma2_cond range: [{sigma2_cond.min().item():.4f}, {sigma2_cond.max().item():.4f}]")
        # Check V and K properties
        V_eig = torch.linalg.eigvalsh(V)
        K_eig = torch.linalg.eigvalsh(K_zz)
        print(f"  V eigenvalues: [{V_eig.min().item():.4f}, {V_eig.max().item():.4f}]")
        print(f"  K eigenvalues: [{K_eig.min().item():.6f}, {K_eig.max().item():.4f}]")
        print(f"  K condition number: {K_eig.max().item() / K_eig.min().item():.2e}")
        print(f"  max |u_*|: {u_star.abs().max().item():.4f}")

    return mu_cond, sigma2_cond


def train_gp_fixed_lengthscale(model, likelihood, train_x, train_y,
                                n_iterations=N_ITERATIONS, lr=0.1):
    """Train GP but keep lengthscale and outputscale fixed."""
    # Set and freeze lengthscale
    model.covar_module.base_kernel.lengthscale = FIXED_LENGTHSCALE
    model.covar_module.base_kernel.raw_lengthscale.requires_grad = False

    # Set and freeze outputscale
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

            # ELBO components
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
    SEED = 1
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Training data (SAME as diagnose_posterior_conditioning.py) ---
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

    # --- Compute quantities using GPyTorch ---
    with torch.no_grad():
        all_x = torch.cat([x_sample, x_plot])
        full_posterior = model(all_x)
        full_covar = full_posterior.covariance_matrix
        full_mean = full_posterior.mean

        Sigma_ii = full_covar[0, 0]
        Sigma_star_star = full_covar[1:, 1:].diag()
        Sigma_i_star = full_covar[0, 1:]
        mu_star = full_mean[1:]

        Sigma_cond = Sigma_star_star - (Sigma_i_star ** 2) / Sigma_ii
        innovation = lambda_sample - full_mean[0]
        mu_cond = mu_star + (Sigma_i_star / Sigma_ii) * innovation

    # --- Compute marginal entropy (GPyTorch) ---
    H_marg = compute_H_marginal(model, x_plot)

    # --- Compute marginal entropy (manual formula with unwhitening) ---
    H_marg_manual, mu_star_manual, sigma2_star_manual = compute_H_marginal_manual(
        model, x_plot, debug=True, unwhiten=True
    )
    # Get GPyTorch marginal moments for comparison
    with torch.no_grad():
        gpytorch_marg = model(x_plot)
        mu_star_gpytorch = gpytorch_marg.mean
        sigma2_star_gpytorch = gpytorch_marg.variance
    print(f"GPyTorch marginal mean range: [{mu_star_gpytorch.min().item():.4f}, {mu_star_gpytorch.max().item():.4f}]")
    print(f"GPyTorch marginal variance range: [{sigma2_star_gpytorch.min().item():.4f}, {sigma2_star_gpytorch.max().item():.4f}]")

    # --- Compute conditional entropy using GPyTorch quantities ---
    r_values = torch.arange(0, MAX_R, dtype=DTYPE, device=DEVICE)
    p_r_cond, log_p_r_cond = laplace_approximations_new(mu=mu_cond, sigma2=Sigma_cond, r=r_values)
    H_cond = -torch.sum(p_r_cond * log_p_r_cond, dim=1)

    # --- Average over multiple lambda samples ---
    N_LAMBDA = 500
    H_cond_list = []
    for i in range(N_LAMBDA):
        lam_i = mu_sample + std_sample * np.random.randn()
        mu_cond_i = mu_star + (Sigma_i_star / Sigma_ii) * (lam_i - full_mean[0])
        p_r_i, log_p_r_i = laplace_approximations_new(mu=mu_cond_i, sigma2=Sigma_cond, r=r_values)
        H_cond_i = -torch.sum(p_r_i * log_p_r_i, dim=1)
        H_cond_list.append(H_cond_i.cpu().numpy())
    H_cond_avg = np.mean(H_cond_list, axis=0)

    # --- Compute H_cond using manual cross-covariance formula (with unwhitening) ---
    with torch.no_grad():
        Sigma_i_star_manual = compute_cross_cov_manual(model, x_sample, x_plot, unwhiten=True)

        # Clamp for stability (same as utility.py)
        max_abs = torch.sqrt(Sigma_ii * Sigma_star_star) * 0.999
        Sigma_i_star_manual_clamped = torch.clamp(Sigma_i_star_manual, -max_abs, max_abs)

        # Conditional variance and mean using manual cross-cov
        Sigma_cond_manual = Sigma_star_star - (Sigma_i_star_manual_clamped ** 2) / Sigma_ii
        Sigma_cond_manual = torch.clamp(Sigma_cond_manual, min=1e-8)
        mu_cond_manual = mu_star + (Sigma_i_star_manual_clamped / Sigma_ii) * innovation

        # Compute H_cond with wrong formula
        p_r_manual, log_p_r_manual = laplace_approximations_new(mu=mu_cond_manual, sigma2=Sigma_cond_manual, r=r_values)
        H_cond_manual = -torch.sum(p_r_manual * log_p_r_manual, dim=1)

    # --- Compute H_cond using Schur complement formula (with unwhitening) ---
    with torch.no_grad():
        mu_cond_correct, sigma2_cond_correct = compute_conditional_moments_correct(
            model, x_sample, x_plot, lambda_sample, debug=True, unwhiten=True
        )
        # Also compare marginal variances to verify formula
        print(f"\nGPyTorch marginal variance range: [{Sigma_star_star.min().item():.4f}, {Sigma_star_star.max().item():.4f}]")
        sigma2_cond_correct = torch.clamp(sigma2_cond_correct, min=1e-8)

        p_r_correct, log_p_r_correct = laplace_approximations_new(
            mu=mu_cond_correct, sigma2=sigma2_cond_correct, r=r_values
        )
        H_cond_correct = -torch.sum(p_r_correct * log_p_r_correct, dim=1)

    # --- Convert to numpy ---
    x_np = x_plot.cpu().numpy()
    mu_before = mu_star.cpu().numpy()
    var_before = Sigma_star_star.cpu().numpy()
    mu_after = mu_cond.cpu().numpy()
    var_after = Sigma_cond.cpu().numpy()
    train_x_np = train_x.cpu().numpy()
    true_lambda_np = lambda_true(x_plot).cpu().numpy()
    H_marg_np = H_marg.cpu().numpy()
    H_marg_manual_np = H_marg_manual.cpu().numpy()
    H_cond_np = H_cond.cpu().numpy()
    H_cond_manual_np = H_cond_manual.cpu().numpy()
    H_cond_correct_np = H_cond_correct.cpu().numpy()
    cross_cov_gpytorch = Sigma_i_star.cpu().numpy()
    cross_cov_manual = Sigma_i_star_manual.cpu().numpy()
    sigma2_cond_correct_np = sigma2_cond_correct.cpu().numpy()
    mu_star_manual_np = mu_star_manual.cpu().numpy()
    sigma2_star_manual_np = sigma2_star_manual.cpu().numpy()
    mu_star_gpytorch_np = mu_star_gpytorch.cpu().numpy()
    sigma2_star_gpytorch_np = sigma2_star_gpytorch.cpu().numpy()

    # --- Compute manual utility ---
    utility_manual_np = H_marg_manual_np - H_cond_correct_np


    # --- Plot ---
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(12, 20), sharex=True)

    std_before = np.sqrt(var_before)
    std_after = np.sqrt(np.maximum(var_after, 0))

    ax1.fill_between(x_np, mu_before - 2*std_before, mu_before + 2*std_before,
                     alpha=0.3, color='blue', label='Before: ±2σ')
    ax1.plot(x_np, mu_before, 'b-', linewidth=2, label='Before: GP mean')

    ax1.fill_between(x_np, mu_after - 2*std_after, mu_after + 2*std_after,
                     alpha=0.3, color='red', label='After: ±2σ')
    ax1.plot(x_np, mu_after, 'r-', linewidth=2, label='After: GP mean')

    ax1.plot(x_np, true_lambda_np, 'k--', linewidth=1.5, label='True λ(x)')

    ax1.axvline(x=X_SAMPLE, color='green', linestyle='--', alpha=0.7)
    ax1.scatter([X_SAMPLE], [lambda_sample], color='green', s=100, zorder=5, marker='*',
                label=f'Observed λ = {lambda_sample:.2f}')

    for i, tx in enumerate(train_x_np):
        ax1.axvline(x=tx, color='red', alpha=0.3, linewidth=1, linestyle=':',
                    label='Training x' if i == 0 else None)

    ax1.axvspan(X_MIN, X_MAX, alpha=0.1, color='gray', label='Training region')
    ax1.set_ylabel('λ(x)')
    ax1.set_title(f'GP Posterior Before vs After Conditioning on λ({X_SAMPLE}) = {lambda_sample:.2f}\n(Fixed lengthscale = {FIXED_LENGTHSCALE})')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_np, var_before, 'b-', linewidth=2, label='Variance before conditioning')
    ax2.plot(x_np, var_after, 'r-', linewidth=2, label='Variance after conditioning')
    ax2.axvline(x=X_SAMPLE, color='green', linestyle='--', alpha=0.7, label='x_sample')
    for tx in train_x_np:
        ax2.axvline(x=tx, color='red', alpha=0.3, linewidth=1, linestyle=':')
    ax2.axvspan(X_MIN, X_MAX, alpha=0.1, color='gray')
    ax2.set_ylabel('Variance')
    ax2.set_title('Variance: Before vs After Conditioning')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: H_marg, H_cond, and Utility
    utility_np = H_marg_np - H_cond_np
    ax3.plot(x_np, H_marg_np, 'b-', linewidth=2, label='H_marg(x*)')
    ax3.plot(x_np, H_cond_np, 'r-', linewidth=2, label='H_cond(x*)')
    ax3.plot(x_np, utility_np, 'g-', linewidth=2, label='Utility = H_marg - H_cond')
    ax3.axvline(x=X_SAMPLE, color='green', linestyle='--', alpha=0.7)
    for tx in train_x_np:
        ax3.axvline(x=tx, color='red', alpha=0.3, linewidth=1, linestyle=':')
    ax3.axvspan(X_MIN, X_MAX, alpha=0.1, color='gray')
    ax3.axhline(y=0, color='gray', linewidth=1, linestyle='-')
    ax3.set_ylabel('Entropy (nats)')
    ax3.set_title('Entropy Components and Utility')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Utility comparison
    utility_avg = H_marg_np - H_cond_avg
    ax4.plot(x_np, utility_np, 'g-', linewidth=1, alpha=0.5, label='Single sample')
    ax4.plot(x_np, utility_avg, 'b-', linewidth=2, label=f'Averaged ({N_LAMBDA} samples)')
    ax4.axvline(x=X_SAMPLE, color='green', linestyle='--', alpha=0.7)
    for tx in train_x_np:
        ax4.axvline(x=tx, color='red', alpha=0.3, linewidth=1, linestyle=':')
    ax4.axvspan(X_MIN, X_MAX, alpha=0.1, color='gray')
    ax4.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax4.set_ylabel('Utility (nats)')
    ax4.set_title('Utility: Single vs Averaged (GPyTorch cross-cov)')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # Subplot 5: H_cond comparison - GPyTorch vs manual formulas
    # Both manual formulas are NUMERICALLY UNSTABLE due to ill-conditioned K matrix
    ax5.plot(x_np, H_cond_np, 'b-', linewidth=2, label='H_cond (GPyTorch) [CORRECT]')
    ax5.plot(x_np, H_cond_manual_np, 'r-', linewidth=2, label='H_cond (cross-cov formula) [UNSTABLE]')
    ax5.plot(x_np, H_cond_correct_np, 'g--', linewidth=2, label='H_cond (Schur formula) [UNSTABLE]')
    ax5.axvline(x=X_SAMPLE, color='green', linestyle='--', alpha=0.7)
    for tx in train_x_np:
        ax5.axvline(x=tx, color='red', alpha=0.3, linewidth=1, linestyle=':')
    ax5.axvspan(X_MIN, X_MAX, alpha=0.1, color='gray')
    ax5.set_ylabel('Entropy (nats)')
    ax5.set_title('H_cond: Manual formulas fail due to ill-conditioned K (cond=3.7e6)')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)

    # Subplot 6: H_marg comparison - GPyTorch vs manual formula
    ax6.plot(x_np, H_marg_np, 'b-', linewidth=2, label='H_marg (GPyTorch)')
    ax6.plot(x_np, H_marg_manual_np, 'r--', linewidth=2, label='H_marg (manual: s* + u*^T V u*)')
    ax6.axvline(x=X_SAMPLE, color='green', linestyle='--', alpha=0.7)
    for tx in train_x_np:
        ax6.axvline(x=tx, color='red', alpha=0.3, linewidth=1, linestyle=':')
    ax6.axvspan(X_MIN, X_MAX, alpha=0.1, color='gray')
    ax6.set_ylabel('Entropy (nats)')
    ax6.set_title('H_marg: GPyTorch vs Manual formula (σ²* = s* + u*^T V u*)')
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)

    # Subplot 7: Utility comparison - GPyTorch vs Manual formulas
    ax7.plot(x_np, utility_np, 'b-', linewidth=2, label='Utility (GPyTorch)')
    ax7.plot(x_np, utility_manual_np, 'r--', linewidth=2, label='Utility (Manual unwhitened)')
    ax7.axvline(x=X_SAMPLE, color='green', linestyle='--', alpha=0.7)
    for tx in train_x_np:
        ax7.axvline(x=tx, color='red', alpha=0.3, linewidth=1, linestyle=':')
    ax7.axvspan(X_MIN, X_MAX, alpha=0.1, color='gray')
    ax7.axhline(y=0, color='gray', linewidth=1, linestyle='-')
    ax7.set_xlabel('x')
    ax7.set_ylabel('Utility (nats)')
    ax7.set_title('Utility: GPyTorch vs Manual (unwhitened) formulas')
    ax7.legend(loc='upper right')
    ax7.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/1D_playground/train_border_bug/simple_utility_gpytorch_result.png', dpi=150)
    plt.close()

    print(f"\nSample point: x={X_SAMPLE}, λ={lambda_sample:.2f}")
    print(f"Posterior mean at x_sample: {mu_sample:.2f}")
    print(f"Innovation (λ - μ): {lambda_sample - mu_sample:.2f}")
    print(f"\nSingle sample utility:")
    print(f"  Min: {utility_np.min():.4f}, Max: {utility_np.max():.4f}")
    print(f"  Negative values: {(utility_np < 0).sum()} / {len(utility_np)}")
    print(f"\nAveraged utility ({N_LAMBDA} samples):")
    print(f"  Min: {utility_avg.min():.4f}, Max: {utility_avg.max():.4f}")
    print(f"  Negative values: {(utility_avg < 0).sum()} / {len(utility_avg)}")
    print(f"\nCross-covariance comparison:")
    print(f"  GPyTorch:     min={cross_cov_gpytorch.min():.4f}, max={cross_cov_gpytorch.max():.4f}")
    print(f"  WRONG manual: min={cross_cov_manual.min():.4f}, max={cross_cov_manual.max():.4f}")
    print(f"\nConditional variance (Schur complement):")
    print(f"  GPyTorch (var_after): min={var_after.min():.6f}, max={var_after.max():.6f}")
    print(f"  CORRECT formula:      min={sigma2_cond_correct_np.min():.6f}, max={sigma2_cond_correct_np.max():.6f}")
    print(f"\nH_cond comparison:")
    print(f"  GPyTorch:       min={H_cond_np.min():.4f}, max={H_cond_np.max():.4f}")
    print(f"  WRONG manual:   min={H_cond_manual_np.min():.4f}, max={H_cond_manual_np.max():.4f}")
    print(f"  CORRECT Schur:  min={H_cond_correct_np.min():.4f}, max={H_cond_correct_np.max():.4f}")
    print(f"\nH_marg comparison:")
    print(f"  GPyTorch: min={H_marg_np.min():.4f}, max={H_marg_np.max():.4f}")
    print(f"  Manual:   min={H_marg_manual_np.min():.4f}, max={H_marg_manual_np.max():.4f}")
    print(f"\nMarginal variance comparison:")
    print(f"  GPyTorch: min={sigma2_star_gpytorch_np.min():.6f}, max={sigma2_star_gpytorch_np.max():.6f}")
    print(f"  Manual:   min={sigma2_star_manual_np.min():.6f}, max={sigma2_star_manual_np.max():.6f}")
    print(f"\nMarginal mean comparison:")
    print(f"  GPyTorch: min={mu_star_gpytorch_np.min():.4f}, max={mu_star_gpytorch_np.max():.4f}")
    print(f"  Manual:   min={mu_star_manual_np.min():.4f}, max={mu_star_manual_np.max():.4f}")
    print(f"\nUtility comparison (GPyTorch vs Manual unwhitened):")
    print(f"  GPyTorch: min={utility_np.min():.4f}, max={utility_np.max():.4f}")
    print(f"  Manual:   min={utility_manual_np.min():.4f}, max={utility_manual_np.max():.4f}")
    print(f"\nPlot saved to simple_utility_gpytorch_result.png")


if __name__ == '__main__':
    main()
