"""
Utility exploration with NORMALIZED arc-cosine kernel.
Created by Claude (copy of explore_utility.py with normalized kernel).

Trains a GP model using ArcCosineKernelNormalized (constant diagonal = 1.0)
and provides helper functions to explore utility behavior. Tests whether
normalized kernel eliminates utility divergence toward high-norm images.

Uses standard_utility() and distribution_aware_utility() from acquisition.py
(single source of truth for acquisition functions).

Usage:
    python investigations/normalized_kernel/explore_utility_normalized.py
"""

import sys
import json
import math
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_gpytorch_dir = _script_dir.parent.parent
sys.path.insert(0, str(_gpytorch_dir))

import importlib.util

# Import from local utils.py via importlib to avoid sys.modules shadowing
_local_utils_path = _gpytorch_dir / 'utils.py'
_spec = importlib.util.spec_from_file_location("gpytorch_porting_utils", str(_local_utils_path))
_local_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_utils)

get_gp_marginal_moments = _local_utils.get_gp_marginal_moments
get_gp_conditional_moments = _local_utils.get_gp_conditional_moments
_diff_laplace_log_probs = _local_utils._diff_laplace_log_probs
compute_adaptive_rmax = _local_utils.compute_adaptive_rmax

# Import from acquisition.py via importlib (same pattern)
_acquisition_path = _gpytorch_dir / 'acquisition.py'
_spec_acq = importlib.util.spec_from_file_location("gpytorch_porting_acquisition", str(_acquisition_path))
_acquisition = importlib.util.module_from_spec(_spec_acq)
_spec_acq.loader.exec_module(_acquisition)

standard_utility = _acquisition.standard_utility
distribution_aware_utility = _acquisition.distribution_aware_utility

from run_single_mode import run_single_config, build_config_from_defaults, load_pnas_data
from kernels import ArcCosineKernelNormalized
from gpy_model import VariationalGPModel
from likelihoods import PoissonLikelihood
from gpy_training import train_gpy_default
from utils import compute_rf_center_from_sta

# ---------------------------------------------------------------------------
# Adaptive r_max params from default_params.json
# ---------------------------------------------------------------------------
with open(_gpytorch_dir / 'default_params.json') as f:
    _defaults = json.load(f)
_ADAPTIVE_RMAX_PARAMS = {
    'adaptive_safety_k': _defaults['utility']['adaptive_safety_k'],
    'adaptive_max_rmax': _defaults['utility']['adaptive_max_rmax'],
    'adaptive_min_rmax': _defaults['utility']['adaptive_min_rmax'],
}

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------
N_TRAIN = 100
M = 100


# ============================================================================
# Setup
# ============================================================================

def setup():
    """Train model with NORMALIZED kernel and return everything needed for exploration.

    ONLY DIFFERENCE from original: uses ArcCosineKernelNormalized instead of ArcCosineKernel.

    Returns:
        dict with keys:
            model: trained GP model (eval mode)
            likelihood: PoissonLikelihood with .A and .lambda0
            X_pool: (N_pool, n_pixels) pool of natural images
            X_train: (N_train, n_pixels) training images
            x_target: (n_pixels,) first pool image (convenient reference)
            config: full config dict
    """
    # === Get config from defaults (ensures all params match) ===
    config = build_config_from_defaults(mode='default_gpy', M=M, n_train=N_TRAIN)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    seed = config['seed']
    cell = config['cell']
    n_px_side = config['n_px_side']

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # === Data loading copied from run_single_mode.py:476-520 (2026-02-09) ===
    data_path = _gpytorch_dir.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    print(f"Loading data from: {data_path}")
    data = load_pnas_data(data_path, dtype=dtype)

    # Combine train + val, flatten
    X = torch.cat([data['X_train'], data['X_val']], dim=0)
    R = torch.cat([data['R_train'], data['R_val']], dim=0)
    X = X.reshape(X.shape[0], -1).to(device)  # (N, 11664)
    R = R.to(device)

    # Select cell
    r = R[:, cell]

    # === Compute RF center from STA ===
    n_samples_sta = config['n_samples_sta']
    if n_samples_sta is not None:
        n_sta = min(n_samples_sta, X.shape[0])
        indices_sta = torch.randperm(X.shape[0], device=device)[:n_sta]
        X_sta = X[indices_sta]
        r_sta = r[indices_sta]
    else:
        X_sta = X
        r_sta = r

    eps_0x_sta, eps_0y_sta = compute_rf_center_from_sta(
        X_sta, r_sta, n_px_side, zscore=True
    )

    # Use STA-computed center if config has None
    eps_0x = config['eps_0x'] if config['eps_0x'] is not None else eps_0x_sta
    eps_0y = config['eps_0y'] if config['eps_0y'] is not None else eps_0y_sta

    print(f"RF center: ({eps_0x:.4f}, {eps_0y:.4f})")

    # === Select training indices (random) ===
    all_indices = torch.randperm(X.shape[0], device=device)
    indices_train = all_indices[:N_TRAIN]

    X_train = X[indices_train]
    r_train = r[indices_train]

    # Select inducing points (same as training for this setup)
    inducing_points = X_train.clone()

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  inducing_points: {inducing_points.shape}")

    # === Model creation copied from run_single_mode.py:843-871 (default_gpy block) ===
    # ONLY CHANGE: ArcCosineKernel → ArcCosineKernelNormalized
    kernel = ArcCosineKernelNormalized(
        sigma_0=config['sigma_0'],
        Amp=config['Amp'],  # Pass Amp in constructor
        n_px_side=n_px_side,
        eps_0x=eps_0x,
        eps_0y=eps_0y,
        beta=config['beta'],
        rho=config['rho'],
        use_mask=config['use_mask'],
        gradient_mode='autograd'  # Normalized kernel only supports autograd
    )

    jitter = config['jitter']
    model = VariationalGPModel(
        inducing_points, kernel, jitter=jitter,
        standard_variational_distribution=not config['unwhitened_variational_dist']
    )

    A_init = config['A_init']
    lambda0_init = config['lambda0_init']
    likelihood = PoissonLikelihood(A_init=A_init, lambda0_init=lambda0_init)

    model = model.to(dtype=dtype, device=device)
    likelihood = likelihood.to(dtype=dtype, device=device)

    print(f"\nInitial parameters:")
    print(f"  A: {likelihood.A.item():.4f}, lambda0: {likelihood.lambda0.item():.4f}")
    print(f"  Amp: {kernel.Amp.item():.6f}")

    # === Train model (reuse existing function - no duplication) ===
    print(f"\nTraining with ArcCosineKernelNormalized:")
    train_result = train_gpy_default(
        model=model,
        likelihood=likelihood,
        train_x=X_train,
        train_y=r_train,
        optimizer_name=config['optimizer'],
        lr=config['lr'],
        n_iterations=config['n_iterations'],
        print_every=10,
        device=device,
        early_stop=config['early_stop'],
        stop_window=config['stop_window'],
        stop_thresh=config['stop_thresh'],
        min_iterations=config['min_iterations'],
        lbfgs_max_iter=config['gpy_lbfgs_max_iter'],
        jitter=jitter,
        cholesky_max_tries=config['cholesky_max_tries'],
    )

    model.eval()

    # === Build pool (all images not in training) ===
    pool_indices = sorted(set(range(X.shape[0])) - set(indices_train.cpu().numpy().tolist()))
    X_pool = X[pool_indices]

    print(f"\nReady. Pool: {X_pool.shape[0]} images, Train: {X_train.shape[0]} images")
    print(f"Final: A={likelihood.A.item():.4f}, lambda0={likelihood.lambda0.item():.4f}")

    return {
        'model': model,
        'likelihood': likelihood,
        'X_pool': X_pool,
        'X_train': X_train,
        'x_target': X_pool[0].clone(),
        'config': config,
    }


# ============================================================================
# Kernel helpers
# ============================================================================

def kernel_value(model, x1, x2):
    """K(x1, x2) — scalar kernel value between two images."""
    with torch.no_grad():
        return model.covar_module(
            x1.unsqueeze(0), x2.unsqueeze(0)
        ).evaluate().squeeze().item()


def kernel_norm(model, x):
    """||x||_C = sqrt(K(x, x)) — the C-weighted norm."""
    return math.sqrt(kernel_value(model, x, x))


def kernel_angle(model, x1, x2):
    """Angle between x1 and x2 in kernel space (radians).

    Computes arccos(K(x1,x2) / sqrt(K(x1,x1)*K(x2,x2))).
    For the arc-cosine kernel this equals arccos(J(theta)/pi),
    which approximates the geometric C-space angle theta well
    for small angles (error O(theta^4)).
    """
    k11 = kernel_value(model, x1, x1)
    k22 = kernel_value(model, x2, x2)
    k12 = kernel_value(model, x1, x2)
    cos_a = k12 / (math.sqrt(k11 * k22) + 1e-30)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.acos(cos_a)


# ============================================================================
# GP moment helpers
# ============================================================================

def gp_moments(model, x):
    """GP posterior moments at x. Returns (mu, sigma2) as floats."""
    with torch.no_grad():
        mu, sigma2 = get_gp_marginal_moments(model, x.unsqueeze(0))
    return mu.item(), sigma2.item()


def logfiring_moments(model, likelihood, x):
    """Log-firing-rate moments: g = A*lambda + lambda0.

    Returns (logf_mean, logf_var) as floats.
    """
    mu, sigma2 = gp_moments(model, x)
    A = likelihood.A.item()
    lam0 = likelihood.lambda0.item()
    return A * mu + lam0, A ** 2 * sigma2


# ============================================================================
# Convenience: describe an image
# ============================================================================

def describe(model, likelihood, x, x_ref=None, label="image"):
    """Print a summary of an image's GP and utility properties.

    Args:
        model, likelihood: trained model
        x: the image to describe
        x_ref: optional reference image for angle and DA utility
        label: name for printing
    """
    mu, sigma2 = gp_moments(model, x)
    logf_mean, logf_var = logfiring_moments(model, likelihood, x)
    norm = kernel_norm(model, x)

    with torch.no_grad():
        u_std_result = standard_utility(
            model, likelihood, x.unsqueeze(0),
            r_max=None, adaptive_r_max=True,
            **_ADAPTIVE_RMAX_PARAMS
        )
    u_std = u_std_result['utility'].item()

    print(f"\n  --- {label} ---")
    print(f"  ||x||_C = {norm:.2f}")
    print(f"  GP:  mu = {mu:.4f},  sigma2 = {sigma2:.4f}")
    print(f"  Log-firing:  mean = {logf_mean:.4f},  var = {logf_var:.4f}")
    print(f"  U_std = {u_std:.6f}")

    if x_ref is not None:
        angle = kernel_angle(model, x, x_ref)
        with torch.no_grad():
            da_result = distribution_aware_utility(
                model, likelihood,
                x_candidates=x.unsqueeze(0),
                x_samples=x_ref.unsqueeze(0),
                r_max=None, adaptive_r_max=True,
                sample_lambda=False,
                **_ADAPTIVE_RMAX_PARAMS
            )
        u_da = da_result['utility'].item()
        h_marg = da_result['H_marg'].item()
        h_cond = da_result['H_cond'].item()
        print(f"  Angle to ref = {angle:.4f} rad ({math.degrees(angle):.1f} deg)")
        print(f"  U_DA = {u_da:.6f}  (H_marg={h_marg:.4f}, H_cond={h_cond:.4f})")


# ============================================================================
# DA utility conditioned on one image
# ============================================================================

def eval_da_conditioned(model, likelihood, X_train, x_cond):
    """Evaluate DA utility of training images conditioned on one image.

    Computes U_DA(x_i) = H_marg(x_i) - H_cond(x_i | lambda(x_cond))
    for each training image x_i and for x_cond itself.

    Prints a table with DA utility, H_marg, H_cond, and kernel angle
    to x_cond for each image, sorted by DA utility descending.
    """
    print("\n" + "=" * 60)
    print("DA UTILITY CONDITIONED ON x_cond")
    print("=" * 60)

    # Candidates = all training images + x_cond itself
    n_train = X_train.shape[0]
    x_candidates = torch.cat([X_train, x_cond.unsqueeze(0)], dim=0)

    with torch.no_grad():
        result = distribution_aware_utility(
            model, likelihood,
            x_candidates=x_candidates,
            x_samples=x_cond.unsqueeze(0),
            r_max=None, adaptive_r_max=True,
            sample_lambda=False,
            **_ADAPTIVE_RMAX_PARAMS
        )

    utilities = result['utility']
    h_margs = result['H_marg']
    h_conds = result['H_cond']

    # Compute angles and norms for each candidate
    angles = []
    norms = []
    for i in range(x_candidates.shape[0]):
        angles.append(kernel_angle(model, x_candidates[i], x_cond))
        norms.append(kernel_norm(model, x_candidates[i]))
    norm_cond = kernel_norm(model, x_cond)

    # Sort by DA utility descending
    sorted_idx = utilities.argsort(descending=True)

    print(f"\n  ||x_cond||_C = {norm_cond:.2f}")
    print(f"\n  {'rank':>4}  {'image':>10}  {'U_DA':>10}  {'H_marg':>10}  "
          f"{'H_cond':>10}  {'||x*||_C':>8}  {'ratio':>6}  {'angle(rad)':>10}  {'angle(deg)':>10}")
    print("  " + "-" * 93)

    for rank, idx in enumerate(sorted_idx):
        idx_val = idx.item()
        label = f"train[{idx_val}]" if idx_val < n_train else "x_cond"
        ratio = norms[idx_val] / norm_cond
        print(f"  {rank+1:>4}  {label:>10}  {utilities[idx_val].item():10.6f}  "
              f"{h_margs[idx_val].item():10.4f}  {h_conds[idx_val].item():10.4f}  "
              f"{norms[idx_val]:8.2f}  {ratio:6.2f}  "
              f"{angles[idx_val]:10.4f}  {math.degrees(angles[idx_val]):10.1f}")

    print(f"\n  Total images evaluated: {x_candidates.shape[0]} "
          f"({n_train} training + x_cond)")


# ============================================================================
# DA utility landscape plot
# ============================================================================

def plot_da_landscape(model, likelihood, X_candidates, x_cond, save_path,
                      n_grid_mu=200, n_grid_sigma2=150):
    """Plot entropy heatmap with marginal/conditional scatter for DA utility.

    Single panel. Background is H(mu_g, sigma2_g). Each image x* appears
    twice: at its marginal moments (before conditioning) and at its
    conditional moments (after conditioning on x_cond). Pairs are
    connected by faint lines.

    Args:
        model, likelihood: trained model in eval mode.
        X_candidates: (N, d) images to evaluate.
        x_cond: (d,) conditioning image.
        save_path: where to save the PNG.
        n_grid_mu, n_grid_sigma2: heatmap grid resolution.
    """
    A = likelihood.A.item()
    lam0 = likelihood.lambda0.item()
    device = next(model.parameters()).device

    # --- Compute scatter point coordinates ---
    with torch.no_grad():
        mu_marg, sigma2_marg = get_gp_marginal_moments(model, X_candidates)
        lambda_cond = model(x_cond.unsqueeze(0)).mean[0]
        mu_cond, sigma2_cond = get_gp_conditional_moments(
            model, X_candidates, x_cond, lambda_cond
        )

    # Transform to g-space
    mu_g_marg = (A * mu_marg + lam0).cpu().numpy()
    s2_g_marg = (A ** 2 * sigma2_marg).cpu().numpy()
    mu_g_cond = (A * mu_cond + lam0).cpu().numpy()
    s2_g_cond = (A ** 2 * sigma2_cond).cpu().numpy()

    # x_cond index (last element)
    n_cand = X_candidates.shape[0]

    # --- Determine axis range from data ---
    all_mu = np.concatenate([mu_g_marg, mu_g_cond])
    all_s2 = np.concatenate([s2_g_marg, s2_g_cond])
    mu_range = all_mu.max() - all_mu.min()
    s2_range = all_s2.max() - all_s2.min()
    pad_mu = max(mu_range * 0.15, 0.1)
    pad_s2 = max(s2_range * 0.15, 0.005)
    mu_g_min = all_mu.min() - pad_mu
    mu_g_max = all_mu.max() + pad_mu
    s2_g_min = max(all_s2.min() - pad_s2, 1e-6)
    s2_g_max = all_s2.max() + pad_s2

    # --- Compute entropy heatmap ---
    mu_g_grid = torch.linspace(mu_g_min, mu_g_max, n_grid_mu)
    s2_g_grid = torch.linspace(s2_g_min, s2_g_max, n_grid_sigma2)

    # Determine safe r_max for the grid's upper corner
    r_max = compute_adaptive_rmax(
        torch.tensor([mu_g_max]), torch.tensor([s2_g_max]),
        safety_k=_ADAPTIVE_RMAX_PARAMS['adaptive_safety_k'],
        max_rmax=_ADAPTIVE_RMAX_PARAMS['adaptive_max_rmax'],
        min_rmax=_ADAPTIVE_RMAX_PARAMS['adaptive_min_rmax'],
    )

    # Compute H on grid (batched)
    r = torch.arange(0, r_max, dtype=torch.float32, device=device)
    H_grid = np.full((n_grid_mu, n_grid_sigma2), np.nan, dtype=np.float32)
    batch_rows = 20
    for start in range(0, n_grid_mu, batch_rows):
        end = min(start + batch_rows, n_grid_mu)
        batch_size = end - start
        mu_flat = mu_g_grid[start:end].repeat_interleave(n_grid_sigma2).to(device)
        s2_flat = s2_g_grid.repeat(batch_size).to(device)
        p_r, log_p_r = _diff_laplace_log_probs(mu_flat, s2_flat, r)
        H_flat = -torch.sum(p_r * log_p_r, dim=1).cpu().numpy()
        H_grid[start:end] = H_flat.reshape(batch_size, n_grid_sigma2)

    # --- Compute H_marg, H_cond per candidate for right panel ---
    mu_g_marg_t = torch.tensor(mu_g_marg, dtype=torch.float32, device=device)
    s2_g_marg_t = torch.tensor(s2_g_marg, dtype=torch.float32, device=device)
    mu_g_cond_t = torch.tensor(mu_g_cond, dtype=torch.float32, device=device)
    s2_g_cond_t = torch.tensor(s2_g_cond, dtype=torch.float32, device=device)

    p_marg, logp_marg = _diff_laplace_log_probs(mu_g_marg_t, s2_g_marg_t, r)
    H_marg_vals = (-torch.sum(p_marg * logp_marg, dim=1)).cpu().numpy()
    p_cond, logp_cond = _diff_laplace_log_probs(mu_g_cond_t, s2_g_cond_t, r)
    H_cond_vals = (-torch.sum(p_cond * logp_cond, dim=1)).cpu().numpy()
    U_DA_vals = H_marg_vals - H_cond_vals

    # Compute kernel angles and norms for right panel
    angles_rad = np.array([kernel_angle(model, X_candidates[i], x_cond)
                           for i in range(n_cand)])
    norms = np.array([kernel_norm(model, X_candidates[i])
                      for i in range(n_cand)])

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # === Left panel: H(R) landscape ===
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='lightgray', alpha=0.5)
    im = ax1.pcolormesh(
        s2_g_grid.numpy(), mu_g_grid.numpy(), H_grid,
        cmap=cmap, shading='auto',
    )
    cbar = plt.colorbar(im, ax=ax1, pad=0.02)
    cbar.set_label('H(R | $\\mu_g$, $\\sigma^2_g$)', fontsize=10)

    # Connecting lines (draw first, behind dots)
    for i in range(n_cand):
        ax1.plot(
            [s2_g_marg[i], s2_g_cond[i]],
            [mu_g_marg[i], mu_g_cond[i]],
            color='gray', linewidth=0.5, alpha=0.4, zorder=3,
        )

    # Marginal dots
    ax1.scatter(
        s2_g_marg, mu_g_marg,
        s=25, c='tab:blue', edgecolors='white', linewidths=0.3,
        zorder=5, label='marginal',
    )

    # Conditional dots
    ax1.scatter(
        s2_g_cond, mu_g_cond,
        s=25, c='tab:orange', edgecolors='white', linewidths=0.3,
        zorder=5, label='conditional',
    )

    # x_cond marker (marginal position, last candidate)
    ax1.scatter(
        [s2_g_marg[-1]], [mu_g_marg[-1]],
        s=80, marker='*', c='red', edgecolors='white', linewidths=0.5,
        zorder=6, label='$x_{cond}$',
    )

    ax1.set_xlabel('$\\sigma^2_g$ (log-firing rate variance)', fontsize=10)
    ax1.set_ylabel('$\\mu_g$ (log-firing rate mean)', fontsize=10)
    ax1.set_title('H(R) Landscape: conditioning shifts from $x_{cond}$', fontsize=11)
    ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)

    # === Right panel: U_DA vs norm, colored by angle ===
    idx_cond = n_cand - 1
    idx_train = list(range(idx_cond))

    sc = ax2.scatter(
        norms[idx_train], U_DA_vals[idx_train],
        s=30, c=angles_rad[idx_train], cmap='plasma',
        vmin=0, vmax=np.pi / 2,
        edgecolors='white', linewidths=0.3, zorder=3,
    )
    ax2.scatter(
        norms[idx_cond], U_DA_vals[idx_cond],
        s=100, marker='*', c='red', edgecolors='white', linewidths=0.5,
        zorder=5, label='$x_{cond}$',
    )
    cbar2 = plt.colorbar(sc, ax=ax2, pad=0.02)
    cbar2.set_label('Angle to $x_{cond}$ (rad)', fontsize=10)

    ax2.set_xlabel('$||x^*||_C$', fontsize=10)
    ax2.set_ylabel('$H_{marg} - H_{cond}$', fontsize=10)
    ax2.set_title('DA utility vs norm (color = angle)', fontsize=11)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Run a pedagogical demonstration."""
    env = setup()
    model = env['model']
    likelihood = env['likelihood']
    X_pool = env['X_pool']
    X_train = env['X_train']
    x_target = env['x_target']

    # === Part 1: A natural image ===
    print("\n" + "=" * 60)
    print("PART 1: A natural image from the pool")
    print("=" * 60)
    describe(model, likelihood, x_target, x_ref=x_target, label="x_target (natural)")

    # === Part 2: Same image, scaled up ===
    print("\n" + "=" * 60)
    print("PART 2: Same image scaled by c=5 (louder, same direction)")
    print("=" * 60)
    x_scaled = 5.0 * x_target
    describe(model, likelihood, x_scaled, x_ref=x_target,
             label="5 * x_target")

    # === Part 3: A different natural image (different direction) ===
    print("\n" + "=" * 60)
    print("PART 3: A different natural image (different direction)")
    print("=" * 60)
    # Find one at moderate angle
    angles = []
    for i in range(min(100, X_pool.shape[0])):
        a = kernel_angle(model, x_target, X_pool[i])
        angles.append((i, a))
    angles.sort(key=lambda x: x[1])
    # Pick one near the median angle
    mid_idx, mid_angle = angles[len(angles) // 2]
    x_other = X_pool[mid_idx]
    describe(model, likelihood, x_other, x_ref=x_target,
             label=f"natural image (pool[{mid_idx}])")

    # === Part 4: That other image, scaled up ===
    print("\n" + "=" * 60)
    print("PART 4: Different natural image scaled by c=5")
    print("=" * 60)
    x_other_scaled = 5.0 * x_other
    describe(model, likelihood, x_other_scaled, x_ref=x_target,
             label=f"5 * pool[{mid_idx}]")

    # === Part 5: Comparison table ===
    print("\n" + "=" * 60)
    print("COMPARISON: Amplitude vs Angle effects")
    print("=" * 60)

    cases = [
        ("x_target", x_target),
        ("5 * x_target", x_scaled),
        (f"pool[{mid_idx}]", x_other),
        (f"5 * pool[{mid_idx}]", x_other_scaled),
    ]

    # Batch compute utilities
    x_batch = torch.stack([x for _, x in cases])
    with torch.no_grad():
        std_result = standard_utility(
            model, likelihood, x_batch,
            r_max=None, adaptive_r_max=True,
            **_ADAPTIVE_RMAX_PARAMS
        )
        da_result = distribution_aware_utility(
            model, likelihood,
            x_candidates=x_batch,
            x_samples=x_target.unsqueeze(0),
            r_max=None, adaptive_r_max=True,
            sample_lambda=False,
            **_ADAPTIVE_RMAX_PARAMS
        )

    print(f"\n  {'Image':>25}  {'||x||_C':>8}  {'angle':>8}  {'U_std':>10}  {'U_DA':>10}")
    print("  " + "-" * 65)

    for i, (label, x) in enumerate(cases):
        norm = kernel_norm(model, x)
        angle = kernel_angle(model, x, x_target)
        u_std = std_result['utility'][i].item()
        u_da = da_result['utility'][i].item()

        print(f"  {label:>25}  {norm:8.1f}  {angle:8.4f}  {u_std:10.6f}  {u_da:10.6f}")

    print(f"""
  Key observations:
  - Scaling UP (same direction): ||x||_C grows, angle stays ~0,
    conditioning stays strong, DA utility grows.
  - Different direction: angle is large, conditioning is weak,
    DA utility is small regardless of amplitude.
  - Standard utility grows with amplitude for ALL images (no angle dependence).
""")

    # === Part 6: DA utility conditioned on one image ===
    eval_da_conditioned(model, likelihood, X_train, x_target)

    # === Part 7: DA utility landscape + scatter plot ===
    x_candidates = torch.cat([X_train, x_target.unsqueeze(0)], dim=0)
    plot_da_landscape(
        model, likelihood, x_candidates, x_target,
        save_path=_script_dir / 'da_utility_landscape_normalized.png',
    )


if __name__ == '__main__':
    demo()
