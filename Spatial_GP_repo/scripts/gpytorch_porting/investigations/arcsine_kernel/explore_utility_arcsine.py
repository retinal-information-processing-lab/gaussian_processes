"""
Workbench for exploring utility function behavior with the ARC-SINE kernel.
Created by Claude.

IMPORTANT: This script fits its OWN model during setup() - it does NOT use a
pre-trained model. The model is trained fresh each time with M=50, n_train=50
(hardcoded below) using ArcSineKernel (Williams 1998).

For fair test_r comparison between kernels, use run_arcsine.py with matched
parameters. This script is for exploring utility behavior patterns, not for
model performance evaluation.

Trains a GP model and provides simple helper functions to compute
kernel values, GP moments, and utilities for any image. Designed for
interactive exploration and pedagogical use.

Uses standard_utility() and distribution_aware_utility() from acquisition.py
(single source of truth for acquisition functions).

The arc-sine kernel K(x,x) saturates at 1 for large inputs (unlike arc-cosine
which grows quadratically). This means kernel_norm = sqrt(K(x,x)) is bounded
by 1, so the right panel of the landscape plot will show a compressed x-axis
compared to the arc-cosine version.

Usage:
    # Run directly for a demo:
    python investigations/arcsine_kernel/explore_utility_arcsine.py

    # Or import in a script/REPL:
    from explore_utility_arcsine import setup, kernel_angle, describe, ...
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

from run_arcsine import run_single_config, build_config_from_defaults, load_pnas_data

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

M = 50
SIGMA_0 = None  # Override kernel sigma_0 AFTER training, before utility eval. None = keep trained value.


# ============================================================================
# Setup
# ============================================================================

def setup():
    """Train model with ArcSineKernel and return everything needed for exploration.

    Returns:
        dict with keys:
            model: trained GP model (eval mode)
            likelihood: PoissonLikelihood with .A and .lambda0
            X_pool: (N_pool, n_pixels) pool of natural images
            X_train: (N_train, n_pixels) training images
            x_target: (n_pixels,) first pool image (convenient reference)
            config: full config dict
    """
    config = build_config_from_defaults(mode='default_gpy', M=M, n_train=N_TRAIN)
    result = run_single_config(config)

    if result is None or result.get('status') != 'success':
        raise RuntimeError("Training failed")

    model = result['_model']
    likelihood = result['_likelihood']
    indices_train = result['_indices_train']
    model.eval()

    # Build pool and training set
    data_path = _gpytorch_dir.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = np.load(data_path)
    dtype = torch.float32
    device = next(model.parameters()).device

    X_all = torch.cat([
        torch.tensor(data['images_train'], dtype=dtype),
        torch.tensor(data['images_val'], dtype=dtype),
    ], dim=0).reshape(-1, config['n_px_side'] ** 2).to(device)

    X_train = X_all[indices_train]
    pool_indices = sorted(set(range(X_all.shape[0])) - set(indices_train.cpu().numpy().tolist()))
    X_pool = X_all[pool_indices]

    print(f"\nReady. Pool: {X_pool.shape[0]} images, Train: {X_train.shape[0]} images, "
          f"A={likelihood.A.item():.4f}, lambda0={likelihood.lambda0.item():.4f}")

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
    """K(x1, x2) -- scalar kernel value between two images."""
    with torch.no_grad():
        return model.covar_module(
            x1.unsqueeze(0), x2.unsqueeze(0)
        ).evaluate().squeeze().item()


def kernel_norm(model, x):
    """sqrt(K(x, x)) -- the kernel self-value square root.

    For the arc-sine kernel, K(x,x) = (2/pi)*arcsin(v/(1+v)) where
    v = x^T C x + sigma_0^2. This saturates at 1 for large v, so
    kernel_norm is bounded by 1.
    """
    return math.sqrt(kernel_value(model, x, x))


def kernel_angle(model, x1, x2):
    """Angle between x1 and x2 in kernel space (radians).

    Computes arccos(K(x1,x2) / sqrt(K(x1,x1)*K(x2,x2))).
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
    print(f"  sqrt(K(x,x)) = {norm:.4f}")
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

    print(f"\n  sqrt(K(x_cond,x_cond)) = {norm_cond:.4f}")
    print(f"\n  {'rank':>4}  {'image':>10}  {'U_DA':>10}  {'H_marg':>10}  "
          f"{'H_cond':>10}  {'sqrt(Kxx)':>9}  {'ratio':>6}  {'angle(rad)':>10}  {'angle(deg)':>10}")
    print("  " + "-" * 98)

    for rank, idx in enumerate(sorted_idx):
        idx_val = idx.item()
        label = f"train[{idx_val}]" if idx_val < n_train else "x_cond"
        ratio = norms[idx_val] / norm_cond
        print(f"  {rank+1:>4}  {label:>10}  {utilities[idx_val].item():10.6f}  "
              f"{h_margs[idx_val].item():10.4f}  {h_conds[idx_val].item():10.4f}  "
              f"{norms[idx_val]:9.4f}  {ratio:6.2f}  "
              f"{angles[idx_val]:10.4f}  {math.degrees(angles[idx_val]):10.1f}")

    print(f"\n  Total images evaluated: {x_candidates.shape[0]} "
          f"({n_train} training + x_cond)")


# ============================================================================
# DA utility landscape plot
# ============================================================================

def plot_da_landscape(model, likelihood, X_candidates, x_cond, save_path,
                      n_train=None, n_grid_mu=200, n_grid_sigma2=150):
    """Plot entropy heatmap with marginal/conditional scatter for DA utility.

    Single panel. Background is H(mu_g, sigma2_g). Each image x* appears
    twice: at its marginal moments (before conditioning) and at its
    conditional moments (after conditioning on x_cond). Pairs are
    connected by faint lines.

    Args:
        model, likelihood: trained model in eval mode.
        X_candidates: (N, d) images to evaluate. Expected order:
            [training images, x_cond, pool images (optional)]
        x_cond: (d,) conditioning image.
        save_path: where to save the PNG.
        n_train: Number of training images. If provided, images beyond
            n_train+1 are assumed to be pool images (plotted with squares).
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

    # Identify image groups for left panel
    if n_train is None:
        idx_cond = n_cand - 1
        idx_train = list(range(idx_cond))
        idx_pool = []
    else:
        idx_train = list(range(n_train))
        idx_cond = n_train
        idx_pool = list(range(n_train + 1, n_cand))

    # Connecting lines (draw first, behind dots)
    for i in range(n_cand):
        ax1.plot(
            [s2_g_marg[i], s2_g_cond[i]],
            [mu_g_marg[i], mu_g_cond[i]],
            color='gray', linewidth=0.5, alpha=0.4, zorder=3,
        )

    # Marginal dots - training (circles)
    ax1.scatter(
        s2_g_marg[idx_train], mu_g_marg[idx_train],
        s=25, c='tab:blue', edgecolors='white', linewidths=0.3,
        zorder=5, label='marginal',
    )

    # Marginal dots - pool (squares)
    if idx_pool:
        ax1.scatter(
            s2_g_marg[idx_pool], mu_g_marg[idx_pool],
            s=30, marker='s', c='tab:blue', edgecolors='white', linewidths=0.3,
            zorder=5,
        )

    # Conditional dots - training (circles)
    ax1.scatter(
        s2_g_cond[idx_train], mu_g_cond[idx_train],
        s=25, c='tab:orange', edgecolors='white', linewidths=0.3,
        zorder=5, label='conditional',
    )

    # Conditional dots - pool (squares)
    if idx_pool:
        ax1.scatter(
            s2_g_cond[idx_pool], mu_g_cond[idx_pool],
            s=30, marker='s', c='tab:orange', edgecolors='white', linewidths=0.3,
            zorder=5,
        )

    # x_cond marker (marginal position)
    ax1.scatter(
        [s2_g_marg[idx_cond]], [mu_g_marg[idx_cond]],
        s=80, marker='*', c='red', edgecolors='white', linewidths=0.5,
        zorder=6, label='$x_{cond}$',
    )

    ax1.set_xlabel('$\\sigma^2_g$ (log-firing rate variance)', fontsize=10)
    ax1.set_ylabel('$\\mu_g$ (log-firing rate mean)', fontsize=10)
    ax1.set_title('H(R) Landscape: conditioning shifts from $x_{cond}$', fontsize=11)
    ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)

    # === Right panel: U_DA vs norm, colored by angle ===

    # Plot training images (circles)
    sc = ax2.scatter(
        norms[idx_train], U_DA_vals[idx_train],
        s=30, c=angles_rad[idx_train], cmap='plasma',
        vmin=0, vmax=np.pi / 2,
        edgecolors='white', linewidths=0.3, zorder=3,
        label='train' if idx_pool else None,
    )

    # Plot pool images (squares) if present
    if idx_pool:
        ax2.scatter(
            norms[idx_pool], U_DA_vals[idx_pool],
            s=40, marker='s', c=angles_rad[idx_pool], cmap='plasma',
            vmin=0, vmax=np.pi / 2,
            edgecolors='white', linewidths=0.3, zorder=3,
            label='pool',
        )

    # Plot conditioning image (red star)
    ax2.scatter(
        norms[idx_cond], U_DA_vals[idx_cond],
        s=100, marker='*', c='red', edgecolors='white', linewidths=0.5,
        zorder=5, label='$x_{cond}$',
    )

    cbar2 = plt.colorbar(sc, ax=ax2, pad=0.02)
    cbar2.set_label('Angle to $x_{cond}$ (rad)', fontsize=10)

    ax2.set_xlabel('$\\sqrt{K(x^*,x^*)}$', fontsize=10)
    ax2.set_ylabel('$H_{marg} - H_{cond}$', fontsize=10)
    ax2.set_title('DA utility vs kernel self-value (color = angle)', fontsize=11)
    ax2.legend(fontsize=9, loc='best')

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

    if SIGMA_0 is not None:
        trained_sigma0 = model.covar_module.sigma_0.item()
        model.covar_module.sigma_0 = SIGMA_0
        print(f"\n** sigma_0 changed: {trained_sigma0:.4f} -> {SIGMA_0} (post-training override)")

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
    print("COMPARISON: Amplitude vs Angle effects (Arc-Sine Kernel)")
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

    print(f"\n  {'Image':>25}  {'sqrt(Kxx)':>9}  {'angle':>8}  {'U_std':>10}  {'U_DA':>10}")
    print("  " + "-" * 70)

    for i, (label, x) in enumerate(cases):
        norm = kernel_norm(model, x)
        angle = kernel_angle(model, x, x_target)
        u_std = std_result['utility'][i].item()
        u_da = da_result['utility'][i].item()

        print(f"  {label:>25}  {norm:9.4f}  {angle:8.4f}  {u_std:10.6f}  {u_da:10.6f}")

    print(f"""
  Key observations (arc-sine kernel):
  - K(x,x) saturates at 1 for large inputs (erf activation).
  - Scaling UP (same direction): sqrt(K(x,x)) should increase but saturate,
    so utility should also saturate rather than diverge.
  - Compare with arc-cosine: sqrt(K(x,x)) grows unboundedly with scaling,
    causing utility to diverge.
  - Different direction: angle is large, conditioning is weak,
    DA utility is small regardless of amplitude.
""")

    # === Part 6: DA utility conditioned on one image ===
    eval_da_conditioned(model, likelihood, X_train, x_target)

    # === Part 7: DA utility landscape + scatter plot ===
    # Add 20 random pool images (not in training) - plotted with square markers
    n_pool_samples = 20
    pool_indices = torch.randperm(X_pool.shape[0])[:n_pool_samples]
    X_pool_sample = X_pool[pool_indices]

    print(f"\nAdding {n_pool_samples} random pool images for landscape plot")
    print(f"  (plotted with square markers to distinguish from training)")

    # Concatenate: [training images, x_cond, pool samples]
    x_candidates = torch.cat([X_train, x_target.unsqueeze(0), X_pool_sample], dim=0)
    plot_da_landscape(
        model, likelihood, x_candidates, x_target,
        n_train=X_train.shape[0],  # Tell plot where training images end
        save_path=_script_dir / 'explore_utility_arcsine.png',
    )


if __name__ == '__main__':
    demo()
