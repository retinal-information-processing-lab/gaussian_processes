"""
Entropy landscape H(R | mu_g, sigma2_g) with calibrated truncation boundaries.
Created by Claude.

Computes the entropy heatmap using adaptive r_max (up to 10000), then
for each milestone r_max computes sum(p_r) on the full grid and draws
the contour where sum(p_r) = 0.99 (empirical truncation boundary).
Also overlays the analytical 3-sigma boundary for comparison.

See entropy_landscape.md for findings and limitations.

Usage:
    python investigations/utility/entropy_landscape.py
"""

import sys
import math
import time
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Path setup — import local _diff_laplace_log_probs from utils.py
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_gpytorch_dir = _script_dir.parent.parent

import importlib.util
_local_utils_path = _gpytorch_dir / 'utils.py'
_spec = importlib.util.spec_from_file_location("gpytorch_porting_utils", str(_local_utils_path))
_local_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_utils)

_diff_laplace_log_probs = _local_utils._diff_laplace_log_probs
compute_H_MC = _local_utils.compute_H_MC

# ---------------------------------------------------------------------------
# Grid parameters (in GP moment / lambda space)
# ---------------------------------------------------------------------------
# GP moment ranges (lambda space)
MU_LAMBDA_MIN = -6.0      # Covers g-space [-5, 12] with lambda0=1
MU_LAMBDA_MAX = 11.0
SIGMA2_LAMBDA_MIN = 0.001
SIGMA2_LAMBDA_MAX = 15.0
N_MU = 400
N_SIGMA2 = 300

# Transformation parameters (for normalized case: A=1, lambda0=1)
A_PLOT = 1.0
LAMBDA0_PLOT = 1.0

# Adaptive computation limits (for the reference entropy heatmap)
MAX_PRACTICAL_RMAX = 10000
MIN_RMAX = 200
SAFETY_K = 3.0

# Milestone r_max values to evaluate
MILESTONE_RMAX = [100, 500, 1000, 10000]

# Relative H error threshold for the empirical boundary

H_ERROR_THRESHOLD = 0.01  # 1%

# Choose which H to plot in Panel 1 heatmap:
#   'adaptive' - use adaptive r_max reference (default)
#   100, 500, 1000, 10000 - use specific fixed r_max -> nb must be int not stirng
PLOT_HEATMAP_MODE = 'adaptive'


def compute_H_grid_fixed_rmax(mu_range, sigma2_range, rmax, device, batch_rows=10):
    """Compute H at every grid point using a fixed r_max.

    Returns:
        H: (N_MU, N_SIGMA2) numpy array
    """
    N_mu = len(mu_range)
    N_s2 = len(sigma2_range)
    H = np.full((N_mu, N_s2), np.nan, dtype=np.float32)

    r = torch.arange(0, rmax, dtype=torch.float32, device=device)
    sigma2_dev = sigma2_range.to(device)

    for start in range(0, N_mu, batch_rows):
        end = min(start + batch_rows, N_mu)
        batch_size = end - start

        mu_flat = mu_range[start:end].repeat_interleave(N_s2).to(device)
        s2_flat = sigma2_dev.repeat(batch_size)

        p_r, log_p_r = _diff_laplace_log_probs(mu_flat, s2_flat, r)
        H_flat = -torch.sum(p_r * log_p_r, dim=1).cpu().numpy()
        H[start:end] = H_flat.reshape(batch_size, N_s2)

    return H


def compute_entropy_row_adaptive(mu_g_val, sigma2_vals, device):
    """Compute entropy for one row with adaptive r_max."""
    N = len(sigma2_vals)
    H = torch.full((N,), float('nan'), dtype=torch.float32)

    max_logf_threshold = math.log(MAX_PRACTICAL_RMAX)
    computable_mask = (mu_g_val + SAFETY_K * torch.sqrt(sigma2_vals)) < max_logf_threshold

    if not computable_mask.any():
        return H

    sigma2_sub = sigma2_vals[computable_mask]

    upper_logf = mu_g_val + SAFETY_K * math.sqrt(sigma2_sub.max().item())
    if upper_logf > 20:
        needed = MAX_PRACTICAL_RMAX
    else:
        upper_rate = math.exp(upper_logf)
        needed = int(upper_rate + 5 * math.sqrt(max(upper_rate, 1))) + 10
    needed = max(min(needed, MAX_PRACTICAL_RMAX), MIN_RMAX)

    r = torch.arange(0, needed, dtype=torch.float32, device=device)
    n_sub = sigma2_sub.shape[0]
    mu_vec = torch.full((n_sub,), mu_g_val, dtype=torch.float32, device=device)

    p_r, log_p_r = _diff_laplace_log_probs(mu_vec, sigma2_sub.to(device), r)
    H_sub = -torch.sum(p_r * log_p_r, dim=1)
    H[computable_mask] = H_sub.cpu()
    return H


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create grid in lambda space (plot coordinates)
    mu_lambda_range = torch.linspace(MU_LAMBDA_MIN, MU_LAMBDA_MAX, N_MU)
    sigma2_lambda_range = torch.linspace(SIGMA2_LAMBDA_MIN, SIGMA2_LAMBDA_MAX, N_SIGMA2)
    mu_lambda_np = mu_lambda_range.numpy()
    sigma2_lambda_np = sigma2_lambda_range.numpy()

    # Transform to g-space (computation coordinates)
    mu_g_range = A_PLOT * mu_lambda_range + LAMBDA0_PLOT
    sigma2_g_range = (A_PLOT ** 2) * sigma2_lambda_range

    print(f"Grid: mu_lambda in [{MU_LAMBDA_MIN}, {MU_LAMBDA_MAX}] ({N_MU} pts)")
    print(f"      -> mu_g in [{mu_g_range[0].item():.1f}, {mu_g_range[-1].item():.1f}]")
    print(f"      sigma2_lambda in [{SIGMA2_LAMBDA_MIN}, {SIGMA2_LAMBDA_MAX}] ({N_SIGMA2} pts)")
    print(f"      -> sigma2_g in [{sigma2_g_range[0].item():.3f}, {sigma2_g_range[-1].item():.1f}]")

    # =========================================================================
    # 1. Compute reference entropy (adaptive r_max)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 1: Computing reference entropy (adaptive r_max)")
    print("=" * 60)

    H_ref = torch.full((N_MU, N_SIGMA2), float('nan'))
    t0 = time.time()
    for i in range(N_MU):
        # Use g-space values for computation
        H_ref[i] = compute_entropy_row_adaptive(mu_g_range[i].item(), sigma2_g_range, device)
        if (i + 1) % 100 == 0:
            print(f"  Row {i+1}/{N_MU} | elapsed: {time.time()-t0:.1f}s")
    H_ref_np = H_ref.numpy()
    print(f"  Done in {time.time()-t0:.1f}s")

    # =========================================================================
    # 2. For each milestone r_max, compute H and relative error vs reference
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Computing H with each fixed r_max")
    print(f"Error threshold: {H_ERROR_THRESHOLD*100:.0f}%")
    print("=" * 60)

    rel_error_grids = {}
    H_fixed_grids = {}  # Store H computed with fixed r_max (for optional plotting)
    for rmax in MILESTONE_RMAX:
        print(f'rmax = {rmax} | ', end='', flush=True)
        t0 = time.time()
        max_batch = max(1, int(2e9 / (N_SIGMA2 * rmax * 4)))
        batch_rows = min(max_batch, 50)

        print(f"\n  r_max = {rmax}: ", end='', flush=True)
        # Use g-space values for computation
        H_fixed = compute_H_grid_fixed_rmax(mu_g_range, sigma2_g_range, rmax, device,
                                            batch_rows=batch_rows)
        elapsed = time.time() - t0

        # Relative error: |H_fixed - H_ref| / H_ref
        # Only where H_ref is valid and > 0
        rel_err = np.full_like(H_ref_np, np.nan)
        valid = (~np.isnan(H_ref_np)) & (H_ref_np > 0.01)
        rel_err[valid] = np.abs(H_fixed[valid] - H_ref_np[valid]) / H_ref_np[valid]
        rel_error_grids[rmax] = rel_err
        H_fixed_grids[rmax] = H_fixed  # Save for optional plotting
        print(f'H_fixed_grids key and vals: {H_fixed_grids.keys()} and {[H_fixed_grids[k].shape for k in H_fixed_grids.keys()]}')
        # Stats
        err_valid = rel_err[valid]
        frac_bad = np.mean(err_valid > H_ERROR_THRESHOLD) * 100
        print(f"{elapsed:.1f}s | "
              f"points with >{H_ERROR_THRESHOLD*100:.0f}% error: {frac_bad:.1f}% "
              f"(of {valid.sum()} reference points)")

    # =========================================================================
    # 3. Print comparison: analytical vs empirical boundaries
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Analytical vs empirical boundary comparison")
    print("=" * 60)

    for rmax in MILESTONE_RMAX:
        rel_err = rel_error_grids[rmax]
        empirical_mu_lambda = []
        analytical_mu_lambda = []
        for j in range(N_SIGMA2):
            s2_lambda_val = sigma2_lambda_range[j].item()
            col = rel_err[:, j]
            # Find last mu where error is below threshold
            good = np.where((~np.isnan(col)) & (col <= H_ERROR_THRESHOLD))[0]
            if len(good) > 0:
                # Get g-space value and transform back to lambda-space
                mu_g_boundary = mu_g_range[good[-1]].item()
                mu_lambda_boundary = (mu_g_boundary - LAMBDA0_PLOT) / A_PLOT
                empirical_mu_lambda.append(mu_lambda_boundary)
            else:
                empirical_mu_lambda.append(MU_LAMBDA_MIN)

            # Transform analytical formula to lambda-space
            # mu_g = log(rmax) - 3*sqrt(sigma2_g)
            # A*mu_lambda + lambda0 = log(rmax) - 3*A*sqrt(sigma2_lambda)
            # mu_lambda = [log(rmax) - lambda0 - 3*A*sqrt(sigma2_lambda)] / A
            analytical_mu_lambda.append(
                (math.log(rmax) - LAMBDA0_PLOT
                 - SAFETY_K * A_PLOT * math.sqrt(s2_lambda_val)) / A_PLOT
            )

        empirical_mu_lambda = np.array(empirical_mu_lambda)
        analytical_mu_lambda = np.array(analytical_mu_lambda)
        diff = empirical_mu_lambda - analytical_mu_lambda
        valid = (empirical_mu_lambda > MU_LAMBDA_MIN + 0.5) & (analytical_mu_lambda > MU_LAMBDA_MIN)
        if valid.any():
            print(f"\n  r_max = {rmax}:")
            print(f"    Empirical (1% H error): "
                  f"[{empirical_mu_lambda[valid].min():.2f}, {empirical_mu_lambda[valid].max():.2f}]")
            print(f"    Analytical (3-sigma):    "
                  f"[{analytical_mu_lambda[valid].min():.2f}, {analytical_mu_lambda[valid].max():.2f}]")
            print(f"    Offset (empirical - analytical): "
                  f"mean={diff[valid].mean():+.2f}, max|diff|={np.abs(diff[valid]).max():.2f}")

    # =========================================================================
    # 4. Compute H with Monte Carlo (for Panel 2) + clip_fraction (affidability)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Computing H with Monte Carlo + affidability metric")
    print("=" * 60)

    H_mc = torch.full((N_MU, N_SIGMA2), float('nan'))
    clip_frac = torch.full((N_MU, N_SIGMA2), float('nan'))
    t0 = time.time()
    for i in range(N_MU):
        # Use g-space values for computation
        mu_val_g = mu_g_range[i].item()
        mu_vec = torch.full((N_SIGMA2,), mu_val_g, dtype=torch.float32, device=device)
        H_row, clip_row = compute_H_MC(mu_vec, sigma2_g_range.to(device), n_samples=2000,
                                       a=1.0, lambda0=0.0, max_log_contrib=50.0,
                                       return_clip_fraction=True)
        H_mc[i] = H_row
        clip_frac[i] = clip_row
        if (i + 1) % 100 == 0:
            print(f"  Row {i+1}/{N_MU} | elapsed: {time.time()-t0:.1f}s")
    H_mc_np = H_mc.numpy()
    clip_frac_np = clip_frac.numpy()
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Clip fraction range: [{np.nanmin(clip_frac_np):.3f}, {np.nanmax(clip_frac_np):.3f}]")
    print(f"  Points with clip_frac > 0.10: {np.sum(clip_frac_np > 0.10)} / {(~np.isnan(clip_frac_np)).sum()}")

    # =========================================================================
    # 5. Plot
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 5: Plotting")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Select which H to plot in Panel 1 based on config
    if PLOT_HEATMAP_MODE == 'adaptive':
        H_panel1 = H_ref_np
        panel1_label = 'Laplace Sum (adaptive r_max + boundaries)'
    else:
        if PLOT_HEATMAP_MODE not in H_fixed_grids:
            raise ValueError(f"PLOT_HEATMAP_MODE={PLOT_HEATMAP_MODE} not in computed grids. "
                           f"Choose from: 'adaptive' or {list(H_fixed_grids.keys())}")
        H_panel1 = H_fixed_grids[PLOT_HEATMAP_MODE]
        panel1_label = f'Laplace Sum (r_max={PLOT_HEATMAP_MODE}) + boundaries'

    # Shared colormap and limits
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='lightgray', alpha=0.5)
    vmin = 0
    vmax = min(np.nanmax(H_panel1), np.nanmax(H_mc_np), 8)

    # -------------------------------------------------------------------------
    # Panel 1: Laplace (chosen mode + boundaries)
    # -------------------------------------------------------------------------
    im1 = ax1.pcolormesh(sigma2_lambda_np, mu_lambda_np, H_panel1, cmap=cmap, shading='auto',
                         vmin=vmin, vmax=vmax)
    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
    cbar1.set_label('Entropy  H(R | $\\mu_\\lambda$, $\\sigma^2_\\lambda$)', fontsize=11)

    # Contours and curves for each milestone r_max (in lambda space)
    sigma2_lambda_smooth = np.linspace(SIGMA2_LAMBDA_MIN, SIGMA2_LAMBDA_MAX, 500)
    sigma2_lambda_mesh, mu_lambda_mesh = np.meshgrid(sigma2_lambda_np, mu_lambda_np)

    curve_styles = [
        (100,   'red',    1.8),
        (500,   'orange', 1.8),
        (1000,  'yellow', 1.8),
        (10000, 'white',  2.5),
    ]

    for rmax, color, lw in curve_styles:
        rel_err = rel_error_grids[rmax]

        # Empirical contour: relative H error = threshold (solid)
        # Mask NaN to avoid contour artifacts
        rel_err_masked = np.where(np.isnan(rel_err), 999.0, rel_err)
        ax1.contour(sigma2_lambda_mesh, mu_lambda_mesh, rel_err_masked,
                    levels=[H_ERROR_THRESHOLD],
                    colors=[color], linestyles=['-'], linewidths=[lw],
                    zorder=6)

        # Analytical 3-sigma curve in lambda-space (thin dotted, same color)
        # Formula: mu_lambda = [log(rmax) - lambda0 - 3*A*sqrt(sigma2_lambda)] / A
        mu_lambda_analytical = ((np.log(rmax) - LAMBDA0_PLOT
                                 - SAFETY_K * A_PLOT * np.sqrt(sigma2_lambda_smooth))
                                / A_PLOT)
        mu_lambda_analytical = np.clip(mu_lambda_analytical, MU_LAMBDA_MIN, MU_LAMBDA_MAX)
        ax1.plot(sigma2_lambda_smooth, mu_lambda_analytical, color=color, linestyle=':',
                 linewidth=1.0, alpha=0.7, zorder=6)

        # Label at the left edge (in lambda-space)
        mu_lambda_at_zero = (math.log(rmax) - LAMBDA0_PLOT) / A_PLOT
        if MU_LAMBDA_MIN <= mu_lambda_at_zero <= MU_LAMBDA_MAX:
            ax1.text(0.3, mu_lambda_at_zero + 0.2, f'$r_{{max}}$={rmax}',
                     color=color, fontsize=9, fontweight='bold',
                     ha='left', va='bottom', zorder=7,
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                               alpha=0.6, edgecolor='none'))

    # Hatched region: beyond adaptive computation (in lambda-space)
    boundary_mu_lambda = np.full(N_SIGMA2, MU_LAMBDA_MAX)
    for j in range(N_SIGMA2):
        col = H_ref_np[:, j]
        valid_idx = np.where(~np.isnan(col))[0]
        if len(valid_idx) > 0 and valid_idx[-1] < N_MU - 1:
            # Transform g-space boundary to lambda-space
            mu_g_boundary = mu_g_range[valid_idx[-1]].item()
            boundary_mu_lambda[j] = (mu_g_boundary - LAMBDA0_PLOT) / A_PLOT
        elif len(valid_idx) == 0:
            boundary_mu_lambda[j] = MU_LAMBDA_MIN
    ax1.fill_between(sigma2_lambda_np, boundary_mu_lambda, MU_LAMBDA_MAX,
                     color='gray', alpha=0.3, hatch='///', zorder=5)

    # Axis labels (lambda-space with transformation note)
    ax1.set_xlabel('$\\sigma^2_\\lambda$ (GP variance)', fontsize=11)
    ax1.set_ylabel('$\\mu_\\lambda$ (GP mean)', fontsize=11)
    ax1.set_title(panel1_label + '\n'
                  '$\\mu_g = \\mu_\\lambda + 1,\\; \\sigma^2_g = \\sigma^2_\\lambda$',
                  fontsize=11)

    # Axis correspondence annotations (expected spike counts at axis limits)
    # Top corner: E[r] at max mu_lambda
    mu_g_max = MU_LAMBDA_MAX + LAMBDA0_PLOT
    E_r_max = np.exp(mu_g_max)
    ax1.text(0.02, 0.98, f'$E[r]\\approx${E_r_max:.0f}',
             transform=ax1.transAxes, fontsize=8, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))

    # Bottom corner: E[r] at min mu_lambda
    mu_g_min = MU_LAMBDA_MIN + LAMBDA0_PLOT
    E_r_min = np.exp(mu_g_min)
    ax1.text(0.02, 0.02, f'$E[r]\\approx${E_r_min:.3f}',
             transform=ax1.transAxes, fontsize=8, va='bottom', ha='left',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))

    # Right corner: Spike count variability at max sigma2_lambda
    # At natural image mean, show std[r] with max variance
    mu_lambda_nat = -2.5
    mu_g_nat = mu_lambda_nat + LAMBDA0_PLOT
    sigma2_g_max = SIGMA2_LAMBDA_MAX
    # Var[r] ≈ exp(mu_g + sigma2_g/2) for Poisson-lognormal
    std_r_max = np.sqrt(np.exp(mu_g_nat + sigma2_g_max / 2))
    ax1.text(0.98, 0.02, f'$\\sigma_r\\approx${std_r_max:.1f}*',
             transform=ax1.transAxes, fontsize=7, va='bottom', ha='right',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
    ax1.text(0.97, 0.06, '*at natural\nimage mean',
             transform=ax1.transAxes, fontsize=6, va='bottom', ha='right',
             style='italic', alpha=0.7)

    # --- Legend (top-right, using available space) ---
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=2, linestyle='-',
               label='Solid: empirical boundary'),
        Line2D([], [], color='none', label=f'  (where $H$ error exceeds '
               f'{H_ERROR_THRESHOLD*100:.0f}% vs'),
        Line2D([], [], color='none',
               label=f'  adaptive $r_{{max}}$ reference)'),
        Line2D([0], [0], color='gray', linewidth=1, linestyle=':',
               alpha=0.7,
               label='Dotted: analytical $3\\sigma$ formula'),
        Line2D([], [], color='none',
               label=f'  ($\\mu_\\lambda = \\log(r_{{max}}) - 1 - 3\\sqrt{{\\sigma^2_\\lambda}}$)'),
        Line2D([], [], color='none', label=''),
        Patch(facecolor='gray', alpha=0.3, hatch='///',
              label='No reference available'),
        Line2D([], [], color='none',
               label=f'  (adaptive $r_{{max}} > {MAX_PRACTICAL_RMAX}$)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=7.5,
               framealpha=0.92, handlelength=2.5, borderpad=0.8,
               labelspacing=0.3)

    ax1.set_xlim(SIGMA2_LAMBDA_MIN, SIGMA2_LAMBDA_MAX)
    ax1.set_ylim(MU_LAMBDA_MIN, MU_LAMBDA_MAX)

    # -------------------------------------------------------------------------
    # Panel 2: Monte Carlo (clipped)
    # -------------------------------------------------------------------------
    im2 = ax2.pcolormesh(sigma2_lambda_np, mu_lambda_np, H_mc_np, cmap=cmap, shading='auto',
                         vmin=vmin, vmax=vmax)
    cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02)
    cbar2.set_label('Entropy  H(R | $\\mu_\\lambda$, $\\sigma^2_\\lambda$)', fontsize=11)

    # Axis labels (lambda-space)
    ax2.set_xlabel('$\\sigma^2_\\lambda$ (GP variance)', fontsize=11)
    ax2.set_ylabel('$\\mu_\\lambda$ (GP mean)', fontsize=11)
    ax2.set_title('Monte Carlo (S=2000, clipped at 50)\n'
                  '$\\mu_g = \\mu_\\lambda + 1,\\; \\sigma^2_g = \\sigma^2_\\lambda$',
                  fontsize=11)

    # Axis correspondence annotations (expected spike counts - same as Panel 1)
    ax2.text(0.02, 0.98, f'$E[r]\\approx${E_r_max:.0f}',
             transform=ax2.transAxes, fontsize=8, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
    ax2.text(0.02, 0.02, f'$E[r]\\approx${E_r_min:.3f}',
             transform=ax2.transAxes, fontsize=8, va='bottom', ha='left',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
    ax2.text(0.98, 0.02, f'$\\sigma_r\\approx${std_r_max:.1f}*',
             transform=ax2.transAxes, fontsize=7, va='bottom', ha='right',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
    ax2.text(0.97, 0.06, '*at natural\nimage mean',
             transform=ax2.transAxes, fontsize=6, va='bottom', ha='right',
             style='italic', alpha=0.7)

    # Overlay affidability contours (clip_fraction thresholds) in lambda-space
    # clip_fraction < 0.05: reliable (green dashed)
    # clip_fraction = 0.10: unreliable boundary (red solid)
    clip_frac_masked = np.where(np.isnan(clip_frac_np), -1.0, clip_frac_np)

    # Reliable threshold (< 5% clipped)
    ax2.contour(sigma2_lambda_mesh, mu_lambda_mesh, clip_frac_masked,
                levels=[0.05], colors=['green'], linestyles=['--'],
                linewidths=[2.0], zorder=7)

    # Unreliable threshold (> 10% clipped)
    ax2.contour(sigma2_lambda_mesh, mu_lambda_mesh, clip_frac_masked,
                levels=[0.10], colors=['red'], linestyles=['-'],
                linewidths=[2.5], zorder=7)

    # Legend for affidability contours
    affid_legend = [
        Line2D([0], [0], color='green', linewidth=2, linestyle='--',
               label='Reliable (clip < 5%)'),
        Line2D([0], [0], color='red', linewidth=2.5, linestyle='-',
               label='Unreliable (clip > 10%)'),
    ]
    ax2.legend(handles=affid_legend, loc='lower right', fontsize=9,
               framealpha=0.9)

    # Clipping annotation
    ax2.text(0.95, 0.95, 'Clipping:\n$-\\log p(r) \\leq 50$\n\n'
                         'Affidability:\nclip fraction',
             transform=ax2.transAxes, fontsize=9, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))

    ax2.set_xlim(SIGMA2_LAMBDA_MIN, SIGMA2_LAMBDA_MAX)
    ax2.set_ylim(MU_LAMBDA_MIN, MU_LAMBDA_MAX)

    # Figure title
    plt.suptitle('Entropy Landscape: GP Moment Space ($\\mu_\\lambda$, $\\sigma^2_\\lambda$)\n'
                 'Laplace Sum vs Monte Carlo',
                 fontsize=13, y=0.98)

    plt.tight_layout()
    save_path = _script_dir / 'entropy_landscape.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
