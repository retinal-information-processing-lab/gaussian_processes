"""
Gradient ascent investigation for DA utility across kernel types.

Validates that DA utility peaks at the conditioning target: when we observe
image A, the query x* = A should have the highest DA utility. Specifically:
1. Interpolation: U_DA increases monotonically from perturbed to A
2. Gradient ascent: starting from perturbed, maximizes U_DA(x | observe A)

Supports --kernel-type {arc_cosine, arc_sine, rbf}. Consolidates the
per-kernel gradient scripts (gradient_unnormalized.py, gradient_arcsine.py,
gradient_rbf.py) into a single investigation script.

Model training via setup() from explore_utility.py (same folder).
All model params from default_params.json via build_config_from_defaults().

Usage:
    # Arc-cosine (default):
    python investigations/utility/gradient.py

    # Arc-sine:
    python investigations/utility/gradient.py --kernel-type arc_sine

    # RBF:
    python investigations/utility/gradient.py --kernel-type rbf
"""

import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

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
compute_H = _local_utils.compute_H

# Import from acquisition.py via importlib (same pattern)
_acquisition_path = _gpytorch_dir / 'acquisition.py'
_spec_acq = importlib.util.spec_from_file_location("gpytorch_porting_acquisition", str(_acquisition_path))
_acquisition = importlib.util.module_from_spec(_spec_acq)
_spec_acq.loader.exec_module(_acquisition)

distribution_aware_utility = _acquisition.distribution_aware_utility

# Import from explore_utility.py in same folder
from explore_utility import setup

# ---------------------------------------------------------------------------
# Investigation-specific constants (not model parameters)
# These are LBFGS optimization tuning knobs, not experiment parameters.
# ---------------------------------------------------------------------------
N_STEPS = 50             # outer LBFGS steps
LR = 0.5                 # LBFGS step size
LBFGS_MAX_ITER = 20      # max iterations per LBFGS step (line search evals)
LBFGS_MAX_EVAL = 25      # max function evaluations per LBFGS step
LBFGS_HISTORY_SIZE = 10  # number of past gradients for Hessian approximation
LOG_EVERY = 1            # print every step

# --- Image creation ---
USE_SYNTHETIC = False    # True: bipartite target + noise start. False: natural + smoothing.
TARGET_INDEX = 0         # pool image index for natural target
SIGMA_SMOOTH = 1.0       # Gaussian smoothing sigma for natural target perturbation
DARK_GRAY = -0.5         # synthetic bipartite: one half pixel value
LIGHT_GRAY = 0.5         # synthetic bipartite: other half pixel value
NOISE_AMP = 0.5          # synthetic: random noise amplitude for starting image

# --- Pixel bounds ---
# 'none': unconstrained pixel optimization
# 'dataset': sigmoid bounds at per-pixel min/max from all images within RF
BOUNDS_MODE = 'none'

N_INTERP = 21            # interpolation points along path


# ============================================================================
# Helper functions
# ============================================================================

def _reconstruct_image(x_rf, rf_mask, n_pixels, dtype, device):
    """Place RF pixel values into full image, zeros elsewhere.

    MUST NOT be called under torch.no_grad() during optimization --
    the gradient from DA utility flows through x_rf into the optimized variable.
    """
    x_full = torch.zeros(n_pixels, dtype=dtype, device=device)
    x_full[rf_mask] = x_rf
    return x_full


def rf_pearson_r(x, target, rf_mask):
    """Pearson correlation between x and target within RF mask."""
    a = x[rf_mask]
    b = target[rf_mask]
    a_c = a - a.mean()
    b_c = b - b.mean()
    num = (a_c * b_c).sum()
    denom = a_c.norm() * b_c.norm()
    if denom < 1e-12:
        return 0.0
    return (num / denom).item()


def rf_proj_coeff(x, target, rf_mask):
    """Projection coefficient: scalar s minimizing ||x - s*target|| within RF.

    s = dot(x, target) / dot(target, target).
    s=1 means same amplitude, s>1 means amplified, s<0 means inverted.
    """
    a = x[rf_mask]
    b = target[rf_mask]
    denom = (b * b).sum()
    if denom < 1e-12:
        return 0.0
    return ((a * b).sum() / denom).item()


# ============================================================================
# Interpolation sweep
# ============================================================================

def interpolation_sweep(model, likelihood, x_target, x_perturbed, n_points, r_max):
    """Compute U_DA along the line from x_perturbed (t=0) to x_target (t=1).

    Uses sample_lambda=False (deterministic mean, no MC sampling).

    Returns:
        ts: (n_points,) numpy array of interpolation parameters
        utilities: (n_points,) numpy array of U_DA values
    """
    ts = np.linspace(0, 1, n_points)
    utilities = []

    with torch.no_grad():
        for t in ts:
            x_t = (1 - t) * x_perturbed + t * x_target
            result = distribution_aware_utility(
                model, likelihood,
                x_t.unsqueeze(0),
                x_target.unsqueeze(0),
                r_max=r_max,
                adaptive_r_max=False,
                sample_lambda=False,
            )
            u = result['utility'].item()
            utilities.append(u)
            print(f"  t={t:.2f}: U_DA={u:.6f}")

    return ts, np.array(utilities)


# ============================================================================
# Gradient ascent (LBFGS)
# ============================================================================

def gradient_ascent(model, likelihood, x_start, x_target, rf_mask, r_max, f_max,
                    pixel_lo, pixel_hi,
                    n_steps, lr, max_iter, max_eval, history_size):
    """LBFGS gradient ascent maximizing U_DA(x | observe A).

    Optimizes RF-masked pixels only (~2,480 out of 11,664). Non-RF pixels
    are always zero. Uses torch.optim.LBFGS with strong_wolfe line search.

    If pixel_lo/pixel_hi are provided, uses sigmoid reparametrization:
    the optimizer works in unconstrained z-space, mapped to pixel space
    via x_rf = lo + (hi - lo) * sigmoid(z). Otherwise optimizes x_rf directly.

    Firing rate guard: if predicted firing rate exceeds f_max, closure returns
    +inf loss so the line search rejects that step.

    Tracks two convergence metrics (RF-masked):
      - pearson_r: structural similarity (1.0 = perfect pattern match)
      - proj_coeff: amplitude along target direction (1.0 = same scale)

    Returns:
        x_final: (n_pixels,) optimized image (zeros outside RF)
        history: dict with lists of per-step metrics
    """
    n_pixels = x_start.shape[0]
    device = x_start.device
    dtype = x_start.dtype

    # Extract RF pixels from starting image
    x_rf_init = x_start[rf_mask]  # (n_rf,)

    # Setup optimization variable: sigmoid reparametrization or direct
    use_sigmoid = pixel_lo is not None and pixel_hi is not None
    if use_sigmoid:
        # Inverse sigmoid (logit) to initialize z in unconstrained space
        eps = 1e-6
        x_clamped = x_rf_init.clamp(pixel_lo + eps, pixel_hi - eps)
        sigmoid_val = (x_clamped - pixel_lo) / (pixel_hi - pixel_lo)
        z_rf = torch.log(sigmoid_val / (1 - sigmoid_val))  # logit
        z_rf = z_rf.detach().requires_grad_(True)
        opt_var = z_rf

        def _to_pixel(z):
            # clamp handles float32 precision: sigmoid(large z) = 1.0 exactly,
            # but lo + (hi - lo) * 1.0f can overshoot hi by ~1 ULP
            return (pixel_lo + (pixel_hi - pixel_lo) * torch.sigmoid(z)).clamp(pixel_lo, pixel_hi)
    else:
        x_rf = x_rf_init.clone().detach().requires_grad_(True)
        opt_var = x_rf

        def _to_pixel(z):
            return z

    optimizer = torch.optim.LBFGS(
        [opt_var],
        lr=lr,
        max_iter=max_iter,
        max_eval=max_eval,
        history_size=history_size,
        line_search_fn='strong_wolfe',
    )

    history = {
        'step': [], 'utility': [], 'grad_norm': [],
        'pearson_r': [], 'proj_coeff': [],
    }

    for step in range(n_steps):
        def closure():
            optimizer.zero_grad()
            x_rf_pixels = _to_pixel(opt_var)
            x_full = _reconstruct_image(x_rf_pixels, rf_mask, n_pixels, dtype, device)
            result = distribution_aware_utility(
                model, likelihood,
                x_full.unsqueeze(0),
                x_target.unsqueeze(0),
                r_max=r_max,
                adaptive_r_max=False,
                sample_lambda=False,
            )
            # Firing rate guard: reject if predicted rate exceeds f_max
            mu_g = result['mu_g_marg']
            if torch.exp(mu_g).item() > f_max:
                return torch.tensor(float('inf'), device=device)
            loss = -result['utility'].squeeze()  # minimize negative = maximize
            loss.backward()
            return loss

        optimizer.step(closure)

        # Verify gradient flow on first step
        if step == 0:
            assert opt_var.grad is not None and opt_var.grad.norm() > 0, \
                "No gradient flow through RF reconstruction"

        # Track metrics AFTER optimizer.step() (no grad needed for logging)
        with torch.no_grad():
            x_rf_pixels = _to_pixel(opt_var)
            x_full = _reconstruct_image(x_rf_pixels, rf_mask, n_pixels, dtype, device)
            result = distribution_aware_utility(
                model, likelihood,
                x_full.unsqueeze(0),
                x_target.unsqueeze(0),
                r_max=r_max,
                adaptive_r_max=False,
                sample_lambda=False,
            )
            utility = result['utility'].item()
            pr = rf_pearson_r(x_full, x_target, rf_mask)
            pc = rf_proj_coeff(x_full, x_target, rf_mask)

        grad_norm = opt_var.grad.norm().item() if opt_var.grad is not None else 0.0

        history['step'].append(step)
        history['utility'].append(utility)
        history['grad_norm'].append(grad_norm)
        history['pearson_r'].append(pr)
        history['proj_coeff'].append(pc)

        # NaN check
        if np.isnan(utility) or np.isnan(grad_norm):
            print(f"  step {step}: NaN detected - stopping")
            break

        if step % LOG_EVERY == 0 or step == n_steps - 1:
            print(f"  step {step:4d}: U={utility:.6f}  "
                  f"r={pr:.4f}  proj={pc:.4f}  |grad|={grad_norm:.4e}")

    # Return full reconstructed image
    with torch.no_grad():
        x_rf_final = _to_pixel(opt_var)
        x_final = _reconstruct_image(x_rf_final, rf_mask, n_pixels, dtype, device)
    return x_final.detach(), history


# ============================================================================
# Visualization
# ============================================================================

def plot_results(x_target, x_perturbed, x_final,
                 ts, utilities_interp, history,
                 rf_mask, kernel, config,
                 vmin, vmax, test_r, reliability,
                 start_label, kernel_type, n_px_side, out_path):
    """Generate the gradient ascent summary figure.

    Top row: target, start, final, difference images (RF-cropped) with RF overlay.
    Bottom row: interpolation sweep + gradient ascent convergence (U_DA + Pearson r + proj_coeff).
    """
    # RF mask in 2D + bounding box for cropping
    mask_2d = rf_mask.cpu().numpy().reshape(n_px_side, n_px_side)
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    r_min, r_max_px = np.where(rows)[0][[0, -1]]
    c_min, c_max_px = np.where(cols)[0][[0, -1]]
    # Add 1-pixel padding
    r_min = max(0, r_min - 1)
    r_max_px = min(n_px_side - 1, r_max_px + 1)
    c_min = max(0, c_min - 1)
    c_max_px = min(n_px_side - 1, c_max_px + 1)

    def masked_crop(x_flat, gray_val):
        """Reshape to 2D, gray out non-RF pixels, crop to RF bounding box."""
        img = x_flat.detach().cpu().numpy().reshape(n_px_side, n_px_side).copy()
        img[~mask_2d] = gray_val
        return img[r_min:r_max_px+1, c_min:c_max_px+1]

    gray_val = (vmin + vmax) / 2

    img_target = masked_crop(x_target, gray_val)
    img_perturbed = masked_crop(x_perturbed, gray_val)
    img_final = masked_crop(x_final, gray_val)

    # Check if images exceed the dataset pixel range
    rf_mask_np = rf_mask.cpu().numpy()
    perturbed_rf_vals = x_perturbed.cpu().numpy()[rf_mask_np]
    final_rf_vals = x_final.cpu().numpy()[rf_mask_np]
    clip_warnings = {}
    for label, vals in [('Start', perturbed_rf_vals), ('Final', final_rf_vals)]:
        below = vals[vals < vmin]
        above = vals[vals > vmax]
        if len(below) > 0 or len(above) > 0:
            clip_warnings[label] = {
                'n_below': len(below), 'min_val': float(vals.min()),
                'n_above': len(above), 'max_val': float(vals.max()),
            }
            print(f"  WARNING: {label} exceeds dataset range [{vmin:.3f}, {vmax:.3f}]: "
                  f"min={float(vals.min()):.3f}, max={float(vals.max()):.3f} "
                  f"({len(below)} px below, {len(above)} px above)")

    # Difference image: final - start
    diff_flat = x_final - x_perturbed
    img_diff = masked_crop(diff_flat, 0.0)

    # RF center and width from trained kernel
    eps_0x = kernel.eps_0x.item()
    eps_0y = kernel.eps_0y.item()
    beta_nat = kernel.beta.item()
    sigma_rf = beta_nat * np.sqrt(2)  # RF width in normalized coords
    # Convert normalized [-1, 1] -> pixel coords
    cx_px = (eps_0x + 1) / 2 * (n_px_side - 1)
    cy_px = (eps_0y + 1) / 2 * (n_px_side - 1)
    sigma_px = sigma_rf * (n_px_side - 1) / 2
    # Offset for cropped images
    cx_crop = cx_px - c_min
    cy_crop = cy_px - r_min

    def draw_rf_overlay(ax):
        """Draw RF center + 1/2-sigma circles on a cropped image axis."""
        ax.plot(cx_crop, cy_crop, 'r+', markersize=8, markeredgewidth=1.5)
        circle_1s = plt.Circle((cx_crop, cy_crop), sigma_px, fill=False,
                               color='red', linewidth=1.5, linestyle='-')
        circle_2s = plt.Circle((cx_crop, cy_crop), 2 * sigma_px, fill=False,
                               color='red', linewidth=1, linestyle='--')
        ax.add_patch(circle_1s)
        ax.add_patch(circle_2s)

    print(f"  RF params: eps_0=({eps_0x:.3f}, {eps_0y:.3f}), beta={beta_nat:.4f}, "
          f"sigma_rf={sigma_rf:.4f} norm = {sigma_px:.1f} px")

    fig = plt.figure(figsize=(18, 8))

    # Top row: 4 images (target, start, final, difference) with RF overlays
    images = [img_target, img_perturbed, img_final]
    u_target = utilities_interp[-1]
    u_start = utilities_interp[0]
    u_final = history['utility'][-1]
    if USE_SYNTHETIC:
        titles = [f'Target (bipartite)\nU_DA={u_target:.4f}',
                  f'Start ({start_label})\nU_DA={u_start:.4f}',
                  f'Final (grad ascent)\nU_DA={u_final:.4f}']
    else:
        titles = [f'Target A\nU_DA={u_target:.4f}',
                  f'Start ({start_label})\nU_DA={u_start:.4f}',
                  f'Final (grad ascent)\nU_DA={u_final:.4f}']
    labels = ['Target', 'Start', 'Final']
    for i, (img, title, label) in enumerate(zip(images, titles, labels)):
        ax = fig.add_subplot(2, 4, i + 1)
        im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        draw_rf_overlay(ax)
        if label in clip_warnings:
            w = clip_warnings[label]
            warn_text = f"CLIPPED: [{w['min_val']:.2f}, {w['max_val']:.2f}]"
            ax.text(0.5, 0.02, warn_text, transform=ax.transAxes, fontsize=8,
                    color='red', fontweight='bold', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    # 4th image: difference (final - start)
    ax_diff = fig.add_subplot(2, 4, 4)
    im_diff = ax_diff.imshow(img_diff, cmap='gray', vmin=vmin, vmax=vmax,
                             aspect='equal')
    ax_diff.set_title('Final - Start', fontsize=10)
    ax_diff.axis('off')
    cb_diff = fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)
    cb_diff.ax.tick_params(labelsize=7)
    draw_rf_overlay(ax_diff)

    # Difference stats
    diff_rf = diff_flat[rf_mask].detach().cpu().numpy()
    print(f"  Difference (final - start) within RF:")
    print(f"    mean={diff_rf.mean():.6f}, std={diff_rf.std():.6f}")
    print(f"    min={diff_rf.min():.6f}, max={diff_rf.max():.6f}")
    print(f"    ||diff||={np.linalg.norm(diff_rf):.4f}")

    # Bottom left: Interpolation sweep
    ax1 = fig.add_subplot(2, 4, 5)
    ax1.plot(ts, utilities_interp, 'b.-', markersize=8, linewidth=1.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(1, color='gray', linestyle='--', alpha=0.5)
    ax1.text(0.02, 0.95, 'start', transform=ax1.transAxes, fontsize=9, color='gray')
    ax1.text(0.85, 0.95, 'target', transform=ax1.transAxes, fontsize=9, color='gray')
    ax1.set_xlabel('t (0 = start, 1 = target)')
    ax1.set_ylabel('U_DA')
    ax1.set_title('Interpolation: monotonicity check')
    ax1.grid(True, alpha=0.3)

    # Bottom right: Gradient ascent convergence (U_DA + Pearson r + proj_coeff)
    ax2 = fig.add_subplot(2, 2, 4)
    steps = history['step']
    color_u = 'tab:blue'
    color_r = 'tab:green'
    color_p = 'tab:red'

    ax2.plot(steps, history['utility'], color=color_u, linewidth=1.5, label='U_DA')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('U_DA', color=color_u)
    ax2.tick_params(axis='y', labelcolor=color_u)

    ax2r = ax2.twinx()
    ax2r.plot(steps, history['pearson_r'], color=color_r, linewidth=1.5,
              alpha=0.7, label='Pearson r (RF)')
    ax2r.plot(steps, history['proj_coeff'], color=color_p, linewidth=1.5,
              alpha=0.7, linestyle='--', label='Proj coeff (RF)')
    ax2r.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
    ax2r.set_ylabel('Structure / Amplitude')
    ax2r.tick_params(axis='y')

    # Combine legends from both axes
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=8)

    ax2.set_title('LBFGS gradient ascent convergence')
    ax2.grid(True, alpha=0.3)

    # Kernel name for display
    kernel_display = {'arc_cosine': 'Arc-Cosine', 'arc_sine': 'Arc-Sine', 'rbf': 'RBF'}
    kname = kernel_display.get(kernel_type, kernel_type)
    title_str = (
        f'DA Utility - {kname} Kernel '
        f'(M={config["M"]}, n_train={config["n_train"]}, '
        f'seed={config["seed"]}, cell={config["cell"]})  '
        f'test_r={test_r:.3f}, reliability={reliability:.3f}'
    )
    fig.suptitle(title_str, fontsize=11)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main(kernel_type=None):
    # === Step 1: Train model ===
    ktype_label = kernel_type or 'default'
    print("=" * 70)
    print(f"Step 1: Training model (kernel_type={ktype_label})")
    print("=" * 70)

    env = setup(kernel_type=kernel_type)
    model = env['model']
    likelihood = env['likelihood']
    X_pool = env['X_pool']
    X_train = env['X_train']
    config = env['config']
    kernel_type = env['kernel_type']
    test_r = env['test_r']
    reliability = env['reliability']
    r_max = config['r_max']
    f_max = config['f_max']
    n_px_side = config['n_px_side']

    print(f"\nTest Pearson r: {test_r:.4f}")
    print(f"Reliability:    {reliability:.4f}")

    # === Step 2: Setup target and starting image ===
    print("\n" + "=" * 70)
    print(f"Step 2: Creating images (USE_SYNTHETIC={USE_SYNTHETIC})")
    print("=" * 70)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_pixels = n_px_side ** 2

    # Ensure RF mask is cached by running a forward pass
    kernel = getattr(model, 'covar_module', None) or getattr(model, 'kernel', None)
    if not (hasattr(kernel, '_cached_mask') and kernel._cached_mask is not None):
        with torch.no_grad():
            _ = model(X_pool[0].unsqueeze(0))
    rf_mask = kernel._cached_mask.squeeze()

    # Dataset pixel bounds (RF pixels only) -- for sigmoid reparametrization + visualization
    rf_mask_np = rf_mask.cpu().numpy()
    X_all = torch.cat([X_pool, X_train], dim=0)
    all_rf_vals = X_all.cpu().numpy()[:, rf_mask_np]
    vmin_dataset = float(all_rf_vals.min())
    vmax_dataset = float(all_rf_vals.max())
    print(f"  Dataset RF pixel range: [{vmin_dataset:.3f}, {vmax_dataset:.3f}] "
          f"(from {X_all.shape[0]} images)")

    if USE_SYNTHETIC:
        # Bipartite target: left=dark, right=light within RF mask
        mask_2d = rf_mask.cpu().numpy().reshape(n_px_side, n_px_side)
        _, mask_cols = np.where(mask_2d)
        col_center = (mask_cols.min() + mask_cols.max()) / 2

        target_2d = np.zeros((n_px_side, n_px_side), dtype=np.float32)
        left = mask_2d & (np.arange(n_px_side)[None, :] <= col_center)
        right = mask_2d & (np.arange(n_px_side)[None, :] > col_center)
        target_2d[left] = DARK_GRAY
        target_2d[right] = LIGHT_GRAY

        x_target = torch.tensor(target_2d.reshape(-1), dtype=dtype, device=device)

        # Noise starting image within RF mask
        torch.manual_seed(config['seed'])
        x_perturbed = NOISE_AMP * torch.randn(n_pixels, dtype=dtype, device=device)
        x_perturbed[~rf_mask] = 0.0

        start_label = "noise"
        print(f"Target: synthetic bipartite (dark={DARK_GRAY}, light={LIGHT_GRAY})")
        print(f"Start: random noise (amp={NOISE_AMP})")
    else:
        # Natural image experiment
        x_target = X_pool[TARGET_INDEX]

        x_target_2d = x_target.cpu().numpy().reshape(n_px_side, n_px_side)
        x_smoothed_2d = gaussian_filter(x_target_2d, sigma=SIGMA_SMOOTH)
        x_perturbed = torch.tensor(
            x_smoothed_2d.reshape(-1), dtype=x_target.dtype, device=x_target.device
        )

        start_label = f"smoothed (sigma={SIGMA_SMOOTH})"
        print(f"Target: pool image {TARGET_INDEX}")
        print(f"Start: Gaussian smoothing sigma={SIGMA_SMOOTH}")

    pixel_dist = (x_perturbed - x_target).norm().item()
    rf_dist = (x_perturbed[rf_mask] - x_target[rf_mask]).norm().item()
    print(f"  pixel_dist={pixel_dist:.4f}, rf_dist={rf_dist:.4f}")
    print(f"  ||target||_RF={x_target[rf_mask].norm().item():.2f}, "
          f"||start||_RF={x_perturbed[rf_mask].norm().item():.2f}")

    # Optimization pixel bounds (sigmoid reparametrization)
    if BOUNDS_MODE == 'none':
        pixel_lo, pixel_hi = None, None
        print(f"  Pixel bounds: none (unconstrained)")
    elif BOUNDS_MODE == 'dataset':
        pixel_lo, pixel_hi = vmin_dataset, vmax_dataset
        print(f"  Pixel bounds (dataset): [{pixel_lo:.3f}, {pixel_hi:.3f}]")
    else:
        raise ValueError(f"Unknown BOUNDS_MODE: {BOUNDS_MODE}. Use 'none' or 'dataset'.")

    # === Step 3: Interpolation sweep ===
    print("\n" + "=" * 70)
    print(f"Step 3: Interpolation sweep ({start_label} -> target)")
    print("=" * 70)

    ts, utilities_interp = interpolation_sweep(
        model, likelihood, x_target, x_perturbed, N_INTERP, r_max
    )

    diffs = np.diff(utilities_interp)
    is_monotone = np.all(diffs >= 0)
    print(f"\nMonotone? {is_monotone} (min diff: {diffs.min():.6e})")
    print(f"U_DA(start | target) = {utilities_interp[0]:.6f}")
    print(f"U_DA(target | target) = {utilities_interp[-1]:.6f}")

    # === Step 4: Gradient ascent ===
    print("\n" + "=" * 70)
    print(f"Step 4: Gradient ascent from {start_label}")
    print("=" * 70)

    x_final, history = gradient_ascent(
        model, likelihood, x_perturbed, x_target, rf_mask, r_max, f_max,
        pixel_lo, pixel_hi,
        N_STEPS, LR, LBFGS_MAX_ITER, LBFGS_MAX_EVAL, LBFGS_HISTORY_SIZE
    )

    print(f"\nGradient ascent summary:")
    print(f"  Steps: {len(history['step'])}")
    print(f"  U_DA: {history['utility'][0]:.6f} -> {history['utility'][-1]:.6f}")
    print(f"  Pearson r (RF): {history['pearson_r'][0]:.4f} -> {history['pearson_r'][-1]:.4f}")
    print(f"  Proj coeff (RF): {history['proj_coeff'][0]:.4f} -> {history['proj_coeff'][-1]:.4f}")
    print(f"  ||final||_RF={x_final[rf_mask].norm().item():.2f}")

    # Firing rate diagnostics
    print(f"\n  Firing rate diagnostics (f_max={f_max})")
    A = likelihood.A.squeeze()
    lam0 = likelihood.lambda0.squeeze()
    with torch.no_grad():
        for label, x_img in [('target', x_target), ('start', x_perturbed), ('final', x_final)]:
            mu, sigma2 = get_gp_marginal_moments(model, x_img.unsqueeze(0))
            mu_g = A * mu + lam0
            firing_rate = torch.exp(mu_g).item()
            H = compute_H(mu, sigma2, r_max=r_max, a=A, lambda0=lam0).item()
            flag = " ** EXCEEDS f_max" if firing_rate > f_max else ""
            print(f"    {label:8s}: lambda_m={mu.item():.4f}, lambda_var={sigma2.item():.4f}, "
                  f"mu_g={mu_g.item():.4f}, firing_rate={firing_rate:.2f}, "
                  f"H_marg={H:.6f}, ||x||_RF={x_img[rf_mask].norm().item():.2f}{flag}")

    # === Step 5: Visualization ===
    print("\n" + "=" * 70)
    print("Step 5: Visualization")
    print("=" * 70)

    plot_results(
        x_target, x_perturbed, x_final,
        ts, utilities_interp, history,
        rf_mask, kernel, config,
        vmin_dataset, vmax_dataset, test_r, reliability,
        start_label, kernel_type, n_px_side,
        out_path=_script_dir / f'gradient_{kernel_type}.png',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DA utility gradient ascent investigation')
    parser.add_argument('--kernel-type', type=str, default=None,
                        choices=['arc_cosine', 'arc_sine', 'rbf'],
                        help='Kernel type (default: from default_params.json)')
    args = parser.parse_args()
    main(kernel_type=args.kernel_type)
