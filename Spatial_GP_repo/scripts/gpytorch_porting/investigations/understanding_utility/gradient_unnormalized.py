"""
Gradient ascent investigation for DA utility with UNNORMALIZED arc-cosine kernel.
Created by Claude.

Companion to investigations/normalized_kernel/gradient_normalized.py — same
visualization structure and experiment logic, but with the standard ArcCosineKernel.
The goal is to compare gradient ascent behavior between normalized and unnormalized
kernels.

Validates that DA utility peaks at the conditioning target: when we observe
image A, the query x* = A should have the highest DA utility. Specifically:
1. Interpolation: U_DA increases monotonically from perturbed to A
2. Gradient ascent: starting from perturbed, maximizes U_DA(x | observe A)

Uses standard ArcCosineKernel (K(x,x) = (1/pi)*||x||_C^2 * J(0)),
which has norm-dependent utility behavior — gradient ascent may exploit
norm growth rather than angular convergence.

Model training reuses setup() from explore_utility.py (M=50, N_TRAIN=50).
All other params from default_params.json via build_config_from_defaults().

Usage:
    python investigations/understanding_utility/gradient_unnormalized.py
"""

import sys
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

from explore_utility import setup
from acquisition import distribution_aware_utility
from run_single_mode import load_pnas_data
from gpy_training import predict
from metrics import compute_pearson_correlation, compute_explained_variance

# ---------------------------------------------------------------------------
# Investigation parameters (visible, explicit)
# Model params (M=50, N_TRAIN=50) are set in explore_utility.py
# ---------------------------------------------------------------------------
# --- Experiment mode (revertible: set False to restore natural image experiment) ---
USE_SYNTHETIC = True     # True: bipartite target + noise start. False: natural + smoothing.
DARK_GRAY = -0.5         # synthetic bipartite: left half pixel value
LIGHT_GRAY = 0.5         # synthetic bipartite: right half pixel value
NOISE_AMP = 0.5          # synthetic: random noise amplitude for starting image

# --- Natural image experiment params (used when USE_SYNTHETIC = False) ---
SIGMA_SMOOTH = 1.0       # Gaussian smoothing sigma for perturbation
TARGET_INDEX = 0         # which pool image to use as target

SIGMA_0 = 1000           # Override kernel sigma_0 AFTER training. None = keep trained value.

N_INTERP = 21            # interpolation points along path
N_STEPS = 5000           # gradient ascent steps
LR = 5.1                 # learning rate for plain gradient ascent
LOG_EVERY = 50           # print interval for gradient ascent


def interpolation_sweep(model, likelihood, x_target, x_perturbed, n_points, r_max):
    """Compute U_DA along the line from x_perturbed (t=0) to x_target (t=1).

    Returns:
        ts: (n_points,) numpy array of interpolation parameters
        utilities: (n_points,) numpy array of U_DA values
    """
    ts = np.linspace(0, 1, n_points)
    utilities = []

    with torch.no_grad():
        for i, t in enumerate(ts):
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


def gradient_ascent(model, likelihood, x_start, x_target, r_max, n_steps, lr):
    """Gradient ascent maximizing U_DA(x | observe A).

    Returns:
        x_final: (n_pixels,) optimized image
        history: dict with lists of per-step metrics
    """
    x_opt = x_start.clone().detach().requires_grad_(True)
    initial_dist = (x_start - x_target).norm().item()

    history = {
        'step': [], 'utility': [], 'dist': [],
        'frac_dist': [], 'grad_norm': [],
    }

    for step in range(n_steps):
        result = distribution_aware_utility(
            model, likelihood,
            x_opt.unsqueeze(0),
            x_target.unsqueeze(0),
            r_max=r_max,
            adaptive_r_max=False,
            sample_lambda=False,
        )
        utility = result['utility'].squeeze()
        utility.backward()

        grad_norm = x_opt.grad.norm().item()
        dist = (x_opt.detach() - x_target).norm().item()
        frac_dist = dist / initial_dist

        history['step'].append(step)
        history['utility'].append(utility.item())
        history['dist'].append(dist)
        history['frac_dist'].append(frac_dist)
        history['grad_norm'].append(grad_norm)

        # NaN check
        if np.isnan(utility.item()) or np.isnan(grad_norm):
            print(f"  step {step}: NaN detected - stopping")
            break

        if step % LOG_EVERY == 0 or step == n_steps - 1:
            print(f"  step {step:4d}: U={utility.item():.6f}  "
                  f"frac_dist={frac_dist:.4f}  |grad|={grad_norm:.4e}")

        # Gradient step
        with torch.no_grad():
            x_opt += lr * x_opt.grad
        x_opt.grad = None

    return x_opt.detach(), history


def main():
    # === Step 1: Train model with unnormalized kernel ===
    print("=" * 70)
    print("Step 1: Training model with ArcCosineKernel (unnormalized)")
    print("=" * 70)

    env = setup()
    model = env['model']
    likelihood = env['likelihood']
    X_pool = env['X_pool']
    config = env['config']
    r_max = config['r_max']
    n_px_side = config['n_px_side']

    # --- Optional sigma_0 override ---
    sigma0_overridden = False
    if SIGMA_0 is not None:
        trained_sigma0 = model.covar_module.sigma_0.item()
        model.covar_module.sigma_0 = SIGMA_0
        sigma0_overridden = True
        print(f"\n** sigma_0 changed: {trained_sigma0:.4f} -> {SIGMA_0} (post-training override)")

    # --- Test evaluation ---
    data_path = _gpytorch_dir.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = load_pnas_data(data_path, dtype=torch.float32)
    device = next(model.parameters()).device
    X_test = data['X_test'].reshape(data['X_test'].shape[0], -1).to(device)
    R_test = data['R_test'].to(device)
    r_test = R_test[:, :, config['cell']]  # (30 repeats, 30 images)

    predictions = predict(model, likelihood, X_test, device=device,
                          jitter=config['jitter'],
                          cholesky_max_tries=config['cholesky_max_tries'])
    f_pred = predictions['f_pred']
    r_test_mean = r_test.mean(dim=0)
    test_r = compute_pearson_correlation(r_test_mean.float(), f_pred.float())
    explained_var, reliability = compute_explained_variance(r_test.float(), f_pred.float())
    print(f"\nTest Pearson r:  {test_r:.4f}")
    print(f"Reliability:     {reliability:.4f}")
    print(f"Explained var:   {explained_var:.4f}")

    # === Step 2: Setup target and starting image ===
    print("\n" + "=" * 70)
    print(f"Step 2: Creating images (USE_SYNTHETIC={USE_SYNTHETIC})")
    print("=" * 70)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_pixels = n_px_side ** 2

    # Ensure RF mask is cached by running a forward pass on any pool image
    kernel = model.covar_module
    if not (hasattr(kernel, '_cached_mask') and kernel._cached_mask is not None):
        with torch.no_grad():
            _ = model(X_pool[0].unsqueeze(0))
    rf_mask = kernel._cached_mask.squeeze()

    if USE_SYNTHETIC:
        # --- Bipartite target: left=dark, right=light within RF mask ---
        mask_2d = rf_mask.cpu().numpy().reshape(n_px_side, n_px_side)
        mask_rows, mask_cols = np.where(mask_2d)
        col_center = (mask_cols.min() + mask_cols.max()) / 2

        target_2d = np.zeros((n_px_side, n_px_side), dtype=np.float32)
        # Left half of mask
        left = mask_2d & (np.arange(n_px_side)[None, :] <= col_center)
        right = mask_2d & (np.arange(n_px_side)[None, :] > col_center)
        target_2d[left] = DARK_GRAY
        target_2d[right] = LIGHT_GRAY

        x_target = torch.tensor(target_2d.reshape(-1), dtype=dtype, device=device)

        # --- Noise starting image within RF mask ---
        torch.manual_seed(config['seed'])
        x_perturbed = NOISE_AMP * torch.randn(n_pixels, dtype=dtype, device=device)
        x_perturbed[~rf_mask] = 0.0

        start_label = "noise"
        print(f"Target: synthetic bipartite (dark={DARK_GRAY}, light={LIGHT_GRAY})")
        print(f"Start: random noise (amp={NOISE_AMP})")
    else:
        # --- Natural image experiment (original code) ---
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
    print(f"  ||target||={x_target.norm().item():.2f}, "
          f"||start||={x_perturbed.norm().item():.2f}")

    # === Step 3: Interpolation sweep ===
    print("\n" + "=" * 70)
    print(f"Step 3: Interpolation sweep ({start_label} -> target)")
    print("=" * 70)

    ts, utilities_interp = interpolation_sweep(
        model, likelihood, x_target, x_perturbed, N_INTERP, r_max
    )

    # Check monotonicity
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
        model, likelihood, x_perturbed, x_target, r_max, N_STEPS, LR
    )

    print(f"\nGradient ascent summary:")
    print(f"  Steps: {len(history['step'])}")
    print(f"  U_DA: {history['utility'][0]:.6f} -> {history['utility'][-1]:.6f}")
    print(f"  frac_dist: {history['frac_dist'][0]:.4f} -> {history['frac_dist'][-1]:.4f}")
    print(f"  Converged toward target: {history['frac_dist'][-1] < history['frac_dist'][0]}")
    print(f"  ||final||={x_final.norm().item():.2f}")

    # === Step 5: Visualization ===
    print("\n" + "=" * 70)
    print("Step 5: Visualization")
    print("=" * 70)

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

    # vmin/vmax from the ENTIRE natural image dataset (RF pixels only)
    rf_mask_np = rf_mask.cpu().numpy()
    X_all = torch.cat([env['X_pool'], env['X_train']], dim=0)
    all_rf_vals = X_all.cpu().numpy()[:, rf_mask_np]  # (N_all, n_rf_pixels)
    vmin = float(all_rf_vals.min())
    vmax = float(all_rf_vals.max())
    gray_val = (vmin + vmax) / 2
    print(f"  Dataset RF pixel range: [{vmin:.3f}, {vmax:.3f}] (from {X_all.shape[0]} images)")

    img_target = masked_crop(x_target, gray_val)
    img_perturbed = masked_crop(x_perturbed, gray_val)
    img_final = masked_crop(x_final, gray_val)

    # Check if smoothed/final images exceed the natural dataset pixel range
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

    print(f"mean={X_all.mean():.3f}, std={X_all.std():.3f}")

    # Difference image: final - start (shows what optimizer changed)
    diff_flat = x_final - x_perturbed
    img_diff = masked_crop(diff_flat, 0.0)  # gray_val=0 for difference

    fig = plt.figure(figsize=(18, 8))

    # Top row: 4 images (target, start, final, difference) with colorbars
    images = [img_target, img_perturbed, img_final]
    if USE_SYNTHETIC:
        titles = ['Target (bipartite)', f'Start ({start_label})', 'Final (grad ascent)']
    else:
        titles = ['Target A', f'Start ({start_label})', 'Final (grad ascent)']
    labels = ['Target', 'Start', 'Final']
    for i, (img, title, label) in enumerate(zip(images, titles, labels)):
        ax = fig.add_subplot(2, 4, i + 1)
        im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        # Add clipping warning on the image if pixels exceed natural range
        if label in clip_warnings:
            w = clip_warnings[label]
            warn_text = f"CLIPPED: [{w['min_val']:.2f}, {w['max_val']:.2f}]"
            ax.text(0.5, 0.02, warn_text, transform=ax.transAxes, fontsize=8,
                    color='red', fontweight='bold', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    # 4th image: difference (final - start), diverging colormap
    ax_diff = fig.add_subplot(2, 4, 4)
    im_diff = ax_diff.imshow(img_diff, cmap='gray', vmin=vmin, vmax=vmax,
                             aspect='equal')
    ax_diff.set_title('Final - Start', fontsize=10)
    ax_diff.axis('off')
    cb_diff = fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)
    cb_diff.ax.tick_params(labelsize=7)

    # Print difference stats
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

    # Bottom right (spanning remaining cols): Gradient ascent
    ax2 = fig.add_subplot(2, 2, 4)
    steps = history['step']
    color_u = 'tab:blue'
    color_d = 'tab:red'

    ax2.plot(steps, history['utility'], color=color_u, linewidth=1.5, label='U_DA')
    ax2.axhline(utilities_interp[-1], color=color_u, linestyle='--', alpha=0.5,
                label=f'U_DA(A|A) = {utilities_interp[-1]:.4f}')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('U_DA', color=color_u)
    ax2.tick_params(axis='y', labelcolor=color_u)
    ax2.legend(loc='center left')

    ax2r = ax2.twinx()
    ax2r.plot(steps, history['frac_dist'], color=color_d, linewidth=1.5, alpha=0.7)
    ax2r.set_ylabel('Fractional distance to A', color=color_d)
    ax2r.tick_params(axis='y', labelcolor=color_d)
    ax2r.axhline(1.0, color=color_d, linestyle=':', alpha=0.3)

    ax2.set_title('Gradient ascent convergence')
    ax2.grid(True, alpha=0.3)

    title_str = (
        f'DA Utility - Unnormalized Kernel '
        f'(M={config["M"]}, n_train={config["n_train"]}, '
        f'seed={config["seed"]}, cell={config["cell"]})  '
        f'test_r={test_r:.3f}, reliability={reliability:.3f}'
    )
    if sigma0_overridden:
        title_str += f'  [sigma_0 override: {SIGMA_0}]'
    fig.suptitle(title_str, fontsize=11)
    fig.tight_layout()

    out_path = _script_dir / 'gradient_ascent_unnormalized.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
