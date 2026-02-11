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
from acquisition import distribution_aware_utility, get_gp_marginal_moments, compute_H
from run_single_mode import load_pnas_data
from gpy_training import predict
from metrics import compute_pearson_correlation, compute_explained_variance

# ---------------------------------------------------------------------------
# Investigation parameters (visible, explicit)
# Model params (M=50, N_TRAIN=50) are set in explore_utility.py
# ---------------------------------------------------------------------------
# --- Experiment mode (revertible: set False to restore natural image experiment) ---
USE_SYNTHETIC = False     # False: bipartite target + noise start. False: natural + smoothing.
DARK_GRAY = -0.5         # synthetic bipartite: left half pixel value
LIGHT_GRAY = 0.5         # synthetic bipartite: right half pixel value
NOISE_AMP = 0.5          # synthetic: random noise amplitude for starting image

# --- Natural image experiment params (used when USE_SYNTHETIC = False) ---
SIGMA_SMOOTH = 1.0       # Gaussian smoothing sigma for perturbation
TARGET_INDEX = 0         # which pool image to use as target

SIGMA_0 = None           # Override kernel sigma_0 AFTER training. None = keep trained value.

# --- Pixel bounds (experimental, may be reverted) ---
USE_SIGMOID_BOUNDS = True   # sigmoid reparametrization: pixels bounded to dataset [vmin, vmax]

N_INTERP = 21            # interpolation points along path

# --- LBFGS optimizer parameters ---
N_STEPS = 50             # outer LBFGS steps
LR = 0.5                 # LBFGS step size (1.0 standard for quasi-Newton)
LBFGS_MAX_ITER = 20      # max iterations per LBFGS step (line search evals)
LBFGS_MAX_EVAL = 25      # max function evaluations per LBFGS step
LBFGS_HISTORY_SIZE = 10  # number of past gradients for Hessian approximation
LOG_EVERY = 1            # print every step (LBFGS steps are expensive)


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


def _reconstruct_image(x_rf, rf_mask, n_pixels, dtype, device):
    """Place RF pixel values into full image, zeros elsewhere."""
    x_full = torch.zeros(n_pixels, dtype=dtype, device=device)
    x_full[rf_mask] = x_rf
    return x_full


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
    f_max = config['f_max']
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

    # Dataset pixel bounds (RF pixels only) — used for sigmoid reparametrization + visualization
    rf_mask_np = rf_mask.cpu().numpy()
    X_all = torch.cat([X_pool, env['X_train']], dim=0)
    all_rf_vals = X_all.cpu().numpy()[:, rf_mask_np]  # (N_all, n_rf_pixels)
    vmin = float(all_rf_vals.min())
    vmax = float(all_rf_vals.max())
    print(f"  Dataset RF pixel range: [{vmin:.3f}, {vmax:.3f}] (from {X_all.shape[0]} images)")

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
    print(f"  ||target||_RF={x_target[rf_mask].norm().item():.2f}, "
          f"||start||_RF={x_perturbed[rf_mask].norm().item():.2f}")

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

    pixel_lo = vmin if USE_SIGMOID_BOUNDS else None
    pixel_hi = vmax if USE_SIGMOID_BOUNDS else None
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

    # --- DEBUG: firing rate diagnostics (target vs optimized) ---
    print(f"\n  DEBUG: Firing rate diagnostics (f_max={f_max})")
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
    # --- END DEBUG ---

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

    # vmin/vmax already computed in Step 2 (reuse for visualization)
    gray_val = (vmin + vmax) / 2

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
    ax2.set_xlabel('Step')
    ax2.set_ylabel('U_DA', color=color_u)
    ax2.tick_params(axis='y', labelcolor=color_u)
    ax2.legend(loc='center left')

    ax2r = ax2.twinx()
    ax2r.plot(steps, history['pearson_r'], color=color_d, linewidth=1.5, alpha=0.7)
    ax2r.set_ylabel('Pearson r (RF)', color=color_d)
    ax2r.tick_params(axis='y', labelcolor=color_d)
    ax2r.axhline(1.0, color=color_d, linestyle=':', alpha=0.3)

    ax2.set_title('LBFGS gradient ascent convergence')
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

    out_path = _script_dir / 'gradient_unnormalized.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
