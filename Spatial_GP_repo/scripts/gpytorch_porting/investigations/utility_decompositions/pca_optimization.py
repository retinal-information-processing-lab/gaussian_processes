"""
PCA-constrained gradient ascent for distribution-aware utility.

Optimizes x* = mu + V_K @ z to maximize U_DA(x* | observe x_target), where:
- mu = mean of training images (RF-masked pixels)
- V_K = top K principal components of training image covariance
- z in R^K is the optimization variable

The gradient in PCA space: nabla_z U = V_K^T @ nabla_x* U

Model training via setup() from explore_utility.py.
All model params from default_params.json via build_config_from_defaults().

Simplifications in this first version:
- Single target image (x_target = first pool image)
- PCA computed on training images only (not pool)
- K chosen by 95% variance explained threshold (CLI-overridable)
- No pixel bounds enforcement (PCA subspace provides implicit constraint)

Usage:
    python investigations/utility_decompositions/pca_optimization.py

    # Custom PCA components:
    python investigations/utility_decompositions/pca_optimization.py --n-components 50

    # Custom variance threshold:
    python investigations/utility_decompositions/pca_optimization.py --var-threshold 0.99
"""

import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

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

# Import from explore_utility.py in utility/ folder
sys.path.insert(0, str(_script_dir.parent / 'utility'))
from explore_utility import setup

from run_single_mode import build_config_from_defaults

# ---------------------------------------------------------------------------
# Investigation-specific constants (optimizer tuning, not model parameters)
# ---------------------------------------------------------------------------
N_STEPS = 50             # outer optimization steps
LR = 0.5                 # LBFGS learning rate
LBFGS_MAX_ITER = 20      # LBFGS inner iterations per step
LBFGS_MAX_EVAL = 25      # LBFGS max function evaluations per step
LBFGS_HISTORY_SIZE = 10  # LBFGS history size
LOG_EVERY = 5            # print every N steps

TARGET_INDEX = 0          # pool image index for target

DEFAULT_VAR_THRESHOLD = 0.95  # fraction of variance to retain
START_NOISE = 5             # noise std relative to ||z_target|| (0 = exact PCA projection)


# ============================================================================
# PCA computation
# ============================================================================

def compute_pca(X_train, rf_mask, var_threshold=0.95, n_components=None):
    """Compute PCA on RF-masked training images.

    Args:
        X_train: (N, d) training images on device
        rf_mask: (d,) boolean mask for RF pixels
        var_threshold: fraction of variance to retain (used if n_components is None)
        n_components: explicit number of components (overrides var_threshold)

    Returns:
        mu_rf: (n_rf,) mean of training images within RF
        V_K: (n_rf, K) top K eigenvectors (columns)
        eigenvalues: (K,) corresponding eigenvalues
        K: number of retained components
        var_explained: fraction of variance explained by K components
    """
    # Extract RF pixels: (N, n_rf)
    X_rf = X_train[:, rf_mask]
    n_train, n_rf = X_rf.shape

    # Mean
    mu_rf = X_rf.mean(dim=0)  # (n_rf,)

    # Center
    X_centered = X_rf - mu_rf.unsqueeze(0)  # (N, n_rf)

    # SVD on centered data for numerical stability
    # X_centered = U @ S @ V^T, so cov = V @ (S^2 / N) @ V^T
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    # S: (min(N, n_rf),), Vt: (min(N, n_rf), n_rf)
    # eigenvalues of covariance = S^2 / N
    eigenvalues_all = S ** 2 / n_train  # (min(N, n_rf),)
    V_all = Vt.T  # (n_rf, min(N, n_rf)) -- columns are eigenvectors

    # Determine K
    total_var = eigenvalues_all.sum().item()
    cumvar = torch.cumsum(eigenvalues_all, dim=0) / total_var

    if n_components is not None:
        K = min(n_components, len(eigenvalues_all))
    else:
        # Find smallest K such that cumvar[K-1] >= var_threshold
        above = (cumvar >= var_threshold).nonzero(as_tuple=True)[0]
        if len(above) > 0:
            K = above[0].item() + 1
        else:
            K = len(eigenvalues_all)

    V_K = V_all[:, :K]  # (n_rf, K)
    eigenvalues = eigenvalues_all[:K]  # (K,)
    var_explained = cumvar[K - 1].item()

    print(f"  PCA: n_train={n_train}, n_rf={n_rf}")
    print(f"  K={K} components, variance explained={var_explained:.4f} "
          f"(threshold={var_threshold})")
    print(f"  Eigenvalue range: [{eigenvalues[-1].item():.6f}, {eigenvalues[0].item():.6f}]")

    return mu_rf, V_K, eigenvalues, K, var_explained


# ============================================================================
# PCA reconstruction helpers
# ============================================================================

def pca_to_image(z, mu_rf, V_K, rf_mask, n_pixels, dtype, device):
    """Convert PCA coordinates z to full image.

    x_rf = mu + V_K @ z
    Then place RF pixels into full image (zeros elsewhere).

    MUST NOT be called under torch.no_grad() during optimization --
    the gradient flows through z -> V_K @ z -> x_rf -> kernel -> utility.
    """
    x_rf = mu_rf + V_K @ z  # (n_rf,)
    x_full = torch.zeros(n_pixels, dtype=dtype, device=device)
    x_full[rf_mask] = x_rf
    return x_full


def image_to_pca(x_full, mu_rf, V_K, rf_mask):
    """Project a full image into PCA coordinates.

    z = V_K^T @ (x_rf - mu)
    """
    x_rf = x_full[rf_mask]
    return V_K.T @ (x_rf - mu_rf)  # (K,)


# ============================================================================
# Gradient ascent in PCA space
# ============================================================================

def gradient_ascent_pca(model, likelihood, z_start, x_target, mu_rf, V_K,
                        rf_mask, r_max, f_max, n_pixels, dtype, device,
                        n_steps, lr, z_max_norm=None):
    """LBFGS gradient ascent maximizing U_DA in PCA space.

    Optimizes z in R^K. The image is reconstructed as x* = mu + V_K @ z.
    Gradient chain: z -> V_K @ z + mu -> x_full -> kernel -> utility.

    If z_max_norm is set, z is projected back onto the ball ||z|| <= z_max_norm
    after each outer step (projected gradient ascent).

    Args:
        z_start: (K,) initial PCA coordinates
        x_target: (n_pixels,) conditioning target for DA utility
        mu_rf, V_K: PCA basis (from compute_pca)
        rf_mask: (n_pixels,) boolean RF mask
        r_max: max spike count for Laplace truncation
        f_max: firing rate guard threshold
        n_pixels: total number of pixels
        dtype, device: tensor specs
        z_max_norm: if set, project z onto ||z|| <= z_max_norm after each step

    Returns:
        x_final: (n_pixels,) optimized image
        z_final: (K,) final PCA coordinates
        history: dict with per-step metrics
    """
    z = z_start.clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [z], lr=lr,
        max_iter=LBFGS_MAX_ITER,
        max_eval=LBFGS_MAX_EVAL,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn='strong_wolfe',
    )

    history = {
        'step': [], 'utility': [], 'grad_norm': [],
        'z_norm': [], 'pearson_r_rf': [],
    }

    for step in range(n_steps):

        def closure():
            optimizer.zero_grad()
            x_full = pca_to_image(z, mu_rf, V_K, rf_mask, n_pixels, dtype, device)
            result = distribution_aware_utility(
                model, likelihood,
                x_full.unsqueeze(0),
                x_target.unsqueeze(0),
                r_max=r_max,
                adaptive_r_max=False,
                sample_lambda=False,
            )
            # Firing rate guard
            mu_g = result['mu_g_marg']
            if torch.exp(mu_g).item() > f_max:
                return torch.tensor(float('inf'), device=device)
            loss = -result['utility'].squeeze()  # NEGATE: maximize utility
            loss.backward()
            return loss

        optimizer.step(closure)

        # Project z back onto the ball ||z|| <= z_max_norm
        if z_max_norm is not None:
            with torch.no_grad():
                current_norm = z.norm()
                if current_norm > z_max_norm:
                    z.mul_(z_max_norm / current_norm)

        # Verify gradient flow on first step
        if step == 0:
            assert z.grad is not None and z.grad.norm() > 0, \
                "No gradient flow through PCA reconstruction"

        # Track metrics
        with torch.no_grad():
            x_full_eval = pca_to_image(z, mu_rf, V_K, rf_mask, n_pixels, dtype, device)
            result_eval = distribution_aware_utility(
                model, likelihood,
                x_full_eval.unsqueeze(0),
                x_target.unsqueeze(0),
                r_max=r_max,
                adaptive_r_max=False,
                sample_lambda=False,
            )
            utility = result_eval['utility'].item()

            # Pearson r within RF
            a = x_full_eval[rf_mask]
            b = x_target[rf_mask]
            a_c = a - a.mean()
            b_c = b - b.mean()
            num = (a_c * b_c).sum()
            denom = a_c.norm() * b_c.norm()
            pr = (num / denom).item() if denom > 1e-12 else 0.0

        grad_norm = z.grad.norm().item() if z.grad is not None else 0.0

        history['step'].append(step)
        history['utility'].append(utility)
        history['grad_norm'].append(grad_norm)
        history['z_norm'].append(z.detach().norm().item())
        history['pearson_r_rf'].append(pr)

        if np.isnan(utility) or np.isnan(grad_norm):
            print(f"  step {step}: NaN detected - stopping")
            break

        if step % LOG_EVERY == 0 or step == n_steps - 1:
            print(f"  step {step:4d}: U={utility:.6f}  "
                  f"r_rf={pr:.4f}  |z|={z.detach().norm().item():.4f}  "
                  f"|grad_z|={grad_norm:.4e}")

    with torch.no_grad():
        x_final = pca_to_image(z, mu_rf, V_K, rf_mask, n_pixels, dtype, device)
        z_final = z.detach().clone()

    return x_final.detach(), z_final, history


# ============================================================================
# Visualization
# ============================================================================

def plot_results(x_target, x_target_pca, x_start, x_final, history, rf_mask,
                 u_target_orig, u_target_pca, target_index,
                 kernel, config, vmin, vmax, test_r, K, var_explained,
                 var_threshold, kernel_type,
                 n_px_side, out_path):
    """Generate PCA optimization summary figure.

    Top row: target, target PCA projection, start (noisy), final optimized.
    Bottom left: utility convergence. Bottom right: Pearson r convergence.

    Each subplot independently checks for OOB pixels (RF-masked only)
    and flags with red title if any exceed [vmin, vmax].
    """
    mask_2d = rf_mask.cpu().numpy().reshape(n_px_side, n_px_side)
    rf_mask_np = rf_mask.cpu().numpy()
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    r_min, r_max_px = np.where(rows)[0][[0, -1]]
    c_min, c_max_px = np.where(cols)[0][[0, -1]]
    r_min = max(0, r_min - 1)
    r_max_px = min(n_px_side - 1, r_max_px + 1)
    c_min = max(0, c_min - 1)
    c_max_px = min(n_px_side - 1, c_max_px + 1)

    def masked_crop(x_flat, gray_val):
        img = x_flat.detach().cpu().numpy().reshape(n_px_side, n_px_side).copy()
        img[~mask_2d] = gray_val
        return img[r_min:r_max_px+1, c_min:c_max_px+1]

    def check_oob(x_flat):
        """Check OOB on RF pixels. Returns (pct_oob, actual_min, actual_max)."""
        vals = x_flat.detach().cpu().numpy()[rf_mask_np]
        n_below = (vals < vmin).sum()
        n_above = (vals > vmax).sum()
        n_oob = n_below + n_above
        pct = 100.0 * n_oob / len(vals)
        return pct, float(vals.min()), float(vals.max())

    gray_val = (vmin + vmax) / 2

    # RF overlay params
    eps_0x = kernel.eps_0x.item()
    eps_0y = kernel.eps_0y.item()
    beta_nat = kernel.beta.item()
    sigma_rf = beta_nat * np.sqrt(2)
    cx_px = (eps_0x + 1) / 2 * (n_px_side - 1)
    cy_px = (eps_0y + 1) / 2 * (n_px_side - 1)
    sigma_px = sigma_rf * (n_px_side - 1) / 2
    cx_crop = cx_px - c_min
    cy_crop = cy_px - r_min

    def draw_rf_overlay(ax):
        ax.plot(cx_crop, cy_crop, 'r+', markersize=8, markeredgewidth=1.5)
        circle_1s = plt.Circle((cx_crop, cy_crop), sigma_px, fill=False,
                               color='red', linewidth=1.5, linestyle='-')
        circle_2s = plt.Circle((cx_crop, cy_crop), 2 * sigma_px, fill=False,
                               color='red', linewidth=1, linestyle='--')
        ax.add_patch(circle_1s)
        ax.add_patch(circle_2s)

    fig = plt.figure(figsize=(16, 8))

    # Top row: 4 images — each with independent OOB check
    x_tensors = [x_target, x_target_pca, x_start, x_final]
    cropped_images = [masked_crop(x, gray_val) for x in x_tensors]

    u_start = history['utility'][0]
    u_final = history['utility'][-1]

    base_titles = [
        f'Target A (original, pool[{target_index}])\nU_DA={u_target_orig:.4f}',
        f'Target A (PCA, K={K})\nU_DA={u_target_pca:.4f}',
        f'Start (noisy)\nU_DA={u_start:.4f}',
        f'Final (PCA optimized)\nU_DA={u_final:.4f}',
    ]

    for i, (img, x_flat, title) in enumerate(zip(cropped_images, x_tensors, base_titles)):
        pct_oob, lo, hi = check_oob(x_flat)
        is_oob = pct_oob > 0
        if is_oob:
            title += f'\nCLIPPED {pct_oob:.0f}% OOB [{lo:.2f},{hi:.2f}]'

        ax = fig.add_subplot(2, 4, i + 1)
        im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(title, fontsize=10, color='red' if is_oob else 'black')
        ax.axis('off')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        draw_rf_overlay(ax)

    # Bottom left: Utility convergence
    ax1 = fig.add_subplot(2, 2, 3)
    steps = history['step']
    ax1.plot(steps, history['utility'], 'b.-', linewidth=1.5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('U_DA')
    ax1.set_title('Utility convergence')
    ax1.grid(True, alpha=0.3)

    # Bottom right: Pearson r + z norm
    ax2 = fig.add_subplot(2, 2, 4)
    color_r = 'tab:green'
    color_z = 'tab:red'
    ax2.plot(steps, history['pearson_r_rf'], color=color_r, linewidth=1.5,
             label='Pearson r (RF)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Pearson r', color=color_r)
    ax2.tick_params(axis='y', labelcolor=color_r)
    ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
    ax2.grid(True, alpha=0.3)

    ax2r = ax2.twinx()
    ax2r.plot(steps, history['z_norm'], color=color_z, linewidth=1.5,
              alpha=0.7, linestyle='--', label='|z|')
    ax2r.set_ylabel('|z|', color=color_z)
    ax2r.tick_params(axis='y', labelcolor=color_z)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=8)
    ax2.set_title('Structure convergence')

    title_str = (
        f'PCA-constrained DA Utility Optimization  '
        f'(K={K}, var_thr={var_threshold:.2f}, var_expl={var_explained:.3f}, '
        f'kernel={kernel_type}, '
        f'M={config["M"]}, n_train={config["n_train"]}, '
        f'seed={config["seed"]}, cell={config["cell"]})  '
        f'test_r={test_r:.3f}'
    )
    fig.suptitle(title_str, fontsize=11)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PCA-constrained DA utility gradient ascent')
    parser.add_argument('--n-components', type=int, default=None,
                        help='Explicit number of PCA components (overrides --var-threshold)')
    parser.add_argument('--var-threshold', type=float, default=DEFAULT_VAR_THRESHOLD,
                        help=f'Variance explained threshold for PCA (default: {DEFAULT_VAR_THRESHOLD})')
    parser.add_argument('--kernel-type', type=str, default=None,
                        choices=['arc_cosine', 'arc_sine', 'rbf'],
                        help='Kernel type (default: from default_params.json)')
    parser.add_argument('--target-index', type=int, default=TARGET_INDEX,
                        help=f'Pool image index for target (default: {TARGET_INDEX})')
    parser.add_argument('--M', type=int, default=None,
                        help='Number of inducing points (default: from explore_utility)')
    parser.add_argument('--n-train', type=int, default=None,
                        help='Number of training points (default: from explore_utility)')
    args = parser.parse_args()

    # === Step 1: Train model ===
    print("=" * 70)
    print("Step 1: Training model")
    print("=" * 70)

    env = setup(kernel_type=args.kernel_type,
                M_override=args.M, n_train_override=args.n_train)
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

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_pixels = n_px_side ** 2

    # === Step 2: Get RF mask ===
    print("\n" + "=" * 70)
    print("Step 2: RF mask and target image")
    print("=" * 70)

    kernel = getattr(model, 'covar_module', None) or getattr(model, 'kernel', None)
    if not (hasattr(kernel, '_cached_mask') and kernel._cached_mask is not None):
        with torch.no_grad():
            _ = model(X_pool[0].unsqueeze(0))
    rf_mask = kernel._cached_mask.squeeze()
    n_rf = rf_mask.sum().item()
    print(f"  RF mask: {n_rf} active pixels out of {n_pixels}")

    # Target image
    x_target = X_pool[args.target_index]
    print(f"  Target: pool image {args.target_index}")

    # Dataset pixel bounds for visualization
    rf_mask_np = rf_mask.cpu().numpy()
    X_all = torch.cat([X_pool, X_train], dim=0)
    all_rf_vals = X_all.cpu().numpy()[:, rf_mask_np]
    vmin_dataset = float(all_rf_vals.min())
    vmax_dataset = float(all_rf_vals.max())
    print(f"  Dataset RF pixel range: [{vmin_dataset:.3f}, {vmax_dataset:.3f}]")

    # === Step 3: Compute PCA ===
    print("\n" + "=" * 70)
    print("Step 3: PCA on all images (RF-masked)")
    print("=" * 70)

    mu_rf, V_K, eigenvalues, K, var_explained = compute_pca(
        X_all, rf_mask,
        var_threshold=args.var_threshold,
        n_components=args.n_components,
    )

    # === Step 4: Project target into PCA space for starting point ===
    print("\n" + "=" * 70)
    print("Step 4: PCA projection and starting point")
    print("=" * 70)

    z_target = image_to_pca(x_target, mu_rf, V_K, rf_mask)
    x_target_reconstructed = pca_to_image(z_target, mu_rf, V_K, rf_mask,
                                          n_pixels, dtype, device)

    # Reconstruction error within RF
    recon_err = (x_target[rf_mask] - x_target_reconstructed[rf_mask]).norm().item()
    target_norm = x_target[rf_mask].norm().item()
    print(f"  Target PCA reconstruction error: {recon_err:.4f} "
          f"(relative: {recon_err / target_norm:.4f})")
    print(f"  ||z_target|| = {z_target.norm().item():.4f}")

    # Compute z-norm statistics from training data for norm constraint
    z_train_norms = []
    for i in range(X_train.shape[0]):
        z_i = image_to_pca(X_train[i], mu_rf, V_K, rf_mask)
        z_train_norms.append(z_i.norm().item())
    z_train_norms = np.array(z_train_norms)
    z_max_norm = float(np.percentile(z_train_norms, 95))
    print(f"  Training z-norms: mean={z_train_norms.mean():.2f}, "
          f"std={z_train_norms.std():.2f}, "
          f"max={z_train_norms.max():.2f}, "
          f"95th percentile={z_max_norm:.2f}")

    # Start from noisy version of target's PCA projection
    z_start = z_target.clone().detach()
    if START_NOISE > 0:
        noise_std = START_NOISE * z_target.norm().item()
        z_start = z_start + noise_std * torch.randn_like(z_start)
    # Clip z_start to norm boundary before optimization (avoids gradient
    # detach from the in-loop projection on the very first step)
    if z_max_norm is not None and z_start.norm() > z_max_norm:
        z_start = z_start * (z_max_norm / z_start.norm())
    x_start = pca_to_image(z_start, mu_rf, V_K, rf_mask, n_pixels, dtype, device)
    print(f"  Starting from z_target + noise (START_NOISE={START_NOISE})")
    print(f"  ||z_start|| = {z_start.norm().item():.4f}")
    print(f"  Norm constraint: ||z|| <= {z_max_norm:.2f} (95th percentile of training)")

    # === Step 5: Gradient ascent ===
    print("\n" + "=" * 70)
    print(f"Step 5: LBFGS gradient ascent in PCA space (K={K})")
    print("=" * 70)

    x_final, z_final, history = gradient_ascent_pca(
        model, likelihood, z_start, x_target, mu_rf, V_K,
        rf_mask, r_max, f_max, n_pixels, dtype, device,
        N_STEPS, LR, z_max_norm=z_max_norm,
    )

    print(f"\nGradient ascent summary:")
    print(f"  Steps: {len(history['step'])}")
    print(f"  U_DA: {history['utility'][0]:.6f} -> {history['utility'][-1]:.6f}")
    print(f"  Pearson r (RF): {history['pearson_r_rf'][0]:.4f} -> {history['pearson_r_rf'][-1]:.4f}")
    print(f"  |z|: {history['z_norm'][0]:.4f} -> {history['z_norm'][-1]:.4f}")

    # PCA space analysis: how does z_final compare to z_target?
    z_similarity = torch.nn.functional.cosine_similarity(
        z_final.unsqueeze(0), z_target.unsqueeze(0)
    ).item()
    print(f"\n  PCA space analysis:")
    print(f"    z_target: |z|={z_target.norm().item():.4f}")
    print(f"    z_final:  |z|={z_final.norm().item():.4f}")
    print(f"    cosine_similarity(z_final, z_target) = {z_similarity:.4f}")

    # Firing rate diagnostics
    A = likelihood.A.squeeze()
    lam0 = likelihood.lambda0.squeeze()
    print(f"\n  Firing rate diagnostics (f_max={f_max})")
    with torch.no_grad():
        for label, x_img in [('target', x_target), ('start', x_start), ('final', x_final)]:
            mu, sigma2 = get_gp_marginal_moments(model, x_img.unsqueeze(0))
            mu_g = A * mu + lam0
            firing_rate = torch.exp(mu_g).item()
            H = compute_H(mu, sigma2, r_max=r_max, a=A, lambda0=lam0).item()
            flag = " ** EXCEEDS f_max" if firing_rate > f_max else ""
            print(f"    {label:8s}: lambda_m={mu.item():.4f}, "
                  f"mu_g={mu_g.item():.4f}, firing_rate={firing_rate:.2f}, "
                  f"H_marg={H:.6f}{flag}")

    # Compute utility for target (original) and target (PCA projection)
    with torch.no_grad():
        result_target_orig = distribution_aware_utility(
            model, likelihood,
            x_target.unsqueeze(0), x_target.unsqueeze(0),
            r_max=r_max, adaptive_r_max=False, sample_lambda=False,
        )
        u_target_orig = result_target_orig['utility'].item()

        result_target_pca = distribution_aware_utility(
            model, likelihood,
            x_target_reconstructed.unsqueeze(0), x_target.unsqueeze(0),
            r_max=r_max, adaptive_r_max=False, sample_lambda=False,
        )
        u_target_pca = result_target_pca['utility'].item()

    print(f"\n  Utility at target images:")
    print(f"    target (original):       U_DA={u_target_orig:.6f}")
    print(f"    target (PCA projection): U_DA={u_target_pca:.6f}")

    # === Step 6: Visualization ===
    print("\n" + "=" * 70)
    print("Step 6: Visualization")
    print("=" * 70)

    plot_results(
        x_target, x_target_reconstructed, x_start, x_final, history, rf_mask,
        u_target_orig, u_target_pca, args.target_index,
        kernel, config, vmin_dataset, vmax_dataset, test_r, K, var_explained,
        args.var_threshold, kernel_type,
        n_px_side,
        out_path=_script_dir / f'pca_optimization_{kernel_type}_vt{args.var_threshold:.2f}.png',
    )


if __name__ == '__main__':
    main()
