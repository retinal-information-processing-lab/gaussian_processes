"""
Subspace-constrained gradient ascent for distribution-aware utility.

Unified script supporting two subspace parameterizations:

  PCA:         x* = mu + V_K @ z   (data manifold constraint)
  C-eigenspace: x* = U_K @ z       (kernel C matrix eigenvectors)

Both methods optimize z in R^K via LBFGS to maximize U_DA(x* | conditioning images).

PCA truncation is a genuine constraint — it restricts the optimizer to directions
with observed variance in the dataset. C-eigenspace truncation only improves
conditioning (removes near-zero gradient directions) without changing the solution.

Model training via setup() from explore_utility.py.
All model params from default_params.json via build_config_from_defaults().

Usage:
    # PCA, single target (default):
    python investigations/utility_decompositions/subspace_optimization.py

    # C-eigenspace, single target:
    python investigations/utility_decompositions/subspace_optimization.py --method c_eigen

    # PCA with custom variance threshold:
    python investigations/utility_decompositions/subspace_optimization.py --var-threshold 0.95

    # C-eigenspace with custom threshold and RBF kernel:
    python investigations/utility_decompositions/subspace_optimization.py --method c_eigen --kernel-type rbf --eigen-threshold 1e-6

    # Multi-conditioning mode (either method):
    python investigations/utility_decompositions/subspace_optimization.py --method pca --n-cond 300 --kernel-type rbf
"""

import sys
import argparse
import math
import time
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

# Import setup from explore_utility.py in utility/ folder
_utility_dir = _script_dir.parent / 'utility'
sys.path.insert(0, str(_utility_dir))
from explore_utility import setup

# ---------------------------------------------------------------------------
# Investigation-specific constants (optimizer tuning, not model parameters)
# ---------------------------------------------------------------------------

# --- Optimizer ---
N_STEPS = 50             # outer LBFGS steps
LR = 0.5                 # LBFGS step size
LBFGS_MAX_ITER = 20      # max iterations per LBFGS step (line search evals)
LBFGS_MAX_EVAL = 25      # max function evaluations per LBFGS step
LBFGS_HISTORY_SIZE = 10  # number of past gradients for Hessian approximation
LOG_EVERY = 1            # print every step

# --- Image selection ---
TARGET_INDEX = 0         # pool image index for single-target mode
SIGMA_SMOOTH = 5.0       # Gaussian smoothing sigma for C-eigen start (single-target)

# --- Multi-conditioning mode ---
N_COND = 1               # 1 = single-target mode. >1 = multi-conditioning.
COND_SEED = 42           # seed for reproducible conditioning sample selection
START_SEED = 123         # seed for random start image selection (multi-cond only)
GRAD_CHUNK_SIZE = 30     # images per gradient accumulation chunk (GPU memory)

# --- PCA-specific ---
DEFAULT_VAR_THRESHOLD = 0.8  # fraction of variance to retain

# --- C-eigenspace-specific ---
# Relative threshold: keep eigenvalues > EIGEN_REL_THRESHOLD * max_eigenvalue.
# Mass-based thresholds (e.g., 95%) are useless for C because the first
# eigenvalue contains 99%+ of the total mass due to the locality mask.
EIGEN_REL_THRESHOLD = 1e-3

# --- Model size (investigation overrides, smaller for wider RF) ---
M_OVERRIDE = 50
N_TRAIN_OVERRIDE = 50


# ============================================================================
# Shared helpers: subspace <-> image conversion
# ============================================================================

def z_to_image(z, basis, offset, rf_mask, n_pixels, dtype, device):
    """Convert subspace coordinates z to full image.

    x_rf = offset + basis @ z, then place into full image (zeros elsewhere).

    For PCA:   offset = mu_rf (data mean), basis = V_K (PCA eigenvectors)
    For C-eigen: offset = zeros,           basis = U_K (C matrix eigenvectors)

    MUST NOT be called under torch.no_grad() during optimization --
    the gradient flows through z -> basis @ z -> kernel -> utility.
    """
    x_rf = offset + basis @ z  # (n_rf,)
    x_full = torch.zeros(n_pixels, dtype=dtype, device=device)
    x_full[rf_mask] = x_rf
    return x_full


def image_to_z(x_full, basis, offset, rf_mask):
    """Project a full image into subspace coordinates.

    z = basis^T @ (x_rf - offset)
    """
    x_rf = x_full[rf_mask]
    return basis.T @ (x_rf - offset)


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


# ============================================================================
# PCA decomposition
# ============================================================================

def compute_pca(X_images, rf_mask, var_threshold=0.95, n_components=None):
    """Compute PCA on RF-masked images.

    Args:
        X_images: (N, d) images on device (all available, not just training)
        rf_mask: (d,) boolean mask for RF pixels
        var_threshold: fraction of variance to retain (used if n_components is None)
        n_components: explicit number of components (overrides var_threshold)

    Returns:
        offset: (n_rf,) mean of images within RF (mu_rf)
        basis: (n_rf, K) top K eigenvectors (columns)
        eigenvalues_all: full eigenvalue spectrum (for plotting)
        K: number of retained components
        meta: dict with PCA-specific metadata
    """
    X_rf = X_images[:, rf_mask]
    n_samples, n_rf = X_rf.shape

    mu_rf = X_rf.mean(dim=0)  # (n_rf,)
    X_centered = X_rf - mu_rf.unsqueeze(0)

    # SVD for numerical stability: X_centered = U @ S @ V^T
    # cov = V @ (S^2 / N) @ V^T
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    eigenvalues_all = S ** 2 / n_samples
    V_all = Vt.T  # (n_rf, min(N, n_rf))

    total_var = eigenvalues_all.sum().item()
    cumvar = torch.cumsum(eigenvalues_all, dim=0) / total_var

    if n_components is not None:
        K = min(n_components, len(eigenvalues_all))
    else:
        above = (cumvar >= var_threshold).nonzero(as_tuple=True)[0]
        if len(above) > 0:
            K = above[0].item() + 1
        else:
            K = len(eigenvalues_all)

    basis = V_all[:, :K]  # (n_rf, K)
    var_explained = cumvar[K - 1].item()

    print(f"  PCA: n_samples={n_samples}, n_rf={n_rf}")
    print(f"  K={K} components, variance explained={var_explained:.4f} "
          f"(threshold={var_threshold})")
    print(f"  Eigenvalue range: [{eigenvalues_all[K-1].item():.6f}, "
          f"{eigenvalues_all[0].item():.6f}]")

    meta = {
        'var_explained': var_explained,
        'var_threshold': var_threshold,
        'n_components': n_components,
    }
    return mu_rf, basis, eigenvalues_all, K, meta


# ============================================================================
# C-eigenspace decomposition
# ============================================================================

def compute_c_eigenspace(kernel, rf_mask, eigen_rel_threshold, no_filter=False):
    """Eigendecompose the kernel's C matrix on masked pixels.

    Uses a RELATIVE threshold: keep eigenvalues > threshold * max_eigenvalue.

    If no_filter=True, keep ALL eigenvectors (K = n_rf).

    Returns:
        offset: (n_rf,) zeros (C-eigenspace has no mean offset)
        basis: (n_rf, K) top K eigenvectors (columns)
        eigenvalues_all: full eigenvalue spectrum (for plotting)
        K: number of retained dimensions
        meta: dict with C-eigen-specific metadata
    """
    with torch.no_grad():
        C, mask = kernel._compute_C_matrix(apply_mask=True)

    n_rf = C.shape[0]
    print(f"  C matrix shape: {C.shape}")
    print(f"  C matrix range: [{C.min().item():.6f}, {C.max().item():.6f}]")

    eigvals, eigvecs = torch.linalg.eigh(C)

    # Reverse to descending order
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)

    # Clamp negative eigenvalues (numerical noise)
    eigvals = eigvals.clamp(min=0.0)

    total_mass = eigvals.sum().item()
    max_eigval = eigvals[0].item()

    print(f"  Total eigenvalue mass: {total_mass:.4f}")
    print(f"  Max eigenvalue: {max_eigval:.6f}")
    print(f"  Top 10 eigenvalues: {eigvals[:10].cpu().numpy()}")
    print(f"  Bottom 5 eigenvalues: {eigvals[-5:].cpu().numpy()}")

    if no_filter:
        K = eigvals.shape[0]
        print(f"  NO FILTER: keeping all {K} eigenvectors")
    else:
        abs_threshold = eigen_rel_threshold * max_eigval
        K = int((eigvals > abs_threshold).sum().item())
        K = max(K, 1)
        print(f"  Relative threshold: {eigen_rel_threshold} * max = {abs_threshold:.6e}")

    cumsum = eigvals.cumsum(0) / total_mass
    print(f"  Dimensions kept (K): {K} out of {eigvals.shape[0]}")
    print(f"  Eigenvalue mass captured: {cumsum[K-1].item():.6f}")
    print(f"  Effective dimensionality reduction: {eigvals.shape[0]} -> {K} "
          f"({100 * K / eigvals.shape[0]:.1f}%)")

    basis = eigvecs[:, :K]  # (n_rf, K)
    offset = torch.zeros(n_rf, dtype=basis.dtype, device=basis.device)

    meta = {
        'eigen_rel_threshold': eigen_rel_threshold,
        'no_filter': no_filter,
    }
    return offset, basis, eigvals, K, meta


# ============================================================================
# Gradient ascent (unified for both methods)
# ============================================================================

def gradient_ascent(model, likelihood, z_start, x_samples,
                    basis, offset, rf_mask, r_max, f_max,
                    n_pixels, dtype, device, n_steps, lr,
                    z_max_norm=None, x_target_for_pearson=None):
    """LBFGS gradient ascent maximizing U_DA in subspace.

    The optimization variable is z in R^K. The image is x* = offset + basis @ z
    in RF-pixel space, then placed into full image for kernel evaluation.

    Supports:
    - Single-target (n_cond=1) and multi-conditioning (n_cond>1)
    - Chunked gradient accumulation for large n_cond (GPU memory)
    - Early stopping when utility converges
    - Optional norm constraint: ||z|| <= z_max_norm (projected gradient)

    Args:
        z_start: (K,) initial subspace coordinates
        x_samples: (n_cond, n_pixels) conditioning images for DA utility
        basis: (n_rf, K) subspace basis vectors
        offset: (n_rf,) mean offset (zeros for C-eigen, mu_rf for PCA)
        z_max_norm: if set, project z onto ||z|| <= z_max_norm after each step
        x_target_for_pearson: (n_pixels,) image to compute Pearson r against
            (single-target: the target, multi-cond: the start image)

    Returns:
        x_final: (n_pixels,) optimized image
        z_final: (K,) final subspace coordinates
        history: dict with per-step metrics
    """
    n_cond = x_samples.shape[0]

    z = z_start.clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [z], lr=lr,
        max_iter=LBFGS_MAX_ITER,
        max_eval=LBFGS_MAX_EVAL,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn='strong_wolfe',
    )

    history = {
        'step': [], 'utility': [], 'grad_norm_z': [],
        'z_norm': [], 'rf_norm': [], 'step_time': [],
    }
    if n_cond == 1:
        history['pearson_r'] = []
    else:
        history['pearson_r_vs_start'] = []
        history['firing_rate'] = []

    A = likelihood.A.squeeze()
    lam0 = likelihood.lambda0.squeeze()

    # Capture last closure utility to avoid expensive recomputation
    last_closure_utility = [None]

    # Evaluate utility at z_start BEFORE any LBFGS step, so history[0]
    # is the true starting-point utility (not after the first line search).
    with torch.no_grad():
        x_start_eval = z_to_image(z, basis, offset, rf_mask,
                                  n_pixels, dtype, device)
        result_start = distribution_aware_utility(
            model, likelihood,
            x_start_eval.unsqueeze(0), x_samples,
            r_max=r_max, adaptive_r_max=False, sample_lambda=False,
        )
        u_start = result_start['utility'].item()
        cur_rf_norm_start = x_start_eval[rf_mask].norm().item()

    history['step'].append(-1)  # sentinel: pre-optimization
    history['utility'].append(u_start)
    history['grad_norm_z'].append(0.0)
    history['z_norm'].append(z.detach().norm().item())
    history['rf_norm'].append(cur_rf_norm_start)
    history['step_time'].append(0.0)

    if n_cond == 1 and x_target_for_pearson is not None:
        history['pearson_r'].append(
            rf_pearson_r(x_start_eval, x_target_for_pearson, rf_mask))
    elif n_cond > 1 and x_target_for_pearson is not None:
        history['pearson_r_vs_start'].append(
            rf_pearson_r(x_start_eval, x_target_for_pearson, rf_mask))
        mu_s, _ = get_gp_marginal_moments(model, x_start_eval.unsqueeze(0))
        history['firing_rate'].append(
            torch.exp(A * mu_s + lam0).item())

    for step in range(n_steps):
        t_step = time.time()

        def closure():
            optimizer.zero_grad()
            x_full = z_to_image(z, basis, offset, rf_mask, n_pixels, dtype, device)

            if n_cond <= GRAD_CHUNK_SIZE:
                result = distribution_aware_utility(
                    model, likelihood,
                    x_full.unsqueeze(0),
                    x_samples,
                    r_max=r_max,
                    adaptive_r_max=False,
                    sample_lambda=False,
                )
                mu_g = result['mu_g_marg']
                if torch.exp(mu_g).item() > f_max:
                    return torch.tensor(float('inf'), device=device)
                loss = -result['utility'].squeeze()
                last_closure_utility[0] = result['utility'].item()
                loss.backward()
                return loss
            else:
                # Chunked gradient accumulation for large n_cond
                n_chunks = math.ceil(n_cond / GRAD_CHUNK_SIZE)
                total_loss_val = 0.0

                for chunk_idx in range(n_chunks):
                    c_start = chunk_idx * GRAD_CHUNK_SIZE
                    c_end = min(c_start + GRAD_CHUNK_SIZE, n_cond)
                    x_chunk = x_samples[c_start:c_end]

                    result = distribution_aware_utility(
                        model, likelihood,
                        x_full.unsqueeze(0),
                        x_chunk,
                        r_max=r_max,
                        adaptive_r_max=False,
                        sample_lambda=False,
                    )

                    if chunk_idx == 0:
                        mu_g = result['mu_g_marg']
                        if torch.exp(mu_g).item() > f_max:
                            return torch.tensor(float('inf'), device=device)

                    chunk_weight = (c_end - c_start) / n_cond
                    chunk_loss = -result['utility'].squeeze() * chunk_weight
                    total_loss_val += chunk_loss.item()

                    is_last = (chunk_idx == n_chunks - 1)
                    chunk_loss.backward(retain_graph=not is_last)

                last_closure_utility[0] = -total_loss_val
                return torch.tensor(total_loss_val, device=device)

        optimizer.step(closure)
        step_time = time.time() - t_step

        # Norm constraint (PCA projected gradient)
        if z_max_norm is not None:
            with torch.no_grad():
                current_norm = z.norm()
                if current_norm > z_max_norm:
                    z.mul_(z_max_norm / current_norm)

        # Verify gradient flow on first step
        if step == 0:
            if z.grad is None or z.grad.norm() == 0:
                raise RuntimeError(
                    f"No gradient at step 0: f_max guard likely rejected all "
                    f"closure evaluations. The starting image's predicted "
                    f"firing rate exceeds f_max={f_max}. Try a different "
                    f"starting point or increase f_max."
                )

        # Track metrics (using captured utility from last closure call)
        utility = last_closure_utility[0]
        with torch.no_grad():
            x_full = z_to_image(z, basis, offset, rf_mask, n_pixels, dtype, device)
            cur_rf_norm = x_full[rf_mask].norm().item()

            if n_cond == 1 and x_target_for_pearson is not None:
                pr = rf_pearson_r(x_full, x_target_for_pearson, rf_mask)
            elif n_cond > 1 and x_target_for_pearson is not None:
                pr_vs_start = rf_pearson_r(x_full, x_target_for_pearson, rf_mask)
                mu, sigma2 = get_gp_marginal_moments(model, x_full.unsqueeze(0))
                fr = torch.exp(A * mu + lam0).item()

        grad_norm_z = z.grad.norm().item() if z.grad is not None else 0.0

        history['step'].append(step)
        history['utility'].append(utility)
        history['grad_norm_z'].append(grad_norm_z)
        history['z_norm'].append(z.detach().norm().item())
        history['rf_norm'].append(cur_rf_norm)
        history['step_time'].append(step_time)

        if n_cond == 1 and x_target_for_pearson is not None:
            history['pearson_r'].append(pr)
        elif n_cond > 1 and x_target_for_pearson is not None:
            history['pearson_r_vs_start'].append(pr_vs_start)
            history['firing_rate'].append(fr)

        # NaN check
        if np.isnan(utility) or np.isnan(grad_norm_z):
            raise RuntimeError(
                f"NaN detected at step {step}. LBFGS line search likely "
                f"proposed a step that caused firing rate overflow "
                f"(arc_cosine kernel: k(x,x) ~ ||Cx||^2). "
                f"Try RBF kernel or a different starting point."
            )

        # Early stopping: utility converged
        if step >= 5:
            recent = history['utility'][-5:]
            if max(recent) - min(recent) < 1e-8:
                print(f"  step {step}: utility converged (no change for 5 steps) "
                      f"- stopping early")
                break

        if step % LOG_EVERY == 0 or step == n_steps - 1:
            elapsed_steps = step + 1
            avg_time = sum(history['step_time']) / elapsed_steps
            eta = avg_time * (n_steps - elapsed_steps)

            if n_cond == 1:
                pr_str = f"r={pr:.4f}  " if x_target_for_pearson is not None else ""
                print(f"  step {step:4d}: U={utility:.6f}  "
                      f"{pr_str}|z|={z.detach().norm().item():.4f}  "
                      f"|grad_z|={grad_norm_z:.4e}  "
                      f"[{step_time:.1f}s, ETA {eta:.0f}s]")
            else:
                pr_str = (f"r_start={pr_vs_start:.4f}  "
                          if x_target_for_pearson is not None else "")
                fr_str = (f"FR={fr:.1f}  "
                          if x_target_for_pearson is not None else "")
                print(f"  step {step:4d}: U={utility:.6f}  "
                      f"{pr_str}||x||_RF={cur_rf_norm:.2f}  "
                      f"{fr_str}|grad_z|={grad_norm_z:.4e}  "
                      f"[{step_time:.1f}s, ETA {eta:.0f}s]")

    with torch.no_grad():
        x_final = z_to_image(z, basis, offset, rf_mask, n_pixels, dtype, device)
        z_final = z.detach().clone()

    return x_final.detach(), z_final, history


# ============================================================================
# Plotting helpers (shared between single-target and multi-cond)
# ============================================================================

def _setup_plot_helpers(rf_mask, kernel, n_px_side, vmin, vmax):
    """Create helper functions and data for plotting.

    Returns dict with: mask_2d, rf_mask_np, crop_bounds, gray_val,
    and closures: masked_crop, check_oob, draw_rf_overlay.
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
    gray_val = (vmin + vmax) / 2

    def masked_crop(x_flat, gv=gray_val):
        img = x_flat.detach().cpu().numpy().reshape(n_px_side, n_px_side).copy()
        img[~mask_2d] = gv
        return img[r_min:r_max_px+1, c_min:c_max_px+1]

    def check_oob(x_flat):
        vals = x_flat.detach().cpu().numpy()[rf_mask_np]
        n_below = (vals < vmin).sum()
        n_above = (vals > vmax).sum()
        n_oob = n_below + n_above
        pct = 100.0 * n_oob / len(vals)
        return pct, float(vals.min()), float(vals.max())

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

    return {
        'masked_crop': masked_crop,
        'check_oob': check_oob,
        'draw_rf_overlay': draw_rf_overlay,
    }


# ============================================================================
# Visualization: single-target mode
# ============================================================================

def plot_results_single(x_target, x_target_proj, x_start, x_final,
                        history, eigenvalues_all, K, method, method_meta,
                        target_index, u_target_orig, u_target_proj,
                        rf_mask, kernel, config, vmin, vmax, test_r,
                        n_px_side, out_path):
    """Summary figure for single-target subspace optimization.

    Top row: target original, target projected, start, final.
    Bottom left: eigenvalue spectrum. Bottom right: utility + Pearson r.
    """
    ph = _setup_plot_helpers(rf_mask, kernel, n_px_side, vmin, vmax)
    masked_crop = ph['masked_crop']
    check_oob = ph['check_oob']
    draw_rf_overlay = ph['draw_rf_overlay']

    fig = plt.figure(figsize=(20, 10))

    # --- Top row: 4 images ---
    u_start = history['utility'][0]
    u_final = history['utility'][-1]

    x_tensors = [x_target, x_target_proj, x_start, x_final]
    cropped_images = [masked_crop(x) for x in x_tensors]

    method_label = 'PCA' if method == 'pca' else 'C-eigen'
    base_titles = [
        f'Target (original, pool[{target_index}])\nU_DA={u_target_orig:.4f}',
        f'Target ({method_label}, K={K})\nU_DA={u_target_proj:.4f}',
        f'Start\nU_DA={u_start:.4f}',
        f'Final ({method_label} opt)\nU_DA={u_final:.4f}',
    ]

    for i, (img, x_flat, title) in enumerate(
            zip(cropped_images, x_tensors, base_titles)):
        pct_oob, lo, hi = check_oob(x_flat)
        is_oob = pct_oob > 0
        if is_oob:
            title += f'\nCLIPPED {pct_oob:.0f}% OOB [{lo:.2f},{hi:.2f}]'

        ax = fig.add_subplot(2, 4, i + 1)
        im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(title, fontsize=10,
                     color='red' if is_oob else 'black')
        ax.axis('off')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        draw_rf_overlay(ax)

    # --- Bottom left: Eigenvalue spectrum ---
    ax_eig = fig.add_subplot(2, 4, 5)
    eigvals_np = eigenvalues_all.cpu().numpy()
    ax_eig.semilogy(eigvals_np, 'b-', linewidth=1.5)
    # Build threshold label
    if method == 'pca':
        vt = method_meta.get('var_threshold', DEFAULT_VAR_THRESHOLD)
        ve = method_meta.get('var_explained', 0.0)
        threshold_label = f'K={K} (var={ve:.2f}, thr={vt:.2f})'
    else:
        if method_meta.get('no_filter', False):
            threshold_label = f'K={K} (no filter)'
        else:
            et = method_meta.get('eigen_rel_threshold', EIGEN_REL_THRESHOLD)
            threshold_label = f'K={K} (thresh={et:.0e})'
    ax_eig.axvline(K - 1, color='red', linestyle='--', linewidth=1,
                   label=threshold_label)
    ax_eig.set_xlabel('Component index')
    ax_eig.set_ylabel('Eigenvalue (log scale)')
    spectrum_name = 'PCA' if method == 'pca' else 'C matrix'
    ax_eig.set_title(f'{spectrum_name} eigenvalue spectrum')
    ax_eig.legend(fontsize=9)
    ax_eig.grid(True, alpha=0.3)

    # --- Bottom right: Utility + Pearson r ---
    ax_util = fig.add_subplot(2, 4, (6, 8))
    steps = history['step']
    color_u = 'tab:blue'
    color_r = 'tab:green'

    ax_util.plot(steps, history['utility'], color=color_u, linewidth=1.5,
                 label='U_DA')
    ax_util.set_xlabel('Step')
    ax_util.set_ylabel('U_DA', color=color_u)
    ax_util.tick_params(axis='y', labelcolor=color_u)

    if 'pearson_r' in history and len(history['pearson_r']) > 0:
        ax_r = ax_util.twinx()
        ax_r.plot(steps, history['pearson_r'], color=color_r, linewidth=1.5,
                  alpha=0.7, label='Pearson r vs target (RF)')
        ax_r.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
        ax_r.set_ylabel('Pearson r', color=color_r)
        ax_r.tick_params(axis='y', labelcolor=color_r)
        lines1, labels1 = ax_util.get_legend_handles_labels()
        lines2, labels2 = ax_r.get_legend_handles_labels()
        ax_util.legend(lines1 + lines2, labels1 + labels2,
                       loc='center left', fontsize=9)

    ax_util.set_title(f'LBFGS convergence ({method_label})')
    ax_util.grid(True, alpha=0.3)

    # Suptitle
    n_rf = rf_mask.sum().item()
    if method == 'pca':
        vt = method_meta.get('var_threshold', DEFAULT_VAR_THRESHOLD)
        ve = method_meta.get('var_explained', 0.0)
        method_str = f'PCA (var_thr={vt:.2f}, var_expl={ve:.3f})'
    else:
        if method_meta.get('no_filter', False):
            method_str = 'C-eigen (no filter)'
        else:
            et = method_meta.get('eigen_rel_threshold', EIGEN_REL_THRESHOLD)
            method_str = f'C-eigen (thresh={et:.0e})'

    title_str = (
        f'{method_str} DA Utility Optimization  '
        f'K={K}/{n_rf}  '
        f'M={config["M"]}, n_train={config["n_train"]}, '
        f'kernel={config.get("kernel_type", "?")}, '
        f'test_r={test_r:.3f}'
    )
    fig.suptitle(title_str, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ============================================================================
# Visualization: multi-conditioning mode
# ============================================================================

def plot_results_multicond(x_start, x_final, history, eigenvalues_all, K,
                           method, method_meta, n_cond,
                           rf_mask, kernel, config, vmin, vmax, test_r,
                           n_px_side, out_path,
                           start_index=None):
    """Summary figure for multi-conditioning subspace optimization.

    Top row: start, final, diff, text summary.
    Bottom: eigenvalue spectrum + convergence curves.
    """
    ph = _setup_plot_helpers(rf_mask, kernel, n_px_side, vmin, vmax)
    masked_crop = ph['masked_crop']
    check_oob = ph['check_oob']
    draw_rf_overlay = ph['draw_rf_overlay']

    fig = plt.figure(figsize=(20, 10))

    u_start = history['utility'][0]
    u_final = history['utility'][-1]

    # --- Top row: images + text summary ---
    x_tensors = [x_start, x_final, x_final - x_start]
    cropped_images = [masked_crop(x) for x in x_tensors]
    start_label = (f'Start (pool #{start_index})'
                   if start_index is not None else 'Start')
    method_label = 'PCA' if method == 'pca' else 'C-eigen'
    base_titles = [
        f'{start_label}\nU_DA={u_start:.4f}',
        f'Final ({method_label} opt)\nU_DA={u_final:.4f}',
        f'Final - Start',
    ]
    use_clamp = [True, True, False]

    for i, (img, x_flat, title) in enumerate(
            zip(cropped_images, x_tensors, base_titles)):
        pct_oob, lo, hi = check_oob(x_flat)
        is_oob = pct_oob > 0
        if is_oob and use_clamp[i]:
            title += f'\nCLIPPED {pct_oob:.0f}% OOB [{lo:.2f},{hi:.2f}]'

        ax = fig.add_subplot(2, 4, i + 1)
        v = (vmin, vmax) if use_clamp[i] else (None, None)
        im = ax.imshow(img, cmap='gray', vmin=v[0], vmax=v[1], aspect='equal')
        ax.set_title(title, fontsize=10,
                     color='red' if (is_oob and use_clamp[i]) else 'black')
        ax.axis('off')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        if use_clamp[i]:
            draw_rf_overlay(ax)

    # Panel 4: Text summary
    ax = fig.add_subplot(2, 4, 4)
    ax.axis('off')
    total_time = sum(history['step_time'])
    avg_step = total_time / len(history['step_time'])
    final_fr = (history['firing_rate'][-1]
                if 'firing_rate' in history else 'N/A')
    final_rf_norm = history['rf_norm'][-1]
    final_oob, _, _ = check_oob(x_final)
    summary_lines = [
        f'Method: {method_label}',
        f'n_cond = {n_cond}',
        f'n_steps = {len(history["step"])}',
        f'',
        f'U_DA: {u_start:.6f} -> {u_final:.6f}',
        f'delta_U: {u_final - u_start:.6f}',
        f'',
        f'||x||_RF: {history["rf_norm"][0]:.2f} -> {final_rf_norm:.2f}',
        (f'Firing rate: {final_fr:.1f}'
         if isinstance(final_fr, float) else f'Firing rate: {final_fr}'),
        f'OOB pixels: {final_oob:.1f}%',
        f'',
        f'Total time: {total_time:.0f}s ({avg_step:.1f}s/step)',
        f'test_r: {test_r:.3f}',
    ]
    ax.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    # --- Bottom left: Eigenvalue spectrum ---
    ax_eig = fig.add_subplot(2, 4, 5)
    eigvals_np = eigenvalues_all.cpu().numpy()
    ax_eig.semilogy(eigvals_np, 'b-', linewidth=1.5)
    if method == 'pca':
        vt = method_meta.get('var_threshold', DEFAULT_VAR_THRESHOLD)
        ve = method_meta.get('var_explained', 0.0)
        threshold_label = f'K={K} (var={ve:.2f}, thr={vt:.2f})'
    else:
        if method_meta.get('no_filter', False):
            threshold_label = f'K={K} (no filter)'
        else:
            et = method_meta.get('eigen_rel_threshold', EIGEN_REL_THRESHOLD)
            threshold_label = f'K={K} (thresh={et:.0e})'
    ax_eig.axvline(K - 1, color='red', linestyle='--', linewidth=1,
                   label=threshold_label)
    ax_eig.set_xlabel('Component index')
    ax_eig.set_ylabel('Eigenvalue (log scale)')
    spectrum_name = 'PCA' if method == 'pca' else 'C matrix'
    ax_eig.set_title(f'{spectrum_name} eigenvalue spectrum')
    ax_eig.legend(fontsize=9)
    ax_eig.grid(True, alpha=0.3)

    # --- Bottom right: Utility + Pearson r vs start ---
    ax_util = fig.add_subplot(2, 4, (6, 8))
    steps = history['step']
    color_u = 'tab:blue'
    color_r = 'tab:green'

    ax_util.plot(steps, history['utility'], color=color_u, linewidth=1.5,
                 label='U_DA')
    ax_util.set_xlabel('Step')
    ax_util.set_ylabel('U_DA', color=color_u)
    ax_util.tick_params(axis='y', labelcolor=color_u)

    if 'pearson_r_vs_start' in history and len(history['pearson_r_vs_start']) > 0:
        ax_r = ax_util.twinx()
        ax_r.plot(steps, history['pearson_r_vs_start'], color=color_r,
                  linewidth=1.5, alpha=0.7, label='Pearson r vs start (RF)')
        ax_r.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
        ax_r.set_ylabel('Pearson r', color=color_r)
        ax_r.tick_params(axis='y', labelcolor=color_r)
        lines1, labels1 = ax_util.get_legend_handles_labels()
        lines2, labels2 = ax_r.get_legend_handles_labels()
        ax_util.legend(lines1 + lines2, labels1 + labels2,
                       loc='center left', fontsize=9)

    ax_util.set_title('LBFGS convergence')
    ax_util.grid(True, alpha=0.3)

    # Suptitle
    n_rf = rf_mask.sum().item()
    if method == 'pca':
        vt = method_meta.get('var_threshold', DEFAULT_VAR_THRESHOLD)
        method_str = f'PCA (var_thr={vt:.2f})'
    else:
        if method_meta.get('no_filter', False):
            method_str = 'C-eigen (no filter)'
        else:
            et = method_meta.get('eigen_rel_threshold', EIGEN_REL_THRESHOLD)
            method_str = f'C-eigen (thresh={et:.0e})'

    title_str = (
        f'Multi-Cond {method_str} DA Utility Optimization (N={n_cond})  '
        f'K={K}/{n_rf}  M={config["M"]}, n_train={config["n_train"]}, '
        f'kernel={config.get("kernel_type", "?")}, '
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
        description='Subspace-constrained DA utility gradient ascent (PCA or C-eigenspace)')
    parser.add_argument('--method', type=str, default='pca',
                        choices=['pca', 'c_eigen'],
                        help='Subspace method (default: pca)')
    # PCA-specific
    parser.add_argument('--var-threshold', type=float, default=DEFAULT_VAR_THRESHOLD,
                        help=f'Variance explained threshold for PCA (default: {DEFAULT_VAR_THRESHOLD})')
    parser.add_argument('--n-components', type=int, default=None,
                        help='Explicit number of PCA components (overrides --var-threshold)')
    # C-eigen-specific
    parser.add_argument('--eigen-threshold', type=float, default=None,
                        help=f'Relative eigenvalue threshold for C-eigen '
                             f'(fraction of max, default: {EIGEN_REL_THRESHOLD})')
    parser.add_argument('--no-filter', action='store_true',
                        help='C-eigen: keep ALL eigenvectors (no truncation)')
    # Shared
    parser.add_argument('--kernel-type', type=str, default=None,
                        choices=['arc_cosine', 'arc_sine', 'rbf'],
                        help='Kernel type (default: from default_params.json)')
    parser.add_argument('--target-index', type=int, default=TARGET_INDEX,
                        help=f'Pool image index for target (default: {TARGET_INDEX})')
    parser.add_argument('--M', type=int, default=M_OVERRIDE,
                        help=f'Number of inducing points (default: {M_OVERRIDE})')
    parser.add_argument('--n-train', type=int, default=N_TRAIN_OVERRIDE,
                        help=f'Number of training points (default: {N_TRAIN_OVERRIDE})')
    # Multi-conditioning
    parser.add_argument('--n-cond', type=int, default=N_COND,
                        help=f'Number of conditioning images '
                             f'(default: {N_COND}, 1 = single target)')
    parser.add_argument('--cond-seed', type=int, default=COND_SEED,
                        help=f'Seed for conditioning image selection (default: {COND_SEED})')
    parser.add_argument('--start-index', type=int, default=None,
                        help='Pool image index for starting point (multi-cond)')
    parser.add_argument('--start-seed', type=int, default=START_SEED,
                        help=f'Seed for random start image (default: {START_SEED})')
    args = parser.parse_args()

    t0 = time.time()
    method = args.method
    n_cond = args.n_cond
    eigen_rel_threshold = (args.eigen_threshold
                           if args.eigen_threshold is not None
                           else EIGEN_REL_THRESHOLD)

    method_label = 'PCA' if method == 'pca' else 'C-eigenspace'

    # === Step 1: Train model ===
    print("=" * 70)
    print(f"Step 1: Training model (default_gpy, M={args.M}, "
          f"n_train={args.n_train})")
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
    print("Step 2: RF mask")
    print("=" * 70)

    kernel = (getattr(model, 'covar_module', None)
              or getattr(model, 'kernel', None))
    if not (hasattr(kernel, '_cached_mask') and kernel._cached_mask is not None):
        with torch.no_grad():
            _ = model(X_pool[0].unsqueeze(0))
    rf_mask = kernel._cached_mask.squeeze()
    n_rf = rf_mask.sum().item()
    print(f"  RF mask: {n_rf} active pixels out of {n_pixels}")

    # Dataset pixel bounds for visualization
    rf_mask_np = rf_mask.cpu().numpy()
    X_all = torch.cat([X_pool, X_train], dim=0)
    all_rf_vals = X_all.cpu().numpy()[:, rf_mask_np]
    vmin_dataset = float(all_rf_vals.min())
    vmax_dataset = float(all_rf_vals.max())
    print(f"  Dataset RF pixel range: [{vmin_dataset:.3f}, {vmax_dataset:.3f}]")

    # === Step 3: Subspace decomposition ===
    print("\n" + "=" * 70)
    print(f"Step 3: {method_label} decomposition")
    print("=" * 70)

    if method == 'pca':
        offset, basis, eigenvalues_all, K, method_meta = compute_pca(
            X_all, rf_mask,
            var_threshold=args.var_threshold,
            n_components=args.n_components,
        )
    else:  # c_eigen
        offset, basis, eigenvalues_all, K, method_meta = compute_c_eigenspace(
            kernel, rf_mask, eigen_rel_threshold,
            no_filter=args.no_filter,
        )

    # === Step 4: Create conditioning images and starting point ===
    print("\n" + "=" * 70)
    print("Step 4: Setting up images")
    print("=" * 70)

    x_target = None       # only set for single-target
    x_target_proj = None   # target projected into subspace
    z_max_norm = None      # norm constraint (PCA single-target only)
    actual_start_index = None

    if n_cond == 1:
        # --- Single-target mode ---
        x_target = X_pool[args.target_index]
        x_samples = x_target.unsqueeze(0)  # (1, n_pixels)

        # Project target into subspace for visualization
        z_target = image_to_z(x_target, basis, offset, rf_mask)
        x_target_proj = z_to_image(z_target, basis, offset, rf_mask,
                                   n_pixels, dtype, device)

        recon_err = (x_target[rf_mask] - x_target_proj[rf_mask]).norm().item()
        target_norm = x_target[rf_mask].norm().item()
        print(f"  Mode: single-target")
        print(f"  Target: pool image {args.target_index}")
        print(f"  Reconstruction error: {recon_err:.4f} "
              f"(relative: {recon_err / target_norm:.4f})")

        if method == 'pca':
            # PCA: start from mean (z=0), with norm constraint
            z_start = torch.zeros(K, dtype=dtype, device=device)
            x_start = z_to_image(z_start, basis, offset, rf_mask,
                                 n_pixels, dtype, device)

            # Compute norm constraint from training data
            z_train_norms = []
            for i in range(X_train.shape[0]):
                z_i = image_to_z(X_train[i], basis, offset, rf_mask)
                z_train_norms.append(z_i.norm().item())
            z_train_norms = np.array(z_train_norms)
            z_max_norm = float(np.percentile(z_train_norms, 95))

            print(f"  Start: dataset mean (z=0)")
            print(f"  Norm constraint: ||z|| <= {z_max_norm:.2f} "
                  f"(95th percentile of training)")
            print(f"  Training z-norms: mean={z_train_norms.mean():.2f}, "
                  f"std={z_train_norms.std():.2f}, "
                  f"max={z_train_norms.max():.2f}")
        else:
            # C-eigen: start from smoothed target projected into eigenspace
            x_target_2d = x_target.cpu().numpy().reshape(n_px_side, n_px_side)
            x_smoothed_2d = gaussian_filter(x_target_2d, sigma=SIGMA_SMOOTH)
            x_smoothed = torch.tensor(
                x_smoothed_2d.reshape(-1), dtype=dtype, device=device)
            z_start = image_to_z(x_smoothed, basis, offset, rf_mask)
            x_start = z_to_image(z_start, basis, offset, rf_mask,
                                 n_pixels, dtype, device)

            pixel_dist = (x_start - x_target).norm().item()
            rf_dist = (x_start[rf_mask] - x_target[rf_mask]).norm().item()
            print(f"  Start: Gaussian smoothing sigma={SIGMA_SMOOTH}")
            print(f"  pixel_dist={pixel_dist:.4f}, rf_dist={rf_dist:.4f}")

    else:
        # --- Multi-conditioning mode ---
        n_pool = X_pool.shape[0]
        rng = torch.Generator(device='cpu').manual_seed(args.cond_seed)
        cond_indices = torch.randperm(n_pool, generator=rng)[:n_cond]
        x_samples = X_pool[cond_indices]  # (n_cond, n_pixels)

        # Select starting image
        if args.start_index is not None:
            actual_start_index = args.start_index
            x_start_raw = X_pool[args.start_index].clone()
        else:
            start_rng = torch.Generator(device='cpu').manual_seed(args.start_seed)
            cond_set = set(cond_indices.tolist())
            remaining = sorted(set(range(n_pool)) - cond_set)
            rand_idx = remaining[
                torch.randint(len(remaining), (1,), generator=start_rng).item()
            ]
            actual_start_index = rand_idx
            x_start_raw = X_pool[rand_idx].clone()

        x_start_raw = x_start_raw.to(device=device, dtype=dtype)

        # Project start into subspace and reconstruct
        z_start = image_to_z(x_start_raw, basis, offset, rf_mask)
        x_start = z_to_image(z_start, basis, offset, rf_mask,
                              n_pixels, dtype, device)

        overlap = set(cond_indices.tolist()) & {actual_start_index}
        print(f"  Mode: multi-conditioning (n_cond={n_cond})")
        print(f"  Conditioning: {n_cond} images from pool "
              f"(seed={args.cond_seed})")
        print(f"  Start: pool image {actual_start_index}"
              f"{'  (user-specified)' if args.start_index is not None else f'  (random, seed={args.start_seed})'}")
        print(f"  Start in conditioning set: "
              f"{'YES' if overlap else 'no'}")

    print(f"  Dataset RF pixel range: [{vmin_dataset:.3f}, {vmax_dataset:.3f}]")

    # Firing rate sanity check
    with torch.no_grad():
        mu_start, _ = get_gp_marginal_moments(model, x_start.unsqueeze(0))
        A_check = likelihood.A.squeeze()
        lam0_check = likelihood.lambda0.squeeze()
        fr_start = torch.exp(A_check * mu_start + lam0_check).item()
    if fr_start > f_max or np.isnan(fr_start):
        raise RuntimeError(
            f"Starting image has firing rate {fr_start:.1f} (f_max={f_max}). "
            f"Cannot optimize — utility will be NaN. "
            f"Arc-cosine kernel amplifies image norm via k(x,x) ~ ||Cx||^2. "
            f"Try a different starting point or use RBF kernel."
        )
    print(f"  Starting image firing rate: {fr_start:.2f} (f_max={f_max})")

    # === Step 5: Gradient ascent ===
    print("\n" + "=" * 70)
    print(f"Step 5: LBFGS gradient ascent in {method_label} space "
          f"(K={K}, n_cond={n_cond})")
    print("=" * 70)

    # Pearson r reference: target for single-target, start for multi-cond
    if n_cond == 1:
        pearson_ref = x_target
    else:
        pearson_ref = x_start

    x_final, z_final, history = gradient_ascent(
        model, likelihood, z_start, x_samples,
        basis, offset, rf_mask, r_max, f_max,
        n_pixels, dtype, device, N_STEPS, LR,
        z_max_norm=z_max_norm,
        x_target_for_pearson=pearson_ref,
    )

    # === Step 6: Diagnostics ===
    print(f"\nGradient ascent summary:")
    print(f"  Steps: {len(history['step'])}")
    print(f"  U_DA: {history['utility'][0]:.6f} -> {history['utility'][-1]:.6f}")
    if n_cond == 1 and 'pearson_r' in history:
        print(f"  Pearson r (RF): {history['pearson_r'][0]:.4f} -> "
              f"{history['pearson_r'][-1]:.4f}")
    if n_cond > 1:
        if 'pearson_r_vs_start' in history:
            print(f"  Pearson r vs start (RF): "
                  f"{history['pearson_r_vs_start'][0]:.4f} -> "
                  f"{history['pearson_r_vs_start'][-1]:.4f}")
        print(f"  ||x||_RF: {history['rf_norm'][0]:.2f} -> "
              f"{history['rf_norm'][-1]:.2f}")
        if 'firing_rate' in history:
            print(f"  Firing rate: {history['firing_rate'][0]:.1f} -> "
                  f"{history['firing_rate'][-1]:.1f}")

    print(f"  |z|: {history['z_norm'][0]:.4f} -> {history['z_norm'][-1]:.4f}")

    # PCA space analysis (single-target only)
    if n_cond == 1:
        z_target = image_to_z(x_target, basis, offset, rf_mask)
        z_similarity = torch.nn.functional.cosine_similarity(
            z_final.unsqueeze(0), z_target.unsqueeze(0)
        ).item()
        print(f"\n  Subspace analysis:")
        print(f"    z_target: |z|={z_target.norm().item():.4f}")
        print(f"    z_final:  |z|={z_final.norm().item():.4f}")
        print(f"    cosine_similarity(z_final, z_target) = {z_similarity:.4f}")

    # Pixel bounds check
    x_final_rf = x_final[rf_mask]
    final_min = x_final_rf.min().item()
    final_max = x_final_rf.max().item()
    n_below = (x_final_rf < vmin_dataset).sum().item()
    n_above = (x_final_rf > vmax_dataset).sum().item()
    n_oob = n_below + n_above
    pct_oob = 100.0 * n_oob / n_rf
    print(f"\n  Pixel bounds check (dataset RF range: "
          f"[{vmin_dataset:.3f}, {vmax_dataset:.3f}]):")
    print(f"    Final image RF range: [{final_min:.3f}, {final_max:.3f}]")
    print(f"    Out-of-bounds pixels: {n_oob}/{n_rf} ({pct_oob:.1f}%)"
          f"  [{n_below} below, {n_above} above]")

    # Firing rate diagnostics
    print(f"\n  Firing rate diagnostics (f_max={f_max})")
    A = likelihood.A.squeeze()
    lam0 = likelihood.lambda0.squeeze()
    with torch.no_grad():
        if n_cond == 1:
            diag_imgs = [('target', x_target), ('start', x_start),
                         ('final', x_final)]
        else:
            diag_imgs = [('start', x_start), ('final', x_final)]
        for label, x_img in diag_imgs:
            mu, sigma2 = get_gp_marginal_moments(model, x_img.unsqueeze(0))
            mu_g = A * mu + lam0
            firing_rate = torch.exp(mu_g).item()
            H = compute_H(mu, sigma2, r_max=r_max, a=A, lambda0=lam0).item()
            flag = " ** EXCEEDS f_max" if firing_rate > f_max else ""
            print(f"    {label:8s}: lambda_m={mu.item():.4f}, "
                  f"mu_g={mu_g.item():.4f}, firing_rate={firing_rate:.2f}, "
                  f"H_marg={H:.6f}, ||x||_RF={x_img[rf_mask].norm().item():.2f}"
                  f"{flag}")

    # === Step 7: Visualization ===
    print("\n" + "=" * 70)
    print("Step 7: Visualization")
    print("=" * 70)

    # Build output filename
    kernel_suffix = f'_{kernel_type}' if kernel_type else ''
    cond_suffix = f'_cond{n_cond}' if n_cond > 1 else ''
    if method == 'pca':
        threshold_suffix = f'_vt{args.var_threshold:.2f}'
    else:
        if args.no_filter:
            threshold_suffix = '_nofilter'
        else:
            threshold_suffix = f'_thresh{eigen_rel_threshold:.0e}'
    out_name = f'subspace_{method}{kernel_suffix}{cond_suffix}{threshold_suffix}.png'

    if n_cond == 1:
        # Compute utility for target (original and projected)
        with torch.no_grad():
            result_target_orig = distribution_aware_utility(
                model, likelihood,
                x_target.unsqueeze(0), x_target.unsqueeze(0),
                r_max=r_max, adaptive_r_max=False, sample_lambda=False,
            )
            u_target_orig = result_target_orig['utility'].item()

            result_target_proj = distribution_aware_utility(
                model, likelihood,
                x_target_proj.unsqueeze(0), x_target.unsqueeze(0),
                r_max=r_max, adaptive_r_max=False, sample_lambda=False,
            )
            u_target_proj = result_target_proj['utility'].item()

        print(f"  Utility at target images:")
        print(f"    target (original):       U_DA={u_target_orig:.6f}")
        print(f"    target ({method_label} proj): U_DA={u_target_proj:.6f}")

        plot_results_single(
            x_target, x_target_proj, x_start, x_final,
            history, eigenvalues_all, K, method, method_meta,
            args.target_index, u_target_orig, u_target_proj,
            rf_mask, kernel, config,
            vmin_dataset, vmax_dataset, test_r,
            n_px_side,
            out_path=_script_dir / out_name,
        )
    else:
        plot_results_multicond(
            x_start, x_final, history, eigenvalues_all, K,
            method, method_meta, n_cond,
            rf_mask, kernel, config,
            vmin_dataset, vmax_dataset, test_r,
            n_px_side,
            out_path=_script_dir / out_name,
            start_index=actual_start_index,
        )

    elapsed = time.time() - t0
    print(f"\nTotal elapsed time: {elapsed:.1f}s")

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Method: {method_label}")
    print(f"  Kernel: {kernel_type}")
    print(f"  Conditioning: n_cond={n_cond}")
    print(f"  Subspace: K={K} out of {n_rf} ({100*K/n_rf:.1f}%)")
    if method == 'pca':
        ve = method_meta.get('var_explained', 0.0)
        print(f"  Variance explained: {ve:.4f} "
              f"(threshold={args.var_threshold})")
    else:
        if args.no_filter:
            print(f"  Eigenvalue filter: NONE (all eigenvectors kept)")
        else:
            print(f"  Eigenvalue relative threshold: {eigen_rel_threshold:.0e}")
    print(f"  Final utility: {history['utility'][-1]:.6f}")
    if n_cond == 1 and 'pearson_r' in history:
        print(f"  Final Pearson r to target: {history['pearson_r'][-1]:.4f}")
    print(f"  Output: {out_name}")


if __name__ == '__main__':
    main()
