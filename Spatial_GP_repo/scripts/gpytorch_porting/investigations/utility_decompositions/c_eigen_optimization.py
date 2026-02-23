"""
C-eigenspace utility optimization.

Optimizes image x* to maximize distribution-aware utility, where x* is
parameterized in the dominant eigenspace of the kernel's C matrix:

    C = U Gamma U^T    (eigendecomposition of C from trained kernel)
    Keep U_K = eigenvectors with gamma_i > threshold
    x* = U_K @ z       (z in R^K)
    Gradient: nabla_z U = U_K^T @ nabla_x* U

Mathematical insight (from pca_vs_c_eigenspace.tex):
  C-eigenspace optimization does NOT change the solution compared to
  pixel-space optimization -- it only improves conditioning by removing
  directions where the gradient is already near-zero (proportional to
  gamma_i). This is a reformulation of what gradient ascent already does
  implicitly.

Simplifications:
  - Single target image (pool image 0)
  - Start from smoothed version of target projected into C-eigenspace
  - Optimization in RF-masked pixel space only (~2500 pixels)
  - C matrix computed on masked pixels (not full 11664)
  - Eigenvalue threshold: keep eigenvectors with eigenvalue > fraction
    of max eigenvalue (default 1e-6). The eigenvalue mass threshold
    (e.g., 95%) is not useful because the first eigenvalue contains
    99%+ of the mass due to the locality mask alpha in C.
  - No pixel bounds (unconstrained optimization)
  - sample_lambda=False in DA utility (deterministic, no MC noise)
  - M=50, n_train=50 (smaller model for wider RF and more interesting
    eigenstructure; M=300 gives an extremely tight RF with only ~72
    masked pixels)

Model training via setup() from explore_utility.py (same folder as gradient.py).
All model params from default_params.json via build_config_from_defaults().

Usage:
    # Single-target mode (legacy, default):
    python investigations/utility_decompositions/c_eigen_optimization.py

    # Custom relative eigenvalue threshold:
    python investigations/utility_decompositions/c_eigen_optimization.py --eigen-threshold 1e-4

    # Multi-conditioning mode (N=300 pool images, random start):
    python investigations/utility_decompositions/c_eigen_optimization.py --n-cond 300

    # Multi-conditioning with RBF kernel:
    python investigations/utility_decompositions/c_eigen_optimization.py --n-cond 300 --kernel-type rbf
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

# Import setup from explore_utility.py (in utility/ folder, not this folder)
_utility_dir = _script_dir.parent / 'utility'
sys.path.insert(0, str(_utility_dir))
from explore_utility import setup

# ---------------------------------------------------------------------------
# Investigation-specific constants (not model parameters)
# These are LBFGS optimization tuning knobs for THIS investigation.
# ---------------------------------------------------------------------------
N_STEPS = 50             # outer LBFGS steps
LR = 0.5                 # LBFGS step size
LBFGS_MAX_ITER = 20      # max iterations per LBFGS step (line search evals)
LBFGS_MAX_EVAL = 25      # max function evaluations per LBFGS step
LBFGS_HISTORY_SIZE = 10  # number of past gradients for Hessian approximation
LOG_EVERY = 1            # print every step

# --- Image creation ---
TARGET_INDEX = 5         # pool image index for natural target
SIGMA_SMOOTH = 5.0       # Gaussian smoothing sigma for perturbation

# --- Multi-conditioning mode ---
N_COND = 1               # 1 = legacy single-target mode. >1 = multi-conditioning.
COND_SEED = 42           # seed for reproducible conditioning sample selection
START_SEED = 123         # seed for random start image selection (multi-cond only)
GRAD_CHUNK_SIZE = 30     # n_cond images per gradient accumulation chunk (GPU memory)

# --- C-eigenspace ---
# Relative threshold: keep eigenvalues > EIGEN_REL_THRESHOLD * max_eigenvalue.
# Mass-based thresholds (e.g., 95%) are useless here because the first
# eigenvalue of C contains 99%+ of the total mass due to the locality mask.
EIGEN_REL_THRESHOLD = 1e-3   # relative to largest eigenvalue

# --- Model size (investigation overrides) ---
# Smaller than explore_utility.py's M=300/N_TRAIN=300 to get a wider RF mask
# and more interesting eigenstructure.
M_OVERRIDE = 50
N_TRAIN_OVERRIDE = 50


# ============================================================================
# Helper functions
# ============================================================================

def _reconstruct_image_from_rf(x_rf, rf_mask, n_pixels, dtype, device):
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
    """Projection coefficient: scalar s minimizing ||x - s*target|| within RF."""
    a = x[rf_mask]
    b = target[rf_mask]
    denom = (b * b).sum()
    if denom < 1e-12:
        return 0.0
    return ((a * b).sum() / denom).item()


# ============================================================================
# C-eigenspace decomposition
# ============================================================================

def compute_c_eigenspace(kernel, rf_mask, eigen_rel_threshold, no_filter=False):
    """Eigendecompose the kernel's C matrix on masked pixels.

    Uses a RELATIVE threshold: keep eigenvalues > eigen_rel_threshold * max_eigenvalue.
    Mass-based thresholds are not useful because the first eigenvalue dominates.

    If no_filter=True, keep ALL eigenvectors (K = n_rf). The eigendecomposition
    is still computed (it's a rotation), but no truncation is applied.

    Returns:
        U_K: (n_rf, K) top eigenvectors (columns)
        gamma_K: (K,) corresponding eigenvalues
        all_eigvals: (n_rf,) full eigenvalue spectrum (for plotting)
        K: number of retained dimensions
    """
    # Extract C matrix on masked pixels
    with torch.no_grad():
        C, mask = kernel._compute_C_matrix(apply_mask=True)

    print(f"  C matrix shape: {C.shape}")
    print(f"  C matrix range: [{C.min().item():.6f}, {C.max().item():.6f}]")

    # Eigendecompose (eigh returns ascending order)
    eigvals, eigvecs = torch.linalg.eigh(C)

    # Reverse to descending order
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)

    # Keep negative eigenvalues as zero (numerical noise)
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
        # Keep all eigenvalues above the relative threshold
        K = int((eigvals > abs_threshold).sum().item())
        K = max(K, 1)  # always keep at least 1
        print(f"  Relative threshold: {eigen_rel_threshold} * max = {abs_threshold:.6e}")

    cumsum = eigvals.cumsum(0) / total_mass

    print(f"  Dimensions kept (K): {K} out of {eigvals.shape[0]}")
    print(f"  Eigenvalue mass captured: {cumsum[K-1].item():.6f}")
    print(f"  Effective dimensionality reduction: {eigvals.shape[0]} -> {K} "
          f"({100 * K / eigvals.shape[0]:.1f}%)")

    U_K = eigvecs[:, :K]   # (n_rf, K)
    gamma_K = eigvals[:K]   # (K,)

    return U_K, gamma_K, eigvals, K


# ============================================================================
# Gradient ascent in C-eigenspace
# ============================================================================

def gradient_ascent_c_eigen(model, likelihood, x_start, x_samples,
                            rf_mask, U_K, r_max, f_max,
                            n_steps, lr, max_iter, max_eval, history_size):
    """LBFGS gradient ascent maximizing U_DA, parameterized in C-eigenspace.

    The optimization variable is z in R^K. The image is x* = U_K @ z
    (in RF-pixel space), then placed into full image for kernel evaluation.

    Args:
        x_start: (n_pixels,) starting image in full pixel space
        x_samples: (n_cond, n_pixels) conditioning images for DA utility
        U_K: (n_rf, K) top K eigenvectors of C (on masked pixels)

    Returns:
        x_final: (n_pixels,) optimized image (full pixel space)
        history: dict with per-step metrics
    """
    n_pixels = x_start.shape[0]
    device = x_start.device
    dtype = x_start.dtype
    n_rf = rf_mask.sum().item()
    n_cond = x_samples.shape[0]

    # Project starting RF pixels into C-eigenspace: z_0 = U_K^T @ x_rf
    x_rf_start = x_start[rf_mask]  # (n_rf,)
    z_init = U_K.T @ x_rf_start    # (K,)

    # Check reconstruction quality
    x_rf_reconstructed = U_K @ z_init
    recon_error = (x_rf_start - x_rf_reconstructed).norm().item()
    relative_error = recon_error / x_rf_start.norm().item()
    print(f"  Projection reconstruction error: {recon_error:.6f} "
          f"(relative: {relative_error:.6f})")

    # Optimization variable
    z = z_init.clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [z],
        lr=lr,
        max_iter=max_iter,
        max_eval=max_eval,
        history_size=history_size,
        line_search_fn='strong_wolfe',
    )

    history = {
        'step': [], 'utility': [], 'grad_norm_z': [], 'grad_norm_pixel': [],
        'rf_norm': [], 'step_time': [],
    }
    if n_cond == 1:
        history['pearson_r'] = []
        history['proj_coeff'] = []
    else:
        history['pearson_r_vs_start'] = []
        history['firing_rate'] = []

    A = likelihood.A.squeeze()
    lam0 = likelihood.lambda0.squeeze()

    # Capture last closure utility to avoid recomputing it for logging
    last_closure_utility = [None]

    for step in range(n_steps):
        t_step = time.time()

        def closure():
            optimizer.zero_grad()
            # z -> RF pixels -> full image
            x_rf = U_K @ z             # (n_rf,) - gradient flows through U_K @ z
            x_full = _reconstruct_image_from_rf(x_rf, rf_mask, n_pixels, dtype, device)

            if n_cond <= GRAD_CHUNK_SIZE:
                # Small enough to fit in GPU memory in one pass
                result = distribution_aware_utility(
                    model, likelihood,
                    x_full.unsqueeze(0),
                    x_samples,
                    r_max=r_max,
                    adaptive_r_max=False,
                    sample_lambda=False,
                )
                # Firing rate guard
                mu_g = result['mu_g_marg']
                if torch.exp(mu_g).item() > f_max:
                    return torch.tensor(float('inf'), device=device)
                loss = -result['utility'].squeeze()
                last_closure_utility[0] = result['utility'].item()
                loss.backward()
                return loss
            else:
                # Chunked gradient accumulation for large n_cond.
                # Each chunk computes DA utility with a subset of x_samples,
                # calls backward to free the graph, and accumulates gradients.
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

                    # Firing rate guard (only need to check once, same x*)
                    if chunk_idx == 0:
                        mu_g = result['mu_g_marg']
                        if torch.exp(mu_g).item() > f_max:
                            return torch.tensor(float('inf'), device=device)

                    # Weight by chunk fraction for correct average
                    chunk_weight = (c_end - c_start) / n_cond
                    chunk_loss = -result['utility'].squeeze() * chunk_weight
                    total_loss_val += chunk_loss.item()

                    # Backward with retain_graph for all but last chunk
                    # (x_full graph from z needed by subsequent chunks)
                    is_last = (chunk_idx == n_chunks - 1)
                    chunk_loss.backward(retain_graph=not is_last)

                last_closure_utility[0] = -total_loss_val
                return torch.tensor(total_loss_val, device=device)

        optimizer.step(closure)
        step_time = time.time() - t_step

        # Verify gradient flow on first step
        if step == 0:
            if z.grad is None or z.grad.norm() == 0:
                raise RuntimeError(
                    f"No gradient at step 0: f_max guard likely rejected all "
                    f"closure evaluations. The starting image's predicted "
                    f"firing rate exceeds f_max={f_max}. Try a different "
                    f"starting point or increase f_max."
                )

        # Track metrics (no grad needed for logging).
        # Use captured utility from last closure call to avoid expensive recomputation.
        utility = last_closure_utility[0]
        with torch.no_grad():
            x_rf = U_K @ z
            x_full = _reconstruct_image_from_rf(x_rf, rf_mask, n_pixels, dtype, device)
            cur_rf_norm = x_full[rf_mask].norm().item()

            if n_cond == 1:
                pr = rf_pearson_r(x_full, x_samples[0], rf_mask)
                pc = rf_proj_coeff(x_full, x_samples[0], rf_mask)
            else:
                pr_vs_start = rf_pearson_r(x_full, x_start, rf_mask)
                mu, sigma2 = get_gp_marginal_moments(model, x_full.unsqueeze(0))
                fr = torch.exp(A * mu + lam0).item()

        grad_norm_z = z.grad.norm().item() if z.grad is not None else 0.0
        # Compute pixel-space gradient norm for comparison
        # nabla_x U = U_K @ nabla_z U (since nabla_z = U_K^T @ nabla_x, and U_K^T U_K = I)
        if z.grad is not None:
            grad_pixel = U_K @ z.grad  # (n_rf,) - pixel-space gradient from z-space
            grad_norm_pixel = grad_pixel.norm().item()
        else:
            grad_norm_pixel = 0.0

        history['step'].append(step)
        history['utility'].append(utility)
        history['grad_norm_z'].append(grad_norm_z)
        history['grad_norm_pixel'].append(grad_norm_pixel)
        history['rf_norm'].append(cur_rf_norm)
        history['step_time'].append(step_time)

        if n_cond == 1:
            history['pearson_r'].append(pr)
            history['proj_coeff'].append(pc)
        else:
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

        # Early stopping: utility converged (no change for 5 consecutive steps)
        if step >= 5:
            recent = history['utility'][-5:]
            if max(recent) - min(recent) < 1e-8:
                print(f"  step {step}: utility converged (no change for 5 steps) - stopping early")
                break

        if step % LOG_EVERY == 0 or step == n_steps - 1:
            elapsed_steps = step + 1
            avg_time = sum(history['step_time']) / elapsed_steps
            eta = avg_time * (n_steps - elapsed_steps)

            if n_cond == 1:
                print(f"  step {step:4d}: U={utility:.6f}  "
                      f"r={pr:.4f}  proj={pc:.4f}  |grad_z|={grad_norm_z:.4e}  "
                      f"|grad_px|={grad_norm_pixel:.4e}  "
                      f"[{step_time:.1f}s, ETA {eta:.0f}s]")
            else:
                print(f"  step {step:4d}: U={utility:.6f}  "
                      f"r_start={pr_vs_start:.4f}  ||x||_RF={cur_rf_norm:.2f}  "
                      f"FR={fr:.1f}  |grad_z|={grad_norm_z:.4e}  "
                      f"[{step_time:.1f}s, ETA {eta:.0f}s]")

    # Return full reconstructed image
    with torch.no_grad():
        x_rf_final = U_K @ z
        x_final = _reconstruct_image_from_rf(x_rf_final, rf_mask, n_pixels, dtype, device)
    return x_final.detach(), history


# ============================================================================
# Visualization
# ============================================================================

def plot_results(x_target, x_start, x_final, history, all_eigvals, K,
                 eigen_rel_threshold, target_index,
                 rf_mask, kernel, config,
                 vmin, vmax, test_r, reliability, n_px_side, out_path,
                 u_target=None):
    """Generate summary figure for C-eigenspace optimization.

    Top row: target, start, final, difference images.
    Bottom row: eigenvalue spectrum, utility convergence, gradient norms.

    Each image subplot independently checks for OOB pixels (RF-masked only)
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

    fig = plt.figure(figsize=(20, 10))

    # --- Top row: images with per-image OOB checks ---
    u_start = history['utility'][0]
    u_final = history['utility'][-1]

    x_tensors = [x_target, x_start, x_final, x_final - x_start]
    cropped_images = [masked_crop(x, gray_val) for x in x_tensors]
    target_title = f'Target A (pool[{target_index}])'
    if u_target is not None:
        target_title += f'\nU_DA={u_target:.4f}'
    base_titles = [
        target_title,
        f'Start (smoothed)\nU_DA={u_start:.4f}',
        f'Final (C-eigen opt)\nU_DA={u_final:.4f}',
        f'Final - Start',
    ]

    for i, (img, x_flat, title) in enumerate(zip(cropped_images, x_tensors, base_titles)):
        pct_oob, lo, hi = check_oob(x_flat)
        is_oob = pct_oob > 0
        if is_oob and i < 3:  # skip diff image for OOB annotation
            title += f'\nCLIPPED {pct_oob:.0f}% OOB [{lo:.2f},{hi:.2f}]'

        ax = fig.add_subplot(2, 4, i + 1)
        v = (vmin, vmax) if i < 3 else (None, None)
        im = ax.imshow(img, cmap='gray', vmin=v[0], vmax=v[1], aspect='equal')
        ax.set_title(title, fontsize=10, color='red' if (is_oob and i < 3) else 'black')
        ax.axis('off')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        if i < 3:
            draw_rf_overlay(ax)

    # --- Bottom left: Eigenvalue spectrum ---
    ax_eig = fig.add_subplot(2, 4, 5)
    eigvals_np = all_eigvals.cpu().numpy()
    ax_eig.semilogy(eigvals_np, 'b-', linewidth=1.5)
    ax_eig.axvline(K - 1, color='red', linestyle='--', linewidth=1,
                   label=f'K={K} (thresh={eigen_rel_threshold:.0e})')
    ax_eig.set_xlabel('Eigenvalue index')
    ax_eig.set_ylabel('Eigenvalue (log scale)')
    ax_eig.set_title('C matrix eigenvalue spectrum')
    ax_eig.legend(fontsize=9)
    ax_eig.grid(True, alpha=0.3)

    # --- Bottom right: Utility + Pearson r vs target ---
    ax_util = fig.add_subplot(2, 4, (6, 8))
    steps = history['step']
    color_u = 'tab:blue'
    color_r = 'tab:green'

    ax_util.plot(steps, history['utility'], color=color_u, linewidth=1.5, label='U_DA')
    ax_util.set_xlabel('Step')
    ax_util.set_ylabel('U_DA', color=color_u)
    ax_util.tick_params(axis='y', labelcolor=color_u)

    ax_r = ax_util.twinx()
    ax_r.plot(steps, history['pearson_r'], color=color_r, linewidth=1.5,
              alpha=0.7, label='Pearson r vs target (RF)')
    ax_r.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
    ax_r.set_ylabel('Pearson r', color=color_r)
    ax_r.tick_params(axis='y', labelcolor=color_r)

    lines1, labels1 = ax_util.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax_util.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=9)
    ax_util.set_title('LBFGS convergence (C-eigen)')
    ax_util.grid(True, alpha=0.3)

    title_str = (
        f'C-Eigenspace DA Utility Optimization  '
        f'(M={config["M"]}, n_train={config["n_train"]}, K={K}/{rf_mask.sum().item()})  '
        f'test_r={test_r:.3f}, reliability={reliability:.3f}'
    )
    fig.suptitle(title_str, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_results_multicond(x_start, x_final, history, all_eigvals, K,
                           eigen_rel_threshold, no_filter, n_cond,
                           rf_mask, kernel, config,
                           vmin, vmax, test_r, n_px_side, out_path,
                           start_index=None):
    """Summary figure for multi-conditioning C-eigenspace optimization.

    Top row: start image, final image, final-start diff, text summary.
    Bottom row: eigenvalue spectrum, utility convergence, firing rate + RF norm, gradient norms.

    Each image subplot independently checks for OOB pixels (RF-masked only)
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

    fig = plt.figure(figsize=(20, 10))

    u_start = history['utility'][0]
    u_final = history['utility'][-1]

    # --- Top row: images + text summary ---
    # Panels 1-3: Start, Final, Diff — each with independent OOB check
    x_tensors = [x_start, x_final, x_final - x_start]
    cropped_images = [masked_crop(x, gray_val) for x in x_tensors]
    start_label = f'Start (pool #{start_index})' if start_index is not None else 'Start'
    base_titles = [
        f'{start_label}\nU_DA={u_start:.4f}',
        f'Final (C-eigen opt)\nU_DA={u_final:.4f}',
        f'Final - Start',
    ]
    use_clamp = [True, True, False]  # clamp to vmin/vmax for first two only

    for i, (img, x_flat, title) in enumerate(zip(cropped_images, x_tensors, base_titles)):
        pct_oob, lo, hi = check_oob(x_flat)
        is_oob = pct_oob > 0
        if is_oob and use_clamp[i]:
            title += f'\nCLIPPED {pct_oob:.0f}% OOB [{lo:.2f},{hi:.2f}]'

        ax = fig.add_subplot(2, 4, i + 1)
        v = (vmin, vmax) if use_clamp[i] else (None, None)
        im = ax.imshow(img, cmap='gray', vmin=v[0], vmax=v[1], aspect='equal')
        ax.set_title(title, fontsize=10, color='red' if (is_oob and use_clamp[i]) else 'black')
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
    final_fr = history['firing_rate'][-1] if 'firing_rate' in history else 'N/A'
    final_rf_norm = history['rf_norm'][-1]
    final_oob, _, _ = check_oob(x_final)
    summary_lines = [
        f'n_cond = {n_cond}',
        f'n_steps = {len(history["step"])}',
        f'',
        f'U_DA: {u_start:.6f} -> {u_final:.6f}',
        f'delta_U: {u_final - u_start:.6f}',
        f'',
        f'||x||_RF: {history["rf_norm"][0]:.2f} -> {final_rf_norm:.2f}',
        f'Firing rate: {final_fr:.1f}' if isinstance(final_fr, float) else f'Firing rate: {final_fr}',
        f'OOB pixels: {final_oob:.1f}%',
        f'',
        f'Total time: {total_time:.0f}s ({avg_step:.1f}s/step)',
        f'test_r: {test_r:.3f}',
    ]
    ax.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    # --- Bottom row ---
    # Panel 5: Eigenvalue spectrum
    ax_eig = fig.add_subplot(2, 4, 5)
    eigvals_np = all_eigvals.cpu().numpy()
    ax_eig.semilogy(eigvals_np, 'b-', linewidth=1.5)
    if no_filter:
        ax_eig.axvline(K - 1, color='red', linestyle='--', linewidth=1,
                       label=f'K={K} (no filter)')
    else:
        ax_eig.axvline(K - 1, color='red', linestyle='--', linewidth=1,
                       label=f'K={K} (thresh={eigen_rel_threshold:.0e})')
    ax_eig.set_xlabel('Eigenvalue index')
    ax_eig.set_ylabel('Eigenvalue (log scale)')
    ax_eig.set_title('C matrix eigenvalue spectrum')
    ax_eig.legend(fontsize=9)
    ax_eig.grid(True, alpha=0.3)

    # Panels 6-8: Utility + Pearson r vs start (wide subplot)
    ax_util = fig.add_subplot(2, 4, (6, 8))
    steps = history['step']
    color_u = 'tab:blue'
    color_r = 'tab:green'

    ax_util.plot(steps, history['utility'], color=color_u, linewidth=1.5, label='U_DA')
    ax_util.set_xlabel('Step')
    ax_util.set_ylabel('U_DA', color=color_u)
    ax_util.tick_params(axis='y', labelcolor=color_u)

    ax_r = ax_util.twinx()
    ax_r.plot(steps, history['pearson_r_vs_start'], color=color_r, linewidth=1.5,
              alpha=0.7, label='Pearson r vs start (RF)')
    ax_r.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
    ax_r.set_ylabel('Pearson r', color=color_r)
    ax_r.tick_params(axis='y', labelcolor=color_r)

    lines1, labels1 = ax_util.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax_util.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=9)
    ax_util.set_title('LBFGS convergence')
    ax_util.grid(True, alpha=0.3)

    filter_str = 'no filter' if no_filter else f'thresh={eigen_rel_threshold:.0e}'
    title_str = (
        f'Multi-Cond DA Utility Optimization (N={n_cond})  '
        f'M={config["M"]}, n_train={config["n_train"]}, K={K}/{rf_mask.sum().item()} ({filter_str})  '
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

def main(eigen_rel_threshold=None, no_filter=False, kernel_type=None,
         n_cond=None, cond_seed=COND_SEED, start_index=None, start_seed=START_SEED):
    t0 = time.time()

    if eigen_rel_threshold is None:
        eigen_rel_threshold = EIGEN_REL_THRESHOLD
    if n_cond is None:
        n_cond = N_COND

    # === Step 1: Train model ===
    print("=" * 70)
    print(f"Step 1: Training model (default_gpy, M={M_OVERRIDE}, n_train={N_TRAIN_OVERRIDE})")
    print("=" * 70)

    env = setup(kernel_type=kernel_type, M_override=M_OVERRIDE, n_train_override=N_TRAIN_OVERRIDE)
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

    # === Step 2: Get RF mask and C matrix ===
    print("\n" + "=" * 70)
    print("Step 2: C matrix eigendecomposition")
    print("=" * 70)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_pixels = n_px_side ** 2

    kernel = getattr(model, 'covar_module', None) or getattr(model, 'kernel', None)
    # Ensure RF mask is cached
    if not (hasattr(kernel, '_cached_mask') and kernel._cached_mask is not None):
        with torch.no_grad():
            _ = model(X_pool[0].unsqueeze(0))
    rf_mask = kernel._cached_mask.squeeze()

    n_rf = rf_mask.sum().item()
    print(f"  RF mask: {n_rf} pixels out of {n_pixels}")

    U_K, gamma_K, all_eigvals, K = compute_c_eigenspace(
        kernel, rf_mask, eigen_rel_threshold, no_filter=no_filter
    )

    # === Step 3: Create conditioning images and starting image ===
    print("\n" + "=" * 70)
    print("Step 3: Creating images")
    print("=" * 70)

    # Dataset pixel bounds for visualization (needed by both branches)
    rf_mask_np = rf_mask.cpu().numpy()
    X_all = torch.cat([X_pool, X_train], dim=0)
    all_rf_vals = X_all.cpu().numpy()[:, rf_mask_np]
    vmin_dataset = float(all_rf_vals.min())
    vmax_dataset = float(all_rf_vals.max())

    # Track which start index was actually used (for plotting labels)
    actual_start_index = None

    if n_cond == 1:
        # --- Legacy single-target mode ---
        x_target = X_pool[TARGET_INDEX]
        x_samples = x_target.unsqueeze(0)  # (1, n_pixels)

        # Smoothed starting image (same as gradient.py)
        x_target_2d = x_target.cpu().numpy().reshape(n_px_side, n_px_side)
        x_smoothed_2d = gaussian_filter(x_target_2d, sigma=SIGMA_SMOOTH)
        x_start = torch.tensor(
            x_smoothed_2d.reshape(-1), dtype=dtype, device=device
        )

        pixel_dist = (x_start - x_target).norm().item()
        rf_dist = (x_start[rf_mask] - x_target[rf_mask]).norm().item()
        print(f"  Mode: single-target (legacy)")
        print(f"  Target: pool image {TARGET_INDEX}")
        print(f"  Start: Gaussian smoothing sigma={SIGMA_SMOOTH}")
        print(f"  pixel_dist={pixel_dist:.4f}, rf_dist={rf_dist:.4f}")
    else:
        # --- Multi-conditioning mode ---
        n_pool = X_pool.shape[0]
        rng = torch.Generator(device='cpu').manual_seed(cond_seed)
        cond_indices = torch.randperm(n_pool, generator=rng)[:n_cond]
        x_samples = X_pool[cond_indices]  # (n_cond, n_pixels)

        # Select starting image
        if start_index is not None:
            actual_start_index = start_index
            x_start = X_pool[start_index].clone()
        else:
            # Random start, avoiding conditioning set
            start_rng = torch.Generator(device='cpu').manual_seed(start_seed)
            cond_set = set(cond_indices.tolist())
            remaining = sorted(set(range(n_pool)) - cond_set)
            rand_idx = remaining[torch.randint(len(remaining), (1,), generator=start_rng).item()]
            actual_start_index = rand_idx
            x_start = X_pool[rand_idx].clone()

        x_start = x_start.to(device=device, dtype=dtype)

        overlap = set(cond_indices.tolist()) & {actual_start_index}
        print(f"  Mode: multi-conditioning (n_cond={n_cond})")
        print(f"  Conditioning: {n_cond} images from pool (seed={cond_seed})")
        print(f"  Start: pool image {actual_start_index}"
              f"{'  (user-specified)' if start_index is not None else f'  (random, seed={start_seed})'}")
        print(f"  Start in conditioning set: {'YES' if overlap else 'no'}")

    print(f"  Dataset RF pixel range: [{vmin_dataset:.3f}, {vmax_dataset:.3f}]")

    # Firing rate sanity check on starting image before optimization
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

    # === Step 4: Gradient ascent in C-eigenspace ===
    print("\n" + "=" * 70)
    print(f"Step 4: Gradient ascent in C-eigenspace (K={K}, n_cond={n_cond})")
    print("=" * 70)

    x_final, history = gradient_ascent_c_eigen(
        model, likelihood, x_start, x_samples,
        rf_mask, U_K, r_max, f_max,
        N_STEPS, LR, LBFGS_MAX_ITER, LBFGS_MAX_EVAL, LBFGS_HISTORY_SIZE,
    )

    print(f"\nGradient ascent summary:")
    print(f"  Steps: {len(history['step'])}")
    print(f"  U_DA: {history['utility'][0]:.6f} -> {history['utility'][-1]:.6f}")
    if n_cond == 1:
        print(f"  Pearson r (RF): {history['pearson_r'][0]:.4f} -> {history['pearson_r'][-1]:.4f}")
        print(f"  Proj coeff (RF): {history['proj_coeff'][0]:.4f} -> {history['proj_coeff'][-1]:.4f}")
    else:
        print(f"  Pearson r vs start (RF): {history['pearson_r_vs_start'][0]:.4f} -> {history['pearson_r_vs_start'][-1]:.4f}")
        print(f"  ||x||_RF: {history['rf_norm'][0]:.2f} -> {history['rf_norm'][-1]:.2f}")
        print(f"  Firing rate: {history['firing_rate'][0]:.1f} -> {history['firing_rate'][-1]:.1f}")

    # Pixel bounds check
    x_final_rf = x_final[rf_mask]
    final_min = x_final_rf.min().item()
    final_max = x_final_rf.max().item()
    n_below = (x_final_rf < vmin_dataset).sum().item()
    n_above = (x_final_rf > vmax_dataset).sum().item()
    n_oob = n_below + n_above
    pct_oob = 100.0 * n_oob / n_rf
    print(f"\n  Pixel bounds check (dataset RF range: [{vmin_dataset:.3f}, {vmax_dataset:.3f}]):")
    print(f"    Final image RF range: [{final_min:.3f}, {final_max:.3f}]")
    print(f"    Out-of-bounds pixels: {n_oob}/{n_rf} ({pct_oob:.1f}%)"
          f"  [{n_below} below, {n_above} above]")
    if n_oob > 0:
        print(f"    ** IMAGE IS CLIPPED in plot (vmin/vmax set to dataset range)")

    # Firing rate diagnostics
    print(f"\n  Firing rate diagnostics (f_max={f_max})")
    A = likelihood.A.squeeze()
    lam0 = likelihood.lambda0.squeeze()
    with torch.no_grad():
        if n_cond == 1:
            diag_imgs = [('target', X_pool[TARGET_INDEX]), ('start', x_start), ('final', x_final)]
        else:
            diag_imgs = [('start', x_start), ('final', x_final)]
        for label, x_img in diag_imgs:
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

    cond_suffix = f'_cond{n_cond}' if n_cond > 1 else ''
    kernel_suffix = f'_{kernel_type}' if kernel_type else ''
    if no_filter:
        out_name = f'c_eigen_optimization{kernel_suffix}{cond_suffix}_nofilter.png'
    else:
        out_name = f'c_eigen_optimization{kernel_suffix}{cond_suffix}_thresh{eigen_rel_threshold:.0e}.png'

    # Compute target utility for plot title (single-target mode only)
    u_target = None
    if n_cond == 1:
        with torch.no_grad():
            x_target = X_pool[TARGET_INDEX]
            result_target = distribution_aware_utility(
                model, likelihood,
                x_target.unsqueeze(0),
                x_target.unsqueeze(0),
                r_max=r_max, adaptive_r_max=False, sample_lambda=False,
            )
            u_target = result_target['utility'].item()
        print(f"  Target utility (U_DA): {u_target:.6f}")

    if n_cond == 1:
        plot_results(
            X_pool[TARGET_INDEX], x_start, x_final, history, all_eigvals, K,
            eigen_rel_threshold, TARGET_INDEX,
            rf_mask, kernel, config,
            vmin_dataset, vmax_dataset, test_r, reliability, n_px_side,
            out_path=_script_dir / out_name,
            u_target=u_target,
        )
    else:
        plot_results_multicond(
            x_start, x_final, history, all_eigvals, K,
            eigen_rel_threshold, no_filter, n_cond,
            rf_mask, kernel, config,
            vmin_dataset, vmax_dataset, test_r, n_px_side,
            out_path=_script_dir / out_name,
            start_index=actual_start_index,
        )

    elapsed = time.time() - t0
    print(f"\nTotal elapsed time: {elapsed:.1f}s")

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Kernel: {kernel_type}")
    print(f"  Conditioning: n_cond={n_cond}")
    print(f"  C matrix: {n_rf} x {n_rf} (RF-masked)")
    print(f"  Eigenspace: K={K} out of {n_rf} ({100*K/n_rf:.1f}%)")
    if no_filter:
        print(f"  Eigenvalue filter: NONE (all eigenvectors kept)")
    else:
        print(f"  Eigenvalue relative threshold: {eigen_rel_threshold:.0e}")
    print(f"  Final utility: {history['utility'][-1]:.6f}")
    if n_cond == 1:
        print(f"  Final Pearson r to target: {history['pearson_r'][-1]:.4f}")
        print(f"  Final proj coeff: {history['proj_coeff'][-1]:.4f}")
        print(f"\n  Theoretical prediction: C-eigenspace optimization should")
        print(f"  produce the SAME final image as pixel-space optimization")
        print(f"  (only improves conditioning, does not constrain solution).")
        print(f"  Compare with gradient.py results to verify.")
    else:
        print(f"  Final ||x||_RF: {history['rf_norm'][-1]:.2f}")
        print(f"  Final firing rate: {history['firing_rate'][-1]:.1f}")
        print(f"  OOB pixels: {pct_oob:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='C-eigenspace DA utility optimization')
    parser.add_argument('--eigen-threshold', type=float, default=None,
                        help=f'Relative eigenvalue threshold for truncation '
                             f'(fraction of max eigenvalue, default: {EIGEN_REL_THRESHOLD})')
    parser.add_argument('--no-filter', action='store_true',
                        help='Keep ALL eigenvectors (no truncation). '
                             'Equivalent to full rotation without dimensionality reduction.')
    parser.add_argument('--kernel-type', type=str, default=None,
                        choices=['arc_cosine', 'arc_sine', 'rbf'],
                        help='Kernel type (default: from default_params.json)')
    parser.add_argument('--n-cond', type=int, default=N_COND,
                        help='Number of conditioning images for DA utility '
                             f'(default: {N_COND}, 1 = single target mode)')
    parser.add_argument('--cond-seed', type=int, default=COND_SEED,
                        help=f'Seed for conditioning image selection (default: {COND_SEED})')
    parser.add_argument('--start-index', type=int, default=None,
                        help='Pool image index for starting point '
                             '(default: TARGET_INDEX for n_cond=1, random for n_cond>1)')
    parser.add_argument('--start-seed', type=int, default=START_SEED,
                        help=f'Seed for random start image selection (default: {START_SEED})')
    args = parser.parse_args()
    main(eigen_rel_threshold=args.eigen_threshold, no_filter=args.no_filter,
         kernel_type=args.kernel_type, n_cond=args.n_cond, cond_seed=args.cond_seed,
         start_index=args.start_index, start_seed=args.start_seed)
