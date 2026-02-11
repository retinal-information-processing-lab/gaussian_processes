"""
Multi-image DA utility gradient ascent with LocalRBFKernel.
Created by Claude.

Companion to gradient_rbf.py (single-target) and explore_utility_rbf.py (workbench).

Instead of conditioning on a single target image, this script conditions on
N_SAMPLE natural images from the pool. The DA utility becomes:

    U(x*) = H_marg(x*) - (1/N) sum_i H(r* | lambda(x_i), D)

This is the "proper" distribution-aware utility — the optimizer must find x*
that's informative about responses across many natural images, not just one.
The hypothesis is that multi-image conditioning acts as a natural regularizer,
preventing the pixel saturation seen with single-target optimization.

Model training reuses setup() from explore_utility_rbf.py (M=50, N_TRAIN=50).
RF-only optimization reuses _reconstruct_image() from gradient_rbf.py.
All other params from default_params.json via build_config_from_defaults().

Usage:
    python investigations/rbf_kernel/distribution_gradient.py
"""

import sys
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

from explore_utility_rbf import setup
from gradient_rbf import _reconstruct_image
from acquisition import distribution_aware_utility, get_gp_marginal_moments, compute_H, compute_adaptive_rmax
from run_rbf import load_pnas_data
from gpy_training import predict
from metrics import compute_pearson_correlation, compute_explained_variance

# ---------------------------------------------------------------------------
# Investigation parameters (visible, explicit)
# Model params (M=50, N_TRAIN=50) are set in explore_utility_rbf.py¡
# ---------------------------------------------------------------------------
N_SAMPLE = 500            # number of conditioning images from pool
NOISE_AMP = 0.01         # amplitude of starting noise on top of mean gray
SAMPLE_LAMBDA = False    # True: stochastic lambda samples; False: deterministic (mean)

# --- Pixel bounds ---
USE_SIGMOID_BOUNDS = True   # sigmoid reparametrization: pixels bounded to dataset [vmin, vmax]

# --- LBFGS optimizer parameters ---
N_STEPS = 2             # outer LBFGS steps
LR = 0.01                 # LBFGS step size
LBFGS_MAX_ITER = 20      # max iterations per LBFGS step (line search evals)
LBFGS_MAX_EVAL = 25      # max function evaluations per LBFGS step
LBFGS_HISTORY_SIZE = 10  # number of past gradients for Hessian approximation
LOG_EVERY = 1            # print every step


def gradient_ascent_da(model, likelihood, x_start, x_samples, rf_mask,
                       f_max, pixel_lo, pixel_hi,
                       n_steps, lr, max_iter, max_eval, history_size,
                       sample_lambda,
                       adaptive_safety_k, adaptive_max_rmax, adaptive_min_rmax):
    """LBFGS gradient ascent maximizing DA utility conditioned on N images.

    Optimizes RF-masked pixels only. The utility conditions on all images
    in x_samples (N_SAMPLE, d) rather than a single target.

    Args:
        x_start: (n_pixels,) starting image
        x_samples: (N_SAMPLE, d) conditioning images from pool
        rf_mask: (n_pixels,) boolean mask for RF pixels
        pixel_lo, pixel_hi: sigmoid bounds (None to disable)
        sample_lambda: if True, sample lambda at conditioning images; False uses mean

    Returns:
        x_final: (n_pixels,) optimized image (zeros outside RF)
        history: dict with lists of per-step metrics
    """
    n_pixels = x_start.shape[0]
    device = x_start.device
    dtype = x_start.dtype

    # Extract RF pixels from starting image
    x_rf_init = x_start[rf_mask]

    # Setup optimization variable: sigmoid reparametrization or direct
    use_sigmoid = pixel_lo is not None and pixel_hi is not None
    if use_sigmoid:
        eps = 1e-6
        x_clamped = x_rf_init.clamp(pixel_lo + eps, pixel_hi - eps)
        sigmoid_val = (x_clamped - pixel_lo) / (pixel_hi - pixel_lo)
        z_rf = torch.log(sigmoid_val / (1 - sigmoid_val))  # logit
        z_rf = z_rf.detach().requires_grad_(True)
        opt_var = z_rf

        def _to_pixel(z):
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
        'step': [], 'utility': [], 'H_marg': [], 'H_cond': [], 'grad_norm': [],
    }

    for step in range(n_steps):
        def closure():
            optimizer.zero_grad()
            x_rf_pixels = _to_pixel(opt_var)
            x_full = _reconstruct_image(x_rf_pixels, rf_mask, n_pixels, dtype, device)
            result = distribution_aware_utility(
                model, likelihood,
                x_full.unsqueeze(0),
                x_samples,
                r_max=None,
                adaptive_r_max=True,
                sample_lambda=sample_lambda,
                adaptive_safety_k=adaptive_safety_k,
                adaptive_max_rmax=adaptive_max_rmax,
                adaptive_min_rmax=adaptive_min_rmax,
            )
            # Firing rate guard
            mu_g = result['mu_g_marg']
            if torch.exp(mu_g).item() > f_max:
                return torch.tensor(float('inf'), device=device)
            loss = -result['utility'].squeeze()
            loss.backward()
            return loss

        optimizer.step(closure)

        # Verify gradient flow on first step
        if step == 0:
            assert opt_var.grad is not None and opt_var.grad.norm() > 0, \
                "No gradient flow through RF reconstruction"

        # Track metrics (no grad needed)
        with torch.no_grad():
            x_rf_pixels = _to_pixel(opt_var)
            x_full = _reconstruct_image(x_rf_pixels, rf_mask, n_pixels, dtype, device)
            result = distribution_aware_utility(
                model, likelihood,
                x_full.unsqueeze(0),
                x_samples,
                r_max=None,
                adaptive_r_max=True,
                sample_lambda=sample_lambda,
                adaptive_safety_k=adaptive_safety_k,
                adaptive_max_rmax=adaptive_max_rmax,
                adaptive_min_rmax=adaptive_min_rmax,
            )
            utility = result['utility'].item()
            h_marg = result['H_marg'].item()
            h_cond = result['H_cond'].item()

        grad_norm = opt_var.grad.norm().item() if opt_var.grad is not None else 0.0

        history['step'].append(step)
        history['utility'].append(utility)
        history['H_marg'].append(h_marg)
        history['H_cond'].append(h_cond)
        history['grad_norm'].append(grad_norm)

        if np.isnan(utility) or np.isnan(grad_norm):
            print(f"  step {step}: NaN detected - stopping")
            break

        if step % LOG_EVERY == 0 or step == n_steps - 1:
            print(f"  step {step:4d}: U={utility:.6f}  "
                  f"H_marg={h_marg:.4f}  H_cond={h_cond:.4f}  |grad|={grad_norm:.4e}")

    # Return full reconstructed image
    with torch.no_grad():
        x_rf_final = _to_pixel(opt_var)
        x_final = _reconstruct_image(x_rf_final, rf_mask, n_pixels, dtype, device)
    return x_final.detach(), history


def main():
    # === Step 1: Train model with LocalRBFKernel ===
    print("=" * 70)
    print("Step 1: Training model with LocalRBFKernel")
    print("=" * 70)

    env = setup()
    model = env['model']
    likelihood = env['likelihood']
    X_pool = env['X_pool']
    config = env['config']
    f_max = config['f_max']
    n_px_side = config['n_px_side']

    # --- Test evaluation ---
    data_path = Path('/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/notebooks/PNAS_paper_sorted_data.npz')
    data = load_pnas_data(data_path, dtype=torch.float32)
    device = next(model.parameters()).device
    X_test = data['X_test'].reshape(data['X_test'].shape[0], -1).to(device)
    R_test = data['R_test'].to(device)
    r_test = R_test[:, :, config['cell']]

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

    # === Step 2: Setup conditioning images and starting image ===
    print("\n" + "=" * 70)
    print(f"Step 2: Setup (N_SAMPLE={N_SAMPLE})")
    print("=" * 70)

    dtype = next(model.parameters()).dtype
    n_pixels = n_px_side ** 2

    # Ensure RF mask is cached
    kernel = model.covar_module
    if not (hasattr(kernel, '_cached_mask') and kernel._cached_mask is not None):
        with torch.no_grad():
            _ = model(X_pool[0].unsqueeze(0))
    rf_mask = kernel._cached_mask.squeeze()

    # Conditioning images: first N_SAMPLE from pool
    n_pool = X_pool.shape[0]
    n_sample = min(N_SAMPLE, n_pool)
    x_samples = X_pool[:n_sample]
    print(f"  Conditioning images: {n_sample} from pool ({n_pool} total)")

    # Starting image: uniform gray at the mean RF intensity of sampled images, + small noise
    mean_rf_val = x_samples[:, rf_mask].mean().item()
    torch.manual_seed(config['seed'])
    x_start = torch.full((n_pixels,), mean_rf_val, dtype=dtype, device=device)
    x_start[rf_mask] += 0.01 * torch.randn(rf_mask.sum().item(), dtype=dtype, device=device)
    x_start[~rf_mask] = 0.0
    print(f"  Starting image: uniform gray ({mean_rf_val:.4f}) + noise, RF only")
    print(f"  ||start||_RF={x_start[rf_mask].norm().item():.2f}")

    # Dataset pixel bounds (RF pixels only) for sigmoid reparametrization
    rf_mask_np = rf_mask.cpu().numpy()
    X_all = torch.cat([X_pool, env['X_train']], dim=0)
    all_rf_vals = X_all.cpu().numpy()[:, rf_mask_np]
    vmin = float(all_rf_vals.min())
    vmax = float(all_rf_vals.max())
    print(f"  Dataset RF pixel range: [{vmin:.3f}, {vmax:.3f}]")

    # === Step 3: LBFGS gradient ascent ===
    print("\n" + "=" * 70)
    print(f"Step 3: LBFGS gradient ascent ({N_STEPS} steps, {n_sample} conditioning images)")
    print("=" * 70)

    pixel_lo = vmin if USE_SIGMOID_BOUNDS else None
    pixel_hi = vmax if USE_SIGMOID_BOUNDS else None
    x_final, history = gradient_ascent_da(
        model, likelihood, x_start, x_samples, rf_mask, f_max,
        pixel_lo, pixel_hi,
        N_STEPS, LR, LBFGS_MAX_ITER, LBFGS_MAX_EVAL, LBFGS_HISTORY_SIZE,
        SAMPLE_LAMBDA,
        adaptive_safety_k=config['adaptive_safety_k'],
        adaptive_max_rmax=config['adaptive_max_rmax'],
        adaptive_min_rmax=config['adaptive_min_rmax'],
    )

    print(f"\nGradient ascent summary:")
    print(f"  Steps: {len(history['step'])}")
    print(f"  U_DA: {history['utility'][0]:.6f} -> {history['utility'][-1]:.6f}")
    print(f"  H_marg: {history['H_marg'][0]:.4f} -> {history['H_marg'][-1]:.4f}")
    print(f"  H_cond: {history['H_cond'][0]:.4f} -> {history['H_cond'][-1]:.4f}")
    print(f"  ||final||_RF={x_final[rf_mask].norm().item():.2f}")

    # === Step 4: Post-hoc analysis ===
    print("\n" + "=" * 70)
    print("Step 4: Post-hoc pool comparison")
    print("=" * 70)

    # Pearson r of final image vs each conditioning image (within RF)
    final_rf = x_final[rf_mask]
    final_rf_c = final_rf - final_rf.mean()
    final_rf_norm = final_rf_c.norm()

    correlations = []
    for i in range(n_sample):
        pool_rf = x_samples[i][rf_mask]
        pool_rf_c = pool_rf - pool_rf.mean()
        pool_rf_norm = pool_rf_c.norm()
        if pool_rf_norm < 1e-12 or final_rf_norm < 1e-12:
            correlations.append(0.0)
        else:
            correlations.append(((final_rf_c * pool_rf_c).sum() / (final_rf_norm * pool_rf_norm)).item())

    correlations = np.array(correlations)
    best_idx = np.argmax(correlations)
    print(f"  Best match: pool[{best_idx}] with Pearson r = {correlations[best_idx]:.4f}")
    print(f"  Correlation range: [{correlations.min():.4f}, {correlations.max():.4f}]")
    print(f"  Mean correlation: {correlations.mean():.4f}")

    # Top 5 matches
    top5 = np.argsort(correlations)[::-1][:5]
    print(f"  Top 5: {[(int(i), f'{correlations[i]:.3f}') for i in top5]}")

    # Firing rate diagnostics
    print(f"\n  Firing rate diagnostics (f_max={f_max})")
    A = likelihood.A.squeeze()
    lam0 = likelihood.lambda0.squeeze()
    with torch.no_grad():
        for label, x_img in [('start', x_start), ('final', x_final),
                              (f'best_pool[{best_idx}]', x_samples[best_idx])]:
            mu, sigma2 = get_gp_marginal_moments(model, x_img.unsqueeze(0))
            mu_g = A * mu + lam0
            sigma2_g = A ** 2 * sigma2
            firing_rate = torch.exp(mu_g).item()
            r_max_diag = compute_adaptive_rmax(mu_g, sigma2_g,
                                               safety_k=config['adaptive_safety_k'],
                                               max_rmax=config['adaptive_max_rmax'],
                                               min_rmax=config['adaptive_min_rmax'])
            H = compute_H(mu, sigma2, r_max=r_max_diag, a=A, lambda0=lam0).item()
            flag = " ** EXCEEDS f_max" if firing_rate > f_max else ""
            print(f"    {label:20s}: lambda_m={mu.item():.4f}, lambda_var={sigma2.item():.4f}, "
                  f"mu_g={mu_g.item():.4f}, rate={firing_rate:.2f}, "
                  f"H_marg={H:.6f}, ||x||_RF={x_img[rf_mask].norm().item():.2f}{flag}")

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
    r_min = max(0, r_min - 1)
    r_max_px = min(n_px_side - 1, r_max_px + 1)
    c_min = max(0, c_min - 1)
    c_max_px = min(n_px_side - 1, c_max_px + 1)

    def masked_crop(x_flat, gray_val):
        img = x_flat.detach().cpu().numpy().reshape(n_px_side, n_px_side).copy()
        img[~mask_2d] = gray_val
        return img[r_min:r_max_px+1, c_min:c_max_px+1]

    gray_val = (vmin + vmax) / 2

    img_start = masked_crop(x_start, gray_val)
    img_final = masked_crop(x_final, gray_val)
    img_best = masked_crop(x_samples[best_idx], gray_val)
    img_diff = masked_crop(x_final - x_start, 0.0)

    fig = plt.figure(figsize=(18, 8))

    # Top row: 4 images
    images = [img_start, img_final, img_best, img_diff]
    titles = [
        f'Start (mean gray + noise)',
        f'Final (grad ascent)\nU_DA={history["utility"][-1]:.4f}',
        f'Best match (pool[{best_idx}])\nr={correlations[best_idx]:.4f}',
        'Final - Start',
    ]
    cmaps = ['gray', 'gray', 'gray', 'RdBu_r']
    for i, (img, title, cmap) in enumerate(zip(images, titles, cmaps)):
        ax = fig.add_subplot(2, 4, i + 1)
        if cmap == 'RdBu_r':
            vlim = max(abs(img.min()), abs(img.max()))
            im = ax.imshow(img, cmap=cmap, vmin=-vlim, vmax=vlim, aspect='equal')
        else:
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)

    # Bottom left: U_DA convergence
    ax1 = fig.add_subplot(2, 4, 5)
    steps = history['step']
    ax1.plot(steps, history['utility'], 'b.-', markersize=4, linewidth=1.5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('U_DA')
    ax1.set_title(f'DA utility ({n_sample} images)')
    ax1.grid(True, alpha=0.3)

    # Bottom center: H_marg and H_cond
    ax2 = fig.add_subplot(2, 4, 6)
    ax2.plot(steps, history['H_marg'], 'r.-', markersize=4, linewidth=1.5, label='H_marg')
    ax2.plot(steps, history['H_cond'], 'b.-', markersize=4, linewidth=1.5, label='H_cond')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Entropy (nats)')
    ax2.set_title('Marginal vs conditional entropy')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Bottom right: pool correlations histogram
    ax3 = fig.add_subplot(2, 4, 7)
    ax3.hist(correlations, bins=20, edgecolor='black', alpha=0.7)
    ax3.axvline(correlations[best_idx], color='red', linestyle='--', linewidth=2,
                label=f'best: pool[{best_idx}]')
    # (start is uniform gray — not a pool image, so no start marker here)
    ax3.set_xlabel('Pearson r (RF)')
    ax3.set_ylabel('Count')
    ax3.set_title('Final vs pool images')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Bottom far right: gradient norm
    ax4 = fig.add_subplot(2, 4, 8)
    ax4.semilogy(steps, history['grad_norm'], 'k.-', markersize=4, linewidth=1.5)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('|grad|')
    ax4.set_title('Gradient norm')
    ax4.grid(True, alpha=0.3)

    title_str = (
        f'DA Utility - RBF Kernel (multi-image conditioning) '
        f'N_sample={n_sample}, M={config["M"]}, n_train={config["n_train"]}, '
        f'seed={config["seed"]}, cell={config["cell"]}  '
        f'test_r={test_r:.3f}'
    )
    fig.suptitle(title_str, fontsize=10)
    fig.tight_layout()

    out_path = _script_dir / 'distribution_gradient.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
