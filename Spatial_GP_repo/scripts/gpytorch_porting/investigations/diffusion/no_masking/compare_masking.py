"""
Compare masked vs full-image optimization for diffusion-guided utility.

Question: Does the UNet work better when the entire 64x64 image is coherent
(all pixels optimized, kernel.use_mask=False) vs the current approach where
only RF pixels are optimized and the rest is frozen natural content?

The spatial non-uniformity (natural outside RF, modified inside) may confuse
the UNet, which was trained on images with uniform statistics at each noise level.

This script:
1. Trains one GP model (shared for both modes)
2. For each target image, runs both:
   a. Masked: gradient_ascent_guided() as-is (optimize RF pixels only)
   b. Full: gradient_ascent_full() (optimize ALL pixels, use_mask=False)
3. Produces per-target comparison plots + summary

Usage:
    python investigations/diffusion/no_masking/compare_masking.py --lambda-diff 1.0
"""

import sys
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Path setup: import from guided_optimization.py and gpytorch_porting
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_diffusion_dir = _script_dir.parent
_gpytorch_dir = _diffusion_dir.parent.parent
sys.path.insert(0, str(_gpytorch_dir))
sys.path.insert(0, str(_diffusion_dir))

from guided_optimization import (
    setup_gp,
    load_diffusion_model,
    gradient_ascent_guided,
    compute_tweedie_denoised,
    rf_pearson_r,
    rf_proj_coeff,
    _reconstruct_image,
    N_STEPS, LR, LBFGS_MAX_ITER, LBFGS_MAX_EVAL, LBFGS_HISTORY_SIZE,
    SIGMA_SMOOTH, PNAS_SIZE, T_SCORE, LAMBDA_DIFF,
)
from acquisition import distribution_aware_utility

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_INDICES = [0, 10, 20, 30, 50]


# ============================================================================
# Full-image optimization (no RF masking)
# ============================================================================

def gradient_ascent_full(model, likelihood, x_start, x_target, rf_mask,
                         r_max, f_max, n_px_side,
                         n_steps, lr, max_iter, max_eval, history_size,
                         diffusion_env, lambda_diff, t_score):
    """Gradient ascent on ALL pixels with kernel.use_mask=False.

    Same combined loss as gradient_ascent_guided but:
    - Optimization variable is x_full (all n_pixels), not x_rf
    - kernel.use_mask = False so utility gradient flows through ALL pixels
    - Diffusion L2 penalty on ALL pixels: ||x_full - x_0_hat||^2
    - Metrics still computed on RF mask for fair comparison with masked mode

    Args: same as gradient_ascent_guided (rf_mask used only for metrics)

    Returns:
        x_final: (n_pixels,) optimized image
        history: dict with per-step metrics (RF-only for comparison)
    """
    n_pixels = x_start.shape[0]
    device = x_start.device
    dtype = x_start.dtype

    # Get kernel and toggle masking OFF
    kernel = getattr(model, 'covar_module', None) or getattr(model, 'kernel', None)
    original_use_mask = kernel.use_mask
    kernel.use_mask = False
    print(f"  kernel.use_mask set to False (was {original_use_mask})")

    x_full = x_start.clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [x_full], lr=lr, max_iter=max_iter, max_eval=max_eval,
        history_size=history_size, line_search_fn='strong_wolfe',
    )
    print(f"  Optimizer: LBFGS (lr={lr}, max_iter={max_iter})")
    print(f"  Optimizing ALL {n_pixels} pixels (no RF mask)")

    history = {
        'step': [], 'utility': [], 'grad_norm': [],
        'pearson_r': [], 'proj_coeff': [],
        'loss_utility': [], 'loss_diffusion': [],
    }

    # Record initial state
    with torch.no_grad():
        try:
            result_init = distribution_aware_utility(
                model, likelihood,
                x_full.unsqueeze(0),
                x_target.unsqueeze(0),
                r_max=r_max,
                adaptive_r_max=False,
                sample_lambda=False,
            )
            utility_init = result_init['utility'].item()
        except Exception:
            utility_init = float('nan')
        pr_init = rf_pearson_r(x_full, x_target, rf_mask)
        pc_init = rf_proj_coeff(x_full, x_target, rf_mask)

    history['step'].append(-1)
    history['utility'].append(utility_init)
    history['grad_norm'].append(0.0)
    history['pearson_r'].append(pr_init)
    history['proj_coeff'].append(pc_init)
    history['loss_utility'].append(-utility_init if not np.isnan(utility_init) else float('nan'))
    history['loss_diffusion'].append(0.0)
    print(f"  init   : U={utility_init:.6f}  r={pr_init:.4f}  proj={pc_init:.4f}")

    for step in range(n_steps):
        # Compute Tweedie target ONCE per outer step (fixed for line search)
        with torch.no_grad():
            x_0_hat = compute_tweedie_denoised(
                x_full.detach(),
                diffusion_env['unet'],
                diffusion_env['schedule'],
                diffusion_env['scale_factor'],
                t_score, device,
                seed=42,
            )

        # LBFGS closure: combined loss on ALL pixels
        def closure():
            optimizer.zero_grad()

            try:
                result = distribution_aware_utility(
                    model, likelihood,
                    x_full.unsqueeze(0),
                    x_target.unsqueeze(0),
                    r_max=r_max,
                    adaptive_r_max=False,
                    sample_lambda=False,
                )
            except Exception:
                return torch.tensor(float('inf'), device=device)

            mu_g = result['mu_g_marg']
            if torch.exp(mu_g).item() > f_max:
                return torch.tensor(float('inf'), device=device)

            loss = -result['utility'].squeeze()
            if torch.isnan(loss):
                return torch.tensor(float('inf'), device=device)

            # Diffusion L2 on ALL pixels
            loss_diff = 0.5 * ((x_full - x_0_hat.detach()) ** 2).sum()
            loss = loss + lambda_diff * loss_diff

            loss.backward()
            return loss

        optimizer.step(closure)

        # Verify gradient on first step
        if step == 0:
            assert x_full.grad is not None and x_full.grad.norm() > 0, \
                "No gradient flow"
            print(f"  Step 0: |grad|={x_full.grad.norm().item():.4e}")

        # Track metrics on RF mask (for comparison with masked mode)
        with torch.no_grad():
            try:
                result = distribution_aware_utility(
                    model, likelihood,
                    x_full.unsqueeze(0),
                    x_target.unsqueeze(0),
                    r_max=r_max,
                    adaptive_r_max=False,
                    sample_lambda=False,
                )
                utility = result['utility'].item()
            except Exception:
                utility = float('nan')
            pr = rf_pearson_r(x_full, x_target, rf_mask)
            pc = rf_proj_coeff(x_full, x_target, rf_mask)

            x_0_hat_track = compute_tweedie_denoised(
                x_full.detach(), diffusion_env['unet'],
                diffusion_env['schedule'],
                diffusion_env['scale_factor'],
                t_score, device, seed=42)
            ld = 0.5 * ((x_full - x_0_hat_track) ** 2).sum().item()

        grad_norm = x_full.grad.norm().item() if x_full.grad is not None else 0.0

        history['step'].append(step)
        history['utility'].append(utility)
        history['grad_norm'].append(grad_norm)
        history['pearson_r'].append(pr)
        history['proj_coeff'].append(pc)
        history['loss_utility'].append(-utility if not np.isnan(utility) else float('nan'))
        history['loss_diffusion'].append(ld)

        if np.isnan(utility) or np.isnan(grad_norm):
            print(f"  step {step}: NaN detected - stopping")
            break

        if step % 5 == 0 or step == n_steps - 1:
            print(f"  step {step:4d}: U={utility:.6f}  "
                  f"r={pr:.4f}  proj={pc:.4f}  |grad|={grad_norm:.4e}  L_diff={ld:.4f}")

    # Restore kernel masking
    kernel.use_mask = original_use_mask
    print(f"  kernel.use_mask restored to {original_use_mask}")

    return x_full.detach(), history


# ============================================================================
# Plotting
# ============================================================================

def plot_comparison(target_idx, x_target, x_start,
                    x_final_masked, history_masked,
                    x_final_full, history_full,
                    rf_mask, vmin, vmax, n_px_side, out_path):
    """Side-by-side comparison plot for one target image."""
    mask_2d = rf_mask.cpu().numpy().reshape(n_px_side, n_px_side)

    def to_image(x_flat):
        return x_flat.detach().cpu().numpy().reshape(n_px_side, n_px_side)

    def check_oob(x_flat, label):
        vals = x_flat[rf_mask].cpu().numpy()
        n_oob = np.sum(vals < vmin) + np.sum(vals > vmax)
        pct = 100.0 * n_oob / len(vals) if len(vals) > 0 else 0.0
        if pct > 0:
            print(f"  {label}: {pct:.1f}% OOB")
        return pct > 0

    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Top row: images
    panels = [
        ('Target', x_target),
        ('Start', x_start),
        ('Masked final', x_final_masked),
        ('Full final', x_final_full),
    ]
    for i, (label, x) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        oob = check_oob(x, label)
        ax.imshow(to_image(x), cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
        ax.contour(mask_2d, levels=[0.5], colors='cyan', linewidths=0.8)
        ax.set_title(label, fontsize=10, color='red' if oob else 'black')
        ax.axis('off')

    # Bottom row: convergence curves (overlay masked vs full)
    ax_u = fig.add_subplot(gs[1, 0:2])
    ax_u.plot(history_masked['step'], history_masked['utility'], 'b-',
              linewidth=1.5, label='Masked')
    ax_u.plot(history_full['step'], history_full['utility'], 'r--',
              linewidth=1.5, label='Full')
    ax_u.set_xlabel('Step')
    ax_u.set_ylabel('U_DA')
    ax_u.set_title('Utility convergence')
    ax_u.legend(fontsize=8)
    ax_u.grid(True, alpha=0.3)

    ax_r = fig.add_subplot(gs[1, 2])
    ax_r.plot(history_masked['step'], history_masked['pearson_r'], 'b-',
              linewidth=1.5, label='Masked')
    ax_r.plot(history_full['step'], history_full['pearson_r'], 'r--',
              linewidth=1.5, label='Full')
    ax_r.set_xlabel('Step')
    ax_r.set_ylabel('Pearson r (RF)')
    ax_r.set_title('Structural similarity')
    ax_r.legend(fontsize=8)
    ax_r.grid(True, alpha=0.3)

    ax_p = fig.add_subplot(gs[1, 3])
    ax_p.plot(history_masked['step'], history_masked['proj_coeff'], 'b-',
              linewidth=1.5, label='Masked')
    ax_p.plot(history_full['step'], history_full['proj_coeff'], 'r--',
              linewidth=1.5, label='Full')
    ax_p.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
    ax_p.set_xlabel('Step')
    ax_p.set_ylabel('Proj coeff (RF)')
    ax_p.set_title('Amplitude')
    ax_p.legend(fontsize=8)
    ax_p.grid(True, alpha=0.3)

    fig.suptitle(f'Masked vs Full optimization  (target={target_idx})', fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_summary(results, out_path):
    """Summary bar chart across all target images."""
    indices = [r['target_idx'] for r in results]
    u_masked = [r['u_masked'] for r in results]
    u_full = [r['u_full'] for r in results]
    r_masked = [r['r_masked'] for r in results]
    r_full = [r['r_full'] for r in results]

    x = np.arange(len(indices))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x - width/2, u_masked, width, label='Masked', color='steelblue')
    ax1.bar(x + width/2, u_full, width, label='Full', color='indianred')
    ax1.set_xlabel('Target index')
    ax1.set_ylabel('Final U_DA')
    ax1.set_title('Utility')
    ax1.set_xticks(x)
    ax1.set_xticklabels(indices)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(x - width/2, r_masked, width, label='Masked', color='steelblue')
    ax2.bar(x + width/2, r_full, width, label='Full', color='indianred')
    ax2.set_xlabel('Target index')
    ax2.set_ylabel('Final Pearson r (RF)')
    ax2.set_title('Structural similarity')
    ax2.set_xticks(x)
    ax2.set_xticklabels(indices)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Masked vs Full: Summary across targets', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary: {out_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare masked vs full-image diffusion-guided optimization')
    parser.add_argument('--crop-size', type=int, default=64)
    parser.add_argument('--lambda-diff', type=float, default=LAMBDA_DIFF)
    parser.add_argument('--t-score', type=int, default=T_SCORE)
    parser.add_argument('--checkpoint', type=str,
                        default=str(_diffusion_dir / 'checkpoints' / 'ddpm_epoch1000.pt'))
    parser.add_argument('--targets', type=int, nargs='+', default=TARGET_INDICES,
                        help='Pool indices for target images')
    args = parser.parse_args()

    # === Train GP once (shared model) ===
    env = setup_gp(crop_size=args.crop_size)
    model = env['model']
    likelihood = env['likelihood']
    X_pool = env['X_pool']
    config = env['config']
    test_r = env['test_r']

    n_px_side = config['n_px_side']
    n_pixels = n_px_side ** 2
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    r_max = config['r_max']
    f_max = config['f_max']

    # Get RF mask
    kernel = getattr(model, 'covar_module', None) or getattr(model, 'kernel', None)
    if not (hasattr(kernel, '_cached_mask') and kernel._cached_mask is not None):
        with torch.no_grad():
            _ = model(X_pool[0].unsqueeze(0))
    rf_mask = kernel._cached_mask.squeeze()
    rf_mask_np = rf_mask.cpu().numpy()

    # Dataset pixel bounds
    X_all = torch.cat([X_pool, env['X_train']], dim=0)
    all_rf_vals = X_all.cpu().numpy()[:, rf_mask_np]
    vmin = float(all_rf_vals.min())
    vmax = float(all_rf_vals.max())
    print(f"\nDataset RF pixel range: [{vmin:.3f}, {vmax:.3f}]")

    # === Load diffusion model ===
    print(f"\n{'=' * 60}")
    print("Loading diffusion model")
    print(f"{'=' * 60}")
    diffusion_env = load_diffusion_model(args.checkpoint, device)

    # === Run comparison for each target ===
    results = []
    sigma_scaled = SIGMA_SMOOTH * (PNAS_SIZE / n_px_side)

    for target_idx in args.targets:
        if target_idx >= X_pool.shape[0]:
            print(f"\nSkipping target {target_idx}: only {X_pool.shape[0]} pool images")
            continue

        print(f"\n{'=' * 60}")
        print(f"TARGET {target_idx}")
        print(f"{'=' * 60}")

        x_target = X_pool[target_idx]
        x_target_2d = x_target.cpu().numpy().reshape(n_px_side, n_px_side)
        x_smoothed_2d = gaussian_filter(x_target_2d, sigma=sigma_scaled)

        # Start image: smooth only RF pixels, keep original outside RF
        x_start_np = x_target_2d.copy().reshape(-1)
        x_start_np[rf_mask_np] = x_smoothed_2d.reshape(-1)[rf_mask_np]
        x_start = torch.tensor(x_start_np, dtype=dtype, device=device)

        print(f"  ||target||_RF={x_target[rf_mask].norm().item():.2f}, "
              f"||start||_RF={x_start[rf_mask].norm().item():.2f}")

        # --- Mode A: Masked optimization ---
        print(f"\n  --- Masked optimization (use_mask=True) ---")
        x_final_masked, history_masked = gradient_ascent_guided(
            model, likelihood, x_start, x_target, rf_mask,
            r_max, f_max, n_px_side,
            N_STEPS, LR, LBFGS_MAX_ITER, LBFGS_MAX_EVAL, LBFGS_HISTORY_SIZE,
            diffusion_env=diffusion_env,
            lambda_diff=args.lambda_diff,
            t_score=args.t_score,
            x_background=x_start,
        )

        # --- Mode B: Full optimization ---
        print(f"\n  --- Full optimization (use_mask=False) ---")
        x_final_full, history_full = gradient_ascent_full(
            model, likelihood, x_start, x_target, rf_mask,
            r_max, f_max, n_px_side,
            N_STEPS, LR, LBFGS_MAX_ITER, LBFGS_MAX_EVAL, LBFGS_HISTORY_SIZE,
            diffusion_env=diffusion_env,
            lambda_diff=args.lambda_diff,
            t_score=args.t_score,
        )

        # --- Comparison plot ---
        out_path = _script_dir / f'compare_target{target_idx}.png'
        plot_comparison(
            target_idx, x_target, x_start,
            x_final_masked, history_masked,
            x_final_full, history_full,
            rf_mask, vmin, vmax, n_px_side, out_path,
        )

        # Collect results for summary
        results.append({
            'target_idx': target_idx,
            'u_masked': history_masked['utility'][-1],
            'u_full': history_full['utility'][-1],
            'r_masked': history_masked['pearson_r'][-1],
            'r_full': history_full['pearson_r'][-1],
        })

        print(f"\n  Summary for target {target_idx}:")
        print(f"    Masked: U={results[-1]['u_masked']:.6f}, r={results[-1]['r_masked']:.4f}")
        print(f"    Full:   U={results[-1]['u_full']:.6f}, r={results[-1]['r_full']:.4f}")

    # === Summary plot ===
    if len(results) > 1:
        plot_summary(results, _script_dir / 'summary.png')

    # === Print summary table ===
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Target':>6}  {'U_masked':>10}  {'U_full':>10}  {'r_masked':>10}  {'r_full':>10}")
    for r in results:
        print(f"  {r['target_idx']:>6}  {r['u_masked']:>10.6f}  {r['u_full']:>10.6f}  "
              f"{r['r_masked']:>10.4f}  {r['r_full']:>10.4f}")


if __name__ == '__main__':
    main()
