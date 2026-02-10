"""
Gradient investigation for variational GP utility functions.
Created by Claude.

Trains a GP model and evaluates utility functions with gradient flow enabled,
computing dU/dx* for each candidate image. This verifies that:

1. Gradients flow correctly from utility values back to input pixels
2. Gradient norms are non-trivial (not all zero, not all same)
3. Gradient structure reflects the receptive field mask (non-zero only within RF)
4. Standard and distribution-aware utilities produce consistent gradient patterns

Additionally, performs gradient ascent on a single image to maximize utility:
5. DA utility with single conditioning target — should converge toward target
6. Standard utility as control — no reason to converge toward target

Technical details:
- For standard utility, dU(x_i)/dx_j = 0 for i != j (each candidate's utility
  depends only on its own kernel values). So .sum().backward() gives correct
  per-candidate gradients.
- For DA utility: same property holds. Conditioning on x_sample involves
  cross-covariance k(x_sample, x_star_i) which depends only on x_star_i.
- Arc-cosine kernel gradient w.r.t. pixels: non-zero only within the RF mask.

Model: default_gpy mode. All params read from default_params.json via
build_config_from_defaults(), with explicit overrides at script top.

Usage:
    python investigations/validate_utility/gradient.py
"""

import sys
import time
import torch
import numpy as np
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

from run_single_mode import run_single_config, build_config_from_defaults
from acquisition import standard_utility, distribution_aware_utility

# ---------------------------------------------------------------------------
# Investigation parameters (visible, explicit overrides from defaults)
# ---------------------------------------------------------------------------
N_CANDIDATES = 20          # Number of natural images to evaluate utility on
N_MC_SAMPLES = 100         # MC samples for DA utility (kept small for memory with grads)
N_TRAIN = 50               # Training set size
M = 50                     # Number of inducing points
MIN_TEST_R = 0.3           # Minimum model quality to proceed

# --- Gradient ascent parameters ---
TARGET_INDEX = 0           # Which pool image to use as target (index into X_pool)
N_STEPS = 1000000             # Number of gradient ascent steps
LR = 1.0                   # Learning rate for plain gradient ascent
NOISE_SCALE = 0.1          # Perturbation std (relative to pixel std within RF)
LOG_EVERY = 10             # Print diagnostics every N steps


def check_rf_mask_structure(grad, model, label=""):  # investigation needed
    """Check that gradient sparsity matches the kernel's RF mask.

    The ArcCosineKernel stores the mask as kernel._cached_mask (boolean,
    computed on first forward pass) when use_mask=True.

    Args:
        grad: (N_candidates, n_pixels) gradient tensor.
        model: GP model with kernel that may have an RF mask.
        label: String label for print output.
    """
    kernel = model.covar_module
    if hasattr(kernel, '_cached_mask') and kernel._cached_mask is not None:
        mask = kernel._cached_mask  # (n_pixels,) boolean
        n_masked = int(mask.sum().item())
        n_pixels = mask.shape[0]

        # Check: are non-zero gradients confined to masked pixels?
        mask_float = mask.float()
        grad_nonzero = (grad.abs() > 1e-10).float()  # (N_candidates, n_pixels)
        outside_rf = grad_nonzero * (1 - mask_float.unsqueeze(0))
        n_outside = int(outside_rf.sum().item())
        n_inside = int((grad_nonzero * mask_float.unsqueeze(0)).sum().item())

        print(f"  [{label}] RF mask: {n_masked}/{n_pixels} pixels active")
        print(f"  [{label}] Non-zero grads inside RF: {n_inside}")
        print(f"  [{label}] Non-zero grads outside RF: {n_outside} (should be 0)")
    elif hasattr(kernel, 'use_mask') and not kernel.use_mask:
        print(f"  [{label}] Kernel has use_mask=False — no RF mask applied")
    else:
        print(f"  [{label}] No cached RF mask found (run model forward first?)")


def compute_closeness_metrics(x_opt, x_target, model, rf_mask, initial_pixel_dist):
    """Compute closeness metrics between optimized and target images.

    Args:
        x_opt: (n_pixels,) current optimized image.
        x_target: (n_pixels,) target image.
        model: GP model (for kernel evaluation).
        rf_mask: (n_pixels,) boolean mask for RF pixels.
        initial_pixel_dist: scalar, ||x_start - x_target||_2 at step 0.

    Returns:
        dict with metric values.
    """
    with torch.no_grad():
        pixel_dist = (x_opt - x_target).norm().item()
        frac_dist = pixel_dist / initial_pixel_dist if initial_pixel_dist > 0 else 0.0
        rf_pixel_dist = (x_opt[rf_mask] - x_target[rf_mask]).norm().item()
        kernel_sim = model.covar_module(
            x_opt.unsqueeze(0), x_target.unsqueeze(0)
        ).evaluate().squeeze().item()
        pixel_min = x_opt.min().item()
        pixel_max = x_opt.max().item()

    return {
        'pixel_dist': pixel_dist,
        'frac_dist': frac_dist,
        'rf_pixel_dist': rf_pixel_dist,
        'kernel_sim': kernel_sim,
        'pixel_min': pixel_min,
        'pixel_max': pixel_max,
    }


def run_gradient_ascent(model, likelihood, x_start, x_target, rf_mask,
                        initial_pixel_dist, r_max, use_da=True, label=""):
    """Run gradient ascent to maximize utility starting from x_start.

    Args:
        model: Trained GP model in eval mode.
        likelihood: PoissonLikelihood with .A and .lambda0.
        x_start: (n_pixels,) starting image (perturbed target).
        x_target: (n_pixels,) target image (for DA conditioning and metrics).
        rf_mask: (n_pixels,) boolean RF mask.
        initial_pixel_dist: scalar, ||x_start - x_target||_2.
        r_max: Max spike count for Laplace truncation.
        use_da: If True, use DA utility with x_target as single conditioning
            sample. If False, use standard utility.
        label: String label for print output.

    Returns:
        dict with 'x_final' (optimized image), 'history' (list of per-step dicts).
    """
    x_opt = x_start.clone().detach().requires_grad_(True)
    history = []

    for step in range(N_STEPS):
        # Forward pass
        if use_da:
            result = distribution_aware_utility(
                model, likelihood,
                x_opt.unsqueeze(0),
                x_target.unsqueeze(0),
                r_max=r_max, sample_lambda=False,
            )
        else:
            result = standard_utility(model, likelihood, x_opt.unsqueeze(0), r_max=r_max)

        utility = result['utility'].squeeze()

        # Backward
        utility.backward()

        # Log metrics
        grad_norm = x_opt.grad.norm().item()
        metrics = compute_closeness_metrics(
            x_opt.detach(), x_target, model, rf_mask, initial_pixel_dist
        )
        metrics['utility'] = utility.item()
        metrics['grad_norm'] = grad_norm
        metrics['step'] = step
        history.append(metrics)

        # NaN check — stop early if optimization diverged
        if np.isnan(metrics['utility']) or np.isnan(grad_norm):
            print(f"  [{label}] step {step:4d}: NaN detected — stopping early")
            break

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            print(f"  [{label}] step {step:4d}: U={metrics['utility']:.6f}  "
                  f"frac_dist={metrics['frac_dist']:.4f}  "
                  f"rf_dist={metrics['rf_pixel_dist']:.4f}  "
                  f"k_sim={metrics['kernel_sim']:.4f}  "
                  f"|grad|={grad_norm:.4e}  "
                  f"px=[{metrics['pixel_min']:.3f}, {metrics['pixel_max']:.3f}]")

        # Gradient step (in-place update under no_grad)
        with torch.no_grad():
            x_opt += LR * x_opt.grad

        # Reset grad
        x_opt.grad = None

    return {'x_final': x_opt.detach(), 'history': history, 'diverged': np.isnan(history[-1]['utility'])}


def main():
    # =========================================================================
    # Step 1: Train model
    # =========================================================================
    print("=" * 70)
    print("Step 1: Training default_gpy model")
    print("=" * 70)

    config = build_config_from_defaults(
        mode='default_gpy',
        M=M,
        n_train=N_TRAIN,
        n_mc_samples=N_MC_SAMPLES,
    )

    result = run_single_config(config)

    if result is None or result.get('status') != 'success':
        print("ERROR: Training failed")
        return

    model = result['_model']
    likelihood = result['_likelihood']
    indices_train = result['_indices_train']
    test_r = result['test_r']

    print(f"\nModel trained: test_r={test_r:.4f}")
    print(f"A={likelihood.A.item():.4f}, lambda0={likelihood.lambda0.item():.4f}")

    if test_r < MIN_TEST_R:
        print(f"\nBLOCKING: test_r={test_r:.4f} < {MIN_TEST_R}")
        print("Model too poor for meaningful gradient investigation.")
        print("Consider increasing N_TRAIN or changing seed.")
        return

    model.eval()
    r_max = config['r_max']

    # =========================================================================
    # Step 2: Build candidate and sample sets
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Building candidate and sample sets")
    print("=" * 70)

    data_path = _gpytorch_dir.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = np.load(data_path)
    dtype = torch.float32
    device = next(model.parameters()).device

    X_all = torch.cat([
        torch.tensor(data['images_train'], dtype=dtype),
        torch.tensor(data['images_val'], dtype=dtype),
    ], dim=0).reshape(-1, config['n_px_side'] ** 2).to(device)

    # Pool: everything not in training
    all_indices = set(range(X_all.shape[0]))
    train_indices = set(indices_train.cpu().numpy().tolist())
    pool_indices = sorted(all_indices - train_indices)
    X_pool = X_all[pool_indices]

    # Candidates: first N_CANDIDATES from pool
    X_candidates = X_pool[:N_CANDIDATES].clone()

    # MC samples: next N_MC_SAMPLES from pool (non-overlapping with candidates)
    x_samples = X_pool[N_CANDIDATES:N_CANDIDATES + N_MC_SAMPLES]

    print(f"Pool size: {X_pool.shape[0]} images")
    print(f"Candidates: {N_CANDIDATES} natural images")
    print(f"MC samples: {N_MC_SAMPLES} natural images (non-overlapping with candidates)")

    # =========================================================================
    # Step 3: Standard utility with gradients
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Standard utility with gradient flow")
    print("=" * 70)

    X_cand_std = X_candidates.clone().detach().requires_grad_(True)

    t0 = time.time()
    std_result = standard_utility(model, likelihood, X_cand_std, r_max=r_max)
    std_utility = std_result['utility']  # (N_CANDIDATES,)

    # Backward: sum over candidates (cross-derivatives are zero)
    std_utility.sum().backward()
    std_time = time.time() - t0

    std_grad = X_cand_std.grad.clone()  # (N_CANDIDATES, n_pixels)

    print(f"Computed in {std_time:.1f}s")
    grad_norms_std = std_grad.norm(dim=1)
    print(f"  Utility range: [{std_utility.min().item():.6f}, {std_utility.max().item():.6f}]")
    print(f"  Gradient norm range: [{grad_norms_std.min().item():.6e}, {grad_norms_std.max().item():.6e}]")
    print(f"  Mean gradient norm: {grad_norms_std.mean().item():.6e}")
    n_nonzero_std = (std_grad.abs() > 1e-10).sum().item()
    print(f"  Non-zero grad entries: {n_nonzero_std} / {std_grad.numel()}")

    check_rf_mask_structure(std_grad, model, label="Standard")

    # =========================================================================
    # Step 4: Distribution-aware utility with gradients
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Distribution-aware utility with gradient flow")
    print("=" * 70)

    X_cand_da = X_candidates.clone().detach().requires_grad_(True)

    t0 = time.time()
    da_result = distribution_aware_utility(
        model, likelihood, X_cand_da, x_samples,
        r_max=r_max, sample_lambda=False,  # deterministic for reproducibility
    )
    da_utility = da_result['utility']  # (N_CANDIDATES,)

    da_utility.sum().backward()
    da_time = time.time() - t0

    da_grad = X_cand_da.grad.clone()

    print(f"Computed in {da_time:.1f}s")
    grad_norms_da = da_grad.norm(dim=1)
    print(f"  Utility range: [{da_utility.min().item():.6f}, {da_utility.max().item():.6f}]")
    print(f"  H_marg range: [{da_result['H_marg'].min().item():.6f}, {da_result['H_marg'].max().item():.6f}]")
    print(f"  H_cond range: [{da_result['H_cond'].min().item():.6f}, {da_result['H_cond'].max().item():.6f}]")
    print(f"  Gradient norm range: [{grad_norms_da.min().item():.6e}, {grad_norms_da.max().item():.6e}]")
    print(f"  Mean gradient norm: {grad_norms_da.mean().item():.6e}")
    n_nonzero_da = (da_grad.abs() > 1e-10).sum().item()
    print(f"  Non-zero grad entries: {n_nonzero_da} / {da_grad.numel()}")

    check_rf_mask_structure(da_grad, model, label="DA")

    # =========================================================================
    # Step 5: Compare gradient structures
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Comparison of standard vs DA gradients")
    print("=" * 70)

    # Correlation of gradient norms across candidates
    corr = torch.corrcoef(torch.stack([grad_norms_std, grad_norms_da]))[0, 1]
    print(f"Correlation of gradient norms (std vs DA): {corr.item():.4f}")

    # Per-candidate cosine similarity between gradient directions
    cos_sim = torch.nn.functional.cosine_similarity(std_grad, da_grad, dim=1)
    print(f"Cosine similarity range: [{cos_sim.min().item():.4f}, {cos_sim.max().item():.4f}]")
    print(f"Mean cosine similarity: {cos_sim.mean().item():.4f}")

    # =========================================================================
    # Step 6: Summary table
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary: Per-candidate results")
    print("=" * 70)
    print(f"{'Idx':>4}  {'Std U':>10}  {'DA U':>10}  "
          f"{'|grad| Std':>12}  {'|grad| DA':>12}  {'cos_sim':>8}")
    print("-" * 62)
    for i in range(N_CANDIDATES):
        print(f"{i:4d}  {std_utility[i].item():10.6f}  {da_utility[i].item():10.6f}  "
              f"{grad_norms_std[i].item():12.6e}  {grad_norms_da[i].item():12.6e}  "
              f"{cos_sim[i].item():8.4f}")

    # =========================================================================
    # Step 7: Setup gradient ascent experiment
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 7: Setup gradient ascent experiment")
    print("=" * 70)

    # Target image
    x_target = X_pool[TARGET_INDEX].clone()
    print(f"Target image: pool index {TARGET_INDEX}")
    print(f"  Pixel range: [{x_target.min().item():.4f}, {x_target.max().item():.4f}]")

    # RF mask
    kernel = model.covar_module
    if not (hasattr(kernel, '_cached_mask') and kernel._cached_mask is not None):
        print("ERROR: No RF mask found. Cannot confine perturbation to RF.")
        return
    rf_mask = kernel._cached_mask  # (n_pixels,) boolean
    n_rf = int(rf_mask.sum().item())
    print(f"  RF mask: {n_rf}/{rf_mask.shape[0]} pixels active")

    # Perturbation: Gaussian noise within RF only
    rf_pixel_std = x_target[rf_mask].std().item()
    noise = torch.zeros_like(x_target)
    noise[rf_mask] = torch.randn(n_rf, dtype=dtype, device=device) * NOISE_SCALE * rf_pixel_std
    x_start = x_target + noise

    initial_pixel_dist = noise.norm().item()
    initial_rf_dist = noise[rf_mask].norm().item()
    print(f"  Perturbation: NOISE_SCALE={NOISE_SCALE}, rf_pixel_std={rf_pixel_std:.4f}")
    print(f"  Initial pixel distance: {initial_pixel_dist:.4f}")
    print(f"  Initial RF pixel distance: {initial_rf_dist:.4f}")

    # Baseline metrics at x_start and x_target
    metrics_start = compute_closeness_metrics(x_start, x_target, model, rf_mask, initial_pixel_dist)
    metrics_at_target = compute_closeness_metrics(x_target, x_target, model, rf_mask, initial_pixel_dist)

    print(f"\n  Baseline at x_start:")
    print(f"    kernel_sim = {metrics_start['kernel_sim']:.6f}")

    # Utility at target (theoretical peak for DA)
    with torch.no_grad():
        da_at_target = distribution_aware_utility(
            model, likelihood, x_target.unsqueeze(0), x_target.unsqueeze(0),
            r_max=r_max, sample_lambda=False,
        )
        std_at_target = standard_utility(model, likelihood, x_target.unsqueeze(0), r_max=r_max)
    print(f"\n  Reference utility at x_target:")
    print(f"    DA utility  = {da_at_target['utility'].item():.6f}")
    print(f"    Std utility = {std_at_target['utility'].item():.6f}")
    print(f"    kernel_sim(target, target) = {metrics_at_target['kernel_sim']:.6f}")

    # =========================================================================
    # Step 8: Gradient ascent — DA utility
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 8: Gradient ascent — DA utility (single target)")
    print("=" * 70)
    print(f"  N_STEPS={N_STEPS}, LR={LR}, sample_lambda=False")

    t0 = time.time()
    da_ascent = run_gradient_ascent(
        model, likelihood, x_start, x_target, rf_mask,
        initial_pixel_dist, r_max, use_da=True, label="DA",
    )
    da_time = time.time() - t0
    print(f"\n  Completed in {da_time:.1f}s")

    # =========================================================================
    # Step 9: Gradient ascent — standard utility (control)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 9: Gradient ascent — standard utility (control)")
    print("=" * 70)
    print(f"  N_STEPS={N_STEPS}, LR={LR}")

    t0 = time.time()
    std_ascent = run_gradient_ascent(
        model, likelihood, x_start, x_target, rf_mask,
        initial_pixel_dist, r_max, use_da=False, label="Std",
    )
    std_time = time.time() - t0
    print(f"\n  Completed in {std_time:.1f}s")

    # =========================================================================
    # Step 10: Compare trajectories
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 10: Trajectory comparison")
    print("=" * 70)

    da_h = da_ascent['history']
    std_h = std_ascent['history']

    # Find last non-NaN entry for each
    da_h_valid = [h for h in da_h if not np.isnan(h['utility'])]
    std_h_valid = [h for h in std_h if not np.isnan(h['utility'])]

    print(f"\n  DA: {len(da_h_valid)}/{len(da_h)} valid steps"
          f"{'  (DIVERGED — NaN at step ' + str(da_h[-1]['step']) + ')' if da_ascent['diverged'] else ''}")
    print(f"  Std: {len(std_h_valid)}/{len(std_h)} valid steps"
          f"{'  (DIVERGED — NaN at step ' + str(std_h[-1]['step']) + ')' if std_ascent['diverged'] else ''}")

    if da_h_valid and std_h_valid:
        print(f"\n{'':>8}  {'DA utility':>14}  {'Std utility':>14}")
        print(f"{'Metric':>8}  {'start':>7} {'final':>7}  {'start':>7} {'final':>7}")
        print("-" * 55)

        for key, fmt in [('utility', '.6f'), ('frac_dist', '.4f'),
                         ('rf_pixel_dist', '.4f'), ('kernel_sim', '.4f')]:
            da_s, da_f = da_h_valid[0][key], da_h_valid[-1][key]
            std_s, std_f = std_h_valid[0][key], std_h_valid[-1][key]
            print(f"{key:>14}  {da_s:>7{fmt}} {da_f:>7{fmt}}  {std_s:>7{fmt}} {std_f:>7{fmt}}")
    elif da_h_valid:
        print(f"\n  DA results (Std diverged):")
        for key, fmt in [('utility', '.6f'), ('frac_dist', '.4f'),
                         ('rf_pixel_dist', '.4f'), ('kernel_sim', '.4f')]:
            print(f"    {key}: {da_h_valid[0][key]:{fmt}} -> {da_h_valid[-1][key]:{fmt}}")

    # Pixel range check (only from valid entries)
    with torch.no_grad():
        pool_min = X_pool.min().item()
        pool_max = X_pool.max().item()
    print(f"\n  Pixel range (natural images): [{pool_min:.3f}, {pool_max:.3f}]")

    for label_name, h_valid, ascent_result in [("DA", da_h_valid, da_ascent),
                                                ("Std", std_h_valid, std_ascent)]:
        if h_valid:
            px_min = min(h['pixel_min'] for h in h_valid)
            px_max = max(h['pixel_max'] for h in h_valid)
            print(f"  {label_name} pixel range during optimization: [{px_min:.3f}, {px_max:.3f}]")
            if px_min < pool_min - 0.1 or px_max > pool_max + 0.1:
                flag = "WARNING" if label_name == "DA" else "NOTE"
                print(f"  {flag}: {label_name} utility pushed pixels outside natural image bounds")
        else:
            print(f"  {label_name}: no valid steps (diverged immediately)")

    # Convergence check
    if da_h_valid:
        da_converged = da_h_valid[-1]['frac_dist'] < da_h_valid[0]['frac_dist']
        print(f"\n  DA frac_dist: {da_h_valid[0]['frac_dist']:.4f} -> {da_h_valid[-1]['frac_dist']:.4f}  "
              f"({'CONVERGING' if da_converged else 'NOT CONVERGING'})")
    if std_h_valid:
        print(f"  Std frac_dist: {std_h_valid[0]['frac_dist']:.4f} -> {std_h_valid[-1]['frac_dist']:.4f}")

    # =========================================================================
    # Step 11: Visualization
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 11: Visualization")
    print("=" * 70)

    n_px_side = config['n_px_side']
    save_dir = _script_dir
    x_final_da = da_ascent['x_final']
    x_final_std = std_ascent['x_final']
    da_ok = not da_ascent['diverged']
    std_ok = not std_ascent['diverged']

    if da_ok:
        print(f"  DA: {len(da_h)} steps, final utility={da_h[-1]['utility']:.6f}")
    else:
        print(f"  DA: DIVERGED at step {da_h[-1]['step']} (NaN)")
    if std_ok:
        print(f"  Std: {len(std_h)} steps, final utility={std_h[-1]['utility']:.6f}")
    else:
        print(f"  Std: DIVERGED at step {std_h[-1]['step']} (NaN)")

    # Filter out NaN entries for plotting
    da_h_clean = [h for h in da_h if not np.isnan(h['utility'])]
    std_h_clean = [h for h in std_h if not np.isnan(h['utility'])]

    # --- Figure 1: Images side-by-side ---
    # Only include non-NaN final images
    images_row1 = [
        (x_target, "Target"),
        (x_start, "Start (perturbed)"),
    ]
    diffs_row2 = [
        (torch.zeros_like(x_target), "Target (reference)"),
        (x_start - x_target, "Start - Target"),
    ]
    if da_ok:
        images_row1.append((x_final_da, "Final (DA)"))
        diffs_row2.append((x_final_da - x_target, "DA Final - Target"))
    else:
        images_row1.append((None, "Final (DA) — NaN"))
        diffs_row2.append((None, "DA — NaN"))
    if std_ok:
        images_row1.append((x_final_std, "Final (Std)"))
        diffs_row2.append((x_final_std - x_target, "Std Final - Target"))
    else:
        images_row1.append((None, "Final (Std) — NaN"))
        diffs_row2.append((None, "Std — NaN"))

    n_cols = len(images_row1)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    # Row 1: Images — color scale from non-NaN images only
    valid_pixels = [img for img, _ in images_row1 if img is not None]
    all_pixels = torch.cat(valid_pixels)
    vmin, vmax = all_pixels.min().item(), all_pixels.max().item()

    for ax, (img, title) in zip(axes[0], images_row1):
        if img is not None:
            im = ax.imshow(img.cpu().reshape(n_px_side, n_px_side).numpy(),
                           cmap='gray', vmin=vmin, vmax=vmax)
        else:
            ax.text(0.5, 0.5, "NaN\n(diverged)", ha='center', va='center',
                    fontsize=14, transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    fig.colorbar(im, ax=axes[0, -1], fraction=0.046, pad=0.04)

    # Row 2: Differences — color scale from non-NaN diffs only
    valid_diffs = [d for d, _ in diffs_row2 if d is not None]
    max_diff = max(d.abs().max().item() for d in valid_diffs) if valid_diffs else 1.0

    for ax, (diff, title) in zip(axes[1], diffs_row2):
        if diff is not None:
            im = ax.imshow(diff.cpu().reshape(n_px_side, n_px_side).numpy(),
                           cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
        else:
            ax.text(0.5, 0.5, "NaN\n(diverged)", ha='center', va='center',
                    fontsize=14, transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    fig.colorbar(im, ax=axes[1, -1], fraction=0.046, pad=0.04)

    # Suptitle with available info
    suptitle_parts = [f"Gradient Ascent: LR={LR}, N_STEPS={N_STEPS}, NOISE_SCALE={NOISE_SCALE}"]
    if da_h_clean:
        suptitle_parts.append(f"DA frac_dist: {da_h_clean[0]['frac_dist']:.4f} -> {da_h_clean[-1]['frac_dist']:.4f}")
    else:
        suptitle_parts.append("DA: diverged")
    if std_h_clean:
        suptitle_parts.append(f"Std frac_dist: {std_h_clean[0]['frac_dist']:.4f} -> {std_h_clean[-1]['frac_dist']:.4f}")
    else:
        suptitle_parts.append("Std: diverged")
    fig.suptitle("\n".join(suptitle_parts), fontsize=12)
    fig.tight_layout()
    fig1_path = save_dir / 'gradient_ascent_images.png'
    fig.savefig(fig1_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig1_path}")

    # --- Figure 2: Metrics over steps (only non-NaN data) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Utility
    ax = axes[0, 0]
    if da_h_clean:
        ax.plot([h['step'] for h in da_h_clean], [h['utility'] for h in da_h_clean], 'b-', label='DA')
    if std_h_clean:
        ax.plot([h['step'] for h in std_h_clean], [h['utility'] for h in std_h_clean], 'r-', label='Std')
    ax.axhline(da_at_target['utility'].item(), color='b', linestyle='--', alpha=0.5,
               label=f'DA at target ({da_at_target["utility"].item():.6f})')
    ax.axhline(std_at_target['utility'].item(), color='r', linestyle='--', alpha=0.5,
               label=f'Std at target ({std_at_target["utility"].item():.6f})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Utility')
    ax.set_title('Utility over steps')
    ax.legend(fontsize=8)

    # Fractional distance
    ax = axes[0, 1]
    if da_h_clean:
        ax.plot([h['step'] for h in da_h_clean], [h['frac_dist'] for h in da_h_clean], 'b-', label='DA')
    if std_h_clean:
        ax.plot([h['step'] for h in std_h_clean], [h['frac_dist'] for h in std_h_clean], 'r-', label='Std')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('frac_dist (1.0 = start)')
    ax.set_title('Fractional distance to target')
    ax.legend()

    # Kernel similarity
    ax = axes[1, 0]
    if da_h_clean:
        ax.plot([h['step'] for h in da_h_clean], [h['kernel_sim'] for h in da_h_clean], 'b-', label='DA')
    if std_h_clean:
        ax.plot([h['step'] for h in std_h_clean], [h['kernel_sim'] for h in std_h_clean], 'r-', label='Std')
    ax.axhline(metrics_at_target['kernel_sim'], color='gray', linestyle='--', alpha=0.5,
               label=f'k(target, target) = {metrics_at_target["kernel_sim"]:.1f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('k(x_opt, x_target)')
    ax.set_title('Kernel similarity to target')
    ax.legend(fontsize=8)

    # Gradient norm
    ax = axes[1, 1]
    if da_h_clean:
        ax.plot([h['step'] for h in da_h_clean], [h['grad_norm'] for h in da_h_clean], 'b-', label='DA')
    if std_h_clean:
        ax.plot([h['step'] for h in std_h_clean], [h['grad_norm'] for h in std_h_clean], 'r-', label='Std')
    ax.set_xlabel('Step')
    ax.set_ylabel('||grad||')
    ax.set_title('Gradient norm')
    ax.legend()

    fig.suptitle(f"Gradient Ascent Trajectories (LR={LR}, N_STEPS={N_STEPS})", fontsize=12)
    fig.tight_layout()
    fig2_path = save_dir / 'gradient_ascent_trajectories.png'
    fig.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig2_path}")

    # =========================================================================
    # Step 12: Save artifacts for diagnostic script
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 12: Saving artifacts")
    print("=" * 70)

    artifacts = {
        'x_target': x_target.cpu(),
        'x_start': x_start.cpu(),
        'x_final_da': x_final_da.cpu(),
        'x_final_std': x_final_std.cpu(),
        'da_history': da_ascent['history'],
        'std_history': std_ascent['history'],
        'da_diverged': da_ascent['diverged'],
        'std_diverged': std_ascent['diverged'],
        'config': {
            'mode': 'default_gpy', 'M': M, 'n_train': N_TRAIN,
            'seed': config['seed'], 'cell': config['cell'],
            'r_max': r_max, 'n_px_side': n_px_side,
            'LR': LR, 'N_STEPS': N_STEPS, 'NOISE_SCALE': NOISE_SCALE,
            'TARGET_INDEX': TARGET_INDEX,
        },
    }
    artifacts_path = save_dir / 'gradient_ascent_artifacts.pt'
    torch.save(artifacts, artifacts_path)
    print(f"  Saved: {artifacts_path}")


if __name__ == '__main__':
    main()
