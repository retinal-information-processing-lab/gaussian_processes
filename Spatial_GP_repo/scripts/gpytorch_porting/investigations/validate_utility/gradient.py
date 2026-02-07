"""
Gradient investigation for variational GP utility functions.
Created by Claude.

Trains a GP model and evaluates utility functions with gradient flow enabled,
computing dU/dx* for each candidate image. This verifies that:

1. Gradients flow correctly from utility values back to input pixels
2. Gradient norms are non-trivial (not all zero, not all same)
3. Gradient structure reflects the receptive field mask (non-zero only within RF)
4. Standard and distribution-aware utilities produce consistent gradient patterns

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


def check_rf_mask_structure(grad, model, label=""):
    """Check that gradient sparsity matches the kernel's RF mask.

    Args:
        grad: (N_candidates, n_pixels) gradient tensor.
        model: GP model with kernel that may have an RF mask (alpha).
        label: String label for print output.
    """
    kernel = model.covar_module
    if hasattr(kernel, 'alpha') and kernel.alpha is not None:
        mask = kernel.alpha  # (n_pixels,) — 0 outside RF, >0 inside
        mask_binary = (mask > 0).float()
        n_masked = int(mask_binary.sum().item())
        n_pixels = mask_binary.shape[0]

        # Check: are non-zero gradients confined to masked pixels?
        grad_nonzero = (grad.abs() > 1e-10).float()  # (N_candidates, n_pixels)
        outside_rf = grad_nonzero * (1 - mask_binary.unsqueeze(0))
        n_outside = int(outside_rf.sum().item())
        n_inside = int((grad_nonzero * mask_binary.unsqueeze(0)).sum().item())

        print(f"  [{label}] RF mask: {n_masked}/{n_pixels} pixels active")
        print(f"  [{label}] Non-zero grads inside RF: {n_inside}")
        print(f"  [{label}] Non-zero grads outside RF: {n_outside} (should be 0)")
    else:
        print(f"  [{label}] No RF mask found on kernel (use_mask=False?)")


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


if __name__ == '__main__':
    main()
