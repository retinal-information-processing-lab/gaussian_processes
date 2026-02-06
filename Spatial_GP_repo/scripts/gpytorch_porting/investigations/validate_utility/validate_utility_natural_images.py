"""
Validate utility functions on natural images vs synthetic stimuli.

Tests the hypothesis that:
- Standard utility peaks for synthetic edge stimuli (high GP uncertainty)
- Distribution-aware utility peaks for natural images (weights by p(x))

Usage:
    python investigations/validate_utility/validate_utility_natural_images.py

Uses default_gpy mode. All other params (cell, M, n_train, seed) from default_params.json.
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
# Investigation parameters (visible, not buried in function body)
# ---------------------------------------------------------------------------
N_NATURAL_CANDIDATES = 50  # Number of natural images to include as candidates


def main():
    # =========================================================================
    # 1. Train model
    # =========================================================================
    print("=" * 70)
    print("Step 1: Training default_gpy model")
    print("=" * 70)

    # All parameters read from default_params.json via build_config_from_defaults.
    # Only 'mode' is specified here; seed, cell, M, n_train come from defaults.
    config = build_config_from_defaults(mode='default_gpy')

    result = run_single_config(config)

    model = result['_model']
    likelihood = result['_likelihood']
    indices_train = result['_indices_train']

    model.eval()
    n_mc = config['n_mc_samples']
    r_max = config['r_max']
    print(f"\nModel trained: test_r={result['test_r']:.4f}")
    print(f"A={likelihood.A.item():.4f}, lambda0={likelihood.lambda0.item():.4f}")
    print(f"Utility params from config: n_mc_samples={n_mc}, r_max={r_max}")

    # =========================================================================
    # 2. Build candidate sets
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Building candidate sets")
    print("=" * 70)

    # Load full dataset to get remaining images
    data_path = _gpytorch_dir.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = np.load(data_path)
    dtype = torch.float32
    device = next(model.parameters()).device

    X_all = torch.cat([
        torch.tensor(data['images_train'], dtype=dtype),
        torch.tensor(data['images_val'], dtype=dtype),
    ], dim=0)
    X_all = X_all.reshape(X_all.shape[0], -1).to(device)  # (3160, 11664)

    # Remaining pool (not used in training)
    all_indices = set(range(X_all.shape[0]))
    train_indices = set(indices_train.cpu().numpy().tolist())
    pool_indices = sorted(all_indices - train_indices)
    X_pool = X_all[pool_indices]  # view, no copy
    print(f"Pool size: {X_pool.shape[0]} images (total {X_all.shape[0]} - {len(train_indices)} train)")

    # Natural candidates
    n_natural = N_NATURAL_CANDIDATES
    X_natural = X_pool[:n_natural]

    # Synthetic stimuli — pixel bounds derived from actual data
    n_pixels = X_all.shape[1]  # 11664
    pixel_min = X_all.min().item()
    pixel_max = X_all.max().item()
    print(f"Dataset pixel range: [{pixel_min:.4f}, {pixel_max:.4f}]")

    torch.manual_seed(999)  # fixed seed for noise
    X_black = torch.full((1, n_pixels), pixel_min, dtype=dtype, device=device)
    X_white = torch.full((1, n_pixels), pixel_max, dtype=dtype, device=device)
    X_gray = torch.full((1, n_pixels), 0.0, dtype=dtype, device=device)
    X_noise = torch.randn(1, n_pixels, dtype=dtype, device=device)

    # Combine all candidates
    X_candidates = torch.cat([X_natural, X_black, X_white, X_gray, X_noise], dim=0)
    n_candidates = X_candidates.shape[0]
    print(f"Candidates: {n_natural} natural + 4 synthetic = {n_candidates} total")

    # Labels for printing
    candidate_labels = [f"natural_{i}" for i in range(n_natural)]
    candidate_labels += ["FULL_BLACK", "FULL_WHITE", "UNIFORM_GRAY", "RANDOM_NOISE"]

    # x_samples for distribution-aware utility (draw from pool, excluding candidates)
    X_mc_pool = X_pool[n_natural:]  # exclude the natural candidates
    mc_indices = torch.randperm(X_mc_pool.shape[0])[:n_mc]
    x_samples = X_mc_pool[mc_indices]
    print(f"MC samples for dist-aware: {n_mc} images from remaining pool")

    # =========================================================================
    # 3. Evaluate standard utility
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Computing standard utility")
    print("=" * 70)

    t0 = time.time()
    with torch.no_grad():
        std_result = standard_utility(model, likelihood, X_candidates, r_max=r_max)
    std_time = time.time() - t0
    print(f"Standard utility computed in {std_time:.1f}s")

    # =========================================================================
    # 4. Evaluate distribution-aware utility
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Computing distribution-aware utility")
    print("=" * 70)

    torch.manual_seed(42)  # reproducible lambda sampling
    t0 = time.time()
    with torch.no_grad():
        da_result = distribution_aware_utility(
            model, likelihood, X_candidates, x_samples,
            r_max=r_max, sample_lambda=True,
        )
    da_time = time.time() - t0
    print(f"Distribution-aware utility computed in {da_time:.1f}s")

    # =========================================================================
    # 5. Print results
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    std_u = std_result['utility']
    da_u = da_result['utility']
    da_hmarg = da_result['H_marg']
    da_hcond = da_result['H_cond']

    # Summary by type
    natural_std = std_u[:n_natural]
    natural_da = da_u[:n_natural]

    print(f"\n--- Summary by candidate type ---")
    print(f"{'Type':<20} {'Std Utility':>12} {'DA Utility':>12} {'H_marg':>10} {'H_cond':>10}")
    print("-" * 66)
    print(f"{'Natural (mean)':<20} {natural_std.mean().item():>12.6f} {natural_da.mean().item():>12.6f} "
          f"{da_hmarg[:n_natural].mean().item():>10.4f} {da_hcond[:n_natural].mean().item():>10.4f}")
    print(f"{'Natural (std)':<20} {natural_std.std().item():>12.6f} {natural_da.std().item():>12.6f}")
    print(f"{'Natural (min)':<20} {natural_std.min().item():>12.6f} {natural_da.min().item():>12.6f}")
    print(f"{'Natural (max)':<20} {natural_std.max().item():>12.6f} {natural_da.max().item():>12.6f}")

    for name_idx, name in [("FULL_BLACK", n_natural), ("FULL_WHITE", n_natural+1),
                            ("UNIFORM_GRAY", n_natural+2), ("RANDOM_NOISE", n_natural+3)]:
        i = name
        print(f"{name_idx:<20} {std_u[i].item():>12.6f} {da_u[i].item():>12.6f} "
              f"{da_hmarg[i].item():>10.4f} {da_hcond[i].item():>10.4f}")

    # Top-5 by each utility
    print(f"\n--- Top-5 by STANDARD utility ---")
    top5_std = torch.argsort(std_u, descending=True)[:5]
    for rank, idx in enumerate(top5_std):
        print(f"  {rank+1}. {candidate_labels[idx]:<20} std_u={std_u[idx].item():.6f}")

    print(f"\n--- Top-5 by DISTRIBUTION-AWARE utility ---")
    top5_da = torch.argsort(da_u, descending=True)[:5]
    for rank, idx in enumerate(top5_da):
        print(f"  {rank+1}. {candidate_labels[idx]:<20} da_u={da_u[idx].item():.6f}")

    # Bottom-5 by dist-aware (should include synthetics?)
    print(f"\n--- Bottom-5 by DISTRIBUTION-AWARE utility ---")
    bot5_da = torch.argsort(da_u, descending=False)[:5]
    for rank, idx in enumerate(bot5_da):
        print(f"  {rank+1}. {candidate_labels[idx]:<20} da_u={da_u[idx].item():.6f}")

    # Quick sanity check
    print(f"\n--- Sanity checks ---")
    print(f"Standard utility: any NaN={torch.any(torch.isnan(std_u)).item()}, "
          f"any Inf={torch.any(torch.isinf(std_u)).item()}")
    print(f"DA utility: any NaN={torch.any(torch.isnan(da_u)).item()}, "
          f"any Inf={torch.any(torch.isinf(da_u)).item()}")
    print(f"DA utility min={da_u.min().item():.6f} (should be ~>=0)")


if __name__ == "__main__":
    main()
