"""
Validate utility with better synthetic stimuli.
Created by Claude.

Previous validation used pathological synthetics (full-black, full-white) which
map to very negative g_mean → low firing rate → low entropy → uninformative
comparison. This script uses structurally-degraded natural images that stay in
the same entropy regime as real natural images.

Synthetic types:
- Pixel-shuffled: same pixel distribution, destroyed spatial structure
- Phase-scrambled: same power spectrum, destroyed phase (edges/objects)
- Matched noise: per-pixel N(mean, std) from training set, no spatial correlation

Usage:
    python investigations/validate_utility/validate_utility_better_synthetics.py
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

# Import compute_H from local utils.py via importlib (avoids sys.modules shadowing)
import importlib.util
_local_utils_path = _gpytorch_dir / 'utils.py'
_spec = importlib.util.spec_from_file_location("gpytorch_porting_utils", str(_local_utils_path))
_local_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_utils)

compute_H = _local_utils.compute_H

# ---------------------------------------------------------------------------
# Investigation parameters
# ---------------------------------------------------------------------------
N_NATURAL_CANDIDATES = 50
N_SYNTHETIC_PER_TYPE = 10
N_PX_SIDE = 108


def make_pixel_shuffled(X_source, seed=42):
    """Shuffle pixels of each image independently. Destroys spatial structure."""
    gen = torch.Generator(device=X_source.device).manual_seed(seed)
    X_shuffled = X_source.clone()
    for i in range(X_shuffled.shape[0]):
        perm = torch.randperm(X_shuffled.shape[1], generator=gen, device=X_source.device)
        X_shuffled[i] = X_shuffled[i][perm]
    return X_shuffled


def make_phase_scrambled(X_source, seed=43):
    """Randomize phase, keep magnitude. Preserves power spectrum."""
    gen = torch.Generator(device=X_source.device).manual_seed(seed)
    n = X_source.shape[0]
    X_scrambled = torch.empty_like(X_source)
    for i in range(n):
        img = X_source[i].reshape(N_PX_SIDE, N_PX_SIDE)
        F = torch.fft.rfft2(img)
        # Random phase
        phase = torch.rand(F.shape, generator=gen, dtype=X_source.dtype,
                           device=X_source.device) * 2 * torch.pi
        F_scrambled = F.abs() * torch.exp(1j * phase)
        img_out = torch.fft.irfft2(F_scrambled, s=(N_PX_SIDE, N_PX_SIDE))
        X_scrambled[i] = img_out.flatten()
    return X_scrambled


def make_matched_noise(X_train, n_images, seed=44):
    """Per-pixel N(mean, std) from training set. Same marginals, no structure."""
    gen = torch.Generator(device=X_train.device).manual_seed(seed)
    pixel_mean = X_train.mean(dim=0)
    pixel_std = X_train.std(dim=0)
    noise = torch.randn(n_images, X_train.shape[1], generator=gen,
                        dtype=X_train.dtype, device=X_train.device)
    return pixel_mean.unsqueeze(0) + pixel_std.unsqueeze(0) * noise


def main():
    # =========================================================================
    # 1. Train model
    # =========================================================================
    print("=" * 70)
    print("Step 1: Training default_gpy model")
    print("=" * 70)

    config = build_config_from_defaults(M=50, n_train=200, n_mc_samples=1000)
    result = run_single_config(config)

    model = result['_model']
    likelihood = result['_likelihood']
    indices_train = result['_indices_train']
    n_mc = config['n_mc_samples']
    r_max = config['r_max']

    model.eval()
    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()
    print(f"\nModel trained: test_r={result['test_r']:.4f}")
    print(f"A={A.item():.4f}, lambda0={lambda0.item():.4f}")
    print(f"Utility params: n_mc_samples={n_mc}, r_max={r_max}")

    # =========================================================================
    # 2. Build candidate sets
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Building candidate sets")
    print("=" * 70)

    data_path = _gpytorch_dir.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = np.load(data_path)
    dtype = torch.float32
    device = next(model.parameters()).device

    X_all = torch.cat([
        torch.tensor(data['images_train'], dtype=dtype),
        torch.tensor(data['images_val'], dtype=dtype),
    ], dim=0)
    X_all = X_all.reshape(X_all.shape[0], -1).to(device)

    all_indices = set(range(X_all.shape[0]))
    train_indices = set(indices_train.cpu().numpy().tolist())
    pool_indices = sorted(all_indices - train_indices)
    X_pool = X_all[pool_indices]
    X_train_set = X_all[list(train_indices)]

    n_nat = N_NATURAL_CANDIDATES
    n_syn = N_SYNTHETIC_PER_TYPE

    # Natural candidates
    X_natural = X_pool[:n_nat]

    # Source images for synthetic generation (from pool, after naturals)
    X_source = X_pool[n_nat:n_nat + n_syn]

    # Better synthetics
    X_shuffled = make_pixel_shuffled(X_source, seed=42)
    X_phase = make_phase_scrambled(X_source, seed=43)
    X_matched = make_matched_noise(X_train_set, n_syn, seed=44)

    print(f"Candidates: {n_nat} natural + {n_syn} shuffled + {n_syn} phase-scrambled "
          f"+ {n_syn} matched-noise = {n_nat + 3*n_syn} total")

    # Combine all candidates
    X_candidates = torch.cat([X_natural, X_shuffled, X_phase, X_matched], dim=0)
    n_candidates = X_candidates.shape[0]

    # Labels
    labels = [f"natural_{i}" for i in range(n_nat)]
    labels += [f"shuffled_{i}" for i in range(n_syn)]
    labels += [f"phase_{i}" for i in range(n_syn)]
    labels += [f"matched_{i}" for i in range(n_syn)]

    # Type slices
    slices = {
        'Natural': slice(0, n_nat),
        'Pixel-shuffled': slice(n_nat, n_nat + n_syn),
        'Phase-scrambled': slice(n_nat + n_syn, n_nat + 2*n_syn),
        'Matched-noise': slice(n_nat + 2*n_syn, n_nat + 3*n_syn),
    }

    # =========================================================================
    # 3. Kernel norm diagnostic
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Kernel norm diagnostic (v_x)")
    print("=" * 70)

    with torch.no_grad():
        v_x = model.covar_module(X_candidates, X_candidates, diag=True)
        posterior = model(X_candidates)
        g_mean = A * posterior.mean + lambda0
        g_var = A ** 2 * posterior.variance

    for name, sl in slices.items():
        print(f"  {name:<20} v_x: mean={v_x[sl].mean().item():.2f}, "
              f"std={v_x[sl].std().item():.2f}, "
              f"g_mean: mean={g_mean[sl].mean().item():.3f}, "
              f"g_var: mean={g_var[sl].mean().item():.6f}")

    # =========================================================================
    # 4. Standard utility
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Computing standard utility")
    print("=" * 70)

    t0 = time.time()
    with torch.no_grad():
        std_result = standard_utility(model, likelihood, X_candidates, r_max=r_max)
    std_time = time.time() - t0
    print(f"Standard utility computed in {std_time:.1f}s")

    # =========================================================================
    # 5. Distribution-aware utility
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Computing distribution-aware utility")
    print("=" * 70)

    # MC samples from pool (excluding candidates and source images)
    mc_start = n_nat + n_syn  # after natural candidates and source images
    X_mc_pool = X_pool[mc_start:]
    mc_indices = torch.randperm(X_mc_pool.shape[0])[:n_mc]
    x_samples = X_mc_pool[mc_indices]
    print(f"MC samples: {n_mc} images from remaining pool")

    torch.manual_seed(42)
    t0 = time.time()
    with torch.no_grad():
        da_result = distribution_aware_utility(
            model, likelihood, X_candidates, x_samples,
            r_max=r_max, sample_lambda=True,
        )
    da_time = time.time() - t0
    print(f"Distribution-aware utility computed in {da_time:.1f}s")

    # =========================================================================
    # 6. Print results
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    std_u = std_result['utility']
    da_u = da_result['utility']
    da_hmarg = da_result['H_marg']
    da_hcond = da_result['H_cond']

    print(f"\n--- Summary by candidate type ---")
    header = (f"{'Type':<20} {'Std Util':>10} {'DA Util':>10} {'H_marg':>8} "
              f"{'H_cond':>8} {'v_x':>8} {'g_mean':>8}")
    print(header)
    print("-" * len(header))

    for name, sl in slices.items():
        print(f"{name + ' (mean)':<20} "
              f"{std_u[sl].mean().item():>10.6f} "
              f"{da_u[sl].mean().item():>10.6f} "
              f"{da_hmarg[sl].mean().item():>8.4f} "
              f"{da_hcond[sl].mean().item():>8.4f} "
              f"{v_x[sl].mean().item():>8.2f} "
              f"{g_mean[sl].mean().item():>8.4f}")
        print(f"{name + ' (std)':<20} "
              f"{std_u[sl].std().item():>10.6f} "
              f"{da_u[sl].std().item():>10.6f} "
              f"{da_hmarg[sl].std().item():>8.4f} "
              f"{da_hcond[sl].std().item():>8.4f} "
              f"{v_x[sl].std().item():>8.2f} "
              f"{g_mean[sl].std().item():>8.4f}")

    # Top-5 by each utility
    print(f"\n--- Top-10 by STANDARD utility ---")
    top10_std = torch.argsort(std_u, descending=True)[:10]
    for rank, idx in enumerate(top10_std):
        tp = "NAT" if idx < n_nat else ("SHF" if idx < n_nat+n_syn else ("PHS" if idx < n_nat+2*n_syn else "MTC"))
        print(f"  {rank+1:>2}. {labels[idx]:<20} [{tp}] std_u={std_u[idx].item():.6f}")

    print(f"\n--- Top-10 by DISTRIBUTION-AWARE utility ---")
    top10_da = torch.argsort(da_u, descending=True)[:10]
    for rank, idx in enumerate(top10_da):
        tp = "NAT" if idx < n_nat else ("SHF" if idx < n_nat+n_syn else ("PHS" if idx < n_nat+2*n_syn else "MTC"))
        print(f"  {rank+1:>2}. {labels[idx]:<20} [{tp}] da_u={da_u[idx].item():.6f}")

    # Count types in top-20
    print(f"\n--- Type composition of top-20 ---")
    for util_name, util_tensor in [("Standard", std_u), ("Dist-Aware", da_u)]:
        top20 = torch.argsort(util_tensor, descending=True)[:20]
        counts = {'NAT': 0, 'SHF': 0, 'PHS': 0, 'MTC': 0}
        for idx in top20:
            idx = idx.item()
            if idx < n_nat:
                counts['NAT'] += 1
            elif idx < n_nat + n_syn:
                counts['SHF'] += 1
            elif idx < n_nat + 2*n_syn:
                counts['PHS'] += 1
            else:
                counts['MTC'] += 1
        print(f"  {util_name:<12}: NAT={counts['NAT']}, SHF={counts['SHF']}, "
              f"PHS={counts['PHS']}, MTC={counts['MTC']}")

    # Sanity checks
    print(f"\n--- Sanity checks ---")
    print(f"Standard utility: NaN={torch.any(torch.isnan(std_u)).item()}, "
          f"Inf={torch.any(torch.isinf(std_u)).item()}")
    print(f"DA utility: NaN={torch.any(torch.isnan(da_u)).item()}, "
          f"Inf={torch.any(torch.isinf(da_u)).item()}, "
          f"min={da_u.min().item():.6f}")


if __name__ == "__main__":
    main()
