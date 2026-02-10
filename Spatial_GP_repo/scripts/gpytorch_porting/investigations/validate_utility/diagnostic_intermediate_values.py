"""
Diagnostic: intermediate values in the utility pipeline.
Created by Claude.

Shows the full numerical chain from kernel self-covariance through GP posterior
to log-firing rate to entropy for each candidate type. Goal: understand WHY
FULL_BLACK dominates utility rankings despite being pathological.

Usage:
    python investigations/validate_utility/diagnostic_intermediate_values.py
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

# Import compute_H and nd_utility_new from local utils.py via importlib (avoids sys.modules shadowing)
import importlib.util
_local_utils_path = _gpytorch_dir / 'utils.py'
_spec = importlib.util.spec_from_file_location("gpytorch_porting_utils", str(_local_utils_path))
_local_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_utils)

compute_H = _local_utils.compute_H
nd_utility_new = _local_utils.nd_utility_new

# ---------------------------------------------------------------------------
# Investigation parameters
# ---------------------------------------------------------------------------
N_NATURAL_CANDIDATES = 50


def main():
    # =========================================================================
    # 1. Train model
    # =========================================================================
    print("=" * 70)
    print("Step 1: Training default_gpy model")
    print("=" * 70)

    config = build_config_from_defaults(M=50, n_train=200)
    result = run_single_config(config)

    model = result['_model']
    likelihood = result['_likelihood']
    indices_train = result['_indices_train']
    r_max = config['r_max']

    model.eval()
    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()
    print(f"\nModel trained: test_r={result['test_r']:.4f}")
    print(f"A={A.item():.4f}, lambda0={lambda0.item():.4f}")

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

    n_natural = N_NATURAL_CANDIDATES
    X_natural = X_pool[:n_natural]

    n_pixels = X_all.shape[1]
    pixel_min = X_all.min().item()
    pixel_max = X_all.max().item()
    print(f"Dataset pixel range: [{pixel_min:.4f}, {pixel_max:.4f}]")

    torch.manual_seed(999)
    X_black = torch.full((1, n_pixels), pixel_min, dtype=dtype, device=device)
    X_white = torch.full((1, n_pixels), pixel_max, dtype=dtype, device=device)
    X_gray = torch.full((1, n_pixels), 0.0, dtype=dtype, device=device)
    X_noise = torch.randn(1, n_pixels, dtype=dtype, device=device)

    X_candidates = torch.cat([X_natural, X_black, X_white, X_gray, X_noise], dim=0)
    n_candidates = X_candidates.shape[0]

    candidate_labels = [f"natural_{i}" for i in range(n_natural)]
    candidate_labels += ["FULL_BLACK", "FULL_WHITE", "UNIFORM_GRAY", "RANDOM_NOISE"]

    # =========================================================================
    # 3. Compute all intermediate values
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Computing intermediate values")
    print("=" * 70)

    t0 = time.time()
    with torch.no_grad():
        # Raw kernel self-covariance (prior, no variational correction)
        v_x = model.covar_module(X_candidates, X_candidates, diag=True)

        # GP posterior moments
        posterior = model(X_candidates)
        lambda_mean = posterior.mean
        lambda_var = posterior.variance

        # Log-firing rate transform
        g_mean = A * lambda_mean + lambda0
        g_var = A ** 2 * lambda_var

        # Predicted firing rate (approximate)
        firing_rate = torch.exp(g_mean)

        # Marginal entropy
        H_marg = compute_H(lambda_mean, lambda_var, r_max=r_max, a=A, lambda0=lambda0)

        # Standard utility
        std_utility = nd_utility_new(g_mean, g_var, r_max=r_max)

    elapsed = time.time() - t0
    print(f"All computations done in {elapsed:.1f}s")

    # =========================================================================
    # 4. Print diagnostic table
    # =========================================================================
    print("\n" + "=" * 90)
    print("DIAGNOSTIC TABLE: Intermediate Values at Each Pipeline Stage")
    print("=" * 90)
    print(f"Model params: A={A.item():.4f}, lambda0={lambda0.item():.4f}, r_max={r_max}")
    print(f"Transform: g_mean = {A.item():.4f} * lambda_mean + {lambda0.item():.4f}")
    print(f"           g_var  = {A.item():.4f}^2 * lambda_var = {(A**2).item():.6f} * lambda_var")

    header = (f"{'Type':<20} {'v_x':>8} {'lam_m':>8} {'lam_var':>8} "
              f"{'g_mean':>8} {'g_var':>8} {'fire_rt':>8} {'H_marg':>8} {'std_util':>8}")
    print(f"\n{header}")
    print("-" * len(header))

    # Natural image statistics
    nat = slice(0, n_natural)
    for stat_name, fn in [("Natural (mean)", torch.mean), ("Natural (std)", torch.std),
                          ("Natural (min)", torch.min), ("Natural (max)", torch.max)]:
        print(f"{stat_name:<20} "
              f"{fn(v_x[nat]).item():>8.2f} "
              f"{fn(lambda_mean[nat]).item():>8.3f} "
              f"{fn(lambda_var[nat]).item():>8.4f} "
              f"{fn(g_mean[nat]).item():>8.4f} "
              f"{fn(g_var[nat]).item():>8.6f} "
              f"{fn(firing_rate[nat]).item():>8.4f} "
              f"{fn(H_marg[nat]).item():>8.4f} "
              f"{fn(std_utility[nat]).item():>8.6f}")

    # Synthetic stimuli (individual values)
    for name, idx in [("FULL_BLACK", n_natural), ("FULL_WHITE", n_natural + 1),
                      ("UNIFORM_GRAY", n_natural + 2), ("RANDOM_NOISE", n_natural + 3)]:
        print(f"{name:<20} "
              f"{v_x[idx].item():>8.2f} "
              f"{lambda_mean[idx].item():>8.3f} "
              f"{lambda_var[idx].item():>8.4f} "
              f"{g_mean[idx].item():>8.4f} "
              f"{g_var[idx].item():>8.6f} "
              f"{firing_rate[idx].item():>8.4f} "
              f"{H_marg[idx].item():>8.4f} "
              f"{std_utility[idx].item():>8.6f}")

    # =========================================================================
    # 5. Kernel scale ratios
    # =========================================================================
    print(f"\n--- Kernel self-covariance ratios (v_x / natural mean) ---")
    v_nat_mean = v_x[:n_natural].mean().item()
    for name, idx in [("FULL_BLACK", n_natural), ("FULL_WHITE", n_natural + 1),
                      ("UNIFORM_GRAY", n_natural + 2), ("RANDOM_NOISE", n_natural + 3)]:
        ratio = v_x[idx].item() / v_nat_mean
        print(f"  {name:<20} v_x={v_x[idx].item():.2f}, ratio={ratio:.2f}x")
    print(f"  Natural mean v_x = {v_nat_mean:.2f}")

    # =========================================================================
    # 6. Per-pixel norm analysis (to understand v_x differences)
    # =========================================================================
    print(f"\n--- Raw pixel norm analysis (full image, before masking) ---")
    for name, X in [("Natural (mean)", X_natural), ("FULL_BLACK", X_black),
                    ("FULL_WHITE", X_white), ("UNIFORM_GRAY", X_gray),
                    ("RANDOM_NOISE", X_noise)]:
        norms = (X ** 2).sum(dim=1)
        print(f"  {name:<20} ||x||^2 = {norms.mean().item():.1f}")

    # Also show norms in the masked region only
    mask = model.covar_module._cached_mask
    if mask is not None:
        n_masked = mask.sum().item()
        print(f"\n--- Masked pixel norm ({int(n_masked)} / {n_pixels} pixels in RF) ---")
        for name, X in [("Natural (mean)", X_natural), ("FULL_BLACK", X_black),
                        ("FULL_WHITE", X_white), ("UNIFORM_GRAY", X_gray),
                        ("RANDOM_NOISE", X_noise)]:
            masked_norms = (X[:, mask] ** 2).sum(dim=1)
            print(f"  {name:<20} ||x_masked||^2 = {masked_norms.mean().item():.1f}")


if __name__ == "__main__":
    main()
