"""
Entropy landscape check: overlay candidate positions on entropy heatmap.
Created by Claude.

Computes (g_mean, g_var) for each candidate and plots them on top of the
H(R | g_mean, g_var) entropy landscape. This shows whether candidates fall
in the entropy dead zone (H ~ 0) where utility is trivially zero.

Usage:
    python investigations/validate_utility/entropy_landscape_check.py
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_gpytorch_dir = _script_dir.parent.parent
sys.path.insert(0, str(_gpytorch_dir))

from run_single_mode import run_single_config, build_config_from_defaults

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
N_GRID = 300  # entropy grid resolution


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

    torch.manual_seed(999)
    X_black = torch.full((1, n_pixels), pixel_min, dtype=dtype, device=device)
    X_white = torch.full((1, n_pixels), pixel_max, dtype=dtype, device=device)
    X_gray = torch.full((1, n_pixels), 0.0, dtype=dtype, device=device)
    X_noise = torch.randn(1, n_pixels, dtype=dtype, device=device)

    X_candidates = torch.cat([X_natural, X_black, X_white, X_gray, X_noise], dim=0)

    # =========================================================================
    # 3. Compute (g_mean, g_var) for all candidates
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Computing GP posterior → log-firing rate moments")
    print("=" * 70)

    with torch.no_grad():
        posterior = model(X_candidates)
        lambda_mean = posterior.mean
        lambda_var = posterior.variance

    g_mean = (A * lambda_mean + lambda0).cpu()
    g_var = (A ** 2 * lambda_var).cpu()
    firing_rate = torch.exp(g_mean)

    # Compute H at exact candidate points
    with torch.no_grad():
        H_at_candidates = compute_H(
            lambda_mean, lambda_var, r_max=r_max, a=A, lambda0=lambda0
        ).cpu()

    # =========================================================================
    # 4. Print table
    # =========================================================================
    print(f"\n--- Candidate positions on entropy landscape ---")
    print(f"{'Type':<20} {'g_mean':>8} {'g_var':>10} {'exp(g_m)':>9} {'H':>8}")
    print("-" * 58)

    nat = slice(0, n_natural)
    print(f"{'Natural (mean)':<20} {g_mean[nat].mean().item():>8.4f} "
          f"{g_var[nat].mean().item():>10.6f} "
          f"{firing_rate[nat].mean().item():>9.4f} "
          f"{H_at_candidates[nat].mean().item():>8.4f}")
    print(f"{'Natural (std)':<20} {g_mean[nat].std().item():>8.4f} "
          f"{g_var[nat].std().item():>10.6f} "
          f"{firing_rate[nat].std().item():>9.4f} "
          f"{H_at_candidates[nat].std().item():>8.4f}")
    print(f"{'Natural (min)':<20} {g_mean[nat].min().item():>8.4f} "
          f"{g_var[nat].min().item():>10.6f} "
          f"{firing_rate[nat].min().item():>9.4f} "
          f"{H_at_candidates[nat].min().item():>8.4f}")
    print(f"{'Natural (max)':<20} {g_mean[nat].max().item():>8.4f} "
          f"{g_var[nat].max().item():>10.6f} "
          f"{firing_rate[nat].max().item():>9.4f} "
          f"{H_at_candidates[nat].max().item():>8.4f}")

    for name, idx in [("FULL_BLACK", n_natural), ("FULL_WHITE", n_natural + 1),
                      ("UNIFORM_GRAY", n_natural + 2), ("RANDOM_NOISE", n_natural + 3)]:
        print(f"{name:<20} {g_mean[idx].item():>8.4f} "
              f"{g_var[idx].item():>10.6f} "
              f"{firing_rate[idx].item():>9.4f} "
              f"{H_at_candidates[idx].item():>8.4f}")

    # =========================================================================
    # 5. Compute entropy grid
    # =========================================================================
    print(f"\n" + "=" * 70)
    print("Step 4: Computing entropy grid for heatmap")
    print("=" * 70)

    # Determine range from data with margin
    all_g = g_mean
    all_gv = g_var
    mu_min = min(all_g.min().item() - 1.0, -5.0)
    mu_max = max(all_g.max().item() + 1.0, 5.0)
    sigma2_min = 0.0
    sigma2_max = max(all_gv.max().item() * 2.0, 0.5)

    print(f"Grid range: mu=[{mu_min:.2f}, {mu_max:.2f}], sigma2=[{sigma2_min:.4f}, {sigma2_max:.4f}]")

    mu_range = torch.linspace(mu_min, mu_max, N_GRID, dtype=torch.float32, device=device)
    sigma2_range = torch.linspace(sigma2_min, sigma2_max, N_GRID, dtype=torch.float32, device=device)

    mu_grid, sigma2_grid = torch.meshgrid(mu_range, sigma2_range, indexing='ij')
    mu_flat = mu_grid.flatten()
    sigma2_flat = sigma2_grid.flatten()

    t0 = time.time()
    with torch.no_grad():
        # Grid is already in log-firing rate space, so a=1, lambda0=0
        H_flat = compute_H(mu_flat, sigma2_flat, r_max=r_max, a=1.0, lambda0=0.0)
    elapsed = time.time() - t0
    print(f"Entropy grid ({N_GRID}x{N_GRID} = {N_GRID**2} points) computed in {elapsed:.1f}s")

    H_grid = H_flat.reshape(N_GRID, N_GRID).cpu().numpy()
    mu_np = mu_range.cpu().numpy()
    sigma2_np = sigma2_range.cpu().numpy()

    # =========================================================================
    # 6. Plot
    # =========================================================================
    print(f"\n" + "=" * 70)
    print("Step 5: Generating plot")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Heatmap background
    im = ax.imshow(H_grid, origin='lower', aspect='auto',
                   extent=[sigma2_np.min(), sigma2_np.max(), mu_np.min(), mu_np.max()],
                   cmap='viridis', interpolation='bilinear')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Entropy H(R | g_mean, g_var)', fontsize=11)

    # Natural images scatter
    ax.scatter(g_var[:n_natural].numpy(), g_mean[:n_natural].numpy(),
               c='white', marker='o', s=40, alpha=0.7, edgecolors='gray',
               linewidths=0.5, label=f'Natural ({n_natural})', zorder=4)

    # Synthetic markers
    synth_styles = [
        ("FULL_BLACK", n_natural, 'red', 's'),
        ("FULL_WHITE", n_natural + 1, 'blue', 'D'),
        ("UNIFORM_GRAY", n_natural + 2, 'lime', '^'),
        ("RANDOM_NOISE", n_natural + 3, 'orange', 'v'),
    ]
    for name, idx, color, marker in synth_styles:
        ax.scatter(g_var[idx].item(), g_mean[idx].item(),
                   c=color, marker=marker, s=150, edgecolors='black',
                   linewidths=1.5, label=name, zorder=5)

    ax.set_xlabel('g_var = A^2 * lambda_var', fontsize=12)
    ax.set_ylabel('g_mean = A * lambda_mean + lambda0', fontsize=12)
    ax.set_title(f'Entropy landscape with candidate positions\n'
                 f'(A={A.item():.4f}, lambda0={lambda0.item():.4f}, '
                 f'M=50, n_train=200, test_r={result["test_r"]:.3f})',
                 fontsize=13)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    save_path = _script_dir / 'entropy_landscape_check.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # =========================================================================
    # 7. Diagnosis summary
    # =========================================================================
    print(f"\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    print(f"Entropy non-zero band (from heatmap): approx g_mean in [-3, +5]")
    print(f"Natural images g_mean range: [{g_mean[:n_natural].min().item():.2f}, "
          f"{g_mean[:n_natural].max().item():.2f}]")
    print(f"FULL_BLACK g_mean: {g_mean[n_natural].item():.2f} "
          f"({'IN band' if -3 < g_mean[n_natural].item() < 5 else 'OUT of band'})")
    print(f"FULL_WHITE g_mean: {g_mean[n_natural + 1].item():.2f} "
          f"({'IN band' if -3 < g_mean[n_natural + 1].item() < 5 else 'OUT of band'})")
    print(f"UNIFORM_GRAY g_mean: {g_mean[n_natural + 2].item():.2f}")
    print(f"RANDOM_NOISE g_mean: {g_mean[n_natural + 3].item():.2f}")

    print(f"\nConclusion:")
    if g_mean[n_natural].item() < -2 or g_mean[n_natural + 1].item() < -2:
        print(f"  Extreme stimuli map to LOW g_mean (low predicted firing rate).")
        print(f"  Poisson concentrated at r=0 → low entropy → low utility.")
        print(f"  This is correct GP behavior, but makes extreme stimuli poor")
        print(f"  candidates for testing utility function separation.")
    else:
        print(f"  All candidates fall within the non-zero entropy band.")
        print(f"  The entropy landscape is not the primary issue.")


if __name__ == "__main__":
    main()
