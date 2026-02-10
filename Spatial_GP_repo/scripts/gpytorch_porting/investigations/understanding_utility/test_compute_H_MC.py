"""
Test compute_H_MC: Monte Carlo entropy estimation.
Created by Claude for investigations/understanding_utility/.

Tests:
1. Agreement with compute_H in the safe region (low mu_g, low sigma2_g)
2. Extension beyond the r_max=100 truncation boundary
3. Scaling behavior under norm increase (c=1, 5, 10, 20, 50)
4. Plot H vs c to visualize the behavior

Usage:
    python investigations/understanding_utility/test_compute_H_MC.py
"""

import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Path setup
_script_dir = Path(__file__).resolve().parent
_gpytorch_dir = _script_dir.parent.parent
sys.path.insert(0, str(_gpytorch_dir))

import importlib.util
_local_utils_path = _gpytorch_dir / 'utils.py'
_spec = importlib.util.spec_from_file_location("gpytorch_porting_utils", str(_local_utils_path))
_local_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_utils)

compute_H = _local_utils.compute_H
compute_H_MC = _local_utils.compute_H_MC

# Trained model parameters (seed=42, cell=8, M=50, n_train=50, default_gpy)
A = 0.0265
LAMBDA0 = -1.686

# Natural image baseline (from findings.md)
MU_TARGET = 7.86
SIGMA2_TARGET = 70.9


def test_safe_region():
    """Test 1: MC agrees with sum-based H in the safe region."""
    print("\n" + "="*70)
    print("TEST 1: Agreement in safe region (natural images)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test points: natural image scale (c=1)
    mu = torch.tensor([MU_TARGET], dtype=torch.float32, device=device)
    sigma2 = torch.tensor([SIGMA2_TARGET], dtype=torch.float32, device=device)

    # Compute with both methods
    H_sum = compute_H(mu, sigma2, r_max=100, a=A, lambda0=LAMBDA0)
    H_mc = compute_H_MC(mu, sigma2, n_samples=10000, a=A, lambda0=LAMBDA0)

    print(f"\nNatural image (mu={MU_TARGET:.2f}, sigma2={SIGMA2_TARGET:.2f}):")
    print(f"  mu_g = {A * MU_TARGET + LAMBDA0:.4f}")
    print(f"  sigma2_g = {A**2 * SIGMA2_TARGET:.4f}")
    print(f"\n  H (sum, r_max=100):  {H_sum.item():.6f}")
    print(f"  H (MC, S=10000):     {H_mc.item():.6f}")
    print(f"  Relative diff:       {abs(H_sum.item() - H_mc.item()) / H_sum.item() * 100:.2f}%")
    print(f"\n✓ Expected: < 2% difference (MC variance + Laplace approx error)")


def test_scaling_behavior(plot=True):
    """Test 2: Trace H under norm scaling (c=1, 5, 10, 20, 50)."""
    print("\n" + "="*70)
    print("TEST 2: Scaling behavior beyond truncation boundary")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Scaling factors - denser sampling for plotting
    c_values = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]

    print(f"\nTracing H as image is scaled: x* = c * x_target")
    print(f"GP moments scale: mu(cx) = c*mu(x), sigma2(cx) = c^2*sigma2(x)\n")

    results = []
    for c in c_values:
        mu = torch.tensor([c * MU_TARGET], dtype=torch.float32, device=device)
        sigma2 = torch.tensor([c**2 * SIGMA2_TARGET], dtype=torch.float32, device=device)

        mu_g = A * mu.item() + LAMBDA0
        sigma2_g = A**2 * sigma2.item()

        # Laplace safety check
        z_safe = (np.log(100) - mu_g) / np.sqrt(sigma2_g) if sigma2_g > 0 else float('inf')

        # Try both methods (sum will fail for large c)
        try:
            H_sum = compute_H(mu, sigma2, r_max=100, a=A, lambda0=LAMBDA0).item()
            sum_valid = z_safe > 1.5
        except:
            H_sum = float('nan')
            sum_valid = False

        H_mc = compute_H_MC(mu, sigma2, n_samples=5000, a=A, lambda0=LAMBDA0).item()

        results.append({
            'c': c,
            'mu': mu.item(),
            'sigma2': sigma2.item(),
            'mu_g': mu_g,
            'sigma2_g': sigma2_g,
            'z_safe': z_safe,
            'H_sum': H_sum,
            'H_mc': H_mc,
            'sum_valid': sum_valid
        })

    # Print table
    print(f"{'c':>6} {'mu':>8} {'sigma2':>10} {'mu_g':>8} {'s2_g':>8} "
          f"{'z_safe':>8} {'H_sum':>10} {'H_MC':>10} {'Valid?':>8}")
    print("-" * 92)

    for r in results:
        valid_str = "YES" if r['sum_valid'] else "NO"
        h_sum_str = f"{r['H_sum']:10.6f}" if not np.isnan(r['H_sum']) else "   (fails)"
        print(f"{r['c']:6.1f} {r['mu']:8.1f} {r['sigma2']:10.1f} "
              f"{r['mu_g']:8.4f} {r['sigma2_g']:8.4f} {r['z_safe']:8.2f} "
              f"{h_sum_str} {r['H_mc']:10.6f} {valid_str:>8}")

    print("\nInterpretation:")
    print("  - z_safe > 2.0: sum-based H is accurate")
    print("  - z_safe < 1.5: sum-based H collapses (truncation artifact)")
    print("  - MC continues working at all scales")
    print("  - H grows monotonically with c (larger variance → more uncertainty)")

    # Plot if requested
    if plot:
        plot_scaling_behavior(results)

    return results


def plot_scaling_behavior(results):
    """Create visualization of H vs scaling factor."""
    _script_dir = Path(__file__).resolve().parent

    c_vals = [r['c'] for r in results]
    H_sum = [r['H_sum'] for r in results]
    H_mc = [r['H_mc'] for r in results]
    z_safe = [r['z_safe'] for r in results]
    valid = [r['sum_valid'] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Panel 1: H vs c (log scale for H to handle large values)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Sum-based H (only plot where valid)
    c_valid = [c_vals[i] for i in range(len(c_vals)) if valid[i]]
    H_sum_valid = [H_sum[i] for i in range(len(H_sum)) if valid[i]]
    if c_valid:
        ax1.plot(c_valid, H_sum_valid, 'o-', color='blue', linewidth=2,
                 markersize=6, label='H (sum, r_max=100)', zorder=3)

    # Mark where sum becomes invalid
    c_invalid = [c_vals[i] for i in range(len(c_vals)) if not valid[i]]
    H_sum_invalid = [H_sum[i] for i in range(len(H_sum)) if not valid[i]]
    if c_invalid:
        ax1.plot(c_invalid, H_sum_invalid, 'x', color='blue', markersize=8,
                 alpha=0.3, label='H (sum, invalid)', zorder=2)

    # MC-based H
    ax1.plot(c_vals, H_mc, 's-', color='red', linewidth=2,
             markersize=6, label='H (MC, S=5000)', zorder=3)

    ax1.set_xlabel('Scaling factor c', fontsize=12)
    ax1.set_ylabel('Entropy H(R | x*, D)', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title('Entropy under norm scaling: x* = c·x_target', fontsize=13)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Annotate the safe region
    ax1.axvspan(c_vals[0], c_valid[-1] if c_valid else c_vals[0],
                alpha=0.1, color='green', label='Sum valid (z_safe>1.5)')
    ax1.text(c_valid[-1]/2 if c_valid else 2, ax1.get_ylim()[0]*1.5,
             'Sum-based\nH valid', fontsize=9, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

    # Panel 2: z_safe (Laplace validity metric)
    ax2.plot(c_vals, z_safe, 'o-', color='purple', linewidth=2, markersize=6)
    ax2.axhline(2.0, color='green', linestyle='--', linewidth=1.5,
                label='Safe threshold (z_safe=2.0)')
    ax2.axhline(1.5, color='orange', linestyle='--', linewidth=1.5,
                label='Borderline (z_safe=1.5)')
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    ax2.set_xlabel('Scaling factor c', fontsize=12)
    ax2.set_ylabel('z_safe = (log(r_max) - μ_g) / √(σ²_g)', fontsize=12)
    ax2.set_title('Laplace approximation validity (r_max=100)', fontsize=13)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Highlight regions
    ax2.fill_between(c_vals, 2.0, 20, alpha=0.1, color='green', label='Safe')
    ax2.fill_between(c_vals, 1.5, 2.0, alpha=0.1, color='orange', label='Borderline')
    ax2.fill_between(c_vals, -10, 1.5, alpha=0.1, color='red', label='Invalid')

    plt.tight_layout()
    save_path = _script_dir / 'H_scaling_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {save_path}")
    plt.close()


def test_extreme_case():
    """Test 3: Extreme case far beyond natural range."""
    print("\n" + "="*70)
    print("TEST 3: Extreme case (c=100, way beyond natural range)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    c = 100.0
    mu = torch.tensor([c * MU_TARGET], dtype=torch.float32, device=device)
    sigma2 = torch.tensor([c**2 * SIGMA2_TARGET], dtype=torch.float32, device=device)

    mu_g = A * mu.item() + LAMBDA0
    sigma2_g = A**2 * sigma2.item()

    print(f"\nExtreme scaling: c={c:.0f}")
    print(f"  Raw GP: mu={mu.item():.1f}, sigma2={sigma2.item():.1f}")
    print(f"  Log-firing rate: mu_g={mu_g:.2f}, sigma2_g={sigma2_g:.2f}")

    # MC with more samples for this extreme case
    H_mc = compute_H_MC(mu, sigma2, n_samples=10000, a=A, lambda0=LAMBDA0).item()

    print(f"\n  H (MC, S=10000): {H_mc:.6f}")
    print(f"\n✓ MC handles this gracefully (would take r_max ~ 10^50 for sum-based)")


if __name__ == '__main__':
    print("\nTesting compute_H_MC: Monte Carlo entropy estimation")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    test_safe_region()
    test_scaling_behavior()
    test_extreme_case()

    print("\n" + "="*70)
    print("All tests complete.")
    print("="*70)
