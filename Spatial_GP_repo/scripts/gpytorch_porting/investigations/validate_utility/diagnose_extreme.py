"""
Diagnose utility values at extreme (gradient-ascent-optimized) images.
Created by Claude.

Loads artifacts from gradient.py (optimized images), retrains the same model
(deterministic, same seed), and dissects the utility computation step by step
to identify whether high utility at extreme images is:
  (a) numerically correct but a kernel extrapolation artifact
  (b) numerically wrong (overflow, truncation, Laplace failure)
  (c) both

Checks:
1. GP posterior moments (marginal + conditional) — are they in the useful band?
2. Kernel values — self-kernel, cross-kernel, angular distance
3. Laplace approximation validity — does sum(p_r) ≈ 1?
4. Entropy decomposition — H_marg, H_cond, utility
5. Variance reduction ratio — does conditioning actually help?

Usage:
    python investigations/validate_utility/diagnose_extreme.py

Requires: gradient_ascent_artifacts.pt (produced by gradient.py)
"""

import sys
import math
import torch
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_gpytorch_dir = _script_dir.parent.parent
sys.path.insert(0, str(_gpytorch_dir))

import importlib.util
_local_utils_path = _gpytorch_dir / 'utils.py'
_spec = importlib.util.spec_from_file_location("gpytorch_porting_utils", str(_local_utils_path))
_local_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_utils)

get_gp_marginal_moments = _local_utils.get_gp_marginal_moments
get_gp_conditional_moments = _local_utils.get_gp_conditional_moments
compute_H = _local_utils.compute_H
nd_utility_new = _local_utils.nd_utility_new
_diff_laplace_log_probs = _local_utils._diff_laplace_log_probs

from run_single_mode import run_single_config, build_config_from_defaults

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_NATURAL = 3    # Number of natural images to include in comparison
ARTIFACTS_PATH = _script_dir / 'gradient_ascent_artifacts.pt'


def main():
    # =========================================================================
    # Load artifacts
    # =========================================================================
    print("=" * 70)
    print("Loading artifacts")
    print("=" * 70)

    if not ARTIFACTS_PATH.exists():
        print(f"ERROR: {ARTIFACTS_PATH} not found. Run gradient.py first.")
        return

    artifacts = torch.load(ARTIFACTS_PATH, weights_only=False)
    cfg = artifacts['config']
    x_target = artifacts['x_target']
    x_start = artifacts['x_start']
    x_final_da = artifacts['x_final_da']
    x_final_std = artifacts['x_final_std']

    print(f"  Config: mode={cfg['mode']}, M={cfg['M']}, n_train={cfg['n_train']}, "
          f"seed={cfg['seed']}, cell={cfg['cell']}")
    print(f"  Gradient ascent: LR={cfg['LR']}, N_STEPS={cfg['N_STEPS']}, "
          f"NOISE_SCALE={cfg['NOISE_SCALE']}")
    print(f"  DA diverged: {artifacts['da_diverged']}")
    print(f"  Std diverged: {artifacts['std_diverged']}")

    # =========================================================================
    # Retrain model (same seed = identical model)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Retraining model (same seed for reproducibility)")
    print("=" * 70)

    config = build_config_from_defaults(
        mode=cfg['mode'], M=cfg['M'], n_train=cfg['n_train'],
    )
    result = run_single_config(config)

    if result is None or result.get('status') != 'success':
        print("ERROR: Training failed")
        return

    model = result['_model']
    likelihood = result['_likelihood']
    model.eval()

    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()
    r_max = config['r_max']
    device = next(model.parameters()).device
    dtype = torch.float32

    print(f"\n  A={A.item():.4f}, lambda0={lambda0.item():.4f}, r_max={r_max}")

    # Move artifacts to device
    x_target = x_target.to(device)
    x_start = x_start.to(device)
    x_final_da = x_final_da.to(device)
    x_final_std = x_final_std.to(device)

    # Get natural images for comparison
    data_path = _gpytorch_dir.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = np.load(data_path)
    X_all = torch.cat([
        torch.tensor(data['images_train'], dtype=dtype),
        torch.tensor(data['images_val'], dtype=dtype),
    ], dim=0).reshape(-1, config['n_px_side'] ** 2).to(device)

    indices_train = result['_indices_train']
    pool_indices = sorted(set(range(X_all.shape[0])) - set(indices_train.cpu().numpy().tolist()))
    # Pick natural images NOT used as target (target is pool index 0)
    natural_images = [X_all[pool_indices[i + 1]] for i in range(N_NATURAL)]

    # Build test set: [x_target, x_final_da, x_final_std, natural_1, natural_2, natural_3]
    test_images = [
        ("x_target", x_target),
        ("x_final_da", x_final_da),
        ("x_final_std", x_final_std),
    ] + [(f"natural_{i+1}", img) for i, img in enumerate(natural_images)]

    # Lambda at target (for conditioning) — use posterior mean
    with torch.no_grad():
        post_target = model(x_target.unsqueeze(0))
        lambda_at_target = post_target.mean[0]
    print(f"  lambda(x_target) = {lambda_at_target.item():.4f} (posterior mean)")

    # =========================================================================
    # Check 1: GP posterior moments — marginal AND conditional
    # =========================================================================
    print("\n" + "=" * 70)
    print("Check 1: GP posterior moments (marginal + conditional)")
    print("=" * 70)

    print(f"\n  Table A: Marginal moments")
    print(f"  {'Image':>14}  {'mu':>10}  {'sigma2':>12}  {'logf_mean':>10}  {'logf_var':>10}  "
          f"{'exp(lfm)':>10}  {'in_band':>8}  {'px_range':>20}")
    print("  " + "-" * 105)

    marginal_data = {}
    for name, x in test_images:
        with torch.no_grad():
            mu, sigma2 = get_gp_marginal_moments(model, x.unsqueeze(0))
            mu_val = mu.item()
            s2_val = sigma2.item()
            logf_mean = (A * mu + lambda0).item()
            logf_var = (A ** 2 * sigma2).item()
            exp_lfm = math.exp(logf_mean) if logf_mean < 80 else float('inf')
            in_band = -5 < logf_mean < 10
            px_min, px_max = x.min().item(), x.max().item()

        marginal_data[name] = {
            'mu': mu_val, 'sigma2': s2_val,
            'logf_mean': logf_mean, 'logf_var': logf_var,
        }
        band_str = "YES" if in_band else "NO"
        print(f"  {name:>14}  {mu_val:10.4f}  {s2_val:12.4f}  {logf_mean:10.4f}  {logf_var:10.4f}  "
              f"{exp_lfm:10.2f}  {band_str:>8}  [{px_min:.2f}, {px_max:.2f}]")

        # Flags
        if logf_mean + 0.5 * logf_var > 80:
            print(f"    FLAG: logf_mean + 0.5*logf_var = {logf_mean + 0.5*logf_var:.1f} > 80 (exp overflow risk)")
        if exp_lfm > r_max:
            print(f"    FLAG: exp(logf_mean) = {exp_lfm:.1f} > r_max={r_max} (truncation risk)")

    print(f"\n  Table B: Conditional moments (conditioned on x_target)")
    print(f"  {'Image':>14}  {'mu_cond':>10}  {'s2_cond':>12}  {'lfm_cond':>10}  {'lfv_cond':>10}  "
          f"{'var_ratio':>10}")
    print("  " + "-" * 75)

    conditional_data = {}
    for name, x in test_images:
        with torch.no_grad():
            mu_cond, s2_cond = get_gp_conditional_moments(
                model, x.unsqueeze(0), x_target, lambda_at_target
            )
            mu_c = mu_cond.item()
            s2_c = s2_cond.item()
            lfm_cond = (A * mu_cond + lambda0).item()
            lfv_cond = (A ** 2 * s2_cond).item()
            s2_marg = marginal_data[name]['sigma2']
            var_ratio = s2_c / s2_marg if s2_marg > 0 else 0.0

        conditional_data[name] = {
            'mu_cond': mu_c, 's2_cond': s2_c,
            'lfm_cond': lfm_cond, 'lfv_cond': lfv_cond,
            'var_ratio': var_ratio,
        }
        print(f"  {name:>14}  {mu_c:10.4f}  {s2_c:12.4f}  {lfm_cond:10.4f}  {lfv_cond:10.4f}  "
              f"{var_ratio:10.6f}")

    # =========================================================================
    # Check 2: Kernel values
    # =========================================================================
    print("\n" + "=" * 70)
    print("Check 2: Kernel values")
    print("=" * 70)

    print(f"\n  {'Image':>14}  {'k(x,x)':>12}  {'k(x,tgt)':>12}  {'angle(rad)':>12}")
    print("  " + "-" * 55)

    for name, x in test_images:
        with torch.no_grad():
            k_xx = model.covar_module(x.unsqueeze(0), x.unsqueeze(0)).evaluate().squeeze().item()
            k_xt = model.covar_module(x.unsqueeze(0), x_target.unsqueeze(0)).evaluate().squeeze().item()
            # Angular distance: cos(angle) = k(x,y) / sqrt(k(x,x)*k(y,y))
            k_tt = model.covar_module(x_target.unsqueeze(0), x_target.unsqueeze(0)).evaluate().squeeze().item()
            cos_angle = k_xt / (math.sqrt(k_xx * k_tt) + 1e-30)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp for arccos
            angle = math.acos(cos_angle)

        print(f"  {name:>14}  {k_xx:12.2f}  {k_xt:12.2f}  {angle:12.6f}")

    # =========================================================================
    # Check 3: Laplace approximation validity
    # =========================================================================
    print("\n" + "=" * 70)
    print("Check 3: Laplace approximation validity")
    print("=" * 70)

    r = torch.arange(0, r_max, dtype=dtype, device=device)
    print(f"\n  {'Image':>14}  {'sum(p_r)':>10}  {'max(p_r)':>10}  {'argmax':>8}  "
          f"{'any_nan':>8}  {'any_inf':>8}")
    print("  " + "-" * 65)

    for name, x in test_images:
        with torch.no_grad():
            mu, sigma2 = get_gp_marginal_moments(model, x.unsqueeze(0))
            logf_mean = A * mu + lambda0
            logf_var = A ** 2 * sigma2
            p_r, log_p_r = _diff_laplace_log_probs(logf_mean, logf_var, r)

            sum_pr = p_r.sum().item()
            max_pr = p_r.max().item()
            argmax_r = p_r.argmax().item()
            any_nan = bool(torch.isnan(p_r).any().item())
            any_inf = bool(torch.isinf(p_r).any().item())

        flag = ""
        if abs(sum_pr - 1.0) > 0.01:
            flag = " FLAG: sum != 1"
        if any_nan:
            flag += " FLAG: NaN"
        if any_inf:
            flag += " FLAG: inf"
        print(f"  {name:>14}  {sum_pr:10.6f}  {max_pr:10.6f}  {argmax_r:8d}  "
              f"{'YES' if any_nan else 'no':>8}  {'YES' if any_inf else 'no':>8}{flag}")

    # =========================================================================
    # Check 4: Entropy decomposition
    # =========================================================================
    print("\n" + "=" * 70)
    print("Check 4: Entropy decomposition")
    print("=" * 70)

    print(f"\n  {'Image':>14}  {'H_marg':>10}  {'H_cond':>10}  {'U_DA':>10}  "
          f"{'U_std':>10}  {'flags':>10}")
    print("  " + "-" * 70)

    for name, x in test_images:
        with torch.no_grad():
            mu, sigma2 = get_gp_marginal_moments(model, x.unsqueeze(0))
            H_marg = compute_H(mu, sigma2, r_max=r_max, a=A, lambda0=lambda0).item()

            mu_cond, s2_cond = get_gp_conditional_moments(
                model, x.unsqueeze(0), x_target, lambda_at_target
            )
            H_cond = compute_H(mu_cond, s2_cond, r_max=r_max, a=A, lambda0=lambda0).item()

            U_da = H_marg - H_cond

            logf_mean = A * mu + lambda0
            logf_var = A ** 2 * sigma2
            U_std = nd_utility_new(logf_mean, logf_var, r_max=r_max).item()

        flags = []
        if math.isnan(H_marg) or math.isnan(H_cond):
            flags.append("NaN")
        if math.isinf(H_marg) or math.isinf(H_cond):
            flags.append("inf")
        if H_marg < -0.001:
            flags.append("H<0")
        if abs(H_marg) < 0.001 and abs(U_da) > 0.001:
            flags.append("H~0,U>0")
        flag_str = ",".join(flags) if flags else "ok"

        print(f"  {name:>14}  {H_marg:10.6f}  {H_cond:10.6f}  {U_da:10.6f}  "
              f"{U_std:10.6f}  {flag_str:>10}")

    # =========================================================================
    # Check 5: Variance reduction ratio
    # =========================================================================
    print("\n" + "=" * 70)
    print("Check 5: Variance reduction ratio")
    print("=" * 70)

    print(f"\n  {'Image':>14}  {'sigma2':>12}  {'sigma2_cond':>12}  {'ratio':>10}  "
          f"{'interpretation':>20}")
    print("  " + "-" * 75)

    for name, x in test_images:
        s2 = marginal_data[name]['sigma2']
        s2_c = conditional_data[name]['s2_cond']
        ratio = conditional_data[name]['var_ratio']

        if ratio < 0.1:
            interp = "STRONG conditioning"
        elif ratio < 0.5:
            interp = "moderate conditioning"
        elif ratio < 0.9:
            interp = "weak conditioning"
        else:
            interp = "NO conditioning"

        print(f"  {name:>14}  {s2:12.4f}  {s2_c:12.4f}  {ratio:10.6f}  {interp:>20}")

    print("\n" + "=" * 70)
    print("Diagnosis complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
