"""
Verify conclusions from gradient investigation.
Created by Claude.

Numerical tests for the mathematical claims in norm_scaling_analysis.tex and
the empirical claims in findings.md. Each test corresponds to a claim ID from
the verification plan (C1-C8).

Tests:
  C1: Kernel proportionality K(cx, z) = c * K(x, z)  [Theorem 1]
  C2: Posterior moment scaling mu(cx) = c*mu(x), var(cx) = c^2*var(x)  [Corollary 2]
  C3: Conditioning effectiveness rho^2 -> 1 for x* = c*x_target  [Corollary 1]
  C4: DA utility growth with c (Poisson-GP growth rate)
  C7: sigma_0^2 correction magnitude
  C8: Fixed-angle norm-scaling utility test

Usage:
    python investigations/validate_utility/verify_conclusions/verify_conclusions.py
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
_investigation_dir = _script_dir.parent
_gpytorch_dir = _investigation_dir.parent.parent
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
# Investigation parameters (explicit overrides)
# ---------------------------------------------------------------------------
N_TRAIN = 50
M = 50
TARGET_INDEX = 0    # Same as gradient.py
C_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
N_INDUCING_SAMPLE = 5  # Inducing points to sample for C1/C7
LAPLACE_SUM_THRESHOLD = 0.95  # sum(p_r) below this = unreliable


def compute_laplace_z_safe(logf_mean, logf_var, r_max):
    """Compute z-score predicting Laplace approximation validity.

    z_safe = (log(r_max) - logf_mean) / sqrt(logf_var)

    This measures how many standard deviations of the log-firing-rate
    distribution lie below log(r_max). When g ~ N(logf_mean, logf_var),
    the fraction of rates exp(g) exceeding r_max is Phi(-z_safe).

    Rules of thumb:
        z_safe > 3.0:  very safe (< 0.1% mass beyond r_max)
        z_safe > 2.0:  safe (< 2.3% mass beyond r_max)
        z_safe 1.5-2:  borderline (3-7% mass beyond r_max)
        z_safe < 1.5:  unreliable (> 7% mass beyond r_max)
        z_safe < 0.5:  broken (> 30% mass beyond r_max)

    Can be computed from GP marginal moments BEFORE running Laplace —
    cheap pre-check to decide whether utility is trustworthy.
    """
    if logf_var <= 0:
        return float('inf')  # Exact Poisson, no Laplace needed
    return (math.log(r_max) - logf_mean) / math.sqrt(logf_var)


def compute_kernel_angle(model, x1, x2):
    """Compute the kernel-based angle metric arccos(K(x,y)/sqrt(K(x,x)*K(y,y))).

    Note: this equals arccos(J(theta)/pi), NOT the geometric C-space angle theta.
    For small angles the approximation is excellent (error O(theta^4)).
    """
    with torch.no_grad():
        k_11 = model.covar_module(x1.unsqueeze(0), x1.unsqueeze(0)).evaluate().squeeze().item()
        k_22 = model.covar_module(x2.unsqueeze(0), x2.unsqueeze(0)).evaluate().squeeze().item()
        k_12 = model.covar_module(x1.unsqueeze(0), x2.unsqueeze(0)).evaluate().squeeze().item()
        cos_a = k_12 / (math.sqrt(k_11 * k_22) + 1e-30)
        cos_a = max(-1.0, min(1.0, cos_a))
        return math.acos(cos_a)


def compute_posterior_correlation(model, x_star, x_target):
    """Compute posterior correlation rho^2 between x_star and x_target."""
    with torch.no_grad():
        all_x = torch.stack([x_target, x_star])
        posterior = model(all_x)
        cov = posterior.covariance_matrix
        sigma_tt = cov[0, 0].item()
        sigma_ss = cov[1, 1].item()
        sigma_ts = cov[0, 1].item()
        rho_sq = sigma_ts ** 2 / (sigma_tt * sigma_ss + 1e-30)
    return rho_sq


def test_c1_kernel_proportionality(model, x_target, X_inducing):
    """C1: Test K(cx, z) = c * K(x, z) [Theorem 1]."""
    print("\n" + "=" * 70)
    print("C1: Kernel proportionality K(c*x_target, z) / K(x_target, z) = c")
    print("=" * 70)

    # Select a few inducing points to test against
    n_test = min(N_INDUCING_SAMPLE, X_inducing.shape[0])
    test_indices = list(range(n_test))

    with torch.no_grad():
        # K(x_target, z_j) for each inducing point
        k_base = model.covar_module(
            x_target.unsqueeze(0), X_inducing[test_indices]
        ).evaluate().squeeze()  # (n_test,)

    print(f"\n  {'c':>6}  ", end="")
    for j in range(n_test):
        print(f"{'z_' + str(j) + ' ratio':>12}  ", end="")
    print(f"{'mean_ratio':>12}  {'expected':>10}  {'rel_err':>10}")
    print("  " + "-" * (6 + 14 * n_test + 36))

    all_pass = True
    for c in C_VALUES:
        if c == 0:
            continue
        x_scaled = c * x_target
        with torch.no_grad():
            k_scaled = model.covar_module(
                x_scaled.unsqueeze(0), X_inducing[test_indices]
            ).evaluate().squeeze()

        ratios = (k_scaled / k_base).cpu().numpy()
        mean_ratio = ratios.mean()
        rel_err = abs(mean_ratio - c) / abs(c)

        print(f"  {c:6.1f}  ", end="")
        for j in range(n_test):
            print(f"{ratios[j]:12.6f}  ", end="")
        print(f"{mean_ratio:12.6f}  {c:10.1f}  {rel_err:10.6f}", end="")

        # Tolerance: sigma_0^2 correction should be small
        if rel_err > 0.01:
            print("  FAIL")
            all_pass = False
        else:
            print("  ok")

    print(f"\n  C1 verdict: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_c2_posterior_moment_scaling(model, x_target):
    """C2: Test mu(cx) = c*mu(x), var(cx) = c^2*var(x) [Corollary 2]."""
    print("\n" + "=" * 70)
    print("C2: Posterior moment scaling")
    print("=" * 70)

    with torch.no_grad():
        mu_base, var_base = get_gp_marginal_moments(model, x_target.unsqueeze(0))
        mu_base = mu_base.item()
        var_base = var_base.item()

    print(f"  Base: mu(x_t) = {mu_base:.6f}, var(x_t) = {var_base:.6f}")
    print(f"\n  {'c':>6}  {'mu(cx)':>12}  {'c*mu(x)':>12}  {'mu_err%':>10}  "
          f"{'var(cx)':>14}  {'c^2*var(x)':>14}  {'var_err%':>10}")
    print("  " + "-" * 90)

    all_pass = True
    for c in C_VALUES:
        if c == 0:
            continue
        x_scaled = c * x_target
        with torch.no_grad():
            mu_c, var_c = get_gp_marginal_moments(model, x_scaled.unsqueeze(0))
            mu_c = mu_c.item()
            var_c = var_c.item()

        mu_expected = c * mu_base
        var_expected = c ** 2 * var_base

        mu_err = abs(mu_c - mu_expected) / (abs(mu_expected) + 1e-30) * 100
        var_err = abs(var_c - var_expected) / (abs(var_expected) + 1e-30) * 100

        print(f"  {c:6.1f}  {mu_c:12.4f}  {mu_expected:12.4f}  {mu_err:10.4f}  "
              f"{var_c:14.4f}  {var_expected:14.4f}  {var_err:10.4f}", end="")

        if mu_err > 1.0 or var_err > 1.0:
            print("  FAIL")
            all_pass = False
        else:
            print("  ok")

    print(f"\n  C2 verdict: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_c3_conditioning_effectiveness(model, x_target):
    """C3: Test rho^2 -> 1 and var_ratio -> 0 for x* = c*x_target [Corollary 1]."""
    print("\n" + "=" * 70)
    print("C3: Conditioning effectiveness (rho^2 and variance ratio)")
    print("=" * 70)

    # Lambda at target for conditioning
    with torch.no_grad():
        post_t = model(x_target.unsqueeze(0))
        lambda_target = post_t.mean[0]

    print(f"  lambda(x_target) = {lambda_target.item():.6f}")
    print(f"\n  {'c':>6}  {'rho^2':>10}  {'var_marg':>12}  {'var_cond':>12}  "
          f"{'var_ratio':>10}  {'1-rho^2':>12}")
    print("  " + "-" * 75)

    all_pass = True
    for c in C_VALUES:
        if c == 0:
            continue
        x_scaled = c * x_target

        rho_sq = compute_posterior_correlation(model, x_scaled, x_target)

        with torch.no_grad():
            _, var_marg = get_gp_marginal_moments(model, x_scaled.unsqueeze(0))
            _, var_cond = get_gp_conditional_moments(
                model, x_scaled.unsqueeze(0), x_target, lambda_target
            )
            var_marg = var_marg.item()
            var_cond = var_cond.item()

        var_ratio = var_cond / (var_marg + 1e-30)

        print(f"  {c:6.1f}  {rho_sq:10.6f}  {var_marg:12.4f}  {var_cond:12.4f}  "
              f"{var_ratio:10.6f}  {1 - rho_sq:12.2e}", end="")

        # rho^2 should be very close to 1 (sigma_0 correction only)
        if rho_sq < 0.99:
            print("  FAIL (rho^2 < 0.99)")
            all_pass = False
        else:
            print("  ok")

    print(f"\n  C3 verdict: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_c4_da_utility_growth(model, likelihood, x_target, r_max):
    """C4: DA utility growth with c for x* = c*x_target."""
    print("\n" + "=" * 70)
    print("C4: DA utility growth with c (x* = c * x_target)")
    print("=" * 70)

    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()

    # Lambda at target for conditioning
    with torch.no_grad():
        post_t = model(x_target.unsqueeze(0))
        lambda_target = post_t.mean[0]

    r = torch.arange(0, r_max, dtype=x_target.dtype, device=x_target.device)

    print(f"\n  {'c':>6}  {'U_DA':>10}  {'H_marg':>10}  {'H_cond':>10}  "
          f"{'sum(p_r)':>10}  {'z_safe':>8}  {'logf_mean':>10}  {'logf_var':>10}  {'var_ratio':>10}")
    print("  " + "-" * 105)

    results = []
    for c in C_VALUES:
        x_scaled = c * x_target

        with torch.no_grad():
            mu, sigma2 = get_gp_marginal_moments(model, x_scaled.unsqueeze(0))
            H_marg = compute_H(mu, sigma2, r_max=r_max, a=A, lambda0=lambda0).item()

            mu_cond, s2_cond = get_gp_conditional_moments(
                model, x_scaled.unsqueeze(0), x_target, lambda_target
            )
            H_cond = compute_H(mu_cond, s2_cond, r_max=r_max, a=A, lambda0=lambda0).item()

            U_da = H_marg - H_cond

            logf_mean = (A * mu + lambda0).item()
            logf_var = (A ** 2 * sigma2).item()

            # Laplace validity: actual check + z-score predictor
            p_r, _ = _diff_laplace_log_probs(
                A * mu + lambda0, A ** 2 * sigma2, r
            )
            sum_pr = p_r.sum().item()
            z_safe = compute_laplace_z_safe(logf_mean, logf_var, r_max)

            var_ratio = s2_cond.item() / (sigma2.item() + 1e-30)

        results.append({
            'c': c, 'U_da': U_da, 'H_marg': H_marg, 'H_cond': H_cond,
            'sum_pr': sum_pr, 'z_safe': z_safe,
            'logf_mean': logf_mean, 'logf_var': logf_var,
            'var_ratio': var_ratio,
        })

        flag = ""
        if sum_pr < LAPLACE_SUM_THRESHOLD:
            flag = "  LAPLACE_BROKEN"
        if math.isnan(U_da) or math.isinf(U_da):
            flag = "  NUMERICAL_FAIL"

        print(f"  {c:6.1f}  {U_da:10.6f}  {H_marg:10.6f}  {H_cond:10.6f}  "
              f"{sum_pr:10.6f}  {z_safe:8.2f}  {logf_mean:10.4f}  {logf_var:10.4f}  "
              f"{var_ratio:10.6f}{flag}")

    # Analysis: only use Laplace-valid data
    valid = [r for r in results
             if r['sum_pr'] >= LAPLACE_SUM_THRESHOLD
             and not math.isnan(r['U_da'])
             and not math.isinf(r['U_da'])]
    broken = [r for r in results if r['sum_pr'] < LAPLACE_SUM_THRESHOLD]

    print(f"\n  Laplace-valid points: {len(valid)}/{len(results)} "
          f"(threshold: sum(p_r) >= {LAPLACE_SUM_THRESHOLD})")

    if len(valid) >= 2:
        monotonic = all(valid[i+1]['U_da'] >= valid[i]['U_da'] for i in range(len(valid)-1))
        print(f"  Monotonicity (valid points only): {'YES' if monotonic else 'NO'}")

        # Growth fit on valid data with c >= 2 and U > 0
        fit_pts = [r for r in valid if r['c'] >= 2.0 and r['U_da'] > 0]
        if len(fit_pts) >= 2:
            log_c = np.array([np.log(r['c']) for r in fit_pts])
            log_u = np.array([np.log(r['U_da']) for r in fit_pts])
            coeffs = np.polyfit(log_c, log_u, 1)
            print(f"  Growth exponent (valid data, c >= 2): alpha = {coeffs[0]:.3f}")
            print(f"    (alpha=1 = linear, alpha=2 = quadratic)")
            print(f"    Fit points: c = {[r['c'] for r in fit_pts]}")
        else:
            print(f"  Not enough valid points for growth fit (need >= 2 with c >= 2, U > 0)")

    if broken:
        print(f"\n  Laplace breakdown summary:")
        print(f"    First broken at c = {broken[0]['c']:.1f}")
        print(f"    z_safe at breakdown: {broken[0]['z_safe']:.2f}")
        print(f"    logf_mean at breakdown: {broken[0]['logf_mean']:.2f}")
        print(f"    logf_var at breakdown: {broken[0]['logf_var']:.2f}")

    print(f"\n  C4 verdict: see numbers above")
    return results


def test_c7_sigma0_correction(model, x_target, X_inducing):
    """C7: sigma_0^2 correction magnitude."""
    print("\n" + "=" * 70)
    print("C7: sigma_0^2 correction magnitude")
    print("=" * 70)

    sigma0_sq = model.covar_module.sigma_0.item() ** 2
    print(f"  sigma_0^2 = {sigma0_sq:.6f}")

    n_test = min(N_INDUCING_SAMPLE, X_inducing.shape[0])

    # For each c, compute fractional error |K(cx, z) - c*K(x,z)| / |c*K(x,z)|
    with torch.no_grad():
        k_base = model.covar_module(
            x_target.unsqueeze(0), X_inducing[:n_test]
        ).evaluate().squeeze()

        # Also compute v_t = K(x_t, x_t) (self-kernel)
        v_t = model.covar_module(
            x_target.unsqueeze(0), x_target.unsqueeze(0)
        ).evaluate().squeeze().item()

    q = v_t - sigma0_sq  # x_t^T C x_t
    print(f"  v_t = K(x_t, x_t) = {v_t:.4f}")
    print(f"  q = v_t - sigma_0^2 = {q:.4f}")
    print(f"  sigma_0^2 / v_t = {sigma0_sq / v_t:.6f}")

    print(f"\n  {'c':>6}  {'mean_frac_err':>14}  {'max_frac_err':>14}  "
          f"{'sigma0^2/(c^2*q)':>18}  {'sigma0^2/v_t':>14}")
    print("  " + "-" * 75)

    for c in C_VALUES:
        if c == 0:
            continue
        x_scaled = c * x_target
        with torch.no_grad():
            k_scaled = model.covar_module(
                x_scaled.unsqueeze(0), X_inducing[:n_test]
            ).evaluate().squeeze()

        frac_err = ((k_scaled - c * k_base).abs() / (c * k_base).abs()).cpu().numpy()
        mean_err = frac_err.mean()
        max_err = frac_err.max()
        theoretical_1 = sigma0_sq / (c ** 2 * q) if c > 0 else float('inf')
        theoretical_2 = sigma0_sq / v_t

        print(f"  {c:6.1f}  {mean_err:14.8f}  {max_err:14.8f}  "
              f"{theoretical_1:18.8f}  {theoretical_2:14.8f}")

    print(f"\n  C7 verdict: compare mean_frac_err column with theoretical bounds")
    print(f"  If errors track sigma0^2/(c^2*q): errors decrease with c^2")
    print(f"  If errors track sigma0^2/v_t: errors are constant across c")


def test_c8_fixed_angle_scaling(model, likelihood, x_target, X_pool, r_max):
    """C8: Fixed-angle norm-scaling utility test."""
    print("\n" + "=" * 70)
    print("C8: Fixed-angle norm-scaling utility test")
    print("=" * 70)

    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()

    # Lambda at target for conditioning
    with torch.no_grad():
        post_t = model(x_target.unsqueeze(0))
        lambda_target = post_t.mean[0]

    r = torch.arange(0, r_max, dtype=x_target.dtype, device=x_target.device)

    # Select natural images at varying angles from x_target
    # Compute angles for a batch of pool images, pick ones spanning the range
    n_scan = min(200, X_pool.shape[0])
    angles = []
    for i in range(n_scan):
        angle = compute_kernel_angle(model, x_target, X_pool[i])
        angles.append((i, angle))

    angles.sort(key=lambda x: x[1])

    # Pick ~4 images spanning the angle range
    # Target for small angle, then 25th, 50th, 75th percentile
    percentiles = [0, 25, 50, 75]
    selected = []
    for p in percentiles:
        idx = int(p / 100 * (len(angles) - 1))
        pool_idx, angle = angles[idx]
        selected.append((pool_idx, angle))

    print(f"  Scanned {n_scan} pool images, angles range: "
          f"[{angles[0][1]:.4f}, {angles[-1][1]:.4f}] rad")
    print(f"\n  Selected images:")
    for pool_idx, angle in selected:
        print(f"    Pool index {pool_idx}: angle = {angle:.4f} rad ({math.degrees(angle):.1f} deg)")

    # For each selected image, sweep c values
    c_values_c8 = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    for img_num, (pool_idx, base_angle) in enumerate(selected):
        x_img = X_pool[pool_idx]
        print(f"\n  --- Image {img_num} (pool idx {pool_idx}, base angle {base_angle:.4f} rad) ---")
        print(f"  {'c':>6}  {'U_DA':>10}  {'H_marg':>10}  {'H_cond':>10}  "
              f"{'var_ratio':>10}  {'angle':>8}  {'sum(p_r)':>10}  {'z_safe':>8}  {'logf_mean':>10}")
        print("  " + "-" * 105)

        prev_valid_u = None
        img_results = []
        for c in c_values_c8:
            x_scaled = c * x_img

            with torch.no_grad():
                # Marginal moments
                mu, sigma2 = get_gp_marginal_moments(model, x_scaled.unsqueeze(0))
                H_marg = compute_H(mu, sigma2, r_max=r_max, a=A, lambda0=lambda0).item()

                # Conditional moments
                mu_cond, s2_cond = get_gp_conditional_moments(
                    model, x_scaled.unsqueeze(0), x_target, lambda_target
                )
                H_cond = compute_H(mu_cond, s2_cond, r_max=r_max, a=A, lambda0=lambda0).item()

                U_da = H_marg - H_cond

                logf_mean = (A * mu + lambda0).item()
                logf_var = (A ** 2 * sigma2).item()

                # Laplace validity
                p_r, _ = _diff_laplace_log_probs(
                    A * mu + lambda0, A ** 2 * sigma2, r
                )
                sum_pr = p_r.sum().item()
                z_safe = compute_laplace_z_safe(logf_mean, logf_var, r_max)

                var_ratio = s2_cond.item() / (sigma2.item() + 1e-30)

                # Angle check (should stay ~constant)
                angle = compute_kernel_angle(model, x_scaled, x_target)

            laplace_ok = sum_pr >= LAPLACE_SUM_THRESHOLD
            flag = ""
            if not laplace_ok:
                flag += " LAPLACE_BROKEN"
            elif prev_valid_u is not None and U_da < prev_valid_u - 1e-6:
                flag += " NON_MONOTONIC"
            if math.isnan(U_da) or math.isinf(U_da):
                flag += " NUMERICAL_FAIL"

            print(f"  {c:6.1f}  {U_da:10.6f}  {H_marg:10.6f}  {H_cond:10.6f}  "
                  f"{var_ratio:10.6f}  {angle:8.4f}  {sum_pr:10.6f}  {z_safe:8.2f}  "
                  f"{logf_mean:10.4f}{flag}")

            img_results.append({
                'c': c, 'U_da': U_da, 'sum_pr': sum_pr, 'z_safe': z_safe,
                'laplace_ok': laplace_ok,
            })
            if laplace_ok and not math.isnan(U_da):
                prev_valid_u = U_da

        # Per-image summary
        valid_pts = [r for r in img_results if r['laplace_ok'] and not math.isnan(r['U_da'])]
        if len(valid_pts) >= 2:
            mono = all(valid_pts[i+1]['U_da'] >= valid_pts[i]['U_da'] - 1e-6
                       for i in range(len(valid_pts) - 1))
            max_c_valid = valid_pts[-1]['c']
            print(f"  >> Valid range: c <= {max_c_valid:.0f}, "
                  f"monotonic (valid only): {'YES' if mono else 'NO'}, "
                  f"max U_DA = {max(r['U_da'] for r in valid_pts):.6f}")

    print(f"\n  C8 verdict: see tables above")
    print(f"  Key checks:")
    print(f"    1. Does U_DA increase monotonically with c (within Laplace-valid range)?")
    print(f"    2. Is the growth rate larger for smaller angles?")
    print(f"    3. Does the angle stay approximately constant across c values?")
    print(f"    4. At what c/z_safe does Laplace break?")


def main():
    # =========================================================================
    # Train model
    # =========================================================================
    print("=" * 70)
    print("Training model (same config as gradient.py)")
    print("=" * 70)

    config = build_config_from_defaults(
        mode='default_gpy',
        M=M,
        n_train=N_TRAIN,
    )
    result = run_single_config(config)

    if result is None or result.get('status') != 'success':
        print("ERROR: Training failed")
        return

    model = result['_model']
    likelihood = result['_likelihood']
    indices_train = result['_indices_train']
    model.eval()

    A = likelihood.A.squeeze()
    lambda0 = likelihood.lambda0.squeeze()
    r_max = config['r_max']
    device = next(model.parameters()).device
    dtype = torch.float32

    print(f"\n  test_r = {result['test_r']:.4f}")
    print(f"  A = {A.item():.4f}, lambda0 = {lambda0.item():.4f}")
    print(f"  sigma_0 = {model.covar_module.sigma_0.item():.4f}")
    print(f"  seed = {config['seed']}, cell = {config['cell']}")

    # Build pool and get x_target
    data_path = _gpytorch_dir.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    data = np.load(data_path)
    X_all = torch.cat([
        torch.tensor(data['images_train'], dtype=dtype),
        torch.tensor(data['images_val'], dtype=dtype),
    ], dim=0).reshape(-1, config['n_px_side'] ** 2).to(device)

    pool_indices = sorted(set(range(X_all.shape[0])) - set(indices_train.cpu().numpy().tolist()))
    X_pool = X_all[pool_indices]
    x_target = X_pool[TARGET_INDEX].clone()

    # Get inducing points
    X_inducing = model.variational_strategy.inducing_points.data

    print(f"  Pool: {X_pool.shape[0]} images")
    print(f"  Inducing points: {X_inducing.shape[0]}")

    # =========================================================================
    # Run tests
    # =========================================================================
    test_c1_kernel_proportionality(model, x_target, X_inducing)
    test_c2_posterior_moment_scaling(model, x_target)
    test_c3_conditioning_effectiveness(model, x_target)
    c4_results = test_c4_da_utility_growth(model, likelihood, x_target, r_max)
    test_c7_sigma0_correction(model, x_target, X_inducing)
    test_c8_fixed_angle_scaling(model, likelihood, x_target, X_pool, r_max)

    # =========================================================================
    # Laplace Breakdown Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("Laplace Breakdown Analysis")
    print("=" * 70)

    print("""
  The Laplace approximation truncates at r_max (default 100). It breaks when
  the firing rate distribution has significant mass at r > r_max.

  Root cause: g ~ N(logf_mean, logf_var). When g is large, rate = exp(g) >> r_max.
  The fraction of g above log(r_max) determines the missed probability mass.

  Pre-check metric (no Laplace computation needed):

      z_safe = (log(r_max) - logf_mean) / sqrt(logf_var)

  where logf_mean = A*mu + lambda0, logf_var = A^2*sigma2 (from GP marginals).

  Calibration from C4/C8 data:""")

    # Collect all (z_safe, sum_pr) pairs from C4
    print(f"\n  {'z_safe':>8}  {'sum(p_r)':>10}  {'c':>6}  {'status':>14}")
    print("  " + "-" * 45)
    for r in c4_results:
        status = "OK" if r['sum_pr'] >= LAPLACE_SUM_THRESHOLD else "BROKEN"
        print(f"  {r['z_safe']:8.2f}  {r['sum_pr']:10.6f}  {r['c']:6.1f}  {status:>14}")

    print(f"""
  Recommendation:
    z_safe > 2.0:  trust the utility value
    z_safe 1.5-2:  borderline — flag but don't discard
    z_safe < 1.5:  do not trust — Laplace truncation corrupts entropy
    z_safe < 0.5:  catastrophically broken — utility may be negative/NaN

  To compute z_safe for a candidate image x:
    1. mu, sigma2 = get_gp_marginal_moments(model, x)
    2. logf_mean = A * mu + lambda0
    3. logf_var = A^2 * sigma2
    4. z_safe = (log(r_max) - logf_mean) / sqrt(logf_var)

  This is O(1) cost (no Laplace integral) and can be used to filter
  candidates before computing the full utility.
""")

    print("=" * 70)
    print("All tests complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
