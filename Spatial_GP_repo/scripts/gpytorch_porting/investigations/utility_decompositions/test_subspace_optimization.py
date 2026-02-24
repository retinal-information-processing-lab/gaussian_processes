"""
Strict tests for subspace_optimization.py.

Tests mathematical invariants of the PCA and C-eigenspace decompositions,
gradient flow through the reconstruction pipeline, norm constraint
enforcement, and that optimization actually improves utility.

Uses REAL data (PNAS dataset) — trains a model once, then runs all checks.

Usage:
    python investigations/utility_decompositions/test_subspace_optimization.py
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_gpytorch_dir = _script_dir.parent.parent
sys.path.insert(0, str(_gpytorch_dir))

import importlib.util

# Import subspace_optimization functions under test
_sub_path = _script_dir / 'subspace_optimization.py'
_spec = importlib.util.spec_from_file_location("subspace_optimization", str(_sub_path))
_sub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sub)

z_to_image = _sub.z_to_image
image_to_z = _sub.image_to_z
rf_pearson_r = _sub.rf_pearson_r
compute_pca = _sub.compute_pca
compute_c_eigenspace = _sub.compute_c_eigenspace
gradient_ascent = _sub.gradient_ascent

# Also need setup and utility
_utility_dir = _script_dir.parent / 'utility'
sys.path.insert(0, str(_utility_dir))
from explore_utility import setup

_local_utils_path = _gpytorch_dir / 'utils.py'
_spec_u = importlib.util.spec_from_file_location("gpytorch_porting_utils", str(_local_utils_path))
_local_utils = importlib.util.module_from_spec(_spec_u)
_spec_u.loader.exec_module(_local_utils)
get_gp_marginal_moments = _local_utils.get_gp_marginal_moments

_acquisition_path = _gpytorch_dir / 'acquisition.py'
_spec_a = importlib.util.spec_from_file_location("gpytorch_porting_acquisition", str(_acquisition_path))
_acquisition = importlib.util.module_from_spec(_spec_a)
_spec_a.loader.exec_module(_acquisition)
distribution_aware_utility = _acquisition.distribution_aware_utility

# ============================================================================

M_OVERRIDE = 50
N_TRAIN_OVERRIDE = 50
ATOL_ORTHO_EIGH = 1e-5   # orthonormality tolerance for eigh (symmetric, stable)
ATOL_ORTHO_SVD = 5e-3    # orthonormality tolerance for SVD on large non-square matrices
                         # (float32 SVD of 3160x2356 matrix: ~1e-3 typical error)
ATOL_ROUNDTRIP_EIGH = 1e-4  # round-trip reconstruction for eigh-based decomposition
ATOL_ROUNDTRIP_SVD = 1e-2   # round-trip for SVD-based decomposition (cascades from ortho error)
N_OPT_STEPS = 5       # short optimization for gradient/utility tests

passed = 0
failed = 0
errors = []


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        failed += 1
        errors.append(name)


def main():
    global passed, failed
    t0 = time.time()

    # === Train model once (RBF for well-behaved optimization) ===
    print("=" * 70)
    print("Training model (RBF, M=50, n_train=50)")
    print("=" * 70)
    env = setup(kernel_type='rbf', M_override=M_OVERRIDE,
                n_train_override=N_TRAIN_OVERRIDE)
    model = env['model']
    likelihood = env['likelihood']
    X_pool = env['X_pool']
    X_train = env['X_train']
    config = env['config']
    test_r = env['test_r']
    r_max = config['r_max']
    f_max = config['f_max']
    n_px_side = config['n_px_side']

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_pixels = n_px_side ** 2

    kernel = (getattr(model, 'covar_module', None)
              or getattr(model, 'kernel', None))
    if not (hasattr(kernel, '_cached_mask') and kernel._cached_mask is not None):
        with torch.no_grad():
            _ = model(X_pool[0].unsqueeze(0))
    rf_mask = kernel._cached_mask.squeeze()
    n_rf = rf_mask.sum().item()

    X_all = torch.cat([X_pool, X_train], dim=0)

    print(f"\nModel ready. n_rf={n_rf}, n_pool={X_pool.shape[0]}, "
          f"test_r={test_r:.4f}")

    # =====================================================================
    # TEST GROUP 1: PCA decomposition mathematical properties
    # =====================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 1: PCA decomposition")
    print("=" * 70)

    # 1a: PCA basis orthonormality
    offset_pca, basis_pca, eigvals_pca, K_pca, meta_pca = compute_pca(
        X_all, rf_mask, var_threshold=0.80)

    gram = basis_pca.T @ basis_pca  # Should be I_K
    identity_err = (gram - torch.eye(K_pca, device=device, dtype=dtype)).abs().max().item()
    check("PCA basis orthonormality (V^T V = I, float32 SVD)",
          identity_err < ATOL_ORTHO_SVD,
          f"max |V^T V - I| = {identity_err:.2e}, tol={ATOL_ORTHO_SVD:.0e}")

    # 1b: PCA eigenvalues are non-negative and descending
    eigvals_np = eigvals_pca.cpu().numpy()
    check("PCA eigenvalues non-negative",
          np.all(eigvals_np >= -1e-7),
          f"min eigenvalue = {eigvals_np.min():.6e}")
    check("PCA eigenvalues descending",
          np.all(np.diff(eigvals_np[:K_pca]) <= 1e-7),
          f"max ascending diff = {np.diff(eigvals_np[:K_pca]).max():.6e}")

    # 1c: variance explained is correct
    total_var = eigvals_pca.sum().item()
    retained_var = eigvals_pca[:K_pca].sum().item()
    actual_ve = retained_var / total_var
    check("PCA variance explained matches metadata",
          abs(actual_ve - meta_pca['var_explained']) < 1e-6,
          f"computed={actual_ve:.6f}, meta={meta_pca['var_explained']:.6f}")
    check("PCA variance explained >= threshold",
          meta_pca['var_explained'] >= 0.80 - 1e-6,
          f"var_explained={meta_pca['var_explained']:.6f}, threshold=0.80")

    # 1d: PCA with var_threshold=1.0 gives full rank when n_samples > n_rf
    offset_full, basis_full, eigvals_full, K_full, meta_full = compute_pca(
        X_all, rf_mask, var_threshold=1.0)
    n_samples = X_all.shape[0]
    expected_full_rank = min(n_samples, n_rf)
    check("PCA full-rank: K = min(n_samples, n_rf) at var_threshold=1.0",
          K_full == expected_full_rank,
          f"K={K_full}, expected={expected_full_rank} "
          f"(n_samples={n_samples}, n_rf={n_rf})")

    # 1e: Full-rank PCA reconstruction is lossless for dataset images
    test_img = X_pool[7]  # arbitrary image
    z_test = image_to_z(test_img, basis_full, offset_full, rf_mask)
    x_recon = z_to_image(z_test, basis_full, offset_full, rf_mask,
                         n_pixels, dtype, device)
    recon_err = (test_img[rf_mask] - x_recon[rf_mask]).abs().max().item()
    check("PCA full-rank reconstruction is lossless (float32 SVD)",
          recon_err < ATOL_ROUNDTRIP_SVD,
          f"max |x - recon(x)| = {recon_err:.2e}, tol={ATOL_ROUNDTRIP_SVD:.0e}")

    # 1f: Truncated PCA reconstruction is lossy (non-trivial truncation)
    z_trunc = image_to_z(test_img, basis_pca, offset_pca, rf_mask)
    x_recon_trunc = z_to_image(z_trunc, basis_pca, offset_pca, rf_mask,
                               n_pixels, dtype, device)
    trunc_err = (test_img[rf_mask] - x_recon_trunc[rf_mask]).norm().item()
    check("PCA truncated reconstruction is lossy (K < full rank)",
          trunc_err > 0.1,
          f"recon error norm = {trunc_err:.4f} (should be non-trivial)")

    # 1g: PCA projection is idempotent: project(reconstruct(project(x))) == reconstruct(project(x))
    z_once = image_to_z(test_img, basis_pca, offset_pca, rf_mask)
    x_proj = z_to_image(z_once, basis_pca, offset_pca, rf_mask,
                        n_pixels, dtype, device)
    z_twice = image_to_z(x_proj, basis_pca, offset_pca, rf_mask)
    x_proj2 = z_to_image(z_twice, basis_pca, offset_pca, rf_mask,
                          n_pixels, dtype, device)
    idempotent_err = (x_proj[rf_mask] - x_proj2[rf_mask]).abs().max().item()
    check("PCA projection is idempotent (float32 SVD)",
          idempotent_err < ATOL_ROUNDTRIP_SVD,
          f"max |proj(proj(x)) - proj(x)| = {idempotent_err:.2e}")

    # 1h: z=0 reconstructs to the mean image
    z_zero = torch.zeros(K_pca, dtype=dtype, device=device)
    x_mean = z_to_image(z_zero, basis_pca, offset_pca, rf_mask,
                        n_pixels, dtype, device)
    # Compare with actual mean of RF pixels
    actual_mean_rf = X_all[:, rf_mask].mean(dim=0)
    mean_err = (x_mean[rf_mask] - actual_mean_rf).abs().max().item()
    check("PCA z=0 reconstructs to dataset mean",
          mean_err < 1e-6,
          f"max |z_to_image(0) - mean| = {mean_err:.2e}")

    # =====================================================================
    # TEST GROUP 2: C-eigenspace decomposition mathematical properties
    # =====================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 2: C-eigenspace decomposition")
    print("=" * 70)

    offset_ceig, basis_ceig, eigvals_ceig, K_ceig, meta_ceig = compute_c_eigenspace(
        kernel, rf_mask, eigen_rel_threshold=1e-3)

    # 2a: C-eigen basis orthonormality
    gram_c = basis_ceig.T @ basis_ceig
    identity_err_c = (gram_c - torch.eye(K_ceig, device=device, dtype=dtype)).abs().max().item()
    check("C-eigen basis orthonormality (U^T U = I, eigh)",
          identity_err_c < ATOL_ORTHO_EIGH,
          f"max |U^T U - I| = {identity_err_c:.2e}")

    # 2b: C-eigen eigenvalues non-negative and descending
    ceig_np = eigvals_ceig.cpu().numpy()
    check("C-eigen eigenvalues non-negative",
          np.all(ceig_np >= -1e-7),
          f"min eigenvalue = {ceig_np.min():.6e}")
    check("C-eigen eigenvalues descending",
          np.all(np.diff(ceig_np[:K_ceig]) <= 1e-7),
          f"max ascending diff = {np.diff(ceig_np[:K_ceig]).max():.6e}")

    # 2c: C-eigen offset is zeros
    check("C-eigen offset is zeros",
          offset_ceig.abs().max().item() == 0.0,
          f"max |offset| = {offset_ceig.abs().max().item():.2e}")

    # 2d: C-eigen no_filter gives full rank
    offset_nf, basis_nf, eigvals_nf, K_nf, meta_nf = compute_c_eigenspace(
        kernel, rf_mask, eigen_rel_threshold=1e-3, no_filter=True)
    check("C-eigen no_filter gives K = n_rf",
          K_nf == n_rf,
          f"K={K_nf}, n_rf={n_rf}")

    # 2e: C-eigen no_filter reconstruction is lossless (just a rotation)
    z_ceig_full = image_to_z(test_img, basis_nf, offset_nf, rf_mask)
    x_ceig_recon = z_to_image(z_ceig_full, basis_nf, offset_nf, rf_mask,
                              n_pixels, dtype, device)
    ceig_recon_err = (test_img[rf_mask] - x_ceig_recon[rf_mask]).abs().max().item()
    check("C-eigen no_filter reconstruction is lossless (eigh)",
          ceig_recon_err < ATOL_ROUNDTRIP_EIGH,
          f"max |x - recon(x)| = {ceig_recon_err:.2e}")

    # 2f: C-eigen truncated reconstruction IS lossy
    z_ceig_trunc = image_to_z(test_img, basis_ceig, offset_ceig, rf_mask)
    x_ceig_trunc = z_to_image(z_ceig_trunc, basis_ceig, offset_ceig, rf_mask,
                              n_pixels, dtype, device)
    ceig_trunc_err = (test_img[rf_mask] - x_ceig_trunc[rf_mask]).norm().item()
    check("C-eigen truncated reconstruction is lossy",
          ceig_trunc_err > 0.1,
          f"recon error norm = {ceig_trunc_err:.4f}")

    # 2g: C-eigen with lower threshold retains MORE dimensions
    _, _, _, K_loose, _ = compute_c_eigenspace(
        kernel, rf_mask, eigen_rel_threshold=1e-6)
    check("Lower eigen threshold => more dimensions",
          K_loose > K_ceig,
          f"K(1e-6)={K_loose}, K(1e-3)={K_ceig}")

    # =====================================================================
    # TEST GROUP 3: Gradient flow through z_to_image
    # =====================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 3: Gradient flow")
    print("=" * 70)

    # 3a: PCA gradient flow — z_to_image produces a tensor with grad_fn
    z_grad = torch.randn(K_pca, dtype=dtype, device=device, requires_grad=True)
    x_out = z_to_image(z_grad, basis_pca, offset_pca, rf_mask,
                       n_pixels, dtype, device)
    check("PCA z_to_image output has grad_fn",
          x_out.grad_fn is not None,
          f"grad_fn = {x_out.grad_fn}")

    # 3b: PCA gradient flows back to z through a scalar loss
    scalar_loss = x_out.sum()
    scalar_loss.backward()
    check("PCA gradient reaches z (z.grad is not None and non-zero)",
          z_grad.grad is not None and z_grad.grad.norm().item() > 0,
          f"grad norm = {z_grad.grad.norm().item() if z_grad.grad is not None else 'None'}")

    # 3c: PCA gradient is correct (d/dz sum(offset + basis @ z) = basis^T @ ones)
    # sum(x_full) = sum(x_rf) = sum(offset + basis @ z) over RF pixels
    # d/dz = basis^T @ ones_rf
    expected_grad = basis_pca.T @ torch.ones(n_rf, dtype=dtype, device=device)
    actual_grad = z_grad.grad
    grad_err = (actual_grad - expected_grad).abs().max().item()
    check("PCA gradient is mathematically correct",
          grad_err < 1e-5,
          f"max |grad - basis^T @ 1| = {grad_err:.2e}")

    # 3d: C-eigen gradient flow
    z_grad_c = torch.randn(K_ceig, dtype=dtype, device=device, requires_grad=True)
    x_out_c = z_to_image(z_grad_c, basis_ceig, offset_ceig, rf_mask,
                         n_pixels, dtype, device)
    scalar_loss_c = x_out_c.sum()
    scalar_loss_c.backward()
    check("C-eigen gradient reaches z",
          z_grad_c.grad is not None and z_grad_c.grad.norm().item() > 0,
          f"grad norm = {z_grad_c.grad.norm().item() if z_grad_c.grad is not None else 'None'}")

    # 3e: Gradient flows through the FULL pipeline (z -> image -> utility)
    # This is the critical test: does gradient survive through kernel evaluation?
    z_pipe = torch.zeros(K_pca, dtype=dtype, device=device, requires_grad=True)
    x_pipe = z_to_image(z_pipe, basis_pca, offset_pca, rf_mask,
                        n_pixels, dtype, device)
    x_target_pipe = X_pool[0]
    result_pipe = distribution_aware_utility(
        model, likelihood,
        x_pipe.unsqueeze(0),
        x_target_pipe.unsqueeze(0),
        r_max=r_max, adaptive_r_max=False, sample_lambda=False,
    )
    utility_pipe = result_pipe['utility'].squeeze()
    utility_pipe.backward()
    check("Gradient flows through full pipeline (z -> image -> kernel -> utility)",
          z_pipe.grad is not None and z_pipe.grad.norm().item() > 0,
          f"grad norm = {z_pipe.grad.norm().item() if z_pipe.grad is not None else 'None'}")

    # 3f: torch.no_grad() BLOCKS gradient (sanity check that test 3e is real)
    z_nograd = torch.zeros(K_pca, dtype=dtype, device=device, requires_grad=True)
    with torch.no_grad():
        x_nograd = z_to_image(z_nograd, basis_pca, offset_pca, rf_mask,
                              n_pixels, dtype, device)
    check("torch.no_grad() blocks grad_fn (sanity check)",
          x_nograd.grad_fn is None,
          f"grad_fn = {x_nograd.grad_fn}")

    # =====================================================================
    # TEST GROUP 4: rf_pearson_r correctness
    # =====================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 4: rf_pearson_r")
    print("=" * 70)

    # 4a: Pearson r of identical images is 1.0
    img_a = X_pool[3]
    check("Pearson r(x, x) = 1.0",
          abs(rf_pearson_r(img_a, img_a, rf_mask) - 1.0) < 1e-5,
          f"r = {rf_pearson_r(img_a, img_a, rf_mask):.6f}")

    # 4b: Pearson r of scaled image is still 1.0 (scale invariant)
    img_scaled = img_a * 3.7 + 1.2
    check("Pearson r is scale-invariant: r(x, a*x+b) = 1.0",
          abs(rf_pearson_r(img_a, img_scaled, rf_mask) - 1.0) < 1e-4,
          f"r = {rf_pearson_r(img_a, img_scaled, rf_mask):.6f}")

    # 4c: Pearson r is symmetric
    img_b = X_pool[5]
    r_ab = rf_pearson_r(img_a, img_b, rf_mask)
    r_ba = rf_pearson_r(img_b, img_a, rf_mask)
    check("Pearson r is symmetric: r(a,b) = r(b,a)",
          abs(r_ab - r_ba) < 1e-6,
          f"r(a,b)={r_ab:.6f}, r(b,a)={r_ba:.6f}")

    # 4d: Pearson r against constant image is 0 (or handled gracefully)
    img_const = torch.ones_like(img_a) * 0.5
    r_const = rf_pearson_r(img_a, img_const, rf_mask)
    check("Pearson r against constant image is 0",
          abs(r_const) < 1e-5,
          f"r = {r_const:.6f}")

    # =====================================================================
    # TEST GROUP 5: Norm constraint enforcement
    # =====================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 5: Norm constraint in optimization")
    print("=" * 70)

    # Run a short PCA optimization WITH norm constraint
    z_max_norm = 20.0  # deliberately tight
    z_start = torch.zeros(K_pca, dtype=dtype, device=device)
    x_target_opt = X_pool[0]

    x_final, z_final, history = gradient_ascent(
        model, likelihood, z_start, x_target_opt.unsqueeze(0),
        basis_pca, offset_pca, rf_mask, r_max, f_max,
        n_pixels, dtype, device, n_steps=N_OPT_STEPS, lr=0.5,
        z_max_norm=z_max_norm,
        x_target_for_pearson=x_target_opt,
    )

    final_z_norm = z_final.norm().item()
    check("Norm constraint enforced: ||z_final|| <= z_max_norm",
          final_z_norm <= z_max_norm + 1e-5,
          f"||z_final|| = {final_z_norm:.4f}, z_max_norm = {z_max_norm:.4f}")

    # 5b: Without norm constraint, z can grow beyond that limit
    x_final_nc, z_final_nc, history_nc = gradient_ascent(
        model, likelihood, z_start, x_target_opt.unsqueeze(0),
        basis_pca, offset_pca, rf_mask, r_max, f_max,
        n_pixels, dtype, device, n_steps=N_OPT_STEPS, lr=0.5,
        z_max_norm=None,
        x_target_for_pearson=x_target_opt,
    )
    nc_z_norm = z_final_nc.norm().item()
    # The unconstrained optimizer should explore further
    check("Without norm constraint, z grows larger",
          nc_z_norm >= final_z_norm - 1e-3,
          f"unconstrained ||z|| = {nc_z_norm:.4f}, "
          f"constrained ||z|| = {final_z_norm:.4f}")

    # =====================================================================
    # TEST GROUP 6: Optimization actually improves utility
    # =====================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 6: Utility improvement")
    print("=" * 70)

    # 6a: Final utility >= starting utility (PCA)
    u_start_pca = history['utility'][0]
    u_final_pca = history['utility'][-1]
    check("PCA optimization improves utility (U_final >= U_start)",
          u_final_pca >= u_start_pca - 1e-8,
          f"U_start={u_start_pca:.6f}, U_final={u_final_pca:.6f}")

    # 6b: Run C-eigen optimization and verify utility improves
    # Use smoothed target as start (C-eigen default)
    from scipy.ndimage import gaussian_filter
    x_target_c = X_pool[0]
    x_target_2d = x_target_c.cpu().numpy().reshape(n_px_side, n_px_side)
    x_smoothed_2d = gaussian_filter(x_target_2d, sigma=5.0)
    x_smoothed = torch.tensor(x_smoothed_2d.reshape(-1), dtype=dtype, device=device)
    z_start_c = image_to_z(x_smoothed, basis_ceig, offset_ceig, rf_mask)

    x_final_c, z_final_c, history_c = gradient_ascent(
        model, likelihood, z_start_c, x_target_c.unsqueeze(0),
        basis_ceig, offset_ceig, rf_mask, r_max, f_max,
        n_pixels, dtype, device, n_steps=N_OPT_STEPS, lr=0.5,
        z_max_norm=None,
        x_target_for_pearson=x_target_c,
    )
    u_start_c = history_c['utility'][0]
    u_final_c = history_c['utility'][-1]
    check("C-eigen optimization improves utility",
          u_final_c >= u_start_c - 1e-8,
          f"U_start={u_start_c:.6f}, U_final={u_final_c:.6f}")

    # 6c: Utility values are positive (DA utility should always be >= 0)
    check("PCA utility is non-negative",
          u_final_pca >= -1e-8,
          f"U_final = {u_final_pca:.6f}")
    check("C-eigen utility is non-negative",
          u_final_c >= -1e-8,
          f"U_final = {u_final_c:.6f}")

    # 6d: History has expected keys and lengths
    expected_keys = {'step', 'utility', 'grad_norm_z', 'z_norm', 'rf_norm',
                     'step_time', 'pearson_r'}
    check("History has all expected keys (single-target)",
          expected_keys.issubset(set(history.keys())),
          f"missing: {expected_keys - set(history.keys())}")

    n_recorded = len(history['step'])
    for key in ['utility', 'grad_norm_z', 'z_norm', 'rf_norm', 'step_time',
                'pearson_r']:
        if key in history:
            check(f"History['{key}'] length matches n_steps",
                  len(history[key]) == n_recorded,
                  f"len={len(history[key])}, expected={n_recorded}")

    # =====================================================================
    # TEST GROUP 7: Reconstructed image is in the subspace
    # =====================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 7: Optimized image lives in the subspace")
    print("=" * 70)

    # 7a: PCA optimized image round-trips through PCA exactly
    z_roundtrip = image_to_z(x_final, basis_pca, offset_pca, rf_mask)
    x_roundtrip = z_to_image(z_roundtrip, basis_pca, offset_pca, rf_mask,
                             n_pixels, dtype, device)
    roundtrip_err = (x_final[rf_mask] - x_roundtrip[rf_mask]).abs().max().item()
    check("PCA optimized image round-trips (lives in subspace, float32 SVD)",
          roundtrip_err < ATOL_ROUNDTRIP_SVD,
          f"max error = {roundtrip_err:.2e}")

    # 7b: C-eigen optimized image round-trips through C-eigenspace exactly
    z_rt_c = image_to_z(x_final_c, basis_ceig, offset_ceig, rf_mask)
    x_rt_c = z_to_image(z_rt_c, basis_ceig, offset_ceig, rf_mask,
                        n_pixels, dtype, device)
    rt_err_c = (x_final_c[rf_mask] - x_rt_c[rf_mask]).abs().max().item()
    check("C-eigen optimized image round-trips exactly (eigh)",
          rt_err_c < ATOL_ROUNDTRIP_EIGH,
          f"max error = {rt_err_c:.2e}")

    # 7c: Pixels outside RF mask are zero for both methods
    check("PCA: pixels outside RF are zero",
          x_final[~rf_mask].abs().max().item() < 1e-8,
          f"max non-RF pixel = {x_final[~rf_mask].abs().max().item():.2e}")
    check("C-eigen: pixels outside RF are zero",
          x_final_c[~rf_mask].abs().max().item() < 1e-8,
          f"max non-RF pixel = {x_final_c[~rf_mask].abs().max().item():.2e}")

    # =====================================================================
    # TEST GROUP 8: Cross-method consistency
    # =====================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 8: Cross-method consistency")
    print("=" * 70)

    # 8a: Both methods produce valid images (finite, no NaN)
    check("PCA final image is finite (no NaN/inf)",
          torch.isfinite(x_final).all().item(),
          f"has NaN: {torch.isnan(x_final).any().item()}, "
          f"has inf: {torch.isinf(x_final).any().item()}")
    check("C-eigen final image is finite",
          torch.isfinite(x_final_c).all().item(),
          f"has NaN: {torch.isnan(x_final_c).any().item()}")

    # 8b: Both methods start from z_to_image(z_start) and the result matches
    # x_start in history (the utility was computed on it)
    x_start_pca_check = z_to_image(z_start, basis_pca, offset_pca, rf_mask,
                                   n_pixels, dtype, device)
    # Verify the start utility from history matches a fresh computation
    with torch.no_grad():
        result_fresh = distribution_aware_utility(
            model, likelihood,
            x_start_pca_check.unsqueeze(0),
            x_target_opt.unsqueeze(0),
            r_max=r_max, adaptive_r_max=False, sample_lambda=False,
        )
    u_fresh = result_fresh['utility'].item()
    u_hist_start = history['utility'][0]
    check("Start utility from history matches fresh computation",
          abs(u_fresh - u_hist_start) < 1e-4,
          f"fresh={u_fresh:.6f}, history[0]={u_hist_start:.6f}")

    # 8c: z_final from gradient_ascent reconstructs to x_final
    x_from_z = z_to_image(z_final, basis_pca, offset_pca, rf_mask,
                          n_pixels, dtype, device)
    zf_err = (x_final[rf_mask] - x_from_z[rf_mask]).abs().max().item()
    check("z_final reconstructs to x_final",
          zf_err < 1e-5,
          f"max |x_final - z_to_image(z_final)| = {zf_err:.2e}")

    # =====================================================================
    # TEST GROUP 9: Early stopping and step timing
    # =====================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 9: Early stopping and timing")
    print("=" * 70)

    # 9a: All step times are non-negative (step_time[0] is the pre-opt entry = 0)
    check("All step times are non-negative",
          all(t >= 0 for t in history['step_time']),
          f"min step_time = {min(history['step_time']):.4f}")

    # 9a2: Actual optimization steps have positive time
    opt_step_times = history['step_time'][1:]  # skip pre-opt entry
    check("Optimization step times are positive",
          all(t > 0 for t in opt_step_times),
          f"min opt step_time = {min(opt_step_times):.4f}" if opt_step_times else "no opt steps")

    # 9b: Steps start with -1 (pre-opt) then sequential from 0
    steps = history['step']
    expected = [-1] + list(range(len(steps) - 1))
    check("Steps are [-1, 0, 1, 2, ...] (pre-opt then sequential)",
          steps == expected,
          f"steps = {steps}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed ({elapsed:.1f}s)")
    print("=" * 70)
    if errors:
        print("\nFailed tests:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == '__main__':
    main()
