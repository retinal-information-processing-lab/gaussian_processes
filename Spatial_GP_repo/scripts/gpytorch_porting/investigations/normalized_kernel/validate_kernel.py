#!/usr/bin/env python3
"""
Pre-training validation suite for ArcCosineKernelNormalized.
Created by Claude.

Loads real PNAS images and runs validation checks comparing the normalized
kernel against the unnormalized one. All parameters read from default_params.json
via build_config_from_defaults().

Run BEFORE training to verify correctness:
    python investigations/normalized_kernel/validate_kernel.py

Groups:
  1. Mathematical correctness (normalization identity, diagonal, symmetry, PSD)
  2. Parameter sensitivity (gradient flow, Amp/sigma0 effect, mask consistency)
  3. GP pipeline integration (K_tilde conditioning, eigenspectrum, forward pass, LBFGS step)
  4. Edge cases (identical images via full path, zero image, pivoted Cholesky)
"""

import sys
import torch
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_gpytorch_dir = _script_dir.parent.parent
sys.path.insert(0, str(_gpytorch_dir))

from run_single_mode import build_config_from_defaults, load_pnas_data
from kernels import ArcCosineKernel, ArcCosineKernelNormalized
from likelihoods import PoissonLikelihood
from gpy_model import VariationalGPModel
from gpy_training import train_gpy_default
from utils import select_inducing_points_pivoted, compute_rf_center_from_sta
from eigenspace_utils import EIGVAL_TOL, eigendecompose_K_tilde
from tests.test_utils import set_reproducible_seed

# ---------------------------------------------------------------------------
# Investigation parameters (visible, explicit overrides from defaults)
# ---------------------------------------------------------------------------
N_IMAGES = 30        # Number of images for kernel matrix tests
M_INDUCING = 20      # Number of inducing points for integration tests
SEED = 42            # Reproducibility


def load_test_data(config):
    """Load PNAS data and prepare a small batch for testing."""
    data_path = _gpytorch_dir.parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
    dtype = torch.float32
    data = load_pnas_data(data_path, dtype=dtype)

    X = torch.cat([data['X_train'], data['X_val']], dim=0)
    R = torch.cat([data['R_train'], data['R_val']], dim=0)
    X = X.reshape(X.shape[0], -1)  # (N, 11664)

    cellid = config['cell']
    R_cell = R[:, cellid]

    return X, R_cell, data, dtype


def create_kernels(config, dtype, eps_0x, eps_0y):
    """Create both unnormalized and normalized kernels with same parameters."""
    kwargs = dict(
        n_px_side=config['n_px_side'],
        sigma_0=config['sigma_0'],
        Amp=config['Amp'],
        beta=config['beta'],
        rho=config['rho'],
        eps_0x=eps_0x,
        eps_0y=eps_0y,
        use_mask=config['use_mask'],
    )
    k_unnorm = ArcCosineKernel(**kwargs, gradient_mode='autograd').to(dtype=dtype)
    k_norm = ArcCosineKernelNormalized(**kwargs).to(dtype=dtype)
    return k_unnorm, k_norm


# ===== Test Functions =====

def test_normalization_identity(k_unnorm, k_norm, X_batch):
    """K_bar[i,j] == K[i,j] / sqrt(K[i,i] * K[j,j]) for real images."""
    print("\n--- Test: Normalization identity ---")
    with torch.no_grad():
        K = k_unnorm(X_batch, X_batch).evaluate()
        K_bar = k_norm(X_batch, X_batch).evaluate()
        K_diag = k_unnorm(X_batch, diag=True)

    # Manual normalization
    D_inv = 1.0 / torch.sqrt(K_diag)
    K_manual_norm = K * D_inv.unsqueeze(-1) * D_inv.unsqueeze(-2)

    max_err = (K_bar - K_manual_norm).abs().max().item()
    print(f"  Max |K_bar - K/sqrt(K_ii*K_jj)|: {max_err:.2e}")
    passed = max_err < 1e-5
    print(f"  {'PASS' if passed else 'FAIL'} (tol 1e-5)")
    return passed


def test_diagonal_ones(k_norm, X_batch):
    """K_bar(X, diag=True) returns exactly ones."""
    print("\n--- Test: Diagonal = 1.0 ---")
    with torch.no_grad():
        diag = k_norm(X_batch, diag=True)

    max_err = (diag - 1.0).abs().max().item()
    print(f"  Max |diag - 1.0|: {max_err:.2e}")
    passed = max_err < 1e-7
    print(f"  {'PASS' if passed else 'FAIL'} (tol 1e-7)")
    return passed


def test_diagonal_consistency(k_norm, X_batch):
    """diag(K_bar(X, X)) matches K_bar(X, diag=True)."""
    print("\n--- Test: Diagonal consistency (full vs diag path) ---")
    with torch.no_grad():
        K_bar_full = k_norm(X_batch, X_batch).evaluate()
        K_bar_diag = k_norm(X_batch, diag=True)

    full_diag = torch.diagonal(K_bar_full)
    max_err = (full_diag - K_bar_diag).abs().max().item()
    print(f"  Max |diag(K_full) - K_diag|: {max_err:.2e}")
    # The full path uses arccos clamping, so J(eps)/pi ~= 1 - O(eps^2)
    # With eps=1e-7, expect deviation ~1e-14 (negligible)
    passed = max_err < 1e-4
    # Also check full path diagonal is close to 1.0
    max_err_from_one = (full_diag - 1.0).abs().max().item()
    print(f"  Max |diag(K_full) - 1.0|: {max_err_from_one:.2e}")
    passed = passed and max_err_from_one < 1e-4
    print(f"  {'PASS' if passed else 'FAIL'} (tol 1e-4)")
    return passed


def test_symmetry(k_norm, X_batch):
    """K_bar(x1, x2) == K_bar(x2, x1).T"""
    print("\n--- Test: Symmetry ---")
    x1 = X_batch[:10]
    x2 = X_batch[10:20]
    with torch.no_grad():
        K12 = k_norm(x1, x2).evaluate()
        K21 = k_norm(x2, x1).evaluate()
    max_err = (K12 - K21.T).abs().max().item()
    print(f"  Max |K(x1,x2) - K(x2,x1).T|: {max_err:.2e}")
    passed = max_err < 1e-6
    print(f"  {'PASS' if passed else 'FAIL'} (tol 1e-6)")
    return passed


def test_value_range(k_norm, X_batch):
    """All K_bar values in [0, 1]."""
    print("\n--- Test: Value range [0, 1] ---")
    with torch.no_grad():
        K_bar = k_norm(X_batch, X_batch).evaluate()
    min_val = K_bar.min().item()
    max_val = K_bar.max().item()
    print(f"  K_bar range: [{min_val:.6f}, {max_val:.6f}]")
    passed = min_val >= -1e-6 and max_val <= 1.0 + 1e-6
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_positive_semidefinite(k_norm, X_batch):
    """All eigenvalues of K_bar matrix >= -1e-6."""
    print("\n--- Test: Positive semi-definite ---")
    with torch.no_grad():
        K_bar = k_norm(X_batch, X_batch).evaluate()
    eigvals = torch.linalg.eigvalsh(K_bar)
    min_eig = eigvals.min().item()
    max_eig = eigvals.max().item()
    print(f"  Eigenvalue range: [{min_eig:.6e}, {max_eig:.6e}]")
    passed = min_eig >= -1e-6
    print(f"  {'PASS' if passed else 'FAIL'} (min eigenvalue >= -1e-6)")
    return passed


def test_gradient_flow(k_norm_fresh_fn, X_batch):
    """All 6 hyperparams have non-zero gradients from sum(K_bar)."""
    print("\n--- Test: Gradient flow ---")
    k = k_norm_fresh_fn()  # Fresh kernel with grad enabled
    x = X_batch[:5].detach().requires_grad_(False)

    K_bar = k(x, x).evaluate()
    loss = K_bar.sum()
    loss.backward()

    param_names = ['raw_sigma_0', 'raw_Amp', 'raw_m2log2beta',
                   'raw_mlog2rho2', 'eps_0x', 'eps_0y']
    all_ok = True
    for name in param_names:
        param = getattr(k, name)
        if param.grad is None:
            print(f"  {name}: grad is None  FAIL")
            all_ok = False
        elif param.grad.abs().max().item() == 0.0:
            print(f"  {name}: grad is zero  FAIL")
            all_ok = False
        else:
            print(f"  {name}: grad = {param.grad.item():.6e}  OK")
    print(f"  {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test_amp_effect(k_norm, X_batch, config, dtype, eps_0x, eps_0y):
    """K_bar changes when Amp is doubled."""
    print("\n--- Test: Amp effect ---")
    k2 = ArcCosineKernelNormalized(
        n_px_side=config['n_px_side'],
        sigma_0=config['sigma_0'],
        Amp=config['Amp'] * 2.0,
        beta=config['beta'], rho=config['rho'],
        eps_0x=eps_0x, eps_0y=eps_0y,
        use_mask=config['use_mask'],
    ).to(dtype=dtype)
    x = X_batch[:5]
    with torch.no_grad():
        K1 = k_norm(x, x).evaluate()
        K2 = k2(x, x).evaluate()
    diff = (K1 - K2).abs().max().item()
    print(f"  Max |K_bar(Amp) - K_bar(2*Amp)|: {diff:.6e}")
    passed = diff > 1e-6
    print(f"  {'PASS' if passed else 'FAIL'} (should be > 1e-6)")
    return passed


def test_sigma0_effect(k_norm, X_batch, config, dtype, eps_0x, eps_0y):
    """K_bar changes when sigma_0 changes."""
    print("\n--- Test: sigma_0 effect ---")
    k2 = ArcCosineKernelNormalized(
        n_px_side=config['n_px_side'],
        sigma_0=config['sigma_0'] * 3.0,
        Amp=config['Amp'],
        beta=config['beta'], rho=config['rho'],
        eps_0x=eps_0x, eps_0y=eps_0y,
        use_mask=config['use_mask'],
    ).to(dtype=dtype)
    x = X_batch[:5]
    with torch.no_grad():
        K1 = k_norm(x, x).evaluate()
        K2 = k2(x, x).evaluate()
    diff = (K1 - K2).abs().max().item()
    print(f"  Max |K_bar(sigma0) - K_bar(3*sigma0)|: {diff:.6e}")
    passed = diff > 1e-6
    print(f"  {'PASS' if passed else 'FAIL'} (should be > 1e-6)")
    return passed


def test_mask_consistency(k_unnorm, k_norm, X_batch):
    """Same params produce same pixel mask."""
    print("\n--- Test: Mask consistency ---")
    with torch.no_grad():
        # Trigger mask computation
        k_unnorm(X_batch[:2], X_batch[:2])
        k_norm(X_batch[:2], X_batch[:2])
    mask_unnorm = k_unnorm._cached_mask
    mask_norm = k_norm._cached_mask
    if mask_unnorm is None and mask_norm is None:
        print("  Both masks are None (masking disabled)")
        passed = True
    elif mask_unnorm is not None and mask_norm is not None:
        match = torch.equal(mask_unnorm, mask_norm)
        n_pixels = mask_unnorm.sum().item()
        print(f"  Masks equal: {match}, active pixels: {n_pixels}")
        passed = match
    else:
        print(f"  Mismatch: unnorm mask={mask_unnorm is not None}, norm mask={mask_norm is not None}")
        passed = False
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_ktilde_conditioning(k_unnorm, k_norm, X_inducing):
    """Compare condition numbers of K_tilde."""
    print("\n--- Test: K_tilde conditioning ---")
    with torch.no_grad():
        K_tilde_unnorm = k_unnorm(X_inducing, X_inducing).evaluate()
        K_tilde_norm = k_norm(X_inducing, X_inducing).evaluate()

    cond_unnorm = torch.linalg.cond(K_tilde_unnorm).item()
    cond_norm = torch.linalg.cond(K_tilde_norm).item()
    print(f"  Unnormalized cond(K_tilde): {cond_unnorm:.2e}")
    print(f"  Normalized   cond(K_tilde): {cond_norm:.2e}")
    print(f"  Ratio (unnorm/norm): {cond_unnorm / max(cond_norm, 1e-10):.2f}")
    # Normalized should be better conditioned (or at least not catastrophically worse)
    passed = cond_norm < 1e10
    print(f"  {'PASS' if passed else 'FAIL'} (cond < 1e10)")
    return passed


def test_eigenspectrum(k_unnorm, k_norm, X_inducing):
    """Compare eigenvalue spectra and kept eigenvalue count."""
    print("\n--- Test: Eigenspectrum comparison ---")
    with torch.no_grad():
        K_tilde_unnorm = k_unnorm(X_inducing, X_inducing).evaluate()
        K_tilde_norm = k_norm(X_inducing, X_inducing).evaluate()

    eigvals_unnorm = torch.linalg.eigvalsh(K_tilde_unnorm)
    eigvals_norm = torch.linalg.eigvalsh(K_tilde_norm)

    n_kept_unnorm = (eigvals_unnorm > EIGVAL_TOL).sum().item()
    n_kept_norm = (eigvals_norm > EIGVAL_TOL).sum().item()

    print(f"  Unnormalized eigenvalues: [{eigvals_unnorm.min().item():.4e}, {eigvals_unnorm.max().item():.4e}]")
    print(f"  Normalized   eigenvalues: [{eigvals_norm.min().item():.4e}, {eigvals_norm.max().item():.4e}]")
    print(f"  Kept eigenvalues (tol={EIGVAL_TOL}): unnorm={n_kept_unnorm}/{len(eigvals_unnorm)}, norm={n_kept_norm}/{len(eigvals_norm)}")

    # Both should keep a reasonable number of eigenvalues
    passed = n_kept_norm > 0
    print(f"  {'PASS' if passed else 'FAIL'} (norm keeps > 0 eigenvalues)")
    return passed


def test_model_forward_pass(k_norm, X_inducing, X_batch, config):
    """Create VariationalGPModel with normalized kernel, do one forward pass."""
    print("\n--- Test: Model forward pass ---")
    model = VariationalGPModel(
        X_inducing,
        k_norm,
        jitter=config['jitter'],
        standard_variational_distribution=not config['unwhitened_variational_dist']
    )
    model.eval()
    with torch.no_grad():
        output = model(X_batch[:5])
    mean = output.mean
    var = output.variance

    print(f"  Posterior mean range: [{mean.min().item():.4f}, {mean.max().item():.4f}]")
    print(f"  Posterior var range:  [{var.min().item():.4f}, {var.max().item():.4f}]")
    has_nan = torch.isnan(mean).any() or torch.isnan(var).any()
    passed = not has_nan and var.min().item() > 0
    print(f"  {'PASS' if passed else 'FAIL'} (no NaN, positive variance)")
    return passed


def test_single_training_step(X_batch, R_batch, X_inducing, config, dtype):
    """One LBFGS step: verify loss is finite."""
    print("\n--- Test: Single LBFGS step ---")
    from linear_operator import settings as lo_settings

    # Create fresh kernel and model for this test
    k = ArcCosineKernelNormalized(
        n_px_side=config['n_px_side'],
        sigma_0=config['sigma_0'],
        Amp=config['Amp'],
        beta=config['beta'], rho=config['rho'],
        eps_0x=0.0, eps_0y=0.0,
        use_mask=config['use_mask'],
    ).to(dtype=dtype)

    model = VariationalGPModel(
        X_inducing,
        k,
        jitter=config['jitter'],
        standard_variational_distribution=not config['unwhitened_variational_dist']
    )
    likelihood = PoissonLikelihood(
        A_init=config['A_init'],
        lambda0_init=config['lambda0_init']
    ).to(dtype=dtype)

    model.train()
    likelihood.train()

    all_params = list(model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.LBFGS(all_params, lr=0.1, max_iter=5, line_search_fn='strong_wolfe')

    jitter = config['jitter']
    with torch.enable_grad(), \
         lo_settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
         lo_settings.cholesky_max_tries(config['cholesky_max_tries']):

        def closure():
            optimizer.zero_grad()
            output = model(X_batch)
            ell = likelihood.expected_log_prob(R_batch, output)
            kl = model.variational_strategy.kl_divergence()
            loss = -ell + kl
            loss.backward()
            return loss

        loss_before = closure().item()
        optimizer.step(closure)
        loss_after = closure().item()

    print(f"  Loss before: {loss_before:.4f}")
    print(f"  Loss after:  {loss_after:.4f}")
    finite = np.isfinite(loss_before) and np.isfinite(loss_after)
    passed = finite
    print(f"  {'PASS' if passed else 'FAIL'} (finite loss)")
    return passed


def test_identical_images_full_path(k_norm, X_batch):
    """K_bar(x, x) via full matrix path should be ~1.0 despite arccos clamping."""
    print("\n--- Test: Identical images via full matrix path ---")
    x = X_batch[:5]
    with torch.no_grad():
        K_full = k_norm(x, x).evaluate()
    full_diag = torch.diagonal(K_full)
    max_err = (full_diag - 1.0).abs().max().item()
    print(f"  Full-path diagonal values: {full_diag.tolist()}")
    print(f"  Max |diag - 1.0|: {max_err:.2e}")
    # arccos clamping at 1-1e-7 means cos_theta = 1-1e-7, theta ~ 4.5e-4
    # J(theta)/pi = (sin(theta) + (pi-theta)*cos(theta))/pi ~ 1 - theta^2/(2*pi)
    # ~ 1 - 1e-7/(2*pi) ~ 1 - 1.6e-8
    passed = max_err < 1e-4
    print(f"  {'PASS' if passed else 'FAIL'} (tol 1e-4)")
    return passed


def test_zero_image(k_norm, config, dtype):
    """K_bar for zero image (V1 = sigma_0^2 only)."""
    print("\n--- Test: Zero image ---")
    n_px = config['n_px_side'] ** 2
    x_zero = torch.zeros(1, n_px, dtype=dtype)
    x_other = torch.randn(1, n_px, dtype=dtype) * 0.01

    with torch.no_grad():
        K_diag = k_norm(x_zero, diag=True)
        K_cross = k_norm(x_zero, x_other).evaluate()

    print(f"  K_bar(zero, zero) diag: {K_diag.item():.6f}")
    print(f"  K_bar(zero, other): {K_cross.item():.6f}")
    has_nan = torch.isnan(K_diag).any() or torch.isnan(K_cross).any()
    diag_ok = abs(K_diag.item() - 1.0) < 1e-6
    range_ok = 0 <= K_cross.item() <= 1.0
    passed = not has_nan and diag_ok and range_ok
    print(f"  {'PASS' if passed else 'FAIL'} (no NaN, diag=1, cross in [0,1])")
    return passed


def test_pivoted_cholesky(k_norm, X_batch):
    """Pivoted inducing point selection with normalized K_tilde."""
    print("\n--- Test: Pivoted Cholesky selection ---")
    try:
        _, indices = select_inducing_points_pivoted(
            X_batch, k_norm, n_inducing=M_INDUCING, seed=SEED
        )
        n_unique = len(set(indices.tolist()))
        print(f"  Selected {len(indices)} inducing points, {n_unique} unique")
        passed = n_unique == len(indices)
        if not passed:
            print(f"  WARNING: Duplicate inducing points!")
        print(f"  {'PASS' if passed else 'FAIL'}")
    except Exception as e:
        print(f"  Exception: {e}")
        passed = False
        print(f"  FAIL (crashed)")
    return passed


# ===== Main =====

def main():
    print("=" * 70)
    print("Validation Suite: ArcCosineKernelNormalized")
    print("=" * 70)

    set_reproducible_seed(SEED)

    # Build config from defaults
    config = build_config_from_defaults(mode='default_gpy')

    # Load data
    X, R_cell, data, dtype = load_test_data(config)
    print(f"Loaded data: X shape {X.shape}, cell {config['cell']}")

    # Compute STA for RF center
    n_samples_sta = min(200, X.shape[0])
    eps_0x, eps_0y = compute_rf_center_from_sta(
        X[:n_samples_sta], R_cell[:n_samples_sta], config['n_px_side'], zscore=True
    )
    print(f"STA RF center: ({eps_0x:.3f}, {eps_0y:.3f})")

    # Create kernels
    k_unnorm, k_norm = create_kernels(config, dtype, eps_0x, eps_0y)

    # Select test images
    X_batch = X[:N_IMAGES]
    R_batch = R_cell[:N_IMAGES]
    X_inducing = X[:M_INDUCING]

    # Run tests
    results = {}

    # Group 1: Mathematical correctness
    print("\n" + "=" * 50)
    print("GROUP 1: Mathematical Correctness")
    print("=" * 50)
    results['normalization_identity'] = test_normalization_identity(k_unnorm, k_norm, X_batch)
    results['diagonal_ones'] = test_diagonal_ones(k_norm, X_batch)
    results['diagonal_consistency'] = test_diagonal_consistency(k_norm, X_batch)
    results['symmetry'] = test_symmetry(k_norm, X_batch)
    results['value_range'] = test_value_range(k_norm, X_batch)
    results['psd'] = test_positive_semidefinite(k_norm, X_batch)

    # Group 2: Parameter sensitivity
    print("\n" + "=" * 50)
    print("GROUP 2: Parameter Sensitivity")
    print("=" * 50)

    def make_fresh_kernel():
        return ArcCosineKernelNormalized(
            n_px_side=config['n_px_side'],
            sigma_0=config['sigma_0'],
            Amp=config['Amp'],
            beta=config['beta'], rho=config['rho'],
            eps_0x=eps_0x, eps_0y=eps_0y,
            use_mask=config['use_mask'],
        ).to(dtype=dtype)

    results['gradient_flow'] = test_gradient_flow(make_fresh_kernel, X_batch)
    results['amp_effect'] = test_amp_effect(k_norm, X_batch, config, dtype, eps_0x, eps_0y)
    results['sigma0_effect'] = test_sigma0_effect(k_norm, X_batch, config, dtype, eps_0x, eps_0y)
    results['mask_consistency'] = test_mask_consistency(k_unnorm, k_norm, X_batch)

    # Group 3: Integration with GP pipeline
    print("\n" + "=" * 50)
    print("GROUP 3: GP Pipeline Integration")
    print("=" * 50)
    results['ktilde_conditioning'] = test_ktilde_conditioning(k_unnorm, k_norm, X_inducing)
    results['eigenspectrum'] = test_eigenspectrum(k_unnorm, k_norm, X_inducing)
    results['model_forward'] = test_model_forward_pass(k_norm, X_inducing, X_batch, config)
    results['single_step'] = test_single_training_step(X_batch, R_batch, X_inducing, config, dtype)

    # Group 4: Edge cases
    print("\n" + "=" * 50)
    print("GROUP 4: Edge Cases")
    print("=" * 50)
    results['identical_full_path'] = test_identical_images_full_path(k_norm, X_batch)
    results['zero_image'] = test_zero_image(k_norm, config, dtype)
    results['pivoted_cholesky'] = test_pivoted_cholesky(k_norm, X_batch)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_pass = sum(v for v in results.values())
    n_total = len(results)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
    print(f"\n{n_pass}/{n_total} tests passed")

    if n_pass < n_total:
        print("\nFAILED TESTS:")
        for name, passed in results.items():
            if not passed:
                print(f"  - {name}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
