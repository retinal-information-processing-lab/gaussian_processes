#!/usr/bin/env python3
"""
Strict tests for validation-based early stopping implementation.

Tests use real PNAS data, not synthetic. Each test has hard assertions that
raise errors on failure. No "check visually" or soft warnings.

Usage:
    python tests/test_early_stopping.py

All tests use cell 8, seed 42, M=50, vargp_direct mode for speed.
Total runtime target: < 2 minutes on GPU.
"""

import sys
import math
import time
from pathlib import Path

# Add project root to path
PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ))

import torch
import numpy as np

from run_single_mode import build_config_from_defaults, run_single_config
from eigenspace_training import _compute_val_log_lik, predict_eigenspace
from tests.test_utils import set_reproducible_seed


# ===== Shared config builder =====
def _make_config(**overrides):
    """Build a vargp_direct config with small M for fast testing."""
    config = build_config_from_defaults(
        mode='vargp_direct',
        M=50,
        n_train=500,
        seed=42,
        cell=8,
        n_iterations=20,
        ip_selection='random',
    )
    config.update(overrides)
    return config


# ===== Test functions =====

def test_1_data_flow_integrity():
    """Validation data never leaks into training set."""
    print("Test 1: Data flow integrity...", end=' ', flush=True)

    config = _make_config(early_stop=False)
    result = run_single_config(config)

    assert result is not None, "run_single_config returned None"
    # n_train should be <= 2910 (no val data in training)
    assert result['n_train'] <= 2910, (
        f"n_train={result['n_train']} > 2910: validation data leaked into training")

    # With n_train=3160, effective should be capped to 2910
    config2 = _make_config(n_train=3160, early_stop=False)
    result2 = run_single_config(config2)
    assert result2['n_train'] == 2910, (
        f"n_train={result2['n_train']} != 2910 when requesting 3160 (val not held out)")

    print("PASS")


def test_2_val_log_lik_formula():
    """_compute_val_log_lik returns the correct expected log-likelihood."""
    print("Test 2: Validation log-likelihood formula...", end=' ', flush=True)

    config = _make_config(n_iterations=6, early_stop=False)
    result = run_single_config(config)
    model = result['_model']
    likelihood = result['_likelihood']

    # Load validation data the same way run_single_config does
    data_path = Path(config['data_path'])
    if not data_path.is_absolute():
        data_path = PROJ / data_path
    data = np.load(data_path)
    dtype = torch.float32
    device = torch.device(config['device'])
    X_val = torch.tensor(data['images_val'], dtype=dtype).reshape(
        data['images_val'].shape[0], -1).to(device)
    R_val = torch.tensor(data['responses_val'], dtype=dtype).to(device)
    r_val = R_val[:, config['cell']]

    # Compute using the function
    val_ll_func = _compute_val_log_lik(model, X_val, r_val)

    # Compute manually
    with torch.no_grad():
        preds = predict_eigenspace(model, X_val)
        A = model.likelihood.A.squeeze()
        lambda0 = model.likelihood.lambda0.squeeze()
        mu = preds['lambda_m']
        f_mean = preds['f_pred']
        val_ll_manual = (r_val * (A * mu + lambda0) - f_mean).sum().item()

    rel_err = abs(val_ll_func - val_ll_manual) / max(abs(val_ll_manual), 1e-8)
    assert rel_err < 1e-5, (
        f"val_ll_func={val_ll_func:.6f} != val_ll_manual={val_ll_manual:.6f}, "
        f"rel_err={rel_err:.2e}")

    print(f"PASS (rel_err={rel_err:.2e})")


def test_3_elbo_decomposition():
    """train_log_lik - train_kl == -train_loss at every iteration."""
    print("Test 3: ELBO decomposition consistency...", end=' ', flush=True)

    config = _make_config(n_iterations=10, early_stop=False)
    result = run_single_config(config)
    curves = result['curves']

    for i in range(len(curves['train_loss'])):
        train_ll = curves['train_log_lik'][i]
        train_kl = curves['train_kl'][i]
        train_loss = curves['train_loss'][i]
        # ELBO = log_lik - KL, loss = -ELBO = KL - log_lik
        # So train_ll - train_kl should equal -train_loss
        reconstructed = train_ll - train_kl
        expected = -train_loss
        abs_err = abs(reconstructed - expected)
        assert abs_err < 1e-2, (
            f"Iter {i}: train_ll({train_ll:.4f}) - train_kl({train_kl:.4f}) = "
            f"{reconstructed:.4f} != -{train_loss:.4f} = {expected:.4f}, "
            f"abs_err={abs_err:.4f}")

    print(f"PASS ({len(curves['train_loss'])} iterations checked)")


def test_4_patience_fires_correctly():
    """With patience=3, stopping happens 3 iters after best validation."""
    print("Test 4: Patience fires at correct iteration...", end=' ', flush=True)

    config = _make_config(
        n_iterations=50,
        early_stop=True,
        patience=3,
        min_delta_rel=0.001,
        min_iterations=5,
        restore_best=False,  # Don't restore, so we can check timing cleanly
    )
    result = run_single_config(config)

    assert result['stopped_early'], "Expected early stopping but it didn't trigger"
    best_iter = result['best_iteration']
    final_iter = result['n_iterations_run']

    # Final iteration should be best_iteration + patience
    # (unless min_iterations pushed it later)
    expected_stop = max(best_iter + 3, config['min_iterations'])
    assert final_iter == expected_stop, (
        f"final_iter={final_iter} != expected {expected_stop} "
        f"(best_iter={best_iter}, patience=3, min_iter={config['min_iterations']})")

    print(f"PASS (best={best_iter}, stopped={final_iter})")


def test_5_min_iterations_hard_floor():
    """Early stopping NEVER triggers before min_iterations."""
    print("Test 5: min_iterations is a hard floor...", end=' ', flush=True)

    config = _make_config(
        n_iterations=50,
        early_stop=True,
        patience=1,
        min_delta_rel=0.001,
        min_iterations=20,
    )
    result = run_single_config(config)

    assert result['n_iterations_run'] >= 20, (
        f"Stopped at iteration {result['n_iterations_run']} < min_iterations=20")

    print(f"PASS (stopped at iter {result['n_iterations_run']})")


def test_6_min_delta_rel_filters():
    """Absurdly high min_delta_rel means no improvement ever counts."""
    print("Test 6: min_delta_rel filters tiny improvements...", end=' ', flush=True)

    config = _make_config(
        n_iterations=50,
        early_stop=True,
        patience=3,
        min_delta_rel=0.5,  # 50% improvement required -- impossible
        min_iterations=5,
    )
    result = run_single_config(config)

    assert result['stopped_early'], "Expected early stopping with impossible delta"
    # Iteration 1 always sets baseline (first observation). After that, no improvement
    # ever exceeds 50%, so patience fills up. Stop = max(1 + patience, min_iterations).
    expected_stop = max(1 + 3, 5)  # max(best_iter + patience, min_iterations)
    assert result['n_iterations_run'] == expected_stop, (
        f"Stopped at {result['n_iterations_run']} != expected {expected_stop} "
        f"(best_iter=1, patience=3, min_iter=5)")

    print(f"PASS (stopped at iter {result['n_iterations_run']})")


def test_7_restore_best_matches():
    """After restore, model params match curves at best_iteration."""
    print("Test 7: Restore-best model state matches...", end=' ', flush=True)

    config = _make_config(
        n_iterations=50,
        early_stop=True,
        patience=3,
        min_delta_rel=0.001,
        min_iterations=5,
        restore_best=True,
    )
    result = run_single_config(config)
    model = result['_model']
    curves = result['curves']
    best_iter = result['best_iteration']

    assert result['stopped_early'], "Expected early stopping to trigger"
    assert best_iter > 0, f"best_iteration={best_iter} should be > 0"

    # Curves are 0-indexed, iterations are 1-indexed
    idx = best_iter - 1

    checks = {
        'A': model.likelihood.A.item(),
        'lambda0': model.likelihood.lambda0.item(),
        'beta': model.kernel.beta.item(),
        'rho': model.kernel.rho.item(),
        'sigma_0': model.kernel.sigma_0.item(),
        'eps_0x': model.kernel.eps_0x.item(),
        'eps_0y': model.kernel.eps_0y.item(),
        'Amp': model.kernel.Amp.item(),
    }

    for name, current_val in checks.items():
        saved_val = curves[name][idx]
        assert current_val == saved_val, (
            f"Parameter '{name}' after restore: {current_val} != "
            f"curves['{name}'][{idx}]={saved_val} (best_iter={best_iter})")

    print(f"PASS (8 params match at best_iter={best_iter})")


def test_8_curve_lengths():
    """All curve arrays have exactly final_iteration entries."""
    print("Test 8: Curve lengths are consistent...", end=' ', flush=True)

    config = _make_config(n_iterations=15, early_stop=False)
    result = run_single_config(config)
    curves = result['curves']
    final_iter = result['n_iterations_run']

    for key, vals in curves.items():
        assert len(vals) == final_iter, (
            f"curves['{key}'] has length {len(vals)} != final_iteration={final_iter}")
        # Check no NaN/inf (None is OK for val_log_lik when no val data)
        for i, v in enumerate(vals):
            if v is not None:
                assert not math.isnan(v) and not math.isinf(v), (
                    f"curves['{key}'][{i}] = {v} (NaN or inf)")

    print(f"PASS ({len(curves)} curves, {final_iter} entries each)")


def test_9_early_stop_false_still_logs():
    """Disabling early stopping runs full iterations but still produces curves."""
    print("Test 9: early_stop=False still logs curves...", end=' ', flush=True)

    config = _make_config(n_iterations=12, early_stop=False)
    result = run_single_config(config)

    # Loop runs range(1, n_iterations), so final_iteration = n_iterations - 1
    assert result['n_iterations_run'] == 11, (
        f"final_iteration={result['n_iterations_run']} != 11 (n_iterations=12)")
    assert not result['stopped_early'], "stopped_early should be False"
    assert 'curves' in result and result['curves'], "curves dict should exist"

    curves = result['curves']
    for key, vals in curves.items():
        assert len(vals) == 11, (
            f"curves['{key}'] length {len(vals)} != 11")

    # best_iteration should be the iter with max val_log_lik
    val_lls = curves['val_log_lik']
    if val_lls[0] is not None:  # has validation data
        max_idx = max(range(len(val_lls)), key=lambda i: val_lls[i])
        expected_best = max_idx + 1  # 1-indexed
        assert result['best_iteration'] == expected_best, (
            f"best_iteration={result['best_iteration']} != {expected_best} "
            f"(iter with max val_ll)")

    print(f"PASS (best_iter={result['best_iteration']})")


def test_10_reproducibility():
    """Two runs with identical config produce identical curves."""
    print("Test 10: Reproducibility...", end=' ', flush=True)

    config = _make_config(n_iterations=10, early_stop=False)

    result1 = run_single_config(config.copy())
    result2 = run_single_config(config.copy())

    curves1 = result1['curves']
    curves2 = result2['curves']

    assert result1['n_iterations_run'] == result2['n_iterations_run'], (
        f"Iteration counts differ: {result1['n_iterations_run']} vs {result2['n_iterations_run']}")

    for key in ['train_loss', 'val_log_lik', 'A', 'lambda0']:
        vals1 = curves1[key]
        vals2 = curves2[key]
        for i in range(len(vals1)):
            if vals1[i] is not None and vals2[i] is not None:
                assert vals1[i] == vals2[i], (
                    f"curves['{key}'][{i}]: {vals1[i]} != {vals2[i]} "
                    f"(non-deterministic)")

    assert result1['best_iteration'] == result2['best_iteration'], (
        f"best_iteration differs: {result1['best_iteration']} vs {result2['best_iteration']}")

    print("PASS")


# ===== Main =====
if __name__ == '__main__':
    tests = [
        test_1_data_flow_integrity,
        test_2_val_log_lik_formula,
        test_3_elbo_decomposition,
        test_4_patience_fires_correctly,
        test_5_min_iterations_hard_floor,
        test_6_min_delta_rel_filters,
        test_7_restore_best_matches,
        test_8_curve_lengths,
        test_9_early_stop_false_still_logs,
        test_10_reproducibility,
    ]

    print(f"\n{'='*60}")
    print("Early Stopping Tests (10 tests, real PNAS data)")
    print(f"{'='*60}\n")

    start = time.time()
    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"FAIL: {e}")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed ({elapsed:.1f}s)")
    if errors:
        print("\nFailures:")
        for name, msg in errors:
            print(f"  {name}: {msg}")
    print(f"{'='*60}")

    sys.exit(1 if failed > 0 else 0)
