"""Phase 3: NGD prototype / sweep.

SUPERSEDED (Phase 3D, 2026-04-22): the investigation monkey-patch is
no longer necessary — `mode='ngd'` is a first-class training mode in
`run_single_mode.py`. For new sweeps use `build_config_from_defaults(mode='ngd', ...)`
+ `run_single_config(config)` directly (the pattern in step3_beta_sweep.py
but with `mode='ngd'`). This file is preserved as the Phase 3
investigation artifact; do not use it in new code.


Usage:
    python run_ngd_sweep.py --prototype     # 5 cells x 1 seed (seed 42)
    python run_ngd_sweep.py --full          # 16 cells x 3 seeds

Strategy:
    For each (cell, seed):
      1. Monkey-patch gpy_training.train_gpy_default to a capture-and-no-op
         that stashes (model, likelihood, X_train, r_train). This reuses the
         production data/IP/RF/kernel/likelihood setup EXACTLY.
      2. Run run_single_config(config) — does setup, fake-trains, returns
         garbage metrics (discarded).
      3. Extract kernel + inducing_points + likelihood from captured model.
      4. Build a fresh NGDVariationalGPModel with TrilNaturalVariationalDistribution
         around the same kernel/inducing_points.
      5. Train via NGD+Adam (no ES, full-batch, 200 iters).
      6. Load X_test/R_test directly from the .npz and compute test_r.
      7. Append one JSON line to ngd_results.jsonl.

Output: investigations/default_gpy_gap_v2/ngd/ngd_results.jsonl

All hyperparameters match the Phase 2 main sweep defaults (see SCRAPBOOK.md
§3.2) except the optimizer — this is the apples-to-apples comparison needed
for the paired-Δ analysis.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# --- Path setup ---
NGD_DIR = Path(__file__).parent
INVESTIGATION_DIR = NGD_DIR.parent
ROOT = INVESTIGATION_DIR.parent.parent  # .../gpytorch_porting
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NGD_DIR))

# --- Production imports (no edits to these files) ---
import gpy_training as _gpy_training  # module-level ref for monkey-patch
from run_single_mode import build_config_from_defaults, run_single_config
import run_single_mode as _rsm  # for monkey-patch of its imported symbol
from gpy_training import predict
from metrics import (
    compute_pearson_correlation,
    compute_explained_variance,
    compute_adjusted_r_squared,
)

# --- Investigation-local imports ---
from ngd_model import NGDVariationalGPModel
from ngd_training import train_ngd


# --- Fixed charter config (matches Phase 2 main sweep §3.2) ---
DATA_PATH = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
M = 300
N_TRAIN = 1500
N_ITERATIONS = 1000  # raised from 200 after prototype showed loss still falling
# at iter 200 on all 5 cells (slopes -0.3 to -1.9/iter). NGD+Adam is first-order
# so it needs substantially more steps than LBFGS's 50-outer x ~20-inner budget.

# ELBO-based early stopping parameters. Tuned via simulate_es.py on the 48
# saved full-iter trajectories. Relative to vargp_direct's (patience=15,
# min_delta_rel=1e-3) defaults from `default_params.json`, these are scaled
# by ~10x on both axes to account for NGD's finer per-iter step size (one
# NGD+Adam step ≈ 1 grad eval; one vargp LBFGS outer iter ≈ ~7 inner grad
# evals per strong-Wolfe step). Chosen so ES triggers in the loss plateau
# region for ≥75% of (cell, seed) pairs without reducing final test_r.
# Expected ~37% wall-clock reduction vs the no-ES 1000-iter baseline, with
# mean Δ test_r ~ 0.
NGD_ES_PATIENCE = 200
NGD_ES_MIN_DELTA_REL = 1e-2
NGD_ES_MIN_ITERATIONS = 50
NGD_ES_RESTORE_BEST = True
NGD_LR = 0.1
ADAM_LR = 0.01
CELLS_USED_PATH = INVESTIGATION_DIR / 'cells_used.json'
NGD_RESULTS_PATH = NGD_DIR / 'ngd_results.jsonl'

PROTOTYPE_CELLS = [40, 38, 16, 30, 29]
PROTOTYPE_SEEDS = [42]

FULL_SEEDS = [42, 123, 789]


# ------------------------------------------------------------------
# Monkey-patch scaffolding: capture the setup, skip the training.
# ------------------------------------------------------------------

_captured = {}


def _capture_and_no_train(model, likelihood, X_train, r_train, **kwargs):
    """Replacement for gpy_training.train_gpy_default.

    Captures the model/likelihood/data built by run_single_config's setup
    code without performing any optimization. Returns a result dict with
    the minimum keys required by run_single_config's downstream code.
    """
    _captured['model'] = model
    _captured['likelihood'] = likelihood
    _captured['X_train'] = X_train
    _captured['r_train'] = r_train
    _captured['kwargs'] = kwargs
    return {
        'losses': [float('nan')],
        'stopped_early': True,
        'final_iteration': 0,
        'best_iteration': 0,
        'curves': {},
    }


def _install_monkey_patch():
    """Replace train_gpy_default in both the module and its importer."""
    _gpy_training.train_gpy_default = _capture_and_no_train
    _rsm.train_gpy_default = _capture_and_no_train


def _restore_monkey_patch(original):
    _gpy_training.train_gpy_default = original
    _rsm.train_gpy_default = original


# ------------------------------------------------------------------
# Test-data loading (deterministic, seed-independent).
# ------------------------------------------------------------------

def load_test_tensors(cell, device, dtype=torch.float32):
    data_path = ROOT / DATA_PATH
    data = np.load(data_path)
    X_test = torch.tensor(data['images_test'], dtype=dtype)
    R_test = torch.tensor(data['responses_test'], dtype=dtype)
    X_test = X_test.reshape(X_test.shape[0], -1).to(device)
    R_test = R_test.to(device)
    r_test = R_test[:, :, cell]  # (30 repeats, 30 images)
    return X_test, r_test


# ------------------------------------------------------------------
# Single-run.
# ------------------------------------------------------------------

def run_one(cell, seed, n_iterations=N_ITERATIONS, ngd_lr=NGD_LR, adam_lr=ADAM_LR,
            early_stop=True, patience=NGD_ES_PATIENCE,
            min_delta_rel=NGD_ES_MIN_DELTA_REL,
            min_iterations=NGD_ES_MIN_ITERATIONS,
            restore_best=NGD_ES_RESTORE_BEST,
            probe_test_r=True):
    """Run a single NGD fit on (cell, seed). Returns a dict (one JSONL line)."""
    os.chdir(ROOT)
    # Explicit GPU cleanup before each run: the sweep reuses this process
    # for all (cell, seed) pairs, so between-run reference leaks would
    # accumulate. Clearing the captured dict + empty_cache releases the
    # previous run's model/kernel/likelihood tensors. This fixes the OOM
    # observed on the 2nd run of the first 1000-iter sweep (see SCRAPBOOK
    # Phase 3 Finding P3-2).
    _captured.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 1. Build config exactly as the Phase 2 main sweep did.
    config = build_config_from_defaults(
        mode='default_gpy',
        data_path=DATA_PATH,
        M=M, n_train=N_TRAIN, cell=cell, seed=seed,
        ip_selection='random',
        # All other params (beta=0.1, rho=0.1, A_init=0.01, lambda0=1.0,
        # rf_init='ground_truth', ES on, n_iterations=50) come from defaults.
    )
    jitter_val = config['jitter']
    cholesky_max_tries = config['cholesky_max_tries']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Monkey-patch + run setup via run_single_config.
    original = _gpy_training.train_gpy_default
    _captured.clear()
    _install_monkey_patch()
    t_setup_start = time.time()
    try:
        _ = run_single_config(config)  # setup runs; fake-training returns garbage
    finally:
        _restore_monkey_patch(original)
    t_setup = time.time() - t_setup_start

    # 3. Extract setup artefacts.
    model_orig = _captured['model']
    likelihood = _captured['likelihood']
    X_train = _captured['X_train']
    r_train = _captured['r_train']

    kernel = model_orig.covar_module  # already initialised (beta=0.1 etc)
    # inducing_points is registered as a buffer/parameter of the variational
    # strategy; grab a clean copy so the fresh NGD model owns its own tensor.
    inducing_points = model_orig.variational_strategy.inducing_points.detach().clone()

    # 4. Build fresh NGD model around the extracted kernel and IPs.
    ngd_model = NGDVariationalGPModel(
        inducing_points=inducing_points,
        kernel=kernel,
        jitter=jitter_val,
        learn_inducing_locations=False,
    )
    ngd_model = ngd_model.to(device=device, dtype=X_train.dtype)
    likelihood = likelihood.to(device=device, dtype=X_train.dtype)

    # 5. Train via NGD+Adam.
    print(f"\n=== cell={cell} seed={seed} — NGD+Adam, {n_iterations} iters ===", flush=True)

    # Optional test_r probe for overfitting diagnosis. Never used as ES signal.
    probe_fn = None
    if probe_test_r:
        X_test_probe, r_test_probe = load_test_tensors(cell, device=device, dtype=X_train.dtype)
        r_test_mean_probe = r_test_probe.mean(dim=0)
        def probe_fn(model, likelihood):  # noqa: E306
            preds = predict(
                model, likelihood, X_test_probe, device=device,
                jitter=jitter_val, cholesky_max_tries=cholesky_max_tries,
            )
            return compute_pearson_correlation(r_test_mean_probe, preds['f_pred'])

    t_train_start = time.time()
    train_result = train_ngd(
        ngd_model, likelihood, X_train, r_train,
        n_iterations=n_iterations,
        ngd_lr=ngd_lr,
        adam_lr=adam_lr,
        jitter=jitter_val,
        cholesky_max_tries=cholesky_max_tries,
        device=device,
        print_every=50,
        early_stop=early_stop,
        patience=patience,
        min_delta_rel=min_delta_rel,
        min_iterations=min_iterations,
        restore_best=restore_best,
        test_r_probe=probe_fn,
        test_r_every=25,
    )
    t_train = time.time() - t_train_start

    # 6. Test-time metrics.
    X_test, r_test = load_test_tensors(cell, device=device, dtype=X_train.dtype)
    try:
        predictions = predict(
            ngd_model, likelihood, X_test, device=device,
            jitter=jitter_val, cholesky_max_tries=cholesky_max_tries,
        )
        f_pred = predictions['f_pred']
        r_test_mean = r_test.mean(dim=0)
        test_r = compute_pearson_correlation(r_test_mean, f_pred)
        explained_var, reliability = compute_explained_variance(r_test, f_pred)
        adjusted_r2 = compute_adjusted_r_squared(r_test, f_pred)

        train_pred_out = predict(
            ngd_model, likelihood, X_train, device=device,
            jitter=jitter_val, cholesky_max_tries=cholesky_max_tries,
        )
        train_r = compute_pearson_correlation(r_train, train_pred_out['f_pred'])
        predict_ok = True
        predict_error = None
    except RuntimeError as e:
        print(f"  Prediction failed: {e}", flush=True)
        test_r = float('nan')
        train_r = float('nan')
        explained_var = float('nan')
        adjusted_r2 = float('nan')
        reliability = float('nan')
        predict_ok = False
        predict_error = repr(e)

    # 7. Record.
    final_A = float(likelihood.A.item())
    final_lambda0 = float(likelihood.lambda0.item())
    final_beta = float(kernel.beta.item()) if hasattr(kernel, 'beta') else None
    final_rho = float(kernel.rho.item()) if hasattr(kernel, 'rho') else None
    final_eps_0x = float(kernel.eps_0x.item()) if hasattr(kernel, 'eps_0x') else None
    final_eps_0y = float(kernel.eps_0y.item()) if hasattr(kernel, 'eps_0y') else None
    final_loss = train_result['losses'][-1] if train_result['losses'] else None

    # Compute "oracle ES" test_r: best test_r over the probe trajectory (not
    # actually usable as ES because it looks at test info, but useful to
    # quantify the upper bound NGD could reach with validation ES).
    probe_iters = train_result['curves'].get('test_r_iter', [])
    probe_vals = train_result['curves'].get('test_r', [])
    if probe_vals:
        best_i, best_tr = max(enumerate(probe_vals), key=lambda kv: kv[1])
        best_test_r_probe = float(best_tr)
        best_test_r_iter = int(probe_iters[best_i])
    else:
        best_test_r_probe = None
        best_test_r_iter = None

    record = {
        'cell': cell,
        'seed': seed,
        'mode': 'ngd',
        'M': M, 'n_train': N_TRAIN,
        'n_iterations_requested': n_iterations,
        'n_iterations_run': train_result['final_iteration'],
        'stopped_early': train_result['stopped_early'],
        'best_iteration': train_result.get('best_iteration'),
        'diverged': train_result['diverged'],
        'diverged_at': train_result['diverged_at'],
        'best_test_r_probe': best_test_r_probe,    # oracle-ES upper bound
        'best_test_r_iter_probe': best_test_r_iter,
        'ngd_lr': ngd_lr,
        'adam_lr': adam_lr,
        'test_r': float(test_r) if test_r == test_r else None,  # NaN -> None
        'train_r': float(train_r) if train_r == train_r else None,
        'explained_var': float(explained_var) if explained_var == explained_var else None,
        'adjusted_r2': float(adjusted_r2) if adjusted_r2 == adjusted_r2 else None,
        'reliability': float(reliability) if reliability == reliability else None,
        'predict_ok': predict_ok,
        'predict_error': predict_error,
        'final_loss': final_loss,
        'final_A': final_A,
        'final_lambda0': final_lambda0,
        'final_beta': final_beta,
        'final_rho': final_rho,
        'final_eps_0x': final_eps_0x,
        'final_eps_0y': final_eps_0y,
        'setup_time_s': t_setup,
        'train_time_s': t_train,
        's_per_iter': t_train / max(train_result['final_iteration'], 1),
        'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'curves': train_result['curves'],
    }
    print(
        f"  -> test_r={record['test_r']}  final_A={final_A:.4g}  "
        f"final_beta={final_beta if final_beta is None else f'{final_beta:.4g}'}  "
        f"final_loss={final_loss if final_loss is None else f'{final_loss:.3f}'}  "
        f"train_time={t_train:.1f}s",
        flush=True,
    )
    return record


# ------------------------------------------------------------------
# Sweep drivers.
# ------------------------------------------------------------------

def load_cells_full():
    with open(CELLS_USED_PATH) as f:
        return [c['cell_id'] for c in json.load(f)['cells']]


def sweep(cells, seeds, output_path=NGD_RESULTS_PATH):
    """Run a sweep, writing one JSONL record per run to output_path."""
    # Rename any existing results file (never overwrite silently).
    if output_path.exists():
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = output_path.with_suffix(f'.backup_{ts}.jsonl')
        output_path.rename(backup)
        print(f"[renamed existing results file to {backup.name}]", flush=True)

    total = len(cells) * len(seeds)
    idx = 0
    wall_start = time.time()
    for cell in cells:
        for seed in seeds:
            idx += 1
            print(f"\n[{idx}/{total}] ============================================", flush=True)
            try:
                rec = run_one(cell, seed)
            except Exception as e:
                import traceback
                traceback.print_exc()
                rec = {
                    'cell': cell,
                    'seed': seed,
                    'mode': 'ngd',
                    'status': 'crashed',
                    'error': repr(e),
                    'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
                }
            with open(output_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
    wall_elapsed = time.time() - wall_start
    print(f"\n=== Sweep done. {idx}/{total} runs in {wall_elapsed / 60:.1f} min. ===", flush=True)
    print(f"Results: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prototype', action='store_true',
                        help=f"5 cells x 1 seed (cells={PROTOTYPE_CELLS}, seed=42)")
    parser.add_argument('--full', action='store_true',
                        help=f"16 cells x 3 seeds ({FULL_SEEDS})")
    parser.add_argument('--cells', type=int, nargs='+', default=None,
                        help="Custom cell list (overrides --prototype/--full)")
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help="Custom seed list (overrides --prototype/--full)")
    parser.add_argument('--output', type=str, default=str(NGD_RESULTS_PATH),
                        help="Output JSONL path")
    args = parser.parse_args()

    if args.cells is not None or args.seeds is not None:
        cells = args.cells if args.cells is not None else PROTOTYPE_CELLS
        seeds = args.seeds if args.seeds is not None else PROTOTYPE_SEEDS
    elif args.full:
        cells = load_cells_full()
        seeds = FULL_SEEDS
    else:
        cells = PROTOTYPE_CELLS
        seeds = PROTOTYPE_SEEDS

    out = Path(args.output)
    print(f"Running NGD sweep: {len(cells)} cells x {len(seeds)} seeds "
          f"= {len(cells) * len(seeds)} runs")
    print(f"  cells: {cells}")
    print(f"  seeds: {seeds}")
    print(f"  output: {out}")
    sweep(cells, seeds, output_path=out)


if __name__ == '__main__':
    main()
