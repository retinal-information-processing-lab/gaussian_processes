#!/usr/bin/env python3
"""
Confirmation run: verify test_r is preserved with abs_tol=1e-7.

Same grid as tolerance sweep (10 cells x 3 seeds = 30 runs) but with
lbfgs_tolerance_change_rel=1e-7. Compare test_r against baseline
(tolerance_sweep_results.jsonl which used the legacy absolute 1e-9).

Also collects diagnostics to verify the tolerance is actually triggering
and saving closure calls as predicted by the offline analysis.
"""

import datetime
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, PROJ)

from run_single_mode import build_config_from_defaults  # noqa: E402

DATA_64 = os.path.join(PROJ, 'datasets', 'PNAS_64x64_center_crop_no_renorm.npz')
RF_PATH = os.path.join(PROJ, 'datasets', 'rf_centers_ground_truth.npz')
BASELINE_FILE = os.path.join(SCRIPT_DIR, 'tolerance_sweep_results.jsonl')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'confirmation_abs1em3_results.jsonl')

ABS_TOL = 1e-3

CELLS = [8, 10, 3, 7, 22, 20, 30, 0, 15, 39]
SEEDS = [1, 2, 3]

rf = np.load(RF_PATH)


def load_completed_runs(path):
    completed = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed.add((r.get('cell'), r.get('seed')))
                except json.JSONDecodeError:
                    continue
    return completed


def load_baseline():
    baseline = {}
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    baseline[(r['cell'], r['seed'])] = r
                except (json.JSONDecodeError, KeyError):
                    continue
    return baseline


def run_single_cell(cell_id, seed):
    eps_0x, eps_0y = rf['norm_64'][cell_id]

    config = build_config_from_defaults(
        mode='vargp_direct',
        M=250, n_train=3160, n_val_split=0,
        seed=seed, cell=cell_id,
        data_path=DATA_64,
        eps_0x=float(eps_0x), eps_0y=float(eps_0y),
        beta=0.1, rho=0.1,
        A_init=1e-4, lambda0_init=-1.0,
        n_iterations=80, n_estep=50, n_mstep=20,
        es_metric='elbo', patience=15, min_delta_rel=0.001, min_iterations=10,
    )
    config['early_stop'] = True
    config['ip_selection'] = 'random'
    config['fix_Amp'] = True
    config['interleave_fstep'] = True
    config['collect_mstep_diagnostics'] = True
    config['lbfgs_tolerance_change_abs'] = ABS_TOL

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False,
                                     dir='/tmp') as f:
        json.dump(config, f)
        tmp_path = f.name

    try:
        cmd = [sys.executable, os.path.join(PROJ, 'run_single_mode.py'),
               '--from-config', tmp_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        for line in proc.stdout.split('\n'):
            if line.startswith('RESULT_JSON:'):
                return json.loads(line[len('RESULT_JSON:'):])
        print(f"    ERROR: No RESULT_JSON for cell {cell_id} seed {seed}",
              flush=True)
        if proc.stderr:
            print(f"    STDERR: {proc.stderr[-500:]}", flush=True)
        return None
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT: cell {cell_id} seed {seed}", flush=True)
        return None
    except Exception as e:
        print(f"    EXCEPTION: cell {cell_id} seed {seed}: {e}", flush=True)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def main():
    start_time = time.time()
    total = len(CELLS) * len(SEEDS)
    baseline = load_baseline()

    print(f"=== Confirmation: abs_tol={ABS_TOL:.0e} vs baseline ===")
    print(f"  {len(CELLS)} cells x {len(SEEDS)} seeds = {total} runs")
    print(f"  Baseline: {len(baseline)} runs from {BASELINE_FILE}")
    print(f"  Results: {RESULTS_FILE}")
    print(f"  Started: {datetime.datetime.now().isoformat(timespec='seconds')}")

    completed = load_completed_runs(RESULTS_FILE)
    if completed:
        print(f"  Resuming: {len(completed)} done, {total - len(completed)} remaining")
    print(flush=True)

    jobs = [(c, s) for c in CELLS for s in SEEDS if (c, s) not in completed]
    if not jobs:
        print("All runs already completed.")
    else:
        print(f"  Running {len(jobs)} jobs\n", flush=True)
        for i, (cell_id, seed) in enumerate(jobs):
            print(f"[{i+1}/{len(jobs)}] cell={cell_id} seed={seed} ...",
                  end='', flush=True)
            result = run_single_cell(cell_id, seed)
            if result:
                tr = result.get('test_r', 0)
                n_iter = result.get('n_iterations_run', '?')
                n_diag = len(result.get('mstep_diagnostics') or [])
                bl = baseline.get((cell_id, seed))
                bl_tr = bl.get('test_r', 0) if bl else '?'
                bl_iter = bl.get('n_iterations_run', '?') if bl else '?'
                delta = tr - bl_tr if isinstance(bl_tr, float) else '?'
                print(f" test_r={tr:.4f} (baseline={bl_tr:.4f}, delta={delta:+.4f}), "
                      f"iters={n_iter} (bl={bl_iter}), mstep_entries={n_diag}",
                      flush=True)
                with open(RESULTS_FILE, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            else:
                print(f" FAILED", flush=True)

    # --- Summary ---
    confirm = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    confirm[(r['cell'], r['seed'])] = r
                except (json.JSONDecodeError, KeyError):
                    continue

    print(f"\n{'='*70}")
    print(f"COMPARISON: abs_tol={ABS_TOL:.0e} vs baseline (abs tol=1e-9)")
    print(f"{'='*70}")
    print(f"{'Cell':>6} {'Seed':>4} {'bl_test_r':>10} {'new_test_r':>11} "
          f"{'delta':>8} {'bl_iters':>9} {'new_iters':>10}")

    deltas = []
    iter_diffs = []
    for cell in CELLS:
        for seed in SEEDS:
            bl = baseline.get((cell, seed))
            nw = confirm.get((cell, seed))
            if bl and nw:
                d = nw['test_r'] - bl['test_r']
                deltas.append(d)
                bl_it = bl.get('n_iterations_run', '?')
                nw_it = nw.get('n_iterations_run', '?')
                if isinstance(bl_it, int) and isinstance(nw_it, int):
                    iter_diffs.append(nw_it - bl_it)
                print(f"{cell:>6} {seed:>4} {bl['test_r']:>10.4f} {nw['test_r']:>11.4f} "
                      f"{d:>+8.4f} {bl_it:>9} {nw_it:>10}")

    if deltas:
        deltas = np.array(deltas)
        print(f"\ntest_r delta: mean={np.mean(deltas):+.4f}, "
              f"median={np.median(deltas):+.4f}, "
              f"min={np.min(deltas):+.4f}, max={np.max(deltas):+.4f}")
        print(f"  |delta| > 0.005: {np.sum(np.abs(deltas) > 0.005)}/{len(deltas)} runs")
        print(f"  |delta| > 0.01:  {np.sum(np.abs(deltas) > 0.01)}/{len(deltas)} runs")
    if iter_diffs:
        iter_diffs = np.array(iter_diffs)
        print(f"Iterations: mean_diff={np.mean(iter_diffs):+.1f}, "
              f"median_diff={np.median(iter_diffs):+.0f}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
