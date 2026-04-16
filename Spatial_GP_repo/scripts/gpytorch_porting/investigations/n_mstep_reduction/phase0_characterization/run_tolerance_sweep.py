#!/usr/bin/env python3
"""
Tolerance Characterization Sweep: full M-step telemetry for offline analysis.

================================================================================
PURPOSE
================================================================================

Record per-closure-call loss, gradient, and parameter traces from the M-step
LBFGS to determine principled float32 tolerances for tolerance_change and
tolerance_grad.

Current defaults (tolerance_change=1e-9, tolerance_grad=1e-7) are below or
at the float32 noise floor, effectively disabled. This sweep runs with
these loose tolerances and n_mstep=20 to capture the full convergence
trajectory. In post-analysis, we can replay traces offline and ask:
"if tolerance had been X, at which call would LBFGS have stopped, and
how much ELBO improvement would have been forfeited?"

Enhanced telemetry per closure call:
  - loss (ELBO)
  - grad_max: max |g_i| across kernel params (LBFGS compares this to tolerance_grad)
  - grad_norm: L2 gradient norm
  - params: current kernel parameter values (6 floats)
  - time_s, rejected

Per M-step call:
  - n_lbfgs_iters, n_func_evals, final_step_size, termination
  - elbo_before_mstep, elbo_after_reproject (NET M-step benefit after
    eigenspace reprojection)

================================================================================
GRID (10 cells x 3 seeds = 30 runs)
================================================================================

Cells chosen to span difficulty range on 64x64:
  Easy:   8, 10, 3, 7        (test_r > 0.85)
  Mid:    22, 20, 30          (test_r 0.75-0.85)
  Hard:   0, 15, 39           (test_r < 0.70 or known issues)

Config: intl_fixAmp + ELBO ES (canonical best), matching Phase 0.
  M=250, n_train=3160, n_mstep=20, n_iterations=80

Expected runtime: ~30 runs x ~60s = ~30 minutes.

================================================================================
OUTPUT
================================================================================

Results: .../phase0_characterization/tolerance_sweep_results.jsonl
"""

import datetime
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

# -- Path setup ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, PROJ)

from run_single_mode import build_config_from_defaults  # noqa: E402

# -- Fixed resources ----------------------------------------------------------
DATA_64 = os.path.join(PROJ, 'datasets', 'PNAS_64x64_center_crop_no_renorm.npz')
RF_PATH = os.path.join(PROJ, 'datasets', 'rf_centers_ground_truth.npz')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'tolerance_sweep_results.jsonl')

# 10 cells spanning the difficulty range on 64x64
CELLS = [8, 10, 3, 7, 22, 20, 30, 0, 15, 39]
SEEDS = [1, 2, 3]

rf = np.load(RF_PATH)


def load_completed_runs():
    completed = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed.add((r.get('cell'), r.get('seed')))
                except json.JSONDecodeError:
                    continue
    return completed


def run_single_cell(cell_id, seed):
    eps_0x, eps_0y = rf['norm_64'][cell_id]

    config = build_config_from_defaults(
        mode='vargp_direct',
        M=250,
        n_train=3160,
        n_val_split=0,
        seed=seed,
        cell=cell_id,
        data_path=DATA_64,
        eps_0x=float(eps_0x),
        eps_0y=float(eps_0y),
        beta=0.1,
        rho=0.1,
        A_init=1e-4,
        lambda0_init=-1.0,
        n_iterations=80,
        n_estep=50,
        n_mstep=20,
        es_metric='elbo',
        patience=15,
        min_delta_rel=0.001,
        min_iterations=10,
    )
    config['early_stop'] = True
    config['ip_selection'] = 'random'
    config['fix_Amp'] = True
    config['interleave_fstep'] = True
    config['collect_mstep_diagnostics'] = True

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
        print(f"    TIMEOUT: cell {cell_id} seed {seed} (>600s)", flush=True)
        return None
    except Exception as e:
        print(f"    EXCEPTION: cell {cell_id} seed {seed}: {e}", flush=True)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def append_result(result):
    with open(RESULTS_FILE, 'a') as f:
        f.write(json.dumps(result) + '\n')


def main():
    start_time = time.time()
    total = len(CELLS) * len(SEEDS)

    print(f"=== Tolerance Characterization Sweep (64x64, intl_fixAmp) ===")
    print(f"  Config: M=250, n_train=3160, n_mstep=20, n_iterations=80")
    print(f"  Enhanced telemetry: grad_max, grad_norm, params, ELBO bracket")
    print(f"  {len(CELLS)} cells x {len(SEEDS)} seeds = {total} runs")
    print(f"  Cells: {CELLS}")
    print(f"  Started: {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"  Results: {RESULTS_FILE}")

    completed = load_completed_runs()
    if completed:
        print(f"  Resuming: {len(completed)} runs already completed, "
              f"{total - len(completed)} remaining")
    print(flush=True)

    jobs = [(c, s) for c in CELLS for s in SEEDS if (c, s) not in completed]

    if not jobs:
        print("All runs already completed.")
        return

    print(f"  Running {len(jobs)} jobs sequentially\n", flush=True)

    done = 0
    failed = 0

    for cell_id, seed in jobs:
        done += 1
        print(f"[{done}/{len(jobs)}] cell={cell_id} seed={seed} ...",
              end='', flush=True)

        result = run_single_cell(cell_id, seed)

        if result:
            tr = result.get('test_r', 0)
            se = result.get('stopped_early', False)
            n_diag = len(result.get('mstep_diagnostics') or [])
            print(f" test_r={tr:.4f}, {'ES' if se else 'full'}, "
                  f"mstep_entries={n_diag}", flush=True)
            append_result(result)
        else:
            failed += 1
            print(f" FAILED", flush=True)

    elapsed = time.time() - start_time
    print(f"\n=== Done: {done - failed}/{done} succeeded, "
          f"{failed} failed, {elapsed:.0f}s total ===")


if __name__ == '__main__':
    main()
