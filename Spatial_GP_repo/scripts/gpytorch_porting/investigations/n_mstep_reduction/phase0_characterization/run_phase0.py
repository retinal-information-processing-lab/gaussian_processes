#!/usr/bin/env python3
"""
Phase 0: Characterize M-step LBFGS behavior with n_mstep=20.

================================================================================
PURPOSE
================================================================================

Understand what the M-step LBFGS is actually doing today before proposing any
changes to n_mstep. Specifically:

1. How many LBFGS iterations actually run before tolerance triggers?
   (vs the max_iter budget of 20)
2. How many closure calls does Wolfe line search make per iteration?
3. Does the ELBO improvement plateau after a few LBFGS iterations?
4. How does M-step behavior change over the course of training
   (early outer iterations vs late)?
5. Is the answer cell-dependent?

================================================================================
GRID (5 cells x 3 seeds = 15 runs)
================================================================================

Cells chosen to span the difficulty range:
  - Cell 8:  well-behaved baseline (test_r ~ 0.88)
  - Cell 0:  known stuck-near-init failure mode
  - Cell 10: default_gpy struggles, vargp_direct handles it
  - Cell 15: mid-difficulty, STA edge artifact on 108x108 but clean on 64x64
  - Cell 22: mid-difficulty, same STA edge artifact caveat

Config: intl_fixAmp + ELBO ES (current best documented config).
  - M=250, n_train=3160 (all images, no val carving)
  - interleave_fstep=True, fix_Amp=True, A_init=1e-4
  - n_estep=50, n_mstep=20 (deliberately above default=10 to see full
    convergence curve), n_iterations=80
  - ELBO ES: patience=15, min_delta_rel=0.001, min_iterations=10
  - ground-truth RF centers from rf_centers_ground_truth.npz

n_mstep=20 matches the ELBO ES sweep baseline (run_sweep_elbo_es_64x64.py).

collect_mstep_diagnostics=True is set for all runs. This records per-closure-
call ELBO, wall time, and LBFGS termination reason for every outer EM
iteration.

================================================================================
OUTPUT
================================================================================

Results: investigations/n_mstep_reduction/phase0_characterization/results.jsonl

Each JSONL record contains all standard fields from run_single_config PLUS
'mstep_diagnostics': a list of per-outer-iteration M-step diagnostic dicts
with keys: outer_iter, n_lbfgs_iters, termination, total_time_s,
closure_calls (list of {loss, time_s, rejected}).

Expected runtime: ~15 runs x ~60s each = ~15 minutes.
"""

import datetime
import json
import os
import subprocess
import sys
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
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'results.jsonl')

CELLS = [8, 0, 10, 15, 22]
SEEDS = [1, 2, 3]

rf = np.load(RF_PATH)


def load_completed_runs():
    """Load already-completed (cell, seed) tuples. Enables resume."""
    completed = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    key = (r.get('cell'), r.get('seed'))
                    completed.add(key)
                except json.JSONDecodeError:
                    continue
    return completed


def run_single_cell(cell_id, seed):
    """Run a single (cell, seed) fit as a subprocess."""
    import tempfile

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
                result = json.loads(line[len('RESULT_JSON:'):])
                return result

        print(f"    ERROR: No RESULT_JSON for cell {cell_id} seed {seed}",
              flush=True)
        if proc.stderr:
            # Print last 500 chars of stderr for debugging
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

    print(f"=== Phase 0: M-step Characterization (64x64, intl_fixAmp) ===")
    print(f"  Config: M=250, n_train=3160, n_mstep=20, n_iterations=80")
    print(f"  ELBO ES: patience=15, min_delta_rel=0.001")
    print(f"  {len(CELLS)} cells x {len(SEEDS)} seeds = {total} runs")
    print(f"  Cells: {CELLS}")
    print(f"  Started: {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"  Results: {RESULTS_FILE}")

    completed = load_completed_runs()
    if completed:
        print(f"  Resuming: {len(completed)} runs already completed, "
              f"{total - len(completed)} remaining")
    print(flush=True)

    jobs = []
    for cell_id in CELLS:
        for seed in SEEDS:
            key = (cell_id, seed)
            if key not in completed:
                jobs.append((cell_id, seed))

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
            tt = result.get('time_total_s', 0)
            se = result.get('stopped_early', False)
            n_diag = len(result.get('mstep_diagnostics') or [])
            print(f" test_r={tr:.4f}, time={tt:.1f}s, "
                  f"{'ES' if se else 'full'}, "
                  f"mstep_diag_entries={n_diag}",
                  flush=True)
            append_result(result)
        else:
            failed += 1
            print(f" FAILED", flush=True)

    elapsed = time.time() - start_time
    print(f"\n=== Done: {done - failed}/{done} succeeded, "
          f"{failed} failed, {elapsed:.0f}s total ===")


if __name__ == '__main__':
    main()
