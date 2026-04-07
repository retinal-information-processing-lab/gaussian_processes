#!/usr/bin/env python3
"""
ELBO Early Stopping Sweep: 4-config Amp x Interleave grid on 64x64.

================================================================================
PURPOSE
================================================================================

This sweep tests whether using ELBO staleness as an early stopping criterion
recovers the ~0.02 test_r gap we see with val_log_lik-based ES.

The idea is from investigations/optimization/possible_optimizations.md
(Investigation 2, "ELBO Convergence as Early Stopping"):

- The ELBO = log_lik - KL is the objective we maximize. Its KL term acts as
  a built-in regularizer, so overfitting in the neural-network sense is
  unlikely with M/N ~ 0.09 (250 inducing points, 2910 training images).
- val_ll-based ES was unstable because the interleaved F-step causes rapid
  A transients, making predictions (and hence val_ll) noisy in early iters.
- ELBO averages over all N training points, so it's robust to A transients.
- Stopping when the ELBO stops increasing is mathematically principled for
  variational inference.

================================================================================
GRID (4 configs x 41 cells x 3 seeds = 492 runs)
================================================================================

| # | config_name                   | interleave | fix_Amp | A_init | Amp_init |
|---|-------------------------------|------------|---------|--------|----------|
| 1 | 64_elbo_intl_fixAmp           | yes        | yes     | 1e-4   | 1.0 (frozen) |
| 2 | 64_elbo_intl_freeAmp_Amp1     | yes        | no      | 1e-4   | 1.0 (free)   |
| 3 | 64_elbo_no_intl_fixAmp        | no         | yes     | 0.01   | 1.0 (frozen) |
| 4 | 64_elbo_no_intl_freeAmp_Amp1  | no         | no      | 0.01   | 1.0 (free)   |

Each config uses:
  - M = 250 inducing points
  - n_train = 3160 (NO validation carving — all images used for training)
  - seeds = [1, 2, 3]
  - beta init = 0.1, rho init = 0.1, lambda0 init = -1.0
  - Amp init = 1.0 (the default; frozen or free depending on fix_Amp)
  - ground-truth RF centers (from datasets/rf_centers_ground_truth.npz)
  - ip_selection = 'random'
  - n_estep = 50, n_mstep = 20, n_iterations = 80

Why different A_init values (1e-4 for interleaved, 0.01 for non-interleaved):

  The E-step Newton update has gradient magnitude ~ A * N_train * max(r).
  With A=0.01 and N_train=3160, the Newton step on (m, V) overshoots,
  producing large mu values. Then f_mean = exp(A*mu + lambda0) blows past
  the stability threshold (500), the E-step reverts, and training stalls.
  Empirically verified on cell 8: A_init=0.01 + interleave gives 19/19
  E-step divergences and test_r=0.64 (vs 0.88 with A_init=1e-4).

  With A_init=1e-4, the initial Newton gradient is 100x smaller, keeping
  the first E-step stable. A then grows via the interleaved damped Newton
  as (m, V) stabilize.

  For non-interleaved configs, A is frozen at its init value throughout
  the 50 E-step iterations, so the E-step is stable for A=1e-4 OR A=0.01.
  But after the E-step, the LBFGS F-step runs only 10 iterations; the
  gradient of the expected log-likelihood w.r.t. log(A) vanishes as A->0,
  so LBFGS can't bootstrap A from 1e-4. This was confirmed by the p=30
  sweep where no_intl_fixAmp_es_p30 (with A_init=1e-4) got only 0.754
  mean test_r vs 0.838 for the interleaved version.

  So the A_init choice is a physical constraint of the optimizer, not a
  tuning preference: interleaved needs A_init small enough for the first
  Newton step to be stable; non-interleaved needs A_init large enough for
  LBFGS to see a gradient.

================================================================================
EARLY STOPPING
================================================================================

  es_metric = 'elbo'      # uses -train_loss (no val data needed)
  patience = 15           # Match previous val_ll sweeps for fair comparison
  min_delta_rel = 0.001   # 0.1% relative ELBO improvement to reset counter
  min_iterations = 10
  restore_best = True

NO VALIDATION CARVING: This sweep sets n_val_split=0, so the full 3160
training images are used for training. ELBO ES uses the training loss
directly (-train_loss) and doesn't need a held-out validation set. The
training curves still log val_log_lik / val_r / val_rho as None (they're
not computed when X_val is None).

This is one of the key benefits claimed for ELBO ES in
possible_optimizations.md Investigation 2: recovering the 8% data that
was previously held out for val_ll ES.

================================================================================
BASELINES FOR COMPARISON
================================================================================

See experiments/2026-04-06_es_sweeps_64x64/README.md for the reference
table. Key numbers to beat or match:

  Best no-ES baseline    (intl_fixAmp, 80 iters): test_r=0.8382, ev=0.8994, 37/41
  Best val_ll ES p=15    (intl_fixAmp, ~33 iters): test_r=0.8143, ev=0.8735, 36/41
  Best val_ll ES p=30    (intl_fixAmp, ~50 iters): test_r=0.8185, ev=0.8780, 36/41

This sweep's goal: ELBO ES config 1 (intl_fixAmp) should close the gap
toward 0.8382 while stopping meaningfully earlier than 80 iters.

================================================================================
OUTPUT
================================================================================

Results: investigations/optimization/sweep_64x64_elbo_es_results.jsonl

Each JSONL record contains: config, all kernel/likelihood params, training
curves (train_loss, train_log_lik, train_kl, val_log_lik, train_r, val_r,
val_rho, per-iteration parameters, iter_time), final metrics (test_r,
explained_var, adjusted_r2, reliability), timing, seed, cell.

Expected runtime: ~7-8 hours sequential on one GPU (~55-60s per run).

================================================================================
"""

import datetime
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

# -- Path setup --------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# investigations/optimization/ is 2 levels below gpytorch_porting/
PROJ = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJ)

from run_single_mode import build_config_from_defaults  # noqa: E402

# -- Fixed resources ---------------------------------------------------------
DATA_64 = os.path.join(PROJ, 'datasets', 'PNAS_64x64_center_crop_no_renorm.npz')
RF_PATH = os.path.join(PROJ, 'datasets', 'rf_centers_ground_truth.npz')
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'sweep_64x64_elbo_es_results.jsonl')

CELLS = list(range(41))
SEEDS = [1, 2, 3]

rf = np.load(RF_PATH)

# -- 4-config grid -----------------------------------------------------------
# Clean 2x2 grid on (interleave_fstep, fix_Amp).
# A_init differs by F-step method (see docstring rationale): interleaved
# configs require 1e-4 for E-step stability, non-interleaved require 0.01
# for LBFGS bootstrap. Amp_init is always 1.0 (free or frozen).
SHARED = {
    'beta': 0.1,
    'n_estep': 50,
    'n_mstep': 20,
    'n_iterations': 80,
}

# A_init required by F-step method (physical constraint, not a tuning choice)
A_INIT_INTERLEAVED = 1e-4   # small so first E-step Newton doesn't overshoot
A_INIT_NON_INTERLEAVED = 0.01  # large enough for LBFGS to see a gradient

CONFIGS = [
    {
        'name': '64_elbo_intl_fixAmp',
        'notes': 'Interleaved F-step + Amp frozen at 1.0',
        'interleave_fstep': True,
        'fix_Amp': True,
        'A_init': A_INIT_INTERLEAVED,
        **SHARED,
    },
    {
        'name': '64_elbo_intl_freeAmp_Amp1',
        'notes': 'Interleaved F-step + free Amp starting at 1.0',
        'interleave_fstep': True,
        'fix_Amp': False,
        'A_init': A_INIT_INTERLEAVED,
        **SHARED,
    },
    {
        'name': '64_elbo_no_intl_fixAmp',
        'notes': 'Standard F-step + Amp frozen at 1.0',
        'interleave_fstep': False,
        'fix_Amp': True,
        'A_init': A_INIT_NON_INTERLEAVED,
        **SHARED,
    },
    {
        'name': '64_elbo_no_intl_freeAmp_Amp1',
        'notes': 'Standard F-step + free Amp starting at 1.0',
        'interleave_fstep': False,
        'fix_Amp': False,
        'A_init': A_INIT_NON_INTERLEAVED,
        **SHARED,
    },
]

# -- ELBO ES settings (fixed across configs) ---------------------------------
ES_METRIC = 'elbo'
PATIENCE = 15
MIN_DELTA_REL = 0.001
MIN_ITERATIONS = 10


def load_completed_runs():
    """Load already-completed (config_name, cell, seed) tuples. Enables resume."""
    completed = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    key = (r.get('config_name'), r.get('cell'), r.get('seed'))
                    completed.add(key)
                except json.JSONDecodeError:
                    continue
    return completed


def run_single_cell(cell_id, seed, cfg):
    """Run a single (cell, seed, config) fit as a subprocess.

    Uses a subprocess so each run has its own Python process — this
    isolates GPU memory and prevents any leak from accumulating across runs.
    """
    eps_0x, eps_0y = rf['norm_64'][cell_id]

    config = build_config_from_defaults(
        mode='vargp_direct',
        M=250,
        n_train=3160,           # all training images — no val carving
        n_val_split=0,          # CRITICAL: skip val carving, use full pool
        seed=seed,
        cell=cell_id,
        data_path=DATA_64,
        eps_0x=float(eps_0x),
        eps_0y=float(eps_0y),
        beta=cfg['beta'],
        rho=0.1,
        A_init=cfg['A_init'],
        lambda0_init=-1.0,
        n_iterations=cfg['n_iterations'],
        n_estep=cfg['n_estep'],
        n_mstep=cfg['n_mstep'],
        # ELBO ES settings
        es_metric=ES_METRIC,
        patience=PATIENCE,
        min_delta_rel=MIN_DELTA_REL,
        min_iterations=MIN_ITERATIONS,
    )
    config['early_stop'] = True
    config['ip_selection'] = 'random'
    config['fix_Amp'] = cfg['fix_Amp']
    config['interleave_fstep'] = cfg['interleave_fstep']

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir='/tmp') as f:
        json.dump(config, f)
        tmp_path = f.name

    try:
        cmd = [sys.executable, os.path.join(PROJ, 'run_single_mode.py'),
               '--from-config', tmp_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        for line in proc.stdout.split('\n'):
            if line.startswith('RESULT_JSON:'):
                result = json.loads(line[len('RESULT_JSON:'):])
                result['config_name'] = cfg['name']
                return result

        print(f"    ERROR: No RESULT_JSON for cell {cell_id} seed {seed} "
              f"config {cfg['name']}", flush=True)
        if proc.stderr:
            print(f"    STDERR: {proc.stderr[-300:]}", flush=True)
        return None
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT: cell {cell_id} seed {seed} config {cfg['name']} (>600s)",
              flush=True)
        return None
    except Exception as e:
        print(f"    EXCEPTION: cell {cell_id} seed {seed} config {cfg['name']}: {e}",
              flush=True)
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
    total = len(CONFIGS) * len(CELLS) * len(SEEDS)

    print(f"=== ELBO Early Stopping Sweep (64x64) ===")
    print(f"  ES metric: {ES_METRIC}, patience={PATIENCE}, "
          f"min_delta_rel={MIN_DELTA_REL}, min_iterations={MIN_ITERATIONS}")
    print(f"  {len(CONFIGS)} configs x {len(CELLS)} cells x {len(SEEDS)} seeds = {total} runs")
    print(f"  Started: {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"  Results: {RESULTS_FILE}")

    completed = load_completed_runs()
    if completed:
        print(f"  Resuming: {len(completed)} runs already completed, "
              f"{total - len(completed)} remaining")
    print(flush=True)

    jobs = []
    for cfg in CONFIGS:
        for cell_id in CELLS:
            for seed in SEEDS:
                key = (cfg['name'], cell_id, seed)
                if key not in completed:
                    jobs.append((cell_id, seed, cfg))

    if not jobs:
        print("All runs already completed.")
        return

    print(f"  Running {len(jobs)} jobs sequentially\n", flush=True)

    done = 0
    failed = 0

    for cell_id, seed, cfg in jobs:
        done += 1
        result = run_single_cell(cell_id, seed, cfg)

        if result:
            tr = result.get('test_r', 0)
            tt = result.get('train_time', 0)
            se = result.get('stopped_early', False)
            ni = result.get('n_iterations_run', '?')
            bi = result.get('best_iteration', '?')
            print(f"  [{done}/{len(jobs)}] {cfg['name']} cell={cell_id} seed={seed} "
                  f"test_r={tr:.4f} iters={ni} best={bi} es={se} time={tt:.1f}s",
                  flush=True)
            append_result(result)
        else:
            failed += 1
            print(f"  [{done}/{len(jobs)}] {cfg['name']} cell={cell_id} seed={seed} FAILED",
                  flush=True)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Sweep complete: {done - failed}/{done} successful, {failed} failed")
    print(f"Total time: {total_time / 3600:.1f}h")
    print(f"Results: {RESULTS_FILE}")


if __name__ == '__main__':
    main()
