"""Phase 2 Step 2: three targeted interventions on outlier cells.

Three hypotheses, three interventions (see SCRAPBOOK.md section 13 for
motivation):

  H-a: aggressive inner LBFGS -> intervention I_maxiter5
  H-b: lambda0_init far from equilibrium -> intervention I_lambda0_eq
  H-c: beta=0.01 attractor -> intervention I_beta0p2

Applied to cells 40 and 38, seed 42. 6 runs total.

Runs are direct overrides of build_config_from_defaults; no library changes.
Results in step2_results.jsonl (local to investigation folder).
"""
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

INVESTIGATION_DIR = Path(__file__).parent
ROOT = INVESTIGATION_DIR.parent.parent
sys.path.insert(0, str(ROOT))
from run_single_mode import build_config_from_defaults, run_single_config  # noqa: E402

STEP2_RESULTS = INVESTIGATION_DIR / 'step2_results.jsonl'
DATA_PATH = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
M = 300
N_TRAIN = 1500
SEED = 42
OUTLIER_CELLS = [40, 38]


def compute_lambda0_equilibrium(cell):
    """Approximate equilibrium lambda0: value such that exp(A_init * 0 + lambda0)
    matches the training-pool mean firing rate. Lambda0_eq = log(mean_fr).
    (Approximate because A_init*0 = 0 at initial zero variational mean.)"""
    data = np.load(ROOT / DATA_PATH)
    R = np.concatenate([data['responses_train'], data['responses_val']], axis=0)
    fr = float(R[:, cell].mean())
    return math.log(max(fr, 1e-3))


def make_interventions(cell):
    return [
        ('I_maxiter5',     dict(gpy_lbfgs_max_iter=5)),
        ('I_lambda0_eq',   dict(lambda0_init=compute_lambda0_equilibrium(cell))),
        ('I_beta0p2',      dict(beta=0.2)),
    ]


def run_one(label, cell, overrides):
    config = build_config_from_defaults(
        mode='default_gpy',
        data_path=DATA_PATH,
        M=M, n_train=N_TRAIN, cell=cell, seed=SEED,
        ip_selection='random',
        **overrides,
    )
    print(f"\n=== {label}  cell={cell}  overrides={overrides} ===")
    t0 = time.time()
    started = datetime.datetime.now().isoformat(timespec='seconds')
    try:
        result = run_single_config(config)
        err = None
    except Exception as e:
        result = None
        err = repr(e)
    elapsed = time.time() - t0

    if result is None:
        record = {
            'label': label, 'cell': cell, 'status': 'failed',
            'error': err, 'wall_time_s': elapsed,
            'started_at': started, 'overrides': overrides,
        }
    else:
        curves = result.get('curves') or {}
        record = {
            'label': label, 'cell': cell, 'seed': SEED,
            'status': result.get('status', 'unknown'),
            'wall_time_s': elapsed, 'started_at': started,
            'overrides': overrides,
            'test_r': result.get('test_r'),
            'train_r': result.get('train_r'),
            'final_loss': result.get('final_loss'),
            'n_iterations_run': result.get('n_iterations_run'),
            'stopped_early': result.get('stopped_early'),
            'best_iteration': result.get('best_iteration'),
            'final_A': result.get('final_A'),
            'final_lambda0': result.get('final_lambda0'),
            'final_beta': result.get('final_beta'),
            'final_rho': result.get('final_rho'),
            'final_eps_0x': result.get('final_eps_0x'),
            'final_eps_0y': result.get('final_eps_0y'),
            'train_loss_curve': curves.get('train_loss'),
            'A_curve': curves.get('A'),
            'beta_curve': curves.get('beta'),
        }

    with open(STEP2_RESULTS, 'a') as f:
        f.write(json.dumps(record) + '\n')
    tr = record.get('test_r')
    fl = record.get('final_loss')
    print(f"  -> test_r={tr if tr is None else f'{tr:.4f}'} "
          f"final_loss={fl if fl is None else f'{fl:.2f}'} "
          f"iters={record.get('n_iterations_run')} "
          f"final_A={record.get('final_A')} "
          f"wall={elapsed:.1f}s")
    return record


def main():
    os.chdir(ROOT)
    if STEP2_RESULTS.exists():
        STEP2_RESULTS.unlink()
    all_records = []
    for cell in OUTLIER_CELLS:
        print(f"\nlambda0_eq for cell {cell} = log(mean_fr) "
              f"= {compute_lambda0_equilibrium(cell):.3f}")
        for label, overrides in make_interventions(cell):
            all_records.append(run_one(label, cell, overrides))

    print("\n" + "=" * 108)
    print("STEP 2 SUMMARY (seed 42)")
    print("=" * 108)
    print(f"  {'cell':>4}  {'intervention':<18}  {'test_r':>7}  "
          f"{'final_loss':>12}  {'final_A':>9}  {'final_beta':>10}  "
          f"{'best_iter':>9}  {'stopped_early':>13}")
    for r in all_records:
        def fmt(x, f):
            return (f.format(x) if x is not None else 'None')
        print(f"  {r['cell']:>4}  {r['label']:<18}  "
              f"{fmt(r.get('test_r'), '{:>7.4f}')}  "
              f"{fmt(r.get('final_loss'), '{:>12.2f}')}  "
              f"{fmt(r.get('final_A'), '{:>9.4g}')}  "
              f"{fmt(r.get('final_beta'), '{:>10.4g}')}  "
              f"{str(r.get('best_iteration')):>9}  "
              f"{str(r.get('stopped_early')):>13}")
    print()
    print("Reference (main sweep, seed 42):")
    print("  vargp_direct cell=40: test_r=0.8553 loss=387.21 A=0.0414 beta=0.1116")
    print("  default_gpy  cell=40: test_r=0.3663 loss=459.11 A=0.5657 beta=0.0109")
    print("  vargp_direct cell=38: test_r=0.7776 loss=660.79 A=0.0320 beta=0.1291")
    print("  default_gpy  cell=38: test_r=0.3106 loss=1030.78 A=0.0631 beta=0.0630")


if __name__ == '__main__':
    main()
