"""Phase 2 Step 1: diagnostic runs on outlier cells.

Two variations x 2 outlier cells x 1 seed = 4 runs.

  Run A (rules out premature ES): default_gpy with early_stop=False on cells
        40 and 38 at seed 42. Everything else identical to the main sweep.
        Expected result: loss stays at the plateau across all 50 iterations.

  Run B (tests basin-of-attraction): default_gpy warm-started from
        vargp_direct's final (A, lambda0, beta, rho, eps_0x, eps_0y) on the
        same cell/seed. If default_gpy stays at the good ELBO, the problem
        is the init's basin. If it drifts back toward the bad optimum, the
        problem is structural to joint-LBFGS parameterization.

Outputs JSONL to step1_results.jsonl (separate from the main sweep) and
prints a comparison table. Does not write to or modify results.jsonl.
"""

import datetime
import json
import os
import sys
import time
from pathlib import Path

INVESTIGATION_DIR = Path(__file__).parent
ROOT = INVESTIGATION_DIR.parent.parent
sys.path.insert(0, str(ROOT))
from run_single_mode import build_config_from_defaults, run_single_config  # noqa: E402

MAIN_RESULTS = INVESTIGATION_DIR / 'results.jsonl'
STEP1_RESULTS = INVESTIGATION_DIR / 'step1_results.jsonl'
DATA_PATH = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
M = 300
N_TRAIN = 1500
SEED = 42
OUTLIER_CELLS = [40, 38]


def load_vargp_finals(cell, seed):
    """Read vargp_direct's final params for (cell, seed) from results.jsonl."""
    with open(MAIN_RESULTS) as f:
        for line in f:
            r = json.loads(line)
            if r['mode'] == 'vargp_direct' and r['cell'] == cell and r['seed'] == seed:
                return {
                    'A': r['final_A'],
                    'lambda0': r['final_lambda0'],
                    'beta': r['final_beta'],
                    'rho': r['final_rho'],
                    'eps_0x': r['final_eps_0x'],
                    'eps_0y': r['final_eps_0y'],
                }
    raise RuntimeError(f"no vargp_direct record for cell={cell} seed={seed}")


def run_and_save(label, config):
    print(f"\n=== {label} ===")
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
            'label': label,
            'status': 'failed',
            'error': err,
            'wall_time_s': elapsed,
            'started_at': started,
        }
    else:
        curves = result.get('curves') or {}
        record = {
            'label': label,
            'cell': config['cell'],
            'seed': config['seed'],
            'status': result.get('status', 'unknown'),
            'wall_time_s': elapsed,
            'started_at': started,
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
            'rho_curve': curves.get('rho'),
            'config_overrides': {
                k: config.get(k) for k in [
                    'mode', 'early_stop', 'A_init', 'lambda0_init',
                    'beta', 'rho', 'eps_0x', 'eps_0y',
                    'M', 'n_train', 'cell', 'seed', 'ip_selection',
                ]
            },
        }

    with open(STEP1_RESULTS, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(f"  -> test_r={record.get('test_r')} final_loss={record.get('final_loss')} "
          f"iters_run={record.get('n_iterations_run')} wall={elapsed:.1f}s")
    return record


def main():
    os.chdir(ROOT)
    if STEP1_RESULTS.exists():
        STEP1_RESULTS.unlink()

    all_records = []

    # --- Run A: early_stop=False ---
    for cell in OUTLIER_CELLS:
        config = build_config_from_defaults(
            mode='default_gpy',
            data_path=DATA_PATH,
            M=M, n_train=N_TRAIN, cell=cell, seed=SEED,
            ip_selection='random',
            early_stop=False,
        )
        all_records.append(
            run_and_save(f'A_no_es_cell{cell}', config)
        )

    # --- Run B: warm-start from vargp_direct's final params ---
    for cell in OUTLIER_CELLS:
        finals = load_vargp_finals(cell, SEED)
        config = build_config_from_defaults(
            mode='default_gpy',
            data_path=DATA_PATH,
            M=M, n_train=N_TRAIN, cell=cell, seed=SEED,
            ip_selection='random',
            A_init=finals['A'],
            lambda0_init=finals['lambda0'],
            beta=finals['beta'],
            rho=finals['rho'],
            eps_0x=finals['eps_0x'],
            eps_0y=finals['eps_0y'],
        )
        all_records.append(
            run_and_save(f'B_warmstart_cell{cell}', config)
        )

    print("\n" + "=" * 80)
    print("STEP 1 SUMMARY")
    print("=" * 80)
    print(f"  {'label':<22}  {'test_r':>7}  {'final_loss':>12}  "
          f"{'iters':>5}  {'stopped_early':>13}  {'final_A':>9}  {'final_beta':>10}")
    for r in all_records:
        print(f"  {r['label']:<22}  "
              f"{r.get('test_r', 0):>7.4f}  "
              f"{r.get('final_loss', 0):>12.2f}  "
              f"{r.get('n_iterations_run', 0):>5}  "
              f"{str(r.get('stopped_early')):>13}  "
              f"{r.get('final_A', 0):>9.4g}  "
              f"{r.get('final_beta', 0):>10.4g}")
    print()
    print("Reference (from main sweep, seed 42):")
    print("  vargp_direct cell=40: test_r=0.8553, final_loss=387.21, A=0.0414, beta=0.1116")
    print("  default_gpy  cell=40: test_r=0.3663, final_loss=459.11, A=0.5657, beta=0.0109")
    print("  vargp_direct cell=38: test_r=0.7776, final_loss=660.79, A=0.0320, beta=0.1291")
    print("  default_gpy  cell=38: test_r=0.3106, final_loss=1030.78, A=0.0631, beta=0.0630")


if __name__ == '__main__':
    main()
