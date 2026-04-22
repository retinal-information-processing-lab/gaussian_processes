"""Phase 2 Step 4: seed-replicate beta_init=0.2 across seeds {123, 789}.

Step 3 already has seed 42. This completes 3-seed coverage for the
beta_init=0.2 intervention, giving 48 paired observations (16 cells x 3 seeds)
to compare against the main sweep's 48 default_gpy baselines.

Motivation: at seed 42, beta_init=0.2 showed a cell-level pattern (40 and 38
recover big; 29 and 8 regress). Whether that pattern is robust or seed-noise
requires more seeds. 32 additional runs.

Output: step4_results.jsonl (32 records, seeds 123 and 789 only).
Combined with step3_results.jsonl (seed 42), gives the full 3-seed picture.
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

CELLS_USED_PATH = INVESTIGATION_DIR / 'cells_used.json'
STEP4_RESULTS = INVESTIGATION_DIR / 'step4_results.jsonl'
DATA_PATH = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
M = 300
N_TRAIN = 1500
SEEDS = [123, 789]
BETA_INIT = 0.2


def load_cells():
    with open(CELLS_USED_PATH) as f:
        return [c['cell_id'] for c in json.load(f)['cells']]


def run_one(cell, seed):
    config = build_config_from_defaults(
        mode='default_gpy',
        data_path=DATA_PATH,
        M=M, n_train=N_TRAIN, cell=cell, seed=seed,
        ip_selection='random',
        beta=BETA_INIT,
    )
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
        record = {'cell': cell, 'seed': seed, 'status': 'failed',
                  'error': err, 'wall_time_s': elapsed, 'started_at': started}
    else:
        curves = result.get('curves') or {}
        record = {
            'cell': cell, 'seed': seed,
            'status': result.get('status', 'unknown'),
            'wall_time_s': elapsed, 'started_at': started,
            'beta_init': BETA_INIT,
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
            'train_loss_curve': curves.get('train_loss'),
            'A_curve': curves.get('A'),
            'beta_curve': curves.get('beta'),
        }
    with open(STEP4_RESULTS, 'a') as f:
        f.write(json.dumps(record) + '\n')
    return record


def main():
    os.chdir(ROOT)
    if STEP4_RESULTS.exists():
        STEP4_RESULTS.unlink()
    cells = load_cells()
    total = len(cells) * len(SEEDS)
    print(f"Running beta_init={BETA_INIT} on {len(cells)} cells x {len(SEEDS)} seeds = {total} runs")
    i = 0
    t_start = time.time()
    for cell in cells:
        for seed in SEEDS:
            i += 1
            print(f"  [{i:2d}/{total}] cell={cell} seed={seed} ...", end=' ', flush=True)
            r = run_one(cell, seed)
            print(f"test_r={r.get('test_r', 0):.4f}  wall={r['wall_time_s']:.1f}s")
    print(f"\nDone. {total} runs in {(time.time()-t_start)/60:.1f} min.")


if __name__ == '__main__':
    main()
