"""Phase 2 Step 3: test beta_init=0.2 across ALL 16 cells, seed 42.

Motivation: Step 2 found beta_init=0.2 recovers both outlier cells (40, 38)
dramatically. Before calling it a fix, must verify it does not harm the 14
cells where default_gpy already worked at baseline (beta_init=0.1, the
default).

Design: 16 cells x 1 seed (42). Each run = default_gpy with beta=0.2, all
other settings identical to main sweep. Compare paired delta vs baseline
default_gpy (main sweep, seed 42) and vs vargp_direct (seed 42).

Output: step3_results.jsonl (one record per cell), printed summary table.
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
STEP3_RESULTS = INVESTIGATION_DIR / 'step3_results.jsonl'
CELLS_USED_PATH = INVESTIGATION_DIR / 'cells_used.json'
DATA_PATH = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
M = 300
N_TRAIN = 1500
SEED = 42
BETA_INIT = 0.2


def load_cells():
    with open(CELLS_USED_PATH) as f:
        return [c['cell_id'] for c in json.load(f)['cells']]


def load_baseline(mode):
    """(cell, seed)-keyed dict of baseline test_r for the given mode."""
    out = {}
    with open(MAIN_RESULTS) as f:
        for line in f:
            r = json.loads(line)
            if r['mode'] == mode:
                out[(r['cell'], r['seed'])] = r
    return out


def run_one(cell):
    config = build_config_from_defaults(
        mode='default_gpy',
        data_path=DATA_PATH,
        M=M, n_train=N_TRAIN, cell=cell, seed=SEED,
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
        record = {'cell': cell, 'status': 'failed', 'error': err,
                  'wall_time_s': elapsed, 'started_at': started}
    else:
        curves = result.get('curves') or {}
        record = {
            'cell': cell, 'seed': SEED,
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
    with open(STEP3_RESULTS, 'a') as f:
        f.write(json.dumps(record) + '\n')
    return record


def main():
    os.chdir(ROOT)
    if STEP3_RESULTS.exists():
        STEP3_RESULTS.unlink()

    cells = load_cells()
    baseline_gpy = load_baseline('default_gpy')
    baseline_vargp = load_baseline('vargp_direct')

    print(f"Running default_gpy with beta_init={BETA_INIT} on {len(cells)} cells, seed {SEED}")
    print("=" * 80)

    recs = []
    for i, cell in enumerate(cells, 1):
        print(f"  [{i:2d}/{len(cells)}] cell={cell} ...", end=' ', flush=True)
        r = run_one(cell)
        recs.append(r)
        tr = r.get('test_r')
        print(f"test_r={'N/A' if tr is None else f'{tr:.4f}'}  "
              f"final_beta={r.get('final_beta')}  "
              f"wall={r['wall_time_s']:.1f}s")

    # Summary
    print()
    print("=" * 108)
    print("STEP 3 PER-CELL TABLE (seed 42)")
    print("=" * 108)
    print(f"  {'cell':>4}  {'tr_beta0p2':>11}  {'tr_default_gpy':>15}  "
          f"{'tr_vargp_direct':>16}  {'d_vs_gpy':>10}  {'d_vs_vargp':>12}  "
          f"{'f_beta':>7}  {'f_A':>8}")
    deltas_vs_gpy = []
    deltas_vs_vargp = []
    for r in recs:
        tr_new = r.get('test_r')
        cell = r['cell']
        tr_gpy = baseline_gpy.get((cell, SEED), {}).get('test_r')
        tr_vargp = baseline_vargp.get((cell, SEED), {}).get('test_r')
        d_gpy = (tr_new - tr_gpy) if (tr_new is not None and tr_gpy is not None) else None
        d_vargp = (tr_new - tr_vargp) if (tr_new is not None and tr_vargp is not None) else None
        if d_gpy is not None: deltas_vs_gpy.append(d_gpy)
        if d_vargp is not None: deltas_vs_vargp.append(d_vargp)
        def fmt(x, f): return f.format(x) if x is not None else 'N/A'
        print(f"  {cell:>4}  {fmt(tr_new, '{:>11.4f}')}  "
              f"{fmt(tr_gpy, '{:>15.4f}')}  {fmt(tr_vargp, '{:>16.4f}')}  "
              f"{fmt(d_gpy, '{:>+10.4f}')}  {fmt(d_vargp, '{:>+12.4f}')}  "
              f"{fmt(r.get('final_beta'), '{:>7.4f}')}  "
              f"{fmt(r.get('final_A'), '{:>8.4g}')}")
    print("-" * 108)
    import numpy as np
    print(f"  AGGREGATE delta_vs_baseline_default_gpy: "
          f"mean={np.mean(deltas_vs_gpy):+.4f}  "
          f"std={np.std(deltas_vs_gpy, ddof=1):.4f}  "
          f"n_improved={(np.array(deltas_vs_gpy) > 0.02).sum()}  "
          f"n_harmed={(np.array(deltas_vs_gpy) < -0.02).sum()}")
    print(f"  AGGREGATE delta_vs_vargp_direct:         "
          f"mean={np.mean(deltas_vs_vargp):+.4f}  "
          f"std={np.std(deltas_vs_vargp, ddof=1):.4f}  "
          f"(pre-registered threshold: |mean_delta|>0.02 -> GAP EXISTS)")
    print()
    print("  Reference (from main sweep, paired seed-42 only, 16 cells):")
    baseline_deltas = []
    for cell in cells:
        tv = baseline_vargp.get((cell, SEED), {}).get('test_r')
        tg = baseline_gpy.get((cell, SEED), {}).get('test_r')
        if tv is not None and tg is not None:
            baseline_deltas.append(tv - tg)
    print(f"    mean delta (vargp - baseline_gpy) seed 42 = "
          f"{np.mean(baseline_deltas):+.4f}   n={len(baseline_deltas)}")


if __name__ == '__main__':
    main()
