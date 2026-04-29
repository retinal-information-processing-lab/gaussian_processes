"""Exp C — Low n_train regime (M = n_train) on 64x64, both modes.

Goal: check stability of vargp_direct and NGD+Adam in the sparse-data
regime relevant to active learning (n_train ≈ M, not n_train >> M).

Grid: 41 cells x seed=1 x 3 (M,n_train) pairs x 2 modes = 246 runs.
  (M=50,  n_train=50 )
  (M=150, n_train=150)
  (M=300, n_train=300)

fix_Amp=True for both modes (matches Phase 3C for clean comparison;
free-Amp is tested separately in Exp B).

No existing baseline — both modes run fresh; compared against each other.

Note: with M=n_train, ip_selection='random' selects all training points
as inducing points (exact variational inference at the inducing level).
ES patience unchanged (vargp=15, NGD=200) — testing default config behaviour
at low n_train, not retuning ES.
"""
from __future__ import annotations
import datetime, json, os, sys, time, traceback
from pathlib import Path
import numpy as np, torch

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))
from run_single_mode import build_config_from_defaults, run_single_config

DATA_PATH = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
SEED      = 1
FIX_AMP   = True
CELLS     = list(range(41))
GRID      = [(50, 50), (150, 150), (300, 300)]   # (M, n_train)
MODES     = ['vargp_direct', 'ngd']
RESULTS   = EXP_DIR / 'results_C.jsonl'


def _nan_to_none(x):
    try: return float(x) if np.isfinite(float(x)) else None
    except: return None


def run_one(mode, cell, M, n_train):
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    config = build_config_from_defaults(
        mode=mode, data_path=DATA_PATH, M=M, n_train=n_train,
        cell=cell, seed=SEED, ip_selection='random', fix_Amp=FIX_AMP,
        **(dict(ngd_n_iterations=1500) if mode == 'ngd' else {}),
    )
    t0 = time.time()
    try:
        result = run_single_config(config); err = None
    except Exception as e:
        traceback.print_exc(); result = None; err = repr(e)
    wall = time.time() - t0
    if result is None:
        return {'exp': 'C', 'mode': mode, 'cell': cell, 'seed': SEED,
                'M': M, 'n_train': n_train, 'fix_Amp': FIX_AMP,
                'status': 'crashed', 'error': err, 'wall_time_s': wall}
    return {'exp': 'C', 'mode': mode, 'cell': cell, 'seed': SEED,
            'M': M, 'n_train': n_train, 'fix_Amp': FIX_AMP,
            'status': result.get('status'), 'wall_time_s': wall,
            'test_r': _nan_to_none(result.get('test_r')),
            'explained_var': _nan_to_none(result.get('explained_var')),
            'final_A': result.get('final_A'), 'final_Amp': result.get('final_Amp'),
            'final_beta': result.get('final_beta'),
            'n_iterations_run': result.get('n_iterations_run'),
            'stopped_early': result.get('stopped_early')}


def main():
    os.chdir(ROOT)
    done = set()
    if RESULTS.exists():
        for l in open(RESULTS):
            try:
                r = json.loads(l)
                done.add((r['mode'], r['cell'], r['M'], r['n_train']))
            except: pass

    all_runs = [(mode, cell, M, n_train)
                for (M, n_train) in GRID
                for mode in MODES
                for cell in CELLS]
    remaining = [r for r in all_runs if r not in done]
    total = len(all_runs)
    print(f"Exp C (low n_train, 64x64): {total} total, {len(done)} done, "
          f"{len(remaining)} to run", flush=True)

    t0 = time.time()
    for i, (mode, cell, M, n_train) in enumerate(remaining, 1):
        print(f"  [{len(done)+i}/{total}] {mode} cell={cell} M={M} n_train={n_train}",
              end='', flush=True)
        rec = run_one(mode, cell, M, n_train)
        with open(RESULTS, 'a') as f: f.write(json.dumps(rec) + '\n')
        done.add((mode, cell, M, n_train))
        tr = rec.get('test_r')
        print(f"  test_r={tr if tr is None else f'{tr:.4f}'}  wall={rec['wall_time_s']:.0f}s",
              flush=True)

    print(f"\nExp C done. {len(done)}/{total} in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == '__main__': main()
