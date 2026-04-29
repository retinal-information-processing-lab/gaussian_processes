"""Exp B — NGD with free Amp on 64x64.

Goal: verify NGD works correctly with Amp trainable (fix_Amp=False).
Phase 3C and M-sweep both froze Amp=1.0. This checks that the frozen-Amp
choice was not hiding a pathology.

Grid: 41 cells x seed=1 = 41 runs (NGD only).
Reference: experiments/2026-04-06_es_sweeps_64x64/sweep_64x64_elbo_es_results.jsonl
           config_name='64_elbo_intl_freeAmp_Amp1', seed=1 (41 paired obs).
           vargp mean test_r = 0.836 on that config.
"""
from __future__ import annotations
import datetime, json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np, torch

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))
from run_single_mode import build_config_from_defaults, run_single_config

DATA_PATH  = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
M          = 250
N_TRAIN    = 3160
SEED       = 1
FIX_AMP    = False       # <-- the thing being tested
CELLS      = list(range(41))
RESULTS    = EXP_DIR / 'results_B.jsonl'


def _nan_to_none(x):
    try: return float(x) if np.isfinite(float(x)) else None
    except: return None


def run_one(cell):
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    config = build_config_from_defaults(
        mode='ngd', data_path=DATA_PATH, M=M, n_train=N_TRAIN,
        cell=cell, seed=SEED, ip_selection='random', fix_Amp=FIX_AMP,
        ngd_n_iterations=1500,
    )
    t0 = time.time()
    try:
        result = run_single_config(config); err = None
    except Exception as e:
        traceback.print_exc(); result = None; err = repr(e)
    wall = time.time() - t0
    if result is None:
        return {'exp': 'B', 'cell': cell, 'seed': SEED, 'M': M,
                'fix_Amp': FIX_AMP, 'status': 'crashed', 'error': err, 'wall_time_s': wall}
    return {'exp': 'B', 'cell': cell, 'seed': SEED, 'M': M, 'fix_Amp': FIX_AMP,
            'status': result.get('status'), 'wall_time_s': wall,
            'test_r': _nan_to_none(result.get('test_r')),
            'explained_var': _nan_to_none(result.get('explained_var')),
            'final_A': result.get('final_A'), 'final_Amp': result.get('final_Amp'),
            'final_beta': result.get('final_beta'), 'n_iterations_run': result.get('n_iterations_run'),
            'stopped_early': result.get('stopped_early')}


def main():
    os.chdir(ROOT)
    done = set()
    if RESULTS.exists():
        for l in open(RESULTS):
            try: done.add(json.loads(l)['cell'])
            except: pass
    total = len(CELLS); remaining = [c for c in CELLS if c not in done]
    print(f"Exp B (NGD freeAmp 64x64): {total} cells, {len(done)} already done, "
          f"{len(remaining)} to run", flush=True)
    t0 = time.time()
    for i, cell in enumerate(remaining, 1):
        print(f"  [{len(done)+i}/{total}] cell={cell}", end='', flush=True)
        rec = run_one(cell)
        with open(RESULTS, 'a') as f: f.write(json.dumps(rec) + '\n')
        done.add(cell)
        tr = rec.get('test_r')
        print(f"  test_r={tr if tr is None else f'{tr:.4f}'}  "
              f"Amp={rec.get('final_Amp', '?'):.3f}  wall={rec['wall_time_s']:.0f}s", flush=True)
    print(f"\nExp B done. {len(done)}/{total} in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == '__main__': main()
