"""Exp A — NGD on 108x108 images.

Goal: verify NGD works on the larger (11664-pixel) dataset that existing
sweeps never touched.

Grid: 41 cells x seed=1 = 41 runs (NGD only).
Reference: experiments/2026-03-20_massive_allcells_108/results.jsonl
           mode=vargp_direct, seed=1, M=300 (41 paired obs).
           vargp mean test_r ≈ 0.745 on 108x108.

Config matches the existing 108x108 vargp baseline:
  M=300, n_train=2910 (full pool on 108x108), fix_Amp=False
  (existing baseline trained Amp freely; final_Amp ≈ 2.1 — must match).

fix_Amp=False verified separately on 64x64 in Exp B before this runs.
"""
from __future__ import annotations
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np, torch

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))
from run_single_mode import build_config_from_defaults, run_single_config

# 108x108 default from build_config_from_defaults — no data_path override needed.
M        = 300
N_TRAIN  = 2910
SEED     = 1
FIX_AMP  = False    # matches existing vargp baseline (Amp was trained)
CELLS    = list(range(41))
RESULTS  = EXP_DIR / 'results_A.jsonl'


def _nan_to_none(x):
    try: return float(x) if np.isfinite(float(x)) else None
    except: return None


def run_one(cell):
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    config = build_config_from_defaults(
        mode='ngd', M=M, n_train=N_TRAIN,
        cell=cell, seed=SEED, ip_selection='random', fix_Amp=FIX_AMP,
        ngd_n_iterations=1500,
        # data_path intentionally omitted — defaults to PNAS_108x108_original.npz
    )
    t0 = time.time()
    try:
        result = run_single_config(config); err = None
    except Exception as e:
        traceback.print_exc(); result = None; err = repr(e)
    wall = time.time() - t0
    if result is None:
        return {'exp': 'A', 'cell': cell, 'seed': SEED, 'M': M, 'n_train': N_TRAIN,
                'fix_Amp': FIX_AMP, 'status': 'crashed', 'error': err, 'wall_time_s': wall}
    return {'exp': 'A', 'cell': cell, 'seed': SEED, 'M': M, 'n_train': N_TRAIN,
            'fix_Amp': FIX_AMP, 'status': result.get('status'), 'wall_time_s': wall,
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
            try: done.add(json.loads(l)['cell'])
            except: pass
    remaining = [c for c in CELLS if c not in done]
    total = len(CELLS)
    print(f"Exp A (NGD 108x108): {total} cells, {len(done)} done, "
          f"{len(remaining)} to run", flush=True)
    t0 = time.time()
    for i, cell in enumerate(remaining, 1):
        print(f"  [{len(done)+i}/{total}] cell={cell}", end='', flush=True)
        rec = run_one(cell)
        with open(RESULTS, 'a') as f: f.write(json.dumps(rec) + '\n')
        done.add(cell)
        tr = rec.get('test_r')
        print(f"  test_r={tr if tr is None else f'{tr:.4f}'}  "
              f"Amp={rec.get('final_Amp') or '?'}  wall={rec['wall_time_s']:.0f}s", flush=True)
    print(f"\nExp A done. {len(done)}/{total} in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == '__main__': main()
