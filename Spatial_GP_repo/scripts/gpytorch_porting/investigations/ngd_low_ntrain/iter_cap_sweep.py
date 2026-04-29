"""Direct test of overfitting: rerun cells at varying iter caps.

If hypothesis correct: hard cells (35, 0) should peak test_r at low iter cap,
degrade with high cap. Easy cells (16) should improve monotonically.

Cells: 35 (NGD disaster), 0 (NGD disaster), 16 (NGD healthy).
Caps: 100, 300, 500, 1500. ES is DISABLED — runs to cap exactly.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from run_single_mode import build_config_from_defaults, run_single_config

CELLS = [0, 35, 16]
CAPS  = [100, 300, 500, 1500]
RESULTS = EXP_DIR / 'iter_cap_sweep.jsonl'


def run_one(cell, cap):
    os.chdir(ROOT)
    config = build_config_from_defaults(
        mode='ngd',
        data_path='datasets/PNAS_64x64_center_crop_no_renorm.npz',
        M=50, n_train=50, cell=cell, seed=1,
        ip_selection='random', fix_Amp=True,
        ngd_n_iterations=cap,
    )
    config['early_stop'] = False  # disable ES — run exactly `cap` iters
    t0 = time.time()
    result = run_single_config(config)
    return {
        'cell': cell, 'cap': cap,
        'test_r': result.get('test_r'),
        'final_A': result.get('final_A'),
        'final_beta': result.get('final_beta'),
        'n_iter': result.get('n_iterations_run'),
        'wall': time.time() - t0,
    }


def main():
    done = set()
    if RESULTS.exists():
        for l in open(RESULTS):
            try:
                r = json.loads(l); done.add((r['cell'], r['cap']))
            except: pass

    for cell in CELLS:
        for cap in CAPS:
            if (cell, cap) in done:
                continue
            print(f"\n=== cell={cell} cap={cap} ===", flush=True)
            rec = run_one(cell, cap)
            with open(RESULTS, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            tr = rec['test_r']
            print(f"  test_r={tr:.4f}  A={rec['final_A']:.4f}  wall={rec['wall']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
