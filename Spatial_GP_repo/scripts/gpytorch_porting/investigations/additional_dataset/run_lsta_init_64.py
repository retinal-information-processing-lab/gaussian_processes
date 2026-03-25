#!/usr/bin/env python3
"""Run 64x64 vargp_direct M=2910 with LSTA-derived RF centers for all 41 cells x 3 seeds."""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.parent.parent
PYTHON = sys.executable
RESULTS_PATH = Path(__file__).parent / 'results_lsta_init_64.jsonl'

sys.path.insert(0, str(SCRIPT_DIR))
from run_single_mode import build_config_from_defaults

rf = np.load(SCRIPT_DIR / 'datasets' / 'rf_centers_lsta.npz')
norm_64 = rf['norm_64']  # (41, 2)

N_CELLS = 41
SEEDS = [1, 2, 3]
total = N_CELLS * len(SEEDS)

print(f"Running {total} fits: 41 cells x 3 seeds, vargp_direct M=2910, 64x64, LSTA RF init")
print(f"Results: {RESULTS_PATH}")
print()

t_start = time.time()
done = 0

with open(RESULTS_PATH, 'w') as results_file:
    for cell in range(N_CELLS):
        eps_x, eps_y = float(norm_64[cell, 0]), float(norm_64[cell, 1])
        for seed in SEEDS:
            done += 1
            config = build_config_from_defaults(
                mode='vargp_direct', M=2910, n_train=2910, seed=seed, cell=cell,
                data_path='datasets/PNAS_64x64_center_crop_no_renorm.npz',
                eps_0x=eps_x, eps_0y=eps_y)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                              dir=SCRIPT_DIR, delete=False) as f:
                json.dump(config, f)
                config_path = f.name

            try:
                proc = subprocess.run(
                    [PYTHON, str(SCRIPT_DIR / 'run_single_mode.py'), '--from-config', config_path],
                    capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=600)

                record = None
                for line in proc.stdout.splitlines():
                    if line.startswith('RESULT_JSON:'):
                        record = json.loads(line[len('RESULT_JSON:'):])
                        break

                if record:
                    record['init_eps_0x'] = eps_x
                    record['init_eps_0y'] = eps_y
                    results_file.write(json.dumps(record) + '\n')
                    results_file.flush()
                    tr = record.get('test_r')
                    tr_s = f'{tr:.4f}' if tr is not None else 'DIVERG'
                    print(f'  [{done}/{total}] cell={cell:2d} seed={seed} test_r={tr_s} time={record["train_time"]:.1f}s')
                else:
                    print(f'  [{done}/{total}] cell={cell:2d} seed={seed} FAILED (no result)')
                    fail = {'cell': cell, 'seed': seed, 'mode': 'vargp_direct', 'M': 2910,
                            'status': 'error', 'init_eps_0x': eps_x, 'init_eps_0y': eps_y}
                    results_file.write(json.dumps(fail) + '\n')
                    results_file.flush()
            finally:
                Path(config_path).unlink(missing_ok=True)

elapsed = time.time() - t_start
print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
