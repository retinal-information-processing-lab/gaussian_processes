#!/usr/bin/env python
"""
64x64 sweep WITH early stopping (fixed M-step timing, val_from_train).

Same 3 configs x 41 cells x 3 seeds as sweep_64x64_results_ntrain3160.jsonl,
but with early stopping enabled. val_from_train=True carves 250 images from
training as validation (effective n_train=2910 vs 3160 without ES).

Run in background:
  nohup python investigations/paper_gap/run_sweep_64x64_es.py > investigations/paper_gap/sweep_64x64_es.log 2>&1 &

RESUME-SAFE: skips already-completed (config_name, cell, seed) tuples.

Results saved to:
  investigations/paper_gap/sweep_64x64_es_results.jsonl
"""
import sys, os, json, subprocess, time, tempfile, datetime, argparse
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
from run_single_mode import build_config_from_defaults

CELLS = list(range(41))
SEEDS = [1, 2, 3]
DATA_64 = os.path.join(PROJ, 'datasets', 'PNAS_64x64_center_crop_no_renorm.npz')
RF_PATH = os.path.join(PROJ, 'datasets', 'rf_centers_ground_truth.npz')
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_64x64_es_results.jsonl')

rf = np.load(RF_PATH)

# =========================================================================
# Same 3 configs as sweep_64x64_results_ntrain3160.jsonl (no-ES reference)
# =========================================================================
CONFIGS = [
    {
        'name': '64_A01_free_no_intl_n3160_es',
        'notes': '64x64 free Amp, no interleaving, A=0.01. ES version of 64_A01_free_no_intl_n3160.',
        'beta': 0.1, 'A_init': 0.01, 'interleave_fstep': False,
        'fix_Amp': False,
        'n_estep': 50, 'n_mstep': 20, 'n_iterations': 80,
    },
    {
        'name': '64_intl_fixAmp_n3160_es',
        'notes': '64x64 fix Amp=1, interleaved, A=1e-4. ES version of 64_intl_fixAmp_n3160.',
        'beta': 0.1, 'A_init': 1e-4, 'interleave_fstep': True,
        'fix_Amp': True,
        'n_estep': 50, 'n_mstep': 20, 'n_iterations': 80,
    },
    {
        'name': '64_intl_freeAmp_n3160_es',
        'notes': '64x64 free Amp, interleaved, A=1e-4. ES version of 64_intl_freeAmp_n3160.',
        'beta': 0.1, 'A_init': 1e-4, 'interleave_fstep': True,
        'fix_Amp': False,
        'n_estep': 50, 'n_mstep': 20, 'n_iterations': 80,
    },
]


def load_completed_runs():
    """Load already-completed (config_name, cell, seed) tuples from results file."""
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
    """Run a single cell fit as subprocess. Returns result dict or None."""
    eps_0x, eps_0y = rf['norm_64'][cell_id]

    config = build_config_from_defaults(
        mode='vargp_direct',
        M=250,
        n_train=3160,
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
    )
    # Early stopping + val_from_train (the fix under test)
    config['early_stop'] = True
    config['val_from_train'] = True
    config['ip_selection'] = 'random'
    config['fix_Amp'] = cfg['fix_Amp']
    if cfg['interleave_fstep']:
        config['interleave_fstep'] = True

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir='/tmp') as f:
        json.dump(config, f)
        tmp_path = f.name

    try:
        cmd = [sys.executable, os.path.join(PROJ, 'run_single_mode.py'), '--from-config', tmp_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        for line in proc.stdout.split('\n'):
            if line.startswith('RESULT_JSON:'):
                result = json.loads(line[len('RESULT_JSON:'):])
                result['config_name'] = cfg['name']
                return result

        print(f"    ERROR: No RESULT_JSON for cell {cell_id} seed {seed} config {cfg['name']}",
              flush=True)
        if proc.stderr:
            print(f"    STDERR: {proc.stderr[-300:]}", flush=True)
        return None
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT: cell {cell_id} seed {seed} config {cfg['name']} (>600s)", flush=True)
        return None
    except Exception as e:
        print(f"    EXCEPTION: cell {cell_id} seed {seed} config {cfg['name']}: {e}", flush=True)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def append_result(result):
    """Append a result to the JSONL file."""
    with open(RESULTS_FILE, 'a') as f:
        f.write(json.dumps(result) + '\n')


def main():
    start_time = time.time()
    total = len(CONFIGS) * len(CELLS) * len(SEEDS)

    print(f"=== Early Stopping Sweep (64x64, sequential) ===")
    print(f"  {len(CONFIGS)} configs x {len(CELLS)} cells x {len(SEEDS)} seeds = {total} runs")
    print(f"  Early stopping: patience=15, min_iter=10, val_from_train=True")
    print(f"  Started: {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"  Results: {RESULTS_FILE}")

    completed = load_completed_runs()
    if completed:
        print(f"  Resuming: {len(completed)} runs already completed, {total - len(completed)} remaining")
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
            bi = result.get('best_iter', '?')
            print(f"  [{done}/{len(jobs)}] {cfg['name']} cell={cell_id} seed={seed} "
                  f"test_r={tr:.4f} iters={ni} best={bi} es={se} time={tt:.1f}s", flush=True)
            append_result(result)
        else:
            failed += 1
            print(f"  [{done}/{len(jobs)}] {cfg['name']} cell={cell_id} seed={seed} FAILED",
                  flush=True)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Sweep finished in {total_time:.0f}s ({total_time/3600:.1f} hours)")
    print(f"  Completed: {done - failed}, Failed: {failed}")


if __name__ == '__main__':
    main()
