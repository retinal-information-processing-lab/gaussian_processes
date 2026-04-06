#!/usr/bin/env python
"""
Parameter sweep on 64x64 images (parallelized).

All 41 cells x 3 seeds x N configs.

Designed to run unattended in background:
  nohup python investigations/paper_gap/run_sweep_64x64.py > investigations/paper_gap/sweep_64x64.log 2>&1 &

RESUME-SAFE: If interrupted and restarted, skips already-completed runs
by checking results file for existing (config_name, cell, seed) tuples.

PARALLEL: Runs --workers cells simultaneously (default 4). Each is a separate
subprocess with its own GPU memory. RTX 4090 uses ~2.5 GB per cell.

TODO: No GPU memory isolation between workers. If one cell's memory usage spikes,
it can push total GPU usage over the limit and OOM ALL concurrent workers (CUDA
OOM is device-global). Possible fixes: CUDA_MEM_FRACTION per worker, or monitor
nvidia-smi and pause submissions when usage is high.

Results saved to:
  investigations/paper_gap/sweep_64x64_results.jsonl
"""
import sys, os, json, subprocess, time, tempfile, datetime, gc, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
from run_single_mode import build_config_from_defaults

CELLS = list(range(41))
SEEDS = [1, 2, 3]
DATA_64 = os.path.join(PROJ, 'datasets', 'PNAS_64x64_center_crop_no_renorm.npz')
RF_PATH = os.path.join(PROJ, 'datasets', 'rf_centers_ground_truth.npz')
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_64x64_results_ntrain3160.jsonl')

rf = np.load(RF_PATH)

# Lock for thread-safe JSONL writes
_write_lock = threading.Lock()

# =========================================================================
# Sweep configurations
# =========================================================================
CONFIGS = [
    {
        'name': '64_intl_freeAmp_n3160',
        'notes': '64x64 free Amp + interleaved, A=1e-4. Tests if free Amp helps/hurts with interleaving.',
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
        n_train=3160,  # train+val combined (same as 108x108 sweep)
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
    # Config flags
    config['early_stop'] = False
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
    """Thread-safe append of a result to the JSONL file."""
    with _write_lock:
        with open(RESULTS_FILE, 'a') as f:
            f.write(json.dumps(result) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    args = parser.parse_args()

    start_time = time.time()
    total = len(CONFIGS) * len(CELLS) * len(SEEDS)

    print(f"=== Parameter Sweep (64x64, {args.workers} workers) ===")
    print(f"  {len(CONFIGS)} configs x {len(CELLS)} cells x {len(SEEDS)} seeds = {total} runs")
    print(f"  Started: {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"  Results: {RESULTS_FILE}")

    completed = load_completed_runs()
    if completed:
        print(f"  Resuming: {len(completed)} runs already completed, {total - len(completed)} remaining")
    print(flush=True)

    # Build list of jobs to run
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

    print(f"  Submitting {len(jobs)} jobs to {args.workers} workers\n", flush=True)

    done = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {}
        for cell_id, seed, cfg in jobs:
            future = executor.submit(run_single_cell, cell_id, seed, cfg)
            future_to_job[future] = (cell_id, seed, cfg)

        for future in as_completed(future_to_job):
            cell_id, seed, cfg = future_to_job[future]
            done += 1

            try:
                result = future.result()
            except Exception as e:
                print(f"  [{done}/{len(jobs)}] {cfg['name']} cell={cell_id} seed={seed} EXCEPTION: {e}",
                      flush=True)
                failed += 1
                continue

            if result:
                adj = result.get('adjusted_r2', 0)
                tr = result.get('test_r', 0)
                tt = result.get('train_time', 0)
                print(f"  [{done}/{len(jobs)}] {cfg['name']} cell={cell_id} seed={seed} "
                      f"adj_r2={adj:.4f} test_r={tr:.4f} time={tt:.1f}s", flush=True)
                append_result(result)
            else:
                failed += 1
                print(f"  [{done}/{len(jobs)}] {cfg['name']} cell={cell_id} seed={seed} FAILED",
                      flush=True)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Sweep finished in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Completed: {done - failed}, Failed: {failed}")


if __name__ == '__main__':
    main()
