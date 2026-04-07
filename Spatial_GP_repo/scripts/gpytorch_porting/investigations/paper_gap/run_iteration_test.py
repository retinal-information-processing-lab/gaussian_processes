"""
Experiment 1: Inner iteration count test.

Tests whether increasing n_estep (10 -> 50) and n_mstep (10 -> 20)
to match the paper's original varGP() defaults improves performance.

Cells: 18, 14, 9, 28, 39 (selected subset spanning adj_r2 range)
Configs: baseline (10/10/50), paper-like (50/20/50), extended (50/20/150)
"""
import sys, os, json, subprocess, time, tempfile
import numpy as np

# Project root
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
from run_single_mode import build_config_from_defaults

CELLS = [18, 14, 9, 28, 39]
DATA_PATH = os.path.join(PROJ, 'datasets', 'PNAS_64x64_center_crop_no_renorm.npz')
RF_PATH = os.path.join(PROJ, 'datasets', 'rf_centers_ground_truth.npz')
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment1_results.jsonl')

# Load ground-truth RF centers
rf = np.load(RF_PATH)

CONFIGS = {
    'baseline': dict(n_estep=10, n_mstep=10, n_iterations=50),
    'paper_like': dict(n_estep=50, n_mstep=20, n_iterations=50),
    'extended': dict(n_estep=50, n_mstep=20, n_iterations=150),
}


def run_single_cell(cell_id, config_name, overrides):
    """Run a single cell fit as subprocess. Returns result dict or None."""
    eps_0x, eps_0y = rf['norm_64'][cell_id]

    config = build_config_from_defaults(
        mode='vargp_direct',
        M=2910,
        n_train=2910,
        seed=1,
        cell=cell_id,
        data_path=DATA_PATH,
        eps_0x=float(eps_0x),
        eps_0y=float(eps_0y),
        n_iterations=overrides['n_iterations'],
        n_estep=overrides['n_estep'],
        n_mstep=overrides['n_mstep'],
    )
    # Disable early stopping for this experiment -- we want to see full convergence
    config['early_stopping_enabled'] = False

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir='/tmp') as f:
        json.dump(config, f)
        tmp_path = f.name

    try:
        cmd = [sys.executable, os.path.join(PROJ, 'run_single_mode.py'), '--from-config', tmp_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        # Parse RESULT_JSON line from stdout
        for line in proc.stdout.split('\n'):
            if line.startswith('RESULT_JSON:'):
                result = json.loads(line[len('RESULT_JSON:'):])
                result['config_name'] = config_name
                result['n_estep'] = overrides['n_estep']
                result['n_mstep'] = overrides['n_mstep']
                return result

        print(f"  ERROR: No RESULT_JSON found for cell {cell_id}, config {config_name}")
        print(f"  STDOUT: {proc.stdout[-500:]}")
        print(f"  STDERR: {proc.stderr[-500:]}")
        return None
    finally:
        os.unlink(tmp_path)


def main():
    print(f"=== Experiment 1: Inner Iteration Count Test ===")
    print(f"Cells: {CELLS}")
    print(f"Configs: {list(CONFIGS.keys())}")
    print(f"Results: {RESULTS_FILE}")
    print()

    total = len(CELLS) * len(CONFIGS)
    done = 0

    for config_name, overrides in CONFIGS.items():
        print(f"\n--- Config: {config_name} (n_estep={overrides['n_estep']}, "
              f"n_mstep={overrides['n_mstep']}, n_iter={overrides['n_iterations']}) ---")

        for cell_id in CELLS:
            done += 1
            print(f"  [{done}/{total}] Cell {cell_id}...", end=' ', flush=True)
            t0 = time.time()
            result = run_single_cell(cell_id, config_name, overrides)
            elapsed = time.time() - t0

            if result:
                adj_r2 = result.get('adjusted_r2', '?')
                test_r = result.get('test_r', '?')
                n_iter = result.get('n_iterations_run', '?')
                print(f"adj_r2={adj_r2:.4f}, test_r={test_r:.4f}, "
                      f"n_iter={n_iter}, time={elapsed:.1f}s")

                # Append to results file
                with open(RESULTS_FILE, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            else:
                print(f"FAILED ({elapsed:.1f}s)")

    # Print summary
    print("\n\n=== SUMMARY ===")
    if os.path.exists(RESULTS_FILE):
        results = []
        with open(RESULTS_FILE) as f:
            for line in f:
                results.append(json.loads(line))

        print(f"\n{'Config':<14} {'Cell':>4} {'adj_r2':>8} {'test_r':>8} {'n_iter':>6} {'time_s':>7}")
        print("-" * 55)
        for r in results:
            print(f"{r['config_name']:<14} {r['cell']:>4} {r.get('adjusted_r2', 0):>8.4f} "
                  f"{r.get('test_r', 0):>8.4f} {r.get('n_iterations_run', '?'):>6} "
                  f"{r.get('train_time', 0):>7.1f}")

        # Per-config averages
        print(f"\n{'Config':<14} {'avg_adj_r2':>10} {'avg_test_r':>10}")
        print("-" * 38)
        for cname in CONFIGS:
            cr = [r for r in results if r['config_name'] == cname]
            if cr:
                avg_a = np.mean([r['adjusted_r2'] for r in cr])
                avg_t = np.mean([r['test_r'] for r in cr])
                print(f"{cname:<14} {avg_a:>10.4f} {avg_t:>10.4f}")


if __name__ == '__main__':
    main()
