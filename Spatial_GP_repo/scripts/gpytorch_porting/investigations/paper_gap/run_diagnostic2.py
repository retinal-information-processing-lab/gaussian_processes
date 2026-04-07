"""
Experiment 2: Cross-mode and cross-resolution diagnostics.

Two groups:
A) vargp_old mode on 64x64 — tests whether the original varGP code path
   reproduces paper-level performance on the same data the gpytorch port uses.
   M=n_train=2910 (all training images are inducing points), ground-truth RF
   centers, n_estep=10, n_mstep=10, seed=1.

B) vargp_direct mode on 108x108 — tests whether the full-resolution images
   (the paper's original resolution) yield better performance.
   M=n_train=2910, ground-truth RF centers (norm_108), seed=1.

Cells: 18, 14, 9, 28, 39 (same subset as experiment 1).

Note on vargp_old and M=n_train: the code path uses random inducing point
selection with ntilde = min(M, n_train). When M == n_train, all training points
become inducing points, which is valid (full GP limit).
"""
import sys, os, json, subprocess, time, tempfile
import numpy as np

# Project root (gpytorch_porting/)
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
from run_single_mode import build_config_from_defaults

CELLS = [18, 14, 9, 28, 39]
DATA_64 = os.path.join(PROJ, 'datasets', 'PNAS_64x64_center_crop_no_renorm.npz')
DATA_108 = os.path.join(PROJ, 'datasets', 'PNAS_108x108_original.npz')
RF_PATH = os.path.join(PROJ, 'datasets', 'rf_centers_ground_truth.npz')
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment2_results.jsonl')

# Load ground-truth RF centers
rf = np.load(RF_PATH)


def run_single_cell(cell_id, group_name, mode, data_path, rf_key, extra_overrides=None):
    """Run a single cell fit as subprocess. Returns result dict or None."""
    eps_0x, eps_0y = rf[rf_key][cell_id]

    config = build_config_from_defaults(
        mode=mode,
        M=2910,
        n_train=2910,
        seed=1,
        cell=cell_id,
        data_path=data_path,
        eps_0x=float(eps_0x),
        eps_0y=float(eps_0y),
    )
    # Disable early stopping (correct flat config key)
    config['early_stop'] = False

    # Apply any group-specific overrides
    if extra_overrides:
        config.update(extra_overrides)

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir='/tmp') as f:
        json.dump(config, f)
        tmp_path = f.name

    try:
        cmd = [sys.executable, os.path.join(PROJ, 'run_single_mode.py'), '--from-config', tmp_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

        # Parse RESULT_JSON line from stdout
        for line in proc.stdout.split('\n'):
            if line.startswith('RESULT_JSON:'):
                result = json.loads(line[len('RESULT_JSON:'):])
                result['group'] = group_name
                return result

        print(f"  ERROR: No RESULT_JSON found for cell {cell_id}, group {group_name}")
        print(f"  STDOUT (last 500): {proc.stdout[-500:]}")
        print(f"  STDERR (last 500): {proc.stderr[-500:]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: cell {cell_id}, group {group_name} (>1200s)")
        return None
    finally:
        os.unlink(tmp_path)


# =========================================================================
# Group definitions
# =========================================================================
# Group A: vargp_old on 64x64
# n_estep=10, n_mstep=10 are already defaults from default_params.json
# n_iterations=50 is already the default
GROUP_A = {
    'name': 'vargp_old_64',
    'mode': 'vargp_old',
    'data_path': DATA_64,
    'rf_key': 'norm_64',
    'overrides': {},  # defaults are already n_estep=10, n_mstep=10, n_iter=50
}

# Group B: vargp_direct on 108x108
GROUP_B = {
    'name': 'vargp_direct_108',
    'mode': 'vargp_direct',
    'data_path': DATA_108,
    'rf_key': 'norm_108',
    'overrides': {},
}

GROUPS = [GROUP_A, GROUP_B]


def main():
    print("=== Experiment 2: Cross-Mode & Cross-Resolution Diagnostics ===")
    print(f"Cells: {CELLS}")
    print(f"Groups: {[g['name'] for g in GROUPS]}")
    print(f"All groups: M=2910, n_train=2910, seed=1, early_stop=False")
    print(f"Results: {RESULTS_FILE}")
    print()

    total = len(CELLS) * len(GROUPS)
    done = 0

    for group in GROUPS:
        gname = group['name']
        print(f"\n--- Group: {gname} (mode={group['mode']}, "
              f"data={os.path.basename(group['data_path'])}) ---")

        for cell_id in CELLS:
            done += 1
            print(f"  [{done}/{total}] Cell {cell_id}...", end=' ', flush=True)
            t0 = time.time()
            result = run_single_cell(
                cell_id,
                group_name=gname,
                mode=group['mode'],
                data_path=group['data_path'],
                rf_key=group['rf_key'],
                extra_overrides=group['overrides'],
            )
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

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n\n=== SUMMARY ===")
    if not os.path.exists(RESULTS_FILE):
        print("No results file found.")
        return

    results = []
    with open(RESULTS_FILE) as f:
        for line in f:
            results.append(json.loads(line))

    # Per-cell table
    print(f"\n{'Group':<22} {'Cell':>4} {'adj_r2':>8} {'test_r':>8} {'n_iter':>6} {'time_s':>7}")
    print("-" * 60)
    for r in results:
        print(f"{r['group']:<22} {r['cell']:>4} {r.get('adjusted_r2', 0):>8.4f} "
              f"{r.get('test_r', 0):>8.4f} {r.get('n_iterations_run', '?'):>6} "
              f"{r.get('train_time', 0):>7.1f}")

    # Per-group averages
    group_names = [g['name'] for g in GROUPS]
    print(f"\n{'Group':<22} {'avg_adj_r2':>10} {'avg_test_r':>10}")
    print("-" * 45)
    for gname in group_names:
        gr = [r for r in results if r['group'] == gname]
        if gr:
            avg_a = np.mean([r['adjusted_r2'] for r in gr if r.get('adjusted_r2') is not None])
            avg_t = np.mean([r['test_r'] for r in gr if r.get('test_r') is not None])
            print(f"{gname:<22} {avg_a:>10.4f} {avg_t:>10.4f}")

    # Side-by-side comparison per cell
    print(f"\n--- Per-Cell Comparison ---")
    print(f"{'Cell':>4}  ", end='')
    for gname in group_names:
        print(f"  {gname:>18}", end='')
    print()
    print("-" * (6 + 20 * len(group_names)))
    for cell_id in CELLS:
        print(f"{cell_id:>4}  ", end='')
        for gname in group_names:
            cr = [r for r in results if r['group'] == gname and r['cell'] == cell_id]
            if cr:
                print(f"  {cr[0].get('test_r', 0):>18.4f}", end='')
            else:
                print(f"  {'---':>18}", end='')
        print()


if __name__ == '__main__':
    main()
