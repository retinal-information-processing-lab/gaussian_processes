"""
Experiment 3: Paper-exact configuration.

Tests the configuration closest to the paper's setup:
- 108x108 images (paper's resolution)
- M=250 inducing points (paper's M)
- n_train=3160 (paper uses all available images)
- nEstep=50, nMstep=20 (paper's original defaults in varGP code)
- Ground-truth RF centers
- No early stopping

Also runs vargp_old with the same config for direct comparison.

Cells: 18, 14, 9, 28, 39
"""
import sys, os, json, subprocess, time, tempfile
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
from run_single_mode import build_config_from_defaults

CELLS = [18, 14, 9, 28, 39]
DATA_108 = os.path.join(PROJ, 'datasets', 'PNAS_108x108_original.npz')
RF_PATH = os.path.join(PROJ, 'datasets', 'rf_centers_ground_truth.npz')
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment3_results.jsonl')

rf = np.load(RF_PATH)

# Paper configuration: M=250, n_train=3160, nEstep=50, nMstep=20, 108x108
PAPER_OVERRIDES = dict(
    n_estep=50,
    n_mstep=20,
    n_iterations=50,
    n_train=3160,
    M=250,
)

GROUPS = [
    {'name': 'vargp_direct_paper', 'mode': 'vargp_direct', 'overrides': PAPER_OVERRIDES},
    {'name': 'vargp_old_paper', 'mode': 'vargp_old', 'overrides': PAPER_OVERRIDES},
]


def run_single_cell(cell_id, group_name, mode, overrides):
    eps_0x, eps_0y = rf['norm_108'][cell_id]

    config = build_config_from_defaults(
        mode=mode,
        M=overrides['M'],
        n_train=overrides['n_train'],
        seed=1,
        cell=cell_id,
        data_path=DATA_108,
        eps_0x=float(eps_0x),
        eps_0y=float(eps_0y),
        n_iterations=overrides['n_iterations'],
        n_estep=overrides['n_estep'],
        n_mstep=overrides['n_mstep'],
    )
    config['early_stop'] = False

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir='/tmp') as f:
        json.dump(config, f)
        tmp_path = f.name

    try:
        cmd = [sys.executable, os.path.join(PROJ, 'run_single_mode.py'), '--from-config', tmp_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        for line in proc.stdout.split('\n'):
            if line.startswith('RESULT_JSON:'):
                result = json.loads(line[len('RESULT_JSON:'):])
                result['group'] = group_name
                return result

        print(f"  ERROR: No RESULT_JSON for cell {cell_id}, group {group_name}")
        print(f"  STDOUT (last 500): {proc.stdout[-500:]}")
        print(f"  STDERR (last 500): {proc.stderr[-500:]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: cell {cell_id}, group {group_name}")
        return None
    finally:
        os.unlink(tmp_path)


def main():
    print("=== Experiment 3: Paper-Exact Configuration ===")
    print(f"Config: 108x108, M=250, n_train=3160, nEstep=50, nMstep=20")
    print(f"Cells: {CELLS}")
    print(f"Groups: {[g['name'] for g in GROUPS]}")
    print()

    total = len(CELLS) * len(GROUPS)
    done = 0

    for group in GROUPS:
        gname = group['name']
        print(f"\n--- Group: {gname} (mode={group['mode']}) ---")

        for cell_id in CELLS:
            done += 1
            print(f"  [{done}/{total}] Cell {cell_id}...", end=' ', flush=True)
            t0 = time.time()
            result = run_single_cell(cell_id, gname, group['mode'], group['overrides'])
            elapsed = time.time() - t0

            if result:
                adj_r2 = result.get('adjusted_r2', '?')
                test_r = result.get('test_r', '?')
                n_iter = result.get('n_iterations_run', '?')
                print(f"adj_r2={adj_r2:.4f}, test_r={test_r:.4f}, "
                      f"n_iter={n_iter}, time={elapsed:.1f}s")
                with open(RESULTS_FILE, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            else:
                print(f"FAILED ({elapsed:.1f}s)")

    # Summary
    print("\n\n=== SUMMARY ===")
    if not os.path.exists(RESULTS_FILE):
        return

    results = []
    with open(RESULTS_FILE) as f:
        for line in f:
            results.append(json.loads(line))

    print(f"\n{'Group':<25} {'Cell':>4} {'adj_r2':>8} {'test_r':>8} {'n_iter':>6} {'time_s':>7}")
    print("-" * 62)
    for r in results:
        print(f"{r['group']:<25} {r['cell']:>4} {r.get('adjusted_r2', 0):>8.4f} "
              f"{r.get('test_r', 0):>8.4f} {r.get('n_iterations_run', '?'):>6} "
              f"{r.get('train_time', 0):>7.1f}")

    group_names = [g['name'] for g in GROUPS]
    print(f"\n{'Group':<25} {'avg_adj_r2':>10} {'avg_test_r':>10}")
    print("-" * 48)
    for gname in group_names:
        gr = [r for r in results if r['group'] == gname]
        if gr:
            avg_a = np.mean([r['adjusted_r2'] for r in gr])
            avg_t = np.mean([r['test_r'] for r in gr])
            print(f"{gname:<25} {avg_a:>10.4f} {avg_t:>10.4f}")

    # Compare with baseline
    print(f"\n--- Comparison with baseline (vargp_direct 64x64, M=2910, 10/10/50) ---")
    baseline = {'18': 0.870, '14': 0.769, '9': 0.663, '28': 0.535, '39': 0.354}
    for gname in group_names:
        print(f"\n  {gname}:")
        for cell_id in CELLS:
            cr = [r for r in results if r['group'] == gname and r['cell'] == cell_id]
            if cr:
                new = cr[0]['adjusted_r2']
                old = baseline[str(cell_id)]
                diff = new - old
                print(f"    Cell {cell_id}: {old:.3f} -> {new:.3f} ({diff:+.3f})")


if __name__ == '__main__':
    main()
