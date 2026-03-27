"""
Experiment 5: Paper-exact init + Amp fixed at 1.0.

The paper's code has NO Amp parameter. C = alpha * Csmooth * alpha.
Our code adds Amp: C = Amp * alpha * Csmooth * alpha.
Amp=1.0 frozen matches the paper's architecture.

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
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment5_results.jsonl')

rf = np.load(RF_PATH)

PAPER_INIT = dict(
    beta=0.0452,
    rho=0.0821,
    A_init=1e-4,
    lambda0_init=-1.0,
    n_iterations=80,
    n_estep=50,
    n_mstep=20,
    n_train=3160,
    M=250,
)


def run_single_cell(cell_id, group_name, mode, fix_amp):
    eps_0x, eps_0y = rf['norm_108'][cell_id]

    config = build_config_from_defaults(
        mode=mode,
        data_path=DATA_108,
        cell=cell_id,
        seed=1,
        eps_0x=float(eps_0x),
        eps_0y=float(eps_0y),
        **PAPER_INIT,
    )
    config['early_stop'] = False
    if fix_amp:
        config['fix_Amp'] = True

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

        print(f"  ERROR: No RESULT_JSON for cell {cell_id}, {group_name}")
        print(f"  STDOUT (last 800): {proc.stdout[-800:]}")
        print(f"  STDERR (last 300): {proc.stderr[-300:]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: cell {cell_id}, {group_name}")
        return None
    finally:
        os.unlink(tmp_path)


def main():
    print("=== Experiment 5: Paper Init + Amp Fixed ===")
    print(f"Cells: {CELLS}")
    print()

    groups = [
        {'name': 'direct_fix_amp', 'mode': 'vargp_direct', 'fix_amp': True},
    ]

    total = len(CELLS) * len(groups)
    done = 0

    for group in groups:
        gname = group['name']
        print(f"\n--- {gname} ---")

        for cell_id in CELLS:
            done += 1
            print(f"  [{done}/{total}] Cell {cell_id}...", end=' ', flush=True)
            t0 = time.time()
            result = run_single_cell(cell_id, gname, group['mode'], group['fix_amp'])
            elapsed = time.time() - t0

            if result:
                adj = result.get('adjusted_r2', 0)
                tr = result.get('test_r', 0)
                sig0 = result.get('final_sigma_0', 0)
                amp = result.get('final_Amp', 0)
                beta = result.get('final_beta', 0)
                A = result.get('final_A', 0)
                lam0 = result.get('final_lambda0', 0)
                n_iter = result.get('n_iterations_run', '?')
                print(f"adj_r2={adj:.4f}, test_r={tr:.4f}, "
                      f"sig0={sig0:.3f}, Amp={amp:.3f}, beta={beta:.4f}, A={A:.6f}, lam0={lam0:.2f}, "
                      f"n_iter={n_iter}, time={elapsed:.1f}s")
                with open(RESULTS_FILE, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            else:
                print(f"FAILED ({elapsed:.1f}s)")

    # Comparison
    print("\n\n=== COMPARISON ===")
    baseline = {'18': 0.870, '14': 0.769, '9': 0.663, '28': 0.535, '39': 0.354}
    exp4_direct = {'18': 0.871, '14': 0.810, '9': 0.773, '28': 0.573, '39': 0.215}

    if os.path.exists(RESULTS_FILE):
        results = []
        with open(RESULTS_FILE) as f:
            for line in f:
                results.append(json.loads(line))

        print(f"\n{'Cell':>4} {'baseline':>10} {'exp4(free)':>12} {'exp5(fixed)':>12} {'paper_tgt':>10}")
        print("-" * 52)
        adj_vals = []
        for r in sorted(results, key=lambda x: x['cell']):
            c = str(r['cell'])
            adj = r['adjusted_r2']
            adj_vals.append(adj)
            print(f"{r['cell']:>4} {baseline[c]:>10.4f} {exp4_direct[c]:>12.4f} {adj:>12.4f} {'> 0.8':>10}")
        print(f" Avg {np.mean(list(baseline.values())):>10.4f} "
              f"{np.mean(list(exp4_direct.values())):>12.4f} "
              f"{np.mean(adj_vals):>12.4f}")


if __name__ == '__main__':
    main()
