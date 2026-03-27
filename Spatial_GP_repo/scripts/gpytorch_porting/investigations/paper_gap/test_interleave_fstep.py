"""
Test: Interleaved damped Newton F-step vs current approach.

Compares three configs on 5-cell subset:
1. vargp_direct, interleave_fstep=False (current)
2. vargp_direct, interleave_fstep=True (new)
3. vargp_old (reference from experiment 3)

All use paper config: 108x108, M=250, n_train=3160, nEstep=50, nMstep=20,
paper init (beta=0.0452, rho=0.0821, A=1e-4, lambda0=-1), fix_Amp=True.

Also runs with our A=0.01 init for comparison.
"""
import sys, os, json, subprocess, time, tempfile
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
from run_single_mode import build_config_from_defaults

CELLS = [18, 14, 9, 28, 39]
DATA_108 = os.path.join(PROJ, 'datasets', 'PNAS_108x108_original.npz')
RF_PATH = os.path.join(PROJ, 'datasets', 'rf_centers_ground_truth.npz')
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment7_results.jsonl')

rf = np.load(RF_PATH)

PAPER_BASE = dict(
    n_estep=50, n_mstep=20, n_iterations=80,
    n_train=3160, M=250,
    beta=0.0452, rho=0.0821,
)

GROUPS = [
    # Paper A=1e-4 configs
    {'name': 'direct_A1e4',      'mode': 'vargp_direct', 'A_init': 1e-4, 'interleave': False},
    {'name': 'direct_A1e4_intl', 'mode': 'vargp_direct', 'A_init': 1e-4, 'interleave': True},
    {'name': 'old_A1e4',         'mode': 'vargp_old',    'A_init': 1e-4, 'interleave': False},
    # Our A=0.01 configs
    {'name': 'direct_A01',       'mode': 'vargp_direct', 'A_init': 0.01, 'interleave': False},
    {'name': 'direct_A01_intl',  'mode': 'vargp_direct', 'A_init': 0.01, 'interleave': True},
]


def run_cell(cell_id, group):
    eps_0x, eps_0y = rf['norm_108'][cell_id]
    config = build_config_from_defaults(
        mode=group['mode'], data_path=DATA_108, cell=cell_id, seed=1,
        eps_0x=float(eps_0x), eps_0y=float(eps_0y),
        A_init=group['A_init'], lambda0_init=-1.0,
        **PAPER_BASE,
    )
    config['early_stop'] = False
    config['fix_Amp'] = True
    if group['interleave']:
        config['interleave_fstep'] = True

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir='/tmp') as f:
        json.dump(config, f)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(PROJ, 'run_single_mode.py'), '--from-config', tmp_path],
            capture_output=True, text=True, timeout=600)
        for line in proc.stdout.split('\n'):
            if line.startswith('RESULT_JSON:'):
                r = json.loads(line[len('RESULT_JSON:'):])
                r['group'] = group['name']
                return r
        print(f"    ERROR: {proc.stderr[-200:]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT")
        return None
    finally:
        os.unlink(tmp_path)


def main():
    print("=== Experiment 7: Interleaved Damped Newton F-step ===\n")

    # Clean results file
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    total = len(CELLS) * len(GROUPS)
    done = 0

    for group in GROUPS:
        print(f"\n--- {group['name']} (mode={group['mode']}, A={group['A_init']}, "
              f"interleave={group['interleave']}) ---")
        for cell_id in CELLS:
            done += 1
            print(f"  [{done}/{total}] Cell {cell_id}...", end=' ', flush=True)
            t0 = time.time()
            result = run_cell(cell_id, group)
            elapsed = time.time() - t0
            if result:
                print(f"adj_r2={result['adjusted_r2']:.4f}, test_r={result['test_r']:.4f}, "
                      f"A={result['final_A']:.6f}, sig0={result['final_sigma_0']:.3f}, "
                      f"beta={result['final_beta']:.4f}, time={elapsed:.1f}s")
                with open(RESULTS_FILE, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            else:
                print(f"FAILED ({elapsed:.1f}s)")

    # Summary
    print("\n\n=== SUMMARY ===")
    results = []
    with open(RESULTS_FILE) as f:
        for line in f:
            results.append(json.loads(line))

    group_names = [g['name'] for g in GROUPS]
    print(f"\n{'Group':<22} | {'C18':>6} {'C14':>6} {'C9':>6} {'C28':>6} {'C39':>6} | {'Avg':>6}")
    print("-" * 72)
    for gname in group_names:
        gr = [r for r in results if r['group'] == gname]
        if not gr:
            continue
        vals = {r['cell']: r['adjusted_r2'] for r in gr}
        avg = np.mean(list(vals.values()))
        print(f"{gname:<22} | {vals.get(18,0):>6.3f} {vals.get(14,0):>6.3f} "
              f"{vals.get(9,0):>6.3f} {vals.get(28,0):>6.3f} {vals.get(39,0):>6.3f} | {avg:>6.3f}")

    print(f"\n{'Baseline (64,defaults)':<22} | {0.870:>6.3f} {0.769:>6.3f} "
          f"{0.663:>6.3f} {0.535:>6.3f} {0.354:>6.3f} | {0.638:>6.3f}")
    print(f"{'Paper target':<22} | {'>0.8':>6} {'>0.8':>6} "
          f"{'>0.8':>6} {'?':>6} {'?':>6} | {'36/41':>6}")


if __name__ == '__main__':
    main()
