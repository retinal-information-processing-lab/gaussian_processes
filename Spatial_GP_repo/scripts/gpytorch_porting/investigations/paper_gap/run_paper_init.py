"""
Experiment 4: Paper-exact initialization values.

The paper's GitHub code reveals very different default init values:
  beta_nat:  0.0452  (ours: 0.1)  -- paper's RF is 2.2x tighter
  rho_nat:   0.0821  (ours: 0.1)  -- paper's smoothness is 1.2x tighter
  A_init:    1e-4    (ours: 0.01) -- paper starts 100x smaller
  lambda0:   -1      (ours: +1)   -- opposite sign
  maxiter:   80      (ours: 50)   -- 60% more iterations

Also includes the softplus->exp fix for sigma_0 and Amp.

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
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment4_results.jsonl')

rf = np.load(RF_PATH)

# Paper-exact init values (verified parameterization mapping)
# Paper: 0.5*exp(5.5) = 122.35 in exponent -> beta_nat = 0.0452
# Paper: 0.5*exp(5.0) = 74.2 in exponent -> rho_nat = 0.0821
PAPER_INIT = dict(
    beta=0.0452,       # paper: exp(5.5)=244.7, equiv beta_nat=0.0452 (RF sigma ~3.5px)
    rho=0.0821,        # paper: exp(5.0)=148.4, equiv rho_nat=0.0821
    A_init=1e-4,       # paper: A=1e-4 (vs our 0.01)
    lambda0_init=-1.0, # paper: lambda0=-1 (vs our +1)
    n_iterations=80,   # paper: maxiter=80
    n_estep=50,        # paper: nEstep=50
    n_mstep=20,        # paper: nMstep=20
    n_train=3160,      # paper: uses all available images
    M=250,             # paper: ntilde=250
)

GROUPS = [
    {'name': 'direct_paper_init', 'mode': 'vargp_direct'},
    {'name': 'old_paper_init', 'mode': 'vargp_old'},
]


def run_single_cell(cell_id, group_name, mode):
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
    print("=== Experiment 4: Paper-Exact Initialization ===")
    print(f"Paper init: beta={PAPER_INIT['beta']}, rho={PAPER_INIT['rho']}, "
          f"A={PAPER_INIT['A_init']}, lam0={PAPER_INIT['lambda0_init']}")
    print(f"Schedule: iter={PAPER_INIT['n_iterations']}, estep={PAPER_INIT['n_estep']}, "
          f"mstep={PAPER_INIT['n_mstep']}, M={PAPER_INIT['M']}, n_train={PAPER_INIT['n_train']}")
    print(f"Cells: {CELLS}")
    print()

    total = len(CELLS) * len(GROUPS)
    done = 0

    for group in GROUPS:
        gname = group['name']
        print(f"\n--- {gname} (mode={group['mode']}) ---")

        for cell_id in CELLS:
            done += 1
            print(f"  [{done}/{total}] Cell {cell_id}...", end=' ', flush=True)
            t0 = time.time()
            result = run_single_cell(cell_id, gname, group['mode'])
            elapsed = time.time() - t0

            if result:
                adj = result.get('adjusted_r2', 0)
                tr = result.get('test_r', 0)
                sig0 = result.get('final_sigma_0', 0)
                amp = result.get('final_Amp', 0)
                beta = result.get('final_beta', 0)
                A = result.get('final_A', 0)
                n_iter = result.get('n_iterations_run', '?')
                print(f"adj_r2={adj:.4f}, test_r={tr:.4f}, "
                      f"sig0={sig0:.3f}, Amp={amp:.3f}, beta={beta:.4f}, A={A:.6f}, "
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

    # Comparison with previous experiments
    baseline_64 = {'18': 0.870, '14': 0.769, '9': 0.663, '28': 0.535, '39': 0.354}
    old_paper = {'18': 0.857, '14': 0.835, '9': 0.756, '28': 0.524, '39': 0.421}

    for gname in [g['name'] for g in GROUPS]:
        gr = [r for r in results if r['group'] == gname]
        if not gr:
            continue
        print(f"\n--- {gname} ---")
        print(f"{'Cell':>4} {'adj_r2':>8} {'vs baseline':>12} {'vs old_paper':>12}")
        print("-" * 42)
        adj_vals = []
        for r in sorted(gr, key=lambda x: x['cell']):
            adj = r['adjusted_r2']
            adj_vals.append(adj)
            bl = baseline_64[str(r['cell'])]
            op = old_paper[str(r['cell'])]
            print(f"{r['cell']:>4} {adj:>8.4f} {adj-bl:>+12.4f} {adj-op:>+12.4f}")
        print(f" Avg {np.mean(adj_vals):>8.4f}")


if __name__ == '__main__':
    main()
