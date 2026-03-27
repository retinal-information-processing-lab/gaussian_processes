#!/usr/bin/env python
"""
Parameter sweep: find best vargp_direct configuration.

Designed to run unattended in background:
  nohup python investigations/paper_gap/run_sweep.py > investigations/paper_gap/sweep.log 2>&1 &

Results saved to:
  investigations/paper_gap/sweep_results.jsonl  (machine-readable, one JSON per run)
  investigations/paper_gap/sweep_summary.txt    (human-readable summary table)

Investigation rules (MANDATORY):
  - ip_selection = 'random' (Finding 18: pivoted confounds mode comparisons)
  - fix_Amp = True (paper has no Amp; free Amp confounds training dynamics)
  - sigma_0 = direct parameterization (Finding 19: exp transform causes stagnation)

All configs use lambda0_init=-1 (paper's value).
"""
import sys, os, json, subprocess, time, tempfile, datetime
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
from run_single_mode import build_config_from_defaults

CELLS = [18, 14, 9, 28, 39]
DATA_108 = os.path.join(PROJ, 'datasets', 'PNAS_108x108_original.npz')
RF_PATH = os.path.join(PROJ, 'datasets', 'rf_centers_ground_truth.npz')
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_results.jsonl')
SUMMARY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_summary.txt')

rf = np.load(RF_PATH)

# =========================================================================
# Sweep configurations
# =========================================================================
CONFIGS = [
    {
        'name': 'our_defaults',
        'beta': 0.1, 'A_init': 0.01, 'interleave_fstep': False,
        'n_estep': 10, 'n_mstep': 10, 'n_iterations': 50,
    },
    {
        'name': 'our+paper_inner',
        'beta': 0.1, 'A_init': 0.01, 'interleave_fstep': False,
        'n_estep': 50, 'n_mstep': 20, 'n_iterations': 80,
    },
    {
        'name': 'paper_init',
        'beta': 0.0452, 'A_init': 1e-4, 'interleave_fstep': False,
        'n_estep': 50, 'n_mstep': 20, 'n_iterations': 80,
    },
    {
        'name': 'paper+interleave',
        'beta': 0.0452, 'A_init': 1e-4, 'interleave_fstep': True,
        'n_estep': 50, 'n_mstep': 20, 'n_iterations': 80,
    },
    {
        'name': 'broad+interleave',
        'beta': 0.1, 'A_init': 1e-4, 'interleave_fstep': True,
        'n_estep': 50, 'n_mstep': 20, 'n_iterations': 80,
    },
    {
        'name': 'broad+A01+intl',
        'beta': 0.1, 'A_init': 0.01, 'interleave_fstep': True,
        'n_estep': 50, 'n_mstep': 20, 'n_iterations': 80,
    },
]

# Rho mapped from beta: paper's rho=0.0821 for beta=0.0452, our rho=0.1 for beta=0.1
BETA_TO_RHO = {0.0452: 0.0821, 0.1: 0.1}


def run_single_cell(cell_id, cfg):
    """Run a single cell fit as subprocess. Returns result dict or None."""
    eps_0x, eps_0y = rf['norm_108'][cell_id]
    rho = BETA_TO_RHO[cfg['beta']]

    config = build_config_from_defaults(
        mode='vargp_direct',
        M=250,
        n_train=3160,
        seed=1,
        cell=cell_id,
        data_path=DATA_108,
        eps_0x=float(eps_0x),
        eps_0y=float(eps_0y),
        beta=cfg['beta'],
        rho=rho,
        A_init=cfg['A_init'],
        lambda0_init=-1.0,
        n_iterations=cfg['n_iterations'],
        n_estep=cfg['n_estep'],
        n_mstep=cfg['n_mstep'],
    )
    # Investigation rules
    config['early_stop'] = False
    config['fix_Amp'] = True
    config['ip_selection'] = 'random'
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

        print(f"    ERROR: No RESULT_JSON for cell {cell_id}, config {cfg['name']}")
        if proc.stderr:
            print(f"    STDERR: {proc.stderr[-300:]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT: cell {cell_id}, config {cfg['name']} (>600s)")
        return None
    except Exception as e:
        print(f"    EXCEPTION: cell {cell_id}, config {cfg['name']}: {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def write_summary(all_results):
    """Write human-readable summary to file."""
    lines = []
    lines.append("=" * 80)
    lines.append("PARAMETER SWEEP SUMMARY")
    lines.append(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Cells: {CELLS}")
    lines.append(f"Rules: ip_selection=random, fix_Amp=True, sigma_0=direct")
    lines.append(f"Fixed: 108x108, M=250, n_train=3160, seed=1, ground-truth RF, lambda0=-1")
    lines.append("=" * 80)

    # Config descriptions
    lines.append("\nConfigurations:")
    for cfg in CONFIGS:
        intl = "yes" if cfg['interleave_fstep'] else "no"
        lines.append(f"  {cfg['name']:<22} beta={cfg['beta']}, A={cfg['A_init']}, "
                      f"intl={intl}, {cfg['n_estep']}/{cfg['n_mstep']}/{cfg['n_iterations']}")

    # Results table
    lines.append(f"\n{'Config':<22} | ", )
    header = f"{'Config':<22} |"
    for c in CELLS:
        header += f" C{c:>2} |"
    header += f" {'Avg':>6} |"
    lines.append(header)
    lines.append("-" * len(header))

    config_avgs = {}
    for cfg in CONFIGS:
        name = cfg['name']
        cr = [r for r in all_results if r.get('config_name') == name]
        vals = {r['cell']: r['adjusted_r2'] for r in cr}
        row = f"{name:<22} |"
        cell_vals = []
        for c in CELLS:
            v = vals.get(c, None)
            if v is not None:
                row += f" {v:>.3f} |"
                cell_vals.append(v)
            else:
                row += f" {'FAIL':>5} |"
        avg = np.mean(cell_vals) if cell_vals else 0
        config_avgs[name] = avg
        row += f" {avg:>6.3f} |"
        lines.append(row)

    # Best config
    if config_avgs:
        best_name = max(config_avgs, key=config_avgs.get)
        lines.append(f"\nBest config: {best_name} (avg adj_r2 = {config_avgs[best_name]:.4f})")

    # Comparison baselines
    lines.append("\nReference baselines:")
    lines.append("  Previous baseline (64x64, M=2910, defaults): avg adj_r2 = 0.720, 13/41 > 0.8")
    lines.append("  Paper target: 36/41 cells > 0.8 adj_r2")

    # Trained hyperparameters for best config
    if config_avgs:
        best_results = [r for r in all_results if r.get('config_name') == best_name]
        if best_results:
            lines.append(f"\nTrained hyperparameters for best config ({best_name}):")
            lines.append(f"  {'Cell':>4} {'adj_r2':>8} {'A':>10} {'sig0':>8} {'beta':>8} {'rho':>8}")
            for r in sorted(best_results, key=lambda x: x['cell']):
                lines.append(f"  {r['cell']:>4} {r['adjusted_r2']:>8.4f} {r['final_A']:>10.6f} "
                              f"{r['final_sigma_0']:>8.3f} {r['final_beta']:>8.4f} {r['final_rho']:>8.4f}")

    lines.append("\n" + "=" * 80)
    lines.append("SWEEP COMPLETE")
    lines.append("=" * 80)

    with open(SUMMARY_FILE, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    start_time = time.time()
    print(f"=== Parameter Sweep: {len(CONFIGS)} configs x {len(CELLS)} cells = {len(CONFIGS)*len(CELLS)} runs ===")
    print(f"Started: {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"Results: {RESULTS_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    print()

    # Clean results file for fresh sweep
    if os.path.exists(RESULTS_FILE):
        backup = RESULTS_FILE + '.bak'
        os.rename(RESULTS_FILE, backup)
        print(f"  Previous results backed up to {backup}")

    all_results = []
    total = len(CONFIGS) * len(CELLS)
    done = 0

    for cfg in CONFIGS:
        intl = "yes" if cfg['interleave_fstep'] else "no"
        print(f"\n--- {cfg['name']} (beta={cfg['beta']}, A={cfg['A_init']}, intl={intl}, "
              f"{cfg['n_estep']}/{cfg['n_mstep']}/{cfg['n_iterations']}) ---")

        for cell_id in CELLS:
            done += 1
            print(f"  [{done}/{total}] Cell {cell_id}...", end=' ', flush=True)
            t0 = time.time()
            result = run_single_cell(cell_id, cfg)
            elapsed = time.time() - t0

            if result:
                adj = result.get('adjusted_r2', 0)
                tr = result.get('test_r', 0)
                print(f"adj_r2={adj:.4f}, test_r={tr:.4f}, time={elapsed:.1f}s")
                all_results.append(result)
                # Append to JSONL immediately (crash-safe)
                with open(RESULTS_FILE, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            else:
                print(f"FAILED ({elapsed:.1f}s)")

    total_time = time.time() - start_time
    print(f"\n\nTotal sweep time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Successful runs: {len(all_results)}/{total}")

    # Write summary
    write_summary(all_results)
    print(f"\nSummary written to: {SUMMARY_FILE}")

    # Also print summary to stdout
    with open(SUMMARY_FILE) as f:
        print(f.read())


if __name__ == '__main__':
    main()
