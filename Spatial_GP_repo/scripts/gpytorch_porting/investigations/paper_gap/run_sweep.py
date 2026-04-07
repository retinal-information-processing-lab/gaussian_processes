#!/usr/bin/env python
"""
Parameter sweep: find best vargp_direct configuration.

All 41 cells x 3 seeds x 6 configs = 738 runs.

Designed to run unattended in background:
  nohup python investigations/paper_gap/run_sweep.py > investigations/paper_gap/sweep.log 2>&1 &

RESUME-SAFE: If interrupted and restarted, skips already-completed runs
by checking sweep_results.jsonl for existing (config_name, cell, seed) tuples.

Results saved to:
  investigations/paper_gap/sweep_results.jsonl  (machine-readable, one JSON per run)
  investigations/paper_gap/sweep_summary.txt    (human-readable summary table)

Investigation rules (MANDATORY):
  - ip_selection = 'random' (Finding 18: pivoted confounds mode comparisons)
  - fix_Amp = True (paper has no Amp; free Amp confounds training dynamics)
  - sigma_0 = direct parameterization (Finding 19: exp transform causes stagnation)

All configs use lambda0_init=-1 (paper's value).
"""
import sys, os, json, subprocess, time, tempfile, datetime, gc
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
from run_single_mode import build_config_from_defaults

CELLS = list(range(41))
SEEDS = [1, 2, 3]
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
        'notes': 'Our original defaults (10/10/50, A=0.01). Baseline before any paper-inspired changes',
        'beta': 0.1, 'A_init': 0.01, 'interleave_fstep': False,
        'n_estep': 10, 'n_mstep': 10, 'n_iterations': 50,
    },
    {
        'name': 'our+paper_inner',
        'notes': 'Our defaults but with paper iteration counts (50/20/80). Isolates iteration count effect',
        'beta': 0.1, 'A_init': 0.01, 'interleave_fstep': False,
        'n_estep': 50, 'n_mstep': 20, 'n_iterations': 80,
    },
    {
        'name': 'paper_init',
        'notes': 'Paper init values (tight RF beta=0.0452, A=1e-4) without interleaving. Tests init effect alone',
        'beta': 0.0452, 'A_init': 1e-4, 'interleave_fstep': False,
        'n_estep': 50, 'n_mstep': 20, 'n_iterations': 80,
    },
    {
        'name': 'paper+interleave',
        'notes': 'Paper init + interleaved F-step. Tests if interleaving helps bootstrap A=1e-4 with tight RF',
        'beta': 0.0452, 'A_init': 1e-4, 'interleave_fstep': True,
        'n_estep': 50, 'n_mstep': 20, 'n_iterations': 80,
    },
    {
        'name': 'broad+interleave',
        'notes': 'Best config: broad RF + interleaving + A=1e-4. Our best result (36/41 expl_var > 0.8)',
        'beta': 0.1, 'A_init': 1e-4, 'interleave_fstep': True,
        'n_estep': 50, 'n_mstep': 20, 'n_iterations': 80,
    },
    {
        'name': 'broad+A01+intl',
        'notes': 'Broad RF + interleaving but A=0.01. Tests if larger A init helps or hurts with interleaving',
        'beta': 0.1, 'A_init': 0.01, 'interleave_fstep': True,
        'n_estep': 50, 'n_mstep': 20, 'n_iterations': 80,
    },
]

# Rho mapped from beta: paper's rho=0.0821 for beta=0.0452, our rho=0.1 for beta=0.1
BETA_TO_RHO = {0.0452: 0.0821, 0.1: 0.1}


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
    eps_0x, eps_0y = rf['norm_108'][cell_id]
    rho = BETA_TO_RHO[cfg['beta']]

    config = build_config_from_defaults(
        mode='vargp_direct',
        M=250,
        n_train=3160,
        seed=seed,
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
                if 'notes' in cfg:
                    result['notes'] = cfg['notes']
                return result

        print(f"    ERROR: No RESULT_JSON for cell {cell_id} seed {seed} config {cfg['name']}")
        if proc.stderr:
            print(f"    STDERR: {proc.stderr[-300:]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT: cell {cell_id} seed {seed} config {cfg['name']} (>600s)")
        return None
    except Exception as e:
        print(f"    EXCEPTION: cell {cell_id} seed {seed} config {cfg['name']}: {e}")
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
    lines.append(f"Cells: 0-40 (all 41)")
    lines.append(f"Seeds: {SEEDS}")
    lines.append(f"Rules: ip_selection=random, fix_Amp=True, sigma_0=direct")
    lines.append(f"Fixed: 108x108, M=250, n_train=3160, ground-truth RF, lambda0=-1")
    lines.append("=" * 80)

    # Config descriptions
    lines.append("\nConfigurations:")
    for cfg in CONFIGS:
        intl = "yes" if cfg['interleave_fstep'] else "no"
        lines.append(f"  {cfg['name']:<22} beta={cfg['beta']}, A={cfg['A_init']}, "
                      f"intl={intl}, {cfg['n_estep']}/{cfg['n_mstep']}/{cfg['n_iterations']}")

    # Compute per-cell mean adj_r2 (averaged over seeds)
    lines.append("\n\nPer-config summary (mean over cells and seeds):")
    lines.append(f"{'Config':<22} | {'mean':>6} {'median':>7} {'std':>6} | {'n>0.8':>5} {'n>0.6':>5} | {'n_runs':>6}")
    lines.append("-" * 75)

    config_stats = {}
    for cfg in CONFIGS:
        name = cfg['name']
        cr = [r for r in all_results if r.get('config_name') == name]
        if not cr:
            continue

        # Average over seeds per cell, then stats over cells
        cell_means = {}
        for cell_id in CELLS:
            cell_runs = [r['adjusted_r2'] for r in cr if r['cell'] == cell_id]
            if cell_runs:
                cell_means[cell_id] = np.mean(cell_runs)

        if not cell_means:
            continue

        vals = list(cell_means.values())
        avg = np.mean(vals)
        med = np.median(vals)
        std = np.std(vals)
        n_above_08 = sum(1 for v in vals if v > 0.8)
        n_above_06 = sum(1 for v in vals if v > 0.6)
        config_stats[name] = {'mean': avg, 'median': med, 'n_above_08': n_above_08,
                               'n_cells': len(vals), 'n_runs': len(cr)}

        lines.append(f"{name:<22} | {avg:>6.3f} {med:>7.3f} {std:>6.3f} | "
                      f"{n_above_08:>5}/41 {n_above_06:>5}/41 | {len(cr):>6}")

    # Best config
    if config_stats:
        best_name = max(config_stats, key=lambda k: config_stats[k]['mean'])
        best = config_stats[best_name]
        lines.append(f"\nBest config: {best_name}")
        lines.append(f"  mean adj_r2 = {best['mean']:.4f}, {best['n_above_08']}/41 cells > 0.8")

    # Per-cell breakdown for best config
    if config_stats:
        best_results = [r for r in all_results if r.get('config_name') == best_name]
        if best_results:
            lines.append(f"\nPer-cell results for best config ({best_name}):")
            lines.append(f"  {'Cell':>4} {'mean_adj_r2':>11} {'std':>6} {'mean_test_r':>11} "
                          f"{'mean_A':>8} {'mean_sig0':>9} {'mean_beta':>9}")
            lines.append("  " + "-" * 65)
            for cell_id in CELLS:
                cr = [r for r in best_results if r['cell'] == cell_id]
                if cr:
                    adj = np.mean([r['adjusted_r2'] for r in cr])
                    adj_std = np.std([r['adjusted_r2'] for r in cr])
                    tr = np.mean([r['test_r'] for r in cr])
                    A = np.mean([r['final_A'] for r in cr])
                    sig0 = np.mean([r['final_sigma_0'] for r in cr])
                    beta = np.mean([r['final_beta'] for r in cr])
                    marker = " *" if adj > 0.8 else ""
                    lines.append(f"  {cell_id:>4} {adj:>11.4f} {adj_std:>6.4f} {tr:>11.4f} "
                                  f"{A:>8.5f} {sig0:>9.3f} {beta:>9.4f}{marker}")

    # Comparison baselines
    lines.append("\nReference baselines:")
    lines.append("  Previous baseline (64x64, M=2910, our defaults): avg adj_r2 = 0.720, 13/41 > 0.8")
    lines.append("  Paper target: 36/41 cells > 0.8 adj_r2")

    lines.append("\n" + "=" * 80)
    lines.append("SWEEP COMPLETE")
    lines.append("=" * 80)

    with open(SUMMARY_FILE, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    start_time = time.time()
    total = len(CONFIGS) * len(CELLS) * len(SEEDS)

    print(f"=== Parameter Sweep ===")
    print(f"  {len(CONFIGS)} configs x {len(CELLS)} cells x {len(SEEDS)} seeds = {total} runs")
    print(f"  Started: {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"  Results: {RESULTS_FILE}")
    print(f"  Summary: {SUMMARY_FILE}")

    # Load already-completed runs for resume
    completed = load_completed_runs()
    if completed:
        print(f"  Resuming: {len(completed)} runs already completed, {total - len(completed)} remaining")
    print()

    all_results = []
    # Reload existing results for summary computation
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    done = len(completed)
    skipped = 0
    failed = 0

    for cfg in CONFIGS:
        intl = "yes" if cfg['interleave_fstep'] else "no"
        print(f"\n--- {cfg['name']} (beta={cfg['beta']}, A={cfg['A_init']}, intl={intl}, "
              f"{cfg['n_estep']}/{cfg['n_mstep']}/{cfg['n_iterations']}) ---")

        for cell_id in CELLS:
            for seed in SEEDS:
                key = (cfg['name'], cell_id, seed)

                # Skip if already completed (resume support)
                if key in completed:
                    skipped += 1
                    continue

                done += 1
                print(f"  [{done}/{total}] {cfg['name']} cell={cell_id} seed={seed}...",
                      end=' ', flush=True)
                t0 = time.time()
                result = run_single_cell(cell_id, seed, cfg)
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
                    failed += 1
                    print(f"FAILED ({elapsed:.1f}s)")

                # Explicit cleanup between runs
                gc.collect()

    total_time = time.time() - start_time
    print(f"\n\n{'=' * 60}")
    print(f"Sweep finished in {total_time:.0f}s ({total_time/60:.1f} min, {total_time/3600:.1f} hr)")
    print(f"  Completed: {len(all_results)}, Skipped (resume): {skipped}, Failed: {failed}")

    # Write summary
    write_summary(all_results)
    print(f"\nSummary written to: {SUMMARY_FILE}")

    # Print summary to stdout too
    with open(SUMMARY_FILE) as f:
        print(f.read())


if __name__ == '__main__':
    main()
