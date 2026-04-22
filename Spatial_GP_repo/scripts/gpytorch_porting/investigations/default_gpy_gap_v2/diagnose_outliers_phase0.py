"""Phase 0 diagnostic: compare trajectories on outlier cells (40, 38) vs
controls (30, 16) using existing sweep data only.

Produces:
  - Console table of final params, ES state, trajectory summaries.
  - diagnose_outliers_phase0.png: 4 cells x 2 subplots (ELBO and A) showing
    vargp_direct (seed 42) vs default_gpy (seed 42) trajectories side-by-side.

Input:  results.jsonl (written by run_sweep.py)
Output: diagnose_outliers_phase0.png in the investigation folder.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INVESTIGATION_DIR = Path(__file__).parent
RESULTS_PATH = INVESTIGATION_DIR / 'results.jsonl'
PLOT_PATH = INVESTIGATION_DIR / 'diagnose_outliers_phase0.png'

FOCUS_CELLS = [
    ('outlier', 40),
    ('outlier', 38),
    ('control', 30),
    ('control', 16),
]
SEED = 42


def load():
    out = {}
    with open(RESULTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            out[(r['mode'], r['cell'], r['seed'])] = r
    return out


def trajectory_summary(curve):
    if not curve:
        return 'N/A'
    arr = np.array(curve)
    return (f"n={len(arr):2d}  start={arr[0]:.3g}  "
            f"min={arr.min():.3g}  max={arr.max():.3g}  end={arr[-1]:.3g}")


def print_table(recs):
    print("=" * 100)
    print(f"Phase 0 — outlier diagnostic (seed {SEED})")
    print("=" * 100)
    for tag, cell in FOCUS_CELLS:
        print(f"\nCELL {cell}  ({tag})")
        print("-" * 100)
        for mode in ['vargp_direct', 'default_gpy']:
            r = recs.get((mode, cell, SEED))
            if r is None:
                print(f"  {mode}: <missing>")
                continue
            print(f"  [{mode}]")
            print(f"    test_r={r['test_r']:.4f}  train_r={r['train_r']:.4f}  "
                  f"final_loss={r['final_loss']:.2f}  "
                  f"stopped_early={r['stopped_early']}  "
                  f"best_iter={r['best_iteration']}  "
                  f"n_iter_run={r['n_iterations_run']}")
            print(f"    final params: A={r['final_A']:.4g}  "
                  f"lambda0={r['final_lambda0']:.3g}  "
                  f"beta={r['final_beta']:.4g}  rho={r['final_rho']:.4g}  "
                  f"eps_0=({r['final_eps_0x']:.3f}, {r['final_eps_0y']:.3f})")
            print(f"    train_loss curve: {trajectory_summary(r['train_loss_curve'])}")
            print(f"    A curve:          {trajectory_summary(r['A_curve'])}")
    print("=" * 100)


def plot(recs):
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=True)
    for row, (tag, cell) in enumerate(FOCUS_CELLS):
        ax_loss = axes[row, 0]
        ax_A = axes[row, 1]
        for mode, color in [('vargp_direct', 'C0'), ('default_gpy', 'C1')]:
            r = recs.get((mode, cell, SEED))
            if r is None or not r.get('train_loss_curve'):
                continue
            loss = np.array(r['train_loss_curve'])
            A = np.array(r['A_curve'])
            x = np.arange(len(loss))
            ax_loss.plot(x, loss, color=color, label=mode, lw=1.5)
            ax_A.plot(np.arange(len(A)), A, color=color, label=mode, lw=1.5)
            if r['stopped_early']:
                bi = r['best_iteration']
                ax_loss.axvline(bi, color=color, ls='--', alpha=0.4, lw=0.8)
        title_prefix = '[OUTLIER] ' if tag == 'outlier' else '[control] '
        ax_loss.set_title(f"{title_prefix}cell {cell}  -- train_loss (-ELBO)")
        ax_A.set_title(f"{title_prefix}cell {cell}  -- A")
        ax_loss.set_ylabel('train_loss')
        ax_A.set_ylabel('A')
        ax_A.set_yscale('log')
        ax_loss.legend(fontsize=8, loc='upper right')
        ax_loss.grid(alpha=0.3)
        ax_A.grid(alpha=0.3, which='both')
    axes[-1, 0].set_xlabel('outer iteration')
    axes[-1, 1].set_xlabel('outer iteration')
    fig.suptitle(f"Phase 0 outlier diagnostic -- seed {SEED}\n"
                 f"dashed vertical = best_iteration when ES fired",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=110, bbox_inches='tight')
    print(f"\nWrote {PLOT_PATH}")


def main():
    recs = load()
    print_table(recs)
    plot(recs)


if __name__ == '__main__':
    main()
