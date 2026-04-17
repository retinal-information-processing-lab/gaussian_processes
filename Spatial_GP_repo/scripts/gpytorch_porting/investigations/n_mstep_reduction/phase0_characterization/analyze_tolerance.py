#!/usr/bin/env python3
"""
Tolerance Analysis: replay M-step traces to find principled float32 tolerances.

Reads tolerance_sweep_results.jsonl and performs offline what-if analysis:
for each candidate tolerance pair, compute how many closure calls would have
been saved and how much ELBO improvement would have been forfeited.

Key plots:
1. grad_max distribution across all closure calls (where to set tolerance_grad)
2. |loss_k - loss_{k-1}| distribution (where to set tolerance_change for loss)
3. |param_k - param_{k-1}| distribution (where to set tolerance_change for step)
4. ELBO forfeited vs closure calls saved for candidate tolerances
5. elbo_before vs elbo_after_reproject: how much M-step gain survives reprojection

Usage:
    python analyze_tolerance.py [--results tolerance_sweep_results.jsonl]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    results = []
    with open(path) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def extract_all_traces(results):
    """Extract per-M-step closure-call traces with full telemetry."""
    traces = []
    for r in results:
        cell, seed = r.get('cell'), r.get('seed')
        for entry in (r.get('mstep_diagnostics') or []):
            calls = entry.get('closure_calls', [])
            valid_calls = [c for c in calls if not c.get('rejected', True)]
            if len(valid_calls) < 2:
                continue
            traces.append({
                'cell': cell,
                'seed': seed,
                'outer_iter': entry['outer_iter'],
                'n_lbfgs_iters': entry['n_lbfgs_iters'],
                'n_func_evals': entry.get('n_func_evals'),
                'elbo_before': entry.get('elbo_before_mstep'),
                'elbo_after_reproject': entry.get('elbo_after_reproject'),
                'valid_calls': valid_calls,
            })
    return traces


def plot_grad_max_distribution(traces, out_dir):
    """Distribution of grad_max across all closure calls."""
    all_gm = []
    for t in traces:
        for c in t['valid_calls']:
            gm = c.get('grad_max')
            if gm is not None and gm > 0:
                all_gm.append(gm)

    if not all_gm:
        print("  No grad_max data found, skipping grad_max plot")
        return

    all_gm = np.array(all_gm)
    log_gm = np.log10(all_gm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(log_gm, bins=80, edgecolor='black', alpha=0.7, linewidth=0.3)
    ax.set_xlabel('log10(grad_max)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of max |gradient| across all closure calls')
    # Mark candidate tolerance_grad values
    for tg, color, ls in [(1e-7, 'red', '--'), (1e-5, 'orange', '--'),
                           (1e-3, 'green', '--'), (1e-1, 'blue', '--')]:
        ax.axvline(np.log10(tg), color=color, linestyle=ls,
                   label=f'tol_grad={tg:.0e}')
    ax.legend(fontsize=8)

    # Right: grad_max vs closure call index within M-step
    ax = axes[1]
    for t in traces[:100]:  # limit to 100 traces for readability
        gms = [c.get('grad_max') for c in t['valid_calls']
               if c.get('grad_max') is not None]
        if gms:
            ax.semilogy(range(len(gms)), gms, alpha=0.1, color='steelblue',
                        linewidth=0.5)
    ax.set_xlabel('Closure call index within M-step')
    ax.set_ylabel('grad_max (log scale)')
    ax.set_title('Gradient convergence within each M-step')
    for tg, color, ls in [(1e-7, 'red', '--'), (1e-5, 'orange', '--'),
                           (1e-3, 'green', '--')]:
        ax.axhline(tg, color=color, linestyle=ls, label=f'{tg:.0e}')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'grad_max_distribution.png'), dpi=150)
    plt.close()
    print(f"  Saved grad_max_distribution.png")


def plot_loss_change_distribution(traces, out_dir):
    """Distribution of |loss_k - loss_{k-1}| between consecutive closure calls."""
    all_diffs = []
    all_rel_diffs = []
    for t in traces:
        losses = [c['loss'] for c in t['valid_calls']]
        for i in range(1, len(losses)):
            d = abs(losses[i] - losses[i-1])
            if d > 0:
                all_diffs.append(d)
                if abs(losses[i-1]) > 1e-10:
                    all_rel_diffs.append(d / abs(losses[i-1]))

    if not all_diffs:
        print("  No loss diff data, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute loss change
    ax = axes[0]
    log_diffs = np.log10(np.array(all_diffs))
    ax.hist(log_diffs, bins=80, edgecolor='black', alpha=0.7, linewidth=0.3)
    ax.set_xlabel('log10(|loss_k - loss_{k-1}|)')
    ax.set_ylabel('Count')
    ax.set_title('Absolute loss change between consecutive closure calls')
    for tc, color, ls in [(1e-9, 'red', '--'), (1e-4, 'orange', '--'),
                           (1e-2, 'green', '--'), (1.0, 'blue', '--')]:
        ax.axvline(np.log10(tc), color=color, linestyle=ls,
                   label=f'tol={tc:.0e}')
    ax.legend(fontsize=8)

    # Right: relative loss change
    ax = axes[1]
    if all_rel_diffs:
        log_rel = np.log10(np.array(all_rel_diffs))
        ax.hist(log_rel, bins=80, edgecolor='black', alpha=0.7, linewidth=0.3)
        ax.set_xlabel('log10(|loss_k - loss_{k-1}| / |loss_{k-1}|)')
        ax.set_ylabel('Count')
        ax.set_title('Relative loss change between consecutive calls')
        for tc, color, ls in [(1e-9, 'red', '--'), (1e-6, 'orange', '--'),
                               (1e-4, 'green', '--')]:
            ax.axvline(np.log10(tc), color=color, linestyle=ls,
                       label=f'{tc:.0e}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_change_distribution.png'), dpi=150)
    plt.close()
    print(f"  Saved loss_change_distribution.png")


def plot_param_step_distribution(traces, out_dir):
    """Distribution of max |param_k - param_{k-1}| between consecutive calls."""
    all_steps = []
    for t in traces:
        calls = t['valid_calls']
        for i in range(1, len(calls)):
            p_prev = calls[i-1].get('params')
            p_curr = calls[i].get('params')
            if p_prev and p_curr:
                max_step = max(abs(p_curr[k] - p_prev[k])
                               for k in p_curr if k in p_prev)
                if max_step > 0:
                    all_steps.append(max_step)

    if not all_steps:
        print("  No param step data, skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    log_steps = np.log10(np.array(all_steps))
    ax.hist(log_steps, bins=80, edgecolor='black', alpha=0.7, linewidth=0.3)
    ax.set_xlabel('log10(max |param_k - param_{k-1}|)')
    ax.set_ylabel('Count')
    ax.set_title('Max parameter step between consecutive closure calls')
    # float32 eps for typical param magnitudes (~0.1)
    f32_eps_param = 0.1 * 1.2e-7
    ax.axvline(np.log10(f32_eps_param), color='red', linestyle='--',
               label=f'float32 floor (~{f32_eps_param:.1e})')
    ax.axvline(np.log10(1e-9), color='orange', linestyle='--',
               label='current tol_change=1e-9')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'param_step_distribution.png'), dpi=150)
    plt.close()
    print(f"  Saved param_step_distribution.png")


def plot_elbo_bracket(traces, out_dir):
    """Decomposed ELBO bracket: M-step internal gain vs reprojection cost."""
    befores, afters, outers = [], [], []
    mstep_gains, reproj_costs = [], []
    for t in traces:
        eb = t.get('elbo_before')
        ea = t.get('elbo_after_reproject')
        if eb is None or ea is None:
            continue
        befores.append(eb)
        afters.append(ea)
        outers.append(t['outer_iter'])
        # Last non-rejected closure call loss = -ELBO in old eigenspace after M-step
        valid = t['valid_calls']
        elbo_after_mstep = -valid[-1]['loss']  # M-step's final ELBO (old eigenspace)
        mstep_gains.append(elbo_after_mstep - eb)   # should be >= 0
        reproj_costs.append(ea - elbo_after_mstep)   # can be negative

    if not befores:
        print("  No ELBO bracket data, skipping")
        return

    net_gains = np.array(afters) - np.array(befores)
    mstep_gains = np.array(mstep_gains)
    reproj_costs = np.array(reproj_costs)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Left: M-step internal gain (should be >= 0)
    ax = axes[0]
    ax.scatter(outers, mstep_gains, alpha=0.2, s=10, color='green')
    ax.set_xlabel('Outer EM iteration')
    ax.set_ylabel('ELBO change')
    ax.set_title('M-step internal gain\n(LBFGS improvement in old eigenspace)')
    ax.axhline(0, color='red', linestyle='-', alpha=0.5)
    pct_pos = 100 * np.sum(mstep_gains > 0) / len(mstep_gains)
    ax.text(0.95, 0.95, f'{pct_pos:.0f}% positive',
            transform=ax.transAxes, ha='right', va='top', fontsize=11)

    # Middle: reprojection cost (often negative)
    ax = axes[1]
    ax.scatter(outers, reproj_costs, alpha=0.2, s=10, color='orange')
    ax.set_xlabel('Outer EM iteration')
    ax.set_ylabel('ELBO change')
    ax.set_title('Reprojection cost\n(eigenspace rebuild effect)')
    ax.axhline(0, color='red', linestyle='-', alpha=0.5)
    pct_neg = 100 * np.sum(reproj_costs < 0) / len(reproj_costs)
    ax.text(0.95, 0.95, f'{pct_neg:.0f}% negative',
            transform=ax.transAxes, ha='right', va='top', fontsize=11)

    # Right: net gain (= mstep_gain + reproj_cost)
    ax = axes[2]
    ax.scatter(outers, net_gains, alpha=0.2, s=10, color='steelblue')
    ax.set_xlabel('Outer EM iteration')
    ax.set_ylabel('ELBO change')
    ax.set_title('Net ELBO gain\n(M-step + reprojection combined)')
    ax.axhline(0, color='red', linestyle='-', alpha=0.5)
    pct_positive = 100 * np.sum(net_gains > 0) / len(net_gains)
    ax.text(0.95, 0.95, f'{pct_positive:.0f}% positive',
            transform=ax.transAxes, ha='right', va='top', fontsize=11)
    ax.set_ylabel('Net ELBO gain')
    ax.set_title('M-step net ELBO gain vs training progress')
    ax.axhline(0, color='red', linestyle='-', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'elbo_bracket.png'), dpi=150)
    plt.close()
    print(f"  Saved elbo_bracket.png")


def plot_elbo_trajectory(results, out_dir):
    """Per-cell ELBO trajectory showing M-step / reprojection / E+F step structure.

    For each outer iteration K, plots 3 segments:
      Orange (dotted): M-step gain — elbo_before_mstep[K] to elbo_after_mstep[K]
      Gray:            Reprojection — elbo_after_mstep[K] to elbo_after_reproject[K]
      Blue:            E+F recovery — elbo_after_reproject[K] to elbo_before_mstep[K+1]

    One subplot per (cell, seed) run.
    """
    # Group diagnostics by (cell, seed) and sort by outer_iter
    runs = []
    for r in results:
        cell, seed = r.get('cell'), r.get('seed')
        diag = r.get('mstep_diagnostics')
        if not diag:
            continue
        # Sort entries by outer_iter
        entries = sorted(diag, key=lambda e: e['outer_iter'])
        runs.append({'cell': cell, 'seed': seed, 'entries': entries,
                     'test_r': r.get('test_r')})

    if not runs:
        print("  No diagnostic data for trajectory plot")
        return

    # Pick a subset: one seed per cell to keep plots readable
    cells_seen = set()
    selected = []
    for run in sorted(runs, key=lambda r: (r['cell'], r['seed'])):
        if run['cell'] not in cells_seen:
            cells_seen.add(run['cell'])
            selected.append(run)

    n_plots = len(selected)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows),
                              squeeze=False)

    for idx, run in enumerate(selected):
        ax = axes[idx // n_cols][idx % n_cols]
        entries = run['entries']

        # Build the 4 values per iteration
        for i, entry in enumerate(entries):
            eb = entry.get('elbo_before_mstep')
            ea = entry.get('elbo_after_reproject')
            if eb is None:
                continue

            oi = entry['outer_iter']
            valid_calls = [c for c in entry.get('closure_calls', [])
                           if not c.get('rejected', True)]
            if not valid_calls:
                continue
            elbo_after_mstep = -valid_calls[-1]['loss']

            # x positions: spread each iteration across [oi, oi+1)
            x_before = oi
            x_after_mstep = oi + 0.33
            x_after_reproj = oi + 0.66

            # Orange dotted: M-step gain
            ax.plot([x_before, x_after_mstep], [eb, elbo_after_mstep],
                    color='darkorange', linestyle=':', linewidth=1.5,
                    marker='.', markersize=4, zorder=3)

            # Gray: reprojection
            if ea is not None:
                ax.plot([x_after_mstep, x_after_reproj], [elbo_after_mstep, ea],
                        color='gray', linewidth=1.2, marker='.', markersize=4,
                        zorder=2)

                # Blue: E+F recovery (connect to next iteration's elbo_before)
                if i + 1 < len(entries):
                    next_eb = entries[i + 1].get('elbo_before_mstep')
                    if next_eb is not None:
                        x_next = entries[i + 1]['outer_iter']
                        ax.plot([x_after_reproj, x_next],
                                [ea, next_eb],
                                color='steelblue', linewidth=1.5,
                                marker='.', markersize=4, zorder=3)

        ax.set_xlabel('Outer EM iteration')
        ax.set_ylabel('ELBO')
        ax.set_title(f'Cell {run["cell"]} seed {run["seed"]}  '
                     f'(test_r={run["test_r"]:.3f})')

    # Legend in first axis
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkorange', linestyle=':', linewidth=1.5,
               marker='.', label='M-step (LBFGS)'),
        Line2D([0], [0], color='gray', linewidth=1.2, marker='.',
               label='Reprojection'),
        Line2D([0], [0], color='steelblue', linewidth=1.5, marker='.',
               label='E+F step'),
    ]
    axes[0][0].legend(handles=legend_elements, fontsize=9, loc='lower right')

    # Hide unused subplots
    for idx in range(n_plots, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle('ELBO trajectory: per-iteration decomposition', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'elbo_trajectory.png'), dpi=150)
    plt.close()
    print(f"  Saved elbo_trajectory.png")


def what_if_tolerance(traces):
    """Offline replay: for candidate tolerances, compute savings and cost.

    For each M-step trace, walk the closure calls and find the first call
    where the candidate tolerance would have triggered. Compute:
    - calls saved (how many fewer closure calls)
    - ELBO forfeited (ELBO at early-stop point vs ELBO at actual end)
    """
    # Candidate tolerance_change values (absolute loss diff)
    tol_changes = [1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    # Candidate tolerance_grad values (max absolute gradient)
    tol_grads = [1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    print("\n" + "=" * 70)
    print("WHAT-IF TOLERANCE ANALYSIS")
    print("=" * 70)

    # -- tolerance_change analysis --
    print("\n--- tolerance_change (loss diff) ---")
    print(f"{'tol_change':>12} {'med_%_calls':>12} "
          f"{'med_%_of_mstep_gain':>20} {'med_%_of_elbo':>15} {'pct_affected':>14}")
    for tc in tol_changes:
        calls_saved_list = []
        pct_of_gain_list = []
        pct_of_elbo_list = []
        n_affected = 0
        for t in traces:
            losses = [c['loss'] for c in t['valid_calls']]
            total_calls = len(losses)
            stop_idx = total_calls
            for i in range(1, len(losses)):
                if abs(losses[i] - losses[i-1]) < tc:
                    stop_idx = i
                    break
            saved = total_calls - stop_idx
            calls_saved_list.append(saved)
            if saved > 0:
                n_affected += 1
                abs_forfeited = losses[stop_idx] - losses[-1]  # positive = lost ELBO
                # As % of M-step's total internal gain
                mstep_total_gain = losses[0] - losses[-1]  # total loss reduction
                if abs(mstep_total_gain) > 1e-10:
                    pct_of_gain_list.append(100 * abs_forfeited / abs(mstep_total_gain))
                else:
                    pct_of_gain_list.append(0.0)
                # As % of ELBO magnitude
                if abs(losses[0]) > 1e-10:
                    pct_of_elbo_list.append(100 * abs_forfeited / abs(losses[0]))
                else:
                    pct_of_elbo_list.append(0.0)
            else:
                pct_of_gain_list.append(0.0)
                pct_of_elbo_list.append(0.0)

        med_pct_calls = 100 * np.median(calls_saved_list) / np.median(
            [len(t['valid_calls']) for t in traces])
        med_pct_gain = np.median(pct_of_gain_list)
        med_pct_elbo = np.median(pct_of_elbo_list)
        pct_aff = 100 * n_affected / len(traces)
        print(f"{tc:>12.0e} {med_pct_calls:>11.1f}% "
              f"{med_pct_gain:>19.2f}% {med_pct_elbo:>14.4f}% {pct_aff:>13.1f}%")

    print("\n  Columns: med_%_calls = median % of closure calls saved")
    print("           med_%_of_mstep_gain = median % of M-step's own ELBO improvement forfeited")
    print("           med_%_of_elbo = median % of total ELBO magnitude forfeited")

    # -- tolerance_change by training phase --
    phases = [
        ('Early (iter 1-10)', lambda t: t['outer_iter'] <= 10),
        ('Late  (iter 20+)',  lambda t: t['outer_iter'] >= 20),
    ]
    for phase_name, phase_filter in phases:
        phase_traces = [t for t in traces if phase_filter(t)]
        if not phase_traces:
            continue
        print(f"\n--- tolerance_change — {phase_name} ({len(phase_traces)} M-steps) ---")
        print(f"{'tol_change':>12} {'med_%_calls':>12} "
              f"{'med_%_of_mstep_gain':>20} {'med_%_of_elbo':>15} "
              f"{'med_abs_forfeited':>18}")
        for tc in tol_changes:
            pct_calls_list = []
            pct_of_gain_list = []
            pct_of_elbo_list = []
            abs_forfeited_list = []
            for t in phase_traces:
                losses = [c['loss'] for c in t['valid_calls']]
                total_calls = len(losses)
                stop_idx = total_calls
                for i in range(1, len(losses)):
                    if abs(losses[i] - losses[i-1]) < tc:
                        stop_idx = i
                        break
                saved = total_calls - stop_idx
                pct_calls_list.append(100 * saved / total_calls if total_calls > 0 else 0)
                abs_forf = losses[stop_idx] - losses[-1] if saved > 0 else 0.0
                abs_forfeited_list.append(abs_forf)
                mstep_total = losses[0] - losses[-1]
                if saved > 0 and abs(mstep_total) > 1e-10:
                    pct_of_gain_list.append(100 * abs_forf / abs(mstep_total))
                else:
                    pct_of_gain_list.append(0.0)
                if abs(losses[0]) > 1e-10:
                    pct_of_elbo_list.append(100 * abs_forf / abs(losses[0]))
                else:
                    pct_of_elbo_list.append(0.0)

            print(f"{tc:>12.0e} {np.median(pct_calls_list):>11.1f}% "
                  f"{np.median(pct_of_gain_list):>19.2f}% "
                  f"{np.median(pct_of_elbo_list):>14.4f}% "
                  f"{np.median(abs_forfeited_list):>17.3f}")

    # -- tolerance_grad analysis --
    print("\n--- tolerance_grad (max |gradient|) ---")
    print(f"{'tol_grad':>12} {'med_%_calls':>12} {'pct_affected':>14}")
    for tg in tol_grads:
        n_affected = 0
        calls_saved_list = []
        for t in traces:
            calls = t['valid_calls']
            total_calls = len(calls)
            stop_idx = total_calls
            for i, c in enumerate(calls):
                gm = c.get('grad_max')
                if gm is not None and gm <= tg:
                    stop_idx = i
                    break
            saved = total_calls - stop_idx
            calls_saved_list.append(saved)
            if saved > 0:
                n_affected += 1

        med_pct_calls = 100 * np.median(calls_saved_list) / np.median(
            [len(t['valid_calls']) for t in traces])
        pct_aff = 100 * n_affected / len(traces)
        print(f"{tg:>12.0e} {med_pct_calls:>11.1f}% {pct_aff:>13.1f}%")
    print("  (grad_max never drops below 0.6 — tolerance_grad is inert in float32)")


def print_summary(results, traces):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("TOLERANCE SWEEP SUMMARY")
    print("=" * 70)

    print(f"\n{len(results)} runs, {len(traces)} M-step traces with >=2 valid calls")

    # Aggregate grad_max at final (accepted) closure call
    final_gms = []
    for t in traces:
        last_gm = None
        for c in reversed(t['valid_calls']):
            gm = c.get('grad_max')
            if gm is not None:
                last_gm = gm
                break
        if last_gm is not None:
            final_gms.append(last_gm)

    if final_gms:
        final_gms = np.array(final_gms)
        print(f"\ngrad_max at LBFGS termination (tolerance actually triggered):")
        print(f"  Mean:   {np.mean(final_gms):.4e}")
        print(f"  Median: {np.median(final_gms):.4e}")
        print(f"  P10:    {np.percentile(final_gms, 10):.4e}")
        print(f"  P90:    {np.percentile(final_gms, 90):.4e}")
        print(f"  Min:    {np.min(final_gms):.4e}")
        print(f"  Max:    {np.max(final_gms):.4e}")

    # ELBO bracket — decomposed
    mstep_gains, reproj_costs, net_gains = [], [], []
    for t in traces:
        eb = t.get('elbo_before')
        ea = t.get('elbo_after_reproject')
        if eb is None or ea is None:
            continue
        valid = t['valid_calls']
        elbo_after_mstep = -valid[-1]['loss']
        mstep_gains.append(elbo_after_mstep - eb)
        reproj_costs.append(ea - elbo_after_mstep)
        net_gains.append(ea - eb)
    if net_gains:
        mstep_gains = np.array(mstep_gains)
        reproj_costs = np.array(reproj_costs)
        net_gains = np.array(net_gains)
        print(f"\nELBO bracket (decomposed):")
        print(f"  M-step internal gain (LBFGS in old eigenspace):")
        print(f"    Mean:   {np.mean(mstep_gains):.2f}")
        print(f"    Median: {np.median(mstep_gains):.2f}")
        print(f"    % positive: {100*np.sum(mstep_gains>0)/len(mstep_gains):.0f}%")
        print(f"  Reprojection cost (eigenspace rebuild):")
        print(f"    Mean:   {np.mean(reproj_costs):.2f}")
        print(f"    Median: {np.median(reproj_costs):.2f}")
        print(f"    % negative: {100*np.sum(reproj_costs<0)/len(reproj_costs):.0f}%")
        print(f"  Net gain (M-step + reprojection):")
        print(f"    Mean:   {np.mean(net_gains):.2f}")
        print(f"    Median: {np.median(net_gains):.2f}")
        print(f"    % positive: {100*np.sum(net_gains>0)/len(net_gains):.0f}%")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Analyze tolerance sweep')
    parser.add_argument('--results', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'tolerance_sweep_results.jsonl'))
    parser.add_argument('--out-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'tolerance_plots'))
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"ERROR: {args.results} not found. Run run_tolerance_sweep.py first.")
        sys.exit(1)

    results = load_results(args.results)
    print(f"Loaded {len(results)} runs")

    traces = extract_all_traces(results)
    print(f"Extracted {len(traces)} M-step traces")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nGenerating plots in {args.out_dir}/")

    plot_elbo_trajectory(results, args.out_dir)
    plot_grad_max_distribution(traces, args.out_dir)
    plot_loss_change_distribution(traces, args.out_dir)
    plot_param_step_distribution(traces, args.out_dir)
    plot_elbo_bracket(traces, args.out_dir)

    print_summary(results, traces)
    what_if_tolerance(traces)


if __name__ == '__main__':
    main()
