"""Analyze the default_gpy_gap_v2 sweep.

Reads results.jsonl and reports (per PROMPT.md "Pre-registered decision
criterion"):

  1. Per-cell table: 16 rows x {vargp_direct, default_gpy} x 3 seeds, showing
     per-mode mean±std of test_r, plus the cell's 3-seed mean Δ
     (= test_r_vargp_direct − test_r_default_gpy).
  2. Paired-Δ summary: mean, std, range, IQR, per-cell sign count.
  3. Binary decision: |mean_Δ| > 0.02 → GAP; else NO GAP.
  4. Timing: per-mode mean/std of train_time and train_time / n_iterations_run.
  5. Final-A distribution: per-mode {min, p25, median, p75, max, count(<0.001)}
     to catch freeze/explosion.
  6. default_gpy_alt sanity: seed-42 single-seed results per cell; compare to
     default_gpy (seed 42) as an at-a-glance sanity.

Numbers only. No interpretation, no speculation, no fix suggestions.
"""

import json
from pathlib import Path

import numpy as np

INVESTIGATION_DIR = Path(__file__).parent
RESULTS_PATH = INVESTIGATION_DIR / 'results.jsonl'
CELLS_USED_PATH = INVESTIGATION_DIR / 'cells_used.json'

GAP_THRESHOLD = 0.02  # locked in PROMPT.md


def load_records():
    records = []
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_cells_ordered():
    """Return cell_ids in the order they appear in cells_used.json
    (stratified bucket order + user-added appended)."""
    with open(CELLS_USED_PATH) as f:
        p = json.load(f)
    return [c['cell_id'] for c in p['cells']], p['cells']


def group_by_mode_cell_seed(records):
    out = {}
    for r in records:
        key = (r['mode'], r['cell'], r['seed'])
        if key in out:
            raise ValueError(f"duplicate (mode, cell, seed) {key}")
        out[key] = r
    return out


def fmt(x, prec=4):
    if x is None:
        return '  -   '
    return f"{x:.{prec}f}"


# --------------------------------------------------------------------- tables


def print_status_overview(records):
    print("=" * 78)
    print("STATUS OVERVIEW")
    print("=" * 78)
    modes = sorted({r['mode'] for r in records})
    for mode in modes:
        sub = [r for r in records if r['mode'] == mode]
        ok = sum(1 for r in sub if r.get('status') != 'failed')
        failed = len(sub) - ok
        print(f"  {mode:<18}  total={len(sub):3d}  ok={ok:3d}  failed={failed}")
    print()


def print_per_cell_table(by_key, cells_ordered, cells_meta):
    print("=" * 104)
    print("PER-CELL TEST_R (main modes, 3 seeds)")
    print("=" * 104)
    print(f"  {'cell':>4} {'bucket':>6} {'fr':>6}  "
          f"{'vargp_direct (3 seeds)':<30}  "
          f"{'default_gpy (3 seeds)':<30}  {'mean_Δ':>8}")
    print("-" * 104)
    meta_by_cell = {c['cell_id']: c for c in cells_meta}
    for cell in cells_ordered:
        m = meta_by_cell[cell]
        direct_trs = []
        gpy_trs = []
        for seed in [42, 123, 789]:
            d = by_key.get(('vargp_direct', cell, seed))
            g = by_key.get(('default_gpy', cell, seed))
            if d is not None and d.get('test_r') is not None:
                direct_trs.append(d['test_r'])
            if g is not None and g.get('test_r') is not None:
                gpy_trs.append(g['test_r'])
        # pairwise Δ: only pairs where both exist
        deltas = []
        for seed in [42, 123, 789]:
            d = by_key.get(('vargp_direct', cell, seed))
            g = by_key.get(('default_gpy', cell, seed))
            if d and g and d.get('test_r') is not None and g.get('test_r') is not None:
                deltas.append(d['test_r'] - g['test_r'])
        mean_delta = np.mean(deltas) if deltas else None

        def seed_row(vals):
            vals_ext = vals + [None] * (3 - len(vals))
            fvals = [fmt(v) for v in vals_ext]
            if vals:
                tag = f"  μ={np.mean(vals):.4f}"
            else:
                tag = "  (none)"
            return " ".join(fvals) + tag

        extra = ' U' if m.get('user_added') else '  '
        print(f"  {cell:>4} {m['bucket']:>6}{extra}{m['firing_rate']:>5.2f}  "
              f"{seed_row(direct_trs):<30}  {seed_row(gpy_trs):<30}  "
              f"{fmt(mean_delta):>8}")
    print("=" * 104)
    print()


def compute_paired_deltas(by_key, cells_ordered):
    """Return list of (cell, seed, delta) for successful pairs."""
    out = []
    for cell in cells_ordered:
        for seed in [42, 123, 789]:
            d = by_key.get(('vargp_direct', cell, seed))
            g = by_key.get(('default_gpy', cell, seed))
            if d and g and d.get('test_r') is not None and g.get('test_r') is not None:
                out.append((cell, seed, d['test_r'] - g['test_r']))
    return out


def print_paired_summary(deltas_full, cells_ordered, by_key):
    vals = np.array([d for _, _, d in deltas_full])
    n = len(vals)
    mean_d = vals.mean()
    std_d = vals.std(ddof=1) if n > 1 else float('nan')
    lo, hi = vals.min(), vals.max()
    q25, med, q75 = np.percentile(vals, [25, 50, 75])
    pos = int((vals > 0).sum())
    neg = int((vals < 0).sum())
    zero = int((vals == 0).sum())

    print("=" * 78)
    print("PAIRED Δ SUMMARY  (Δ = test_r_vargp_direct − test_r_default_gpy)")
    print("=" * 78)
    print(f"  n (paired runs):        {n}")
    print(f"  mean Δ:                 {mean_d:+.4f}")
    print(f"  std Δ (ddof=1):         {std_d:.4f}")
    print(f"  range [min, max]:       [{lo:+.4f}, {hi:+.4f}]")
    print(f"  IQR [q25, median, q75]: [{q25:+.4f}, {med:+.4f}, {q75:+.4f}]")
    print(f"  sign counts:            Δ>0: {pos},  Δ<0: {neg},  Δ=0: {zero}")
    print()

    # Per-cell mean deltas
    by_cell = {}
    for cell, seed, d in deltas_full:
        by_cell.setdefault(cell, []).append(d)
    per_cell_mean = {c: float(np.mean(v)) for c, v in by_cell.items()}
    cell_favor_direct = sum(1 for v in per_cell_mean.values() if v > GAP_THRESHOLD)
    cell_favor_gpy = sum(1 for v in per_cell_mean.values() if v < -GAP_THRESHOLD)
    cell_within = sum(1 for v in per_cell_mean.values() if abs(v) <= GAP_THRESHOLD)
    print(f"  per-cell 3-seed mean Δ (threshold ±{GAP_THRESHOLD:.2f}):")
    print(f"     cells favoring vargp_direct (Δ >  {GAP_THRESHOLD}): {cell_favor_direct}/{len(per_cell_mean)}")
    print(f"     cells within band    (|Δ| ≤  {GAP_THRESHOLD}):      {cell_within}/{len(per_cell_mean)}")
    print(f"     cells favoring default_gpy   (Δ < -{GAP_THRESHOLD}): {cell_favor_gpy}/{len(per_cell_mean)}")
    print()

    print("=" * 78)
    if abs(mean_d) > GAP_THRESHOLD:
        direction = 'vargp_direct' if mean_d > 0 else 'default_gpy'
        print(f"  DECISION: GAP EXISTS at this config.  "
              f"|mean_Δ| = {abs(mean_d):.4f} > {GAP_THRESHOLD:.2f}")
        print(f"            mean_Δ = {mean_d:+.4f} favors {direction}.")
    else:
        print(f"  DECISION: NO GAP DETECTED.  "
              f"|mean_Δ| = {abs(mean_d):.4f} ≤ {GAP_THRESHOLD:.2f}")
    print("=" * 78)
    print()

    return mean_d, std_d, per_cell_mean


# --------------------------------------------------------------------- timing


def print_timing(records):
    print("=" * 78)
    print("TIMING  (wall_time_s per run, and train_time / n_iterations_run)")
    print("=" * 78)
    print(f"  {'mode':<18} {'n':>4}  {'wall_s mean':>14} {'± std':>8}  "
          f"{'s/iter mean':>14} {'± std':>8}")
    print("-" * 78)
    for mode in ['vargp_direct', 'default_gpy', 'default_gpy_alt']:
        sub = [r for r in records
               if r['mode'] == mode and r.get('status') != 'failed']
        walls = np.array([r['wall_time_s'] for r in sub])
        per_iter = np.array([
            r['train_time'] / r['n_iterations_run']
            for r in sub
            if r.get('train_time') and r.get('n_iterations_run')
        ])
        if len(walls) == 0:
            continue
        print(f"  {mode:<18} {len(walls):>4}  "
              f"{walls.mean():>14.2f} {walls.std(ddof=1):>8.2f}  "
              f"{per_iter.mean():>14.3f} {per_iter.std(ddof=1):>8.3f}")
    print()


# --------------------------------------------------------------------- final_A


def print_final_A(records):
    print("=" * 78)
    print("FINAL A DISTRIBUTION  (catches freeze near init 0.01 / explosion)")
    print("=" * 78)
    print(f"  {'mode':<18} {'n':>4}  "
          f"{'min':>9} {'p25':>9} {'median':>9} {'p75':>9} {'max':>9}  "
          f"{'≤ 0.012':>8}")
    print("-" * 78)
    for mode in ['vargp_direct', 'default_gpy', 'default_gpy_alt']:
        sub = [r for r in records
               if r['mode'] == mode and r.get('status') != 'failed'
               and r.get('final_A') is not None]
        A = np.array([r['final_A'] for r in sub])
        if len(A) == 0:
            continue
        near_init = int((A <= 0.012).sum())
        q = np.percentile(A, [0, 25, 50, 75, 100])
        print(f"  {mode:<18} {len(A):>4}  "
              f"{q[0]:>9.4g} {q[1]:>9.4g} {q[2]:>9.4g} {q[3]:>9.4g} {q[4]:>9.4g}  "
              f"{near_init:>8d}")
    print()


# --------------------------------------------------------------------- alt check


def print_alt_sanity(by_key, cells_ordered):
    print("=" * 78)
    print("default_gpy_alt (alternating_fstep=True) seed-42 SANITY CHECK")
    print("-" * 78)
    print("  Paired against default_gpy (seed 42) to see if ALT systematically")
    print("  breaks or helps. This is a per-cell observation, not a decision.")
    print("=" * 78)
    print(f"  {'cell':>4}  {'default_gpy':>12}  {'default_gpy_alt':>16}  "
          f"{'Δ(alt−joint)':>14}  {'final_A_alt':>13}  {'iters_alt':>10}")
    print("-" * 78)
    deltas = []
    for cell in cells_ordered:
        j = by_key.get(('default_gpy', cell, 42))
        a = by_key.get(('default_gpy_alt', cell, 42))
        if a is None or a.get('status') == 'failed':
            print(f"  {cell:>4}  {'-':>12}  {'FAILED':>16}")
            continue
        tr_j = j.get('test_r') if j else None
        tr_a = a['test_r']
        d = (tr_a - tr_j) if (tr_j is not None and tr_a is not None) else None
        if d is not None:
            deltas.append(d)
        print(f"  {cell:>4}  {fmt(tr_j):>12}  {fmt(tr_a):>16}  "
              f"{fmt(d, 4):>14}  {a['final_A']:>13.4g}  "
              f"{a['n_iterations_run']:>10}")
    if deltas:
        d = np.array(deltas)
        print("-" * 78)
        print(f"  Δ(alt−joint) over {len(d)} cells: "
              f"mean={d.mean():+.4f}  std={d.std(ddof=1):.4f}  "
              f"range=[{d.min():+.4f}, {d.max():+.4f}]  "
              f"cells with |Δ|>0.02: {(np.abs(d) > 0.02).sum()}")
    print()


# --------------------------------------------------------------------- main


def main():
    records = load_records()
    cells_ordered, cells_meta = load_cells_ordered()
    by_key = group_by_mode_cell_seed(records)

    print_status_overview(records)
    print_per_cell_table(by_key, cells_ordered, cells_meta)
    deltas_full = compute_paired_deltas(by_key, cells_ordered)
    print_paired_summary(deltas_full, cells_ordered, by_key)
    print_timing(records)
    print_final_A(records)
    print_alt_sanity(by_key, cells_ordered)


if __name__ == '__main__':
    main()
