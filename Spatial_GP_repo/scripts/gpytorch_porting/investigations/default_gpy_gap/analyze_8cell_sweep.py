"""Analyze the 8-cell × 3-seed × 3-mode comparison sweep.

Reads results.jsonl and produces:
  1. Per-cell mean ± std test_r for each mode
  2. Per-mode aggregate (mean test_r across all cells/seeds)
  3. Gap table: vargp_direct − default_gpy(joint), vargp_direct − default_gpy_alt
  4. final_A summary per mode (catches A-explosion / A-freeze)
  5. ES behavior (n_iterations_run summary)
  6. A flagged-cell list: cells where default_gpy_alt fails to match vargp_direct
     within 0.02 test_r (a rough threshold; seed spread is ~0.005–0.015).

Usage:
    python investigations/default_gpy_gap/analyze_8cell_sweep.py
"""

import json
import statistics
from pathlib import Path
from collections import defaultdict

RESULTS = Path(__file__).parent / 'results.jsonl'


def load(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return [r for r in out if r.get('status') == 'success']


def group_by(recs, *keys):
    groups = defaultdict(list)
    for r in recs:
        k = tuple(r[key] for key in keys)
        groups[k].append(r)
    return groups


def msd(xs):
    if not xs:
        return float('nan'), float('nan')
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def fmt_msd(xs, fmt='{:.4f} ± {:.4f}'):
    m, s = msd(xs)
    return fmt.format(m, s)


def main():
    recs = load(RESULTS)
    if not recs:
        print('No successful runs in results.jsonl')
        return

    modes = sorted({r['mode'] for r in recs})
    cells = sorted({r['cell'] for r in recs})
    seeds = sorted({r['seed'] for r in recs})

    print(f'Loaded {len(recs)} successful runs: '
          f'{len(modes)} modes × {len(cells)} cells × {len(seeds)} seeds')
    print(f'  modes: {modes}')
    print(f'  cells: {cells}')
    print(f'  seeds: {seeds}')

    # --- 1. Per-cell test_r table ---
    print()
    print('=' * 90)
    print('TEST_R: per-cell mean ± std across 3 seeds')
    print('=' * 90)
    hdr = f'{"cell":>4}  '
    for mode in modes:
        hdr += f'{mode:<22}'
    print(hdr)
    print('-' * 90)
    by_mc = group_by(recs, 'mode', 'cell')
    for cell in cells:
        row = f'{cell:>4}  '
        for mode in modes:
            rs = by_mc.get((mode, cell), [])
            tr = [r['test_r'] for r in rs]
            row += f'{fmt_msd(tr):<22}'
        print(row)

    # --- 2. Aggregate per-mode (pooled across cells+seeds) ---
    print()
    print('=' * 70)
    print('AGGREGATE test_r: pooled across all cells × seeds')
    print('=' * 70)
    for mode in modes:
        rs = [r for r in recs if r['mode'] == mode]
        tr = [r['test_r'] for r in rs]
        ev = [r['explained_var'] for r in rs]
        print(f'  {mode:<20} N={len(rs):>3}  '
              f'test_r={fmt_msd(tr)}  exp_var={fmt_msd(ev)}')

    # --- 3. Paired gaps per (cell, seed) — vargp_direct vs alt / joint ---
    print()
    print('=' * 90)
    print('PAIRED GAP (vargp_direct − other) per (cell, seed): negative = other is better')
    print('=' * 90)
    by_mcs = group_by(recs, 'mode', 'cell', 'seed')

    def gap_rows(other_mode):
        gaps = []
        for cell in cells:
            for seed in seeds:
                d = by_mcs.get(('vargp_direct', cell, seed), [])
                o = by_mcs.get((other_mode, cell, seed), [])
                if d and o:
                    gaps.append((cell, seed, d[0]['test_r'] - o[0]['test_r']))
        return gaps

    for target in ['default_gpy', 'default_gpy_alt']:
        gaps = gap_rows(target)
        gap_vals = [g for _, _, g in gaps]
        m, s = msd(gap_vals)
        worst_alt = max(gaps, key=lambda x: x[2]) if gaps else None
        best_alt = min(gaps, key=lambda x: x[2]) if gaps else None
        print(f'  vargp_direct − {target}:  mean={m:+.4f} std={s:.4f}  '
              f'range=[{min(gap_vals):+.4f}, {max(gap_vals):+.4f}]')
        if worst_alt:
            print(f'    worst gap (target most behind):  cell={worst_alt[0]} seed={worst_alt[1]}  Δ={worst_alt[2]:+.4f}')
            print(f'    best gap  (target closest/ahead): cell={best_alt[0]} seed={best_alt[1]}  Δ={best_alt[2]:+.4f}')

    # --- 4. final_A per mode (A-explosion / freeze diagnostic) ---
    print()
    print('=' * 70)
    print('final_A per mode: ideal range ~0.005–0.05 (vargp_direct regime)')
    print('=' * 70)
    for mode in modes:
        rs = [r for r in recs if r['mode'] == mode]
        As = [r.get('final_A') for r in rs if r.get('final_A') is not None]
        m, s = msd(As)
        mn, mx = (min(As), max(As)) if As else (float('nan'), float('nan'))
        exploded = sum(1 for a in As if a > 0.1)
        frozen = sum(1 for a in As if a < 0.007)   # init=0.01; <0.007 suggests no movement from init_low
        print(f'  {mode:<20}  mean={m:.4f} std={s:.4f}  range=[{mn:.4f}, {mx:.4f}]  '
              f'exploded(>0.1)={exploded}  frozen(<0.007)={frozen}')

    # --- 5. ES behavior ---
    print()
    print('=' * 70)
    print('Iterations run (out of 50) per mode')
    print('=' * 70)
    for mode in modes:
        rs = [r for r in recs if r['mode'] == mode]
        iters = [r['n_iterations_run'] for r in rs]
        stopped = sum(1 for r in rs if r.get('stopped_early'))
        print(f'  {mode:<20}  mean iters={statistics.mean(iters):.1f}  '
              f'stopped_early={stopped}/{len(rs)}')

    # --- 6. Flagged cells: where default_gpy_alt underperforms vargp_direct by > 0.02 ---
    print()
    print('=' * 90)
    print('Cells where default_gpy_alt UNDERPERFORMS vargp_direct by > 0.02 (mean across seeds)')
    print('=' * 90)
    flagged = []
    for cell in cells:
        d_trs = [r['test_r'] for r in by_mc.get(('vargp_direct', cell), [])]
        a_trs = [r['test_r'] for r in by_mc.get(('default_gpy_alt', cell), [])]
        j_trs = [r['test_r'] for r in by_mc.get(('default_gpy', cell), [])]
        if not (d_trs and a_trs):
            continue
        d_mean = statistics.mean(d_trs)
        a_mean = statistics.mean(a_trs)
        j_mean = statistics.mean(j_trs) if j_trs else float('nan')
        if d_mean - a_mean > 0.02:
            flagged.append((cell, d_mean, a_mean, j_mean))
    if flagged:
        print(f'{"cell":>4}  {"vargp_direct":>14}  {"default_gpy_alt":>18}  {"default_gpy":>14}  {"gap (alt)":>10}')
        for cell, d, a, j in flagged:
            print(f'{cell:>4}  {d:>14.4f}  {a:>18.4f}  {j:>14.4f}  {d-a:>+10.4f}')
    else:
        print('  NONE. default_gpy_alt matches vargp_direct within 0.02 on every cell mean.')

    # --- 7. Success criterion summary ---
    print()
    print('=' * 70)
    print('SUCCESS CRITERIA FOR FLIPPING THE DEFAULT')
    print('=' * 70)
    print('  1. default_gpy_alt closes the gap on hard cells (mean gap ≤ 0.02)')
    print('  2. default_gpy_alt does not degrade easy cells beyond seed noise')
    print('  3. No A-explosion / A-freeze pathologies in default_gpy_alt')
    print()
    print('Status:')
    crit1 = 'PASS' if not flagged else f'FAIL ({len(flagged)} cell(s) flagged)'
    print(f'  (1) Closes gap on hard cells:   {crit1}')
    # Criterion 2: any easy cell where alt is worse than joint by > 0.02?
    degraded = []
    for cell in cells:
        a_trs = [r['test_r'] for r in by_mc.get(('default_gpy_alt', cell), [])]
        j_trs = [r['test_r'] for r in by_mc.get(('default_gpy', cell), [])]
        if a_trs and j_trs:
            a_m = statistics.mean(a_trs)
            j_m = statistics.mean(j_trs)
            # "Easy" proxy: joint mode > 0.95 test_r
            if j_m > 0.95 and (j_m - a_m) > 0.02:
                degraded.append((cell, j_m, a_m))
    crit2 = 'PASS' if not degraded else f'FAIL ({len(degraded)} easy cell(s) degraded)'
    print(f'  (2) No degradation on easy:     {crit2}')
    alt_As = [r.get('final_A') for r in recs if r['mode'] == 'default_gpy_alt' and r.get('final_A') is not None]
    alt_exploded = sum(1 for a in alt_As if a > 0.1)
    alt_frozen = sum(1 for a in alt_As if a < 0.007)
    crit3 = 'PASS' if (alt_exploded == 0 and alt_frozen == 0) else f'FAIL (exploded={alt_exploded}, frozen={alt_frozen})'
    print(f'  (3) No A pathologies in alt:    {crit3}')


if __name__ == '__main__':
    main()
