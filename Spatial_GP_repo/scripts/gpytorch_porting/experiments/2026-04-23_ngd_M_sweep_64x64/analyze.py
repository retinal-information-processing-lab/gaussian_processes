"""Analysis for NGD+Adam M-sweep.

Loads this sweep (results.jsonl) and the vargp M-sweep reference
(experiments/2026-04-13_M_sweep_64x64/M_sweep_results.jsonl) and reports:

  1. Per-M summary: mean/median/std test_r across 41 cells × 3 seeds
  2. M-degradation check: cells where test_r drops > 0.05 from their peak
  3. Paired Δ(NGD − vargp) at M values present in both sweeps
  4. Consistency check: M=250 here vs Phase 3C (experiments/2026-04-22...)

Run after at least one M completes for an early progress check, or after
all 1107 runs for the full picture.

Usage:
    python analyze.py              # all completed M values
    python analyze.py --M 50 250  # specific M values only
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
NGD_JSONL = EXP_DIR / 'results.jsonl'
VARGP_JSONL = ROOT / 'experiments/2026-04-13_M_sweep_64x64/M_sweep_results.jsonl'
PHASE3C_JSONL = ROOT / 'experiments/2026-04-22_ngd_final_verdict_64x64/results.jsonl'

M_VALUES_EXPECTED = [50, 100, 200, 250, 300, 500, 750, 1000, 1500]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def by_key(recs, keys):
    d = {}
    for r in recs:
        k = tuple(r[kk] for kk in keys)
        d[k] = r
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, nargs='*', help='M values to analyze (default: all)')
    args = parser.parse_args()

    ngd = [r for r in load_jsonl(NGD_JSONL) if r.get('test_r') is not None]
    print(f"NGD records loaded: {len(ngd)} successful")

    M_present = sorted({r['M'] for r in ngd})
    M_filter = set(args.M) if args.M else set(M_present)
    M_show = sorted(M_present & M_filter)
    print(f"M values present: {M_present}")
    print(f"M values to report: {M_show}")

    # Group by M
    by_M = defaultdict(list)
    for r in ngd:
        by_M[r['M']].append(r)

    # --- 1. Per-M summary ---
    print()
    print('=' * 80)
    print('NGD+Adam PER-M SUMMARY (mean ± std, median, n_runs)')
    print('=' * 80)
    print(f"{'M':>5}  {'n':>4}  {'mean_r':>7}  {'std':>6}  {'median':>7}  "
          f"{'min':>7}  {'max':>7}  {'wall_mean':>10}")
    peak_mean = {}
    for M in M_show:
        recs = by_M[M]
        trs = [r['test_r'] for r in recs]
        walls = [r['wall_time_s'] for r in recs if r.get('wall_time_s')]
        m = stats.mean(trs)
        peak_mean[M] = m
        print(f"{M:>5}  {len(trs):>4}  {m:>7.4f}  {stats.stdev(trs):>6.4f}  "
              f"{stats.median(trs):>7.4f}  {min(trs):>7.4f}  {max(trs):>7.4f}  "
              f"{stats.mean(walls):>9.1f}s")

    # --- 2. M-degradation check ---
    print()
    print('=' * 80)
    print('M-DEGRADATION CHECK (cells where test_r drops > 0.05 from peak)')
    print('=' * 80)
    by_cell_M = defaultdict(dict)
    for r in ngd:
        by_cell_M[r['cell']][r['M']] = r

    degraded = []
    for c in sorted(by_cell_M):
        cell_recs = by_cell_M[c]
        m_tr = {M: stats.mean(r['test_r'] for r in [cell_recs[M]])
                if M in cell_recs else None
                for M in M_show}
        valid = {M: v for M, v in m_tr.items() if v is not None}
        if not valid:
            continue
        peak = max(valid.values())
        peak_M = max(valid, key=lambda m: valid[m])
        low_Ms = [M for M, v in valid.items() if M > peak_M and v < peak - 0.05]
        if low_Ms:
            degraded.append((c, peak, peak_M, low_Ms,
                             min(valid[M] for M in low_Ms)))

    if degraded:
        print(f"  {'cell':>4}  {'peak_r':>7}  {'peak_M':>7}  "
              f"{'degrad_at_M':>12}  {'min_r':>7}  {'drop':>7}")
        for c, pk, pm, bad_Ms, min_r in sorted(degraded, key=lambda x: -x[1]):
            print(f"  {c:>4}  {pk:>7.4f}  {pm:>7}  "
                  f"{str(bad_Ms):>12}  {min_r:>7.4f}  {pk-min_r:>+7.4f}")
    else:
        print(f"  None (no cell drops > 0.05 from peak across M values analysed).")
    print(f"\n  Degraded cells: {len(degraded)} / {len(by_cell_M)}")

    # --- 3. Paired Δ(NGD − vargp) ---
    print()
    print('=' * 80)
    print('PAIRED Δ(NGD − vargp_direct) AT MATCHING M VALUES')
    print('  (vargp ref: experiments/2026-04-13_M_sweep_64x64/)')
    print('=' * 80)
    if not VARGP_JSONL.exists():
        print('  vargp JSONL not found — skipping.')
    else:
        vargp = [r for r in load_jsonl(VARGP_JSONL) if r.get('test_r') is not None]
        vargp_M = sorted({r['M'] for r in vargp})
        overlap_M = sorted(set(M_show) & set(vargp_M))
        print(f"  vargp M values available: {vargp_M}")
        print(f"  overlap with this sweep:  {overlap_M}")
        print()
        by_ngd_cMs = by_key(ngd, ['cell', 'M', 'seed'])
        by_v_cMs = by_key(vargp, ['cell', 'M', 'seed'])
        print(f"  {'M':>5}  {'n_pairs':>7}  {'mean_Δ':>8}  {'SEM':>6}  "
              f"{'std_Δ':>7}  {'range':>18}  {'Δ>+0.02':>7}  {'|Δ|≤0.02':>8}  {'Δ<-0.02':>7}")
        for M in overlap_M:
            pairs = [(by_ngd_cMs[(c, M, s)]['test_r'], by_v_cMs[(c, M, s)]['test_r'])
                     for c in range(41) for s in SEEDS
                     if (c, M, s) in by_ngd_cMs and (c, M, s) in by_v_cMs]
            if not pairs:
                continue
            ds = [n - v for n, v in pairs]
            n = len(ds)
            m = stats.mean(ds)
            sd = stats.stdev(ds)
            sem = sd / math.sqrt(n)
            print(f"  {M:>5}  {n:>7}  {m:>+8.4f}  {sem:>6.4f}  "
                  f"{sd:>7.4f}  [{min(ds):+.3f}, {max(ds):+.3f}]  "
                  f"{sum(1 for d in ds if d>0.02):>7}  "
                  f"{sum(1 for d in ds if abs(d)<=0.02):>8}  "
                  f"{sum(1 for d in ds if d<-0.02):>7}")

    # --- 4. Phase 3C consistency check ---
    print()
    print('=' * 80)
    print('CONSISTENCY CHECK: M=250 here vs Phase 3C (seeds 1,2,3 vs 0,1,2)')
    print('  (Phase 3C ref: experiments/2026-04-22_ngd_final_verdict_64x64/)')
    print('=' * 80)
    if not PHASE3C_JSONL.exists():
        print('  Phase 3C JSONL not found — skipping.')
    elif 250 not in by_M:
        print('  M=250 not yet in this sweep.')
    else:
        p3c = [r for r in load_jsonl(PHASE3C_JSONL) if r.get('test_r') is not None]
        this_250 = [r['test_r'] for r in by_M[250]]
        p3c_250 = [r['test_r'] for r in p3c]
        print(f"  This sweep M=250 (seeds 0,1,2): "
              f"n={len(this_250)} mean={stats.mean(this_250):.4f} std={stats.stdev(this_250):.4f}")
        print(f"  Phase 3C M=250  (seeds 1,2,3): "
              f"n={len(p3c_250)} mean={stats.mean(p3c_250):.4f} std={stats.stdev(p3c_250):.4f}")
        print(f"  Δ means = {stats.mean(this_250)-stats.mean(p3c_250):+.4f}  "
              f"(expect ~0 if seed-stable)")


if __name__ == '__main__':
    main()
