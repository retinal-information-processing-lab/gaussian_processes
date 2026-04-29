"""Compare NGD scaled-ES vs Exp C baseline (NGD default-ES + vargp).

Reports for each M = n_train ∈ {50, 150, 300}:
  - vargp baseline (from Exp C)
  - NGD default-ES (from Exp C)
  - NGD scaled-ES (this sweep)
  - paired Δ vs vargp (closes the gap?)
  - paired Δ vs default-ES NGD (does the fix help?)
  - Disaster comparison (cell-level 'NGD-only' disasters fixed?)
"""
from __future__ import annotations
import json, math, statistics as stats
from collections import defaultdict
from pathlib import Path

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent

NEW   = EXP_DIR / 'results_scaled_es.jsonl'
EXP_C = ROOT / 'experiments/2026-04-28_ngd_validation/results_C.jsonl'

if not NEW.exists() or NEW.stat().st_size == 0:
    print(f"No data yet at {NEW}")
    raise SystemExit

new = [json.loads(l) for l in open(NEW)]
exp_c = [json.loads(l) for l in open(EXP_C)]

new_by = {(r['cell'], r['M']): r for r in new if r.get('test_r') is not None}
v_by = {(r['cell'], r['M']): r for r in exp_c
        if r['mode']=='vargp_direct' and r.get('test_r') is not None}
n_default_by = {(r['cell'], r['M']): r for r in exp_c
                if r['mode']=='ngd' and r.get('test_r') is not None}

def sem(d): return stats.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0

print(f"Scaled-ES sweep: {len(new)} records, {len(new_by)} successful")
print()

for M in [50, 150, 300]:
    pairs = sorted([k for k in new_by if k[1] == M
                    and k in v_by and k in n_default_by])
    if not pairs:
        print(f"M={M}: no overlapping cells yet ({sum(1 for k in new_by if k[1]==M)} new records)")
        continue

    new_trs = [new_by[k]['test_r'] for k in pairs]
    v_trs = [v_by[k]['test_r'] for k in pairs]
    nd_trs = [n_default_by[k]['test_r'] for k in pairs]

    print(f'=== M = n_train = {M}  (n={len(pairs)} paired cells) ===')
    print(f'  vargp baseline       mean={stats.mean(v_trs):.4f}  std={stats.stdev(v_trs):.4f}')
    print(f'  NGD default-ES (1500 iter): mean={stats.mean(nd_trs):.4f}  std={stats.stdev(nd_trs):.4f}')
    print(f'  NGD scaled-ES   ({EXP_DIR.name}): mean={stats.mean(new_trs):.4f}  std={stats.stdev(new_trs):.4f}')

    d_vs_vargp = [n - v for n, v in zip(new_trs, v_trs)]
    d_vs_default = [n - d for n, d in zip(new_trs, nd_trs)]
    d_default_vs_vargp = [d - v for d, v in zip(nd_trs, v_trs)]
    print(f'  Δ scaled NGD − vargp:        {stats.mean(d_vs_vargp):+.4f}  SEM={sem(d_vs_vargp):.4f}')
    print(f'  Δ scaled NGD − default NGD:  {stats.mean(d_vs_default):+.4f}  SEM={sem(d_vs_default):.4f}')
    print(f'  Δ default NGD − vargp:       {stats.mean(d_default_vs_vargp):+.4f}  (the original gap)')
    gap_closed = stats.mean(d_vs_vargp) - stats.mean(d_default_vs_vargp)
    print(f'  → gap reduction by scaling ES: {gap_closed:+.4f}')

    # Disaster check
    new_dis = sum(1 for t in new_trs if t < 0.3)
    v_dis = sum(1 for t in v_trs if t < 0.3)
    nd_dis = sum(1 for t in nd_trs if t < 0.3)
    print(f'  disasters (test_r<0.3): vargp={v_dis} NGD-default={nd_dis} NGD-scaled={new_dis}')

    # NGD-only disasters that may now be fixed
    new_only_dis = [k[0] for k in pairs
                    if new_by[k]['test_r'] < 0.3 and v_by[k]['test_r'] >= 0.3]
    n_default_only_dis = [k[0] for k in pairs
                          if n_default_by[k]['test_r'] < 0.3 and v_by[k]['test_r'] >= 0.3]
    print(f'  scaled-NGD-only disasters: {sorted(new_only_dis)}')
    print(f'  default-NGD-only disasters: {sorted(n_default_only_dis)}')
    fixed = set(n_default_only_dis) - set(new_only_dis)
    print(f'  → cells where scaling FIXED the NGD-only-disaster: {sorted(fixed)} ({len(fixed)} cells)')
    print()

# Wall time and iter stats
print('=== Compute cost ===')
for M in [50, 150, 300]:
    rs = [r for r in new if r.get('M')==M and r.get('wall_time_s')]
    if not rs: continue
    walls = [r['wall_time_s'] for r in rs]
    iters = [r['n_iterations_run'] for r in rs if r.get('n_iterations_run')]
    es_count = sum(1 for r in rs if r.get('stopped_early'))
    print(f'  M={M:>3}: n={len(rs)}  wall mean={stats.mean(walls):.0f}s  '
          f'iters mean={stats.mean(iters) if iters else 0:.0f}  ES rate='
          f'{es_count}/{len(rs)}')
