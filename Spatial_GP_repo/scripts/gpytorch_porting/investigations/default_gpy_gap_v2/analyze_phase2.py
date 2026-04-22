"""Phase 2 analysis: 3-seed picture of beta_init=0.2 intervention vs baselines.

Combines step3_results.jsonl (seed 42) + step4_results.jsonl (seeds 123, 789)
for the beta=0.2 intervention, and compares against:
  - baseline default_gpy (3 seeds) from main sweep results.jsonl
  - vargp_direct (3 seeds) from main sweep results.jsonl

Produces:
  1. Per-cell table: 3-seed means of test_r under each mode + paired deltas.
  2. Aggregate: mean Δ(beta=0.2 - baseline default_gpy), mean Δ(beta=0.2 - vargp).
  3. Per-cell consistency: does beta=0.2 consistently help/harm each cell
     across seeds, or is it noisy?
  4. Final beta/A distributions per mode.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
SEEDS = [42, 123, 789]


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path)]


def collect_beta0p2():
    """Return (cell, seed) -> record for all beta=0.2 runs (seeds 42, 123, 789)."""
    records = load_jsonl(ROOT / 'step3_results.jsonl')  # seed 42
    records += load_jsonl(ROOT / 'step4_results.jsonl')  # seeds 123, 789
    return {(r['cell'], r['seed']): r for r in records}


def collect_main(mode):
    records = [r for r in load_jsonl(ROOT / 'results.jsonl') if r['mode'] == mode]
    return {(r['cell'], r['seed']): r for r in records}


def main():
    beta0p2 = collect_beta0p2()
    baseline_gpy = collect_main('default_gpy')
    vargp = collect_main('vargp_direct')

    with open(ROOT / 'cells_used.json') as f:
        cells = [c['cell_id'] for c in json.load(f)['cells']]

    print("=" * 116)
    print("Phase 2 — 3-seed beta_init=0.2 analysis")
    print("=" * 116)
    header = (f"  {'cell':>4}  "
              f"{'beta=0.2 (3 seeds)':<26}  "
              f"{'baseline gpy (3 seeds)':<26}  "
              f"{'d_gpy 3s':>9}  {'d_vargp 3s':>11}  {'sign(d_gpy)':>11}")
    print(header)
    print("-" * 116)

    per_cell_deltas_gpy = []
    per_cell_deltas_vargp = []
    paired_deltas_gpy = []
    paired_deltas_vargp = []
    per_cell_results = []

    for cell in cells:
        tr_new = []
        tr_gpy = []
        tr_vargp = []
        signs_gpy = []  # per-seed sign of d vs baseline gpy
        for s in SEEDS:
            r_new = beta0p2.get((cell, s))
            r_gpy = baseline_gpy.get((cell, s))
            r_vargp = vargp.get((cell, s))
            if r_new and r_new.get('test_r') is not None:
                tr_new.append(r_new['test_r'])
            if r_gpy and r_gpy.get('test_r') is not None:
                tr_gpy.append(r_gpy['test_r'])
            if r_vargp and r_vargp.get('test_r') is not None:
                tr_vargp.append(r_vargp['test_r'])
            if r_new and r_gpy and r_new.get('test_r') is not None and r_gpy.get('test_r') is not None:
                d = r_new['test_r'] - r_gpy['test_r']
                paired_deltas_gpy.append((cell, s, d))
                signs_gpy.append(np.sign(d))
            if r_new and r_vargp and r_new.get('test_r') is not None and r_vargp.get('test_r') is not None:
                paired_deltas_vargp.append((cell, s, r_new['test_r'] - r_vargp['test_r']))

        mu_new = np.mean(tr_new)
        sd_new = np.std(tr_new, ddof=1) if len(tr_new) > 1 else 0
        mu_gpy = np.mean(tr_gpy)
        sd_gpy = np.std(tr_gpy, ddof=1) if len(tr_gpy) > 1 else 0
        mu_vargp = np.mean(tr_vargp)
        d_gpy_3s = mu_new - mu_gpy
        d_vargp_3s = mu_new - mu_vargp
        per_cell_deltas_gpy.append((cell, d_gpy_3s))
        per_cell_deltas_vargp.append((cell, d_vargp_3s))
        sign_str = ('+' if s > 0 else ('-' if s < 0 else '0') for s in signs_gpy)
        sign_summary = ''.join(sign_str)
        per_cell_results.append((cell, mu_new, sd_new, mu_gpy, sd_gpy, mu_vargp, d_gpy_3s, d_vargp_3s, sign_summary))

    # sort by d_gpy descending to show winners first
    per_cell_results.sort(key=lambda x: -x[6])
    for cell, mu_new, sd_new, mu_gpy, sd_gpy, mu_vargp, d_gpy, d_vargp, signs in per_cell_results:
        print(f"  {cell:>4}  "
              f"{mu_new:.4f} ± {sd_new:.4f}         "
              f"{mu_gpy:.4f} ± {sd_gpy:.4f}         "
              f"{d_gpy:>+9.4f}  {d_vargp:>+11.4f}  {signs:>11}")

    print("-" * 116)

    # Aggregate
    vals_gpy = np.array([d for _, _, d in paired_deltas_gpy])
    vals_vargp = np.array([d for _, _, d in paired_deltas_vargp])
    print()
    print(f"PAIRED Δ (beta=0.2 − baseline default_gpy) over {len(vals_gpy)} (cell×seed) pairs:")
    print(f"   mean = {vals_gpy.mean():+.4f}   std = {vals_gpy.std(ddof=1):.4f}   "
          f"range = [{vals_gpy.min():+.4f}, {vals_gpy.max():+.4f}]")
    print(f"   sign counts: +>0.02: {(vals_gpy > 0.02).sum()},  "
          f"|Δ|<=0.02: {((vals_gpy >= -0.02) & (vals_gpy <= 0.02)).sum()},  "
          f"<-0.02: {(vals_gpy < -0.02).sum()}")
    print()
    print(f"PAIRED Δ (beta=0.2 − vargp_direct) over {len(vals_vargp)} (cell×seed) pairs:")
    print(f"   mean = {vals_vargp.mean():+.4f}   std = {vals_vargp.std(ddof=1):.4f}   "
          f"range = [{vals_vargp.min():+.4f}, {vals_vargp.max():+.4f}]")
    print(f"   pre-registered threshold (|Δ|>0.02 = GAP): "
          f"{'GAP' if abs(vals_vargp.mean()) > 0.02 else 'NO GAP'}")
    print()

    # Reference: original baseline vargp vs default_gpy gap
    base_deltas = []
    for cell in cells:
        for s in SEEDS:
            vr = vargp.get((cell, s)) or {}
            gr = baseline_gpy.get((cell, s)) or {}
            if vr.get('test_r') is not None and gr.get('test_r') is not None:
                base_deltas.append(vr['test_r'] - gr['test_r'])
    base = np.array(base_deltas)
    print(f"Reference: mean Δ(vargp_direct − baseline default_gpy) over {len(base)} pairs:")
    print(f"   {base.mean():+.4f}  ← original v2 decision: GAP EXISTS")
    print()

    # Consistency analysis: which cells are robust winners/losers under beta=0.2?
    print("=" * 116)
    print("Per-cell CONSISTENCY of beta=0.2 effect across seeds:")
    print("=" * 116)
    print(f"  {'cell':>4}  {'signs(d_gpy)':>13}  {'robust_winner':>14}  {'robust_loser':>13}  "
          f"{'3s d_gpy':>9}  {'3s d_vargp':>11}")
    # Build sign matrix
    for cell in cells:
        seed_ds = []
        for s in SEEDS:
            r_new = beta0p2.get((cell, s))
            r_gpy = baseline_gpy.get((cell, s))
            if r_new and r_gpy and r_new.get('test_r') is not None and r_gpy.get('test_r') is not None:
                seed_ds.append(r_new['test_r'] - r_gpy['test_r'])
        signs = ['+' if d > 0.02 else ('-' if d < -0.02 else '0') for d in seed_ds]
        robust_winner = all(d > 0.02 for d in seed_ds)
        robust_loser = all(d < -0.02 for d in seed_ds)
        mean_d_gpy = np.mean(seed_ds) if seed_ds else 0
        # vargp side
        seed_ds_v = []
        for s in SEEDS:
            r_new = beta0p2.get((cell, s))
            r_v = vargp.get((cell, s))
            if r_new and r_v and r_new.get('test_r') is not None and r_v.get('test_r') is not None:
                seed_ds_v.append(r_new['test_r'] - r_v['test_r'])
        mean_d_vargp = np.mean(seed_ds_v) if seed_ds_v else 0
        print(f"  {cell:>4}  {''.join(signs):>13}  "
              f"{('YES' if robust_winner else 'no'):>14}  "
              f"{('YES' if robust_loser else 'no'):>13}  "
              f"{mean_d_gpy:>+9.4f}  {mean_d_vargp:>+11.4f}")


if __name__ == '__main__':
    main()
