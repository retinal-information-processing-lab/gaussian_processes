"""
Analyze the hyperparam prior validation sweep against the baseline
experiment 2026-04-13_M_sweep_64x64.

Produces:
 - Per-cell per-M table: baseline test_r, fix test_r, delta.
 - Per-cell trend (delta M=50 -> M=1500) baseline vs fix.
 - Success criteria summary (which cells saved, which controls preserved).
 - Hyperparameter drift table (final_A M=50 vs M=1500, baseline vs fix).

Run from gpytorch_porting/:
  python experiments/2026-04-14_hyperparam_prior_validation/analyze.py
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
FIX_JSONL = HERE / 'results.jsonl'
BASELINE_JSONL = HERE.parent / '2026-04-13_M_sweep_64x64' / 'M_sweep_results.jsonl'

DEGRADERS = [39, 35, 16, 13, 10, 33, 15, 14, 27]
CONTROLS = [1, 8]
CELLS = DEGRADERS + CONTROLS
MS = [50, 300, 1500]

SUCCESS_TREND_TOL = 0.005   # degrader "saved" if M=50->M=1500 delta > -SUCCESS_TREND_TOL
SUCCESS_CONTROL_TOL = 0.005 # control "preserved" if |fix - baseline| < SUCCESS_CONTROL_TOL


def load(path, cells, Ms):
    by_cm = defaultdict(list)
    if not path.exists():
        return by_cm
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
            except Exception:
                continue
            if r.get('test_r') is None:
                continue
            if r.get('cell') in cells and r.get('M') in Ms:
                by_cm[(r['cell'], r['M'])].append(r)
    return by_cm


def mean_over_seeds(by_cm, cell, M, field):
    rows = by_cm.get((cell, M), [])
    vals = [r[field] for r in rows if r.get(field) is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def main():
    base = load(BASELINE_JSONL, CELLS, MS)
    fix = load(FIX_JSONL, CELLS, MS)

    print("=" * 90)
    print("Hyperparameter Prior Validation — analysis against baseline")
    print("=" * 90)
    print(f"Baseline: {BASELINE_JSONL}")
    print(f"Fix:      {FIX_JSONL}")
    print()

    # How many runs in each?
    total_base = sum(len(rs) for rs in base.values())
    total_fix = sum(len(rs) for rs in fix.values())
    print(f"Records used: baseline={total_base}  fix={total_fix}  "
          f"(grid = {len(CELLS)} cells x {len(MS)} M x 3 seeds = {len(CELLS)*len(MS)*3})")
    print()

    # ========= Per-cell test_r table =========
    print("TEST_R (mean over 3 seeds)")
    print(f"{'cell':>5}  {'| baseline M=50/300/1500':>35}  "
          f"{'| fix M=50/300/1500':>35}  {'| delta_1500':>12}")
    print("-" * 95)

    cell_summary = {}  # cell -> {'saved_or_preserved': bool, 'notes': str}
    for cell in CELLS:
        bl = [mean_over_seeds(base, cell, M, 'test_r') for M in MS]
        fx = [mean_over_seeds(fix, cell, M, 'test_r') for M in MS]

        def fmt(lst):
            return "  ".join(f"{v:.4f}" if v is not None else "  N/A " for v in lst)

        delta1500 = None
        if bl[-1] is not None and fx[-1] is not None:
            delta1500 = fx[-1] - bl[-1]
        star = ""
        # Degrader: compare fix M=1500 vs fix M=50 (is the trend saved?)
        if cell in DEGRADERS and fx[0] is not None and fx[-1] is not None:
            trend = fx[-1] - fx[0]
            if trend >= -SUCCESS_TREND_TOL:
                star = " *SAVED*"
                cell_summary[cell] = {'saved': True, 'trend': trend}
            else:
                cell_summary[cell] = {'saved': False, 'trend': trend}
        elif cell in CONTROLS and bl[-1] is not None and fx[-1] is not None:
            if abs(delta1500) <= SUCCESS_CONTROL_TOL:
                star = " *OK*"
                cell_summary[cell] = {'preserved': True, 'delta1500': delta1500}
            else:
                star = " *HURT*" if delta1500 < 0 else ""
                cell_summary[cell] = {'preserved': False, 'delta1500': delta1500}

        dstr = f"{delta1500:+.4f}" if delta1500 is not None else "  N/A "
        print(f"{cell:>5}  | {fmt(bl):>34}  | {fmt(fx):>34}  | {dstr:>12}{star}")

    # ========= Trend summary =========
    print()
    print("TREND (fix test_r @ M=1500 − fix test_r @ M=50) per cell")
    print(f"{'cell':>5}  {'type':>12}  {'trend':>8}  {'criterion':>12}")
    print("-" * 45)
    saved = 0
    preserved = 0
    for cell in DEGRADERS:
        f50 = mean_over_seeds(fix, cell, 50, 'test_r')
        f1500 = mean_over_seeds(fix, cell, 1500, 'test_r')
        if f50 is None or f1500 is None:
            print(f"{cell:>5}  {'degrader':>12}  {'  N/A  ':>8}  {'N/A':>12}")
            continue
        trend = f1500 - f50
        ok = trend >= -SUCCESS_TREND_TOL
        if ok: saved += 1
        print(f"{cell:>5}  {'degrader':>12}  {trend:+.4f}  {'SAVED' if ok else 'FAILED':>12}")
    for cell in CONTROLS:
        b1500 = mean_over_seeds(base, cell, 1500, 'test_r')
        f1500 = mean_over_seeds(fix, cell, 1500, 'test_r')
        if b1500 is None or f1500 is None:
            print(f"{cell:>5}  {'control':>12}  {'  N/A  ':>8}  {'N/A':>12}")
            continue
        delta = f1500 - b1500
        ok = abs(delta) <= SUCCESS_CONTROL_TOL
        if ok: preserved += 1
        print(f"{cell:>5}  {'control':>12}  {delta:+.4f}  {'OK' if ok else 'HURT':>12}")
    print()
    print(f"Degraders saved:   {saved} / {len(DEGRADERS)}")
    print(f"Controls preserved: {preserved} / {len(CONTROLS)}")

    # ========= Final A drift =========
    print()
    print("FINAL A DRIFT (final_A, mean over 3 seeds)")
    print(f"{'cell':>5}  {'| baseline M=50->1500':>30}  {'| fix M=50->1500':>30}  {'| fix dA ratio':>14}")
    print("-" * 90)
    for cell in CELLS:
        bA = [mean_over_seeds(base, cell, M, 'final_A') for M in MS]
        fA = [mean_over_seeds(fix, cell, M, 'final_A') for M in MS]

        def fmt(lst):
            parts = []
            for v in lst:
                parts.append(f"{v:.4f}" if v is not None else "  N/A ")
            return "  ".join(parts)

        ratio = None
        if fA[0] is not None and fA[-1] is not None and fA[0] > 0:
            ratio = fA[-1] / fA[0]
        rstr = f"{ratio:.3f}x" if ratio is not None else "  N/A "
        print(f"{cell:>5}  | {fmt(bA):>29}  | {fmt(fA):>29}  | {rstr:>14}")

    # ========= Grand means =========
    print()
    print("GRAND MEAN test_r across the 11 validation cells")
    for M in MS:
        bl_vals = [mean_over_seeds(base, c, M, 'test_r') for c in CELLS]
        fx_vals = [mean_over_seeds(fix, c, M, 'test_r') for c in CELLS]
        bl_vals = [v for v in bl_vals if v is not None]
        fx_vals = [v for v in fx_vals if v is not None]
        print(f"  M={M:>4d}  baseline={np.mean(bl_vals):.4f} (n={len(bl_vals)})  "
              f"fix={np.mean(fx_vals):.4f} (n={len(fx_vals)})  "
              f"delta={np.mean(fx_vals) - np.mean(bl_vals):+.4f}")


if __name__ == '__main__':
    main()
