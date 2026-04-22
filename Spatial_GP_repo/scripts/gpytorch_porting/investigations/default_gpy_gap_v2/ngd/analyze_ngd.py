"""Phase 3 analysis: NGD vs. vargp_direct vs. default_gpy at the locked
(M=300, n_train=1500, 64x64, seeds {42, 123, 789}) operating point.

Computes:
  - Per-cell mean test_r for each mode (3-seed mean + std)
  - Per-(cell, seed) paired delta vs vargp_direct and vs default_gpy
  - Aggregate mean delta + pre-registered decision (|mean_delta| > 0.02 = GAP)
  - Both "final test_r" (ELBO-argmax, honest apples-to-apples)
    and "best test_r over trajectory" (oracle ES upper bound)
  - Final A distribution (catches A-explosion / freeze)
  - Timing summary

Inputs (paired with the Phase 2 sweep):
  ../results.jsonl             - Phase 2 main sweep (vargp_direct + default_gpy)
  ./ngd_results.jsonl          - Phase 3 NGD sweep

Output: printed tables + ngd_summary.json
"""
from __future__ import annotations

import json
import statistics as stats
from pathlib import Path

import numpy as np

NGD_DIR = Path(__file__).parent
INV_DIR = NGD_DIR.parent
MAIN_JSONL = INV_DIR / 'results.jsonl'
NGD_JSONL = NGD_DIR / 'ngd_results.jsonl'
CELLS_USED = INV_DIR / 'cells_used.json'
SUMMARY_JSON = NGD_DIR / 'ngd_summary.json'


def load_jsonl(path):
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def idx_by(mode, recs):
    return {(r['cell'], r['seed']): r for r in recs if r.get('mode') == mode}


def summarize():
    main = load_jsonl(MAIN_JSONL)
    ngd = load_jsonl(NGD_JSONL)

    with open(CELLS_USED) as f:
        cells_info = json.load(f)['cells']
    cell_order = [c['cell_id'] for c in cells_info]
    cell_bucket = {c['cell_id']: c.get('bucket', '?') for c in cells_info}
    cell_fr = {c['cell_id']: c.get('firing_rate', float('nan')) for c in cells_info}

    vargp = idx_by('vargp_direct', main)
    defgpy = idx_by('default_gpy', main)
    ngd_by = idx_by('ngd', ngd)
    # drop crashed entries
    ngd_by = {k: v for k, v in ngd_by.items() if v.get('test_r') is not None}

    seeds = sorted({s for (_, s) in ngd_by})

    print(f"Cells: {len(cell_order)}  Seeds: {seeds}")
    print(f"Runs: vargp={len(vargp)}  default_gpy={len(defgpy)}  ngd={len(ngd_by)}")
    print()

    # === Per-cell table ===
    print("=" * 130)
    print("PER-CELL 3-SEED MEAN test_r  (FINAL = at iter 1000, BEST = oracle-ES peak over probe trajectory)")
    print("=" * 130)
    print(f"  {'cell':>4} {'bkt':>4} {'fr':>6}  "
          f"{'vargp_μ':>8} {'vargp_σ':>8}   "
          f"{'defgpy_μ':>9} {'defgpy_σ':>9}   "
          f"{'ngdF_μ':>7} {'ngdF_σ':>7}   "
          f"{'ngdB_μ':>7} {'ngdB_σ':>7}   "
          f"{'Δ(F-vg)':>9} {'Δ(B-vg)':>9}")

    per_cell = {}
    for c in cell_order:
        rows_vg = [vargp.get((c, s), {}).get('test_r') for s in seeds]
        rows_dg = [defgpy.get((c, s), {}).get('test_r') for s in seeds]
        rows_nf = [ngd_by.get((c, s), {}).get('test_r') for s in seeds]
        rows_nb = [ngd_by.get((c, s), {}).get('best_test_r_probe') for s in seeds]
        def stat(xs):
            xs = [x for x in xs if x is not None]
            if not xs:
                return None, None
            return stats.mean(xs), stats.stdev(xs) if len(xs) >= 2 else 0.0
        m_vg, s_vg = stat(rows_vg)
        m_dg, s_dg = stat(rows_dg)
        m_nf, s_nf = stat(rows_nf)
        m_nb, s_nb = stat(rows_nb)
        d_fvg = (m_nf - m_vg) if (m_nf is not None and m_vg is not None) else None
        d_bvg = (m_nb - m_vg) if (m_nb is not None and m_vg is not None) else None
        per_cell[c] = dict(
            bucket=cell_bucket[c], fr=cell_fr[c],
            vargp=(m_vg, s_vg), defgpy=(m_dg, s_dg),
            ngdF=(m_nf, s_nf), ngdB=(m_nb, s_nb),
            delta_F_vg=d_fvg, delta_B_vg=d_bvg,
        )
        def fmt(x, f):
            return f.format(x) if x is not None else ' N/A'
        print(f"  {c:>4} {cell_bucket[c]:>4} {cell_fr[c]:>6.2f}  "
              f"{fmt(m_vg, '{:>8.4f}')} {fmt(s_vg, '{:>8.4f}')}   "
              f"{fmt(m_dg, '{:>9.4f}')} {fmt(s_dg, '{:>9.4f}')}   "
              f"{fmt(m_nf, '{:>7.4f}')} {fmt(s_nf, '{:>7.4f}')}   "
              f"{fmt(m_nb, '{:>7.4f}')} {fmt(s_nb, '{:>7.4f}')}   "
              f"{fmt(d_fvg, '{:>+9.4f}')} {fmt(d_bvg, '{:>+9.4f}')}")

    # === Paired Δ summary ===
    print()
    print("=" * 90)
    print("PAIRED Δ  (per (cell, seed))")
    print("=" * 90)

    def paired(a, b, key_a='test_r', key_b='test_r'):
        """Return list of a[k]-b[k] over common keys."""
        out = []
        for k in a:
            if k in b and a[k].get(key_a) is not None and b[k].get(key_b) is not None:
                out.append(a[k][key_a] - b[k][key_b])
        return out

    # Final (ELBO-argmax)
    d_final_vs_vargp = paired(ngd_by, vargp, 'test_r', 'test_r')
    d_final_vs_defgpy = paired(ngd_by, defgpy, 'test_r', 'test_r')
    # Oracle (best over probe)
    d_best_vs_vargp = paired(ngd_by, vargp, 'best_test_r_probe', 'test_r')
    d_best_vs_defgpy = paired(ngd_by, defgpy, 'best_test_r_probe', 'test_r')

    for label, d in [
        ('Δ(ngd_final − vargp_direct)',  d_final_vs_vargp),
        ('Δ(ngd_final − default_gpy)',   d_final_vs_defgpy),
        ('Δ(ngd_best  − vargp_direct)',  d_best_vs_vargp),
        ('Δ(ngd_best  − default_gpy)',   d_best_vs_defgpy),
    ]:
        arr = np.array(d)
        if len(arr) == 0:
            print(f"  {label}: no data"); continue
        print(f"  {label}:  n={len(arr):>3}   "
              f"mean={arr.mean():>+.4f}   "
              f"std={arr.std(ddof=1):>.4f}   "
              f"range=[{arr.min():>+.4f}, {arr.max():>+.4f}]   "
              f"pos/neg={int((arr > 0.02).sum())}/{int((arr < -0.02).sum())}/{int(((arr >= -0.02) & (arr <= 0.02)).sum())}")

    # === Pre-registered decision ===
    print()
    print("=" * 90)
    print("PRE-REGISTERED DECISION (|mean_Δ| > 0.02  =>  GAP EXISTS)")
    print("=" * 90)
    for label, d in [
        ('NGD_final vs vargp_direct',  d_final_vs_vargp),
        ('NGD_best  vs vargp_direct',  d_best_vs_vargp),
    ]:
        if not d:
            continue
        m = np.mean(d)
        decision = "GAP EXISTS" if abs(m) > 0.02 else "NO GAP"
        sign = "favors vargp" if m < 0 else "favors NGD"
        print(f"  {label:<28}  mean_Δ={m:>+.4f}   |mean_Δ|={abs(m):>.4f}   {decision} ({sign})")

    print()
    # Phase 2 baseline for comparison
    baseline_d = paired(vargp, defgpy, 'test_r', 'test_r')
    if baseline_d:
        m_bl = np.mean(baseline_d)
        print(f"  (Phase 2 baseline: vargp − default_gpy  mean_Δ={m_bl:+.4f}, n={len(baseline_d)})")

    # === Final A distribution ===
    print()
    print("=" * 90)
    print("FINAL A DISTRIBUTION  (catches explosion / freeze)")
    print("=" * 90)
    print(f"  {'mode':<24} {'n':>4} {'min':>9} {'p25':>9} {'median':>9} {'p75':>9} {'max':>9} {'≤0.012':>7}")
    for mode_name, src in [
        ('vargp_direct',     [r for r in main if r['mode'] == 'vargp_direct']),
        ('default_gpy',      [r for r in main if r['mode'] == 'default_gpy']),
        ('ngd',              list(ngd_by.values())),
    ]:
        As = [r['final_A'] for r in src if r.get('final_A') is not None]
        if not As:
            continue
        arr = np.array(As)
        print(f"  {mode_name:<24} {len(arr):>4} "
              f"{arr.min():>9.4g} {np.percentile(arr, 25):>9.4g} "
              f"{np.median(arr):>9.4g} {np.percentile(arr, 75):>9.4g} "
              f"{arr.max():>9.4g} {int((arr <= 0.012).sum()):>7}")

    # === Timing ===
    print()
    print("=" * 90)
    print("TIMING")
    print("=" * 90)
    print(f"  {'mode':<24} {'n':>4} {'wall_s mean±std':>22} {'s/iter mean±std':>22}")
    for mode_name, src in [
        ('vargp_direct',     [r for r in main if r['mode'] == 'vargp_direct']),
        ('default_gpy',      [r for r in main if r['mode'] == 'default_gpy']),
        ('ngd',              list(ngd_by.values())),
    ]:
        walls = []
        perit = []
        for r in src:
            w = r.get('train_time') or r.get('train_time_s') or r.get('wall_time_s')
            n = r.get('n_iterations_run') or r.get('final_iteration') or None
            if w is not None:
                walls.append(w)
            if w is not None and n is not None and n > 0:
                perit.append(w / n)
        if not walls:
            continue
        arrw = np.array(walls)
        arri = np.array(perit) if perit else np.array([float('nan')])
        print(f"  {mode_name:<24} {len(walls):>4}  "
              f"{arrw.mean():>8.2f} ± {arrw.std(ddof=1 if len(arrw) > 1 else 0):<7.2f}  "
              f"{arri.mean():>8.4f} ± {arri.std(ddof=1 if len(arri) > 1 else 0):<7.4f}")

    # === Where does NGD peak? (probe trajectory summary) ===
    print()
    print("=" * 90)
    print("NGD PEAK test_r ITER DISTRIBUTION  (where oracle ES would have stopped)")
    print("=" * 90)
    iters = [r['best_test_r_iter_probe'] for r in ngd_by.values() if r.get('best_test_r_iter_probe') is not None]
    if iters:
        arr = np.array(iters)
        print(f"  n={len(arr)}  min={arr.min()}  p25={np.percentile(arr, 25):.0f}  "
              f"median={np.median(arr):.0f}  p75={np.percentile(arr, 75):.0f}  max={arr.max()}")

    # === Save summary JSON ===
    summary = {
        'n_seeds': len(seeds), 'seeds': seeds,
        'n_cells': len(cell_order),
        'n_runs_ngd': len(ngd_by),
        'mean_delta_final_vs_vargp':  (float(np.mean(d_final_vs_vargp)) if d_final_vs_vargp else None),
        'mean_delta_final_vs_defgpy': (float(np.mean(d_final_vs_defgpy)) if d_final_vs_defgpy else None),
        'mean_delta_best_vs_vargp':   (float(np.mean(d_best_vs_vargp)) if d_best_vs_vargp else None),
        'mean_delta_best_vs_defgpy':  (float(np.mean(d_best_vs_defgpy)) if d_best_vs_defgpy else None),
        'per_cell': {
            c: {k: (list(v) if isinstance(v, tuple) else v) for k, v in per_cell[c].items()}
            for c in per_cell
        },
    }
    with open(SUMMARY_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written to: {SUMMARY_JSON.name}")


if __name__ == '__main__':
    summarize()
