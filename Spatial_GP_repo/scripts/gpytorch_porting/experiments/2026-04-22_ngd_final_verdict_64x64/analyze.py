"""Phase 3C analysis: NGD+Adam (this sweep) vs vargp_direct intl_fixAmp (reference).

Loads:
  ./results.jsonl                                                 — this sweep (NGD, 123 runs)
  ../2026-04-06_es_sweeps_64x64/sweep_64x64_elbo_es_results.jsonl — vargp best-ever

Applies the pre-registered §27 criterion: |mean_Δ(NGD − vargp_intl_fixAmp)| > 0.02 = GAP.
Reports §27 primary + §28 secondary diagnostics.
"""
from __future__ import annotations

import json
import math
import statistics as stats
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
NGD_JSONL = EXP_DIR / 'results.jsonl'
VARGP_JSONL = ROOT / 'experiments/2026-04-06_es_sweeps_64x64/sweep_64x64_elbo_es_results.jsonl'
VARGP_CONFIG_NAME = '64_elbo_intl_fixAmp'


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def keyed(recs, key_fn):
    return {key_fn(r): r for r in recs}


def main():
    ngd = load_jsonl(NGD_JSONL)
    vargp_all = load_jsonl(VARGP_JSONL)
    vargp = [r for r in vargp_all if r.get('config_name') == VARGP_CONFIG_NAME]

    # Filter by status; drop crashes.
    ngd_ok = [r for r in ngd if r.get('test_r') is not None]
    vargp_ok = [r for r in vargp if r.get('test_r') is not None]

    print(f"NGD runs loaded: {len(ngd)}  successful: {len(ngd_ok)}  "
          f"crashes: {len(ngd) - len(ngd_ok)}")
    print(f"Vargp (intl_fixAmp, elbo_es_p15) runs loaded: {len(vargp_ok)}/{len(vargp)}")
    print()

    ngd_by = keyed(ngd_ok, lambda r: (r['cell'], r['seed']))
    vargp_by = keyed(vargp_ok, lambda r: (r['cell'], r['seed']))
    pairs = sorted(set(ngd_by) & set(vargp_by))
    print(f"Paired (cell, seed) observations: {len(pairs)}")
    cells = sorted({c for (c, _) in pairs})
    seeds = sorted({s for (_, s) in pairs})
    print(f"  unique cells: {len(cells)}   unique seeds: {sorted(seeds)}")
    print()

    # ---------------------------------------------------------------
    # Headline (§27 primary): paired Δ test_r
    # ---------------------------------------------------------------
    d_test_r = []
    d_expl_var = []
    for (c, s) in pairs:
        n = ngd_by[(c, s)]
        v = vargp_by[(c, s)]
        d_test_r.append(n['test_r'] - v['test_r'])
        if n.get('explained_var') is not None and v.get('explained_var') is not None:
            d_expl_var.append(n['explained_var'] - v['explained_var'])

    arr = np.array(d_test_r)
    mean_d = arr.mean()
    std_d = arr.std(ddof=1)
    print("=" * 80)
    print("HEADLINE (pre-registered §27 decision)")
    print("=" * 80)
    decision = "NO GAP (|Δ| ≤ 0.02)" if abs(mean_d) <= 0.02 else "GAP EXISTS (|Δ| > 0.02)"
    sign = "favors NGD" if mean_d > 0 else ("favors vargp" if mean_d < 0 else "tie")
    print(f"  mean Δ(NGD_final − vargp_intl_fixAmp) = {mean_d:+.4f}  (n={len(arr)})")
    print(f"  std                                   = {std_d:.4f}")
    print(f"  SEM                                   = {std_d / math.sqrt(len(arr)):.4f}")
    print(f"  range                                 = [{arr.min():+.4f}, {arr.max():+.4f}]")
    print(f"  sign counts                           = "
          f"pos/within/neg = "
          f"{int((arr > 0.02).sum())}/{int(((arr >= -0.02) & (arr <= 0.02)).sum())}/{int((arr < -0.02).sum())}")
    print()
    print(f"  DECISION: {decision} — {sign}")
    print()

    # ---------------------------------------------------------------
    # §27 additional scalars
    # ---------------------------------------------------------------
    print("=" * 80)
    print("AGGREGATE SCALARS (both modes; from paired subset)")
    print("=" * 80)
    def scalars(recs_dict, label):
        tr = [recs_dict[k]['test_r'] for k in pairs]
        ev = [recs_dict[k].get('explained_var') for k in pairs
              if recs_dict[k].get('explained_var') is not None]
        # cells > 0.8 exp_var on 3-seed max per cell (vargp paper convention)
        per_cell_max_ev = {}
        for k in pairs:
            c = k[0]
            v = recs_dict[k].get('explained_var')
            if v is None: continue
            if c not in per_cell_max_ev or v > per_cell_max_ev[c]:
                per_cell_max_ev[c] = v
        n_cells_over_08 = sum(1 for v in per_cell_max_ev.values() if v > 0.8)
        return dict(
            mean_test_r=stats.mean(tr),
            std_test_r=stats.stdev(tr),
            mean_exp_var=(stats.mean(ev) if ev else None),
            cells_over_08_ev=n_cells_over_08,
            n_cells_tracked=len(per_cell_max_ev),
        )

    s_n = scalars(ngd_by, 'NGD')
    s_v = scalars(vargp_by, 'vargp')
    print(f"  {'mode':<30} {'mean_test_r':>12} {'mean_exp_var':>13} "
          f"{'cells>0.8_ev':>13} (/{s_n['n_cells_tracked']})")
    print(f"  {'NGD (this sweep)':<30} {s_n['mean_test_r']:>12.4f} "
          f"{s_n['mean_exp_var']:>13.4f} {s_n['cells_over_08_ev']:>13}")
    print(f"  {'vargp intl_fixAmp (ELBO p=15)':<30} {s_v['mean_test_r']:>12.4f} "
          f"{s_v['mean_exp_var']:>13.4f} {s_v['cells_over_08_ev']:>13}")
    print()

    if d_expl_var:
        arr_ev = np.array(d_expl_var)
        print(f"  mean Δ explained_var = {arr_ev.mean():+.4f}  "
              f"(n={len(arr_ev)}, std={arr_ev.std(ddof=1):.4f})")
        print()

    # ---------------------------------------------------------------
    # Timing
    # ---------------------------------------------------------------
    print("=" * 80)
    print("TIMING")
    print("=" * 80)
    def timing(recs):
        walls = [r.get('train_time') or r.get('wall_time_s') for r in recs]
        walls = [w for w in walls if w is not None]
        iters = [r.get('n_iterations_run') for r in recs if r.get('n_iterations_run')]
        return dict(
            wall_mean=stats.mean(walls), wall_std=stats.stdev(walls),
            iter_mean=stats.mean(iters) if iters else None,
            iter_med=stats.median(iters) if iters else None,
            iter_max=max(iters) if iters else None,
            iter_min=min(iters) if iters else None,
        )
    t_n = timing([ngd_by[k] for k in pairs])
    t_v = timing([vargp_by[k] for k in pairs])
    print(f"  {'mode':<30} {'wall_s mean±std':>22} {'iters mean':>12} {'iters [min, med, max]':>25}")
    print(f"  {'NGD':<30} {t_n['wall_mean']:>10.2f} ± {t_n['wall_std']:<8.2f}  "
          f"{t_n['iter_mean']:>12.1f}  "
          f"[{t_n['iter_min']}, {t_n['iter_med']}, {t_n['iter_max']}]")
    print(f"  {'vargp intl_fixAmp':<30} {t_v['wall_mean']:>10.2f} ± {t_v['wall_std']:<8.2f}  "
          f"{t_v['iter_mean']:>12.1f}  "
          f"[{t_v['iter_min']}, {t_v['iter_med']}, {t_v['iter_max']}]")
    print()

    # ---------------------------------------------------------------
    # ES trigger rate + stop-iter distribution
    # ---------------------------------------------------------------
    print("=" * 80)
    print("EARLY STOPPING (NGD only)")
    print("=" * 80)
    stopped = sum(1 for k in pairs if ngd_by[k].get('stopped_early'))
    print(f"  ES trigger rate: {stopped}/{len(pairs)} ({100*stopped/len(pairs):.0f}%)")
    stop_iters = [ngd_by[k]['n_iterations_run'] for k in pairs
                  if ngd_by[k].get('stopped_early')]
    if stop_iters:
        a = np.array(stop_iters)
        print(f"  stop iters (when ES fired): min={a.min()} p25={int(np.percentile(a,25))} "
              f"median={int(np.median(a))} p75={int(np.percentile(a,75))} max={a.max()}")
    hit_cap = sum(1 for k in pairs if not ngd_by[k].get('stopped_early'))
    print(f"  hit cap (1500 iters): {hit_cap}/{len(pairs)}")
    print()

    # ---------------------------------------------------------------
    # Final-A distribution
    # ---------------------------------------------------------------
    print("=" * 80)
    print("FINAL A DISTRIBUTION")
    print("=" * 80)
    def a_dist(recs):
        As = [r.get('final_A') for r in recs if r.get('final_A') is not None]
        arr = np.array(As)
        return (arr.min(), np.percentile(arr, 25), np.median(arr),
                np.percentile(arr, 75), arr.max(),
                int((arr <= 0.012).sum()), int((arr > 0.3).sum()))
    print(f"  {'mode':<30} {'min':>9} {'p25':>9} {'med':>9} {'p75':>9} {'max':>9} "
          f"{'≤0.012':>7} {'>0.3':>6}")
    n_min, n_p25, n_med, n_p75, n_max, n_low, n_hi = a_dist([ngd_by[k] for k in pairs])
    v_min, v_p25, v_med, v_p75, v_max, v_low, v_hi = a_dist([vargp_by[k] for k in pairs])
    print(f"  {'NGD':<30} {n_min:>9.4g} {n_p25:>9.4g} {n_med:>9.4g} "
          f"{n_p75:>9.4g} {n_max:>9.4g} {n_low:>7} {n_hi:>6}")
    print(f"  {'vargp intl_fixAmp':<30} {v_min:>9.4g} {v_p25:>9.4g} {v_med:>9.4g} "
          f"{v_p75:>9.4g} {v_max:>9.4g} {v_low:>7} {v_hi:>6}")
    print()

    # ---------------------------------------------------------------
    # Per-cell table (3-seed mean, signed Δ, largest winners/losers)
    # ---------------------------------------------------------------
    print("=" * 80)
    print("PER-CELL Δ (3-seed mean) — top winners & losers for NGD")
    print("=" * 80)
    per_cell = {}
    for c in cells:
        n_trs = [ngd_by[(c, s)]['test_r'] for s in seeds if (c, s) in ngd_by]
        v_trs = [vargp_by[(c, s)]['test_r'] for s in seeds if (c, s) in vargp_by]
        if not n_trs or not v_trs: continue
        per_cell[c] = (stats.mean(n_trs) - stats.mean(v_trs),
                       stats.mean(n_trs), stats.mean(v_trs))
    sorted_cells = sorted(per_cell.items(), key=lambda kv: kv[1][0])
    print(f"  {'cell':>4} {'d_3s':>8} {'NGD_μ':>8} {'vargp_μ':>8}")
    print("  -- 5 biggest NGD losses:")
    for c, (d, n, v) in sorted_cells[:5]:
        print(f"  {c:>4} {d:>+8.4f} {n:>8.4f} {v:>8.4f}")
    print("  -- 5 biggest NGD wins:")
    for c, (d, n, v) in sorted_cells[-5:]:
        print(f"  {c:>4} {d:>+8.4f} {n:>8.4f} {v:>8.4f}")
    in_band = sum(1 for _, (d, _, _) in per_cell.items() if abs(d) <= 0.02)
    favors_ngd = sum(1 for _, (d, _, _) in per_cell.items() if d > 0.02)
    favors_vargp = sum(1 for _, (d, _, _) in per_cell.items() if d < -0.02)
    print(f"  per-cell sign counts (3-seed mean): "
          f"favors_NGD={favors_ngd}, within_band={in_band}, favors_vargp={favors_vargp}")


if __name__ == '__main__':
    main()
