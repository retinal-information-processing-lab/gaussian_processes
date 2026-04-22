"""Simulate ELBO-based early stopping on the existing 48-run trajectories.

Reads ngd_results.jsonl (full 1000-iter trajectories), applies the same
patience-based ES logic as gpy_training.train_gpy_default, and reports
what the wall-time + test_r would have been if we had enabled ES during
training. No retraining needed — this is a cheap parameter-sensitivity
check.

Logic (identical to gpy_training line 526-582):
  es_value = -train_loss[i]
  is_first = (best_es_value == -inf)
  if is_first or es_value > best_es_value:
      best_es_value = es_value
      best_iter = i + 1
  if is_first:
      patience_reference = es_value
      patience_counter = 0
  else:
      rel_improvement = (es_value - patience_reference) / max(|patience_reference|, 1e-8)
      if rel_improvement > min_delta_rel:
          patience_reference = es_value
          patience_counter = 0
      else:
          patience_counter += 1
  if patience_counter >= patience and (i + 1) >= min_iterations:
      stop at i; restore_best puts test_r at best_iter.

Because restore_best uses the argmax of (−loss), and our loss is monotone
most of the time, the restored iter is close to the stopping iter. We
report both (in the simulation): stop_iter and best_iter.

test_r approximation: the probe records test_r every 25 iters. For a
given best_iter, we pick the probe entry at the nearest probe iter
(off by at most 12 iters). This is close enough for a sanity check.
"""
from __future__ import annotations

import json
import statistics as stats
import sys
from pathlib import Path

import numpy as np

NGD_DIR = Path(__file__).parent
RES_JSONL = NGD_DIR / 'ngd_results.jsonl'


def simulate_es(losses, patience=100, min_delta_rel=1e-3, min_iterations=50):
    """Return (stopped_at_iter, best_iter, stopped_early)."""
    best_es_value = float('-inf')
    best_iter = 0
    patience_reference = float('-inf')
    patience_counter = 0
    for i, loss in enumerate(losses):
        es = -loss
        is_first = (best_es_value == float('-inf'))
        if is_first or es > best_es_value:
            best_es_value = es
            best_iter = i + 1
        if is_first:
            patience_reference = es
            patience_counter = 0
        else:
            rel = (es - patience_reference) / max(abs(patience_reference), 1e-8)
            if rel > min_delta_rel:
                patience_reference = es
                patience_counter = 0
            else:
                patience_counter += 1
        if patience_counter >= patience and (i + 1) >= min_iterations:
            return (i + 1, best_iter, True)
    return (len(losses), best_iter if best_iter else len(losses), False)


def test_r_at_iter(probe_iters, probe_vals, target_iter):
    """Nearest-probe test_r at target_iter (probe recorded every 25 iters)."""
    if not probe_iters:
        return None
    # nearest probe
    dists = [abs(p - target_iter) for p in probe_iters]
    k = int(np.argmin(dists))
    return probe_vals[k]


def main():
    recs = []
    with open(RES_JSONL) as f:
        for line in f:
            recs.append(json.loads(line))

    # Drop crashed
    recs = [r for r in recs if r.get('test_r') is not None]

    # Params to try — pick one to report in detail, report others briefly.
    configs = [
        ('p= 50 δ=1e-3 minit=50', 50, 1e-3, 50),
        ('p=100 δ=1e-3 minit=50', 100, 1e-3, 50),
        ('p= 50 δ=5e-3 minit=50', 50, 5e-3, 50),
        ('p=100 δ=5e-3 minit=50', 100, 5e-3, 50),
        ('p=150 δ=5e-3 minit=50', 150, 5e-3, 50),
        ('p=200 δ=5e-3 minit=50', 200, 5e-3, 50),
        ('p= 50 δ=1e-2 minit=50', 50, 1e-2, 50),
        ('p=100 δ=1e-2 minit=50', 100, 1e-2, 50),
        ('p=200 δ=1e-2 minit=50', 200, 1e-2, 50),
        ('p=100 δ=2e-2 minit=50', 100, 2e-2, 50),
        ('p=200 δ=2e-2 minit=50', 200, 2e-2, 50),
        # Alternative: coarsen ES check window (compare over `patience` iters)
    ]

    print(f"{'config':<30}  {'med stop':>9} {'med best':>9}  "
          f"{'tr_final_μ':>12} {'tr_best_μ':>12}  "
          f"{'tr_sim_μ':>10} {'Δ_sim-final':>13}  "
          f"{'stopped_frac':>14}  {'avg_wall_s':>11}")

    for label, patience, mdr, minit in configs:
        sim_iters = []
        sim_best = []
        sim_test_rs = []
        stopped = []
        wall_estimates = []
        for rec in recs:
            losses = rec['curves']['train_loss']
            probe_iters = rec['curves'].get('test_r_iter', [])
            probe_vals = rec['curves'].get('test_r', [])
            stop_it, best_it, stopped_early = simulate_es(losses, patience, mdr, minit)
            tr_sim = test_r_at_iter(probe_iters, probe_vals, best_it)
            if tr_sim is None:
                continue
            sim_iters.append(stop_it)
            sim_best.append(best_it)
            sim_test_rs.append(tr_sim)
            stopped.append(1 if stopped_early else 0)
            spi = rec.get('s_per_iter', 0.017)
            wall_estimates.append(stop_it * spi)

        tr_final = [r['test_r'] for r in recs]
        tr_best = [r.get('best_test_r_probe') for r in recs if r.get('best_test_r_probe') is not None]

        print(f"{label:<30}  "
              f"{int(np.median(sim_iters)):>9} {int(np.median(sim_best)):>9}  "
              f"{stats.mean(tr_final):>12.4f} {stats.mean(tr_best):>12.4f}  "
              f"{stats.mean(sim_test_rs):>10.4f} "
              f"{stats.mean(sim_test_rs) - stats.mean(tr_final):>+13.4f}  "
              f"{sum(stopped)}/{len(stopped):<12}"
              f"  {stats.mean(wall_estimates):>11.2f}s")


if __name__ == '__main__':
    main()
