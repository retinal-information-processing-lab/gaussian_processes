# NGD low-n_train investigation

**Status**: OPEN (2026-04-29) — first hypothesis (config tuning) falsified;
the gap is real and needs more thought before a fix is found.
**Question**: at low n_train (M = n_train = 50, 150, 300), does NGD with
properly-scaled ES parameters close the gap to vargp_direct that the
default Phase 3C config produces?

## TL;DR (added after results)

**The hypothesis was wrong.** Scaling NGD's ES to vargp-style values
(patience=15, min_delta_rel=1e-3, iter_cap=200) made the gap *wider*,
not narrower:

| M=n_train | Δ default vs vargp | Δ scaled-ES vs vargp |
|-----------|--------------------|--------------------|
| 50  | −0.118 | −0.157 |
| 150 | −0.083 | −0.098 |
| 300 | −0.064 | −0.133 |

Mechanism: patience=15 with min_delta_rel=1e-3 essentially never fires
at low n_train (loss keeps improving by >0.1% every iter), so the iter
cap of 200 dominates. cap=200 fixes the ~7 disaster cells per M (where
overfitting was the issue) but undertrains the other ~30 cells (where
the kernel hyperparams need 500+ iters). Net effect: net regression.

**Real conclusion**: the low-n_train gap is NOT a simple config issue.
It's an algorithmic limitation of Adam at low n_train. Different cells
need different amounts of training and there's no single iter cap that
works. A real fix would require per-cell adaptive ES (e.g. via held-out
validation) or a different optimizer entirely.

The 3-cell mini-sweep (cells 0, 35, 16) that motivated this run was
biased — those were specifically the worst-overfitting cells. They
benefit from cap=100 but most cells don't. The 41-cell test was
necessary to surface this.

See `analyze_scaled_es.py` output and `results_scaled_es.jsonl` for
full data.

## Wall-time picture (also unfavorable for NGD at low n_train)

| M=n_train | vargp wall | NGD-default wall | NGD-default vs vargp |
|-----------|------------|------------------|----------------------|
| 50  | 47s | 113s | NGD 2.4× **slower** |
| 150 | 58s | 125s | NGD 2.1× **slower** |
| 300 | 69s | 140s | NGD 2.0× **slower** |

Reverses the high-n_train pattern (Phase 3C: NGD 4× faster at n_train=3160).
Reason: vargp's LBFGS converges in ~45 outer iters via Newton-like steps;
NGD's first-order Adam needs ~1050 iters because ES doesn't fire early.

So at low n_train, vargp wins on **both accuracy AND speed**. Decisive.

## What's still untried (open avenues for future investigation)

1. **Held-out validation ES** — real validation set to detect overfitting.
   The project documents that small-validation ES is noisy (you wrote
   this in CLAUDE.md), but at low n_train the alternative (train ELBO)
   is also broken. Would need a fair test to see which is less bad.

2. **Mode-switching** — use `vargp_direct` for low n_train, `ngd` once
   n_train > some threshold. Hybrid solution; ugly but practical.

3. **Different optimizer** — try AdamW with high weight decay on A,
   or a 2nd-order optimizer just for hyperparams (separate from the
   Phase 3E LBFGS-on-hyperparams which had its own A→0 problem).

4. **Cell-by-cell diagnosis** — the 7 NGD-only-disaster cells at M=50
   (0, 10, 15, 23, 28, 35, 36) might share a structural property
   (sparse firing rate? bad STA?) that could be detected upfront.

5. **Per-cell adaptive iter cap** — train each cell with held-out 5%
   for ES; let cap range from 100 to 1500 based on per-cell behavior.

The project's authoritative comparison remains: vargp_direct's
`intl_fixAmp` at production n_train is the strongest baseline. NGD
matches that at production n_train (Phase 3C: Δ=+0.007). Below
n_train ~ 500, NGD is genuinely worse and faster fixes haven't
been found.

## Background

The Exp C sweep (`experiments/2026-04-28_ngd_validation/results_C.jsonl`)
showed NGD trails vargp at low n_train:

| M=n_train | vargp test_r | NGD test_r (default) | Δ NGD−vargp |
|-----------|--------------|----------------------|-------------|
| 50  | 0.512 | 0.394 | −0.118 |
| 150 | 0.545 | 0.462 | −0.083 |
| 300 | 0.637 | 0.573 | −0.064 |

Compare to Phase 3C (n_train=3160): Δ = +0.007. The gap is specific to the
low-n_train regime.

## Hypothesis from preliminary investigation

A 3-cell × 4-iter-cap mini-sweep (`iter_cap_sweep.py`,
`iter_cap_sweep.jsonl`) showed:

- Hard cells (0, 35) degrade monotonically with iter count
- Easy cell (16) robust
- Train ELBO improves monotonically (A inflates) → `restore_best` selects
  most-overfit iteration

Mechanism: NGD's default ES (patience=200, min_delta_rel=1e-2,
n_iterations=1500) was tuned for n_train=3160 in Phase 3C. At n_train=50
those parameters let NGD overtrain and inflate A.

Mini-sweep evidence at cap=100 vs 1500:
- cell 0: +0.19 vs −0.10  (recovered ~0.29 of the gap)
- cell 35: +0.01 vs −0.15 (recovered ~0.16 of the gap)
- cell 16: +0.87 vs +0.81 (no harm)

But even at cap=100, NGD on hard cells still trails vargp by ~0.20. So
overfitting is part of the story but not all of it. We need the 41-cell
check to know:

1. Does NGD with vargp-scale ES (patience=15) close the gap on average?
2. Are NGD-only-disaster cells (0, 10, 15, 23, 28, 35, 36 at M=50) fixed?

## What this run (41-cell scaled-ES check) does

Re-run NGD at low n_train with **vargp-scale ES** to see if the gap
closes at the 41-cell level. Compare against existing Exp C vargp results.

### Config (one explicit deviation from Phase 3C)

- mode: `ngd`
- M = n_train ∈ {50, 150, 300}
- seed = 1 (matches Exp C vargp baseline)
- fix_Amp = True (matches Exp C)
- 64x64, ip_selection=random, rf_init=ground_truth
- **ES override** (the thing being tested):
  - `ngd_es_patience = 15` (was 200 — matches vargp)
  - `ngd_es_min_delta_rel = 1e-3` (was 1e-2 — matches vargp)
  - `ngd_es_min_iterations = 10` (was 50 — matches vargp)
- `ngd_n_iterations = 200` (was 1500 — safety cap; ES expected to fire
  well before this)
- All other NGD params (lr=0.1, adam_lr=0.01, restore_best=True): defaults

### Grid

41 cells × 3 (M=n_train) × 1 seed = **123 runs**, NGD only. Output:
`results_scaled_es.jsonl`.

### Reference baselines (already exist, don't re-run)

- `experiments/2026-04-28_ngd_validation/results_C.jsonl` — both NGD
  (default ES) and vargp at the same (cell, M, seed=1) tuples.

### Comparison plan after run

For each M=n_train ∈ {50, 150, 300}, compare the new "scaled-ES NGD" to
both:
1. The Exp C vargp result → did the gap close?
2. The Exp C default-ES NGD result → did NGD itself improve?

If "scaled-ES NGD" mean test_r ≈ Exp C vargp mean: hypothesis confirmed,
the gap is purely a config issue.

If gap closes only partially: the residual is structural (Adam vs
LBFGS efficiency on kernel hyperparams), already noted as mechanism 3.

If gap doesn't close at all: hypothesis falsified, look elsewhere.

## Files

- `iter_cap_sweep.py` / `.jsonl` — preliminary 3-cell × 4-cap mini-sweep
  (motivation evidence)
- `diagnose_overfit.py` / `overfit_diagnostic.jsonl` — full-trajectory
  curves on 4 cells (training ELBO + A inflation)
- `run_scaled_es.py` / `results_scaled_es.jsonl` — this 41-cell check
- `analyze_scaled_es.py` — comparison vs Exp C vargp + default-NGD

## SCRAPBOOK pointer

When this finishes, results will be appended to
`investigations/default_gpy_gap_v2/SCRAPBOOK.md` §62 (or new section)
as the validation of the low-n_train mechanism.
