# ES Sweeps: 64x64 — Early Stopping Investigation Results

**Date**: 2026-04-05 / 2026-04-06 / 2026-04-07
**Mode**: vargp_direct
**Image resolution**: 64x64 (center crop, no renorm)
**Dataset**: `datasets/PNAS_64x64_center_crop_no_renorm.npz`

**Branch**: `pietro/investigate-paper-gap`

## TL;DR

After comparing four early stopping methods on 64x64 PNAS data:

- **ELBO-based ES is the chosen default** (April 2026). It matches the
  no-ES baseline within 0.0007 test_r while saving ~53% compute.
- val_ll-based ES (and val_r/val_rho variants) underperform the no-ES
  baseline by ~0.02 test_r because of A-transient noise on the held-out
  validation set.
- All ELBO ES runs use **the full 3160 training images** (n_val_split=0
  — no validation carving needed) which contributes part of the
  improvement vs val_ll ES.
- Sweep script: `run_sweep_elbo_es_64x64.py` (in this folder).
- Bug fix during the investigation: ES best-tracking was conflated with
  the patience threshold in the original code; fixed to track the true
  argmax of the ES metric. See SESSION_LOG.md and DECISION_LOG.md Q33.

---

## Reference values for ES method comparison

All values computed over 3 seeds x 41 cells, M=250, vargp_direct,
ground-truth RF init, n_estep=50, n_mstep=20, n_iterations=80 max.
`cells>0.8` takes the max explained_var across the 3 seeds for each cell,
then counts cells above 0.8.

| Config | Interleave | Amp | A_init | ES | Patience | n_train | mean test_r | mean exp_var | cells>0.8 ev | mean iters | Notes |
|--------|-----------|-----|--------|-----|----------|---------|-------------|--------------|--------------|-----------|-------|
| A01_free_no_intl | no | free | 0.01 | off | -- | 2910 | 0.8272 | 0.8880 | 35/41 | 80 | baseline |
| intl_fixAmp | yes | 1.0 | 1e-4 | off | -- | 2910 | **0.8382** | **0.8994** | **37/41** | 80 | best baseline |
| intl_freeAmp | yes | free | 1e-4 | off | -- | 2910 | 0.8354 | 0.8965 | 37/41 | 80 | baseline |
| A01_free_no_intl | no | free | 0.01 | val_ll | 15 | 2910 | 0.8100 | 0.8694 | 35/41 | 37.3 | val_ll p=15 |
| intl_fixAmp | yes | 1.0 | 1e-4 | val_ll | 15 | 2910 | 0.8143 | 0.8735 | 36/41 | 33.1 | val_ll p=15 |
| intl_freeAmp | yes | free | 1e-4 | val_ll | 15 | 2910 | 0.8128 | 0.8719 | 35/41 | 30.6 | val_ll p=15 |
| A01_free_no_intl | no | free | 0.01 | val_ll | 30 | 2910 | 0.8096 | 0.8690 | 35/41 | 52.9 | val_ll p=30 |
| intl_fixAmp | yes | 1.0 | 1e-4 | val_ll | 30 | 2910 | 0.8185 | 0.8780 | 36/41 | 50.7 | val_ll p=30 |
| intl_freeAmp | yes | free | 1e-4 | val_ll | 30 | 2910 | 0.8154 | 0.8748 | 35/41 | 49.2 | val_ll p=30 |
| no_intl_fixAmp | no | 1.0 | 1e-4 | val_ll | 30 | 2910 | 0.7542 | 0.8055 | 31/41 | 63.8 | A_init too small for LBFGS |
| no_intl_freeAmp | no | free | 0.01 | elbo | 15 | **3160** | 0.8274 | 0.8883 | 35/41 | 58.6 | ELBO ES |
| no_intl_fixAmp | no | 1.0 | 0.01 | elbo | 15 | **3160** | 0.8307 | 0.8913 | 34/41 | 74.7 | ELBO ES |
| intl_freeAmp | yes | free | 1e-4 | elbo | 15 | **3160** | 0.8357 | 0.8968 | 37/41 | 35.2 | ELBO ES |
| **intl_fixAmp** | **yes** | **1.0** | **1e-4** | **elbo** | **15** | **3160** | **0.8375** | **0.8987** | **37/41** | **37.5** | **best ES (chosen default)** |

### Key findings

1. **ELBO ES (intl_fixAmp) closes the gap to the no-ES baseline**:
   ELBO ES = 0.8375 vs baseline 0.8382 -> only -0.0007 test_r. The val_ll
   ES gap was -0.020 to -0.024.
2. **ELBO ES preserves cells>0.8**: 37/41 same as the no-ES baseline.
   val_ll ES dropped to 35-36/41.
3. **Compute savings**: ~53% (37.5 iters vs 80).
4. **The 4th grid cell (no_intl + fix_Amp)** is now testable. With
   A_init=0.01 (instead of the failed 1e-4 used for val_ll p=30) ELBO ES
   gives test_r=0.8307 — clearly better than 0.7542.
5. **Why ELBO ES works where val_ll ES failed**: ELBO is averaged over
   all 3160 training points so it absorbs the A-transient noise from the
   interleaved damped Newton F-step. val_ll on 250 held-out images is
   too noisy in early iterations. See
   `investigations/optimization/possible_optimizations.md` Investigation 2
   for the full reasoning + diagnostic data that led to this conclusion.

### File → Sweep mapping

| JSONL file | Sweep | n_runs | n_train | ES |
|-----------|-------|--------|---------|-----|
| `sweep_64x64_results_ntrain3160.jsonl` | Baseline (no ES) | 369 | 2910 | off |
| `sweep_64x64_es_results.jsonl` | val_ll ES p=15 (FIXED) | 369 | 2910 | val_ll |
| `sweep_64x64_es_results_STALE_n2660.jsonl` | val_ll ES p=15 (n=2660 bug) | 369 | 2660 | val_ll |
| `sweep_64x64_es_p30_results.jsonl` | val_ll ES p=30 (4 configs) | 492 | 2910 | val_ll |
| `sweep_64x64_elbo_es_results.jsonl` | **ELBO ES p=15 (chosen default)** | **492** | **3160** | **elbo** |
| `sweep_64x64_elbo_es_results_BUGGY_best_tracking.jsonl` | ELBO ES with the conflated best-tracking bug (kept as evidence for the fix; do not use for analysis) | 32 | 3160 | elbo |
| `diagnostic_no_es_results.jsonl` | Diagnostic interleaved | 7 | 2910 | off |
| `diagnostic_no_es_no_interleaved_results.jsonl` | Diagnostic non-interleaved | 7 | 2910 | off |
| `sweep_64x64_results.jsonl` | Old stale (n=2910 bug), superseded | 246 | 2910 | off |

### Sweep scripts in this folder

| Script | Sweep produced |
|--------|----------------|
| `run_sweep_64x64.py` | `sweep_64x64_results.jsonl` (stale baseline) |
| `run_sweep_64x64_es.py` | `sweep_64x64_es_results.jsonl` (val_ll p=15) |
| `run_sweep_64x64_es_p30.py` | `sweep_64x64_es_p30_results.jsonl` (val_ll p=30, 4 configs) |
| `run_sweep_elbo_es_64x64.py` | `sweep_64x64_elbo_es_results.jsonl` (**ELBO ES**) |

---

## Curve availability per file

| File | has per-iter curves? | n_iters stored |
|------|---------------------|----------------|
| `sweep_64x64_results.jsonl` (old stale) | NO | — |
| `sweep_64x64_results_ntrain3160.jsonl` (baseline no-ES) | **NO** | — |
| `sweep_64x64_es_results.jsonl` (val_ll p=15 fixed) | yes | 33-37 (stopped) |
| `sweep_64x64_es_results_STALE_n2660.jsonl` | yes | ~36 (stopped) |
| `sweep_64x64_es_p30_results.jsonl` (val_ll p=30) | yes | 30-54 (stopped) |
| `sweep_64x64_elbo_es_results.jsonl` (**ELBO ES**) | yes | 35-75 (stopped) |
| `sweep_64x64_elbo_es_results_BUGGY_best_tracking.jsonl` | yes | 32 (stopped, partial run) |
| `diagnostic_no_es_results.jsonl` | yes | 79 (full) |
| `diagnostic_no_es_no_interleaved_results.jsonl` | yes | 79 (full) |

The baseline sweeps predate the curve logging feature. The ES sweeps embed
`train_loss` (= -ELBO), `train_log_lik`, `train_kl`, `train_r`, and (when
n_val_split > 0) `val_log_lik`, `val_r`, `val_rho` curves per record. The
ELBO ES sweep was run with `n_val_split=0` so all val_* curves are None
(but train_* curves are populated). The diagnostics have full 80-iteration
curves but only cover 7 cells.

## What this comparison is for

This folder is the empirical reference for the **chosen early stopping
mechanism** decision (ELBO ES). Future ES experiments should compare
against `sweep_64x64_elbo_es_results.jsonl` (the new default) and the
no-ES baseline `sweep_64x64_results_ntrain3160.jsonl`.

For background on why val_ll/val_r/val_rho ES was rejected, see:
- `investigations/optimization/possible_optimizations.md` Investigation 2
  (the diagnostic data showing val_ll noise from the A transient)
- `.claude/DECISION_LOG.md` Q32 (the decision rationale)
- The `diagnostic_no_es_*.jsonl` files in this folder (the original
  diagnostic data showing val_ll instability for early-stopper cells)

---

## Purpose

This folder archives the 64x64 sweeps generated during the paper gap
investigation's early-stopping (ES) phase. The sweeps explore:

- Baseline performance with no early stopping (80 iterations)
- ES with patience=15 (original, with and without the data loading bug)
- ES with patience=30 (broadened investigation, 4 configs)
- Diagnostic runs on early-stopper cells to understand val metric noise

All runs use:
- M=250 inducing points
- seeds 1, 2, 3
- vargp_direct mode
- 41 cells (PNAS dataset)
- beta init = 0.1, rho init = 0.1, lambda0 init = -1.0
- ground-truth RF centers from `datasets/rf_centers_ground_truth.npz`
- ip_selection = 'random'

---

## Files

### Sweep scripts

| File | Purpose |
|------|---------|
| `run_sweep_64x64.py` | Original 64x64 baseline sweep (no ES). Generated the ntrain3160 results. |
| `run_sweep_64x64_es.py` | ES patience=15 sweep, 3 configs. Originally had val_from_train bug; re-run after fix. |
| `run_sweep_64x64_es_p30.py` | ES patience=30 sweep, 4 configs (includes missing no_intl_fixAmp combo). |

### Result JSONL files

Each record contains: config, params, final kernel values, test_r,
explained_var, adjusted_r2, reliability, full training curves (train_loss,
train_log_lik, train_kl, val_log_lik, train_r, val_r, kernel params,
iter_time), timing, seed, cell.

---

## Sweep 1: Baseline no-ES (FIXED, n_train=3160)

**File**: `sweep_64x64_results_ntrain3160.jsonl`
**Runs**: 369 (3 configs x 41 cells x 3 seeds)
**n_train**: 3160 (all available training images)
**Early stopping**: OFF — runs full 80 iterations

Reference baseline for ES comparisons. Fixes the earlier `n_train=2910` bug
by using all 3160 training images.

### Configurations

| Config | Amp | Interleave | A_init | n_estep | n_mstep | n_iters |
|--------|-----|-----------|--------|---------|---------|---------|
| `64_A01_free_no_intl_n3160` | free | no | 0.01 | 50 | 20 | 80 |
| `64_intl_fixAmp_n3160` | fixed=1.0 | yes (damped Newton) | 1e-4 | 50 | 20 | 80 |
| `64_intl_freeAmp_n3160` | free | yes (damped Newton) | 1e-4 | 50 | 20 | 80 |

### Results (mean over 3 seeds x 41 cells)

| Config | mean test_r | mean explained_var |
|--------|-------------|-------------------|
| 64_A01_free_no_intl_n3160 | 0.8272 | 0.8880 |
| 64_intl_fixAmp_n3160 | **0.8382** | **0.8994** |
| 64_intl_freeAmp_n3160 | 0.8354 | 0.8965 |

**Finding**: Interleaved F-step with fix_Amp=True is best, ~0.01 ahead of
the non-interleaved A=0.01 config.

---

## Sweep 2: ES patience=15 (FIXED data loading)

**File**: `sweep_64x64_es_results.jsonl`
**Runs**: 369 (3 configs x 41 cells x 3 seeds)
**n_train**: 2910 (3160 minus 250 carved validation)
**Early stopping**: ON, patience=15, min_delta_rel=0.001, min_iterations=10,
metric=val_log_lik, restore_best=True

Same 3 configs as Sweep 1 but with ES enabled. This is the corrected version
after fixing the `val_from_train` data loading bug.

### Results

| Config | mean test_r | mean_iters | stopped_early |
|--------|-------------|-----------|---------------|
| 64_A01_free_no_intl_n3160_es | 0.8100 | 37.3 | 112/123 |
| 64_intl_fixAmp_n3160_es | 0.8143 | 33.1 | 120/123 |
| 64_intl_freeAmp_n3160_es | 0.8128 | 30.6 | 121/123 |

**Delta vs Sweep 1 baseline** (paired, per cell x seed):
- 64_A01_free_no_intl: mean delta = -0.017
- 64_intl_fixAmp: mean delta = -0.024
- 64_intl_freeAmp: mean delta = -0.023

**Finding**: ES costs ~0.02 test_r vs full 80-iteration training but saves
~55% compute (avg 33 iters vs 80). The gap is fundamental — driven by
premature stopping on val_ll during the interleaved F-step's A transient.

---

## Sweep 3: ES patience=15 STALE (val_from_train bug, n_train=2660)

**File**: `sweep_64x64_es_results_STALE_n2660.jsonl`
**Runs**: 369 (3 configs x 41 cells x 3 seeds)
**n_train**: 2660 (BUG: should have been 2910)
**Early stopping**: ON, patience=15

**STATUS**: STALE. Preserved for reference showing the impact of the
`val_from_train` data loading bug. Commit `1be8b8a` replaced
`X = torch.cat([X_train, X_val])` (3160) with `X = X_train` (2910).
With `val_from_train=True` carving another 250, only 2660 images remained
for training — 16% less than intended.

### Results

| Config | mean test_r |
|--------|-------------|
| 64_A01_free_no_intl_n3160_es | 0.8073 |
| 64_intl_fixAmp_n3160_es | 0.8210 |
| 64_intl_freeAmp_n3160_es | 0.8202 |

**Finding**: Fixing the bug (2660 -> 2910) gave only marginal improvement
(+0.003 for A01_free, -0.007 for interleaved). The interleaved configs
actually performed slightly worse with more data — the bug's effect was
smaller than expected, suggesting the 250-image difference is not the
dominant factor in the ES gap.

**Do not use for new analysis**. Use `sweep_64x64_es_results.jsonl` instead.

---

## Sweep 4: ES patience=30, 4 configs

**File**: `sweep_64x64_es_p30_results.jsonl`
**Runs**: 492 (4 configs x 41 cells x 3 seeds)
**n_train**: 2910
**Early stopping**: ON, patience=30, min_delta_rel=0.001, min_iterations=10

Doubled patience to test whether ES was stopping prematurely. Added the
4th missing combination (no interleave + fix_Amp).

### Configurations

| Config | Amp | Interleave | A_init | Purpose |
|--------|-----|-----------|--------|---------|
| `64_A01_free_no_intl_es_p30` | free | no | 0.01 | Same as Sweep 2 with p=30 |
| `64_intl_fixAmp_es_p30` | fixed=1.0 | yes | 1e-4 | Same as Sweep 2 with p=30 |
| `64_intl_freeAmp_es_p30` | free | yes | 1e-4 | Same as Sweep 2 with p=30 |
| `64_no_intl_fixAmp_es_p30` | fixed=1.0 | no | 1e-4 | NEW: 4th grid cell |

### Results

| Config | mean test_r | mean_iters | stopped_early |
|--------|-------------|-----------|---------------|
| 64_A01_free_no_intl_es_p30 | 0.8096 | 52.9 | 94/123 |
| 64_intl_fixAmp_es_p30 | **0.8185** | 50.7 | 108/123 |
| 64_intl_freeAmp_es_p30 | 0.8154 | 49.2 | 111/123 |
| 64_no_intl_fixAmp_es_p30 | 0.7542 | 63.8 | 58/123 |

**Delta vs Sweep 2 (p=15)** (paired):
- 64_A01_free_no_intl: -0.0004 (essentially no change)
- 64_intl_fixAmp: +0.0042
- 64_intl_freeAmp: +0.0027

**Findings**:
1. Doubling patience helped interleaved configs marginally (+0.003-0.004)
   but not the non-interleaved config. Interleaving is the source of
   premature stopping.
2. The 4th config (`no_intl_fixAmp`) performs **much worse** (0.754 vs
   0.819). Without interleaving, A_init=1e-4 is too small — the standard
   LBFGS F-step can't ramp A fast enough to recover. Only 47% of runs
   triggered ES at all; many ran to 80 iters without finding a useful
   configuration.
3. The 0.02 gap to Sweep 1 baseline persists across patience settings —
   this is a fundamental cost of ES + 8% validation holdout, not a
   patience tuning issue.

---

## Diagnostic runs: val_ll recovery check

Small-scale diagnostic runs on 7 cells to understand why val_ll is noisy.
5 "early stopper" cells (where ES patience=15 stopped at iteration 1) +
2 healthy control cells.

### Diagnostic 1: Interleaved F-step, no ES

**File**: `diagnostic_no_es_results.jsonl`
**Runs**: 7 (matching the ES sweep's interleaved config)
**Config**: intl_fixAmp, A_init=1e-4, interleave_fstep=True, 80 iters, no ES

**Cells tested**: 9, 24, 2, 4, 8 (early stoppers in Sweep 2), 10, 30 (healthy)

**Finding**: For some cells (9, 4, 8), val_ll dips during the A transient
then **recovers** above its iteration-1 value by iteration 40-50. ES was
stopping prematurely. For other cells (24, 2), val_ll drops and does NOT
recover, but test_r is still excellent at iteration 80 — val_ll is a poor
proxy for test_r on these cells.

| Cell | Seed | val_ll/sample iter1 | val_ll/sample best | iter best | test_r |
|------|------|--------------------|--------------------|-----------|--------|
| 9 | 1 | +0.194 | +0.270 | 41 | 0.8054 |
| 24 | 2 | +0.121 | +0.121 | 1 | 0.8878 |
| 2 | 1 | -0.158 | -0.141 | 3 | 0.8955 |
| 4 | 3 | -0.476 | -0.446 | 44 | 0.7734 |
| 8 | 1 | -0.896 | -0.877 | 30 | 0.7462 |
| 10 | 3 | -0.803 | -0.792 | 79 | 0.8966 |
| 30 | 2 | -0.576 | -0.555 | 21 | 0.8138 |

### Diagnostic 2: Non-interleaved F-step, no ES

**File**: `diagnostic_no_es_no_interleaved_results.jsonl`
**Runs**: 7 (same cells as Diagnostic 1)
**Config**: A01_free_no_intl, A_init=0.01, interleave_fstep=False,
fix_Amp=False, 80 iters, no ES

**Finding**: Without the interleaved F-step, A evolves smoothly (0.003 to
0.015 over 80 iters vs jumping from 1e-4 to 0.03 in iteration 1 with
interleaving). Val metrics are much smoother — **no A transient, no val_ll
crash**. This confirms that the val metric instability is caused by the
interleaved F-step's rapid A changes, not by intrinsic Poisson noise on
the 250-image validation set.

Mean test_r across the 7 cells: **0.851** (vs 0.831 with interleaved) —
non-interleaved actually performed better on this small sample, though
3 seeds x 7 cells is too small to draw strong conclusions.

---

## Stale/superseded sweep: 64x64 with n_train=2910 bug

**File**: `sweep_64x64_results.jsonl`
**Runs**: 246 (2 configs x 41 cells x 3 seeds)
**n_train**: 2910 (should have been 3160)
**Early stopping**: OFF

**STATUS**: STALE. Superseded by `sweep_64x64_results_ntrain3160.jsonl`.
Preserved for reference showing the original n_train bug in the 64x64
sweep script. The 64x64 baseline sweep was run with hardcoded
`n_train=2910` (only the `images_train` part of the .npz), while the
108x108 sweep correctly used `n_train=3160` (train+val combined).

### Results

| Config | mean test_r | mean explained_var |
|--------|-------------|-------------------|
| 64_free_amp_no_intl | 0.8111 | 0.8703 |
| 64_free_amp_no_intl_A01 | 0.8275 | 0.8882 |

**Do not use for new analysis**. Use `sweep_64x64_results_ntrain3160.jsonl`.

---

## Training curve files

For plotting and post-hoc analysis. Each contains per-iteration curves
(train_loss, val_log_lik, kernel params, etc.) extracted from the sweep
result files.

| File | Source | Config |
|------|--------|--------|
| `curves_es_64_A01_free_no_intl_n3160_es.jsonl` | Sweep 3 STALE (n=2660) | A01_free_no_intl |
| `curves_es_64_intl_fixAmp_n3160_es.jsonl` | Sweep 3 STALE | intl_fixAmp |
| `curves_es_64_intl_freeAmp_n3160_es.jsonl` | Sweep 3 STALE | intl_freeAmp |
| `curves_fixed_es_64_A01_free_no_intl_n3160_es.jsonl` | Sweep 2 (n=2910 FIXED) | A01_free_no_intl |
| `curves_fixed_es_64_intl_fixAmp_n3160_es.jsonl` | Sweep 2 FIXED | intl_fixAmp |
| `curves_fixed_es_64_intl_freeAmp_n3160_es.jsonl` | Sweep 2 FIXED | intl_freeAmp |

Curves for Sweep 4 (p=30) and the diagnostic runs are embedded directly in
the result JSONL files (in the `curves` field per record).

---

## Key findings summary

1. **Baseline is reproducible**: 64_intl_fixAmp_n3160 mean test_r = 0.8382,
   matching the 108x108 canonical result (0.8387 on paper fits).

2. **ES costs ~0.02 test_r** across all configurations and patience values.
   Not a tuning issue — fundamental to the approach.

3. **Interleaved F-step causes val metric instability**: confirmed by
   diagnostic runs. Val_ll drops during A transients (iterations 1-20),
   fooling ES into premature stopping for ~27% of runs.

4. **Fixing Amp helps** (counter to the intuition that more free parameters
   should always help). Confirmed by the full sweep data: Amp is
   near-unidentifiable because mu(x) is independent of Amp in the
   large-Amp limit.

5. **Doubling patience (15 -> 30) only helps interleaved configs** by
   ~0.004, because only they experience the transient. Non-interleaved
   configs are unaffected.

6. **The 4th grid cell (no_intl + fixAmp with A_init=1e-4) fails**: without
   interleaving, LBFGS F-step can't bootstrap A from 1e-4 quickly enough.

---

## Related files outside this folder

- Investigation context: `investigations/paper_gap/INVESTIGATION_LOG.md`
  (Findings 1-21, compounding factors table)
- Metric discussion: `investigations/paper_gap/METRICS_COMPARISON.md`
- Paper gap resolution: `.claude/DECISION_LOG.md` Q27-Q31
- Follow-up optimization ideas: `investigations/optimization/possible_optimizations.md`
