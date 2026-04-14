# M Degradation Investigation — Findings

**Branch**: `pietro/investigate-M-degradation`
**Date**: 2026-04-13
**Status**: RESOLVED

> **Canonical results** for this sweep live at
> `experiments/2026-04-13_M_sweep_64x64/` (README + JSONL + scripts).
> This file records the *investigation story* — how the problem was
> found, the wrong turns, and why the final config is what it is.

---

## Problem Statement

Increasing M (inducing points) from 250 to 1500 with `vargp_direct + interleave_fstep=True
+ fix_Amp=True` appeared to degrade mean test_r from **0.838 to 0.829** across 41 cells.
This contradicts sparse GP theory.

Pre-existing evidence used to motivate the investigation:
- M=250: `experiments/2026-04-06_es_sweeps_64x64/sweep_64x64_elbo_es_results.jsonl`
  Config `64_elbo_intl_fixAmp`, 41 cells × 3 seeds (1,2,3), mean test_r=0.838
- M=1500: `checkpoints/64x64_ceiling_M1500_intl_fixAmp/sweep_results.jsonl`
  41 cells × 3 seeds (0,1,2), mean test_r=0.829 (38 cells; cells 0,12,30 failed)

---

## Root Cause: Config Mismatch

**The original M=250 vs M=1500 comparison used different training configurations.
The performance gap was primarily a CONFIG effect, not an M effect.**

The M=250 ES sweep (`run_sweep_elbo_es_64x64.py`) overrides 5 parameters for interleaved
training. The M=1500 ceiling fit (`train_ceiling_models.py`, line 117) uses
`build_config_from_defaults()` without those overrides, getting weaker defaults.

| Parameter | ES sweep (M=250) | Ceiling fit (M=1500) | Impact |
|-----------|-------------------|----------------------|--------|
| A_init | **1e-4** | 0.01 | Critical: 0.01 causes E-step Newton overshoot |
| lambda0_init | **-1.0** | 1.0 | Better starting point for bias |
| n_estep | **50** | 10 | 5x more E-step iterations per outer loop |
| n_mstep | **20** | 10 | 2x more M-step iterations |
| n_iterations | **80** | 50 | 1.6x more outer iterations |

**Verification** (Cell 8, seed=1, M=250):

| Code version | Config | test_r |
|-------------|--------|--------|
| Old (pre-Apr-9) | ES sweep config | 0.8740 |
| New (current) | ES sweep config | 0.8743 |
| New (current) | Default config | 0.8378 |

Old vs new code with same config: identical. The April 9 commit did not introduce a
performance change. Same code, ES vs default config: -0.036. The config alone explains
the gap. See `run_sweep_elbo_es_64x64.py` lines 51-56 for the documented rationale:
A_init=0.01 with interleaved F-step causes the E-step Newton gradient (which scales as
`A * N_train * max(r)`) to overshoot on the first iteration.

---

## Corrected Sweep (41 cells × 8 M × 3 seeds = 984 runs)

**Canonical record**: `experiments/2026-04-13_M_sweep_64x64/` (README + JSONL).

Config: `A_init=1e-4, lambda0_init=-1.0, n_estep=50, n_mstep=20, n_iterations=80,
interleave_fstep=True, fix_Amp=True`, ELBO ES (patience=15), `ip_selection='random'`,
`rf_init='ground_truth'`, `n_val_split=0`, 64×64 data. All other parameters from
`default_params.json`.

**All 984 runs succeeded. Zero failures.**

### Population summary

- **27/41 cells improve with M** (median +0.036)
- **5/41 flat**
- **9/41 degrade** — test_r drops from M=50 to M=1500 by more than 0.005

| M | Grand mean test_r |
|---|-------------------|
| 50 | 0.817 |
| 100 | 0.830 |
| 200 | 0.838 |
| 300 | 0.838 (peak) |
| 500 | 0.834 (dip, dragged down by Cell 39) |
| 750 | 0.838 |
| 1000 | 0.840 |
| 1500 | 0.840 |

**Population sweet spot: M=200-300.** Beyond that, gains are negligible and some cells
start overfitting.

### Degrading cells (9/41)

| Cell | M=50 | M=500 | M=1500 | delta | train_r 50→1500 |
|------|------|-------|--------|-------|----------------|
| **39** | 0.699 | 0.563 | **0.501** | **-0.199** | 0.457 → 0.545 |
| **35** | 0.806 | 0.768 | 0.749 | -0.057 | 0.386 → 0.434 |
| 16 | 0.955 | 0.940 | 0.927 | -0.028 | 0.811 → 0.777 |
| 13 | 0.951 | 0.926 | 0.934 | -0.017 | 0.562 → 0.597 |
| 10 | 0.905 | 0.892 | 0.890 | -0.015 | 0.493 → 0.525 |
| 33 | 0.969 | 0.959 | 0.955 | -0.014 | 0.668 → 0.695 |
| 15 | 0.704 | 0.698 | 0.691 | -0.013 | 0.402 → 0.514 |
| 14 | 0.909 | 0.901 | 0.896 | -0.013 | 0.622 → 0.652 |
| 27 | 0.905 | 0.843 | 0.898 | -0.007 | 0.653 → 0.727 |

**Every degrading cell shows the same signature: train_r rises while test_r falls.**
This is classic overfitting — the model uses the additional capacity from more inducing
points to fit training noise.

### Mechanism (validated on Cell 35)

With more inducing points, the ELBO becomes a tighter bound on the marginal likelihood,
revealing finer structure in the training data — including noise. The ELBO's KL term
regularizes the variational distribution but **not the hyperparameters**. The M-step
(optimizing A, beta, rho) has no regularizer, so more capacity lets A drift upward to
amplify predictions, improving train_r but hurting test_r.

Tests performed on Cell 35 (details in this folder's scripts + loss decomposition):
- **Multiple metrics confirm the pattern** — test_r, adjusted_r2, and held-out val_r all
  drop with M. Not a single-metric artifact.
- **Loss decomposition**: at M=1500, ELL improves by more than at M=50 (better training fit)
  while KL grows (variational distribution pushed further from prior). KL regularization
  is not strong enough to prevent the drift.
- **Warm-init experiment**: initializing M=1500 training from the M=50-optimal
  hyperparameters still ends up with A drifting upward and test_r dropping. Not a local
  optimum issue — the ELBO genuinely prefers the high-A solution at M=1500.
- **Noise ceiling**: degrading cells tend to be near their explained-variance ceiling
  already at M=50, leaving no real signal for larger M to capture. The extra capacity
  fits noise instead.

See `experiments/2026-04-13_M_sweep_64x64/README.md` for the canonical results table
and reproduction commands.

---

## Wrong-Config Data (historical, for reference)

**Results**: `results/phase1_results.jsonl` (243 records)

Collected by the previous session using `build_config_from_defaults()` without the ES
sweep overrides. Cells 0, 12, 30 failed at all seeds and M values (78 failures).
With the correct config, those cells succeed at all seeds and all M values.

Do not use this file for conclusions — kept only as part of the investigation story.

---

## Hypothesis Assessment

| ID | Hypothesis | Status | Evidence |
|----|-----------|--------|----------|
| **H0** | **Config mismatch** | **ROOT CAUSE of original 0.838→0.829 finding** | ES sweep used A_init=1e-4; ceiling fit used A_init=0.01. Verified by reproducing Cell 8 at M=250 with both configs. |
| **H4** | **Hyperparameter overfitting** | **Real secondary effect, 9/41 cells** | train_r↑ while test_r↓, monotonic, seed-consistent. Confirmed by held-out val_r test on Cell 35. |
| H1 | IP selection quality | Not needed | Random IP selection works with correct config |
| H2 | Eigenspace rank saturation | Confirmed, harmless | n_b/M drops to 0.03-0.08 at M=1500 but absolute n_b grows and most cells improve |
| H3 | Optimization insufficiency | Not confirmed | With correct config, training converges at all M |
| H5 | Train-test divergence (classic overfitting) | Subsumed by H4 | Same phenomenon, different framing |

---

## Open follow-ups

- **Data-adaptive A_init** to replace the hardcoded 1e-4. See `ToDo.md` "Data-adaptive
  A initialization for interleaved F-step stability" for the analysis and candidate
  formulas.
- **Hyperparameter regularization** (prior on A) to prevent the drift that causes H4
  overfitting for near-ceiling sparse cells. Not implemented; would need user decision
  on the prior strength.
- **Cell 39** shows catastrophic M-degradation (-0.199). Mechanism likely the same as
  Cell 35 but more extreme. Deferred.

---

## Files (this folder)

| File | Purpose |
|------|---------|
| `FINDINGS.md` | This document — investigation story |
| `phase1_correct_config.py` | Main sweep script (13 cells, correct config) |
| `phase1_remaining_cells.py` | Follow-up sweep script (28 remaining cells) |
| `phase1_m_sweep.py` | Original wrong-config sweep (historical) |
| `phase1_extended_sweep.py` | Original wrong-config extended sweep (historical) |
| `phase2_seed_sweep.py` | Original wrong-config seed sweep (historical) |
| `results/phase1_correct_config.jsonl` | **Canonical 984-record results** (also in experiments/) |
| `results/phase1_results.jsonl` | Wrong-config historical data |
| `results/probe_cells_8_25.jsonl` | Parallel probe during main sweep |
| `phase1_correct_config.log` | Main sweep stdout |
| `phase1_remaining_cells.log` | Follow-up sweep stdout |
| `HANDOFF.md` | Handoff from the previous session (superseded) |
