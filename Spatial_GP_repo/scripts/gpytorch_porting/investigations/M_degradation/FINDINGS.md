# M Degradation Investigation — Findings

**Branch**: `pietro/investigate-M-degradation`
**Started**: 2026-04-13
**Status**: Phase 1 complete (wrong config). Corrected sweep running.

---

## Problem Statement

Increasing M (inducing points) from 250 to 1500 with `vargp_direct + interleave_fstep=True
+ fix_Amp=True` appeared to degrade mean test_r from **0.838 to 0.829** across 41 cells.
This contradicts sparse GP theory.

Pre-existing evidence:
- M=250: `experiments/2026-04-06_es_sweeps_64x64/sweep_64x64_elbo_es_results.jsonl`
  Config: `64_elbo_intl_fixAmp`, 41 cells x 3 seeds (1,2,3), mean test_r=0.838
- M=1500: `checkpoints/64x64_ceiling_M1500_intl_fixAmp/`
  41 cells x 3 seeds (0,1,2), mean test_r=0.829 (38 cells; cells 0,12,30 failed)

---

## CRITICAL DISCOVERY: Config Mismatch

**The original M=250 vs M=1500 comparison used different training configurations.
The performance gap is a CONFIG effect, not an M effect.**

The M=250 ES sweep (`run_sweep_elbo_es_64x64.py`) overrides 5 parameters for
interleaved training. The M=1500 ceiling fit (`train_ceiling_models.py`) uses
`build_config_from_defaults()` without those overrides, getting the weaker defaults.

| Parameter | ES sweep (M=250) | Ceiling fit (M=1500) | Impact |
|-----------|-------------------|----------------------|--------|
| A_init | **1e-4** | 0.01 | Critical: 0.01 causes E-step Newton overshoot (documented in ES sweep script lines 51-52) |
| lambda0_init | **-1.0** | 1.0 | Better starting point for bias |
| n_estep | **50** | 10 | 5x more E-step iterations per outer loop |
| n_mstep | **20** | 10 | 2x more M-step iterations |
| n_iterations | **80** | 50 | 1.6x more outer iterations |

### Verification: Cell 8, seed=1, M=250

| Code version | Config | test_r |
|-------------|--------|--------|
| Old (pre-Apr-9) | ES sweep config | 0.8740 |
| New (current) | ES sweep config | **0.8743** |
| New (current) | Default config | 0.8378 |

- Old code vs new code with same config: **identical** (delta=0.000). The April 9
  commit did NOT introduce a systematic performance change.
- Same code, ES config vs default config: **-0.036**. The config difference alone
  explains the gap.

### A-collapse of Cell 0: also a config issue

| Config | Cell 0 test_r |
|--------|---------------|
| Default (A_init=0.01) | FAIL (A -> 1e-10) |
| ES sweep (A_init=1e-4) | **0.586** (succeeds) |

The A-collapse that the first session attributed to "seed-specific fragility" was
actually caused by using A_init=0.01 with interleaved F-step. The ES sweep script
explicitly documents this: "A_init=0.01 + interleave gives 19/19 E-step divergences."

Exception: Cells 12 and 30 fail even with A_init=1e-4 (separate issue, deferred).

---

## Phase 1 Data (Collected with WRONG config — A_init=0.01, n_estep=10)

The previous session's 243 records in `results/phase1_results.jsonl` used the default
config, not the ES sweep config. This data answers "does M matter with the weak config?"
(answer: not much), but does NOT answer the original question.

### Summary (wrong config)

With the same (default) config at all M values, M=250 and M=1500 differ by only +0.005:

| M    | 50    | 100   | 200   | 250   | 300   | 500   | 750   | 1000  | 1500  |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| mean | 0.816 | 0.819 | 0.826 | 0.829 | 0.828 | 0.829 | 0.827 | 0.830 | 0.831 |

(10 non-failing cells, seed=0 for M=100-750, seeds 0-2 pooled for M=50/250/1000/1500)

No M degradation. But this is with the suboptimal config — the model underperforms at
all M values, so there's less room for M to matter.

---

## Corrected Sweep (ES sweep config)

**Script**: `investigations/M_degradation/phase1_correct_config.py`
**Results**: `investigations/M_degradation/results/phase1_correct_config.jsonl`

Config: A_init=1e-4, lambda0_init=-1.0, n_estep=50, n_mstep=20, n_iterations=80,
interleave_fstep=True, fix_Amp=True, ELBO ES (patience=15).

Cells: [0, 1, 3, 5, 8, 9, 12, 25, 26, 28, 30, 35, 36] (13 cells, same as Phase 1)
Seeds: [0, 1, 2]
M: [50, 100, 200, 300, 500, 750, 1000, 1500]
Total: 312 runs (cells 12,30 expected to fail: 48 runs x ~2s each)

*[Results pending]*

---

## Hypothesis Assessment (Updated)

| ID | Hypothesis | Status | Evidence |
|----|-----------|--------|----------|
| H0 | **Config mismatch** | **ROOT CAUSE of original finding** | ES sweep (M=250) used A_init=1e-4, n_estep=50; ceiling fit (M=1500) used A_init=0.01, n_estep=10. Same cell+seed matches perfectly with same config. |
| H1 | IP selection quality | Untested with correct config | Deferred |
| H2 | Eigenspace rank saturation | Confirmed but harmless | n_b/M drops to 0.03-0.06 at M=1500, but absolute n_b still grows |
| H3 | Optimization insufficiency | Plausible with correct config | The ES config has 50 E-steps and 20 M-steps per iteration — large M may need even more. Corrected sweep will test this. |
| H4 | Hyperparameter drift | Minor | Cell 3 dip at M=300 (wrong config); may differ with correct config |
| H5 | Train-test divergence | Not confirmed | |

---

## Files

| File | Purpose |
|------|---------|
| `phase1_m_sweep.py` | Phase 1 sweep (wrong config, A_init=0.01) |
| `phase1_extended_sweep.py` | Extended sweep (wrong config) |
| `phase2_seed_sweep.py` | Seed sweep (wrong config) |
| `phase1_correct_config.py` | **Corrected sweep (ES config)** |
| `results/phase1_results.jsonl` | 243 records — wrong config data |
| `results/phase1_correct_config.jsonl` | Corrected config data (pending) |
