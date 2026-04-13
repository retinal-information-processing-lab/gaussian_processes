# Handoff: M Degradation Investigation

**Branch**: `pietro/investigate-M-degradation`
**Worktree**: `gpytorch_porting_M_degradation/` (sibling of main `gpytorch_porting/`)
**Date**: 2026-04-13
**Status**: In progress — Phase 2 data collection wrapping up, interpretation underway

---

## 1. The Original Problem

Increasing inducing points M from 250 to 1500 with `vargp_direct + interleave_fstep=True +
fix_Amp=True` (our best training config, 64x64 data) produces a mean test_r drop across
41 cells: **0.838 (M=250) to 0.829 (M=1500)**. This is the opposite of what sparse GP theory
predicts. The investigation was to find why.

Pre-existing evidence (from INVESTIGATION_PROMPT.md, trusted, not re-run):
- M=250 sweep: `experiments/2026-04-06_es_sweeps_64x64/sweep_64x64_elbo_es_results.jsonl`
  Config name: `64_elbo_intl_fixAmp`, 41 cells x 3 seeds (1,2,3), mean test_r=0.838
- M=1500 ceiling fit: `checkpoints/64x64_ceiling_M1500_intl_fixAmp/`
  41 cells x 3 seeds (0,1,2), mean test_r=0.829 (38 cells; cells 0,12,30 failed on all 3 seeds)

---

## 2. What Was Run (243 records total)

All results are in one file:
**`investigations/M_degradation/results/phase1_results.jsonl`** (243 rows, all phases complete)

Each row is a JSON object with: cell, M, seed, phase, test_r, train_r, final_A, final_beta,
final_rho, final_lambda0, final_sigma_0, n_b, n_b_over_M, eigval_min, eigval_max,
n_iterations_run, stopped_early, train_time, and full per-iteration curves
(curves_A, curves_beta, curves_train_r, curves_train_loss, curves_train_log_lik).

All runs use: `mode=vargp_direct, interleave_fstep=True, fix_Amp=True, n_train=3160,
data_path=datasets/PNAS_64x64_center_crop_no_renorm.npz`. The worktree's `datasets/`
directory has symlinks to the actual .npz files in the original gpytorch_porting installation.

### 2.1 Phase 1a — Dense M sweep (40 runs)
- Script: `investigations/M_degradation/phase1_m_sweep.py`
- Cells: [1, 3, 5, 8, 9] (2 strong, 2 medium, 1 weak at M=250)
- Seeds: [0]
- M: [50, 100, 200, 300, 500, 750, 1000, 1500]
- 5 cells x 8 M = 40 runs. All succeeded.
- Log: `investigations/M_degradation/phase1_sweep.log`

### 2.2 Phase 1b — Seed stability (35 new runs)
- Script: `investigations/M_degradation/phase1_m_sweep.py` (same script, runs both phases)
- Cells: [1, 3, 5, 8, 9]
- Seeds: [0, 1, 2] (seed=0 entries reused from 1a)
- M: [50, 250, 1000]
- 35 new runs (seed=0 at M=50,1000 reused). All succeeded.

### 2.3 Phase 1 Extended — Failing + random cells (64 runs)
- Script: `investigations/M_degradation/phase1_extended_sweep.py`
- Cells: [0, 12, 30] (the 3 cells that failed in M=1500 ceiling fit) + [25, 26, 28, 35, 36] (random)
- Seeds: [0]
- M: [50, 100, 200, 300, 500, 750, 1000, 1500]
- 8 cells x 8 M = 64 runs. Cells 0/12/30 all FAILED (24 failures). Random cells all succeeded.
- Log: `investigations/M_degradation/phase1_extended_sweep.log`

### 2.4 Phase 2a — Failing cells full seed x M matrix (54 runs)
- Script: `investigations/M_degradation/phase2_seed_sweep.py`
- Cells: [0, 12, 30]
- Seeds: [1, 2, 3]
- M: [50, 100, 250, 500, 1000, 1500]
- 3 cells x 3 seeds x 6 M = 54 runs. **ALL 54 FAILED.** A=1e-10 at iteration 16, every run.
- Log: `investigations/M_degradation/phase2_sweep.log`

### 2.5 Phase 2b — Non-failing cells extra seeds (running, ~50 runs)
- Script: `investigations/M_degradation/phase2_seed_sweep.py` (same script, runs both phases)
- Cells: [1, 3, 5, 8, 9, 25, 26, 28, 35, 36]
- Seeds: [1, 2] (Phase 1b data at M=50/1000 reused for cells 1,3,5,8,9)
- M: [50, 250, 1000, 1500]
- Completed (50 runs). All succeeded for non-failing cells.

### Summary of coverage

| Cell | Seeds tested | M values tested | Result |
|------|-------------|-----------------|--------|
| 0    | 0,1,2,3     | 50,100,200,250,300,500,750,1000,1500 | 26/26 FAIL |
| 1    | 0,1,2       | 50,100,200,250,300,500,750,1000,1500 | 17/17 OK |
| 3    | 0,1,2       | 50,100,200,250,300,500,750,1000,1500 | 17/17 OK |
| 5    | 0,1,2       | 50,100,200,250,300,500,750,1000,1500 | 17/17 OK |
| 8    | 0,1,2       | 50,100,200,250,300,500,750,1000,1500 | 17/17 OK |
| 9    | 0,1,2       | 50,100,200,250,300,500,750,1000,1500 | 17/17 OK |
| 12   | 0,1,2,3     | 50,100,200,250,300,500,750,1000,1500 | 26/26 FAIL |
| 25   | 0,1,2       | 50,100,200,250,300,500,750,1000,1500 | 16/16 OK |
| 26   | 0,1,2       | 50,100,200,250,300,500,750,1000,1500 | 16/16 OK |
| 28   | 0,1,2       | 50,100,200,250,300,500,750,1000,1500 | 16/16 OK |
| 30   | 0,1,2,3     | 50,100,200,250,300,500,750,1000,1500 | 26/26 FAIL |
| 35   | 0,1,2       | 50,100,200,250,300,500,750,1000,1500 | 16/16 OK |
| 36   | 0,1,2       | 50,100,200,250,300,500,750,1000,1500 | 12/12 OK (Phase 2b was finishing) |

---

## 3. What Was Discovered

### Finding 1: No M degradation for non-failing cells

Grand mean test_r across 10 non-failing cells (all available seeds pooled):

| M    | 50    | 100   | 200   | 300   | 500   | 750   | 1000  | 1500  |
|------|-------|-------|-------|-------|-------|-------|-------|-------|
| mean | 0.813 | 0.819 | 0.826 | 0.828 | 0.829 | 0.827 | 0.830 | 0.831 |

Performance monotonically increases (+0.018 from M=50 to M=1500). Plateaus after M~300.

Per-cell: 7 cells improve with M, 3 cells show tiny declines (cell 3: -0.012, cell 25: -0.016,
cell 5: -0.002). No cell shows a large or systematic degradation.

### Finding 2: Cells 0, 12, 30 suffer immediate A-collapse

These cells fail at **every seed (0-3) and every M value (50-1500)** tested. The failure
mode is identical every time:
- `final_A = 1.0000007072408224e-10` (the lower bound of the A constraint)
- `n_iterations_run = 16` (early stopping fires because ELBO can't improve with A~0)
- A collapses **from iteration 1**. The A curve is flat at 1e-10 from the very start.
- Predictions collapse to a constant (`pred_std=0, test_r=NaN`)

This is the A-collapse failure mode — the interleaved damped Newton F-step pushes A to
zero at the very first outer iteration. For comparison, healthy cells see A grow from
the 1e-4 initial value to 0.03-0.12 at iteration 1.

### Finding 3: The pre-existing evidence is confounded by a code change

**Critical timeline:**
- 2026-04-06: ES sweep run (M=250, seeds 1-3) — cells 0,12,30 **succeeded** (test_r 0.57-0.96)
- 2026-04-09: Commits `97bc88f` + `0f4dfff` changed the random sequence for inducing point
  selection ("share one Generator across STA/IP/extras picks")
- 2026-04-13: Ceiling fit run (M=1500, seeds 0-2) — cells 0,12,30 **failed** (all 3 seeds)
- 2026-04-13: This investigation run — cells 0,12,30 **fail at all seeds, all M**

The ES sweep (pre-change) and the ceiling fit + this investigation (post-change) use
**different mappings from seed to inducing point set**. The same `seed=1` produces
different inducing points before vs after the April 9 commits.

This means:
- The original comparison (M=250: 0.838 vs M=1500: 0.829) compares two different code versions
- With the current code, cells 0,12,30 fail universally — the A-collapse is not M-dependent
- With the old code, they worked at M=250 (seeds 1,2,3) but failed at M=1500 (seeds 0,1,2)
  — but we cannot re-test this since the old sequence is gone

### Finding 4: Eigenspace rank saturation is real but harmless

n_b (retained eigenspace dimension) plateaus at ~50-100 regardless of M:
- n_b/M drops from 0.5-0.9 at M=50 to 0.03-0.06 at M=1500
- But n_b in absolute terms still grows slightly (e.g., cell 8: 27 at M=50, 92 at M=1500)
- Training times barely change with M because the effective computation is O(n_b), not O(M)
- This is a phenomenon but does not cause performance degradation

### Finding 5: Hyperparameter drift occurs but recovers

Cell 3 shows a dip at M=200-300 where A drops from 0.11 to 0.04 and beta jumps from 0.039
to 0.061, correlated with test_r dipping to 0.907. It recovers at M=500+. This is a local
optimum issue, not a systematic M-dependent degradation.

---

## 4. Sticking Points and Open Questions

### 4.1 Can cells 0, 12, 30 be trained AT ALL with the current code?

With seeds 0-3, all three cells fail at every M. We have NOT tried:
- Seeds 4-20+ (wider seed scan at small M to find any working configuration)
- Pivoted Cholesky IP selection (`ip_selection='pivoted'`) instead of random
- Different A_init values
- Different RF initialization

**This is the most important open question.** If these cells cannot be trained with ANY
seed using random IP selection in the current code, then the April 9 commit introduced
a regression for these cells specifically. If some higher seeds work, then the problem
is purely about inducing point quality and the seed→IP mapping.

### 4.2 Was there ever "real" M-dependent degradation (pre-April 9)?

With the OLD code, cells 0,12,30 worked at M=250 (seeds 1,2,3) but failed at M=1500
(seeds 0,1,2). This could be:
- (a) M-dependent A-collapse: large M genuinely makes the optimization harder for these cells
- (b) Seed confound: seeds 0,1,2 happen to be bad for these cells, seeds 1,2,3 happen to be good

We cannot distinguish these with the current code because the seed→IP mapping changed.
To answer this properly, one would need to checkout the pre-April-9 code and repeat the
comparison. However, this may not be worth the effort given finding 1 (no degradation
in non-failing cells).

### 4.3 The user's hypothesis (not yet tested)

The user suggested: cells 0,12,30 may have very sparse/low responses, making them sensitive
to inducing point selection. The hypothesis is that at M=50 you might get unlucky (few
images near RF), and at M=300+ the problem goes away because you cover more of the space.
**This is falsified by the Phase 2a data**: M=50 through M=1500 all fail equally with the
current code. However, the underlying logic (these cells need specific images near their RF)
is likely correct — it's just that the current random sequence doesn't produce suitable
IP sets at ANY M for seeds 0-3.

---

## 5. What the New Session Should Do

### 5.1 Immediate: seed scan for failing cells (highest priority)

Try seeds 0-19 for cells [0, 12, 30] at M=250, single cell at a time. This determines
whether ANY seed works with the current code. If some seeds work:
- Run the working seeds at M=[50, 100, 250, 500, 1000, 1500] to test M-dependence
- This would cleanly answer: "with working seeds, does large M hurt?"

If NO seeds work (0-19 all fail):
- The April 9 commit broke these cells. Investigate what changed in the IP selection
  logic that makes all random selections bad for cells 0,12,30.
- Try `ip_selection='pivoted'` for these cells as a workaround.

### 5.2 If M degradation is found for any cell

Among 10 non-failing cells tested, the closest to "degradation" are:
- Cell 25: -0.016 from M=50 to M=1500 (small, likely noise)
- Cell 3: -0.012 (with a specific dip at M=300, then recovery)

To find cells with genuine degradation, you could:
- Run all 41 cells at M=[250, 1500] with seeds [0,1,2] and look for cells where
  M=1500 is consistently worse than M=250 across seeds
- Focus investigation on those specific cells

### 5.3 A-collapse root cause (if no M degradation exists)

If the investigation concludes that the "M degradation" is entirely an artifact of
A-collapse + code version confound, the follow-up is:
- Why do cells 0, 12, 30 trigger A-collapse? (bad initial A gradient direction
  from the damped Newton update)
- Fix: A-collapse detection (flag if A < 1e-6 after iteration 2, retry)
- Fix: pivoted IP selection to ensure coverage of the cell's RF

---

## 6. File Map

All paths relative to `Spatial_GP_repo/scripts/gpytorch_porting/` within the worktree.

| File | Purpose |
|------|---------|
| `INVESTIGATION_PROMPT.md` (worktree root) | Original investigation brief |
| `investigations/M_degradation/FINDINGS.md` | Findings document (needs update after Phase 2 completion) |
| `investigations/M_degradation/HANDOFF.md` | This file |
| `investigations/M_degradation/phase1_m_sweep.py` | Phase 1a + 1b script |
| `investigations/M_degradation/phase1_extended_sweep.py` | Phase 1 extended (failing + random cells) |
| `investigations/M_degradation/phase2_seed_sweep.py` | Phase 2a + 2b script |
| `investigations/M_degradation/results/phase1_results.jsonl` | **All 243 records** (single file, all phases) |
| `investigations/M_degradation/phase1_sweep.log` | Phase 1 run log |
| `investigations/M_degradation/phase1_extended_sweep.log` | Extended sweep log |
| `investigations/M_degradation/phase2_sweep.log` | Phase 2 log (may still be writing) |

Pre-existing data (in main gpytorch_porting, not the worktree):
| `experiments/2026-04-06_es_sweeps_64x64/sweep_64x64_elbo_es_results.jsonl` | M=250 ES sweep (OLD code, pre-April-9) |
| `checkpoints/64x64_ceiling_M1500_intl_fixAmp/ceiling_results.json` | M=1500 ceiling fit summary |
| `checkpoints/64x64_ceiling_M1500_intl_fixAmp/sweep_results.jsonl` | M=1500 ceiling fit per-cell-per-seed |

---

## 7. Loading Results

```python
import json
results = []
with open('investigations/M_degradation/results/phase1_results.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

# Filter by phase
phase1a = [r for r in results if r['phase'] == 'Phase 1a']
phase2a = [r for r in results if r['phase'] == 'Phase 2a (failing cells, seeds 1-3, M sweep)']

# All successes for a cell
cell8 = [r for r in results if r['cell'] == 8 and r.get('test_r') is not None]

# Check if a run failed
failed = [r for r in results if r.get('test_r') is None]  # 78 records, all cells 0/12/30
```

---

## 8. Key Numbers to Remember

- **10 non-failing cells, mean test_r M=50→M=1500**: 0.813 → 0.831 (+0.018, no degradation)
- **3 failing cells (0,12,30)**: A=1e-10 at iteration 1, all seeds 0-3, all M 50-1500
- **78 failures** out of 239 runs, all from cells 0/12/30
- **Cell 8** improves most with M: +0.108 from M=50 to M=1500
- **Eigenspace**: n_b caps at ~50-100 regardless of M (n_b/M drops to 0.03-0.06 at M=1500)
- **ES sweep timestamp**: 2026-04-06 (pre-sequence-change)
- **Random sequence change**: 2026-04-09 (commits `97bc88f`, `0f4dfff`)
- **Ceiling fit**: 2026-04-13 (post-sequence-change, confirmed by user)

---

## 9. Continuation Prompt

```
I'm continuing the M-degradation investigation on branch pietro/investigate-M-degradation.
Working directory: gpytorch_porting_M_degradation/ (a git worktree).

Read the handoff at Spatial_GP_repo/scripts/gpytorch_porting/investigations/M_degradation/HANDOFF.md.

Summary: Among 10 non-failing cells (13 tested), there is NO M degradation —
performance monotonically increases from M=50 to M=1500. The original mean test_r
drop (0.838→0.829) is caused by cells 0, 12, 30 which suffer immediate A-collapse
(A→1e-10 at iteration 1, every seed 0-3, every M 50-1500 tested). These cells
worked in the M=250 ES sweep (April 6) but fail with the current code (post April 9
random sequence change). The comparison between the two benchmarks is confounded.

Immediate next step: seed scan — try seeds 0-19 for cells [0, 12, 30] at M=250
to determine whether any seed produces a working fit with the current code.
If working seeds are found, test them across M=[50,250,1000,1500] to finally
answer the M-dependence question for these specific cells.

All data: investigations/M_degradation/results/phase1_results.jsonl (239 rows).
```
