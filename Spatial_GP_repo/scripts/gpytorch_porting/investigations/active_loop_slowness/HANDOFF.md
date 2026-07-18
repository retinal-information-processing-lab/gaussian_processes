# Investigation: Active Loop Slowness, Eigenvalue Growth, and Rank-1 Update

**Branch**: `pietro/utility_optimization`
**Date**: 2026-04-10
**Status**: Mostly resolved. Rank-1 update, autograd leak, and beta explosion all FIXED. Stuck-near-init remains OPEN.
**Location**: `investigations/active_loop_slowness/`

---

## Problem Statement

The active learning loop (`run_active_loop.py`) grows the model from M=50 to M=500 by adding one inducing point per iteration. A batch of 14 cell-0 runs (7 seeds x 2 strategies) revealed:

1. **5-7x slower than projected** (30-60 min/run vs 8-10 min estimate)
2. **3 of 14 runs OOM'd** during test-set prediction (22.9 GB GPU memory)
3. **6 of 14 runs produced near-zero test_r** despite completing normally (stuck-near-init)
4. **The eigenspace dimension n_b tracks M almost linearly** in healthy runs (n_b/M > 0.95), defeating the purpose of eigenspace compression
5. ~~**The "rank-1 update" (`rank1_update.py`) recomputed K_tilde from scratch**~~ — FIXED in `bb27094`. Now uses efficient O(M) column-append ported from `utils.py:add_one_img_to_kernel`. No wall-time speedup on 64x64 (extend step is negligible vs training). See FINDINGS.md Task 1.

The investigation aims to fix the active loop so it runs cleanly, efficiently, and predictably across cells and seeds.

---

## What Was Tried

### Approach 1: Like-for-like comparison (active loop vs single-run fit)
- **What**: Added `--train-indices-from` flag to `run_single_mode.py` to train on the exact same 300 pool indices the active loop used. Committed in `97bc88f` + `0f4dfff`.
- **Result**: Single run on cell 8's 300 indices: test_r=0.5925 in 7.0s. Active loop endpoint: test_r=0.6236 in ~18 min. Similar kernel params (beta identical to 4 decimals).
- **Interpretation**: Base training code is NOT broken. The active loop reaches a slightly better optimum (0.03 higher test_r) through 1300 incremental EM iterations vs 50 from-scratch EM iterations. The 150x slowness is per-iteration overhead (kernel recompute + utility eval + checkpoint save + test eval), not a training bug.
- **Verdict**: CONFIRMED — base training code is sound.

### Approach 2: 64x64 screening (remove image-size confounder)
- **What**: Added `--data-path` flag to `run_active_loop.py`. Ran 6 screening runs on PNAS_64x64 (cells 0 and 8, seeds 0/1/42, both strategies, n_active=100).
- **Result**:
  - Stuck-near-init PERSISTS on 64x64 (cell 0 seed 1: beta=0.1046, rho=0.1031, identical to 108x108 pattern). NOT an image-size artifact.
  - Beta explosion is ABSENT on 64x64 (cell 0 seed 0 random: beta stays at 0.086 vs 0.448 on 108x108). Image-size IS a factor for beta explosion specifically.
  - Stuck models predict BETTER on 64x64 (test_r 0.27-0.38 vs 0.01-0.27 on 108x108) because the center crop concentrates RF signal.
- **Verdict**: 64x64 is a better default for investigation runs. Stuck-near-init is seed-dependent, not image-dependent.

### Approach 3: n_b vs n_train isolation (apples-to-apples eigenvalue study)
- **What**: Ran 3 fits with the SAME 300 inducing points but different n_train (300, 1500, 3160). Script: `investigations/active_loop_slowness/test_ntrain_vs_nb.py`.
- **Result**:

  | n_train | n_b_final | n_b/M | rho | beta | test_r |
  |---:|---:|---:|---:|---:|---:|
  | 300 | 284 | 0.95 | 0.036 | 0.107 | 0.59 |
  | 1500 | 152 | 0.51 | 0.064 | 0.091 | 0.81 |
  | 3160 | 159 | 0.53 | 0.071 | 0.102 | 0.82 |

  Same 300 inducing points, same init (n_b=99 at init in all three). The M-step converges to different rho depending on how much training data constrains it. Lower n_train → lower rho → less inter-pixel correlation → higher effective rank → more eigenvalues above threshold.
- **Interpretation**: n_b ≈ M in the active loop is NOT a bug in the eigenspace machinery. It is a direct consequence of the M == n_train constraint (only 300 data points to constrain the M-step). With n_train=1500+, n_b saturates at ~155 (52% of M) and test_r jumps to 0.81+.
- **Verdict**: CONFIRMED — eigenvalue growth is driven by under-constrained M-step, not by rank-1 extension.

### Approach 4: Checkpoint staleness fix + tooling
- **What**: Fixed `config['M']` and `config['n_train']` staleness in active-loop checkpoints. Added `load_pool_indices()` helper. Refactored randperm seeding to isolated Generators.
- **Result**: All committed and pushed (`97bc88f`, `0f4dfff`). Anti-slop review found and fixed an identical-permutation bug (H1 in the review).
- **Verdict**: Infrastructure is clean. Ready for investigation use.

---

## Key Findings

1. ~~**CONFIRMED**: `rank1_update.py` recomputed full K_tilde from scratch~~ — FIXED in `bb27094`. Now uses efficient O(M) column-append. K_tilde stored in `DirectVariationalState`, only the new column is computed each iteration. Verified bit-identical to full recompute within float32 precision. No wall-time speedup on 64x64 (extend step is negligible vs training).

2. **CONFIRMED**: n_b ≈ M is caused by M == n_train under-constraining the M-step. With the same 300 inducing points, n_train=300 gives n_b=284 (rho=0.036), n_train=1500 gives n_b=152 (rho=0.064), n_train=3160 gives n_b=159 (rho=0.071). The relationship is smooth and monotonic.

3. **CONFIRMED**: Stuck-near-init (beta ≈ 0.103, rho ≈ 0.103) persists on both 108x108 and 64x64, and uses ground_truth RF init. It is seed-dependent, affecting 6 of 14 cell-0 runs. Root cause still unknown — possibly uninformative initial training set.

4. **CONFIRMED**: Beta explosion (beta 0.12 → 0.45) only happens on 108x108, not on 64x64. At beta=0.45 the C matrix covers all 11664 pixels (519 MB), causing OOM. The wider image provides more room for the positive feedback loop (wider beta → bigger mask → noisier signal → optimizer pushes beta wider).

5. **CONFIRMED**: The unexplained OOM with healthy params (cell 0 seed 5 argmax, beta=0.062 at crash) is NOT explained by any of the above. 22.9 GB GPU memory at crash with healthy kernel params. Likely a memory accumulation bug in the active loop infrastructure (model deepcopy, checkpoint retention, utility graph retention).

6. **CONFIRMED**: Per-iteration wall time on cell 0 random (no utility computation) is 4-12x slower than cell 8 random at the same M. The dominant cost is kernel evaluation (C matrix size scales with mask, mask scales with beta). This affects BOTH strategies equally — the utility computation adds overhead on top but is not the primary bottleneck.

7. **CONFIRMED**: The old active loop (utils.py) also ran the M-step in phase 2 (nMstep=10, nEstep=10). Verified in `utils.py:threaded_train_GP_phase2` and `config.py:198-199`. So the kernel was re-optimized at every step in both old and new implementations.

---

## Why This Was Stopped

Context was running low in the original session (2026-04-10). A follow-up session (2026-04-12) implemented all critical fixes: efficient rank-1 column append (`bb27094`), autograd memory leak 15x speedup (`d771c90`), and beta explosion safeguard (`eaa072e`). Only stuck-near-init diagnosis remains open.

---

## Things Noticed But Not Acted Upon

1. **The `--data-path` flag added to `run_active_loop.py` is uncommitted** (6-line diff, see below). Should be committed.

2. **The old code's incremental K_tilde append (line 443-469 in utils.py) only appends a new COLUMN, then re-eigendecomposes.** It does NOT use Sherman-Morrison or Woodbury to avoid the O(M^3) eigendecomposition. So the O(M^3) cost per iteration is present in BOTH old and new implementations. The savings from column append is in kernel evaluation (O(M) vs O(M^2) kernel calls), not in eigendecomposition. This is important context for the rank-1 implementation task.

3. **n_train=1500 and n_train=3160 give very similar n_b (152 vs 159) and test_r (0.81 vs 0.82).** The marginal benefit of training data beyond ~1500 is small. If the active loop could train on the ~1500 most informative UNLABELED images (using their pixel values to constrain the kernel, not their spike counts), n_b would drop and the model would improve dramatically. This is the "semi-supervised kernel learning" idea — out of scope but potentially very high impact.

4. **The per-iteration timing data reveals that stuck-near-init runs (cell 0 seeds 1/2/3/5) have similar wall time for argmax and random** (~4.5s/iter). The utility computation is nearly free when the model is degenerate (trivial predictions → trivial entropy differences → argmax degenerates). The slowness on those runs is purely from the kernel evaluation on 300+ inducing points through a ~2500-pixel C matrix.

5. **An anti-slop review (opus subagent) found and we fixed several code quality issues** in the `--train-indices-from` implementation. Remaining unfixed items from that review: M2 (`.get()` camouflage pattern), M3 (argparse mutual-exclusion leak), L1 (assert → raise). See commit `0f4dfff` message for the full list.

---

## Uncommitted Changes

```
M run_active_loop.py  — 6 lines: adds --data-path CLI flag (already tested, works on 64x64)
M imgs/default_gpy_M100.png  — plot artifact, not investigation-related
M imgs/vargp_direct_M100.png — plot artifact, not investigation-related
```

The `run_active_loop.py` change should be committed before the next session starts.

---

## Files Created

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| `investigations/active_loop_slowness/README.md` | Full investigation context from prior session (2026-04-09). Hypotheses, reproduction recipes, data locations. | **Keep** — essential context |
| `investigations/active_loop_slowness/HANDOFF.md` | This file | **Keep** |
| `investigations/active_loop_slowness/test_ntrain_vs_nb.py` | Apples-to-apples n_train vs n_b comparison script | **Keep** — reproducible experiment |
| `investigations/active_loop_slowness/phase1_ntrain_vs_nb.log` | 2-way comparison output | Keep (reference) |
| `investigations/active_loop_slowness/phase1_ntrain_vs_nb_3way.log` | 3-way comparison output (300/1500/3160) | Keep (reference) |
| `investigations/active_loop_slowness/phase1_eigenvalue_study/` | Single-run n_b study output | Keep (reference) |
| `results/active_loop/2026-04-10_64x64_screening/` | 6 screening runs on 64x64 | **Keep** — investigation data |

---

## If Someone Revisits This

### What to do next (priority order)

1. ~~**Implement efficient rank-1 K_tilde column append**~~ — DONE (`bb27094`). No wall-time speedup on 64x64 (extend step is negligible vs training). See FINDINGS.md Task 1.

2. ~~**Profile GPU memory accumulation**~~ — DONE (`d771c90`). Root cause: autograd graph chaining through variational params. Fix: detach m_b/V_b, no_grad E-step, detach A/lambda0, clear .grad on deepcopy. **15x speedup** (45.7 min → 3.0 min). See FINDINGS.md Task 2.

3. **Diagnose stuck-near-init** (H2) — ROOT CAUSE FOUND, NO FIX YET. A collapses to 0.0001 during Phase 1. Structural zero-gradient trap. More EM iters don't help. M=100 recovers A but kernel stays at init. See FINDINGS.md Task 3.

4. ~~**Consider beta upper bound**~~ — DONE (`eaa072e`). BETA_MAX tightened 1.0 → 0.3, mask coverage warning added. See FINDINGS.md Task 4.

### What NOT to try again

- **Checking eigval_tol**: verified identical (1e-4) across old and new code, all configs, all YAML files. Finding 1 in README.md. Do not re-investigate.
- **Checking jitter on K_tilde**: there is none, by design. Neither old nor new code adds jitter before eigendecomposition. Finding 1 in README.md.
- **STA edge artifact as cause of stuck-near-init**: ruled out. The active loop uses `rf_init='ground_truth'`, and stuck-near-init persists on 64x64 which has no edge artifact.
- **Image size as cause of stuck-near-init**: ruled out. Same pattern on 108x108 and 64x64.

### Prerequisites

- Commit the `--data-path` change to `run_active_loop.py` before starting new work.
- Read `investigations/active_loop_slowness/README.md` for the full problem characterization (failure modes A-D, reproduction recipes, per-seed timing data).
- Read `notebooks/ACTIVE_LEARNING_LOOP_REFERENCE.md` for the old active loop implementation (especially Section 4 and Section 7 for the incremental kernel update functions).

---

## Continuation Prompt

The following is a ready-to-use prompt for a new Claude Code session that picks up this work as an agent team. It should be pasted as the initial message in a fresh session.

---

