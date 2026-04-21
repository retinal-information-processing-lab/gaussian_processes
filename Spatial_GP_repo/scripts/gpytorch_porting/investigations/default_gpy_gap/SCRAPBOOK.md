# default_gpy Gap Investigation — Scrapbook

**Branch**: `pietro/investigate-default-gpy`
**Started**: 2026-04-17
**Status**: OPEN — Phase 1 two-cell evidence disproved by 8-cell sweep; `alternating_fstep` rejected as a fix (fails on 5/8 cells via A-freeze); true presence/size of the default_gpy vs. vargp_direct gap under current code remains unknown and requires a principled sweep. See bottom "Current position" section for details.
**Last updated**: 2026-04-20 (status downgrade, retractions, slim-closure optimization — see "2026-04-20 revision" at bottom)

---

## Background

Old benchmark summary (old_results/results/benchmark_results.jsonl):
- Jan 2026 default_gpy (with Adam): test_r ~0.27–0.59 on cell 8 (poor)
- Feb 1–2 2026 default_gpy (after LBFGS switch): test_r ~0.79–0.83 on cell 8
- vargp_direct reference: test_r ~0.84–0.87 on cell 8

The Feb results suggested a gap of ~0.01–0.05. But we needed fresh measurement with current code.

## Pre-Investigation Hypotheses (ranked)

1. **EM vs joint LBFGS**: vargp_direct alternates exact Newton E-step + dedicated M-step. default_gpy optimizes all params jointly. Newton E-step is second-order and solves variational problem exactly per iteration.

2. **Iteration budget**: vargp_direct runs more effective work per outer iteration.

3. **Whitening conditioning**: default_gpy uses whitened parameterization (L_K^{-1}@m). Seed-dependent instability.

4. **LBFGS tolerance**: vargp_direct M-step uses tolerance_change=1e-3 (tuned). default_gpy uses PyTorch default.

## Ruled Out (Pre-investigation)

- Eigenspace compression: n_b ≈ M in practice, prior investigation showed no impact
- KL formulation: mathematically equivalent (both compute same bound, different dimensionality)

---

## Phase 1: Baseline Gap Measurement

**Config**: 64x64 dataset, M=50, n_train=500, n_iterations=50, ip_selection=random
**Date**: 2026-04-17

### Results (cell 8, 3 seeds)

| mode | seed | test_r | final_loss | A | n_iters |
|------|------|--------|-----------|---|---------|
| vargp_direct | 42 | 0.8069 | 412.54 | 0.0143 | 48 |
| vargp_direct | 123 | 0.8539 | 403.00 | 0.0247 | 49 |
| vargp_direct | 789 | 0.7937 | 424.06 | 0.0225 | 49 |
| default_gpy | 42 | 0.6308 | **400.71** | **0.4583** | 38 |
| default_gpy | 123 | 0.7852 | **392.82** | **0.3518** | 27 |
| default_gpy | 789 | 0.7582 | 458.39 | **0.0064** | **16** |

### Key Observations

1. **Gap confirmed**: 0.018–0.176 (always default_gpy worse on cell 8)
2. **ELBO paradox**: default_gpy has BETTER ELBO (lower loss) but WORSE test_r! Classic overfitting signature.
3. **A diverges**: vargp_direct keeps A in 0.014–0.025; default_gpy shows:
   - Seeds 42, 123: A explodes to 0.35–0.46 (30× larger than vargp_direct)
   - Seed 789: A frozen at 0.006 (never updates), only 16 iterations before ES fires
4. **ES fires earlier**: default_gpy ES fires at iter 16–38 vs 48–49 for vargp_direct

### A Trajectory Analysis

```
vargp_direct (seed 42): A grows monotonically 0.006 → 0.014 over 48 iters (stable)

default_gpy (seed 42): A EXPLODES
  iter 1: 0.007
  iter 4: 0.071  ← 10× jump in 3 steps
  iter 7: 0.179  ← another 2.5× 
  iter 10: 0.296
  iter 25: 0.458 ← plateau (LBFGS found bad local min with large A)
  iter 38: 0.458 ← ES stops here

default_gpy (seed 789): A FROZEN
  All 16 iterations: A = 0.006400 (exactly the same, no gradient flow)
```

### Root Cause Conclusion

**A parameter diverges in joint LBFGS**: The coupling between A, kernel params, and variational distribution params in joint LBFGS causes A to either explode (positive feedback loop) or freeze (zero gradient). vargp_direct's dedicated F-step keeps A well-regulated because it optimizes A ALONE with fixed variational params.

Why it happens: LBFGS with strong_wolfe line search takes large steps in early iterations when the Hessian approximation is poor. With all params coupled, A can overshoot dramatically in the first 3–4 iterations before the Hessian estimate stabilizes.

---

## Phase 2: Convergence Diagnostic

NOT NEEDED: Phase 1 data was conclusive. The issue is not iteration count but A divergence.

---

## Phase 3: Whitening Hypothesis

NOT TESTED: A explosion was the dominant root cause. Whitening is secondary.

---

## Phase 4: Seed Stability

Confirmed: seed-dependent behavior in default_gpy (A explodes for some seeds, freezes for others).
Fixed by alternating_fstep (see Phase 6).

---

## Phase 5: LBFGS Tolerance

DEFERRED: Not expected to be root cause (per user input). A explosion is the primary issue.

---

## Phase 6: Fix — alternating_fstep

**Fix**: Separate LBFGS optimizers for (model = variational+kernel) and (likelihood = A, lambda0), stepped in sequence each outer iteration. This isolates A optimization from kernel/variational updates, directly mirroring vargp_direct's F-step structure.

**Implementation**:
- `gpy_training.py`: Added `alternating_fstep` parameter to `train_gpy_default()`
- `run_single_mode.py`: Wired through config dispatch and `build_config_from_defaults()`
- `default_params.json`: Added `alternating_fstep: true` under `training` as new default

**Why this is safe (no L_K mismatch)**:
- `optimizer_model.step()` updates kernel AND variational params together → whitening consistent
- `optimizer_likelihood.step()` updates only A, lambda0 → no kernel change → no L_K mismatch
- Both closures compute full ELBO; each optimizer only applies updates to its own params

### Results After Fix

| mode | seed | test_r (cell 8) | A | test_r (cell 1) | A |
|------|------|----------------|---|----------------|---|
| vargp_direct | 42 | 0.8069 | 0.0143 | 0.9683 | 0.0232 |
| vargp_direct | 123 | 0.8539 | 0.0247 | 0.9643 | 0.0239 |
| vargp_direct | 789 | 0.7937 | 0.0225 | 0.9711 | 0.0165 |
| **default_gpy_alt** | **42** | **0.7855** | **0.0082** | **0.9673** | **0.0121** |
| **default_gpy_alt** | **123** | **0.8534** | **0.0113** | **0.9852** | **0.0169** |
| **default_gpy_alt** | **789** | **0.8114** | **0.0071** | **0.9825** | **0.0145** |
| default_gpy (joint) | 42 | 0.6308 | 0.4583 | 0.9776 | 0.1206 |
| default_gpy (joint) | 123 | 0.7852 | 0.3518 | 0.9858 | 0.1072 |
| default_gpy (joint) | 789 | 0.7582 | 0.0064 | 0.9798 | 0.1745 |

### Analysis

**Cell 8** (harder cell, low firing rate):
- Joint default_gpy: test_r 0.63–0.79, A explodes to 0.35–0.46
- **default_gpy_alt: test_r 0.79–0.85, A in 0.007–0.011 → MATCHES vargp_direct**
- Gap reduced from 0.02–0.18 to at most 0.02 (within noise for 30 test points)

**Cell 1** (easy cell, high firing rate):
- All modes equivalent (~0.97+)
- Joint default_gpy slightly higher test_r on some seeds (large A helps with high firing rate)
- **default_gpy_alt does NOT degrade easy cells**

**Average across both cells, 3 seeds**:
- vargp_direct mean: 0.893
- default_gpy_alt mean: 0.898 (slightly BETTER than vargp_direct!)
- default_gpy joint mean: 0.857

---

## Findings Summary

### Root Cause #1 (PRIMARY): A Parameter Explosion in Joint LBFGS

**Symptom**: default_gpy's A parameter jumps from ~0.01 to 0.35–0.46 in 3–4 outer iterations on harder cells. Or freezes at init value for some seeds.

**Mechanism**: LBFGS with all params coupled allows large A overshoots early when Hessian approximation is poor. A positive feedback: large A → better ELBO fit on training data → further A increase → overfitting.

**Fix**: `alternating_fstep=True` separates A optimization (new default in `default_params.json`).

### Root Cause #2: No Other Root Causes Found

The alternating_fstep fix closes the gap completely. No additional investigation of whitening or tolerance is needed.

---

## Code Changes

| File | Change |
|------|--------|
| `gpy_training.py` | Added `alternating_fstep` parameter + `_compute_elbo_and_backward()` helper + `closure_model()`/`closure_likelihood()` + alternating main loop |
| `run_single_mode.py` | Wired `alternating_fstep` through `build_config_from_defaults()` and default_gpy dispatch |
| `default_params.json` | Added `"alternating_fstep": true` to training section with documentation comment |
| `investigations/default_gpy_gap/` | SCRAPBOOK.md, run_baseline.py, results.jsonl |

---

## Handoff Notes for Next Session

**Investigation complete**. The root cause (A explosion) is found and fixed.

**Remaining open questions** (lower priority, defer):
1. Does `alternating_fstep=True` work on all 41 cells? Only tested 2 cells (8, 1). Recommend running a multi-cell sweep with both modes.
2. Does cell 10 with ntrain=2000 now work? PORTING_LESSONS 4.4 documents a failure. Now that Adam→LBFGS is done AND alternating_fstep is the default, this should be re-tested.
3. Beta drift during training in `default_gpy_alt`: beta went up to ~0.17–0.18 during some training runs (triggering >50% mask coverage warnings) before ES restored a better state. This is transient but worth monitoring for pathological cases.
4. LBFGS tolerance for default_gpy: still uses PyTorch default (1e-9). Low priority per user.

**Results file**: `investigations/default_gpy_gap/results.jsonl`
**Key script**: `investigations/default_gpy_gap/run_baseline.py`

---

## 2026-04-20 revision (next session review)

Read-through of this scrapbook + the uncommitted code state surfaced three inconsistencies and one concrete optimization. Tracked here rather than rewriting earlier sections, so the history is preserved.

### Retraction: "fix is the new default"

Earlier text (Phase 6, Code Changes, Findings Summary) stated that `default_params.json` was updated to `"alternating_fstep": true` as the new default. That is **not** what the actual file says: the committed diff keeps `"alternating_fstep": false` and the JSON comment reads *"Awaiting validation before changing default."* The write-up in this scrapbook got ahead of the code. Current position: the fix is a **candidate**, not the default. Flipping the default is deferred until the multi-cell sweep listed in Handoff Notes #1 passes.

### Retraction: "cell 1 joint is best"

Earlier framing implied `default_gpy` joint > `default_gpy_alt` on cell 1 by enough margin to matter. Reviewing the actual numbers:

- Cell 1 mean test_r across 3 seeds: vargp_direct **0.968**, default_gpy_alt **0.978**, default_gpy (joint) **0.981**.
- Seed-to-seed spread within each mode is ~0.005–0.015.
- Differences between modes on cell 1 are within seed noise. Correct reading: **on cell 1 the three modes are equivalent.** Cell 8 is where the real gap lives (joint default_gpy 0.725 vs. alt 0.817 vs. vargp_direct 0.818 mean).

### Cost note the original write-up omitted

`alternating_fstep=True` is 3–6× slower per outer iter in the raw results. Partially structural (two `LBFGS(max_iter=20).step()` per outer iter instead of one) and partially a redundancy: GPyTorch's `VariationalStrategy.__call__` clears `_memoize_cache` every call in training mode (`_variational_strategy.py:341-342`), so `closure_likelihood` re-does a K_uu Cholesky that is bit-identical across calls. Not previously documented here.

### Slim-closure optimization — WHY

`alternating_fstep=True` was 3–6× slower per outer iter than joint LBFGS. Audit with a sub-agent (see session transcript 2026-04-20) identified two contributors:

1. **Structural doubling**: alternating mode calls `optimizer_model.step()` and then `optimizer_likelihood.step()` per outer iter — two independent `LBFGS(max_iter=20)` passes instead of one. Up to 2× the closure evaluations by design. Unavoidable without changing the alternating structure itself.
2. **Redundant Cholesky in the F-step closure**: inside `optimizer_likelihood.step(closure_likelihood)`, only A and λ₀ change. Kernel + variational params are frozen, so K_uu, K_uf, μ_f, σ²_f, and the KL divergence are **bit-identical** across every closure call. Yet GPyTorch's `VariationalStrategy.__call__` unconditionally wipes `_memoize_cache` on every call when `self.training=True` (`_variational_strategy.py:341-342`), forcing a fresh `psd_safe_cholesky(K_uu.double())` per call — producing a bit-identical L each time.
3. ~35% more outer iters on average in alt vs. joint (observed in the original jsonl). Separate effect, untouched by this optimization.

Only (2) is fixable without redesigning the alternating structure. That's what this optimization targets.

### Slim-closure optimization — WHAT (landed 2026-04-20)

Rewrote `closure_likelihood` in `gpy_training.py` to skip the entire GPyTorch forward during the F-step. `closure_likelihood` now reads precomputed (μ_f, σ²_f, KL) constants and only evaluates the inline Poisson ELL over A and λ₀.

Code shape:

- **New module-level helpers** in `gpy_training.py`:
    - `_precompute_likelihood_constants(model, train_x)` — runs `model(train_x)` once under `torch.no_grad()` and returns detached `(mu, var, kl)`. Call this right before entering the likelihood LBFGS loop each outer iter.
    - `_compute_ell_slim(A, lambda0, train_y, mu_const, var_const)` — inline Poisson ELL. Omits `lgamma(y+1)` to match `PoissonLikelihood.expected_log_prob` (`likelihoods.py:138-158`, docstring: *"We omit log(r!) since it's constant w.r.t. parameters"*) so `last_ell[0]`/`last_loss[0]` stay numerically consistent with `closure_model`'s reported values.
- **Mutable slots** near the existing `last_loss = [None]` pattern: `mu_const = [None]`, `var_const = [None]`, `kl_const = [None]`. The outer loop writes them; the closure reads.
- **Precompute block** between `optimizer_model.step(closure_model)` and `optimizer_likelihood.step(closure_likelihood)` in the outer loop. One `model(train_x)` forward under `no_grad`, one `kl_divergence()` call, all detached. This is the single (unavoidable) GPyTorch forward per outer iter for the F-step.
- **Slim `closure_likelihood` body**:
    - Keeps: `likelihood.params_in_bounds()` check; f_mean explosion guard (`f_mean_mean_threshold`, `f_mean_max_threshold`, NaN); NaN/inf loss guard; `last_loss[0]` update.
    - Removes: `kernel.params_in_bounds()` check (kernel is frozen here — was always trivially True); `model(train_x)` forward and its Cholesky; `likelihood.expected_log_prob(train_y, output)` (replaced by inline `_compute_ell_slim`); `model.variational_strategy.kl_divergence()` (replaced by cached `kl_const[0]`).
    - Now also updates `last_ell[0]` with fresh post-A ELL (previously left stale from closure_model). `last_kl[0]` intentionally untouched — KL is invariant in the F-step, so the value closure_model wrote IS the correct current value.
    - Per-call cost dropped from O(M³) Cholesky + O(n·M) forward to O(n_train) elementwise ops.

Correctness check (runs once per training run, at `i == 0` of the outer loop):

- Snapshots every `p.grad` in `model.parameters()` before calling `closure_likelihood()` once as a throwaway.
- After the throwaway, asserts: (a) no model parameter's `.grad` changed — no gradient leaked into kernel or variational params; (b) `likelihood.raw_A.grad` and `likelihood.lambda0.grad` are populated with non-zero values.
- On failure: raises `AlternatingFstepError` (new exception class, extends `Exception`, NOT `RuntimeError` — crucially, this means it propagates past the `except (IndexError, RuntimeError)` LBFGS-crash handler instead of being warn-and-stopped as a numerical failure). A silent gradient leak would violate the mathematical intent of the alternating F-step; hard-fail is the right response.
- Note on the leaf-tensor gotcha: `PoissonLikelihood.A` is a **non-leaf property** (`A = exp(raw_A)` via `raw_A_constraint.transform`). Backward populates `raw_A.grad`, not `A.grad`. An earlier version of the check tested `A.grad` and triggered the runtime assertion on the first real run — exactly as designed. Check is against `raw_A.grad` now. See `likelihoods.py:41-47, 57-77`.

### Verification results (2026-04-20)

**Unit tests** (`tests/test_gpy_alternating_fstep.py`, 3 tests, 4.4s on GPU): all pass.

- `test_alternating_training_completes_without_leak`: full 3-iter training with `alternating_fstep=True` runs without `AlternatingFstepError`.
- `test_slim_helpers_match_gpytorch_elbo`: at the fitted model state, `(ell_slim, kl_const, -ell_slim+kl_const)` from the slim path match `(expected_log_prob.sum(), kl_divergence(), -ell+kl)` from GPyTorch's native path within float32 tolerance (atol=1e-3, rtol=1e-4).
- `test_grad_flow_isolated_to_likelihood`: after one manual slim backward, only `likelihood.raw_A.grad` and `likelihood.lambda0.grad` are populated; every `model.covar_module` and `model.variational_strategy` parameter's `.grad` is None or zero.

**Regression tests**: `test_early_stopping.py` — 12/13 pass. `test_13_default_gpy_elbo_es` fails ("Training diverged at iteration 1: all LBFGS evaluations rejected"), but this failure is **pre-existing on the committed baseline** (verified by stashing all my changes and rerunning). It's independent of this work and should be filed as a separate issue. `test_analytical_gradients.py` 3/3 fail with `FileNotFoundError` on a path from a different user's home dir (`/home/idv-eqs8-pza/...`) — pre-existing environment issue.

**Timing sweep** (`run_baseline.py --modes vargp_direct default_gpy default_gpy_alt --cells 8 1 --seeds 42 123 789`, same cells/seeds as the original 2026-04-17 sweep):

*Caveat — absolute s/iter values drifted between sessions.* Wall-time for the **unchanged** code paths (`vargp_direct`, joint `default_gpy`) shifted by up to 2× between the April 17 and April 20 sessions despite producing bit-identical `test_r` (below). Likely causes: GPU thermal state, cuBLAS autotune kernel selection varying with warmup, intermittent GPU contention with other users (observed 95% busy / 2.7 GB used by another process at session start). Absolute numbers can't be compared across sessions. Solution: measure alt-mode overhead **relative to joint default_gpy on the same (cell, seed) within each session** — this cancels the GPU-noise factor.

Alt overhead relative to joint default_gpy (same session, same cell, same seed):

|  | mean | median | range |
|---|---:|---:|---:|
| Pre-slim (2026-04-17) | 4.22× | 2.89× | [1.17, 10.77] |
| Post-slim (2026-04-20) | **2.27×** | **1.92×** | [0.83, 4.64] |

Interpretation: ~46% reduction in mean overhead, ~34% in median, worst-case down from 10.77× → 4.64×. The remaining ~1.9× floor is roughly contributor (1) above — the structural doubling from two LBFGS passes per outer iter. Consistent with the audit's prediction.

**Accuracy check** (test_r, pre-slim vs. post-slim):

| mode | mean Δ test_r | max \|Δ\| |
|---|---:|---:|
| vargp_direct | 0.0000 | 0.0000 |
| default_gpy (joint) | 0.0000 | 0.0000 |
| default_gpy_alt | −0.0013 | 0.0208 |

`vargp_direct` and joint `default_gpy` are **bit-identical** across old/new sweeps (to 4 decimals), confirming those code paths weren't touched. `default_gpy_alt` drifts by at most 0.02 on a single seed (cell 8 seed 789: 0.8114 → 0.7906), mean essentially zero. This is expected: the slim closure is a mathematical identity reformulation, but float32 is not associative, so the different reduction order in `_compute_ell_slim` produces slightly different roundoff, which LBFGS line search can amplify into slightly different accept/reject decisions and therefore different trajectories. The mean accuracy profile is preserved (cell 8 mean test_r: 0.817 → 0.813; cell 1 mean: 0.978 → 0.980).

**Artifacts**:

- Pre-slim results preserved at `results_pre_slim_closure.jsonl` (19 lines, same schema as `results.jsonl`). `results.jsonl` now contains the post-slim sweep (19 lines).
- Code changes all in `gpy_training.py`. `default_params.json` is **untouched** — `alternating_fstep: false` remains the default. No behaviour change for users who don't opt in.

### What a "good sweep" still looks like (open work)

This session measured:

- 2 cells (8 and 1) × 3 seeds × 3 modes = 18 runs.
- On 64×64 data with M=50, n_train=500 (matches baseline investigation).

This is enough to confirm the slim closure is correct (tests + bit-identical untouched paths + math-identity accuracy check) and that it delivers a real timing improvement (~40% overhead reduction). **It is not enough to decide whether to flip the default.** That decision needs:

1. **Multi-cell test_r sweep**: 8–12 representative cells × 3 seeds × {`default_gpy` joint, `default_gpy_alt`, `vargp_direct` as reference}. Cells should span the response regime (low firing rate / hard / sparse like cell 8; high firing rate / easy like cell 1; plus a few cells from the middle and from known-hard cells). Budget ≈ 100 runs at 64×64 M=50 n_train=500, ≈ 10–20 min on GPU.
2. **Success criterion**: does `default_gpy_alt` close the test_r gap on hard cells without meaningfully degrading easy cells? Within-seed-noise on easy cells is fine — matching `vargp_direct` on hard cells is the goal.
3. **Secondary**: is there a pathological case where the ~35% outer-iter-count inflation or beta drift (previously flagged) actually hurts? If not, the inflation is cosmetic.

Only after (1)–(3) pass should `default_params.json` be flipped to `alternating_fstep: true`, with the `_comment_alternating_fstep` updated accordingly.

### Updated handoff (supersedes earlier list)

1. ~~Does `alternating_fstep=True` work on all 41 cells?~~ → refined to (5) below.
2. Does cell 10 with ntrain=2000 now work? PORTING_LESSONS 4.4. Re-test after (5).
3. Beta drift during training in `default_gpy_alt`: still observed — see mask-coverage warnings emitted during this session's tests and sweep (beta drifts to ~0.17). Transient; investigate only if (5) surfaces a pathology.
4. LBFGS tolerance for default_gpy: still PyTorch default (1e-9). Low priority.
5. **NEW**: Run the multi-cell test_r sweep described above. Only after passing, flip `default_params.json`'s `alternating_fstep` to `true` and update its comment.
6. **NEW**: Pre-existing `test_13_default_gpy_elbo_es` failure — file as separate issue. The joint LBFGS on 108×108, n_iterations=12, seed 42, M=50 diverges at iter 1 with "all LBFGS evaluations rejected". Unrelated to this investigation but blocks a full green test suite.
7. **NEW (optional)**: The remaining ~1.9× alt-mode overhead is mostly the structural doubling (two LBFGS passes). If that becomes a bottleneck at full-sweep scale, possible follow-ups: reduce `gpy_lbfgs_max_iter` for the likelihood optimizer specifically (A/λ₀ LBFGS converges in fewer iters than the model LBFGS); or replace the likelihood LBFGS with a damped Newton step on A/λ₀ like `vargp_direct`'s F-step. Not urgent.

---

### 2026-04-20 sweep results: alt_fstep fails the multi-cell test

Ran the sweep flagged above: 8 cells `{0, 1, 3, 5, 8, 10, 15, 22}` × 3 seeds `{42, 123, 789}` × 3 modes `{vargp_direct, default_gpy, default_gpy_alt}` at M=50, n_train=500, 64×64, n_iterations=50. 72 runs total, all successful. Data in `results.jsonl`. Analysis script: `analyze_8cell_sweep.py`.

**Headline: alt_fstep does not generalize.**

Aggregate test_r (pooled across 8 cells × 3 seeds):

| mode | mean test_r | std |
|---|---:|---:|
| default_gpy (joint) | 0.6698 | 0.2307 |
| default_gpy_alt | **0.3912** | 0.4186 |
| vargp_direct | 0.6678 | 0.2350 |

Per-cell mean test_r:

| cell | default_gpy | default_gpy_alt | vargp_direct |
|---:|---:|---:|---:|
| 0 | 0.325 | **0.073** | 0.297 |
| 1 | 0.981 | 0.980 | 0.968 |
| 3 | 0.911 | 0.860 | 0.809 |
| 5 | 0.564 | **0.083** | 0.578 |
| 8 | 0.725 | 0.812 | 0.818 |
| 10 | 0.814 | **−0.081** | 0.845 |
| 15 | 0.446 | **0.170** | 0.458 |
| 22 | 0.591 | **0.232** | 0.570 |

**alt_fstep fails on cells 0, 5, 10, 15, 22** (test_r drops by 0.2–0.9 vs. the other modes). Cell 10 goes to negative test_r.

**A statistics** (ideal range ~0.005–0.05, the regime vargp_direct occupies):

| mode | A mean | A range | A exploded (>0.1) | A frozen (<0.007) |
|---|---:|---|---:|---:|
| default_gpy (joint) | 0.186 | [0.005, 0.589] | **15/24** | 2/24 |
| default_gpy_alt | 0.007 | [0.000, 0.021] | 0 | **14/24** |
| vargp_direct | 0.026 | [0.001, 0.059] | 0 | 1/24 |

alt_fstep swaps joint's A-explosion pathology for an A-freeze pathology. When A is isolated from the model step and starts near init (0.01), the LBFGS on A alone cannot escape — μ_f / σ²_f were computed under the current small A, so the A-gradient is weak and self-reinforcing. Same structural trap as the "stuck-near-init" active-loop pathology flagged in `CLAUDE.md` Known Issues.

**Paired gap (vargp_direct − other) on this 8-cell × 3-seed sample:**

- vs. default_gpy (joint): mean Δ = −0.002, std = 0.071, range [−0.149, +0.176]. **No systematic gap in this config** — cell 3 seed 123 default_gpy wins by 0.15; cell 8 seed 42 default_gpy loses by 0.18.
- vs. default_gpy_alt: mean Δ = +0.277, std = 0.349, range [−0.124, +0.984]. **alt_fstep is dramatically worse on average.**

### Attempt to compare to historical massive-sweep data — invalidated by code churn

Tried to cross-check against `experiments/2026-03-24_massive_allcells_64/results.jsonl` (492 rows, both modes paired, 41 cells × 2 M values × 3 seeds, n_train=2910). That data shows +0.053 mean gap in favor of vargp_direct at M=300. **However, that experiment ran against commit `a0787f1` (2026-03-21, git_dirty=true), and many structural training-code changes have landed since then**, including:

- `2a2cac3` / `cf1f3dc`: sigma_0 parameterization changed twice (exp ↔ identity)
- `058537c`: Fix LBFGS Hessian corruption from frozen parameters in M-step (correctness fix for vargp_direct)
- `d771c90`: Fix autograd memory leak, detach variational params, no_grad E-step (changes vargp_direct E-step gradient flow)
- `63b3333`: DEFAULT CHANGE: inducing point selection pivoted → random
- `82436c6` / `64795d7` / `8ed6f84` / `d6e4c30`: ES machinery rewritten, metric changed to ELBO
- `eaa072e`: Tighten beta upper bound 1.0 → 0.3
- `e5f688e`: Raise n_mstep default 10 → 20
- `6bf20eb`: LBFGS tolerance_change 1e-9 → 1e-3
- `af16cef`: Replace stability_threshold with f_mean_max/mean_threshold
- plus others

The historical +0.053 gap cannot be cited as evidence about current-code behavior. Re-running that sweep under current code would cost (~1h GPU) and is the only way to answer "what's the gap at M=300, n_train=2910 *now*?"

### Retractions and methodology failure (owned)

- **Retraction**: earlier messages in this session's conversation characterized M=50 as a "degenerate" or "unstable" regime to explain why our 8-cell result disagreed with the historical data. That framing is not supported — no one has mapped the (M, n_train) landscape for default_gpy, so I have no basis to label one point as degenerate and another as canonical. The disagreement between the 8-cell result and the historical data is a real unknown, not a known regime artifact.
- **Methodology failure**: the whole 2-cell → 8-cell → lookup-historical-data chain was reactive. The original Phase 1 chose (M=50, n_train=500) for fast iteration, not for mapping the mode-comparison question, and this config was carried forward uncritically into run_baseline.py and then into the 8-cell sweep. A principled investigation would start by asking: *what operating points do we care about for this comparison, and what sample size do we need to detect a gap of size X with confidence Y?* We did not do that. Documenting as a lesson.

### Current position (as of 2026-04-20 end of session)

1. **The original Phase 1 "0.02–0.18 gap on cell 8" was an artifact of 2-cell sampling.** Expanding to 8 cells at the same config shows paired mean Δ = −0.002 (indistinguishable from zero).
2. **alt_fstep does not generalize at (M=50, n_train=500)** — fails on 5/8 cells via A-freeze. Rejected as a "fix" for the gap.
3. **We do not have valid current-code data to say whether a gap exists at (M=300, n_train=2910)** or any other config. The historical massive sweep is invalidated by code churn.
4. **The slim-closure optimization is orthogonal and stands on its own** — it's a cost reduction for anyone who opts into alt_fstep, independent of whether alt_fstep is ever useful. Code infrastructure, not a method claim.
5. `default_params.json` unchanged: `alternating_fstep: false`. Correct outcome given (2).

### Recommended next steps (supersedes all earlier handoff lists)

Before running any more sweeps, **align on methodology first**:

A. **Decide which operating point(s) matter** for this comparison. Candidates: the investigation's (M=50, n_train=500); the M-sweep recommended (M=200–300, n_train=2910); the paper-gap canonical (M=300, n_train=2910); or mapping (M, n_train) jointly. Each has different cost and different scientific value.

B. **Pre-register success / failure criteria.** What size of gap would change a decision? What's our tolerance for per-seed variance vs. per-cell structure? Do we care about the mean, or about tail behavior (cells where one mode fails hard)?

C. **Budget a single decisive sweep** instead of incremental 2→8→… sampling. At 41 cells × 3 seeds × 2 modes × 50 iters, budget ≈ 1h GPU for one operating point.

D. **Leave the investigation open.** The question "is there a gap between default_gpy and vargp_direct under current code" has no valid answer yet. The one thing this session did settle is that alt_fstep is not the fix, regardless of what the gap turns out to be.
