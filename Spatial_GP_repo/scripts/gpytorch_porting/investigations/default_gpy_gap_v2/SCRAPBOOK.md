# default_gpy_gap_v2 — SCRAPBOOK

## 1. Charter

Answer the single question:

> Under current code, is there a systematic gap in test_r between
> `default_gpy` (joint LBFGS) and `vargp_direct` at operating point
> (M=300, n_train=1500, 64×64, arc_cosine, ground-truth RF init, ELBO ES on)?

This investigation is **not**: a fix for `default_gpy`, a re-validation of
`alternating_fstep`, or a tour of other operating points. Scope was locked
in `PROMPT.md` before data collection.

## 2. Relation to the v1 investigation

v1 (`investigations/default_gpy_gap/`) reached wrong conclusions from an
underpowered sample (2 cells × 3 seeds at M=50, n_train=500) and proposed
`alternating_fstep=True` as a "fix." A follow-up 8-cell sweep disproved
that. v2 exists to replace the missing evidence with a larger, pre-registered
sample at a realistic operating point.

Mistakes from v1 this v2 investigation is designed to avoid (taken from
`PROMPT.md` § "Lessons from the previous investigation"):

1. Do not declare a finding from a 2-cell sample.
2. Do not iterate 2 → 8 → 15 → 41 cells reactively.
3. Do not label operating points "degenerate" or "canonical" without data.
4. Do not conflate "fix works on cell 8" with "fix generalizes."
5. Do not cite historical data without checking code churn.
6. Always carry over a paired-comparison frame.
7. `A = exp(raw_A)` — read `likelihood.raw_A.grad`; read `likelihood.A.item()`.
8. GPU wall-clock is session-noisy — compare within session.

## 3. Pre-registration (written before the sweep ran)

### 3.1 Decision criterion (locked)

```
Δ(cell, seed) = test_r_{vargp_direct}(cell, seed) − test_r_{default_gpy}(cell, seed)
mean_Δ = mean over (cell × seed) of Δ

|mean_Δ| > 0.02  → GAP EXISTS.
|mean_Δ| ≤ 0.02  → NO GAP DETECTED.
```

### 3.2 Experimental design (locked)

| param | value |
|---|---|
| dataset | `datasets/PNAS_64x64_center_crop_no_renorm.npz` |
| M | 300 |
| n_train | 1500 |
| cells | 15 stratified by firing rate (5 low + 5 mid + 5 high) + 1 user-added (cell 8) = 16 |
| seeds | {42, 123, 789} for vargp_direct & default_gpy; {42} for default_gpy_alt |
| modes | vargp_direct, default_gpy (joint), default_gpy_alt (alternating_fstep=True) |
| n_iterations | 50 |
| early stopping | on, ELBO, default patience/min_delta |
| ip_selection | 'random' |
| kernel | arc_cosine |
| rf_init | 'ground_truth' |
| dtype | float32 |
| device | cuda |
| all other params | `build_config_from_defaults()` defaults |

**Run count**: 48 vargp_direct + 48 default_gpy + 16 default_gpy_alt = **112 runs**.

## 4. Cell selection

Stratified by firing rate on the 3160-image pool
(`responses_train` + `responses_val`), split into 3 buckets
(low: rank 0..13, mid: 14..27, high: 28..40), picking cells at bucket-local
positions [1, 3, 6, 9, 12]. Cell 8 added by user request (not in
stratified set; the v1 focus cell).

See `cells_used.json` and `select_cells.py`. Chosen cells
(in the order they appear in `cells_used.json`):

```
low:   40, 30,  4,  7, 23          fr ∈ [0.127, 0.449]
mid:   25, 19, 21, 32, 20          fr ∈ [0.575, 0.964]
high:  10, 24, 38, 29, 16          fr ∈ [1.059, 1.958]
user:   8                           fr = 0.982 (rank 28, lowest-high bucket)
```

## 5. Sweep execution

- Git commit at launch: `d27c4eb` — "Open default_gpy_gap_v2 investigation:
  pre-registered charter."
- Working tree at launch: only the three investigation files
  (`select_cells.py`, `run_sweep.py`, `cells_used.json`) untracked.
  No uncommitted changes to training code.
- Smoke test before launch: `vargp_direct, cell=1, seed=42, M=300, n_train=1500`
  → test_r=0.9748 in 11.2 s.
- Unit test before launch: `tests/test_gpy_alternating_fstep.py` — 3 passed.
- Wall time: 14.6 min, 112/112 runs completed, 0 failures.
- Launch command: `nohup python investigations/default_gpy_gap_v2/run_sweep.py`
- Outputs: `results.jsonl` (112 lines) + `sweep_metadata.json` + `sweep.log`.

## 6. Results

### 6.1 Per-cell test_r (3-seed mean per cell, both main modes)

| cell | bucket | fr | vargp_direct μ | default_gpy μ | Δ |
|---|---|---|---|---|---|
| 40 | low | 0.13 | 0.8612 | 0.3594 | **+0.5018** |
| 30 | low | 0.20 | 0.7796 | 0.7886 | −0.0090 |
| 4  | low | 0.35 | 0.7040 | 0.5797 | +0.1243 |
| 7  | low | 0.40 | 0.6196 | 0.6604 | −0.0409 |
| 23 | low | 0.45 | 0.7228 | 0.6557 | +0.0671 |
| 25 | mid | 0.57 | 0.8232 | 0.8248 | −0.0016 |
| 19 | mid | 0.72 | 0.6810 | 0.6443 | +0.0367 |
| 21 | mid | 0.86 | 0.6438 | 0.7225 | −0.0787 |
| 32 | mid | 0.92 | 0.7921 | 0.7652 | +0.0270 |
| 20 | mid | 0.96 | 0.8293 | 0.8521 | −0.0229 |
| 10 | high | 1.06 | 0.8769 | 0.7637 | +0.1132 |
| 24 | high | 1.07 | 0.8966 | 0.8739 | +0.0227 |
| 38 | high | 1.27 | 0.6943 | 0.2798 | **+0.4145** |
| 29 | high | 1.64 | 0.7383 | 0.7540 | −0.0157 |
| 16 | high | 1.96 | 0.9579 | 0.9198 | +0.0381 |
| 8  | high (user) | 0.98 | 0.8751 | 0.8004 | +0.0748 |

### 6.2 Paired-Δ summary (n=48)

- **mean_Δ = +0.0782**, std=0.1667, range [−0.180, +0.510]
- IQR [q25, median, q75] = [−0.015, **+0.017**, +0.089]
- Sign counts: Δ>0 in 29/48, Δ<0 in 19/48
- Per-cell (3-seed mean): 10/16 favor vargp_direct (>+0.02), 3/16 within band,
  3/16 favor default_gpy (<−0.02).

### 6.3 Decision

```
GAP EXISTS at this config.  |mean_Δ| = 0.0782 > 0.02.
mean_Δ = +0.0782 favors vargp_direct.
```

**Important nuance**: the mean is heavy-tailed. Median Δ = +0.017 is inside the
band. The mean is lifted by two cells (40 and 38) whose 3-seed mean Δ exceeds
+0.40. Dropping those two cells and recomputing the mean over the remaining 14
cells gives paired-mean Δ ≈ +0.02 — at the threshold, not comfortably above it.
The "gap" at this config is primarily a **two-cell phenomenon**, not a uniform
per-cell effect.

### 6.4 Final-A distribution

| mode | n | min | p25 | median | p75 | max | ≤0.012 |
|---|---|---|---|---|---|---|---|
| vargp_direct    | 48 | 0.0143 | 0.0175 | 0.0215 | 0.0271 | 0.0414 | 0 |
| default_gpy     | 48 | 0.0042 | 0.0920 | 0.1283 | 0.1727 | 0.7802 | 3 |
| default_gpy_alt | 16 | 7e-8   | 0.0016 | 0.0050 | 0.0090 | 0.0193 | 13 |

`default_gpy` settles at a **much higher and wider** A than `vargp_direct`.
3 runs freeze near init. `default_gpy_alt` freezes near/below init on
13/16 cells.

### 6.5 Timing

| mode | n | wall_s mean ± std | s/iter mean ± std |
|---|---|---|---|
| vargp_direct    | 48 | 7.22 ± 2.44 | 0.156 ± 0.050 |
| default_gpy     | 48 | 8.18 ± 4.31 | 0.236 ± 0.075 |
| default_gpy_alt | 16 | 8.64 ± 9.93 | 0.258 ± 0.161 |

## 7. default_gpy_alt sanity check

16 cells, seed 42. Mean Δ(alt − joint) = **−0.338**. 10/16 cells show
catastrophic failure (test_r drops by 0.1–1.0 vs joint). 14/16 cells have
|Δ|>0.02. A freezes below 0.01 on 13/16 cells.

This reproduces the v1 follow-up finding that `alternating_fstep=True` is
not a viable fix — it replaces one failure mode with a worse one.

## 8. Conclusions

Data-only conclusions:

- At (M=300, n_train=1500, 64×64), `|mean_Δ| = 0.078` exceeds the pre-registered
  0.02 threshold. Per charter: **GAP EXISTS.**
- The gap is heavy-tailed, driven mostly by two cells (40 and 38) with
  3-seed mean Δ > 0.4. On 11 of 16 cells the 3-seed mean |Δ| is below 0.08.
- `default_gpy`'s final A distribution is 6× higher and much wider than
  `vargp_direct`'s.
- `default_gpy_alt` catastrophically fails on 10/16 cells. Not a usable fallback.

No proposed fixes. No mechanism speculation in this section.

## 9. Lessons carried forward

Adhered to (from v1):
1. Budgeted a decisive sample (16 cells × 3 seeds = 48 paired Δ) up front.
2. Ran the sample once; did not iterate reactively.
3. Used a pre-registered paired-comparison frame.
4. Did not re-investigate `alternating_fstep` beyond the 15-run sanity check.
5. Did not declare a fix after the sweep.
6. Reported the full distribution, not just the mean.

Added lessons:
- **Heavy-tailed distributions deserve median + IQR reporting, not just mean**.
  If we had only reported `mean_Δ = +0.078` without showing the per-cell
  breakdown, we would have overstated the effect size for the typical cell.

---

# Phase 2 — Outlier investigation (cells 40 and 38)

Scope shift approved by user 2026-04-21: narrow the investigation to
understanding why cells 40 and 38 fail so badly for `default_gpy` when other
cells are either competitive or show modest differences. Working hypothesis
(to be tested, not assumed): if we find and fix what's going wrong on these
two cells, the aggregate gap may close substantially.

## 10. Phase 2 charter

- Investigate cells 40 and 38 only. Not all 16 cells, not new operating points.
- Use existing sweep data (train_loss_curve, A_curve, final params) first.
  Only run new experiments after the existing data has been exhausted and
  a specific hypothesis has been formed.
- Think in terms of **minimum viable diagnostic**: what is the cheapest
  measurement that discriminates between two competing hypotheses?
- Proceed step by step. After each diagnostic, decide: do we have enough to
  form a testable hypothesis, or do we need more data?
- Apply the v2 lessons: don't chase a shallow clue into a full re-sweep.

## 11. Phase 2 hypotheses (prior probabilities — to be updated as data arrives)

These are candidate failure modes for `default_gpy` on cells 40 and 38.
No evidence is being weighed yet — these are the hypotheses the diagnostic
steps must discriminate between.

| H | Description | What would confirm it |
|---|---|---|
| H1 | A-parameter explosion: joint LBFGS pushes A far from its vargp_direct value early, lands in a bad optimum. | A_curve shoots up early, stays high; final_A ≫ vargp_direct's; train_loss plateaus at a much worse value. |
| H2 | Kernel-parameter drift (beta/rho/RF center): joint LBFGS drives kernel hyperparameters to pathological values. | final_beta / final_rho on these cells is far outside the population distribution; mask coverage warnings; RF center lands on image edge. |
| H3 | Early stopping fires at a bad intermediate: ES triggers during an A or kernel transient, restoring to a poor best. | stopped_early=True, best_iteration < ~20, with train_loss at best_iteration much worse than at later iterations if allowed to continue. |
| H4 | Whitened variational parameterization is badly conditioned at this operating point for these cells. | Would require a new experiment: swap `default_gpy` to unwhitened parameterization and see if the gap on cells 40/38 narrows. (Last-resort test.) |
| H5 | Cell-specific data pathology (e.g., extremely sparse or saturated responses) that only the alternating E/M structure can handle. | Response-distribution summary of cells 40/38 is qualitatively different from the other 14 cells. |

## 12. Phase 2 plan

Step 0 (FREE — existing data):
  Diagnostic dashboard: for cells 40, 38, and two control cells (one healthy
  low-bucket and one healthy high-bucket), plot/compare:
  - train_loss_curve
  - A_curve
  - final values for A, lambda0, beta, rho, eps_0x, eps_0y
  - n_iterations_run, stopped_early, best_iteration
  Compare `vargp_direct` (seed 42) vs `default_gpy` (seed 42).

Step 1 (CHEAP — only if Step 0 suggests a specific hypothesis):
  Targeted single-run experiment (1-2 runs) to test the hypothesis.
  Example: if H1 (A explosion) looks supported, run `default_gpy` on cell 40
  with `A_init` much lower, or with an A bound, and see whether the gap
  closes.

Step 2 (MODERATE — only if Step 1 confirms the hypothesis):
  Mini-sweep: the targeted fix × cells 40 and 38 × 3 seeds. 6 runs max.
  This is the "would-this-generalize-beyond-one-cell" check.

No Step 3. If the hypothesis holds through Step 2, the finding is reported.
If it does not, we return to Step 0 with updated priors.

## 13. Phase 2 findings (log, appended in order)

### Finding P2-1 (2026-04-21) — Phase 0 diagnostic complete, two distinct failure modes identified

Data source: existing `results.jsonl`. Focus: cells 40 (outlier), 38 (outlier),
30 (low-bucket control), 16 (high-bucket control), seed 42. Full per-iteration
`train_loss_curve` and `A_curve` inspected.

**Cell 40, default_gpy** — A-explosion + β-collapse basin:

```
iter  1: loss=574, A=0.008
iter  2: loss=510, A=0.028    (3x)
iter  3: loss=473, A=0.23     (8x in one step)
iter  4: loss=466, A=0.38
iter  5: loss=465, A=0.41     (oscillates 0.37–0.46 for many iters)
iter 14: best (loss=459)
final: A=0.566, beta=0.011 (10x narrower than vargp_direct), test_r=0.37
```

**Cell 38, default_gpy** — early LBFGS freeze at bad optimum:

```
iter 1: loss=1328, A=0.005
iter 2: loss=1200, A=0.033
iter 3: loss=1116, A=0.043
iter 4: loss=1055, A=0.072
iter 5: loss=1030.78, A=0.063   <- BEST
iter 6-20: loss=1030.78, A=0.063 (IDENTICAL; LBFGS stopped moving)
final: test_r=0.31
```

**Cell 16 (control), default_gpy** — same structural freeze pattern as cell 38:

```
iter 7: loss=-745.19, A=0.036 <- BEST
iter 8-21: loss=-745.19, A=0.036 (IDENTICAL for 14 iters)
final: test_r=0.88
```

**Contrast: vargp_direct cell 40** (smooth, monotonic):

```
iter  1: loss=753, A=0.003
iter  3: loss=469, A=0.017
iter 10: loss=394, A=0.031
iter 49: loss=387, A=0.041
```

**Three observations**:

1. LBFGS freeze-after-plateau is the NORM for default_gpy, not pathology.
   It happens on healthy cells (16, 30) as well as failing ones (38). The
   pattern: LBFGS converges within a few outer iterations, returns
   unchanged parameters, outer loop sees no progress, ES fires. What
   matters is whether this happens at a *good* plateau.

2. Cell 40 is a distinct failure mode: A-explosion (0.008 -> 0.23 in a
   single outer iteration, step 2 -> 3) followed by oscillation between
   A=0.37 and A=0.46. Final beta = 0.011 (10x narrower RF than
   vargp_direct). Model has collapsed the RF to compensate for large A.

3. Cell 38 is the "frozen at bad optimum" mode. ELBO peak is reached at
   iter 5 and never improves; cf. vargp_direct at iter 5 is already
   well below default_gpy's asymptote.

**Working hypothesis for Step 1**: both failure modes are basin-of-attraction
problems. The joint-LBFGS optimizer's first 2–4 outer steps commit to a
basin, and on cells 40 and 38 that basin corresponds to much higher ELBO
than what vargp_direct finds. This is testable via warm-start from
vargp_direct's final params.

**Alternative hypotheses not yet ruled out**:
- ES premature termination (tested in Step 1 Run A)
- Something cell-specific about the response distribution of 40 and 38
  (H5 from section 11, not yet investigated)

### Finding P2-2 (2026-04-21) — Step 1 executed, ES ruled out; warm-start test inconclusive

Script: `step1_diagnostic.py`. Output: `step1_results.jsonl` (4 records).

**Run A (ES disabled) — DEFINITIVE**:

| cell | test_r | final_loss | A | beta | iters |
|---|---|---|---|---|---|
| 40 | 0.3663 | 459.11 | 0.566 | 0.011 | 50 (full) |
| 38 | 0.3106 | 1030.78 | 0.063 | 0.063 | 50 (full) |

Identical to ES-on values. default_gpy ran the full 50 iterations and
produced ZERO additional improvement past the ES-restored values. The
plateau is a real local optimum; LBFGS does not escape it given 2x the
iteration budget.

**Conclusion**: ES is not the cause. Hypothesis H3 rejected.

**Run B (warm-start from vargp_direct finals) — INCONCLUSIVE**:

Both cells: `Training diverged at iteration 1: all LBFGS evaluations rejected`.
Zero training iterations completed.

Design flaw: I warm-started (A, lambda0, beta, rho, eps_0x, eps_0y) but
not the variational posterior. default_gpy stores (m, V) in whitened
parameterization and it was reinitialized to zero. The combination "good
kernel + trained likelihood (A=0.04) + untrained whitened variational
posterior" produces a very bad initial ELBO, and LBFGS's line search
rejects every trial step on the first outer iteration.

This does NOT disprove the basin-of-attraction hypothesis. It says only
that partial warm-start (kernel+likelihood, not variational) is not a
viable probe. A proper basin test requires warm-starting the variational
posterior too — nontrivial, because we'd need to convert vargp_direct's
eigenspace (m_b, V_b) back to default_gpy's whitened (m, L) parameterization.

### What we know after Step 1

Confirmed:
- ES fires early on default_gpy but is not the cause of the bad plateau.
- The bad plateau is a real local optimum in joint-LBFGS-over-default_gpy's
  parameterization. It is reached within 5-14 outer iterations and
  defended against 40+ further iterations of LBFGS.
- The optimum differs per cell in character (cell 40: A exploded, beta
  collapsed; cell 38: stuck early at modest A).

Still not established:
- Whether default_gpy's good basin (the one vargp_direct finds) is
  accessible from ANY init.
- Whether specific initialization (beta, or step-size control) would
  redirect default_gpy into the good basin.
- Whether the two failure modes (40 vs 38) have a common root cause or
  are distinct pathologies.

### Finding P2-3 (2026-04-21) — Step 2 complete; beta_init=0.2 dramatically helps outliers

Script: `step2_diagnostic.py`. Output: `step2_results.jsonl` (6 records).
Three interventions on cells 40, 38 at seed 42:

| cell | intervention        | test_r | final_loss | final_A | final_beta | vs baseline default_gpy |
|---|---|---|---|---|---|---|
| 40 | I_maxiter5         | 0.369 | 459 | 0.535 | 0.011 | ~no change |
| 40 | I_lambda0_eq       | 0.358 | 461 | 0.506 | 0.010 | ~no change |
| 40 | **I_beta0p2**      | **0.822** | **380** | 0.081 | 0.100 | **+0.456** |
| 38 | I_maxiter5         | 0.311 | 1030 | 0.063 | 0.063 | no change |
| 38 | I_lambda0_eq       | 0.273 | 1073 | 0.072 | 0.042 | slightly worse |
| 38 | **I_beta0p2**      | **0.580** | 927 | 0.051 | 0.093 | **+0.269** |

Baselines for reference (seed 42):
- vargp_direct cell=40: test_r=0.855, loss=387, A=0.041, beta=0.112.
- default_gpy  cell=40: test_r=0.366, loss=459, A=0.566, beta=0.011.
- vargp_direct cell=38: test_r=0.778, loss=661, A=0.032, beta=0.129.
- default_gpy  cell=38: test_r=0.311, loss=1030, A=0.063, beta=0.063.

**Observations (single seed, not yet confirmed)**:
- The beta=0.2 override produced a large test_r lift on both outlier cells
  at seed 42: cell 40 0.366 -> 0.822 (near vargp_direct's 0.855), cell 38
  0.311 -> 0.580 (vargp_direct 0.778).
- Cell 40: final beta stopped at 0.10 instead of collapsing to 0.011. Final
  loss (380) is slightly below vargp_direct's (387). No bound violations.
- Cell 38: final beta ended at 0.09 (vargp: 0.129). ELBO plateau still
  worse than vargp (927 vs 661), but test_r is substantially better than
  baseline. Cell 38 never collapsed beta to begin with, so the mechanism
  behind its improvement is less clear than for cell 40.
- `gpy_lbfgs_max_iter=5` and `lambda0_init=log(fr_mean)` produced no
  meaningful improvement (and the latter was slightly worse).

**Interpretation is premature** at one seed on two cells. These are strong
per-cell signals, not yet evidence of a general knob. Possible
confounds / alternate framings: (i) beta=0.2 may land near a different
local minimum that happens to be good here but poor elsewhere; (ii) the
effect may be seed-sensitive; (iii) it may harm currently-healthy cells.
Step 3 tests (iii) at a single seed before proposing more work.

### Step 3 (2026-04-21) — broad beta_init=0.2 check on all 16 cells — DESIGNED, AWAITING USER DECISION

**Motivation**: Step 2 produced a strong per-cell signal for beta_init=0.2
on 2 cells at 1 seed. Before interpreting that as a candidate fix, we'd
want to see whether it HELPS or HARMS on the 14 currently-healthy cells.
A knob that fixes outliers but breaks healthy cells is not useful in
aggregate.

**Design (not run)**: 16 cells x 1 seed (42) x beta_init=0.2. 16 runs, ~1 min.
Compute paired delta vs baseline default_gpy at seed 42 and vs vargp_direct
at seed 42.

**Known confounds**:
- Single seed — one-run-per-cell. Even if it looks clean, a multi-seed
  follow-up is needed for any confirmatory claim.
- beta=0.2 is within BETA_MAX=0.3, so no bound violations expected.
- "Works at seed 42" is not "works in general".

**Not launched pending user decision** on whether this is the right next
step or whether a smaller / different test would be more informative.

### Finding P2-4 (2026-04-21) — Step 3 executed. beta_init=0.2 is NOT a universal fix

Script: `step3_beta_sweep.py`. Output: `step3_results.jsonl` (16 records).

Per-cell table (seed 42):

| cell | tr(beta=0.2) | tr(baseline_gpy) | tr(vargp_direct) | Δ vs gpy | Δ vs vargp | final_beta |
|---|---|---|---|---|---|---|
| 40 | **0.8224** | 0.3663 | 0.8553 | **+0.456** | −0.033 | 0.100 |
| 30 | 0.7973 | 0.7918 | 0.7469 | +0.005 | +0.050 | 0.021 |
| 4  | 0.6177 | 0.6315 | 0.7418 | −0.014 | −0.124 | 0.031 |
| 7  | 0.6392 | 0.6355 | 0.6350 | +0.004 | +0.004 | 0.028 |
| 23 | 0.7056 | 0.6849 | 0.7630 | +0.021 | −0.057 | 0.034 |
| 25 | 0.8111 | 0.7908 | 0.8442 | +0.020 | −0.033 | 0.031 |
| 19 | 0.5656 | 0.5661 | 0.7447 | −0.001 | −0.179 | 0.054 |
| 21 | 0.7062 | 0.6944 | 0.6854 | +0.012 | +0.021 | 0.046 |
| 32 | 0.7696 | 0.7743 | 0.7835 | −0.005 | −0.014 | 0.060 |
| 20 | 0.8290 | 0.8599 | 0.8197 | −0.031 | +0.009 | 0.044 |
| 10 | 0.8516 | 0.8510 | 0.8627 | +0.001 | −0.011 | 0.061 |
| 24 | **0.9250** | 0.7993 | 0.8840 | **+0.126** | +0.041 | 0.027 |
| 38 | **0.5795** | 0.3106 | 0.7776 | **+0.269** | −0.198 | 0.093 |
| 29 | **0.5183** | 0.7527 | 0.7287 | **−0.234** | −0.210 | 0.031 |
| 16 | 0.8766 | 0.8794 | 0.9581 | −0.003 | −0.082 | 0.028 |
| 8  | **0.6948** | 0.8142 | 0.8551 | **−0.119** | −0.160 | 0.041 |

Aggregate (seed 42 only, 16 cells):

- **Mean Δ vs baseline default_gpy = +0.032** (modest improvement)
- **Mean Δ vs vargp_direct = −0.061** (gap still exists, smaller than baseline's −0.093)
- 5 cells improved (Δ > +0.02); **3 cells harmed** (Δ < −0.02), including cell 29 by −0.23 and cell 8 by −0.12.

**Observations (single seed, tentative)**:

- beta_init=0.2 is NOT a universal fix. It trades failures for different
  failures.
- Cell 40 and cell 38 (the original outliers) substantially recover.
- Cell 29 becomes a new outlier (0.75 -> 0.52). Cell 8 moderately worsens.
- Even with beta_init=0.2, beta still collapses on most cells (final beta
  in [0.02, 0.06] on 11 of 16 cells). The effect of beta_init is not
  "prevent collapse" but "change which basin the run lands in."
- The cells where final_beta stays high (0.08-0.10) are cells 40 and 38 —
  exactly the ones the intervention was designed for. Ironic but
  coherent: forcing beta_init further from their failure basin is enough
  to keep them out.

**What this implies about mechanism (tentative)**:

- default_gpy's joint-LBFGS optimization appears highly init-sensitive in
  a basin-of-attraction sense: different beta_init -> different basin
  assignments -> different cell-level outcomes.
- There is no obvious reason to expect a single universal beta_init to be
  optimal; the landscape varies per cell.
- This finding is informational about the mechanism of failure but is NOT
  a ready fix. To turn it into a fix we would need a cell-adaptive strategy,
  which risks overfitting to this dataset.

**What we should NOT do next**:

- Sweep beta_init over {0.15, 0.20, 0.25, 0.30, ...} looking for a value
  that happens to work on all 16 cells at seed 42. That is test-set
  overfitting. beta_init=0.2 was a priori motivated (from cell 40's
  collapse), which is legitimate; iterating on test_r is not.
- Declare "root cause found" based on 2 cells recovering when 3 others
  are harmed.

**Possible legitimate next moves (to discuss with user)**:

- Stop the investigation here with an honest "default_gpy is
  init-sensitive; single-knob fixes transfer failures rather than
  eliminate them." This closes the charter cleanly.
- Seed-replicate the beta_init=0.2 finding: run all 16 cells at seeds
  123, 789 too, to see whether the per-cell winners/losers are consistent
  or noise. (32 more runs.)
- Investigate cell 29 specifically: why does beta_init=0.2 hurt it so
  badly? Same diagnostic as cells 40/38, but with the sign of interest
  reversed. Could reveal a symmetric picture of the landscape.
- Different angle entirely: test whether the failure cells under both
  beta_init settings would be fixed by an alternating optimization scheme
  that ISN'T the failed alternating_fstep (e.g., alternate variational
  and hyperparameter steps separately; requires code).

### Finding P2-5 (2026-04-21) — Step 4 complete; 3-seed picture substantially MORE favorable than Step 3

Steps combined: `step3_results.jsonl` (seed 42) + `step4_results.jsonl`
(seeds 123, 789) = 48 runs at beta_init=0.2, paired against the main
sweep's 48 default_gpy baselines and 48 vargp_direct values.
Analysis script: `analyze_phase2.py`.

**Aggregate (3 seeds, 48 paired obs)**:

- Mean Δ(beta=0.2 − baseline default_gpy) = **+0.049**  (n=48, std 0.148)
- Mean Δ(beta=0.2 − vargp_direct)         = **−0.029**  (n=48, std 0.080)
- Baseline gap (vargp − baseline default_gpy) = **+0.078**  (from §6.2)
- Gap reduction under beta=0.2 intervention: **0.078 → 0.029 (−63%)**
- Pre-registered threshold (|Δ| > 0.02 = GAP): still GAP, but marginally.

**Per-cell consistency across seeds (signs of per-seed Δ vs baseline gpy,
each sign: + means >+0.02, − means <−0.02, 0 means within band)**:

| Category | Cells |
|---|---|
| Robust winners (`+++`) | **40 (+0.474), 25 (+0.053), 38 (+0.293)** |
| Robust losers (`---`) | **NONE** |
| Seed-specific losses (`-00`) | cell 29, cell 8 — only seed-42 regresses, other seeds ≈ neutral |

Cell 40 and cell 38 — the original outliers — recover across all 3 seeds,
not just seed 42. Cell 25 (mid bucket) also improves robustly by a small
amount.

**Cell 29 and 8 single-seed regressions**:

- Cell 29 at seed 42: baseline tr=0.75 (49 iters, smooth), beta=0.2
  tr=0.52 (21 iters, LBFGS frozen at iter 6 — structurally IDENTICAL to
  cell 38's baseline failure mode).
- Cell 29 at seeds 123, 789: both inits land in essentially the same
  good basin.
- Cell 8: seed-42 regression modest (−0.12); other seeds neutral.

So the "new failures" introduced by beta=0.2 are **2 seed-specific
incidents** (cell 29 + cell 8, both at seed 42), whereas the failures
**eliminated** are **6 systematic incidents** (cells 40 and 38 across all
3 seeds). Net trade is favorable.

**Interpretation (tentative, consistent with data)**:

- default_gpy's joint-LBFGS optimization has multiple basins. Beta_init
  determines which basin a (cell, seed) combination lands in.
- At beta_init=0.1, cells 40 and 38 always fall into bad basins.
- At beta_init=0.2, cells 40 and 38 robustly land in good basins, and
  the previously-good cells remain good with minor seed-dependent variation.
- No evidence that any single beta_init eliminates all failures — the
  bad basins exist regardless; init just chooses who lands there.

**Cell 29 mechanism** (from Phase 0-style trajectory comparison, seed 42):

- Baseline (beta_init=0.1) trajectory: loss 346 → 220 → 87 over 35+
  iters. A grows from 0.009 to 0.18. Normal convergence.
- beta=0.2 trajectory: loss 427 → 279 → 254 → 234 → 228 → 226 → FROZEN
  at 226 for 15 iters until ES. A stops at 0.035, beta ends at 0.031
  (same region as baseline's 0.036, but at a much worse loss).
- The beta=0.2 seed-42 trajectory on cell 29 exhibits exactly the "LBFGS
  freezes at a bad plateau" mode that cell 38 shows at baseline. beta
  does collapse from 0.2 → 0.03 before freezing (so the collapse doesn't
  protect cell 29 the way it does on cell 40).
- This is consistent with the basin hypothesis: one basin is the
  "LBFGS-frozen-at-bad-plateau" basin; it's accessed by different inits
  for different cells.

**Summary conclusion of Phase 2 (tentative)**:

- beta_init=0.2 is a real effect: it closes roughly two thirds of the
  aggregate gap and robustly recovers both pre-identified outliers.
- It does not eliminate the gap. The "LBFGS-freezes-early" failure mode
  is a structural feature of default_gpy's optimization, not a removable
  pathology. Init choice determines which cells hit it.
- This finding should NOT be treated as a production fix (test-set
  evidence with 48 paired obs, single beta value motivated a priori).
  It's a mechanism story.
- Option 4 from §13 (alternating variational-then-hyperparameter schedule)
  remains the principled path to a durable fix. Deferred to a future
  session.

### Phase 2 — artifacts / cleanup inventory

Files written during Phase 2 (all under
`investigations/default_gpy_gap_v2/`):

Kept (primary data / scripts):
- `diagnose_outliers_phase0.py`, `diagnose_outliers_phase0.png`
- `step1_diagnostic.py`, `step1_results.jsonl`
- `step2_diagnostic.py`, `step2_results.jsonl`
- `step3_beta_sweep.py`, `step3_results.jsonl`
- `step4_beta_sweep_moreseeds.py`, `step4_results.jsonl`, `step4.log`
- `analyze_phase2.py`

Nothing written outside the investigation folder. `default_params.json`,
training code, tests — untouched. Can be deleted if the folder is archived.

---

# Phase 3 — NGD investigation

Scope shift approved by user 2026-04-22: test whether GPyTorch's built-in
Natural Gradient Descent (NGD) — the canonical solution in the GPyTorch
docs for non-conjugate SVGP — closes the Phase 2 gap between `default_gpy`
(joint LBFGS) and `vargp_direct` (eigenspace EM).

## 14. Phase 3 charter

Answer a **single yes/no question**:

> Under the Phase 2 config (M=300, n_train=1500, 64x64, arc_cosine,
> ground-truth RF, seeds {42, 123, 789}), does training a standard
> GPyTorch SVGP via `TrilNaturalVariationalDistribution` + `gpytorch.optim.NGD`
> for the variational params (and Adam for the hyperparams) close the
> |mean_Δ test_r| = 0.078 gap we observed with joint LBFGS?

Scope:

- Self-contained NGD code in `investigations/default_gpy_gap_v2/ngd/`.
  No edits to production code (`gpy_model.py`, `gpy_training.py`,
  `kernels.py`, `likelihoods.py`, `default_params.json`).
- Same 16 cells × 3 seeds as Phase 2 (48 paired obs) so we can compute
  paired Δ against the existing Phase 2 data without re-running vargp
  or default_gpy.
- Pre-registered decision: `|mean_Δ(ngd − vargp_direct)| > 0.02` = GAP.
  Same threshold as Phase 2.

Hyperparameters (all hardcoded, all flagged per bewary.md — user granted
full autonomy 2026-04-22 and this block is the permanent audit trail):

| # | choice | value | rationale |
|---|---|---|---|
| 1 | variational distribution | `TrilNaturalVariationalDistribution` | stability in float32 + Poisson (GPyTorch docs) |
| 2 | NGD lr | 0.1 | GPyTorch tutorial default |
| 3 | Adam lr | 0.01 | GPyTorch tutorial default for hyperparams |
| 4 | n_iterations | 1000 | first-order NGD+Adam needs ~1000 steps to match LBFGS's ~1000 grad evals |
| 5 | early stopping | OFF | ELBO monotone-decreases even when test_r degrades; ES on ELBO can't catch cell 40's overfit (see Finding P3-1 diagnostics) |
| 6 | batching | full-batch | n_train=1500 fits, matches Phase 2 compute model |
| 7 | hyperparam clamps | `kernel.clamp_hyperparameters()` + `likelihood.clamp_params()` after each Adam step | same projected-gradient pattern as `gpy_training.train_gpy_default`; variational natural params are NOT clamped (unsupported by NGD) |
| 8 | kernel/likelihood init | identical to Phase 2 `build_config_from_defaults(mode='default_gpy', ip_selection='random', M=300, n_train=1500, beta=0.1, rf_init='ground_truth')` | apples-to-apples with existing baselines |

Loss formulation: use `gpytorch.mlls.VariationalELBO(likelihood, model, num_data)`.
This is load-bearing — our first attempt used the sum-scaled loss that
`gpy_training.train_gpy_default` computes directly; combined with
`gpytorch.optim.NGD`'s internal `num_data` multiplier, that scales the
NGD step by `num_data²` and diverges within ~5 iters. See
`ngd_training.py` for the scaling note.

Setup reuse: the prototype monkey-patches `gpy_training.train_gpy_default`
to a capture-and-no-op, runs `run_single_config` to do the full Phase 2
setup (data load, seeded IP selection, RF center, kernel/likelihood init),
extracts the built kernel/likelihood/inducing_points, and drops them
into a fresh NGD model. This guarantees data-flow parity with Phase 2
without duplicating ~80 lines of setup code.

## 15. Phase 3 plan

Step 0 (SMOKE): single cell, 20 iters — verify plumbing.
Step 1 (PROTOTYPE): 5 cells × seed 42 × 1000 iters, no ES, test_r probe
  every 25 iters — inspect trajectories, choose iter budget.
Step 2 (FULL SWEEP): 16 cells × 3 seeds × 1000 iters (48 runs),
  test_r probe every 25 iters — answer the headline question.
Step 3 (WRITE-UP): this section.

## 16. Phase 3 findings (log, appended in order)

### Finding P3-1 (2026-04-22) — NGD+Adam step is num_data² too large with sum-scaled loss

Smoke test on cell 16, 20 iters, `ngd_lr=0.1`. Training diverged at iter 5:
loss went 1523 -> 6627 (4× increase); `natural_vec_norm` jumped 75 -> 1.1e4 -> 1.5e5
in 2 steps.

Root cause: `gpytorch.optim.NGD` computes
`p += -lr * num_data * p.grad` per step (see `gpytorch/optim/ngd.py:42`).
The tutorial uses `VariationalELBO` which returns a per-point ELBO (ELL
mean over batch, KL divided by num_data); the gradient is then at
per-point scale and `-lr * num_data * per_point_grad` is one full-dataset
natural gradient step.

My first attempt reused `gpy_training`'s formula `loss = -expected_log_prob(sum) + kl_divergence`,
which is at full-dataset scale. That makes the gradient num_data× larger
than the mll-computed one; combined with NGD's num_data multiplier the
effective step is num_data² = 2.25M times too large at n=1500.

Fix: switched to `gpytorch.mlls.VariationalELBO`. Divergence gone,
monotone loss descent, cell 16 at 30 iters gives test_r=0.846 (baseline
default_gpy at 50 LBFGS iters = 0.879 for the same cell/seed).

The fix is documented inline in `ngd_training.py` as load-bearing.

### Finding P3-2 (2026-04-22) — Prototype: NGD dramatically recovers outliers but cell 40 overfits at 1000 iters

Prototype: 5 cells (40, 38, 16, 30, 29) × seed 42 × 1000 iters, no ES,
test_r probe every 25 iters.

Per-cell peak test_r and iter of peak (oracle-ES upper bound):

| cell | peak test_r | iter of peak | final test_r @1000 | vargp seed 42 | default_gpy seed 42 |
|---|---|---|---|---|---|
| 40 | 0.821 | 225  | **0.618**  | 0.855 | 0.366 |
| 38 | 0.738 | 925  | 0.733  | 0.778 | 0.311 |
| 16 | 0.970 | 475  | 0.968  | 0.958 | 0.879 |
| 30 | 0.741 | 625  | 0.731  | 0.747 | 0.792 |
| 29 | 0.731 | 975  | 0.729  | 0.729 | 0.753 |

**Cell 40 overfits**: test_r peaks at iter 225 (0.821) then decays steadily
to 0.618 at iter 1000. Training ELBO keeps improving the whole time
(loss 450 at iter 200 -> 384 at iter 1000). Meanwhile `A` grows 0.022 -> 0.039
and `beta` barely moves (0.148 -> 0.154). Classic overfitting of a sparse
cell (firing_rate=0.13); ELBO-based ES cannot catch it because ELBO is
monotone past the test-r peak.

Oracle-ES mean Δ vs vargp across 5 cells: **-0.013** (5-cell mean of peak
- vargp seed-42). Well within the 0.02 threshold if oracle ES were
available. Final-value mean Δ: -0.057.

Decision: run the full sweep at 1000 iters with **no** ES and report
**both** numbers — `final_test_r` (ELBO-argmax, the honest apples-to-apples
metric since Phase 2 also uses ELBO ES) and `best_test_r` over the probe
trajectory (oracle-ES upper bound). The gap between them quantifies
"could-we-get-more-with-better-ES".

### Finding P3-3 (2026-04-22) — Full sweep: NGD closes the Phase 2 gap (|mean_Δ| = 0.012, below 0.02 threshold)

Full sweep: 16 cells × 3 seeds × 1000 iters, no ES. Running time 17 min
wall. One transient CUDA OOM on GB10 on the 2nd run of the 1000-iter
prototype (added `torch.cuda.empty_cache()` between runs; no OOM in the
48-run full sweep).

**Paired Δ summary (n=48):**

| comparison | mean Δ | std | range | pos/neutral/neg |
|---|---|---|---|---|
| NGD_final − vargp_direct  | **+0.0118** | 0.061 | [-0.237, +0.159] | 22 / 14 / 12 |
| NGD_best  − vargp_direct  | **+0.0338** | 0.046 | [-0.040, +0.184] | 28 / 17 /  3 |
| NGD_final − default_gpy   | **+0.0900** | 0.143 | [-0.074, +0.501] | 28 / 13 /  7 |
| NGD_best  − default_gpy   | **+0.1120** | 0.154 | [-0.051, +0.506] | 35 /  7 /  6 |

For reference, the Phase 2 baseline was:

| comparison | mean Δ | (from §6.2) |
|---|---|---|
| vargp_direct − default_gpy | **+0.0782** | GAP EXISTS |

**Pre-registered decision**:

- `|mean_Δ(NGD_final − vargp_direct)| = 0.012 < 0.02`
  → **NO GAP per the pre-registered 0.02 threshold.**
- NGD _final_ beats default_gpy by +0.090 (i.e. closes ~85% of the
  original 0.078 default_gpy gap _plus_ overshoots by 0.012 on average).
- With oracle ES (NGD_best), NGD beats vargp by +0.034 — a GAP in the
  other direction. Note: oracle-ES numbers are informational only; any
  real ES would have to use ELBO or validation data, not test.

**Per-cell table (3-seed mean, all 16 cells):**

| cell | bkt | fr | vargp_μ | default_μ | NGD_final_μ | NGD_best_μ | Δ(F-vg) | Δ(B-vg) |
|---|---|---|---|---|---|---|---|---|
| 40 | low | 0.13 | 0.861 | 0.359 | 0.743 | 0.835 | **-0.118** | -0.026 |
| 30 | low | 0.20 | 0.780 | 0.789 | 0.752 | 0.780 | -0.027 | +0.000 |
|  4 | low | 0.35 | 0.704 | 0.580 | **0.805** | 0.844 | **+0.101** | +0.140 |
|  7 | low | 0.40 | 0.620 | 0.660 | 0.664 | 0.670 | +0.044 | +0.050 |
| 23 | low | 0.45 | 0.723 | 0.656 | 0.749 | 0.761 | +0.026 | +0.038 |
| 25 | mid | 0.57 | 0.823 | 0.825 | **0.899** | 0.899 | **+0.076** | +0.076 |
| 19 | mid | 0.72 | 0.681 | 0.644 | 0.677 | 0.700 | -0.004 | +0.019 |
| 21 | mid | 0.86 | 0.644 | 0.722 | **0.744** | 0.755 | **+0.100** | +0.111 |
| 32 | mid | 0.92 | 0.792 | 0.765 | 0.757 | 0.821 | -0.036 | +0.029 |
| 20 | mid | 0.96 | 0.829 | 0.852 | 0.859 | 0.859 | +0.030 | +0.030 |
| 10 | high| 1.06 | 0.877 | 0.764 | 0.883 | 0.910 | +0.007 | +0.033 |
| 24 | high| 1.07 | 0.897 | 0.874 | 0.926 | 0.936 | +0.030 | +0.040 |
| 38 | high| 1.27 | 0.694 | 0.280 | 0.668 | 0.681 | -0.027 | -0.013 |
| 29 | high| 1.64 | 0.738 | 0.754 | 0.749 | 0.753 | +0.011 | +0.014 |
| 16 | high| 1.96 | 0.958 | 0.920 | 0.958 | 0.966 | -0.000 | +0.008 |
|  8 | high(u)| 0.98 | 0.875 | 0.800 | 0.853 | 0.866 | -0.022 | -0.009 |

Per-cell winners/losers at NGD_final vs vargp (3-seed mean, |Δ|>0.02):
- NGD materially BETTER (+ >= 0.03): cells **4, 25, 21** (by +0.10, +0.08, +0.10)
- NGD roughly EQUAL (|Δ| < 0.03): cells 30, 7, 23, 19, 20, 10, 24, 38, 29, 16, 8
- NGD materially WORSE (≤ -0.03): cells **40, 32** (by -0.118, -0.036)

Cell 40 is the one cell where the Phase 2 outlier _wasn't_ fully
recovered at NGD_final — the overfit observed in P3-2 persists.
With oracle ES (NGD_best), cell 40 is only -0.026 below vargp and the
picture is uniformly + or neutral except for tiny deltas on cells 40,
38, 8.

**Final A distribution (n=48 per mode):**

| mode         | min    | p25    | median | p75    | max    | ≤0.012 |
|---|---|---|---|---|---|---|
| vargp_direct | 0.0143 | 0.0175 | 0.0215 | 0.0271 | 0.0414 | 0 |
| default_gpy  | 0.0042 | 0.0920 | 0.1283 | 0.1727 | 0.7802 | 3 |
| ngd          | 0.0251 | 0.0409 | 0.0540 | 0.0664 | 0.1136 | 0 |

NGD sits between vargp's tight correct-scale distribution and
default_gpy's wide explosion-prone distribution. Zero freezes, no
extreme tails. NGD systematically learns A roughly 2× higher than vargp
— consistent with the overfit story on cell 40 (Adam keeps nudging A
upwards beyond the test-optimum).

**Timing (n=48 per mode):**

| mode         | wall_s mean ± std | s/iter mean ± std |
|---|---|---|
| vargp_direct | 7.15 ± 2.45       | 0.156 ± 0.050 |
| default_gpy  | 8.12 ± 4.31       | 0.236 ± 0.075 |
| ngd          | 16.52 ± 3.01      | 0.0165 ± 0.003 |

NGD wall-clock is ~2x higher (1000 iters vs ~50 outer EM iters), but its
per-iter cost is 10× cheaper. At oracle-ES median stop iter 562, NGD
wall-clock would be ~9s — competitive with vargp_direct and default_gpy.

**Oracle-ES peak iter distribution**: min 125, p25 294, median 562, p75 975,
max 1000. Range is wide (cell-dependent), so a _single_ iter count is
suboptimal. A validation-based ES would in principle catch each cell's
peak, but vargp_direct uses ELBO-based ES by project convention; adding
val-based ES to NGD would be unfair to the comparison.

## 17. Phase 3 conclusions (tentative — data-only)

1. At the pre-registered 0.02 threshold, **NO GAP is detected between
   NGD and vargp_direct** at the main sweep config. Mean paired Δ = +0.012
   (NGD slightly favored), n=48. This reverses the Phase 2 decision
   (|mean_Δ| = 0.078 = GAP, vargp favored) when the variational-step
   optimizer is swapped from whitened-Cholesky + joint LBFGS to
   Tril-natural + NGD.

2. NGD closes ~85% of the Phase 2 aggregate gap and eliminates the
   two outlier-cell failure modes that drove the Phase 2 mean
   (cell 40 A-explosion; cell 38 LBFGS-freeze). Cells 40 and 38 move
   from 0.36/0.28 (default_gpy) to 0.74/0.67 (NGD_final) averaged across
   3 seeds — still slightly below vargp's 0.86/0.69 on cell 40 due to
   cell-40-specific overfit past iter 225.

3. The NGD A distribution is well-controlled: median 0.054, no freezes,
   no explosions (max 0.11). Between vargp's tight 0.02 median and
   default_gpy's wild 0.13 median.

4. Oracle-ES (stop at peak test_r during training) would give NGD an
   additional +0.022 over vargp, reversing the sign decisively. This
   is informational only — scientifically, final_test_r is the honest
   comparison.

5. Cell 40 is the one remaining cell where NGD doesn't close the gap
   (3-seed mean Δ = -0.118 at final, -0.026 at best). Mechanism:
   standard overfitting of a sparse-firing cell by a first-order
   optimizer running past the validation optimum; not an NGD-specific
   pathology.

**The headline answer to the session's question** ("does NGD make GPyTorch
work as well as vargp?"): **Yes, at the 0.02 pre-registered decision
threshold, on this dataset and config.** 3-seed average on 16 cells:
NGD_final within +0.012 of vargp_direct, NGD_final +0.090 above
default_gpy.

## 18. Phase 3 — caveats and what to re-test before calling this a fix

- **Single config.** Tested only at (M=300, n_train=1500, 64x64). No
  evidence the result generalises to other M, n_train, image sizes, or
  datasets.
- **Single dataset.** PNAS-64x64 only.
- **n_iterations=1000 is a pre-registered budget choice**, not learned
  from data. Shortening (e.g. 500) changes per-cell results materially
  — cell 40 would win more, cells 29/30/38 would win less. The aggregate
  picture might differ.
- **No validation-based ES was tried.** A val-ES NGD could plausibly
  match the oracle numbers (Δ ≈ +0.03 vs vargp) — but adding val-ES to
  NGD without also applying it to vargp_direct / default_gpy would be
  asymmetric; the project-wide choice is ELBO ES.
- **No multi-seed significance test was run.** 48 paired obs with
  std=0.061 → standard error of the mean ~ 0.009. +0.012 is
  ~1.3 SEM from zero — consistent with "no detectable gap" but
  NOT "NGD significantly better". Report reflects this: decision is
  "no gap at 0.02 threshold," not "NGD wins".
- **GB10-specific CUDA OOM** observed once in a prototype re-run (not
  in the full sweep). Fixed defensively via `torch.cuda.empty_cache()`
  between runs. Root cause unclear — driver or fragmentation.

## 19. Phase 3 lessons carried forward

Adhered to (from Phase 1 + Phase 2):

1. Budgeted a decisive sample (16 × 3 = 48 paired Δ) from the start —
   used the same (cells, seeds) set as Phase 2 so the comparison is
   immediately paired.
2. Pre-registered the 0.02 threshold and cell/seed list.
3. Reported `final` and `best` test_r separately rather than
   cherry-picking.
4. Documented the GB10 OOM defensively before it could bite the full sweep.
5. Did not modify production code (`gpy_model.py`, `gpy_training.py`,
   `kernels.py`, `likelihoods.py`, `default_params.json`).

New lessons:

- **NGD's `num_data` multiplier interacts with loss scaling — always
  use `VariationalELBO`, not sum-scaled loss** (P3-1). Easy to miss
  when porting NGD code; documented inline in `ngd_training.py`.
- **ELBO ES cannot catch test-r overfit on non-conjugate SVGP** (P3-2 on
  cell 40). A validation split would be necessary if we ever ship NGD
  as a default. Record the peak-test_r-iter distribution as diagnostic;
  do not use it as an ES signal.
- **`TrilNaturalVariationalDistribution` is the right choice for float32
  + Poisson.** No PSD errors in any of 48 runs. The plain
  `NaturalVariationalDistribution` was not tried because the docs
  explicitly warn it is less stable.

## 20. Phase 3 — artifacts / cleanup inventory

Files written during Phase 3 (all under
`investigations/default_gpy_gap_v2/ngd/`):

Kept (primary data / scripts):
- `ngd_model.py` — 50-line fresh SVGP model with `TrilNaturalVariationalDistribution`
- `ngd_training.py` — NGD+Adam training loop with ELBO ES machinery
- `run_ngd_sweep.py` — sweep runner with `--prototype` / `--full` / `--cells` / `--seeds` flags
- `analyze_ngd.py` — paired-Δ analysis + A distribution + timing + oracle-ES diagnostic
- `ngd_results.jsonl` — 48 records, one per (cell, seed)
- `ngd_summary.json` — top-level aggregate JSON
- `full_sweep.log`, `prototype_seed42.log`, `prototype_seed42_1000iter.log`,
  `prototype_seed42_ES.log`, `ngd_results.backup_*.jsonl` — run logs /
  historical prototype results

Not kept (redundant after full sweep):
- None — prototype logs are informative about the P3-1 / P3-2 findings
  and small on disk.

No production code modified.

---

# Phase 3B — ELBO-based early stopping for NGD

Scope: add ES to NGD training so its wall-clock is competitive with
vargp_direct / default_gpy. Pre-registered requirement per user
2026-04-22: ES must be ELBO-based (no validation carve), matching the
project-wide convention; must not materially change the Phase 3
aggregate test_r.

## 21. Phase 3B charter

**Ask**: pick ES parameters, turn them on, rerun the 48-cell/seed sweep,
confirm wall-time improvement without regressing test_r.

Constraints inherited from Phase 3:
- ELBO as the ES signal (no validation split); same machinery as
  `gpy_training.train_gpy_default` and `eigenspace_training`.
- Same 16 cells × 3 seeds as Phase 2 / Phase 3.
- Same NGD+Adam+Tril configuration.
- Decision threshold unchanged: `|mean_Δ(NGD − vargp_direct)| > 0.02`
  = GAP.

Constraints inherited from `.claude/rules/bewary.md`:
- Any hardcoded ES parameter must be traceable to a principled choice
  (no black-box tuning on test set metrics).

## 22. Phase 3B plan

Step 1 — simulate ES on the existing 48 no-ES trajectories:
  `simulate_es.py` loads the full per-iter train_loss curves from
  `ngd_results.jsonl`, runs the exact same patience-based logic as
  `gpy_training.train_gpy_default` (Lightning-style decoupled
  best-tracking + patience counter), and reports:
    - median stop iter
    - estimated wall time at stop iter
    - Δ(simulated test_r − final test_r at iter 1000)
  across 11 (patience, min_delta_rel) combinations.

Step 2 — pick ES parameters from the sweep, justified by ratio to
  vargp_direct's `(patience=15, min_delta_rel=1e-3)` defaults. NGD's
  per-iter step ≈ 1 grad eval; vargp's per-outer-iter step ≈ ~7 grad
  evals. So NGD patience should be ≈ 10× vargp's, and min_delta_rel
  also ≈ 10× because we want the same "relative improvement per
  patience-tick-of-work" threshold.

Step 3 — rerun the 48-cell/seed sweep with ES on. Compare to the no-ES
  baseline (archived at `ngd_results.noES_1000iter.jsonl`).

Step 4 — update SCRAPBOOK, summarise for the user, stop.

## 23. Phase 3B findings

### Finding P3B-1 (2026-04-22) — ELBO ES with δ=1e-3 never fires on our NGD trajectories

Simulated `(patience=100, min_delta_rel=1e-3, min_iterations=50)` — the
exact vargp_direct parameters transplanted verbatim — across all 48
runs. Only 10/48 runs trigger ES before iter 1000. Median stop iter
is 1000 for the remaining 38. Δ test_r essentially zero.

Reason: ELBO keeps improving by >0.1% per iter for the full 1000-iter
budget on most cells. vargp_direct's LBFGS takes ~7 grad evals per
outer iter, so one `patience=15` tick corresponds to ~100 grad evals
of work; a per-tick improvement of 0.1% is a meaningful amount of
progress. NGD's single-grad-eval ticks move the loss by a much smaller
amount each step, so the same 0.1% threshold is never hit on the
no-progress side.

Implication: the `(patience=15, δ=1e-3)` default is not transplantable
as-is. Need to scale both axes.

### Finding P3B-2 (2026-04-22) — Principled rescaling by 10× on both axes matches vargp_direct's stopping behaviour

Simulated 11 (patience × δ) combinations. Sweet-spot summary:

| config                    | median stop | est. wall | Δ test_r vs no-ES | fires |
|---|---|---|---|---|
| p=100 δ=1e-3 (vargp-like) |        1000 |      15.8s |      +0.0002     | 10/48 |
| p=100 δ=5e-3              |         677 |      11.6s |      −0.0018     | 39/48 |
| p=200 δ=5e-3              |         962 |      14.6s |      +0.0004     | 28/48 |
| **p=200 δ=1e-2** (picked) |   **739**   | **12.8s**  |    **+0.0002**   | 36/48 |
| p=200 δ=2e-2              |         639 |      10.5s |      +0.0003     | 43/48 |
| p=100 δ=1e-2              |         525 |       9.3s |      −0.0028     | 43/48 |

Pick: `(patience=200, min_delta_rel=1e-2, min_iterations=50, restore_best=True)`.

Justification (purely by ratio to vargp_direct's `(15, 1e-3, 10, True)`):

- **Patience 200 / 15 ≈ 13×**. Close to the ≈10× ratio between NGD's
  per-iter work unit (1 grad eval) and vargp LBFGS's per-outer-iter
  work unit (~7 grad evals).
- **min_delta_rel 1e-2 / 1e-3 = 10×**. Same ratio, to keep
  "relative improvement required per patience-tick" constant at ~1e-4
  per grad eval in both modes.
- **min_iterations 50 / 10 = 5×**. First-order optimisers need more
  burn-in than LBFGS's big first jumps.
- **restore_best=True**. Same as project-wide default.

Ratios are computed; not tuned on Δ test_r. The sim shows both
(200, 1e-2) and (200, 2e-2) deliver the same headline result; chose the
more conservative value for robustness against trajectories not seen
in the prototype.

### Finding P3B-3 (2026-04-22) — Full sweep with ES: test_r unchanged, wall-clock drops 40 %

Full sweep: 16 cells × 3 seeds × (up to) 1000 iters, ES on with
`(patience=200, min_delta_rel=1e-2, min_iterations=50, restore_best=True)`.
Wall time **8.0 min** total across 48 runs (Phase 3 no-ES baseline was
17 min). Early-stop trigger rate: 36/48 runs (75 %); the 12 remaining
runs hit the 1000-iter cap.

**Paired Δ comparison (vs Phase 3 no-ES baseline and vs Phase 2
baselines, n=48 each):**

| comparison                   | no-ES mean Δ | **ES mean Δ** | Δ change from ES |
|---|---|---|---|
| NGD_final − vargp_direct     |    +0.0118   |  **+0.0120**  |   +0.0002 |
| NGD_final − default_gpy      |    +0.0900   |  **+0.0902**  |   +0.0002 |
| NGD_best  − vargp_direct     |    +0.0338   |  **+0.0299**  |   −0.0039 |
| NGD_best  − default_gpy      |    +0.1120   |  **+0.1081**  |   −0.0039 |

The aggregate metric moves by +0.0002 (noise). Pre-registered decision
is unchanged: `|mean_Δ(NGD_final − vargp_direct)| = 0.012 < 0.02`
→ **NO GAP** between NGD-with-ES and vargp_direct.

Oracle `NGD_best` moved −0.004 because a few runs that would have
converged past their test-r peak are now cut short. This was expected:
ES on ELBO cannot help cases where test_r peaks before ELBO does (e.g.
cell 40). `NGD_best − vargp_direct` still positive at +0.030.

**Timing (n=48 per mode):**

| mode         | wall_s mean ± std    | s/iter mean ± std   |
|---|---|---|
| vargp_direct |      7.15 ± 2.45     |   0.1563 ± 0.0495   |
| default_gpy  |      8.12 ± 4.31     |   0.2364 ± 0.0752   |
| **ngd_ES**   |   **9.93 ± 2.68**    |   0.0131 ± 0.0028   |
| (ngd no-ES)  |     16.52 ± 3.01     |   0.0165 ± 0.0030   |

NGD+ES now within **1.4×** of vargp_direct wall-time and **1.2×** of
default_gpy. Previously was 2.3× / 2.0× respectively. The s/iter
difference vs no-ES (0.013 vs 0.017) is a timing artefact — CUDA
warmup occupies a bigger fraction of the shorter per-run average.

**Per-cell differences from no-ES baseline** (largest |Δ|):

| cell | no-ES ngdF_μ | ES ngdF_μ | Δ |
|---|---|---|---|
|  40 | 0.7431 | 0.7555 | **+0.0124** |
|   4 | 0.8047 | 0.8176 | **+0.0129** |
|  21 | 0.7437 | 0.7313 | −0.0124 |
|  25 | 0.8987 | 0.8790 | −0.0197 |
|  10 | 0.8834 | 0.8946 | +0.0112 |

ES helped cell 40 (`A`-explosion overfit case) slightly — stopped
before the extreme overfit at the 1000-iter mark. Hurt cell 25 slightly.
Net aggregate: +0.0002.

**Final A distribution with ES:**

| mode         | min    | p25    | median | p75    | max    | ≤0.012 |
|---|---|---|---|---|---|---|
| vargp_direct | 0.0143 | 0.0175 | 0.0215 | 0.0271 | 0.0414 | 0 |
| default_gpy  | 0.0042 | 0.0920 | 0.1283 | 0.1727 | 0.7802 | 3 |
| ngd_ES       | 0.0202 | 0.0344 | 0.0457 | 0.0634 | 0.1136 | 0 |
| (ngd no-ES)  | 0.0251 | 0.0409 | 0.0540 | 0.0664 | 0.1136 | 0 |

NGD+ES's A distribution is slightly tighter / lower than NGD no-ES
(median 0.046 vs 0.054), consistent with stopping before A has drifted
all the way up. Still zero freezes, zero explosions, and pulls
slightly closer to vargp_direct's 0.021 median.

## 24. Phase 3B conclusions (data-only)

1. **ELBO-based ES at `(patience=200, min_delta_rel=1e-2,
   min_iterations=50, restore_best=True)` gives a 40 % wall-clock
   reduction with no material change in test_r.** The pre-registered
   gap decision is unchanged: NO GAP between NGD-with-ES and
   vargp_direct at the 0.02 threshold.

2. **Parameter choice was via principled scaling** (NGD-per-iter work ≈
   1/10× LBFGS-per-outer work → scale both patience and min_delta_rel
   by ~10×). Simulation on existing trajectories verified the choice
   before rerunning.

3. **ES triggers in 36/48 runs (75 %)**. The 12 remaining runs hit the
   1000-iter cap. Inspection of those shows loss still meaningfully
   decreasing at iter 1000 — a higher cap would help those runs mildly
   but not materially (their ELBO slope at iter 1000 is ≤1%/100 iters).

4. **NGD-with-ES timing is now competitive with vargp_direct and
   default_gpy** (9.9s vs 7.1s vs 8.1s). No longer a meaningful wall-time
   concern.

**Updated headline answer to the session's original question**:
`NGD with ELBO-based ES (patience=200, min_delta_rel=1e-2) ≈ vargp_direct
on test_r (|mean_Δ|=0.012 < 0.02, n=48) AND within 1.4× vargp_direct
on wall-time. It beats default_gpy by +0.090 on test_r and is within
1.2× on wall-time.`

## 25. Phase 3B — artifacts / cleanup inventory

Added in Phase 3B (all under `investigations/default_gpy_gap_v2/ngd/`):

Kept:
- `simulate_es.py` — ES-parameter simulation over the 48 no-ES trajectories.
- `ngd_results.noES_1000iter.jsonl` — frozen copy of the Phase 3 no-ES
  baseline (before Phase 3B rerun overwrote `ngd_results.jsonl`).
- `ngd_summary.noES_1000iter.json` — summary corresponding to the above.
- `full_sweep_ES.log` — run log of the Phase 3B sweep.

Modified:
- `ngd_training.py` — ES params threaded through (already existed from
  Phase 3 but were unused there); Phase 3B turns them on.
- `run_ngd_sweep.py` — `NGD_ES_PATIENCE=200`, `NGD_ES_MIN_DELTA_REL=1e-2`,
  `NGD_ES_MIN_ITERATIONS=50`, `NGD_ES_RESTORE_BEST=True` as module-level
  constants; `run_one` defaults to `early_stop=True`.
- `ngd_results.jsonl` — now contains Phase 3B (ES-on) results.
- `ngd_summary.json` — summary for Phase 3B results.

No production code modified in Phase 3B either.

---

# Phase 3C — required sweep for a final verdict (spec, do not run in this session)

Phase 3 / 3B answered the question on the Phase 2 16-cell stratified sample.
For a **final, publishable-quality verdict** on whether NGD+ES should be
the new GPyTorch default, we need to reproduce, under NGD, the exact
operating point that gave vargp_direct its best recorded result.

Reference: `experiments/2026-04-06_es_sweeps_64x64/README.md` +
`run_sweep_elbo_es_64x64.py`. The best documented vargp run:
**`intl_fixAmp` + ELBO ES p=15 → test_r=0.8375, exp_var=0.8987, 37/41 cells
with `explained_variance > 0.8`, mean 37.5 iters per run.**

## 26. The sweep that answers the question

**Config (per row of the sweep matrix):**

| parameter | value | source |
|---|---|---|
| mode | `ngd` (Phase 3 code) | new |
| dataset | `datasets/PNAS_64x64_center_crop_no_renorm.npz` | same as vargp best |
| cells | **all 41** | same as vargp best |
| seeds | **{1, 2, 3}** | identical seed set to vargp best |
| M (inducing points) | **250** | identical to vargp best (NB: Phase 3 used M=300) |
| n_train | **3160** | full pool, `n_val_split=0` — identical to vargp best |
| ip_selection | `random` | identical |
| kernel | `arc_cosine` (default) | identical |
| rf_init | `ground_truth` | identical |
| beta init | 0.1 | identical |
| rho init | 0.1 | identical |
| A_init | `0.01` | NGD default (see note) |
| lambda0_init | 1.0 | NGD default (see note) |
| **Amp** | **1.0, FROZEN** (`fix_Amp=True`) | Matches vargp's `intl_fixAmp` config exactly. `fix_Amp` support added to `mode='ngd'` at Phase 3D wiring time (runs `kernel.raw_Amp.requires_grad_(False)` → Adam skips it). Without freezing, Adam learns Amp along with the other kernel params — that would be a deviation from the vargp baseline. **We run with frozen Amp.** |
| n_iterations (cap) | **1500** | higher than vargp's 80 because NGD is first-order; ES expected to fire well before cap |
| dtype | float32 | default |
| device | cuda | default |

**Rationale for A_init / lambda0_init differences from vargp `intl_fixAmp`:**
`intl_fixAmp` uses A_init=1e-4 specifically because the interleaved damped
Newton E-step overshoots if A starts at 0.01. NGD has no E-step (the
variational distribution updates via natural-gradient descent directly),
so that failure mode does not apply. We use the standard NGD init
(A_init=0.01, lambda0=1.0) documented in the Phase 3 charter.

If the user wants the **A_init=1e-4** run as a robustness check, that's a
second row of the sweep; not strictly necessary to answer the main
question, but cheap to add (same compute).

**ES parameters:**

| parameter | value | source |
|---|---|---|
| early_stop | `True` | Phase 3B choice |
| es_metric | `elbo` | project convention |
| patience | **200** | `NGD_ES_PATIENCE` from `run_ngd_sweep.py` |
| min_delta_rel | **1e-2** | `NGD_ES_MIN_DELTA_REL` |
| min_iterations | **50** | `NGD_ES_MIN_ITERATIONS` |
| restore_best | `True` | project convention |

**Matrix**: 41 cells × 3 seeds = **123 runs** (one config). Optional
A_init=1e-4 robustness row: +123 runs → 246 total.

**Expected wall time** at ~10 s per run (with ES) × 123 runs ≈ **~22 min**
sequential on GPU.

## 27. Pre-registered decision criterion

Same as Phase 3 / 3B:

`|mean_Δ(NGD_final − vargp_direct_intl_fixAmp_elbo_p15)| > 0.02 = GAP`

Paired per (cell, seed), n=123. Report paired mean, std, range, IQR,
per-cell sign counts.

Also report, matching the vargp reference:
- `mean test_r` (single scalar)
- `mean exp_var`
- `cells > 0.8 exp_var` (count / 41) — the headline metric in the vargp
  best-sweep documentation.
- wall-time per run distribution
- ES trigger rate and stop-iter distribution

## 28. Secondary checks to include in the final sweep

These are "cheap extras" that the sweep should record alongside the
main metrics, so we don't have to rerun to answer follow-up questions:

1. **Parameter convergence** — final `A`, `beta`, `rho`, `lambda0`,
   `eps_0x`, `eps_0y`. Compare distributions across modes.
2. **Test-time numerical stability** — count of `predict()` runtime
   errors. (Should be zero under NGD+ES; non-zero would flag a
   regression.)
3. **`best_test_r_probe`** trajectory — same probe-every-25-iters
   mechanism as Phase 3. Lets us quantify residual "oracle-vs-ELBO-ES"
   gap on the full 41-cell sample.
4. **Training curves** — per-iter `train_loss`, `A`, `beta` — at least
   stored in the JSONL. Already done in Phase 3.
5. **RNG determinism cross-check** — rerun seed 1 on cell 0 twice;
   results must match bit-exact or near-exact (float32 nondeterminism is
   at most a few LSBs). If not, flag.

## 29. What would make NGD the recommended default

Only the simultaneous combination of:

1. `|mean_Δ(NGD_final − vargp_direct_best)| ≤ 0.02` (no gap), on
   the full 41-cell × 3-seed sample.
2. `cells > 0.8 exp_var` count for NGD ≥ 36/41 (vargp best: 37/41).
3. No crash / numerical failure on any of the 123 runs.
4. Mean wall-time ≤ 2× vargp's on matched hardware.

If any fails, do not wire in as default without a session-level
discussion.

## 30. What would kill the proposal

Any of:

- > 5 cells where NGD_final is > 0.10 below vargp_direct_best at the
  3-seed mean.
- Multiple cells where NGD_final is below 0.5 while vargp_direct is
  > 0.8.
- ES stopping systematically too early on high-firing-rate cells
  (i.e. the Phase 3 pattern reproduced at scale).
- Any `predict()` failures under `float32`.
- Wall-time regression > 3× vs vargp_direct_best.

## 31. Reference script structure (to write in the next session)

Next session should:

1. Create `experiments/<date>_ngd_final_verdict_64x64/` folder.
2. Write `run_sweep_ngd_elbo_es_64x64.py` mirroring the structure of
   `run_sweep_elbo_es_64x64.py` (args, seeds, per-cell loop, JSONL
   output).
3. Write `analyze.py` reusing `investigations/default_gpy_gap_v2/ngd/analyze_ngd.py`
   as a template (but loading the full 41-cell × 3-seed sweep).
4. Run it; allocate ~25 min on a clean GPU.
5. Write a `README.md` in the experiment folder summarising the result
   and the §27–30 decision criteria outcome.

**Do not** use `run_single_mode.py` with `mode='ngd'` until Phase 3D
(wiring task) is complete — at the time this spec was written, the NGD
mode was accessible only via `run_ngd_sweep.py`.

## 32. Explicit non-goals of the final-verdict sweep

- **Hyperparameter search on NGD lr / Adam lr**: locked at
  `(0.1, 0.01)` tutorial defaults. If the main sweep fails by a small
  margin, this is the first thing to revisit, but not in the verdict
  sweep itself (too easy to tune-on-test).
- **ES parameter retune**: fixed from Phase 3B.
- **Minibatching**: full-batch (n_train=3160 fits; matches vargp best).
- **Other datasets (108x108, 48x48)**: deferred; the
  vargp-best reference is 64x64 only.
- **Other kernels**: `arc_cosine` only.
- **Comparing to `default_gpy`**: already done in Phase 3 for 16 cells
  and a smaller config. The Phase 3 default_gpy performance at 64x64
  sets the floor; no new default_gpy run needed in the verdict sweep.

## 33. Phase 3C artifacts / cleanup inventory

None yet — this section is a spec for a future session. No new files.

---

# Phase 3D — NGD wired in as a production mode

Scope: after Phase 3/3B validated NGD's correctness (|mean_Δ vs vargp|=0.012,
NO GAP) and Phase 3C spec'd the final-verdict sweep, the user asked for the
implementation to be promoted to a first-class mode in this branch's
production scripts. This section records the promotion.

## 34. Review summary (via `/simplify` skill)

Three parallel review agents (reuse / quality / efficiency). Key verdicts:

**Blocking (had to fix during wiring)**:
- Hardcoded NGD numerics violated CRITICAL RULE 7 (`NGD_ES_PATIENCE=200`,
  `NGD_LR=0.1`, etc.) — must route through `_constants.py` /
  `default_params.json`.
- `NGDVariationalGPModel` duplicated `VariationalGPModel` — must factor
  out as a `variational_distribution_cls` parameter of the existing class.
- Monkey-patching `gpy_training.train_gpy_default` is fine for an
  investigation prototype but unfit for production — must add a real
  `mode='ngd'` branch in `run_single_mode.py`.
- `probe_test_r=True` default leaked the test set into the training
  loop (diagnostic only, but the wrong default for production) — must
  default to `None`.
- Double docstring in `train_ngd`; dead code (`M_dim` unused,
  `import copy` unused); `standard_variational_distribution=True`
  returned from `NGDVariationalGPModel` was a lie for interop — fix
  during port.

**Non-blocking polish (deferred for now)**:
- ES state-machine duplication between `gpy_training.py`,
  `eigenspace_training.py`, and `ngd_training.py`. Drift risk. Known,
  worth factoring out in a future "shared training utils" PR.
- `.npz` caching / per-cell test-tensor caching — efficiency wins of
  ~10-30 s/sweep; not worth touching the data-loading path right now.
- Redundant `expected_log_prob` + `kl_divergence` recomputation under
  `no_grad` for logging — kept for numerical simplicity; could
  derive ELL+KL from the `mll` scalar algebraically, but fragile
  across GPyTorch versions.
- `os.chdir(ROOT)` in `run_ngd_sweep.py` mutates global CWD — issue
  stays in the investigation code, which is now superseded anyway.

## 35. Wiring changes (this branch only)

All edits landed on `pietro/investigate-default-gpy`; nothing else.

### `default_params.json`
Added top-level `"ngd"` block with `n_iterations=1000`, `lr=0.1`,
`adam_lr=0.01`, `es_patience=200`, `es_min_delta_rel=1e-2`,
`es_min_iterations=50`, `es_restore_best=true`. Comments in-file point
to Phase 3B for derivation.

### `_constants.py`
New exports: `NGD_N_ITERATIONS`, `NGD_LR`, `NGD_ADAM_LR`,
`NGD_ES_PATIENCE`, `NGD_ES_MIN_DELTA_REL`, `NGD_ES_MIN_ITERATIONS`,
`NGD_ES_RESTORE_BEST`. All loaded from `default_params.json["ngd"]` at
import time, matching the project convention (see CRITICAL RULE 7).

### `gpy_model.py`
`VariationalGPModel.__init__` now accepts a `variational_distribution_cls`
kwarg (`'cholesky'` default / `'tril_natural'`). The old
`standard_variational_distribution` bool is preserved for
backwards-compat; the new `variational_distribution_kind` string
attribute is the authoritative flag downstream training loops read.
`TrilNaturalVariationalDistribution` is rejected with
`standard_variational_distribution=False` (unwhitened strategy is
incompatible with natural params).

Backwards compat verified: existing calls still work because
`variational_distribution_cls` defaults to `'cholesky'`.

### `ngd_training.py` (new, project root)
Production port of `investigations/default_gpy_gap_v2/ngd/ngd_training.py`
with review fixes applied:

- Defaults read from `_constants.py` (not inline literals).
- Single docstring, no stale "No early stopping" note.
- `test_r_probe=None` default (was `True` in investigation) — no test
  peek in the training loop.
- `torch.isfinite(loss)` instead of `isnan or isinf`.
- `hasattr(kernel, 'beta')` etc. hoisted out of the loop.
- Explicit runtime check that the model was built with
  `variational_distribution_cls='tril_natural'` — prevents silent
  wrong-optimizer bugs.
- Inline comment on the `VariationalELBO` scaling contract (the
  P3-1 gotcha) preserved verbatim.
- ES state-machine logic kept as-is (intentional duplication with
  `gpy_training.py` for now — future refactor flagged).

### `run_single_mode.py`
New `mode='ngd'` branch after the `default_gpy` block. It:
- Builds `VariationalGPModel(..., variational_distribution_cls='tril_natural')`.
- Reuses `PoissonLikelihood`, `create_kernel`, `apply_rf_center_bounds`,
  `predict`, `compute_*_correlation` — no duplication.
- Calls `train_ngd` from the new `ngd_training` module.
- Reads NGD-specific params from `config['ngd_*']` (populated by
  `build_config_from_defaults` from `default_params.json["ngd"]`).
- `early_stop`, seed, cell, M, n_train come from the same config
  dict as every other mode — a `mode='ngd'` run is a drop-in
  replacement for `mode='default_gpy'`.

`build_config_from_defaults` reads the `ngd` section and populates
`config['ngd_n_iterations']`, `config['ngd_lr']`, `config['ngd_adam_lr']`,
`config['ngd_es_patience']`, `config['ngd_es_min_delta_rel']`,
`config['ngd_es_min_iterations']`, `config['ngd_es_restore_best']`.

`--mode` CLI choices updated to include `'ngd'`.

### Superseded investigation files
Added "SUPERSEDED" headers to:
- `investigations/default_gpy_gap_v2/ngd/ngd_model.py`
- `investigations/default_gpy_gap_v2/ngd/ngd_training.py`
- `investigations/default_gpy_gap_v2/ngd/run_ngd_sweep.py`

The files are kept as the Phase 3 investigation artifact. New code
uses the production path.

## 36. Smoke test

`python run_single_mode.py --mode ngd --data-path datasets/PNAS_64x64_center_crop_no_renorm.npz \
    --ntilde 300 --n-train 1500 --cell 16 --seed 42 --ip-selection random`

Result:
- Trained via NGD+Adam, ES fired at iter 592 (best iter 589).
- Wall time 7.0 s.
- Test Pearson r = 0.9702. (Phase 3B seed-42 cell-16 was 0.9682.)
- Produced `imgs/ngd_M300.png` diagnostic figure via the existing
  plotting pathway — no plotting code changes needed.

Within-run reproducibility of the Phase 3B numbers within ±0.003 on
one (cell, seed). Good signal that the production wiring is
numerically consistent with the investigation.

## 37. What remains (deferred — not in scope here)

1. **The 41-cell × 3-seed final-verdict sweep** (Phase 3C spec). Not
   run in this session. That sweep is the only thing standing between
   NGD-as-an-option and NGD-as-the-gpytorch-default.
2. **Factor out the shared ES state-machine** across `gpy_training.py`,
   `eigenspace_training.py`, `ngd_training.py`. Drift risk. Separate PR.
3. **CLAUDE.md update** to mention `mode='ngd'` in the "Training Modes"
   table and the "File Map". Not done in this session — the scrapbook
   is the authoritative record until the verdict sweep runs.
4. **`ngd` not yet added to YAML experiment system** (`configs/canonical.yaml`,
   `flatten_yaml_config`). If the verdict sweep confirms, that's the
   next step for the canonical experiment workflow.

## 38. Phase 3D artifacts / inventory

Added:
- `default_params.json` — `ngd` block (11 new lines).
- `_constants.py` — 8 new exports (10 new lines).
- `gpy_model.py` — `variational_distribution_cls` parameter + dispatch
  (~40 new lines, net).
- `ngd_training.py` (NEW at project root, 244 lines).
- `run_single_mode.py` — `mode='ngd'` branch (~130 new lines);
  `ngd_*` keys in `build_config_from_defaults` (7 lines);
  CLI `choices` updated (1 line).
- `investigations/default_gpy_gap_v2/ngd/*.py` — SUPERSEDED headers
  added (3 small edits).

Removed: nothing.

Files on other branches: untouched.

---

# Phase 3E — NGD + LBFGS for hyperparameters (spec, future session)

Motivation: the Phase 3/3B/3D choice of Adam for hyperparameters follows
the GPyTorch tutorial verbatim. LBFGS is an architecturally valid
alternative — `gpytorch.optim.NGD` only owns the variational parameters;
the hyperparameter optimiser is the user's call. Swapping Adam for LBFGS
would bring the NGD loop structurally closer to `vargp_direct`'s
E-step / M-step pattern: one natural-gradient step (≈ one E-step iter)
then a full LBFGS inner loop (= the M-step we already trust).

Open question: does NGD + LBFGS close the residual ~0.012 mean Δ and/or
cut wall-time further, without reintroducing the joint-LBFGS basin-trap
that drove the whole Phase 2/3 investigation?

## 39. Phase 3E pre-registered charter

**Question**: at the Phase 3C config (41 cells × 3 seeds, M=250,
n_train=3160, fix_Amp=True), does swapping Adam for LBFGS on the
hyperparameter step change the paired-Δ against vargp_direct
(intl_fixAmp + ELBO ES p=15)?

**Hypotheses**:

| H | Prediction | What would confirm it |
|---|---|---|
| H1 | NGD+LBFGS matches or beats NGD+Adam: fewer outer iters, lower wall-time, same or better test_r. | mean Δ shifts toward zero or positive; ES fires at smaller iter; no new cell regressions. |
| H2 | NGD+LBFGS reintroduces a milder form of Phase 2's LBFGS basin-trap on some cells (likely 40, 38, 32, or a new subset). The variational posterior is handled by NGD so full catastrophic failure shouldn't happen, but LBFGS's Hessian memory could still latch onto a bad kernel configuration at iter 5 that's hard to escape later. | ≥ 1 cell shows new Δ < -0.10 vs NGD+Adam at 3-seed mean; final `beta` collapses to < 0.02 on any cell; LBFGS-freeze-after-plateau on kernel params observable in training curves. |
| H3 | NGD+LBFGS is ~indistinguishable from NGD+Adam. | mean Δ stays within ±0.01 of Phase 3C result; per-cell signs mostly match. |

Prior: **H3 is the most likely**; H2 is what I'd flag as a real risk given Phase 2;
H1 would be a pleasant surprise.

## 40. Phase 3E plan

**Step 0 — implementation (small)**:
- Write `train_ngd_lbfgs` by forking `ngd_training.train_ngd`. The only
  change is the outer loop:

  ```python
  ngd_opt = gpytorch.optim.NGD(model.variational_parameters(),
                                num_data=n_data, lr=ngd_lr)
  lbfgs_opt = torch.optim.LBFGS(
      list(model.hyperparameters()) + list(likelihood.parameters()),
      lr=1.0, max_iter=gpy_lbfgs_max_iter, line_search_fn='strong_wolfe')

  for i in range(n_iterations):
      # 1. NGD step on variational params.
      ngd_opt.zero_grad()
      loss = -mll(model(X), y); loss.backward()
      ngd_opt.step()

      # 2. LBFGS inner loop on hyperparams. Needs closure — see
      #    gpy_training.train_gpy_default for the guard pattern
      #    (params_in_bounds → f_mean threshold → backward).
      def closure():
          lbfgs_opt.zero_grad()
          if not kernel.params_in_bounds(): return inf
          if not likelihood.params_in_bounds(): return inf
          out = model(X)
          ell = likelihood.expected_log_prob(y, out)
          kl  = model.variational_strategy.kl_divergence()
          f_mean = exp(A*out.mean + 0.5*A*A*out.variance + lambda0)
          if f_mean.max() > F_MEAN_MAX or f_mean.mean() > F_MEAN_MEAN_MAX:
              return inf
          loss = (-ell + kl) / n_data  # per-point, for NGD-scale consistency
          loss.backward()
          return loss
      lbfgs_opt.step(closure)

      # 3. Clamp (same as train_ngd).
      kernel.clamp_hyperparameters(); likelihood.clamp_params()

      # 4. ES check (identical to train_ngd).
  ```

  Per-point loss scaling inside closure matches what NGD sees on the
  outer step, so both optimisers agree on units. The inner LBFGS
  iteration budget is `gpy_lbfgs_max_iter=20` (from `_constants.py`),
  matching default_gpy.

**Step 1 — prototype**: 5 cells × 1 seed = 5 runs. Same cells as the
  Phase 3 prototype (40, 38, 16, 30, 29), seed 42 for direct comparison
  with Phase 3's trajectories. Record test_r, final params, loss curve,
  wall-time.

  Pass/fail gate:
  - No run crashes or produces NaN.
  - No cell regresses by more than 0.05 vs NGD+Adam at matched iter
    count.
  - If either fails, write it up as a negative result and stop.

**Step 2 — if Step 1 passes, full 41-cell × 3-seed sweep** mirroring
  Phase 3C exactly except for the optimizer swap. Produces `results.jsonl`
  for a paired-Δ comparison against both Phase 3C (NGD+Adam) and the
  vargp baseline.

**Step 3 — write-up** mirroring Phase 3B structure.

## 41. Phase 3E pre-registered decision criterion

Three-way comparison at n=123 paired (cell, seed):

| comparison | threshold |
|---|---|
| NGD+LBFGS_final − vargp_direct | 0.02 (same pre-registered threshold as Phase 3C) |
| NGD+LBFGS_final − NGD+Adam_final | 0.02 (would this be a meaningful improvement over the current NGD default?) |

Decision matrix:

| ΔvsVargp | ΔvsAdam | action |
|---|---|---|
| within 0.02 | within 0.02 | **Tie** — no reason to prefer LBFGS; keep Adam as default (simpler, no closure). |
| within 0.02 | >+0.02 | **Adopt** — NGD+LBFGS is measurably better; promote to default, keep Adam available. |
| within 0.02 | <−0.02 | **Reject** — LBFGS hurts; preserve finding, do not wire in. |
| outside 0.02 | any | **Investigate** — unexpected; LBFGS for hyperparams is a narrower change than the Phase 2 joint-LBFGS failure, so a large delta would point at a new mechanism. |

Also report (secondary, non-deciding):

- Wall-time distribution — expected LBFGS wall-time to be similar or
  lower (fewer outer iters needed at more expressive inner step).
- ES trigger rate and stop-iter distribution.
- Per-cell kernel-param trajectory for cells 40 and 38 — are they
  recovered, do they show LBFGS freeze-at-plateau?

## 42. Phase 3E what would make this a fix / what would kill it

**Fix signal** (promote to default):
- NGD+LBFGS matches or beats NGD+Adam at p < 0.05 paired test.
- Zero cells with |Δ vs NGD+Adam| > 0.10.
- No convergence failures across 123 runs.
- Wall-time ≤ NGD+Adam's.

**Kill signal** (do not wire in):
- Any cell shows Phase-2-style `A` explosion or `beta` collapse on
  any seed.
- > 5 cells regress by > 0.05 vs NGD+Adam.
- LBFGS crashes (`all evaluations rejected`) on any run.
- Systematic worse-than-Adam on low-firing-rate cells (this would be
  the LBFGS-basin-trap returning).

## 43. Phase 3E caveats

- LBFGS with a CLOSURE is sensitive to guard-clause correctness.
  Phase 2 spent substantial effort getting `params_in_bounds` +
  `f_mean` guards right for `train_gpy_default`. Reuse them verbatim.
- The first iteration will have a zero-initialised variational
  posterior, so the ELBO gradient on kernel params will be small /
  unstable. LBFGS with line search may reject every trial step at iter
  1 (we saw this in Phase 2 Finding P2-2 Run B for warm-start).
  Possible mitigation: run 10 NGD-only warm-up iters before enabling
  LBFGS.
- Closure + strong_wolfe means each outer iter costs ~7x a single
  grad eval. ES patience/min_delta_rel should be rescaled back toward
  vargp_direct's `(15, 1e-3)` — NGD+LBFGS's outer iter is similar in
  compute to vargp's outer iter.
- `fix_Amp` must still be honoured via
  `kernel.raw_Amp.requires_grad_(False)` before constructing the
  LBFGS param list (otherwise LBFGS sees an all-zero Amp gradient
  and the line search may behave strangely).

## 44. Phase 3E non-goals

- Do not tune `ngd_lr`. Locked at 0.1.
- Do not tune `gpy_lbfgs_max_iter`. Use the project default (20).
- Do not attempt minibatching — full-batch only, same as Phase 3/3C.
- Do not change the ES patience/min_delta_rel defaults without running
  a tuning sub-sweep first (Phase 3E-prime or similar).
- Do not touch `default_params.json` or promote to default mode without
  the full 41-cell × 3-seed sweep result in hand.

## 45. Phase 3E rough effort estimate

- Implementation: ~1-2 hours (fork `train_ngd`, reuse guards, add ES
  rescaling option).
- Prototype (5 runs): ~5 min.
- Full sweep (123 runs): ~30 min if LBFGS converges quickly, up to
  ~60 min if it doesn't.
- Analysis + writeup: ~1 hour.

Total: half a session.

## 46. Phase 3E — artifacts / cleanup inventory

None yet — spec only. When implemented, the expected files are:

- `ngd_lbfgs_training.py` at project root (new, ~250 lines forked
  from `ngd_training.py`).
- `experiments/<date>_ngd_lbfgs_verdict_64x64/run_sweep.py` + `results.jsonl`.
- SCRAPBOOK Phase 3E Findings section.

No production-code changes until the verdict sweep passes.

---

# Phase 3F — post-audit fix: NGD loss-logging convention

Triggered by the 2026-04-22 /simplify audit (see §34 for Phase 3D audit
already done; this is a different audit, scoped to the Phase 3D wiring
+ Phase 3C sweep code). The user asked for a full audit after the
sweep completed, with permission to fix real bugs but defer design
decisions.

## 47. The mixed pre/post-step ELBO in the NGD training loop

### The code that shipped in Phase 3D

```python
# ngd_training.py, pre-fix
for i in range(n_iterations):
    variational_ngd_optimizer.zero_grad()
    hyperparameter_optimizer.zero_grad()

    output = model(X_train)               # (θ_i, q_i)  — "pre-step"
    loss = -mll(output, r_train)          # per-point -ELBO at (θ_i, q_i)

    if not torch.isfinite(loss): break

    loss.backward()
    variational_ngd_optimizer.step()      # q_i -> q_{i+1}
    hyperparameter_optimizer.step()       # θ_i -> θ_{i+1}

    kernel.clamp_hyperparameters()
    likelihood.clamp_params()

    # Logging block (THE BUG):
    with torch.no_grad():
        ell_full = likelihood.expected_log_prob(r_train, output)
        #   ^ `output.mean/var` ARE pre-step (cached in the `output`
        #     MVN object), but `likelihood.A` and `likelihood.lambda0`
        #     are POST-step (Adam just updated them above). So
        #     `ell_full` is a FRANKENSTEIN READ:
        #       output part: at (θ_i, q_i)
        #       likelihood part: at θ_{i+1}
        kl_full = model.variational_strategy.kl_divergence()
        #   ^ fully POST-step (reads q_{i+1} and θ_{i+1})
        current_loss = (-ell_full + kl_full).item()
```

The logged `current_loss` at iter i is therefore (writing out the terms
as functions of the state they actually read):

```
current_loss[i] = -ELL(output_i; A_{i+1}, λ₀_{i+1}) + KL(q_{i+1} ‖ p(θ_{i+1}))
```

This is neither the pre-step ELBO (`-ELL(i) + KL(q_i ‖ p(θ_i))`) nor the
post-step ELBO (would require another forward with q_{i+1} and θ_{i+1}).

### The systematic bias

Early in training, `q_{i+1}` has larger KL divergence from the prior
than `q_i` (q is moving away from the prior, toward the posterior).
Likewise, `θ_{i+1}` is usually slightly "better" at describing the data
than θ_i. The logged loss uses:

- ELL term: pre-step `output` (q_i) evaluated with post-step A/λ₀.
  Adam's step on `raw_A` (lr=0.01, gradient magnitude O(1)) changes A by
  at most ~0.01 per iter. The ELL shift is O(A · Δλ · μ) ~ 0.01 × 1500 ×
  (small μ change). Moderate.
- KL term: post-step `kl_divergence()`. KL(q_{i+1} ‖ p(θ_{i+1})) can be
  noticeably larger than KL(q_i ‖ p(θ_i)) in early iters.

Net: the logged `current_loss` is biased **upward** relative to either
the pre-step or post-step ELBO. Bias is largest in the first ~200 iters
where KL is climbing fast; shrinks as KL saturates.

### Why the fit was not affected

Only `loss` (pre-step, returned by `mll`) is fed to `.backward()`. The
gradients computed are pre-step gradients of the true ELBO. The
optimizer steps are unchanged. The final trained model, final params,
and final `test_r` are identical to what the fixed code produces. The
bug is purely a **logging / diagnostic** bug.

### Consequences

1. **`train_loss` curves are biased** — mildly, mostly early. Cross-mode
   plots (NGD vs vargp vs default_gpy) don't line up cleanly because
   vargp and default_gpy log the pre-step ELBO convention.
2. **`train_log_lik` and `train_kl` curves separately** are both
   semantically odd (ell uses post-step A with pre-step output; kl is
   purely post-step).
3. **`best_iteration` (the argmax of -current_loss) may differ by a few
   iters from the true pre-step-ELBO argmax**. ES patience=200 dominates
   this; no practical decision is affected. But `best_iteration`
   semantically "should" point at the best pre-step ELBO, and didn't.

## 48. The fix (landed 2026-04-22)

In `ngd_training.py`, move the `ell_full` / `kl_full` capture BEFORE the
`.backward()` + `.step()` block, and use `loss.item() * n_data` for the
`train_loss` curve (exact algebraic identity with VariationalELBO's
per-point loss — see `_ApproximateMarginalLogLikelihood.forward` in
GPyTorch):

```python
# ngd_training.py, post-fix
output = model(X_train)
loss = -mll(output, r_train)          # per-point -ELBO PRE-step
if not torch.isfinite(loss): break

# Capture PRE-step full-dataset ELL and KL for logging.
with torch.no_grad():
    ell_full_pre = likelihood.expected_log_prob(r_train, output).item()
    kl_full_pre  = model.variational_strategy.kl_divergence().item()

loss.backward()
variational_ngd_optimizer.step()
hyperparameter_optimizer.step()
kernel.clamp_hyperparameters()
likelihood.clamp_params()

# loss.item() * n_data == -ELL_sum_pre + KL_full_pre (algebraic identity).
current_loss = loss.item() * n_data
curves['train_loss'].append(current_loss)
curves['train_log_lik'].append(ell_full_pre)
curves['train_kl'].append(kl_full_pre)
```

Net: all three logged quantities are pre-step full-dataset values at
iter i. Same convention as `gpy_training.py` / `eigenspace_training.py`
so cross-mode curves align.

### Verification

Smoke test (mode=ngd, M=250, n_train=3160, cell=16, seed=1, fix_Amp=False):
runs cleanly, ES fires at iter 882, test_r=0.9537. Previously
(Phase 3D wiring + cell 16 seed 42, M=300, n_train=1500): test_r=0.9702.
Different config, not a regression check; just confirms the fix doesn't
break the fit.

## 49. Stamping old sweep artifacts

Two convention-mismatch stamps added:

- `experiments/2026-04-22_ngd_final_verdict_64x64/LOSS_CONVENTION.md`
  — applies to Phase 3C's 123-run verdict sweep.
- `investigations/default_gpy_gap_v2/ngd/LOSS_CONVENTION.md` — applies
  to Phase 3 (48 no-ES records) and Phase 3B (48 with-ES records) and
  the historical backup files.

Each stamp documents:
1. The mixed convention in the logged `train_loss` / `train_log_lik` /
   `train_kl` curves.
2. Which fields are NOT affected (`test_r`, `final_A`, `A_curve`, etc.).
3. How to produce apples-to-apples cross-mode curves if needed (rerun
   with the fixed code; Phase 3C reruns in ~27 min, Phase 3B in ~10 min).

Phase 3B and Phase 3C VERDICTS stand — they are based on `test_r`,
which is unaffected.

## 50. What future Claude needs to know

Gotcha class: "computing a scalar for logging from a cached distribution
object after in-place optimizer steps have changed other parts of the
model". The cached `output` (an `MultivariateNormal`) carries its mean
and variance tensors from the pre-step forward, but any likelihood or
model method called ON THAT OUTPUT reads live module attributes
(`likelihood.A`, `self.variational_strategy.kl_divergence()`, etc.) —
which are now post-step.

If you port this loop to another variational-GP training mode, put all
logging reads either:
- BEFORE the `.step()` calls (pre-step convention, the chosen default,
  matches `gpy_training.train_gpy_default`), OR
- AFTER a second forward pass with the post-step state (post-step
  convention, ~2x cost).

Do NOT mix pre-step `output` with post-step likelihood reads — it
produces the Frankenstein quantity above.


---

# Phase 3E — LBFGS for hyperparameters: investigation findings (2026-04-22 / 2026-04-23)

Scope: execution of the Phase 3E charter pre-registered in §39-46. Four
variants of NGD+LBFGS were prototyped on 5 cells × 3 seeds each. All
four failed the §40 pass gate. No full 41×3 verdict sweep has been run.
This section is the canonical handoff document — read it first if you
are picking up this investigation after a gap.

## 51. Current status (top of section, for quick orientation)

- **PHASE 3E DEFERRED (2026-04-23)**: user decision after five failed
  variants — "nothing came out of it, adam seems enough". The
  investigation is paused. No production change. NGD+Adam (Phase 3C,
  `experiments/2026-04-22_ngd_final_verdict_64x64/`) remains the
  recommended GPyTorch-native training mode.
- **Five variants tested** (V0 joint no-warmup, V1 separate no-warmup,
  V2 joint + warm-up, V3 separate + warm-up, V5 joint + warm-up +
  damped inner loop). None pass the §40 pass gate on the 5 × 3
  prototype. V4 (LBFGS cadence) was discussed and deprioritized.
- **V2 is the frontier** we found: 4/5 prototype cells within ±0.05
  of NGD+Adam, but cell 30 persistently disasters across every LBFGS
  variant. See §54 for the tables, §55 for per-variant mechanism,
  §60 for V5.
- **Dominant failure mechanism**: LBFGS with strong_wolfe line search
  aggressively exploits the A→0 local optimum of the Poisson-exp
  likelihood (where `f_mean = exp(λ₀) = const`). Adam's small steps
  cannot reach this basin; LBFGS can and does. Damping LBFGS (V5,
  max_iter=1) prevents the collapse but prevents training too — no
  clean middle ground on the `lbfgs_max_iter` axis.
- **No production code has been changed.** Everything is in
  `investigations/default_gpy_gap_v2/ngd_lbfgs/` as an investigation.
- **To reopen**: see §57 for untried avenues (LBFGS cadence, longer
  warm-up, heavier reparameterizations). None looked promising enough
  to pursue given Adam already works.

## 52. Charter reminder

The Phase 3E charter (§39-46) pre-registers:

- Question: does swapping Adam for LBFGS on NGD's hyperparameter step
  change the paired-Δ against `vargp_direct` (intl_fixAmp + ELBO ES p=15)?
- Pass gate (§40 Step 1): 5 cells × 1 seed prototype — no crashes, no
  cell regresses > 0.05 vs NGD+Adam. (We used 5 cells × 3 seeds to
  reduce seed noise; see §53.)
- Kill signals (§42): "Any cell shows Phase-2-style A-explosion or
  β-collapse on any seed"; ">5 cells regress by >0.05"; "LBFGS crashes
  on any run".
- If pass: 41×3 full verdict sweep; apply §41 decision matrix.

The 5 prototype cells are {40, 38, 16, 30, 29} — same as Phase 3's
prototype. Seeds are {42, 123, 789} (the Phase 3B set — not the Phase
3C final-sweep set of {1, 2, 3}).

Dataset: `PNAS_64x64_center_crop_no_renorm.npz`. M=250. n_train=3160.
`fix_Amp=True`. `ip_selection='random'`. All variants share this.

## 53. Prototype-gate modification: 3 seeds instead of 1

The §40 charter specified 5 cells × 1 seed. We used 3 seeds per cell
because V0's single-seed result (seed 42) showed cell 38/40's NGD+Adam
reference has huge seed variance (std 0.18 on cell 38 across seeds
42/123/789 for NGD+Adam+ES). The single-seed gate is uninterpretable
there. 3 seeds gives 3-seed paired means that map cleanly onto the
§42 "any seed shows A-collapse" kill signal.

Trade-off: 3× the prototype cost (~9 min vs ~3 min), same decision
quality for the hard cells, strictly better for the noisy cells.

## 54. Variants tested — results summary

All four variants share `ngd_lbfgs_training.py` with two flags:
`separate_likelihood` (False = joint, True = split kernel vs A/λ₀) and
`n_warmup` (0 = no warmup, 50 = 50 NGD-only iters with kernel +
likelihood frozen before the main loop).

| Variant | `separate_likelihood` | `n_warmup` | JSONL | Log |
|---------|----------------------|-----------|-------|-----|
| V0 joint no-warmup | False | 0 | `prototype_V0_joint_3seed.jsonl` | `log_V0_joint_3seed.log` |
| V1 separate no-warmup | True | 0 | `prototype_V1_separate_3seed.jsonl` | `log_V1_separate_3seed.log` |
| V2 joint + warmup | False | 50 | `prototype_V2_warmup_joint_3seed.jsonl` | `log_V2_warmup_joint_3seed.log` |
| V3 separate + warmup | True | 50 | `prototype_V3_separate_warmup_3seed.jsonl` | `log_V3_separate_warmup_3seed.log` |

Additional file: `prototype_V0_joint_seed42only.jsonl` — 5 runs from
the initial single-seed V0 test before we switched to 3 seeds. Kept
as historical.

**3-seed paired Δ vs NGD+Adam+ES** (reference:
`investigations/default_gpy_gap_v2/ngd/ngd_results.jsonl`, Phase 3B
sweep with ELBO ES p=200; same 15 (cell, seed) pairs):

| cell | V0 Δ | V1 Δ | V2 Δ | V3 Δ | Adam mean ref |
|------|------|------|------|------|---------------|
| 40   | −0.01 | −0.30 | **+0.01** | −0.10 | 0.756 |
| 38   | +0.06 | −0.01 | −0.02 | **+0.08** | 0.662 |
| 16   | −0.02 | **+0.00** | −0.01 | +0.00 | 0.959 |
| 30   | −0.50 | −0.75 | −0.99 | −0.75 | 0.752 |
| 29   | −0.06 | −0.10 | −0.05 | −0.07 | 0.749 |
| **passing cells (\|Δ\|≤0.05)** | **3/5** | **2/5** | **4/5** | **2/5** | — |

**Per-variant disaster counts** (test_r < 0.3, excluding NaN):

| Variant | disasters / 15 | NaN / 15 | hit 200-iter cap / 15 |
|---------|---------------|----------|-----------------------|
| V0 joint | 2 | 0 | 0 |
| V1 separate | 5 | 1 | 0 |
| V2 warmup joint | 3 | 0 | 0 |
| V3 separate+warmup | 4 | 1 | 7 |

**Cell 30 detailed**, all three seeds (the cell that drove the whole exercise):

| seed | V0 joint | V1 separate | V2 warmup joint | V3 separate+warmup | Adam+ES ref |
|------|----------|-------------|-----------------|--------------------|-------------|
| 42   | 0.744    | 0.000       | −0.157          | 0.000              | 0.732 |
| 123  | −0.207   | 0.000       | −0.405          | 0.000              | 0.751 |
| 789  | 0.227    | 0.000       | −0.145          | NaN                | 0.774 |

Cell 30 fails every LBFGS variant. Adam+ES handles it cleanly (3/3 seeds
in [0.73, 0.78]). Per §42 kill signal "A-collapse on any seed",
triggered on all four variants.

## 55. Mechanism analysis per variant

### V0: joint LBFGS, no warm-up

The first variant tried (pre-registered default per §40). Joint LBFGS
optimizes (kernel + A + λ₀) on a single `.step()` per outer iter with
`lbfgs_max_iter=20` strong_wolfe.

Observed on cell 30 seed 789: LBFGS takes a big step at iter 1 (A
drops from 0.01 to 0.013, β jumps from init 0.10 to 0.15), then
**freezes** for all 87 subsequent outer iters. A ends at 8.7e-4 →
test_r = 0.23. LBFGS closure hits `+inf` rejection 99% of outer iters
(line search probes invalid regions), but still finds valid points;
the final point just happens to be a near-collapsed basin.

**Why freezing?** LBFGS's Hessian approximation latches onto a
direction at iter 1 that is orthogonal to any useful update once q has
been updated (via NGD). Subsequent line searches find no improvement
along that direction. This is the H2 prediction from §39.

### V1: separate LBFGS, no warm-up

Split the joint LBFGS into two separate LBFGS optimizers: one over
kernel hyperparams, one over (A, λ₀). Hypothesis (user): A-kernel
coupling is the root cause; separating them cleans the problem up
(vargp's E→M→F pattern).

Observed: **worse than V0**. Cell 30 now collapses on 3/3 seeds
(A ∼ 1e-9, fully trivial predictions, test_r = 0.000). Cell 40 also
collapses on 2/3 seeds; 1 NaN.

**Mechanism**: the Poisson-exp likelihood has a trivial local
optimum at A=0 where `f_mean = exp(λ₀) = const`, and ELBO is
locally maximized by setting `λ₀ = log(mean(r))`. The joint LBFGS
had a *weakly* constraining A-kernel coupling that slowed A's descent
into this basin; once separated, the F-step LBFGS sees only ∂(-ell)/∂(A, λ₀)
and line search hits A=0 in a handful of evals.

At zero-init q, ∂(-ell)/∂A ≈ A · Σ σ² · f_mean (penalty only; signal
term vanishes because μ ≈ 0). So the gradient pushes A to 0, and line
search finds the A=0 optimum in 1-2 probes.

### V2: joint LBFGS + 50-iter NGD-only warm-up

Hypothesis (Pietro): warm-starting q(λ̃) gives μ structure correlating
with r, so ∂(-ell)/∂A = -Σ μ(r−f_mean) + Σ A σ² f_mean gains a
non-vanishing signal term that counteracts the A → 0 attractor.
Implementation: freeze kernel + likelihood `requires_grad=False` for
50 NGD steps, then re-enable and run the main NGD+LBFGS loop.

Observed: **frontier of the investigation**. 4/5 prototype cells pass
the §40 gate. Cells 16/29/38/40 are within ±0.05 of Adam mean.
Cell 30 STILL disasters (3/3 seeds now NEGATIVE test_r,
−0.157, −0.405, −0.145).

**Why 4/5 works**: warm-up drops loss ~65% on each cell (e.g. 8657 →
2973 on cell 30 seed 789). For cells where the init kernel is
reasonable, q acquires signal, and ∂(-ell)/∂A acquires the
counter-attractor term. A stabilizes at a non-trivial value.

**Why cell 30 still fails**: warm-up fits q *to the bad init kernel*.
Once LBFGS then moves the kernel, q can't re-adapt fast enough (1
NGD step per outer iter). The post-warmup trajectory lands in a
region where the kernel gradient pushes toward a direction q can't
follow, ELBO regresses, and ES fires early on a worse-than-init
state. Note cell 30 seed 42 went 0.74 (V0) → −0.16 (V2), i.e. V2
*destroyed* the one seed that worked in V0.

### V3: separate LBFGS + warm-up

The obvious combination. Hypothesis: warm-up defuses the A→0
attractor *and* separation prevents A-kernel coupling in the
kernel LBFGS. Together they should work.

Observed: **not the magic combo**. Cell 30 collapses on 3/3 seeds
again (A → 0, identical behavior to V1). Cell 40 seed 789 regresses
from V2's 0.83 to V3's 0.25. Cell 29 seed 42 regresses from 0.79
to 0.53. 7/15 runs hit the 200-iter cap (stopping-behaviour worse
than other variants). Disasters: 4/15 (worse than V2's 3/15).

**Mechanism**: warm-up defusing was effective in V2 *because the
joint closure couples A with kernel*. Once separation exists, the
F-step LBFGS sees only the likelihood's A=0 local optimum — the
warm-up state of q is irrelevant because ∂(-ell)/∂(A, λ₀) still
has A=0 as a near-optimal point once line search probes there.
Essentially V3 = V1-like A-collapse with a wasted warm-up cost.

## 56. Synthesis: the dominant failure mode is structural

Across all four variants, the pattern on cell 30 is consistent:

- **Separate F-step LBFGS** (V1, V3): A goes to ~0 in 1-5 evals and
  stays there.
- **Joint LBFGS without warm-up** (V0): A drifts to 0.001 and freezes.
- **Joint LBFGS with warm-up** (V2): warm-up forces q to fit bad
  kernel, post-warmup trajectory ends in a worse place than init.

Adam does not have these problems because:
1. Adam's step is `lr * grad / sqrt(v)`, bounded by ~lr per step. A
   can't jump to 0 in one iter; requires ~100 iters of consistent
   negative grad, which fails to occur in practice (other gradients
   shift q and the kernel, changing the A landscape).
2. Adam's second-moment tracking adapts step sizes per-param — A with
   a small gradient gets a small step.

LBFGS + strong_wolfe explicitly seeks the line-search optimum, which
IS the A=0 trivial optimum when q is poorly informed.

**This is not a numerical or tuning issue — it is a property of the
Poisson-exp likelihood landscape interacting with exact line-search
second-order methods.**

The cells where LBFGS works (16, 29, 38, 40 under V2) share the
property that the init kernel is reasonable enough for q to pick up
signal during warm-up. The cells where LBFGS fails (notably 30 on
this dataset) have an init kernel mismatch that warm-up can't fix
without kernel training, which in turn needs a working LBFGS.
Chicken-and-egg.

## 57. Remaining options — decision matrix for the user

**Stop and write up** (aligned with §40/§42 strict reading):

> "NGD+LBFGS for SVGP hyperparams shows a structural failure mode on
> cell-specific basins (A-collapse at A=0 local optimum of Poisson-exp
> likelihood). V2 (joint LBFGS + 50-iter warm-up) is the best found
> variant, passing the gate on 4/5 prototype cells but failing on
> cell 30 under all four variants tried. The A-collapse mechanism is
> structurally incompatible with LBFGS line search; fixes would
> require either a different optimizer (Adam, which defeats the
> GPyTorch-native-LBFGS premise) or a different parameterization
> (`lambda0_given_A` à la vargp, which is heavier than an
> optimizer-swap investigation). Recommend keeping NGD+Adam as the
> production default. Phase 3E findings informative but do not
> change production mode."

**Run V2 on the full 41 × 3 sweep** (~5 min):

The §40 gate was violated, but §41's decision matrix operates on the
123-paired-obs level. If cell 30 is an isolated failure (say 1-3 cells
fail out of 41), the aggregate mean-Δ might still be near zero. We
need the actual number to know whether V2 is "almost adoptable with
known failures" vs "broken". Cost: ~5 min on the GPU at the V2
walltime we saw (2.7 min for 15 runs → ~22 min for 123 runs, but
likely much less in practice because cap is 200 iters and many
converge much sooner).

**Try remaining untried variants** — my read is these don't address
the dominant failure mode but are cheap to test:

- **V4: LBFGS cadence** (every N NGD iters, e.g. N=5). Addresses V0's
  iter-1 latching. Would only help *joint* mode (separate mode's
  failure is in the F-step itself, not the cadence). Should be
  prototyped as V2-like + cadence. Not expected to fix cell 30 for
  the same reason V2 doesn't.
- **V5: `lbfgs_max_iter=1-3`** (damp the inner LBFGS loop). Similar
  scope to V4 — addresses latching, not A-collapse.
- **Longer warm-up** (100-200 iters instead of 50). Tested only at 50.
  Cheap but unlikely to cross the chicken-and-egg on cell 30.

**Heavier changes the user has de-scoped**:

- Hybrid Adam(A, λ₀) + LBFGS(kernel). Rejected — defeats the native-
  GPyTorch-native premise.
- `lambda0_given_A` reparameterization. Rejected earlier for scope; it
  would change more than the optimizer.
- Damped Newton F-step (à la `eigenspace_fstep`). Same as above —
  heavier than optimizer-swap and essentially re-derives vargp.

## 58. Artifact map (for a future session)

All files under `investigations/default_gpy_gap_v2/ngd_lbfgs/`:

```
ngd_lbfgs_training.py                          # the training loop, parameterised by
                                                # `separate_likelihood` and `n_warmup`
run_prototype.py                               # 5 cells × 3 seeds runner
                                                # monkey-patches ngd_training.train_ngd
                                                # → train_ngd_lbfgs, then calls
                                                # run_single_config with mode='ngd'
run_sweep.py                                   # 41 × 3 runner (written, never run)

prototype_V0_joint_3seed.jsonl                 # V0 results
prototype_V1_separate_3seed.jsonl              # V1 results
prototype_V2_warmup_joint_3seed.jsonl          # V2 results
prototype_V3_separate_warmup_3seed.jsonl       # V3 results
prototype_V0_joint_seed42only.jsonl            # V0 5-run single-seed smoke (pre-3-seed)

prototype_results.jsonl                        # most-recent run (currently V3,
                                                # will be overwritten by next run)

log_V0_joint_3seed.log
log_V1_separate_3seed.log
log_V2_warmup_joint_3seed.log
log_V3_separate_warmup_3seed.log

__pycache__/                                   # .pyc files
```

The current `ngd_lbfgs_training.py` defaults are V3 (separate=True,
n_warmup=50). The file's §51-style docstring records this. To run a
different variant, modify the defaults in the function signature OR
pass overrides via config keys (but run_single_mode.py currently
doesn't forward `separate_likelihood` / `n_warmup` from config —
either pass kwargs to `run_single_config` via the monkey-patch
wrapper, or edit the defaults directly).

Reference baseline (Phase 3B/3C):
- Phase 3B paired reference:
  `investigations/default_gpy_gap_v2/ngd/ngd_results.jsonl`
  (48 runs = 16 cells × 3 seeds, NGD+Adam+ES p=200, seeds {42,123,789}).
  This is what the prototype Δ compares against.
- Phase 3C full sweep (for context on the full-sweep scale):
  `experiments/2026-04-22_ngd_final_verdict_64x64/results.jsonl`
  (123 runs = 41 cells × 3 seeds, seeds {1,2,3}, NGD+Adam+ES).

Pre-fix LOSS_CONVENTION note: both the Phase 3B JSONL and the Phase 3C
JSONL use the mixed pre/post-step logging convention (§47). The loss
curves are biased upward but `test_r` is correct. The ngd_lbfgs
prototype JSONLs here all use the POST-fix pre-step convention — they
are apples-to-apples internally but the loss curves are not directly
comparable to the Phase 3B/3C JSONLs.

## 59. What to tell the user on re-entry

"You were investigating Phase 3E — whether LBFGS can replace Adam on
the NGD hyperparameter step. Four variants were prototyped on 5 cells
× 3 seeds. None passed the §40 gate because of a persistent A-collapse
on cell 30 that all four variants hit. V2 (joint LBFGS + 50-iter
warm-up) was the best, passing 4/5 cells. You were deciding between
writing up Phase 3E as a negative result, running V2 on the full
41×3 sweep to measure the real bad-cell fraction, or trying V4/V5
(expected not to help but cheap). See SCRAPBOOK §51-59 for the full
context and §57 for the decision matrix."


## 60. V5 addendum — damped LBFGS inner loop (2026-04-23)

Ran V5 = V2 config (joint LBFGS + 50-iter warm-up) with `lbfgs_max_iter=1`
instead of 20. Hypothesis: limit strong_wolfe to a single line-search
step per outer iter, preventing the aggressive jump into the A=0 basin.

**Result: fails all 5 cells of the gate, but for a different mechanism.**

| cell | V5 3-seed mean_r | V2 mean_r | Adam ref mean_r | V5 Δ vs Adam |
|------|------------------|-----------|-----------------|--------------|
| 40 | 0.418 | 0.763 | 0.756 | −0.337 |
| 38 | 0.327 | 0.645 | 0.662 | −0.336 |
| 16 | 0.815 | 0.949 | 0.959 | −0.144 |
| 30 | 0.148 | −0.236 | 0.752 | −0.604 |
| 29 | 0.508 | 0.702 | 0.749 | −0.242 |

Disasters (tr<0.3): 3/15. Wall time: 2.4s/run (V2 was 10.7s, Adam 11.5s).
V5 iters mean: 46 (ES fires early because loss barely moves).

**Mechanism shift**: with max_iter=1 the inner LBFGS loop does one
line-search step per outer iter. This prevents the A→0 jump (cell 30
now has A=0.008 instead of 1e-4) but also prevents any meaningful
kernel or likelihood training — **A stays at init (≈0.01) on essentially
every cell**. ES fires at ~40-50 outer iters because ELBO barely moves.

**Trade-off curve for `lbfgs_max_iter`**: 20 = too aggressive (A-collapse
on cell 30), 1 = too damped (no training on any cell). A value in
{3..10} was not tested — it is exactly the "tune the knob until one
cell works" territory. Without a principled a-priori reason to pick a
specific value, testing it would be overfitting.

File: `prototype_V5_damp_joint_warmup_3seed.jsonl`, log:
`log_V5_damp_joint_warmup_3seed.log`.

## 61. Final Phase 3E status (2026-04-23 end-of-session)

Five variants tested — V0, V1, V2, V3, V5. **None pass the §40 pass
gate on the 5 × 3 prototype.** V2 remains the frontier (4/5 cells pass,
cell 30 disasters). V5 fails universally via an orthogonal mechanism
(under-training). V4 (LBFGS cadence) was discussed and deprioritized —
it addresses latching which V2 already addressed via warm-up; not
expected to fix the cell-30 A-collapse.

**Investigation is paused, not closed.** A complete negative writeup
would say: "NGD+LBFGS for SVGP hyperparams has a structural failure
mode (A→0 attractor of Poisson-exp likelihood) that is unavoidable
under any tested LBFGS configuration with reasonable inner-loop budget.
The failure affects a minority of cells (1/5 in the prototype);
`lbfgs_max_iter` tuning trades off catastrophic failure vs.
under-training without a clean middle ground. Keep NGD+Adam as the
production default."

**Remaining untried-but-discussed path**: LBFGS cadence (every N NGD
iters instead of every iter), possibly combined with V2. Not attempted
— see §57. Also: longer warm-up (100-200 NGD iters), init/kernel-init
engineering (rejected as overfitting per the user). Those would be the
next experiments if this is reopened.

---

# Phase 3G — NGD validation pipeline (2026-04-23 → 2026-04-29)

After Phase 3E (LBFGS investigation, deferred), the focus shifted from
"can NGD be improved" to "is NGD ready to be the GPyTorch-native
default?" Three follow-up validation experiments + a deep-dive
investigation. This section is the canonical recap of all post-3E work
on the NGD-as-default question.

## 62. Why this exists

Phase 3C validated NGD vs vargp at *one* operating point (M=250,
n_train=3160, 64×64, fix_Amp=True). To promote NGD as default, we
needed to know:

1. **M-dependence**: does NGD's accuracy hold across M? Does it have
   the same M-degradation as vargp on 9/41 cells at large M?
2. **fix_Amp=False**: does NGD work when Amp is trainable, or did
   `fix_Amp=True` mask a problem?
3. **108×108 dataset**: NGD was only validated on 64×64. The harder
   dataset (vargp mean ≈ 0.745 vs 0.838 on 64×64) may stress the
   algorithm differently.
4. **Low n_train (active-learning regime)**: with M = n_train very
   small (50–300), how does NGD compare?

## 63. M-sweep (`experiments/2026-04-23_ngd_M_sweep_64x64/`)

41 cells × 9 M values {50, 100, 200, 250, 300, 500, 750, 1000, 1500}
× seeds {0, 1, 2} = 1107 runs. Same grid as
`experiments/2026-04-13_M_sweep_64x64/` (vargp), so paired comparison.

Result: **NGD ≥ vargp at every M ≥ 200**, plateau at M ≈ 200–300.
No M-degradation (unlike vargp's 9/41 cells).

| M | NGD mean_r | vargp mean_r | Δ |
|---|---|---|---|
| 50 | 0.815 | 0.817 | −0.002 |
| 100 | 0.828 | 0.830 | −0.002 |
| 200 | 0.843 | 0.838 | +0.005 |
| 300 | 0.845 | 0.838 | +0.007 |
| 500 | 0.848 | 0.834 | +0.014 |
| 750 | 0.853 | 0.838 | +0.015 |
| 1000 | 0.846 | 0.840 | +0.007 |
| 1500 | 0.846 | 0.840 | +0.007 |

Wall-time: NGD ~95–160s/run, vargp ~45–55s/run. NGD 2–3× *slower*
**due to GPU contention** (matteo's grid_search ran simultaneously,
consuming ~8 GB GPU). On clean GPU (Phase 3C measurement), NGD was
4× faster than vargp at M=250.

Reproducible from `experiments/2026-04-23_ngd_M_sweep_64x64/run_sweep.py`.

## 64. Validation pipeline (`experiments/2026-04-28_ngd_validation/`)

Three back-to-back experiments via `run_pipeline.py`. Total wall
9.3h. Each crash-safe.

**Exp B — free Amp on 64×64.** 41 cells × seed=1, M=250, n_train=3160,
fix_Amp=False. Compared to existing vargp `64_elbo_intl_freeAmp_Amp1`
seed=1 baseline (mean=0.838). NGD mean=0.848, **Δ = +0.011**, paired
SEM 0.005. Amp values [1.07, 2.37] — Adam trains correctly. ✓

**Exp A — NGD on 108×108.** 41 cells × seed=1, M=300, n_train=2910,
fix_Amp=False. Compared to existing
`experiments/2026-03-20_massive_allcells_108/` seed=1 baseline.
NGD mean=0.773, vargp mean=0.729, **Δ = +0.043**. Crucially, NGD std
(0.17) is *half* of vargp's (0.29). Particularly handles the 6 cells
with STA edge artefacts better. ✓

**Exp C — low n_train (M = n_train).** 41 cells × seed=1 × {50, 150, 300}
= 246 runs, both modes from scratch (no existing baseline). NGD trails
vargp at every value, gap shrinks with n_train but does not invert:

| M=n_train | vargp mean | NGD mean | Δ | NGD-only disasters |
|-----------|------------|----------|---|--------------------|
| 50  | 0.512 | 0.394 | −0.118 | 7/41 |
| 150 | 0.545 | 0.462 | −0.083 | 7/41 |
| 300 | 0.637 | 0.573 | −0.064 | 2/41 |

⚠️ Real and consistent. Triggered an open investigation (§65).

## 65. Low-n_train deep dive (`investigations/ngd_low_ntrain/`)

Investigation of WHY NGD trails at low n_train, and whether the gap
can be closed by config tuning.

**Mini-investigation (3 cells × 4 iter caps)**: cells 0, 35 (NGD-only
disasters at M=50) and cell 16 (healthy control). Iter caps 100, 300,
500, 1500 with ES disabled. Result: hard cells degrade monotonically
with iter cap (cell 0: cap=100 → +0.19, cap=1500 → −0.10). Easy cell
robust. Confirmed: NGD's first-order Adam over 1500 iters inflates A
on hard cells, where there's not enough data signal to support large A.
Train ELBO improves monotonically (loss 143 → 12 on cell 35) so
`restore_best` selects the most-overfit iteration.

**Hypothesis from mini**: gap is a config issue. Phase 3C's
patience=200, min_delta_rel=1e-2, n_iterations=1500 were tuned for
n_train=3160; at n_train=50 those let NGD overtrain.

**41-cell scaled-ES test (`run_scaled_es.py`, 2026-04-29)**: rerun the
same Exp C grid (41 cells × {M=50,150,300} × seed=1 = 123 runs) with
NGD using vargp-scale ES (patience=15, min_delta_rel=1e-3, cap=200).

**Hypothesis FALSIFIED.** Scaling ES made the gap *wider*:

| M=n_train | Δ default vs vargp | Δ scaled-ES vs vargp |
|-----------|--------------------|--------------------|
| 50  | −0.118 | **−0.157** |
| 150 | −0.083 | **−0.098** |
| 300 | −0.064 | **−0.133** |

ES rate at scaled config: 1/41, 1/41, 0/41 — patience=15 with
min_delta_rel=1e-3 essentially never fires (loss keeps improving by
>0.1% every iter). All runs hit cap=200. Result: cap=200 fixed the
~7 disaster cells (over-trainers) but undertrained the other ~30
cells (which need 500+ iters for the kernel to fit).

**Refined diagnosis**: there's no single iter cap that wins. Some
cells need short caps, others need long. This is an algorithmic
limitation of Adam at low n_train, not a config issue.

**Wall time also unfavorable for NGD at low n_train**:

| M=n_train | vargp | NGD-default | NGD/vargp |
|-----------|-------|-------------|-----------|
| 50  | 47s | 113s | 2.4× slower |
| 150 | 58s | 125s | 2.1× slower |
| 300 | 69s | 140s | 2.0× slower |

Reverses the high-n_train pattern (Phase 3C: NGD 4× faster). Reason:
vargp's LBFGS converges in ~45 outer iters via Newton-like steps;
NGD needs ~1050 iters because Adam is first-order.

**Investigation status: OPEN.** First hypothesis (config tuning)
falsified. Open avenues listed in
`investigations/ngd_low_ntrain/README.md`, summarized:
- Held-out validation ES (would naturally find per-cell stop point;
  caveat: known noisy at small validation sets)
- Mode-switching (vargp early, NGD late)
- Different optimizer (AdamW with weight decay on A?)
- Per-cell adaptive iter cap

## 66. Where the project stands on the NGD-default question

| Operating point | Δ NGD−vargp | NGD speed vs vargp | Verdict |
|-----------------|-------------|--------------------|---------|
| Phase 3C (M=250, n=3160, 64×64, fix_Amp=True) | +0.007 | 4× faster (clean GPU) | ✓ NGD wins |
| M-sweep (n=3160, M ∈ [50,1500], 64×64, fix_Amp=True) | +0.000 to +0.015 | (contested GPU) | ✓ NGD ≥ vargp |
| Exp B (M=250, n=3160, 64×64, fix_Amp=False) | +0.011 | (not measured) | ✓ NGD wins |
| Exp A (M=300, n=2910, 108×108, fix_Amp=False) | +0.043 | (not measured) | ✓ NGD wins, lower variance |
| Exp C (M=n=50,150,300, 64×64, fix_Amp=True) | −0.118, −0.083, −0.064 | NGD 2× **slower** | ✗ vargp wins |

**Headline**: NGD wins for "production" workflows (n_train ≥ ~500).
NGD loses for "active-learning early-iteration" workflows (n_train
< ~300).

`default_params.json["run"]["mode"]` is still `'default_gpy'`. NGD is
*available* as a first-class mode (Phase 3D wiring) but not the
default. Promoting NGD to default before resolving the low-n_train
question would regress active-learning users.

## 67. Files (canonical inventory)

```
experiments/2026-04-23_ngd_M_sweep_64x64/      # M-sweep (1107 runs)
  README.md                                    # final results inline
  run_sweep.py / metadata.json / results.jsonl
  effective_config.json                        # frozen resolved config

experiments/2026-04-28_ngd_validation/          # B+C+A pipeline
  README.md                                    # final results inline
  run_pipeline.py
  run_B_freeamp_64x64.py / results_B.jsonl
  run_C_lowntrain_64x64.py / results_C.jsonl
  run_A_108x108.py / results_A.jsonl
  pipeline.log                                 # huge; grep test_r/Exp.*done

investigations/ngd_low_ntrain/                  # open investigation
  README.md                                    # current hypothesis status
  iter_cap_sweep.py / .jsonl                  # 3-cell mini-sweep (motivation)
  diagnose_overfit.py / overfit_diagnostic.jsonl  # A inflation curves
  run_scaled_es.py / results_scaled_es.jsonl  # 41-cell test (falsified H1)
  analyze_scaled_es.py
```

## 68. What to tell future Claude on re-entry

"You were validating NGD before promoting it to GPyTorch-native default.
The M-sweep (1107 runs) and Exp A/B (82 runs at production points)
showed NGD wins or ties vargp on accuracy at every test EXCEPT low
n_train. Exp C (246 runs at M=n_train=50/150/300) showed NGD trails
vargp by 0.06–0.12 in this regime, AND is 2× slower wall-time.
`investigations/ngd_low_ntrain/` is OPEN — first hypothesis (NGD's
ES patience=200 is wrong for low n_train) was tested with a 41-cell
follow-up and falsified (made the gap wider). Open avenues are
listed in that folder's README."

The default mode hasn't been changed. Don't change it without
resolving §65 (low-n_train) first.

**If asked to merge this branch into `pietro/workingbranch`**, read
`MERGE_CHECKLIST.md` at the gpytorch_porting/ root first. It lists
production files changed, what NOT to do (don't flip default mode),
pre-merge verification commands, gotchas (108×108 default path,
ngd_n_iterations 1000 vs 1500 discrepancy, loss-logging convention),
and a suggested PR structure with explicit known-limitation disclosures.

