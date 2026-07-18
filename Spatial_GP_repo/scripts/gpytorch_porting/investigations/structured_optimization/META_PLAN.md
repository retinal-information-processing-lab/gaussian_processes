# Meta-Plan: Utility-Guided Stimulus Optimization Evaluation

**Created**: 2026-03-31
**Status**: Phase A not started
**Detailed reference**: `EVALUATION_FRAMEWORK.md` (same folder)

This document is the session-start roadmap. Read it first, then consult
EVALUATION_FRAMEWORK.md for metric definitions and technical details.

---

## Overview

Three phases, strictly ordered. Each phase has a decision gate that must pass
before proceeding. A session picks up where the last one left off within a phase.

```
Phase A: Foundation         -->  Phase B: Characterization  -->  Phase C: Comparison
(can we measure reliably?)      (what does each method do?)     (which method is best?)
```

---

## Phase A: Foundation

**Goal**: Build the active learning loop, establish baselines, validate that our
metrics and utility function are trustworthy. Everything downstream depends on this.

### A1. Cell Selection

Screen all 41 cells at 64x64 to determine test_r at reference configuration.
Select 5 cells spanning the performance range:
- 2 high-performing (test_r > 0.8 at full training)
- 2 medium (test_r 0.5-0.8)
- 1 low (test_r < 0.5)

**Status**: NOT STARTED
**Deliverable**: List of 5 cell IDs with reference test_r values

UNDECIDED:
- Training configuration for screening (M, kernel_type, n_iterations, early stopping)
- Whether to use existing massive experiment results (if available at 64x64) or run fresh
- How many seeds per cell for screening (1 is fast but noisy; 3 is safer)

DECIDED:
- 64x64 images (not 108x108 — avoids STA edge artifact for 6 cells)
- 5 cells, spanning performance range (not just "most reliable")

### A2. Active Learning Loop Infrastructure

Build the sequential train-select-respond-retrain loop. This is shared infrastructure
used by random baseline (A3), pool-based selection (A4), and generative comparison (C2).

The loop:
1. Train GP on initial N_init images (e.g., 50)
2. Select next image (strategy varies: random, utility-ranked, generated)
3. Obtain response (real data if image is in pool, oracle if generated)
4. Add (image, response) to training set
5. Retrain GP
6. Record test_r on held-out test set
7. Repeat from step 2

**Status**: NOT STARTED
**Deliverable**: `active_learning_loop.py` — accepts a selection strategy as argument

UNDECIDED:
- Retrain from scratch each time, or warm-start from previous parameters?
  (From scratch = cleaner but slower. Warm-start = faster but introduces path dependence.)
  -> User FIRM decision: warm-start -> we need to discuss how to not invert a N+1xN+1 matrix each time we add an image. we should have code for that


- How often to measure test_r (every image? every 5? every 10?)
  -> User FINAL decision: evaluate test every step. -> to consider: timing training and testing separately so we know the overhead of each
- Fixed held-out test set or expanding? (Fixed = simpler, expanding = more data for test_r)
  -> User FINAL decision: Testing of r is always done only on the fixed test set of repeated images. no held out set for now
- N_init: 50 <- User decision
- End point: 500 images? User decision: ok for now. lets first get first results like this and we will reasses

DECIDED:
- Loop structure as described above
- Same infrastructure for all selection strategies
- Multiple seeds per cell for error bars

### A3. Random Selection Baseline

Run the active learning loop with random image selection from pool.
Uses REAL recorded responses (images are in the PNAS dataset, spike counts available).
No simulation or oracle needed.

This is the null hypothesis: what happens when we add images without any
optimization? All methods must beat this curve.

Note that the "reference performance" irrespective of the seed should be the model trained on the full training set, and full inducing points.

**Status**: NOT STARTED
**Deliverable**: test_r(n) curves with error bands for 5 cells

UNDECIDED:
- Number of seeds per cell -?> User decision: 5 seeds

- How to handle the randomness: fix the sequence of random images per seed, or reshuffle?
    User comment: I have code for this , lets discuss

DECIDED:
- Use real recorded responses from PNAS dataset
- Random selection from the REMAINING pool (images not yet in training set)

### A4. Pool-Based Utility Selection Baseline

Run the active learning loop, but instead of random selection, rank all remaining
pool images by utility and pick the best one. Still uses REAL recorded responses.

**Why this matters**: This tests whether the utility function itself is informative,
BEFORE we ask whether generation methods help. Two questions being separated:
1. Does the utility function correctly identify informative images? (A4 vs A3)
2. Can generation produce images MORE informative than any pool image? (Phase C vs A4)

If utility-ranked selection doesn't beat random, then generation methods have a
deeper problem — the utility function is not measuring what we think.

**Status**: NOT STARTED
**Deliverable**: test_r(n) curves for utility-ranked selection, compared to random baseline

UNDECIDED:
- Which utility to use for ranking: U_std or U_DA?
  U_std is much cheaper (no MC sampling). Start with U_std for ranking,
  -> USer input: this has already been tested extensively but with the old gp implementation, we will make it formal here
- n_cond for DA utility ranking (if used): depends on A5 results
- Recompute utility ranking after each retrain, or use initial ranking throughout?
    User final decision: RECOMPUTE UTILITY. the whole point is for utility rankings to adapt to the model as it gets better.

DECIDED:
- Use same loop infrastructure as A3
- Compare directly against random baseline (same cells, same seeds, same initial split)

--------- Stop here, planning for furhter steps only allowed when useful to consider design choices we need to make now, not specific parameters------


### A5. Multi-Conditioning Prerequisite Experiment

Quick experiment: same method (LBFGS gradient ascent), same GP model.
Vary n_cond = 1, 5, 10, 50, 200 images drawn from pool.

**What we're testing**: Does multi-target conditioning change the optimized image
and stabilize the gradient? Hypothesis: n_cond=1 produces target-specific images
(utility peaks around conditioning image structure); n_cond>=50 produces
population-informative images (consensus structure). Gradient variance should
decrease as 1/sqrt(n_cond).

**Why prerequisite**: If multi-target conditioning fundamentally changes results,
all subsequent phases must use multi-target, not single-target evaluation.

**Status**: NOT STARTED
**Deliverable**: Report showing optimized images, utility values, gradient stability
as function of n_cond. Decision on n_cond for Phases B and C.

UNDECIDED:
- Which model to use (cell, M, ntrain, seed)
- Whether to test with sigmoid-bounded optimization or unconstrained
- Whether to also test the gradient stability quantitatively (variance of gradient
  direction across random conditioning set selections at fixed n_cond)

DECIDED:
- Use LBFGS gradient ascent (method A) — simplest, fastest
- Vary n_cond = 1, 5, 10, 50, 200
- This must complete BEFORE Phase B begins

### A6. Metric Pipeline

Build the Tier 1 metric computation so every generated image gets the full diagnostic
suite automatically.

Metrics (see EVALUATION_FRAMEWORK.md Section 5, Tier 1 for definitions):
- OOB fraction
- norm_C ratio
- U_DA with n_cond >= 50
- U_clipped (utility after pixel clipping)
- Preservation ratio (U_clipped / U_raw)
- cos_C structural distortion
- Utility efficiency (U_DA at fixed norm)
- Mean variance reduction across test set
- Power spectrum distance

**Status**: NOT STARTED
**Deliverable**: `evaluate_image.py` or function library callable from any method script

UNDECIDED:
- Standalone script vs module importable from method scripts
- Whether to include power spectrum distance in v1 or defer

DECIDED:
- All metrics computed for every generated image going forward
- n_cond >= 50 for utility evaluation (pending A5 confirmation)

### Phase A Decision Gate

**Proceed to Phase B when ALL of these are true:**
- [ ] 5 cells selected with reference performance established
- [ ] Active learning loop runs reliably (tested on 1 cell, 50->100, random selection)
- [ ] Random baseline curves exist for all 5 cells with error bands
- [ ] Pool-based utility selection tested (at least 1 cell) — does it beat random?
- [ ] Multi-conditioning experiment complete — n_cond for evaluation decided
- [ ] Metric pipeline computes all Tier 1 metrics without errors

**Key question answered**: Does utility-ranked pool selection beat random selection?
- YES -> utility function is informative, proceed to test whether generation adds value
- NO -> investigate why. Possible issues: wrong utility type, wrong conditioning,
  model too poor, utility doesn't capture what we think. Do NOT proceed to Phase B
  until this is resolved.

---

## Phase B: Method Characterization

**Goal**: Understand each method's behavior, variance, and parameter sensitivity.
Determine whether methods are distinguishable given their intrinsic variance.

### B1. Within-Method Variance

For each method (A-D), fix one model (1 cell, 1 seed), run 50-100 times
varying optimization seed. Compute all Tier 1 metrics for each run.

**Deliverable**: Distribution plots of each metric per method. Mean, std, confidence
intervals. Determines which metrics have within-method variance small enough
to distinguish between methods.

### B2. Parameter Sensitivity

For each method, vary its key parameters and measure effect on Tier 1 metrics:
- Method A: no method-specific params (just optimizer settings)
- Method B: var_threshold (PCA rank), n_components
- Method C: eigen_threshold
- Method D: guidance_scale (w), n_seeds (K), DDIM steps, sampler type

Also vary shared parameters across all methods:
- n_cond (unless fixed by A5)
- Model quality (test different cells from the 5-cell testbed)

**Deliverable**: Sensitivity report. Which parameters matter most for each method?
Which can be fixed at default values?

### Phase B Decision Gate

**Proceed to Phase C when:**
- [ ] Within-method variance characterized for all 4 methods
- [ ] At least 2 Tier 1 metrics have within-method variance < between-method difference
- [ ] Key parameters identified and either fixed or systematically varied

**If no metric discriminates between methods**: methods may be equivalent at this
model quality / configuration. This is itself a finding. Consider:
- Testing at different model quality points (different cells)
- Testing with more training data (ntrain=300 vs 50)
- Reporting equivalence as the conclusion

---

## Phase C: Comparison

**Goal**: Determine which method produces the most informative stimuli, quantitatively.

### C1. Controlled Method Comparisons

Pairwise comparisons on the discriminative metrics from Phase B.
All methods evaluated on the SAME models, SAME conditioning sets, SAME initial conditions.
Only the optimization method varies.

All methods must produce physically realizable images (in-bounds).
Unconstrained gradient ascent (method A without bounds) is a diagnostic upper bound,
not a competing method.

**Deliverable**: Comparison table with confidence intervals. Rankings per metric.

### C2. Simulated Closed-Loop with Generated Images

Run the active learning loop (from A2) with each generation method.
Generated images are NOT in the PNAS pool, so responses must be simulated.

**Response simulation**: Use an oracle GP (trained on all available data for each cell)
to generate synthetic spike counts: y* ~ Poisson(exp(A_oracle * mu_oracle(x*) + lambda0_oracle)).

Compare learning curves (test_r vs n_images) against:
- Random selection baseline (from A3)
- Pool-based utility selection (from A4)
- Each generation method

**Deliverable**: Learning curve plots showing test_r(n) for all strategies,
with error bands, for all 5 cells.

### C3. Pool Selection vs Generation

Direct comparison: does generating a NEW image (methods A-D) produce better
learning curves than selecting the BEST image from the existing pool (A4)?

This separates "utility function works" (already shown in Phase A)
from "generation adds value beyond selection."

If pool selection matches or beats generation, the practical recommendation
is: just rank pool images, don't bother generating.

**Deliverable**: Delta test_r curves (generation - pool selection) with significance tests.

### Phase C Decision Gate (final)

**The project answers these questions:**
1. Does utility-guided selection beat random? (A4 vs A3)
2. Does generation beat pool selection? (C3)
3. Which generation method is best? (C1, C2)
4. Does any of this depend on model quality? (tested across 5 cells)

---

## Cross-Phase Notes

### Computational Budget

Rough estimates (at ~5s per GP retrain):
- A3 (random baseline): 5 cells * 5 seeds * 45 retrains = 1125 fits = ~1.5 hours
- A4 (pool selection): same = ~1.5 hours + utility evaluation cost
- B1 (variance): 4 methods * 100 runs = 400 generated images (no retraining)
- C2 (closed loop): 4 methods * 5 cells * 5 seeds * 45 retrains = 4500 fits = ~6 hours

Total GPU time: ~10-15 hours spread across multiple sessions.
Start with 50->100 (10 steps) for feasibility before committing to 50->500.

### What Can Run in Parallel

Within Phase A: A1 (cell selection) must finish first. Then A3, A4, A5, A6 are
partially parallelizable (A3 and A4 share infrastructure from A2).

Phase B is mostly independent per method — could run B1 for methods A and D
in parallel if GPU allows.

### Session Protocol

Each session:
1. Read this META_PLAN.md (30 seconds — know where we are)
2. Read EVALUATION_FRAMEWORK.md if you need metric definitions or technical details
3. Check the status markers in the current phase
4. Pick up where the last session left off
5. Update status markers before ending

### Relationship to EVALUATION_FRAMEWORK.md

This document = WHAT to do and in WHAT ORDER (roadmap).
EVALUATION_FRAMEWORK.md = HOW to do it and WHY (reference).

Do not duplicate content between them. If you need metric definitions,
confound analysis, or mathematical details, see EVALUATION_FRAMEWORK.md.

---

## Status Tracking

**Current phase**: A (Foundation)
**Current step**: Not started — begin with A1 (cell selection)

| Step | Status | Notes |
|------|--------|-------|
| A1. Cell selection | NOT STARTED | |
| A2. AL loop infrastructure | NOT STARTED | |
| A3. Random baseline | NOT STARTED | Depends on A1, A2 |
| A4. Pool-based utility selection | NOT STARTED | Depends on A1, A2 |
| A5. Multi-conditioning experiment | NOT STARTED | Independent of A1-A4 |
| A6. Metric pipeline | NOT STARTED | Independent |
| Phase A gate | -- | |
| B1. Within-method variance | NOT STARTED | |
| B2. Parameter sensitivity | NOT STARTED | |
| Phase B gate | -- | |
| C1. Controlled comparisons | NOT STARTED | |
| C2. Simulated closed-loop | NOT STARTED | |
| C3. Pool selection vs generation | NOT STARTED | |
| Phase C gate (final) | -- | |
