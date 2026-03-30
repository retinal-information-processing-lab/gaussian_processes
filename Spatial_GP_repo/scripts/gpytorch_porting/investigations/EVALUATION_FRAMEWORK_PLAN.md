# Evaluation Framework for Utility-Guided Image Generation

**Created**: 2026-03-30 (thinking session, no code written)
**Status**: Plan only. No implementation started.
**Context**: We have 4 methods for generating informative stimuli. We cannot currently draw trustworthy conclusions about which is best because the evaluation is qualitative and confounded.

---

## 1. The Problems

### 1.1 No valid comparison metric

Raw utility values (U_raw) are not comparable across methods when images have different out-of-bounds (OOB) percentages. The arc-cosine kernel's homogeneity makes utility grow as ~c^1.9 when pixel values are amplified. Methods that push pixels past the physical display range [vmin, vmax] achieve artificially high utility that is physically unrealizable.

The projector clips OOB pixels. Clipping is not just an information loss -- it distorts the spatial structure of the image. Where the optimizer intended a smooth gradient, the neuron sees flat regions pinned to the rail. With 40% OOB, the clipped image is a fundamentally different stimulus from what the optimizer designed.

**Current state**: We compare raw utility values and visually inspect images. This cannot distinguish "genuinely informative stimulus" from "exploited kernel extrapolation."

### 1.2 Single-target evaluation tests the wrong thing

DA utility was designed to condition on a DISTRIBUTION of images: U_DA(x*) = H_marg(x*) - E_{x~p(x)}[H_cond(x* | x)]. Our current tests condition on a single image (the "target"): U_DA(x* | x_target). These measure different things:

- **Single-target**: finds x* maximally correlated with x_target in the GP's feature space. Optimized images "resemble" the target in the RF because the optimizer seeks angular proximity (high rho^2).
- **Multi-target**: finds x* that reduces uncertainty about responses to MANY natural images on average. The optimized image should capture common features shared across the pool, not mimic one specific image.

**We have never evaluated DA utility with more than 1 conditioning image.** The entire "does it look like the target" evaluation paradigm may be testing a degenerate case.

### 1.3 Too many uncontrolled variables

Results vary across: random seed, target image, kernel type, beta (RF size), guidance scale (w), training set size, utility type. We cannot attribute results to the method vs the configuration.

Sources of variance by category:

| Category | Variables | How to handle |
|----------|-----------|---------------|
| Model identity (Tier 1) | kernel, beta, ntrain, M, training seed | Fix across methods |
| Method-specific (Tier 2) | w, n_PCA, n_C_eigen, DDIM steps | Tune per method |
| Evaluation (Tier 3) | random seed, target/conditioning images | Average over |

### 1.4 No ground-truth evaluation

Currently: show optimized image to GP, check utility value. This is circular -- the same GP that guided the optimization evaluates the result. We need an independent evaluation:

- **Simulated closed loop**: use a separate "oracle" GP (trained on full dataset) as a synthetic neuron. The learner GP (trained on a subset) generates stimuli, the oracle provides synthetic responses. Metric: how fast does the learner's test_r improve?

### 1.5 Code is scattered across worktrees

The 4 methods live in 3 different worktrees on 3 branches:

| Method | Location | Branch |
|--------|----------|--------|
| A. Gradient ascent (LBFGS) | main worktree: `investigations/utility/gradient.py` | `pietro/workingbranch` |
| B. PCA subspace | main worktree: `investigations/utility_decompositions/subspace_optimization.py` | `pietro/workingbranch` |
| C. C-eigenspace | main worktree: `investigations/utility_decompositions/subspace_optimization.py` | `pietro/workingbranch` |
| D. Diffusion-guided | approach-d worktree: `investigations/diffusion/guided_reverse.py` | `pietro/approach-d-guided-sampling` |

A fair comparison requires all methods accessible from one codebase. The branches diverged 16 commits ago and must be merged before evaluation work begins.

---

## 2. Proposed Evaluation Metrics

### 2.1 Primary: Clipped utility (U_clipped)

After optimization, clip x* to [vmin, vmax], re-evaluate utility on the clipped image. This is the utility the neuron would actually experience.

### 2.2 Utility preservation ratio

ratio = U_clipped / U_raw

| Ratio | Interpretation |
|-------|---------------|
| > 0.95 | Robust -- almost all utility is physically realizable |
| 0.5 - 0.95 | Moderate OOB exploitation |
| < 0.5 | Method relies fundamentally on extrapolation |

### 2.3 Structural distortion under clipping

cos_C(x*_raw, x*_clipped) = x*_raw^T C x*_clipped / (||x*_raw||_C * ||x*_clipped||_C)

Measures whether clipping changed the spatial pattern the kernel cares about (direction in C-space) or only the amplitude. Close to 1.0 = structure preserved.

### 2.4 Angular similarity to target (single-target only)

cos_C(x*, x_target) -- directly related to rho^2 in the DA utility formula. Separates directional alignment (scientifically meaningful) from amplitude amplification (the divergence pathology).

### 2.5 Simulated closed-loop learning curve (Phase 3)

test_r of the learner GP as a function of number of stimuli added, evaluated against oracle-generated responses.

---

## 3. Proposed Phases

### Phase 0: Multi-conditioning transition experiment

**Question**: How does the optimized image change as we go from 1 to N conditioning images?

**Design**: Same method (LBFGS gradient ascent with sigmoid bounds), same GP model. Vary n_conditioning = 1, 5, 10, 50, 200 images drawn from pool. For each n, examine: final optimized image, utility value, angular similarity to each conditioning image, gradient stability across seeds.

**Why first**: If multi-target conditioning fundamentally changes the optimized image (prediction: it will), the entire single-target evaluation paradigm is invalid. No point running Phases 1-2 under a paradigm we know is wrong.

**Effort**: Small -- only needs to modify how x_samples is constructed in the utility call.

### Phase 1: Add quantitative metrics to existing scripts

**What**: For every optimized image, compute and report: U_clipped, preservation ratio, cos_C(raw, clipped), OOB% in RF.

**Why**: Makes all existing and future runs quantitative at zero extra cost.

**Effort**: Small -- a few utility function calls after optimization.

### Phase 2: Systematic method comparison

**Design**: Fix one GP model (e.g., arc_cosine, beta=0.1, ntrain=50, M=50, cell 8). All methods use sigmoid/clipping constraints to produce in-bounds images. Run 20 conditioning targets x 50 seeds per method. Report: distributions of U_clipped, cos_C, convergence trajectories.

**Key decision**: unconstrained gradient ascent is an upper bound and diagnostic, NOT a competing method. All compared methods must produce in-bounds images.

**Effort**: Medium -- needs all methods runnable from one codebase.

### Phase 3: Simulated closed loop

**Design**:
- Oracle: best-ever GP for a given cell (trained on ALL data, M=200)
- Learner: GP trained on small initial subset (ntrain_init ~ 50)
- Loop: generate stimulus -> clip -> query oracle (Poisson sample) -> add to training -> retrain -> measure test_r on held-out set
- Repeat for T=20 rounds per run

**Average over**: 5 cells x 5 initial splits x 4 methods x 10 seeds = 1000 learning curves of 20 steps each.

**Why last**: Requires all prior phases (metrics, merged codebase, method comparison infrastructure) plus new sequential-experiment code.

**Effort**: Large -- new infrastructure for sequential training and oracle management.

---

## 4. Session Plan and Codebase Organization

### 4.1 What must happen BEFORE any evaluation code is written

**Step A: Merge approach-d into workingbranch.**

The approach-d branch has 4 commits ahead (diffusion investigation + guided_reverse.py). Workingbranch has 16 commits ahead (LSTA, stability fixes, inference package, utility consolidation). These must be merged so all methods are accessible from one branch.

This is a prerequisite for everything. Without it, the evaluation code would need to duplicate or import across worktrees.

**Step B: Decide where evaluation code lives.**

Two options:

| Option | Location | Pros | Cons |
|--------|----------|------|------|
| Investigation folder | `investigations/evaluation/` | Clean separation, easy cleanup | Another investigation folder to maintain |
| Shared module | `evaluation.py` (alongside `acquisition.py`) | Reusable by all scripts, proper infrastructure | Harder to remove if we change approach |

Recommendation: **`evaluation.py`** for the core metrics (U_clipped, preservation ratio, cos_C). These are permanent tools, not a temporary investigation. The simulated closed loop gets its own investigation folder (`investigations/evaluation/`) because it's an experiment, not a library.

### 4.2 Session breakdown

**Session S1 (this session)**: Planning. DONE. This document is the output.

**Session S2: Merge and metrics infrastructure** (~1 session)
- Branch: `pietro/workingbranch`
- Merge `pietro/approach-d-guided-sampling` into `pietro/workingbranch`
- Create `evaluation.py` with core metric functions:
  - `clip_image(x, vmin, vmax)` -> x_clipped
  - `evaluate_clipped_utility(model, likelihood, x_raw, vmin, vmax, ...)` -> dict with U_raw, U_clipped, ratio, OOB%
  - `structural_distortion(x_raw, x_clipped, C_matrix)` -> cos_C value
  - `angular_similarity(x, y, C_matrix)` -> cos_C value
- Wire metrics into `investigations/utility/gradient.py` (LBFGS method)
- Verify f_max guard consistency (> vs >=, unify to >=)
- Test: run one gradient ascent with old pipeline, confirm new metrics compute correctly

**Session S3: Phase 0 -- Multi-conditioning experiment** (~1 session)
- Branch: `pietro/workingbranch` (after S2 merge)
- Location: `investigations/evaluation/multi_conditioning_test.py`
- Modify the utility call to accept N conditioning images
- Run n_conditioning = 1, 5, 10, 50, 200 with LBFGS + sigmoid bounds
- Analyze: does the optimized image converge? Does gradient variance decrease?
- Document findings in `investigations/evaluation/MULTI_CONDITIONING_FINDINGS.md`
- **Decision point**: based on findings, decide whether single-target or multi-target paradigm should be used going forward

**Session S4: Phase 1+2 -- Systematic comparison** (~1-2 sessions)
- Branch: `pietro/workingbranch`
- Add metrics to all method scripts (gradient, subspace, diffusion)
- Ensure all methods use sigmoid/clipping constraints
- Design and run the comparison matrix (20 targets x 50 seeds x 4 methods)
- May need to batch this across multiple GPU runs
- Analysis script in `investigations/evaluation/analyze_comparison.py`

**Session S5: Phase 3 -- Simulated closed loop** (~2 sessions)
- Branch: `pietro/workingbranch`
- Location: `investigations/evaluation/simulated_closed_loop.py`
- Build oracle (best-ever GP for each cell)
- Build sequential training loop
- Run and analyze learning curves
- This is the most complex phase; may need to split into design + execution

### 4.3 Total estimate: 5-6 sessions, spread over time

```
S1 [DONE]  Planning (this session)
S2         Merge + metrics infrastructure        (prerequisite for all else)
S3         Multi-conditioning experiment          (informs Phase 2 design)
   [decision point: single vs multi-target?]
S4a        Wire metrics into all methods          (mechanical)
S4b        Run systematic comparison              (compute-heavy, may run overnight)
S5a        Simulated closed loop infrastructure   (design + implement)
S5b        Run closed loop experiments            (compute-heavy)
```

### 4.4 Anti-patterns to avoid

- **Do NOT create evaluation code in the approach-d worktree.** This worktree should be merged, not extended.
- **Do NOT scatter metric functions across investigation scripts.** Core metrics go in `evaluation.py`. Investigation scripts import from there.
- **Do NOT add new investigation folders in worktrees that will be merged.** The merge will be cleaner if worktree-specific changes are minimal.
- **Do NOT run multi-method comparisons before the merge.** Cross-worktree imports are fragile and unreproducible.
- **Do NOT implement Phase 3 before Phase 0.** If multi-conditioning changes the picture, the closed-loop design may need adjustment.

---

## 5. Open Questions (to resolve during implementation)

1. **Which cells for the simulated closed loop?** Need cells where the "best-ever" GP achieves high test_r (good oracle). Cells with low test_r have unreliable oracles.

2. **Oracle model specification**: Same kernel type as learner, or the best kernel regardless? Using the best kernel creates more realistic misspecification, but confounds kernel comparison.

3. **How to handle the GP retraining cost in Phase 3?** 1000 learning curves x 20 retraining steps = 20,000 GP fits. At ~5s each = ~28 hours. May need to batch, parallelize, or reduce the matrix.

4. **Subspace methods (B, C) status**: Are these fully implemented and runnable? The `subspace_optimization.py` file exists but may be at investigation stage, not production.

5. **Should the evaluation use standard or DA utility?** DA is more informative but more expensive (MC sampling). For Phase 2 (50 seeds x 20 targets x 4 methods = 4000 evaluations), DA utility cost matters.

---

## 6. Dependencies Diagram

```
S1: Planning (this document)
 |
 v
S2: Merge branches + evaluation.py
 |
 +-------+-------+
 |               |
 v               v
S3: Phase 0     S4a: Wire metrics
 |               |
 v               v
 [decision]     S4b: Systematic comparison
 |               |
 +-------+-------+
         |
         v
     S5: Simulated closed loop
```

S3 and S4a can run in parallel after S2 (they're independent). S4b depends on both S3 (to know single vs multi-target) and S4a (metrics wired). S5 depends on everything.

---

*This document should be committed and referenced from CLAUDE.md. It will be the entry point for any session working on the evaluation framework.*
