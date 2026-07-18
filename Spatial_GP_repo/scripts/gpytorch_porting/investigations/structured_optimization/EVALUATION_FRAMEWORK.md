# Evaluation Framework for Utility-Guided Stimulus Generation

**Created**: 2026-03-31
**Status**: Planning (discussion session, no code yet)
**Location**: `investigations/structured_optimization/`

---

## 1. Problem Statement

We have multiple methods (A-D) for generating images that maximize GP-based utility.
Current evaluation is qualitative (visual inspection) and uncontrolled (too many free variables).
We need a quantitative framework that lets us draw trustworthy conclusions.

### Methods Under Evaluation

| Method | Description | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| A. Direct gradient ascent | LBFGS on pixel values | Fast, high utility | Unnatural, OOB pixels, norm exploitation |
| B. PCA subspace | Gradient ascent in top-K PCA components | Restricts to data variance directions | Only 2nd-order stats, no naturalness |
| C. C-eigenspace | Gradient ascent in kernel eigenvectors | Convergence aid, kernel-aligned | Reparameterization, not constraint |
| D. Diffusion-guided | DDPM reverse + utility gradient nudging | Natural images, full distribution | Slow, guidance scale tuning, seed lottery |

All methods live on `pietro/workingbranch` (approach-d merged in commit 44c11d9):

| Method | Script |
|--------|--------|
| A | `investigations/utility/gradient_ascent.py` |
| B, C | `investigations/utility/subspace_optimization.py` (`--method pca` / `--method c_eigen` / `--method combined`) |
| D | `investigations/diffusion/guided_reverse.py` |

### Why Raw Utility Comparison Fails

For the arc-cosine kernel, utility scales as ~c^1.9 with image norm (see `da_utility_theory.md`).
Methods that don't constrain norm (A) always achieve higher raw utility than norm-bounded
methods (D). Comparing raw utility across methods with different norm constraints is
meaningless -- it conflates norm inflation with genuine information gain.

---

## 2. Evaluation Target

**What we want to measure**: Expected information gain about the neuron's response function
across the population of natural images.

This is what DA utility approximates. The gold standard test is:
does adding the generated image to training data improve test_r faster than adding
a random image?

### Why active learning needs to work for poor models

Active learning is specifically designed for the regime where the model is poor.
A model trained on 50 images has high uncertainty almost everywhere -- this is exactly
when good stimulus selection matters most. Image 51 needs to be maximally informative
precisely because the model doesn't yet know much.

This means the evaluation framework MUST work across the model quality spectrum.
We cannot restrict testing to well-fit models. The utility function correctly identifies
where uncertainty is highest even for weak models, but the utility landscape is flatter,
making between-method differences smaller and harder to detect.

This motivates using performance-based metrics (delta_test_r) over visual inspection.

### Known approximation issues (deferred)

- Laplace approximation accuracy depends on firing rate regime. Well-characterized in
  the safe regime (z_safe > 2.0, which holds for natural images). Revisit only if
  anomalies appear. Could verify with finite differences if needed.
- Single-target conditioning (n_cond=1) is NOT the same as population conditioning --
  see Section 4 below.

---

## 3. Key Confounds

### 3.1 Norm confound (the biggest one)

For the arc-cosine kernel, K(x,x) ~ ||x||_C^2. Utility scales as ~c^1.9 with image norm.
Methods that don't constrain norm always "win" on raw utility by exploiting this.
This is NOT a bug -- it's an intrinsic property of the DA utility formula with
homogeneous kernels (proof in `proof_divergence_theorems.tex`).

Any method comparison MUST either:
- Normalize for norm (evaluate at fixed ||x*||_C), or
- Use norm-invariant metrics (mean variance reduction), or
- Constrain all methods to produce in-bounds images (evaluate U_clipped)

### 3.2 Conditioning set mismatch

We test DA utility with n_cond=1 (single target), but the formula is designed for
a distribution. Single-target measures "how informative is x* about x_target specifically."
Multi-target (n_cond>=50) measures "how informative is x* about the population."
These have different optima, different gradients, different sensitivities.

We have never systematically evaluated DA utility with n_cond > 1.

### 3.3 Uncontrolled variables

Results vary across: random seed, target image, kernel type, beta (RF size),
guidance scale (w), training set size, utility type, M, cell identity, inducing
point selection, specific training images selected.

| Category | Variables | How to handle |
|----------|-----------|---------------|
| Model identity | kernel, beta, ntrain, M, cell, training seed | Fix across methods |
| Method-specific | w, n_PCA, n_C_eigen, DDIM steps | Tune per method |
| Evaluation | optimization seed, target/conditioning images | Average over |

### 3.4 Convergence confound

Different methods converge at different rates. Comparing at a fixed iteration count
conflates "method quality" with "convergence speed."

### 3.5 Response sampling asymmetry (for Tier 3)

For the random baseline, we can use ACTUAL recorded responses from the PNAS dataset
(the images are in the pool, we have their spike counts). For optimized images, we
MUST simulate responses (they're not in the dataset). To ensure fairness, the
simulated closed-loop should use the same response-generation mechanism for both:
either an oracle GP for both, or real data for the baseline and oracle GP for
optimized images (with the asymmetry documented).

---

## 4. Single-Target to Multi-Target Transition

### The hypothesis

Our single-target sanity check showed that optimizing U_DA(x* | x_target) produces
images whose RF pixels resemble x_target -- the utility landscape peaks around the
conditioning image's structure. The generalization hypothesis: when conditioning on a
SET of images, the utility should "pull" toward natural image regions that are
informative about the population, rather than toward any single target.

This is a hypothesis, not a proven fact. It needs explicit testing.

### What changes with multiple conditioning images

With n_cond=1 (current):
- U_DA(x* | x_target) = H_marg(x*) - H_cond(x* | x_target)
- Gradient points toward images informative about THIS ONE target
- High variance across targets and seeds

With n_cond>=50:
- U_DA(x* | {x_1,...,x_50}) = H_marg(x*) - (1/50) * sum_i H_cond(x* | x_i)
- Gradient is the AVERAGE of 50 single-target gradients
- Variance decreases as 1/sqrt(n_cond)
- More stable, more robust optimum

### Why this is a prerequisite

If multi-target conditioning fundamentally changes the optimized image (prediction: it will),
the entire single-target evaluation paradigm is invalid. Must test this BEFORE running
systematic method comparisons.

Quick experiment: same method (LBFGS gradient ascent), same GP model. Vary
n_cond = 1, 5, 10, 50, 200 from pool. Examine: final optimized image, utility value,
gradient stability across seeds.

### For evaluation

ALWAYS evaluate final utility with a large conditioning set (n_cond >= 50), even if
optimizing with fewer. The single-target utility is a noisy estimate of the population
utility. Report the stable version.

---

## 5. Tiered Evaluation Plan

### Tier 0: Testbed Construction

**Goal**: Build a reliable baseline against which ALL methods are compared.

Components:
1. Select 5 cells spanning the performance range:
   - 2 high-performing (test_r > 0.8 at full training on 64x64)
   - 2 medium (test_r 0.5-0.8)
   - 1 low (test_r < 0.5)
2. For each cell: establish 64x64 model performance at reference configuration (TBD)
3. For each cell: run random-selection active learning baseline
   - Train on 50 images
   - Add random images one-at-a-time up to 500
   - Record test_r(n) curve at each step
   - Multiple seeds to get error bars
4. Establish reference performance (model trained on ALL available images)

**Output**: Baseline test_r(n) curves with error bands for 5 cells.
Any optimized method must beat these curves.

### Prerequisite experiment: multi-conditioning transition

Before Tier 1, run the quick n_cond experiment described in Section 4.
This determines whether single-target or multi-target should be used in Tiers 1-3.
Effort: small (1 method, 1 model, ~5 configurations).

### Tier 1: Fixed-Model Image Diagnostics

**Goal**: Cheap per-image metrics, computed without retraining (seconds per image).

| Metric | What it measures | Notes |
|--------|-----------------|-------|
| OOB fraction | Physical realizability | Hard constraint -- image must be displayable |
| norm_C ratio | ||x*||_C / mean(||x_pool||_C) | Detects norm exploitation |
| U_DA(x*, pool, n_cond>=50) | Population-average DA utility | Use large conditioning set |
| U_clipped | Utility after clipping x* to [vmin, vmax] | The utility the neuron would actually experience |
| Preservation ratio | U_clipped / U_raw | How much the method relies on extrapolation (>0.95 = robust, <0.5 = fundamentally exploits OOB) |
| cos_C(x*_raw, x*_clipped) | Structural distortion under clipping | Whether clipping changed the pattern in C-space or only the amplitude |
| Utility efficiency | U_DA evaluated at x* scaled to mean(||x_pool||_C) | Strips norm exploitation, measures pure angular informativeness |
| Mean variance reduction | Avg posterior var decrease across test set | Approximately norm-invariant (see note below) |
| Power spectrum distance | Spectral naturalness | Radially-averaged 2D FFT vs natural images |

**Note on mean variance reduction**: For the GP, conditioning on a hypothetical
observation at x* reduces posterior variance at test point x_j by:

    Delta_j = k(x_j, x*)^2 / [k(x*, x*) + noise_var]

    Mean delta = (1/N_test) * sum_j Delta_j

For the arc-cosine kernel, the ||x*||_C terms approximately cancel between numerator
(k^2 ~ ||x*||^2) and denominator (K(x*,x*) ~ ||x*||^2), making this metric naturally
insensitive to norm exploitation.

### Tier 2: Within-Method Variance Decomposition

**Goal**: Determine whether methods are distinguishable given their intrinsic variance.

Protocol:
1. Fix model (1 cell, 1 seed)
2. Run each method N=50-100 times (varying optimization seed)
3. Plot distribution of each Tier 1 metric
4. Compute mean, std, and confidence intervals
5. Determine which metrics have within-method variance < between-method difference

If error bars overlap between methods, no evaluation framework will help --
need to reduce variance first (larger conditioning sets, more seeds, better convergence).

### Tier 3: Simulated Closed-Loop

**Goal**: Does optimized selection actually beat random selection?

Protocol (for each generated image x*):
1. Sample y* ~ Poisson(exp(A * mu(x*) + lambda0)) using oracle model
2. Add (x*, y*) to training data
3. Retrain GP from same initial conditions (fixed seed)
4. Measure test_r on held-out test set
5. Repeat 20x with different Poisson samples
6. Compare delta_test_r to random-selection baseline from Tier 0

Start with 50->60 (10 optimized images) to check if signal exists before running 50->500.

---

## 6. Deferred Ideas (not for current framework, may revisit)

- **Hessian analysis**: Compute top-K eigenvectors of the Hessian of U(x) at a reference
  image. These are directions where utility has most curvature. Interesting for
  understanding but not for evaluation (local analysis, expensive, ambiguous interpretation).
- **Frequency-domain analysis**: Radially-averaged 2D power spectrum comparison.
  Cheap diagnostic. Already in Tier 1 metrics as "power spectrum distance."
  Worth implementing once methods are running.
- **Input warping**: Modify kernel to saturate at pixel bounds.
  Separate investigation (see `investigations/input_warping/`), not part of evaluation.

---

## 7. Codebase Status

All methods merged into `pietro/workingbranch`:
- Approach-d merge: commit 44c11d9
- Diffusion model: `diffusion_model_imagenet/` (99.5M UNet, ImageNet pretrained, PNAS finetuned)

n_cond > 1 infrastructure: `subspace_optimization.py` already supports `--n-cond N`.
Tested only a few times. Needs reconsideration for structured analysis.

---

## 8. Open Questions

1. **Cell selection**: Which 5 cells? Need to check 64x64 performance across all 41.
2. **Training config for testbed**: M, ntrain_initial, kernel_type, early stopping params.
3. **Random baseline protocol**: How many seeds? Retrain every image or every N images?
4. **Computational budget**: Full 50->500 loop = ~450 retrains/cell/method. Start with 50->60.
5. **Oracle design**: Same kernel type as learner? Best kernel regardless?
   Same kernel = fair comparison. Best kernel = more realistic misspecification.
6. **Multi-cond for optimization vs evaluation**: Optimize with n_cond=1 but evaluate with
   n_cond=50? Or optimize with n_cond=50 too? The answer depends on the multi-conditioning
   prerequisite experiment.

---

## 9. File Inventory

| File | Purpose |
|------|---------|
| `EVALUATION_FRAMEWORK.md` | This document (master reference) |
| (future) `build_testbed.py` | Select cells and train reference models |
| (future) `random_baseline.py` | Random-selection active learning loop |
| (future) `evaluate_image.py` | Compute all Tier 1 metrics for a generated image |

### Source documents consulted

- `investigations/utility/docs/da_utility_theory.md` -- DA utility math, scaling theorems
- `investigations/utility/docs/subspace_theory.md` -- PCA vs C-eigenspace analysis
- `investigations/utility/docs/subspace_operations.md` -- Three subspace optimization methods
- `investigations/diffusion/motivation_diffusion_guided_optimization.md` -- Why diffusion
- `investigations/diffusion/guided_diffusion_summary.tex` -- DDIM + guidance algorithm
- `investigations/input_warping/INPUT_WARPING_REFERENCE.md` -- OOB pixel problem
- `.claude/handoffs/` -- Session summaries of prior work on each method

---

*This document is the single source of truth for the evaluation framework.*
*Update it as decisions are made. Do not scatter information across other files.*
