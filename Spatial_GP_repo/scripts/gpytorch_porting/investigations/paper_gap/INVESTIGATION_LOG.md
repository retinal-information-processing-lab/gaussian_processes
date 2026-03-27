# Investigation: Why Does Our Model Underperform the Paper?

**Branch**: pietro/investigate-paper-gap
**Started**: 2026-03-26
**Baseline**: avg adjusted_r2 = 0.720, 13/41 cells > 0.8 (paper claims 36/41)

---

## Possible Compounding Factors

The gap is likely not a single cause but multiple compounding factors. Each may
contribute a few percent. Listed roughly by category, not priority.

### Optimization / Training Procedure
| # | Factor | Status | Effect |
|---|--------|--------|--------|
| O1 | Inner iteration counts (nEstep=10 vs 50, nMstep=10 vs 20) | TESTED | Small (+0.016 avg) |
| O2 | Outer EM iterations (50 vs 200) | TESTED (old only) | Small (+0.015 avg) |
| O3 | sigma_0 softplus -> exp transform (stuck near init) | FIX APPLIED | sigma_0 still doesn't move much; see exp 5 |
| O4 | Amp exp transform causes runaway growth (16-284x) | TESTED, REVERTED | Exp transform BAD for Amp; softplus restored |
| O5 | F-step stability threshold (ours: max>1000, old: mean>100) | NOT TESTED | Unknown |
| O6 | E-step divergence recovery (old lowers logA, ours reverts) | NOT TESTED | Unknown |
| O7 | Early stopping criteria difference (relative vs absolute) | NOT ISOLATED | Unknown |
| O8 | LBFGS history size in F-step (n_fstep vs fixed 100 in M-step) | NOT TESTED | Likely small |
| O9 | F-step interleaving: paper updates A INSIDE each E-step iter | PAPER ONLY (neither vargp_old nor direct has this) | Unknown (see Finding 18) |
| O10 | Amp parameter: paper has NO Amp, ours adds it to C matrix | CONFIRMED | fix_Amp works; free Amp grows 5-284x |
| O11 | M-step optimizer: paper uses scipy L-BFGS-B, ours torch LBFGS | CONFIRMED DIFFERENT | Unknown |
| O12 | Float64 M-step (paper) vs float32 (ours) | CONFIRMED DIFFERENT | Unknown |
| O13 | LBFGS corrupted by frozen params (fix_Amp broke M-step) | **BUG FIXED** | Filter requires_grad=True |

### Data / Preprocessing
| # | Factor | Status | Effect |
|---|--------|--------|--------|
| D1 | Image resolution (108x108 vs 64x64) | TESTED | No meaningful effect |
| D2 | 64x64 normalization shift (mean=-0.069, std=0.934 vs 0,1) | NOT ISOLATED | Unknown (confounded with D1 test) |
| D3 | Training set size (2910 vs 3160 -- we exclude 250 val images) | NOT TESTED | Likely small |
| D4 | Image normalization method (global z-score) | VERIFIED SAME | Both use global z-score |
| D5 | Downscaling method (skimage downscale_local_mean 8x) | NOT VERIFIED | Unknown |

### Model / Architecture
| # | Factor | Status | Effect |
|---|--------|--------|--------|
| M1 | Inducing point selection (pivoted Cholesky vs random+noise) | CONFIRMED DIFFERENT | Paper uses random + 1e-6 noise |
| M2 | Eigenvalue truncation (paper: NONE, ours: 1e-4) | TESTED 1e-4 to 1e-6 | No effect (n_b matters less than LBFGS bug) |
| M3 | lambda_var clamping (ours: 1e-6, old: none) | NOT TESTED | Likely small |
| M4 | V initialization (both: V=K_tilde, same) | SAME | Not a factor |
| M5 | Kernel bounds (ours: finite bounds, old: unbounded) | NOT HITTING BOUNDS | Not a factor currently |
| M6 | Paper has NO eigenspace projection at all | CONFIRMED DIFFERENT | Unknown; could affect M-step |

### Evaluation / Metric
| # | Factor | Status | Effect |
|---|--------|--------|--------|
| E1 | Adjusted R^2 formula | VERIFIED | Matches paper |
| E2 | Prediction formula (E[exp(A*lambda+lambda0)]) | VERIFIED | Correct |
| E3 | Even/odd split for reliability | VERIFIED | Correct |

### Initialization (NEW -- from paper's GitHub code)
| # | Factor | Status | Effect |
|---|--------|--------|--------|
| I1 | beta init: paper=0.0452 (3.5px RF), ours=0.1 (7.6px RF) | TESTED (exp 4-5) | Large (cells 9,14,18 improve) |
| I2 | rho init: paper=0.0821, ours=0.1 | TESTED (exp 4-5) | Part of I1 test |
| I3 | A init: paper=1e-4, ours=0.01 (100x larger) | TESTED (exp 4-5) | Creates chicken-and-egg with O9 |
| I4 | lambda0 init: paper=-1, ours=+1 (opposite sign) | TESTED (exp 4-5) | Part of I3 test |
| I5 | Amp init: paper=N/A (no Amp), ours=1.0 | SEE O10 | Amp absorbs scale meant for A |

### Formerly Unknown (RESOLVED from paper's GitHub code)
| # | Factor | Old status | Resolution |
|---|--------|-----------|------------|
| U1 | Paper's init values | Was UNKNOWN | RESOLVED: see I1-I5 above |
| U2 | Paper's EM iterations | Was UNKNOWN | RESOLVED: maxiter=80 |
| U3 | Paper's code vs utils.py | Was NOT DIFFED | RESOLVED: paper has no Amp, no eigenspace, different F-step |
| U4 | Paper averages over seeds | Still UNKNOWN | Not addressed |
| U5 | Paper's inducing point selection | Was UNKNOWN | RESOLVED: random + 1e-6 noise |
| U6 | Paper's LBFGS settings | Was UNKNOWN | RESOLVED: scipy L-BFGS-B, default history |

### Interaction effects
- ~~I3 + O9 (chicken-and-egg)~~: FALSIFIED. M-step stagnation was caused by
  LBFGS Hessian corruption (O13), not A being too small.
- O3 + O10: sigma_0 stuck + Amp absorbing scale = kernel expressiveness reduced
- I1 + O10: tight beta + free Amp = Amp explodes to compensate for small kernel values
- O13 + O10: Frozen Amp corrupts LBFGS for ALL other params (bug, now fixed)

---

## Phase 0: Information Gathering

### Status
- [x] Paper methods extraction (Agent A) -- DONE
- [x] Baseline analysis + cell subset selection (Agent B) -- DONE
- [x] Data preprocessing comparison (Agent C) -- DONE

---

## Selected Cell Subset (5 cells for investigation)

| Slot | Cell | adj_r2 | test_r | pred_std | Why |
|------|------|--------|--------|----------|-----|
| Top | 18 | 0.870 | 0.925 | 4.37 | High accuracy, highest firing rate |
| Upper-mid | 14 | 0.769 | 0.869 | 1.19 | Typical "good cell" |
| Lower-mid | 9 | 0.663 | 0.809 | 4.46 | High firing rate but moderate fit |
| Low | 28 | 0.535 | 0.697 | 0.62 | Model struggles here |
| Worst | 39 | 0.354 | 0.587 | 1.38 | Worst cell despite good reliability (0.946) |

---

## Finding 1: Paper Uses 108x108, M=250

The paper uses 108x108 images downsampled from 864x864, NOT 64x64.
They use M=250 inducing points, NOT 2910.
They use N~3160 training images total (3190 minus 30 test).

**Implication**: Our M=2910 run uses ALL training images as inducing points -- zero
sparse approximation error. We should be AT LEAST as good as the paper (M=250).
The fact that we're WORSE means the gap is NOT from sparse approximation. Something
else in our training procedure is wrong.

## Finding 2: Paper Omits All Training Details

NOT in the paper:
- Initialization values for ANY hyperparameter (beta, rho, sigma_0, eps_0, A, lambda0)
- Number of EM iterations
- Number of E-step Newton iterations
- Number of M-step gradient steps
- Optimizer name (just says "gradient descent")
- Learning rate
- Early stopping criteria
- Eigenspace truncation
- Jitter/numerical stability
- Float precision

All these details must come from the CODE, not the paper.

## Finding 3: Paper Has No Separate F-step

Paper describes TWO steps, not three:
- E-step: update m, V
- M-step: update ALL hyperparameters (kernel params + A + lambda0) via "gradient descent"

We separate into E-step + F-step (LBFGS for A, analytical lambda0) + M-step (LBFGS for kernel).
This could cause convergence differences -- joint vs separate optimization.

## Finding 4: The Amp Parameter Question

Paper has 5 kernel hyperparameters: eps_0x, eps_0y, beta, rho, sigma_0.
Our code has 6: eps_0x, eps_0y, beta, rho, sigma_0, AND Amp.

The paper's kernel is defined up to "proportionality" -- no explicit amplitude.
In the paper, the overall scale is controlled by A (gain).
In our code, BOTH A and Amp control scale, creating a potential identifiability issue.

NEEDS VERIFICATION: Does the original varGP() code in utils.py have an Amp parameter?

## Finding 5: 64x64 Dataset Has Shifted Statistics

The 108x108 dataset was globally z-scored (mean=0, std=1).
The 64x64 center crop was NOT re-normalized, so it has mean=-0.069, std=0.934.

For arc-cosine kernel where K(x,x) ~ ||x||^2, this shift could affect kernel values.
The sigma_0 bias term may partially compensate, but this is worth checking.

## Finding 6: 13/41 Cells Hit Early Stopping

13 cells trigger early stopping between iterations 30-47 (out of 50 max).

This doesn't correlate with poor performance -- spans adj_r2 from 0.44 to 0.91.
BUT: are the remaining cells also undertrained at 50 iterations?

## Finding 7: Zero Seed Variance with M=2910

All 3 seeds give identical results because M=n_train=2910 (full training set, no
random subset). Training instability cannot be assessed from these results.

---

## Finding 8: CRITICAL -- Inner Iteration Mismatch

The original varGP() code defaults to:
- nEstep = 50 (Newton steps per E-step)
- nMstep = 20 (LBFGS steps per M-step)
- nFparamstep = 10 (LBFGS steps per F-step)
- maxiter = 50 (EM iterations)

Our default_params.json uses:
- n_estep = 10 (5x fewer Newton steps!)
- n_mstep = 10 (2x fewer M-step LBFGS steps!)
- n_fstep = 10 (same)
- n_iterations = 50 (same)

When we run vargp_old through run_single_mode.py, we PASS our defaults (10/10),
overriding the paper's defaults (50/20). So both modes are equally undertrained.

The original code also has inner E-step convergence (norm < 1e-5 breaks early).
Our vargp_direct always runs all n_estep iterations.

## Finding 9: Additional Code Differences (vargp_old vs vargp_direct)

| Difference | vargp_old | vargp_direct | Impact |
|-----------|-----------|--------------|--------|
| F-step stability | f_mean.mean() > 100 | f_mean.max() > 1000 | Old is 10x more conservative |
| Outer early stop | absolute < 1e-4, window=5 | relative < 0.5%, window=20 | Different criteria |
| sigma_0/Amp param | direct (unconstrained) | softplus transform | Different optim landscape |
| E-step inner stop | norm < 1e-5 | none | Old can stop Newton early |
| lambda_var clamp | none | min=1e-6 | Minor numerical difference |
| Kernel bounds | unbounded | finite (beta, rho, Amp) | Not hitting bounds currently |

## Finding 10: Trained Parameters Analysis

No parameters hitting bounds (beta [0.07, 0.16], rho [0.03, 0.10], Amp [1.1, 3.0]).
sigma_0 barely moves from init (stays ~1.0). Amp learns significantly (1.0 -> ~1.8 avg).
28/41 cells run all 49 iterations, 13 stop early (30-47 iterations).

---

## Priority Hypotheses (UPDATED after code analysis)

1. **CRITICAL: Inner iteration counts** -- 5x fewer E-steps, 2x fewer M-steps than paper defaults.
   This is the single most likely cause. Easy to test.
2. **Undertraining** -- 50 outer EM iterations may be too few. Combined with fewer inner steps,
   total gradient steps is much less than paper's code.
3. **F-step threshold difference** -- our 1000 vs paper's 100 may allow A to overshoot.
4. **sigma_0 stagnation** -- not learning. Softplus transform or insufficient M-steps?
5. **Image resolution** -- 108x108 vs 64x64 (user thinks unlikely, test on 108 too).
6. **Amp identifiability** -- Amp and A both control scale. Paper has Amp too, so probably OK.

---

## Experiment 1: Inner Iteration Count Test

**Hypothesis**: Increasing n_estep from 10 to 50 and n_mstep from 10 to 20
(matching paper's original defaults) will significantly improve performance.

**Config**: 5 cells (18, 14, 9, 28, 39), vargp_direct, 64x64, M=2910, ground-truth init

| Setting | Baseline | Paper-like | Extended |
|---------|----------|------------|----------|
| n_estep | 10 | 50 | 50 |
| n_mstep | 10 | 20 | 20 |
| n_iterations | 50 | 50 | 150 |

---

## Experiment 1 Results: Inner Iteration Count

| Config | Cell 18 | Cell 14 | Cell 9 | Cell 28 | Cell 39 | Avg |
|--------|---------|---------|--------|---------|---------|-----|
| baseline (10/10/50) | 0.870 | 0.769 | 0.663 | 0.535 | 0.354 | 0.638 |
| paper_like (50/20/50) | 0.878 | 0.787 | 0.699 | 0.575 | 0.330 | 0.654 |
| extended (50/20/150) | 0.880 | 0.787 | 0.611 | 0.575 | 0.330 | 0.637 |

**CONCLUSION**: More inner iterations help modestly (+0.016 avg). NOT the main cause
of the gap. Extended training can HURT (cell 9: overfitting, 0.699->0.611).

**Bug**: early_stopping was NOT properly disabled (wrong config key: used
'early_stopping_enabled' instead of 'early_stop'). Some cells stopped early in
extended config. Results are still valid for the paper_like config.

**Updated hypothesis ranking**:
1. ~~Inner iteration counts~~ -> TESTED, small effect (+0.016)
2. **Image resolution (108 vs 64)** -> untested, could be large
3. **Metric computation** -> untested, if wrong invalidates everything
4. **Reimplementation correctness** -> untested, need vargp_old comparison
5. Training overfitting -> observed in cell 9

---

## Experiment 2 Results: Cross-Mode + Cross-Resolution

vargp_old 64x64 crashed on RESULT_JSON (fixed bug: model var unbound for vargp_old).
Extracted from stdout. vargp_direct 108x108 ran normally.

| Config | Cell 18 | Cell 14 | Cell 9 | Cell 28 | Cell 39 | Avg |
|--------|---------|---------|--------|---------|---------|-----|
| baseline (direct,64,M=2910) | 0.870 | 0.769 | 0.663 | 0.535 | 0.354 | 0.638 |
| vargp_old (64,M=2910) | ~0.870 | ~0.686 | ~0.674 | ~0.522 | ~0.334 | ~0.617 |
| vargp_direct (108,M=2910) | 0.862 | 0.765 | 0.717 | 0.510 | 0.361 | 0.643 |

**CONCLUSION**: 108x108 does NOT improve over 64x64 (avg 0.643 vs 0.638).
Resolution is NOT the cause of the gap.

Metric verification: our adjusted_r2 formula MATCHES the paper (Eq. 5). Confirmed.

## Experiment 3 Results: Paper-Exact Configuration (KEY FINDING)

Config: 108x108, M=250, n_train=3160, nEstep=50, nMstep=20, no early stopping.

| Config | Cell 18 | Cell 14 | Cell 9 | Cell 28 | Cell 39 | Avg |
|--------|---------|---------|--------|---------|---------|-----|
| vargp_direct_paper | 0.862 | 0.806 | 0.731 | 0.531 | 0.256 | 0.637 |
| vargp_old_paper | 0.857 | 0.835 | 0.756 | 0.524 | 0.421 | 0.678 |

**vargp_old BEATS vargp_direct** with paper settings (avg +0.041). The gap is
largest on cell 39 (+0.165) and cell 14 (+0.029).

BUT: even vargp_old_paper (avg=0.678, 2/5 cells >0.8) doesn't match the paper
(36/41 >0.8). There are TWO gaps to close:
- Gap A: vargp_direct vs vargp_old (reimplementation difference)
- Gap B: vargp_old vs paper (something else)

## Finding 11: sigma_0 Stagnation in vargp_direct (ROOT CAUSE CANDIDATE)

Trained sigma_0 values:
| Cell | vargp_old | vargp_direct | Gap in adj_r2 |
|------|-----------|-------------|---------------|
| 18 | 1.77 | 1.05 | -0.005 |
| 14 | 1.07 | 1.02 | -0.029 |
| 9 | 1.48 | 1.04 | -0.025 |
| 28 | 1.65 | 1.01 | +0.007 |
| 39 | 1.00 | 0.99 | -0.165 |

sigma_0 LEARNS in vargp_old (up to 1.77) but is STUCK in vargp_direct (~1.0).
Probable cause: softplus parameterization changes LBFGS optimization landscape.
The old code optimizes sigma_0 directly (unconstrained).

Note: cell 39 has BOTH sigma_0 stuck AND the largest performance gap, but
sigma_0 doesn't move in EITHER mode for cell 39. So sigma_0 is not the full story.

## Experiment 4 Results: vargp_old with 200 Iterations

Config: 108x108, M=250, n_train=3160, nEstep=50, nMstep=20, 200 EM iterations.

| Cell | 50 iter | 200 iter | Change | sigma_0 (200) |
|------|---------|----------|--------|---------------|
| 18 | 0.857 | 0.859 | +0.002 | 1.42 |
| 14 | 0.835 | 0.839 | +0.004 | 0.93 |
| 9 | 0.756 | 0.753 | -0.003 | 1.51 |
| 28 | 0.524 | 0.534 | +0.010 | 3.22 |
| 39 | 0.421 | 0.482 | +0.061 | 0.97 |
| Avg | 0.678 | 0.693 | +0.015 | - |

More iterations help modestly for vargp_old (+0.015 avg). Cell 39 benefits most
(+0.061). But still only 2/5 cells > 0.8. Far from paper's 36/41 > 0.8.

Cell 28: sigma_0 moves to 3.22 at 200 iter (was 1.65 at 50 iter). Large learning.
Cell 39: sigma_0 STILL doesn't move (0.97) despite more iterations.

## Finding 12: sigma_0 Softplus -- Root Cause Analysis

The sigma_0 stagnation in vargp_direct is caused by the softplus parameterization
(GPyTorch Positive() constraint). In vargp_old, sigma_0 is optimized directly.

The softplus transform warps the LBFGS optimization landscape:
- Gradient attenuation: d(softplus)/d(raw) = sigmoid(raw) = 0.63 at init
- Curvature distortion: LBFGS Hessian approximation in raw-space doesn't match
  the true Hessian in sigma_0-space
- Scale mismatch: sigma_0 and Amp use softplus, other params are direct

**Minimal fix (Option A)**: Change softplus to exp transform in kernels.py:
```
Positive(transform=torch.exp, inv_transform=torch.log)
```
This is a single-line change per parameter. Needs analytical gradient correction
in eigenspace_mstep.py (sigmoid -> sigma_0).

## Summary: Master Comparison Table

| Config | Cell 18 | Cell 14 | Cell 9 | Cell 28 | Cell 39 | Avg |
|--------|---------|---------|--------|---------|---------|-----|
| baseline (direct,64,M=2910,10/10/50) | 0.870 | 0.769 | 0.663 | 0.535 | 0.354 | 0.638 |
| paper_like inner (direct,64,50/20/50) | 0.878 | 0.787 | 0.699 | 0.575 | 0.330 | 0.654 |
| direct_paper (108,M=250,50/20/50) | 0.862 | 0.806 | 0.731 | 0.531 | 0.256 | 0.637 |
| old_paper (108,M=250,50/20/50) | 0.857 | 0.835 | 0.756 | 0.524 | 0.421 | 0.678 |
| old_paper_200 (108,M=250,50/20/200) | 0.859 | 0.839 | 0.753 | 0.534 | 0.482 | 0.693 |
| **PAPER TARGET** | >0.8 | >0.8 | >0.8 | ? | ? | **36/41>0.8** |

## Hypotheses Eliminated

1. ~~Inner iteration counts~~ -- small effect (+0.016), not the main cause
2. ~~Image resolution (108 vs 64)~~ -- no meaningful difference
3. ~~Metric computation~~ -- formula matches the paper
4. ~~Reimplementation correctness (for M=N)~~ -- modes match when M=n_train
5. ~~More outer iterations~~ -- modest effect with vargp_old (+0.015 at 200 iter)
6. ~~Parameter bounds~~ -- not hitting bounds

## Active Hypotheses

A. **sigma_0 softplus stagnation** -- explains part of Gap A (direct vs old).
   Fix: change to exp transform. Easy to test.
B. **Something else in M=250 regime** -- vargp_direct degrades much more than
   vargp_old when M < n_train. Possibly eigenspace truncation, inducing point
   selection, or other numerical differences.
C. **Gap B partially explained** -- paper's GitHub code reveals MASSIVE differences
   from our defaults (Findings 13-15 below). Our utils.py ADDED Amp to the paper's code.

## Finding 13: Paper's GitHub Code -- Fundamental Differences

Paper's code (Jupyter notebook) vs our code:

| Parameter | Paper | Ours | Impact |
|-----------|-------|------|--------|
| beta init | beta_nat=0.0452 (RF ~3.5px) | 0.1 (RF ~7.6px) | RF 2.2x wider |
| rho init | rho_nat=0.0821 | 0.1 | smoothness 1.2x wider |
| A init | 1e-4 | 0.01 | 100x larger |
| lambda0 init | -1 | +1 | opposite sign |
| Amp | DOES NOT EXIST | 1.0 (learnable) | identifiability issue |
| maxiter | 80 | 50 | 60% more iterations |
| F-step | Newton inside EACH E-step iter | LBFGS once after E-step | structural |
| M-step optimizer | scipy L-BFGS-B | torch LBFGS | different impl |
| Precision | float64 (M-step) | float32 | less precision |
| Eigenspace | NONE | eigenspace projection | fundamental |

**Critical**: Paper has NO Amp. Our utils.py added it.

## Finding 14: F-step Structure (CORRECTED in Finding 18)

Paper: (A, lambda0) updated via damped Newton (alpha=0.25) at EVERY E-step
iteration. This is a PAPER-ONLY feature.

BOTH vargp_old and vargp_direct update A ONCE per EM cycle, after all Newton
steps. The vargp_old `for i_estep in range(1)` loop is hardcoded to 1.
See Finding 18 for the full three-way comparison.

## Experiment 4: Paper Init (Amp free)

| Config | C18 | C14 | C9 | C28 | C39 | Avg |
|--------|-----|-----|-----|-----|-----|-----|
| direct_paper_init | 0.871 | 0.810 | 0.773 | 0.573 | 0.215 | 0.648 |
| old_paper_init | 0.861 | 0.800 | 0.738 | 0.497 | 0.441 | 0.667 |

Amp EXPLODED in vargp_direct (16-284). Exp transform backfired for Amp.

## Experiment 5: Paper Init + Amp Fixed at 1.0

| Config | C18 | C14 | C9 | C28 | C39 | Avg |
|--------|-----|-----|-----|-----|-----|-----|
| direct_fix_amp | **0.890** | 0.792 | **0.789** | 0.379 | 0.232 | 0.616 |

Cell 18 best-ever (0.890). But M-step STUCK: beta=0.0452 unchanged, sigma_0=1.0
unchanged. Only F-step (A, lambda0) learned. Kernel frozen at init.

## Finding 15: LBFGS Hessian Corruption Bug (FIXED)

When fix_Amp=True freezes raw_Amp (requires_grad=False), including it in the
LBFGS parameter list corrupted the Hessian approximation. ALL kernel params
got zero effective step size. The M-step was silently a no-op.

**Fix**: Filter kernel.parameters() to only include requires_grad=True.
Applied to both autograd and analytical M-step paths.

After the fix, kernel params learn again with frozen Amp:
beta 0.0452 -> 0.049-0.098, rho 0.0821 -> 0.025-0.043, sigma_0 moves slightly.

## Experiment 6b: Paper Init + Amp Fixed + LBFGS Fix

| Config | C18 | C14 | C9 | C28 | C39 | Avg |
|--------|-----|-----|-----|-----|-----|-----|
| baseline (our defaults) | 0.870 | 0.769 | 0.663 | 0.535 | 0.354 | 0.638 |
| exp5 (fix_Amp, broken LBFGS) | 0.890 | 0.792 | 0.789 | 0.379 | 0.232 | 0.616 |
| exp6b (fix_Amp, LBFGS fixed) | 0.878 | 0.773 | **0.796** | 0.471 | 0.222 | 0.628 |

M-step now works but avg (0.628) still doesn't beat baseline (0.638).
Paper init values aren't universally better -- cell 9 beta grew to 0.098
(nearly our default 0.1), showing the optimizer wants broader RF.

## Finding 16: Eigenspace Truncation NOT the Issue

Tested EIGVAL_TOL = 1e-4, 1e-5, 1e-6. No difference (all give avg ~0.62).
With tight beta, n_b = 36/250 at 1e-4, 250/250 at 1e-5. Full eigenspace
doesn't help when LBFGS is broken (Finding 15).

## Finding 17: Chicken-and-Egg Hypothesis FALSIFIED

Tested A=0.01 (our default, 100x larger than paper's 1e-4) with fix_Amp.
M-step still frozen (before LBFGS fix). The issue was NOT A being too small
for gradient signal -- it was the corrupted LBFGS.

## Finding 18: Three-Way Codebase Comparison (CRITICAL CORRECTION)

There are THREE distinct codebases, not two:
1. **Paper (GitHub notebook)**: The actual published code. We cannot run it.
2. **vargp_old (utils.py)**: Our APPROXIMATION of the paper, with modifications.
3. **vargp_direct (eigenspace_*)**: Our GPyTorch-based reimplementation.

vargp_old is NOT the paper's code. It was written to replicate the paper but
has significant modifications. Key differences between all three:

### F-step Location (MAJOR CORRECTION)

vargp_old does NOT interleave the F-step inside the E-step. The outer loop
`for i_estep in range(1)` is HARDCODED TO 1 (comment: "NOTE THAT THIS LOOP
IS FAKE. ITS A HARD CODED 1", utils.py line 5649).

| Code | F-step location |
|------|----------------|
| Paper | INSIDE each E-step Newton iter (damped Newton, alpha=0.25) |
| vargp_old | AFTER all Newton steps (LBFGS on logA, once per EM cycle) |
| vargp_direct | AFTER all Newton steps (LBFGS on raw_A, once per EM cycle) |

vargp_old and vargp_direct have the SAME F-step structure. Only the paper
interleaves. F-step interleaving is a paper-only feature neither of our codes has.

### Full Three-Way Comparison

| Feature | Paper (GitHub) | vargp_old (utils.py) | vargp_direct |
|---------|---------------|---------------------|--------------|
| F-step | Inside each Newton iter | After all Newton steps | After all Newton steps |
| F-step method | Damped Newton (2x2 Hessian) | LBFGS on logA | LBFGS on raw_A |
| F-step threshold | N/A (self-regulating) | f_mean.mean() > 100 | f_mean.max() > 1000 |
| Amp | NO | YES (direct, unconstrained) | YES (softplus) |
| sigma_0 | exp(sigma_b) stored as sigma_b | direct (unconstrained) | exp(raw_sigma_0) |
| Eigenspace | NONE (full M x M) | YES (EIGVAL_TOL=1e-4) | YES (same) |
| M-step optimizer | scipy L-BFGS-B | torch LBFGS | torch LBFGS |
| Float precision | float64 | float32 | float32 |
| Diverge recovery | Damped Newton self-reg | Lowers logA actively | Reverts m,V only |
| E-step convergence | Unknown | norm < 1e-5 break | None (runs all) |
| Default nEstep | 50 | 50 (but we pass 10) | 10 |
| Default nMstep | 20 (scipy internal) | 20 | 10 |
| Kernel params | 5 (no Amp) | 6 (with Amp) | 6 (with Amp) |

### What Explains Each Performance Gap

**Gap: vargp_old (0.678) vs vargp_direct (0.638) = +0.040**

Both have same F-step structure. Differences that explain the gap:
- sigma_0: direct vs exp transform (sigma_0 learns in old, stuck in direct)
- Amp: direct vs softplus (old Amp grows to 5-13, direct Amp more constrained)
- Diverge recovery: old actively lowers logA (safety mechanism)
- F-step threshold: old more conservative (mean>100 vs max>1000)

**Gap: paper vs vargp_old = UNKNOWN (can't run paper code)**

Additional differences that could explain further gap:
- F-step interleaving (paper only)
- No Amp (paper only)
- No eigenspace projection (paper only)
- scipy L-BFGS-B (paper only)
- Float64 precision (paper only)

## Current Status

To close Gap A (vargp_direct → vargp_old, +0.040), the most impactful changes:
1. sigma_0 direct parameterization (matching vargp_old)
2. Conservative F-step threshold (mean>100)
3. Divergence recovery (lower logA on instability)

To close Gap B (vargp_old → paper, unknown magnitude):
4. F-step interleaving (paper-only, NEW capability)
5. Remove Amp parameter (match paper architecture)
6. Optionally: remove eigenspace projection when M is small
