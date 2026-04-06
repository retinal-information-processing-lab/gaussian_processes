# Investigation: Why Does Our Model Underperform the Paper?

**Branch**: pietro/investigate-paper-gap
**Started**: 2026-03-26
**Baseline**: avg adjusted_r2 = 0.720, 13/41 cells > 0.8 (paper claims 36/41)

---

## Possible Compounding Factors

The gap is likely not a single cause but multiple compounding factors. Each may
contribute a few percent. Listed roughly by category, not priority.

**INVESTIGATION RULE: All experiments MUST use `ip_selection='random'`.**
Pivoted Cholesky is not supported by vargp_old, so it silently falls back to
random -- confounding all mode comparisons. With random IPs the gap between
vargp_direct and vargp_old narrows from 0.042 to 0.014 (Finding 18).

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
| M1 | Inducing point selection (pivoted vs random) | **CONFOUND FOUND** | Pivoted HURTS with tight beta; gap 0.042->0.014 when controlled |
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

## Finding 3: Paper's Optimization Structure (CORRECTED in Finding 18)

The paper TEXT describes two steps (E-step, M-step) but the actual CODE has:
- E-step: Newton on (m,V) with interleaved damped Newton on (A, lambda0)
- M-step: kernel hyperparameters only (NOT A/lambda0) via scipy L-BFGS-B

Our structure: E-step (m,V only) -> F-step (A via LBFGS, lambda0 analytical) -> M-step (kernel).
See Finding 18 for the full three-way comparison.

## Finding 4: The Amp Parameter Question

Paper has 5 kernel hyperparameters: eps_0x, eps_0y, beta, rho, sigma_0.
Our code has 6: eps_0x, eps_0y, beta, rho, sigma_0, AND Amp.

The paper's kernel is defined up to "proportionality" -- no explicit amplitude.
In the paper, the overall scale is controlled by A (gain).
In our code, BOTH A and Amp control scale, creating a potential identifiability issue.

VERIFIED: vargp_old (utils.py) DOES have Amp. The paper's code does NOT.
Amp was added to utils.py as a modification. See Finding 18.

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

## Priority Hypotheses (SUPERSEDED -- see Finding 18 for current status)

~~These were written before the paper's GitHub code was analyzed. Several are wrong.~~

1. ~~Inner iteration counts~~ -> TESTED, small effect (+0.016)
2. ~~Undertraining~~ -> modest effect (+0.015 at 200 iter)
3. F-step threshold -> NOT TESTED, still valid (O5)
4. ~~sigma_0 stagnation via softplus~~ -> exp transform applied, sigma_0 still barely moves
5. ~~Image resolution~~ -> no effect
6. ~~"Paper has Amp too"~~ -> **WRONG**: paper has NO Amp (Finding 13/18)

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

## Experiment 3b Results: vargp_old with 200 Iterations

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

## Finding 12: sigma_0 Parameterization (PARTIALLY ADDRESSED)

The sigma_0 stagnation in vargp_direct was initially attributed to the softplus
parameterization. We applied the exp transform fix (O3).

**Result**: sigma_0 now uses exp(raw_sigma_0) -- same as the paper's code uses
exp(sigma_b). But sigma_0 STILL barely moves in vargp_direct experiments.

In vargp_old, sigma_0 is optimized directly (no transform, unconstrained with
lower bound 0 checked by M-step closure). sigma_0 learns well there (1.0 -> 1.77).

The exp transform is mathematically equivalent to the paper's parameterization,
but vargp_old's DIRECT parameterization (no transform at all) works even better.
This suggests the issue is not just the transform function but possibly the
interaction with other differences (M-step gradient computation, autograd vs
analytical, or other factors). Further investigation needed.

## Summary: Master Comparison Table (all adjusted_r2)

All paper-config rows: 108x108, M=250, n_train=3160, nEstep=50, nMstep=20.

| # | Config | C18 | C14 | C9 | C28 | C39 | Avg |
|---|--------|-----|-----|-----|-----|-----|-----|
| 1 | baseline (direct,64,M=2910,10/10/50) | 0.870 | 0.769 | 0.663 | 0.535 | 0.354 | **0.638** |
| 1 | +paper inner iters (50/20/50) | 0.878 | 0.787 | 0.699 | 0.575 | 0.330 | 0.654 |
| 3 | direct, paper config (our init) | 0.862 | 0.806 | 0.731 | 0.531 | 0.256 | 0.637 |
| 3 | vargp_old, paper config (our init) | 0.857 | 0.835 | 0.756 | 0.524 | 0.421 | **0.678** |
| 3b | vargp_old, 200 iter | 0.859 | 0.839 | 0.753 | 0.534 | 0.482 | 0.693 |
| 4 | direct, paper init, Amp free (exp) | 0.871 | 0.810 | 0.773 | 0.573 | 0.215 | 0.648 |
| 4 | vargp_old, paper init, Amp free | 0.861 | 0.800 | 0.738 | 0.497 | 0.441 | 0.667 |
| 5 | direct, paper init, fixAmp (broken LBFGS) | 0.890 | 0.792 | 0.789 | 0.379 | 0.232 | 0.616 |
| 6b | direct, paper init, fixAmp, LBFGS fixed | 0.878 | 0.773 | 0.796 | 0.471 | 0.222 | 0.628 |
| | **PAPER TARGET** | >0.8 | >0.8 | >0.8 | ? | ? | **36/41>0.8** |

## Hypotheses Eliminated

1. ~~Inner iteration counts~~ -- small effect (+0.016), not the main cause
2. ~~Image resolution (108 vs 64)~~ -- no meaningful difference
3. ~~Metric computation~~ -- formula matches the paper
4. ~~Reimplementation correctness (for M=N)~~ -- modes match when M=n_train
5. ~~More outer iterations~~ -- modest effect with vargp_old (+0.015 at 200 iter)
6. ~~Parameter bounds~~ -- not hitting bounds

## Active Hypotheses (SUPERSEDED -- see Finding 18 "Current Status")

A. ~~sigma_0 softplus stagnation~~ -> exp transform applied (O3). sigma_0 still
   barely moves. Exp transform is necessary but not sufficient. vargp_old uses
   direct (unconstrained) parameterization, which works better.
B. Gap A (vargp_old vs vargp_direct) explained by: sigma_0 direct vs exp,
   Amp direct vs softplus, F-step threshold, divergence recovery. See Finding 18.
C. Gap B (vargp_old vs paper) partially explained: F-step interleaving,
   no Amp, no eigenspace, scipy optimizer, float64. See Finding 18.

## Finding 13: Paper's GitHub Code -- Fundamental Differences

Paper vs BOTH our implementations (vargp_old and vargp_direct share these defaults):

| Parameter | Paper (GitHub) | vargp_old AND vargp_direct | Impact |
|-----------|---------------|---------------------------|--------|
| beta init | beta_nat=0.0452 (RF ~3.5px) | 0.1 (RF ~7.6px) | RF 2.2x wider |
| rho init | rho_nat=0.0821 | 0.1 | smoothness 1.2x wider |
| A init | 1e-4 | 0.01 | 100x larger |
| lambda0 init | -1 | +1 | opposite sign |
| Amp | DOES NOT EXIST | 1.0 (learnable, in both) | identifiability issue |
| maxiter | 80 | 50 | 60% more iterations |
| F-step | Inside each E-step Newton iter | After all Newton steps (both) | paper-only |
| M-step optimizer | scipy L-BFGS-B | torch LBFGS (both) | different impl |
| Precision | float64 | float32 (both) | less precision |
| Eigenspace | NONE | EIGVAL_TOL=1e-4 (both) | fundamental |

**Critical**: Paper has NO Amp. Both vargp_old and vargp_direct added it.
See Finding 18 for the full three-way comparison including differences
BETWEEN vargp_old and vargp_direct.

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

## Finding 18: Inducing Point Selection Confound (CRITICAL)

vargp_old ALWAYS uses random IP selection (line 672: `mode != 'vargp_old'`
bypasses pivoted Cholesky). vargp_direct uses pivoted Cholesky by default.
All prior M=250 comparisons between modes used DIFFERENT inducing points.

With SAME random IPs, the gap narrows from 0.042 to 0.014:

| Cell | direct (random) | old (random) | Diff |
|------|----------------|-------------|------|
| 18 | **0.874** | 0.861 | +0.013 |
| 14 | 0.763 | **0.800** | -0.037 |
| 9 | **0.805** | 0.738 | +0.067 |
| 28 | 0.463 | **0.497** | -0.034 |
| 39 | 0.362 | **0.441** | -0.079 |
| Avg | 0.653 | 0.667 | -0.014 |

Pivoted Cholesky HURTS vargp_direct with tight beta: cell 39 goes from
0.229 (pivoted) to 0.362 (random). The concentrated IP selection doesn't
suit the tight-RF configuration.

## Finding 19: Gap A FULLY CLOSED -- vargp_direct matches vargp_old

Controlled experiment: same random IPs, free Amp, paper init (beta=0.0452,
rho=0.0821, A=1e-4, lambda0=-1), 108x108, M=250, n_train=3160, 80 iters.

Three configurations tested to isolate each factor:

| Cell | direct (exp sig0) | direct (identity sig0) | vargp_old |
|------|-------------------|----------------------|-----------|
| 18 | 0.861 | **0.864** | 0.861 |
| 14 | 0.797 | **0.817** | 0.800 |
| 9 | 0.737 | 0.740 | 0.738 |
| 28 | 0.490 | 0.490 | **0.497** |
| 39 | 0.430 | 0.434 | **0.441** |
| Avg | 0.663 | **0.669** | **0.667** |

With identity (direct) sigma_0 parameterization: **avg difference = +0.002** (within noise).
Cell 14 improved notably (0.797 -> 0.817) because sigma_0 can now learn (0.893 vs stuck at 0.985).

**The original 0.042 gap was caused by three confounds:**

| Source | Contribution | How identified |
|--------|-------------|----------------|
| Inducing point selection (pivoted vs random) | ~0.028 | Finding 18 |
| Frozen vs free Amp | ~0.010 | fix_Amp experiment |
| sigma_0 exp vs direct parameterization | ~0.004 | This experiment |
| **Total explained** | **~0.042** | |

**CONCLUSION: vargp_direct is a correct reimplementation of vargp_old.**
The training algorithms produce equivalent results when given identical inputs.
All differences were in the preprocessing/configuration pipeline, not the math.
- Paper has NO Amp at all -- fix_Amp is the "correct" architecture
  but needs F-step interleaving to work well for all cells
- vargp_old can NOT have fix_Amp (varGP code doesn't support it)

## Finding 20: Full 41-Cell Sweep Results (6 configs x 41 cells x 3 seeds)

Sweep script: `investigations/paper_gap/run_sweep.py`
Results: `investigations/paper_gap/sweep_results.jsonl`, `sweep_summary.txt`

All configs used: fix_Amp=True, ip_selection=random, sigma_0=direct,
108x108, M=250, n_train=3160, ground-truth RF, lambda0=-1, no early stopping.

| Config | beta | A_init | intl | inner | Mean | Median | n>0.8 | n>0.6 |
|--------|------|--------|------|-------|------|--------|-------|-------|
| our_defaults | 0.1 | 0.01 | no | 10/10/50 | 0.678 | 0.717 | 11/41 | 28/41 |
| our+paper_inner | 0.1 | 0.01 | no | 50/20/80 | 0.695 | 0.735 | 13/41 | 29/41 |
| paper_init | 0.0452 | 1e-4 | no | 50/20/80 | 0.700 | 0.759 | 13/41 | 30/41 |
| paper+interleave | 0.0452 | 1e-4 | yes | 50/20/80 | 0.713 | 0.706 | **17/41** | 30/41 |
| **broad+interleave** | 0.1 | 1e-4 | yes | 50/20/80 | **0.730** | **0.751** | 15/41 | **34/41** |
| broad+A01+intl | 0.1 | 0.01 | yes | 50/20/80 | 0.609 | 0.684 | 11/41 | 25/41 |

**Best config: broad+interleave** (beta=0.1, A=1e-4, interleave_fstep=True, 50/20/80).
Mean adj_r2=0.730, 15/41 > 0.8, 34/41 > 0.6.

**Key observations from sweep:**
1. More inner iterations help modestly: our_defaults (0.678) -> our+paper_inner (0.695), +0.017.
2. Paper init alone is comparable: paper_init (0.700) vs our+paper_inner (0.695).
3. Interleaving with tight beta helps: paper+interleave (0.713) vs paper_init (0.700), +0.013.
   Gets most cells above 0.8 (17/41) but some cells collapse with tight beta.
4. Interleaving with broad beta is best overall: broad+interleave (0.730), best mean and n>0.6.
5. A=0.01 + interleaving is unstable: broad+A01+intl (0.609). Damped Newton (alpha=0.25)
   too conservative for A=0.01 -- often converges immediately without updating.
6. The best combo uses our broad beta + paper's small A init + interleaving.

**Paper target (36/41 > 0.8) still far on adj_r2.** Our best is 15-17/41.
BUT see Finding 21 below -- the paper likely reports a different metric.

## Finding 21: Metric Definition Mismatch (PROBABLE)

The paper claims "adjusted R^2 > 0.8 for 36/41 cells" and defines Eq. 5 as:
  adjusted_r2 = (mean_accuracy / sqrt(reliability))^2

Our best config (broad+interleave) gives:
  - 15/41 > 0.8 on adjusted_r2 (the squared Eq. 5 formula)
  - **36/41 > 0.8 on explained_var** (= mean_accuracy / reliability, unsquared)

Evidence that the paper reports the unsquared metric:

1. Figure 2F caption says "explained variance" while y-axis says "adjusted r^2"
2. Figure 2F shows GP data points ABOVE 1.0 -- the squared formula bounds results
   more tightly; the unsquared `accuracy/reliability` exceeds 1.0 more easily
   (needs accuracy > reliability, not accuracy > sqrt(reliability))
3. The only evaluation code in the repo (`regular_cnn.py:get_model_table()`)
   computes `accuracy / reliability` (unsquared) -- stored as
   `explained_variance_fractions_bis`. No code implements the squared Eq. 5.
4. Our 36/41 count matches the paper's claim exactly on the unsquared metric.

The CNN code also computes a VARIANCE-BASED metric:
  `explained_variance_fractions` = (var_total - MSE) / (var_total - var_noise)
which is yet another formula. We did not compute this for our GP fits.

**This is our best hypothesis, NOT confirmed ground truth.** The GP evaluation
code is not in the public repo (it's in a private `pyretina_systemidentification`
package). Ways to verify:
- Refit a CNN on the same data and reproduce Figure 2F
- Fit a GP with flat prior (C=I) and check if both metrics match their Figure 2F
  "GP flat prior" column
- Contact the authors
- Compute all three metric variants and produce a figure matching Fig 2F layout

Full metric comparison: see `investigations/paper_gap/METRICS_COMPARISON.md`

## Current Status

**Gap A FULLY CLOSED.** vargp_direct matches vargp_old within 0.002 avg adj_r2
when given identical inputs. The 0.042 gap was from confounds (Finding 19).

**Gap B likely closed (metric mismatch).** On adjusted_r2 (squared, Eq. 5): our
best is 15/41 > 0.8. On explained_var (unsquared, likely what the paper reports):
our best is **36/41 > 0.8**, matching the paper's claim exactly.

**Permanent code changes made on this branch (pietro/investigate-paper-gap):**
- sigma_0 direct (identity) parameterization -- committed, BUT SEE CAVEAT BELOW
- LBFGS frozen-param filter (M-step) -- committed
- f_mean thresholds: mean>100 + max>500 -- committed
- fix_Amp flag for freezing Amp at 1.0 -- committed
- interleave_fstep flag for damped Newton inside E-step -- committed
- vargp_old model/likelihood reference bug fix -- committed

**CAVEAT: sigma_0 direct parameterization may need reverting.**
sigma_0 appears SQUARED in the kernel: v_x = x^T C x + sigma_0^2. This means:
- The sign of sigma_0 is irrelevant (symmetric around zero)
- The gradient vanishes at sigma_0=0 (saddle point)
- The mathematically natural parameter is sigma_0^2 (or log(sigma_0^2))
- The paper uses exp(sigma_b), effectively optimizing in log(sigma_0) space
- Our exp transform (raw = log(sigma_0)) was mathematically equivalent to the paper
- The +0.006 improvement from direct parameterization was measured AFTER fixing
  the IP selection and Amp confounds, so it's a clean comparison -- but small
- It's possible exp was fine all along and the stagnation we saw earlier was
  caused by the other confounds, not by the transform itself

DECISION (April 2026): Keep direct parameterization. Controlled comparison
(exp vs direct vs optimize sigma_0^2 directly) planned in the optimization
phase. See investigations/optimization/possible_optimizations.md.

**Investigation rules established:**
- ip_selection='random' for all mode comparisons (pivoted silently differs for vargp_old)
- fix_Amp=True for paper comparisons (paper has no Amp parameter)

## Three-Way Codebase Comparison (Finding 18 addendum)

| Feature | Paper (GitHub notebook) | vargp_old (utils.py) | vargp_direct |
|---------|------------------------|---------------------|--------------|
| F-step | Inside each E-step iter (damped Newton) | After all E-steps (LBFGS) | After (LBFGS) or inside (damped Newton via interleave_fstep) |
| Amp | NO | YES (direct, learnable) | YES (softplus, learnable, or frozen via fix_Amp) |
| Eigenspace | NONE | YES (EIGVAL_TOL=1e-4) | YES (same) |
| sigma_0 | exp(sigma_b) | direct | direct (identity, after investigation fix) |
| M-step | scipy L-BFGS-B | torch LBFGS | torch LBFGS |
| Precision | float64 | float32 | float32 |
| IP selection | random + 1e-6 noise | random (forced) | random or pivoted |
| Init A | 1e-4 | from config | from config |
| Init beta | 0.0452 (tight) | from config | from config |
| Init lambda0 | -1 | from config | from config |
