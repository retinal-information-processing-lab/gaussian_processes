# Regularization Proposal — Principled Fix for M Degradation + A Initialization

**Status**: CONCLUDED — proposal implemented, validated on 99 runs, result
was a modest partial fix (1/9 degraders saved, Cell 8 improver regressed);
code reverted. See FINDINGS.md "Resolution" section and
`experiments/2026-04-14_hyperparam_prior_validation/` for data. This
document is retained as the record of the design choices and the
σ-sensitivity smoke tests that drove the parameter values, so that a
future session does not need to re-derive them.

**Branch**: `pietro/investigate-M-degradation`
**Date**: 2026-04-14
**Supersedes**: The "hyperparameter overfitting" and "data-adaptive A_init" follow-ups
listed in `FINDINGS.md` Open Follow-Ups and `ToDo.md`.

---

## 1. Problem recap (what the previous session found)

Two coupled issues, same root cause:

1. **Problem 1 — unprincipled A_init.** The ES-sweep config hardcodes `A_init=1e-4`.
   This was empirically tuned to keep the first E-step Newton step safe at
   `N_train=3160`, where the gradient scales as `A·N·mean(r)` and the Hessian
   scales as `A²·N·mean(f)`. For different `N` (e.g. `N=50` in the active loop)
   the choice is arbitrary. Analysis and three candidate data-adaptive formulas
   are in `ToDo.md`.

2. **Problem 2 — hyperparameter overfitting at large M (9/41 cells degrade).**
   The 984-run sweep (`experiments/2026-04-13_M_sweep_64x64/`) confirmed
   monotonic `test_r` degradation with M for 9 cells, with `train_r` rising.
   The ELBO has a KL regularizer on `q(λ̃)` but **no prior on
   hyperparameters**; with more inducing points, the optimizer uses the extra
   capacity to fit training noise.

**Common mechanism**: the joint MLE of (A, λ₀, β, ρ, σ₀) over the ELBO has no
regularizer. More effective capacity (larger M) means a tighter bound, which
reveals finer structure in the responses — including noise. The optimizer
pushes hyperparameters to fit that noise.

---

## 2. Mechanism verification (done in this session)

I re-inspected the sweep JSONL directly (`M_sweep_results.jsonl`) for all 9
degraders and 2 controls. Findings:

### 2.1 Hyperparameter drift with M

Final hyperparameters averaged over 3 seeds, M=50 → M=1500:

| cell | A@50 | A@1500 | ΔlogA | β@50 | β@1500 | λ₀@50 | λ₀@1500 |
|------|------|--------|-------|------|--------|-------|---------|
| 39 | 0.0498 | 0.0556 | +0.11 | 0.0708 | **0.1042** | −2.38 | −2.49 |
| 35 | 0.0834 | **0.1123** | +0.30 | 0.0595 | 0.0663 | −2.81 | −3.00 |
| 16 | 0.0769 | 0.0415 | −0.62 | 0.0430 | **0.1053** | −2.23 | −0.98 |
| 13 | 0.0739 | 0.0762 | +0.03 | 0.0472 | 0.0600 | −0.15 | −0.28 |
| 10 | 0.0464 | **0.0686** | +0.39 | 0.0586 | 0.0626 | −1.05 | −1.16 |
| 33 | 0.0726 | 0.0645 | −0.12 | 0.0507 | 0.0708 | −0.86 | −1.50 |
| 15 | 0.1156 | 0.1104 | −0.05 | 0.0528 | 0.0755 | −2.06 | −2.01 |
| 14 | 0.0552 | **0.0866** | +0.45 | 0.0607 | 0.0646 | −2.01 | −2.20 |
| 27 | 0.1343 | 0.1064 | −0.23 | 0.0490 | **0.0957** | −3.40 | −3.68 |
| 1 (flat) | 0.1001 | 0.0979 | −0.02 | 0.0374 | 0.0457 | −0.23 | −0.38 |
| 8 (improver) | 0.0777 | 0.0335 | **−0.84** | 0.0498 | 0.1028 | −0.64 | −0.35 |

**Observations:**
- **A-drift is NOT universal** across degraders. Cells 35, 10, 14 grow A;
  cells 16, 27, 33 *shrink* A; some are essentially flat.
- **β grows for almost every cell** with M — degraders AND the improver.
  β alone does not distinguish overfitting from beneficial capacity use.
- **The improver (Cell 8) shows the LARGEST A-shrink (−0.84 on log-scale)** —
  its dynamics are opposite to the typical degrader.
- **Overfitting happens via different hyperparameter channels per cell.** It
  is not a single "A grows too much" story. A and β both get used.

### 2.2 ELBO vs test decomposition (Cell 35)

| M | ELL (final) | ~KL | −loss = ELBO | test_r |
|---|-------------|-----|--------------|--------|
| 50 | −1369.7 | 37.2 | −1406.9 | 0.806 |
| 300 | −1343.5 | 46.4 | −1389.9 | 0.781 |
| 1500 | −1319.5 | **60.5** | **−1380.0** | 0.749 |

ELBO genuinely improves by ~27 nats from M=50 to M=1500. The **ELL improves
more than the KL penalizes it** — training likelihood goes up by 50 nats for
only 23 nats of KL increase. The variational KL is simply not a strong enough
prior to counteract the hyperparameter drift at this M.

Warm-init already ruled out a local-optimum explanation — the ELBO *genuinely
prefers the higher-A solution at M=1500*.

### 2.3 Empirical distribution of A at the "safe" operating point (M=50)

Across 41 cells, averaged over 3 seeds each:

- A range: [0.016, 0.171] — only a 10x spread
- log(A) mean: −2.59 (A ≈ 0.075)
- log(A) std: **0.46** (A ranges ~1.6x from the mean per std)
- A quartiles: [0.058, 0.083, 0.102]

**This is a remarkably tight distribution.** At the best operating point, A
has a narrow natural scale — `A ≈ 0.05–0.15` for essentially every cell. This
is the empirical basis for any prior.

### 2.4 log(A_1500 / A_50) across 41 cells

- Min −0.84, max +0.66, mean +0.06

The **typical drift is small on log-scale** (|Δlog A| ≤ 0.5 for most cells).
A prior with σ ≈ 1 on log(A) would be loose enough to let this drift happen
when the data supports it, and tight enough to prevent unbounded excursion.

---

## 3. Proposed fix — principled hyperparameter prior (MAP instead of MLE)

**Core change.** Replace the current joint MLE over hyperparameters with a
*weakly-informative MAP* estimate. Add a log-normal prior on A (and,
optionally, on β) to the ELBO. Everywhere the current code minimizes
`−ELBO`, it will minimize `−ELBO + prior_penalty`.

This is the textbook ML response to hyperparameter overfitting — Bayesian
type-II inference with a weakly-informative prior.

### 3.1 Where the penalty hooks in

Three places minimize a loss that needs the prior added:

| Function | Current loss | Optimizer | Where to add |
|---|---|---|---|
| `eigenspace_fstep.fstep_eigenspace` | `−log_lik` | LBFGS on raw_A | closure loss + analytical grad |
| `eigenspace_fstep.damped_newton_update_A_lambda0` | `−log_lik` (gradient/Hessian) | Damped Newton | add ∂prior/∂A to R (and ∂²/∂A² to H[0,0]) |
| `eigenspace_mstep.mstep_eigenspace_{autograd,analytical}` | `−log_lik + KL` | LBFGS on kernel params | loss + (grad only for β prior if enabled) |

**Complexity**: ~20–40 lines total across three files. Minimal. The prior
has analytical gradient, so no autograd hassle.

### 3.2 Prior formulation

**Log-normal prior on A** (log(A) ~ Normal(μ_A, σ_A²)):

```
    penalty_A(A) = 0.5 · (log A − μ_A)² / σ_A²
    ∂penalty_A/∂A = (log A − μ_A) / (A · σ_A²)
    ∂²penalty_A/∂A² = (1 − (log A − μ_A)) / (A² · σ_A²)
```

The penalty is added to `−ELBO` (loss being minimized). In the damped-Newton
F-step, add `∂penalty/∂A` to `R[0]` and `∂²penalty/∂A²` to `H[0,0]`.

**Optional log-normal prior on β** (symmetric form, same pattern). My
recommendation: defer the β prior until the A prior alone is evaluated.
β-drift is observed in improvers AND degraders, so a β prior may hurt good
cells.

### 3.3 Choosing (μ_A, σ_A) — **KEY DECISION POINT FOR USER**

There are three distinct options. Each has different tradeoffs.

**Option A1 — Fixed weakly-informative prior.**
- μ_A = log(0.05) = −3.00   (mid-range of observed M=50 optima)
- σ_A = 1.0   (covers ~2.7× range either side; effective range ~0.02–0.15)
- Derivation: rough match to the 41-cell empirical distribution, but
  **hardcoded constants — no data peeking**.
- Pro: Fully universal. No cross-cell information used.
- Con: The chosen constants are somewhat arbitrary. Must be justified.
- **User would need to approve the two constants.**

**Option A2 — Empirical-Bayes prior from the 41-cell M=50 sweep.**
- μ_A = mean of log(A_opt) across 41 cells at M=50 = **−2.59**
- σ_A = std of log(A_opt) across 41 cells at M=50 = **0.46**
- Derivation: the 41-cell sweep at small M is the best estimate we have of
  the natural A-scale for this dataset. At M=50 the overfitting mechanism is
  weak (population mean test_r = 0.817).
- Pro: Derived from data; σ tuned to actual biological scale.
- Con: Mild form of data-peeking — the validation cells (including Cell 35,
  Cell 39) contribute to μ_A/σ_A. Can be defended because (a) it's done once
  globally, not per-cell, and (b) the statistic is about the scale of A,
  not about any specific cell's test_r.
- Less conservative (tighter σ_A = 0.46) may over-constrain cells that
  legitimately want larger A.

**Option A3 — Hybrid.** μ_A from A2 (data-derived), σ_A from A1 (looser).
- μ_A = −2.59,  σ_A = 1.0
- Pro: Center the prior where the empirical A-scale is; leave the scale
  loose so legitimate cell-level variation is not suppressed.
- Con: Still uses 41-cell statistic for the center (same data-peeking
  concern as A2, but weaker).

**My recommendation: Option A3.** It uses the 41-cell mean to pick a
reasonable center (this is legitimate empirical-Bayes for a dataset-level
prior), and a loose σ so cells with legitimately different A are not
penalized. Drift at M=1500 is typically |Δlog A| ≤ 0.5, so σ=1.0 allows
that to happen when data supports it, but prevents unbounded A growth
that fits noise.

**Alternative scenario worth considering.** If the user wants full
universality (no data-peeking at all), then **A1 with σ_A = 1.0, μ_A = −3
(A ≈ 0.05)** is a clean default. The test results will look slightly
different but the story is cleaner to publish.

### 3.4 Do we also need a data-adaptive A_init?

**Yes, but separately from the prior.** They serve different purposes:
- `A_init` determines the first Newton step's stability. Analysis is in
  `ToDo.md`: danger threshold is `A²·N ≳ 0.3`.
- The prior determines *where A settles* over the whole training run.

My recommendation: **combine A_init = Formula 2 (safest) with Option A3
prior.** Formula 2 is:

```
    A_init = c / sqrt(N · sum(r_i²))
```

Choosing `c` so that at N=3160 and a typical cell (sum(r²) ≈ 8000), we
recover the current A_init=1e-4:

```
    c = 1e-4 · sqrt(3160 · 8000) = 1e-4 · 5030 ≈ 0.5
```

So `A_init = 0.5 / sqrt(N · sum(r²))`.

Sanity checks for this `c`:
- Cell 1 (N=3160, sum(r²)=19204): A_init = 0.5/sqrt(3160·19204) = 0.5/7791 = **6.4e-5**
- Cell 35 (N=3160, sum(r²)=1648): A_init = 0.5/sqrt(3160·1648) = 0.5/2282 = **2.2e-4**
- Cell 27 (N=3160, sum(r²)=10186): A_init = 0.5/sqrt(3160·10186) = 0.5/5674 = **8.8e-5**
- N=50, sum(r²)=20 (sparse active-loop case): A_init = 0.5/sqrt(50·20) = **0.0158**
  → A²·N = 0.0125. Still safely below 0.3 danger threshold.
- N=50, sum(r²)=125 (5 bursty images at r=5): A_init = 0.5/sqrt(6250) = **6.3e-3**
  → A²·N = 2.0e-3. Safe.

Formula 2 gives `A_init` values that span a **3.5× range across cells**,
compared to the hardcoded 1e-4 which is the same for every cell. This is
more natural given the observed variation in cell response statistics.

**Alternative: keep A_init=1e-4 hardcoded and rely entirely on the prior**
to prevent drift. Simpler. But the A_init issue for small N (active loop)
would remain unaddressed.

### 3.5 What I'm NOT proposing (and why)

- **Per-cell priors or thresholds.** Violates the "no per-cell tuning"
  constraint.
- **Capacity caps (smaller EIGVAL_TOL, smaller M_max, etc.).** That's a
  bandaid masking the capacity — not principled regularization.
- **Validation-based M-step early stopping.** Adds complexity, interacts
  with ELBO ES, requires validation data which we currently don't carve
  (`n_val_split=0` is the default).
- **Prior on β or ρ.** Improvers and degraders both grow β; a β prior is
  likely to hurt improvers. Deferred until A-prior alone is tested.
- **Prior on λ₀.** λ₀ has an analytical closed-form expression given A
  in `lambda0_given_A()`. Any prior on λ₀ would need to modify that
  analytical step, which is more invasive.

---

## 4. Validation protocol

Per Section 7 of the investigation prompt:

**Cells (11 total).** All 9 degraders {39, 35, 16, 13, 10, 33, 15, 14, 27}
plus 2 controls:
- Cell 1 (flat, near ceiling): test_r ≈ 0.98 at all M. Universality check.
- Cell 8 (improver): test_r 0.75 @ M=50 → 0.87 @ M=1500. Must not be hurt.

**M grid**: {50, 300, 1500}. Three seeds each {0, 1, 2}. Total: 11 × 3 × 3 = **99 runs**.

**Config**: identical to the baseline sweep (see
`experiments/2026-04-13_M_sweep_64x64/README.md`) except:
- A_init changes from hardcoded `1e-4` to `0.5/sqrt(N·sum(r²))`
- A prior added with chosen (μ_A, σ_A)

**Success criteria:**
- All 9 degraders: M=1500 test_r ≥ (M=50 test_r − 0.005), **OR** at minimum
  Cells 39 and 35 fully saved.
- Cell 1: |M=1500 test_r − baseline| < 0.005.
- Cell 8: M=1500 test_r ≥ baseline − 0.005 (must not regress).
- Grand mean over 11 cells at M=1500 ≥ baseline mean − 0.003.

**If the A-prior fails for some degraders:** propose adding the β-prior
(secondary iteration). If the prior helps but hurts Cell 8, weaken σ_A.

**Runtime estimate**: 99 runs × ~90s = 2.5 h sequential on one GPU. Fine for
iterating.

**Output location**: a new experiment folder
`experiments/2026-04-YY_hyperparam_prior_validation/` with README.md +
JSONL + the run script, following the convention in
`experiments/2026-04-13_M_sweep_64x64/`.

---

## 5. Minimal code-change summary

If approved, implementation touches three library files plus a config wiring
and one validation script.

| File | Change |
|------|--------|
| `default_params.json` | **NO change** (user rule). New config keys will live in the ES-sweep overrides. |
| `eigenspace_fstep.py` | Add `hyperparam_prior` param to `fstep_eigenspace` and `damped_newton_update_A_lambda0`; add prior term to loss + gradient/Hessian. |
| `eigenspace_mstep.py` | Add `hyperparam_prior` param to both M-step variants; add prior term to closure loss (autograd picks up the gradient automatically). |
| `eigenspace_training.py` | Thread `hyperparam_prior` config dict through to F-step and M-step calls. |
| `run_single_mode.py` | Read `hyperparam_prior` from config; compute `A_init` from `sum(r²)` formula if `A_init='adaptive'`. |
| `experiments/2026-04-YY_*/run_validation.py` | New script: 11 cells × 3 M × 3 seeds with the new config. |

No structural refactoring. No new abstractions. One config flag turns the
feature on/off so all baselines remain reproducible.

---

## 6. Open questions for user (BLOCKING)

Per `.claude/rules/bewary.md`: these must be resolved **before** any code
change.

1. **Prior choice (Section 3.3)** — Option A1 (universal constants),
   A2 (pure EB), A3 (hybrid, my recommendation), or a different proposal?
2. **A_init change (Section 3.4)** — adopt Formula 2 with `c = 0.5` as
   recommended, or keep hardcoded 1e-4 and rely on the prior alone?
3. **β prior** — defer to a second iteration (my recommendation) or include
   from the start? If included, what (μ_β, σ_β)?
4. **Constant `c` in Formula 2** — `c = 0.5` is chosen to reproduce the
   current 1e-4 default at a typical cell (N=3160, sum(r²)≈8000). Is this
   matching-target reasonable, or should `c` be derived from the
   `A²·N < threshold` stability analysis independently?
5. **Validation cell & M grid (Section 4)** — 11 cells × {50, 300, 1500} ×
   3 seeds OK, or do you want different M values (e.g. add M=750)?

Nothing else is being changed. No silent simplifications.

---

## 7. What happens next (after user decides)

1. User approves a (μ_A, σ_A) choice and A_init formula (or proposes
   modifications).
2. I implement the minimal code changes, commit.
3. I create the validation experiment folder with README + run script.
4. Run the validation sweep.
5. If success criteria met: commit results, update FINDINGS.md, update
   ToDo.md (mark both follow-ups resolved), update CLAUDE.md Authoritative
   Sources. Final deliverable.
6. If not met: second iteration. Either weaken/tighten the prior, or
   propose adding the β prior. Second check-in with user before iterating.

---

**End of proposal. Awaiting user decision on Section 6.**
