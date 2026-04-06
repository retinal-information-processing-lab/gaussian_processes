# Possible Optimizations for Variational GP Training

**Context**: The paper gap investigation is resolved (Gap A closed, metric
mismatch explains Gap B). This document collects optimization opportunities
identified during that investigation, aimed at making the model better for
future datasets — not at matching the paper.

**Branch**: Work planned for a new optimization branch off `pietro/workingbranch`.

---

## 1. Remove Amp Parameter (High Priority)

**Problem**: Amp multiplies the C matrix inside the arc-cosine kernel. In the
large-Amp limit (Amp * q_x >> sigma_0^2), the posterior mean mu(x) is
**completely independent of Amp** — it cancels between k(x) and K_tilde^{-1}.
Amp only affects predictions through a second-order variance correction
(0.5 * A^2 * Amp * s_base) and the KL divergence (M/2 * log(Amp)).

This makes Amp a near-unidentifiable parameter: the M-step spends LBFGS
iterations adjusting it with negligible effect on predictions. Different seeds
find different Amp values, introducing variance without improving the model.
The sweep data confirms that fixing Amp=1 **improves** performance.

The paper (Goldin et al. 2023) does not use Amp.

**Math summary** (derived in session 2026-04-06):
- mu(x) = k_base^T K_base_tilde^{-1} m — Amp cancels exactly
- sigma^2(x) = Amp * s_base(x) + u^T V u — linear in Amp
- KL ~ (M/2) * log(Amp) — logarithmic in Amp
- Amp is partially redundant with A through the coupling A^2 * Amp in the
  variance correction, but NOT redundant for the mean prediction

**Non-identifiability reference**: This is a well-known phenomenon in Bayesian
modeling where adding a redundant or near-redundant parameter makes inference
harder despite the model class being strictly larger. The extra dimension
creates a ridge in the objective landscape that slows convergence and increases
sensitivity to initialization and seed. See:
- Gelman et al. (2013) Bayesian Data Analysis, 3rd ed., Chapter 11
  (hierarchical models and non-identifiability)
- Rasmussen & Williams (2006) GPML, Section 5.4.1 (kernel hyperparameter
  identifiability)
- General discussion in Neal (1996) on funnel geometries in hierarchical
  Bayesian models

**Empirical verification before removal**: Using existing free-Amp sweep data
(41 cells x 3 seeds), check that A * sqrt(Amp) has lower coefficient of
variation across seeds than A alone. This confirms the variance-correction
coupling. Also verify that test_r is stable across seeds even when Amp varies
widely.

**Action**: Remove Amp from ArcCosineKernel (fix at 1.0, remove from M-step
optimization, reduce kernel params from 6 to 5). Run 41-cell comparison sweep.

---

## 2. ELBO Convergence as Early Stopping (High Priority)

**Problem**: Current ES uses validation log-likelihood (val_ll), which is
noisy due to:
- A transients from the interleaved F-step (exp(A*mu) swings when A changes)
- Small validation set (250 images) with Poisson noise (many zeros)
- 8% data cost: 250 images held out from training

We also tried val Pearson r and Spearman rho as alternatives. Both are
equally noisy — the issue is not the metric choice but the A transient
affecting all predictions on held-out data.

**Proposed alternative**: Monitor training ELBO staleness. The ELBO =
log_lik - KL is the objective we maximize. The KL term acts as a built-in
regularizer (penalizes the posterior for deviating from the prior), so the
training ELBO already balances fit vs. complexity. Overfitting in the
neural-network sense is unlikely with M/N ~ 0.09.

Stop when: |ELBO(t) - ELBO(t-k)| / |ELBO(t)| < threshold for several
consecutive iterations.

**Advantages**:
- Uses all 3160 images for training (no holdout cost)
- Not affected by A transients (ELBO averages over all N training points)
- Mathematically principled for variational inference
- Already logged in training curves

**Implementation status (in progress, 2026-04-06)**:
- 'elbo' added as an `es_metric` option in `eigenspace_training.py` and
  `gpy_training.py`. Triggers on relative improvement of `-train_loss`
  (= ELBO), same patience logic as val_ll/val_r ES.
- `n_val_split=0` support added to `run_single_mode.py`: when set, no
  validation is carved and all 3160 images are used for training. Only
  safe when ES doesn't need val (i.e., ELBO-based ES).
- Sweep script: `investigations/optimization/run_sweep_elbo_es_64x64.py`
  runs 4-config grid (interleave × fix_Amp) on 64x64, 41 cells, 3 seeds.
  Uses ELBO ES with patience=15, n_val_split=0. Output:
  `investigations/optimization/sweep_64x64_elbo_es_results.jsonl`.

**Note on A_init** (physical constraint, not tuning): interleaved configs
MUST use `A_init=1e-4` because the E-step Newton gradient scales with A,
and A=0.01 causes the first E-step to overshoot and hit the f_mean
stability threshold (verified empirically: 19/19 E-step divergences on
cell 8 with A_init=0.01 + interleave, test_r collapses to 0.64). Non-
interleaved configs must use `A_init=0.01` because LBFGS can't bootstrap
A from 1e-4 in the 10 iterations per EM cycle. See the sweep script
docstring for the full explanation.

**Verification approaches**:

*The ELBO ES sweep itself* (the in-progress sweep above) is the primary
verification: it runs the 4 configs with ELBO ES from scratch and
compares directly with the baselines in
`experiments/2026-04-06_es_sweeps_64x64/README.md`.

*Post-hoc analysis of existing ES data*: The val_ll-based ES sweeps
(`sweep_64x64_es_results.jsonl`, `sweep_64x64_es_p30_results.jsonl`)
embed per-iteration `train_loss` (= -ELBO) and `val_log_lik` curves.
You can post-process to check when ELBO staleness would have stopped —
limited because runs were truncated at 33-50 iters.

*Full post-hoc analysis*: Would require re-running the baseline sweep
(no ES, 80 iters) with curve logging enabled, which wasn't available
when the current baseline was generated. Not needed if the ELBO ES
sweep above produces clean results.

**Existing baseline data**: `experiments/2026-04-06_es_sweeps_64x64/` —
see README for detailed file inventory and reference values table.
Best no-ES baseline: `intl_fixAmp` at test_r=0.8382, 37/41 cells > 0.8.
Current val_ll ES gap: ~0.02 test_r.

**TODO once this investigation is complete**: If ELBO ES works well,
the experiment system and quick-run system need to be updated to
accommodate it:
- `run_experiment.py` + `configs/canonical.yaml` + `configs/quick.yaml`:
  the YAML schema needs to accept `es_metric: elbo` and `n_val_split: 0`.
  Currently `es_metric` is in the yaml (val_ll default) but the full
  elbo+no-val flow is untested via the experiment system.
- `run_single_mode.py` quick-dev path: works already via the CLI
  (`--es-metric elbo --n-val-split 0`), but the default in
  `default_params.json` (`es_metric: val_ll`) should be revisited if
  ELBO ES becomes the recommended default.
- Validation: update `tests/test_early_stopping.py` to include ELBO ES
  test cases.

---

## 3. E-step Convergence Check (Medium Priority)

**Problem**: The E-step runs a fixed number of Newton iterations (default 10,
sweep configs use 50) without any convergence check. Newton's method on this
log-concave problem converges quadratically near the optimum. In later EM
iterations, 3-5 Newton steps may suffice, making the remaining steps wasteful.

With n_estep=50 (the sweep config), this waste is substantial — potentially
90% of E-step compute is spent on iterations that barely change (m, V).

**Action**: Log the E-step gradient norm at each Newton iteration for a few
representative cells. Implement early exit when
||g_b|| < tol * ||g_b_initial||. Verify no performance regression.

---

## 4. F-step Method Comparison (Medium Priority)

**Background**: Two F-step implementations exist:

Standard (non-interleaved):
- Runs once per EM iteration, AFTER all E-step iterations complete
- LBFGS on a 1D problem (A only, lambda0 computed analytically)
- lambda0 analytical formula: lambda0 = log(sum(r)) - log(sum(exp(A*mu)))
- LBFGS is overkill for 1D — converges in 1-2 iterations

Interleaved (damped Newton):
- Runs at EVERY E-step iteration
- Joint 2x2 damped Newton on (A, lambda0) with alpha=0.25
- lambda0 does NOT need joint optimization — it has a closed-form solution
  for any A. The paper uses joint optimization for implementation simplicity,
  not mathematical necessity.

**Key insight**: The performance difference between interleaved and
non-interleaved comes from WHEN the F-step runs (every E-step iter vs once
per EM iter), not from HOW it optimizes (joint vs profile). A third option —
interleaved LBFGS on A with analytical lambda0 — could combine advantages.

**Investigation needed**:
- Time both F-step methods in otherwise-identical runs
- Compare A trajectories (A vs iteration)
- Check if LBFGS F-step converges in 1-2 iterations (confirming it's
  overkill for 1D)
- Test interleaved profile approach: damped step on A only, analytical
  lambda0 after each step

---

## 5. Redundant Kernel Computation (Low-Medium Priority)

**Finding**: The M-step's final LBFGS closure evaluates the kernel matrices
(K_tilde, K, Kvec) with the final kernel parameters. At the start of the
next EM iteration, recompute_eigenspace() recomputes these same matrices
to do the eigendecomposition. This is one redundant full kernel computation
per EM iteration.

For M=250, N=2910, ~2000 active pixels, this is O(N * M * n_pixels) —
significant.

**Action**: Cache the M-step's final kernel matrices and pass them to
recompute_eigenspace(). Requires careful code change to thread the cached
matrices through.

---

## 6. Adaptive F-step Damping (Low Priority, High Complexity)

**Problem**: The fixed alpha=0.25 damping in the interleaved F-step is
conservative. It prevents A from diverging but also slows convergence.
A larger alpha early in training would let A find its value faster,
reducing the transient period that confuses validation-based ES.

**Proposed**: Levenberg-Marquardt style adaptive damping. Try full Newton
step; if ELBO improves, accept and increase alpha; if not, reduce alpha
and retry.

**Effort**: High — needs math derivation, careful implementation, and
validation that it doesn't introduce instability.

**Relevance**: If ELBO-based ES (Investigation 2) works well, the A
transient is no longer a problem and this optimization becomes unnecessary.

---

## 7. M-step Iteration Count (Low Priority)

**Current**: 10 LBFGS iterations on 6 (or 5) kernel parameters.

The LBFGS history_size is set to 100, which is wasteful for 5-6 parameters
(LBFGS only needs history >= dimension for exact Hessian approximation).

10 LBFGS iterations may not fully converge, but in the EM framework partial
convergence is acceptable — the next EM cycle gets another M-step. The sweep
configs use 20 iterations.

**Question**: Is 10 too many or too few? Profiling needed — if most LBFGS
runs converge in 5 iterations, we can save half the M-step cost.

---

## 8. sigma_0 Parameterization (Low Priority, Controlled Test)

**Background**: sigma_0 enters the kernel squared: v_x = x^T C x + sigma_0^2.
During the paper gap investigation, the parameterization was changed from
exp transform (raw = log(sigma_0), optimizing in log-space) to direct
(identity, raw = sigma_0). This closed a 0.004 gap with vargp_old.

The paper uses exp(sigma_b), which is equivalent to the exp transform.

**Concerns with direct parameterization**:
- sigma_0 appears squared, so the sign is irrelevant (symmetric around 0)
- The gradient vanishes at sigma_0=0 (saddle point)
- Log-space (exp transform) is arguably more natural for a scale parameter

**Concerns with exp parameterization**:
- The 0.004 gap when using exp was measured with all confounds fixed
- May have been fine all along if confounds caused the original stagnation

**Controlled test**: Run 41-cell sweep with three parameterizations:
1. Direct (current): raw_sigma_0 = sigma_0
2. Exp (paper-like): raw_sigma_0 = log(sigma_0), sigma_0 = exp(raw)
3. Squared: raw = sigma_0^2, sigma_0 = sqrt(raw) — optimizes the quantity
   that enters the kernel directly

Compare test_r, convergence speed, and sigma_0 trajectory stability.

---

## Priority Order

1. Remove Amp (simplifies everything downstream)
2. ELBO convergence ES (eliminates validation set, enables using all data)
3. E-step convergence check (easy compute savings)
4. F-step comparison (informs default config choice)
5. Redundant kernel computation (moderate compute savings)
6. M-step iteration count (minor tuning)
7. Adaptive damping (only if ELBO ES doesn't solve the transient problem)
8. sigma_0 parameterization (controlled comparison, low priority)
