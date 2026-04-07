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

## 2. ELBO Convergence as Early Stopping — **DONE (April 2026)**

**Status**: Closed and merged into the codebase as the chosen default.

**Outcome**: ELBO-based ES is the default `es_metric` in
`default_params.json`, `configs/canonical.yaml`, and `configs/quick.yaml`.
The val_ll/val_r/val_rho options were removed entirely. See
`.claude/DECISION_LOG.md` Q32 (decision rationale) and Q33 (the bug fix
in best-tracking semantics that surfaced during this investigation).

**Empirical result** (from the 4-config x 41-cell x 3-seed sweep at 64x64):
- Best ELBO ES config (`intl_fixAmp` + `n_val_split=0` + patience=15):
  test_r = **0.8375**, exp_var = **0.8987**, **37/41 cells > 0.8**,
  mean iters = **37.5** (vs 80 for no ES).
- Best no-ES baseline (`intl_fixAmp`, 80 iters): test_r = 0.8382,
  exp_var = 0.8994, 37/41 cells > 0.8.
- **Gap**: ELBO ES is within **0.0007 mean test_r** of the no-ES baseline
  while saving ~53% compute.
- val_ll ES (the original method): test_r = 0.8143-0.8185 across patience
  values, gap of ~0.020 vs baseline.

Full comparison in `experiments/2026-04-06_es_sweeps_64x64/README.md`.

### Why ELBO works where val metrics failed

The interleaved damped Newton F-step causes A transients in the first
~20 iterations: A jumps from 1e-4 to ~0.03 quickly, which makes
predictions `exp(A*mu + lambda0)` swing dramatically. On the small
validation set (250 images), val_log_lik / val_r / val_rho all become
noisy because individual prediction swings dominate the per-image average.
Patience-based ES on these noisy metrics triggers prematurely for ~27% of
runs (best_iter at iteration 1, restored to a barely-trained model).

The training ELBO is computed over all 3160 training points, so the per-image
A-transient noise averages out. ELBO converges smoothly even during the
A transient, allowing patience-based ES to wait for genuine plateaus.

Diagnostic data showing val_ll instability is in
`experiments/2026-04-06_es_sweeps_64x64/diagnostic_no_es_results.jsonl`
(7-cell diagnostic with full 80-iter curves, intl_fixAmp config).

### Implementation summary

- `eigenspace_training.py` and `gpy_training.py`: `es_metric ∈ {'elbo', 'none'}`.
  When `'elbo'`, `es_value = -train_loss`. When `'none'`, ES is disabled.
  Tracks best_es_value (true argmax) and patience_reference (last meaningful
  improvement) as **two separate state variables** (matching PyTorch
  Lightning's design).
- `run_single_mode.py`: `n_val_split=0` path skips validation carving
  entirely. All 3160 training images used.
- `default_params.json`, `configs/canonical.yaml`, `configs/quick.yaml`:
  defaults `es_metric=elbo`, `n_val_split=0`.
- `tests/test_early_stopping.py`: 10 tests, all pass under new defaults.
- Sweep script: `experiments/2026-04-06_es_sweeps_64x64/run_sweep_elbo_es_64x64.py`
  (492 runs, ~7h sequential GPU).

### Notes / loose ends

**A_init constraint (physical, not tuning)**: Interleaved configs MUST
use `A_init=1e-4` (else the first E-step Newton overshoots — verified
empirically with 19/19 E-step divergences on cell 8 when using A_init=0.01
with interleave). Non-interleaved configs MUST use `A_init=0.01` (else LBFGS
F-step can't bootstrap A in 10 iters). This is documented in the sweep
script docstring.

**Tolerance (`min_delta_rel=0.001`)**: Inherited from val_ll ES, not
empirically motivated for ELBO specifically. The 4-config sweep results
show it's working well in practice (matches baseline within 0.001 test_r),
but it might be improvable. Post-hoc analysis of existing curves shows
~37% of iterations have rel improvement > 0.1%; tighter tolerance
(e.g., 0.0001) might let training run longer in the tail. Revisit if
future ELBO ES experiments underperform.

**Sub-investigation: why does ELBO sometimes decrease?** In 3/32 runs
of the buggy partial sweep, ELBO actually decreased after its peak,
violating the coordinate ascent guarantee. The current best-tracking
fix (Q33) handles this correctly via restoration, but the root cause
deserves a separate investigation. Likely culprits:
- Interleaved damped Newton's fixed alpha=0.25 not guaranteeing ELBO
  increase at every step
- LBFGS M-step line search accepting numerically bad steps
- Kernel hyperparameter changes triggering eigenspace recomputation
Future work: identify WHICH optimization step causes the decrease, add
adaptive damping in the interleaved F-step (reject steps that decrease
ELBO), or tighten LBFGS convergence criteria in M-step.

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
2. ~~ELBO convergence ES~~ — **DONE (April 2026)**: chosen as the default
3. E-step convergence check (easy compute savings)
4. F-step comparison (informs default config choice)
5. Redundant kernel computation (moderate compute savings)
6. M-step iteration count (minor tuning)
7. Adaptive damping / "why does ELBO sometimes decrease?" sub-investigation
   (was originally a fallback for ELBO ES; now a separate question about
   optimizer correctness — see Investigation 2 notes)
8. sigma_0 parameterization (controlled comparison, low priority)
