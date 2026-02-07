# Handoff: Acquisition Function Utility Investigation

**Branch**: `pietro/acquisition-functions` (branched from `pietro/workingbranch`)
**Date**: 2026-02-06 (session 1), 2026-02-07 (session 2)
**Status**: Utility functions validated; experimental design issues identified and resolved

---

## Session 1 Summary (2026-02-06)

Built `acquisition.py` with `standard_utility()` and `distribution_aware_utility()`.
Ran first validation (M=50, n_train=500, cell 8, seed 123). Results were inconclusive:
FULL_BLACK topped both utility rankings, no separation between standard and dist-aware.
See git history and below for full context.

---

## Session 2 Findings (2026-02-07)

### Three diagnostic scripts identified root causes

#### Script A: `diagnostic_intermediate_values.py`
Exposed the full numerical pipeline for each candidate type:

```
Type                v_x    lam_m   lam_var   g_mean   g_var  fire_rt   H_marg  std_util
Natural (mean)     4.37   -1.584    0.41     0.014   0.053    1.11     1.35    0.023
FULL_BLACK        41.33   -5.640    0.99    -1.438   0.127    0.24     0.62    0.016
FULL_WHITE        44.01   -7.644    2.04    -2.156   0.262    0.12     0.41    0.016
UNIFORM_GRAY       0.00   -0.021    0.00     0.574   0.000    1.77     1.64    0.000
RANDOM_NOISE       2.17    0.225    0.65     0.662   0.084    1.94     1.76    0.077
```

**Key insight**: Arc-cosine kernel self-covariance v_x = x^T C x + sigma_0^2 is 10x
larger for FULL_BLACK/WHITE than for natural images. But the GP posterior compresses
this through the variational correction. The real driver is g_mean = A*lambda + lambda0:
extreme stimuli map to very negative g_mean → low predicted firing rate → low Poisson
entropy → low utility. The original hypothesis that extreme stimuli would have HIGH
utility was wrong — the model correctly predicts they don't drive the neuron.

Also: sigma_0 was learned to ~0, so UNIFORM_GRAY has v_x ≈ 0.

#### Script C: `entropy_landscape_check.py`
Overlaid candidate (g_mean, g_var) positions on the Poisson entropy heatmap H(R|mu,sigma2).
Saved as `entropy_landscape_check.png`.

All candidates fall within the non-zero entropy band (g_mean ∈ [-2.2, 0.9]), but
FULL_BLACK/WHITE are at the low-entropy fringe while natural images and RANDOM_NOISE
are in the moderate-to-high entropy zone. This confirms the Laplace approximation is
well-behaved for all candidates — no dead-zone artifacts.

#### Script B: `validate_utility_better_synthetics.py`
Replaced pathological extreme stimuli with structurally-degraded natural images:
- Pixel-shuffled: same pixel values, randomly permuted (destroys spatial structure)
- Phase-scrambled: FFT → random phase, keep magnitude → IFFT (preserves power spectrum)
- Matched noise: per-pixel N(mean, std) from training set

Results with comparable kernel norms:

```
Type composition of top-20:
  Standard    : NAT=5,  SHF=9, PHS=0, MTC=6
  Dist-Aware  : NAT=9,  SHF=5, PHS=3, MTC=3
```

**This is the separation we were looking for.** Distribution-aware utility puts 9/20
natural images in its top-20 vs only 5/20 for standard utility. The effect is moderate
but real, likely weakened by the poor model fit (test_r=0.091).

### Sanity checks performed

1. **Single-image conditioning** (sample_lambda=False, x_samples = one natural image):
   That image correctly ranks #1 in DA utility. Its DA utility equals its standard
   utility (diff = 1.1e-4, numerical noise from different code paths).

2. **MC lambda sampling** (sample_lambda=True, 1000 copies of same image):
   Match got worse (diff = 3.0e-3) because MC noise O(1/sqrt(N)) ≈ 0.003 exceeds
   the Jensen gap O(1e-4). The deterministic mode is more accurate for this test.

3. **MC samples as candidates** (10 images from x_samples added to candidates):
   These behave like regular natural images. Self-conditioning contributes only
   1/1000 of the total — negligible among 1000 MC iterations.

4. **All natural candidates as x_samples** (50 natural images, sample_lambda=False):
   Unexpectedly, FULL_BLACK/WHITE still ranked #2-3 in DA utility, above most naturals.
   The cross-covariance between synthetics and natural images (mediated through the
   variational posterior) is non-negligible. Each natural gets 1/50 self-conditioning,
   but the self-contribution is diluted.

### Model quality concern

All experiments used n_train=200, M=50, seed 42 → test_r=0.091 (very poor fit).
The GP has not learned strong cross-covariance structure between natural images.
This weakens the DA utility signal. A better-fit model (more training data, different
cell, or more iterations) would likely show stronger separation.

---

## Uncommitted Changes

```
M  acquisition.py                        — sys.path save/restore fix
M  run_single_mode.py                    — _model, _likelihood, _indices_train return + _validate_model_params()
M  tests/test_acquisition.py             — sys.path save/restore pattern
M  investigations/validate_utility/validate_utility_natural_images.py — updated docstring + mc_sample candidates
?? investigations/validate_utility/HANDOFF_SESSION_NOTES.md           — this file
?? investigations/validate_utility/diagnostic_intermediate_values.py  — Script A
?? investigations/validate_utility/entropy_landscape_check.py         — Script C
?? investigations/validate_utility/entropy_landscape_check.png        — Script C output
?? investigations/validate_utility/validate_utility_better_synthetics.py — Script B
```

---

## What Was Built (Both Sessions)

### `acquisition.py` — Two utility functions for active learning

| Function | Purpose |
|----------|---------|
| `standard_utility(model, likelihood, x_candidates, r_max)` | U = H_marg - H_noise. No conditioning, no p(x). Works with any model mode. |
| `distribution_aware_utility(model, likelihood, x_candidates, x_samples, r_max, sample_lambda)` | U = H_marg - E[H_cond]. MC over p(x) samples. Requires `.covariance_matrix` (default_gpy only). |

**Key design choices**:
- **Imports from old codebase** (not modified): `compute_H` (1D playground), `get_conditional_moments_nd` (2D playground), `nd_utility_new` (utility.py)
- **p(x) as pre-drawn tensor**: Caller provides `x_samples` tensor
- **`sample_lambda=True/False`**: When False, uses posterior mean — deterministic, useful for debugging
- **Returns dict**: `{'utility': ..., 'H_marg': ..., 'H_cond': ...}`
- **No `torch.no_grad()` wrapper**: Caller decides. Door open for gradient-based optimization.
- **sys.path save/restore**: Guards against import side effects

**Critical subtlety**:
- `compute_H(mu, sigma2, a, lambda0)` takes RAW GP moments, transforms internally
- `nd_utility_new(mu, sigma2)` expects ALREADY-TRANSFORMED log-firing rate moments

### `tests/test_acquisition.py` — 6/6 tests pass

### Investigation scripts in `investigations/validate_utility/`

| Script | Purpose |
|--------|---------|
| `validate_utility_natural_images.py` | Main validation: natural vs synthetic vs mc_sample candidates |
| `diagnostic_intermediate_values.py` | Pipeline diagnostic: v_x, lambda, g, H, utility per candidate type |
| `entropy_landscape_check.py` | Overlay candidates on entropy heatmap (generates .png) |
| `validate_utility_better_synthetics.py` | Better controls: pixel-shuffled, phase-scrambled, matched-noise |

### Things noticed but not acted upon

1. **n_train=50 failed with Cholesky error** (session 1). The user originally wanted
   n_train=50, M=50 (inducing = training). This failed in float32. We moved to n_train=500
   (session 1) then n_train=200 (session 2). The 1D analogy (few training points, big
   unexplored region) would be better served by n_train=50 or n_train=100. Requires
   either float64 or higher jitter.

2. **Candidate selection is not random**. "First 50 from pool" is deterministic based on
   pool index ordering after removing training indices. A random subset might be more
   representative.

3. **The arc-cosine kernel treats synthetic stimuli differently than RBF**. Full-field
   constant images have a specific angle relationship to natural images that depends on
   the kernel's RF center and masking. The 1D RBF kernel intuition from the playground
   does not directly transfer.

4. **Black/white asymmetry** (session 1, n_train=500): FULL_BLACK had much higher standard
   utility (0.122) than FULL_WHITE (0.020). This was explained in session 2 by the
   diagnostic table — both map to negative g_mean but at different magnitudes, placing
   them at different points on the entropy landscape.

5. **We never ran with `sample_lambda=False` on the full MC pool**. The deterministic
   sanity checks used single images or self-conditioning. A full run with 1000 MC pool
   images and sample_lambda=False would eliminate MC noise and show the "true" DA utility
   ranking, at the cost of underestimating the conditioning effect (Jensen gap is small
   with A=0.358, so this cost is negligible).

6. **Gradient-based x* optimization was not attempted**. The high-dim codebase
   (`optimize_with_conditioned_utility` in utility.py) optimizes x* to maximize utility.
   The most informative stimulus might not be any natural image or simple synthetic.

---

## Suggested Next Steps

1. **Improve model quality**: Use n_train=500 or more, different seed, or different cell
   to get test_r > 0.5. Re-run better-synthetics script to see if separation strengthens.

2. **Commit the accumulated changes**: acquisition.py sys.path fix, run_single_mode.py
   model return, test file updates, investigation scripts.

3. **Consider the conditioning mechanism**: H_cond ≈ H_marg everywhere suggests the
   GP's cross-covariance between candidate and MC sample points is weak. With more
   inducing points (M=100, 200) or a better-fit model, conditioning should be stronger.

4. **Gradient-based x* optimization**: The door is open (no torch.no_grad wrapper).
   Optimizing x* to maximize DA utility would test whether the function landscape is
   correct even when individual natural images don't show strong separation.

---

## File Structure

```
scripts/gpytorch_porting/
├── acquisition.py
├── tests/test_acquisition.py
├── run_single_mode.py                      # Modified: returns _model, _likelihood, _indices_train
├── investigations/
│   └── validate_utility/
│       ├── validate_utility_natural_images.py
│       ├── diagnostic_intermediate_values.py
│       ├── entropy_landscape_check.py
│       ├── entropy_landscape_check.png
│       ├── validate_utility_better_synthetics.py
│       └── HANDOFF_SESSION_NOTES.md
```
