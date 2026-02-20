# Handoff: Unified Gradient Investigation Script (Phase 3)

**Branch**: `pietro/workingbranch`
**Date**: 2026-02-20
**Status**: Partial implementation (Phases 1-2 complete, Phase 3-4 remaining)
**Plan file**: `~/.claude/plans/cozy-humming-wadler.md`

---

## Motivation

This is the continuation of the "unify kernel selection and utility investigation" work. The overall goal is to consolidate per-kernel investigation folders (understanding_utility/, arcsine_kernel/, rbf_kernel/) into a single set of scripts in `investigations/utility/` that accept `--kernel-type`.

Phase 1 (kernel selection in `run_single_mode.py`) and Phase 2 (unified `explore_utility.py`) are committed/implemented. Phase 3 creates the unified `gradient.py` — the LBFGS gradient ascent script for DA utility optimization. Phase 4 (cleanup of old folders) follows after Phase 3 is verified.

---

## What Was Accomplished This Session

### Phase 1: Kernel selection in run_single_mode.py (COMMITTED: 8d54cf1)
- Added `kernel.type` and `kernel.lengthscale` to `default_params.json`, `canonical.yaml`, `quick.yaml`
- `create_kernel()` factory function in `kernels.py` (user moved it there from run_single_mode.py)
- `KERNEL_TYPES = ('arc_cosine', 'arc_sine', 'rbf')` in `kernels.py`
- Replaced all 3 hardcoded `ArcCosineKernel()` calls in `run_single_mode.py` with `create_kernel()`
- `--kernel-type` and `--lengthscale` CLI args added
- `build_config_from_defaults()` and `flatten_yaml_config()` wired with `kernel_type` and `lengthscale`
- Validation guards: vargp_old requires arc_cosine; vjp/jacobian require arc_cosine
- Verified: all 3 kernels train in both default_gpy and vargp_direct modes

### Phase 2: Unified explore_utility.py (UNCOMMITTED)
- Created `investigations/utility/explore_utility.py`
- `setup(kernel_type=None)` uses `build_config_from_defaults()`
- `_get_kernel_type(model)` auto-detects from kernel class (isinstance checks)
- Right panel auto-detected: ||x||_C (arc_cosine), sqrt(K(x,x)) (arc_sine), K(x*,x_cond) (rbf)
- Kernel-specific `describe()`, `eval_da_conditioned()`, `plot_da_landscape()`, `demo()`
- Verified: all 3 kernel types produce output and save plots

---

## Decisions and Rationale

### 1. Single `distribution_aware_utility()` call per LBFGS step (NOT RBF's decomposed approach)

The RBF script (`gradient_rbf.py`) decomposed utility into separate H_marg + MC loop over H_cond with per-sample backward — needed for memory efficiency with many lambda samples. The user explicitly said "none of the MC sampling complications." The unified script calls `distribution_aware_utility()` once per step with `sample_lambda=False`, exactly like the unnormalized and arcsine scripts.

**Rejected**: Keeping both patterns with a flag. The decomposed approach was experimental and its key finding (gradients from N=50+ images cancel) suggests it's a dead end.

### 2. Bounds: sigmoid + 2 modes only (none, dataset)

The RBF script had 4 modes: none, dataset, target, percentile. The user chose to keep only `none` (unconstrained) and `dataset` (per-pixel min/max from training images within RF). The `target` and `percentile` modes were experimental and not used in final analyses.

### 3. Hyperparams hardcoded at script level, NOT kernel-specific

N_STEPS, LR, LBFGS settings are single values, same for all kernels. The user explicitly rejected per-kernel defaults. These are investigation tuning knobs, not model parameters. The plan suggests values from the unnormalized baseline: N_STEPS=50, LR=0.5, LBFGS_MAX_ITER=20, LBFGS_MAX_EVAL=25, LBFGS_HISTORY_SIZE=10.

### 4. `kernel_value()` uses `getattr(model, 'covar_module', None) or getattr(model, 'kernel', None)`

DirectVGPModel uses `.kernel`, VariationalGPModel uses `.covar_module`. The explore_utility.py established this pattern and gradient.py should follow it.

### 5. Output filenames include kernel type

`explore_utility.py` saves to `explore_utility_{kernel_type}.png`. The gradient script should save to `gradient_{kernel_type}.png`.

---

## Critical Subtleties for Phase 3 Implementation

### RF overlay circles require kernel attribute access

The arcsine and RBF gradient scripts draw RF center + sigma circles on images. The formula is:
```
kernel = getattr(model, 'covar_module', None) or getattr(model, 'kernel', None)
eps_0x = kernel.eps_0x.item()
eps_0y = kernel.eps_0y.item()
beta_nat = kernel.beta.item()
sigma_px = beta_nat * math.sqrt(2) * (n_px_side - 1) / 2
cx_px = (eps_0x + 1) / 2 * (n_px_side - 1)
cy_px = (eps_0y + 1) / 2 * (n_px_side - 1)
```
The unnormalized script did NOT have this. The unified script should include it (it works for all kernels since they all inherit the same RF structure).

### The `_cached_mask` access pattern is fragile

To get the RF mask, the scripts run a forward pass to trigger mask caching:
```python
kernel = getattr(model, 'covar_module', None) or getattr(model, 'kernel', None)
if not (hasattr(kernel, '_cached_mask') and kernel._cached_mask is not None):
    with torch.no_grad():
        _ = model(X_pool[0].unsqueeze(0))
rf_mask = kernel._cached_mask.squeeze()
```
If the model hasn't done a forward pass, `_cached_mask` is None and you get an error. Always trigger a dummy forward pass first.

### f_max guard in LBFGS closure

All three gradient scripts check firing rate in the LBFGS closure:
```python
mu_g = A * mu_marg + lam0
if torch.exp(mu_g) > f_max:
    return torch.tensor(float('inf'))
```
This is identical across scripts. `f_max` comes from the config (default 100.0).

### Sigmoid bounds: logit initialization and float32 overflow

The sigmoid transform maps unconstrained z to [pixel_lo, pixel_hi]:
```python
x_rf = pixel_lo + (pixel_hi - pixel_lo) * torch.sigmoid(z)
```
Initialization: `z_init = log(sigmoid_val / (1 - sigmoid_val))` where `sigmoid_val = (x_rf_init - pixel_lo) / (pixel_hi - pixel_lo)`. Must clamp sigmoid_val to (1e-6, 1-1e-6) before logit to avoid inf in float32.

### `_reconstruct_image()` pattern

All scripts use the same pattern: allocate a zero tensor of size n_pixels, then assign RF pixels:
```python
def _reconstruct_image(x_rf, rf_mask, n_pixels, dtype, device):
    x_full = torch.zeros(n_pixels, dtype=dtype, device=device)
    x_full[rf_mask] = x_rf
    return x_full
```
This must NOT be called with `torch.no_grad()` — the gradient from DA utility flows through x_rf into z (the optimized variable).

### The gradient script imports `setup()` from explore_utility.py

Per the plan, gradient.py imports `setup()` from the unified explore_utility.py (same folder). It does NOT duplicate the model training code.

### x_target synthetic bipartite pattern

All scripts support USE_SYNTHETIC=True for a half-dark/half-light target image. The pixel values differ per kernel (unnormalized/rbf: -0.5/0.5, arcsine: 0.2/-0.2). The unified script should use a single set of values. Suggest -0.5/0.5 (matches the majority).

### Per-kernel tuning footnote

The user accepted that the unified hyperparams (N_STEPS=50, LR=0.5) may not be optimal for all kernels. The arcsine investigation originally needed 500 steps with LR=0.1 (10x more steps, 5x smaller LR). If the unified defaults don't converge for arc_sine, the user will manually tune. This is documented in the plan.

---

## Phase 3 Implementation Guide

### Target file
`investigations/utility/gradient.py`

### Structure (from plan + comparison of existing scripts)

**Script-level constants** (investigation-specific, hardcoded is OK):
```python
N_STEPS = 50
LR = 0.5
LBFGS_MAX_ITER = 20
LBFGS_MAX_EVAL = 25
LBFGS_HISTORY_SIZE = 10
LOG_EVERY = 1

USE_SYNTHETIC = False       # True = bipartite target, False = smoothed natural image
TARGET_INDEX = 0            # Pool image index for natural target
SIGMA_SMOOTH = 1.0          # Gaussian smoothing sigma for natural target
DARK_GRAY = -0.5            # Bipartite dark half value
LIGHT_GRAY = 0.5            # Bipartite light half value

BOUNDS_MODE = 'none'        # 'none' = unconstrained, 'dataset' = sigmoid with per-pixel min/max
```

**Functions to implement** (ordered by dependency):

1. `_reconstruct_image(x_rf, rf_mask, n_pixels, dtype, device)` — Place RF pixels into full image
2. `rf_pearson_r(x, target, rf_mask)` — Pearson r within RF pixels
3. `rf_proj_coeff(x, target, rf_mask)` — Projection coefficient within RF
4. `interpolation_sweep(model, likelihood, x_target, x_perturbed, n_points, r_max)` — U_DA along t=[0,1] from perturbed to target. Uses `sample_lambda=False`.
5. `gradient_ascent(model, likelihood, x_start, x_target, rf_mask, r_max, f_max, pixel_lo, pixel_hi, n_steps, lr, max_iter, max_eval, history_size)` — LBFGS loop. Returns history dict. The LBFGS closure:
   - Reconstructs full image from z (sigmoid if bounds, raw if no bounds)
   - Calls `distribution_aware_utility()` with `sample_lambda=False`
   - Checks f_max firing rate guard
   - Logs utility, grad_norm, pearson_r, proj_coeff per step
6. `plot_results(...)` — 2-row figure: top row = images (target, start, final, difference with RF overlays), bottom row = interpolation + convergence (U_DA + Pearson r on twinx)
7. `main(kernel_type=None)` — Entry point: calls `setup()` from explore_utility.py, creates target, runs interpolation + gradient ascent, prints diagnostics, saves plot

**CLI**:
```python
parser = argparse.ArgumentParser()
parser.add_argument('--kernel-type', choices=['arc_cosine', 'arc_sine', 'rbf'], default=None)
```

**Imports** (from explore_utility.py in same folder):
```python
from explore_utility import (setup, kernel_value, kernel_norm, kernel_angle,
                              gp_moments, logfiring_moments, _ADAPTIVE_RMAX_PARAMS,
                              _get_kernel_type)
```
Plus importlib pattern for `distribution_aware_utility` from acquisition.py and `get_gp_marginal_moments` from utils.py (needed for firing rate diagnostics in the closure).

**Key differences from existing scripts to handle**:
- `pixel_lo`/`pixel_hi` set to None when BOUNDS_MODE='none', skip sigmoid transform entirely
- When BOUNDS_MODE='dataset': compute per-pixel min/max from all training+pool images within RF mask
- RF overlay on all 4 top-row images (draw_rf_overlay helper function)
- Output saved as `gradient_{kernel_type}.png`

### What to verify after implementation
1. `python investigations/utility/gradient.py` runs with default kernel (arc_cosine)
2. `python investigations/utility/gradient.py --kernel-type arc_sine` runs
3. `python investigations/utility/gradient.py --kernel-type rbf` runs
4. Output PNG has 2 rows: images on top, interpolation + convergence on bottom
5. RF overlay circles appear on images
6. Interpolation shows monotonic or near-monotonic U_DA increase

---

## Uncommitted Changes

```
 M SESSION_LOG.md                          — Updated with this session's entry
 M ../2D_playground/utility_2d_rbf_base.*  — Pre-existing (unrelated)
 D imgs/*                                  — Old plots from deprecated modes (pre-existing)
 M investigations/arcsine_kernel/HANDOFF.md — Pre-existing
 M investigations/understanding_utility/gradient_unnormalized.py — Pre-existing minor edit
 M pietro_plan.md                          — Pre-existing
 M ../one_cell_active_training_distribution_aware.py — Pre-existing
?? .claude/handoffs/HANDOFF_2026-02-20_*   — Handoff files from this session
?? experiments/exploratory/2026-02-07_*    — Pre-existing experiment folders
?? imgs/default_gpy_M{150,300,3000}.png   — Pre-existing
?? investigations/*/explore_utility_*.png  — Pre-existing output images
?? investigations/utility/                 — NEW: Phase 2 unified explore_utility.py + output PNGs
```

The Phase 2 `investigations/utility/` folder is uncommitted. It should be committed before starting Phase 3.

---

## Files to Read First

1. **This handoff** — decisions, Phase 3 implementation guide, critical subtleties
2. **Plan file** (`~/.claude/plans/cozy-humming-wadler.md`) — full 4-phase plan, Phase 3 section
3. `investigations/utility/explore_utility.py` — Phase 2 unified script, provides `setup()` that Phase 3 imports
4. `investigations/understanding_utility/gradient_unnormalized.py` — simplest gradient template (no MC, no complex bounds)
5. `investigations/rbf_kernel/gradient_rbf.py` — reference for `BOUNDS_MODE='dataset'` implementation and RF overlay code
6. `investigations/arcsine_kernel/gradient_arcsine.py` — reference for RF overlay circles + proj_coeff in convergence plot
7. `kernels.py` — `create_kernel()`, `KERNEL_TYPES`, kernel class hierarchy
8. `acquisition.py` — `distribution_aware_utility()` signature (the main function called in LBFGS closure)

---

## Caveats and Open Questions

1. **Unified hyperparams may not converge for all kernels**: N_STEPS=50, LR=0.5 works for arc_cosine. Arc-sine originally needed 500 steps at LR=0.1. RBF used LR=5.1. The user accepted this tradeoff — manual tuning of script constants is expected.

2. **SIGMA_SMOOTH varies per kernel**: Unnormalized=1.0, arc-sine=3.0, RBF=10.0. The unified script picks a single value. 1.0 is the simplest default. If results look bad for other kernels, the user tunes it.

3. **Phase 2 explore_utility.py is uncommitted**: It was verified working for all 3 kernels but not yet committed. Commit it before starting Phase 3 to keep the diff clean.

4. **convergence plot proj_coeff**: The arc-sine script plotted proj_coeff as a secondary metric on the convergence plot. The RBF script did not. The unified script should include it (it's a useful metric) but this wasn't explicitly discussed with the user.

5. **Pre-existing uncommitted changes**: Many old image deletions, investigation artifacts from previous sessions. These should ideally be committed separately to keep the kernel unification diff clean.

---

## Continuation Prompt

```
I'm continuing work on unifying kernel selection and utility investigation scripts.

Read these files first:
1. .claude/handoffs/HANDOFF_2026-02-20_unify-kernel-phase3-gradient.md (Phase 3 guide)
2. ~/.claude/plans/cozy-humming-wadler.md (full 4-phase plan)
3. investigations/utility/explore_utility.py (Phase 2 — provides setup() for gradient.py)
4. investigations/understanding_utility/gradient_unnormalized.py (simplest gradient template)

Phases 1-2 are done. Phase 3: create investigations/utility/gradient.py.
Key constraints:
- Single distribution_aware_utility() call per step (NO MC decomposition)
- Bounds: sigmoid + 2 modes (none, dataset)
- Imports setup() from explore_utility.py in same folder
- Hardcoded hyperparams at script level (N_STEPS=50, LR=0.5, etc.)
- RF overlay circles on all 4 top-row images
- Output: gradient_{kernel_type}.png

Check git status and git branch before starting.
First commit the Phase 2 explore_utility.py, then implement Phase 3.
```
