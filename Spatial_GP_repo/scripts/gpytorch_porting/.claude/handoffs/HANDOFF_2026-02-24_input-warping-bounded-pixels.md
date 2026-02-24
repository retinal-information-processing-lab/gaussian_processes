# Handoff: Input Warping for Bounded Pixel Domain

**Branch**: `pietro/workingbranch` (implementation should create `pietro/input-warping` from here)
**Date**: 2026-02-24
**Status**: Ready for implementation
**Plan file**: `.claude/plans/playful-wandering-pine.md`

---

## Motivation

When optimizing images to maximize GP utility (distribution-aware active learning), the optimizer can push pixel values outside the physical display range [vmin, vmax]. The DMD projector clips at [0, 255]; in normalized data, this maps to a finite [dataset_global_min, dataset_global_max]. The kernel's C matrix is a spatial smoother -- it sees `C @ x`, not individual pixels -- so individual pixel bound violations are invisible to the kernel. The utility landscape has no "walls" at pixel bounds.

Current mitigation is sigmoid reparameterization in the optimizer (`x = vmin + (vmax-vmin) * sigmoid(z)`). This constrains the search but not the model. The GP still predicts high utility past the bounds; the sigmoid just prevents the optimizer from reaching there. You are fighting the model's landscape.

The idea: apply a per-pixel nonlinear warping w(x) INSIDE the kernel, before the C matrix, so the GP trains on warped pixels and its learned parameters adapt to a world where pixel values saturate at bounds. The model genuinely cannot distinguish "at the bound" from "past the bound." No optimizer constraint needed. This emerged from the subspace optimization investigation (see `investigations/utility_decompositions/HANDOFF_COMBINED_SUBSPACE.md`, "Open Idea: Input Warping" section).

---

## Decisions and Rationale

### Decision 1: Fixed scaled tanh (not learned CDF warping)

**Decided**: Use `w(x) = mid + (range/2) * tanh(a * (x - mid) / (range/2))` with fixed steepness `a`.

**Alternatives rejected**:
- **Beta CDF warping (Snoek et al. 2014)**: Designed for learning non-stationarity per input dimension. Would add 2 * 11664 = 23328 learnable parameters. Our pixels all share the same physical bounds -- we don't need per-pixel adaptive warping. Also, the Beta CDF has no closed form (requires incomplete beta function), making gradients harder.
- **Kumaraswamy CDF (BoTorch)**: Same per-dimension philosophy as Beta CDF but computationally cleaner. Still overkill for uniform bound enforcement.
- **Multiplicative penalty kernel**: `k_bounded(x,y) = k(x,y) * prod_i phi(x_i) * phi(y_i)`. Product over 11664 dimensions causes catastrophic underflow. Would need log-space tricks and is non-standard.

**Why tanh**: Zero extra learnable parameters. Same function for all pixels (same physical bounds). Trivially differentiable. Well-understood. The steepness can be made learnable later (single shared parameter) if needed -- documented in `INPUT_WARPING_REFERENCE.md` Section 8.7.

### Decision 2: Steepness is a fixed constant (not learned)

**Decided**: Steepness `a` stored in config, not optimized during training.

**Rationale**: Simplest first version. Adding one learnable parameter is trivial later if the fixed value proves insufficient. The config already supports it -- just change from a plain attribute to an nn.Parameter.

**Value**: `a = 3.0` in `default_params.json`. This gives nearly identity within bounds (~0.5% compression at boundaries) and strong saturation outside.

### Decision 3: Warping goes at the top of kernel.forward(), before all branching

**Decided**: Two lines at the very top of `forward()`, before the VJP/Jacobian/autograd branch.

**Why not inside `_compute_C_matrix()`**: The warping is on pixel INTENSITY values (x1, x2), not on spatial coordinates (xcord, ycord). The C matrix computation uses spatial coordinates for the RF structure -- those should not be warped. This distinction is critical and easy to confuse.

**Why before gradient branching**: All three gradient modes then receive warped inputs consistently. For VJP/Jacobian, the analytical backward computes dk/d(kernel_params) at warped input points -- mathematically correct (we want gradients of the warped kernel w.r.t. hyperparameters). The chain rule through w() for dk/dx is handled by autograd when needed (utility optimization).

### Decision 4: All gradient modes work with warping (no NotImplementedError)

**Discussed**: User initially suggested deprecating VJP/Jacobian when warping is enabled (conservative).

**Decided**: Allow all modes. The reasoning: warping at the top of forward() means GradFunction.apply() receives warped tensors. Analytical gradients compute dk/d(params) at warped points -- this is the correct gradient of the warped kernel. The VJP backward returns dk/d(warped_x1), and autograd chains through w() if x-gradients are needed. During M-step, x1/x2 don't require grad, so no chain-through occurs. During utility optimization, autograd mode is used anyway.

### Decision 5: Warping is togglable, disabled by default

**Decided**: `kernel.input_warping.enabled: false` in `default_params.json`. CLI: `--input-warping` / `--no-input-warping`.

**Rationale**: Existing behavior unchanged. Allows A/B comparison. When disabled, the `forward()` check is a single boolean test (zero overhead).

**Important caveat**: This is a model-level toggle, not a runtime switch. Kernel hyperparameters adapt to the warped space during training. Comparing warped vs unwarped requires separate training runs. No cross-session consistency check needed because there is no model serialization -- each run trains fresh.

### Decision 6: Both training modes (vargp_direct and default_gpy)

**Decided**: Implement for both modes.

**Rationale**: The warping is in the kernel, which is shared. No mode-specific changes needed. The ELBO requires no Jacobian correction (input warping, not output warping). Eigenspace decomposition in vargp_direct works identically on warped K_tilde.

### Decision 7: Separate investigation branch

**Decided**: Create `pietro/input-warping` from `pietro/workingbranch`.

**Rationale**: This is a modeling change (the model trains differently). Should be validated before merging into the working branch.

---

## Critical Subtleties

1. **Warping is on pixel INTENSITIES, not spatial coordinates.** The C matrix computation in `_compute_C_matrix()` uses `xcord`, `ycord` (spatial pixel grid). These must NOT be warped. Only the image data vectors x1, x2 in `forward()` get warped. Getting this wrong would distort the RF structure instead of bounding pixel values.

2. **Subclass forward() methods need the warping preamble too.** ArcSineKernel.forward() (line 746) and LocalRBFKernel.forward() (line 890) override the base forward(). The plan recommends a `_maybe_warp()` helper that each forward() calls at the top. Forgetting one subclass would silently skip warping for that kernel type.

3. **vmin/vmax come from data, not config.** The warping bounds [vmin, vmax] are the global pixel range across the entire dataset (train + val + test). They are computed at data loading time and passed to the kernel at creation time. They should NOT be hardcoded in `default_params.json` -- they are data-dependent. Only `enabled` and `steepness` are config parameters.

4. **No `.get('key', fallback)` for warping config.** Per project parameter discipline rules, missing config keys should crash with KeyError, not silently fall back to a default. The one exception is `create_kernel()` which may receive configs from different sources -- see plan for exact pattern.

5. **The `INPUT_WARPING_REFERENCE.md` document is NOT needed for implementation.** It is a general literature survey (400 lines, 11 sections, 10 references). The plan file contains everything needed for implementation. The reference document is for future consultation if alternative approaches are reconsidered.

---

## Uncommitted Changes

```
Untracked files:
  investigations/input_warping/           <- NEW: reference document (INPUT_WARPING_REFERENCE.md)
  experiments/exploratory/2026-02-07_*    <- OLD: jitter investigation experiment outputs (6 folders)
  imgs/default_gpy_M*.png                <- OLD: experiment plot outputs (3 files)
```

No modified tracked files. The `investigations/input_warping/INPUT_WARPING_REFERENCE.md` was created this session. The experiment outputs and PNGs predate this session.

---

## Files to Read First

1. **`.claude/plans/playful-wandering-pine.md`** -- The implementation plan. Contains exact file-by-file changes, code snippets, and verification steps. This is the primary instruction document.

2. **`kernels.py`** (lines 190-260 for `__init__`, lines 543-629 for `forward()`, lines 71-103 for `create_kernel()`) -- Where the core implementation goes. Understand the existing forward() flow before modifying.

3. **`default_params.json`** -- Existing config structure. The new `input_warping` section nests inside `kernel`.

4. **`run_single_mode.py`** (lines 242-292 for `build_config_from_defaults()`, lines 1107-1196 for argparse) -- Config wiring from JSON to kernel.

5. **`investigations/input_warping/INPUT_WARPING_REFERENCE.md`** -- General literature survey. NOT needed for implementation, but useful background if questions arise about why this approach was chosen over alternatives.

---

## Caveats and Open Questions

1. **Steepness value `a=3.0` is untested.** Chosen based on mathematical analysis (0.5% compression at bounds) but not validated on the PNAS dataset. The first training run will reveal whether this is appropriate. Too high may cause gradient issues at boundaries; too low may not enforce bounds strongly enough.

2. **Effect on test_r is unknown.** Warping changes the kernel -- hyperparameters will adapt, but test_r may improve, degrade, or stay the same. This is an investigation, not a guaranteed improvement.

3. **Analytical gradient correctness with warping is argued mathematically but not yet tested.** The argument (warped inputs flow through GradFunction, analytical backward computes correct dk/d(params)) is sound in principle. Verification Step 2 in the plan will confirm or reveal issues.

4. **Data vmin/vmax computation location is approximate.** The plan says "after data is loaded and combined (~line 586-589)" but the exact code path may differ. The implementer should trace where `X_combined` is available and compute vmin/vmax there.

5. **The plan uses `config.get('warping_enabled', False)` in `create_kernel()`.** This technically violates the "no fallback defaults" rule. Justification: `create_kernel()` is called from multiple code paths (run_single_mode, run_experiment, investigation scripts). Using `.get()` with False as default ensures backward compatibility -- old config dicts without the warping key still work. This is the ONE exception noted in the plan.

---

## Continuation Prompt

```
I am implementing input warping for bounded pixel domain in the GP kernel.

Read these files in order:
1. .claude/handoffs/HANDOFF_2026-02-24_input-warping-bounded-pixels.md (this handoff -- decisions and rationale)
2. .claude/plans/playful-wandering-pine.md (implementation plan -- exact file changes)
3. kernels.py (lines 71-103, 190-260, 543-630 -- where changes go)

Task: Create branch pietro/input-warping from pietro/workingbranch, then implement
per-pixel tanh input warping in the kernel, togglable via config (disabled by default).

Key constraints:
- Warping goes at top of forward(), BEFORE gradient mode branching
- Warping is on pixel intensity values, NOT spatial coordinates
- vmin/vmax come from data at runtime, not from config files
- All kernel subclasses (ArcSine, RBF, Normalized) need the warping preamble
- Follow parameter discipline: no hardcoded values, read from default_params.json

Check git status and git branch before starting.
Do NOT modify pietro/workingbranch.
```
