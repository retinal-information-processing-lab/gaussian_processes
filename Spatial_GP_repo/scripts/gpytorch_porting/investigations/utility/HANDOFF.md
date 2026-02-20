# Investigation: Unified Utility Investigation Scripts

**Branch**: `pietro/workingbranch`
**Date**: 2026-02-20
**Status**: Complete (Phases 1-4)
**Location**: `investigations/utility/`

---

## What This Folder Contains

- `explore_utility.py` — Utility exploration workbench for all kernel types. Trains model, provides kernel/GP/utility helpers, generates DA utility landscape plots. Supports `--kernel-type {arc_cosine, arc_sine, rbf}`.
- `gradient.py` — LBFGS gradient ascent on DA utility across kernel types. Interpolation monotonicity check + gradient ascent with RF overlay visualization. Supports `--kernel-type {arc_cosine, arc_sine, rbf}`.

These replace the per-kernel scripts that were in `understanding_utility/`, `arcsine_kernel/`, `rbf_kernel/`, and `normalized_kernel/` (deleted, retrievable from git history).

---

## Consolidated Key Findings (From Per-Kernel Investigations)

### Arc-Cosine Kernel (unnormalized)
- K(x,x) ~ ||x||_C^2 — norm grows without bound.
- DA utility grows monotonically with norm: higher ||x|| -> higher lambda_m -> higher firing rate -> higher H_marg -> higher utility.
- Gradient ascent exploits this by amplifying images rather than finding angularly similar ones.
- This is intrinsic to the DA utility formula, not a kernel bug.

### Normalized Arc-Cosine Kernel (DEPRECATED)
- K(x,x) = 1.0 exactly — eliminates norm dependence.
- Utility depends only on angular structure (RF alignment).
- But test_r drops ~25% (0.79 -> 0.59 on PNAS cell 8, M=100). Image norm is genuinely informative for neural encoding.
- Not included as a kernel option in the unified scripts.

### Arc-Sine Kernel (Williams 1998)
- K(x,x) saturates toward 1 via erf activation. Training images: K_sat ranges 0.07-0.95.
- Saturation limits but does NOT eliminate norm-driven utility growth. The transition zone before saturation still allows lambda_m growth.
- Firing rates are modest (~6 spikes). The f_max guard doesn't fire.
- LBFGS converges fast (2-3 outer steps), then flat.

### LocalRBF Kernel
- K(x,x) = 1.0 (stationary). Utility depends on distance to conditioning image.
- With multi-image conditioning (50 images), gradients cancel out. Utility barely moves.
- Test_r lower than arc-cosine (0.25 vs 0.78 at M=50/N_TRAIN=50).

### Cross-Kernel Conclusions
- The core issue is shared across all kernels: DA utility rewards high marginal entropy H_marg, which correlates with predicted firing rate, not epistemic uncertainty.
- Only the normalized kernel truly eliminates norm dependence, but it sacrifices predictive accuracy.
- The f_max firing rate guard (default 100.0) prevents extreme rate exploitation. It's effective for arc-cosine but unnecessary for arc-sine/RBF where rates are naturally bounded.

---

## Deferred: distribution_gradient.py (Multi-Image DA Conditioning)

**Deleted** with the rbf_kernel/ folder. Retrievable from git history (`rbf_kernel/distribution_gradient.py`).

**Re-implementation guide** (if needed):
- Decompose utility into H_marg + MC loop over H_cond with per-sample backward for O(1) memory.
- Per-sample backward: `(-H_marg).backward()`, then `(H_cond_i / n_mc).backward()` per sample with fresh x_query each time.
- Must `.detach()` A_val/lam0_val to avoid double-backward through likelihood parameters.
- Key finding: gradients from N=50+ conditioning images cancel out, producing a flat utility landscape. The averaging across diverse images eliminates directional signal.

---

## Files Deleted (Retrievable from Git)

| Folder | Contents | Reason |
|--------|----------|--------|
| `investigations/normalized_kernel/` | validate_kernel.py, run_normalized.py, explore/gradient scripts, HANDOFFs, PNGs | Deprecated kernel, unified scripts cover arc_cosine/arc_sine/rbf |
| `investigations/arcsine_kernel/` | run_arcsine.py, explore/gradient scripts, HANDOFF, PNGs | Superseded by unified scripts + kernel selection in run_single_mode.py |
| `investigations/rbf_kernel/` | run_rbf.py, explore/gradient/distribution_gradient scripts, HANDOFF | Superseded by unified scripts + kernel selection in run_single_mode.py |
| `investigations/understanding_utility/explore_utility.py` | Arc-cosine explore script | Superseded by utility/explore_utility.py |
| `investigations/understanding_utility/gradient_unnormalized.py` | Arc-cosine gradient script | Superseded by utility/gradient.py |

**Kept in `understanding_utility/`**: entropy_landscape.*, test_compute_H_MC.py, *.tex math docs, key_facts.md, HANDOFF.md, da_utility_landscape.png.
