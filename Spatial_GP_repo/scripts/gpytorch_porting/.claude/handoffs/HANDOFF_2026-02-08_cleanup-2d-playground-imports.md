# Handoff: Clean Up 2D Playground Imports

**Branch**: pietro/acquisition-functions
**Date**: 2026-02-08
**Status**: Ready for implementation
**Plan file**: `.claude/plans/partitioned-scribbling-hanrahan.md`

---

## Motivation

The 2D playground folder (`scripts/2D_playground/`) is used to investigate utility function behavior on simplified 2D inputs (x,y coordinates instead of 11,664-pixel images). It grew organically from the 1D playground, inheriting a deep import chain: 2D scripts import from 1D playground, which imports from old `utility.py`, which imports `torchlambertw`. Meanwhile, `gpytorch_porting/` now has clean, maintained, differentiable versions of all the same math functions (Laplace approximation, entropy, GP conditioning, utility computation).

The user's primary complaints:
1. Kernel and likelihood are "imported weirdly" -- hard to trace where they come from
2. A and lambda0 parameters are overridden via fallback patterns (`hasattr(...) else 1.0`)
3. Wrappers and duplicated functions that exist in both 2D and gpytorch_porting

The goal: make gpytorch_porting the single source of truth for kernel, likelihood, utility math, and Laplace approximation. The 2D folder should import from gpytorch_porting for all math, keeping only 2D-specific things (plotting, ground truth function, grid generation, constants).

## Decisions and Rationale

### 1. Arc-Cosine Kernel: Separate SimpleArcCosineKernel class (not refactoring ArcCosineKernel)

The gpytorch_porting `ArcCosineKernel` is deeply tied to high-D images: requires `n_px_side`, registers RF parameters (beta, rho, eps_0x, eps_0y), computes C matrix from RF structure, supports pixel masking. For 2D playground inputs (simple x,y coordinates), none of this applies -- C=identity.

We chose to add a separate `SimpleArcCosineKernel` (~40 lines) to `gpytorch_porting/kernels.py` rather than refactoring ArcCosineKernel to handle both cases. Rationale:
- ArcCosineKernel is validated and working; adding conditional logic risks regressions
- The simple kernel is genuinely simple (~40 lines vs ~400) and doesn't share much infrastructure
- Keeps both classes focused and readable

**Critical**: SimpleArcCosineKernel has only `sigma_0`. No `Amp` parameter. The user confirmed that Amp is a parameter of the C matrix (`C = Amp * alpha * C_smooth * alpha^T`), not a standalone kernel amplitude. When C=I, Amp doesn't apply.

### 2. Model class: Keep 1D playground's VariationalGP

The `VariationalGP` from `gp_utility_playground.py` is a simple GPyTorch ApproximateGP with RBF kernel. It works, it's simple, and it's not the source of the import mess. The 2D playground will still import it from the 1D playground. This is intentional -- the model class isn't the problem.

### 3. PoissonLikelihood: Switch to gpytorch_porting's version everywhere

Currently two different PoissonLikelihood classes:
- 1D playground version: No A or lambda0 parameters. Formula: `r ~ Poisson(exp(f))`
- gpytorch_porting version: Has learnable A and lambda0. Formula: `r ~ Poisson(exp(A*f + lambda0))`

RBF 2D scripts use the 1D version, then pass A/lambda0 manually to every utility function via fallback patterns. Arc-cosine scripts already use the gpytorch_porting version.

Decision: use gpytorch_porting's PoissonLikelihood everywhere. For RBF (synthetic data, A=1, lambda0=0): init with `PoissonLikelihood(A_init=1.0, lambda0_init=0.0)` which matches the defaults.

**Compatibility note**: The 1D `compute_elbo()` doesn't call `likelihood.expected_log_prob()` -- it computes ELL inline as `target * mean - exp(mean + var/2)`, which hardcodes A=1, lambda0=0. This is numerically identical to gpytorch_porting's PoissonLikelihood with A=1, lambda0=0, so training behavior is unchanged.

### 4. adaptive_r_max: Move to gpytorch_porting/utils.py

`compute_adaptive_rmax()` is currently defined in `utility_2d_rbf_base.py`. It's generally useful -- prevents entropy collapse when mu_g is high (common with non-stationary kernels). Moving it to gpytorch_porting makes it available to the high-D acquisition functions too.

### 5. Distribution-aware utility: Consolidate into acquisition.py

Currently two implementations:
- `acquisition.py:distribution_aware_utility()`: Takes pre-drawn `x_samples`, fixed r_max
- `utility_2d_rbf_base.py:evaluate_distribution_aware_utility_2d()`: Draws samples from Gaussian p(x) internally, adaptive r_max

Decision: consolidate into acquisition.py by adding `adaptive_r_max=False` flag. The caller draws samples from whatever p(x) they have (Gaussian, image pool, etc.) and passes them. The sampling is the caller's responsibility; the math is acquisition.py's responsibility.

### 6. Checkpoints: Clean break, no backward compatibility

Old `.pt` checkpoint files will be invalidated by the PoissonLikelihood change. The user explicitly said backward compat is not a priority -- just retrain. New checkpoint format stores `likelihood.state_dict()` alongside `model.state_dict()`.

## Critical Subtleties

### compute_adaptive_rmax has suspicious defaults

The current code has `max_rmax=100, min_rmax=200`. Since the function does `max(min(needed, max_rmax), min_rmax)`, this always returns 200 (min > max). Either the defaults are wrong or the naming is misleading. Verify the intended behavior before moving the function. Check if callers override these defaults. Getting this wrong would silently break entropy truncation.

### importlib.util is still needed for 2D -> gpytorch_porting imports

Even after cleanup, the 2D folder imports VariationalGP from the 1D playground, which adds `Spatial_GP_repo/` to sys.path. This causes `import utils` to resolve to the repo-root `utils.py` instead of `gpytorch_porting/utils.py`. The `importlib.util` pattern (used by `acquisition.py`) is the correct workaround. Don't try to simplify to plain `import`.

### Two PoissonLikelihoods coexist during transition

After cleanup, the gpytorch_porting PoissonLikelihood is used by 2D scripts. But `compute_elbo()` in the 1D playground still computes ELL inline (doesn't call `likelihood.expected_log_prob()`). This means A/lambda0 are NOT used during training -- only during utility computation. This is correct for the RBF case (A=1, lambda0=0 = identity transform) but could be confusing. A comment in the code would help.

### Amp parameter misconception risk

Someone looking at `ArcCosineKernel` and `SimpleArcCosineKernel` side by side might wonder why the simple version has no Amp. The reason: Amp scales the C matrix (`C = Amp * alpha * C_smooth * alpha^T`). With C=I, there is no C to scale. If someone needs output amplitude scaling, they should use GPyTorch's `ScaleKernel` wrapper, which is a different (linear) operation.

### Adaptive r_max computes different values for marginal and conditional

The 2D implementation computes adaptive r_max separately for the marginal entropy (one r_max for all candidates) and for each MC conditional entropy step (different r_max per sample). The acquisition.py consolidation must preserve this: `compute_adaptive_rmax` is called both before the MC loop (for marginal) and inside the loop (for each conditional). Using a single r_max for everything would be wrong.

## Uncommitted Changes

```
modified:   ../2D_playground/arccosine/utility_acos_2d_base.png  — plot output (binary)
modified:   ../2D_playground/arccosine/utility_acos_2d_base.py   — previous edits (pre-cleanup)
modified:   ../2D_playground/utility_2d_rbf_base.py              — previous edits (added adaptive r_max, DA utility, etc.)
modified:   imgs/default_gpy_M50.png                             — plot output (binary)
modified:   investigations/validate_utility/gradient.py          — gradient investigation script
modified:   utils.py                                             — added differentiable Laplace pipeline
```

Plus untracked: experiment results (2026-02-07), investigation scripts/artifacts, session-wrap-up skill, incident_report.md.

**Note**: The uncommitted changes to `utility_2d_rbf_base.py` and `utility_acos_2d_base.py` contain the functions that will be refactored in this cleanup. The cleanup replaces these changes rather than building on top of them.

## Files to Read First

1. **`.claude/plans/partitioned-scribbling-hanrahan.md`** — The implementation plan. Read this first for the step-by-step.
2. **`scripts/2D_playground/utility_2d_rbf_base.py`** — The main file being cleaned up. Understand its current structure, what's being removed vs kept.
3. **`scripts/gpytorch_porting/acquisition.py`** — Where utility functions are consolidated. Understand current API before adding adaptive_r_max.
4. **`scripts/gpytorch_porting/utils.py`** — Where compute_adaptive_rmax moves to, and where compute_H/get_gp_conditional_moments live.
5. **`scripts/gpytorch_porting/kernels.py`** — Where SimpleArcCosineKernel will be added. Read ArcCosineKernel.forward() lines 440-491 for the math to extract.
6. **`scripts/gpytorch_porting/likelihoods.py`** — PoissonLikelihood that all scripts will use.
7. **`scripts/1D_playground/gp_utility_playground.py`** — Source of VariationalGP, train_gp, generate_poisson_data (kept imports).
8. **`scripts/2D_playground/arccosine/utility_acos_2d_base.py`** — Secondary file being updated.

## Caveats and Open Questions

1. **compute_adaptive_rmax defaults (max_rmax=100, min_rmax=200)**: Almost certainly a bug or confusing naming. Needs investigation before moving. The function may always return 200 with default args, which would mean all callers override the defaults.

2. **DEVICE/DTYPE still imported from 1D**: The plan marks these as "TBD — define locally or keep importing". They're trivial (`torch.device('cuda'...)`, `torch.float32`) but the user hasn't explicitly decided. Defining locally would reduce the 1D dependency further.

3. **Training functions don't use likelihood.expected_log_prob**: This is correct but potentially confusing. If someone later makes A/lambda0 non-trivial for the RBF case, training would silently ignore them because compute_elbo computes ELL inline. This is a known limitation of keeping the 1D training functions.

4. **Scope of "other scripts" updates (Step 5e)**: `rbf_2d_conditioning.py`, `diagnose_acos_2d_conditioning.py`, `kernel_and_utility_comparison.py`, `diagnose_acos_2d_utility.py` -- these need import updates but haven't been read in detail. Some may call removed functions. Read each before editing.

---

## Continuation Prompt

```
Continue implementation from handoff:
.claude/handoffs/HANDOFF_2026-02-08_cleanup-2d-playground-imports.md

Plan file: .claude/plans/partitioned-scribbling-hanrahan.md

Task: Clean up 2D playground folder imports to use gpytorch_porting as
single source of truth for kernel, likelihood, and utility functions.

Before starting:
1. Check git status and git branch (should be pietro/acquisition-functions)
2. Read the handoff file for decisions, rationale, and subtleties
3. Read the plan file for step-by-step implementation
4. Read utility_2d_rbf_base.py to understand current state

Key constraints:
- SimpleArcCosineKernel: sigma_0 only, no Amp (Amp is part of C)
- Keep VariationalGP from 1D playground (model class not changing)
- No backward compat for checkpoints -- retrain after cleanup
- Verify compute_adaptive_rmax defaults before moving (suspected bug)
- importlib.util pattern required for gpytorch_porting imports (sys.path collision)

Implementation order: Steps 1-3 (gpytorch_porting additions) first,
then Steps 4-5 (2D cleanup), then Step 6 (verify by running scripts).
```
