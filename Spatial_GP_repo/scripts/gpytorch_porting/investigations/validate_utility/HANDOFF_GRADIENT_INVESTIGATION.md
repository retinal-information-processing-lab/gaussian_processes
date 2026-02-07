# Investigation: Gradient Flow Through Utility Functions

**Branch**: `pietro/acquisition-functions`
**Date**: 2026-02-07
**Status**: Continuing
**Location**: `investigations/validate_utility/`

---

## Problem Statement

The goal is to "walk along the utility landscape" — optimize input images x* via gradient ascent on the acquisition function U(x*). This requires end-to-end gradient flow from the utility value back through the Laplace approximation, GP posterior, and kernel to the input pixels.

The original motivation: understand how the utility landscape varies with pixel values, using gradients to find maximally informative stimuli for active learning with neural data.

This session was the first milestone: (1) enable gradient flow through both standard and distribution-aware (DA) utility, (2) create a diagnostic script that trains a model and verifies gradients are correct.

## What Was Tried

### Approach 1: Trace gradient flow through existing code

- **What**: Read all LaTeX derivations, `acquisition.py`, all imported functions (`get_marginal_moments`, `get_conditional_moments_nd`, `compute_H`, `nd_utility_new`, `laplace_approximations_new`, `LambertWLogFunction`) to map the full computational graph and identify gradient blockers.
- **Result**: Found **three independent gradient blockers**:
  1. `get_marginal_moments()` in `gp_utility_playground.py:328` wraps in `torch.no_grad()`
  2. `get_conditional_moments_nd()` in `utility_2d_rbf_base.py:121` wraps in `torch.no_grad()`
  3. `distribution_aware_utility()` in `acquisition.py:139,141` calls `.item()` on `lambda_i`, converting tensor to Python float
  4. **CRITICAL (found during testing)**: `laplace_approximations_new()` in `utility.py:328` uses `log_p = torch.empty(N, R)` then `log_p[idx] = expr` — indexed assignment into a pre-allocated leaf tensor breaks autograd silently. Both `nd_utility_new` and `compute_H` call this function.
- **Interpretation**: The comment at `acquisition.py:96-98` claiming "keeps the door open for gradient-based optimization" was wrong. Gradients were completely broken at multiple levels.
- **Verdict**: Identified all blockers. All fixed (see Approach 2).

### Approach 2: Create local differentiable replacements

- **What**: Since we cannot modify playground/utility.py files, created local differentiable versions in `utils.py`:
  - `get_gp_marginal_moments()` — model(x) without `torch.no_grad()` (3 lines)
  - `get_gp_conditional_moments()` — Gaussian conditioning without `torch.no_grad()`, accepts tensor lambda_sample (20 lines)
  - Full differentiable Laplace pipeline: `_LambertWLogFunction` (copied from utility.py), `_lambertw0_log`, `_diff_argmax_g`, `_diff_laplace_log_probs` (uses `torch.where` instead of indexed assignment), `compute_entropy_diff` (replaces `compute_H`), `compute_utility_diff` (replaces `nd_utility_new`)
- **Result**: All 6 existing tests pass (one tolerance adjusted from `atol=0` to `atol=1e-14` due to `torch.log1p` vs `GP_utils.safe_log` float path difference — max observed diff was 4.4e-16). Gradient flow verified end-to-end.
- **Interpretation**: The Laplace pipeline duplication was larger than originally planned, but it eliminates ALL playground imports from `acquisition.py`, making it fully self-contained.
- **Verdict**: Complete success. `acquisition.py` now has zero external dependencies.

### Approach 3: Run gradient investigation script

- **What**: Created `investigations/validate_utility/gradient.py` that trains a default_gpy model (M=50, n_train=50) and computes both standard and DA utility with gradient flow enabled. Reports utility values, gradient norms, RF mask structure, and comparison statistics.
- **Result** (from single run, seed=42, cell=8):
  - Model quality: test_r=0.7283 (well above MIN_TEST_R=0.3 threshold)
  - Standard utility: range [0.0015, 0.191], gradient norms [1.2e-3, 4.8e-2], 0.2s
  - DA utility: range [0.0001, 0.011], gradient norms [1.1e-4, 2.2e-3], 0.7s
  - Non-zero gradients: ~23,000 / 233,280 (~10% of pixels) for both utilities
  - Gradient norm correlation (std vs DA): 0.9779
  - Cosine similarity (std vs DA): mean 0.82, range [0.55, 0.94]
  - Candidate #15 is clear outlier — highest utility and gradient norm in both metrics
- **Interpretation**: Gradients are non-zero, structured, and confined to the RF region. DA gradients are ~10-20x smaller (DA utility is smaller). High correlation between std and DA gradient norms suggests they identify similar high-utility regions, but the cosine similarity spread (0.55-0.94) shows DA adds its own directional structure from the conditioning term.
- **Verdict**: Milestone 1 complete. Ready for gradient-based optimization experiments.

## Key Findings

1. **CONFIRMED**: `laplace_approximations_new` in `utility.py` breaks autograd via indexed assignment (`torch.empty` + `__setitem__`). This is the root cause that prevents gradient-based optimization of x*. Fix: use `torch.where` for differentiable branching between Laplace and Poisson paths.

2. **CONFIRMED**: GPyTorch's `model(X)` in eval mode preserves gradients when X has `requires_grad=True`. The posterior `.mean` and `.variance` have proper `grad_fn` (AddBackward0, ExpandBackward0). No special handling needed for the GP model itself.

3. **CONFIRMED**: `dU(x_i)/dx_j = 0` for `i != j` — each candidate's utility depends only on its own kernel values. Verified empirically: `.sum().backward()` gives correct per-candidate gradients.

4. **CONFIRMED**: n_train=50, M=50, default_gpy gives test_r=0.7283 (seed=42, cell=8). This is good enough for gradient investigation.

5. **CONFIRMED**: The `importlib.util` approach is required to import from local `utils.py` because playground imports cache the repo-root `utils.py` in `sys.modules`. Simple `sys.path` manipulation is insufficient.

6. **HYPOTHESIS**: The ~10% non-zero gradient pixels correspond to the RF mask region. The RF mask check in the script printed "No RF mask found on kernel (use_mask=False?)" — the kernel stores the mask as `alpha` but the check `hasattr(kernel, 'alpha')` might not find it at the right level. The 10% sparsity is consistent with RF masking though.

7. **CONFIRMED**: `acquisition.py` now has zero playground imports. All Laplace/entropy computation is local. The `sys.path` manipulation block, the `gp_utility_playground` import, the `utility_2d_rbf_base` import, and the `utility` import are all gone.

8. **CONFIRMED**: The local Lambert W implementation (`_LambertWLogFunction`) with custom backward `dW/dy = W/(1+W)` matches the original to machine precision (test_standard_utility passes with atol=1e-14).

## Why This Was Stopped

Context running out. Milestone 1 (gradient flow verification) is complete. The next step — actually walking along the utility landscape via gradient ascent — is the natural continuation.

## Things Noticed But Not Acted Upon

1. **RF mask check needs fixing**: The `check_rf_mask_structure` function in `gradient.py` looks for `model.covar_module.alpha` but the arc-cosine kernel stores the mask differently. The gradient sparsity (~10%) strongly suggests RF masking works, but the explicit check should be fixed to verify pixel-level agreement.

2. **DA utility is ~10-20x smaller than standard utility**: This is consistent with prior session findings (see HANDOFF_SESSION_NOTES.md). The conditioning step reduces the entropy difference. For gradient ascent, this means DA gradients will be weaker — may need different step sizes.

3. **Memory for DA with gradients**: With n_mc=100, n_candidates=20, the DA utility with gradient tracking completed in 0.7s. For larger n_mc or n_candidates, memory could become an issue since the computational graph accumulates across the MC loop. Gradient checkpointing or chunk-wise backward could help.

4. **Candidate #15 is a strong outlier**: Standard utility 0.191 vs mean 0.015. Worth investigating what makes this image special — it might be near the edge of the RF or in a low-data region.

5. **The `sample_lambda=True` path with gradients**: We used `sample_lambda=False` for deterministic gradients. With `sample_lambda=True`, `lambda_i = mu_i + sigma2_i.sqrt() * randn()` — the `randn` noise doesn't depend on x_candidates, so gradients still flow correctly. But the gradient estimate will be noisy. Could use reparameterization trick with fixed noise seeds for reproducible stochastic gradients.

6. **test_acquisition.py still imports from playground**: The test file imports `nd_utility_new` from `utility.py` to compare against. This import works but still has the `sys.path` side effect within the test file. Not urgent since tests are isolated, but worth noting.

## Uncommitted Changes

```
modified:   acquisition.py              # Rewritten: zero playground imports, uses local differentiable pipeline
modified:   utils.py                    # Added: GP moment functions + full Laplace pipeline (~254 lines)
modified:   tests/test_acquisition.py   # Tolerance adjustment: atol=0 -> atol=1e-14 (justified: different float path)
modified:   .claude/rules/working_guidelines.md  # Session wrap-up section updated (user edit)
modified:   imgs/default_gpy_M50.png    # Image file change (pre-existing, not from this session)

untracked:  investigations/validate_utility/gradient.py   # NEW: gradient investigation script
```

**All code changes are from THIS session and should be committed together.**

The `imgs/default_gpy_M50.png` and `.claude/rules/working_guidelines.md` changes are pre-existing (from earlier sessions).

## Files Created

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| `investigations/validate_utility/gradient.py` | Gradient investigation script: trains model, computes utilities with grads, reports statistics | Keep |
| `investigations/validate_utility/HANDOFF_GRADIENT_INVESTIGATION.md` | This handoff document | Keep |

## If Someone Revisits This

**What to try next** (most promising direction first):

1. **Gradient ascent on x***: Start from a natural image, compute dU/dx*, take a step in the gradient direction, repeat. Visualize the trajectory. Both standard and DA gradients are available. Use `sample_lambda=False` initially for reproducibility.

2. **Fix RF mask check**: Update `check_rf_mask_structure` in gradient.py to find the mask at the right kernel attribute (may be `model.covar_module.base_kernel.alpha` or similar). Verify gradient sparsity matches RF mask exactly.

3. **Visualize gradients as images**: Reshape gradient (11664,) to (108, 108) and display. Should show structure within the RF. Compare standard vs DA gradient images for the same candidate.

4. **Explore step size and convergence**: DA gradients are ~10-20x smaller than standard. May need separate step sizes. Consider normalized gradient ascent (step along gradient direction with fixed pixel magnitude).

5. **Stochastic DA gradients**: Enable `sample_lambda=True` with fixed noise seeds. Compare gradient variance across MC realizations.

**What NOT to try again**:

- Do not try to fix gradient flow by modifying playground files — the project rule is to not modify imported files. The local pipeline approach is cleaner and is already done.
- Do not try simple `sys.path` manipulation to import local `utils.py` — it fails because `sys.modules` caching from playground imports. The `importlib.util` approach is required and working.

**Prerequisites that are already met**:

- Gradient flow works end-to-end for both standard and DA utility
- Model quality with n_train=50, M=50 is sufficient (test_r=0.73)
- All existing tests pass
- `acquisition.py` is fully self-contained (no playground imports)

---

## Continuation Prompt

```
I'm continuing the gradient investigation for utility functions from a previous
session. Read the handoff at:
  investigations/validate_utility/HANDOFF_GRADIENT_INVESTIGATION.md

Summary of where we left off:
- Milestone 1 COMPLETE: gradient flow through both standard_utility and
  distribution_aware_utility is working end-to-end
- acquisition.py was rewritten to eliminate all playground imports, using a local
  differentiable Laplace pipeline in utils.py
- gradient.py script trains a model (M=50, n_train=50, default_gpy) and verified
  non-zero, structured gradients for both utilities
- All 6 tests in test_acquisition.py pass

UNCOMMITTED CHANGES exist on branch pietro/acquisition-functions. Check
git status and git diff before starting.

Next step: gradient ascent on x* — start from a natural image, walk uphill in
utility using the gradients. The gradient.py script already has the model
training + utility evaluation infrastructure. We need to add an optimization loop.
```
