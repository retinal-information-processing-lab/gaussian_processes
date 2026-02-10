# Handoff: Enforce Explicit r_max — No Silent Defaults

**Branch**: pietro/acquisition-functions
**Date**: 2026-02-08
**Status**: Ready for implementation
**Plan file**: `.claude/plans/partitioned-scribbling-hanrahan.md`

---

## Motivation

After the 2D playground import cleanup (completed earlier this session), a deep audit revealed that `r_max` (the Poisson spike count truncation for entropy computation) still silently defaults to 100 in several function signatures. The user's principle is strict: **callers must explicitly choose either a fixed r_max value or adaptive_r_max**. No function should silently fall back to a hardcoded value. The same applies to the `a`/`lambda0` parameters of `compute_H` — if the likelihood has A != 1.0 or lambda0 != 0.0, calling `compute_H` without these produces wrong entropy values.

The immediate bug: `compute_mc_diagnostics_2d()` calls `compute_H()` without passing `a` or `lambda0`, relying on defaults `a=1.0, lambda0=0.0`. If a trained likelihood has different values, the entropy computation is silently wrong.

## Decisions and Rationale

### 1. Mutually exclusive r_max vs adaptive_r_max (not implicit via None)

The user said: "we should make it clear that either we set r_max or we set the adaptive r_max, to avoid mistakes."

We chose to keep both parameters (`r_max=None`, `adaptive_r_max=False`) with explicit validation:
```python
if r_max is None and not adaptive_r_max:
    raise ValueError("Must specify either r_max=<int> or adaptive_r_max=True")
if r_max is not None and adaptive_r_max:
    raise ValueError("Cannot specify both r_max and adaptive_r_max=True")
```

Alternative rejected: collapsing into a single parameter (`r_max=None` means adaptive). This hides the distinction and makes it easy to accidentally use adaptive behavior when you wanted a specific value.

### 2. Remove all defaults from compute_H — make r_max, a, lambda0 required

`compute_H(mu, sigma2, r_max=100, a=1.0, lambda0=0.0)` becomes `compute_H(mu, sigma2, r_max, a, lambda0)`. This is the low-level math function. Its callers (acquisition.py, test_acquisition.py, all investigation scripts) already pass every argument explicitly — verified by grep across the entire codebase. The only caller that relies on defaults is `compute_mc_diagnostics_2d`, which is exactly the bug we're fixing.

Alternative rejected: keeping the defaults and relying on callers to "do the right thing". The user's parameter discipline rule is clear: if a default can silently produce wrong results, remove it.

### 3. Add likelihood parameter to compute_mc_diagnostics_2d

The function needs A and lambda0 to pass to `compute_H`. Rather than passing `a` and `lambda0` as separate floats (which could go stale if the likelihood changes), we pass the `likelihood` object itself and extract A/lambda0 inside the function. This mirrors how `acquisition.py` does it.

### 4. compute_adaptive_rmax internal defaults are OK

`compute_adaptive_rmax(mu_g, sigma2_g, safety_k=3.0, max_rmax=10000, min_rmax=200)` — the user did NOT flag these as problematic. These are algorithm-level parameters (how aggressive the truncation is), not model parameters. They don't change with the likelihood or trained model. Leaving them as defaults is acceptable.

### 5. 1D playground is out of scope

The 1D playground has its own `compute_H` in `gp_utility_playground.py:373` with the same defaults. It is a separate, simpler codebase. The user did not ask to fix it, and doing so would create cross-session scope creep.

## Critical Subtleties

### compute_H defaults are positional-capable but keyword-only in practice

After removing defaults from `compute_H`, all three parameters (r_max, a, lambda0) become positional-or-keyword. Existing callers use keyword syntax (`r_max=75, a=A, lambda0=lambda0`), so no breakage. But if any caller passed them positionally, the signature change could silently reorder arguments. Verify no positional calls exist.

### adaptive_r_max minimum is 200, not 100

When `adaptive_r_max=True`, `compute_adaptive_rmax()` enforces `min_rmax=200`. This means switching from `r_max=100` to `adaptive_r_max=True` can INCREASE the effective r_max, making computation slower but more correct. This is intentional — 200 is a safety floor.

### compute_mc_diagnostics_2d needs adaptive r_max for BOTH marginal and conditional

Just like `distribution_aware_utility()`, the function must compute separate adaptive r_max for the marginal entropy and for each MC conditional step (different mu_g/sigma2_g). Using a single r_max for all would be incorrect because conditioning changes the moments.

### The 1-line change already in utility_2d_rbf_base.py

The current git diff shows a 1-line addition to `utility_2d_rbf_base.py`: re-exporting `compute_adaptive_rmax` from gpytorch_porting utils. This was started by the previous session but the function update was not completed. The next session should build on this.

### nd_utility_new takes pre-transformed moments

`nd_utility_new(mu_g, sigma2_g, r_max)` works on log-firing rate moments (g = A*lambda + lambda0). It does NOT need a or lambda0 — the caller has already done the transform. Only `r_max` needs to be made required. Don't add a/lambda0 parameters to it.

## Uncommitted Changes

```
modified:   ../2D_playground/arccosine/diagnose_acos_2d_conditioning.png  — plot output (binary, from running script)
modified:   ../2D_playground/arccosine/kernel_and_utility_comparison.png  — plot output (binary)
modified:   ../2D_playground/arccosine/trained_arccosine_2d_checkpoint.pt — retrained checkpoint (binary)
modified:   ../2D_playground/arccosine/utility_acos_2d_base.png          — plot output (binary)
modified:   ../2D_playground/rbf_2d_conditioning.png                     — plot output (binary)
modified:   ../2D_playground/trained_rbf_2d_checkpoint.pt                — retrained checkpoint (binary)
modified:   ../2D_playground/utility_2d_rbf_base.png                     — plot output (binary)
modified:   ../2D_playground/utility_2d_rbf_base.py                      — 1-line: added compute_adaptive_rmax re-export
modified:   imgs/default_gpy_M50.png                                     — plot output (binary)
modified:   investigations/validate_utility/gradient.py                  — gradient investigation script
```

Plus untracked: experiment results (2026-02-07), investigation scripts/artifacts, session-wrap-up skill, incident_report.md, SESSION_LOG.md, handoff files.

**Note**: The user implemented the 2D cleanup plan themselves and committed it. Most code changes from the cleanup are already committed. Only the 1-line `compute_adaptive_rmax` re-export and binary outputs remain.

## Files to Read First

1. **`.claude/plans/partitioned-scribbling-hanrahan.md`** — The implementation plan. Step-by-step with exact code changes.
2. **`scripts/gpytorch_porting/utils.py:471`** — `compute_H` signature to modify (remove defaults).
3. **`scripts/gpytorch_porting/utils.py:624`** — `nd_utility_new` signature to modify.
4. **`scripts/gpytorch_porting/acquisition.py:41, 83`** — `standard_utility` and `distribution_aware_utility` signatures to modify.
5. **`scripts/2D_playground/utility_2d_rbf_base.py:225`** — `compute_mc_diagnostics_2d` to overhaul (add likelihood, r_max validation, pass a/lambda0).
6. **`scripts/2D_playground/rbf_2d_conditioning.py:278`** — Caller to update.
7. **`scripts/2D_playground/arccosine/diagnose_acos_2d_conditioning.py:277`** — Caller to update.
8. **`scripts/gpytorch_porting/tests/test_acquisition.py`** — Tests that already pass r_max explicitly (should still pass).

## Caveats and Open Questions

1. **No test for the validation errors**: The plan adds `ValueError` raises for missing r_max / conflicting r_max+adaptive_r_max. There are no existing tests that verify these errors are raised. Consider adding 1-2 quick pytest tests (e.g., `test_standard_utility_no_rmax_raises`).

2. **compute_H_MC also has a=1.0, lambda0=0.0 defaults**: This is investigation-only and all callers pass explicitly, so it was left out of scope. But for consistency, it could be updated too. Low priority.

3. **Partial edit already in tree**: `utility_2d_rbf_base.py` has the `compute_adaptive_rmax` re-export added but the `compute_mc_diagnostics_2d` function was not yet updated. The next session should complete the function update.

4. **1D playground callers of compute_H**: The 1D playground imports its OWN `compute_H` from `gp_utility_playground.py`, not from `gpytorch_porting/utils.py`. Removing defaults from the gpytorch_porting version will NOT break 1D playground scripts. Verified by grep.

---

## Continuation Prompt

```
Continue implementation from handoff:
.claude/handoffs/HANDOFF_2026-02-08_enforce-explicit-rmax.md

Plan file: .claude/plans/partitioned-scribbling-hanrahan.md

Task: Enforce explicit r_max in all utility/entropy functions. No silent
defaults. Callers must specify either r_max=<int> or adaptive_r_max=True.
Also remove a/lambda0 defaults from compute_H.

Before starting:
1. Check git status and git branch (should be pietro/acquisition-functions)
2. Read the handoff file for decisions and subtleties
3. Read the plan file for step-by-step implementation
4. Note: utility_2d_rbf_base.py already has compute_adaptive_rmax re-export
   added but compute_mc_diagnostics_2d is NOT yet updated

Implementation order: Layer 1 (utils.py) first, then Layer 2 (acquisition.py),
then Layer 3 (compute_mc_diagnostics_2d), then Layer 4 (callers), then verify.

Key constraints:
- compute_H: remove ALL defaults (r_max, a, lambda0) -> required args
- nd_utility_new: remove r_max default -> required arg
- standard_utility / distribution_aware_utility: r_max=None + validation
- compute_mc_diagnostics_2d: add likelihood param, same r_max pattern
- Run tests/test_acquisition.py after Layer 2 changes
```
