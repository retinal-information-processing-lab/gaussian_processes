# Handoff: SimpleArcCosineNormalizedKernel for 2D Playground

**Branch**: pietro/acquisition-functions
**Date**: 2026-02-09
**Status**: Plan approved, ready for implementation
**Plan file**: `.claude/plans/mutable-finding-shell.md`

---

## Motivation

The arc-cosine kernel has a fundamental problem for gradient-based stimulus optimization: its self-kernel `K(x,x) = x^T C x + sigma_0^2` grows quadratically with input norm. When the distribution-aware utility `U(x*) = H_marg - H_cond` is optimized via gradient ascent, the marginal entropy term (which depends on GP posterior variance, which depends on `K(x*,x*)`) dominates the correlation term. The optimizer inflates `||x*||` to increase variance rather than finding structurally informative stimuli. This is visible in the current `utility_acos_2d_base.png` — standard utility peaks at domain corners `(30, -30)` where `||x||` is maximal.

The mathematical analysis in `investigations/understanding_utility/proposed_solutions_for_arccosine_k_norm.tex.txt` (Section 3) proposes a normalized kernel that removes the magnitude factor while preserving the angular structure. This session planned the implementation as a 2D playground experiment to verify the fix before applying it to the full pipeline.

## Decisions and Rationale

### 1. Kernel name: `SimpleArcCosineNormalizedKernel`

Chosen to match the existing `SimpleArcCosineKernel` naming pattern. "Simple" distinguishes it from the full `ArcCosineKernel` (which has RF structure, C matrix, pixel masking). Alternatives considered:
- `NormalizedArcCosineKernel` — shorter but doesn't match the "Simple" prefix pattern
- `ArcCosineInvariantKernel` — emphasizes scale-invariance but inconsistent naming

### 2. Kernel class location: `gpytorch_porting/kernels.py`

Placed next to `SimpleArcCosineKernel` (single source of truth for kernels). The alternative was a standalone file in `arccosine_normalized/`, but this would scatter kernel definitions across the project.

### 3. New folder `2D_playground/arccosine_normalized/` for scripts

The user explicitly requested a separate folder rather than adding scripts to the existing `arccosine/` folder. This keeps the two kernel experiments cleanly separated.

### 4. Retrain from scratch (not checkpoint swap)

A new training script creates a fresh checkpoint with the normalized kernel. The alternative (load existing arccosine checkpoint, swap kernel, recompute utilities) would produce wrong results because the variational parameters (m, V) were optimized for the unnormalized kernel and would be mismatched.

### 5. Only Solution 1 (normalization), not Solution 2 (sigmoid/arcsin kernel)

The LaTeX document proposes two solutions. The user explicitly said to ignore Solution 2 (the arc-sin/sigmoid saturation kernel). Only the normalization approach is being implemented.

## Critical Subtleties

### The normalization math

The key derivation that makes this simple:
```
K_bar(x, x') = K(x,x') / sqrt(K(x,x) * K(x',x'))
             = [(1/pi) * sqrt(v_x * v_x') * J(theta)] / [sqrt(v_x) * sqrt(v_x')]
             = (1/pi) * J(theta)
```

The magnitude terms `sqrt(v_x * v_x')` cancel exactly. The implementation only needs to drop the `M` multiplication in the final line of `forward()`. Getting this wrong (e.g., also changing how theta is computed) would break the kernel's angular structure.

### Diagonal case must return 1, not v_x

`SimpleArcCosineKernel.forward(diag=True)` returns `V1 = ||x||^2 + sigma_0^2`. The normalized version must return `torch.ones_like(V1)`. Returning `V1` in the diagonal case would silently produce the wrong posterior variance while the full kernel matrix looks correct. GPyTorch uses `diag=True` in posterior variance computations, so this would manifest as variance that still scales with norm.

### V1, V2, M still needed internally

Even though M is not multiplied into the output, it must still be computed because `cos_theta = C12 / M`. Do not optimize M away.

### sigma_0 still matters

The normalization does NOT remove sigma_0 from the kernel. It still appears inside the angle computation `cos(theta) = (x^T x' + sigma_0^2) / sqrt((||x||^2 + sigma_0^2)(||x'||^2 + sigma_0^2))`. This is what preserves the alignment peak at `c=1` and breaks scale invariance of the angle. If someone removed sigma_0 thinking "the kernel is now scale-invariant," the kernel would lose its ability to distinguish scale.

### The re-export chain

The 2D scripts import kernels through `utility_2d_rbf_base.py` which uses `importlib.util` to load from `gpytorch_porting/kernels.py`. The new kernel must be re-exported in `utility_2d_rbf_base.py` for the training and visualization scripts to import it.

## Uncommitted Changes

The branch has significant uncommitted changes from prior work (acquisition function rewrite, utility investigation, 2D playground updates). These are pre-existing and unrelated to this task. The branch is 1 commit ahead of origin.

Key modified files relevant to this task:
- `2D_playground/utility_2d_rbf_base.py` — the import hub that re-exports gpytorch_porting symbols (will need edit)
- `gpytorch_porting/kernels.py` — where the new kernel class goes (currently clean, not in diff)

## Files to Read First

1. `.claude/plans/mutable-finding-shell.md` — the implementation plan
2. `gpytorch_porting/kernels.py:494-569` — `SimpleArcCosineKernel` class (the template to copy/modify)
3. `2D_playground/arccosine/train_acos_2d.py` — training script to copy
4. `2D_playground/arccosine/utility_acos_2d_base.py` — utility script to copy
5. `2D_playground/utility_2d_rbf_base.py:59-73` — import/re-export section to update
6. `investigations/understanding_utility/proposed_solutions_for_arccosine_k_norm.tex.txt` — the math (Section 3 only)

## Caveats and Open Questions

1. **Training may behave differently**: The normalized kernel has `K(x,x) = 1` everywhere, which is a very different prior variance structure than `||x||^2 + sigma_0^2`. The same training hyperparameters (lr=0.1, 500 iterations, A_init=0.01) may not converge as well. The plan says to retrain with the same settings, but convergence should be monitored.

2. **ELBO comparison not meaningful across kernels**: A normalized kernel will have a different KL divergence magnitude than the unnormalized one, so comparing ELBO values between the two is not informative. Focus on utility behavior, not ELBO.

3. **2D playground is a toy problem**: Success in 2D does not guarantee the normalized kernel works well for high-dimensional image data. The 2D experiment validates the mathematical fix (bounded variance, no corner divergence) but the kernel's representational power for real neural data would need separate investigation.

4. **No tests planned**: The plan does not include a unit test for `SimpleArcCosineNormalizedKernel`. This is consistent with the project's "simplicity first" philosophy — the 2D visualization IS the test. A formal test could be added later if the kernel graduates to production use.

---

## Continuation Prompt

```
Continue implementing the SimpleArcCosineNormalizedKernel for the 2D playground.

Read these files first:
1. .claude/plans/mutable-finding-shell.md (implementation plan)
2. .claude/handoffs/HANDOFF_2026-02-09_normalized-arccosine-kernel-2d.md (this handoff — decisions and subtleties)
3. gpytorch_porting/kernels.py lines 494-569 (SimpleArcCosineKernel to use as template)

Check git status and git branch before starting.

Task summary: Add SimpleArcCosineNormalizedKernel to kernels.py (K_bar = J(theta)/pi,
diagonal returns 1.0), create 2D_playground/arccosine_normalized/ folder with
train_acos_normalized_2d.py and utility_acos_normalized_2d_base.py, update
utility_2d_rbf_base.py re-exports. Then train and visualize.

Key constraint: V1/V2/M still computed internally for theta; only the final output
drops M. Diagonal must return ones, not v_x.
```
