# Investigation: DA Utility Gradient Ascent with Normalized Kernel

**Branch**: `pietro/acquisition-functions`
**Date**: 2026-02-10
**Status**: Continuing
**Location**: `investigations/normalized_kernel/`

---

## Problem Statement

We want to validate that the distribution-aware (DA) utility function behaves correctly under gradient-based optimization with `ArcCosineKernelNormalized`. The normalized kernel (K(x,x)=1) eliminates norm-driven utility divergence that plagued the unnormalized kernel, so gradient ascent on stimuli should be well-behaved.

The core experiment: given a target image A that we condition on, compute U_DA(x*|observe A) = H_marg(x*) - H_cond(x*|A) and verify that:
1. U_DA increases monotonically as x* moves from a perturbed version toward A (interpolation sweep)
2. Gradient ascent from the perturbed image converges toward A

Motivation: understanding how the DA utility landscape behaves is prerequisite to using gradient-based stimulus optimization in the active learning loop.

## What Was Tried

### Approach 1: Small additive noise perturbation (NOISE_SCALE=0.1)
- **What**: Target = natural pool image, start = target + small Gaussian noise within RF mask
- **Result**: Interpolation MONOTONE (U_DA: 0.1005 -> 0.1019). Gradient ascent DIVERGES from target (frac_dist 1.0 -> 1.58 after 500 steps), utility increases to 0.248.
- **Interpretation**: Along the perturbation direction, ρ² dominates (correct monotonicity). But in the full space, the gradient exploits H_marg by moving to high posterior-variance regions. A is NOT a global or local maximum of U_DA(x*|A).
- **Verdict**: Confirmed interpolation monotonicity. Gradient ascent divergence is a real phenomenon, not a bug.

### Approach 2: Gaussian smoothing perturbation (sigma=3.0, then 1.0)
- **What**: Target = natural pool image, start = Gaussian-smoothed version (sigma=3.0)
- **Result**: Interpolation NOT monotone -- smoothed image has HIGHER utility (0.111) than target (0.102). The smoothing creates an image in a higher posterior-variance region.
- **Interpretation**: Gaussian smoothing is too aggressive at sigma=3.0 -- it changes both angular structure AND pushes the image into under-explored regions. The utility difference is dominated by H_marg, not ρ².
- **Verdict**: Smoothing creates a perturbation that's too far from the target's posterior variance regime. Interesting finding but not the right perturbation for testing convergence.

### Approach 3: Synthetic bipartite target + noise start
- **What**: Target = synthetic left-dark/right-light image (±0.5) within RF mask. Start = random noise (amp=0.5). Model trained on natural images.
- **Result**: Interpolation MONOTONE (U_DA: 0.0002 -> 0.265, 1000x range). Gradient ascent barely moves (U: 0.0002 -> 0.026 after 5000 steps, only 10% of target utility). frac_dist slowly increases (1.0 -> 1.05).
- **Interpretation**: The bipartite pattern is very structured relative to the GP posterior, giving a strong interpolation signal. But gradient ascent from random noise can't find the pattern -- the gradient is tiny and mostly orthogonal to the target direction.
- **Verdict**: Confirms interpolation works spectacularly. Gradient ascent from far-away starts doesn't converge.

### Approach 4: Scaled target as start (user's latest edit)
- **What**: Start = 0.5 * target (same structure, half amplitude). LR raised to 5.5.
- **Result**: Not yet run at time of handoff.
- **Interpretation**: Tests whether gradient ascent works when the start has the correct angular structure but different amplitude.
- **Verdict**: Pending.

## Key Findings

1. **CONFIRMED**: Interpolation along the line from perturbed to target is monotone for small perturbations (additive noise) and for synthetic images (noise -> bipartite). The DA utility correctly increases as x* approaches the conditioning target along these paths.

2. **CONFIRMED**: Gradient ascent on U_DA(x*|observe A) does NOT converge to A in the unconstrained pixel space. The optimizer consistently moves AWAY from A to exploit high posterior-variance regions (increasing H_marg). This happened in all 3 experiments.

3. **CONFIRMED**: The DA utility has two competing components: H_marg(x*) (marginal entropy, grows with posterior variance) and the ρ² correlation term (grows as x* approaches A). In unconstrained optimization, H_marg dominates -- the gradient pushes toward under-explored regions rather than toward A.

4. **HYPOTHESIS**: The normalized kernel eliminates the norm divergence (K(x,x) bounded at 1), but the posterior variance σ²(x*) = 1 - k*^T K_uu^{-1} k* still varies across x* and creates a similar exploitation incentive through H_marg.

5. **CONFIRMED**: The `distribution_aware_utility()` function from `acquisition.py` works correctly with the normalized kernel. Gradient flow is intact. The function supports single-target conditioning via `x_samples = target.unsqueeze(0)` with `sample_lambda=False`.

## Why This Was Stopped

Context running out. The investigation produced clear results on interpolation monotonicity and gradient ascent behavior. The next step is to reproduce this analysis with the unnormalized kernel for comparison, and to discuss what these findings mean for the active learning stimulus optimization strategy.

## Things Noticed But Not Acted Upon

1. The model trained with `ArcCosineKernelNormalized` early-stops at iteration 31 (loss barely improves after iter 20). This is much faster than unnormalized kernel training. Worth checking if the normalized kernel model quality is sufficient.

2. The synthetic bipartite experiment showed a 1000x utility range (0.0002 to 0.265) vs ~1% for natural images. This suggests the DA utility is much more discriminative for structured synthetic patterns than for natural images.

3. The gradient norm decays rapidly during gradient ascent (from ~0.06 to ~0.0004 over 5000 steps), suggesting the optimizer reaches a plateau rather than a divergence. The utility landscape may be very flat away from the conditioning target.

4. The user changed LR from 0.1 to 0.5 to 5.5 during the session, and also changed the start image from noise to 0.5*target. These experiments weren't run before context ended.

## Uncommitted Changes

The script `gradient_normalized.py` is untracked (new file). All changes are local, nothing committed this session.

Key new/modified files in `investigations/normalized_kernel/`:
- `gradient_normalized.py` -- the main investigation script (untracked)
- `gradient_ascent_normalized.png` -- latest output figure (untracked)
- `gradient_ascent_normalized50k.png` -- 50k-step run output (untracked)

## Files Created

| File | Purpose | Keep/Delete |
|------|---------|-------------|
| `investigations/normalized_kernel/gradient_normalized.py` | Main gradient investigation script. Supports USE_SYNTHETIC flag for switching between natural and synthetic experiments. | Keep |
| `investigations/normalized_kernel/gradient_ascent_normalized.png` | Latest visualization output (bipartite synthetic experiment) | Keep (will be overwritten on next run) |
| `investigations/normalized_kernel/gradient_ascent_normalized50k.png` | 50k-step run from earlier in session | Delete (superseded) |

## If Someone Revisits This

### Next steps
- Reproduce `gradient_normalized.py` in `investigations/understanding_utility/` using the unnormalized kernel. Same visualization structure and experiment logic, but with `ArcCosineKernel` instead of `ArcCosineKernelNormalized`. This will allow direct comparison of DA utility gradient ascent behavior between normalized and unnormalized kernels.
- Discuss what the findings mean for stimulus optimization in active learning.

### What NOT to try again
- Gaussian smoothing with sigma >= 3.0 as a perturbation for natural images. It pushes the image too far from the target's posterior variance regime, causing the interpolation to go the wrong direction. If smoothing is needed, use sigma <= 1.0.
- Gradient ascent with LR=1.0 and N_STEPS=1M (the original gradient.py settings). The normalized utility is bounded and converges/plateaus quickly. LR=0.1-0.5 with 500-5000 steps is sufficient.

### Script structure notes
- `gradient_normalized.py` imports `setup()` from `explore_utility_normalized.py` for model training (M=100, N_TRAIN=100, normalized kernel)
- `distribution_aware_utility()` imported from `acquisition.py` -- fully differentiable, works with both kernel types
- `USE_SYNTHETIC = True/False` flag switches between bipartite+noise and natural+smoothing experiments
- The unnormalized version would import `setup()` from `explore_utility.py` (or use `run_single_config()` directly) and use `ArcCosineKernel`

---

## Continuation Prompt

```
Read the handoff at investigations/normalized_kernel/HANDOFF_GRADIENT.md

This session continues the DA utility gradient ascent investigation.
The normalized kernel version is at investigations/normalized_kernel/gradient_normalized.py.

Task: create a version of gradient_normalized.py in investigations/understanding_utility/
that uses the UNNORMALIZED ArcCosineKernel (standard kernel from the library).
Same visualization structure and experiment logic. The goal is to compare
gradient ascent behavior between normalized and unnormalized kernels.

Before starting, check git branch and git status.
Read gradient_normalized.py to understand the current structure.
Read explore_utility.py in investigations/understanding_utility/ for the
unnormalized kernel setup pattern.
```
