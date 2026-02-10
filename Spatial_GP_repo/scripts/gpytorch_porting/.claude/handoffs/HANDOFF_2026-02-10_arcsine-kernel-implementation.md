# Handoff: Arc-Sine Kernel Implementation

**Branch**: `pietro/acquisition-functions` (create `pietro/arcsine-kernel` from here)
**Date**: 2026-02-10
**Status**: Ready for implementation
**Plan file**: `.claude/plans/merry-nibbling-sutton.md`

---

## Motivation

The distribution-aware (DA) utility diverges under gradient ascent when using the arc-cosine kernel because K(x,x) = x^T C x + sigma_0^2 grows quadratically with input norm. The optimizer inflates ||x|| to maximize GP posterior variance instead of finding informative stimuli. The normalized kernel (K_bar(x,x) = 1) fixes this but drops test_r by ~25% because image norm carries genuine neural encoding signal.

The arc-sine kernel (Williams 1998) offers a middle ground: it uses the same C matrix / RF structure but replaces the ReLU-derived nonlinearity with an erf-derived one. The prior variance saturates smoothly at 1 for large inputs while retaining amplitude sensitivity for moderate norms. The goal is to evaluate whether this kernel can prevent utility divergence without the normalized kernel's predictive performance loss.

## Decisions and Rationale

### 1. Arc-sine kernel formula (fixed "+1", no sigma_f)

**Decided**: Use the canonical Williams (1998) formula with the "+1" term fixed (not learnable), and no output scale sigma_f.

```
K_sat(x, x') = (2/pi) * arcsin( (x^T C x' + sigma_0^2) / sqrt((1 + v_x)(1 + v_x')) )
```

**Why**:
- The "+1" falls directly out of the Gaussian integration for erf activations -- it is intrinsic to the derivation, not a design choice.
- sigma_f does not appear in the Williams derivation (it would correspond to a second-layer weight variance). Adding it is just wrapping in ScaleKernel.
- The user wants to evaluate the bare kernel first. A and lambda0 in the Poisson likelihood can absorb output scale. Adding sigma_f or learnable ell^2 is deferred to a follow-up if test_r is poor.

**Rejected alternative**: Learnable ell^2 replacing "+1" (too much complexity for first version). Learnable sigma_f built into the kernel (unnecessary -- ScaleKernel can be added later).

### 2. Same C matrix structure, same parameters

**Decided**: The arc-sine kernel uses the exact same C matrix and RF parameters as the arc-cosine kernel: Amp, beta, rho, eps_0x, eps_0y, sigma_0.

**Why**: The C matrix encodes receptive field structure (locality, smoothness, position). This is independent of the nonlinearity applied to the inner product. Both kernels compute v_x = x^T C x + sigma_0^2 the same way -- they differ only in what happens after.

**Key insight about Amp's changed role**: In the arc-cosine kernel, Amp scales K(x,x) without bound (higher Amp = higher prior variance). In the arc-sine kernel, Amp controls the rate of saturation toward the ceiling of 1. The output scale is capped regardless of Amp. This means training will learn different Amp values.

### 3. Subclass ArcCosineKernel (TEMPORARY approach)

**Decided**: ArcSineKernel subclasses ArcCosineKernel to inherit _compute_C_matrix(), masking, parameter properties, and clamp_hyperparameters(). Only forward() is overridden.

**Why this is TEMPORARY**: Subclassing is the fastest way to get a working kernel for evaluation. If the arc-sine kernel proves useful, the C matrix / RF structure should be factored out into a shared base class rather than inheriting from a specific kernel. The current inheritance means ArcSineKernel "is an" ArcCosineKernel, which is semantically wrong -- they share infrastructure but are different kernels.

**What "factoring out" would look like**: A `RFStructuredKernel` base class containing _compute_C_matrix(), compute_mask(), clamp_hyperparameters(), pixel coords, and all RF parameters. Both ArcCosineKernel and ArcSineKernel would subclass it. This is deferred until we know the arc-sine kernel is worth keeping.

### 4. Sandboxed investigation (no mainline changes)

**Decided**: The investigation lives in `investigations/arcsine_kernel/` with its own `run_arcsine.py` (copy of `run_normalized.py` pattern). No modifications to `run_single_mode.py` or `default_params.json`.

**Why**: This follows the established pattern from the normalized kernel investigation. It keeps the mainline code clean until we have empirical evidence the kernel works. All defaults still come from `default_params.json` via `build_config_from_defaults()`.

### 5. Default initialization (Amp=1.0, trust the optimizer)

**Decided**: Start with the same default parameter values as the arc-cosine kernel (Amp=1.0, sigma_0=1.0, etc.). Do not pre-adjust Amp for the arc-sine's saturation behavior.

**Why**: The user explicitly chose "decide after first run." We will inspect v_x statistics from the first training run to understand where images land on the saturation curve, then adjust initialization if needed.

### 6. Same depth as arc-cosine (NOT less expressive from depth)

**Clarified during discussion**: Both the arc-cosine and arc-sine kernels correspond to a single hidden layer network with infinite width. The "2-layer" in the arc-cosine docstring refers to weight matrices (input->hidden, hidden->output), not hidden layers. The only difference is the activation function (ReLU vs erf). The arc-sine RKHS is more constrained (bounded features) but this is not a depth issue.

## Critical Subtleties

### Saturation regime for PNAS data

With current trained RF parameters, v_x ~ 669 for typical PNAS images. At this scale:
- argument = v_x/(1+v_x) = 669/670 = 0.9985
- K_sat(x,x) = (2/pi) * arcsin(0.9985) = 0.965

ALL PNAS images would land in the near-saturated regime (K ~ 0.91-0.97), making the kernel behave almost identically to the normalized kernel (which dropped test_r by 25%). The optimizer is free to learn smaller Amp to bring v_x ~ O(1), but this is an empirical question. The diagnostic code in the plan prints v_x statistics specifically to monitor this.

If test_r is similar to the normalized kernel's 0.59, the saturation is too aggressive and we should consider: (a) smaller initial Amp, or (b) learnable ell^2 replacing "+1".

### Diagonal is NOT constant

Unlike the normalized kernel where K_bar(x,x) = 1 everywhere, the arc-sine kernel has:
```
K_sat(x,x) = (2/pi) * arcsin(v_x / (1 + v_x))
```
This varies with input. The `forward()` method's `diag=True` branch must compute this properly -- do NOT return `torch.ones(...)`.

### arcsin numerical stability

arcsin is defined on [-1, 1] and its gradient diverges at the boundaries. The argument is theoretically bounded by Cauchy-Schwarz, but numerical imprecision can push it outside. Clamp with eps=1e-7 (same pattern as arccos clamping in the arc-cosine kernel).

### The "+1" and ell^2 connection

The "+1" in (1 + v_x) acts like a length-scale. Replacing it with learnable ell^2:
- ell^2 = 1: canonical erf-network value
- ell^2 -> 0: recovers a form of normalized kernel (but with arcsin angular weighting, not J(theta))
- Large ell^2: more dynamic range before saturation

If we add ell^2 later, it creates an identifiability concern with Amp (both affect when saturation kicks in). Would need careful parameterization.

## Uncommitted Changes

The working tree on `pietro/acquisition-functions` has extensive uncommitted changes from previous sessions (2D playground, acquisition.py rewrite, normalized kernel investigation, utility exploration scripts). These are NOT related to the arc-sine kernel work. The handoff plan starts by branching from the current state.

## Files to Read First

1. **`.claude/plans/merry-nibbling-sutton.md`** -- The implementation plan. Contains Step 0-3 with code snippets.
2. **`kernels.py`** -- Where ArcSineKernel will be added. Read ArcCosineKernel (line 64-491) and ArcCosineKernelNormalized (line 494-567) to understand the inheritance pattern.
3. **`investigations/normalized_kernel/run_normalized.py`** -- The template for `run_arcsine.py`. Copy this, replace kernel class, add v_x diagnostics.
4. **`investigations/understanding_utility/proposed_solutions_for_arccosine_k_norm.tex`** -- Solution 2 (lines 150-176) is the mathematical source for the arc-sine kernel.
5. **`investigations/understanding_utility/key_facts.md`** -- Trained model parameters and v_x values for context on the saturation concern.

## Caveats and Open Questions

1. **Test_r prediction is uncertain**: The arc-sine kernel may perform as poorly as the normalized kernel (~0.59) if the optimizer cannot learn RF parameters that keep v_x ~ O(1). The first run is explicitly an evaluation, not a deployment.

2. **The angular weighting is different**: The arc-cosine uses J(theta) = sin(theta) + (pi-theta)*cos(theta). The arc-sine uses arcsin(cos_theta_modified). These are different functions of angle. We have not analyzed whether the arc-sine's angular weighting is worse for neural data. If test_r is poor even with good v_x range, this could be the cause.

3. **Amp bounds may need adjustment**: The arc-cosine kernel clamps Amp at 1000. With the arc-sine, Amp=1000 would put v_x deep in saturation, making gradients vanish. The existing clamp_hyperparameters() is inherited as-is, but the effective useful range of Amp is smaller.

4. **Only autograd gradient mode**: VJP and Jacobian analytical gradients are specific to the arc-cosine formula. The arc-sine kernel is autograd-only. This is fine for default_gpy mode (which uses autograd anyway) but would need new derivations for vargp_direct M-step.

---

## Continuation Prompt

```
I'm continuing from a planning session for the arc-sine kernel implementation.

Read these files first:
1. `.claude/handoffs/HANDOFF_2026-02-10_arcsine-kernel-implementation.md` (this handoff -- decisions, rationale, caveats)
2. `.claude/plans/merry-nibbling-sutton.md` (the implementation plan -- Steps 0-3)
3. `kernels.py` (where ArcSineKernel goes -- read ArcCosineKernel and ArcCosineKernelNormalized)
4. `investigations/normalized_kernel/run_normalized.py` (template for run_arcsine.py)

Then execute the plan:
- Step 0: Branch from pietro/acquisition-functions into pietro/arcsine-kernel
- Step 1: Add ArcSineKernel to kernels.py (subclass of ArcCosineKernel, override forward())
- Step 2: Create investigations/arcsine_kernel/run_arcsine.py (copy run_normalized.py, swap kernel, add v_x diagnostics)
- Step 3: Run on PNAS data with default_gpy mode, report test_r and v_x statistics

Key constraint: Do NOT modify run_single_mode.py or default_params.json. This is a sandboxed investigation.
Key constraint: The ArcCosineKernel subclassing is TEMPORARY -- if the kernel works, we'll factor out a shared base class later.

Check git status and git branch before starting.
```
