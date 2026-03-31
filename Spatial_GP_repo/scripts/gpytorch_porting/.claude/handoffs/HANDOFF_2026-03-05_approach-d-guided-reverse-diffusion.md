# Handoff: Approach D — Guided Reverse Diffusion Implementation

**Branch**: `pietro/approach-d-guided-sampling`
**Date**: 2026-03-05
**Status**: Ready for implementation
**Plan file**: `investigations/diffusion/PLAN_guided_reverse.md` (also at `.claude/plans/vast-painting-boot.md`)

---

## Motivation

We have two working systems: a GP utility optimization pipeline (distribution-aware utility for neural data) and a trained 64x64 DDPM (2.16M param UNet, cosine schedule, T=1000). Previous work on branch `pietro/diffusion-investigation` implemented "Approach A": optimizing an existing image with an L2 penalty toward the Tweedie-denoised estimate. Approach A works but has fundamental weaknesses: no gradient signal near the natural image manifold, and it cannot prevent image norm amplification during optimization.

Approach D flips the paradigm: instead of modifying an existing image, we **generate** an image through the diffusion model's reverse process, nudging each DDIM denoising step toward high GP utility via classifier guidance (Dhariwal & Nichol, 2021). The UNet always operates on inputs at the correct noise level it was trained for, staying in-distribution by construction.

The user wrote a detailed LaTeX document specifying the exact algorithm (`investigations/diffusion/approach_d_guided_reverse_diffusion.tex`). This session read all source materials, explored the codebase, designed the implementation plan, and resolved ambiguities. The next session should implement the single new file `investigations/diffusion/guided_reverse.py`.

---

## Decisions and Rationale

### D1: Single new file, no modifications to existing code

The entire implementation goes into one file: `investigations/diffusion/guided_reverse.py`. Helper functions (setup_gp, load_diffusion_model, gradient_ascent_guided, etc.) are **copied** from `guided_optimization.py` in the other worktree (`gpytorch_porting_diffusion`), not imported from it. Rationale: `guided_optimization.py` does not exist in this worktree (it is uncommitted on branch `pietro/diffusion-investigation`), so importing from it is impossible. Copying ~400 lines of well-tested code is the simplest approach.

### D2: TARGET_INDEX = 5

The user's prompt contained conflicting values: 50 in the "INVESTIGATION CONSTANTS" section, 5 elsewhere. The actual `guided_optimization.py` in the other worktree uses `TARGET_INDEX = 9`. User explicitly chose **5** when asked. This means guided_reverse.py will use a different target image than guided_optimization.py — this is intentional.

### D3: Approximate gradient mode as default

Two gradient computation modes exist:
- **Approximate** (default): UNet runs under `torch.no_grad()`, eps_hat is detached. Gradient flows only through `x_t -> (x_t - detached_eps) / sqrt(abar_t) -> x_0_hat -> utility`. Skips the UNet Jacobian entirely.
- **Full**: UNet forward pass tracked by autograd. Gradient flows through UNet's Jacobian into x_t. More accurate but ~2x slower (requires UNet backward pass).

The tex document recommends starting with approximate mode. The `--use-full-gradient` flag enables full mode for comparison. This was not debated — it follows the tex document's recommendation directly.

### D4: DDIM (deterministic, sigma=0) instead of DDPM stochastic sampling

DDIM with sigma=0 is deterministic given initial noise. This means different seeds produce different images, but the same seed always produces the same image (reproducibility). The tex document specifies DDIM. No stochastic noise is added during the reverse process.

### D5: Checkpoints at each stage

User requested confirmation checkpoints at each milestone. The plan includes `CHECKPOINT: confirm with user before continuing` markers at Stages 1-4.

### D6: Approach A baseline included for comparison

The script includes the full Approach A LBFGS optimization pipeline (copied from guided_optimization.py) as a baseline comparison. Skippable via `--no-baseline`. This allows direct utility comparison between the two approaches in the same run.

---

## Critical Subtleties

### S1: Space conversions at the boundary

The UNet operates in diffusion space (raw / scale_factor, approximately [-1,1]). The GP operates in raw pixel space. The conversion `x_0_hat_raw = x_0_hat.squeeze() * scale_factor` MUST be in the computation graph for gradients to flow from utility back through Tweedie to x_t. If scale_factor is applied under `torch.no_grad()` or the multiplication is detached, the gradient will be zero. Symptom: `g_t` is always zero even though utility is nonzero.

### S2: Schedule tensors are on CPU

The cosine schedule dict from `diffusion_model.py` keeps all tensors on CPU. Every access must index on CPU, then `.to(device)`. Symptom: RuntimeError about tensors on different devices.

### S3: UNet weight freezing vs input gradient

`unet.requires_grad_(False)` freezes the UNet parameters but still allows gradient flow through the UNet's forward computation with respect to its **input** x_t. This is correct behavior — we want `d(eps_hat)/d(x_t)` (input Jacobian) but not `d(eps_hat)/d(theta)` (parameter gradient). Symptom of getting this wrong: either no gradient (if UNet is fully detached) or massive memory usage (if parameter gradients accumulate).

### S4: Detach after every DDIM step

`x_t = x_t_next.detach()` after each step is critical. Without detaching, the computation graph grows across all S steps, causing OOM and incorrect gradients. Each step should be an independent gradient computation. Symptom: CUDA OOM after ~10 steps.

### S5: f_max guard at early steps

At t=999, abar_t ~ 0.000002, so the Tweedie estimate amplifies by ~1000x. The resulting pixel values produce extreme firing rates. The f_max guard (zero gradient when exp(mu_g) >= f_max) prevents the algorithm from chasing impossible firing rates at these early steps. Without it, early-step guidance can produce NaN. The guard should fire frequently at high t and rarely at low t — if it fires at low t, something is wrong.

### S6: The T-1 sampling fix

The DDIM subsequence must start from T-1=999, NOT T=1000. At t=1000, sqrt(1/alpha[1000]) ~ 31.6, causing divergence. The `torch.linspace(T-1, 1, ddim_steps)` construction naturally avoids t=1000. This is documented in `investigations/diffusion/REFERENCE.md` and was the key fix for unconditional generation.

### S7: Checkpoint file is in another worktree

The trained DDPM checkpoint (`ddpm_epoch1000.pt`) exists only in the `gpytorch_porting_diffusion` worktree (gitignored). The default checkpoint path must be the absolute path:
`/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting_diffusion/Spatial_GP_repo/scripts/gpytorch_porting/investigations/diffusion/checkpoints/ddpm_epoch1000.pt`

### S8: distribution_aware_utility x_samples argument

`distribution_aware_utility(model, likelihood, x_candidates, x_samples, ...)` requires `x_samples` as the second positional arg (conditioning samples). In the existing investigation code, this is `x_target.unsqueeze(0)` — a single sample. Follow this exact pattern.

---

## Uncommitted Changes

```
?? investigations/diffusion/approach_d_guided_reverse_diffusion.tex
```
This is the LaTeX document specifying the algorithm. It is the primary mathematical reference. It should be committed (it is tracked content, not a temporary file).

---

## Files to Read First

1. **`.claude/plans/vast-painting-boot.md`** — The implementation plan. Contains exact algorithm pseudocode, function signatures, CLI arguments, and verification steps.

2. **`investigations/diffusion/approach_d_guided_reverse_diffusion.tex`** — The LaTeX algorithm specification (Algorithm 2: Guided DDIM). Single source of truth for the math. Read equations 2, 15-16, 29, 34, 40 especially.

3. **`investigations/diffusion/REFERENCE.md`** — Diffusion investigation reference. Contains normalization pipeline (scale_factor=2.478), T-1 fix explanation, schedule diagnostics, DDPM architecture summary.

4. **Other worktree's `guided_optimization.py`** at:
   `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting_diffusion/Spatial_GP_repo/scripts/gpytorch_porting/investigations/diffusion/guided_optimization.py`
   — Source for all helper functions to copy (setup_gp, load_diffusion_model, gradient_ascent_guided, etc.). Lines 1-714 are the functions to copy.

5. **`investigations/diffusion/diffusion_model.py`** — UNet class, cosine_schedule(), q_sample(), p_sample(), sample(). The guided_reverse function replaces sample() with a guided DDIM loop.

6. **`acquisition.py`** — `distribution_aware_utility()` signature and return dict format. Called at each DDIM step.

---

## Caveats and Open Questions

1. **TARGET_INDEX mismatch across scripts**: guided_reverse.py will use TARGET_INDEX=5 (user's choice), while guided_optimization.py uses 9. If results are compared across the two scripts, this difference must be accounted for. The target image is different.

2. **Approximate gradient quality unknown**: The approximate gradient (skipping UNet Jacobian) may or may not produce useful guidance. The tex document argues it should work because the UNet Jacobian is "near identity" at low noise, but this is an approximation. The `--use-full-gradient` flag exists for comparison, but if approximate mode produces zero utility improvement, full mode should be tried before declaring the approach broken.

3. **Guidance scale calibration**: The optimal value of `guidance_scale` (w) is unknown a priori. The tex document suggests w in [0.1, 10]. The natural ramp factor `(1-abar_t)/sqrt(abar_t)` already modulates guidance strength by noise level. Start with w=1.0 and sweep.

4. **Utility evaluation at early DDIM steps**: At high t (high noise), the Tweedie estimate is very rough. The utility evaluation may fail (NaN, exception) or produce meaningless values. The f_max guard and try/except handle this, but it means the utility trajectory will likely be meaningless for the first ~20 steps and only become informative in the last ~30 steps. This is expected, not a bug.

5. **Memory with full gradient mode**: Each full-gradient step stores UNet intermediate activations for backprop. For 2.16M params at 64x64, this should be ~50-100MB per step. Since we detach after each step, only one step's graph exists at a time. This should fit in GPU memory, but has not been tested.

---

## Continuation Prompt

```
I am implementing Approach D (guided reverse diffusion) for the diffusion-GP
investigation. The planning session is complete and all decisions are made.

Branch: pietro/approach-d-guided-sampling (git worktree)
Worktree root: Spatial_GP_repo/scripts/gpytorch_approach_d/Spatial_GP_repo/scripts/gpytorch_porting/

Read these files IN ORDER before writing any code:
1. .claude/handoffs/HANDOFF_2026-03-05_approach-d-guided-reverse-diffusion.md (decisions + subtleties)
2. .claude/plans/vast-painting-boot.md (implementation plan with pseudocode)
3. investigations/diffusion/approach_d_guided_reverse_diffusion.tex (algorithm math)
4. The other worktree's guided_optimization.py (source for helper function copies):
   /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting_diffusion/Spatial_GP_repo/scripts/gpytorch_porting/investigations/diffusion/guided_optimization.py

Key constraints:
- Create ONE new file: investigations/diffusion/guided_reverse.py
- Do NOT modify any existing files
- Copy helper functions from guided_optimization.py (do not import — it is in another worktree)
- TARGET_INDEX = 5 (confirmed by user)
- The plan has CHECKPOINT markers at each stage — confirm with me before proceeding

Check git branch before starting. Confirm you are on pietro/approach-d-guided-sampling.
```
