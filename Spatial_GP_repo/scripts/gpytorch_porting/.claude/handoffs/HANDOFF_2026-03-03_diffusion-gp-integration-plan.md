# Handoff: Diffusion Model + GP Utility Integration

**Branch**: `pietro/diffusion-investigation`
**Date**: 2026-03-03
**Status**: Ready for implementation
**Plan file**: `investigations/diffusion/PLAN_guided_optimization.md`

---

## Motivation

We have two working systems that operate at different resolutions:
- A **64x64 DDPM diffusion model** trained on center-cropped PNAS natural images (1000 epochs, loss plateau ~0.12, generation quality verified)
- A **GP utility optimization pipeline** operating on **108x108** images (the full PNAS resolution)

The goal is to combine them: optimize an image to maximize the GP's distribution-aware utility (information gain about a target neuron) while keeping the image "natural" using the diffusion model's learned distribution as a regularizer. This requires first solving the resolution mismatch by training a GP on 64x64 center-cropped images.

This was a planning-only session. No code was written.

## Decisions and Rationale

### D1: Replicate pipeline vs modify `run_single_config()`

**Decision**: Replicate the training pipeline steps manually in the new script using importable building blocks, rather than modifying `run_single_config()`.

**Why**: `run_single_config()` (line 579) hardcodes the data path and loads 108x108 images. Adding a `data_path` or pre-loaded data parameter would be a broader change affecting the core pipeline. Since this is an investigation script, replicating the steps (STA -> kernel -> inducing points -> model -> train -> predict) using the same importable functions is cleaner and local. All building blocks are individually importable: `compute_rf_center_from_sta`, `create_kernel`, `select_inducing_points_pivoted`, `VariationalGPModel`, `train_gpy_default`, `predict`.

**Alternative rejected**: Monkey-patching the data path, or creating a temporary npz with pre-cropped data. Both are hacky.

### D2: Parameterized crop size, not hardcoded 64

**Decision**: `setup_gp(crop_size=64)` accepts any square crop size, defaulting to 64.

**Why**: The user explicitly asked for this. The GP pipeline is already resolution-agnostic via `n_px_side` — only the data loading was locked to 108x108. Making the crop size a parameter costs nothing and makes the function reusable for future work at other resolutions.

**Note**: The diffusion model is architecture-locked to 64x64 (3 downsampling levels: 64->32->16->8). So the diffusion score function is 64-only, but the GP setup is general.

### D3: Tweedie denoising for score function, not raw score

**Decision**: Use Tweedie's formula to compute a denoised image estimate, then use the direction `(x_0_hat - x)` as the "naturalness pull", rather than computing the raw score `nabla_x log p(x)`.

**Why**: The raw DDPM score is `score(x_t, t) = -eps_theta(x_t, t) / sqrt(1 - alpha_bar_t)`. At clean images (t~0), `sqrt(1-alpha_bar_t)` is near zero, causing numerical issues. Tweedie's formula avoids this:
```
x_0_hat = (x_t - sqrt(1-abar_t) * eps_hat) / sqrt(abar_t)
```
At small t (~50), abar is ~0.99 so the formula is well-conditioned. The direction `(x_0_hat - x)` has a clear interpretation: "pull toward the nearest natural image as seen by the diffusion model."

**Alternative rejected**: Classifier guidance (adding utility gradient during reverse sampling). More principled but requires the GP to evaluate noisy intermediate images, which may give meaningless predictions. Too complex for first version.

### D4: Detach x_0_hat from computation graph

**Decision**: The Tweedie-denoised estimate is detached — no backpropagation through the UNet.

**Why**: Two reasons:
1. LBFGS does not handle stochastic gradients. Tweedie involves random noise, so the score would be stochastic if the UNet were in the graph.
2. Backprop through the UNet is expensive and unnecessary for the first version.

The resulting gradient is simply `lambda * (x_0_hat - x)` — a fixed direction toward the denoised image, recomputed at each outer step.

### D5: Optimization in raw pixel space

**Decision**: All optimization happens in raw pixel space (the GP's native space). Conversion to diffusion scaled space (`raw / scale_factor`) happens only inside the score function.

**Why**: The GP model, kernel, RF mask, and acquisition functions all operate in raw pixel space. Converting everything to scaled space would require changes throughout the pipeline. The diffusion conversion is local to one function.

### D6: Use `default_gpy` mode (not `vargp_direct`)

**Decision**: Train with `default_gpy` mode for the 64x64 GP.

**Why**: `distribution_aware_utility()` requires `model(X).covariance_matrix` for Gaussian conditioning. `vargp_direct` mode uses `EigenspacePosterior` which doesn't expose full covariance. This is the same requirement as the existing `gradient.py`.

### D7: Investigation-specific constants match explore_utility.py

**Decision**: N_TRAIN=300, M=300 (same as `explore_utility.py`), not the production defaults (n_train=500, M=100).

**Why**: The utility investigation scripts use larger M for a richer, more stable utility landscape. Matching the existing investigation constants ensures comparability.

## Critical Subtleties

1. **Neural responses are for 108x108 images, not 64x64 crops.** The neuron was shown the full 108x108 image when spike counts were recorded. After center-cropping to 64x64, the GP sees a subset of the stimulus but the response is for the full stimulus. This is scientifically valid (the RF is usually in the center) but means the 64x64 model will be a slightly worse predictor. If test_r drops below ~0.5, the RF may be outside the crop region and a different cell should be chosen. Symptom: very low test_r at 64x64 but normal at 108x108.

2. **STA is recomputed on 64x64 data.** The RF center in normalized [-1,1] coordinates will differ between 108x108 and 64x64 because the STA computation operates on different pixel grids. This is correct behavior — the STA should be computed on the data the model actually sees. But it means the RF center won't be identical to the 108x108 version.

3. **Scale factor is from the PNAS dataset, not from the diffusion model.** The diffusion model uses scale_factor=2.4780 (computed from all 3160 full 108x108 images). The center-cropped 64x64 images may have a slightly different pixel range. For the score function, use the checkpoint's scale_factor (2.4780) since the UNet was trained with that normalization. For the GP, use raw pixel values as always.

4. **The diffusion model was trained on RANDOM crops, not center crops.** The UNet learned the distribution of arbitrary 64x64 patches (with D4 augmentation), not just center crops. This is fine — the score function should generalize to center-cropped images. But the learned distribution may slightly differ from the distribution of center crops specifically.

5. **Import pattern must follow `gradient.py`'s importlib approach.** The gpytorch_porting `utils.py` shadows Python's built-in `utils` module. Use `importlib.util.spec_from_file_location()` for `utils.py` and `acquisition.py`, and `sys.path.insert` for the rest. The exact pattern is in `gradient.py` lines 40-64.

6. **Schedule tensors live on CPU.** The cosine schedule dict from `diffusion_model.py` returns CPU tensors. In `q_sample()`, indexing is done on CPU then moved to device (line 107-109). The score function must do the same.

## Uncommitted Changes

Working tree is clean. All previous work (T-1 sampling fix, REFERENCE.md) was committed in `d5f02f4`.

The plan file `investigations/diffusion/PLAN_guided_optimization.md` is a NEW file that needs to be committed.

## Files to Read First

1. **`investigations/diffusion/PLAN_guided_optimization.md`** — The implementation plan. Start here for what to build.
2. **`investigations/diffusion/REFERENCE.md`** — Full reference for the diffusion model (architecture, data pipeline, normalization, T-1 fix, training results).
3. **`investigations/utility/gradient.py`** — Template for the optimization script. The `gradient_ascent()` function (lines 173-304) and `main()` (lines 502-671) are the primary code to adapt.
4. **`investigations/utility/explore_utility.py`** — Template for the `setup()` function (lines 113-179). The new `setup_gp(crop_size)` follows this pattern.
5. **`investigations/diffusion/diffusion_model.py`** — UNet, `cosine_schedule()`, `q_sample()`. Needed for the score function.
6. **`run_single_mode.py`** — `build_config_from_defaults()` (line 242) and `run_single_config()` (line 508). Reference for the training pipeline steps that `setup_gp()` must replicate.
7. **`acquisition.py`** — `distribution_aware_utility()`. The DA utility function called in the optimization closure.

## Caveats and Open Questions

1. **test_r at 64x64 is unknown.** We expect >= 0.70 for cell 8 based on the RF being roughly centered, but this hasn't been verified. If it drops significantly, a different cell or a larger crop may be needed. This is the first thing to test.

2. **LAMBDA_DIFF=0.01 is a guess.** The relative magnitude of utility gradient vs diffusion direction is unknown. The plan includes printing both gradient norms at step 0 so the user can calibrate. May need several orders of magnitude adjustment.

3. **T_SCORE=50 is a starting point.** The noise level for the Tweedie score approximation affects how much the denoised image differs from the input. Too small t: score is noisy (network not well-calibrated). Too large t: score is too smooth (blurs detail). t=50 is a common choice but may need tuning.

4. **Detached score limits the integration.** By detaching x_0_hat, we lose the full gradient through the diffusion model. The optimization alternates between "utility step" and "naturalness projection" rather than truly jointly optimizing. This is acceptable for a first version but a proper classifier-guidance approach would be more principled.

5. **The reusable GP fitting function (`setup_gp`) has not been tested at crop_size=108 for equivalence.** The plan includes this validation but it's not guaranteed to match `run_single_config` exactly due to potential differences in execution order or random state.

---

## Continuation Prompt

```
I am continuing the diffusion model investigation, implementing
diffusion-GP utility integration.
Branch: pietro/diffusion-investigation (git worktree)

Read these files in order:
1. investigations/diffusion/PLAN_guided_optimization.md (implementation plan)
2. .claude/handoffs/HANDOFF_2026-03-03_diffusion-gp-integration-plan.md (decisions + context)
3. investigations/diffusion/REFERENCE.md (diffusion model reference)

Key context:
- Two-stage plan: (1) reusable GP fitting for any square crop size,
  (2) combined utility+diffusion gradient optimization
- Stage 1 is a KEY DELIVERABLE -- setup_gp(crop_size) trains a GP on
  center-cropped images of any square size
- Template scripts: gradient.py (optimization), explore_utility.py (setup)
- No code written yet, working tree should be clean except PLAN file

Check git status and git branch before starting.
Start with Stage 1: implement setup_gp(crop_size) and validate
test_r at 64x64 vs 108x108.
```
