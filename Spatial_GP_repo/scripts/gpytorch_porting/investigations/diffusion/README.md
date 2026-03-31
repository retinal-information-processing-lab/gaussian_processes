# Diffusion Model Investigation

**Branch**: `pietro/diffusion-investigation`
**Status**: Concluded -- superseded by pretrained model approach (see `pietro/approach-d-guided-sampling`)

## What this is

A self-contained investigation into training a small DDPM (denoising diffusion probabilistic model) on 64x64 PNAS natural images, then using it as a naturalness regularizer during GP utility optimization.

This was the FIRST diffusion investigation. It trained a 2.16M-parameter UNet from scratch on 3160 images with random crop + D4 augmentation. The model was later superseded by a 99.5M ImageNet-pretrained model (separate branch) which works with both DDPM and DDIM sampling.

## What we learned

1. The tiny model produces decent unconditional samples with DDPM (999 stochastic steps) but FAILS with DDIM (deterministic, fewer steps) -- output is blurry gradients, not natural images. This limitation motivated switching to the larger pretrained model.

2. The T-1 sampling fix matters: starting reverse process at t=999 instead of t=1000 avoids 31.6x amplification from the cosine schedule endpoint.

3. For diffusion-guided utility optimization, four guidance approaches were analyzed (see `GUIDANCE_POSSIBILITIES.tex`):
   - **Approach A** (L2 toward Tweedie estimate): Implemented here. Cheap but weak near the natural manifold.
   - **Approach D** (full guided reverse diffusion): Most principled, implemented on the `approach-d` branch with the pretrained model.

4. The L2 Tweedie penalty (Approach A) prevents structural degradation but not norm amplification. The arc-cosine kernel's K(x,x)~||x||^2 drives utility higher by amplifying pixel norms, and the Tweedie estimate of an amplified image is itself amplified.

5. A reusable `setup_gp(crop_size)` function was developed that trains a GP at any square resolution (not just 108x108). Validated at 64x64: test_r=0.77.

## Files

### Core DDPM (complete, self-contained)
| File | Purpose |
|------|---------|
| `diffusion_model.py` | UNet (2.16M params), cosine schedule, forward/reverse process |
| `train.py` | Training loop with random crop + D4 augmentation |
| `sample.py` | Unconditional generation + evaluation plots |
| `checkpoints/` | Trained model weights (gitignored) |
| `samples/` | Generated images (gitignored) |

### Diffusion-guided optimization (Approach A, exploratory)
| File | Purpose |
|------|---------|
| `guided_optimization.py` | GP training at any crop size + combined utility+diffusion optimization |
| `no_masking/compare_masking.py` | Comparison: RF-masked vs full-image optimization |
| `no_masking/*.png` | Comparison results (inconclusive, needs lambda tuning) |

### Documentation
| File | Purpose |
|------|---------|
| `REFERENCE.md` | Detailed technical reference (architecture, training, schedule, all code locations) |
| `GUIDANCE_POSSIBILITIES.tex` | Analysis of 4 diffusion guidance approaches with tradeoffs |
| `unet_and_ddpm_introduction.tex` | Introductory notes on UNet architecture and DDPM mechanics |
| `investigation_log.md` | Architecture decision log |
| `SESSION_2026-03-31.md` | Session log: zero-padding fix, LBFGS unification, masking comparison |

## How it connects to the broader project

- `diffusion_model.py` is referenced by `guided_reverse.py` (on `pietro/approach-d-guided-sampling`, merged into `pietro/workingbranch`) for the Approach A baseline's Tweedie denoising, though that code path is now dead.
- `setup_gp(crop_size)` in `guided_optimization.py` is a reusable building block for any work that needs a GP at non-108x108 resolution.
- The conceptual analysis in `GUIDANCE_POSSIBILITIES.tex` informed the decision to pursue Approach D (full guided sampling) on the separate branch.
