# 08 — Diffusion Models & Guided Generation (Synthesizing High-Utility Natural-Image Stimuli)

**Cluster scope.** This note consolidates the theory that bridges the GP
active-learning *utility* to image **synthesis**: instead of only *selecting*
the next natural-image stimulus from a fixed pool, we *generate* a new
natural-image-like stimulus that maximizes the GP utility for a target retinal
ganglion cell. A denoising diffusion model supplies a learned prior over natural
images; the GP utility gradient steers generation toward informative stimuli.

**Status up front (this is an exploratory investigation, represented honestly).**
Two diffusion models and two guidance paradigms exist in the tree. The
*currently adopted* generation path is **guided reverse diffusion ("Approach D")**
in `guided_reverse.py`, using a 99.5M-parameter ImageNet-pretrained,
PNAS-finetuned UNet. An earlier **L2-Tweedie regularizer ("Approach A")** in
`guided_optimization.py` uses a small 2.16M-parameter from-scratch DDPM and is
now largely a baseline. As of the 2026-07-09/10 runtime audit, **both scripts are
stale against the current GP engine and need small fixes to run, and generation
is fragile** (details in §7). Nothing here has been validated on real neurons.

**Math rendering.** This is a Markdown note (not compiled LaTeX), so equations
are written in plain-text unicode. Conventions used throughout:
- `ᾱ_t` = ā_t = the cumulative signal-retention product ∏_{s=1..t}(1−β_s).
- `√` = square root; `∇_x` = gradient w.r.t. x; `·` = scalar/matmul; `≈` ≈;
  `∝` proportional; `‖·‖` = Euclidean norm; `𝒩(·; μ, Σ)` = Gaussian.
- **Every symbol clash with the GP side is tabulated in §8 — read it.** The most
  dangerous: **β is the diffusion noise-schedule variance here AND the arc-cosine
  kernel's RF-locality parameter in the GP.**

---

## 1. Motivation — why synthesize instead of select

Source: `motivation_diffusion_guided_optimization.md`,
`diffusion_models_introduction.tex` §6 & §8, `guided_diffusion_summary.tex` §1.

### 1.1 The problem with direct gradient ascent on the utility

We optimize an image x\* to maximize a GP-based utility U(x\*) (information gain
about the neuron's response). The utility gradient flows through the GP kernel,
whose derivative w.r.t. the image involves a matrix C:

    ∇_{x*} k(x*, z_m) = −k(x*, z_m) · C · (x* − z_m) / l²

where z_m are inducing points and l a lengthscale. C contains a smoothing block
`C_smooth`, a Gaussian kernel on pixel locations:

    C_smooth[i,j] = exp(−ρ · ‖ξ_i − ξ_j‖²)

Multiplying by `C_smooth` is spatial convolution with a Gaussian = a **low-pass
filter**. Consequence: **the utility gradient is inherently smooth**, regardless
of inducing points, training data, or utility definition. So gradient-ascent
images:
- lack high-frequency natural-image structure (edges, textures, local contrast);
- push pixels outside the physical display (DMD) range (norm amplification —
  worsened by the arc-cosine kernel, whose K(x,x) ∝ ‖x‖² rewards larger norms).

These images cannot serve as valid physiological stimuli: they must be
indistinguishable from natural images to be shown to real retinal ganglion cells.

### 1.2 Why subspace methods are insufficient

- **PCA-space optimization** constrains the power spectrum but captures only
  *second-order* statistics. Under approximate stationarity of natural-image
  statistics, PCA eigenvectors → Fourier modes (Szegő's theorem), so PCA-space
  optimization ≈ Fourier low-pass filtering. Neither enforces higher-order
  structure (phase coherence, edge sparsity) that makes images look natural.
- **C-eigenspace optimization** only restricts to kernel-sensitive directions,
  which gradient ascent already implicitly follows — a convergence aid, not a
  naturalness constraint.

### 1.3 The decomposition, and why diffusion

The task has two objectives:
1. **Informativeness** — x\* maximizes utility (reduces response uncertainty).
2. **Naturalness** — x\* is a plausible sample from the natural-image distribution.

Gradient ascent handles (1) not (2). A diffusion model trained on natural images
learns the **score** s(x,t) = ∇_x log p_t(x) — the gradient of the data
log-density at noise level t. At low noise it encodes the *full* statistical
structure of natural images (higher-order statistics PCA/Fourier miss). Guided
generation combines them:

    score_guided = s_θ(x_t, t) + w · ∇_{x_t} U(x̂_0)

The score model supplies **high-frequency naturalness**; the utility gradient
supplies a **low-frequency directional bias** toward informative regions. The
very smoothness that hurt direct optimization becomes acceptable: the utility
only needs to *bias* generation toward informative regions, not prescribe pixels.

**Connection to the closed-loop active-learning goal.** In the fixed-pool
closed-loop experiment, active learning *picks* the pool image of highest
expected information gain. Diffusion-guided synthesis is the natural extension:
*manufacture* a stimulus of even higher utility that is still natural enough to
present on the DMD to a retinal ganglion cell — closing the gap between "best
available image" and "best possible informative stimulus".

**Three science questions framing the package** (`HANDOFF_diffusion_package.md`):
1. Can we generate images that are simultaneously natural-looking AND high-utility?
2. How does that image change as the GP sharpens from n_train = 50 → 300?
3. How much utility does the naturalness constraint cost — the gap between the
   unconstrained "pure utility" optimum and the diffusion-guided one, vs guidance
   strength w?

> **Doc-vs-code staleness:** `motivation_*.md` says "30×30 grayscale (~900 px)"
> and "No implementation exists yet." Both are **stale**: the implemented
> resolution is **64×64 (d = 4096)**, and both Approach A and Approach D are
> implemented. Treat the *reasoning* as current, the *numbers/status* as historic.

---

## 2. Diffusion fundamentals — forward (noising) process

Sources: `diffusion_models_introduction.tex` §2–3, `unet_and_ddpm_introduction.tex`
§2, `diffusion_model.py`.

Let x_0 ∈ ℝ^d be a clean image. Choose T noise levels with a variance schedule
β_1,…,β_T ∈ (0,1). Define α_t = 1 − β_t and the cumulative retention
ᾱ_t = ∏_{s=1..t} α_s = ∏_{s=1..t}(1 − β_s).

**Per-step forward kernel** (scales signal by √(1−β_t), adds variance β_t):

    q(x_t | x_{t−1}) = 𝒩( x_t ; √(1 − β_t) · x_{t−1} , β_t · I )        (F1)

**Closed-form marginal at arbitrary t** (the key property — no intermediate steps):

    q(x_t | x_0) = 𝒩( x_t ; √ᾱ_t · x_0 , (1 − ᾱ_t) · I )                (F2)

**Reparameterization** (one-step sampling):

    x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε ,   ε ~ 𝒩(0, I)                   (F3)

The schedule is chosen so ᾱ_T ≈ 0 ⇒ q(x_T | x_0) ≈ 𝒩(0, I): at the end of the
forward process all information about x_0 is destroyed.

---

## 3. The score function and its link to noise prediction

Source: `diffusion_models_introduction.tex` §3.

**Score** of a density p is ∇_x log p(x): a vector field pointing toward higher
density. Time-dependent version at noise level t: ∇_{x_t} log p_t(x_t), where
p_t is the marginal of x_t.

Intuition by noise level:
- high t (≈T): p_t ≈ Gaussian, score ≈ −x_t (toward origin), little data info;
- moderate t: score captures large-scale layout, brightness, coarse features;
- low t (≈0): score encodes fine structure — edges, textures, local contrasts
  (the full natural-image statistics).

**Score ↔ noise identity.** Differentiating the conditional (F2):

    ∇_{x_t} log q(x_t | x_0) = −(x_t − √ᾱ_t · x_0)/(1 − ᾱ_t) = −ε/√(1 − ᾱ_t)  (S1)

So knowing the score at x_t is equivalent to knowing which noise ε produced x_t.
A network trained to predict ε therefore approximates the score:

    ∇_{x_t} log p_t(x_t) ≈ s_θ(x_t, t) = −ε_θ(x_t, t) / √(1 − ᾱ_t)          (S2)

The network is *trained to predict noise*; the score emerges as a consequence.

---

## 4. Training objective — denoising score matching (simplified L2)

Source: `diffusion_models_introduction.tex` §4, `unet_and_ddpm_introduction.tex`
§4.2.

Train a noise predictor ε_θ(x_t, t) to recover the noise added to x_0:

    ℒ = 𝔼_{ t~U(1,T) } 𝔼_{ x_0~p(x_0) } 𝔼_{ ε~𝒩(0,I) } [ ‖ ε_θ(x_t, t) − ε ‖² ]   (T1)

with x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε. Per step: sample x_0, sample t, sample
ε, form x_t, minimize ‖ε_θ(x_t,t) − ε‖² by gradient descent (Adam). Different
images in a batch get different t; the model learns all noise levels though it
sees each infrequently.

**Data pipeline for the small model** (`REFERENCE.md`, `train.py`): PNAS dataset,
~3160 images (108×108×1). On-the-fly augmentation each epoch: random 64×64 crop
(power-of-2 for clean UNet down-sampling) + random D4 symmetry (4 rotations × 2
reflections = 8 variants — natural patches at this scale are ~D4-invariant).
Together → effectively unlimited training examples. Normalize to ≈[−1,1] via a
global scale factor c = max|pixels| = 2.478 (see §6.4). Training MSE plateaus
≈0.12 (min 0.117 at epoch 909).

---

## 5. Reverse (denoising) process — generating images

Sources: `diffusion_models_introduction.tex` §5, `unet_and_ddpm_introduction.tex`
§4.3, `diffusion_model.py:p_sample/sample`.

### 5.1 Reverse transition and mean

The generative reverse process is Gaussian:

    p_θ(x_{t−1} | x_t) = 𝒩( x_{t−1} ; μ_θ(x_t, t) , σ_t² · I )              (R1)

with mean derived from the noise predictor:

    μ_θ(x_t, t) = (1/√(1−β_t)) · [ x_t − (β_t / √(1 − ᾱ_t)) · ε_θ(x_t, t) ]  (R2)

and fixed variance σ_t², typically σ_t² = β_t (simpler) or the posterior
σ_t² = β̃_t = β_t·(1 − ᾱ_{t−1})/(1 − ᾱ_t) (theoretically optimal). In practice
σ_t² = β_t works well.

### 5.2 Sampling algorithm (DDPM)

1. x_T ~ 𝒩(0, I).
2. For t = T, T−1, …, 1: predict ε̂ = ε_θ(x_t, t); compute μ_θ (R2);
   sample x_{t−1} = μ_θ + σ_t·z, z ~ 𝒩(0,I) for t>1, z = 0 for t=1.
3. Return x_0 — a *novel* image whose statistics match p(x), not in the training set.

**T−1 start fix (important, both models).** The cosine schedule at t=T=1000 has
α_T ≈ 0.001 ⇒ 1/√α_T ≈ 31.6, so the first reverse step amplifies any prediction
error ~31×, causing divergence. Start the loop at **t = T−1 = 999** (1/√α ≈ 2.0,
stable). Implemented in `diffusion_model.py:sample` (`for t in range(T-1,0,-1)`)
and mirrored in `guided_reverse.py`.

### 5.3 The Tweedie estimate (posterior mean of the clean image)

At any t, the noise prediction yields an estimate of the clean image:

    x̂_0(x_t, t) = ( x_t − √(1 − ᾱ_t) · ε_θ(x_t, t) ) / √ᾱ_t                (TW)

This is 𝔼[x_0 | x_t] under the model. Coarse at high noise, increasingly detailed
as t↓. **(TW) is the workhorse of guidance** (§6): it maps a noisy intermediate
to a clean image on which the GP utility can be evaluated, and it is a
differentiable function of x_t.

---

## 6. U-Net parameterization of the noise predictor

Sources: `unet_and_ddpm_introduction.tex` §1, `diffusion_model.py`,
`guided_diffusion_summary.tex` §2.2, `GUIDED_REVERSE_REFERENCE.md`.

The score network ε_θ(x_t, t) maps a noisy image to a same-size tensor (the
predicted noise). This same-size mapping motivates an **encoder–decoder with skip
connections** (U-Net): a plain bottleneck discards the fine detail that separates
natural images from noise; skip connections carry per-resolution detail directly
to the decoder, so the decoder sees both "what is there" (skip) and "what should
be there" (upsampled bottleneck).

**Building blocks** (small model, `diffusion_model.py`):
- **ConvBlock**: Conv2d(3×3, pad 1) → GroupNorm → SiLU, ×2, with a residual
  connection; time embedding added after the first norm+activation.
- **Downsample**: stride-2 4×4 conv (learnable ×½).
- **Upsample**: nearest-neighbor ×2 + 3×3 conv (avoids checkerboard artifacts).
- **Skip**: encoder features concatenated (channel axis) with decoder input at
  the same resolution; the doubled channels are projected back by the next conv.

**Time conditioning.** Scalar t → sinusoidal embedding (Transformer-style):

    emb(t)_{2k}   = sin( t / 10000^{2k/d_emb} )
    emb(t)_{2k+1} = cos( t / 10000^{2k/d_emb} )

passed through a small MLP (Linear→SiLU→Linear), then **added** (broadcast over
space) to each block's feature map — a global "how much to activate at this noise
level" modulation. Addition (not concatenation) is the dominant convention.

**Channel progression** (small model): 1 → 32 → 64 → 128, spatial 64→32→16→8
(bottleneck 8×8×128), then symmetric decode back to 64×64×1. GroupNorm (8 groups)
+ SiLU throughout; d_emb = 128; ~2.16M params.

### 6.1 The two models (ground-truth reality — flag)

The `.tex`/`.md` disagree on model size and schedule because **two different
models exist**; the code is ground truth:

| | Small "from-scratch" | Large "pretrained" |
|---|---|---|
| Params | 2.16M | 99.5M |
| Arch | tiny U-Net (32/64/128) | 5-level `UNet2DModel` [64,128,256,512,512], self-attention |
| Schedule | **cosine** (Nichol & Dhariwal), T=1000 | **linear** β∈[1e-4, 0.02], T=1000 |
| Provenance | trained on PNAS only | ImageNet-pretrained (1.28M grayscale) → PNAS-finetuned (3190 img, 50 ep) |
| Code | `diffusion_model.py`, `train.py`, `sample.py` | HuggingFace `diffusers` `DDPMPipeline` |
| Used by | Approach A (`guided_optimization.py`) | Approach D (`guided_reverse.py`) |
| File | `checkpoints/ddpm_epoch1000.pt` (25 MB) | `.../ddpm-pnas-finetuned/` (380 MB safetensors) |
| Why | first investigation | replaced small model — small model FAILS at DDIM (blurry gradients, not images) |

**Why the swap:** the small model produces decent *unconditional* samples with
DDPM (999 stochastic steps) but fails with DDIM (deterministic, few steps). The
99.5M model produces sharp natural images with **both** DDPM and DDIM, which
guidance needs.

> **Doc-vs-doc conflict (flag).** `guided_reverse_diffusion.tex` §2.1 states the
> Approach-D UNet is "2.16M parameters" with a **cosine** schedule — those are the
> *small* model's specs, but Approach D actually runs the **99.5M / linear** model.
> `guided_diffusion_summary.tex` §2.2 and `GUIDED_REVERSE_REFERENCE.md` correctly
> describe the 99.5M / linear model and match the code. Trust the summary + code.
> (`guided_diffusion_summary.tex` also mis-states "99.5M" in prose while the
> abstract math is model-agnostic — the *number* to trust is 99.5M from the code.)

### 6.2 Noise schedule details

**Cosine schedule** (small model) defines ᾱ_t directly:

    ᾱ_t = f(t)/f(0),   f(t) = cos²( ((t/T + s)/(1 + s)) · π/2 ),   s = 0.008
    β_t = 1 − ᾱ_t/ᾱ_{t−1}   (clipped to ≤ 0.999)

Cosine preserves more signal at low/high noise than linear (fewer wasted steps),
which helps fine natural-image structure. **Linear schedule** (large model):
β_t rises linearly 1e-4 → 0.02. All downstream constants (√ᾱ_t, √(1−ᾱ_t),
β_t/√(1−ᾱ_t), 1/√α_t, β̃_t) are precomputed once from ᾱ_t and stored.

Boundary values (cosine, from `guided_reverse_diffusion.tex` §2.1):

| t | ᾱ_t | √ᾱ_t | √(1−ᾱ_t) | β_t |
|---|---|---|---|---|
| 1 | 0.9999 | 1.000 | 0.010 | 0.0000 |
| 50 | 0.9939 | 0.997 | 0.078 | 0.0001 |
| 250 | 0.8445 | 0.919 | 0.394 | 0.0011 |
| 500 | 0.4938 | 0.703 | 0.711 | 0.0031 |
| 750 | 0.1310 | 0.362 | 0.930 | 0.0070 |
| 999 | 2e-6 | 0.001 | 1.000 | 0.7500 |

### 6.3 DDIM acceleration

Source: `guided_reverse_diffusion.tex` §6, `guided_diffusion_summary.tex` §3.4.

Full DDPM needs T−1 = 999 sequential UNet passes. **DDIM** (Song et al. 2021)
uses a subsequence of S timesteps τ_S > … > τ_1 (e.g. S=50, uniform) with a
deterministic update. Going from t=τ_i to t'=τ_{i−1}:

    x_{t'} = √ᾱ_{t'} · x̂_0 + √(1 − ᾱ_{t'} − σ²) · [ (x_t − √ᾱ_t · x̂_0)/√(1 − ᾱ_t) ] + σ·z   (DDIM)

with x̂_0 from (TW). σ controls stochasticity: **σ = 0 ⇒ deterministic** (fully
reproducible given x_T), σ = √((1−ᾱ_{t'})/(1−ᾱ_t))·√β_t recovers DDPM. Quality
degrades below S≈20; S=50 is the default starting point. At the final step set
x_0 = x̂_0 directly.

---

## 7. Guidance — the bridge from GP utility to synthesis

This is the crux of the cluster: how the GP utility gradient steers reverse
diffusion toward high-utility images.

### 7.1 Classifier-guidance conditional score

Sources: `diffusion_models_introduction.tex` §6.2, `guided_reverse_diffusion.tex`
§4, `guided_diffusion_summary.tex` §5.

Model "high utility" as a label y with p(y | x_0) ∝ exp( w · U(x_0) ), w > 0.
Bayes' rule splits the *conditional* score into naturalness + guidance:

    ∇_{x_t} log p(x_t | y) = ∇_{x_t} log p_t(x_t) + ∇_{x_t} log p(y | x_t)
                           = s_θ(x_t,t)          + w · ∇_{x_t} U( x̂_0(x_t,t) )    (G1)

- first term = unconditional score (from the trained UNet, (S2));
- second term = classifier/utility gradient.

**Separation of concerns:** the score model handles naturalness (all orders of
image statistics, trained once, reused across cells/utilities); the guidance
gradient handles the cell-specific criterion. They operate at complementary
frequency scales (§1.3).

### 7.2 Evaluating the utility on noisy intermediates — the Tweedie trick

U is defined on *clean* images, but x_t is noisy. The standard fix (Dhariwal &
Nichol 2021) evaluates U on the Tweedie estimate x̂_0 (TW):

    ∇_{x_t} log p(y | x_t) ≈ w · ∇_{x_t} U( x̂_0(x_t, t) )                     (G2)

Define the **guidance gradient**:

    g_t = ∇_{x_t} U( x̂_0(x_t, t) )                                            (G3)

which flows through two stages by the chain rule:
1. ∇_{x̂_0} U  — utility gradient w.r.t. the clean-image estimate (through the GP);
2. ∂x̂_0/∂x_t — Jacobian of the Tweedie formula.

**Chain rule through Tweedie.** From (TW):

    ∂x̂_0/∂x_t = (1/√ᾱ_t) · ( I − √(1 − ᾱ_t) · ∂ε_θ/∂x_t )                    (G4)

∂ε_θ/∂x_t is the d×d UNet Jacobian (d=4096), never formed explicitly — PyTorch
computes the vector–Jacobian product via `loss.backward()` with x_t a leaf
tensor (`requires_grad=True`), loss = −U(x̂_0), UNet weights frozen.

### 7.3 The modified reverse update (two equivalent forms)

Sources: `guided_reverse_diffusion.tex` §4.4–4.5, `guided_diffusion_summary.tex`
§5.2.

**Form 1 — guided noise.** Convert the conditional score back to a noise
prediction (s + w·g_t, then ε = −√(1−ᾱ_t)·score):

    ε_guided = ε_θ(x_t, t) − w · √(1 − ᾱ_t) · g_t                            (G5)

**Form 2 — guided Tweedie shift** (used in code). Substitute (G5) into (TW):

    x̂_0^guided = x̂_0 + [ w · (1 − ᾱ_t) / √ᾱ_t ] · g_t                       (G6)

The two are algebraically identical. (G6) shows guidance shifts the clean-image
estimate along g_t with the **ramp factor** (1 − ᾱ_t)/√ᾱ_t:
- large at high noise (t≈T, ramp ~158) — bold "utility steering" while structure
  is being decided;
- small at low noise (t≈1, ramp ~0.3, shift ~1e-4) — preserve fine detail.

The guided reverse step uses ε_guided in the DDPM mean (R2), or x̂_0^guided in
the DDIM update (DDIM). **The guidance signal is the GP utility gradient** — see
§7.6 for the explicit GP link.

### 7.4 Full vs approximate guidance gradient

Source: `guided_reverse_diffusion.tex` §7, `guided_diffusion_summary.tex` §5.1.

- **Full gradient** (`--use-full-gradient`): backprop through the UNet forward
  pass — retains ∂ε_θ/∂x_t in (G4). Cost ≈ 2× a plain step (one UNet fwd + bwd).
- **Approximate gradient** (default): treat the UNet Jacobian as negligible (run
  the UNet under `torch.no_grad()`, ε̂ detached). Then x̂_0 depends on x_t only
  through the direct x_t/√ᾱ_t term, so

    g_t^approx = (1/√ᾱ_t) · ∇_{x̂_0} U(x̂_0)                                   (G7)

  No UNet backward pass (≈½ the cost). Justification: ε_θ is a "local
  correction" whose Jacobian is ~zero-mean (doesn't systematically align with or
  oppose the utility gradient). Excellent at **low** noise (√(1−ᾱ_t) small ⇒ the
  UNet term is negligible); breaks down at high noise / when U is very sensitive
  to the exact ε. Recommended: start approximate; a hybrid (approx for early
  t>200, full for late t<200) captures most of the benefit cheaply. The ramp in
  (G6) absorbs the 1/√ᾱ_t scaling.

### 7.5 The guidance-possibilities catalogue (A / B / C / D)

Source: `GUIDANCE_POSSIBILITIES.tex` (the main catalogue). All four aim to add a
"naturalness" term that counteracts the utility gradient's drift off the natural
manifold ℳ. Here x_c denotes the current image being optimized.

**Approach A — L2 penalty toward the Tweedie estimate (Approach A code).**
Per step: corrupt x_c at a *fixed* noise level t\* (e.g. 50):
x_t = √ᾱ_{t*}·x_c + √(1−ᾱ_{t*})·ε; predict ε̂; form x̂_0 (TW); **detach** x̂_0;
add L2 penalty:

    ℒ = −U(x_c) + (λ/2)·‖x_c − x̂_0‖²                                        (A1)
    update:  x_c ← x_c + η·∇U + η·λ·( x̂_0 − x_c )

**Score-Distillation connection:** substituting definitions,
x_c − x̂_0 = (√(1−ᾱ_{t*})/√ᾱ_{t*})·(ε̂ − ε), i.e. ∝ (ε̂ − ε), the Score
Distillation Sampling (SDS) gradient with a different weighting (SDS uses
w(t)·√ᾱ_t·(ε̂−ε); this uses √((1−ᾱ_t)/ᾱ_t)·(ε̂−ε)).
*Problems:* (i) **no signal near ℳ** — for natural x_c, x̂_0 ≈ x_c so ε̂−ε is
UNet reconstruction noise (no systematic direction): the penalty is *reactive*,
not preventive; (ii) **norm amplification not prevented** — the Tweedie estimate
of an amplified image is itself amplified (preserves texture, not amplitude);
(iii) arbitrary t\*; (iv) single-noise-sample variance (fixed seed → biased).

**Approach B — direct score as gradient term.** Use s_θ(x_t,t\*) = −ε̂/√(1−ᾱ_{t*})
directly, chain-ruled to x_c space (∂x_t/∂x_c = √ᾱ_{t*}):
update adds η·w·(√ᾱ_{t*}/√(1−ᾱ_{t*}))·(−ε̂). Uses ε̂ *alone* (not ε̂−ε), so it
has signal even on ℳ but **high variance** (the actual noise ε rides along; it
cancels in A). Estimates the score of p_{t*} (blurred), not p_0; scale-mismatched
with ∇U through the 1/√(1−ᾱ_{t*}) factor.

**Approach C — score without added noise (evaluate at x_c, t=1).** Feed x_c
directly at t=1, s_θ(x_c,1) = −ε̂/√(1−ᾱ_1). Appealing (no artificial noise, no
Tweedie) since p_1 ≈ p_0. **Fails in practice:** at t=1 the UNet expects
"natural + σ≈0.01 noise"; a utility-distorted x_c (O(1) amplification) is far
out-of-distribution — the UNet must extrapolate, ε̂ is unreliable, and dividing
by √(1−ᾱ_1)≈0.01 amplifies that garbage ~100×. The distortion is *invisible* to
the network at this tiny noise budget.

**Approach D — full guided reverse diffusion (the adopted method).** Don't
optimize an existing image — **generate** from noise, nudging each denoising
step by the utility gradient (§7.1–7.3). The UNet is *always* in-distribution
(x_t is genuinely at noise level t because it came from the reverse process);
traverses coarse→fine over all noise levels; no t\* to pick; output is a proper
sample from the guided distribution. *Costs:* requires the UNet forward with
gradients (full-grad mode), T (or ~50 DDIM) sequential steps, utility on rough
early Tweedie estimates, stochastic output (best-of-K), and **no start-point
control** (starts from noise, not a chosen image).

**A/B/C/D comparison** (`GUIDANCE_POSSIBILITIES.tex` §6):

| | A: L2 Tweedie | B: Direct score | C: No noise | D: Guided sampling |
|---|---|---|---|---|
| Noise required | Yes (t\*) | Yes (t\*) | No | Inherent |
| Backprop through UNet | No | No | No | Yes |
| Score of p_0 or p_t | Neither (L2) | p_{t*} | p_1 (unreliable) | All p_t |
| Signal near ℳ | Weak | Noisy | Unreliable | Strong |
| Prevents norm growth | No | Partially | Unreliable | Yes |
| Cost/step | 1 UNet fwd | 1 UNet fwd | 1 UNet fwd | T fwd(+bwd) |
| Start point | Chosen | Chosen | Chosen | Random noise |

Middle ground for a chosen start: **SDEdit** (Meng et al. 2022) — encode a
starting image to intermediate t\* via the forward process, then guided-reverse
from t\* to 0. Preserves coarse structure while refining under guidance. Noted as
a natural extension, **not implemented**.

### 7.6 The explicit GP link (what the guidance signal actually is)

Sources: `guided_reverse_diffusion.tex` §2.3, `guided_diffusion_summary.tex`
§3 & §7, `motivation_*.md`.

The guidance signal g_t (G3) is the gradient of the **GP information-utility**
w.r.t. the image, back-propagated through Tweedie. The GP models log-firing-rate:

    λ(x) ~ 𝒢𝒫(μ(x), k(x,x')),   firing rate  f(x) = exp( A·λ(x) + λ_0 )

with learned likelihood params A, λ_0. Two utilities are supported (`acquisition.py`):

- **Standard utility:**  U_std(x*) = H[R | x*, D] − H_noise(x*)  — marginal
  response entropy minus irreducible Poisson-noise entropy.
- **Distribution-aware (DA) utility** (default):
  U_DA(x*) = H[R | x*, D] − 𝔼_{x~p(x), λ~q(λ(x)|D)} [ H[R | x*, λ(x), D] ] —
  the second term averages the *conditional* response entropy at x\* after
  hypothetically observing λ at a random pool stimulus x, crediting the fact that
  a response at x\* also reduces uncertainty at nearby stimuli.

Both entropies use a Laplace approximation to the Poisson predictive, truncated
at r_max = 100 (see §7.7 for the accuracy caveat). **Both are differentiable
w.r.t. the image** — the gradient flows through the GP kernel. The kernel is an
**arc-cosine kernel with RF structure**; its gradient w.r.t. the image is the
smooth, low-pass signal of §1.1 (∇_{x*} k = −k·C·(x*−z_m)/l², C carrying the
Gaussian spatial smoother `C_smooth`). *This is exactly why guidance works:* the
utility supplies a **low-frequency** informative direction; the score model
supplies the **high-frequency** naturalness the smooth utility gradient can never
produce on its own.

**RF mask ⇒ guidance touches only RF pixels.** The arc-cosine kernel applies a
Gaussian RF envelope α_j = exp(−d_j²/4β²) (d_j = pixel j's distance from RF
center; **β = kernel RF-locality parameter**, a *different* β from the diffusion
schedule — see §8). Pixels with α_j < 0.001 are masked out (zero contribution).
So ∇_{x*}U is nonzero **only at RF pixels** (~21% of the 64×64 image at β=0.1):
the utility guides only the RF region; **the diffusion model freely generates all
remaining pixels** for naturalness. This is an advantage over Approach A (whose
non-RF pixels are frozen at a start image). Mask area scales ~β²:

| β | Mask pixels (64×64) | Coverage |
|---|---|---|
| 0.03 | 69 | 1.7% |
| 0.05 | 221 | 5.4% |
| 0.10 (default) | 869 | 21.2% |

Tradeoff: small β → natural images (low out-of-bounds) but little utility to gain
(few pixels); large β → high achievable utility but more distortion. β is a
*learnable* kernel parameter — freeze it during GP training or the mask size
drifts from the intended value (`guided_reverse.py` `--beta` override does
`raw_m2log2beta = −2·log(2·β)` then `requires_grad_(False)`).

### 7.7 Guards, space conversion, and best-of-K

- **Firing-rate guard.** At high noise x̂_0 is rough and can give extreme pixels ⇒
  unphysical firing. If exp(μ_g) ≥ f_max (f_max = 100, "Hz"; μ_g = A·λ_mean + λ_0
  the predicted log-firing), **zero the guidance gradient** for that step. Fires
  mostly in the first ~10 steps (high t). This keeps the optimizer near the safe
  natural-image firing regime where the (truncated) utility is accurate.
  > **Utility-accuracy caveat (runtime-verified, `HANDOFF` C5).** The r_max=100
  > truncation makes *standard* utility **diverge to +inf** at high firing
  > (unbounded analytic E_H_noise term → overflow), and makes *DA* utility stay
  > finite but **silently wrong** (H_marg collapses toward 0 past firing ≈50).
  > Both share the truncation; the f_max guard is what keeps generation in the
  > firing ≈0.01–1 zone where both match Monte-Carlo truth within ~3%. An exact
  > **LUT** (lookup table) cures standard's inf and DA's truncation error; it is
  > wired for `--utility standard` in the *standalone package only*, **NOT** on
  > the working branch and **NOT** for the DA path (deliberate — see §7.8).
- **Space conversion (must stay in the graph).** The UNet works in model space
  ≈[−1,1]; the GP in raw PNAS pixels. Convert x̂_0^raw = x̂_0 · s with the scale
  factor **s = 2.478** before calling the utility. This multiply MUST be inside
  the autograd graph (not under `no_grad`/detached) or the gradient from U back to
  x_t is zero.
- **Best-of-K.** Each seed x_T gives a different image of varying utility (the
  utility landscape is sparse — most seeds land at U≈0). Run K seeds (default
  K=100 in the summary; the script default `--n-samples` is 5) and keep the
  highest-utility image. ~1 s/image (DDIM-50) ⇒ K=10 in ~10 s.

### 7.8 Why guided diffusion beats naive bounding (the manifold argument)

Source: `diffusion_models_introduction.tex` §8.

Naive pixel clipping Π_ℬ(x)_i = clip(x_i, x_min, x_max) acts on each pixel
**independently** and ignores the joint distribution p(x). Natural images lie on
a low-dimensional manifold ℳ ⊂ ℬ (local correlations, 1/f spectra, sparse edges);
clipping flattens structured gradients into uniform blocks — the result sits on
the boundary of ℬ but *far from ℳ*, invalid as a stimulus. In the guided score

    ∇_{x_t} log p_t(x_t | high U) = ∇_{x_t} log p_t(x_t)  +  w·∇_{x_t} U_t(x_t)
                                    └ restoring force ┘      └ utility force ┘

the utility force may push off ℳ, but the **score is a restoring force on the
joint distribution**: if a patch saturates and loses its natural spectrum,
p_t(x_t) drops sharply and the score exerts a *structured* counter-gradient that
rearranges pixels to keep natural correlations while accommodating the utility
bias. **Limitation (OOD):** the restoring force is a neural approximation
ε_θ trained only on the forward trajectory (natural + Gaussian noise). Too large
a w drives x_t out-of-distribution, where ε_θ extrapolates and the restoring
force becomes unreliable/artifact-prone. So w must be large enough to explore
high-utility regions yet small enough to keep x_t where the score approximation
is valid.

### 7.9 Guided algorithm (DDPM and DDIM forms)

**Guided DDPM** (`guided_reverse_diffusion.tex` Alg. 1): start x_{T−1} ~ 𝒩(0,I);
for t = T−1…1: x_t.requires_grad; ε̂ = ε_θ(x_t,t); x̂_0 (TW); convert to raw;
U, μ_g = utility(x̂_0); if exp(μ_g) < f_max: (−U).backward(), g_t = −x_t.grad,
else g_t = 0; ε_guided = ε̂ − w√(1−ᾱ_t)·g_t; μ_t = (1/√α_t)(x_t −
(β_t/√(1−ᾱ_t))·ε_guided); x_{t−1} = μ_t + √β_t·z (z=0 at t=1); **detach x_{t−1}**.

**Guided DDIM** (Alg. 2, deterministic σ=0): same guidance block, then form
x̂_0^guided (G6); direction = (x_t − √ᾱ_t·x̂_0^guided)/√(1−ᾱ_t);
x_{t'} = √ᾱ_{t'}·x̂_0^guided + √(1−ᾱ_{t'})·direction; final step returns
x̂_0^guided. **Detach after every step** — each step is an independent gradient
computation; do NOT backprop through the whole T-step chain (would need O(T)
memory; causes OOM after ~10 steps).

---

## 8. IMPLEMENTATION STATUS — code vs plans, open issues

Sources: `guided_reverse.py`, `guided_optimization.py`, `diffusion_model.py`,
`GUIDED_REVERSE_REFERENCE.md`, `REFERENCE.md`, `SESSION_2026-03-31.md`,
`HANDOFF_diffusion_package.md` (runtime-verified 2026-07-09/10),
`investigation_log.md`.

### 8.1 What Approach D (`guided_reverse.py`) actually does

A single script: **train GP → load diffusion model → guided reverse diffusion
(multi-seed) → select best → optional pure-utility LBFGS baseline → figure.**
Core = `guided_reverse_diffusion()` + `_guidance_step()`. Matches the `.tex`
algorithms, with these implemented specifics:
- Both **DDPM** (`--sampler ddpm`, all 999 steps, stochastic) and **DDIM**
  (`--sampler ddim`, S steps, deterministic) share `_guidance_step`.
- Both gradient modes: approximate (default, UNet under `no_grad`) and full
  (`--use-full-gradient`). Extra `--noisy-gradient` flag evaluates U at the noisy
  x_t instead of x̂_0 (bypasses the Tweedie chain rule — an experimental variant
  not in the `.tex`).
- Both utilities: `--utility {da,standard}`, default **da**. Note the DA path
  here conditions on a **single** sample = the target image
  (x_samples = x_target.unsqueeze(0), n_mc = 1) — "conditioned on the target",
  not an MC average over p(x).
- `unet.requires_grad_(False)` freezes weights while allowing input-gradient flow
  (want ∂ε_θ/∂x_t, not ∂ε_θ/∂θ). Detach x_t after each step (prevents graph
  growth / OOM). Space-conversion multiply kept in-graph.
- Constants: N_TRAIN=50, M=50, TARGET_INDEX=5, SIGMA_SMOOTH=5.0, DDIM_STEPS=50,
  GUIDANCE_SCALE=1.0, N_SEEDS=5, PNAS_ABS_MAX=2.478047. Seeds = 42..42+n−1.

### 8.2 What Approach A (`guided_optimization.py`) actually does

Combined utility + diffusion-regularizer optimization on RF pixels only:

    loss = −U_DA(x) + λ_diff · 0.5 · ‖x_rf − x̂_0_rf‖²

- `compute_tweedie_denoised()`: raw→model space, add artificial noise at t=50
  (fixed **seed=42**), UNet predict, Tweedie (TW), back to raw — all under
  `no_grad` (no UNet gradients). This is the L2 target x̂_0, not a score.
- **Optimizer is LBFGS** (strong_wolfe), proximal/MM pattern: x̂_0 computed once
  per outer step (outside the closure) so function/gradient are consistent for the
  line search (`guided_optimization.py:559-562, 604-650`).
  > **Doc-vs-doc conflict (flag):** `REFERENCE.md` says "Utility+diffusion: SGD
  > with momentum 0.9" and lists LBFGS as a dead-end. That is **stale** — the
  > SESSION log (2026-03-31) records unifying to the LBFGS proximal pattern *after
  > the zero-padding fix*, and the code confirms LBFGS. `REFERENCE.md` constants
  > (N_TRAIN=300, M=300, TARGET_INDEX=0, SIGMA_SMOOTH=1.0) are also stale vs code
  > (50, 50, 5→0, 5.0). Trust code + SESSION.
- **Zero-padding fix (2026-03-04):** `_reconstruct_image()` takes a `background`
  (the smoothed start x_start) so the UNet always sees a *full natural image*
  (previously ~79% zeros = OOD input that biased the score). GP kernel masks
  non-RF pixels regardless.
- Loads the **small** custom DDPM (`checkpoints/ddpm_epoch1000.pt`,
  `guided_optimization.py:869`) — NOT the diffusers model.
- Result (post-fix, λ=1.0): diffusion reg keeps 0% out-of-bounds, proj_coeff
  0.32→0.44 (no norm blow-up), Pearson r 0.74→0.65 (structure preserved), but
  U only 0.0014→0.0017 vs utility-only 0.042 (a ~25× utility sacrifice, 48% OOB).

### 8.3 Verified runtime status (2026-07-09/10) — honest current state

- **Both scripts are STALE and crash as-is** against the current GP engine:
  they pass `stop_window`/`stop_thresh` to `train_gpy_default`, removed by the
  ELBO early-stopping refactor (now `patience`/`min_delta_rel`/`es_metric`).
  → `KeyError: 'stop_window'`. **Fix = delete those 2 kwargs per script**
  (`guided_reverse.py:363-364`, `guided_optimization.py:326-327`). After the fix
  the GP trains and ELBO ES fires ("Early stopping at iteration 24").
- **`guided_reverse.py` `DEFAULT_MODEL_PATH` is BROKEN** (points at a
  nonexistent nested `gpytorch_imagenet_diffusion/...` path). The real 380 MB
  finetuned diffusers model is at
  `.../gpytorch_porting/diffusion_model_imagenet/pietro/ddpm-pnas-finetuned/`.
  → must pass `--model-path <real path>` (or fix the constant).
- **Generation is FRAGILE.** At w=50, DDIM-50, 3 seeds: **1/3 collapsed** to
  salt-and-pepper noise (U=0, 78% out-of-display-range) because the f_max guard
  fired on 49/50 steps and guidance never engaged; the "best" seed gave a
  textured-but-not-clearly-natural image. Matches the reference caveat "use
  n_samples 10+, select best". The utility landscape over seeds is sparse.
- **Guidance-scale calibration is kernel-dependent** (`GUIDED_REVERSE_REFERENCE.md`
  S5): arc-cosine (test_r~0.73) needs w~50; RBF (test_r~0.28) effective at w~5,
  overshoots by w~50. GP at 64×64 center-crop validates well (cell 8: test_r
  0.77 at crop 64, beating 0.72 at crop 108, because fewer pixels = easier fit).

### 8.4 Open issues / not-yet-done (from the logs)

- **DA-path LUT NOT done** (deliberate NON-goal per user, 2026-07-10): DA's
  H_marg/H_cond still use the truncated `compute_H`; the f_max guard is relied on
  to stay in the accurate regime. Only `--utility standard` is LUT-backed, and
  only in the standalone package.
- **Norm amplification not fully solved** (Approach A): the L2-Tweedie pull
  preserves structure, not amplitude; small LBFGS steps just delay blow-up. Untried
  fixes: sigmoid pixel bounds, norm-matching x̂_0, unit-vector score direction.
- **t\* / λ_diff / sigma-smoothing are hand-tuned heuristics** (Approach A);
  masked-vs-full-image optimization comparison was **INCONCLUSIVE** (both struggle;
  full mode has worse OOB because the L2 over all 4096 px overwhelms utility).
- **Approach D open items:** stochastic best-of-K is a cost; utility on rough
  early Tweedie estimates can mislead; SDEdit start-from-image not implemented;
  full-gradient vs approximate comparison at 64×64 not systematically swept.
- **Science not evaluated:** does the naturalness constraint actually buy
  better experimental outcomes on real retinal ganglion cells? No neuroscience
  validation done; closed-loop / oracle validation explicitly out of scope for
  the current package.

---

## 9. Notation table & symbol clashes (READ — GP↔diffusion collisions)

| Symbol | Meaning | Source / domain |
|---|---|---|
| x_0 | clean image (diffusion) = the stimulus | diffusion |
| x_t | noisy image at diffusion timestep t | diffusion |
| x\* , x_c | candidate / current image for the utility (x_c = being optimized in A/B/C) | GP-utility |
| x̂_0 | Tweedie estimate 𝔼[x_0\|x_t] = (x_t−√(1−ᾱ_t)ε_θ)/√ᾱ_t | diffusion (TW) |
| x̂_0^guided | utility-shifted Tweedie estimate (G6) | guidance |
| z_m | GP inducing points (kernel gradient) | GP |
| **t** | **diffusion timestep** (1…T). *No temporal-RF variable exists — spatial GP only* | diffusion |
| t\* | fixed corruption level in A/B (e.g. 50) | guidance |
| T | total diffusion timesteps (1000) | diffusion |
| **β_t** | **diffusion noise-schedule variance** at step t | diffusion |
| **β** | **arc-cosine kernel RF-locality parameter** (mask ~β²; default 0.1) | **GP — CLASH with β_t** |
| β̃_t | posterior reverse variance β_t(1−ᾱ_{t−1})/(1−ᾱ_t) | diffusion |
| **α_t** | **diffusion signal retention** 1−β_t | diffusion |
| ᾱ_t | cumulative retention ∏(1−β_s) (a.k.a. alpha-bar) | diffusion |
| **α_j** | **kernel per-pixel Gaussian RF envelope weight** exp(−d_j²/4β²) | **GP — CLASH with α_t** |
| ε | true Gaussian noise added in the forward process | diffusion |
| ε_θ , ε̂ | UNet noise prediction / its value | diffusion |
| ε_guided | guidance-modified noise prediction (G5) | guidance |
| **s_θ** | **score function** −ε_θ/√(1−ᾱ_t) | diffusion |
| **s** | **scale_factor** 2.478 (raw pixels ↔ model space) | diffusion code — CLASH |
| **s** | **cosine-schedule offset** 0.008 | diffusion — CLASH (3 uses of "s") |
| σ_t | reverse-step noise std (√β_t etc.) | diffusion |
| σ (DDIM) | DDIM stochasticity knob (0 = deterministic) | diffusion |
| **σ²_g** | **GP marginal variance of the log-firing rate** | GP — CLASH with σ_t |
| **w** | **guidance scale** (classifier-guidance weight) | guidance |
| w(t) | SDS time-weighting (mentioned once) | guidance — CLASH with w |
| **λ(x)** | **GP latent log-firing rate**; f = exp(A·λ + λ_0) | GP |
| λ_0 | GP likelihood log-firing offset | GP |
| **λ_diff (λ)** | **Approach-A diffusion-regularizer weight** (config `--lambda-diff`) | guidance — CLASH with GP λ |
| A | GP likelihood gain (f = exp(A·λ + λ_0)) | GP |
| μ_g | predicted log-firing mean A·λ_mean + λ_0 (f_max guard) | GP |
| U, U_std, U_DA | GP information utility (standard / distribution-aware) | GP |
| g_t | guidance gradient ∇_{x_t}U(x̂_0) (G3) | guidance |
| H_marg, H_cond, H_noise | marginal / conditional / Poisson-noise response entropies | GP |
| r_max | Poisson-sum truncation (100) — accuracy caveat §7.7 | GP |
| f_max | firing-rate guard cap (100) | guidance |
| ℳ | natural-image manifold | both |
| C, C_smooth | kernel-gradient matrix / its Gaussian spatial smoother | GP |
| ρ, l | pixel-smoother rate / kernel lengthscale | GP |
| d | image dimensionality (64×64 = 4096) | both |
| S, K | # DDIM steps (50) / # seeds for best-of-K | diffusion |

**Highest-risk clashes to keep straight:**
1. **β** — diffusion noise variance β_t *vs* GP kernel RF-locality β. (Flagged in
   the brief; genuine and pervasive.)
2. **α** — diffusion α_t/ᾱ_t (signal retention) *vs* kernel per-pixel envelope α_j.
3. **s** — score s_θ *vs* scale_factor s=2.478 *vs* cosine offset s=0.008.
4. **λ** — GP latent log-firing λ(x) *vs* Approach-A regularizer weight λ_diff.
5. **σ** — reverse-step std σ_t / DDIM σ *vs* GP log-firing variance σ²_g.
6. **t** — always diffusion time here; there is **no** temporal/RF time (spatial
   GP only), so t is unambiguous *within* this cluster but must not be conflated
   with any temporal notion from other clusters.

---

## 10. Cross-doc conflicts & scope exclusions (quick index)

**Doc-vs-doc / doc-vs-code (detail in §6.1, §8):**
- Model size/schedule: `guided_reverse_diffusion.tex` says 2.16M/cosine for
  Approach D; reality (code + `guided_diffusion_summary.tex` +
  `GUIDED_REVERSE_REFERENCE.md`) = 99.5M/linear. **Trust code + summary.**
- Approach-A optimizer: `REFERENCE.md` says SGD; code + `SESSION_2026-03-31.md`
  say LBFGS (proximal/MM). **Trust code + SESSION.** `REFERENCE.md` constants also
  stale (300/300/0/1.0 vs 50/50/5/5.0).
- `motivation_*.md`: "30×30, no implementation yet" — stale; real = 64×64, both
  approaches implemented. Reasoning current, status historic.
- Runtime breakage: both scripts KeyError on the removed `stop_window/stop_thresh`;
  `guided_reverse.py` model path broken (`HANDOFF` C1–C2). Fixes are trivial but
  **the scripts do not run out-of-the-box on the current engine.**
- `GUIDED_REVERSE_REFERENCE.md` cites the algorithm spec as
  `approach_d_guided_reverse_diffusion.tex`; the file in-tree is
  `guided_reverse_diffusion.tex` (renamed) — same content.

**Scope exclusions (per cluster constraints):**
- **Spatial GP only** — no spatiotemporal machinery. The diffusion `t` is
  *diffusion time*, NOT a temporal receptive-field axis; there is no temporal
  stimulus dimension anywhere in this cluster. Nothing spatiotemporal was present
  to skip.
- **Terminology preserved** — natural-image stimuli, receptive field (RF), firing
  rate, retinal ganglion cell kept verbatim; "active learning" used throughout
  (never "experimental design").
- **Abandoned/superseded, retained honestly:** the small 2.16M cosine model
  (superseded by 99.5M for generation, still used by Approach A); Approach A as a
  guidance method (superseded by Approach D but kept as the L2 baseline);
  Approaches B & C (analyzed, rejected, never adopted); SDEdit (proposed, not
  implemented). All summarized as "tried vs adopted" per the brief.
- **Utility internals** (Laplace/entropy machinery, LUT quadrature, arc-cosine
  kernel algebra) belong to the utility/kernel clusters; here only the *interface*
  (U differentiable w.r.t. image; smooth kernel gradient; RF mask) is carried, to
  keep the GP→synthesis bridge explicit without duplicating that math.

---

*Primary sources distilled:* `diffusion_models_introduction.tex`,
`unet_and_ddpm_introduction.tex`, `GUIDANCE_POSSIBILITIES.tex`,
`guided_reverse_diffusion.tex`, `guided_diffusion_summary.tex`,
`motivation_diffusion_guided_optimization.md`.
*Secondary reconciled:* `README.md`, `REFERENCE.md`, `GUIDED_REVERSE_REFERENCE.md`,
`SESSION_2026-03-31.md`, `PLAN_guided_reverse.md`, `investigation_log.md`,
`HANDOFF_diffusion_package.md` (runtime-verified status).
*Code skimmed for ground truth:* `guided_reverse.py`, `guided_optimization.py`,
`diffusion_model.py`.
