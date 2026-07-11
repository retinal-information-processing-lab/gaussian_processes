# Transform brief — biology-free, math/ML-only rendering

**Mission.** Produce a domain-neutral (pure mathematics / machine-learning) version
of the consolidated theory document, for a math-and-code LLM assistant that will
analyze and try to solve the **diverging-utility problem** (below). The assistant
degrades on domain-specific terms, so the output must contain **zero** application-
domain (neuroscience/biology/vision-application) vocabulary. The original document
is untouched; every transformed file goes under `math_version/`.

**Golden rule.** Preserve ALL mathematics exactly — every equation, definition,
theorem, lemma, proof, corollary, remark, `\label`, and cross-reference to KEPT
content stays byte-identical in its math. You change (a) the surrounding **prose /
interpretation** to domain-neutral language, and (b) you DROP the specified
out-of-scope material. Do not re-derive, re-number, or "improve" any math.

---

## 1. Term mapping (apply everywhere; symbols are UNCHANGED)

Keep every symbol (`λ, K, C, K̃, μ, σ², β, ρ, ξ₀, ε₀ₓ, ε₀ᵧ, A, λ₀, Amp, U, H, m,
V, mᵦ, Vᵦ, B, Λ, nᵦ, W₀, r_max, …`) and every math/ML-standard word (kernel, prior,
posterior, Gaussian process / GP, variational, ELBO, inducing points, utility,
entropy, mutual information, **active learning**, Poisson, count, rate, intensity,
diffusion, score, gradient, eigenspace, Cholesky, Lambert-W). Replace ONLY the
domain vocabulary:

| Source (domain) term | Neutral replacement |
|---|---|
| neuron, cell, unit, RGC, "each cell/neuron" | **the latent function** / **the target** / **the model** (usually just refer to `λ` / the model) |
| receptive field (RF) | **the localized (input) region** / **the localized support of the prior** |
| RF center | center `ξ₀` of the localized region |
| RF width / size | width (`β`) of the localized region |
| spike, spike count | **count** / **count observation** `y` |
| firing rate, firing, fire | **rate** / **intensity** `f` |
| response(s) (as the modeled output) | **output** / **count** |
| tuning, tuning curve | **the latent function** / **the input–output map** |
| stimulus, stimuli | **input** |
| image, natural image, image stimulus | **input** (an input is a vector on a 2-D grid) |
| pixel | **grid point** / **input coordinate** |
| "natural images" as a distribution | **the input (data) distribution** `p(x)` |
| saturated image | **saturated input** / input driven to the boundary of the box |
| closed-loop experiment, the experiment | **the (sequential) active-learning setting** / **the active-learning loop** |
| MEA, multi-electrode array, electrode, DMD, micromirror, display, recording | **DROP** (data-acquisition hardware; irrelevant) |
| retina(l), ganglion, visual, brain, cortex, photoreceptor, biological, neural, neuro- | **DROP** |

**Input model to state once (in the intro/setup):** an input `x ∈ ℝ^d` is a
real-valued signal on a 2-D grid of `d` grid points (e.g. a 108×108 grid, `d=11664`);
grid coordinates are `ξ_i`. The input domain is a bounded box (e.g. `[0,1]^d`). This
2-D-grid structure is load-bearing (the structured prior `C` uses grid distances
`‖ξ_i − ξ_j‖`) — keep it, just call it a grid, never an image.

### Dual-use HAZARDS (do not over-strip)
- **KEEP** "**natural parameters**" (the `β,ρ` natural parameterization vs. raw) —
  math term, unrelated to "natural image".
- **KEEP** "**Gaussian field / random field**" (= a GP) — math.
- **KEEP** "spatial" (as in spatial/grid structure of `C`) — geometric, fine.
- "**response**": strip only the modeled-output meaning; reword incidental "in
  response to …".
- "**field**": strip in "receptive field"; keep in "random/Gaussian field".
- "**image**": replace the input meaning with "input"; the math sense "image of a
  map / preimage" (unlikely here) would stay — but there is essentially none.

---

## 2. Keep / drop policy (chosen: STRIP BIOLOGY + DROP CLUTTER ONLY)

Keep the full mathematical content biology-free. DROP only clearly out-of-scope
clutter. Err toward keeping if unsure.

**DROP (out of scope for the math assistant):**
- The VJP / analytical-gradient *derivations* and the "two gradient systems"
  discussion (the M-step chain machinery).
- Code / function / file references and any "code realization / conventions"
  passages (function names, `file.py:line`, code-var names, code caveats such as
  "the docstring says …", "kernels.py:180 comment is wrong", "the handoff claim …").
- Implementation status (broken scripts, model-size specs, run status).
- The **two-implementations** contrast, the checkpoint/reload seam, the rank-1
  online-update plumbing, the `recompute_eigenspace` seam, and the "why manage
  parameters directly / L_K whitening mismatch" implementation rationale.
- Caveats that are code-vs-doc reconciliations. **KEEP** caveats that are genuine
  *mathematical* open questions (e.g. the subspace offset not being mathematically
  motivated; the utility `r_max` accuracy ceiling; the Poisson entropy bound being
  plausible-but-unproved; divergence under free gradient ascent).

**KEEP (biology-free, math intact):** the generative model; the arc-cosine kernel
and structured prior `C` in full (definition, `∇_x K`, `∂C/∂θ`, self-kernel
identity, non-stationarity, bounds, the amplitude `Amp`); the variational posterior
moments, ELBO, expected rate, prediction, the cross-covariance pitfall, whitening;
the **eigenspace representation math** (eigendecomposition of `K̃`, posterior
moments and KL in the eigenbasis) — but NOT the impl plumbing above; the E-step
(Newton update) and F-step in full, and the M-step **objective** (what is
optimized: `{σ₀, Amp, β, ρ, ε₀ₓ, ε₀ᵧ}`) — but drop its analytical-gradient
derivation/VJP; the utility (standard + distribution-aware), entropies, why-sample-
`λ`, the entropy landscape; the entire deep layer (Gaussian conditioning; predictive
distribution; subspace theory; **divergence theorems**; **kernel solutions —
normalized + saturation**; numerical analysis, keeping the math: Lambert-W, `r_max`
truncation, the failure modes); and the diffusion/guidance **math** (forward/reverse
process, score/noise, DSM loss, Tweedie, U-Net parameterization, guidance via the
utility gradient, the four approaches, the link `∇_x U` through `∇_x K`, and the
saturation phenomenon) — dropping only its implementation-status subsection.

**Cross-references:** keep `\ref`/`\eqref` to KEPT content. If you drop a labeled
block, delete or reroute any reference to it. If you reference a label that lives
in another file (kept), keep it; the orchestrator resolves cross-file refs at
assembly.

---

## 3. The problem the assistant will attack (state neutrally where relevant)

The arc-cosine kernel is positively homogeneous in the input: `K(x,x)=xᵀCx+σ₀²`
grows with `‖x‖`. Under active-learning utility maximization this makes the
posterior variance grow like the square of the input scale, so the utility diverges
and the optimizer drives inputs to the boundary of the bounded box (saturation).
The document's proposed fixes (a normalized arc-cosine kernel; a saturating arc-sin
kernel) are presented, and it is an **open problem** to find a kernel that both (i)
keeps the utility bounded on the bounded input domain and (ii) fits well. Keep this
framing crisp and domain-neutral; the intro carries a short problem statement.

---

## 4. Self-check before returning (every transformer)

Grep your output for a domain-term blacklist and confirm **zero** hits (excluding
the allowed "natural parameters", "Gaussian/random field", "spatial"):
`neuron, neural, neuro, cell, spike, spik, firing, fire, receptive, RF, retina,
ganglion, RGC, MEA, electrode, DMD, micromirror, stimulus, stimuli, tuning,
response, biolog, brain, cortex, photoreceptor, image, images, pixel, natural image,
visual`.
Also confirm: every equation/label/proof preserved; only prose changed + specified
drops; no dangling `\ref` to a block you removed.
