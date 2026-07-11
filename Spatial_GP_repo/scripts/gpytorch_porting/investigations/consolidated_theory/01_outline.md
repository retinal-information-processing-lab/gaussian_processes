# Phase-2 Outline + Unified Notation — Unified Theory Consolidation

**Deliverable.** The structure of the single wide LaTeX document
`unified_theory.tex`: (1) a detailed table of contents with the source docs
feeding each section; (2) a unified notation table that resolves every symbol
clash from `00_understanding_memo.md` §3 into one symbol per concept, with the
LaTeX macro each section must use; (3) the composition plan (drafting units,
reviewer, assembly) for Phase 3.

**Document title.** *A Unified Theory of Closed-Loop Active Learning for Retinal
Ganglion Cells: Gaussian-Process Inference, Information Utility, and Guided
Diffusion.*

**Design goals.** Self-contained (a reader needs nothing but this document);
faithful (every in-scope result, derivation, proof, and algorithm detail is
preserved); non-redundant (each derivation appears once, cross-referenced
elsewhere); wide (full depth, not summaries); one consistent notation throughout,
enforced by shared LaTeX macros.

---

## 1. Table of contents (parts → sections → subsections) with source mapping

Legend for sources: `Nxx` = `notes/xx_*.md`; original docs named where useful;
"code" = the current ground-truth implementation.

### Front matter
- **Abstract / overview** — the closed-loop active-learning story end to end.
  *(synthesis of all)*
- **How to read this document; scope statement** — spatial GP only; current
  implementations (`vargp_direct`, `default_gpy`); "active learning". *(memo)*
- **Notation reference** — the unified table (§2 below), placed early as a
  reference. *(this outline)*

### Part I — The model and its prior
1. **Introduction: the closed-loop active-learning problem**
   - 1.1 The experiment: RGCs, the MEA, natural images, the closed loop. *(N01, N06 framing)*
   - 1.2 The active-learning goal: learn each neuron's RF/tuning; choose maximally
     informative images. *(N06, N08 motivation)*
   - 1.3 Roadmap of the document. *(this outline)*
2. **The generative model**
   - 2.1 Latent Gaussian process over image space, `λ ~ GP(0, K)`. *(N01)*
   - 2.2 The exponential link and Poisson spike counts, `f = exp(Aλ+λ₀)`,
     `y ~ Poisson(f)`. *(N01, likelihoods.py)*
   - 2.3 The GP models the latent `λ`, not the rate — why. *(N01)*
3. **The arc-cosine kernel and the structured spatial prior**
   - 3.1 The order-1 arc-cosine kernel: definition, magnitude/angle form, the
     ReLU-network (Cho–Saul) interpretation. *(N05)*
   - 3.2 Non-stationarity and the self-kernel identity `K(x,x) = xᵀCx + σ₀²`. *(N05, N07)*
   - 3.3 The structured spatial prior `C(β, ρ, ε₀)`: local (β), smooth (ρ),
     RF-centered (ε₀); the final agreed form. *(N05)*
   - 3.4 Raw parameterization and constraints (`raw_m2log2beta`,
     `raw_mlog2rho2`, `σ₀=exp(raw)`, `Amp=softplus(raw)`, bounds). *(N05, code)*
   - 3.5 Kernel gradient w.r.t. inputs `∇_x K` (feeds the utility and diffusion
     guidance). *(N05)*
   - 3.6 Kernel gradients w.r.t. hyperparameters `∂C/∂{β,ρ,ε₀}` (feed the M-step). *(N05, N04 hookup)*
   - 3.7 Numerical stability: C symmetrization, forward clamps, Cholesky jitter. *(N05)*

### Part II — Inference
4. **Sparse variational Gaussian-process inference**
   - 4.1 Poisson breaks conjugacy → the variational approximation. *(N01)*
   - 4.2 Inducing points and the variational posterior `q(λ̃) = N(m, V)`. *(N01)*
   - 4.3 Projected posterior moments `μ(x)`, `σ²(x)`. *(N01)*
   - 4.4 The ELBO: expected log-likelihood − KL. *(N01)*
   - 4.5 Expected firing rate via the Gaussian moment-generating function. *(N01)*
   - 4.6 Prediction: expected spike count. *(N01)*
   - 4.7 The posterior cross-covariance pitfall and the fix (slice GPyTorch's
     joint covariance). *(N01, N07 predictive)*
   - 4.8 Whitening: purpose and the whitened↔unwhitened maps. *(N01)*
5. **The eigenspace representation and the two implementations**
   - 5.1 `vargp_direct`: eigendecomposition of `K̃`, the `DirectVariationalState`
     objects, kept eigenvalues, `n_b`. *(N02)*
   - 5.2 Posterior moments and KL in the eigenbasis. *(N02, N04)*
   - 5.3 The `recompute_eigenspace` reprojection seam. *(N02)*
   - 5.4 `default_gpy`: the standard GPyTorch variational path (Cholesky,
     whitening, ZeroMean). *(N02)*
   - 5.5 Why `vargp_direct` manages parameters directly: the L_K whitening
     mismatch. *(N02, DECISION_LOG)*
   - 5.6 Shared math vs divergent representation — the contrast table. *(N02)*
   - 5.7 Online growth: the rank-1 update and the checkpoint seam (with the
     basis-drift caveat). *(N02)*
6. **The training algorithm (EM-style coordinate ascent)**
   - 6.1 Overview: three coordinate blocks on one ELBO; the loop order
     (recompute → E → F → metrics/early-stop → M). *(N04)*
   - 6.2 The E-step: Newton update of `q`
     - 6.2.1 The variational objective; gradient and Hessian w.r.t. `m`. *(N03)*
     - 6.2.2 The Poisson expected log-likelihood and its derivatives. *(N03)*
     - 6.2.3 The helpers `g_E` (gradient) and `G_E` (Fisher curvature). *(N03)*
     - 6.2.4 The `V`-update (corrected symmetric) and the `m`-update (code form;
       the fixed-point equivalence to the corrected Newton step). *(N03)*
     - 6.2.5 Cheapness in eigenspace; damping and the divergence guard. *(N03)*
   - 6.3 The F-step: fitting the firing-rate parameters `A, λ₀`
     - 6.3.1 Closed-form `λ₀` given `A`. *(N04)*
     - 6.3.2 LBFGS on `A` with the analytic `dL/dA`. *(N04)*
     - 6.3.3 The interleaved damped Newton on `(A, λ₀)`. *(N04)*
   - 6.4 The M-step: updating kernel hyperparameters
     - 6.4.1 Objective and six-member parameter set `{σ₀, Amp, β, ρ, ε₀ₓ, ε₀ᵧ}`
       (Amp optimized by default; `fix_Amp` reproduces the paper). *(N04, code)*
     - 6.4.2 The analytical gradient chain `dC → dK/dK̃/dKvec → dλ_m/dλ_var →
       dloglik+dKL → dL`. *(N04)*
     - 6.4.3 Constraint-transform Jacobians. *(N04)*
     - 6.4.4 The two gradient systems; the VJP formulation and why it is cheaper. *(N04)*
     - 6.4.5 Numerical guardrails. *(N04)*

### Part III — Active learning: the information utility
7. **The acquisition objective**
   - 7.1 Utility as expected entropy reduction / mutual information. *(N06)*
   - 7.2 The marginal spike-count entropy `H_marg` (Poisson-lognormal; Laplace
     approximation; the Lambert-W mode). *(N06, N07)*
   - 7.3 The standard (production) utility `U_std = H_marg − E[H_noise]`
     - 7.3.1 The aleatoric noise entropy `E[H_noise]` (closed form). *(N06)*
   - 7.4 The distribution-aware (research) utility `U_DA = H_marg − E_{x∼p(x)}[H_cond]`
     - 7.4.1 Conditional GP moments via Gaussian conditioning. *(N06, N07)*
     - 7.4.2 The deterministic special case. *(N06, N07)*
   - 7.5 Why sample `λ` instead of using its mean: the nonlinearity bias. *(N06)*
   - 7.6 The entropy landscape over image space. *(N06)*
   - 7.7 Code realization and conventions: `standard_utility`,
     `distribution_aware_utility`, `compute_H`, `nd_utility_new`,
     `get_gp_marginal_moments`, `get_gp_conditional_moments`; the A/λ₀ convention
     split and the `r_max` off-by-one. *(N06, N07)*
8. **The deep layer: conditioning, subspace, divergence, numerics**
   - 8.1 GP moments and Gaussian conditioning (Proof III). *(N07)*
   - 8.2 The predictive distribution conditioned on a new observation
     - 8.2.1 The rank-1 posterior update. *(N07)*
     - 8.2.2 Sparse vs augmented predictive. *(N07)*
     - 8.2.3 Theorem: equivalence of predictive means. *(N07)*
   - 8.3 Subspace theory: optimizing the utility in a low-dimensional subspace
     - 8.3.1 The kernel touches `x` only through the C-eigenspace. *(N07)*
     - 8.3.2 Three bases (PCA / C-eigenspace / combined); the Szegő/Fourier
       equivalence. *(N07)*
     - 8.3.3 Subspace ≠ naturalness constraint; the c-eigenspace offset caveat. *(N07)*
   - 8.4 Divergence theorems (Proof I): norm scaling and the utility blow-up
     - 8.4.1 Cross-kernel proportionality; perfect posterior correlation. *(N07)*
     - 8.4.2 Moment scaling `μ ∝ κ`, `σ² ∝ κ²`; the divergence. *(N07)*
     - 8.4.3 The `σ₀² > 0` damping and the alignment peak at `κ = 1`. *(N07)*
   - 8.5 Kernel solutions (Proof II): the normalized arc-cosine and arc-sin
     saturation kernels as fixes. *(N07)*
   - 8.6 Numerical analysis of the utility: four failure modes, log-space
     Lambert-W, `r_max` truncation and the adaptive guard. *(N07, N06)*

### Part IV — From selection to synthesis
9. **Guided diffusion for stimulus synthesis**
   - 9.1 Motivation: synthesize high-utility natural images vs select from a
     fixed pool; the smooth-utility / high-frequency-prior division of labor. *(N08)*
   - 9.2 Diffusion fundamentals
     - 9.2.1 The forward noising process. *(N08)*
     - 9.2.2 Score/noise parameterization and the denoising-score-matching
       objective. *(N08)*
     - 9.2.3 The reverse process and the Tweedie estimate. *(N08)*
     - 9.2.4 The U-Net noise predictor. *(N08)*
   - 9.3 Guidance: steering the reverse process with the GP utility gradient
     - 9.3.1 Classifier-style guidance and the modified reverse update. *(N08)*
     - 9.3.2 The GP link: `∇_x U` through `∇_x K`; the RF mask. *(N08, N05)*
     - 9.3.3 Full vs approximate gradient (through the U-Net Jacobian). *(N08)*
     - 9.3.4 The four approaches (A–D) and the adopted method D. *(N08)*
   - 9.4 Implementation status and open issues (honest, exploratory). *(N08)*

### Back matter
- **Appendix A — Notation reference (full unified table).** *(this outline §2)*
- **Appendix B — Reconciled discrepancies and open caveats.** The doc-vs-code
  resolutions and the honestly-flagged open issues from memo §4. *(memo §4)*
- **Appendix C — Glossary of neuroscience terms** (RGC, RF, MEA, spike count,
  firing rate, tuning, natural image). *(synthesis)*

---

## 2. Unified notation table (one symbol per concept)

The **Macro** column is the LaTeX `\newcommand` every section must use, so the
notation is mechanically uniform. The preamble (written in Phase 3) defines all
macros. "⚠" marks a resolved clash a reader/composer could otherwise reintroduce.

### 2.1 Model, kernel, prior
| Concept | Symbol | Macro | Notes |
|---|---|---|---|
| Stimulus image (vectorized) | `x` | `\x` | one natural image; `x*` = candidate |
| Pixel coordinate of pixel i | `ξ_i` | `\pix_i` | on the image grid |
| RF center coordinate | `ξ₀ = (ε₀ₓ, ε₀ᵧ)` | `\rfc` | on [−1,1]² |
| Latent GP function | `λ(x)` | `\lat` | the tuning latent; ⚠ not the offset |
| Firing-rate offset | `λ₀` | `\latz` | Poisson log-baseline |
| Firing rate | `f(x) = e^{Aλ+λ₀}` | `\rate` | ⚠ f is the rate, never the latent |
| Log-firing rate | `g = Aλ + λ₀` | `\lgr` | so `f = e^g`; used in likelihood/utility |
| Gain | `A` | `\gain` | ⚠ distinct from projection `a` |
| Spike count | `y` (`= r`) | `\spk` | synonyms r, n, target |
| GP prior kernel (function) | `K(x,x')` | `\Kf` | ⚠ inducing gram is always `K̃` |
| Kernel magnitude | `𝓜 = √(v_x v_{x'})` | `\Kmag` | ⚠ NOT M (inducing count) |
| Kernel angle | `θ` | `\kang` | `cosθ = (xᵀCx'+σ₀²)/𝓜` |
| Angular term | `J(θ) = sinθ + (π−θ)cosθ` | `\Jang` | ⚠ J is never a Jacobian |
| Self-magnitude | `v_x = xᵀCx + σ₀²` | `\vx` | `K(x,x)=v_x` |
| Structured prior covariance | `C` | `\Cmat` | `C = Amp·α·C_smooth·αᵀ` (Amp = 6th hyperparameter) |
| Locality mask / envelope | `α_i` | `\loc_i` | ⚠ GP mask; NOT diffusion `α_t` |
| Smoothing factor | `C_smooth` | `\Csm` | RBF over pixel coords |
| RF half-width | `β` | `\bwid` | ⚠ GP prior; NOT diffusion `β_t` |
| Smoothness SD | `ρ` | `\smooth` | ⚠ GP prior; NOT correlation `ρ_λ` |
| Bias variance | `σ₀²` | `\sigz` | `σ₀ = exp(raw_sigma_0)` |
| Amplitude | `Amp` | `\Amp` | sixth kernel hyperparameter; softplus, bounds (0,1000], optimized by default |

### 2.2 Variational inference and eigenspace
| Concept | Symbol | Macro | Notes |
|---|---|---|---|
| Inducing points | `z̃` (matrix `Z̃`) | `\ipt`,`\Ipt` | ⚠ z̃, not the subspace coord |
| Inducing values | `λ̃` (`= u`) | `\latt` | `q(λ̃)=N(m,V)` |
| Number of inducing points | `M` | `\M` | ⚠ M, not magnitude |
| Number of training images | `N` | `\N` | active loop: `M = N` |
| Inducing gram | `K̃ = K(Z̃,Z̃)` | `\Kt` | code `K_tilde`, `K_uu` |
| Cross-covariance vector | `k(x) = K(Z̃,x)` | `\kvec` | |
| Projection vector | `u(x) = K̃⁻¹k(x)` | `\proj` | ⚠ lowercase; the SVGP projection |
| Variational mean / cov | `m`, `V` | `\vm`,`\vV` | inducing-space |
| Posterior mean / var | `μ(x)`, `σ²(x)` | `\pmean`,`\pvar` | of the latent λ |
| ELBO | `ℒ` | `\ELBO` | ⚠ script L; NOT the Cholesky factor |
| Cholesky factor of K̃ | `L_{K̃}` | `\Lchol` | ⚠ upright L |
| Eigenbasis of K̃ | `B` | `\eb` | columns = eigenvectors |
| Eigenvalues of K̃ | `Λ` (diag) | `\eval` | code `eigvals_b`; ⚠ capital, not λ |
| Kept eigen-dimension | `n_b` | `\nb` | dynamic, ≈10–11 |
| Eigenspace mean / cov | `m_b`, `V_b` | `\mb`,`\Vb` | `V_b` dense; `K̃_b = Λ` |
| Eigenspace projection | `a = K K̃⁻¹` | `\amat` | code `KKtilde_inv_b` ("Matthew's a") |
| Diagonal prior variance | `Kvec` | `\Kvec` | `Kvec_i = K(x_i,x_i)` |
| Expected firing rate | `f̄ = e^{Aμ+½A²σ²+λ₀}` | `\fbar` | code `f_mean` |

### 2.3 Training-step helpers
| Concept | Symbol | Macro | Notes |
|---|---|---|---|
| E-step gradient helper | `g_E = A Σ k_i(y_i − f̄_i)` | `\gE` | ⚠ subscript E; NOT log-rate g |
| E-step Fisher curvature | `G_E = A² Σ f̄_i k_i k_iᵀ` | `\GE` | PSD |
| Eigenbasis subscript | `(·)_b` | — | e.g. `g_{E,b} = K̃⁻¹ g_E` |
| Hyperparameter differential | `dC, dK, dK̃` | `\dd C` etc. | M-step gradient chain |
| ELBO gradient (M-step) | `dL/dθ` | `\dL` | θ ∈ {σ₀,Amp,β,ρ,ε₀} |

### 2.4 Utility and information
| Concept | Symbol | Macro | Notes |
|---|---|---|---|
| Utility (standard, production) | `U_std` | `\Ustd` | `= H_marg − E[H_noise]` |
| Utility (distribution-aware) | `U_DA` | `\Uda` | `= H_marg − E_{x∼p(x)}[H_cond]` |
| Marginal entropy | `H_marg` | `\Hmarg` | ⚠ always subscripted; never bare H |
| Aleatoric noise entropy | `H_noise` (`E[H_noise]`) | `\Hnoise` | Poisson noise, std utility |
| Conditional entropy (over p(x)) | `H_cond` | `\Hcond` | DA utility |
| Natural-image distribution | `p(x)` | `\px` | |
| Log-rate moments | `μ_g = Aμ+λ₀`, `σ²_g = A²σ²` | `\mug`,`\varg` | |
| Posterior latent correlation | `ρ_λ(x,x*)` | `\corr` | ⚠ NOT kernel smoothness ρ; `ρ_λ²` = fractional variance reduction |
| Prior conditional covariance | `c = k(x,x*) − uᵀK̃u*` | `\ccond` | the "correction" term |
| Input scaling factor | `κ` (`x* = κ x_t`) | `\scal` | ⚠ divergence proofs; NOT `c` |
| Residual / aleatoric variance | `σ²_r` | `\varr` | ⚠ NOT the scale constant `s` |
| Subspace coordinate | `ζ` | `\subc` | ⚠ NOT an inducing point |
| Lambert-W (principal) | `W₀` | `\LW` | |
| Laplace mode | `ḡ_r` | `\gmode` | `= rσ²+μ − W₀(σ² e^{rσ²+μ})` |
| Count-truncation bound | `r_max` | `\rmax` | default 100; adaptive floor 200 |
| Saturation-kernel amplitude | `σ_f²` | `\satf` | Proof II |

### 2.5 Diffusion (introduced only in Part IV)
| Concept | Symbol | Macro | Notes |
|---|---|---|---|
| Diffusion timestep | `t` | `\dt` | ⚠ diffusion time; there is NO temporal RF |
| Clean / noisy image | `x₀ / x_t` | `\xz`,`\xt` | |
| Tweedie estimate | `x̂₀` | `\xhat` | `= (x_t − √(1−ᾱ_t)ε_θ)/√ᾱ_t` |
| Noise-schedule variance | `β_t` | `\bsched` | ⚠ subscript t; NOT GP β |
| Signal retention | `α_t`, `ᾱ_t` | `\aret`,`\abar` | ⚠ subscript t; NOT GP mask α_i |
| Noise predictor (U-Net) | `ε_θ(x_t,t)` | `\eps` | |
| Score function | `s_θ = ∇ log p_t` | `\score` | `≈ −ε_θ/√(1−ᾱ_t)` |
| Guidance scale | `w` | `\gw` | |
| Guidance gradient | `g_t = ∇_{x_t} U(x̂₀)` | `\ggrad` | the GP utility gradient |
| A-approach reg. weight | `γ_reg` | `\greg` | ⚠ NOT the GP latent λ |
| DDIM std / scale constant | written inline | — | avoid a bare `s` / `σ` |

### 2.6 Synonyms collapsed (write the left form only)
`y` ⟵ r, n, target · `f̄` ⟵ ⟨f⟩, f_mean, f_pred · `μ(x),σ²(x)` ⟵ lambda_m,
lambda_var, posterior.mean/.variance · `K̃` ⟵ K_uu, K_tilde · `Z̃` ⟵ z, X_tilde ·
`a` ⟵ KKtilde_inv_b · `λ̃` ⟵ u (inducing values) · `μ_g,σ²_g` ⟵ g-moments.

---

## 3. Composition plan (Phase 3)

**Shared preamble (I write first, before dispatch).** A `preamble.tex` defining:
`amsmath, amssymb, amsthm, mathtools, geometry (wide margins), booktabs, hyperref,
xcolor, listings (for code snippets), a theorem environment set (Theorem/Lemma/
Corollary/Proposition/Proof), and every `\newcommand` from §2`. All composers
receive the macro list and must use it — this is the primary coherence device.

**Drafting units (subagents, parallel).** Each writes a self-contained LaTeX
section fragment (no preamble; `\input`-ready) to `sections/`:
- **U1 — Part I** (§1–3: intro, generative model, kernel + prior) ← N01(partial), N05, memo.
- **U2 — Part II inference** (§4–5: variational GP, eigenspace, two impls) ← N01, N02.
- **U3 — Part II training** (§6: E/F/M steps, gradients, VJP, loop) ← N03, N04.
- **U4 — Part III utility core** (§7) ← N06.
- **U5 — Part III deep layer** (§8: proofs, subspace, divergence, numerics) ← N07.
- **U6 — Part IV diffusion** (§9) ← N08.
- Front matter + appendices A/B/C: I write directly (they are synthesis/reference).

**Reviewer (persistent agent).** One reviewer agent kept alive across the whole
compose phase (continued via SendMessage). For each drafted fragment it enforces:
(1) unified notation / macro usage; (2) no contradiction with other sections or
the memo's resolutions; (3) no duplicated derivation (cross-reference instead);
(4) correct `\ref`/`\label` cross-references; (5) smooth logical flow into the
neighbours. A fragment is "done" only after the reviewer signs off. A final
whole-document coherence pass runs after assembly.

**Assembly + compile.** I write `unified_theory.tex` = preamble + title +
`\tableofcontents` + `\input{sections/*}` in order + appendices, then compile with
`/home/idv-eqs8-pza/anaconda3/envs/tex/bin/tectonic unified_theory.tex`, fix any
LaTeX errors, and iterate until `unified_theory.pdf` builds clean.

**Cross-cutting rules for every composer:**
- Ground truth = current code; where a doc disagreed, the code wins (memo §4).
- Keep neuroscience framing; "active learning" only.
- Spatial GP only; no temporal/Kronecker/warping.
- Present `Amp` as the sixth optimized kernel hyperparameter
  (`C = Amp·α·C_smooth·αᵀ`); note `fix_Amp=True` reproduces the paper's Amp-free
  model.
- Standard utility = production; DA utility = clearly-labeled research variant.
- Use `get_gp_marginal_moments` / `get_gp_conditional_moments` (never the
  nonexistent `lambda_moments`).
- Flag the honest open caveats (E-step m-form transient; checkpoint basis drift;
  utility `r_max` ceiling; c-eigenspace offset) in-place and collect them in
  Appendix B.

**Status:** Phase 2 complete. Proceeding to Phase 3 (preamble → parallel section
drafts under a persistent reviewer → assemble → compile).
