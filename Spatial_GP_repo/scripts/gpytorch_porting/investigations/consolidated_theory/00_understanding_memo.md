# Phase-1 Understanding Memo — Unified Theory Consolidation

**Purpose.** This memo is the Phase-1 deliverable: it maps the whole in-scope
corpus, lists the key mathematical objects, records the notation each source
uses (with the symbol clashes that must be resolved), and catalogues every
inconsistency / redundancy / conflict found across the docs and against the
current code. Resolutions are previewed here and finalized in `01_outline.md`
(the unified notation table) and enforced during composition.

**Method.** Eight reader passes each distilled one cluster of the corpus into a
persistent notes file under `notes/` (equations preserved, ground truth = the
CURRENT code). This memo synthesizes those eight returns, including the
**cross-cluster** clashes no single reader could see. Plain-text unicode math is
used throughout (this is a `.md`, not compiled LaTeX).

**Scope (applied everywhere).** Spatial GP only (input = one image; latent = a
scalar function over one image); current implementations only — the eigenspace
variational GP (`vargp_direct`: `eigenspace_*.py`) and the standard GPyTorch
variational path (`default_gpy`: `gpy_*.py`); neuroscience framing kept verbatim;
the paradigm is "active learning" (never "experiment(al) design"). Excluded:
spatiotemporal machinery (temporal covariance, Kronecker C = C_spatial ⊗
C_temporal, temporal warping) and the deprecated `vargp_old` code path (repo-root
`utils.py:varGP()/Estep()`, the old torch-LBFGS setup, the "our approximation of
the paper" / paper-vs-vargp_old framing). The underlying variational-GP math is
kept — it is what the current code runs.

---

## 1. The end-to-end picture (how the corpus connects)

A closed-loop active-learning experiment for retinal ganglion cells (RGCs). A
multi-electrode array records spikes while natural grayscale images are shown; a
Gaussian-process model learns each neuron's receptive field (RF) / tuning from
spike counts, and images are chosen by active learning to maximize information
gain about that tuning.

The pipeline, in the order the final document will present it:

1. **Generative model.** Latent `λ ~ GP(0, K)` over image space; firing rate
   `f(x) = exp(A·λ(x) + λ₀)`; spike counts `y ~ Poisson(f(x))`. The GP learns the
   *latent* `λ`, not the rate directly.
2. **Kernel & structured prior.** `K` is the order-1 arc-cosine (ReLU) kernel
   built over a structured spatial prior covariance `C(β, ρ, ε₀)` that encodes
   local, smooth, RF-centered weightings. Two gradients matter downstream: ∇_x K
   (used by the utility and by diffusion guidance) and ∂K/∂hyperparameters (used
   by the M-step).
3. **Variational inference.** The Poisson likelihood breaks conjugacy → sparse
   variational GP with `M` inducing points, Gaussian variational posterior
   `q(λ̃) = N(m, V)`, objective `ELBO = Σᵢ E_q[log Poisson(yᵢ | λᵢ)] − KL(q‖prior)`.
   The expected log-likelihood and expected firing rate are closed-form via the
   Gaussian moment-generating function; whitening improves conditioning.
4. **Eigenspace representation + two implementations.** `vargp_direct` projects
   the posterior into the eigenbasis `B` of the inducing kernel `K̃` (diagonal
   `K̃_b`, custom EM loop); `default_gpy` realizes the *same* model via standard
   GPyTorch variational inference with Cholesky whitening. A reload/checkpoint
   seam and a rank-1 update grow the model one image at a time for the online
   loop.
5. **Training = EM-style coordinate ascent on one ELBO.** E-step (Newton update
   of `m, V`), F-step (fit firing-rate params `A, λ₀`), M-step (update kernel
   hyperparameters via analytical ELBO gradients + a VJP formulation).
6. **Active learning: the utility.** Pick the image maximizing expected entropy
   reduction (mutual information) about the neuron's tuning. Two utilities:
   **standard** (production) `= H_marg − E[H_noise]` and **distribution-aware**
   (research) `= H_marg − E_{x∼p(x)}[H_cond]`. Entropies are Poisson-lognormal,
   computed by Laplace + Lambert-W.
7. **The deep layer.** Proofs (Gaussian conditioning; norm-scaling divergence of
   the utility from arc-cosine homogeneity; kernel fixes; predictive-conditioning
   mean-equivalence), subspace theory (optimize the utility in a low-dim
   subspace), and the numerical analysis of the utility.
8. **From selection to synthesis: guided diffusion.** Instead of only selecting
   from a fixed pool, synthesize a natural-image-like stimulus that maximizes the
   GP utility, using a DDPM prior guided by the utility gradient ∇_x U (which
   flows through ∇_x K).

The single load-bearing thread from §2 to §8: the arc-cosine kernel is positively
homogeneous (`K(x,x) = ‖x‖²_C + σ₀²`), which makes the utility's variance grow
∝ c² under unconstrained input scaling — the object of the divergence proofs, the
subspace constraint, the kernel fixes, and the reason diffusion guidance needs a
naturalness prior.

---

## 2. Key objects (by cluster / notes file)

### 2.1 Generative model & variational framework — `notes/01`
- Generative model `λ ~ GP(0,K)`, `f = exp(A·λ + λ₀)`, `y ~ Poisson(f)`.
- Variational posterior `q(λ̃) = N(m, V)` on `M` inducing points; projected
  moments `μ(x) = k(x)ᵀK̃⁻¹m`, `σ²(x) = k(x,x) + k(x)ᵀK̃⁻¹(V − K̃)K̃⁻¹k(x)`.
- ELBO `= Σᵢ E_q[log p(yᵢ|λᵢ)] − KL(q‖p)`; expected log-lik (closed form)
  `= yᵢ(Aμᵢ + λ₀) − exp(Aμᵢ + ½A²σᵢ² + λ₀)`; KL `= ½[log(|K̃|/|V|) + tr(K̃⁻¹V) +
  mᵀK̃⁻¹m − M]`.
- Expected firing rate `f̄ᵢ = exp(Aμᵢ + ½A²σᵢ² + λ₀)` (Gaussian MGF; uncertainty
  inflates the rate — Jensen).
- Two reparameterizations: eigenspace projection (`vargp_direct`) and Cholesky
  whitening (`default_gpy`).
- Posterior cross-covariance: the manual formula `Σ_{x,x*} = k(x,x*) + u(x)ᵀ(V −
  K̃)u(x*)` is numerically unreliable outside the inducing hull (violates
  Cauchy–Schwarz, ~ −4 vs GPyTorch's ~0) → spurious utility peaks; fix = slice
  GPyTorch's joint covariance.

### 2.2 Eigenspace representation & the two implementations — `notes/02`
- Both paths realize the SAME sparse variational GP; differ only in how `q` is
  represented and optimized.
- `vargp_direct` (`DirectVGPModel`, plain class): eigendecompose `K̃ = BΛBᵀ`
  (`torch.linalg.eigh`, keep eigenvalues above `max(λ_max·tol, tol)`, `tol=1e-4`,
  dynamic kept count `n_b ≈ 10–11`); store eigenspace mean `m_b` and DENSE cov
  `V_b`, with `K̃_b = diag(eigvals)`; init at prior `m_b = 0, V_b = K̃_b`. GPyTorch
  is used only as a kernel calculator; a custom EM loop runs. `recompute_eigenspace()`
  reprojects `m_b, V_b` onto the new basis after every kernel change.
- `default_gpy` (`VariationalGPModel(ApproximateGP)`): `CholeskyVariationalDistribution`,
  whitened `VariationalStrategy`, `ZeroMean`, one joint ELBO (LBFGS/Adam).
- Reason `vargp_direct` manages parameters directly: the L_K whitening mismatch
  (closed-form E-step + kernel-changing M-step makes GPyTorch's whitened params
  stale) — DECISION_LOG Q26/Q27.
- Rank-1 online update (`extend_model_with_new_point`): O(M) column-append to the
  stored `K̃`, re-eigendecompose, warm-start; active-loop invariant `X_train ==
  X_tilde` ⟹ `K = K̃`.
- Checkpoint (`eigenspace_checkpoint.py`) saves `m_b, V_b` + kernel/likelihood
  state but NOT the basis `B` (recomputed on load) — a documented basis-drift
  fragility across torch versions.

### 2.3 Kernel & structured prior — `notes/05`
- Order-1 arc-cosine kernel `K(x,x') = (1/π)·M·J(θ)`, with magnitude `M =
  √(v_x v_{x'})`, `v_x = xᵀCx + σ₀²`, `cos θ = (xᵀCx' + σ₀²)/M`, angular term
  `J(θ) = sin θ + (π − θ)cos θ`. Interpretation: covariance of an infinite-width
  single-hidden-layer ReLU network with input weights `w ~ N(0, C)` and bias
  variance σ₀². Diagonal `K(x,x) = v_x` (non-stationary; grows with input norm).
- Structured spatial prior (final agreed form):
  `C_ij = exp(−‖ξᵢ−ξ₀‖²/(4β²)) · exp(−‖ξᵢ−ξⱼ‖²/(2ρ²)) · exp(−‖ξⱼ−ξ₀‖²/(4β²))`,
  then symmetrized. `β` = RF half-width; `ρ` = smoothness SD; `ξ₀ = (ε₀ₓ, ε₀ᵧ)` =
  RF center on [−1,1]². Stored raw: `raw_m2log2beta = −2log(2β)` (so
  `exp(raw) = 1/(4β²)`), `raw_mlog2rho2 = −log(2ρ²)` (so `exp(raw) = 1/(2ρ²)`).
- Input gradient (utility/guidance): `∇_x K = (1/π)[(π−θ)Cx' + sin θ ·
  √(v_{x'}/v_x)·Cx]`.
- Hyperparameter gradients (M-step): `∂C_ij/∂β`, `∂C_ij/∂ρ`, `∇_{ξ₀}C_ij` (closed
  forms in `notes/05`).
- Numerical stability: C symmetrization; forward clamps (eps=1e-7 on cos θ /
  sin θ); Cholesky jitter (default 1e-4, escalating retries) on gram matrices
  `K_uu`/`K_XX` — NOT on `C` itself.

### 2.4 Training algorithm — E-step `notes/03`, M/F-step `notes/04`
- **E-step** (`eigenspace_estep.py`): closed-form Newton update of `q(λ̃) =
  N(m, V)` at fixed hyperparameters. Helpers `g = A Σ kᵢ(yᵢ − f̄ᵢ)` (gradient) and
  `G = A² Σ f̄ᵢ kᵢkᵢᵀ` (PSD Fisher curvature). Updates: `V = K̃(K̃+G)⁻¹K̃`
  (corrected symmetric form — matches code), `m_new = V_new(G K̃⁻¹m + g)` (code
  uses the OLD form; see §4). Cheap in eigenspace (`K̃_b` diagonal, `n_b ≈ 10`).
- **F-step** ("F" = the firing-rate function `f`; `eigenspace_fstep.py`): fit the
  Poisson rate params `A, λ₀` holding `m, V` and the kernel fixed. Closed-form
  `λ₀ = log(Σ y) − log(Σ exp(Aλ_m + ½A²λ_var))`; LBFGS on `A` with analytic
  `dL/dA = Σ[yλ_m − (λ_m + Aλ_var)f̄]`; optional interleaved damped Newton on
  `(A, λ₀)` (α = 0.25).
- **M-step** (`eigenspace_mstep.py` + `eigenspace_gradients.py`): maximize the
  ELBO over the kernel hyperparameters holding `m, V` and `A, λ₀` and the basis
  `B` fixed. Analytical gradient chain (under `no_grad`): `dC → dK/dK̃/dKvec →
  dλ_m/dλ_var → dloglik + dKL → dL → param.grad`, with a constraint-transform
  Jacobian (`σ₀ = exp(raw)`, `Amp = softplus(raw)`; `ε₀`, the two `raw_*` set
  directly). Two gradient *systems* exist: kernel-level autograd.Function
  (`analytical_gradients.py` Jacobian / `analytical_gradients_vjp.py` VJP,
  selected by `gradient_mode`) and the M-step-level `eigenspace_gradients.py`
  (the one that actually differentiates the ELBO during training). VJP is ~10–15×
  cheaper (compute `dL/dC` once, then chain) and is used for the kernel-level
  path, NOT by the M-step.
- **Loop order** per iteration: `recompute_eigenspace` (sync `B`) → E-step →
  F-step → metrics + ELBO + early-stop (deliberately BEFORE the M-step) → M-step
  (defers the `B` update to the next iteration).

### 2.5 Utility / active learning (core) — `notes/06`
- Utility of a candidate image `x*` = expected reduction in entropy (information
  gain) about the neuron's tuning.
- **Standard utility (PRODUCTION):** `U_std(x*) = H_marg(x*) − E[H_noise(x*)] =
  I(f(x*); R | x*, D)` — information about the latent firing rate *at x* itself*.
  This is what `run_active_loop.py:241` calls (`standard_utility`).
- **Distribution-aware (DA) utility (RESEARCH / investigations only):**
  `U_DA(x*) = H_marg(x*) − E_{x∼p(x)}[H_cond(x*)] = I(f(X); R | X, x*, D)` —
  information about firing rates at natural images `x ∼ p(x)`. "Distribution-aware"
  = aware of the natural-image distribution `p(x)`. This is the object the `.tex`
  summaries derive.
- `H_marg` = marginal spike-count (Poisson-lognormal) entropy via a Laplace
  approximation with Lambert-W closed-form mode `ḡ_r = rσ² + μ − W₀(σ²·e^{rσ²+μ})`.
- `E[H_noise]` (aleatoric Poisson noise entropy, closed form) `=
  −exp(μ_g + ½σ²_g)(μ_g + σ²_g − 1) + Σ p_r·log r!`.
- Conditional λ-moments after a hypothetical observation via Gaussian
  conditioning; log-firing moments `μ_g = Aμ_λ + λ₀`, `σ²_g = A²σ²_λ`.
- Why sample λ (not use its mean): a nonlinearity bias `≈ (ρ²/2)σ²_λ(x*)·∂²H/∂μ²`
  that does not vanish with N and is largest exactly when the correlation ρ→1
  (most relevant); one λ-sample per image is nested-MC-optimal.
- Entropy landscape: H rises monotonically in both `μ_g` and `σ²_g` → the utility
  mixes predicted firing and epistemic uncertainty (not pure epistemic);
  `r_max = 100` truncation collapses H for `μ_g ≳ 4.6` (adaptive guard fixes it).

### 2.6 Utility deep layer — proofs / subspace / numerics — `notes/07`
- **Proof III (moments & conditioning):** self-kernel identity `K(x,x) = ‖x‖²_C`;
  conditioning result `σ²_cond(x*) = σ²(x*)(1 − ρ²)`; deterministic DA utility
  `U_DA = H(μ_g, σ²_g) − H(μ_g, σ²_g(1 − ρ²))`.
- **Predictive distribution conditioned on a new observation:** rank-1 posterior
  update `m′ = m + (Vu/(s + uᵀVu))(λ(x) − uᵀm)`, `V′ = V − VuuᵀV/(s + uᵀVu)`;
  sparse vs augmented predictive; **Theorem (equivalence of predictive means):**
  augmented-integration mean ≡ joint-conditioning mean (full block-inversion
  proof).
- **Proof I (divergence theorems):** with σ₀² = 0 and `x* = c·x_t`, cross-kernel
  proportionality `K(x*, z) = c·K(x_t, z)` → perfect posterior correlation ρ² = 1,
  moment scaling `μ ∝ c`, `σ² ∝ c²` → the utility diverges (∝ c² Poisson-heuristic,
  numerically ~ c^1.9; ~log c in the Gaussian-obs case). With σ₀² > 0 the effect
  is slightly damped (ρ² → ~0.987, alignment peak strictly at c = 1).
- **Proof II (kernel solutions):** two fixes — the normalized arc-cosine kernel
  `K̄ = J(θ)/π` (invariant `K̄(cx, cx) = 1`) and the arc-sin saturation kernel
  (bounded `lim K_sat(x,x) = σ_f²`).
- **Subspace theory:** the kernel touches `x` only via `xᵀCx = Σ γᵢ(uᵢᵀx)²`, and
  ∇K carries a factor `C`, so gradient ascent already concentrates on the top
  C-eigenvectors; truncating to that subspace is an exact reparameterization plus
  a convergence aid (three bases: PCA / C-eigenspace / combined; Szegő/Fourier
  low-frequency equivalence under stationarity). A subspace constraint is NOT a
  distributional (naturalness) constraint.
- **Numerical analysis:** four failure modes of the standard utility; log-space
  Lambert-W; the `r_max` truncation and the adaptive-`r_max` guard.

### 2.7 Guided diffusion (selection → synthesis) — `notes/08`
- Motivation: synthesize a natural-image-like stimulus maximizing the GP utility,
  instead of only selecting from a fixed pool. The utility gradient (through the
  arc-cosine kernel) is smooth / low-pass; the diffusion score supplies the
  high-frequency naturalness the utility gradient cannot.
- Forward noising `q(x_t | x_0) = N(√ᾱ_t·x_0, (1 − ᾱ_t)I)`; score↔noise
  `∇ log p_t ≈ −ε_θ/√(1 − ᾱ_t)`; simplified DSM loss `L = E‖ε_θ(x_t, t) − ε‖²`;
  reverse mean + Tweedie estimate `x̂_0 = (x_t − √(1 − ᾱ_t)ε_θ)/√ᾱ_t`; U-Net noise
  predictor with sinusoidal time embedding.
- Guidance (classifier-style): conditional score = unconditional score +
  `w·∇_{x_t} U(x̂_0)`; the guidance signal is the GP utility gradient, so it flows
  through ∇_x K; the RF mask means guidance only touches the ~21% of pixels in the
  RF, and the diffusion prior generates the rest.
- Four approaches catalogued (A = L2-Tweedie, B = direct score, C =
  score-no-noise, D = full guided reverse); **adopted = D (guided reverse
  diffusion).** Honest status: exploratory, mid-flight; the two scripts are stale
  against the current engine and generation is fragile (some seeds collapse).

---

## 3. Notation: per-source usage and the master clash table

Each cluster's own notation table is in its notes file. The critical work for the
unified document is resolving the symbols that **collide across clusters**. Below
is the master clash table; the "Resolution (proposed)" column is finalized in
`01_outline.md`.

| Symbol | Meaning A (where) | Meaning B (where) | Other meanings | Resolution (proposed) |
|---|---|---|---|---|
| **β** | GP-prior RF half-width, in `C` (05, 04) | diffusion noise-schedule variance `β_t` (08) | — | GP-prior stays **β**; diffusion schedule always subscripted **β_t** and introduced only in the diffusion part |
| **ρ** | kernel smoothness SD, in `C` (05, 04) | posterior correlation `ρ² = corr(λ(x), λ(x*))` (06, 07) | — | kernel smoothness stays **ρ**; the utility correlation is renamed **ϱ** (or spelled "corr"), defined once in the utility part |
| **α** | locality mask / RF envelope `α_i = e^{−…}` (05, 07) | diffusion signal-retention `α_t / ᾱ_t` (08) | residual-corr coeff (07 predictive) | GP mask **α_i**; diffusion **α_t/ᾱ_t** (subscript t, diffusion part only); the predictive coeff renamed |
| **A** vs **a** | Poisson gain **A** (01, 03, 04, 06) | projection matrix **a = K K̃⁻¹** i.e. `KKtilde_inv_b` (02, 03, 04) | weight-cov `A` in an NN aside (01) | keep gain **A**; keep projection **a** (lowercase); drop the NN-aside `A` |
| **L** | ELBO (01, 03, 04) | Cholesky factor of `K̃` (01, 02) | — | ELBO = **ℒ**; Cholesky factor = **L** (or **L_K**) |
| **H** | marginal entropy **H_marg** | aleatoric noise entropy **E[H_noise]** (std utility) | conditional entropy **H_cond** over p(x) (DA utility) | always subscript: **H_marg / H_noise / H_cond**; never a bare `H`. THE most load-bearing clash (06, 07) |
| **M** | kernel magnitude `√(v_x v_{x'})` (05) | number of inducing points (01, 02) | — | inducing count = **M**; kernel magnitude renamed **𝓜** (or written `√(v_x v_{x'})`) |
| **λ** | GP latent `λ(x)` (all) | Poisson offset `λ₀` (all) | eigenvalues `Λ` (02, 03); diffusion reg-weight (08) | latent **λ(x)**, offset **λ₀**, eigenvalues capital **Λ**/`eigvals`; rename the diffusion reg-weight |
| **g** | log-firing rate `g = Aλ + λ₀` (06) | E-step gradient helper `g = AΣk(y−f̄)` (03, 04) | guidance gradient `g_t` (08) | log-firing → keep as **g** in the utility part (it IS `Aλ+λ₀`); E-step helper renamed to avoid the collide; guidance stays **g_t** |
| **σ / σ²** | posterior variance `σ²(x)` (01) | bias variance `σ₀²` (05) | log-firing var `σ²_g` (06); reverse-step std `σ_t` (08) | subscript everything: **σ²(x), σ₀², σ²_g, σ_t** — no bare σ |
| **s** | score function `s_θ` (08) | scale_factor 2.478 (08) | cosine-schedule offset (08); residual var `s` (07) | score = **s_θ**; the constant scale renamed; residual var renamed |
| **c** | prior conditional covariance `c = k(x,x*) − uᵀK̃u*` (06, 07) | input scaling factor `x* = c·x_t` (07) | — | conditional cov = **c**; scaling factor renamed **γ** |
| **z** | inducing point `z̃` (all) | subspace optimization coordinate `z` (07) | — | inducing point **z̃**; subspace coord renamed |
| **K** | kernel function / prior gram `K(X,X)` | inducing gram `K̃ = K(Z,Z)` (some docs write bare `K`) | — | kernel function `K(·,·)`; inducing gram always **K̃** |
| **t** | diffusion timestep (08) | — | (no temporal RF — spatial only) | **t** = diffusion time, unambiguous (introduced only in the diffusion part) |
| **J** | angular term `J(θ)` (05, 04) | (avoid using J for Jacobian) | — | keep **J(θ)**; never use J for a Jacobian |
| **b** | eigenbasis subscript (·)_b (02, 03, 04) | vector `b = K̃_b⁻¹ m_b` in dKL (04) | — | keep subscript **(·)_b**; rename the local dKL vector |

**Synonyms to collapse (same concept, many spellings):**
`y = r = n = target` (spike count); `f̄ = ⟨f⟩ = f_mean = f_pred` (expected firing
rate); `μ(x), σ²(x) = lambda_m, lambda_var = posterior.mean/.variance`;
`K̃ = K_uu = K_tilde`; `Z̃ = z = X̃ = X_tilde`; `a = KKtilde_inv_b` ("Matthew's
a"); `u = λ̃` (inducing values); `m_b, V_b` = eigenspace variational params.

---

## 4. Conflicts, redundancies, inconsistencies — the reconciliation worklist

Every item here is resolved (or explicitly flagged as an open issue) in the final
document. Grouped by type. **Ground-truth rule: where a doc disagrees with the
current code, the code wins and the discrepancy is flagged.**

### 4.1 Doc-vs-code discrepancies (code wins)
1. **E-step m-update (the one substantive math conflict).** The "corrected"
   Newton derivation gives `m ← m + K̃(K̃+G)⁻¹(g − m)`; the code
   (`eigenspace_estep.py`) uses the OLD form `m ← K̃(K̃+G)⁻¹(G K̃⁻¹m + g)`. They
   differ per-iteration when `K̃` and `G` don't commute, BUT both maps share the
   fixed point `m* = g`, so the **converged posterior is identical** — the
   discrepancy is transient only. Present the code form as THE algorithm; state
   the fixed-point equivalence; flag the transient difference. (03)
2. **σ₀ constraint transform.** `MSTEP_ANALYTICAL_HANDOFF.md` says "softplus for
   sigma_0 and Amp" — wrong for σ₀. Code: **σ₀ = exp(raw)**, **Amp =
   softplus(raw)**. (04, 05)
3. **`lambda_moments` does not exist** in the local `gpytorch_porting/utils.py`
   (it is a deprecated varGP routine). Current equivalents:
   **`get_gp_marginal_moments`** and **`get_gp_conditional_moments`**. The brief
   listed `lambda_moments` as a helper to map — it must be replaced by these.
   (06, 07)
4. **Production utility is `standard_utility`, not the DA utility.** The `.tex`
   summaries derive the distribution-aware utility as "the" acquisition, but
   `run_active_loop.py:241` calls `standard_utility`. Both are in scope (both use
   current-code math); the document must label standard = production, DA =
   research variant. (06)
5. **Cross-covariance manual formula is unreliable;** the code reads GPyTorch's
   full joint `covariance_matrix` (which already includes the conditional term
   `c`), so the "correction" caution applies to hand-assembled cross-covariances,
   not the shipped predict path. (01, 06, 07)
6. **Eigenspace reference staleness.** `EIGENSPACE_REFERENCE.md` §3 lists
   `train.py/estep.py/fstep.py/mstep.py` — these do not exist; real names are
   `eigenspace_training/estep/fstep/mstep.py`. Its §7.2
   `EigenspacePosterior.expected_firing_rate()` does not exist (the method is on
   `PoissonLikelihood`). Its §4 "autograd M-step BUGGY" label is stale (diagonal-KL
   bug fixed Jan 2025). (02)
7. **M-step "single dK contraction" vs code.** The handoff/`gradients.md` sketch a
   compute-`dK`-once, single `(dL/dK · dK).sum()` reduction; the shipped M-step
   recomputes `C/dC/K/dK` inside the LBFGS closure and runs the full
   moment→loglik/KL chain. The single-contraction form is the kernel-level
   autograd.Function's backward, not the M-step — the docs conflated the two
   gradient systems. (04)
8. **Kernel code-comment errors.** `kernels.py:180` comment `RAW_BETA_MIN ≈ 0.18`
   actually evaluates to ~1.02; `clamp_hyperparameters` docstring says β ∈
   [0.01, 1.0] but the enforced bound is **0.3** (tightened 2026-04-10 after a
   β-drift OOM). `BETA_RHO_MATH_VERIFICATION.md` predates the tightening. (05)
9. **Diffusion model specs.** `guided_reverse_diffusion.tex` §2.1 says Approach D
   uses a 2.16M / cosine U-Net (those are the *small* model's specs); Approach D
   actually loads the 99.5M / linear model. Approach-A optimizer: `REFERENCE.md`
   says SGD, but code + `SESSION_2026-03-31.md` say LBFGS. `motivation_*.md` is
   stale (30×30 / "no implementation yet" vs actual 64×64 / both implemented).
   Both diffusion scripts fail as-is (KeyError on removed `stop_window/stop_thresh`;
   broken `DEFAULT_MODEL_PATH`). Trust code + summary. (08)

### 4.2 Doc-vs-doc / historical errors
10. **Old-LaTeX algebra error:** `Gaussian_process_theory.tex` claimed the old
    m-update equals the corrected one — an invalid step; superseded, not ground
    truth. (03)
11. **V-update typo:** `Gaussian_process_theory.tex:695` prints `V = (K̃+G)⁻¹K̃`
    (missing leading `K̃`); its own expansion two lines up and `math.md §2.1` and
    the code all give `V = K̃(K̃+G)⁻¹K̃`. (01, 03)
12. **Two near-identical utility `.tex`:** `distribution_aware_utility_pietro.tex`
    ≈ `conditional_entropy_corrected.tex` (same title/eqs; the pietro version has
    an eigenspace section and a σ²_cond typo). `1D_conditional_entropy_derivation.tex`
    is the cleanest statement — prefer it. (06)

### 4.3 Redundancies (cite once)
13. Full sparse-GP derivation appears 3× (`Gaussian_process_theory.tex` active +
    its commented duplicate; `spatiotemporal_gp_theory.tex` §6). (01)
14. The three utility proof docs each independently state cross-kernel
    proportionality / ρ²=1 / μ∝c, σ²∝c² — one phenomenon, three treatments. (07)
15. `f̄` computed by both `likelihoods.py:expected_firing_rate` and the loop's
    `compute_f_mean` — same formula, two entry points. (03)

### 4.4 Stale line/file references (formulas fine, citations drifted — W4 class)
16. `utility_numerical_analysis.tex` line numbers stale (audit 2026-05-22; all
    formulas match code). (07)
17. `ESTEP_MATH_ANALYSIS.md`, `ANALYTICAL_GRADIENTS_MATH.md` cite legacy
    repo-root `direct_vargp.py` / `utils.py:acosker()` layout; live homes are the
    `eigenspace_*.py` files. (03, 04)

### 4.5 Open issues to flag honestly (not resolved by us)
18. **`Amp` is the sixth kernel hyperparameter (RESOLVED — optimized by default).**
    The original brief listed "the Amp parameter" as `vargp_old` baggage to
    exclude; that was **wrong**. The current `kernels.py` registers `raw_Amp` as a
    trainable `nn.Parameter` (softplus-constrained via `Positive()`, bounds
    (0, 1000], default `Amp = 1.0`) and the M-step computes and applies its
    gradient (`eigenspace_mstep.py:324`:
    `raw_Amp.grad = dL['Amp'] · sigmoid(raw_Amp)`), so by default `Amp` is
    **optimized** — one of six kernel hyperparameters {σ₀, Amp, β, ρ, ε₀ₓ, ε₀ᵧ},
    with `C = Amp · α ⊙ C_smooth ⊙ αᵀ`. The opt-in flag `fix_Amp` (default
    `False`, `eigenspace_training.py:181`) freezes `Amp = 1` to reproduce the
    reference paper's Amp-free model. (Reader 04's "one of six optimized
    hyperparameters" was correct; reader 02's "frozen at 1.0" was not.) The final
    document presents `Amp` as a first-class prefactor of `C`, not footnoted.
    (01, 02, 04, 05)
19. **c_eigenspace offset = mean is NOT mathematically motivated** — forced by
    limiter precision; `subspace_operations.md` top banner flags "should be
    investigated." Carry as an open caveat. (07)
20. **Utility accuracy ceiling:** the standard utility → +∞ and the DA utility
    silently collapses (both via the `r_max = 100` truncation) at high firing;
    the `f_max` guard keeps generation in the accurate regime; a DA-path LUT was
    deliberately not built. Present as a known limitation. (06, 08)
21. **Checkpoint omits the basis `B`** → cross-torch-version reload drift (the
    documented ~5% basis-mismatch risk). Present as a known fragility. (02)

---

## 5. Scope exclusions actually applied

- **Spatiotemporal machinery** — excluded. Present in `spatiotemporal_gp_theory.tex`
  only (frame-concatenation stimulus, `C_temporal`/`ρ_t`, temporal warping
  `τ(t)`, Kronecker `C = C_spatial ⊗ C_temporal` and its trace trick). The
  spatial `x` = one vectorized image; `C = C_spatial`. Absent from all other
  sources.
- **`vargp_old`** — excluded as an implementation (repo-root `utils.py:varGP()/
  Estep()`, old torch-LBFGS, paper-vs-vargp_old framing, the commented-out
  `Gaussian_process_theory.tex` block). Its ELBO / gradient / entropy MATH is
  kept where the current code reuses it (e.g. the current E-step numerically
  reproduces the legacy m-update; the KL `−n_b` offset).
- **`Amp`** — **NOT excluded** (the original brief's exclusion instruction was a
  mistake). It is a genuine optimized kernel hyperparameter in the current code —
  the sixth of {σ₀, Amp, β, ρ, ε₀ₓ, ε₀ᵧ}, with `C = Amp · α ⊙ C_smooth ⊙ αᵀ`
  (see §4.5 item 18); `fix_Amp=True` reproduces the paper's Amp-free model.
- **Deprecated utility routines** (`lambda_moments`, `nd_utility_MC/_NUMERICAL/
  _hybrid`, `argmax_g_old`, the `clamp(max=85)` kludge) — named only as
  not-on-active-path; the current-code math is what's carried.

---

## 6. Notes-file index (source material for composition)

| Notes file | Cluster | Feeds document part |
|---|---|---|
| `notes/01_generative_model_and_variational_framework.md` | generative model, ELBO, whitening, cross-cov | Model + Inference |
| `notes/02_eigenspace_and_implementations.md` | eigenspace repr, both impls, reload, rank-1 | Inference (implementations) |
| `notes/03_estep.md` | E-step Newton update | Training algorithm |
| `notes/04_mstep_fstep_gradients.md` | M-step, F-step, analytical/VJP gradients, loop | Training algorithm |
| `notes/05_kernel_and_structured_prior.md` | arc-cosine kernel, C(β,ρ,ε₀), gradients, jitter | Kernel + prior |
| `notes/06_utility_core.md` | standard vs DA utility, entropy, why-sample-λ | Active learning (utility) |
| `notes/07_utility_subspace_proofs_numerics.md` | proofs I–III, subspace, predictive, numerics | Active learning (deep layer) |
| `notes/08_diffusion.md` | DDPM, guidance, GP link, approaches, status | Synthesis (guided diffusion) |

**Status:** Phase 1 complete. All eight notes files written and cross-checked
against current code. Proceeding to Phase 2 (outline + unified notation table).
