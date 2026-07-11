# 01 — Generative Model & Variational GP Framework

**Cluster:** the math foundation — the probabilistic generative model (GP prior over
image space + Poisson spike-count likelihood), the variational posterior and its
parameterization, the ELBO objective, the two numerical reparameterizations (eigenspace
projection and Cholesky whitening), and the posterior cross-covariance formula (with its
known failure mode). This note establishes the notation (`λ`, `f`, `C`/`K`, likelihood,
`m`/`V`, ELBO) that all later sections build on.

**Sources distilled (all read fully):**
- `Papers/latex_summaries/Gaussian_process_theory.tex` — GP foundations + full variational / sparse-GP derivation (the primary math source).
- `Papers/latex_summaries/spatiotemporal_gp_theory.tex` — clean modern statement of the generative model, sparse variational inference, ELBO, prediction. **SPATIAL-GENERAL parts only** (temporal machinery skipped — see §9).
- `.../gpytorch_porting/.claude/rules/math.md` — the porting project's math reference (model, ELBO, EM, eigenspace).
- `Papers/latex_summaries/whitened_parameterization_explanation.tex` — whitening in GPyTorch's variational strategy and how to unwhiten.
- `Papers/latex_summaries/cross_covariance_bug_analysis.tex` — failure of the manual posterior cross-covariance formula outside the inducing region.

**Ground-truth code consulted (trumps the notes on any conflict):**
- `eigenspace_model.py` — `DirectVGPModel` (the `vargp_direct` eigenspace path).
- `gpy_model.py` — `VariationalGPModel` (the `default_gpy` standard-GPyTorch path).
- `likelihoods.py` — `PoissonLikelihood`.

**Domain framing kept verbatim from sources:** a multi-electrode array records
retinal-ganglion-cell **spike counts** while natural grayscale **images** (the
**stimulus**) are shown; a GP learns each neuron's **receptive field (RF) / tuning**; the
paradigm is **active learning** (choosing the next image to maximize information gain).

---

## 1. The generative model (spatial)

The observed **spike count** `y_i` (also written `r_i`, or `n` for a novel test image) of a
neuron in response to image `x_i` is a **Poisson** draw whose rate is an exponential link
applied to a latent Gaussian-process function `λ`:

    λ(·)  ~  GP(0, K)                          [GP prior — latent tuning function]
    f(x)  =  exp( A·λ(x) + λ₀ )                [exponential link → firing rate]
    y_i   ~  Poisson( f(x_i) )                 [spike-count observation]

[spatiotemporal_gp_theory.tex eq:gp_prior / eq:link / eq:obs; Gaussian_process_theory.tex
§"Exponential firing rate" eq:474–476; math.md §1.1; likelihoods.py:1–11, 21–27]

Symbols:
- `λ(x)` — **latent GP function** over image space (zero-mean prior). This is what the GP
  learns; it is NOT the firing rate. (In the general GP-theory background of
  `Gaussian_process_theory.tex` the latent is called `f`; in the neural model the latent is
  `λ` and `f` is reserved for the firing rate — see the notation clash in §7.)
- `f(x)` — **firing rate** = the Poisson rate parameter. Always positive because of the
  exponential link. When `λ = 0`, `f = exp(λ₀)`, so `exp(λ₀)` is the baseline rate.
- `y_i` / `r_i` — observed **spike count** for image `x_i` (a non-negative integer).
- `A > 0` — **gain**: scales how strongly the latent GP modulates firing rate. Learnable.
  (Code default `A_init = 1.0`; comment notes "typical A ~ 0.01", i.e. cells are usually
  low-gain — `likelihoods.py:50, 54`.)
- `λ₀ ∈ ℝ` — **bias / log-baseline firing rate**. Learnable.
- `K` — the **structured, arc-cosine-based prior covariance kernel** over images. Defined
  only STRUCTURALLY here (§2.1); its internal `(β, ρ, ε₀, σ₀², Amp)` parameterization and
  gradients belong to the **kernel cluster**.

The latent is **log-normally distributed** through the link: modelling `λ` as Gaussian and
`f = exp(A·λ + λ₀)` means we learn (a scaled, shifted) log of the Poisson rate rather than
the rate directly [Gaussian_process_theory.tex:478].

**Input `x` (spatial model):** a single natural grayscale image, vectorized,
`x ∈ ℝ^{n_pix}` (`n_pix = 108×108 = 11664` for this project; the general GP formulas below
never assume the image dimensionality). The full spatiotemporal model instead concatenates
`n_w` consecutive frames into `x_t ∈ ℝ^{n_pix·n_w}` — **excluded here** (§9).

**Code realization (`likelihoods.py`, `PoissonLikelihood`):**
- Link and sampling: `forward(function_samples)` returns `Poisson(rate = exp(A·λ + λ₀))`
  [likelihoods.py:160–174].
- `A = exp(raw_A)` with a positivity constraint (matches varGP's `logA`), `λ₀` unconstrained
  [likelihoods.py:57–72]. Bounds guard: `A ∈ (0, 10]`, `λ₀ ∈ [−50, 50]`
  [likelihoods.py:50–52, 86–136].

---

## 2. GP prior — foundations (function space)

A Gaussian process `λ ~ GP(m, k)` is defined by the property that any finite set of latent
values `{λ(x_1), …, λ(x_n)}` is jointly Gaussian — the **GP prior**
[Gaussian_process_theory.tex §"In function space", eq:fprior]:

    λ(X)  ~  N( m(X), K ),      K_ij = k(x_i, x_j) = Cov[λ(x_i), λ(x_j)]

Our model uses **zero mean** `m(X) = 0` [confirmed in code: `gpy_model.py:82`
`mean_module = ZeroMean()`; the eigenspace path has no mean module at all]. So the prior is
`λ ~ N(0, K)`.

The kernel `K` is fixed in shape before seeing data but its actual entries depend on the
inputs `X` (the images), because `k` is evaluated on them [Gaussian_process_theory.tex:142].

### 2.1 The prior covariance `K` is a first-order arc-cosine kernel (STRUCTURAL only)

The GP prior covariance between two images `x, x'` is a **first-order arc-cosine kernel**
built on a structured input-covariance matrix `C`
[spatiotemporal_gp_theory.tex eq:kernel–eq:J; math.md §1.5]:

    K(x, x') = (1/π) · √(v_x · v_{x'}) · J(θ)
    v_x  = xᵀ C x + σ₀²
    cos θ = ( xᵀ C x' + σ₀² ) / √(v_x · v_{x'})
    J(θ) = sin θ + (π − θ) cos θ

with `J(0) = π`, so `K(x, x) = v_x`. This kernel is **non-stationary** (depends on input
magnitudes `v_x, v_{x'}`, not only on `x − x'`) and is the covariance of an infinite-width
single-hidden-layer ReLU network whose input weights are drawn `~ N(0, C)` with bias
variance `σ₀²` [spatiotemporal_gp_theory.tex §"Neural network interpretation"].

**`C` is the structured prior covariance encoding receptive-field structure. Its internal
parameterization — RF center `ε₀ = (ε₀ₓ, ε₀ᵧ)`, RF size `β`, spatial smoothness `ρ`, bias
variance `σ₀²`, amplitude `Amp` — and all kernel gradients are OUT OF SCOPE for this cluster
(kernel cluster's job).** Established here only so downstream notation (`K`, `K̃`, `C`,
`k(x)`) is unambiguous. See §8 for a doc-vs-code note on `Amp`.

### 2.2 Tractable (Gaussian-noise) case — reference/contrast only

For a *Gaussian* likelihood `y = λ + ε`, `ε ~ N(0, σ²I)`, everything is closed-form: the
marginal is `y|X ~ N(0, K + σ²I)`, and the predictive at a test point is Gaussian with

    mean  f̄_* = k_*ᵀ (K + σ²I)⁻¹ y
    var   V[f_*] = k_** − k_*ᵀ (K + σ²I)⁻¹ k_*

[Gaussian_process_theory.tex eq:predictive distribution w noise, single-test-point box
eq:306–308]. The log-marginal-likelihood (the "evidence") is

    log p(y|X) = −½ yᵀ(K+σ²I)⁻¹y − ½ log|K+σ²I| − (n/2) log 2π

[Gaussian_process_theory.tex eq:logmarginal from gaussians explicit]. **This case is
included only to establish the concepts (prior / likelihood / marginal / predictive /
posterior).** Our model's likelihood is Poisson, so none of these closed forms apply — the
posterior integral over `λ` is intractable and we go variational (§3–§4).

### 2.3 Why Poisson breaks tractability

With the Poisson likelihood the marginal

    p(y|X, θ) = ∫ p_Poisson(y | f(x)) · N(λ | 0, K_θ) dλ

has no closed form [Gaussian_process_theory.tex eq:marginal poisson LK, :395], so neither
the posterior `p(λ|y,X,θ)` nor the exact predictive can be computed directly. The variational
approach approximates the posterior by a tractable Gaussian.

---

## 3. Variational posterior & sparse (inducing-point) GP

### 3.1 Variational approximation

Define a tractable Gaussian **variational posterior** approximating the true (intractable)
posterior over the latent:

    q(λ) = N(λ | m, V)

and choose `(m, V)` to minimize `KL( q(λ) ‖ p(λ|y,X,θ) )`. Because
`KL(q‖p) = ⟨log q⟩_q − ⟨log p(λ|y)⟩_q` and `log p(y)` is constant in `(m,V)`, minimizing the
KL is equivalent to maximizing a lower bound on the log-marginal (the ELBO, §4)
[Gaussian_process_theory.tex eq:KL divergence simple poisson noise case → eq:m and V argmax;
spatiotemporal eq:elbo; math.md §1.3].

### 3.2 Sparse GP: inducing points

Exact inference needs the `N×N` kernel over all `N` training images — `O(N³)` and infeasible
[Gaussian_process_theory.tex:799]. Introduce `M ≪ N` **inducing points**
`Z̃ = {z̃_1, …, z̃_M}` with latent values `λ̃ = λ(Z̃)`, and put the variational Gaussian on
the inducing values:

    q(λ̃) = N(λ̃ | m, V),     m ∈ ℝ^M,   V ∈ 𝕊₊₊^M (SPD)

[spatiotemporal eq:variational; math.md §1.2; Gaussian_process_theory.tex §"Sparse GPs"].
NOTE: `q(λ̃)` approximates the posterior over the **inducing** values, not directly over all
`λ` — the full-input posterior is recovered by marginalizing the conditional prior
`p(λ|λ̃)` against `q(λ̃)` [Gaussian_process_theory.tex:529–545].

### 3.3 Projected posterior at any image `x`

Marginalizing `p(λ(x)|λ̃) · q(λ̃)` gives a Gaussian `q(λ(x)) = N(μ(x), σ²(x))` whose moments
are the inducing-point moments projected through the kernel [spatiotemporal
eq:mu_proj/eq:sigma_proj; math.md §1.2; Gaussian_process_theory.tex eq:meanlambda/varlambda]:

    μ(x)  = k(x)ᵀ K̃⁻¹ m
    σ²(x) = k(x,x) + k(x)ᵀ K̃⁻¹ (V − K̃) K̃⁻¹ k(x)

where
- `k(x) = K(Z̃, x) ∈ ℝ^M` — cross-covariance vector from inducing points to `x`,
- `K̃ = K(Z̃, Z̃) ∈ ℝ^{M×M}` — inducing-point kernel matrix,
- `k(x,x) = K(x,x)` — prior variance at `x`.

**Equivalent algebraic form** (used by the whitened doc): expand `(V − K̃)`:

    σ²(x) = [ k(x,x) − k(x)ᵀ K̃⁻¹ k(x) ]  +  k(x)ᵀ K̃⁻¹ V K̃⁻¹ k(x)
             └── residual (Nyström/Schur) ──┘    └── epistemic term ──┘

[whitened_parameterization_explanation.tex eq:var_standard; cross_covariance_bug_analysis.tex
eq:manual_expanded]. The two forms are identical since
`k(x)ᵀK̃⁻¹(V−K̃)K̃⁻¹k(x) = k(x)ᵀK̃⁻¹VK̃⁻¹k(x) − k(x)ᵀK̃⁻¹k(x)`. This reduces cost to `O(NM²)`.

Defining the **projection vector** `u(x) = K̃⁻¹ k(x)` and **Schur complement**
`s(x) = k(x,x) − k(x)ᵀK̃⁻¹k(x)`, the moments read `μ(x) = u(x)ᵀ m` and
`σ²(x) = s(x) + u(x)ᵀ V u(x)` (zero-mean case) [whitened eq:mean_proj/eq:var_proj].

### 3.4 Code realization — eigenspace path (`DirectVGPModel`)

`eigenspace_model.py` computes exactly these moments, in the eigenbasis of `K̃` (see §5.1).
With `a = K̃⁻¹ k(x)` stored as `KKtilde_inv_b` (`= K_b / eigvals_b`, element-wise), for the
training set [eigenspace_model.py `_lambda_moments_eigenspace`, :256–295]:

    lambda_m   = a @ m_b                                   ≡  μ(x) = k(x)ᵀK̃⁻¹m
    lambda_var = Kvec + (a * (a @ (V_b − K̃_b))).sum(1)    ≡  σ²(x) = k(x,x)+k(x)ᵀK̃⁻¹(V−K̃)K̃⁻¹k(x)
    lambda_var = clamp(lambda_var, min = LAMBDA_VAR_CLAMP)  [numerical floor]

Test points recompute `k(x), k(x,x)` fresh and project the same way
[eigenspace_model.py:340–375]. **Code matches theory exactly** (verified term-by-term:
`a_i = K̃⁻¹k_i`, so `a_i(V−K̃)a_iᵀ = k_iᵀK̃⁻¹(V−K̃)K̃⁻¹k_i`).

---

## 4. The ELBO objective

All parameters are learned by maximizing the **evidence lower bound (ELBO)** on the
log-marginal likelihood `log p(y)`:

    L  =  Σ_i  E_{q(λ_i)}[ log p(y_i | λ_i) ]   −   KL( q(λ̃) ‖ p(λ̃) )

[spatiotemporal eq:elbo; math.md §1.3; Gaussian_process_theory.tex eq:L]. It is a lower
bound on `log p(y)`, **tight when `q` equals the true posterior**. The likelihood factorizes
over images, which makes the first term tractable term-by-term.

### 4.1 Expected log-likelihood term (Poisson + exponential link)

Because `q(λ_i) = N(μ_i, σ_i²)` is Gaussian and the link is exponential, the per-image
expected log-likelihood is closed-form [spatiotemporal eq:ell; Gaussian_process_theory.tex
eq:log likelihood single i general :512; math.md §1.3; likelihoods.py:138–158]:

    E_q[ log p(y_i|λ_i) ]  =  y_i·(A·μ_i + λ₀)  −  exp( A·μ_i + ½A²σ_i² + λ₀ )  +  const

    where const = −log(y_i!)   (independent of all parameters; dropped in code)

The exponential term is the **expected firing rate** under `q`:

    f̄_i  =  E_q[ exp(A·λ_i + λ₀) ]  =  exp( A·μ_i + ½A²σ_i² + λ₀ )

[spatiotemporal eq:fbar; likelihoods.py `expected_firing_rate` :176–192]. This closed form is
the Gaussian **moment-generating function** `E[e^{tX}] = exp(tμ + ½t²σ²)` applied with
`X = A·λ_i + λ₀`, `t = 1` (equivalently `M_λ(A)` with `t=A`)
[Gaussian_process_theory.tex:721].

**Code (`PoissonLikelihood.expected_log_prob`, :138–158):**
```
log_prob = target*(A*mu + lambda0) − exp(A*mu + 0.5*A**2*var + lambda0)
return log_prob.sum(-1)
```
matches the equation exactly (sum over images, `−log(y!)` dropped).

### 4.2 KL term (two Gaussians)

The KL between `q(λ̃) = N(m, V)` and the GP prior `p(λ̃) = N(0, K̃)` is the standard
Gaussian–Gaussian KL [spatiotemporal eq:kl; Gaussian_process_theory.tex :642, :856;
math.md §1.3]:

    KL( q ‖ p )  =  ½ [ log( |K̃| / |V| )  +  Tr( K̃⁻¹ V )  +  mᵀ K̃⁻¹ m  −  M ]

equivalently `−KL = ½ log|V| − ½ log|K̃| − ½ mᵀK̃⁻¹m − ½ Tr(K̃⁻¹V) + const`. All three
sources agree (the `−M` is absorbed into `const` in the `Gaussian_process_theory` /
`math.md` forms).

### 4.3 Full objective (sparse case) and how it is optimized

    θ, m, V, A, λ₀  =  argmax [ Σ_i E_q[log p(y_i|λ_i)]  −  KL(q(λ̃)‖p(λ̃)) ]

[Gaussian_process_theory.tex :580]. Optimization is an **EM-style alternation** (details are
the training-algorithm cluster's job — listed here only to place the ELBO):
- **E-step** — update variational `(m, V)` for fixed hyperparameters (closed-form Newton
  updates).
- **M-step** — update kernel hyperparameters `θ` (gradient-based; requires kernel recompute).
- **F-step** — update likelihood params `(A, λ₀)`; folded into the E-step loop since it needs
  no kernel recompute [math.md §1.4; Gaussian_process_theory.tex :517, :699–701].

The Newton update forms that appear in the sources (given for cross-reference; their
derivation/verification is the E-step cluster's job):

    g = A · Σ_i k_i ( r_i − f̄_i )              G = A² · Σ_i k_i k_iᵀ f̄_i
    V ← K̃ (K̃ + G)⁻¹ K̃
    m ← K̃ (K̃ + G)⁻¹ ( G K̃⁻¹ m + g )

[math.md §1.4, §2.1; Gaussian_process_theory.tex :677, :695]. **Two flagged
inconsistencies around these (see §8):** (i) `Gaussian_process_theory.tex:695` prints
`V_new = (K̃+G)⁻¹K̃`, which is missing a leading `K̃` — its own `V⁻¹` expansion two lines up
(`:690`, α=1) actually yields `V = K̃(K̃+G)⁻¹K̃`, matching `math.md` and the code; (ii)
`math.md §2.2` records that the code's `m` update differs from the "rigorous" Newton
`m ← m + K̃(K̃+G)⁻¹(g − m)` and explains why it still converges (eigenspace + correct `V`).

### 4.4 Prediction / inference after training

The trained posterior moments `(μ, σ²)` do **not depend on the observed `y`** — they can be
evaluated at any new image `x*`, which is what makes the same formula serve as the
predictive distribution [Gaussian_process_theory.tex :538–540, :878]. The expected spike
count for a novel image `x*` is the expected firing rate:

    ⟨n*⟩ = ⟨f(x*)⟩ = exp( A·μ(x*) + ½A²σ²(x*) + λ₀ )
    μ(x*)  = k(x*)ᵀ K̃⁻¹ m
    σ²(x*) = k(x*,x*) + k(x*)ᵀ K̃⁻¹ (V − K̃) K̃⁻¹ k(x*)

[Gaussian_process_theory.tex eq:firing rate expectation :709–730; spatiotemporal
eq:f_pred; likelihoods.py:176–192]. Firing-rate variance (log-normal):

    var( f(x) ) = ⟨f(x)⟩² · ( exp(A²·σ²(x)) − 1 )

[Gaussian_process_theory.tex :972]. **Uncertainty inflates the predicted rate:** `σ²(x*)>0`
raises `f̄_*` above the point estimate `exp(A·μ(x*)+λ₀)` by Jensen's inequality (convex
`exp`) [spatiotemporal §Prediction, note after eq:f_pred].

---

## 5. Two reparameterizations for numerical stability (DISTINCT — do not conflate)

The two current code paths use two *different* reparameterizations of the same variational
GP. Both exist "for numerical stability" but are not the same operation.

### 5.1 Eigenspace projection — `vargp_direct` (`DirectVGPModel`)

Eigendecompose the inducing kernel and keep only the well-conditioned directions
[math.md §1.6; eigenspace_model.py `_compute_eigenspace_quantities` :81–150]:

    K̃ = B Λ Bᵀ,   keep eigenpairs with eigenvalue > EIGVAL_TOL  → B ∈ ℝ^{M×n_b}, Λ = diag(eigvals_b)

Projected state (`DirectVariationalState`, eigenspace_model.py:45–74):

    K̃_b = Λ = diag(eigvals_b)      ← DIAGONAL (this is the whole point)
    m_b  = Bᵀ m                     (variational mean in eigenspace, shape n_b)
    V_b  = Bᵀ V B                   (variational covariance in eigenspace, n_b×n_b, NOT diagonal)
    K_b  = K(X_train, Z̃) @ B        (cross-kernel, N×n_b)
    KKtilde_inv_b = K_b / eigvals_b  = K K̃⁻¹  in eigenspace (element-wise division, no solve)
    Kvec = diag k(x_i, x_i)

Because `K̃_b` is diagonal, `K̃⁻¹` is a reciprocal — no explicit inverse or Cholesky solve
[math.md §1.6; eigenspace_model.py:142–148].

**Initialization at the prior:** `m_b = 0`, `V_b = K̃_b` (i.e. `m=0`, `V=K̃`, so the KL term
is 0 at start) [eigenspace_model.py `_compute_initial_eigenspace` :181–184].

**After an M-step** (kernel params changed → `K̃` and its eigenbasis change),
`recompute_eigenspace()` rebuilds `(B, Λ, …)` and **reprojects** the variational params from
the old to the new eigenbasis: `m_b ← Bₙₑwᵀ B_old m_b`,
`V_b ← Bₙₑwᵀ (B_old V_b B_oldᵀ) Bₙₑw` [eigenspace_model.py `_recompute_eigenspace` :200–249].

`m_b, V_b` are updated by **closed-form Newton** and are `detach()`-ed from autograd (they
are not gradient-trained; keeping the graph caused unbounded memory growth — a documented
19.9 GB blowup at M=419) [eigenspace_model.py `update_variational_params` :500–517]. `V_b` is
symmetrized after each update. The full-`M` params are recovered as
`mean = B @ m_b`, `covariance = B @ V_b @ Bᵀ` [eigenspace_model.py:398–428].

> **These `m_b, V_b` are the ACTUAL (unwhitened) variational parameters, merely rotated into
> `K̃`'s eigenbasis.** Eigenspace projection is NOT the Cholesky whitening of §5.2 (whitening
> would make the prior `N(0,I)`; eigenspace projection leaves the prior as `N(0, Λ)`,
> diagonal but not identity).

### 5.2 Cholesky whitening — `default_gpy` standard strategy (`VariationalGPModel`)

`gpy_model.py` builds a GPyTorch `ApproximateGP` with a `CholeskyVariationalDistribution`
`q(u) = N(m, LLᵀ)` and a choice of strategy [gpy_model.py:49–79]:
- `standard_variational_distribution = True`  → `VariationalStrategy` = **whitened**.
- `standard_variational_distribution = False` → `UnwhitenedVariationalStrategy` (stores
  natural params directly). Comment: *"Use False for EM-style optimization where the kernel
  changes between steps."* [gpy_model.py:32–34, 58–73].

**Why whiten** [whitened_parameterization_explanation.tex §"Why Whiten?"]: reparameterize so
the *prior* becomes a standard normal. With `K̃ = L Lᵀ` (Cholesky) and

    f̃_z = L⁻¹ ( f_z − μ_z )     ⟹     f̃_z ~ N(0, I) under the prior

optimization conditions much better. GPyTorch then stores the **whitened** variational
params `(m̃, Ṽ)`, related to the actual ones by

    m − μ_z = L m̃                V = L Ṽ Lᵀ
    m̃ = L⁻¹ (m − μ_z)            Ṽ = L⁻¹ V L⁻ᵀ

[whitened eq:unwhiten_m/eq:unwhiten_V]. The whitened mean encodes the **deviation from the
mean function**, not absolute function values.

**Consequence for anyone reading GPyTorch's stored params in a manual formula** [whitened
§"Summary of Required Corrections"]:
1. Unwhiten covariance: `V = L Ṽ Lᵀ`.
2. Unwhiten mean: `m − μ_z = L m̃`.
3. Add the mean function back: `μ* = μ(x*) + u(x*)ᵀ L m̃`, and
   `σ²* = s* + u(x*)ᵀ L Ṽ Lᵀ u(x*)`.
Skipping unwhitening or forgetting the mean function gives systematically wrong predictions
(empirically the offset equals the learned mean constant, ≈1.92 in the doc's test)
[whitened §"Empirical Verification"].

> **Caveat — the whitening doc's example uses a constant mean `μ(x)=c ≈ 1.92`, but the
> production model uses `ZeroMean` (`gpy_model.py:82`; the eigenspace path has no mean
> module).** So for the production model `μ_z = 0` and the "add the mean function" step is
> trivially zero — but the *whitening itself* (`m = L m̃`, `V = L Ṽ Lᵀ`) still applies
> whenever `VariationalStrategy` (whitened) is selected. The doc's mean-function machinery is
> retained here because it explains WHY whitening needs care, and guards against a future
> non-zero mean.

The conditional-moment update (conditioning `f(x*)` on an observed `λ` at `x`), in whitened
form, is `μ*_cond = μ(x*) + u(x*)ᵀ L m̃′` with
`m̃′ = m̃ + [ Ṽ Lᵀ u / (s + uᵀ L Ṽ Lᵀ u) ] · (λ − μ_x)` [whitened §"Conditional Moments"] —
this feeds the active-learning utility (utility cluster's territory; see §6).

---

## 6. Posterior cross-covariance between two distinct images — the manual formula's failure

The self-variance formula of §3.3 generalizes to the **cross-covariance** between two
distinct images `x` (a candidate sample location) and `x*` (a query location):

    Σ_{x,x*} = Cov[ λ(x), λ(x*) | D ]  =  k(x, x*) + u(x)ᵀ (V − K̃) u(x*),    u(·) = K̃⁻¹ k(·)

[cross_covariance_bug_analysis.tex eq:manual_formula/eq:manual_expanded]. This is the exact
off-diagonal analogue of the posterior variance; at `x = x*` it reduces to §3.3's `σ²(x)`,
which is correct and used everywhere.

**Failure mode (the "cross-covariance bug"):** for a query `x*` OUTSIDE the inducing-point
region, the manual formula produces cross-covariances that **violate the Cauchy–Schwarz
bound** `|Σ_{x,x*}| ≤ √(Σ_{x,x} Σ_{x*,x*})`. Measured example (RBF, ℓ=0.2, 20 inducing pts in
[−1,1], sample `x=0`): at `x* = ±1.5` the manual formula gives `Σ ≈ −4` while GPyTorch's
covariance gives `≈ 0`; with self-variances `~0.04` the valid max is `~0.04`, so `−4` is
`100×` out of range [cross_covariance_bug_analysis.tex §"Empirical Comparison"].

**Root cause:** in the residual term `k(x,x*) − u(x)ᵀ K̃ u(x*)`, outside the inducing hull the
direct kernel `k(x,x*) → 0` (finite lengthscale) but `u(x*) = K̃⁻¹ k(x*)` can have large
entries via `K̃⁻¹`, and the two pieces fail to cancel [cross_covariance_bug_analysis.tex
§"Root Cause Analysis"]. GPyTorch's internal (stabilized / Nyström-consistent) covariance
does not have this problem.

**Where it bites the model / downstream:** the pathological `Σ_{x,x*}` was used inside
`distribution_aware_utility_gpytorch()` to form conditional moments
`μ_cond(x*) = μ(x*) + (Σ_{x,x*}/Σ_{x,x})·(λ_obs − μ(x))` and
`σ²_cond(x*) = Σ_{x*,x*} − Σ_{x,x*}²/Σ_{x,x}`; a `Σ_{x,x*} ≈ −4` yields a regression
coefficient `≈ −100`, nonsensical conditional means (e.g. `λ = −70`), and **spurious utility
peaks at the edges of the inducing region** (large fake variance reduction → low `H_cond` →
high `U = H_marg − H_cond`) [cross_covariance_bug_analysis.tex §"Impact on Utility"]. A
`clamp` to `±0.999·√(Σ_ii Σ**)` was a band-aid that prevents invalid matrices but does not
restore correct values.

**Fix / recommendation:** compute the joint covariance with GPyTorch and slice the
cross-block instead of the manual formula
(`full_covar = model(cat([x_samples, x_star])).covariance_matrix; Σ = full_covar[:N, N:]`)
[cross_covariance_bug_analysis.tex §"Recommended Fix"].

**Bearing on the model definition (this cluster's scope):** the *self*-variance form of
§3.3–§3.4 is correct and is what both code paths use for posterior variance and for the
expected firing rate. The manual *cross*-covariance extension is numerically unreliable
outside the inducing hull; the utility computation that consumes it is the active-learning /
utility cluster's detail.

---

## 7. Notation table

| Symbol | Meaning | Source(s) |
|---|---|---|
| `λ(x)`, `λ̃` | latent GP function; its values at inducing points | all docs + code |
| `f(x)` | firing rate = Poisson rate = `exp(Aλ+λ₀)` | all docs + code |
| `y_i` / `r_i` / `n*` | observed spike count (train / train / novel test) | Gaussian_process_theory, spatiotemporal, math.md, code (`target`) |
| `A` | gain (link), learnable, `A = exp(raw_A) > 0` | spatiotemporal, math.md, likelihoods.py |
| `λ₀` | bias / log-baseline firing rate, learnable | all + likelihoods.py |
| `K` / `k(x,x')` | GP prior covariance (arc-cosine kernel over `C`) | spatiotemporal, math.md |
| `C` | structured input covariance inside the kernel (RF structure) — STRUCTURAL only here | spatiotemporal, math.md, kernels.py |
| `σ₀²` | kernel bias variance (fixed) | spatiotemporal, math.md |
| `Z̃` / `z` / `X̃` | inducing points | math.md, whitened, cross-cov, Gaussian_process_theory |
| `M` (`n_inducing`) | number of inducing points | spatiotemporal, whitened, math.md, code |
| `K̃` = `K(Z̃,Z̃)` | inducing-point kernel (`M×M`); code `K_tilde` | spatiotemporal (`Kuu`), math.md, Gaussian_process_theory, code |
| `k(x)` = `K(Z̃,x)` | cross-covariance vector (`M`) | all sparse docs |
| `u(x)` = `K̃⁻¹k(x)` | projection vector | whitened, cross-cov |
| `m`, `V` | variational mean / covariance of `q(λ̃)` | all + code (`m_b`,`V_b` eigenspace) |
| `μ(x)`, `σ²(x)` | posterior mean / variance of `λ` at `x` | spatiotemporal, math.md, code (`lambda_m`,`lambda_var`) |
| `f̄_i` / `⟨f⟩` / `f_mean` | expected firing rate under `q` | spatiotemporal, Gaussian_process_theory, code |
| `L` (ELBO) | evidence lower bound / lower bound on log-marginal | spatiotemporal (`𝓛`), math.md, Gaussian_process_theory |
| `g`, `G` | E-step Newton gradient / curvature aggregates | math.md, Gaussian_process_theory |
| `θ` | kernel hyperparameters | all |
| `B`, `Λ`=`eigvals_b`, `n_b` | eigenvectors / kept eigenvalues / kept count of `K̃` | math.md, eigenspace_model.py |
| `m_b`, `V_b`, `K̃_b`, `K_b` | eigenspace-projected quantities | eigenspace_model.py |
| `L` (Cholesky), `m̃`, `Ṽ` | Cholesky factor `K̃=LLᵀ`; whitened variational params | whitened_parameterization_explanation.tex, gpy_model.py |
| `μ_z` | mean function at inducing points (=0 for ZeroMean production model) | whitened; code `ZeroMean` |
| `Σ_{x,x*}` | posterior cross-covariance between two distinct images | cross_covariance_bug_analysis.tex |

**CLASHES (same symbol, different meaning) — flag for the writer:**
- **`f`** = general GP latent (Gaussian_process_theory background) **vs** firing rate
  (neural model, everywhere else + code). In the neural model the latent is `λ`.
- **`K`** = training-set kernel `K(X,X)` (`N×N`, general GP theory) **vs** inducing kernel
  `K(Z̃,Z̃)` (`M×M`) — the whitened & cross-cov docs write plain `K` for what everyone else
  calls `K̃`.
- **`A`** = posterior covariance of weights `N(w̄,A)` in feature-space GP theory
  (Gaussian_process_theory eq ~110) **vs** the gain in the link (everywhere else + code).
- **`L`** = ELBO / lower bound (Gaussian_process_theory, spatiotemporal, math.md) **vs**
  Cholesky factor of `K̃` (whitened doc, gpy code). Also `L` = `L u` sampling factor in
  Gaussian_process_theory §Sampling.
- **`σ`/`σ²`** = Gaussian observation-noise variance (general theory) **vs** posterior
  variance of `λ` (neural model) **vs** `σ₀²` kernel bias variance. Three distinct uses.
- **`m`** = variational mean **vs** `m(x)` GP mean function (general theory); **`M`** = #
  inducing points (case-sensitive collision with `m`).
- **`μ`** = posterior mean of `λ` **vs** `μ(x)`/`μ_z` mean function (whitened doc).

**SYNONYMS (different symbol, same concept):**
- spike count: `y` = `r` = `n` (and code `target`).
- latent function: `λ` (neural) — was `f` in the general GP background.
- inducing kernel: `K̃` = `Kuu` = `K` (whitened/cross-cov) = `K_tilde`/`K_tilde_b` (code).
- inducing points: `Z̃` = `z` = `X̃` = `X_tilde`/`inducing_points`.
- projection vector: `u(x)ᵀ` (docs) = a row of `a = KKtilde_inv_b` (eigenspace code).
- expected firing rate: `f̄_i` = `⟨f⟩` = `f_mean`.
- # inducing points: `M` (upper) = `m` (Gaussian_process_theory lowercase).

---

## 8. Conflicts / redundancies / inconsistencies

**Doc-vs-doc**
1. **`V`-update algebra slip.** `Gaussian_process_theory.tex:695` prints
   `V_new = (K̃+G)⁻¹K̃` (missing a leading `K̃`). Its own `V⁻¹` line (`:690`, α=1) gives
   `V⁻¹ = K̃⁻¹(K̃+G)K̃⁻¹` ⟹ `V = K̃(K̃+G)⁻¹K̃`, which is what `math.md §2.1` and the code use.
   The printed `:695` form is a typo. (E-step cluster should own the resolution.)
2. **Two algebraic forms of `σ²(x)`** appear (`k+kᵀK̃⁻¹(V−K̃)K̃⁻¹k` in
   spatiotemporal/math.md/code; `k − kᵀK̃⁻¹k + kᵀK̃⁻¹VK̃⁻¹k` in whitened/cross-cov). Not a
   conflict — algebraically identical (§3.3). Note both so a reader isn't tripped.
3. **KL constant bookkeeping**: spatiotemporal writes `… − M`; Gaussian_process_theory /
   math.md fold `−M` into `const`. Same quantity.
4. **Large commented-out block** in `Gaussian_process_theory.tex:752–1007`
   (`\begin{comment}…\end{comment}`) duplicates the sparse-GP derivation and carries a
   "Wrong" annotation on its M-step (`:925`) and a different KL sign convention (`:980`,
   `+½mᵀK⁻¹m`). It is NOT rendered; I distilled the **active** text (≤ :751 + the ELBO/EM/
   inference sections that are outside the comment). Treat the commented block as historical.

**Doc-vs-code**
5. **`Amp` (amplitude) — the brief's "vargp_old baggage" assumption is contradicted by
   current code.** The task brief lists `Amp` as deprecated vargp_old baggage to exclude, and
   spatiotemporal_gp_theory.tex's parameter table indeed has NO amplitude (kernel scale comes
   from `√(v_x v_{x'})`). BUT the **current** gpytorch-port `ArcCosineKernel` registers a live
   `raw_Amp` positive parameter that multiplies `C` directly: `C = Amp·α·C_smooth·αᵀ`
   (`kernels.py:19, 87, 195, 212–220, 283–285`), and `math.md §1.5` also lists "Amp:
   amplitude". So `Amp` is **present in current code**, not purely deprecated. This is
   kernel-internal (out of scope for this cluster) — flag to the **kernel cluster / writer**:
   decide whether current-code `Amp` is the same object the brief meant to drop, or a live
   parameter the consolidated doc must keep.
6. **Mean function**: whitened doc's worked example assumes a constant mean `≈1.92`; both
   production paths use zero mean (`gpy_model.py:82 ZeroMean`; eigenspace path has none). The
   whitening formulas still hold with `μ_z = 0` (§5.2 caveat).
7. **`m`-update discrepancy (already documented in `math.md §2.2`)**: code uses
   `m ← K̃(K̃+G)⁻¹(GK̃⁻¹m+g)`; the "rigorous" Newton is `m ← m + K̃(K̃+G)⁻¹(g−m)`. `math.md`
   argues convergence is unaffected (eigenspace diagonal `K̃_b`, correct `V`, iteration). This
   is an **E-step** matter — noted here only because it surfaces in this cluster's sources.

**Redundancy**: the full sparse-GP variational derivation appears three times (active
Gaussian_process_theory §"Sparse GPs", its commented duplicate, and spatiotemporal §6). They
agree on the final ELBO / moments / KL; the writer should cite one and note the others.

---

## 9. Scope exclusions (what I deliberately dropped)

**Spatiotemporal-specific (from `spatiotemporal_gp_theory.tex` — distilled spatial-general
parts only):**
- §Stimulus Representation — frame concatenation `x_t = vec([s_t, s_{t−1}, …, s_{t−n_w+1}])`,
  `n_x = n_pix·n_w`, `N = T − n_w + 1`, the `X ∈ ℝ^{n_pix×n_w}` reshape. **Dropped.** For the
  spatial model `x` is a single vectorized image `∈ ℝ^{n_pix}`.
- §Temporal Component — temporal covariance `C_temporal` (eq:C_temporal, `ρ_t`), the temporal
  warping `τ(t)` and its parameter `α_t` (eq:warping). **Dropped entirely.**
- §Kronecker Structure — `C = C_spatial ⊗ C_temporal` (eq:kronecker) and the
  trace-factorization efficiency trick `xᵀ(C_s⊗C_t)x = Tr(Xᵀ C_s X C_t)` (eq:kron_efficient).
  **Dropped** (temporal-only). The spatial model uses `C = C_spatial` directly.
- The parameter table's `ρ_t`, `α_t` rows. **Dropped**; kept only the spatial/likelihood/
  variational rows conceptually.

**vargp_old / deprecated (per scope constraint #4):**
- The repo-root `utils.py` `varGP()` / `Estep()` code path, the old torch-LBFGS setup, and any
  "our approximation of the paper" / paper-vs-vargp_old comparison framing — **not described**.
  The model is stated as realized by the CURRENT paths only: eigenspace `DirectVGPModel`
  (`vargp_direct`) and standard-GPyTorch `VariationalGPModel` (`default_gpy`).
- `Gaussian_process_theory.tex:752–1007` commented block — treated as historical (§8.4).
- **`Amp`**: the brief flags it as vargp_old baggage, but current code keeps it live (§8.5).
  I did NOT describe its role (kernel-internal, out of scope) — only flagged the tension.

**Other clusters' territory (mentioned, not derived here):**
- Kernel internals: `C`'s `(β, ρ, ε₀, σ₀², Amp)`, the pixel grid, mask logic, log-space
  parameterization, and all kernel gradients (`math.md §1.5`, `kernels.py`) → **kernel cluster**.
- E-step / M-step / F-step **update-rule derivations and the `m`/`V` formula verification**
  (`math.md §1.4, §2`, `Gaussian_process_theory.tex` E-step boxes, `estep.py`/`fstep.py`) →
  **training-algorithm cluster**. I state the ELBO objective and its two terms; I do not
  derive the Newton updates.
- Active-learning **utility** (mutual information `U = H_marg − H_cond`, Lambert-`W₀`
  overflow-safe computation `w + log w = y`, `Gaussian_process_theory.tex §"Active learning"`
  :736–751) → **active-learning / utility cluster**. Covered here only via the posterior
  cross-covariance formula it consumes (§6).
