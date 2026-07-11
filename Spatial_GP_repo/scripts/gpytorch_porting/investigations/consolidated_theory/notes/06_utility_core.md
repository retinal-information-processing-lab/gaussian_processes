# 06 — Utility & Entropy Core (the active-learning acquisition objective)

**What this note owns.** The utility/acquisition function used to pick the next
image to show — *utility of a candidate image = the expected reduction in entropy
(information gain) about the neuron's tuning* — plus the entropy machinery it is
built from: the marginal response entropy, the conditional-entropy derivation (1D
and n-d), the "distribution-aware" variant, why we sample the firing-rate latent
λ instead of using a point estimate, and the code that realizes all of this.

**Scope guards for this note.**
- Neuroscience framing is literal: the neuron is a retinal ganglion cell; λ(x) is
  the latent that sets its firing rate to natural image x; "choosing the next
  image" means choosing the image whose observed spike count most reduces our
  uncertainty about the cell's receptive field / tuning.
- **SPATIAL GP only.** No spatiotemporal machinery appears here; where a source
  had it, it is out of scope and skipped.
- **CURRENT implementation only.** Ground truth = `acquisition.py`,
  `utils.py` (LOCAL `gpytorch_porting/utils.py`), `metrics.py`. Deprecated varGP
  helpers (repo-root `utils.py`, `vargp_style_estep.py`) are flagged, not carried.
- The **subspace/eigenbasis proofs** (K = B Kᵦ Bᵀ block-inversion, sign-flip /
  near-degenerate numerics) and the **divergence theorems** are owned by the
  sibling note — referenced here, not re-derived.

**Plain-text math convention.** This is a `.md` reference, not compiled LaTeX, so
math is written in unicode (λ, σ², μ, ρ², ∑, ∫, ≈, →, ᵀ, ∂). The `.tex` sources
use the LaTeX forms of the same symbols.

---

## 0. TL;DR — two utilities, one idea

Both utilities express the SAME idea: *utility(x*) = information the spike-count
observation R at candidate image x* gives about the neuron's tuning = a drop in
entropy.* They differ in **what uncertainty they score**.

| | **Standard utility** (PRODUCTION) | **Distribution-aware (DA) utility** (research) |
|---|---|---|
| Symbol | U_std(x*) | U_DA(x*) |
| Formula | H_marg(x*) − E[H_noise(x*)] | H_marg(x*) − E_{x∼p(x)}[H_cond(x*)] |
| Meaning | info R gives about the latent firing rate **at x* itself** | info R gives about firing rates at **natural images x∼p(x)** |
| = mutual information | I(f(x*); R \| x*, D) | I(f(X); R \| X, x*, D), X∼p(x) |
| Depends on | marginal GP moments at x* only | marginal moments **and** cross-covariance to natural images |
| Code | `standard_utility()` → `nd_utility_new()` | `distribution_aware_utility()` → `compute_H()` + `get_gp_conditional_moments()` |
| Where used | `run_active_loop.py:241` — the real experiment | `investigations/` only (workbench, diffusion, subspace) |

**Ground-truth fact to hold onto:** the closed-loop experiment selects images with
**`standard_utility`** (verified: `run_active_loop.py:241`). The DA utility is the
theoretically richer object derived in the `.tex` summaries, but in the current
codebase it is an investigation tool, not the production acquisition. (It also has
a known norm-exploitation divergence under free-space gradient ascent — see §9.)

---

## 1. The acquisition objective as mutual information (DA derivation)

Source: `distribution_aware_utility_pietro.tex`, `conditional_entropy_corrected.tex`.

**Set-up.** Spike count r to image x is Poisson with mean firing rate f(x):

    p(r | x, f) = Poisson(r | f(x)),   f(x) = exp(A·λ(x) + λ₀).

λ(x) is a smooth latent function with a GP posterior given data
D_n = {(xᵢ, rᵢ)}. A and λ₀ are fixed scalars (the PoissonLikelihood's `.A`,
`.lambda0`). We want to predict the neuron's responses to natural stimuli
x ∼ p(x), and to learn them as fast as possible by choosing the next image x*.

**Objective.** Define the random variable Z := (X, f(X)) with X ∼ p(x) a natural
image and f(X) its mean firing rate. Choose x* to maximize the information the
observation R (at x*) carries about Z:

    U(x*) := I(Z; R | x*, D) = I( (X, f(X)); R | x*, D ).            (eq. 1)

**Chain rule for mutual information:**

    I((X, f(X)); R | x*, D) = I(X; R | x*, D) + I(f(X); R | X, x*, D).

X (which natural image we imagine) is independent of R (the response at x*) given
x* and D, so the first term is 0. Hence

    U(x*) = I(f(X); R | X, x*, D) = E_{x∼p(x)}[ I(f(x); R | x, x*, D) ].   (eq. 2)

**Entropy decomposition of each MI term.** For a fixed natural image x,

    I(f(x); R | x, x*, D) = H(R | x*, D) − E_{f(x)}[ H(R | x*, f(x), D) ],

because (i) H(R | x, x*, D) = H(R | x*, D) — the marginal of R at x* does not
depend on which natural image we intend to ask about, and (ii) conditioning R on
the latent value f(x) leaves the expected entropy E_{f(x)}[H(R | x*, f(x), D)].
Averaging over x∼p(x):

    U(x*) = H_marg(x*) − H_cond(x*),                                  (eq. 3)

    H_marg(x*) := H(R | x*, D),                                       (eq. 4)
    H_cond(x*) := E_{x∼p(x), λ(x)∼p(λ|D)}[ H(R | x*, f(x), D) ].      (eq. 5)

**Note on the two expectations in H_cond.** The main decomposition
(`…pietro.tex` eq:hcond) writes H_cond with the outer E_{x∼p(x)} only; the
"Conditional Entropy" section restores the inner E_{λ(x)∼p(λ|D)}. Both are
required and both are honored in code — f(x) = exp(A·λ(x)+λ₀) is itself uncertain
(only the λ(x) posterior is known), so we must integrate over λ(x). The
**necessity** of the inner sampling (not just evaluating at the mean) is the whole
point of `why_sample_lambda.tex` — see §7.

By symmetry of mutual information, U(x*) equals the reduction in uncertainty about
the firing rates {f(x): x∼p(x)} produced by observing R at x*. That is the precise
sense of "how much does image x* teach us about the cell's tuning on natural
images."

---

## 2. Marginal entropy H_marg (shared by BOTH utilities)

Source: `…pietro.tex` / `conditional_entropy_corrected.tex` §"Marginal entropy";
code: `utils.py:compute_H`, `nd_utility_new`, `_diff_laplace_log_probs`,
`_diff_argmax_g`.

The marginal predictive entropy of the spike count at x* is

    H_marg(x*) = − ∑_{r=0}^∞ p(r | x*, D) · log p(r | x*, D).          (eq. 6)

Define the latent **log-firing-rate** g(x) := A·λ(x) + λ₀, so f = exp(g). The
marginal predictive of r integrates the Poisson likelihood against the Gaussian
posterior of g(x*):

    p(r | x*, D) = ∫ Poisson(r | eᵍ) · N(g | μ, σ²) dg
                 = (1/r!) ∫ exp(r·g − eᵍ) · N(g | μ, σ²) dg,           (eq. 7)

with the **log-firing-rate posterior moments**

    μ = μ(x*) = A·E[λ(x*) | D] + λ₀,        (transform of the raw GP mean)
    σ² = σ²(x*) = A²·Var[λ(x*) | D].         (transform of the raw GP variance)  (eq. 8)

The integral has no closed form; use a **Laplace expansion** of the integrand
about its mode ḡ_r:

    log p(r | x*, D) ≈ ḡ_r·r − e^{ḡ_r} − (ḡ_r − μ)²/(2σ²)
                       − ½·log(1 + σ²·e^{ḡ_r}) − log r!.               (eq. 9)

**Mode in closed form (Lambert W).**

    ḡ_r = argmax_g { r·g − eᵍ − (g−μ)²/(2σ²) }.

Stationarity: r − eᵍ − (g−μ)/σ² = 0 ⇒ g = r·σ² + μ − σ²·eᵍ. Substituting
w = σ²·eᵍ gives w + log w = log σ² + r·σ² + μ, i.e. w = W₀(σ²·e^{rσ²+μ}), hence

    ḡ_r = r·σ² + μ − W₀( σ²·e^{ r·σ² + μ } ),                          (eq. 10)

W₀ = principal branch of the Lambert W function.

**Truncation.** The r-sum in eq. 6 is formally infinite but decays fast (retinal
firing rates are low), so it is truncated at r_max. This truncation is the source
of the entropy-landscape artifacts in §9; the adaptive-r_max guard handles it.

**Code map (marginal entropy).**
- `_diff_argmax_g(r, σ², μ)` → eq. 10. Computes y = log(σ²) + r·σ² + μ and returns
  r·σ² + μ − W₀(exp(y)) via `_lambertw0_log` (a custom-autograd Lambert-W-of-exp,
  Newton-iterated, backward dW/dy = W/(1+W)).
- `_diff_laplace_log_probs(μ, σ², r)` → eq. 9 (log_p_laplace). For σ² < 1e-6 it
  falls back to the exact Poisson log-pmf via `torch.where` (differentiable
  branch) — the Laplace form is unstable at tiny variance.
- `compute_H(mu, sigma2, r_max, a, lambda0)` → eq. 6. Takes **RAW λ moments**,
  transforms internally (logf_mean = a·mu + lambda0, logf_var = a²·sigma2), sums
  −∑ p_r·log p_r over r = 0..r_max−1.
- `nd_utility_new(mu_g, sigma2_g, r_max)` also builds H_marg from
  `_diff_laplace_log_probs`, but takes **already-transformed g moments** and sums
  over r = 0..r_max (note the off-by-one vs compute_H; both are truncations, see
  §8 gotcha).

---

## 3. Standard (production) utility: H_marg − E[H_noise]

Source: `da_utility_theory.md` §"The Two Utilities"; code:
`acquisition.py:standard_utility`, `utils.py:nd_utility_new`.

**Definition.**

    U_std(x*) = H_marg(x*) − E[H_noise(x*)]
              = H(R | x*, D) − E_{f∼posterior}[ H(R | f) ]
              = I( f(x*); R | x*, D ).                                 (eq. 11)

This is the mutual information between the response and the latent firing rate **at
the query point x* itself** — no p(x), no cross-covariance. It scores "how much
would a spike count at x* pin down the firing rate at x*." `da_utility_theory.md`:
"only depends on marginal GP moments at x*; no awareness of other stimuli in the
world."

**The E[H_noise] second term in closed form.** H_noise is the aleatoric Poisson
noise entropy given the rate f: the entropy of Poisson(f) is

    H(Poisson(f)) = f·(1 − log f) + E_{r∼Poisson(f)}[ log r! ].

With g = log f and g ∼ N(μ_g, σ²_g), use the lognormal identities
E[eᵍ] = exp(μ_g + ½σ²_g) and E[g·eᵍ] = exp(μ_g + ½σ²_g)·(μ_g + σ²_g):

    E_g[ f(1 − log f) ] = E_g[ eᵍ(1 − g) ]
                        = exp(μ_g + ½σ²_g)·(1 − (μ_g + σ²_g))
                        = − exp(μ_g + ½σ²_g)·(μ_g + σ²_g − 1).

The remaining E[log r!] is taken under the marginal predictive p(r|x*,D):

    E[H_noise] = − exp(μ_g + ½σ²_g)·(μ_g + σ²_g − 1) + ∑_r p(r|x*,D)·log r!.  (eq. 12)

**Code map (standard utility).**
- `nd_utility_new(mu_g, sigma2_g, r_max)`:
  - `H_marg = −∑ p_r·log p_r` (eq. 6 via Laplace),
  - `E_H_noise = −exp(mu_g + 0.5·sigma2_g)·(mu_g + sigma2_g − 1) + ∑ p_r·lgamma(r+1)`
    (eq. 12; the code comment cites "Eq. 33 PNAS" — Goldin et al. 2023),
  - returns `H_marg − E_H_noise`.
- `standard_utility(model, likelihood, x_candidates, r_max, adaptive_r_max, …)`:
  1. `lambda_mean, lambda_var = get_gp_marginal_moments(model, x_candidates)` —
     RAW λ moments (§6),
  2. transform: `mu_g = A·lambda_mean + lambda0`, `sigma2_g = A²·lambda_var`
     (eq. 8),
  3. optional `compute_adaptive_rmax(mu_g, sigma2_g, …)` (§8),
  4. `utility = nd_utility_new(mu_g, sigma2_g, r_max)`,
  5. returns `{'utility', 'mu_g'}`.

---

## 4. Distribution-aware utility: H_marg − E_{x∼p(x)}[H_cond]

Source: `da_utility_theory.md`; `…pietro.tex`; code:
`acquisition.py:distribution_aware_utility`.

**Definition** (eq. 3 with the H_cond of eq. 5):

    U_DA(x*) = H_marg(x*) − E_{x∼p(x), λ(x)∼q}[ H(R | x*, λ(x), D) ].   (eq. 13)

Where standard utility only reduces uncertainty about f at x*, DA utility reduces
uncertainty about f at the **natural images we actually want to predict**. The
"distribution-aware" qualifier means exactly *aware of p(x)*, the natural-image
distribution — it prefers a candidate x* whose observation is informative about
responses to typical natural images, weighting by cross-covariance to those
images.

**Monte-Carlo algorithm (one λ per image; see §7 for why exactly one).** Given a
pre-drawn sample set {xᵢ}_{i=1}^{N_mc} ∼ p(x):

    H_marg(x*)  = compute_H( μ_marg, σ²_marg, r_max, A, λ₀ )        (marginal at x*)
    for i = 1..N_mc:
        (μ_i, σ²_i)  = GP posterior λ-moments at xᵢ
        λ_i          = μ_i + σ_i·ε,  ε∼N(0,1)      (if sample_lambda; else λ_i = μ_i)
        (μ_cond, σ²_cond) = conditional λ-moments at x* given λ(xᵢ)=λ_i   (§5)
        H_cond_i     = compute_H( μ_cond, σ²_cond, r_max, A, λ₀ )
    H_cond = mean_i H_cond_i
    U_DA   = H_marg − H_cond

**Code map (DA utility).** `distribution_aware_utility(model, likelihood,
x_candidates, x_samples, r_max, adaptive_r_max, sample_lambda=True, …)`:
- Step 1: `mu_marg, sigma2_marg = get_gp_marginal_moments(model, x_candidates)`;
  `H_marg = compute_H(mu_marg, sigma2_marg, r_max_marg, a=A, lambda0=λ₀)`. Note
  `compute_H` takes **RAW λ moments** and transforms internally (contrast with
  `standard_utility`, which transforms before calling `nd_utility_new` — §8 gotcha).
- Step 2 loop over `x_samples`: the GP posterior at xᵢ and the reparameterized
  draw λ_i are computed **inside `torch.no_grad()`** (they do not depend on
  x_candidates, so excluding them from the graph avoids O(N_mc) graph
  accumulation). Then `get_gp_conditional_moments(model, x_candidates, x_i,
  lambda_i)` (differentiable w.r.t. x_candidates), then `H_cond_i =
  compute_H(mu_cond, sigma2_cond, …)`. Accumulate, divide by N_mc.
- Returns `{'utility': H_marg − H_cond, 'H_marg', 'H_cond', 'mu_g_marg'}`.
- **Mode restriction:** DA utility currently needs `model(X).covariance_matrix`
  (a GPyTorch VariationalGP / "default_gpy"); the vargp_direct/eigenspace path is
  deferred (would need the augmented-matrix approach of §5).
- `sample_lambda` is **not a debug flag** — `False` uses the posterior mean λ_i =
  μ_i (deterministic, reproducible sanity check); `True` is the correct unbiased
  estimator (§7).

---

## 5. Conditional GP moments given a hypothetical λ(x) (the "corrected" derivation)

Sources: `1D_conditional_entropy_derivation.tex` (cleanest),
`conditional_entropy_corrected.tex`, `…pietro.tex`; code:
`utils.py:get_gp_conditional_moments`.

To compute H_cond we need the predictive moments of λ(x*) after hypothetically
observing λ(x) = λ at a natural image x. This is Gaussian conditioning on the
joint posterior of [λ(x), λ(x*)].

**Building blocks** (inducing values λ̃ with variational posterior
q(λ̃) = N(m, V); inducing kernel matrix K; kernel vector k(x) = [k(x, x̃_j)]):

    u   = K⁻¹ k(x),     s   = k(x,x)   − k(x)ᵀ K⁻¹ k(x)      (projection, Schur compl.)
    u*  = K⁻¹ k(x*),    s*  = k(x*,x*) − k(x*)ᵀ K⁻¹ k(x*)

s, s* are the **prior** conditional variances of λ(x), λ(x*) given the inducing
values.

**Marginal posterior moments** (integrate prior conditional N(uᵀλ̃, s) against
q(λ̃)):

    μ_x  = E[λ(x) | D]   = uᵀ m,        Σ_xx = Var[λ(x) | D]   = s  + uᵀ V u
    μ_*  = E[λ(x*) | D]  = u*ᵀ m,       Σ_** = Var[λ(x*) | D]  = s* + u*ᵀ V u*   (eq. 14)

**The critical term — prior conditional covariance** between λ(x) and λ(x*) given
the inducing values (this is the term the ORIGINAL formulation missed):

    c = k(x, x*) − k(x)ᵀ K⁻¹ k(x*) = k(x, x*) − uᵀ K u*.               (eq. 15)

**Cross-covariance** (law of total covariance = mean of prior conditional cov +
cov of conditional means):

    Σ_x* = c + uᵀ V u* = k(x, x*) + uᵀ (V − K) u*.                     (eq. 16)

**Conditional moments** (standard Gaussian conditioning on the joint of eq. 14–16):

    μ_cond(x*) = μ_* + (Σ_x* / Σ_xx)·(λ − μ_x)                          (eq. 17)
    σ²_cond(x*) = Σ_** − Σ_x*² / Σ_xx.                                  (eq. 18)

These λ-moments are transformed to g-space (μ_g = A·μ_cond + λ₀, σ²_g =
A²·σ²_cond) and fed to the Laplace entropy (eq. 6, 9, 10) to give
H(R | x*, λ(x), D).

**The correction (doc-vs-doc history).** An ORIGINAL formulation
(`active_learning_pietro.tex` Eq. 217) used cross-covariance u*ᵀ V u only —
**missing c**. The 1D doc proves the gap is exactly
Σ_x* − uᵀ V u* = k(x,x*) − uᵀ K u* = c (eq. 15). Omitting c overestimates the
conditional variance, which can make conditioning *increase* variance and yield
**negative utility** in the mean-value (deterministic) case. `V_cond` (the
rank-1-corrected inducing-point posterior covariance),

    V_cond = V − (V u uᵀ V) / (s + uᵀ V u),                            (eq. 19)

with σ²_cond = s* + u*ᵀ V_cond u*, is the equivalent "corrected" statement
(`conditional_entropy_corrected.tex` §"Corrected Variance Update"). The
augmented-matrix appendix in the 1D doc verifies eq. 17 algebraically via
K_aug = [[k(x,x), k(x)ᵀ], [k(x), K]] and the block inverse (Schur complement s).

**How the CODE realizes this (and why it is correct-by-construction).**
`get_gp_conditional_moments(model, x_star, x_sample, lambda_sample)`:
- builds `all_x = [x_sample; x_star]`, calls `posterior = model(all_x)`, reads the
  **full joint** `full_covar = posterior.covariance_matrix`,
- extracts `mu_sample, var_sample = full_covar[0,0], mu_star, var_star =
  full_covar.diag()[1:], cross_cov = full_covar[0, 1:]`,
- `mu_cond = mu_star + cross_cov·(innovation / var_sample)` (eq. 17),
  `sigma2_cond = var_star − cross_cov² / var_sample`, clamped ≥ 1e-8 (eq. 18).

The GPyTorch variational posterior covariance between any two points a,b is
Cov(λ(a),λ(b)|D) = k(a,b) − uₐᵀ K u_b + uₐᵀ V u_b = c + uₐᵀ V u_b — **exactly**
Σ_x* of eq. 16, c included. So by taking `cross_cov` from the true joint
`covariance_matrix`, the code sidesteps the manual-assembly bug entirely; it never
forms the buggy u*ᵀ V u. The `.tex` "correction" is a warning about hand-assembled
cross-covariances, not a defect in the current code. `lambda_sample` must be a
**tensor** (not a Python float) so gradient flows through the reparameterization.

---

## 6. GP marginal & conditional moments → code (and the deprecated `lambda_moments`)

Code: `utils.py:get_gp_marginal_moments`, `get_gp_conditional_moments`.

- `get_gp_marginal_moments(model, x_star)` → `model.eval(); posterior =
  model(x_star); return posterior.mean, posterior.variance`. These are the **raw
  λ-posterior moments** μ_x = u*ᵀ m and Σ_** = s* + u*ᵀ V u* (eq. 14), read
  straight off the GPyTorch posterior. Deliberately NOT wrapped in
  `torch.no_grad()` so gradients flow x_star → kernel → (μ, σ²) for gradient-based
  x* optimization (contrast the playground version, which blocks gradients).
- `get_gp_conditional_moments(...)` → eq. 16–18 via the full covariance (§5).

**`lambda_moments` — DEPRECATED, flag.** The task brief lists a `lambda_moments`
helper. There is **no `lambda_moments` in the current
`gpytorch_porting/utils.py`** (verified: only `get_gp_marginal_moments` /
`get_gp_conditional_moments` exist). `lambda_moments` is a **legacy varGP
analytic λ-moment routine** in the repo-root deprecated `utils.py`
(referenced as "utils.py:lambda_moments() lines 3937-3952" by `eigenspace_model.py`
/ `eigenspace_gradients.py`, and called in `deprecated/vargp_style_estep.py`). Its
current-code role is filled by `get_gp_marginal_moments` (marginal μ_λ, σ²_λ) and
`get_gp_conditional_moments` (conditional μ_λ|x, σ²_λ|x). Use those; do not
resurrect `lambda_moments`.

---

## 7. Why we SAMPLE λ(x) instead of using its mean (bias/variance)

Source: `why_sample_lambda.tex`.

H_cond is a **double expectation** (eq. 5): over natural images x∼p(x) and over the
posterior λ(x) ∼ N(μ_λ(x), σ²_λ(x)). The tempting shortcut is to replace the inner
expectation by evaluation at the mean:

    H̃_cond(x*) = E_{x∼p(x)}[ H(R | x*, λ(x)=μ_λ(x), D) ].              (eq. 20)

**Claim (Proposition).** eq. 20 is **biased**:
E_{λ(x)}[H(R|x*,λ(x),D)] ≠ H(R|x*,λ(x)=μ_λ(x),D) in general.

**Derivation of the bias.** The predictive mean of λ(x*) given λ(x) is linear in
λ(x):

    E[λ(x*) | λ(x), D] = μ_λ(x*) + α·(λ(x) − μ_λ(x)),
    α := Cov(λ(x*), λ(x) | D) / σ²_λ(x)    (regression coefficient).

The predictive **variance** does not depend on the value of λ(x) (only the mean
shifts). But entropy H is **nonlinear** in the predictive mean. Taylor-expand H
around λ(x) = μ_λ(x) and take E over λ(x) ∼ N(μ_λ(x), σ²_λ(x)) (the first-order
term vanishes, E[λ(x)−μ_λ(x)] = 0):

    E_{λ(x)}[H] ≈ H(μ_λ(x*), σ²_{*|x}) + ½·α²·σ²_λ(x)·∂²H/∂μ_{*|x}².

**The σ²_λ(x) cancels.** With ρ the correlation, Cov(λ(x*),λ(x)) = ρ·σ_λ(x*)·σ_λ(x),
so α = ρ·σ_λ(x*)/σ_λ(x) and α²·σ²_λ(x) = ρ²·σ²_λ(x*). Hence

    Bias ≈ (ρ²/2)·σ²_λ(x*)·∂²H/∂μ_{*|x}².                              (eq. 21)

**When the bias vanishes** (only): ρ = 0 (x uninformative about x* — but then the
utility contribution is ~0 anyway), or σ²_λ(x*) = 0 (nothing to learn), or
∂²H/∂μ² = 0 (entropy linear in the mean — false for Poisson-lognormal). Notably
σ²_λ(x) = 0 is **not** a zero-bias condition — it cancels (Remark).

**The bias is largest exactly when it matters.** Bias ∝ ρ²·σ²_λ(x*): maximal when
the sampled image x is most correlated with x* (most relevant to the utility) and
when x* is most uncertain (where the acquisition should be guiding us). A
systematic bias does not shrink with more MC samples — it converges to the WRONG
value and corrupts the optimizer's direction. Sampling λ is unbiased; its variance
falls as O(1/N). Hence **sample λ**.

**One λ per image is optimal (nested MC).** With g(x,λ) = H(R|x*,λ,D), define
τ² = Var_x[E_λ g] (between-image) and E_x[σ²_x] (within-image). For a fixed budget
of M entropy evaluations, using N_λ samples per image gives
Var = (N_λ·τ² + E_x[σ²_x]) / M — strictly worse for N_λ > 1. So Strategy A
(N_x = M images, one λ each) is optimal; the implementation's single λ per xᵢ is
correct.

**Practical magnitude — the A² knob (flagged "review carefully" in the source).**
The added "Practical Relevance" section (self-flagged as possibly containing
errors) argues the bias scales as A²: since Δg(x*) = A·α·(λ(x)−μ_λ(x)),
∂²H/∂μ_{*|x}² = A²·∂²H/∂g², so Bias ≈ (ρ²/2)·σ²_λ(x*)·A²·∂²H/∂g². For a fitted PNAS
cell (A ≈ 0.36) the empirical worst-case (ρ=1) bias was ~1.1e-4 vs an MC standard
error ~3.0e-3 at N=1000 — i.e. ~30× smaller than MC noise; you'd need N ≈ 750,000
for the stochastic estimator's variance to fall below the deterministic bias. So
for small A the bias is negligible and sampling merely adds honest noise; the
regime guide: A ≪ 1 negligible, A ∼ 1 moderate, A ≫ 1 essential. **In this
codebase A is typically small** (§9: A ≈ 0.0265 for one fitted cell), so in
practice the deterministic and stochastic DA utilities nearly coincide — but the
*correct* estimator remains the sampled one. (Also: with small A, the dominant MC
noise is which images xᵢ are drawn, not the λ draw.)

**`DEBUG_FIX_LAMBDA_i` / sample_lambda=False.** Fixing λ(x)=μ_λ(x) computes
U_debug = H_marg − H(R|x*,λ(x)=μ_λ(x),D), the deterministic DA of §9. It is
guaranteed non-negative (with the corrected c-term of §5) but estimates a
*different quantity* than the true information gain.

---

## 8. Code realization details, conventions, and gotchas

### 8.1 The A/λ₀ transform inconsistency (documented silent-bug source)
Source: `.claude/rules/acquisition.md` "Critical: A/lambda0 transform inconsistency".

The imported helpers disagree on whether they take RAW λ moments or already
transformed g = A·λ + λ₀ moments:

| function | input convention |
|---|---|
| `compute_H(mu, sigma2, a, lambda0)` | RAW GP λ moments; transforms internally |
| `nd_utility_new(mu, sigma2)` | LOG-FIRING-RATE g moments (pre-transformed) |
| `_diff_laplace_log_probs(mu, sigma2, r)` | g moments |

So `distribution_aware_utility` passes RAW moments + A, λ₀ to `compute_H`, while
`standard_utility` transforms manually (mu_g = A·λ + λ₀, sigma2_g = A²·σ²) before
calling `nd_utility_new`. Both deliver the same math to the Laplace pipeline, but
mixing the conventions up is a silent bug. Keep the mapping explicit.

### 8.2 r truncation off-by-one
`compute_H` sums r = arange(0, r_max) (0..r_max−1); `nd_utility_new` sums
r = arange(0, r_max+1) (0..r_max). Both are finite truncations of eq. 6; the
one-bin difference is immaterial for low firing rates but is a real inconsistency
between the two entry points — do not treat r_max as bit-identical across them.

### 8.3 Adaptive r_max (truncation guard)
`compute_adaptive_rmax(mu_g, sigma2_g, safety_k, max_rmax, min_rmax)`:

    upper_logf = μ_g + safety_k·√σ²_g            (3σ upper tail of g, default k=3)
    upper_rate = exp(max upper_logf)             (clamped at exp(80) to avoid overflow)
    needed     = int(upper_rate + 5·√upper_rate + 10)   (+5 Poisson SD margin)
    return max(needed, min_rmax);  raise if needed > max_rmax

It **raises** rather than silently truncating when the required r_max would exceed
max_rmax (default 10000) — an aggressive-truncation guard: a wrong entropy would
otherwise pass unnoticed. This is what keeps H from collapsing when μ_g is high
(§9). Non-adaptive callers pass a fixed r_max (production default r_max = 100).

### 8.4 Gradient compatibility
All production entropy/utility functions are differentiable w.r.t. x_candidates:
the Laplace path uses `torch.where` (not indexed assignment, which breaks
autograd), and the Lambert-W has a proper custom backward (dW/dy = W/(1+W)). The
caller decides whether to track gradients (pass requires_grad tensors) or suppress
(wrap in `torch.no_grad()`), which is what enables gradient-ascent x*
optimization — and what exposes the divergence of §9.

### 8.5 `compute_H_MC` — analysis only, hard-coded values, RAISE TO USER
`compute_H_MC` (MC entropy: sample g∼N, r∼Poisson(eᵍ), evaluate Laplace log p(r),
clip −log p at max_log_contrib=50, average) exists **only for pedagogy/analysis**.
It is **not differentiable** (discrete Poisson sampling) and **biased** (clipping),
so it is NOT a production acquisition. The source carries an in-code warning:
"THIS FUNCTION IS USING HARD CODED VALUES. RAISE TO USER IMMEDIATELY." Honor it.

---

## 9. Entropy landscape — shape of utility over image space, and its limits

Sources: `entropy_landscape.md`, `da_utility_theory.md`.

**Monotone shape.** The response entropy H(R | μ_g, σ²_g) increases monotonically
with **both** μ_g (more predicted firing) and σ²_g (more GP/epistemic uncertainty).
So a candidate image scores high on utility for a mixture of two reasons — it is
predicted to fire a lot AND it is epistemically uncertain. Utility is therefore
**not** a pure epistemic-uncertainty acquisition; predicted firing rate is baked
in. (Whether this is desirable is a modeling choice — see the normalized-kernel
trade-off below.)

**Truncation artifact.** With a fixed r_max = 100, the Laplace sum misses the
probability mass above r_max once the Poisson peak climbs past it, so H **collapses
to ~0** for μ_g ≳ log(100) ≈ 4.6 (and at lower μ_g for large σ²_g). This is a
numerical artifact, not real entropy. The computable region is roughly triangular
in (μ_g, σ²_g). The adaptive r_max (§8.3) removes the collapse (zero monotonicity
violations confirmed). A pre-check without running Laplace:
z_safe = (log r_max − μ_g)/√σ²_g; z_safe > 2 safe, < 1.5 unreliable, < 0.5 broken.
**Natural images from the pool sit at z_safe ≫ 10 — always safe**; trouble only
arises with artificially amplified images (norm scaling c ≫ 1).

**Divergence under free-space gradient ascent (DA utility; sibling owns proofs).**
The arc-cosine kernel is positively 1-homogeneous: K(c·x, y) = c·K(x, y) when the
bias σ₀² = 0. Scaling x* → c·x* scales μ(x*) ∼ c and σ²(x*) ∼ c², so H_marg grows,
while for a directionally aligned conditioning image ρ ≈ 1 keeps σ²_cond ≈ 0. So
U_DA grows ~ c^1.9 (numerically) — gradient ascent "diverges" by **amplifying image
norm** rather than finding angularly informative images. This is intrinsic to the
DA formula with homogeneous kernels, **not a bug**; the f_max firing-rate guard
(default 100) bounds the exploitation. Over the finite natural-image POOL
(production selection over discrete candidates) norms are bounded and this never
fires — the divergence is a free-space-optimization phenomenon. (Full theorems:
sibling note / `proof_divergence_theorems.tex`, `proof_kernel_solutions.tex`.)

**Deterministic DA special case (sample_lambda=False)** — the cleanest lens on
what conditioning does. With λ_t = μ(x_t) the innovation (λ(x)−μ_x) = 0, so
μ_cond = μ_* (mean unchanged) and σ²_cond = Σ_**·(1 − ρ²), where
ρ² = Σ_x*² / (Σ_xx·Σ_**). In g-space:

    U_DA(x*) = H(μ_g, σ²_g) − H(μ_g, σ²_g·(1 − ρ²)).                   (eq. 22)

DA utility then depends on exactly three quantities: μ_g (operating point on the
entropy landscape), σ²_g (marginal variance / how much entropy to start with), and
ρ² (fraction of that variance conditioning removes). Conditioning moves
(μ_g, σ²_g) → (μ_g, σ²_g(1−ρ²)): same mean, reduced variance; utility is the
entropy difference. ρ² is governed by the **angle** between x* and the conditioning
image in C-space (small angle → ρ² ≈ 1 → near-total variance removal; large angle
→ ρ² ≈ 0 → no conditioning). This ρ² is the same ρ² that sets the sampling bias of
eq. 21 (α²σ²_λ(x) = ρ²σ²_λ(x*)).

**Normalized-kernel trade-off (why we keep the norm dependence).** A normalized
arc-cosine kernel forces K(x,x) = 1, killing the norm channel so utility tracks
only angular/RF alignment (pure epistemic). But test_r drops ~25% (0.79 → 0.59,
PNAS cell 8, M=100): image norm is genuinely informative for neural encoding. So
the production kernel keeps the norm dependence and accepts that utility mixes
predicted firing with epistemic uncertainty.

**Representative fitted-model scale** (seed 42, cell 8, M=50, n_train=50):
A ≈ 0.0265 (very small — compresses GP moments, so the λ-sampling bias of §7 is
negligible here), λ₀ ≈ −1.686, σ₀ ≈ 0.989, K(x,x) ≈ 669 for a typical natural
image; natural-image logf_mean ∈ [−1.8, −1.3], logf_var ∈ [0.04, 0.19].

---

## 10. metrics.py — evaluation metrics (NOT acquisition; adjacent, for orientation)

`metrics.py` holds post-hoc **evaluation** metrics, not the acquisition objective;
included so the two are not confused.
- `compute_pearson_correlation`, `compute_r_squared`, `compute_spearman_correlation`
  — standard.
- `compute_explained_variance(r_test, f_pred)` → correlation ratio
  accuracy/reliability, where reliability = |corr(r_even, r_odd)| (even/odd split
  of repetitions) and accuracy = ½(corr(f_pred, r_even) + corr(f_pred, r_odd)).
  **NOT** the paper's adjusted R².
- `compute_adjusted_r_squared(r_test, f_pred)` → Goldin et al. 2023 PNAS Eq. 5:
  adjusted R² = mean_accuracy² / reliability (Spearman-Brown correction; divides by
  reliability, i.e. by (√reliability)²). Relationship: adjusted_r2 =
  explained_var²·reliability.

These score how well the fitted tuning predicts held-out responses; the utility of
§§3–4 scores how much a *new* image would improve that tuning.

---

## 11. Conflicts, redundancies, and deprecated-flags (ground-truth reconciliation)

1. **Doc-vs-code — production utility.** The `.tex` summaries derive the
   **distribution-aware** utility (U = H_marg − H_cond over p(x)) as "the"
   acquisition function. The **production** loop uses **`standard_utility`**
   (H_marg − E[H_noise], no p(x)) — `run_active_loop.py:241`. DA utility appears
   only under `investigations/`. Both are "information gain = entropy reduction,"
   but about different targets (x* itself vs natural images). Do not conflate.

2. **Doc-redundancy — two near-identical "corrected" tex.**
   `distribution_aware_utility_pietro.tex` and `conditional_entropy_corrected.tex`
   share the title "Corrected Derivation of the Distribution-Aware Acquisition
   Function" and the same eq. 1–10. Differences: the pietro version adds an
   eigenspace-projection section (subspace machinery — sibling's domain) and
   contains a **typo** in its σ²_cond equation (the second line of eq. (219–220)
   drops an "=" / minus, reading "...) k(x*,x*) − ..."); the corrected version adds
   the explicit V_cond rank-1 update (eq. 19) and the "Summary of Correction."
   Treat `1D_conditional_entropy_derivation.tex` as the cleanest, unambiguous
   statement of eq. 14–18.

3. **Doc-vs-doc — the "correction" narrative & code status.** The correction (add
   the prior conditional covariance c, eq. 15) fixes `active_learning_pietro.tex`
   Eq. 217, which used cross-covariance u*ᵀ V u only. **The current code
   (`get_gp_conditional_moments`) is not affected** — it reads `cross_cov` from the
   true joint `covariance_matrix`, which already equals Σ_x* = c + uᵀ V u*. The
   correction is a caution about hand-assembled cross-covariances (and about the
   deferred vargp_direct augmented-matrix path), not a live bug.

4. **Deprecated — `lambda_moments`.** Not in current `gpytorch_porting/utils.py`;
   it is a legacy varGP routine (repo-root deprecated `utils.py` L3937-3952,
   `deprecated/vargp_style_estep.py`). Superseded by `get_gp_marginal_moments` /
   `get_gp_conditional_moments`. (§6.)

5. **Code-vs-code — A/λ₀ convention split** between `compute_H` (raw λ) and
   `nd_utility_new` (g). (§8.1.) And the r_max off-by-one between them (§8.2).

6. **Doc self-flag — `why_sample_lambda.tex` §"Practical Relevance"** is marked by
   its own author as possibly containing errors ("review carefully before being
   relied upon"). The A²-scaling conclusion and the 30×-smaller-than-MC-noise
   numbers are plausibility arguments, not settled results. The core
   bias-does-not-vanish-with-N argument (§7, eq. 21) is solid; the practical
   magnitude section is provisional.

7. **Hard-coded-values flag — `compute_H_MC`** carries an in-code
   "RAISE TO USER IMMEDIATELY" warning about hard-coded values; analysis-only,
   non-differentiable, biased. (§8.5.)

---

## 12. Notation table (with clashes/synonyms flagged)

| symbol | meaning | source / code |
|---|---|---|
| x, x* | natural stimulus (image); candidate/query image | all |
| xᵢ, x_sample | a natural image drawn from p(x) for the MC conditioning | tex / `distribution_aware_utility` |
| p(x) | natural-image distribution (what "distribution-aware" is aware of) | tex, da_utility_theory |
| r, R | spike count observation (Poisson) | all |
| **λ(x)** | **GP latent function** (sets firing rate). CLASH: also appears as offset λ₀ | tex; code `lambda_mean/var`, `lambda_i` |
| **λ₀** | fixed firing-rate **offset** (link g = A·λ + λ₀). SYNONYM: `lambda0`, `likelihood.lambda0` | tex, code |
| λ̃ (lambda-tilde) | inducing values (latent at inducing points) | tex |
| A | fixed firing-rate **scaling** in the link; `likelihood.A`. Small in practice (≈0.03) | tex, code |
| g(x) | **log-firing rate** g = A·λ + λ₀, f = eᵍ | tex, code (`mu_g`, `sigma2_g`) |
| f(x) | mean firing rate = exp(A·λ + λ₀) | tex |
| **μ, σ²** | CLASH: (a) in `compute_H`/tex eq. 8 = RAW λ posterior moments; (b) inside Laplace = g moments. Always check which space | code, tex |
| μ_g, σ²_g | **log-firing-rate** posterior moments = A·μ_λ+λ₀, A²·σ²_λ | code, tex |
| μ_x, μ_* ; Σ_xx, Σ_** | marginal λ posterior mean/var at image x and at x* | 1D tex eq. 14 |
| Σ_x* | posterior **cross-covariance** of λ(x), λ(x*) = c + uᵀ V u* | 1D tex eq. 16; code `cross_cov` |
| μ_cond, σ²_cond | **conditional** λ-moments at x* given λ(x) | tex eq. 17-18; code `mu_cond/sigma2_cond` |
| u, u* | projection vectors K⁻¹k(x), K⁻¹k(x*) | tex |
| s, s* | Schur complements = **prior** conditional variances | tex |
| **c** | **prior conditional covariance** k(x,x*)−uᵀK u* (the "correction" term). SYNONYM: α in the 1D augmented-matrix appendix | tex eq. 15 |
| m, V | variational posterior mean/cov of inducing values, q(λ̃)=N(m,V) | tex |
| V_cond | rank-1-corrected inducing-point posterior cov (eq. 19) | corrected tex |
| K, k(x), Kᵦ, B | inducing kernel matrix; kernel vector; eigenvalues; eigenbasis (**subspace = sibling note**) | tex |
| **ρ, ρ²** | correlation between λ(x) and λ(x*); ρ² = Σ_x*²/(Σ_xx Σ_**). Sets both variance-reduction (eq. 22) AND sampling bias (eq. 21) | da_utility_theory, why_sample_lambda |
| α | regression coeff Cov/Var = ρ·σ_*/σ_x (why_sample_lambda). CLASH: = c in 1D augmented-matrix appendix | why_sample_lambda / 1D tex |
| **U, U(x*)** | **utility / acquisition value**. CLASH by variant: U_std (production) vs U_DA (research) | all |
| I(·;·) | mutual information (the utility IS a conditional MI) | tex |
| **H** | **entropy**. CLASH across THREE roles: H_marg (marginal response entropy), H_cond (DA second term, over p(x)), E[H_noise] (aleatoric Poisson noise entropy, the STANDARD-utility second term). H_cond ≠ E[H_noise] — different second terms | all |
| ḡ_r | Laplace mode of the g-integrand (eq. 10, Lambert-W closed form) | tex, `_diff_argmax_g` |
| W₀ | principal branch Lambert W; code computes W₀(eʸ) via `_lambertw0_log` | tex, code |
| r_max | Laplace-sum truncation (prod default 100; adaptive guard to ≤10000) | code |
| z_safe | (log r_max − μ_g)/√σ²_g, Laplace-validity pre-check | da_utility_theory |
| N_mc, N | number of MC image samples for H_cond | code, tex |
| sample_lambda | flag: draw λ_i∼N(μ_i,σ²_i) (True, correct) vs λ_i=μ_i (False, deterministic) | code |

---

## 13. Pointers (sibling-owned; do not re-derive here)

- Subspace/eigenbasis realization (K = B Kᵦ Bᵀ, block-inverse K_{b,x}⁻¹, projected
  u_b, updates without reconstructing the full matrix): `…pietro.tex`
  §"Computation in the Eigenspace"; 1D tex §"GPyTorch Quantities"; sibling note.
- Divergence theorems / kernel homogeneity / scaling corollaries / normalized- &
  arc-sine-kernel analysis: `proof_divergence_theorems.tex`,
  `proof_kernel_solutions.tex`, `proof_moments_and_conditioning.tex`; sibling note.
- Numerical stabilization of the Laplace/MC entropy (r_max convergence tests,
  sum(p_r) noise, clipping bias): `entropy_landscape.md`; sibling note.
- Companion "Predictive distribution conditioned on observation" derivation
  (full V_cond integration): referenced by the corrected tex.
