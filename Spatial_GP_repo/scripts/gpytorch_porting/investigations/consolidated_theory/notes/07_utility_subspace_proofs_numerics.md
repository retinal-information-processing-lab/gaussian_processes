# 07 — Utility: Subspace Theory, the Three Proofs, Predictive Conditioning, and Numerical Analysis

**Cluster:** the *deep proof / subspace / numerical layer beneath the utility.*
This file supplies the theorems, proofs, and numerical analysis that sit
**underneath** the utility definitions. The companion file
`06_utility_core.md` owns the top layer — the two utilities, the
mutual-information derivation, the marginal/conditional-entropy definitions,
and the code mapping. Here we prove *why* those formulas behave as they do:
the subspace validity argument, the norm-scaling divergence theorems, the
kernel fixes, the exact predictive-conditioning derivation, and the
four-failure-mode numerical audit.

**Math notation.** Plain-text / unicode throughout (this is a `.md` reference,
not a compiled `.tex`). Multi-line derivations are in fenced blocks.

**Ground truth = current code.** All code claims here were checked against the
live local engine at
`gpytorch_porting/utils.py` and `gpytorch_porting/acquisition.py`
(§8 records the exact line numbers and every doc-vs-code drift found).

---

## 0. Orientation — the two utilities and where this file's proofs attach

There are **two** acquisition utilities in play. `06_utility_core.md §0–§4`
defines both; recalled here only enough to attach the proofs.

- **Standard / production utility** (mutual information between spike count and
  its own latent rate), computed by `utils.py:nd_utility_new`:

  ```
  U(x*) = H_marg(x*) − E_g[H(R | g)]  =  I(R ; g | x*, D)
  ```

  The subtracted term is the **expected Poisson noise entropy**
  E_g[H(R|g)] (written `E_H_noise` in code). No second image is involved.
  → §6 (numerics) is entirely about this object.

- **Distribution-aware (DA) utility**, conditioned on a hypothetical
  observation of the latent at a *target* image x_t:

  ```
  U_DA(x*) = H_marg(x*) − H_cond(x* | λ(x_t))
  ```

  The subtracted term is the entropy of r(x*) **after Gaussian-conditioning**
  the GP on an observed λ(x_t). → §1 gives the moments, §2 gives the exact
  predictive object being conditioned, §4–§5 prove how U_DA behaves under
  image scaling.

**Notation clash to keep straight (flagged for the pool):** the symbol "the
term after the minus sign" is **not** the same across the two utilities.
Standard utility subtracts E_g[H(R|g)] (an average over the *own* latent);
DA utility subtracts H_cond (entropy after conditioning on *another* point
x_t). They coincide only in the ρ²=1 deterministic limit, where DA's H_cond
collapses to a single Poisson entropy H_Poisson(exp(μ_g)) (§4). Do not fuse
them.

The proofs in §4–§5 are all about the DA utility diverging under
*unconstrained gradient ascent on the image x\**. That is a
**super-stimulus / image-optimization** setting (find an x* maximizing
utility), distinct from the production loop, which selects the best image
from a **fixed pool** via `standard_utility`. The subspace machinery (§3) and
the kernel fixes (§5) are proposed remedies for that gradient-ascent
divergence.

---

## 1. GP moments and Gaussian conditioning — Proof III
### Source: `proof_moments_and_conditioning.tex`

This is the moments reference that grounds `get_gp_marginal_moments` and
`get_gp_conditional_moments`. (The sibling file states these formulas and
maps them to code in `06 §5–§6`; here they are the foundation for §2, §4, §5.)

### 1.1 Kernel (arc-cosine, order 1)

```
K(x, y) = (1/π) · ‖x‖_C · ‖y‖_C · J(θ)

‖x‖_C   = sqrt( xᵀ C x + σ₀² )          (C-norm, "amplitude")
cos θ   = ( xᵀ C y + σ₀² ) / (‖x‖_C ‖y‖_C)   ("angle" between x and y)
J(θ)    = sin θ + (π − θ) cos θ          (angular factor, monotone ↓ on [0,π])
```

**Self-kernel.** Set y = x: cos θ = ‖x‖_C² / ‖x‖_C² = 1 ⟹ θ = 0, and
J(0) = sin 0 + π cos 0 = π. Hence

```
K(x, x) = (1/π) · ‖x‖_C · ‖x‖_C · π = ‖x‖_C².
```

The **prior variance at any point equals its squared C-norm.** This single
identity is the seed of the entire norm-scaling divergence (§4–§5).

### 1.2 Variational GP posterior

M inducing points {z̃_j}, variational posterior q(λ̃) = N(m, V). Shorthands:

```
K̃      = K(Z̃, Z̃)                      (M×M inducing-point kernel matrix)
k(x)    = [K(x, z̃₁), …, K(x, z̃_M)]ᵀ    (M-vector, cross-kernel to inducing pts)
W       = K̃⁻¹ (V − K̃) K̃⁻¹             ("correction" matrix, FIXED after training)
```

**Marginal moments at x\*** (grounds `get_gp_marginal_moments`, code returns
`posterior.mean, posterior.variance`):

```
μ(x*)   = k(x*)ᵀ K̃⁻¹ m
σ²(x*)  = K(x*,x*) + k(x*)ᵀ W k(x*)
        = ‖x*‖_C²   + k(x*)ᵀ W k(x*)
```

First term = prior variance; second = posterior correction (negative when
V ≺ K̃, which shrinks variance).

**Posterior covariance between x\* and x_t:**

```
Σ_q(x*, x_t) = K(x*, x_t) + k(x*)ᵀ W k(x_t)
```

with Σ_q(x*,x*) = σ²(x*) and Σ_q(x_t,x_t) = σ²(x_t).

**Posterior correlation:**

```
ρ²(x*, x_t) = Σ_q(x*, x_t)² / ( σ²(x*) σ²(x_t) )        ∈ [0, 1]
```

the squared Pearson correlation of the joint Gaussian [λ(x*), λ(x_t)] under
the variational posterior — how much knowing λ(x_t) tells you about λ(x*).

### 1.3 Conditioning on an observed λ_t (grounds `get_gp_conditional_moments`)

Standard Gaussian conditioning on observing λ_t at x_t:

```
μ_cond(x*)  = μ(x*) + [ Σ_q(x*, x_t) / σ²(x_t) ] · ( λ_t − μ(x_t) )
σ²_cond(x*) = σ²(x*) − Σ_q(x*, x_t)² / σ²(x_t)
            = σ²(x*) · ( 1 − ρ² )                        ← key simplification
```

So **ρ² is directly the fractional variance reduction** from conditioning:
ρ²=0.9 removes 90% of the variance at x*. The code
(`utils.py:get_gp_conditional_moments`, lines 360–364) computes exactly

```
innovation  = lambda_sample − mu_sample
mu_cond     = mu_star + cross_cov * (innovation / var_sample)
sigma2_cond = var_star − cross_cov² / var_sample     (then clamp ≥ 1e-8)
```

i.e. it takes the **full joint covariance** of [x_sample ; x_star] from
`model(all_x).covariance_matrix` and applies Method-1 joint conditioning
(§2.5) — the exact route, not the sparse approximation.

### 1.4 Log-firing-rate transform

Observation model r ~ Poisson(exp(g)), g = A·λ + λ₀. GP moments in g-space:

```
μ_g            = A μ + λ₀,            σ²_g            = A² σ²        (marginal)
μ_{g,cond}     = A μ_cond + λ₀,       σ²_{g,cond}     = A² σ²_cond   (conditional)
```

Variance reduction carries through: σ²_{g,cond} = σ²_g (1 − ρ²).

### 1.5 DA utility in these terms

```
U_DA(x*) = H(μ_g, σ²_g) − H(μ_{g,cond}, σ²_{g,cond})
```

where H(μ_g, σ²_g) = −Σ_r p(r) log p(r) is the Poisson–log-normal mixture
entropy (Laplace-approximated; see §6). In **deterministic mode** (λ_t = μ(x_t),
`sample_lambda=False`) the conditional mean is unchanged, μ_{g,cond} = μ_g, so

```
U_DA(x*) = H(μ_g, σ²_g) − H(μ_g, σ²_g (1 − ρ²))       (boxed result)
```

The DA utility then depends on exactly **three** scalars:
1. **μ_g = A μ(x*) + λ₀** — the operating point (where on the H landscape;
   fixed only in the deterministic treatment),
2. **σ²_g = A² σ²(x*)** — the marginal variance (how much entropy to begin with),
3. **ρ²** — how much of that variance conditioning removes (horizontal shift on
   the landscape).

Conditioning moves the point (μ_g, σ²_g) → (μ_g, σ²_g(1−ρ²)): same μ_g, reduced
variance; the utility is the H-gap between them.

### 1.6 How scaling affects ρ² (bridge to §4)

Conditioning on x_t, evaluate at x* = c·x_t (same image, scaled by c > 0):

- **Exact (σ₀² = 0).** Scaling preserves the angle to every point z:
  ```
  cos θ(c x_t, z) = (c x_tᵀ C z) / sqrt(c² q · v_z)
                  = (x_tᵀ C z) / sqrt(q · v_z) = cos θ(x_t, z),   q ≜ x_tᵀ C x_t
  ```
  J(θ) unchanged and ‖c x_t‖_C = c‖x_t‖_C ⟹ k(c x_t) = c·k(x_t). Substituting:
  μ(c x_t)=c μ(x_t), σ²(c x_t)=c² σ²(x_t), Σ_q(c x_t, x_t)=c σ²(x_t). Therefore
  ```
  ρ² = (c σ²(x_t))² / (c² σ²(x_t) · σ²(x_t)) = 1     exactly, for any c.
  ```
  **But the moments still grow** (μ_g ∝ c, σ²_g ∝ c² ⟹ H_marg → ∞), so even
  though conditioning removes 100% of the variance, **U_DA still → ∞.**
- **σ₀² > 0 (trained model).** The angle is no longer exactly preserved; σ₀²
  adds a constant offset to numerator and denominator of cos θ. Fractional error
  is O(σ₀² / ‖x_t‖_C²), does not vanish with c. Net: ρ² → ρ²_∞ < 1 as c → ∞
  (a fixed *fraction* removed, not all). This does **not** prevent divergence:
  σ²_g(1 − ρ²_∞) is a fixed fraction of σ²_g ∝ c², H grows ~linearly in σ²_g, so
  ```
  U_DA = H(μ_g,σ²_g) − H(μ_g, σ²_g(1−ρ²_∞)) ∝ ρ²_∞ · σ²_g ∝ c².
  ```
- **Two different images (not parallel).** ρ² depends on the angle θ(x*, x_t).
  At large angles (θ → π/2), Σ_q(x*,x_t) → 0, so ρ² → 0 and conditioning has no
  effect.

### 1.7 Normalized kernel — what it fixes and where it fails (preview of §5.1)

The divergence comes from the prior variance K(x,x) = ‖x‖_C² growing unbounded.
Normalizing,

```
K̄(x,y) = K(x,y) / sqrt( K(x,x) K(y,y) )
        = [ (1/π) ‖x‖_C ‖y‖_C J(θ) ] / ( ‖x‖_C ‖y‖_C ) = J(θ)/π
```

the norms cancel exactly; K̄ depends only on θ. **Fixes:** K̄(x,x)=J(0)/π=1 for
all x, cross-kernel entries K̄(x,z̃_j)=J(θ_j)/π ∈ [0,1] bounded ⟹ σ̄²(x) bounded
regardless of ‖x‖_C ⟹ U_DA cannot diverge by scaling; the only way up is to
improve ρ² (find small angle to x_t; with σ₀²>0 minimized at c=1, in both
direction and magnitude). **Fails:** K̄ discards all norm information, but image
norm carries genuine neural-encoding signal (neurons respond differently to
high/low contrast). Empirically, training with K̄ drops test correlation by
**~25%** — the norm encodes real structure the GP needs.

---

## 2. Exact predictive distribution conditioned on a new observation — focus (c)
### Source: `predictive_distribution_conditioned_on_observation.tex`

**This is the object whose entropy the DA utility integrates.** The DA utility
needs p(λ(x*) | λ(x), D): the predictive over the latent at a query x* after a
**hypothetical** new latent observation λ(x). This section derives it two ways
(sparse-approximated and exact/augmented) and proves the two agree in mean. It
is the rigorous derivation beneath the conditional moments of §1.3 and beneath
`06 §5`.

### 2.1 The predictive integral

Goal: p(λ(x*) | λ(x), D). Marginalize the inducing points λ̃:

```
p(λ(x*) | λ(x), D) = ∫ p(λ(x*) | λ(x), λ̃) · p(λ̃ | λ(x), D) dλ̃
```

**Sparse GP assumption (conditional independence):** given λ̃, the prediction at
x* is independent of the sample λ(x) and data D, so the first factor collapses
to the conditional prior:

```
p(λ(x*) | λ(x), λ̃) ≈ p(λ(x*) | λ̃)

⟹  p(λ(x*) | λ(x), D) = ∫ p(λ(x*) | λ̃)          · p(λ̃ | λ(x), D)          dλ̃
                          └ Conditional Prior ┘     └ Updated Posterior ┘
```

### 2.2 The updated posterior p(λ̃ | λ(x), D)

Build the joint of the sample λ(x) and λ̃ under the variational approximation.
Given
- Variational posterior: p(λ̃ | D) = N(m, V)
- Conditional likelihood: p(λ(x) | λ̃) = N(uᵀ λ̃, s), with
  ```
  u = K⁻¹ k(x),      s = k(x,x) − k(x)ᵀ K⁻¹ k(x)
  ```
the joint is Gaussian:

```
[ λ(x) ]      ( [ uᵀ m ]   [ s + uᵀV u    uᵀV ] )
[ λ̃   ] ~ N ( [  m   ] ,  [   V u          V  ] )
```

Conditioning on the observed λ(x) gives p(λ̃ | λ(x), D) = N(m′, V′) with

```
m′ = m + [ V u / (s + uᵀV u) ] · ( λ(x) − uᵀ m )
V′ = V − ( V u uᵀ V ) / ( s + uᵀV u )
```

### 2.3 Predictive moments (sparse route)

Conditional-prior moments at x* given λ̃: E[λ(x*)|λ̃] = u*ᵀ λ̃, Var[λ(x*)|λ̃] = s*,
with u* = K⁻¹ k(x*), s* = k(x*,x*) − k(x*)ᵀ K⁻¹ k(x*). Writing
λ(x*)|λ̃ = u*ᵀ λ̃ + η√s* and averaging over the conditioned posterior:

```
μ_pred = u*ᵀ m′
       = u*ᵀ ( m + [ V u / (s + uᵀV u) ] ( λ(x) − uᵀ m ) )       (boxed)

σ²_pred = s* + Var_λ̃[ u*ᵀ λ̃ ]  = s* + u*ᵀ V′ u*         (law of total variance)
        = k(x*,x*) − u*ᵀ (K − V′) u*                             (boxed)
```

### 2.4 Exact (augmented) route — removes the sparse artifact

Keep the direct λ(x)–λ(x*) correlation: do **not** approximate the conditional
prior. Augmented latent λ_aug = [λ(x) ; λ̃], augmented kernel

```
K̃_x = [ k(x,x)   k(x)ᵀ ]        k_aug(x*) = [ k(x, x*) ]
       [ k(x)     K     ]                    [ k(x*)    ]
```

Non-approximated conditional prior p(λ(x*) | λ_aug) = N(u_augᵀ λ_aug, S*) with

```
u_aug = K̃_x⁻¹ k_aug(x*)
S*    = k(x*,x*) − k_aug(x*)ᵀ K̃_x⁻¹ k_aug(x*) = k(x*,x*) − u_augᵀ K̃_x u_aug
```

Since λ(x) is **observed** (variance 0) while λ̃ follows the updated posterior,
the augmented posterior is

```
p(λ_aug | λ(x), D) ~ N( [λ(x) ; m′], V″ ),    V″ = [ 0   0ᵀ ]
                                                    [ 0   V′ ]
```

Predictive moments:

```
μ_pred  = u_augᵀ [ λ(x) ; m′ ]
σ²_pred = S* + u_augᵀ V″ u_aug = k(x*,x*) − u_augᵀ ( K̃_x − V″ ) u_aug     (boxed)
```

**Key property:** when x = x*, u_aug aligns with the first column of K̃_x, and
because the (1,1) block of V″ is 0, the variance is forced to **0** — resolving
the sparse approximation's spurious nonzero self-variance. The exact route is
what a *self-consistent* conditional predictive requires.

### 2.5 Equivalence of predictive means (theorem + full proof)

**Claim.** The augmented-integration mean equals the mean from standard Gaussian
conditioning on the joint predictive [λ(x), λ(x*)] — **exactly**, no
cross-covariance approximation.

**Method 1 (joint conditioning).** Joint moments given D:

```
E[λ(x)]=μ_x=uᵀ m,   E[λ(x*)]=μ_{x*}=u*ᵀ m
Σ₁₁ = s + uᵀ V u                                   (var at x)
Σ₁₂ = (k(x,x*) − uᵀ K u*) + u*ᵀ V u                (cross-cov: residual s_cross + inducing)
      └──── residual s_cross ────┘   └ inducing ┘
```

Standard Gaussian identity:

```
E[λ(x*) | λ(x), D] = μ_{x*} + (Σ₁₂/Σ₁₁)(λ(x) − μ_x)
                   = u*ᵀ m + (Σ₁₂/Σ₁₁)(λ(x) − uᵀ m)        (target)
```

(Aside from the .tex: s_cross = 0 would enforce the sparse factorization
p(λ*,λ|λ̃) ≈ p(λ*|λ̃)p(λ|λ̃); the inducing/epistemic cross-cov can still be ≠ 0
because a change of belief about an inducing point moves both λ* and λ.)

**Method 2 (augmented integration).** μ_aug = u_augᵀ [λ(x) ; m′]. Block-invert
K̃_x; with residual correlation coefficient α = s⁻¹ s_cross,

```
u_aug = [ α ; u* − α u ],       α = (k(x,x*) − uᵀ K u*) / s
```

Expand:

```
μ_aug = α λ(x) + (u*ᵀ − α uᵀ) m′,   m′ = m + (V u / Σ₁₁)(λ(x) − uᵀ m)
```

Group by prior mean u*ᵀ m and innovation (λ(x) − uᵀ m); add and subtract
α uᵀ m:

```
μ_aug = α(λ(x) − uᵀ m) + u*ᵀ m + [ (u*ᵀ − α uᵀ) V u / Σ₁₁ ] (λ(x) − uᵀ m)
```

Factor the innovation; its coefficient C is

```
C = α + ( u*ᵀ V u − α uᵀ V u ) / Σ₁₁
```

Common denominator Σ₁₁ = s + uᵀ V u:

```
C = ( α s + α uᵀV u + u*ᵀV u − α uᵀV u ) / Σ₁₁ = ( α s + u*ᵀ V u ) / Σ₁₁
```

Substitute α s = s_cross = k(x,x*) − uᵀ K u*:

```
C = ( s_cross + u*ᵀ V u ) / Σ₁₁ = Σ₁₂ / Σ₁₁       (numerator is exactly Σ₁₂)
```

Therefore

```
μ_aug = u*ᵀ m + (Σ₁₂/Σ₁₁)(λ(x) − uᵀ m)   ≡   Method 1.      ∎
```

**Consequence.** The production conditional-moment code (§1.3,
`get_gp_conditional_moments`) uses Method-1 joint conditioning on the full
covariance — which this theorem certifies is the exact augmented predictive
mean, so the code is *not* incurring the sparse cross-covariance error.

---

## 3. Subspace theory and operations — focus (a)
### Sources: `subspace_theory.md`, `subspace_operations.md`

**Question answered:** how (and why validly/efficiently) the utility can be
optimized in a **low-dimensional subspace** of image space instead of full
pixel space (11664 dims for the full image; ~900–2356 for the RF-masked
region). Setting: optimize an image x* to maximize DA utility U(x*) by gradient
ascent, but constrain x* to a K-dim subspace x_rf = μ + basis·z, optimizing over
z. Three bases; all share one optimizer/plotting/diagnostics.

> **⚠ REMIND-USER flag carried verbatim from `subspace_operations.md` (top):**
> the **c_eigenspace mode has a non-zero offset that is NOT mathematically
> motivated** — it is forced by *limiter precision*: the non-capturable
> eigenvectors collapse to 0 instead of to the mean, so the code compensates by
> setting offset = mean. **This should be investigated.**

### 3.1 Setup and the C matrix

d pixels in the masked RF region. The kernel's structured PSD matrix:

```
C = diag(α) · C_smooth · diag(α)
C_smooth[i,j] = exp( −ρ ‖ξ_i − ξ_j‖² )      (spatial pixel correlations)
α_i           = exp( −β ‖ξ_i − ξ_0‖² )      (locality mask about RF center ξ_0)
```

Dataset of N natural images with empirical mean μ and covariance
Σ_data = (1/N) X_centeredᵀ X_centered. Goal: x* maximizing U(x*) while resembling
a natural image.

### 3.2 The three bases (math)

`z_to_image(z, basis, offset) = offset + basis·z`;
`image_to_z(x, basis, offset) = basisᵀ(x_rf − offset)`. All use
**offset = μ_rf** (dataset mean over RF pixels), so z=0 maps to the mean image.

**(1) PCA (`--method pca`).** x_rf = μ + V_K z, V_K = top-K eigenvectors of
Σ_data (Σ_data = V Λ Vᵀ, λ₁ ≥ λ₂ ≥ …).
- Offset μ is **mathematically intrinsic** (centering before SVD).
- Truncation is a **genuine constraint**: excludes low-data-variance
  (unnatural) directions. K set by `--var-threshold` (variance fraction) or
  `--n-components`. At var_threshold 0.80: K=197 of 2356 RF pixels.
- Gradient projects: ∇_z U = V_Kᵀ ∇_x U.
- **What PCA captures = second-order statistics only** (mean + covariance). Blind
  to phase correlations (edges/contours), sparsity, non-Gaussian marginals.
  N(μ, Σ_data) has the same PCA as the true image distribution but samples look
  like correlated noise (right power spectrum, random phases).

**(2) C-eigenspace (`--method c_eigen`).** x_rf = μ + U_K z, U_K = top-K
eigenvectors of the kernel's C matrix (C = U Γ Uᵀ, γ₁ ≥ … ≥ γ_d ≥ 0).
- Offset μ is **NOT intrinsic** — the C-eigendecomposition passes through the
  origin. μ is a practical choice (keeps reconstructions in pixel bounds, gives
  the kernel a realistic operating point). It **changes the utility landscape**
  (kernel sees C(μ + U_K z), not C(U_K z)). ← the ⚠ flag above.
- Truncation is a **reparameterization** at full rank (no information loss); at
  reduced rank it drops near-zero-gradient directions. K by `--eigen-threshold`
  (relative to max eigenvalue, default 1e-3). At 1e-3: K=28 of 2356, capturing
  99.7% of eigenvalue mass. Spectrum extremely skewed (top eigenvalue ~44% of
  total).
- **Float32 noise floor:** eigenvalues below ~1e-7·max are numerical noise;
  thresholds below ~1e-6 pull in garbage eigenvectors and degrade
  reconstruction. Default 1e-3 is safely above.

**(3) Combined (`--method combined`).** x_rf = μ + (V_K W) z, project C into PCA
space then eigendecompose:

```
C_pca = V_Kᵀ C V_K   (K_pca × K_pca);   C_pca = W Γ_W Wᵀ;   keep W above eigen-threshold
basis = V_K · W_retained
```

Directions that are BOTH natural (PCA) AND kernel-visible (C-eigen). Two
thresholds (`--var-threshold`, `--eigen-threshold`). Offset μ from the PCA step
(intrinsic). At (0.80, 1e-3): 2356 → K_pca=197 → K_combined=28. **Combined
achieves near-identical utility to PCA with ~7× fewer dims** (28 vs 197).

### 3.3 WHY the subspace is valid and efficient (the load-bearing argument)

**C-eigenspace validity (the core reason optimizing in a subspace is sound).**
Every kernel evaluation touches x only through the quadratic forms xᵀ C x and
xᵀ C x′. In the C-eigenbasis, xᵀ C x = Σ_i γ_i (u_iᵀ x)². Directions with γ_i ≈ 0
contribute ≈ 0 to *every* kernel value. Moreover the gradient of a kernel
evaluation carries a factor of C:

```
∇_{x*} k(x*, z_m) = −k(x*, z_m) · C · (x* − z_m) / ℓ²   (RBF example)
```

so the component along eigenvector u_i is scaled by γ_i. **Standard gradient
ascent therefore already concentrates movement along the top C-eigenvectors**;
explicit C-eigenspace projection just makes this filtering explicit. Removing
tiny-γ directions (a) does not change the solution (they had near-zero gradient)
and (b) removes flat directions, giving a **better-conditioned, lower-effective-
dimension** optimization. → C-eigenspace = **convergence aid / exact
reparameterization**, not a naturalness constraint. (Analogous to the EIGVAL_TOL
threshold on K̃ in the direct-VGP eigenspace path.)

**PCA validity.** Truncation restricts to directions where natural images vary,
introducing information the GP does not have — a *data-driven subspace
constraint*.

**PCA ≈ Fourier under stationarity (Szegő).** A 2nd-order-stationary process on a
regular grid has Toeplitz (2D: block-Toeplitz) covariance Σ[i,j] = f(|i−j|). By
Szegő's theorem, as d → ∞ the eigenvectors converge to Fourier modes and the
eigenvalues to the power spectral density S(ω) = Σ_k f(k) e^{−iωk}. Natural
images have ~stationary 2nd-order statistics with S(ω) ~ 1/|ω|² (1/f² law), so
Σ_data's eigenvectors ≈ Fourier basis, eigenvalues ≈ 1/k² ordering, and
**PCA-truncation ≈ keeping the K lowest-frequency Fourier modes.** Caveats
weaken the equivalence: irregular masked region (Fourier needs regular grids),
non-stationarity from the α mask (RF-center pixels have higher variance —
breaks Toeplitz), finite N ≈ 10⁴ in d ≈ 900, and residual non-stationarity of
natural images. At 30×30 patches, approximate stationarity is reasonable.

### 3.4 Subspace ≠ distributional constraint (the honest limitation)

All three bases are **subspace constraints, not distributional constraints.**

| Property | PCA (Σ_data) | C-eigenspace | Fourier |
|---|---|---|---|
| Basis source | Data covariance | Kernel C matrix | Fixed sinusoids |
| Data-dependent? | Yes | No (model) | No |
| 2nd-order stats? | Yes | No (kernel geometry) | Yes |
| Higher-order stats? | No | No | No |
| Constrains to natural dist? | No | No | No |
| Handles irregular mask? | Yes | Yes | Poorly |
| Role in optimization | Subspace constraint | Convergence aid | Subspace constraint |

None guarantee x* is a plausible sample of p(x): an image with the right power
spectrum but wrong phase (correlated noise) satisfies all three. A true
distributional constraint needs higher-order structure (phases, edges, sparsity)
— generative models (diffusion / GAN / normalizing flows). The subspace approach
is a useful **diagnostic** (does restricting to the right power spectrum change
utility? does the GP utility operate primarily at 2nd order?), not a full fix.

### 3.5 Optimization operations (for completeness, not theory)

LBFGS with strong-Wolfe line search; start at the dataset mean projected into the
subspace; **fully unconstrained within the subspace** (no norm/sigmoid/clipping —
OOB pixels flagged visually only); early stop after 5 flat steps; **f_max guard**
rejects an LBFGS step whose predicted firing rate exceeds f_max (100).
Multi-conditioning (`--n-cond N>1`) conditions the DA utility on N random pool
images with chunked gradient accumulation. Key empirical notes: RBF is
well-behaved (K(x,x)=1, no norm explosion); **arc-cosine causes norm-driven
utility explosion (K(x,x) ~ ‖x‖²)** — the theme §4–§5 prove. Subspace projection
can produce OOB pixels (linear projection preserves L2 norm, not element-wise
bounds — expected, not a bug; μ-centering reduces it). An untested **input-warping
idea** (per-pixel soft-clip w(·) applied *before* C, so the kernel is blind to
OOB values) is recorded as a separate future investigation, not implemented.

---

## 4. Divergence theorems — focus (b), Proof 1
### Source: `proof_divergence_theorems.tex`

**Theorem statement (informal).** For the order-1 arc-cosine kernel, scaling an
image x* = c·x_t along a fixed direction inflates the GP posterior variance as c²
while preserving the conditioning benefit, so unconstrained gradient ascent on
the DA utility diverges. Made precise below.

### 4.1 Theorem 1 — Cross-kernel proportionality (σ₀² = 0)

**Statement.** Let σ₀² = 0 and x* = c·x_t, c > 0. Then for any z:
K(x*, z) = c·K(x_t, z), and hence k(x*) = c·k(x_t).

**Proof.** With σ₀²=0, v_{x*} = c²(x_tᵀ C x_t) = c² q (q ≜ x_tᵀ C x_t), v_{x_t}=q.
For any z with v_z = zᵀ C z:

```
cos θ(x*,z) = (c x_t)ᵀ C z / sqrt(c² q · v_z) = c x_tᵀ C z / (c sqrt(q v_z))
            = x_tᵀ C z / sqrt(q v_z) = cos θ(x_t, z).
```

So θ(x*,z)=θ(x_t,z) ⟹ J(θ(x*,z))=J(θ(x_t,z)). Then

```
K(x*,z) = (1/π) sqrt(c² q · v_z) J(θ(x_t,z)) = c · (1/π) sqrt(q v_z) J(θ) = c K(x_t,z).
```

Apply to each inducing point z = z̃_j ⟹ k(x*) = c·k(x_t). ∎

### 4.2 Corollary 1 — Perfect posterior correlation

**Statement.** Under Theorem 1, ρ² ≜ Σ_q(x*,x_t)² / (Σ_q(x*,x*)Σ_q(x_t,x_t)) = 1.

**Proof.** Let W = K̃⁻¹(V−K̃)K̃⁻¹. From Theorem 1, k(x*)=c k(x_t), and
K(x*,x_t) = c K(x_t,x_t) = c v_t (θ(x*,x_t)=0 when x* ∥ x_t ⟹ J(0)=π,
K(x_t,x_t)=v_t). Substituting into Σ_q:

```
Σ_q(x*,x_t) = c v_t + c k(x_t)ᵀ W k(x_t) = c · Σ_q(x_t,x_t)
Σ_q(x*,x*)  = c² v_t + c² k(x_t)ᵀ W k(x_t) = c² · Σ_q(x_t,x_t)
⟹ ρ² = [c Σ_q(x_t,x_t)]² / (c² Σ_q(x_t,x_t) · Σ_q(x_t,x_t)) = 1.  ∎
```

### 4.3 Corollary 2 — Scaling of posterior moments

**Statement.** Under Theorem 1: μ(c x_t) = c·μ(x_t), σ²(c x_t) = c²·σ²(x_t).
**Proof.** Direct substitution of k(x*)=c k(x_t) into μ = k ᵀ K̃⁻¹ m and
σ² = K(x*,x*) + kᵀ W k. ∎

### 4.4 Remark — effect of σ₀² > 0

Results hold only approximately. Exact expressions:

```
v_{c x_t}      = c² q + σ₀²  ≠  c²(q + σ₀²) = c² v_t
cos θ(c x_t,z) = (c x_tᵀ C z + σ₀²) / sqrt((c² q + σ₀²) v_z)  ≠  cos θ(x_t,z)
```

The **fractional error** |K(cx,z) − c K(x,z)| / |c K(x,z)| converges to a
**constant O(σ₀²/v_t)** as c → ∞, *not* to zero (σ₀² shifts the angle by an amount
∝ σ₀²/v_t, independent of c). In the trained model v_t = q + σ₀² ≈ 669,
σ₀² ≈ 0.98 ⟹ σ₀²/v_t ≈ 0.0015. Numerically (verify_conclusions.py, test C7): mean
fractional error in K(c x_t, z_j) → ≈ 0.6% for large c (slightly above σ₀²/v_t
since both magnitude and angle corrections contribute); ρ² → ≈ 0.987 (test C3),
residual variance ratio ≈ 1.3%. Conditioning stays very effective, not perfect.

### 4.5 Consequences for conditioning

From Corollary 1, ρ²=1 ⟹ σ²_cond(x*) = σ²(x*)(1−ρ²) = 0. Deterministic mode
(λ_t = μ(x_t)) with Σ_q(x*,x_t)=c σ²(x_t):

```
μ_cond(x*) = c μ(x_t) + c(μ(x_t) − μ(x_t)) = c μ(x_t) = μ(x*).
```

So in the exact parallel case (θ=0, σ₀²=0): μ_cond = μ, σ²_cond = 0, and H_cond
collapses to a single Poisson entropy H_Poisson(exp(μ_g)), μ_g ≜ A μ(x*) + λ₀.
The DA utility becomes

```
U_DA(c x_t) = H_marg(μ_g, σ²_g) − H_Poisson(exp(μ_g)),   μ_g = A c μ(x_t)+λ₀,  σ²_g = A² c² σ²(x_t).
```

### 4.6 Proposition 1 — Gaussian observation model gives log(c) growth

**Statement.** If observations were Gaussian, r(x) ~ N(λ(x), τ²) with fixed τ²,
then for ρ=1, σ²=c²σ²(x_t), the DA utility grows as log(c).
**Proof.**

```
H_marg = ½ log(2πe(σ² + τ²)),   H_cond = ½ log(2πe(σ²_cond + τ²)) = ½ log(2πe τ²)  [σ²_cond=0]
⟹ U = ½ log(1 + σ²/τ²).   With σ²=c²σ²(x_t), large c:  U ≈ ½ log(c² σ²(x_t)/τ²) = log(c) + const.  ∎
```

**Remark.** Unbounded but slow; dU/dc ~ 1/c → gradient ascent slows but never
stops. Norm scaling is **not** unique to Poisson — the distinction is the *rate*.

### 4.7 Poisson-GP growth — heuristic lower bound, numerically α ≈ 1.9 (not proved)

No closed form for H_marg(μ_g, σ²_g) in the Poisson-GP case (Laplace-approximated,
§6). Heuristic: for large σ²_g, p(r) is a Poisson mixture over log-normal rates
f = e^g, g ~ N(μ_g, σ²_g). Claim H_marg ≥ H_Poisson(E[f]) (plausible: Poisson
mixtures are overdispersed, "more spread"; a rigorous proof is **not** given — and
note the Poisson does *not* maximize entropy on ℕ₀ under a mean constraint alone;
the geometric does. Poisson is max-entropy only under Var=mean, so the naive
max-entropy argument does not directly apply). With E[f]=e^{μ_g+σ²_g/2} and
large-λ Poisson entropy H_Poisson(λ) ≈ ½ log(2πe λ):

```
H_marg ≥ ½ log(2πe) + ½(μ_g + σ²_g/2),   dominant term ¼ σ²_g = ¼ A² c² σ²(x_t).
```

H_cond = H_Poisson(exp(μ_g)) depends only on μ_g (grows as c, not c²). Therefore

```
U_DA = H_marg − H_cond  ≳  ¼ σ²_g  ∝  c²      (faster than the Gaussian log c).
```

**Numerical calibration** (verify_conclusions.py, test C4): fitting U_DA ~ c^α over
the Laplace-valid range c ∈ {2,5,10} (all Σp(r) > 0.95) gives **α ≈ 1.9**,
consistent with c². **Caveat:** Laplace truncates at r_max=100 and breaks once
σ²_g is large enough to put mass at r > r_max. Pre-check via the **Laplace safety
score** z_safe = (log r_max − μ_g)/√σ²_g; z_safe < 2 ⟹ unreliable entropy.
In-model, z_safe < 2 at c ≈ 10 (μ_g=0.4, σ²_g=5.0), limiting the verifiable range;
beyond it, c² growth is predicted by the bound but not computationally confirmed
without raising r_max.

### 4.8 Numerical evidence (trained model: seed 42, cell 8, M=50, n_train=50; A=0.0265, λ₀=−1.686)

| Image | μ | σ² | μ_g | σ²_g | angle(rad) | notes |
|---|---|---|---|---|---|---|
| x_t (target) | 7.86 | 70.9 | −1.48 | 0.050 | 0 | natural |
| x*_DA | 101 | 3 240 | 0.99 | 2.28 | 0.114 | DA-optimized |
| x*_Std | 68 497 | 2.85e8 | 1816 | 2.0e5 | 0.866 | Std-optimized |
| natural | [−6,16] | [57,267] | [−1.8,−1.3] | [0.04,0.19] | [0.6,0.7] | pool |

| Image | Σp(r) | H_marg | H_cond | U_DA | var.ratio |
|---|---|---|---|---|---|
| x_t | 1.000 | 0.593 | 0.583 | 0.010 | 0.000003 |
| x*_DA | 0.984 | 2.836 | 2.036 | 0.800 | 0.060 |
| x*_Std | 7e-6 | 9e-5 | 9e-5 | ~0 (U_std=∞) | 0.9999 |
| natural | ~1.0 | [0.49,0.69] | [0.49,0.69] | [0.0001,0.006] | [0.82,1.0] |

Observations: (1) **DA-optimized (θ=0.114)** after 5000 steps has μ(x*)=101
(~13× target; C-norm ratio ≈ 7.8), variance ratio 0.060 ⟹ ρ²≈0.94, μ_g=0.99 in
the useful band because A=0.0265 compresses the raw GP mean; U_DA=0.800 is
**genuine** (real loophole, not a numerical artifact). (2) **Std-optimized
(θ=0.866)** diverged after 1479 steps (U=∞ from overflow), μ_g=1816 ⟹ Laplace
catastrophically fails (Σp(r)=7e-6); `nd_utility_new`'s exp(μ_g+σ²_g/2) overflows
float32 — and the large angle shows Std, without conditioning to preserve
alignment, diverged in a different direction. (3) **Natural images (θ≈0.65)**:
weak conditioning (var ratio > 0.82) ⟹ tiny U_DA < 0.006. (4) **DA collapse
threshold:** μ_g > 10 needs μ(x*) > (10−λ₀)/A ≈ 441, i.e. c > 441/μ(x_t) ≈ 56 —
at μ=101 there is still ~23% headroom.

### 4.9 Summary of proof 1

Root cause: the arc-cosine kernel is **positively homogeneous**,
K(cx,y)=c K(x,y) when σ₀²=0, decoupling norm from direction in the GP posterior.
Gradient ascent exploits the norm degree of freedom to raise variance (hence
entropy and utility) without sacrificing the conditioning benefit from
directional alignment. Any practical gradient-based image optimization with this
kernel needs a **norm constraint** or a **normalized kernel** K̄ = K/√(K(x,x)K(x′,x′))
that depends only on angle. Proved: Theorem 1, Corollaries 1–2, Proposition 1.
Numerically confirmed: α≈1.9, genuine U_DA=0.800, and (test C8) that utility
growth with norm is strongly **angle-dependent** — effective at small angles
(θ < 0.2, strong conditioning), negligible at large angles (θ > 0.6).

---

## 5. Kernel solutions — focus (b), Proof 2
### Source: `proof_kernel_solutions.tex`

Independent-review restatement of the divergence mechanism + two kernel-level
fixes. Complements §4 (which quantifies the divergence) with the *remedies*.

### 5.1 The divergence mechanism (restated) and the σ₀² alignment peak

Cosine similarity of the scaled candidate x* = c x_t (q = x_tᵀ C x_t):

```
cos θ(c) = (c q + σ₀²) / sqrt( (c² q + σ₀²)(q + σ₀²) )
```

- σ₀² = 0 ⟹ cos θ(c) = 1 for all c (scaling does not affect alignment).
- σ₀² > 0 ⟹ cos θ(c) is **strictly maximized at c = 1**.

**Implication:** the "correlation gradient" points to c=1 — σ₀² breaks scale
invariance, so the kernel *wants* to match the target in pattern **and**
intensity. But kernel magnitude scales quadratically: K(x*,x*) = c² q + σ₀² = O(c²).

### 5.2 Utility decomposition and the optimization conflict

For Poisson with link f = exp(λ), entropy scales with the log-variance of the
firing rate. Decompose:

```
U(x*) ≈  ½ σ²_GP(x*)        +   I(ρ²)
         └ Variance Term ┘      └ Correlation Term ┘  (maximized when ρ² → 1)
```

- Correlation term: maximized at c=1 (bounded, from σ₀²).
- Variance term: O(c²), **unbounded**.

Since c² grows without bound while the correlation term is bounded, ∇_x U is
dominated by the variance term for large c: the optimizer sacrifices the small
misalignment penalty to reap unbounded variance reward (c → ∞). **To fix, bound
the variance term.**

### 5.3 Solution 1 — Normalized arc-cosine kernel (project onto the unit sphere)

**Definition.** K̄(x,x′) = K(x,x′) / √(K(x,x) K(x′,x′)). For order-1 arc-cosine,
K(x,x)=v_x=xᵀ C x + σ₀², so

```
K̄(x,x′) = [ (1/π) √(v_x v_{x′}) J(θ) ] / ( √v_x √v_{x′} ) = J(θ)/π.
```

Removes the magnitude terms; depends only on the angle θ.

**Effect on posterior variance.** σ̄²(x) = K̄(x,x) + k̄(x)ᵀ K̃⁻¹(V−K̃)K̃⁻¹ k̄(x).
- **Prior variance** K̄(x,x) = J(0)/π = 1, constant for all x regardless of norm.
- **Cross terms** k̄(x)_j = K̄(x, z̃_j) are correlation coefficients, |·| ≤ 1.
- **Boundedness** with fixed variational W ⟹ the whole posterior-correction term
  is bounded.

**Sub-theorem — invariance of kernel magnitude under scaling.** For x* = c x,
c > 0: K̄(x*,x*) = K(cx,cx)/√(K(cx,cx)²) = 1 = K̄(x,x). So the prior variance of K̄
is **strictly constant, independent of ‖x‖** — unlike K(cx,cx) ≈ c² K(x,x). No
utility gain from signal amplification. ∎

**Resulting gradient dynamics.** With σ̄² bounded, U can no longer increase by
inflating ‖x‖; optimization relies entirely on the correlation term, and since
σ₀² > 0 is retained inside θ, the angle is minimized only when x → x_t in both
direction and magnitude ⟹ the gradient converges to x_t. (Failure mode: as in
§1.7, discarding norm drops test-r ~25%.)

### 5.4 Solution 2 — Biologically plausible saturation kernel (arc-sin)

The arc-cosine kernel = infinite ReLU networks, which **do not saturate**; real
neurons have refractory periods and maximum firing rates. Model saturation with
the **arc-sin kernel** (step/erf nonlinearity, i.e. probit activation):

```
K_sat(x,x′) = σ_f² · (2/π) · arcsin(  (xᵀ C x′ + σ₀²)
                                      / sqrt( (1 + xᵀ C x + σ₀²)(1 + x′ᵀ C x′ + σ₀²) )  )
```

The extra "+1" in the denominator (generally a length-scale ℓ²) is characteristic
of the arc-sin kernel. **Saturation:** unlike K(x,x) ∝ ‖x‖²,

```
lim_{‖x‖→∞} K_sat(x,x) = σ_f² · (2/π) · arcsin(1) = σ_f².
```

The variance saturates to the constant σ_f². **Why it fixes divergence:** (1)
H_marg cannot grow indefinitely with norm; (2) to maximize utility the optimizer
must instead lower H_cond by finding x* highly correlated with x_t; (3) it does
this **without normalizing** the kernel — "super-stimuli" (infinite norm) yield
diminishing returns, naturally halting gradient ascent, embedding the physical
constraint that contrast beyond a point yields no more signal variance.

---

## 6. Utility numerical analysis — focus (d)
### Source: `utility_numerical_analysis.tex` (audit dated 2026-05-22)

**This section is entirely about the standard/production utility**
U = H_marg − E_g[H(R|g)] = I(R;g) (§0), computed by `nd_utility_new`. It
disentangles several distinct phenomena the older docs lumped under
"divergence." Short answer to "does exp(μ) still overflow?": **not where the
Laplace approximation is built** (fixed by log-space Lambert W). A *different*
exp(·) can still overflow in one exact analytical term — but in practice you lose
precision to **cancellation** and accuracy to **truncation** long before that.

### 6.1 The closed-form decomposition (Eq. 31/33 of the PNAS paper)

GP posterior λ(x*) | D ~ N(μ_λ, σ²_λ); g = A λ + λ₀ ⟹ g | D ~ N(μ_g, σ²_g); R | g
~ Poisson(e^g). Utility U = H_marg − E_g[H(R|g)] = I(R; g | x*, D).

**Marginal entropy** (integral has no closed form; Laplace-approximated, §6.2):

```
H_marg = −Σ_{r=0}^∞ p(r) log p(r),    p(r) = ∫ Poisson(r; e^g) N(g; μ_g, σ²_g) dg.
```

**Expected noise entropy.** Using H(Poisson(ν)) = ν − ν log ν + E_{R~Poisson(ν)} log R!
and taking expectations under g ~ N(μ_g, σ²_g), with the log-normal MGF
E[e^g] = e^{μ_g+σ²_g/2}, E[g e^g] = e^{μ_g+σ²_g/2}(μ_g+σ²_g):

```
E_g[H(R|g)] = −e^{μ_g + σ²_g/2} (μ_g + σ²_g − 1)  +  Σ_{r=0}^∞ p(r) log r!    (boxed)
              └────── exact, closed form (MGF) ──────┘   └── needs p(r), Laplace ──┘
```

**Ground-truth match:** `utils.py:nd_utility_new` line **720** (doc cited 719):

```python
E_H_noise = -torch.exp(mu_g + 0.5*sigma2_g) * (mu_g + sigma2_g - 1) + p_times_logr_sum
```

The first term is **exact**; only the second involves the truncated Laplace sum.
Both sums are truncated at the same finite r_max:

```
U ≈ −Σ_{r=0}^{r_max} p(r) log p(r) − ( −e^{μ_g+σ²_g/2}(μ_g+σ²_g−1) + Σ_{r=0}^{r_max} p(r) log r! ).
```

### 6.2 The Laplace approximation and the log-space Lambert W fix

Saddle-point of p(r) at the mode ḡ(r) of Poisson(r;e^g)·N(g;μ_g,σ²_g):

```
log p(r) ≈ r ḡ − e^{ḡ} − (ḡ − μ_g)²/(2σ²_g) − ½ log(1 + σ²_g e^{ḡ}) − log r!
```

Saddle equation r = e^{ḡ} + (ḡ − μ_g)/σ²_g, solved by

```
ḡ(r) = r σ²_g + μ_g − W₀( σ²_g · e^{ r σ²_g + μ_g } )     (W₀ = principal Lambert branch)
```

**The OLD bug** (legacy `utility.py:argmax_g_old`): formed σ²_g e^{rσ²_g+μ_g}
literally and passed it to a Lambert-W routine. In float32 this **overflows when
rσ²_g + μ_g ≳ 88** (e^{88} ~ 10³⁸) — very easily. The old code masked out
overflowing r, so the truncated Σp(r) could exceed 1 by large factors (docstring
reported up to ~108). **This** is the historic "exp(μ) diverges."

**The FIX (log-space Lambert W)** — current `utils.py:_diff_argmax_g` +
`_LambertWLogFunction` (verified present, lines 386–437). Never forms e^y. Set
y = log σ²_g + r σ²_g + μ_g (finite for finite inputs) and solve

```
w + log w = y    ⟺    w = W₀(e^y)
```

by Newton iterations directly on the log-form (code: fixed **10** iterations,
custom autograd; backward uses dW/dy = W/(1+W), the e^y cancels). Then
ḡ(r) = r σ²_g + μ_g − w. Since w ≈ y − log y for large y, ḡ(r) ≈ −log σ²_g + log y
~ log r for large r ⟹ e^{ḡ(r)} ≈ r (exactly right: Poisson likelihood is
maximized when rate = count) and **stays in float32 range for any r.** Hence
`exp_g_bar = torch.exp(g_bar)` (line **464**, doc cited 463) is **not** an
overflow risk post-fix.

### 6.3 Four independent residual failure modes

Distinct thresholds and remedies (they coincide only because all worsen at large
μ_g / σ²_g). Modes (a) and (b) are the pair older docs conflate.

**(a) Laplace approximation error in log p(r)** (§ code lines 464–469). Local
quadratic model of the integrand; inaccurate when the integrand departs from a
quadratic-log shape. Two regimes: σ²_g → 0 (prior degenerates, saddle equation
singular — code falls back to **exact Poisson** at σ²_g < 1e-6, line **477–478**:
log p = μ_g r − e^{μ_g} − log r!); σ²_g large (flat prior; asymmetric Poisson
factor skews the integrand, over/undershoots p(r) non-monotonically in r).
**Independent of r_max** — each p(r) has its own bias, observable as Σp(r) ≠ 1
even with no missing mass (reported up to 1.015; that is Laplace error, NOT
truncation).
- **Measured (2026-05-25, `laplace_pointwise_error.py` vs scipy.quad at fixed
  single r, no summing):** error < 0.01 nats (< 1%) for σ²_g ≲ 1 at every μ_g;
  grows to **5–20% at the peak by σ²_g ~ 50** for μ_g ≲ 5 (worst where the
  predictive peak slides onto r=0). At μ_g ≳ 6 the model predicts an impossible
  rate (e^{μ_g} ≳ 400 spikes) and the error blows up (≳ 100%, up to ~3 nats at
  μ_g=8) — but only at counts the cell can never produce (a sign of out-of-physical-
  range prediction, not a fixable error). Driver is **σ²_g, not μ_g**; float32 vs
  float64 shifts log p(r) by ≤ 3.7e-4 nats. **Physical-plausibility gate:** a
  retinal ganglion cell cannot fire more than ~100 spikes in the window, so the
  sensible regime is **σ²_g ≲ 6**; within it (a) is a few-percent effect, not a
  live concern. The GP occasionally wanders to σ²_g ~ 15 (rarely ~20) on
  poorly-constrained cells / early reps — explicitly out of scope.

**(b) Truncation of the sums at r_max** (§ `entropy_landscape.py`; code lines 715,
719). Both sums run to r_max−1; if the Poisson-log-normal predictive puts
non-negligible mass above r_max, the positive contributions −p(r)log p(r) and
p(r)log r! are clipped, biasing **both H_marg and the closed-sum term low.**
- Moments: **E[R] = e^{μ_g+σ²_g/2}** (log-normal mean); **Var(R) = e^{μ_g+σ²_g/2}
  (Poisson part) + e^{2μ_g+σ²_g}(e^{σ²_g}−1) (log-normal-of-rate part).** The
  second term dominates once σ²_g ≳ 1 (distribution very wide). As σ²_g grows the
  predictive is strongly right-skewed and its three centres separate: **mean**
  e^{μ_g+σ²_g/2} climbs, **median** stays ≈ e^{μ_g}, **mode** drifts *down* toward
  0 (≈ e^{μ_g−σ²_g}). So "p(r) peaks at r ≈ e^{μ_g}" is only the small-σ²_g limit;
  in the truncation-relevant regime the *peak* sits *lower* than e^{μ_g} even as
  the *mass* reaches high counts.
- **Sufficient benign-truncation condition:** r_max ≳ E[R] + k√Var(R).
- **Code `compute_adaptive_rmax`** (lines 483–531; doc cited 513–521) is stricter,
  a two-layer guard (verified, lines 514/521/522):
  ```
  r_max_adapt = e^{μ_g + k σ_g}  +  5 √( e^{μ_g + k σ_g} )  +  10,   k=3 (safety_k)
  ```
  i.e. a k-σ upper tail in g-space, exponentiated to a rate, plus 5·Poisson-std,
  plus 10; then clamped to [min_rmax, max_rmax] with a **ValueError raise** if
  needed > max_rmax (never silently over-truncates). Defaults: min_rmax **200**,
  max_rmax 10000, and a guard that raises if max_rmax > exp(80) so the overflow
  clamp can't hide failures. More conservative than the sufficient condition
  (converts the g upper tail to a rate *before* taking the Poisson std → worst
  case across supp(g), not typical).
- **Fixed default r_max = 100 (`config.py:234`)** is benign for confident
  predictions but **not safe in general**: fails whenever μ_g ≳ log 100 ≈ 4.6
  **or** σ²_g grows large (the log-normal-of-rate term spreads mass above 100 even
  at low μ_g) — exactly the high-uncertainty regime active learning explores.

**(c) Float32 overflow of the closed-form term** (code line **719/720**). The
exact term −e^{μ_g+σ²_g/2}(μ_g+σ²_g−1) overflows float32 when

```
μ_g + ½ σ²_g ≳ 88     (e^{88} ~ 10³⁸)
```

⟹ E_g[H(R|g)] = −∞ ⟹ U → +∞. **This is the only residual "exp can diverge" issue,
and it is in an exact analytical term, not the Laplace approximation** — distinct
from the historic bug (b/6.2). In practice truncation (b) bites long first (σ_g~1
⟹ truncation at μ_g~1 for r_max=100, vs μ_g~88 here).

**(d) Catastrophic cancellation** — two flavours; inherent to the decomposition
(survive even with exact Laplace and no truncation):
- **Inside E_g[H(R|g)] (large, earliest).** Both −e^{μ_g+σ²_g/2}(μ_g+σ²_g−1) and
  Σ p(r) log r! grow ~e^{μ_g}, but the true E_g[H(R|g)] grows only **linearly**
  (~½ μ_g + const via H(Poisson(ν)) ~ ½ log(2πeν)). At μ_g=10 each term ~2×10⁵
  vs a true value ~6 (ratio ~3×10⁴): ~4.5 leading digits cancel, ~2–3 survive in
  float32's ~7-digit mantissa. At μ_g=15 the terms ~5×10⁶ vs residual ~8: ~6.7
  digits cancel — essentially noise. Truncating the second sum makes this *worse*
  (closed-form term keeps full magnitude, sum is artificially smaller). (μ_g ≳ 6
  is already unphysical, so the μ_g∈[10,30] dominance is a formula property, not a
  live concern.)
- **Between H_marg and E_g[H(R|g)] (smaller, second-level).** By Jensen H_marg ≥
  E_g[H(R|g)] (equality iff σ²_g=0); U = I(R;g) is bounded (asymptotic Gaussian
  U ~ ½ log(1 + σ²_g e^{μ_g+σ²_g/2})) while both entropies grow linearly in μ_g —
  a second cancellation, costing fewer digits (both sides at the same reduced
  precision, common-mode errors cancel).
- **Unfixable without rewriting** the decomposition so cancelling pieces are
  computed together — e.g. summing −p(r)log p(r) − p(r)log r! as a *single* term
  keeps magnitudes commensurate before any large analytical exponential. Flagged,
  out of scope of the audit.

### 6.4 Step-by-step production-chain audit (each dangerous op flagged)

| Step / file:line (as of audit) | Operation | Risk |
|---|---|---|
| `acquisition.py:standard_utility` | μ_g=Aμ_λ+λ₀, σ²_g=A²σ²_λ | None numerically; kernel-driven blow-up of σ²_λ propagates here |
| `utils.py:compute_adaptive_rmax` | sets r_max | Raises before overflow if needed r_max > 10³⁴; no asymptotic fallback |
| `utils.py:_diff_argmax_g` | ḡ(r) via log-space Lambert W | **Safe by construction** (post-fix); no exp of large quantities |
| `_diff_laplace_log_probs:463/464` | exp_g_bar = exp(g_bar) | Safe: ḡ(r)~log r for large r ⟹ exp ḡ ~ r; no overflow in the sum |
| `_diff_laplace_log_probs:467/468` | log(1 + σ² e^{ḡ}) | Uses `log1p`; benign |
| `_diff_laplace_log_probs:464–469` | log p(r) ≈ … (Laplace) | **(a) Laplace error**; indep. of r_max; observable as Σp(r) ≠ 1 |
| `nd_utility_new:714/715` | Σ p(r) log p(r) | **(b) Truncation bias** if r_max too small; indep. of (a) |
| `nd_utility_new:718/719` | Σ p(r) log r! | Same truncation bias (2nd term of E_g[H(R|g)]) |
| `nd_utility_new:719/720` | −e^{μ_g+σ²_g/2}(μ_g+σ²_g−1) | **(c) float32 overflow** at μ_g+½σ²_g ≳ 88; **(d) cancellation** vs Σp(r)log r! (precision dies ~μ_g=15) |
| `nd_utility_new:721/722` | H_marg − E_g[H(R|g)] | Second-level **(d) cancellation** between two degraded quantities |

(Doc line numbers are from 2026-05-22; verified live values in the right slash of
each cell — see §8. `x/y` = doc-cited/actual.)

### 6.5 "Utility diverges under gradient ascent," re-examined

The claim is **two** distinct things, not one. (i) **Kernel-driven input
blow-up:** K(αx,αx)=α²K(x,x) ⟹ ascending U with no norm constraint sends
σ²_λ(x) ∝ ‖x‖² → ∞, so σ²_g → ∞ (this is §4–§5's mechanism). (ii) The **computed**
utility then misbehaves via the four numerical mechanisms — typically in order:
truncation collapse of H_marg (b) first, then Laplace error at large σ²_g (a),
then cancellation inside E_g[H(R|g)] (d) as μ_g grows, finally float32 overflow of
the closed-form term (c) at extreme scales. **The kernel makes the inputs
unbounded; then numerics make the computed utility ill-behaved — not the same
thing, and the four numerical effects are not the same thing either.** Numerical
fixes alone won't cure it: either constrain the optimizer (norm projection, or the
normalized `ArcCosineKernelNormalized`, §5.1) or rewrite the closed-form term in
an asymptotically stable form (combine the two matching exponentials before
evaluating).

### 6.6 What the older investigation gets right / where to push back

**Right:** the empirical truncation boundary in (μ_g, σ²_g) is a *curve* not a
line (the safe region depends on both moments via E[R], Var(R); a fixed-3σ-in-g
rule gives a straight line μ_g+3σ_g = log r_max that ignores the Poisson std);
Laplace-truncated H_marg collapses above the boundary; Monte-Carlo H_marg avoids
truncation at the cost of sample-tail-clipping bias. **Push back:** "Σp(r) can
exceed 1 (up to 1.015)" is normal saddle-point behaviour (p(r) is not a
normalized density), NOT a separate failure mode — conflates (a) with (b);
"exp(μ) overflow is happening inside the Laplace approximation" is **wrong**
post-fix (§6.2); the legacy `nd_utility_MC / _NUMERICAL / _hybrid` variants are
historical, not production; `compute_H_MC` (line 563/562) carries a **"HARD CODED
VALUES. RAISE TO USER IMMEDIATELY"** docstring warning — its clip thresholds are
not settled; "p(r) peaks at e^{μ_g}" is a small-σ²_g approximation only.

---

## 7. Notation table (symbol | meaning | source; clashes flagged)

| Symbol | Meaning | Source doc(s) |
|---|---|---|
| x, x*, x_t | candidate / query / target image (pixel vector) | all |
| C | structured PSD kernel matrix, C = diag(α) C_smooth diag(α) | subspace, proofs |
| C_smooth, α | spatial-correlation part; locality mask (α_i = e^{−β‖ξ_i−ξ_0‖²}) | subspace_theory |
| ‖x‖_C, v_x | C-norm sqrt(xᵀCx+σ₀²); v_x = xᵀCx+σ₀² (= ‖x‖_C²) | moments, divergence |
| σ₀² | kernel bias variance | proofs (all) |
| θ, J(θ) | angle between x,y; angular factor sin θ+(π−θ)cos θ | moments, proofs |
| q | xᵀ C x (quadratic form, σ₀²-free) | divergence, moments |
| K(x,y), K̄ | arc-cosine kernel; **normalized** kernel J(θ)/π | proofs |
| K_sat | arc-sin saturation kernel | kernel_solutions |
| **K̃ / K** | **inducing-point kernel matrix K(Z̃,Z̃)** — ⚠ see clash below | moments / predictive |
| k(x), u | cross-kernel M-vector; u = K⁻¹k(x) (projection vector) | moments / predictive |
| m, V | variational posterior mean, covariance N(m,V) at inducing pts | moments, predictive |
| W | K̃⁻¹(V−K̃)K̃⁻¹ (fixed correction matrix) | moments, kernel_sol |
| μ(x*), σ²(x*) | marginal GP posterior mean, variance (λ-space) | moments |
| Σ_q(x*,x_t) | posterior covariance between two points | moments, divergence |
| ρ² | squared posterior correlation ∈[0,1]; frac. variance reduction | moments, divergence |
| μ_cond, σ²_cond | conditional moments after observing λ_t; σ²_cond=σ²(1−ρ²) | moments |
| A, λ₀ | Poisson-link params: g = Aλ+λ₀ (log firing rate) | all |
| μ_g, σ²_g | log-firing-rate moments: Aμ+λ₀, A²σ² | moments, numerics |
| g, ḡ(r) | log-firing rate; Laplace saddle mode ḡ(r)=rσ²_g+μ_g−W₀(…) | numerics |
| f | firing rate f = e^g | numerics, divergence |
| r, R, r_max | spike count value; count RV; truncation cap | numerics |
| p(r) | Poisson–log-normal predictive (Laplace-approx) | numerics |
| H_marg | marginal entropy −Σ p(r)log p(r) (**shared by both utilities**) | all |
| E_g[H(R\|g)] | **standard** utility's subtracted term (Poisson noise entropy; `E_H_noise`) | numerics |
| H_cond | **DA** utility's subtracted term (entropy after conditioning on λ(x_t)) | moments, divergence |
| U, U_DA | standard utility (=I(R;g)) / distribution-aware utility | numerics / proofs |
| W₀ | principal branch of the Lambert W function | numerics |
| z_safe | Laplace safety score (log r_max − μ_g)/√σ²_g | divergence, numerics |
| **λ̃ / λ_aug** | inducing latents; augmented [λ(x);λ̃] | predictive |
| **m′, V′ / m″(V″)** | updated posterior after observing λ(x); augmented-covariance | predictive |
| u*, s* | u*=K⁻¹k(x*); s*=k(x*,x*)−k(x*)ᵀK⁻¹k(x*) (residual var) | predictive |
| α (predictive) | residual correlation coeff s_cross/s — ⚠ NOT the locality mask α | predictive |
| Σ₁₁, Σ₁₂ | joint predictive var at x; cross-cov (residual + inducing) | predictive |
| **z / z_k** | subspace coordinate vector; PCA score — ⚠ NOT an inducing point | subspace |
| V_K, U_K, W(comb) | PCA basis; C-eigen basis; C_pca eigenvectors | subspace |
| Γ, γ_i | C eigenvalues (C = U Γ Uᵀ) | subspace |
| Σ_data, Λ | dataset covariance; its eigenvalues | subspace |

**Flagged clashes / synonyms:**
- **K** — in `moments`/`divergence` `K(·,·)` is the *kernel function* AND K̃ the
  inducing kernel matrix; in `predictive_distribution…tex` the **bold K is the
  inducing-point kernel matrix** (this file's K̃) and `k(x)` the cross-kernel.
  Same object, different symbol (K̃ vs K). Mapping: predictive-doc **K = K̃**,
  **u = K̃⁻¹k(x)**, **W-of-moments** relates to V′ via the conditioning.
- **α** — TWO unrelated meanings: locality mask α_i (subspace) vs residual
  correlation coefficient α = s_cross/s (predictive §2.5). Disambiguate by source.
- **z** — inducing point z̃_j (moments/proofs) vs **subspace coordinate** z
  (subspace docs). The subspace z is the optimization variable, never an inducing
  point.
- **"the subtracted entropy term"** — E_g[H(R|g)] (standard) vs H_cond (DA). Both
  written near "H_marg − …"; they are different objects (§0). This is the single
  most important non-obvious clash for a pooled reader.
- **σ² overloading** — σ²(x*) is λ-space marginal variance; σ²_g = A²σ² is
  g-space; σ²_cond / σ²_{g,cond} are the conditioned versions; s, s* are
  *residual* (aleatoric) variances in the predictive derivation. Track the
  subscript/space.

---

## 8. Code-grounding map — which result grounds which code (verified live)

Checked against `gpytorch_porting/utils.py` and `acquisition.py` on read.

| Theory result (this file) | Grounds code | Status |
|---|---|---|
| §1.2 marginal moments μ,σ² | `utils.py:get_gp_marginal_moments` (L289) → `posterior.mean, .variance` | ✔ exists, matches |
| §1.3 / §4.5 conditioning μ_cond, σ²_cond=σ²(1−ρ²) | `utils.py:get_gp_conditional_moments` (L316; L360–364) | ✔ exact match to formulas |
| §2.5 mean-equivalence (Method-1 = augmented) | same fn uses **full joint covariance** (L350–351) ⟹ exact route | ✔ certifies no sparse error |
| §6.1 E_g[H(R\|g)] closed form | `nd_utility_new` **L720** `E_H_noise = -exp(mu_g+0.5*sigma2_g)*(mu_g+sigma2_g-1)+p_times_logr_sum` | ✔ matches (doc said L719) |
| §6.1 H_marg = −Σ p log p | `nd_utility_new` L715 | ✔ |
| §6.2 log-space Lambert W fix | `_LambertWLogFunction` (L386), `_diff_argmax_g` (L419) | ✔ exists; 10 Newton iters; dW/dy=W/(1+W) |
| §6.2 exp(ḡ) safe post-fix | `_diff_laplace_log_probs` L464 `exp_g_bar=exp(g_bar)` | ✔ (doc said L463) |
| §6.3(a) exact-Poisson small-σ² branch | `_diff_laplace_log_probs` L472–478 (σ²<1e-6) | ✔ (doc said L476) |
| §6.3(b) two-layer adaptive r_max | `compute_adaptive_rmax` L483–531 (formula L514/521/522) | ✔ (doc said L513–521) |
| §6.6 compute_H_MC hard-coded warning | `compute_H_MC` L563 | ✔ exists (doc said L562) |
| §1/§6 standard-utility entry | `acquisition.py:standard_utility` L41 | ✔ exists |

**Deprecated / not on active path (per task scope "current implementations
only"):** the divergence/numerics docs also reference legacy `utility.py`
(`argmax_g`, `argmax_g_old` with the `exp(rsigma2+mu).clamp(max=85)` kludge,
`nd_utility_MC/_NUMERICAL/_hybrid`). Those live in the **non-local legacy
`utility.py`**, not the active `gpytorch_porting` path; the active path uses the
differentiable ports `_diff_argmax_g` + `_LambertWLogFunction`. Keep the math,
ignore the legacy call sites.

---

## 9. Doc-vs-doc / doc-vs-code discrepancies and flags

1. **`lambda_moments` does NOT exist (confirmed).** Task-flagged and verified: no
   `def lambda_moments` in the local `utils.py`. It is a deprecated varGP routine.
   Current equivalents: **`get_gp_marginal_moments`** (marginal) and
   **`get_gp_conditional_moments`** (conditional). No source in *this* cluster
   references `lambda_moments`; the sibling `06 §6` documents the deprecation.

2. **`utility_numerical_analysis.tex` line numbers have drifted** (audit dated
   2026-05-22; code has since shifted a few lines). **Every formula matches; only
   the line citations are stale** (a W4-class staleness, not a substantive error):
   E_H_noise 719→**720**; exp_g_bar 463→**464**; small-σ² branch 476→**477–478**;
   `compute_adaptive_rmax` 513–521→**483–531** (inner formula 514/521/522);
   `compute_H_MC` 562→**563**. Recorded in §6.4/§8 as `cited/actual`.

3. **`min_rmax` floor (200) vs fixed `r_max` default (100).** Two distinct knobs:
   `compute_adaptive_rmax` floors at **min_rmax=200** (when adaptive_r_max=True),
   while the non-adaptive fixed default is **r_max=100** (`config.py:234`). Not a
   contradiction — different code paths. The doc discusses both; a careless reader
   could conflate them.

4. **Newton iteration count.** Doc says "a few Newton iterations"; code does a
   **fixed 10** (`_LambertWLogFunction.forward`, L399). Cosmetic.

5. **⚠ C-eigenspace non-zero offset (open, unresolved).** `subspace_operations.md`
   top banner: the c_eigenspace offset = mean is **not mathematically motivated**,
   forced by limiter precision (non-capturable eigenvectors collapse to 0 instead
   of mean). Flagged "should be investigated." Carried into §3 verbatim. This is
   an acknowledged modeling wrinkle, not a resolved result.

6. **Heavy redundancy across the three proof docs (not a conflict, but noted).**
   `proof_divergence_theorems.tex`, `proof_kernel_solutions.tex`, and
   `proof_moments_and_conditioning.tex` **all** independently state cross-kernel
   proportionality, ρ²=1 under scaling, and μ∝c/σ²∝c². They agree numerically and
   symbolically. This file preserves each faithfully (§1, §4, §5) but a pooled
   reader should know they are three treatments of one phenomenon, not three
   results. `proof_moments_and_conditioning` is the moments reference; `…divergence`
   is the theorem/proof + numerical evidence; `…kernel_solutions` is the
   mechanism + the two fixes.

7. **Two utilities share H_marg but differ in the subtracted term (§0 clash).**
   `utility_numerical_analysis.tex` is about U = H_marg − E_g[H(R|g)] (standard);
   the proof/subspace docs are about U_DA = H_marg − H_cond (DA). Not a
   contradiction — genuinely different utilities — but the shared "H_marg − …"
   shape invites conflation. They coincide only in the ρ²=1 deterministic limit
   (§4.5), where H_cond → a single Poisson entropy.

8. **Subspace status is exploratory.** `subspace_theory.md` is explicitly
   "ongoing investigation, no implementation yet"; `subspace_operations.md`
   documents a specific research script (`subspace_optimization.py`) on a side
   branch. The C-eigenspace *validity theory* (§3.3) is solid and load-bearing for
   "why utility can be computed/optimized in a subspace"; the *image-optimization
   application* is research, distinct from the production pool-selection loop.

---

## 10. Scope exclusions (per task constraints)

- **Spatiotemporal machinery: none present to exclude.** All seven sources are
  purely **spatial** GP (spatial C matrix, spatial RF, per-image spike counts). No
  temporal kernel / spatiotemporal factorization appears, so nothing was dropped
  on that axis. (Noted for completeness per the spatial-only mandate.)
- **Deprecated varGP baggage: excluded from the code-grounding, math kept.** The
  legacy `utility.py` call sites (`argmax_g_old`, `nd_utility_MC/_NUMERICAL/_hybrid`,
  the `clamp(max=85)` kludge) and the deprecated `lambda_moments` are named only to
  mark them **not on the active path**; the utility/entropy/subspace **math is kept
  in full** (task: "KEEP the utility subspace/entropy MATH").
- **Non-theory operational detail: trimmed.** Full CLI reference, file inventory,
  LBFGS hyperparameter table, continuation prompts, and the dependency table from
  `subspace_operations.md` are summarized to their load-bearing essentials (§3.5),
  not reproduced — they are operational, not proof/theory.
- **Input-warping idea:** recorded as a one-line pointer (§3.5); it is an untested
  future-investigation sketch, out of the current theory scope.
- **Sibling-owned (not re-derived here):** the mutual-information framing of the
  acquisition objective, the H_marg definition/derivation, the standard-vs-DA
  utility top-level exposition, `sample_lambda` bias/variance, and the code
  conventions (A/λ₀ transform, r off-by-one) live in **`06_utility_core.md`**
  (§1–§4, §5–§8). This file provides the proofs, subspace validity, exact
  predictive derivation, and numerical audit **beneath** them.

---

### Provenance
Sources distilled (read in full):
`investigations/utility/docs/subspace_theory.md`,
`investigations/utility/docs/subspace_operations.md`,
`investigations/utility/docs/proof_divergence_theorems.tex`,
`investigations/utility/docs/proof_kernel_solutions.tex`,
`investigations/utility/docs/proof_moments_and_conditioning.tex`,
`Papers/latex_summaries/utility_numerical_analysis.tex` (at
`/home/idv-eqs8-pza/IDV_code/Papers/latex_summaries/`),
`/home/idv-eqs8-pza/IDV_code/Papers/latex_summaries/predictive_distribution_conditioned_on_observation.tex`.
Code cross-checked live:
`gpytorch_porting/utils.py` (L289, 316, 386, 419, 440, 483, 534, 563, 689),
`gpytorch_porting/acquisition.py` (L41).
