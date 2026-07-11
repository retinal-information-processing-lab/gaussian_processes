# 05 — Arc-Cosine Kernel and the Structured Spatial Prior Covariance C

**Cluster owner note.** This file OWNS the definitions of (i) the arc-cosine kernel
`K(x, x')` and (ii) the structured spatial prior covariance `C` (parameters β, ρ, ξ₀)
for the whole consolidated-theory document. Other notes reference `C` only structurally
(as "the n_x × n_x spatial prior covariance"); the parameterization, constraints, and
gradients live here.

**Scope.** SPATIAL GP only. The `C` defined here is the *spatial* prior covariance. In a
full spatiotemporal model this would be the spatial factor of a Kronecker product
`C = C_spatial ⊗ C_temporal`; the temporal factor and Kronecker/warping machinery are
OUT of scope and do **not** appear in any of this cluster's sources (they are purely
spatial). The `Amp` amplitude parameter is treated as deprecated legacy-varGP baggage
and is EXCLUDED from the core math (see §7 for the code-vs-scope flag).

**Ground truth.** Current code `kernels.py` (the `gpytorch_porting` module). Where a
LaTeX summary or a prose note disagrees with the code, the code wins and the discrepancy
is flagged in §7.

**Sources.**
- `Papers/latex_summaries/acosker_kernel_def_and_gradients.tex` — kernel definition,
  structured prior C, hyperparameter gradients, ∇ₓ gradient.
- `Papers/latex_summaries/Acosker_gradient_dx.tex` — the ∇ₓK derivation (utility/guidance).
- `.../gpytorch_porting/investigations/archive/BETA_RHO_MATH_VERIFICATION.md` — β/ρ
  raw-vs-natural reconciliation.
- `.../gpytorch_porting/.claude/rules/jitter.md` — Cholesky jitter (skimmed).
- Code: `.../gpytorch_porting/kernels.py` (`ArcCosineKernel`).

---

## 1. The Arc-Cosine Kernel (order 1)

### 1.1 Definition

The kernel between two stimulus vectors `x, x' ∈ ℝ^{n_x}` (flattened natural images,
`n_x = n_px_side²`, e.g. 108² = 11664 px) is

```
K(x, x') = (1/π) · M · J(θ)
```

with the intermediate scalars

```
v_x    = xᵀ C x  + σ₀²          (self-term / prior variance of x)
v_x'   = x'ᵀ C x' + σ₀²         (self-term of x')
c_xx'  = xᵀ C x' + σ₀²          (cross-term)
M      = √(v_x · v_x')          (magnitude)
cos θ  = c_xx' / M              (normalized inner product in the C-feature space)
J(θ)   = sin θ + (π − θ) cos θ  (order-1 angular term)
```

Equivalently, expanding the angle,

```
θ = cos⁻¹( (xᵀ C x' + σ₀²) / ( √(xᵀ C x + σ₀²) · √(x'ᵀ C x' + σ₀²) ) )
```

`σ₀²` is a constant **bias variance** hyperparameter. `C ∈ ℝ^{n_x × n_x}` is the
structured spatial prior covariance (§3).

**Proportionality constant.** The LaTeX summaries write `K ∝ M·J(θ)`. The code fixes the
constant explicitly at **1/π** (`kernels.py:645`, `K = M * J / torch.pi`). Throughout this
document the constant is 1/π; every `∝` in the LaTeX becomes `= (1/π)·` in code.

### 1.2 Order and neural-network interpretation

This is the arc-cosine kernel of **order n = 1**. The angular term
`J₁(θ) = sin θ + (π − θ) cos θ` is exactly the order-1 (ReLU) member of the
Cho & Saul (2009) arc-cosine family (order 0 = Heaviside/step, order 1 = ReLU).

Interpretation (stated in `acosker_kernel_def_and_gradients.tex` §1 and the code
docstring): `K` is the covariance of the output of an **infinite-width, single-hidden-layer
neural network with ReLU hidden units**. The input-to-hidden weights `w` are drawn from a
Gaussian prior `w ~ N(0, C)`; the bias contributes `σ₀²`. Then

```
K(x, x') = E_w[ ReLU(wᵀx) · ReLU(wᵀx') ]   (up to the 1/π constant, with bias σ₀²)
```

so `C` is literally the **prior covariance of the network weights** — "the covariance
between weights at pixel locations ξᵢ and ξⱼ" (LaTeX §1.1). Shaping `C` (§3) therefore
injects the prior belief that a retinal ganglion cell's receptive field is spatially
**local** and **smooth**.

### 1.3 Diagonal case

When `x = x'`: `c_xx' = v_x = M`, so `cos θ = 1`, `θ = 0`, and
`J(0) = sin 0 + (π − 0)cos 0 = π`. Hence

```
K(x, x) = M · J(0) / π = M · π / π = M = √(v_x·v_x) = v_x = xᵀ C x + σ₀².
```

Code confirms this: the `diag=True` branch returns `V1 = xᵀCx + σ₀²` directly
(`kernels.py:614-618`).

**Non-stationarity.** `K` depends on the *actual input values* through `xᵀCx`, not only on
`x − x'`. Consequently the prior variance `v_x = xᵀCx + σ₀²` grows with the input norm.
This matters for active learning: utility maximization (which uses ∇ₓ, §4) is biased
toward high-norm images unless the kernel is normalized. The code provides normalized and
saturating siblings for this reason (§1.5); this note keeps focus on the base
`ArcCosineKernel`.

### 1.4 Code realization of `forward()` (numerical clamps)

`ArcCosineKernel.forward()` (`kernels.py:561-647`):

```
sigma_0_sq = self.sigma_0 ** 2                      # σ₀² ; self.sigma_0 = exp(raw_sigma_0) > 0
C, mask    = self._compute_C_matrix(apply_mask=…)   # §3
x1, x2     = x1[..., mask], x2[..., mask]            # restrict to active pixels
CX1 = x1 @ C ;  V1 = (CX1*x1).sum(-1) + σ₀²          # v_x
CX2 = x2 @ C ;  V2 = (CX2*x2).sum(-1) + σ₀²          # v_x'
C12 = CX1 @ x2ᵀ + σ₀²                                # c_xx'
M   = √(V1 ⊗ V2)
cos θ   = clamp(C12 / M, −1+eps, 1−eps)              # eps = 1e-7  (keep in arccos domain)
θ       = arccos(cos θ)
sin θ   = √(clamp(1 − cos²θ, min=eps))               # avoid sqrt of ≤0
J       = sin θ + (π − θ)·cos θ
K       = M · J / π
```

Two internal guards (distinct from Cholesky jitter, §5): `eps = 1e-7` clamps `cos θ` into
`(−1, 1)` so `arccos` is finite, and clamps `1 − cos²θ` at `min=eps` so `sin θ` never takes
`√` of a negative float-rounding result.

### 1.5 Sibling kernels (present in code; not this cluster's focus)

Same `C`-matrix / masking infrastructure, different nonlinearity or normalization
(`kernels.py`):
- `ArcCosineKernelNormalized` — `K̄ = J(θ)/π` (magnitude `M` dropped); constant prior
  variance `K̄(x,x)=1`. Removes the norm-scaling incentive that pushes utility to domain
  corners.
- `ArcSineKernel` — erf-network kernel `K = (2/π)·arcsin( c_xx' / √((1+v_x)(1+v_x')) )`;
  saturates at 1 (same depth, different nonlinearity).
- `LocalRBFKernel` — stationary RBF `exp(−(x−x')ᵀ C_base (x−x') / (2ℓ²))`, using `C_base`
  (= `C` **without** `Amp`) plus a lengthscale ℓ.
- `SimpleArcCosineKernel` / `SimpleArcCosineNormalizedKernel` — `C = I`, no RF structure
  (low-dimensional playground only).

---

## 2. (reserved — angular term reused below)

`J(θ) = sin θ + (π − θ) cos θ`, with derivative used repeatedly in §4:

```
dJ/dθ = cos θ − π sin θ − (cos θ − θ sin θ) = −(π − θ) sin θ.
```

---

## 3. Structured Spatial Prior Covariance C

`C` encodes that RGC receptive fields are spatially local and smooth. It is **constant
w.r.t. the stimulus x** (it depends only on hyperparameters β, ρ, ξ₀). Pixel `i` sits at
grid coordinate `ξᵢ = (xcord[i], ycord[i])` on a normalized `[−1,1] × [−1,1]` grid
(`kernels.py:_setup_pixel_coords`, float64). `ξ₀ = (eps_0x, eps_0y)` is the RF center.

### 3.1 LaTeX / paper form (natural parameters)

`acosker_kernel_def_and_gradients.tex` §1.1:

```
C_ij = α_i^local · α_j^local · C_ij^smooth
α_i^local  = exp( − ||ξᵢ − ξ₀||² / (4 β²) )        (locality weight)
C_ij^smooth = exp( − ||ξᵢ − ξⱼ||² / (2 ρ²) )        (smoothness kernel)
```

so, combined,

```
C_ij = exp( − (||ξᵢ − ξ₀||² + ||ξⱼ − ξ₀||²) / (4 β²) − ||ξᵢ − ξⱼ||² / (2 ρ²) ).
```

Here β and ρ are the **natural** ("paper") parameters:
- **β** = receptive-field **size / half-width**: `α` reaches `e⁻¹` at `dist = 2β`.
  `α` is a Gaussian of standard deviation `√2·β` in pixel-distance.
- **ρ** = smoothness **standard deviation** (length scale) of the RBF-like `C_smooth`
  (variance `ρ²`).
- **ξ₀** = RF center (2D).

### 3.2 Code form (raw / unconstrained parameters)

For optimizer stability the code stores **unconstrained log-space raw parameters**, not β
and ρ directly. Registered in `__init__` (`kernels.py:229-242`, all float64):

```
eps_0x, eps_0y   : RF center, direct parameters on [−1, 1]
raw_m2log2beta   = −2·log(2·β)      # init from natural β (default 0.1 → 3.22)
raw_mlog2rho2    = −log(2·ρ²)       # init from natural ρ (default 0.1 → 3.91)
raw_sigma_0      : σ₀ = exp(raw_sigma_0)  (Positive(exp/log) constraint)
```

Inside `_compute_C_matrix` (`kernels.py:486-559`) the exponentials of the raw parameters
are taken and used **directly** as the coefficients in the exponents:

```python
beta = torch.exp(self.raw_m2log2beta)     # local var — NOT natural β  (call it beta_code)
rho2 = torch.exp(self.raw_mlog2rho2)      # local var — NOT ρ²         (call it rho2_code)
dist_sq_center = (xcord - eps_0x)**2 + (ycord - eps_0y)**2
alpha    = exp(-beta * dist_sq_center)                     # α
dx,dy    = pairwise coordinate differences
C_smooth = exp(-rho2 * (dx**2 + dy**2))                    # C_smooth
C        = Amp * alpha[:,None] * C_smooth * alpha[None,:]  # Amp: see §7 (excluded)
C        = (C + C.T) / 2                                   # symmetrize (§5)
```

**The local variable `beta` is 1/(4β²) and `rho2` is 1/(2ρ²)** — not the natural values.

### 3.3 Reconciliation (LaTeX-vs-code) — RESOLVED

`BETA_RHO_MATH_VERIFICATION.md` verifies the two forms are the **same function**; the only
difference is variable naming/parameterization. Substituting the raw→code transforms:

```
beta_code = exp(raw_m2log2beta) = exp(−2·log(2β)) = (2β)⁻²  = 1/(4β²)
rho2_code = exp(raw_mlog2rho2)  = exp(−log(2ρ²)) =            1/(2ρ²)
```

Therefore the code's exponents reduce exactly to the LaTeX ones:

```
α        = exp(−beta_code · dist²) = exp(−dist² / (4β²))     ✓  = LaTeX α^local
C_smooth = exp(−rho2_code · dist²) = exp(−dist² / (2ρ²))     ✓  = LaTeX C^smooth
```

Numerically verified (β=ρ=0.1): `beta_code = 25 = 1/(4·0.1²)`, `rho2_code = 50 = 1/(2·0.1²)`.
The `.beta` / `.rho` **properties** (`kernels.py:294-326`) invert the raw parameters to
recover the natural values, and round-trip cleanly:

```
beta (property) = exp(−raw_m2log2beta / 2) · 0.5   → returns natural β   (0.1 → 0.1 ✓)
rho  (property) = exp(−raw_mlog2rho2 / 2) / √2      → returns natural ρ   (0.1 → 0.1 ✓)
```

(Matching inverse transforms `logbetaexpr_to_beta` / `logrhoexpr_to_rho` in the legacy
`utils.py`, per the verification doc.)

**FINAL AGREED FORM (authoritative for the whole document):**

```
C_ij = exp( −||ξᵢ − ξ₀||² / (4β²) ) · exp( −||ξᵢ − ξⱼ||² / (2ρ²) ) · exp( −||ξⱼ − ξ₀||² / (4β²) )
```

with natural β (RF half-width), natural ρ (smoothness SD), ξ₀ = (eps_0x, eps_0y). (Plus a
symmetrization `(C + Cᵀ)/2`, and — in the code only, excluded per scope — a scalar `Amp`
multiplier; see §7.)

**Naming hazard for downstream sessions.** "beta"/"rho" are OVERLOADED:
- natural β, ρ (LaTeX, the `.beta`/`.rho` properties) — use THESE when reasoning/plotting.
- code locals `beta = 1/(4β²)`, `rho2 = 1/(2ρ²)` inside `_compute_C_matrix` — reciprocal-
  square coefficients, NOT the natural params. Do not confuse.

### 3.4 Physical interpretation (defaults β = ρ = 0.1 on the [−1,1] grid)

- `α` (locality) hits `e⁻¹` at `dist = 2β`; half-max at `1.665·β`; ~99% mass within
  `4.29·β`. For β=0.1 the RF spans roughly ±0.2 (≈20% of the axis).
- `C_smooth` (RBF): pixels closer than `ρ` are strongly correlated (>0.6); pixels beyond
  `3ρ` nearly uncorrelated (<0.01). ρ is the correlation length.
- Together: `α` localizes weight around ξ₀, `C_smooth` correlates nearby pixels — an
  RF-shaped weight prior.

### 3.5 Positivity, bounds, constraints (from code)

| Param | Constraint mechanism | Bounds (natural) | Bounds (raw) |
|---|---|---|---|
| σ₀ | `Positive(exp/log)`: σ₀ = exp(raw_sigma_0) | σ₀ > 0 | — |
| β | via `raw_m2log2beta`, clamped | **[0.01, 0.3]** | RAW_BETA ∈ [1.02, 7.82] |
| ρ | via `raw_mlog2rho2`, clamped | [0.01, 0.5] | RAW_RHO ∈ [0.69, 8.52] |
| ξ₀ | direct clamp | [−1, 1] per axis (or tight radius via `set_center_bounds`) | — |
| Amp | `Positive()`, clamp (excluded, §7) | (0, 1000] | — |

Bounds are enforced two ways: `params_in_bounds()` (read-only reject inside the LBFGS
closure) and `clamp_hyperparameters()` (projected clamp after `optimizer.step()`).

**β upper bound = 0.3, not 1.0.** Tightened from 1.0 → 0.3 on 2026-04-10: β drift
0.12→0.45 made `C` cover all 11664 px (~519 MB) and OOM. At β=0.3, RF σ = 0.3·√2 ≈ 0.42
(~23-px σ on 108²). RAW_BETA_MIN = −2·log(2·0.3) = **1.02**. (Code inline comment on
`kernels.py:180` says "≈ 0.18" — that is WRONG; the block comment on line 172 and the
arithmetic both give 1.02. `clamp_hyperparameters` docstring still says "beta ∈ [0.01, 1.0]"
— also stale. Flagged in §7.)

### 3.6 Pixel masking (locality exploitation)

`MASK_THRESHOLD = 0.001`. Pixels with locality weight `α ≥ 0.001` are kept; the rest are
dropped, shrinking `C` from (n_px, n_px) to (n_masked, n_masked) — typically ~100× smaller
(`use_mask=True` default). The mask (`compute_mask`, `kernels.py:461-484`) is computed with
**DETACHED** parameters (`beta.detach()`, `eps.detach()`) so mask *structure* is
non-differentiable and stable, while the retained `C` entries still carry gradients.

Mask radius: `α ≥ 0.001` ⟹ `dist ≤ 2β·√(ln 1000) ≈ 5.25·β` (natural β). For β=0.1,
radius ≈ 0.525 (~28 px on 108²). `forward()` applies the same mask to the inputs
(`x1 = x1[..., mask]`), so only active pixels contribute to `xᵀCx`.

---

## 4. Kernel Gradients

### 4.1 Gradient w.r.t. the input x — ∇ₓK (utility / active-learning guidance)

Needed to pick the next image (maximize expected information gain requires ∂K/∂x). Full
derivation (`Acosker_gradient_dx.tex`; identical block in the other LaTeX), C symmetric and
constant, σ₀² constant, `v_x' ` constant w.r.t. x:

**Intermediate gradients**
```
∇ₓ v_x   = 2 C x
∇ₓ c_xx' = C x'
∇ₓ M     = √(v_x') · (1/(2√v_x)) · 2Cx = √(v_x'/v_x) · C x
```
**Angular term**
```
dJ/dθ  = −(π − θ) sin θ
∇ₓ J   = (dJ/dθ) ∇ₓθ = −(π − θ) sin θ · ∇ₓθ
∇ₓθ    = (−1/√(1−u²)) ∇ₓu,  u = c_xx'/M = cos θ,  √(1−u²) = sin θ
       = (−1/sin θ) · ∇ₓ(c_xx'/M)
⟹ ∇ₓ J = (π − θ) · ∇ₓ(c_xx'/M)            (the sin θ cancels)
∇ₓ(c_xx'/M) = (1/M) C x' − (c_xx'/M²) √(v_x'/v_x) C x
```
**Product rule on K ∝ M·J(θ):** `∇ₓK ∝ (∇ₓM)J + M(∇ₓJ)`
```
Term1 = √(v_x'/v_x) C x · [ sin θ + (π−θ) cos θ ]
      = sinθ √(v_x'/v_x) Cx  +  (π−θ)cosθ √(v_x'/v_x) Cx      … [A]+[B]
Term2 = M(π−θ)[ (1/M)Cx' − (c_xx'/M²)√(v_x'/v_x)Cx ]
      = (π−θ) Cx'  −  (π−θ) cosθ √(v_x'/v_x) Cx               … [C]−[D]   (c_xx'/M = cos θ)
```
**Cancellation:** [B] and [D] are identical with opposite sign → cancel. Result:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ ∇ₓ K(x, x') = (1/π) [ (π − θ) C x'  +  sin θ · √( v_x' / v_x ) · C x ]        │
│             = (1/π) [ (π − θ) C x'  +  sin θ · √( (x'ᵀCx'+σ₀²)/(xᵀCx+σ₀²) )·Cx ] │
└────────────────────────────────────────────────────────────────────────────┘
```
By symmetry (swap x ↔ x'):
```
∇_{x'} K(x, x') = (1/π) [ (π − θ) C x  +  sin θ · √( (xᵀCx+σ₀²)/(x'ᵀCx'+σ₀²) ) · C x' ].
```
(LaTeX writes `∝`; constant = 1/π to match §1.)

### 4.2 Gradients w.r.t. hyperparameters — ∂C/∂θ (M-step)

`acosker_kernel_def_and_gradients.tex` §2 gives the analytic derivatives of `C_ij` w.r.t.
the **natural** parameters (building blocks; the full ∂K/∂param then chains these through
`v_x = xᵀCx`, `c_xx'`, etc., using the same quadratic-form structure as §4.1 with `C`
replaced by `∂C/∂param`):

```
∂C_ij/∂β  = C_ij · [ (||ξᵢ − ξ₀||² + ||ξⱼ − ξ₀||²) / (2 β³) ]

∂C_ij/∂ρ  = C_ij · [ ||ξᵢ − ξⱼ||² / ρ³ ]

∇_{ξ₀} C_ij = (C_ij / (2 β²)) · [ (ξᵢ − ξ₀) + (ξⱼ − ξ₀) ]        (2D vector)
```
Derivations: each is `C_ij ·` (derivative of its log). For β,
`∂/∂β[ −(||ξᵢ−ξ₀||²+||ξⱼ−ξ₀||²)/(4β²) ] = +(…)/(2β³)`. For ρ,
`∂/∂ρ[ −||ξᵢ−ξⱼ||²/(2ρ²) ] = +||ξᵢ−ξⱼ||²/ρ³`. For ξ₀, using
`∇_{ξ₀}||ξ−ξ₀||² = 2(ξ₀−ξ)`.

**Code note.** These analytic forms are w.r.t. the natural β, ρ, ξ₀. The code optimizes the
**raw** parameters, so the default `gradient_mode='autograd'` differentiates through the
raw→natural transforms automatically (chain rule via `d(raw)/dβ = −2/β`, etc.). The
optional analytical modes reproduce the above explicitly.

### 4.3 Gradient modes (code)

`gradient_mode ∈ {'autograd','vjp','jacobian'}` (`kernels.py:38-61, 561-594`):
- `autograd` (default) — PyTorch autodiff through `forward()`.
- `vjp` — analytical VJP (`analytical_gradients_vjp.py`), same speed as autograd, explicit
  formulas (the §4.1/§4.2 expressions).
- `jacobian` — materialized Jacobian (`analytical_gradients.py`), slow, bit-matches the
  original varGP path.
All three return the same `K`; only the backward path differs. (Implementation files are
outside this cluster's sources — referenced, not read.)

---

## 5. Numerical Stability & Jitter

Three stability mechanisms at three distinct levels — do not conflate them:

**Level 1 — C matrix.** Symmetrize only: `C = (C + Cᵀ)/2` (`kernels.py:557`). Cancels
float-rounding asymmetry from the outer product. **No diagonal jitter is added to C.**

**Level 2 — kernel `forward()`.** `eps = 1e-7` clamps (§1.4): `cos θ` into `(−1, 1)` for
`arccos`, and `1 − cos²θ` at `min=eps` for `sin θ`. Purely to keep transcendental ops in
-domain.

**Level 3 — GP covariance / Cholesky (jitter.md).** Jitter is added to the **kernel gram
matrices** `K_uu` (inducing–inducing) and `K_XX` (predictive), i.e. to the *output* of the
arc-cosine kernel, at the GPyTorch `VariationalStrategy` level — NOT to `C`. Two sub-layers:

- **Layer 1 (pre-Cholesky).** `K_uu.add_jitter(jitter_val)` at
  `variational_strategy.py:196`; also on `K_XX` (lines 226/231). `jitter_val = model.jitter`
  (**default 1e-4**, wired via `gpy_model.py:63`). If `None` → GPyTorch
  `variational_cholesky_jitter` (1e-4 float32, 1e-6 float64). Our model's `forward()` does
  **not** add jitter — GPyTorch does it internally (adding it in `forward()` would
  double-jitter).
- **Layer 2 (retry escalation).** `psd_safe_cholesky` promotes `K_uu` to **float64**
  (intentional: sequential subtractions in Cholesky cause catastrophic cancellation in
  float32; ~15 vs ~7 digits) and, on failure, retries with `jitter_new = jitter · 10^i`
  (`cholesky.py:12-47`). Overrides: `cholesky_jitter` (float64 default 1e-8 → our
  `model.jitter` 1e-4) and `cholesky_max_tries` (default 3). With jitter=1e-4, max_tries=3
  the retries add **1e-4, 1e-3, 1e-2**.

Config wiring: `model.jitter` feeds both Layer 1 `jitter_val` and the Layer 2 retry start;
`model.cholesky_max_tries` sets the retry count (CLI `--jitter`, `--cholesky-max-tries`;
YAML `numerical.jitter`, `numerical.cholesky_max_tries`). Note: an error "…up to 1.0e-06"
means float64 jitter (1e-8·10² after 3 tries), not a float32 value.

---

## 6. Notation Table (with clashes)

| Symbol | Meaning | Source | Clash / note |
|---|---|---|---|
| `x, x'` | stimulus vectors (flattened images), ℝ^{n_x} | LaTeX, code | clashes with pixel coord `xcord`; `x` = image, `xcord` = grid x |
| `n_x` | input dim = #pixels = n_px_side² | LaTeX (`n_x`), code (`n_features`) | |
| `n_px_side` | image side (108) | code | |
| `C` | structured spatial prior covariance (n_x×n_x); weight covariance | LaTeX, code | in full spatiotemporal model = C_spatial (temporal/Kronecker OUT of scope) |
| `σ₀²` | bias variance (added in v_x, c_xx') | LaTeX; code `sigma_0_sq = self.sigma_0**2` | |
| `σ₀`/`sigma_0` | bias std; `= exp(raw_sigma_0) > 0` | code | LaTeX σ₀ ≡ code `self.sigma_0` |
| `raw_sigma_0`/`sigma_b` | unconstrained log-param, σ₀=exp(·) | code | |
| `v_x, v_x'` | xᵀCx+σ₀², x'ᵀCx'+σ₀² | LaTeX; code `V1,V2` | "self-terms" / prior variances |
| `c_xx'` | xᵀCx'+σ₀² (cross-term) | LaTeX; code `C12` | |
| `M` | √(v_x·v_x') magnitude | LaTeX, code | **CLASH:** `M` also = #inducing points (jitter.md). Here always magnitude. |
| `θ` | angle = arccos(c_xx'/M) | LaTeX, code | **CLASH:** legacy varGP uses `theta` as the whole hyperparameter dict (`theta['Amp']`, `theta['sigma_0']`). Here always the kernel angle. |
| `J(θ)` | sin θ + (π−θ)cos θ (order-1 angular term) | LaTeX, code | |
| `K` | kernel = (1/π)·M·J(θ) | LaTeX (∝), code (explicit 1/π) | |
| `n` (order) | arc-cosine order = 1 (ReLU) | inferred from J₁ form + LaTeX "ReLU" | clashes with n = #data points |
| **`β`/`beta` (natural)** | RF size/half-width; α=e⁻¹ at dist=2β | LaTeX, BETA_RHO doc, `.beta` property | **KEY CLASH** ↓ |
| **`ρ`/`rho` (natural)** | smoothness SD (length scale); var ρ² | LaTeX, BETA_RHO doc, `.rho` property | **KEY CLASH** ↓ |
| `beta` (code local) | `= exp(raw_m2log2beta) = 1/(4β²)` — NOT natural β | code, BETA_RHO doc | **reuses the name `beta` for a different quantity** |
| `rho2` (code local) | `= exp(raw_mlog2rho2) = 1/(2ρ²)` — NOT ρ² | code, BETA_RHO doc | |
| `raw_m2log2beta` | `−2·log(2β)` stored unconstrained | code | |
| `raw_mlog2rho2` | `−log(2ρ²)` stored unconstrained | code | |
| `ξ₀`/`(eps_0x,eps_0y)` | RF center on [−1,1]² | LaTeX, code | |
| `ξᵢ`/`(xcord[i],ycord[i])` | pixel i grid coordinate on [−1,1]² | LaTeX, code | |
| `α_i^local`/`alpha` | exp(−‖ξᵢ−ξ₀‖²/(4β²)) locality weight | LaTeX, code | |
| `C_ij^smooth`/`C_smooth` | exp(−‖ξᵢ−ξⱼ‖²/(2ρ²)) smoothness | LaTeX, code | |
| `Amp` | legacy scalar multiplying C; **EXCLUDED** per scope | code (module docstring, `_compute_C_matrix`) | present in code, no-op at default 1.0 — see §7 |
| `MASK_THRESHOLD` | 0.001; keep pixels with α ≥ threshold | code | |
| `jitter`/`jitter_val` | 1e-4; added to K_uu/K_XX pre-Cholesky | jitter.md | added to gram K, NOT to C |
| `A, λ₀` | Poisson-likelihood params (drive) | code docstrings | NOT this cluster (likelihood note owns them) |

**Cross-agent flag on β/ρ/ξ₀:** other notes should use the **natural** β, ρ (RF
half-width, smoothness SD) and ξ₀ (RF center). If any other note reads β/ρ off the code, it
must use the `.beta`/`.rho` *properties*, never the `_compute_C_matrix` locals `beta`/`rho2`
(which are 1/(4β²) and 1/(2ρ²)). Also warn: `θ` and `M` are each reused elsewhere (θ =
hyperparameter dict; M = #inducing points).

---

## 7. Conflicts, Redundancies, Flags (code = ground truth)

1. **`Amp` — code-vs-scope conflict (most important).** This document's scope EXCLUDES
   `Amp` as deprecated legacy-varGP baggage, and the LaTeX summaries omit it entirely
   (`C_ij = α_i α_j C_ij^smooth`, no amplitude). But the **current `kernels.py` still
   carries it**: `C = self.Amp · α · C_smooth · αᵀ` (`_compute_C_matrix`, line 554), plus a
   `raw_Amp` parameter, `AMP_MAX=1000` clamp, and `create_kernel` reading `config['Amp']`.
   Its docstring says it "multiplies C directly … affecting the kernel non-linearly through
   the sqrt and arccos operations (matching legacy varGP)." **Resolution for this
   document:** the core structured-prior math is the Amp-free form in §3.3; `Amp` is a
   scalar prefactor that, at its **default 1.0, is a no-op**. `LocalRBFKernel` already
   drops it (`C_base` excludes Amp). Recorded here per the ground-truth rule: Amp exists in
   code but is out of scope; if ever ≠1 it rescales `C` and thus enters K non-linearly.

2. **β natural bounds — LaTeX-vs-code, and code self-inconsistency.**
   - Authoritative bound: **BETA_MAX = 0.3** (code class constant + block comment
     line 172, with the 2026-04-10 OOM rationale). BETA_RHO_MATH_VERIFICATION.md (Jan 2025)
     predates this and still discusses β up to 1.0 — treat its ranges as advisory/older.
   - `kernels.py:180` inline comment `RAW_BETA_MIN … # ≈ 0.18` is **wrong**:
     `−2·log(2·0.3) = 1.02` (verified). The block comment (line 172) and the code both give
     1.02.
   - `clamp_hyperparameters` docstring (`kernels.py:379`) says "beta ∈ [0.01, 1.0]" —
     **stale**; the enforced bound is [0.01, 0.3]. (Instance of the W4 "stale prose"
     pattern.)

3. **`∝` vs explicit 1/π.** LaTeX uses `K ∝ …` and `∇ₓK ∝ …`. The code fixes the constant
   at **1/π** (`kernels.py:645`). No contradiction — the LaTeX simply omits the constant.
   Resolved: constant = 1/π everywhere (§1, §4.1).

4. **σ₀ nesting.** LaTeX `σ₀²` = code `sigma_0_sq = self.sigma_0**2`, with
   `self.sigma_0 = exp(raw_sigma_0)`. So the added bias variance is `(exp(raw_sigma_0))²`.
   Consistent; just note the double transform (raw → exp → square).

5. **β/ρ naming (redundancy already resolved).** LaTeX and code agree on the final `C`
   (§3.3); the apparent mismatch is only that the code's `_compute_C_matrix` locals reuse
   the names `beta`/`rho2` for `1/(4β²)` / `1/(2ρ²)`. Not a bug — a documented naming
   hazard (BETA_RHO doc §6.1 recommends renaming to `inv_4beta_sq`/`inv_2rho_sq`; not done
   in current code).

6. **No jitter on C.** Some intuition might expect diagonal jitter on the prior `C`; there
   is none. `C` gets only symmetrization; jitter lives on the GP gram matrices K_uu/K_XX
   (§5). Flagged so a reader does not "add jitter to C" thinking it is missing.

---

## 8. Scope Exclusions (explicit)

- **Spatiotemporal machinery** — `C_temporal`, the Kronecker `C = C_spatial ⊗ C_temporal`,
  and warping: OUT of scope AND absent from every source in this cluster (all sources are
  purely spatial). The `C` here is the spatial prior; in a full model it would be
  `C_spatial`.
- **`Amp`** — excluded per scope (deprecated legacy-varGP); presence in code flagged in §7.1.
- **Other deprecated vargp_old baggage** — excluded.
- **Likelihood parameters** (`A`, `λ₀`, Poisson link) — owned by the likelihood note, not
  here (mentioned only where a kernel docstring references them).
- **Variational inference internals** (eigenspace, inducing points, E/M-step machinery) —
  owned by other notes; touched here only for the jitter that lands on K_uu/K_XX (§5) and
  for how the §4.2 C-derivatives feed the M-step.
- **Analytical-gradient implementation files** (`analytical_gradients*.py`) — referenced as
  the `vjp`/`jacobian` backends (§4.3) but not read (outside this cluster's sources).
- **Sibling kernels** (`ArcCosineKernelNormalized`, `ArcSineKernel`, `LocalRBFKernel`,
  `Simple*`) — noted in §1.5 for context; the base `ArcCosineKernel` is the focus.
```
