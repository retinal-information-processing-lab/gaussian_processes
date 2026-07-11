# 03 — The E-Step: Variational Newton Update of the Latent Posterior (Eigenspace)

**Cluster:** the E-step — the closed-form Newton update of the variational
posterior over the latent GP `q(λ̃) = N(m, V)` at **fixed** hyperparameters
(kernel params, gain `A`, bias `λ₀`). This is the "expectation" half of the
EM-style training of the spatial variational GP that learns a retinal ganglion
cell's receptive field from spike counts to natural images.

**Ground truth (current implementation):**
- `scripts/gpytorch_porting/eigenspace_estep.py` — `estep_eigenspace()` (one Newton step)
- `scripts/gpytorch_porting/likelihoods.py` — `PoissonLikelihood` (expected log-lik, f̄, derivatives)
- driven by `scripts/gpytorch_porting/eigenspace_training.py` (the `n_estep` Newton loop)
- state container `DirectVariationalState` in `scripts/gpytorch_porting/eigenspace_model.py`,
  built by `scripts/gpytorch_porting/eigenspace_utils.py`

**Sources distilled:**
- `.claude/archive/ESTEP_MATH_ANALYSIS.md` (Jan 2025 — code-vs-derivation audit)
- `Papers/latex_summaries/Estep_corrected.tex`
- `Papers/latex_summaries/Estep_corrected_mderivation.tex`

Where a note contradicts the ground-truth code, the code wins and the conflict is
flagged (§10). The deprecated repo-root `utils.py:Estep()` is **not** ground truth;
the current eigenspace code is a faithful re-expression of it in the reduced basis.

> **Scope:** SPATIAL GP only. No temporal covariance `C`, no Kronecker/warping
> machinery — see §13. All math below is for the per-image latent, `λ` a function
> of image `x` only.

---

## 1. Model setup (spatial, sparse variational GP + Poisson)

Generative model for a single cell (`Estep_corrected*.tex` §2, `likelihoods.py`):

```
λ(x)  ~  GP(0, k(x, x'))                 latent tuning function (log-drive)
f(xᵢ) =  exp( A·λ(xᵢ) + λ₀ )             expected firing rate (exp link)
yᵢ    ~  Poisson( f(xᵢ) )                spike count for image i
```

- `A` = gain (learnable, positive), `λ₀` = bias / baseline log-firing rate (learnable).
- `yᵢ` = spike count for image `i` (called `r` in the code — **synonym**, see §12).

**Sparse variational approximation.** Introduce `M` inducing points at locations
`Z̃` with latent values `λ̃`, and approximate their posterior by a Gaussian whose
parameters are the E-step unknowns:

```
q(λ̃) = N( λ̃ | m, V ),     m ∈ ℝ^M,   V ∈ ℝ^{M×M}
```

The latent at any training image `i` is Gaussian, projected from the inducing
points through the kernel (`Estep_corrected*.tex` eq. mu_def / sigma_def):

```
q(λᵢ) = N(μᵢ, σᵢ²)
μᵢ   = kᵢᵀ K̃⁻¹ m                                   (posterior mean at i)
σᵢ²  = kᵢᵢ + kᵢᵀ K̃⁻¹ (V − K̃) K̃⁻¹ kᵢ               (posterior variance at i)
```

with `K̃ = k(Z̃, Z̃)` (M×M inducing kernel), `kᵢ = k(Z̃, xᵢ)` (cross-covariance),
`K = [k₁ … k_N]ᵀ` the N×M cross-kernel, `kᵢᵢ = k(xᵢ,xᵢ)`.

---

## 2. The variational objective (ELBO)

The E-step maximises the ELBO over `(m, V)` with everything else fixed
(`Estep_corrected*.tex` §3):

```
L = Σᵢ E_{q(λᵢ)}[ log p(yᵢ | λᵢ) ]  −  D_KL( q(λ̃) ‖ p(λ̃) )
```

**Data term** — the Poisson expected log-likelihood (see §3):

```
L_data = Σᵢ [ yᵢ (A·μᵢ + λ₀) − f̄ᵢ ] + const
f̄ᵢ    = exp( A·μᵢ + ½ A² σᵢ² + λ₀ )          "expected firing rate"
```

`f̄ᵢ` is `E_q[ exp(A·λᵢ + λ₀) ]` using the Gaussian MGF `E[e^X] = exp(μ_X + ½σ_X²)`.
The constant is `−log yᵢ!`, dropped (independent of `m,V,A,λ₀`).

**KL term** — between `q = N(m,V)` and prior `p = N(0, K̃)`:

```
−D_KL = ½ log|V| − ½ log|K̃| − ½ Tr(K̃⁻¹ V) − ½ mᵀ K̃⁻¹ m + M/2
```

Only two pieces depend on the E-step variables through the KL: `−½ mᵀK̃⁻¹m`
(couples to `m`) and `½log|V| − ½Tr(K̃⁻¹V)` (couples to `V`).

---

## 3. Poisson likelihood: expected log-lik and the derivatives the Newton step needs
`likelihoods.py: PoissonLikelihood`

**Expected log-probability** (`expected_log_prob`, lines 138–158) — exactly `L_data`:

```python
log_prob = target*(A*mu + lambda0) − exp(A*mu + 0.5*A**2*var + lambda0)
return log_prob.sum(-1)
```
i.e. `Σᵢ [ yᵢ(A μᵢ + λ₀) − f̄ᵢ ]`, `log yᵢ!` omitted (docstring lines 27–30).

**Expected firing rate f̄** (`expected_firing_rate`, lines 176–192; the loop's
`compute_f_mean` computes the identical scalar):

```
f̄ = exp( A·λ_m + ½ A²·λ_var + λ₀ )
```
`λ_m, λ_var` are the posterior moments `μ, σ²`. **`f_mean` in the code is exactly
`f̄`.** It is computed from the *current* posterior and passed into the E-step; it
is the single quantity that couples successive Newton steps (§11).

**Point derivatives that assemble the E-step (chain rule on `f̄ᵢ`):**

```
∂f̄ᵢ/∂μᵢ   = A f̄ᵢ                    (mean sensitivity → builds g and G)
∂f̄ᵢ/∂σᵢ²  = ½ A² f̄ᵢ                 (variance sensitivity → builds ∇_V term)
∂/∂μᵢ  E_q[log p] = A(yᵢ − f̄ᵢ)       (per-point mean gradient → builds g)
```

**Parameterisation & bounds** (used by the F-step, not the E-step, but part of the
likelihood contract): `A = exp(raw_A)` (positivity via exp/log, matching varGP's
`logA`), `λ₀` unconstrained. Guards: `A ∈ (0, 10]`, `λ₀ ∈ [−50, 50]`
(`A_MAX`, `LAMBDA0_MIN/MAX`; `params_in_bounds`, `clamp_params`). `forward()`
returns `Poisson(exp(A·λ+λ₀))` for sampling. Note `A` and `λ₀` are **fixed** during
the E-step (detached in the training loop).

---

## 4. The two helper matrices g and G

From the derivatives above, define (`Estep_corrected*.tex` §4):

```
g ≡ A  Σᵢ kᵢ (yᵢ − f̄ᵢ)            ∈ ℝ^M     ("standard" gradient vector)
G ≡ A² Σᵢ f̄ᵢ kᵢ kᵢᵀ               ∈ ℝ^{M×M} ("standard" curvature matrix, PSD)
```

`g` is the accumulated mismatch between observed and expected counts, projected
through the kernel; `G` is the Poisson Fisher information (curvature), always
positive semidefinite because `f̄ᵢ > 0`.

---

## 5. Deriving the m-update (gradient, Hessian, Newton)
`Estep_corrected_mderivation.tex` §"Optimizing the Mean m" — full steps preserved.

`σᵢ²` (hence the variance part of `f̄ᵢ`) does **not** depend on `m`, so:

**Step 1 — Gradient.** With `∂μᵢ/∂m = K̃⁻¹kᵢ` and `∂f̄ᵢ/∂μᵢ = A f̄ᵢ`:

```
∇_m L_data = Σᵢ (yᵢ A − f̄ᵢ A) K̃⁻¹kᵢ = K̃⁻¹ [ A Σᵢ kᵢ(yᵢ − f̄ᵢ) ] = K̃⁻¹ g
∇_m (−KL) = −K̃⁻¹ m
⇒  ∇_m L = K̃⁻¹ (g − m)
```

**Step 2 — Hessian.** `g` depends on `m` through `f̄ᵢ`; `∇_m f̄ᵢ = A f̄ᵢ K̃⁻¹kᵢ`, so
`∇_m g = −A² Σᵢ f̄ᵢ kᵢ kᵢᵀ K̃⁻¹ = −G K̃⁻¹`. Therefore:

```
H_m = ∇²_m L = K̃⁻¹(−G K̃⁻¹) − K̃⁻¹ = −K̃⁻¹ (K̃ + G) K̃⁻¹
```

**Step 3 — Newton–Raphson** `m ← m − H_m⁻¹ ∇_m L`, with
`H_m⁻¹ = −K̃(K̃+G)⁻¹K̃`:

```
m_new = m + K̃ (K̃ + G)⁻¹ (g − m)          ← CORRECTED-DERIVATION m-update
```

Equivalent expanded form (used to compare with code, `ESTEP_MATH_ANALYSIS.md`):

```
m_new = G(K̃+G)⁻¹ m + K̃(K̃+G)⁻¹ g
        [uses I − K̃(K̃+G)⁻¹ = G(K̃+G)⁻¹]
```

Stationary point: `∇_m L = K̃⁻¹(g − m) = 0 ⇒ m* = g` (with `g = g(m*)` implicit,
since `g` depends on `m` through `f̄`). Both this Newton map and the code's map
fix `m* = g` — see §10.

---

## 6. Deriving the V-update (∇_V L = 0, closed form)
`Estep_corrected*.tex` §"Optimizing the Covariance V". `μᵢ` does not depend on `V`.

**Data-term gradient** (chain rule `∂f̄ᵢ/∂σᵢ² = ½A²f̄ᵢ`, `∂σᵢ²/∂V = K̃⁻¹kᵢkᵢᵀK̃⁻¹`):

```
∇_V L_data = −Σᵢ ½ A² f̄ᵢ (K̃⁻¹kᵢkᵢᵀK̃⁻¹) = −½ K̃⁻¹ G K̃⁻¹
```

**KL-term gradient:** `∇_V(−KL) = ½V⁻¹ − ½K̃⁻¹`.

**Set total to zero and solve:**

```
½V⁻¹ − ½K̃⁻¹ − ½K̃⁻¹ G K̃⁻¹ = 0
⇒ V⁻¹ = K̃⁻¹ + K̃⁻¹ G K̃⁻¹ = K̃⁻¹ (K̃ + G) K̃⁻¹
⇒ V   = K̃ (K̃ + G)⁻¹ K̃                    ← CORRECTED V-update (symmetric sandwich)
```

`V = K̃·(K̃+G)⁻¹·K̃` is a symmetric "sandwich": `K̃` and `(K̃+G)⁻¹` are symmetric PD,
so the product `K̃ · (·) · K̃` is symmetric PD. This symmetry is the whole point of
the "correction" (§7).

---

## 7. What "corrected" fixed: the non-symmetric V bug

The `Estep_corrected*.tex` documents label the **V-update** as the fix.

| quantity | OLD source formula (WRONG) | CORRECTED / code | why old is wrong |
|---|---|---|---|
| **V** | `V = (K̃ + G)⁻¹ K̃` | `V = K̃(K̃ + G)⁻¹K̃` | product of two symmetric matrices is symmetric only if they **commute**; `K̃` and `(K̃+G)⁻¹` don't ⇒ old `V` is non-symmetric ⇒ not a valid covariance ⇒ Cholesky/PSD failures |

The old (superseded) `Gaussian_process_theory.tex` also carried a **wrong m
formula** `m = K̃(K̃+G)⁻¹(GK̃⁻¹m + g)` and *claimed* it equals the correct
`m + K̃(K̃+G)⁻¹(g−m)` — that algebraic step is invalid (`ESTEP_MATH_ANALYSIS.md`
"OLD LaTeX Formulas"). The corrected derivation keeps the honest Newton form
`m_new = m + K̃(K̃+G)⁻¹(g−m)`. **Crucially, the ground-truth code implements the
corrected V but the OLD-form m — see the conflict flag in §10.**

**Key identity used to connect code to formula** (`ESTEP_MATH_ANALYSIS.md`):

```
(I + A B⁻¹)⁻¹ = B (B + A)⁻¹
proof: (I + AB⁻¹)·B(B+A)⁻¹ = B(B+A)⁻¹ + A(B+A)⁻¹ = (B+A)(B+A)⁻¹ = I
```

---

## 8. The eigenspace representation — why the update is cheap
`eigenspace_utils.py`, `eigenspace_model.py: DirectVariationalState`

The E-step never touches the M-dimensional inducing space directly. It works in the
**eigenbasis of `K̃`**, keeping only the `n_b` leading eigen-directions
(`n_b ≈ 10–11` in practice; `M` can be large). Eigendecompose
`K̃ = B Λ Bᵀ`, keep eigenvalues above `max(λ_max·tol, tol)` (`tol = EIGVAL_TOL = 1e-4`,
`eigendecompose_K_tilde`). Retained pieces (all `_b` = "in eigenbasis"):

| object (code) | definition | shape | note |
|---|---|---|---|
| `B` | kept eigenvectors of `K̃` | (M, n_b) | columns, ascending eigenvalue order |
| `eigvals_b` (Λ) | kept eigenvalues | (n_b,) | |
| `K_tilde_b` | `Bᵀ K̃ B = diag(eigvals_b)` | (n_b, n_b) | **DIAGONAL** = Λ |
| `K_b` | `K B` (cross-kernel in eigenspace) | (N, n_b) | |
| `KKtilde_inv_b` (`a`) | `K_b / eigvals_b` = `K K̃⁻¹` in eigenbasis | (N, n_b) | `K K̃⁻¹ = a Bᵀ` |
| `m_b` | `Bᵀ m` | (n_b,) | variational mean |
| `V_b` | variational covariance | (n_b, n_b) | **NOT** diagonal |

Prior init (`initialize_variational_state`): `m_b = 0`, `V_b = K_tilde_b = Λ`.

**Exact reductions** (basis algebra, on the retained subspace):
```
K̃ = B Λ Bᵀ ,  K̃⁻¹ = B Λ⁻¹ Bᵀ ,  K K̃⁻¹ = K B Λ⁻¹ Bᵀ = a Bᵀ
λ_m  = K K̃⁻¹ m = a m_b
λ_var = diag(Kvec) + diag( a (V_b − Λ) aᵀ )        (predict_eigenspace)
```
Since `K̃_b = Λ` is diagonal, `K̃⁻¹` is trivial (`1/eigvals_b`), and every E-step
quantity (`m_b`, `g_b`∈ℝ^{n_b}, `G_b`∈ℝ^{n_b×n_b}, the Newton solve) lives in the
tiny `n_b`-dimensional space. **That dimensionality collapse (M → n_b ≈ 10), not any
special-casing inside `estep_eigenspace`, is what makes the Newton step cheap.**

---

## 9. GROUND-TRUTH CODE: `estep_eigenspace()` mapped line by line
`eigenspace_estep.py`. Runs entirely under `torch.no_grad()` (the update is
closed-form; leaving autograd on would chain through `A` and leak memory).

```python
A = model.likelihood.A.squeeze()
a = state.KKtilde_inv_b                       # (N, n_b) = K K̃⁻¹ in eigenbasis
g_b = A * (a.T @ (r - f_mean))                # (n_b,)      transformed gradient
G_b = (A*A) * (a.T @ (f_mean[:,None] * a))    # (n_b,n_b)   transformed curvature
eye = I_{n_b}
V_b_new = torch.linalg.solve(eye + K_tilde_b @ G_b, K_tilde_b)   # V update
m_b_new = V_b_new @ (G_b @ m_b + g_b)                            # m update
V_b_new = (V_b_new + V_b_new.T) / 2           # symmetrize for stability
model.update_variational_params(m_b_new, V_b_new)
```

**`g_b`, `G_b` are the TRANSFORMED (pre-multiplied) quantities**, related to the
standard `g, G` of §4 by (verified by expanding `a = K K̃⁻¹`):

```
g_b  =  K̃⁻¹ g            (eigenbasis coords: K̃⁻¹ g = B g_b)
G_b  =  K̃⁻¹ G K̃⁻¹        (eigenbasis coords)
```
because `a = KKtilde_inv_b = K K̃⁻¹` (in-basis) ⇒
`a·(r−f̄) = K̃⁻¹ Σ kᵢ(yᵢ−f̄ᵢ)` and `aᵀ diag(f̄) a = K̃⁻¹ (Σ f̄ᵢ kᵢkᵢᵀ) K̃⁻¹`.

**V update — algebra confirms it equals the corrected symmetric V** (`ESTEP_MATH_ANALYSIS.md`):
```
V_b_new = (I + K̃_b G_b)⁻¹ K̃_b
        = (I + K̃ · K̃⁻¹GK̃⁻¹)⁻¹ K̃
        = (I + G K̃⁻¹)⁻¹ K̃
        = K̃(K̃+G)⁻¹ · K̃                       [identity (I+AB⁻¹)⁻¹ = B(B+A)⁻¹]
        = V_correct  ✓
```

**m update — expands to:**
```
m_b_new = V_b_new (G_b m_b + g_b)
        = K̃(K̃+G)⁻¹K̃ · (K̃⁻¹GK̃⁻¹ m + K̃⁻¹ g)
        = K̃(K̃+G)⁻¹ (G K̃⁻¹ m + g)              ← the OLD-form m-update (see §10)
```

`update_variational_params` writes `m_b_new, V_b_new` back into `state`
(`eigenspace_model.py`).

---

## 10. ⚑ CONFLICT FLAG — the m-update: code ≠ corrected-derivation Newton step

This is the one genuine doc-vs-code (and code-vs-code) discrepancy in the cluster.
Both the code's own `TODO` (lines 34–39) and `ESTEP_MATH_ANALYSIS.md` raise it.

| | m-update |
|---|---|
| CORRECTED derivation (§5, `Estep_corrected_mderivation.tex`) | `m_new = m + K̃(K̃+G)⁻¹(g − m) = G(K̃+G)⁻¹m + K̃(K̃+G)⁻¹g` |
| GROUND-TRUTH CODE (§9, `eigenspace_estep.py`, = deprecated `utils.py:Estep`) | `m_new = V_new(G m + g) = K̃(K̃+G)⁻¹(G K̃⁻¹ m + g)` |

The two differ only in the first term:
```
code:   K̃(K̃+G)⁻¹ G K̃⁻¹ m
Newton: G(K̃+G)⁻¹ m
difference vanishes  ⇔  K̃ and G commute  ⇔  K̃ and (K̃+G)⁻¹ commute
```
In eigenspace `K̃_b = Λ` is diagonal but `G_b` is a full matrix, so they commute
only if the retained eigenvalues are (near-)equal or `G_b` is (near-)diagonal —
generally false. So the eigenspace **reduces** the discrepancy (if eigenvalues are
of similar scale) but does **not** eliminate it. The code's `TODO` states the
"correct" alternative would be `m_new = m + K̃ @ solve(K̃+G, g − m)`.

**Verification added by this consolidation — the discrepancy does NOT change the
converged posterior (both maps share the fixed point `m* = g`):**
```
Newton map:  T_N(g) = g + K̃(K̃+G)⁻¹(g − g) = g                         ✓
Code map:    T_C(g) = K̃(K̃+G)⁻¹(G K̃⁻¹ + I) g
                    = K̃(K̃+G)⁻¹ (G + K̃) K̃⁻¹ g = K̃ K̃⁻¹ g = g          ✓
```
Both are valid fixed-point iterations onto the same stationary condition
`m* = g = A Σ kᵢ(yᵢ − f̄ᵢ)`. They differ only in the **transient path** when a
single E-step is not run to convergence; with `n_estep` iterations both approach
`m* = g`. This resolves the archived doc's "Open Questions" (why the code works
despite the discrepancy): it is a different iteration map, not a different solution.
The **V-update is unambiguously correct** (§9), which is what the utility /
active-learning variance estimates depend on. `V` is also explicitly symmetrised.

**Recommendation status:** left as-is (works empirically; changing it risks
regressions). Flagged here because the code is ground truth and it does **not**
match the "corrected" m-derivation off-convergence.

---

## 11. How the E-step is driven: iteration, coupling, damping, divergence guard
`eigenspace_training.py` (the EM loop that calls `estep_eigenspace`)

`estep_eigenspace()` itself performs a **single, undamped, closed-form Newton
step**. Iteration, coupling, and stabilisation live in the driver:

- **Newton loop:** `for i_estep in range(n_estep): estep_eigenspace(model, r, f_mean)`.
  `n_estep` = E-step Newton iterations per EM iteration.
- **Coupling variable = `f_mean` (= f̄).** After each Newton step the posterior is
  recomputed (`posterior = model(X_train)` → `lambda_m, lambda_var`) and `f_mean`
  is refreshed via `compute_f_mean` (= `f̄ = exp(A λ_m + ½A² λ_var + λ₀)`), then fed
  into the next step's `g_b, G_b`. `A, λ₀` are detached (fixed) during the E-step.
- **Divergence guard (trust-region-style revert):** each iteration snapshots
  `m_b_prev, V_b_prev`; if after the step `f_mean.mean()` or `f_mean.max()` exceeds
  thresholds, or `f_mean` has NaNs, it **reverts** `(m_b, V_b)` (and `A, λ₀` if
  interleaved), recomputes `f_mean`, prints `"f_mean diverged … reverted"`, and
  `break`s the loop. This is the practical "damping/convergence" mechanism around
  the otherwise-full Newton step.
- **Interleaved F-step (optional):** with `interleave_fstep`, `A, λ₀` get a
  **damped** Newton update (`alpha = 0.25`, `damped_newton_update_A_lambda0`) at
  each E-step iteration (the paper's approach); otherwise the F-step optimises `A`
  after the E-step loop.

Redundancy note: `f̄` is computed both by `likelihoods.py:expected_firing_rate`
(from a posterior object) and by the loop's `compute_f_mean(λ_m, λ_var, A, λ₀)` —
identical formula, two entry points.

---

## 12. Notation table (symbol · meaning · source; CLASHES flagged)

| symbol | meaning | source / code name |
|---|---|---|
| `λ(x)`, `λᵢ` | latent GP tuning function (log-drive) at image | `lambda_m` (mean), latent |
| `λ₀` | scalar bias / baseline log-firing (**CLASH** with `λ` and `Λ`) | `lambda0` |
| `Λ` | diagonal matrix of kept `K̃` eigenvalues (**CLASH** with `λ`,`λ₀`) | `eigvals_b`, `K_tilde_b` |
| `A` | gain (positive, `A=exp(raw_A)`) (**CLASH** with lowercase `a`) | `likelihood.A`, `raw_A` |
| `a` | `K K̃⁻¹` in eigenbasis, (N,n_b) (**CLASH** with gain `A`) | `KKtilde_inv_b` |
| `yᵢ` | spike count for image i (**SYNONYM** `r`) | `r`, `target` |
| `f(xᵢ)` | Poisson rate `exp(A λᵢ + λ₀)` | — |
| `f̄ᵢ` | expected firing rate `exp(Aμᵢ+½A²σᵢ²+λ₀)` (**SYNONYM** `f_mean`, `compute_f_mean`) | `f_mean` |
| `m`, `V` | variational mean/cov over inducing values (M-dim) | `m`, `V` (standard form) |
| `m_b`, `V_b` | same, in eigenbasis (n_b-dim); `m_b=Bᵀm` | `state.m_b`, `state.V_b` |
| `μᵢ`, `σᵢ²` | posterior mean/var of `λᵢ` | `lambda_m`, `lambda_var`; `posterior.mean/.variance` |
| `K̃` (K_uu) | inducing kernel `k(Z̃,Z̃)`, (M,M) | `K_tilde` |
| `K̃_b` | `K̃` in eigenbasis = `diag(eigvals_b)` (DIAGONAL) | `state.K_tilde_b` |
| `kᵢ` | cross-cov `k(Z̃,xᵢ)`, (M,) | column of `K` |
| `K` | cross-kernel (N,M); `K_b = K B` | `K_b` (eigenbasis) |
| `B` | kept eigenvectors of `K̃`, (M,n_b) (**CLASH** w/ eigenbasis kernel `B`-matrix elsewhere) | `state.B` |
| `g` | `A Σᵢ kᵢ(yᵢ−f̄ᵢ)`, (M,) standard gradient | — |
| `g_b` | transformed: `K̃⁻¹g` in eigenbasis, (n_b,) | `g_b` |
| `G` | `A² Σᵢ f̄ᵢ kᵢkᵢᵀ`, (M,M) standard curvature (PSD) | — |
| `G_b` | transformed: `K̃⁻¹ G K̃⁻¹` in eigenbasis, (n_b,n_b) | `G_b` |
| `M` | number of inducing points | — |
| `N` | number of training images | — |
| `n_b` | retained eigen-dimension (≈10–11) (**CLASH-ish** with `M`,`N`) | `n_b`, `len(eigvals_b)` |
| `n_estep` | Newton iterations per EM iteration | `n_estep` |
| `H_m` | Hessian of L wrt `m`: `−K̃⁻¹(K̃+G)K̃⁻¹` | — |
| `L`, `D_KL` | ELBO, KL divergence | `expected_log_prob` (data part) |

**Synonym summary:** `y ↔ r ↔ target`; `f̄ ↔ f_mean ↔ compute_f_mean/expected_firing_rate`;
`μ,σ² ↔ lambda_m,lambda_var ↔ posterior.mean/variance`; `K̃ ↔ K_uu ↔ K_tilde`.
**Clash summary:** `λ` (latent) vs `λ₀` (scalar bias) vs `Λ` (eigenvalue diag);
`A` (scalar gain) vs `a` (matrix `KKtilde_inv_b`).

---

## 13. Scope exclusions (recorded, not covered here)

- **Spatiotemporal machinery:** no temporal covariance `C`, no Kronecker structure,
  no time-warping. This cluster is the spatial per-image latent only. Any temporal
  E-step generalisation is out of scope by directive.
- **Deprecated varGP:** repo-root `utils.py:varGP()/Estep()`, `Amp`, old torch-LBFGS,
  and paper-vs-vargp_old framing are excluded. The current E-step is
  `eigenspace_estep.py`; it *reproduces* the deprecated `Estep()` m-update
  numerically (§10) but is the ground-truth implementation.
- **F-step / M-step:** `A, λ₀` optimisation (`eigenspace_fstep.py`) and kernel
  hyperparameter optimisation (`eigenspace_mstep.py`, `eigenspace_gradients.py`) are
  separate clusters; only their coupling to the E-step (fixed `A,λ₀`; `f_mean`
  refresh; interleaved damped F-step) is noted here (§11).
- **Prediction / utility / active-image selection:** `predict_eigenspace`,
  `standard_utility` are downstream consumers of `(m_b, V_b)`; not derived here.
