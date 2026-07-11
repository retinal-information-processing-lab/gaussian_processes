# 04 — M-step, F-step, and Analytical ELBO Gradients

Consolidated theory/math note for the **spatial** Gaussian-Process active-learning
model that learns retinal ganglion cell receptive fields (RFs) from spike counts to
natural images. This note covers the **M-step / F-step** of the EM-style training loop
and the **analytical gradients of the ELBO w.r.t. kernel hyperparameters** (as opposed
to autograd), including the vector–Jacobian-product (VJP) formulation and where the
arc-cosine kernel's own parameter gradients plug into the ELBO gradient.

**Scope of this note.** Spatial GP only. Spatiotemporal machinery (temporal covariance
`C_temporal`, Kronecker structure, time warping) is **excluded** — none of it appears in
the files distilled here; the covariance `C` is purely spatial (pixel × pixel). The
deprecated `vargp_old` engine (repo-root `utils.py`'s `varGP()` / `acosker()` / `Estep()`,
the old ScaleKernel handling of `Amp`, torch-LBFGS-timing framing) is **excluded** as an
engine — but the ELBO / hyperparameter-gradient **math** it contained is current and is
kept here. The current, ground-truth implementations are:

| Concern | Current file (ground truth) |
|---|---|
| E-step (Newton on `m_b, V_b`) | `eigenspace_estep.py:estep_eigenspace` |
| **F-step** (firing-rate params `A, λ₀`) | `eigenspace_fstep.py` |
| **M-step** (kernel hyperparameters) | `eigenspace_mstep.py` |
| M-step analytical gradient chain | `eigenspace_gradients.py` |
| Kernel-level analytical gradients (autograd.Function) | `analytical_gradients.py` (Jacobian), `analytical_gradients_vjp.py` (VJP) |
| Training loop (composes E/M/F) | `eigenspace_training.py:train_eigenspace` |
| Likelihood helpers (`λ₀` closed form, `f_mean`) | `utils.py` (this dir) |

All paths are relative to
`…/Spatial_GP_repo/scripts/gpytorch_porting/`.

> **Two distinct gradient systems live in the codebase** (flagged in
> `eigenspace_gradients.py`'s own module docstring, lines 10-19). They share the dC/dK
> *formulas* but have different APIs and different callers. Keeping them apart is
> essential to reading this note:
>
> - **System 1 — kernel-level** (`analytical_gradients.py` = Jacobian,
>   `analytical_gradients_vjp.py` = VJP). Both are `torch.autograd.Function` wrappers
>   selected by `ArcCosineKernel(..., gradient_mode='jacobian'|'vjp'|'autograd')`. Their
>   backward takes `grad_output = dL/dK` and returns per-hyperparameter grads. Used when
>   the kernel is a node in a general autograd graph.
> - **System 2 — M-step level** (`eigenspace_gradients.py` +
>   `mstep_eigenspace_analytical`). **Not** an autograd.Function. Runs under
>   `@torch.no_grad()`, computes the *entire* loss-gradient chain analytically (through
>   the eigenspace posterior moments, the Poisson log-likelihood, and the KL), and writes
>   `param.grad` directly. **This is the system that actually differentiates the ELBO
>   w.r.t. hyperparameters during training.**
>
> System 2 does **not** use the VJP. The VJP (System 1) cannot be used for the LBFGS
> M-step — see §7.

---

## 1. The objective: the ELBO (in the eigenspace)

Sources: `eigenspace_training.py:compute_elbo_eigenspace`, and the closures in
`eigenspace_mstep.py`.

Per stimulus image `i` (out of `N` shown), the model predicts an expected firing rate
through an exponential (Poisson) link:

```
f_mean_i = exp( A·λ_m,i + ½·A²·λ_var,i + λ₀ )          (utils.py:compute_f_mean)
```

- `λ_m,i`, `λ_var,i` — posterior mean / variance of the latent GP value at image `i`.
- `A` — firing-rate **gain**; `λ₀` — firing-rate **baseline/offset**. Both are
  parameters of the Poisson likelihood.
- The `½·A²·λ_var` term is the log-normal correction: `E[exp(A·λ)] = exp(A·μ + ½A²·σ²)`.

The ELBO (maximized) is expected-Poisson-log-likelihood minus a KL to the GP prior:

```
ELBO = log_lik − KL

log_lik = Σ_i [ r_i·(A·λ_m,i + λ₀) − f_mean_i ]                    (Poisson evidence)

KL = ½·( tr(K̃⁻¹ V) + mᵀ K̃⁻¹ m − n_b + log|K̃| − log|V| )         (q(u)=N(m,V) ‖ p(u)=N(0,K̃))
```

`r_i` = spike count for image `i` (raw counts in a fixed post-stimulus window, not Hz).
Training minimizes `loss = −ELBO`.

**Eigenspace representation.** Everything is carried in an eigenbasis `B` of the inducing
kernel. Subscript `b` = "in the eigenspace basis":
- `K̃_b = Bᵀ K̃ B` (inducing/inducing kernel, `n_b × n_b`), `K_b = K B` (train/inducing,
  `N × n_b`), `Kvec` = diagonal self-kernel `K(x_i,x_i)` (`N`).
- `m_b`, `V_b` — variational mean/covariance in the basis (`n_b`, `n_b × n_b`).
- `eigvals_b` — eigenvalues; `n_b = len(eigvals_b)` = effective inducing dimension.

Posterior moments at the training images (`eigenspace_gradients.py:compute_lambda_moments_and_gradients`,
mirrored in the M-step closures and `predict_eigenspace`):

```
a       = K_b · K̃_b⁻¹                                     (N × n_b, "projection")
λ_m     = a · m_b                                          (N,)
λ_var   = Kvec + rowsum( a ⊙ (a·(V_b − K̃_b)) )            (N,)
        = Kvec + diag( a·(V_b − K̃_b)·aᵀ )
λ_var   = clamp(λ_var, min = lambda_var_clamp)            (numerical floor)
```

---

## 2. The three steps — precise definitions from the code

The training loop is coordinate ascent on the ELBO over three parameter blocks. Each step
optimizes one block while **holding the other two fixed**.

| Step | Optimizes | Holds fixed | Method | File |
|---|---|---|---|---|
| **E** | variational `m_b, V_b` | kernel hyperparams; `A, λ₀` | closed-form Newton (no autograd) | `eigenspace_estep.py` |
| **F** | firing-rate `A` (`λ₀` closed-form given `A`) | `m_b, V_b` (via `λ_m, λ_var`); kernel hyperparams | LBFGS on `raw_A` + analytic `dL/dA`, or damped Newton on `(A,λ₀)` | `eigenspace_fstep.py` |
| **M** | kernel hyperparams `{σ₀, Amp, ε₀ₓ, ε₀ᵧ, raw_m2log2beta, raw_mlog2rho2}` | `m_b, V_b`; `A, λ₀`; eigenbasis `B` | LBFGS + autograd **or** analytical gradients | `eigenspace_mstep.py` |

### 2.1 What "M" is — the hyperparameter step

The **M-step** maximizes the ELBO w.r.t. the **kernel/prior hyperparameters** while the
variational parameters `(m_b, V_b)`, the firing-rate parameters `(A, λ₀)`, **and the
eigenbasis `B`** are held fixed. The six hyperparameters (all owned by `ArcCosineKernel`):

- `σ₀` (`sigma_0`) — bias variance. **`σ₀ = exp(raw_sigma_0)`** (kernels.py:207).
- `Amp` — kernel amplitude. **`Amp = softplus(raw_Amp)`** (kernels.py:219).
- `ε₀ₓ, ε₀ᵧ` (`eps_0x`, `eps_0y`) — RF **center** position (optimized as raw values).
- `raw_m2log2beta` — RF **size**, log-parameterized: `= −2·log(2β)` ⇒ `exp(raw_m2log2beta)=1/(4β²)`.
- `raw_mlog2rho2` — RF **smoothness**, log-parameterized: `= −log(2ρ²)` ⇒ `exp(raw_mlog2rho2)=1/(2ρ²)`.

`B` is deliberately **not** recomputed inside the M-step (it is an approximation to keep
the M-step cheap and stable); it is re-synced to the new hyperparameters at the *start* of
the next EM iteration via `model.recompute_eigenspace()`. Two implementations:
`mstep_eigenspace_autograd` (LBFGS + `torch.autograd.grad`) and
`mstep_eigenspace_analytical` (LBFGS + hand-written gradients). Selected by
`use_analytical_mstep` in `train_eigenspace`.

### 2.2 What "F" is — the firing-rate step ("F" = the firing-rate function `f`)

The **F-step** optimizes the **firing-rate (Poisson-likelihood) parameters** `A` and `λ₀`
that define the rate nonlinearity `f_mean = exp(A·λ + ½A²·λ_var + λ₀)`, holding the
variational parameters (through fixed `λ_m, λ_var`) and the kernel hyperparameters fixed.
Determined directly from `eigenspace_fstep.py`'s docstring (lines 4-6): *"optimization of
firing rate parameter A while holding variational parameters (m_b, V_b) and kernel
hyperparameters fixed. lambda0 is computed analytically given A."* So **"F" denotes the
firing-rate function `f`** — the step that fits the response nonlinearity, distinct from
the M ("maximization"/hyperparameter) step and the E ("expectation"/variational) step. In
this codebase `A` is the only free variable of the F-step; `λ₀` is not searched over but
solved in closed form for the current `A`.

> **Doc-coverage note:** none of the four source docs (`ANALYTICAL_GRADIENTS_MATH.md`,
> `gradients.md`, `MSTEP_ANALYTICAL_HANDOFF.md`, `VJP_ANALYTICAL_GRADIENTS.md`) define or
> discuss the F-step at all — they are entirely about kernel/M-step gradients. The F-step
> definition here is taken **only** from the code (`eigenspace_fstep.py`). See §9-D.

---

## 3. The F-step in full (`eigenspace_fstep.py`)

### 3.1 Closed-form `λ₀` given `A`  (`utils.py:lambda0_given_A`)

`λ₀` maximizes the expected log-likelihood in closed form. From `∂L/∂λ₀ = 0`:

```
∂/∂λ₀ Σ_i[ r_i·λ₀ − exp(λ₀)·exp(A·λ_m,i + ½A²·λ_var,i) ] = 0
⇒  Σ_i r_i = exp(λ₀)·Σ_i exp(A·λ_m,i + ½A²·λ_var,i)
⇒  λ₀ = log(Σ_i r_i) − log( Σ_i exp(A·λ_m,i + ½A²·λ_var,i) )
```

(Requires `Σ r_i > 0`; raises otherwise — a cell with zero spikes is a data problem.)
This is re-evaluated whenever `A` changes (inside the F-step closure and after the step).

### 3.2 Standard F-step — LBFGS on `A` with analytic `dL/dA`

`fstep_eigenspace(model, r, λ_m, λ_var, n_fstep, lr, …)`:

1. Update `λ₀ ← lambda0_given_A(A, r, λ_m, λ_var)`.
2. LBFGS over `raw_A = log A` (`n_fstep` iters). Each closure:
   - re-solves `λ₀` for the current `A`; rejects the step (returns `+inf`) if `λ₀`
     over/underflows or if `f_mean` blows past `f_mean_{mean,max}_threshold`;
   - loss returned is `−log_lik`;
   - **analytical gradient** (no autograd):
     ```
     dL/dA      = r·λ_m − (λ_m + A·λ_var)·f_mean          (dot products over images)
     dL/d(logA) = A · dL/dA
     raw_A.grad = − dL/d(logA)          (sign: LBFGS minimizes −log_lik)
     ```
3. After the step: clamp likelihood params, recompute `λ₀`; **revert** `raw_A, λ₀` to the
   pre-step values if the final `λ₀` overflows (guards the exp link against a too-large `A`).

Note `λ_m, λ_var` are **inputs** — they are frozen for the whole F-step, which is why the
F-step "holds the variational parameters fixed".

### 3.3 Interleaved F-step — joint damped Newton on `(A, λ₀)`

`damped_newton_update_A_lambda0(...)` (matches the reference paper's `updateA`). Used when
`interleave_fstep=True`: called *inside* the E-step loop after every Newton iteration, so
`A, λ₀` track `(m, V)` as they move. Let `ψ = [A, λ₀]`, and per image:

```
f_i   = exp(A·μ_i + ½A²·var_i + λ₀)
d_i   = μ_i + A·var_i                          (= ∂/∂A of the exponent)

gradient  R = [ r·μ − d·f ,  Σr − Σf ]
Hessian   H = − [[ var·f + d²·f ,  d·f ],
                 [ d·f          ,  Σf  ]]       (negative-definite; log-lik is concave)

update    ψ ← ψ − α · solve(H, R),   α = 0.25 (damping)
```

Iterates (≤`max_iter`, tol on `Σ|R|`) with the same `f_mean` stability rejections; writes
`raw_A = log A`, `λ₀` back and returns the updated `f_mean`. When interleaved, the standalone
`fstep_eigenspace` is skipped (`train_eigenspace` lines 373-377).

---

## 4. The M-step analytical gradient chain (`eigenspace_mstep.py` + `eigenspace_gradients.py`)

This is the heart of the cluster: the **analytical** ELBO gradient w.r.t. the six kernel
hyperparameters, used by `mstep_eigenspace_analytical`. It runs the exact same math the
legacy varGP engine used, but under `@torch.no_grad()` and wired into GPyTorch parameters.

**Why analytical instead of autograd (motivation).**
- **Speed / memory.** The closure is `@torch.no_grad()` — it never builds an autograd
  graph. The autograd M-step (`mstep_eigenspace_autograd`) recomputes the kernels *with
  grad tracking* and calls `torch.autograd.grad` every LBFGS line-search evaluation, which
  is what made the legacy autograd path slow. (The source docs quote ~17 ms vs the
  analytical/Jacobian path; those absolute numbers are legacy-engine framing and are not
  reproduced as current benchmarks.)
- **Reference match / stability.** The analytical path reproduces the varGP formulas
  bit-for-bit (the "CRITICAL FIX" comments in `eigenspace_mstep.py` lines 251-256, 280-283
  note where it matches the reference: full-matrix `solve()` for `K̃_b⁻¹`, Cholesky
  log-det, symmetrization), giving controlled, inspectable numerics rather than autograd's.

### Order of the closure (per LBFGS evaluation)

Guardrails (bounds, symmetrization, `f_mean` cap, Cholesky-fail, NaN/Inf) are inline —
listed in §6.

**Step 0 — reject out-of-bounds trial** (`kernel.params_in_bounds()` → `+inf`).

**Step 1 — `C` and `dC`** (`compute_C_and_gradients(kernel)`; this is the *hookup point*
for the arc-cosine kernel's own parameter gradients — see §5). Returns the **masked**
spatial covariance `C` (pixels with negligible RF weight dropped), the pixel `mask`, and
`dC = {Amp, eps_0x, eps_0y, raw_m2log2beta, raw_mlog2rho2}`. Inputs are masked:
`X_masked = X[:, mask]`.

**Step 2 — `K, K̃, Kvec` and their `dK, dK̃, dKvec`** via
`compute_kernel_and_gradients(...)` (the arc-cosine kernel evaluated three times):
`K̃ = k(X̃,X̃)`, `K = k(X, X̃)`, `Kvec = diag k(X,X)`. Each returns a `dict` keyed by the
six hyperparameters (`sigma_0` added on top of the five C-params). Formulas in §5.

**Step 3 — project to the eigenspace** (basis `B` fixed):
```
K̃_b = Bᵀ K̃ B ;  K_b = K B                       (symmetrize K̃, K̃_b)
K̃_b⁻¹ = solve(K̃_b, I)                            (full-matrix inverse, matches varGP)
dK̃_b[θ] = Bᵀ·dK̃[θ]·B ;  dK_b[θ] = dK[θ]·B        (project every gradient matrix too)
```

**Step 4 — posterior moments and their hyperparameter gradients**
(`compute_lambda_moments_and_gradients`). With `a = K_b·K̃_b⁻¹`:

```
λ_m   = a·m_b
λ_var = Kvec + rowsum( a ⊙ (a·(V_b − K̃_b)) )                 (clamped)

For each hyperparameter θ:
  da[θ]        = ( dK_b[θ] − a·dK̃_b[θ] ) · K̃_b⁻¹                       (N × n_b)
  dλ_m[θ]      = da[θ] · m_b                                            (N,)
  dλ_var[θ]    = dKvec[θ]
                 + 2·diag( da[θ]·V_b·aᵀ )        = 2·einsum('ij,ji->i', da, V_b aᵀ)
                 − diag( dK_b[θ]·aᵀ )            = − einsum('ij,ij->i', dK_b, a)
                 − diag( K_b·da[θ]ᵀ )            = − einsum('ij,ij->i', K_b, da)
```

**Step 5 — the loss** `= −log_lik + KL` (formulas in §1); `f_mean` from `λ_m, λ_var, A, λ₀`.

**Step 6 — loss gradients** (`compute_loss_gradients`), split into the log-likelihood and
KL contributions. For each θ:

```
# expected-Poisson-log-likelihood term
dloglik[θ] = A·(r·dλ_m[θ]) − A·(f_mean·dλ_m[θ]) − ½·A²·(f_mean·dλ_var[θ])

# KL term:  with  c = V_b·K̃_b⁻¹ ,  b = K̃_b⁻¹·m_b ,  B = dK̃_b[θ]·K̃_b⁻¹
dKL[θ]     = ½·tr(B) − ½·tr(c·B) − ½·bᵀ·(B·m_b)

dL[θ]      = − dloglik[θ] + dKL[θ]        # gradient of the loss (= −ELBO)
```

(The `dloglik` formula is `∂/∂θ` of `Σ_i[r_i(Aλ_m+λ₀) − f_mean_i]` at fixed `A,λ₀`:
the `r·(A dλ_m)` term from the linear part, and `−A·f_mean·dλ_m − ½A²·f_mean·dλ_var` from
`∂f_mean/∂θ = f_mean·(A dλ_m + ½A² dλ_var)`.)

**Step 7 — write `param.grad`, applying the constraint-transform Jacobian**
(`eigenspace_mstep.py` lines 320-330). `dL[θ]` above is `∂loss/∂(value)`; GPyTorch
optimizes the **raw** parameter, so we multiply by `d(value)/d(raw)`:

```
raw_sigma_0.grad   = dL['sigma_0'] · σ₀                    # σ₀ = exp(raw)  ⇒ dσ₀/draw = σ₀
raw_Amp.grad       = dL['Amp']     · sigmoid(raw_Amp)      # Amp = softplus(raw) ⇒ d/draw = sigmoid(raw)
eps_0x.grad        = dL['eps_0x']                          # optimized directly (no transform)
eps_0y.grad        = dL['eps_0y']
raw_m2log2beta.grad= dL['raw_m2log2beta']                  # dC already differentiated the raw
raw_mlog2rho2.grad = dL['raw_mlog2rho2']
```

`eps_0x/eps_0y` and the two `raw_m*` params need **no** transform factor: for `eps₀` the
value *is* the raw; for `raw_m2log2beta`/`raw_mlog2rho2` the `dC` formulas (§5.1) are
already derivatives w.r.t. the raw log-param. Only `σ₀` and `Amp` carry a transform Jacobian
because their `dC`/`dK` were taken w.r.t. the *values* `σ₀`, `Amp`.

`ArcSineKernel` is explicitly rejected here (`eigenspace_mstep.py` lines 182-188): the
analytical formulas are specific to the arc-cosine `K = M·J(θ)/π`; other kernels must use
the autograd M-step.

> **Where the "compute dK once, cache it" story from the docs is only half-true (§9-C):**
> the source docs (`MSTEP_ANALYTICAL_HANDOFF.md` §"Why VJP cannot be used",
> `gradients.md` §6) sketch a pattern where `K, dK` are computed **once outside** the LBFGS
> closure and a cached `dK` is reused as `grad[key] = (dL_dK·dK[key]).sum()`. The shipped
> `mstep_eigenspace_analytical` does **not** do this: it recomputes `C/dC/K/dK` **inside**
> the closure on every line-search evaluation (correct — the parameters move each
> evaluation, so `dK` genuinely changes), and it does **not** reduce via a single
> `(dL_dK·dK).sum()` — it runs the full moment→log-lik/KL chain of §4. The essential win
> (`@torch.no_grad()`, explicit reference-matching formulas) holds; the "cached outside /
> single dL/dK contraction" wording is a simplification that does not match the code.

### 4.1 Diagonal / eigenvalue branch (present but unused by the shipped M-step)

`compute_lambda_moments_and_gradients` and `compute_loss_gradients` both accept a **1-D**
`K̃_b⁻¹` (diagonal = eigenvalues, element-wise ops) **or** a full 2-D inverse. The shipped
`mstep_eigenspace_analytical` always passes the **full** `solve()`-based inverse (the
"CRITICAL FIX" matching varGP), so the diagonal branch is currently dead code retained for
generality. Do not assume `K̃_b⁻¹` is diagonal in the M-step.

---

## 5. Where the arc-cosine kernel's parameter gradients plug in (chain-rule hookup)

*(The arc-cosine kernel and its `dC`/`dK` derivations are owned elsewhere; here only the
hookup — how those matrices enter the ELBO gradient.)* The chain is:

```
hyperparameter θ  →  dC[θ]        (Step 1, compute_C_and_gradients)
                  →  dK[θ], dK̃[θ], dKvec[θ]   (Step 2, compute_kernel_and_gradients)
                  →  dK_b[θ], dK̃_b[θ]          (Step 3, project by B)
                  →  dλ_m[θ], dλ_var[θ]        (Step 4)         ┐
                  →  dK̃_b[θ] also feeds dKL[θ] (Step 6)         ├→ dL[θ]  →  param.grad
                  →  dloglik[θ], dKL[θ]        (Step 6)         ┘
```

### 5.1 `dC` — the spatial covariance and its five gradients
(`compute_C_and_gradients`, masked pixels). With `β_f = exp(raw_m2log2beta) = 1/(4β²)`,
`ρ_f = exp(raw_mlog2rho2) = 1/(2ρ²)`, pixel coords `(xᵢ,yᵢ)∈[−1,1]²`:

```
d_center,i   = (xᵢ − ε₀ₓ)² + (yᵢ − ε₀ᵧ)²
logα_i       = −β_f · d_center,i ;      α_i = exp(logα_i)          (locality weight)
d_pair,ij    = (xᵢ − xⱼ)² + (yᵢ − yⱼ)²
logCsm_ij    = −ρ_f · d_pair,ij ;       Csm_ij = exp(logCsm_ij)    (smoothness)
C            = Amp · α[:,None] · Csm · α[None,:]        (symmetrized)

dC/dAmp            = C / Amp
dC/dε₀ₓ            = 2·β_f·C·( xᵢ + xⱼ − 2ε₀ₓ )
dC/dε₀ᵧ            = 2·β_f·C·( yᵢ + yⱼ − 2ε₀ᵧ )
dC/d(raw_m2log2beta) = C·( logα_i + logα_j )           (already w.r.t. the raw log-param)
dC/d(raw_mlog2rho2)  = C·logCsm                        (already w.r.t. the raw log-param)
```

### 5.2 `dK` from `dC` — the arc-cosine kernel gradients (full matrix)
(`compute_kernel_and_gradients`, `diag=False`). Forward intermediates, with `x1,x2` the
masked stimulus vectors and `σ₀² = sigma_0²`:

```
V1_i = x1_iᵀ C x1_i + σ₀² ;  V2_j = x2_jᵀ C x2_j + σ₀² ;  X1=√V1, X2=√V2 ;  X1X2 = X1⊗X2
x1x2 = x1ᵀ C x2 + σ₀²
cosδ = clamp( x1x2 / (X1X2+ε), −1+ε, 1−ε ) ;  δ = arccos(cosδ) ;  sinδ = √(1−cos²δ)
J    = ( sinδ + (π − δ)·cosδ ) / π
K    = X1X2 · J
```

For each C-dependent θ (given `dC[θ]`):
```
dX1_i  = ½·(x1_iᵀ dC x1_i)/X1_i ;   dX2_j = ½·(x2_jᵀ dC x2_j)/X2_j
dX1X2  = dX1[:,None]·X2 + X1[:,None]·dX2
dcosδ  = ( x1ᵀ dC x2  − cosδ·dX1X2 ) / X1X2
dJ     = −(δ − π)·dcosδ / π
dK[θ]  = X1X2·dJ + dX1X2·J
```

For `σ₀` (appears in `K` directly, not through `C`):
```
dX1X2_σ = σ₀²·( X2/X1 + X1/X2 )
dcosδ_σ = ( 2σ₀² − cosδ·dX1X2_σ ) / X1X2
dJ_σ    = −(δ − π)·dcosδ_σ / π
dK[σ₀]  = ( X1X2·dJ_σ + dX1X2_σ·J ) / σ₀
```
(The `/σ₀` cancels an implicit `σ₀` factor built into `dX1X2_σ = σ₀²·(…)` and
`dcosδ_σ`'s `2σ₀²`, so `dK[σ₀]` is the true `∂K/∂σ₀` w.r.t. the *value* σ₀.)

### 5.3 Diagonal (`Kvec`) gradients (`diag=True`)
```
Kvec_i     = x1_iᵀ C x1_i + σ₀²
dKvec[σ₀]  = 2·σ₀                                  (true ∂Kvec/∂σ₀)
dKvec[θ]_i = x1_iᵀ dC[θ] x1_i     (for the five C-params)
```

> **Cross-system convention mismatch (flag, §9-E):** System 1's Jacobian file
> (`analytical_gradients.py`, `diag=True`) returns `dK[σ₀] = 2·σ₀·1/σ₀ = 2` (it
> pre-divides by σ₀ "for consistency" with the legacy `acosker`), whereas System 2's
> `eigenspace_gradients.py` returns `2·σ₀`. **The M-step (System 2) is internally
> consistent** — its diagonal and full-matrix `σ₀` gradients are both true `∂/∂σ₀`, and the
> single `×σ₀` transform is applied once at Step 7. The factor-σ₀ difference is a
> per-system bookkeeping convention (each applies its own downstream transform), not a bug
> in the M-step. Do not "unify" them without tracing each system's transform handling.

---

## 6. Numerical guardrails (M-step closure)

Inline in `mstep_eigenspace_analytical` (the "8 guardrails from varGP"):

1. bounds check `params_in_bounds()` → `+inf` reject;
2. `cosδ` clamped to `[−1+ε, 1−ε]` before `arccos`; `sinδ = √(clamp(1−cos²δ, min=ε))`
   (`compute_kernel_and_gradients`); division by `X1X2` uses `+ε` jitter (`ε = 1e-7`);
3. `K̃` and `K̃_b` symmetrized `(M+Mᵀ)/2`;
4. `K̃_b⁻¹` via `torch.linalg.solve` (not eigen-inverse);
5. `f_mean.mean() > f_mean_mean_threshold` or any NaN → reject;
6. `log|K̃_b|` via Cholesky (`2·Σlog diag L`); Cholesky failure → reject;
7. `loss` NaN/Inf → reject; `log|V_b|` sign ≤ 0 → reject;
8. gradient NaN/Inf counted and warned.

`C` symmetrized in `compute_C_and_gradients`; `λ_var` floored at `lambda_var_clamp`;
LBFGS `IndexError/RuntimeError` (line-search crash on NaN) caught → keep pre-step params;
`kernel.clamp_hyperparameters()` after the step. Constants: `ε=1e-7`, eigenvalue threshold
`max(eigmax·1e-4, 1e-4)`; float64 throughout (kernel values can reach ~10⁴).

---

## 7. The VJP formulation (System 1, kernel-level) — and why it is *not* the M-step

`analytical_gradients_vjp.py:ArcCosineVJPGradients` (a `torch.autograd.Function`). It
computes the **same** `dL/d(hyperparameters)` as the Jacobian version but far more cheaply,
by never materializing the per-hyperparameter `dK/dθ` matrices. Instead it computes
`dL/dC` **once** and contracts it against the (cheap) `dC/dθ` structure.

**Motivation (speed).** The Jacobian path materializes 5 `dC` (nx×nx) and 5 `dK` (n1×n2)
matrices → ~15 large matmuls per call, `O(15·nx²·n)`. The VJP is one backward sweep,
`O(nx²·n) + O(nx²)` — the docs quote ~10–15× faster for the gradient, and Jacobian ≈73 ms
vs VJP ≈16 ms (≈ autograd) in the legacy benchmark. **Stability** is inherited from the
same clamps as the forward pass.

**Backward chain.** Given `G = dL/dK` (shape n1×n2), reusing forward intermediates
`X1,X2,X1X2,cosδ,δ,J,x1Cx2,α,S,C,d_center,d_pair` and `Cx1,Cx2`:

```
# K → J, X1X2
dL/dJ        = G ⊙ X1X2
dL/dX1X2|K   = G ⊙ J
# J → cosδ            (dJ/dcosδ = (π − δ)/π)
dL/dcosδ     = dL/dJ ⊙ (π − δ)/π
# cosδ → x1Cx2, X1X2   (cosδ = x1Cx2 / X1X2)
dL/dx1Cx2    = dL/dcosδ / X1X2
dL/dX1X2     = G⊙J  −  dL/dcosδ ⊙ cosδ / X1X2
# X1X2 → X1, X2 → V1, V2
dL/dX1 = dL/dX1X2 · X2 ;  dL/dX2 = dL/dX1X2ᵀ · X1
dL/dV1 = dL/dX1 / (2·X1) ;  dL/dV2 = dL/dX2 / (2·X2)
# V1,V2,x1Cx2 → C   (ONE nx×nx matrix, computed once; xᵀCx ⇒ outer products)
dL/dC  = x1·diag(dL/dV1)·x1ᵀ + x2·diag(dL/dV2)·x2ᵀ + x1·(dL/dx1Cx2)·x2ᵀ     (symmetrized)
# C → hyperparameters   (C = Amp·α ⊗ α ⊙ S)
dL/dAmp      = (dL/dC ⊙ C).sum() / Amp
dL/dα        = 2·( (dL/dC ⊙ Amp ⊙ S) · α )
dL/dbeta_raw = ( dL/dα ⊙ α ⊙ (−d_center) · β ).sum()          # β = exp(beta_raw)
dL/dε₀ₓ      = ( dL/dα ⊙ α · 2β · (x − ε₀ₓ) ).sum() ;  dL/dε₀ᵧ analogous
dL/dS        = dL/dC ⊙ Amp ⊙ (α ⊗ α)
dL/drho_raw  = ( dL/dS ⊙ S ⊙ (−d_pair) · ρ² ).sum()           # ρ² = exp(rho_raw)
dL/dσ₀       = 2σ₀·( Σ dL/dV1 + Σ dL/dV2 + Σ dL/dx1Cx2 )
```

Note the VJP returns `dL/dbeta_raw`, `dL/drho_raw` directly (chain through `β=exp(raw)`,
`ρ²=exp(raw)` gives the extra `·β`, `·ρ²`), consistent with the M-step's `dC` w.r.t. the raw
log-params. `σ₀`/`Amp` VJP grads are w.r.t. the **values**; the caller applies the transform
Jacobian (the VJP's own test uses `sigmoid(raw_Amp)` for Amp and treats σ₀ as fed directly).

**Why the VJP is *not* used for the LBFGS M-step** (docs + `eigenspace_mstep` design). The
VJP needs `dL/dK` (= `G`) available at backward time. The M-step's loss is not a plain
function of `K`; it flows `K → (a, λ_m, λ_var) → log_lik/KL` with the eigenspace projection
and the Poisson likelihood in between, and LBFGS's strong-Wolfe line search re-evaluates the
loss (hence a *different* `dL/dK`) many times per step. The M-step therefore materializes the
`dK/dθ` matrices and runs the explicit moment→log-lik/KL chain (§4) rather than a single
`dL/dK` VJP contraction. In short: **System 1/VJP differentiates `K → loss` when the kernel
is one autograd node; System 2/M-step differentiates the whole eigenspace-ELBO analytically
and never forms a single `dL/dK`.**

---

## 8. Training loop composition (`eigenspace_training.py:train_eigenspace`)

EM-style coordinate ascent. One iteration (`for iteration in 1..n_iterations−1`):

1. **Re-sync eigenspace** (only if `n_mstep>0 and iteration>1`):
   `model.recompute_eigenspace()` rebuilds `B` from the hyperparameters the *previous*
   M-step changed, then recomputes `λ_m, λ_var, f_mean`. (This is where the M-step's
   deferred `B` update lands.)
2. **E-step** (`n_estep` Newton iters): `estep_eigenspace(model, r, f_mean)` updates
   `m_b, V_b` in closed form (no autograd), holding kernel + `A,λ₀` fixed. Re-reads
   `λ_m,λ_var` after each. If `interleave_fstep`: run
   `damped_newton_update_A_lambda0` after every E-iter (fuses the F-step in, §3.3). Divergence
   (`f_mean` over threshold / NaN) → revert `m_b,V_b` (and `A,λ₀` if interleaved), break.
3. **F-step** (only if **not** `interleave_fstep`):
   `fstep_eigenspace(model, r, λ_m, λ_var, n_fstep, lr_f, …)` optimizes `A` (+ closed-form
   `λ₀`), holding `m_b,V_b` (frozen `λ_m,λ_var`) and the kernel fixed (§3.2).
4. **Metrics + ELBO + early-stopping — computed BEFORE the M-step** (deliberate,
   lines 392-398): the M-step changes kernel params without recomputing `B`, so all logged
   quantities must be read while the model state is self-consistent. Early stopping is on the
   **ELBO** (`es_metric='elbo'`, the only supported metric as of Apr 2026; `'none'` disables);
   patience with best-restore.
5. **M-step** (only if `n_mstep>0 and iteration < n_iterations−1`): `mstep_eigenspace_analytical`
   (if `use_analytical_mstep`) else `mstep_eigenspace_autograd`, optimizing the six
   hyperparameters with `m_b,V_b`, `A,λ₀`, **and `B`** fixed. `B` is intentionally *not*
   recomputed here — it is re-synced at step 1 of the next iteration.

**What is fixed in each step (summary):**

| Step | Free | Fixed |
|---|---|---|
| E | `m_b, V_b` | kernel hyperparams; `A, λ₀`; `B` |
| F | `A` (`λ₀` closed-form) | `m_b, V_b` (via `λ_m,λ_var`); kernel hyperparams |
| M | `σ₀, Amp, ε₀ₓ, ε₀ᵧ, raw_m2log2beta, raw_mlog2rho2` | `m_b, V_b`; `A, λ₀`; `B` (until next iter) |

---

## 9. Notation map, conflicts, and stale-doc flags

### A. Symbol → code map

| Math symbol | Code name | Meaning | Source |
|---|---|---|---|
| `A` | `likelihood.A` / `raw_A=log A` | firing-rate gain | fstep, utils |
| `λ₀` | `likelihood.lambda0` | firing-rate baseline/offset | fstep, utils |
| `f_mean` | `f_mean` | expected Poisson rate `exp(Aλ_m+½A²λ_var+λ₀)` | utils:compute_f_mean |
| `r_i` | `r` | spike counts (raw, fixed window) | all |
| `λ_m, λ_var` | `lambda_m, lambda_var` | GP posterior mean/var per image | training, gradients |
| `a` | `a` | projection `K_b·K̃_b⁻¹` | gradients |
| `m_b, V_b` | `state.m_b, state.V_b` | variational mean/cov (eigenspace) | estep, mstep |
| `B` | `state.B` | eigenbasis of inducing kernel | training, mstep |
| `n_b` | `n_b=len(eigvals_b)` | effective inducing dim | all |
| `K̃, K, Kvec` | `K_tilde, K, Kvec` | inducing/cross/diag kernels | mstep, gradients |
| `K̃_b, K_b` | `K_tilde_b, K_b` | …projected by `B` | mstep |
| `σ₀` | `sigma_0` (`raw_sigma_0`) | bias variance; **`σ₀=exp(raw)`** | kernels.py:207 |
| `Amp` | `Amp` (`raw_Amp`) | amplitude; **`Amp=softplus(raw)`** | kernels.py:219 |
| `ε₀ₓ, ε₀ᵧ` | `eps_0x, eps_0y` | RF center | kernels, gradients |
| `β` (RF size) | `raw_m2log2beta = −2log(2β)` | `exp(raw)=β_f=1/(4β²)` | kernels.py:309 |
| `ρ` (smoothness) | `raw_mlog2rho2 = −log(2ρ²)` | `exp(raw)=ρ_f=1/(2ρ²)` | kernels.py:326 |
| `J(δ)` | `J` | angular term `(sinδ+(π−δ)cosδ)/π` | gradients |
| `δ, cosδ` | `delta, cosdelta` | arc-cosine angle | gradients |

### B. Notation clashes / synonyms (be careful)

- **`dK[θ]` dict keys differ across the two systems.** System 1 (`analytical_gradients.py`)
  uses string keys **`'-2log2beta'`, `'-log2rho2'`**; System 2 (`eigenspace_gradients.py`)
  uses **`'raw_m2log2beta'`, `'raw_mlog2rho2'`**. Same parameters, different spellings.
- **`M` is overloaded.** In the docs `M` (a.k.a. `X1X2`) is the "magnitude" `√(V1·V2)` in
  `K = M·J/π`. Elsewhere `M` = number of inducing points. This note uses `X1X2` for the
  magnitude and reserves `M`/`n_b` for the inducing dimension.
- **`J`** = the arc-cosine angular term, unrelated to any Jacobian.
- **"F-step"** = **firing-rate** step (`A, λ₀`), *not* "final" or "function" in any other
  sense; not to be confused with `f`/`f_mean` the rate itself (though it fits exactly that).
- **`b` subscript** = eigenspace basis; also `b = K̃_b⁻¹·m_b` (a vector) inside `dKL`.
  Disambiguate by context.

### C. Doc-vs-code — the "cache dK once / single `dL/dK` contraction" claim
`MSTEP_ANALYTICAL_HANDOFF.md` (§"Why VJP…", §"Quick Reference") and `gradients.md` §6
present the analytical M-step as *"compute `K, dK` once at start; LBFGS closure reuses
cached `dK` via `grad[key]=(dL_dK·dK[key]).sum()`."* The shipped
`mstep_eigenspace_analytical` **recomputes `C/dC/K/dK` inside the closure each evaluation**
(necessary — params move per line-search step) and computes gradients through the **full
moment→log-lik/KL chain**, not a single `(dL_dK·dK).sum()`. Trust the code (§4). The
`(dL_dK·dK).sum()` form is what System 1's **kernel-level** Jacobian backward does
(`analytical_gradients.py:ArcCosineJacobianGradients.backward`), which the docs conflated
with the M-step.

### D. Doc gap — the F-step is undocumented in the four sources
None of the four source docs mention the F-step, `A`, `λ₀`, or `lambda0_given_A`. Its
definition (§2.2, §3) rests entirely on `eigenspace_fstep.py` + `utils.py`. If docs and code
"disagree" on the F-step, it is because the docs are **silent**, not wrong; code is ground
truth.

### E. Doc-vs-code — `σ₀` transform, and the diagonal-`σ₀` factor
- `MSTEP_ANALYTICAL_HANDOFF.md` §2 ("Parameter Transform Gotchas") states *"GPyTorch uses
  softplus for sigma_0 and Amp."* **Wrong for σ₀.** Ground truth (kernels.py:207):
  `σ₀ = exp(raw_sigma_0)`; the M-step correctly multiplies by `kernel.sigma_0` (=`exp(raw)`),
  matching `gradients.md` §6. `analytical_gradients_vjp.py`'s test comment calls σ₀ an
  "identity transform" — a third, different statement, and also not the model's transform (it
  is a test-bookkeeping choice because the test feeds the constrained σ₀ directly).
  **Ground truth: σ₀ = exp, Amp = softplus.**
- Diagonal-`σ₀` factor-of-σ₀ difference between the two systems: see §5.3. Not a bug in the
  M-step (System 2 is internally consistent); it is a per-system convention.

### F. Doc-vs-code — `Amp` "handled by ScaleKernel"
`ANALYTICAL_GRADIENTS_MATH.md` §1.1 marks `Amp` as *"handled by ScaleKernel, gradient is
K/Amp."* In the current engine `Amp` is an **internal `ArcCosineKernel` parameter** (no
ScaleKernel): `dC['Amp'] = C/Amp`, and the M-step sets `raw_Amp.grad` with the softplus
Jacobian. The `K/Amp` relationship survives (`dC/dAmp = C/Amp`), but the "ScaleKernel"
attribution is stale.

### G. Stale file/line references in the docs
The four docs point at the **legacy** layout: `direct_vargp.py:mstep_lbfgs_analytical`,
`utils.py:acosker()` at lines ~3661-3813, `localker()` ~3577-3631, etc. Those line numbers
refer to the **repo-root** legacy `varGP` `utils.py` (the porting *reference*), **not** to
this directory's `utils.py` (which holds only `lambda0_given_A`, `compute_f_mean`, inducing
selection). The live homes today are `eigenspace_mstep.py` / `eigenspace_gradients.py` /
`analytical_gradients*.py`. `eigenspace_gradients.py`'s own docstrings still cite the legacy
`utils.py:...` line numbers as provenance — read them as "ported from", not "look here".

### H. Redundancy (not conflicts)
`ANALYTICAL_GRADIENTS_MATH.md`, `gradients.md`, and `VJP_ANALYTICAL_GRADIENTS.md` restate the
same `dC`/`dK`/VJP formulas at increasing consolidation (`gradients.md` is the merge of the
other two, per its footer). They agree with each other and with the code on the `dC`/`dK`/VJP
math; the only substantive divergences from code are C–G above.

---

## 10. Scope exclusions (explicitly skipped)

- **Spatiotemporal machinery** — temporal covariance, Kronecker products, time warping:
  absent from these files; `C` is spatial (pixel×pixel) only. Skipped by design.
- **`vargp_old` engine** — repo-root `utils.py:varGP()/Estep()/acosker()`, ScaleKernel `Amp`,
  old torch-LBFGS timing/`vargp_direct`-vs-`vargp_old` framing, benchmark absolute times:
  excluded as an engine. Its ELBO/hyperparameter-gradient **math** is retained (it is the
  same math the current analytical path runs).
- **Deriving the arc-cosine kernel `dC`/`dK` from first principles** — owned by the
  kernel-gradient cluster; here only reproduced as the *hookup* (§5).
- **The E-step's internal Newton math** — owned by the E-step cluster; here only its role in
  the loop composition (what it optimizes / holds fixed).
- **Acquisition / active-learning utility** (image selection) — separate cluster.
```
