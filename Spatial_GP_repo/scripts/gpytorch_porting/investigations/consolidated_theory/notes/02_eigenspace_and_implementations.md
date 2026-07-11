# 02 — Eigenspace representation & the two current implementations

**Cluster:** the bridge from the abstract variational-GP math to the two concrete
algorithms that run in this codebase. Establishes precisely which quantities live in
the eigenbasis of the (inducing) kernel, how predictions are formed from the eigenspace
posterior, the reload/checkpoint seam, the rank-1 online update used for active learning,
and how the standard-GPyTorch path realises the *same* model. Does NOT re-derive the
E/M/F-step update equations themselves (other cluster owns those) — it fixes the objects
those steps operate on.

**Ground truth = the current code** (paths relative to
`.../Spatial_GP_repo/scripts/gpytorch_porting/`):
`eigenspace_model.py`, `eigenspace_training.py`, `eigenspace_utils.py`,
`eigenspace_checkpoint.py`, `rank1_update.py` (the vargp_direct / eigenspace path);
`gpy_model.py`, `gpy_training.py` (the default_gpy / standard-GPyTorch path).
Doc sources (secondary, some stale — see §12): `.claude/EIGENSPACE_REFERENCE.md`,
`.claude/DECISION_LOG.md`.

Two current training modes are in scope:
- **`vargp_direct`** — the eigenspace variational GP (`DirectVGPModel`). GPyTorch is used
  as a *kernel calculator only*; variational parameters are managed directly in the
  eigenbasis of the inducing kernel, with a custom EM loop (E-step Newton, F-step, M-step
  LBFGS). Files: `eigenspace_*.py`, `rank1_update.py`.
- **`default_gpy`** — the standard GPyTorch sparse-variational path (`VariationalGPModel`,
  an `ApproximateGP`). Full `VariationalStrategy` + `CholeskyVariationalDistribution`;
  single joint ELBO optimisation. Files: `gpy_model.py`, `gpy_training.py`.

(Excluded modes: `vargp_old` — repo-root `utils.py:varGP()`/`Estep()`, per scope; and
`vargp_style` — a third hybrid mode mentioned in the reference, neither of my two. See §13.)

---

## 1. The shared generative model (both implementations realise the SAME model)

Both `vargp_direct` and `default_gpy` are the identical sparse variational GP for one
retinal ganglion cell's spike counts to natural images. **Spatial only**: each input is
one 108×108 image flattened to a pixel vector `x` (n_pixels = 11664 at full res, 64×64 in
some sweeps); the latent is a scalar function `λ(x)` over one image. *No temporal kernel,
no Kronecker `C = C_spatial ⊗ C_temporal`, no temporal warping appears anywhere in these
files* (grep-confirmed) — that machinery is out of scope and simply absent from this layer.

Ingredients shared by both implementations:

- **Inducing points** `X_tilde` (M points). In the closed-loop active setting the invariant
  is `X_train == X_tilde` (M == N), so training points *are* the inducing points.
- **Zero-mean GP prior** over inducing values `u` (a.k.a. `λ̃`, the function values at the
  inducing points): `p(u) = N(0, K̃)` where `K̃ = K_tilde = kernel(X_tilde, X_tilde)`.
- **Arc-cosine kernel** with a receptive-field-structured `C` matrix (locality β, eps_0x/eps_0y;
  smoothness ρ; offset σ₀; amplitude Amp). Kernel math belongs to the kernel cluster; here
  the kernel is a black box exposing `kernel(X1,X2).to_dense()`, `kernel(X, diag=True)`,
  `kernel._cached_mask`, `kernel.{beta,rho,sigma_0,eps_0x,eps_0y,Amp}`, `clamp_hyperparameters()`.
- **Poisson likelihood** with log-rate linear in the latent: `PoissonLikelihood(A, lambda0)`,
  rate `= exp(A·λ + λ₀)`.
- **Variational posterior** `q(u) = N(m, V)` over inducing values.
- **Expected firing rate** (the prediction), the mean of `exp(A·λ + λ₀)` under a Gaussian
  `λ ~ N(λ_m, λ_var)`:
  ```
  f_mean(x) = E[exp(A·λ + λ₀)] = exp(A·λ_m + 0.5·A²·λ_var + λ₀)
  ```
  Identical in both paths (`utils.compute_f_mean`, `PoissonLikelihood.expected_firing_rate`,
  `predict_eigenspace`, `gpy_training.predict`).
- **ELBO** (maximised): `ELBO = E_q[log p(r | f)] − KL(q(u) ‖ p(u))`, with expected
  log-likelihood
  ```
  E[log p(r|λ)] = Σ_i [ r_i·(A·λ_m,i + λ₀) − exp(A·λ_m,i + 0.5·A²·λ_var,i + λ₀) ]
  ```
  (`PoissonLikelihood.expected_log_prob`, `compute_elbo_eigenspace`).

**Where they diverge is purely the *representation* of `q(u)` and the *optimiser*, not the
model.** §3–§7 cover the eigenspace representation; §8 covers the GPyTorch representation;
§9 contrasts them.

---

## 2. Standard sparse-GP predictive (the formulas both paths compute)

For `q(u) = N(m, V)` at inducing points `X_tilde`, the posterior over `λ` at any query
point (mean μ, variance σ²) is the standard SVGP/Titsias predictive:
```
μ(x)  = k(x)ᵀ K̃⁻¹ m
σ²(x) = k(x,x) + k(x)ᵀ K̃⁻¹ (V − K̃) K̃⁻¹ k(x)
```
where `k(x) = kernel(x, X_tilde)` (M-vector) and `k(x,x) = kernel(x,x)` (scalar).
Sanity check: at the prior (`V = K̃`) the correction vanishes and `σ²(x) = k(x,x)`
(prior variance). As data shrink `V` below `K̃`, `σ²(x) < k(x,x)`.

`vargp_direct` computes these in the eigenbasis (§3.5); `default_gpy` computes them via
GPyTorch's `VariationalStrategy` internals (§8). Same math, two engines.

---

## 3. Eigenspace representation (vargp_direct) — the core of this cluster

### 3.1 Why eigenspace

The inducing kernel `K̃` (M×M) has effective rank ~10–50 (≪ M). Projecting into the
eigenbasis of `K̃`:
1. reduces dimensionality M → n_b (kept eigenvalues), so cost drops O(M³) → O(n_b³);
2. makes `K̃` **diagonal** in the new basis (`K̃_b = diag(eigvals)`), so `K̃⁻¹` is trivial
   (`diag(1/eigvals)`) — no matrix solve;
3. gives implicit regularisation by dropping small eigenvalues.

This is *also* effectively a whitening/re-parameterisation, but one the code recomputes and
reprojects *explicitly* after every kernel change (§3.8) — which is exactly what lets the
closed-form E-step coexist with a changing kernel, and is the design reason `vargp_direct`
bypasses GPyTorch's own whitening (see §8.2, DECISION_LOG Q26/Q27).

### 3.2 The eigendecomposition (`eigenspace_utils.eigendecompose_K_tilde`)

```python
eigvals, eigvecs = torch.linalg.eigh(K_tilde, UPLO='L')   # ASCENDING eigenvalues
threshold = max(eigvals.max().item() * eigval_tol, eigval_tol)   # relative OR abs floor
ikeep = eigvals > threshold
B         = eigvecs[:, ikeep]     # (M, n_b)  eigenvectors = columns
eigvals_b = eigvals[ikeep]        # (n_b,)   kept eigenvalues, ascending
```
- `K̃ = B diag(eigvals) Bᵀ`; `B` is orthonormal (`BᵀB = I_{n_b}`).
- **Threshold is `max(λ_max·tol, tol)`**, not a bare `> tol` — a *relative* cut (fraction of
  the largest eigenvalue) with an absolute floor. `eigval_tol = 1e-4` (default).
- **`n_b` is dynamic**: it changes across training iterations as kernel hyperparameters change
  (and grows by ~1 when a point is appended, §7). Reprojection (§3.4) handles the dimension
  change. Reference/utils docstrings quote n_b ~10–11 typical for the closed-loop cells,
  up to ~50 elsewhere.

### 3.3 Objects that live in eigenspace — `DirectVariationalState` (`eigenspace_model.py`)

The dataclass that IS the eigenspace posterior + all precomputed kernel projections:

| field | shape | meaning |
|---|---|---|
| `m_b` | (n_b,) | variational mean in eigenspace, `= Bᵀ m` |
| `V_b` | (n_b, n_b) | variational covariance in eigenspace, `= Bᵀ V B`. **NOT diagonal** (even though `K̃_b` is) |
| `B` | (M, n_b) | eigenvector matrix of `K̃` (the basis) |
| `eigvals_b` | (n_b,) | kept eigenvalues of `K̃` (ascending) |
| `K_tilde_b` | (n_b, n_b) | inducing kernel in eigenspace `= diag(eigvals_b)` — DIAGONAL |
| `K_b` | (N, n_b) | cross-kernel in eigenspace `= K @ B`, where `K = kernel(X_train, X_tilde)` |
| `KKtilde_inv_b` | (N, n_b) | `K @ K̃⁻¹` in eigenspace `= K_b / eigvals_b` (element-wise). The projection vector, called `a` |
| `Kvec` | (N,) | diagonal self-kernel `k(x_i, x_i)` (prior variance per training point) |
| `K_tilde` | (M, M) | the FULL inducing kernel, retained *only* to enable O(M) column-append in the rank-1 update (§7) |
| `mask` | (n_pixels,) or None | pixel mask from the kernel (RF support), if `use_mask=True` |

Core insight preserved: **`V_b` is dense; only `K̃_b` is diagonal.** Never assume diagonal `V_b`.

`_compute_eigenspace_quantities(kernel, X_train, X_tilde, eigval_tol)` builds all of these
in one pass (under `torch.no_grad()`): computes `K̃`, `K`, `Kvec` from the kernel,
eigendecomposes `K̃`, then forms `K_b = K@B`, `K̃_b = diag(eigvals_b)`,
`KKtilde_inv_b = K_b / eigvals_b`. Shared by both initialisation and post-M-step recompute.

### 3.4 Projection & reprojection (`eigenspace_utils.py`)

- **Project full → eigenspace** (`project_to_eigenspace`):
  `m_b = Bᵀ m`, `V_b = Bᵀ V B`, `K_b = K B`.
- **Initial state** (`_compute_initial_eigenspace`): variational params start at the prior —
  `m_b = 0` (n_b,), `V_b = K̃_b.clone()` (so `q(u) = p(u)` initially).
- **Reproject across a basis change** (`reproject_variational_params`, called after M-step):
  when the kernel changes, `K̃` and hence `B` change (`B_old → B_new`, possibly different n_b).
  Round-trip the params through full M-space:
  ```
  m_b_new = B_newᵀ B_old m_b_old
  V_b_new = B_newᵀ (B_old V_b_old B_oldᵀ) B_new
  ```
  Warning preserved in code: if n_b grows, the new dimensions carry no prior information and
  `V_b_new` may have small eigenvalues; the subsequent E-step restores positive-definiteness.

### 3.5 Posterior moments from the eigenspace posterior

With `a = KKtilde_inv_b = K_b / eigvals_b` (the eigenspace image of `k(x)ᵀ K̃⁻¹`), the §2
formulas become (`_lambda_moments_eigenspace` for training points; identical form with a
freshly projected `a` for test points in `predict_eigenspace` / `EigenspacePosterior`):
```
λ_m   = a @ m_b                                   # (N,)
λ_var = Kvec + rowsum( a * (a @ (V_b − K̃_b)) )    # (N,) = Kvec + diag(a (V_b − K̃_b) aᵀ)
λ_var = clamp(λ_var, min = lambda_var_clamp)       # 1e-6
```
For **test** points (`is_training_data=False`): compute `K_query = kernel(X_test, X_tilde)`,
`Kvec_query = kernel(X_test, diag=True)`, project `K_query_b = K_query @ B`, then
`a = K_query_b / eigvals_b` and apply the same two lines. For **training** points the
precomputed `state.KKtilde_inv_b` is reused (exact equivalence, no kernel recompute).

The firing-rate prediction (`predict_eigenspace` returns `f_pred`, `lambda_m`, `lambda_var`):
```
f_pred = exp(A·λ_m + 0.5·A²·λ_var + λ₀)
```

### 3.6 ELBO / KL in eigenspace (`compute_elbo_eigenspace`)

`ELBO = log_lik − KL`. Log-lik as in §1. KL for `q(u)=N(m,V)` vs `p(u)=N(0,K̃)`, written in
the eigenbasis where `K̃_b = diag(eigvals)`:
```
KL = 0.5·[ tr(K̃⁻¹ V) + mᵀ K̃⁻¹ m − n_b + log|K̃| − log|V| ]
   = 0.5·[ Σ_j V_b[j,j]/eigvals_j  +  Σ_j m_b[j]²/eigvals_j  −  n_b
           + Σ_j log(eigvals_j)  −  logdet(V_b) ]
```
Implementation notes preserved:
- `trace_term` uses only `diag(V_b)` — valid here **because `K̃_b` is guaranteed diagonal**
  when the ELBO is evaluated (fixed eigenspace, after E/F-step, *before* M-step). This is
  the exact assumption that is UNSAFE inside the M-step closure (see §12 / caveat below).
- `log|V|` uses full `torch.linalg.slogdet(V_b)`; a non-PD `V_b` warns and sets `log|V|→0`.
- **Known constant loss offset vs vargp_old (`≈ 0.5·n_b`):** vargp_direct includes the `−n_b`
  term; vargp_old omits it. Zero gradient ⇒ does not affect optimisation or predictions,
  only the reported loss magnitude (EIGENSPACE_REFERENCE §6.8).

### 3.7 What the E/F/M steps operate on (objects only; derivations deferred)

The custom EM loop mutates the eigenspace state; the update *equations* are another cluster's.
The objects each step reads/writes:
- **E-step** (`estep_eigenspace`): Newton update of `(m_b, V_b)` using `a = KKtilde_inv_b`,
  `K̃_b`, the spike counts `r`, gain `A`, and `f_mean`. Writes back via
  `model.update_variational_params(m_b, V_b)` (which **detaches** both — avoids the autograd
  graph chaining across iterations through `A` and blowing up memory — and **symmetrises**
  `V_b ← (V_b+V_bᵀ)/2`). The m-update uses the legacy "old formula"
  `m_new = V_new @ (G m + g)` (empirically stable; the "correct" form collapsed — deferred).
- **F-step** (`fstep_eigenspace` / `damped_newton_update_A_lambda0`): optimises likelihood
  `A` (and computes `λ₀` analytically); reads `λ_m, λ_var`. Operates on `model.likelihood`.
- **M-step** (`mstep_eigenspace_autograd` / `_analytical`): optimises kernel hyperparameters
  by LBFGS on the ELBO; the eigenbasis `B` is held FIXED during the closure, so `K̃_b` becomes
  non-diagonal as soon as the kernel moves (caveat below). Analytical variant is the correct
  one; autograd variant had a diagonal-KL bug (now fixed).

### 3.8 The recompute seam after the M-step (`recompute_eigenspace`)

After the M-step changes kernel hyperparameters, `K̃` (and thus `B`, `eigvals_b`) is stale.
`model.recompute_eigenspace()` → `_recompute_eigenspace()`:
1. recompute all eigenspace quantities from the *new* kernel (`_compute_eigenspace_quantities`);
2. reproject `(m_b, V_b)` from `B_old` to `B_new` (§3.4).
In `train_eigenspace`, this runs at the START of each iteration `> 1` (guarded by
`n_mstep > 0 and iteration > 1`), so metrics/ELBO are always evaluated on a *consistent*
(kernel, eigenspace, params) triple. **Caveat (EIGENSPACE_REFERENCE §6.1):** `K̃_b` is
diagonal ONLY right after init / recompute / during E-/F-step; inside the M-step closure after
a hyperparameter step, `K̃_b = Bᵀ K̃_new B` is NOT diagonal (B is the OLD eigenbasis) — code
there must use `torch.linalg.solve`, not `diag(1/eigvals)`.

---

## 4. `DirectVGPModel` and its wrapper classes (`eigenspace_model.py`)

`DirectVGPModel` owns `kernel`, `likelihood`, `X_train`, `X_tilde`, and the
`DirectVariationalState`. It is a **plain Python class, NOT an `nn.Module`** — hence the
separate checkpoint file (§6). Key surface:

- `__init__(kernel, likelihood, X_train, X_tilde, eigval_tol=1e-4, lambda_var_clamp=1e-6,
  _precomputed_state=None)` — builds the initial eigenspace unless a `_precomputed_state`
  is supplied (the rank-1 update supplies one, §7, to skip redundant kernel eval).
- `model(X_query)` → `EigenspacePosterior` (GPyTorch-like). Uses precomputed
  `KKtilde_inv_b` when `X_query is self.X_train` (identity check), else projects fresh.
- `update_variational_params(m_b, V_b)` — E-step write-back (detaches + symmetrises `V_b`).
- `recompute_eigenspace()` — post-M-step sync (§3.8).
- `.state` (read the `DirectVariationalState`), `.variational_distribution`
  (`EigenspaceVariationalDistribution`).
- `.eval()` / `.train()` — no-ops, present only for API compatibility.

`EigenspacePosterior` computes `(mean=λ_m, variance=λ_var)` ONCE on construction (§3.5) and
exposes `.mean` / `.variance`. (It does NOT itself carry `expected_firing_rate` — that lives on
`PoissonLikelihood.expected_firing_rate(posterior)`; the reference doc's Section 7.2 is stale
on this, see §12.)

`EigenspaceVariationalDistribution` exposes the params in both spaces:
`mean_eigenspace = m_b`, `covariance_eigenspace = V_b` (n_b), and the full-space
`mean = B @ m_b` (M,), `covariance = B @ V_b @ Bᵀ` (M×M, expensive — reconstruct on demand only).

---

## 5. The eigenspace training loop (`train_eigenspace`)

EM-style loop, `for iteration in range(1, n_iterations)` ⇒ **n_iterations − 1 actual
iterations** (matches vargp_old exactly). Per iteration:
1. If `iteration > 1` and `n_mstep>0`: `recompute_eigenspace()` (§3.8), refresh
   `λ_m, λ_var, A, λ₀, f_mean`.
2. **E-step** Newton loop (`n_estep`), with a per-step **divergence guard**: if
   `f_mean.mean() > f_mean_mean_threshold` (100) or `f_mean.max() > f_mean_max_threshold`
   (500) or NaN, revert `(m_b, V_b)` (and A/λ₀ if interleaving) to the pre-step clone and break.
3. **F-step**: either interleaved (damped Newton on A, λ₀ at *every* E-step iteration,
   `alpha=0.25` — the paper's approach, DECISION_LOG Q28) or a single trailing
   `fstep_eigenspace` after the E-loop.
4. Compute ELBO/metrics on the consistent state (BEFORE the M-step); log per-iteration curves
   (loss, train/val log-lik, KL, Pearson r, Spearman ρ, and the params A, λ₀, β, ρ, σ₀,
   eps_0x, eps_0y, Amp).
5. **Early stopping** on ELBO (only supported `es_metric`; `'none'` disables — DECISION_LOG
   Q32). Two decoupled counters (DECISION_LOG Q33, PyTorch-Lightning-style): `best_es_value`
   updates on ANY improvement (drives `restore_best`); `patience_reference` resets only on
   cumulative relative gain `> min_delta_rel` (0.001). `restore_best` uses `_save_model_state`
   / `_restore_model_state`, which DO save `B` and then `recompute_eigenspace()` to reproject
   — i.e. an in-memory basis-consistent restore (contrast the on-disk checkpoint, §6).
6. **M-step** last (`iteration < n_iterations − 1`, so the final M-step is skipped — it would
   build an eigenspace never used by the final params). Analytical or autograd variant.

`fix_Amp=True` freezes `raw_Amp` ("paper's code has no Amp parameter"); `Amp` is fixed at 1.0
in production (DECISION_LOG Q27). Divergence to NaN/inf ELBO ⇒ stop.

---

## 6. Reload / checkpoint seam (`eigenspace_checkpoint.py`)

Save/load for `DirectVGPModel`. Separate from `checkpoint.py` (the default_gpy one) because
`DirectVGPModel` is not an `nn.Module` and has no `variational_strategy`.

**Saved** (`save_eigenspace_checkpoint`): `kernel.state_dict()`, `likelihood.state_dict()`,
**`m_b`, `V_b`**, `pool_indices` (indices into the shared PNAS `X_pool`), integrity tags
`pool_shape / pool_sum / pool_dtype`, `rf_center_bounds` (the RF-centre box, a plain kernel
attribute anchored to the *initial* STA — saved explicitly so a reload keeps the same trainable
region), plus `config`, `metrics`, `hyperparams`, `metadata`.

**NOT saved — and this is the reload seam:**
- `X_train` — reconstructed as `X_pool[pool_indices]` (keeps checkpoints ~1 KB not ~2 MB).
- **The entire derived eigenspace** (`B, eigvals_b, K̃_b, K_b, KKtilde_inv_b, Kvec`). On load,
  `DirectVGPModel.__init__` recomputes it fresh from `(loaded kernel, X_train, eigval_tol)`,
  then the saved `m_b, V_b` are re-injected via `update_variational_params`.

**Critical consequence:** the saved `m_b, V_b` are expressed in the SAVE-time eigenbasis `B`,
but are injected into a FRESHLY-recomputed `B` — with **no reprojection**. The code assumes the
recomputed eigendecomposition equals the save-time one ("deterministic on the same hardware").
This is exactly the basis-mismatch fragility flagged project-wide for torch-version drift
(`analysis/CLAUDE.md`: "saved m_b/V_b are tied to the SAVE-time basis; the LOAD-time basis under
a different torch differs subtly → ~5% drift"; near-degenerate/near-threshold eigenvectors can
flip sign or drop in/out). Note the asymmetry with the in-memory `restore_best` path (§5), which
DOES save `B` and reprojects — the on-disk checkpoint does not. Integrity is guarded only for the
*data pool* (shape exact, dtype warn, sum within tolerance), not for the basis.

Active-loop invariant used at load: `X_tilde = X_train` and `M == n_train == len(pool_indices)`
(`pool_indices.shape[0]` is authoritative; `metadata['M']` is unreliable for old iter_NNN.pt).
`load_pool_indices` is the lightweight variant (indices + integrity only, no model rebuild).

---

## 7. Rank-1 online update for active learning (`rank1_update.extend_model_with_new_point`)

The mechanism by which Phase 2 grows the model by one image per active-learning step. Produces a
**new** `DirectVGPModel` with `M+1` points, warm-started from the M-point model.

Steps (mirrors legacy `set_new_model_variational_params`, utils.py:384):
1. **Efficient column append to `K̃`** — compute ONLY the new column
   `kernel(X_tilde_new, x_new)` (O(M) kernel evals) and stitch it as a new column+row onto the
   stored `state.K_tilde` (this is *why* the full `K_tilde` is retained in the state). Avoids the
   O(M²) full recompute. The eigendecomposition of the (M+1)×(M+1) `K̃` is O(M³) either way.
2. **Eigendecompose** the new `K̃` → `B_new, eigvals_b` (`n_b_new`).
3. **Derived quantities** using the active-loop invariant `X_train == X_tilde` (so `K = K̃`):
   `K_b = K̃_new @ B_new`, `K̃_b = diag(eigvals_b)`, `KKtilde_inv_b = K_b / eigvals_b`.
   `Kvec` extended by the single new diagonal `k(x_new, x_new)`.
4. **Warm-start the variational params** (the legacy 3-move):
   - expand to full M-space: `m_full = B_old m_b_old`, `V_full = B_old V_b_old B_oldᵀ` (symmetrise);
   - pad to M+1: new mean entry `= mean(m_full)`; new covariance via
     `V_full_ext = eye(M+1)` then `V_full_ext[:M,:M] = V_full` (**new point gets variance 1.0,
     zero cross-covariance** — an identity-variance placeholder for the yet-unseen dimension);
   - project into the new basis: `m_b_new = B_newᵀ m_full_ext`, `V_b_new = B_newᵀ V_full_ext B_new`.
5. Build the state and hand it to `DirectVGPModel(..., _precomputed_state=state)` (skips the
   constructor's own eigenspace compute). Kernel/likelihood must be DEEP-COPIED by the caller,
   because `train_eigenspace` mutates the kernel in place.

Net: a single active image appends one column to `K̃`, bumps `n_b` by ~1, and re-warm-starts
`(m_b, V_b)` — the online counterpart of the batch `_compute_initial_eigenspace`.

---

## 8. The default_gpy implementation (`gpy_model.py`, `gpy_training.py`)

Realises the SAME model (§1) via stock GPyTorch, i.e. it keeps the variational posterior in the
FULL M-dimensional inducing space and never touches the eigenbasis.

### 8.1 `VariationalGPModel(ApproximateGP)` (`gpy_model.py`)

- `variational_distribution = CholeskyVariationalDistribution(M)` — stores `q(u) = N(m, L Lᵀ)`
  via the Cholesky factor `L` (numerically stable), in full M-space.
- `variational_strategy = VariationalStrategy(...)` (whitened, when
  `standard_variational_distribution=True`, the default) OR `UnwhitenedVariationalStrategy(...)`
  (natural params, when `False` — for EM-style use, see §8.2). Both get `jitter_val = JITTER`.
- `mean_module = ZeroMean()`; `covar_module = kernel` (the arc-cosine kernel).
- `forward(x)` returns the GP **prior** `MultivariateNormal(mean_module(x), covar_module(x))`;
  GPyTorch's strategy turns that + `q(u)` into `q(f)` at `x` (adding jitter internally — none is
  added in `forward`).
- `get_variational_parameters()` → `{mean: m, covar: V}`; `get_inducing_points()`.
- It IS an `nn.Module`, so kernel + likelihood + variational params are all in `state_dict()`.

### 8.2 Whitened vs unwhitened — the reason `vargp_direct` manages params directly

This is the load-bearing link between the two implementations (DECISION_LOG Q26–Q29):
- **Whitened `VariationalStrategy`** stores params relative to `L_K` (Cholesky of `K_uu`); prior
  becomes `N(0, I)`. It assumes JOINT autograd optimisation, where autograd differentiates through
  `L_K` and keeps the coupling consistent.
- A closed-form Newton E-step + a kernel-changing M-step (EM style) BREAKS this: params stored as
  `m_stored = L_K_old⁻¹ m_natural` become stale once the M-step makes `L_K_old → L_K_new`, so the
  next E-step reads `L_K_new m_stored ≠ m_natural` (~8× errors in λ_m — the "L_K mismatch").
- Two escapes: (a) **`UnwhitenedVariationalStrategy`** stores natural params (`L_K`-independent) so
  EM is safe — but it is slower and empirically less accurate (KL-gradient instability, Q29); or
  (b) **`vargp_direct`'s own eigenspace**, which whitens via `B`/`eigvals` but **recomputes and
  reprojects the basis explicitly** after every M-step (§3.8) instead of relying on autograd.
  `default_gpy` in practice uses the whitened strategy with a single JOINT ELBO optimisation
  (no custom E-step), which is the regime whitening was designed for.

### 8.3 `train_gpy_default` (`gpy_training.py`)

Single joint ELBO maximisation over ALL parameters (kernel + likelihood + variational), no EM
decomposition:
- Optimiser `'lbfgs'` (`line_search_fn='strong_wolfe'`, `max_iter=GPY_LBFGS_MAX_ITER=20`) or
  `'adam'`.
- Loss = `−likelihood.expected_log_prob(train_y, model(train_x)) + model.variational_strategy.kl_divergence()`.
- LBFGS `closure` guards (return `inf` ⇒ reject step): out-of-bounds kernel/likelihood params
  (`params_in_bounds()`), kernel exceptions/NaN `K_uu`, firing-rate divergence (same 100/500
  thresholds as vargp_direct), NaN/inf loss.
- After each step: `kernel.clamp_hyperparameters()`, `likelihood.clamp_params()` (projected GD).
- Cholesky stability via `linear_operator.settings.cholesky_jitter(JITTER)` +
  `cholesky_max_tries(3)` context (retries at 1e-4, 1e-3, 1e-2).
- Same ELBO early-stopping + decoupled best/patience logic as `train_eigenspace` (§5), here
  saving/restoring `model.state_dict()` / `likelihood.state_dict()`.

### 8.4 Prediction (`gpy_training.predict`)

`model.eval()`; `posterior = model(test_x)`; `lambda_mean = posterior.mean`,
`lambda_var = clamp(posterior.variance, 1e-6)`; `f_pred = exp(A·lambda_mean + 0.5·A²·lambda_var + λ₀)`.
Same firing-rate formula as §3.5. **Return-key clash to note:** this returns `'lambda_mean'`
whereas `predict_eigenspace` returns `'lambda_m'` (§12).

---

## 9. Contrast — shared math vs divergent representation

Shared (the model, §1): same GP prior `N(0,K̃)`, same arc-cosine RF kernel, same
`PoissonLikelihood(A, λ₀)`, same `f_mean = exp(Aμ + ½A²σ² + λ₀)`, same ELBO decomposition,
same divergence guards/thresholds, same ELBO early-stopping logic, same `JITTER=1e-4`,
same inducing-points-are-training-points invariant.

| Aspect | vargp_direct (`DirectVGPModel`) | default_gpy (`VariationalGPModel`) |
|---|---|---|
| Posterior rep of `q(u)` | eigenspace: `m_b` (n_b), dense `V_b` (n_b×n_b) in basis `B` of `K̃` | full M-space: `CholeskyVariationalDistribution` `q(u)=N(m, LLᵀ)` |
| `K̃⁻¹` | trivial `diag(1/eigvals)` | via GPyTorch Cholesky of `K_uu`+jitter |
| Whitening | own eigenbasis, **explicit reproject** after each M-step | whitened `N(0,I)` (default) via `VariationalStrategy`; or `Unwhitened` |
| Optimiser | custom **EM**: E (Newton), F (A, λ₀), M (LBFGS kernel) + `recompute_eigenspace` | single **joint ELBO** over all params (LBFGS or Adam) |
| GPyTorch role | **kernel calculator only** (`kernel(...).to_dense()`) | full `ApproximateGP` + `VariationalStrategy` |
| `nn.Module`? | NO (plain class; dataclass state) | YES |
| KL | manual eigenspace formula (`compute_elbo_eigenspace`) | `variational_strategy.kl_divergence()` |
| Dim reduction | M → n_b eigenspace | none (full M) |
| Online rank-1 update | `extend_model_with_new_point` (K̃ column append) | none provided |
| Checkpoint | `eigenspace_checkpoint.py` (`m_b,V_b`; no `B` saved) | `checkpoint.py` (`state_dict`) |
| Predict return keys | `f_pred, lambda_m, lambda_var` | `f_pred, lambda_mean, lambda_var` |

The eigenspace basis `B` and GPyTorch's whitening factor `L_K` play analogous roles (both
diagonalise/whiten the prior); the difference is that `vargp_direct` OWNS and explicitly
re-syncs its `B`, so a closed-form E-step is safe, whereas GPyTorch's `L_K` is managed implicitly
by autograd and goes stale under EM (§8.2).

---

## 10. Constants (`default_params.json` → `_constants.py`)

| name | value | use |
|---|---|---|
| `eigval_tol` (`EIGVAL_TOL`) | `1e-4` | eigenvalue keep-threshold `max(λ_max·tol, tol)` |
| `lambda_var_clamp` (`LAMBDA_VAR_CLAMP`) | `1e-6` | min posterior variance clamp |
| `jitter` (`JITTER`) | `1e-4` | GPyTorch Cholesky jitter (default_gpy) |
| `cholesky_max_tries` | `3` | Cholesky retry count (retries ×10 jitter each) |
| `gpy_lbfgs_max_iter` | `20` | LBFGS inner iters per outer step (default_gpy) |
| `f_mean_max_threshold` | `500` | divergence guard on `f_mean.max()` |
| `f_mean_mean_threshold` | `100` | divergence guard on `f_mean.mean()` |
| early_stopping | `enabled=True, patience=15, min_delta_rel=1e-3, min_iterations=10, restore_best=True, es_metric='elbo'` | shared ES config |

No hidden fallbacks: `_constants.py` reads these from `default_params.json` (KeyError if missing,
by design).

---

## 11. Notation table (symbol | meaning | source; CLASHES/SYNONYMS flagged)

| symbol (code) | math | meaning | source |
|---|---|---|---|
| `X_train` | X | training images (N × n_pixels) | all |
| `X_tilde` | X̃ / Z | inducing points (M × n_pixels); `== X_train` in active loop | eigenspace_model, gpy_model |
| `M` | M | # inducing points | all |
| `N` | N | # training points (`= M` in active loop) | eigenspace_* |
| `n_b` | n_b | eigenspace dim = # kept eigenvalues (dynamic) | eigenspace_utils |
| `u` / `λ̃` | u | GP function values at inducing points (`q(u)=N(m,V)`) | gpy_model docstring |
| `m`, `V` | m, V | full-space variational mean/cov (M) | project_to_eigenspace |
| `m_b` | m_b (=Bᵀm) | eigenspace variational mean (n_b) | DirectVariationalState |
| `V_b` | V_b (=BᵀVB) | eigenspace variational cov (n_b×n_b), **dense** | DirectVariationalState |
| `B` | B | eigenvector matrix of `K̃` (M×n_b), `BᵀB=I` | eigenspace_utils |
| `eigvals_b` | λ_i | kept eigenvalues of `K̃` (ascending) | eigenspace_utils |
| `K_tilde` / `K̃` | K̃ / K_uu | inducing kernel `kernel(X̃,X̃)` (M×M) | _compute_eigenspace_quantities |
| `K_tilde_b` / `K̃_b` | K̃_b | inducing kernel in eigenspace `= diag(eigvals)` (diagonal) | DirectVariationalState |
| `K` (local) / `K_b` | K, K_b | cross-kernel `kernel(X,X̃)` (N×M) and its projection `K@B` (N×n_b) | _compute_eigenspace_quantities |
| `KKtilde_inv_b` = `a` | K K̃⁻¹ | projection vector `= K_b/eigvals` (N×n_b) | eigenspace_model. **SYNONYM: `a` ≡ `state.KKtilde_inv_b`; legacy "Matthew's a"** |
| `Kvec` | k(xᵢ,xᵢ) | diagonal self-kernel / prior variance (N) | _compute_eigenspace_quantities |
| `lambda_m` / `lambda_mean` | μ(x) | posterior mean of λ. **CLASH: `predict_eigenspace`→`lambda_m`; `gpy predict`→`lambda_mean`** | eigenspace_training / gpy_training |
| `lambda_var` | σ²(x) | posterior variance of λ (clamped ≥1e-6) | both |
| `f_mean` / `f_pred` | E[f] | expected firing rate `exp(Aμ+½A²σ²+λ₀)`. **SYNONYM: `f_mean` (train) ≡ `f_pred` (predict)** | utils, both predicts |
| `r` / `train_y` / `target` | r | observed spike counts (N). **SYNONYM across files** | all |
| `A` | A | Poisson-likelihood gain (`= exp(raw_A)`) | likelihoods |
| `lambda0` / `λ₀` | λ₀ | Poisson-likelihood bias/offset | likelihoods |
| `Amp` | Amp | kernel amplitude — **frozen at 1.0** in production (`fix_Amp`); paper has no Amp | eigenspace_training, DECISION_LOG Q27 |
| `sigma_0`, `beta`, `rho`, `eps_0x`, `eps_0y` | σ₀, β, ρ, eps₀ | arc-cosine kernel hyperparams (kernel cluster owns the math) | curves in eigenspace_training |
| `mask` | — | RF pixel-support mask (`kernel._cached_mask`) | DirectVariationalState |

---

## 12. Conflicts / redundancies / inconsistencies (doc-vs-code, doc-vs-doc)

1. **STALE FILE NAMES in `EIGENSPACE_REFERENCE.md` (doc-vs-code).** The reference's §3 "File
   Structure" lists `train.py`, `estep.py`, `fstep.py`, `mstep.py`. These files DO NOT EXIST —
   confirmed by `ls`. Current names: `eigenspace_training.py`, `eigenspace_estep.py`,
   `eigenspace_fstep.py`, `eigenspace_mstep.py` (the reference even admits "Extracted from
   train.py during codebase reorganization (2025-02)"). Trust the `eigenspace_*.py` names.
2. **`EigenspacePosterior.expected_firing_rate()` does NOT exist (doc-vs-code).** Reference §7.2
   shows it as a method on the posterior. In current code the method lives on
   `PoissonLikelihood.expected_firing_rate(posterior)`; the posterior exposes only `.mean` /
   `.variance`. Usage is `model.likelihood.expected_firing_rate(model(X))`.
3. **`autograd` M-step "BUGGY" label is stale (doc-vs-doc within the reference).** §4 M-step
   Details calls `mstep_eigenspace_autograd` "BUGGY, use analytical instead", but §8 "Resolved
   Bugs" records the diagonal-KL-trace bug FIXED (Jan 2025) with post-fix test_r matching. The
   "BUGGY" warning is pre-fix residue; the training loop still defaults to autograd
   (`use_analytical_mstep=False`).
4. **Return-key clash between the two `predict`s (code-vs-code).** `predict_eigenspace` →
   `{'f_pred','lambda_m','lambda_var'}`; `gpy_training.predict` → `{'f_pred','lambda_mean','lambda_var'}`.
   Any code consuming both must handle both key spellings.
5. **On-disk checkpoint does NOT save `B`, but the in-memory restore DOES (code-vs-code, §6/§5).**
   `save_eigenspace_checkpoint` omits `B` and relies on deterministic re-eigendecomposition; the
   `restore_best` path (`_save_model_state`) saves `B` and reprojects. The former is the documented
   torch-version-drift fragility (basis mismatch → ~5% drift); a fresh session should treat a
   cross-torch reload of an eigenspace checkpoint as suspect.
6. **`Amp` scope tension (instruction-vs-code).** The consolidation scope says exclude the "Amp"
   parameter (a vargp_old artefact). `Amp` nonetheless still EXISTS in the current kernel and is
   logged/checkpointed by these files — but is frozen at 1.0 (`fix_Amp`, DECISION_LOG Q27; "the
   paper does not use Amp"; full removal planned). Treated here as a vestigial, frozen kernel
   parameter; its kernel-math belongs to the kernel cluster, not this one.
7. **`EIGENSPACE_REFERENCE.md` header still frames everything against `vargp_old`** ("matches
   vargp_old", loss offset "vs vargp_old", "Amp"). vargp_old is out of scope; those comparisons
   are retained ONLY where they explain a current-code behaviour (the `−n_b` KL offset; the legacy
   `m_new` formula; the legacy `set_new_model_variational_params` that the rank-1 update mirrors).
8. **DECISION_LOG numbering collision (doc-vs-doc).** Two different `Q26/Q27/Q28` blocks exist
   (Session 9 "Whitening" vs "YAML"/"Paper Gap" sections). When citing, disambiguate by topic, not
   number.

---

## 13. Scope exclusions (what I deliberately left out)

- **Spatiotemporal machinery** — none present in these 7 files (grep-confirmed: no temporal
  kernel, no Kronecker `C = C_spatial ⊗ C_temporal`, no temporal warping). This layer is purely
  spatial (input = one image; latent = scalar function over one image). Nothing to exclude here
  beyond noting its absence.
- **`vargp_old`** (repo-root `utils.py:varGP()`/`Estep()`, old torch-LBFGS, the "paper-vs-vargp_old"
  framing) — excluded per scope. Mentioned only where a current behaviour is defined by matching it.
- **`vargp_style`** (the hybrid custom-E-step + GPyTorch-strategy mode named in the reference) —
  not one of my two implementations; excluded.
- **E/M/F-step update derivations** — the Newton E-step formulas, the analytical M-step gradients,
  the F-step A/λ₀ updates. I fixed the *objects* they read/write (§3.7) but leave the update
  equations to the E/M/F cluster. Files `eigenspace_estep.py`, `eigenspace_fstep.py`,
  `eigenspace_mstep.py`, `eigenspace_gradients.py` were intentionally not distilled here.
- **Arc-cosine kernel internals** (the `C = Amp·α αᵀ ⊙ C_smooth` construction, β/ρ/σ₀/eps₀ math,
  the pixel mask derivation, parameter transforms/chain rules) — kernel cluster owns these; treated
  as a black-box calculator throughout.
- **Utility / active-image-selection math** (information gain, `standard_utility`) — a different
  cluster; I covered only the *model-growth* mechanism (rank-1 update) that active learning drives.
