# vargp_direct Implementation Context

**Purpose**: Comprehensive context for implementing a true varGP copy in GPyTorch.
**Created**: January 2025
**Status**: COMPLETE (January 2025)
**Plan file**: `/home/idv-eqs8-pza/.claude/plans/mellow-finding-kitten.md`

---

## Quick Reference

| Item | Value |
|------|-------|
| **Goal** | New training mode `vargp_direct` that matches original varGP structure |
| **Key insight** | Use GPyTorch as kernel calculator only, bypass VariationalStrategy |
| **Files created** | `eigenspace.py`, `direct_vargp.py` |
| **Files modified** | `run_single_mode.py` |
| **E-step formula** | Uses OLD formula (matching vargp_old) - see DEFERRED below |
| **M-step** | LBFGS with autograd (not analytical gradients) |

---

## DEFERRED: Correct E-step m_new Formula

**Decision**: We implemented the OLD m_new formula (matching vargp_old) instead of the
mathematically correct formula. This was done to get a working baseline that matches
vargp_old behavior.

**Reason**: When we tried using the "correct" formula directly, there was a mismatch
in how g_b and G_b are computed (transformed vs standard). The old code's formulas
work correctly with the transformed quantities.

**TODO for future investigation**:
1. The "correct" m_new formula is: `m_new = m + K_tilde @ solve(K_tilde + G, g - m)`
2. The old formula is: `m_new = V_new @ (G @ m + g)`
3. These differ unless K_tilde and G commute
4. See `.claude/ESTEP_MATH_ANALYSIS.md` for full analysis
5. Worth testing if the "correct" formula improves results

**Current implementation** (in `direct_vargp.py:estep_eigenspace()`):
- Uses OLD formula matching utils.py line 4247
- Has a TODO comment noting this for future investigation

---

## 1. Why This Implementation?

### Current State

There are three training modes:

1. **vargp_old** (`utils.py:varGP()`): Original implementation
   - Eigenspace projection (~10-11 dims from M inducing points)
   - LBFGS with analytical gradients for M-step
   - Works well but codebase is complex

2. **default_gpy** (`train.py:train_gpy_default()`): Standard GPyTorch
   - Uses GPyTorch's variational framework
   - Adam optimizer for all parameters
   - Simple but different optimization dynamics

3. **vargp_style** (`train.py:train_varGP_style()`): Hybrid attempt
   - Custom E-step loop but works in full M-dim space
   - Uses GPyTorch VariationalStrategy (requires whitening conversions)
   - Adam for M-step (not LBFGS)

### The Gap

`vargp_style` is missing two critical features:
1. **Eigenspace projection** - reduces dimensionality, makes K̃⁻¹ trivial
2. **LBFGS M-step** - matches original optimization dynamics

### The Solution

Create `vargp_direct` that:
- Uses GPyTorch `ArcCosineKernel` for kernel computation only
- Manages m_b, V_b tensors directly (no GPyTorch VariationalStrategy)
- Implements eigenspace projection exactly like original
- Uses LBFGS for M-step (with autograd, not analytical gradients)

---

## 2. Original varGP Loop Structure

From `utils.py:varGP()` lines 5293-5975:

```
for iteration in range(1, maxiter):

    # 1. KERNEL RECOMPUTATION (after M-step changes hyperparameters)
    if nMstep > 0 and iteration > 1:
        C, mask = localker(theta)           # Recompute C matrix
        K_tilde = acosker(theta, xtilde)    # Inducing kernel (M, M)
        K = acosker(theta, x, xtilde)       # Cross-kernel (N, M)
        Kvec = acosker(theta, x, diag=True) # Self-kernel (N,)

        eigvals, eigvecs = eigh(K_tilde)    # Eigendecomposition
        ikeep = eigvals > threshold
        B = eigvecs[:, ikeep]               # (M, n_b)
        K_tilde_b = diag(eigvals[ikeep])    # DIAGONAL in eigenspace

        # Reproject variational params to new eigenspace
        V_b = B_new.T @ (B_old @ V_b @ B_old.T) @ B_new
        m_b = B_new.T @ B_old @ m_b

    # 2. E-STEP: Newton loop (nEstep iterations, early stops ~3-5)
    for _ in range(nEstep):
        m_b, V_b = Estep(r, KKtilde_inv, m_b, ...)
        lambda_m, lambda_var = lambda_moments(...)
        f_mean = exp(A*lambda_m + 0.5*A²*lambda_var + lambda0)
        # Stability check, early stopping

    # 3. F-STEP: Optimize A with LBFGS (analytical lambda0)
    lambda0 = lambda0_given_logA(logA, r, lambda_m, lambda_var)
    LBFGS([logA], closure=closure_f_params)

    # 4. M-STEP: Optimize kernel hyperparameters (skip last iteration)
    if nMstep > 0 and iteration < maxiter - 1:
        LBFGS(theta.values(), closure=closure_hyperparams)
```

### Key Observations

1. **E-step and F-step are sequential** - Despite code having `for i_estep in range(1):`, this is hard-coded to 1 iteration (explicitly commented as "FAKE" loop)

2. **Eigenspace makes inverse trivial** - K̃_b = diag(eigenvalues), so K̃_b⁻¹ = diag(1/eigenvalues)

3. **V_b is NOT diagonal** - Only K̃_b is diagonal in eigenspace

4. **Reprojection needed after M-step** - When kernel changes, eigenspace changes, must transform m_b, V_b

---

## 3. E-step Math Discrepancy (DEFERRED)

### Background

The original varGP code has a mathematically **incorrect m_new formula** that works empirically.

### What We Implemented

**We used the OLD formula** (matching vargp_old) to get a working baseline:

```python
# V update: V_new = solve(I + K_tilde @ G, K_tilde)
V_b_new = torch.linalg.solve(eye + state.K_tilde_b @ G_b, state.K_tilde_b)

# m update: m_new = V_new @ (G @ m + g)  -- OLD FORMULA
m_b_new = V_b_new @ (G_b @ state.m_b + g_b)
```

### The Correct Formula (NOT IMPLEMENTED)

```python
m_new = m + K̃ @ solve(K̃ + G, g - m)
```

### Why We Deferred

When attempting to use the "correct" formula, the model collapsed (A→0). The issue is
that g_b and G_b are **transformed** quantities (pre-multiplied by K_tilde_inv), and
the old code's formulas are designed to work with these transformed quantities.

### Future Investigation

See `.claude/ESTEP_MATH_ANALYSIS.md` for full mathematical analysis. Testing the
correct formula would require either:
1. Computing g_standard and G_standard (not transformed), or
2. Deriving the equivalent formula for transformed quantities

---

## 4. Eigenspace Projection Details

### Why Eigenspace?

From M inducing points, the effective dimensionality is typically ~10-11 (for M=50-100). Eigenspace projection:
1. Reduces computation (O(n_b) vs O(M))
2. Makes K̃⁻¹ trivial (diagonal)
3. Provides implicit regularization (drops small eigenvalues)

### Key Formulas

```python
# Eigendecomposition
eigvals, eigvecs = torch.linalg.eigh(K_tilde, UPLO='L')
threshold = max(eigvals.max() * 1e-4, 1e-4)  # EIGVAL_TOL = 1e-4
ikeep = eigvals > threshold
B = eigvecs[:, ikeep]       # (M, n_b)
eigvals_b = eigvals[ikeep]  # (n_b,)

# Projected quantities
K_tilde_b = diag(eigvals_b)           # (n_b, n_b) - DIAGONAL!
K_tilde_inv_b = diag(1/eigvals_b)     # (n_b, n_b) - DIAGONAL!
m_b = B.T @ m                         # (n_b,)
V_b = B.T @ V @ B                     # (n_b, n_b) - NOT diagonal
K_b = K @ B                           # (N, n_b)

# Efficient K @ K̃⁻¹ computation
# Since K_tilde_inv_b is diagonal:
KKtilde_inv_b = K_b * (1/eigvals_b)   # Element-wise multiplication!
```

### Reprojection After M-step

When M-step changes kernel hyperparameters, K_tilde changes, so eigenspace changes:

```python
# After M-step, have new B_new from new K_tilde
V_b_new = B_new.T @ (B_old @ V_b_old @ B_old.T) @ B_new
m_b_new = B_new.T @ B_old @ m_b_old
```

---

## 5. GPyTorch Kernel as Standalone Calculator

### The Key Insight

`ArcCosineKernel` can be used WITHOUT GPyTorch's variational framework:

```python
from kernels import ArcCosineKernel

# Create kernel with RF parameters
kernel = ArcCosineKernel(
    n_px_side=108,
    sigma_0=1.0,
    Amp=1.0,
    beta=0.1,
    rho=0.1,
    eps_0x=0.0,
    eps_0y=0.0,
    use_mask=True,
    gradient_mode='autograd'  # Use autograd for M-step
)
kernel = kernel.double().to(device)

# Compute kernel matrices directly
K_tilde = kernel(X_tilde, X_tilde).evaluate()  # (M, M)
K = kernel(X, X_tilde).evaluate()              # (N, M)
Kvec = kernel(X, diag=True)                    # (N,)

# Mask is computed internally
C, mask = kernel._compute_C_matrix(apply_mask=kernel.use_mask)
```

### Key Differences from Original acosker()

| Aspect | Original acosker | GPyTorch ArcCosineKernel |
|--------|-----------------|-------------------------|
| Input shape | (n_features, n_points) transposed | (n_points, n_features) native |
| C matrix | Passed externally from localker() | Computed internally from RF params |
| Mask | Manual pixel selection | Automatic via use_mask flag |
| Gradients | Explicit dK dict | PyTorch autograd |

---

## 6. Lambda Moments in Eigenspace

The posterior moments at training points:

```python
def lambda_moments_eigenspace(state):
    """Compute GP posterior moments using eigenspace quantities.

    lambda_m = K @ K̃⁻¹ @ m
    lambda_var = Kvec + diag(K @ K̃⁻¹ @ (V - K̃) @ K̃⁻¹ @ K^T)

    In eigenspace:
    lambda_m = KKtilde_inv_b @ m_b
    lambda_var = Kvec + diag(a @ (V_b - K_tilde_b) @ a^T)
              = Kvec + sum(a * (a @ (V_b - K_tilde_b)), dim=1)
    """
    a = state.KKtilde_inv_b  # (N, n_b)

    lambda_m = a @ state.m_b  # (N,)

    V_minus_K = state.V_b - state.K_tilde_b  # (n_b, n_b)
    aV = a @ V_minus_K                        # (N, n_b)
    lambda_var = state.Kvec + (a * aV).sum(dim=1)  # (N,)

    lambda_var = torch.clamp(lambda_var, min=1e-6)  # Numerical stability
    return lambda_m, lambda_var
```

---

## 7. M-step with LBFGS and Autograd

### Decision

Use **autograd for gradients** instead of analytical gradients:
- Simpler to implement
- `gradient_mode='autograd'` is default in ArcCosineKernel
- Performance is similar (VJP analytical is same speed)

### Implementation

```python
def mstep_lbfgs_autograd(kernel, likelihood, X, X_tilde, r, state, n_mstep, lr):
    """M-step with LBFGS using PyTorch autograd."""

    optimizer = torch.optim.LBFGS(
        kernel.parameters(),
        lr=lr,
        max_iter=n_mstep,
        tolerance_change=1e-9,
        tolerance_grad=1e-7,
        history_size=100,
        line_search_fn='strong_wolfe'
    )

    def closure():
        optimizer.zero_grad()

        # Check bounds - return inf if out of bounds
        if not _check_kernel_bounds(kernel):
            return torch.tensor(float('inf'))

        # Compute kernels WITH gradients
        K_tilde = kernel(X_tilde_masked, X_tilde_masked).evaluate()
        K = kernel(X_masked, X_tilde_masked).evaluate()
        Kvec = kernel(X_masked, diag=True).evaluate()

        # Project into FIXED eigenspace (state.B unchanged during M-step)
        K_tilde_b = state.B.T @ K_tilde @ state.B
        K_b = K @ state.B
        K_tilde_inv_b = torch.linalg.solve(K_tilde_b, torch.eye(...))

        # Compute moments with fixed m_b, V_b
        # ... (lambda_m, lambda_var, f_mean)

        # Loss = -log_lik + KL
        loss = -log_lik + KL
        loss.backward()  # Autograd computes gradients
        return loss

    optimizer.step(closure)
    kernel.clamp_hyperparameters()
```

---

## 8. Timing Requirements

The training loop MUST track and return timing like existing implementations:

```python
def train_vargp_direct(...) -> Dict:
    time_estep_total = 0.0
    time_mstep_total = 0.0

    for iteration in range(1, n_iterations):
        # E-step timing (includes F-step like vargp_style)
        start_estep = time.time()
        # ... E-step Newton loop ...
        # ... F-step ...
        time_estep_total += time.time() - start_estep

        # M-step timing
        start_mstep = time.time()
        # ... M-step ...
        time_mstep_total += time.time() - start_mstep

    return {
        'losses': losses,
        'state': state,
        'time_estep_total': time_estep_total,
        'time_mstep_total': time_mstep_total,
    }
```

---

## 9. File Structure (IMPLEMENTED)

### Files Created

```
gpytorch_porting/
├── eigenspace.py       # Eigenspace projection utilities
│   ├── EIGVAL_TOL = 1e-4
│   ├── compute_eigenspace(K_tilde)
│   ├── project_to_eigenspace(B, m, V, K)
│   ├── reproject_variational_params(B_old, B_new, m_b, V_b)
│   ├── compute_KKtilde_inv_b(K_b, eigvals_b)
│   ├── compute_K_tilde_b_diagonal(eigvals_b)
│   └── compute_K_tilde_inv_b_diagonal(eigvals_b)
│
├── direct_vargp.py     # Main implementation
│   ├── DirectVariationalState (dataclass)
│   ├── compute_kernels_direct(kernel, X, X_tilde)
│   ├── recompute_kernels_after_mstep(kernel, X, X_tilde, state)
│   ├── lambda_moments_eigenspace(state)
│   ├── estep_eigenspace(state, r, A, f_mean)  # Uses OLD formula
│   ├── compute_f_mean(lambda_m, lambda_var, A, lambda0)
│   ├── compute_elbo_eigenspace(state, r, ...)
│   ├── fstep_direct(likelihood, r, lambda_m, lambda_var, ...)
│   ├── mstep_lbfgs_autograd(kernel, likelihood, ...)
│   ├── train_vargp_direct(...)
│   └── predict_direct(kernel, likelihood, state, X_tilde, X_test)
```

### Files Modified

```
run_single_mode.py
├── Added 'vargp_direct' to --mode choices
├── Added training block for vargp_direct mode
└── Handles prediction with predict_direct()
```

---

## 10. Implementation Stages

### Stage 1: eigenspace.py
- Implement and test eigenspace utilities
- Verify eigendecomposition correctness
- Test reprojection preserves structure

### Stage 2: DirectVariationalState
- Create dataclass for state management
- Implement compute_kernels_direct()
- Implement recompute_kernels_after_mstep()

### Stage 3: E-step
- Implement lambda_moments_eigenspace()
- Implement estep_eigenspace() with CORRECT formula
- Add prominent documentation about math discrepancy

### Stage 4: M-step
- Implement mstep_lbfgs_autograd()
- Implement compute_KL_eigenspace()
- Add bounds checking

### Stage 5: Training Loop
- Implement train_vargp_direct()
- Implement predict_direct()
- Add timing tracking

### Stage 6: Integration
- Add vargp_direct mode to run_single_mode.py
- Run validation tests
- Compare to vargp_old

---

## 11. Validation Results (COMPLETE)

### Comparison: vargp_direct vs vargp_old

Test configuration: M=50, 50 iterations, n_estep=10, n_fstep=10, n_mstep=10, seed=123

| Metric | vargp_old | vargp_direct | Notes |
|--------|-----------|--------------|-------|
| Test Pearson r | **0.84** | 0.81 | Slightly lower |
| Explained var | **0.89** | 0.86 | Slightly lower |
| Final loss | 428 | **402** | Lower loss but less generalization |
| Training time | **6.2s** | 18.8s | 3x slower |
| E-step time | 1.1s | 0.7s | Faster (eigenspace) |
| M-step time | **4.8s** | 17.3s | 3.6x slower (autograd vs analytical) |

### Success Criteria

- [x] Eigenspace dimension reduces from M to ~42-47 (for M=50)
- [x] Test correlation within 0.05 of vargp_old (0.81 vs 0.84)
- [ ] Timing breakdown similar to vargp_old (M-step is 3.6x slower)
- [x] No NaN/crashes on PNAS data

### Known Issues

1. **M-step is slow**: Uses autograd instead of analytical gradients
2. **Slightly lower test performance**: Despite lower training loss, suggesting possible overfitting

### Command to Run

```bash
python run_single_mode.py --mode vargp_direct --ntilde 50 --n-iterations 50 \
    --n-estep 10 --n-fstep 10 --n-mstep 10 --seed 123
```

---

## 12. Key Reference Files

| File | Lines | What to Reference |
|------|-------|-------------------|
| `utils.py` | 5293-5975 | Complete loop structure |
| `utils.py` | 4244-4256 | Original E-step (note m_new is wrong) |
| `utils.py` | 5619-5627 | Reprojection after M-step |
| `utils.py` | 5435-5444 | Eigenspace computation |
| `kernels.py` | 412-514 | ArcCosineKernel forward() |
| `estep.py` | 248-293 | Newton formulas reference |
| `fstep.py` | 130-252 | f_step_lbfgs to reuse |
| `.claude/ESTEP_MATH_ANALYSIS.md` | all | E-step math derivation |

---

## 13. Common Pitfalls Encountered

1. **Don't forget reprojection after M-step** - Eigenspace changes when kernel changes

2. **K̃_b is diagonal, V_b is NOT** - Only K̃_b becomes diagonal in eigenspace

3. **E-step formulas must match transformed g, G** - The g_b and G_b computed from
   KKtilde_inv_b are TRANSFORMED quantities. The old code's formulas work with these.
   Using "correct" formulas with transformed quantities causes collapse.

4. **LBFGS closure called multiple times** - Can't use loss.backward() inside LBFGS
   closure with strong_wolfe line search. Use torch.autograd.grad() instead.

5. **Symmetrize V after update** - `V = (V + V.T) / 2` for numerical stability

6. **Check bounds in M-step closure** - Return inf to reject LBFGS step

7. **Clamp hyperparameters after M-step** - `kernel.clamp_hyperparameters()`

---

## 14. M-step Analytical Gradients - Implementation Requirements

### 14.1 Why Current VJP Cannot Be Used for M-step Precomputation

**Investigation Result (January 2025)**: The existing `analytical_gradients_vjp.py` does NOT compute dK/dθ matrices explicitly. It only does backward-pass chaining.

**How VJP Works:**
```python
# Forward pass: computes K, saves intermediates
K = VJPGradients.apply(x1, x2, sigma_0, Amp, ...)  # Returns K only

# Backward pass (given dL/dK from optimizer):
# 1. Computes dL/dC ONCE via chain rule
# 2. Chains to each hyperparameter via element-wise ops
# dL/dθ = sum(dL/dC * dC/dθ)  for each θ
```

**What M-step Needs (from original vargp_old):**
```python
# Compute K and all dK/dθ matrices ONCE at start
K, dK = acosker(theta, x, xtilde, C, dC, grad=True)
# dK = {'sigma_0': dK/dσ₀, 'Amp': dK/dAmp, ...}  # All explicit matrices

# Then in LBFGS closure, reuse cached dK:
for key in dK:
    grad[key] = (dL_dK * dK[key]).sum()  # Fast element-wise
```

**Why VJP Cannot Work:**
1. VJP requires dL/dK from the optimizer at backward time
2. LBFGS line search evaluates loss at different step sizes - each has different dL/dK
3. VJP would need to run backward for EVERY line search evaluation
4. No way to precompute and cache dK/dθ matrices

### 14.2 What Needs to Be Implemented

**New function needed**: `compute_kernel_with_explicit_grads()` that returns both K and all dK/dθ matrices:

```python
def compute_kernel_with_explicit_grads(kernel, X1, X2, C, dC, diag=False):
    """Compute kernel and all gradient matrices.

    Returns:
        K: Kernel matrix (N1, N2) or (N1,) if diag=True
        dK: Dict of gradient matrices, same shape as K
            {'sigma_0': dK/dσ₀, 'Amp': dK/dAmp, 'beta': dK/dβ,
             'rho': dK/dρ, 'eps_0x': dK/dξₓ, 'eps_0y': dK/dξᵧ}
    """
    # Port formulas from utils.py:acosker() lines 3650-3813
```

**Reference**: `utils.py:acosker()` with `grad=True` computes exactly this.

### 14.3 Diagonal Kernel Gradients (EXACT from utils.py:3784-3813)

**Finding**: The original varGP uses a **direct formula** for diagonal kernel gradients, NOT extracted from full matrix.

**Diagonal kernel Kvec computation:**
```python
# K[i] = x1[i]ᵀ @ C @ x1[i] + σ₀²
K = torch.sum(x1 * torch.matmul(C, x1), dim=0)[:, None] + sigma_0**2
K = K.squeeze()  # Shape (n1,)
```

**dKvec/d(sigma_0) - SPECIAL CASE:**
```python
# dK/dσ₀ = 2·σ₀ (derivative of σ₀² term)
ones = torch.ones((n1, 1), device=DEVICE, dtype=TORCH_DTYPE)
dK['sigma_0'] = (2*sigma_0**2*ones).squeeze() / sigma_0  # = 2*sigma_0
```

**dKvec/d(C-params) - DIRECT FORMULA:**
```python
# dK[key][i] = x1[i]ᵀ @ dC/dθ[key] @ x1[i]
for key in dC.keys():
    if key == 'sigma_0':
        continue
    dK[key] = torch.sum(x1 * torch.matmul(dC[key], x1), dim=0)  # Shape (n1,)
```

**Full K gradients (non-diagonal, for comparison):**
```python
# Complex formula involving arccos chain rule
dX1 = 0.5 * sum(x1 * (dC[key] @ x1), dim=0) / X1
dX2 = 0.5 * sum(x2 * (dC[key] @ x2), dim=0) / X2
dX1X2 = dX1 * X2 + X1 * dX2
dcosdelta = (x1.T @ dC[key] @ x2 - cosdelta * dX1X2) / X1X2
dJ = -(delta - pi) * dcosdelta / pi
dK[key] = X1X2 * dJ + dX1X2 * J  # (N1, N2) matrix
```

**Key differences:**

| Aspect | Full Matrix (diag=False) | Diagonal (diag=True) |
|--------|--------------------------|----------------------|
| sigma_0 | `(X1X2 * dJ_sigma + dX1X2_sigma * J) / sigma_0` | `2*sigma_0` |
| C-params | Complex with arccos chain rule | Simple: `sum(x * (dC @ x))` |
| Shape | `(N1, N2)` | `(N,)` |
| Complexity | O(N1*N2) | O(N) |

**Implementation Note**: When implementing `compute_kernel_with_explicit_grads()`, handle `diag=True` as a SEPARATE code path with the direct formula, not by computing full matrix and taking diagonal.

### 14.4 M-step Loss Function

**Confirmed**: The M-step optimizes the ELBO (Expected Log-Likelihood - KL divergence):

```python
# From utils.py M-step:
loss = -E[log p(r|λ)] + KL(q(u) || p(u))

# Where:
E[log p(r|λ)] = Σᵢ [rᵢ(Aλₘᵢ + λ₀) - exp(Aλₘᵢ + ½A²λᵥᵢ + λ₀)]
KL = 0.5 * (tr(K̃⁻¹V) + m^T K̃⁻¹ m - n_b + log|K̃|/|V|)
```

### 14.5 Gradient Chain for M-step (EXACT FORMULAS)

**Step 1: dlambda_m and dlambda_var (from utils.py:3942-3952)**

```python
# Intermediate: derivative of projection vector a = K @ K_tilde_inv
da[key] = (dK[key] - a @ dK_tilde[key]) @ K_tilde_inv   # (nt, ntilde)

# Mean gradient
dlambda_m[key] = da[key] @ m                            # (nt, 1)

# Variance gradient (4 terms)
dlambda_var[key] = (
    dK_vec[key]                                          # dKvec
    + torch.einsum('ij,ji->i', 2*da[key], V @ a.T)      # 2*diag(da @ V @ a.T)
    - torch.einsum('ij,ij->i', dK[key], a)              # -diag(dK @ a.T)
    - torch.einsum('ij,ij->i', K, da[key])              # -diag(K @ da.T)
)                                                        # (nt,)
```

**Step 2: dloglikelihood (from utils.py:4111-4117)**

```python
dloglikelihood[key] = (
    A * r @ dlambda_m[key]
    - A * f_mean @ dlambda_m[key]
    - 0.5 * A**2 * f_mean @ dlambda_var[key]
)
```

**Step 3: dKL (from utils.py:4143-4150)**

```python
c = V @ K_tilde_inv                     # (ntilde, ntilde)
b = K_tilde_inv @ m                     # (ntilde, 1)

for key in dK_tilde.keys():
    B = dK_tilde[key] @ K_tilde_inv     # (ntilde, ntilde)
    dKL[key] = (
        0.5 * torch.trace(B)            # d(log|K_tilde|)/dθ
        - 0.5 * torch.trace(c @ B)      # d(trace(V @ K_tilde_inv))/dθ
        - 0.5 * b.T @ (B @ m)           # d(m.T @ K_tilde_inv @ m)/dθ
    )
```

**Step 4: Final gradient**

```python
dlogmarginal[key] = dloglikelihood[key] - dKL[key]
theta[key].grad = -dlogmarginal[key]    # Negate for minimization
```

### 14.6 Parameter Transforms (GPyTorch vs Original)

**Original varGP** uses log-space for some parameters:
- `beta = exp(-raw/2) / 2` where raw = `-2log2beta`
- `rho = sqrt(exp(-raw) / 2)` where raw = `-log2rho2`
- `sigma_0`, `Amp` are direct (positive)

**GPyTorch** uses Positive constraint (softplus) for sigma_0, Amp:
- `param = softplus(raw_param)`
- Gradient needs chain rule: `d/d(raw) = d/d(param) * sigmoid(raw)`

**Recommendation**: Work with raw parameters directly like vargp_old does, bypassing GPyTorch constraints during M-step optimization. Then clamp to valid ranges afterward.

---

## 15. Future Work

1. **Implement analytical M-step gradients** - Port `acosker(..., grad=True)` to compute explicit dK/dθ matrices
2. **Investigate correct m_new formula** - May improve generalization
3. **Unit tests for eigenspace utilities** - Not yet implemented

---

*Last updated: January 2025 (Added M-step analytical gradient requirements)*
