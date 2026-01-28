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

## 14. Future Work

1. **Analytical M-step gradients** - Would make M-step 3-4x faster
2. **Investigate correct m_new formula** - May improve generalization
3. **Unit tests for eigenspace utilities** - Not yet implemented

---

*Last updated: January 2025 (Implementation complete)*
