# vargp_direct vs vargp_old Investigation Context

**Purpose**: Comprehensive technical context for investigating why `vargp_direct` (GPyTorch-based) produces different results from `vargp_old` (original utils.py implementation).

**Problem Statement**: On a 10-cell benchmark with ntrain=500,2000 and M=50,250, `vargp_direct --mstep-analytical --float32` shows:
- Some cells match well (8, 18: r ≈ same)
- Some cells completely fail (6, 15: r << expected)
- General tendency to underperform

**Goal**: Identify and fix all discrepancies so both implementations produce identical results.

---

## Table of Contents

1. [File Locations](#1-file-locations)
2. [Algorithm Overview](#2-algorithm-overview)
3. [Detailed Component Comparison](#3-detailed-component-comparison)
4. [Potential Bugs Identified](#4-potential-bugs-identified)
5. [Precise Test Specifications](#5-precise-test-specifications)
6. [How to Run Tests](#6-how-to-run-tests)

---

## 1. File Locations

### Original Implementation (vargp_old)
```
utils.py:localker()          - Lines 3577-3631 (C matrix and dC gradients)
utils.py:acosker()           - Lines 3663-3813 (kernel K and dK gradients)
utils.py:lambda_moments()    - Lines 3906-3956 (posterior moments and gradients)
utils.py:Estep()             - Lines 4217-4277 (Newton update for m, V)
utils.py:compute_loglikelihood() - Lines 4064-4119 (log-likelihood and gradients)
utils.py:compute_KL_div()    - Lines 4121-4152 (KL divergence and gradients)
utils.py:varGP()             - Lines 5291-5975 (main training loop)
```

### New Implementation (vargp_direct)
```
direct_vargp.py:compute_C_and_gradients()              - Lines 52-134
direct_vargp.py:compute_kernel_and_gradients()         - Lines 137-260
direct_vargp.py:compute_lambda_moments_and_gradients() - Lines 263-359
direct_vargp.py:compute_loss_gradients()               - Lines 362-443
direct_vargp.py:estep_eigenspace()                     - Lines 634-691
direct_vargp.py:mstep_lbfgs_analytical()               - Lines 977-1162
direct_vargp.py:train_vargp_direct()                   - Lines 1165-1304
eigenspace.py:compute_eigenspace()                     - Lines 22-60
eigenspace.py:compute_KKtilde_inv_b()                  - Lines 141-168
```

---

## 2. Algorithm Overview

Both implementations follow the same EM structure:

```
for iteration in 1..n_iterations:

    # 1. KERNEL RECOMPUTATION (after iteration 1, if M-step enabled)
    if n_mstep > 0 and iteration > 1:
        C, mask, dC = localker(theta, grad=True)
        K_tilde, dK_tilde = acosker(xtilde, xtilde, C, dC)
        K, dK = acosker(x, xtilde, C, dC)
        Kvec, dKvec = acosker(x, diag=True, C, dC)

        # Eigenspace projection
        eigvals, eigvecs = eigh(K_tilde)
        B = eigvecs[:, eigvals > threshold]
        K_tilde_b = diag(eigvals[kept])   # DIAGONAL

        # Reproject m_b, V_b to new eigenspace
        V_b = B_new.T @ (B_old @ V_b @ B_old.T) @ B_new
        m_b = B_new.T @ B_old @ m_b

    # 2. E-STEP: Newton updates (n_estep iterations)
    for _ in range(n_estep):
        a = K_b @ K_tilde_inv_b          # Projection vector
        g = A * a.T @ (r - f_mean)       # Gradient
        G = A² * a.T @ diag(f_mean) @ a  # Hessian

        V_new = solve(I + K_tilde_b @ G, K_tilde_b)
        m_new = V_new @ (G @ m + g)      # OLD FORMULA

        lambda_m = a @ m_b
        lambda_var = Kvec + diag(a @ (V_b - K_tilde_b) @ a.T)
        f_mean = exp(A * lambda_m + 0.5 * A² * lambda_var + lambda0)

    # 3. F-STEP: Optimize A, compute lambda0 analytically
    lambda0 = log(sum(r) / sum(exp(A * lambda_m + 0.5 * A² * lambda_var)))
    LBFGS(logA, closure)

    # 4. M-STEP: Optimize kernel hyperparameters (skip last iteration)
    if n_mstep > 0 and iteration < n_iterations:
        LBFGS(theta, closure)  # Uses precomputed dK/dθ
```

---

## 3. Detailed Component Comparison

### 3.1 C Matrix Computation

**Original (utils.py:localker lines 3577-3631)**:
```python
# Coordinate grid
ycord, xcord = torch.meshgrid(linspace(-1, 1, n_px_side), linspace(-1, 1, n_px_side), indexing='ij')
xcord = xcord.flatten()  # Shape: (11664,)
ycord = ycord.flatten()

# Locality weights
logalpha = -exp(theta['-2log2beta']) * ((xcord - eps_0x)² + (ycord - eps_0y)²)
alpha_local = exp(logalpha)

# Mask: keep pixels where alpha >= 0.001
mask = alpha_local >= 0.001
alpha_local = alpha_local[mask]  # Cropped
xcord = xcord[mask]
ycord = ycord[mask]

# Smoothness kernel
logCsmooth = -exp(theta['-log2rho2']) * ((xcord[:, None] - xcord[None, :])² + (ycord[:, None] - ycord[None, :])²)
C_smooth = exp(logCsmooth)

# Full C matrix
C = theta['Amp'] * alpha_local[:, None] * C_smooth * alpha_local[None, :]
C = (C + C.T) / 2  # Symmetrize
```

**Our implementation (direct_vargp.py:compute_C_and_gradients lines 52-134)**:
```python
# Get parameters
Amp = kernel.Amp.squeeze()
eps_0x = kernel.eps_0x.squeeze()
eps_0y = kernel.eps_0y.squeeze()
beta_factor = torch.exp(kernel.raw_m2log2beta.squeeze())  # = exp(-2log2beta)
rho2_factor = torch.exp(kernel.raw_mlog2rho2.squeeze())   # = exp(-log2rho2)

# Get pixel coordinates (from kernel)
xcord = kernel.xcord.clone()
ycord = kernel.ycord.clone()

# Compute mask
mask = kernel.compute_mask()
xcord = xcord[mask]
ycord = ycord[mask]

# Same computation as original...
```

**POTENTIAL DIFFERENCE #1: Coordinate System**
- Original: Creates `xcord, ycord` in `localker()` with `indexing='ij'`
- Ours: Uses `kernel.xcord, kernel.ycord` (computed in `_setup_pixel_coords()`)
- **CHECK**: Are the coordinates identical? Same ordering? Same range [-1, 1]?

**POTENTIAL DIFFERENCE #2: Mask Computation**
- Original: Computes mask inside `localker()` using current theta values
- Ours: Calls `kernel.compute_mask()` which may use DETACHED parameters
- **CHECK**: Is the mask computed identically? Does detachment affect anything?

### 3.2 Kernel Computation

**Original (utils.py:acosker lines 3663-3813)**:
```python
# CRITICAL: Inputs are TRANSPOSED
x1 = x1.T  # Shape: (nx, n1) from (n1, nx)
x2 = x2.T  # Shape: (nx, n2)

# Quadratic forms
X1 = sqrt(sum(x1 * (C @ x1), dim=0) + sigma_0²)  # Shape: (n1,)
X2 = sqrt(sum(x2 * (C @ x2), dim=0) + sigma_0²)  # Shape: (n2,)

X1X2 = outer(X1, X2)           # Shape: (n1, n2)
x1x2 = x1.T @ C @ x2 + sigma_0²  # Shape: (n1, n2)

cosdelta = clip(x1x2 / (X1X2 + 1e-7), -1, 1)
delta = arccos(cosdelta)
J = (sqrt(1 - cosdelta²) + π*cosdelta - delta*cosdelta) / π
K = X1X2 * J
```

**Our implementation (direct_vargp.py:compute_kernel_and_gradients lines 137-260)**:
```python
# NO transpose - inputs stay as (n1, nx)
CX1 = x1 @ C     # Shape: (n1, nx)
V1 = (CX1 * x1).sum(dim=-1) + sigma_0²  # Shape: (n1,)

CX2 = x2 @ C     # Shape: (n2, nx)
V2 = (CX2 * x2).sum(dim=-1) + sigma_0²  # Shape: (n2,)

X1 = sqrt(V1)
X2 = sqrt(V2)
X1X2 = X1[:, None] * X2[None, :]  # Shape: (n1, n2)

x1x2 = (CX1 @ x2.T) + sigma_0²    # Shape: (n1, n2)
# ... rest same
```

**POTENTIAL DIFFERENCE #3: Matrix Multiplication Order**
- Original: `x1.T @ C @ x2` where x1 is (nx, n1), C is (nx, nx), x2 is (nx, n2)
- Ours: `(x1 @ C) @ x2.T` where x1 is (n1, nx), C is (nx, nx), x2 is (n2, nx)
- **Mathematically equivalent** BUT numerical precision may differ
- **CHECK**: Compute both and compare values

**POTENTIAL DIFFERENCE #4: Clipping**
- Original: `clip(x1x2 / (X1X2 + 1e-7), -1, 1)`
- Ours: `clamp(x1x2 / (X1X2 + eps), -1.0 + eps, 1.0 - eps)` where eps=1e-7
- **Different at boundary**: We never reach exactly ±1
- **CHECK**: Does this affect arccos computation?

### 3.3 Diagonal Kernel (Kvec)

**Original (utils.py:acosker lines 3783-3800)**:
```python
# Diagonal case
K = sum(x1 * (C @ x1), dim=0)[:, None] + sigma_0²
K = K.squeeze()  # Shape: (n1,)

# Gradients
dK['sigma_0'] = (2*sigma_0² * ones).squeeze() / sigma_0  # = 2*sigma_0
for key in dC.keys():
    dK[key] = sum(x1 * (dC[key] @ x1), dim=0)
```

**Our implementation (direct_vargp.py lines 173-191)**:
```python
# Diagonal case: K[i] = V1[i]
K = V1  # Already computed as (CX1 * x1).sum(-1) + sigma_0²

dK['sigma_0'] = torch.full_like(K, 2 * sigma_0.item())
for key, dC_val in dC.items():
    dCX1 = x1 @ dC_val
    dK[key] = (dCX1 * x1).sum(dim=-1)
```

**MATCH**: Same formula, just different input convention.

### 3.4 E-step

**Original (utils.py:Estep lines 4217-4256)**:
```python
A = exp(f_params['logA'])
g = A * KKtilde_inv.T @ (r - f_mean)
G = A*A * KKtilde_inv.T @ (KKtilde_inv * f_mean[:, None])

V_new = solve(I + K_tilde @ G, K_tilde)
m_new = V_new @ (G @ m + g)
V_new = (V_new + V_new.T) / 2
```

**Our implementation (direct_vargp.py:estep_eigenspace lines 634-691)**:
```python
a = state.KKtilde_inv_b  # Shape: (N, n_b)
g_b = A * (a.T @ (r - f_mean))  # Shape: (n_b,)
G_b = A*A * (a.T @ (f_mean[:, None] * a))  # Shape: (n_b, n_b)

V_b_new = solve(I + K_tilde_b @ G_b, K_tilde_b)
m_b_new = V_b_new @ (G_b @ m_b + g_b)
V_b_new = (V_b_new + V_b_new.T) / 2
```

**POTENTIAL DIFFERENCE #5: Eigenspace Projection**
- Original: Works in full M-dimensional space, then projects
- Ours: Works directly in eigenspace from the start
- **Should be equivalent** because eigenspace is just a basis change
- **CHECK**: Compare g, G values before and after projection

**POTENTIAL DIFFERENCE #6: G computation**
- Original: `KKtilde_inv.T @ (KKtilde_inv * f_mean[:, None])`
  - This is `a.T @ diag(f_mean) @ a` where a = KKtilde_inv
- Ours: `a.T @ (f_mean[:, None] * a)`
  - Same formula, just different expression
- **MATCH**

### 3.5 Lambda Moments

**Original (utils.py:lambda_moments lines 3906-3956)**:
```python
a = KKtilde_inv  # Shape: (nt, ntilde)
lambda_m = matmul(a, m)  # Shape: (nt, 1)

# Variance formula
lambda_var = Kvec + sum(-K.T * a.T + a.T * (V @ a.T), 0)
```

**Our implementation (direct_vargp.py:lambda_moments_eigenspace lines 599-631)**:
```python
a = state.KKtilde_inv_b  # Shape: (N, n_b)
lambda_m = a @ state.m_b  # Shape: (N,)

V_minus_K = state.V_b - state.K_tilde_b
aV = a @ V_minus_K
lambda_var = state.Kvec + (a * aV).sum(dim=1)
```

**POTENTIAL DIFFERENCE #7: Variance Formula**
- Original: `Kvec + sum(-K.T * a.T + a.T * (V @ a.T), 0)`
  - = Kvec - diag(K @ a.T) + diag(a @ V @ a.T)
  - = Kvec + diag(a @ (V - K) @ a.T)  where K here is K_tilde
- Ours: `Kvec + sum(a * (a @ (V_b - K_tilde_b)), dim=1)`
  - = Kvec + diag(a @ (V_b - K_tilde_b) @ a.T)
- **MATCH** (same formula)

### 3.6 M-step Gradient Computation

**Original (utils.py:lambda_moments with gradients, lines 3942-3952)**:
```python
for key in dK.keys():
    da[key] = (dK[key] - a @ dK_tilde[key]) @ K_tilde_inv
    dlambda_m[key] = da[key] @ m
    dlambda_var[key] = (dK_vec[key]
                       + einsum('ij,ji->i', 2*da[key], V @ a.T)
                       - einsum('ij,ij->i', dK[key], a)
                       - einsum('ij,ij->i', K, da[key]))
```

**Our implementation (direct_vargp.py:compute_lambda_moments_and_gradients lines 328-357)**:
```python
for key in dK_b.keys():
    da_key = (dK_b[key] - a @ dK_tilde_b[key]) * K_tilde_inv_b_diag[None, :]
    dlambda_m[key] = da_key @ m_b

    term1 = dKvec[key]
    term2 = 2 * einsum('ij,ji->i', da_key, Va_T)
    term3 = -einsum('ij,ij->i', dK_b[key], a)
    term4 = -einsum('ij,ij->i', K_b, da_key)
    dlambda_var[key] = term1 + term2 + term3 + term4
```

**POTENTIAL DIFFERENCE #8: da computation in eigenspace**
- Original: `da = (dK - a @ dK_tilde) @ K_tilde_inv`
- Ours: `da = (dK_b - a @ dK_tilde_b) * K_tilde_inv_b_diag` (element-wise)
- In eigenspace, K_tilde_inv is diagonal, so matrix mult = element-wise
- **Should be equivalent** but **CHECK**: is `dK_tilde_b` computed correctly?

**POTENTIAL DIFFERENCE #9: Projection of dK_tilde**
- Original: `dK_tilde` is in full M-dim space
- Ours: `dK_tilde_b = B.T @ dK_tilde @ B`
- **CHECK**: Is this projection correct? Does it preserve gradient structure?

### 3.7 KL Divergence Gradient

**Original (utils.py:compute_KL_div lines 4143-4150)**:
```python
c = V @ K_tilde_inv
b = K_tilde_inv @ m

for key in dK_tilde.keys():
    B = dK_tilde[key] @ K_tilde_inv
    dKL[key] = 0.5*trace(B) - 0.5*trace(c @ B) - 0.5*b.T @ (B @ m)
```

**Our implementation (direct_vargp.py:compute_loss_gradients lines 416-436)**:
```python
c = V_b * K_tilde_inv_b_diag[None, :]  # Diagonal on right
b = K_tilde_inv_b_diag * m_b

for key in dK_tilde_b.keys():
    B = dK_tilde_b[key] * K_tilde_inv_b_diag[None, :]
    term1 = 0.5 * trace(B)
    term2 = -0.5 * trace(c @ B)
    term3 = -0.5 * (b @ (B @ m_b))
    dKL[key] = term1 + term2 + term3
```

**POTENTIAL DIFFERENCE #10: Using projected dK_tilde_b**
- Original works in full space, ours in eigenspace
- **Should be equivalent** but need to verify eigenspace projection is correct

### 3.8 Parameter Transforms

**CRITICAL DIFFERENCE #11: Parameter Names and Transforms**

| Parameter | Original Name | Original Transform | Our Name | Our Transform |
|-----------|--------------|-------------------|----------|---------------|
| sigma_0 | `theta['sigma_0']` | Direct (positive) | `kernel.sigma_0` | `softplus(raw_sigma_0)` |
| Amp | `theta['Amp']` | Direct (positive) | `kernel.Amp` | `softplus(raw_Amp)` |
| beta | `theta['-2log2beta']` | `beta = exp(-raw/2) / 2` | `kernel.raw_m2log2beta` | Same |
| rho | `theta['-log2rho2']` | `rho = sqrt(exp(-raw) / 2)` | `kernel.raw_mlog2rho2` | Same |
| eps_0x | `theta['eps_0x']` | Direct | `kernel.eps_0x` | Direct |
| eps_0y | `theta['eps_0y']` | Direct | `kernel.eps_0y` | Direct |

**POTENTIAL DIFFERENCE #12: Softplus Transform for sigma_0 and Amp**
- Original: sigma_0 and Amp are stored directly (assumed positive)
- Ours: Uses `softplus(raw)` transform
- In gradient computation, we apply chain rule: `d/d(raw) = d/d(param) * sigmoid(raw)`
- **CHECK**: Is this applied correctly in `mstep_lbfgs_analytical` lines 1150-1151?

### 3.9 Eigenspace Tolerance

**Original (need to find exact value)**:
```python
# In varGP main loop
threshold = ...  # What is the exact value?
ikeep = eigvals > threshold
```

**Our implementation (eigenspace.py line 19)**:
```python
EIGVAL_TOL = 1e-4
threshold = max(eigvals.max().item() * eigval_tol, eigval_tol)
```

**CONFIRMED MATCH: Eigenvalue Threshold**
- Original: `EIGVAL_TOL = 1e-4` (utils.py line 68)
- Ours: `EIGVAL_TOL = 1e-4` (eigenspace.py line 19)
- Formula: `threshold = max(eigvals.max() * EIGVAL_TOL, EIGVAL_TOL)`
- **MATCH**: Both use identical formula and tolerance

---

## 4. Potential Bugs Identified

### BUG CANDIDATE 1: Input Shape Convention
- Original `acosker` expects `(n, nx)` and transposes internally to `(nx, n)`
- Our `compute_kernel_and_gradients` expects `(n, nx)` and keeps it
- The matrix operations are mathematically equivalent but may have different numerical precision

### BUG CANDIDATE 2: Mask Consistency
- Mask is computed from theta parameters
- During M-step, theta changes
- Does our mask stay consistent with the one used for gradient computation?

### BUG CANDIDATE 3: Eigenspace Projection of Gradients
- We project `dK_tilde_b = B.T @ dK_tilde @ B`
- This projection may not preserve the correct gradient structure
- In original, gradients are in full space, then ELBO gradient flows back to parameters
- In ours, we compute projected gradients directly

### BUG CANDIDATE 4: Softplus Chain Rule
- Lines 1150-1151 in `mstep_lbfgs_analytical`:
```python
kernel.raw_sigma_0.grad = dL['sigma_0'] * torch.sigmoid(kernel.raw_sigma_0)
kernel.raw_Amp.grad = dL['Amp'] * torch.sigmoid(kernel.raw_Amp)
```
- **Question**: Is `dL['sigma_0']` w.r.t. sigma_0 (after softplus) or raw_sigma_0?
- If it's w.r.t. sigma_0, chain rule is correct
- If not, this is wrong

### BUG CANDIDATE 5: M-step Eigenspace Fixed
- During M-step closure, we use fixed `state.B` from before M-step
- Original recomputes eigenspace only AFTER M-step completes
- Our closure uses `K_tilde_b = B.T @ K_tilde @ B` with new K_tilde but old B
- This may cause inconsistency

### BUG CANDIDATE 6: KL Divergence Eigenspace
- We compute KL using fixed `eigvals_b` from state
- But K_tilde_b is recomputed in closure with new kernel
- `log_det_K = log(eigvals_b).sum()` uses OLD eigenvalues
- This is INCONSISTENT

---

## 5. Precise Test Specifications

### TEST 0: Initialization Match
**Purpose**: Verify both implementations start from identical state.

**Setup**:
- Same seed (123)
- Same cell (8)
- ntrain=500, ntilde=50
- n_iterations=1, n_estep=0, n_fstep=0, n_mstep=0

**Compare**:
1. C matrix: `||C_old - C_new|| / ||C_old||` should be < 1e-10
2. Mask: `mask_old == mask_new` exactly
3. K_tilde: `||K_tilde_old - K_tilde_new|| / ||K_tilde_old||` < 1e-10
4. K: Same
5. Kvec: Same
6. Eigenvalues: `||eigvals_old - eigvals_new|| / ||eigvals_old||` < 1e-10
7. B: `||B_old - B_new|| / ||B_old||` < 1e-10 (up to sign)
8. Initial m_b, V_b: Both should be 0 and K_tilde_b

### TEST 1: Single E-step Match
**Purpose**: Verify E-step produces identical updates.

**Setup**:
- Same as TEST 0
- n_iterations=1, n_estep=1, n_fstep=0, n_mstep=0

**Compare** (after 1 E-step):
1. g_b: `||g_old - g_new|| / ||g_old||` < 1e-8
2. G_b: `||G_old - G_new|| / ||G_old||` < 1e-8
3. V_b_new: Same
4. m_b_new: Same
5. lambda_m: Same
6. lambda_var: Same
7. f_mean: Same

### TEST 2: Single F-step Match
**Purpose**: Verify F-step produces identical A updates.

**Setup**:
- After 1 E-step (from TEST 1)
- n_iterations=1, n_estep=1, n_fstep=10, n_mstep=0

**Compare**:
1. Initial A: Same
2. lambda0 (analytical): Same formula, same result
3. Final A after LBFGS: `|A_old - A_new| / |A_old|` < 1e-6
4. Final lambda0: Same

### TEST 3: Single M-step Match (NO kernel update yet)
**Purpose**: Verify M-step gradient computation matches.

**Setup**:
- After 1 E-step + 1 F-step
- n_iterations=1, n_estep=1, n_fstep=10, n_mstep=1 (single LBFGS step)

**Compare** (at start of M-step, before any update):
1. dC matrices: For each key, `||dC_old[key] - dC_new[key]|| / ||dC_old[key]||` < 1e-8
2. dK_tilde matrices: Same
3. dK matrices: Same
4. dKvec vectors: Same
5. dlambda_m: Same
6. dlambda_var: Same
7. dloglikelihood: Same
8. dKL: Same
9. Final gradient for each parameter: Same

### TEST 4: M-step Parameter Update Match
**Purpose**: Verify LBFGS updates parameters identically.

**Setup**:
- Same as TEST 3 but with n_mstep=10

**Compare**:
1. Parameter values after M-step
2. Loss trajectory (if accessible)

### TEST 5: Full Iteration Match
**Purpose**: Verify complete EM iteration matches.

**Setup**:
- n_iterations=1, n_estep=10, n_fstep=10, n_mstep=10

**Compare**:
1. Final m_b, V_b
2. Final A, lambda0
3. Final kernel parameters
4. Final ELBO/loss

### TEST 6: Multi-Iteration Match
**Purpose**: Verify eigenspace reprojection works correctly.

**Setup**:
- n_iterations=5, n_estep=10, n_fstep=10, n_mstep=10

**Compare**:
1. Eigenspace dimension after each iteration
2. Loss trajectory
3. Final test correlation

### TEST 7: Problem Cell Analysis (Cell 6)
**Purpose**: Understand why cell 6 completely fails.

**Setup**:
- cell=6, ntrain=500, ntilde=50
- Run both implementations with verbose output

**Investigate**:
1. At what iteration does divergence begin?
2. What quantity first shows discrepancy?
3. Are there NaN/Inf values appearing?

---

## 6. How to Run Tests

### Command Template
```bash
cd /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting
source ~/anaconda3/bin/activate pytorch_gpytorch

# For vargp_old
python run_single_mode.py --mode vargp_old --cell 8 --ntilde 50 --n-train 500 \
    --n-iterations 1 --n-estep 0 --n-fstep 0 --n-mstep 0 --seed 123

# For vargp_direct (analytical)
python run_single_mode.py --mode vargp_direct --mstep-analytical --float32 --cell 8 \
    --ntilde 50 --n-train 500 --n-iterations 1 --n-estep 0 --n-fstep 0 --n-mstep 0 --seed 123
```

### Creating a Diagnostic Script

Create `tests/test_vargp_direct_match.py` with functions:
```python
def compare_initialization(seed, cell, ntrain, ntilde):
    """TEST 0: Compare initial state"""

def compare_single_estep(seed, cell, ntrain, ntilde):
    """TEST 1: Compare after single E-step"""

def compare_single_fstep(seed, cell, ntrain, ntilde):
    """TEST 2: Compare after single F-step"""

def compare_mstep_gradients(seed, cell, ntrain, ntilde):
    """TEST 3: Compare M-step gradients"""

def compare_full_iteration(seed, cell, ntrain, ntilde, n_iter):
    """TEST 5-6: Compare full iterations"""

def diagnose_failing_cell(cell):
    """TEST 7: Detailed diagnosis of failing cell"""
```

---

## 7. Summary of Required Verification

| Component | Priority | Status |
|-----------|----------|--------|
| C matrix computation | HIGH | Need TEST 0 |
| Mask computation | HIGH | Need TEST 0 |
| Kernel K computation | HIGH | Need TEST 0 |
| Eigenspace projection | HIGH | Need TEST 0 |
| E-step g, G | HIGH | Need TEST 1 |
| E-step V, m update | HIGH | Need TEST 1 |
| Lambda moments | MEDIUM | Need TEST 1 |
| F-step lambda0 | MEDIUM | Need TEST 2 |
| F-step A LBFGS | MEDIUM | Need TEST 2 |
| M-step dC gradients | HIGH | Need TEST 3 |
| M-step dK gradients | HIGH | Need TEST 3 |
| M-step dlambda gradients | HIGH | Need TEST 3 |
| M-step dKL gradients | HIGH | Need TEST 3 |
| M-step softplus chain rule | HIGH | Need TEST 3 |
| M-step LBFGS convergence | MEDIUM | Need TEST 4 |
| Eigenspace reprojection | HIGH | Need TEST 6 |
| Cell-specific failures | HIGH | Need TEST 7 |

---

## 8. Resolution (January 2025)

### Root Cause Confirmed: BUG #6

The M-step closure used **stale eigenvalues** (`state.eigvals_b` from E-step) while computing fresh `K_tilde_b` from updated hyperparameters. This caused:
- Incorrect `K_tilde_inv_b_diag = 1.0 / eigvals_b` (stale)
- Incorrect `log_det_K = log(eigvals_b).sum()` (stale)
- Mathematical inconsistency: new K_tilde_b matrix with old inverse/determinant

### Initial Fix Attempt (FAILED)

I first tried recomputing eigenvalues from the fresh K_tilde_b:

```python
# FAILED APPROACH - causes numerical issues on float32
eigvals_b_fresh = torch.linalg.eigvalsh(K_tilde_b)
eigvals_b_fresh = torch.clamp(eigvals_b_fresh, min=1e-6)
K_tilde_inv_b_diag = 1.0 / eigvals_b_fresh
log_det_K = torch.log(eigvals_b_fresh).sum()
```

**Why this failed**: `torch.linalg.eigvalsh()` on float32 throws `LinAlgError: The algorithm failed to converge because the input matrix is ill-conditioned` for some cells (cell 6).

### Correct Fix: Use solve() Like vargp_old

Looking at the original vargp_old code (utils.py lines 5920-5921), I discovered it uses `torch.linalg.solve()` for the inverse, NOT eigendecomposition:

```python
# ORIGINAL vargp_old (utils.py lines 5920-5921)
eye = torch.eye(K_tilde_b.shape[0], device=DEVICE, dtype=TORCH_DTYPE)
K_tilde_inv_b = torch.linalg.solve(K_tilde_b, eye)
```

**Key insight**: In the M-step closure, `K_tilde_b = B.T @ K_tilde @ B` is NOT diagonal (only K_tilde_b during E-step initialization is diagonal). The original code uses full matrix operations.

### Final Fix Applied

**File**: `direct_vargp.py:mstep_lbfgs_analytical()`

1. **Compute K_tilde_inv_b via solve()** (line 1107):
```python
eye_b = torch.eye(n_b, device=K_tilde_b.device, dtype=K_tilde_b.dtype)
K_tilde_inv_b = torch.linalg.solve(K_tilde_b, eye_b)
```

2. **Use full matrix for KL computation** (lines 1130-1148):
```python
trace_term = torch.trace(K_tilde_inv_b @ V_b)
quad_term = m_b @ K_tilde_inv_b @ m_b
# log_det via Cholesky (matching vargp_old's log_det function)
L_K = torch.linalg.cholesky(K_tilde_b)
log_det_K = 2 * torch.log(torch.diag(L_K)).sum()
```

3. **Updated helper functions** to accept full matrix K_tilde_inv_b:
   - `compute_lambda_moments_and_gradients()` - now handles both diagonal (1D) and full (2D) K_tilde_inv_b
   - `compute_loss_gradients()` - same

### Validation Results

| Cell | vargp_old | vargp_direct (fixed) | Status |
|------|-----------|---------------------|--------|
| 6 | 0.5942 | 0.5869 | **FIXED** (was ~0) |
| 8 | 0.8413 | 0.8429 | Match |
| 15 | -0.1014 | -0.1013 | Match (both fail on this cell) |

### Lessons Learned

1. **Don't assume diagonal structure persists**: K_tilde_b is diagonal only when first computed via eigendecomposition. After projection `B.T @ K_tilde @ B` with changed K_tilde, it's a full symmetric matrix.

2. **Follow the original code exactly**: The original uses `solve()` not eigendecomposition for K_tilde_inv in M-step. This is more numerically stable.

3. **float32 eigendecomposition is fragile**: `eigvalsh()` can fail on ill-conditioned matrices. `solve()` and `cholesky()` are more robust.

---

## 9. Remaining Uninvestigated Items

### Investigated but NOT numerically verified:

1. **BUG #4 (Softplus chain rule)** - Explore agent concluded the math is correct (`dL['sigma_0'] * sigmoid(raw)`), but no numerical gradient check was run to verify this matches autograd.

2. **BUG #3 (Eigenspace projection of dK_tilde)** - Claimed mathematically correct, but no numerical comparison of actual gradient values between implementations.

### Not investigated at all:

3. **Gradient magnitude sanity check** - Are the analytical gradients reasonable in scale? Could help catch subtle bugs.

4. **More cells** - Only tested 3 cells (6, 8, 15). Original benchmark had 10 cells (1, 5, 6, 7, 8, 9, 15, 16, 18, 28).

5. **Different M values** - Only tested M=50. Original benchmark included M=250.

6. **ntrain=2000** - Only tested ntrain=500.

### Unexplained observation:

7. **vargp_direct is faster than vargp_old** (5.0s vs 6.4s on cell 8) despite adding `solve()` which is O(n_b³). Possible explanations:
   - vargp_old has extra overhead (logging, checks, etc.)
   - Different code paths or redundant computations in original
   - Measurement variance

   This warrants investigation if performance parity is important.

---

*Created: January 2025*
*Purpose: Investigation context for new Claude Code session*
*Resolution: January 2025 - Fixed by using solve() instead of eigendecomposition*
