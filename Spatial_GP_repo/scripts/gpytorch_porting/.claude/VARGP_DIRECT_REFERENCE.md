# vargp_direct Implementation Reference

**Purpose**: Comprehensive guide to the `vargp_direct` training mode - a GPyTorch-based implementation that matches the original `varGP()` algorithm structure.

**Status**: ACTIVE - Performance matches vargp_old; known loss offset remains (see Section 6.8)
**Last Updated**: February 2025 (API cleanup: model(X_train) for posterior, shared _compute_eigenspace_quantities)

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Architecture Overview](#2-architecture-overview)
3. [File Structure](#3-file-structure)
4. [The Training Loop](#4-the-training-loop)
5. [Component Details](#5-component-details)
6. [Important Caveats](#6-important-caveats)
7. [GPyTorch Wrapper Classes](#7-gpytorch-wrapper-classes)
8. [Resolved Bugs](#8-resolved-bugs)
9. [Deferred Items](#9-deferred-items)
10. [Unit Tests](#10-unit-tests)
11. [Math Reference](#11-math-reference)
12. [Code-to-Math Mapping](#12-code-to-math-mapping)

---

## 1. Quick Start

### Run vargp_direct

```bash
# Standard usage (autograd M-step)
python run_single_mode.py --mode vargp_direct --float32 \
    --ntilde 50 --n-iterations 50 --n-estep 10 --n-fstep 10 --n-mstep 10 --seed 123

# Alternative: analytical M-step (slightly different optimization path)
python run_single_mode.py --mode vargp_direct --mstep-analytical --float32 \
    --ntilde 50 --n-iterations 50 --seed 123
```

### Performance Comparison

| Mode | Dtype | Total | Test r | Status |
|------|-------|-------|--------|--------|
| vargp_old | float32 | 6.3s | 0.84 | Reference |
| vargp_direct (autograd) | float32 | 5.6s | 0.84 | Matches reference |
| vargp_direct (analytical) | float32 | 5.2s | 0.85 | Matches reference |

**Note**: Loss values differ by ~0.5*n_b due to KL constant term difference (see Section 6.8). This does NOT affect optimization or predictions.

**Key point**: Use `--float32` to match vargp_old speed. float64 is ~3x slower.

---

## 2. Architecture Overview

### Why vargp_direct Exists

There are three training modes in this codebase:

| Mode | Description | Limitation |
|------|-------------|------------|
| `vargp_old` | Original implementation in utils.py | Complex codebase, hard to modify |
| `default_gpy` | Standard GPyTorch variational inference | Different optimization dynamics |
| `vargp_style` | Hybrid (custom E-step + GPyTorch strategy) | No eigenspace projection, Adam M-step |

`vargp_direct` fills the gap: it uses GPyTorch as a **kernel calculator only**, while managing variational parameters directly with eigenspace projection - exactly like the original.

### Key Design Decisions

1. **Eigenspace projection**: Reduces dimensionality from M inducing points to ~10-50 dimensions
2. **K_tilde_b is DIAGONAL**: In eigenspace, the inducing kernel becomes `diag(eigenvalues)` - trivial inverse!
3. **LBFGS for M-step**: Matches original optimization dynamics (not Adam)
4. **GPyTorch for kernels only**: We bypass `VariationalStrategy` entirely

---

## 3. File Structure

The vargp_direct implementation is organized into a **modular structure** with clear separation of concerns:

```
gpytorch_porting/
|
|-- eigenspace_model.py      # STATE & MODEL CLASSES (~500 lines)
|   |-- DirectVariationalState       # Dataclass for eigenspace state
|   |-- _compute_eigenspace_quantities()  # Core shared computation (kernel → eigenspace)
|   |-- _compute_initial_eigenspace()     # Model initialization (m_b=0, V_b=K_tilde_b)
|   |-- _recompute_eigenspace()           # After M-step (reprojects m_b, V_b)
|   |-- DirectVGPModel               # Main model class (owns kernel, likelihood, state)
|   |   |-- .X_train                 # Training data
|   |   |-- .X_tilde                 # Inducing points
|   |   |-- .state                   # DirectVariationalState
|   |   |-- .update_variational_params()  # Update m_b, V_b after E-step
|   |   +-- .recompute_eigenspace()  # Sync eigenspace after M-step
|   |-- EigenspacePosterior          # Posterior at query points (via model(X))
|   +-- EigenspaceVariationalDistribution  # Variational params interface
|
|-- eigenspace.py            # EIGENSPACE UTILITIES (~200 lines, mainly for tests)
|   |-- EIGVAL_TOL = 1e-4            # Eigenvalue threshold
|   |-- eigendecompose_K_tilde()     # Eigendecomposition of inducing kernel
|   |-- project_to_eigenspace()      # m_b = B.T @ m, etc.
|   |-- reproject_variational_params()   # After M-step changes B
|   |-- compute_KKtilde_inv_b()      # K @ K_tilde_inv (for tests)
|   +-- compute_K_tilde_b_diagonal() # diag(eigenvalues) (for tests)
|
|-- train.py                 # TRAINING LOOP (~785 lines total)
|   |-- train_eigenspace()           # Main training loop for vargp_direct
|   |-- predict_eigenspace()         # Prediction at test points
|   +-- compute_elbo_eigenspace()    # ELBO computation
|
|-- estep.py                 # E-STEP (~893 lines total)
|   +-- estep_eigenspace()           # Newton update for (m_b, V_b)
|
|-- fstep.py                 # F-STEP (~361 lines total)
|   |-- compute_f_mean()             # Expected firing rate
|   +-- fstep_eigenspace()           # LBFGS for A, analytical lambda0
|
|-- mstep.py                 # M-STEP (~395 lines total)
|   |-- mstep_eigenspace_autograd()  # LBFGS with PyTorch autograd
|   +-- mstep_eigenspace_analytical()# LBFGS with explicit gradients
|
|-- direct_vargp.py          # GRADIENT FUNCTIONS ONLY (~440 lines)
|   |-- compute_C_and_gradients()    # C matrix and dC/dtheta
|   |-- compute_kernel_and_gradients()   # K matrix and dK/dtheta
|   |-- compute_lambda_moments_and_gradients()  # Posterior moments + gradients
|   +-- compute_loss_gradients()     # dL/dtheta for M-step
|
|-- kernels.py               # ArcCosineKernel (used as calculator)
|-- likelihoods.py           # PoissonLikelihood (A, lambda0 parameters)
|
+-- tests/
    |-- test_vargp_direct_match.py   # Unit tests for gradient functions
    |-- test_direct_vgp_model.py     # Unit tests for wrapper classes
    +-- test_mstep_analytical.py     # M-step gradient tests
```

### Key Points About File Organization

1. **eigenspace_model.py** is the central module - it contains both the state management (`DirectVariationalState`) and the GPyTorch wrapper classes (`DirectVGPModel`)

2. **direct_vargp.py** now ONLY contains gradient functions for the analytical M-step. All other code has been moved to appropriate modules.

3. **Naming convention**: Functions use `_eigenspace` suffix (e.g., `estep_eigenspace`, `fstep_eigenspace`) to distinguish from other training modes

4. **DELETED FILE**: `direct_vargp_wrapper.py` no longer exists - its contents were merged into `eigenspace_model.py`

---

## 4. The Training Loop

### High-Level Structure

The training loop is implemented in `train.py:train_eigenspace()`:

```python
def train_eigenspace(model, r, ...):
    # model is a DirectVGPModel which owns:
    # - kernel, likelihood, X_train, X_tilde
    # - state (initialized with eigenspace projection on construction)

    state = model.state  # Already initialized with m_b=0, V_b=K_tilde_b

    # MAIN LOOP
    # NOTE: Uses range(1, n_iterations) to match vargp_old behavior
    # This means n_iterations=50 gives 49 actual iterations (1-49)
    for iteration in range(1, n_iterations):

        # EIGENSPACE RECOMPUTATION (after iteration 1, when kernel params changed)
        if n_mstep > 0 and iteration > 1:
            model.recompute_eigenspace()  # Sync eigenspace with current kernel params

        # E-STEP: Newton updates for m_b, V_b
        for _ in range(n_estep):
            posterior = model(model.X_train)  # GPyTorch-like: call model to get posterior
            lambda_m, lambda_var = posterior.mean, posterior.variance
            f_mean = compute_f_mean(lambda_m, lambda_var, A, lambda0)
            m_b, V_b = estep_eigenspace(state, r, A, f_mean)

        # F-STEP: Optimize A, compute lambda0 analytically
        fstep_eigenspace(likelihood, r, lambda_m, lambda_var, n_fstep, lr_f)

        # M-STEP: Optimize kernel hyperparameters
        # NOTE: Skips when iteration >= n_iterations - 1 (matches vargp_old)
        if n_mstep > 0 and iteration < n_iterations - 1:
            if use_analytical_mstep:
                mstep_eigenspace_analytical(kernel, ...)
            else:
                mstep_eigenspace_autograd(kernel, ...)

    return {'losses': losses, 'state': state, ...}
```

### CRITICAL: Iteration Count Behavior

**The `n_iterations` parameter results in `n_iterations - 1` actual iterations.**

This matches vargp_old behavior exactly:

| Parameter | vargp_old | vargp_direct (train_eigenspace) |
|-----------|-----------|--------------------------------|
| Loop | `range(1, maxiter)` | `range(1, n_iterations)` |
| With n=50 | iterations 1-49 (49 total) | iterations 1-49 (49 total) |
| M-step skip | `iteration < maxiter - 1` | `iteration < n_iterations - 1` |
| M-step runs | iterations 1-48 | iterations 1-48 |

**Rationale**: The last M-step is skipped because it would create a new eigenspace that won't be used by the final variational parameters (m, V). This matches the original implementation.

**FUTURE CONSIDERATION**: May want to change to `range(1, n_iterations + 1)` for one additional iteration if testing shows improved convergence.

### Detailed Step Explanations

#### Initialization

```python
# In DirectVGPModel.__init__() (via _compute_initial_eigenspace):
K_tilde = kernel(X_tilde, X_tilde).evaluate()  # (M, M)
K = kernel(X_train, X_tilde).evaluate()        # (N, M)
Kvec = kernel(X_train, diag=True)              # (N,)

# Eigendecomposition
B, eigvals_b, _ = eigendecompose_K_tilde(K_tilde)  # B: (M, n_b)
# n_b is typically 10-50, much smaller than M

# Project to eigenspace
K_tilde_b = diag(eigvals_b)      # DIAGONAL!
K_b = K @ B                       # (N, n_b)
KKtilde_inv_b = K_b / eigvals_b  # Element-wise (trivial inverse!)

# Initialize variational params
m_b = zeros(n_b)                 # Start with prior mean
V_b = K_tilde_b.clone()          # Start with prior covariance
```

#### E-step Details

The E-step is implemented in `estep.py:estep_eigenspace()`:

```python
def estep_eigenspace(state, r, A, f_mean):
    a = state.KKtilde_inv_b  # (N, n_b), precomputed K @ K_tilde_inv

    # Gradient and Hessian of log-likelihood w.r.t. natural params
    g_b = A * (a.T @ (r - f_mean))           # (n_b,)
    G_b = A**2 * (a.T @ (f_mean[:, None] * a)) # (n_b, n_b)

    # Newton update for V
    V_b_new = solve(I + K_tilde_b @ G_b, K_tilde_b)

    # Newton update for m (using OLD formula - see Section 9)
    m_b_new = V_b_new @ (G_b @ m_b + g_b)

    # Symmetrize for numerical stability
    V_b_new = (V_b_new + V_b_new.T) / 2

    return m_b_new, V_b_new
```

#### F-step Details

The F-step is implemented in `fstep.py:fstep_eigenspace()`:

```python
# lambda0 is computed analytically given A
lambda0 = log(sum(r) / sum(exp(A*lambda_m + 0.5*A**2*lambda_var)))

# A is optimized with LBFGS
def f_closure():
    log_lik = r @ (A*lambda_m + lambda0) - f_mean.sum()
    return -log_lik  # Minimize negative log-likelihood

LBFGS([logA], closure=f_closure, max_iter=n_fstep)
```

#### M-step Details

The M-step has two implementations in `mstep.py`:

**With autograd** (`mstep_eigenspace_autograd`) - **BUGGY, use analytical instead**:
```python
def closure():
    # Recompute kernels with current hyperparams
    K_tilde = kernel(X_tilde, X_tilde).evaluate()
    K = kernel(X, X_tilde).evaluate()
    # BUG: Uses diagonal approximation for KL trace (see Section 8)
    # ... compute loss using fixed eigenspace B ...
    grads = torch.autograd.grad(loss, kernel_params)
    return loss

LBFGS(kernel.parameters(), closure=closure)
```

**WARNING**: `mstep_eigenspace_autograd` has a known bug in the KL trace computation that degrades test_r by 2-5%. Use `--mstep-analytical` for correct results.

**With analytical gradients** (`mstep_eigenspace_analytical`) - **CORRECT**:
```python
def closure():
    # Compute kernels AND all dK/dtheta matrices
    C, mask, dC = compute_C_and_gradients(kernel)
    K_tilde, dK_tilde = compute_kernel_and_gradients(X_tilde, X_tilde, C, dC, ...)
    K, dK = compute_kernel_and_gradients(X, X_tilde, C, dC, ...)

    # CORRECT: Uses full matrix inverse for KL trace
    K_tilde_inv_b = torch.linalg.solve(K_tilde_b, eye_b)
    trace_term = torch.trace(K_tilde_inv_b @ V_b)

    # Compute loss and gradients explicitly
    dL = compute_loss_gradients(r, f_mean, A, m_b, V_b, ...)

    # Set parameter gradients manually with chain rule
    kernel.raw_sigma_0.grad = dL['sigma_0'] * sigmoid(raw_sigma_0)
    # ... etc ...

    return loss
```

---

## 5. Component Details

### 5.1 DirectVariationalState Dataclass

Defined in `eigenspace_model.py`, this dataclass holds all eigenspace quantities:

```python
@dataclass
class DirectVariationalState:
    m_b: torch.Tensor        # Variational mean (n_b,)
    V_b: torch.Tensor        # Variational covariance (n_b, n_b) - NOT diagonal!
    B: torch.Tensor          # Eigenvector matrix (M, n_b)
    eigvals_b: torch.Tensor  # Eigenvalues (n_b,)
    K_tilde_b: torch.Tensor  # Inducing kernel in eigenspace (n_b, n_b) - diagonal
    K_b: torch.Tensor        # Cross-kernel in eigenspace (N, n_b)
    KKtilde_inv_b: torch.Tensor  # K @ K_tilde_inv = K_b / eigvals_b (N, n_b)
    Kvec: torch.Tensor       # Self-kernel diagonal (N,)
    mask: torch.Tensor       # Pixel mask for RF structure (n_pixels,)
```

### 5.2 Eigenspace Projection

**Why eigenspace?** The inducing point kernel K_tilde (M x M) has effective rank ~10-50. Eigenspace projection:
1. Reduces computation from O(M^3) to O(n_b^3)
2. Makes K_tilde inverse trivial: `diag(1/eigenvalues)`
3. Provides implicit regularization (drops small eigenvalues)

**Key quantities in eigenspace:**

| Symbol | Name | Shape | Formula |
|--------|------|-------|---------|
| B | Eigenvector matrix | (M, n_b) | From `eigh(K_tilde)` |
| K_tilde_b | Inducing kernel | (n_b, n_b) | `diag(eigenvalues)` - DIAGONAL |
| K_b | Cross-kernel | (N, n_b) | `K @ B` |
| m_b | Variational mean | (n_b,) | `B.T @ m` |
| V_b | Variational covariance | (n_b, n_b) | `B.T @ V @ B` - NOT diagonal |
| a | Projection vector | (N, n_b) | `K_b / eigvals_b` (element-wise) |

**Critical insight**: V_b is NOT diagonal even though K_tilde_b is diagonal. Don't assume diagonal V_b!

### 5.3 Posterior Moments (lambda_m, lambda_var)

Access via GPyTorch-like pattern: `posterior = model(model.X_train)`, then `posterior.mean`, `posterior.variance`.
Internally implemented in `eigenspace_model.py:_lambda_moments_eigenspace()` (private):

```python
# In eigenspace:
a = K_b / eigvals_b              # K @ K_tilde_inv, element-wise division
lambda_m = a @ m_b               # Posterior mean
lambda_var = Kvec + (a @ (V_b - K_tilde_b) @ a.T).diag()  # Posterior variance
```

### 5.4 C Matrix (Receptive Field Structure)

The C matrix encodes receptive field (RF) properties:

```
C = Amp * alpha[:, None] * C_smooth * alpha[None, :]

where:
  alpha[i] = exp(-beta_factor * ||pixel_i - center||^2)    # Locality weight
  C_smooth[i,j] = exp(-rho_factor * ||pixel_i - pixel_j||^2)  # Smoothness
  beta_factor = exp(raw_m2log2beta) = 1/(4*beta^2)
  rho_factor = exp(raw_mlog2rho2) = 1/(2*rho^2)
```

### 5.5 Parameter Transforms

| Parameter | Raw Name | Transform | Gradient Chain Rule |
|-----------|----------|-----------|-------------------|
| sigma_0 | `raw_sigma_0` | `softplus(raw)` | `dL/d(raw) = dL/d(sigma_0) * sigmoid(raw)` |
| Amp | `raw_Amp` | `softplus(raw)` | Same as above |
| beta | `raw_m2log2beta` | `beta = exp(-raw/2)/2` | Direct (no transform in gradient) |
| rho | `raw_mlog2rho2` | `rho = sqrt(exp(-raw)/2)` | Direct |
| eps_0x, eps_0y | `eps_0x`, `eps_0y` | Direct | Direct |

---

## 6. Important Caveats

### 6.1 K_tilde_b is Only Diagonal at Specific Points

**CRITICAL**: K_tilde_b = diag(eigenvalues) is diagonal ONLY:
1. Immediately after eigendecomposition (model initialization)
2. After `model.recompute_eigenspace()` (which does fresh eigendecomposition)
3. During E-step and F-step (eigenspace is fixed)

**K_tilde_b is NOT diagonal**:
- Inside M-step LBFGS closure when hyperparameters have been updated

```python
# INSIDE M-step closure (DANGER ZONE)
K_tilde_new = kernel(X_tilde, X_tilde)  # New kernel with updated hyperparams
K_tilde_b = B.T @ K_tilde_new @ B        # NOT diagonal! B is from OLD K_tilde!
```

**Safe vs Unsafe Code Patterns**:

| Location | K_tilde_b diagonal? | Safe to use eigvals? |
|----------|--------------------|--------------------|
| After model initialization | YES | YES |
| After `model.recompute_eigenspace()` | YES | YES |
| Inside E-step | YES (fixed) | YES |
| Inside F-step | YES (fixed) | YES |
| Inside M-step closure (first call) | YES | YES |
| Inside M-step closure (after hyperparam update) | **NO** | **NO** |

**The mstep_eigenspace_autograd bug** (Section 8) occurs because it assumes diagonal K_tilde_b inside the M-step closure where hyperparameters are changing.

**The fix** (`mstep_eigenspace_analytical`): Uses `torch.linalg.solve()` instead of diagonal assumption.

### 6.2 Eigenspace Changes After M-step

After M-step changes kernel hyperparameters, the eigenspace changes. You MUST call `model.recompute_eigenspace()` to sync the state:

```python
# After M-step - call recompute_eigenspace() which:
# 1. Recomputes K_tilde, K, Kvec with current kernel params
# 2. Does fresh eigendecomposition: B_new, eigvals_new
# 3. Reprojects m_b, V_b to new eigenspace
model.recompute_eigenspace()
```

### 6.3 LBFGS Closure Called Multiple Times

LBFGS with `line_search_fn='strong_wolfe'` calls the closure multiple times per step. Don't use `loss.backward()` inside - it frees the graph. Instead, compute gradients manually or use `torch.autograd.grad()`.

### 6.4 Symmetrize V After Updates

Always symmetrize V after Newton updates for numerical stability:
```python
V_b = (V_b + V_b.T) / 2
```

### 6.5 Clamp Hyperparameters After M-step

Call `kernel.clamp_hyperparameters()` after M-step to ensure parameters stay in valid ranges.

### 6.6 Float32 vs Float64

- **Float32**: Matches vargp_old, fast, recommended for production
- **Float64**: More precise but ~3x slower, useful for debugging gradient issues

### 6.7 Iteration Count (n_iterations vs Actual Iterations)

**IMPORTANT**: The `n_iterations` parameter results in `n_iterations - 1` actual iterations.

This matches vargp_old behavior:

| Implementation | Loop | For n=50 | M-step condition | M-step runs |
|----------------|------|----------|------------------|-------------|
| vargp_old | `range(1, maxiter)` | iters 1-49 | `iter < maxiter-1` | 1-48 |
| train_eigenspace | `range(1, n_iterations)` | iters 1-49 | `iter < n_iterations-1` | 1-48 |

**Rationale**: The last M-step is skipped to avoid generating a new eigenspace that won't be used.

**FUTURE**: May want to change to `range(1, n_iterations+1)` for one more iteration.

### 6.8 KL Divergence Formula Discrepancy (CONFIRMED)

When comparing final loss values between vargp_direct and vargp_old, there is a consistent offset of approximately `0.5 * n_b` (around 20-25 units for typical n_b values).

**Confirmed cause**: The KL divergence formula differs by a constant term:
- vargp_direct uses the standard formula: `KL = 0.5 * (tr + quad - n_b + log|K| - log|V|)`
- vargp_old omits the `-n_b` term (utils.py:4141): `KL = 0.5 * (tr + quad + log|K| - log|V|)`

**Verification**: Temporarily removing `-n_b` from vargp_direct reduced the loss difference from ~23 to ~1.6 for Cell 8, M=50.

**Important notes**:
- This does NOT affect optimization (constant terms have zero gradient)
- This does NOT affect predictions (test_r values are nearly identical)
- This only affects the absolute loss value reported

**Remaining discrepancy**: Even after accounting for the `-n_b` term, some configurations show larger residual differences (e.g., Cell 8, M=100: ~15 units). This correlates with dynamic eigenspace dimension changes during training (n_b varying from 76→66). **Further investigation needed** to understand eigenspace reprojection differences between implementations.

---

## 7. GPyTorch Wrapper Classes

The `eigenspace_model.py` module provides GPyTorch-compatible wrapper classes that allow vargp_direct to be used like a standard GPyTorch model.

### 7.1 DirectVGPModel

The main model class for vargp_direct training mode:

```python
class DirectVGPModel:
    """Eigenspace variational GP model for vargp_direct training mode."""

    def __init__(self, kernel, likelihood, X_train, X_tilde, eigval_tol=1e-4):
        self.kernel = kernel
        self.likelihood = likelihood
        self.X_train = X_train      # Training data (stored for eigenspace ops)
        self.X_tilde = X_tilde      # Inducing points
        self._state = ...           # Initialized with eigenspace projection

    def __call__(self, X_query) -> EigenspacePosterior:
        """Returns posterior at query points."""
        ...

    def update_variational_params(self, m_b, V_b):
        """Update m_b, V_b after E-step."""
        ...

    def recompute_eigenspace(self):
        """Sync eigenspace with current kernel hyperparameters. Call after M-step."""
        ...

    @property
    def state(self) -> DirectVariationalState:
        """Access eigenspace state (read-only preferred)."""
        ...

    @property
    def variational_distribution(self) -> EigenspaceVariationalDistribution:
        """Returns variational distribution interface."""
        ...
```

### 7.2 EigenspacePosterior

Represents the GP posterior at query points:

```python
class EigenspacePosterior:
    """Posterior distribution at query points."""

    @property
    def mean(self) -> torch.Tensor:
        """Posterior mean lambda_m at query points."""

    @property
    def variance(self) -> torch.Tensor:
        """Posterior variance lambda_var at query points."""

    def expected_firing_rate(self) -> torch.Tensor:
        """E[f] = exp(A*lambda_m + 0.5*A^2*lambda_var + lambda0)"""
```

### 7.3 EigenspaceVariationalDistribution

Provides access to variational parameters:

```python
class EigenspaceVariationalDistribution:
    """Interface to variational parameters in eigenspace."""

    @property
    def mean_eigenspace(self) -> torch.Tensor:
        """m_b in eigenspace (n_b,)"""
        return self.state.m_b

    @property
    def covariance_eigenspace(self) -> torch.Tensor:
        """V_b in eigenspace (n_b, n_b)"""
        return self.state.V_b

    @property
    def mean(self) -> torch.Tensor:
        """m in full space: B @ m_b (M,)"""
        return self.state.B @ self.state.m_b

    @property
    def covariance(self) -> torch.Tensor:
        """V in full space: B @ V_b @ B.T (M, M)"""
        return self.state.B @ self.state.V_b @ self.state.B.T
```

### 7.4 Usage Example

```python
from eigenspace_model import DirectVGPModel
from kernels import ArcCosineKernel
from likelihoods import PoissonLikelihood
from train import train_eigenspace, predict_eigenspace

# Create model (owns kernel, likelihood, training data, and state)
kernel = ArcCosineKernel(...)
likelihood = PoissonLikelihood(A_init=0.01, lambda0_init=1.0)
model = DirectVGPModel(kernel, likelihood, X_train, X_tilde)

# Train using eigenspace mode
result = train_eigenspace(model, r_train, n_iterations=50, ...)

# Make predictions at test points
predictions = predict_eigenspace(model, X_test)
f_pred = predictions['f_pred']

# Or use model directly for posterior
posterior = model(X_test)
lambda_m = posterior.mean
lambda_var = posterior.variance
```

---

## 8. Resolved Bugs

### RESOLVED: mstep_eigenspace_autograd Diagonal Approximation (FIXED January 2025)

**Status**: FIXED

**Location**: `mstep.py:mstep_eigenspace_autograd()` lines 167-170

**Original Problem**: Inside the LBFGS closure, the KL trace term used a diagonal approximation that was INCORRECT when kernel hyperparameters changed:

```python
# OLD BUGGY CODE
V_diag = torch.diag(state.V_b)
K_tilde_b_diag = torch.diag(K_tilde_b)  # WRONG - K_tilde_b NOT diagonal after hyperparam changes
trace_term = (V_diag / K_tilde_b_diag.clamp(min=1e-10)).sum()
```

**Why it was wrong**:
1. At initialization, `K_tilde_b = diag(eigenvalues)` is diagonal
2. Inside M-step closure, hyperparameters change → `K_tilde_new ≠ K_tilde_init`
3. Fresh `K_tilde_b = B.T @ K_tilde_new @ B` is NOT diagonal (B is eigenvectors of OLD K_tilde)
4. Taking only diagonal elements gave WRONG KL trace → WRONG gradients → suboptimal optimization

**Fix applied**: Use the already-computed full matrix inverse:

```python
# FIXED CODE (mstep.py lines 167-170)
# K_tilde_b_inv was already computed via solve() at line 139-143
trace_term = torch.trace(K_tilde_b_inv @ state.V_b)
```

**Verification** (January 2025):
| Mode | test_r (M=50, Cell 8, 5 iters) | Diff vs vargp_old |
|------|-------------------------------|-------------------|
| vargp_old | 0.7951 | reference |
| vargp_direct (autograd) BEFORE fix | 0.7726 | -0.0225 (BUGGY) |
| vargp_direct (autograd) AFTER fix | 0.7946 | -0.0005 (FIXED) |
| vargp_direct (analytical) | 0.7959 | +0.0008 |

---

### RESOLVED: Stale Eigenvalues in Analytical M-step (FIXED January 2025)

**Status**: FIXED

**Original Problem**: Used stale `state.eigvals_b` while computing fresh `K_tilde_b`:

```python
# OLD BUGGY CODE
K_tilde_b = B.T @ K_tilde_new @ B     # Fresh K_tilde_b
K_tilde_inv_b = diag(1/eigvals_b)     # STALE eigenvalues!
```

**Fix**: Use `torch.linalg.solve()` for the inverse:

```python
# FIXED CODE (in mstep.py:mstep_eigenspace_analytical lines 316-317)
K_tilde_b = B.T @ K_tilde_new @ B
eye_b = torch.eye(n_b, ...)
K_tilde_inv_b = torch.linalg.solve(K_tilde_b, eye_b)  # Works for any SPD matrix
```

---

### Verified Non-Bugs

Unit tests confirmed these are NOT bugs:

- **BUG #4 (Softplus chain rule)**: `dL['sigma_0'] * sigmoid(raw)` is correct
- **BUG #3 (Eigenspace projection)**: `dK_tilde_b = B.T @ dK_tilde @ B` preserves gradient structure
- **compute_elbo_eigenspace trace formula**: Correctly uses diagonal V_b elements when K_tilde_b is guaranteed diagonal (called with fixed eigenspace after E-step, not during M-step)
- **compute_loss_gradients**: Correctly branches based on `is_diagonal` flag

---

## 9. Deferred Items

### 9.1 Correct E-step m_new Formula (NOT IMPLEMENTED)

**Background**: The original varGP uses a mathematically **incorrect** m_new formula that works empirically.

**Current implementation** (OLD formula, matching vargp_old):
```python
m_new = V_new @ (G @ m + g)
```

**Mathematically correct formula** (NOT implemented):
```python
m_new = m + K_tilde @ solve(K_tilde + G, g - m)
```

**Why deferred**: When we tried using the correct formula, the model collapsed. The issue is that `g_b` and `G_b` are **transformed** quantities (pre-multiplied by K_tilde_inv), and the old formulas work with these transformed quantities.

**See**: `.claude/ESTEP_MATH_ANALYSIS.md` for full derivation

### 9.2 Gradient Function Consolidation (DEFERRED)

There are TWO gradient systems in the codebase:
1. **Kernel-level gradients** (`analytical_gradients.py`, `analytical_gradients_vjp.py`) - for `--gradient-mode`
2. **M-step gradients** (`direct_vargp.py`) - for `mstep_eigenspace_analytical()`

These have different APIs but some shared formulas. Consolidation is deferred to avoid breaking anything.

### 9.3 Extended Testing

- Only tested M=50. Original benchmark included M=250.
- Only tested ntrain=500. Could test ntrain=2000.
- Only tested 3 cells (6, 8, 15). Could test all 10 benchmark cells.

---

## 10. Unit Tests

### Test Files

| File | Purpose |
|------|---------|
| `tests/test_vargp_direct_match.py` | Unit tests for gradient functions |
| `tests/test_direct_vgp_model.py` | Unit tests for wrapper classes |
| `tests/test_mstep_analytical.py` | M-step gradient validation |

### Key Tests

| Test | Purpose | Acceptance |
|------|---------|------------|
| `test_softplus_chain_rule` | Verify gradient transform for sigma_0, Amp | rel_err < 1e-4 vs finite diff |
| `test_eigenspace_projection_gradients` | Verify dK_tilde_b preserves gradients | rel_err < 1e-4 for all params |
| `test_ktilde_inv_methods` | Verify solve() vs eigenvalue inverse | Match at init, non-diagonal after change |
| `test_expected_firing_rate` | Verify f_mean formula in wrapper | Exact match |
| `test_prediction_test_points` | Verify wrapper matches predict_eigenspace | Exact match |

### Running Tests

```bash
# All vargp_direct tests
python tests/test_vargp_direct_match.py
python tests/test_direct_vgp_model.py

# Skip slow tests
python tests/test_vargp_direct_match.py --skip-slow

# Individual test
python tests/test_vargp_direct_match.py --test 1 --verbose
```

### Test Results (January 2025)

All tests PASS:
- Softplus chain rule: rel_err < 1e-7
- Eigenspace projection: rel_err < 1e-6 for all 6 hyperparameters
- K_tilde_inv methods: solve() works for non-diagonal K_tilde_b
- Wrapper classes: exact match with standalone functions

---

## 11. Math Reference

### LaTeX Source Documents

| File | Content |
|------|---------|
| `~/IDV_code/Papers/latex_summaries/Gaussian_process_theory.tex` | Full variational GP derivation |
| `~/IDV_code/Papers/latex_summaries/Estep_corrected.tex` | Correct E-step derivation |
| `~/IDV_code/Papers/latex_summaries/acosker_kernel_def_and_gradients.tex` | Arc-cosine kernel math |

### Local Context Documents

| File | Content |
|------|---------|
| `.claude/ESTEP_MATH_ANALYSIS.md` | Analysis of E-step formula discrepancy |
| `.claude/MATH_REFERENCE.md` | Quick math reference for this codebase |

---

## 12. Code-to-Math Mapping

### Variables

| Code | Math | Shape | Description |
|------|------|-------|-------------|
| `m_b` | m | (n_b,) | Variational mean in eigenspace |
| `V_b` | V | (n_b, n_b) | Variational covariance (NOT diagonal) |
| `K_tilde_b` | K_tilde | (n_b, n_b) | Inducing kernel (diagonal in eigenspace) |
| `K_b` | K | (N, n_b) | Cross-kernel to inducing points |
| `Kvec` | k(x,x) | (N,) | Self-kernel (diagonal) |
| `eigvals_b` | lambda_i | (n_b,) | Eigenvalues of K_tilde |
| `B` | B | (M, n_b) | Eigenvector matrix |
| `a` / `KKtilde_inv_b` | K K_tilde_inv | (N, n_b) | Projection vector |
| `lambda_m` | mu(x) | (N,) | Posterior mean |
| `lambda_var` | sigma^2(x) | (N,) | Posterior variance |
| `f_mean` | E[f] | (N,) | Expected firing rate |
| `A` | A | scalar | Gain parameter |
| `lambda0` | lambda_0 | scalar | Bias parameter |

### Key Formulas

**Posterior moments**:
```
mu(x) = k(x).T @ K_tilde_inv @ m
sigma^2(x) = k(x,x) + k(x).T @ K_tilde_inv @ (V - K_tilde) @ K_tilde_inv @ k(x)
```

**Expected log-likelihood**:
```
E[log p(r|lambda)] = r*(A*mu + lambda_0) - exp(A*mu + 0.5*A^2*sigma^2 + lambda_0)
```

**KL divergence**:
```
KL = 0.5 * [tr(K_tilde_inv @ V) + m.T @ K_tilde_inv @ m - n_b + log|K_tilde| - log|V|]
```

**E-step Newton updates**:
```
g = A * a.T @ (r - f)           # Gradient
G = A^2 * a.T @ diag(f) @ a     # Hessian
V_new = solve(I + K_tilde @ G, K_tilde)
m_new = V_new @ (G @ m + g)     # OLD formula (see Section 9.1)
```

---

## Appendix: Original Code References

| Component | utils.py Lines | Description |
|-----------|----------------|-------------|
| Main loop | 5293-5975 | `varGP()` function |
| C matrix | 3577-3631 | `localker()` |
| Kernel | 3663-3813 | `acosker()` |
| E-step | 4217-4277 | `Estep()` |
| lambda moments | 3906-3956 | `lambda_moments()` |
| Log-likelihood | 4064-4119 | `compute_loglikelihood()` |
| KL divergence | 4121-4152 | `compute_KL_div()` |
| Eigenspace | 5435-5444 | Eigendecomposition |
| Reprojection | 5619-5627 | After M-step |

---

*Created: January 2025*
*Last Updated: January 2025 (Reorganized to modular structure)*
*Purpose: Comprehensive reference for vargp_direct implementation*
