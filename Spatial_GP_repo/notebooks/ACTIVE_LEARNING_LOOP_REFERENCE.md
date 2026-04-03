# Active Learning Loop — Detailed Reference

**Purpose**: This document explains exactly how the active learning loop works in the
Spatial_GP_repo notebook codebase, so that another session can understand how it fits
into the full experimental pipeline.

**Source notebooks** (in `gaussian_processes/Spatial_GP_repo/notebooks/`):
- `one_cell_active_training.ipynb` — The **standard** active learning loop (Dec 12 2025). Uses `nd_utility()` (PNAS Eq. 27). 250 iterations.
- `one_cell_active_training_distribution_aware.ipynb` — The **latest** (Dec 16 2025). Compares two utility methods: `batch_utility_w_grad()` and `conditioned_utility_clean()`. 50 iterations.

**Key library files**:
- `utils.py` — Core GP functions: `varGP()`, `nd_utility()`, `lambda_moments()`, `generate_new_active_model()`, `test()`, `Estep()`, kernel functions
- `utility.py` — Advanced utility functions: `nd_utility_new()`, `batch_utility_w_grad()`, `conditioned_utility_clean()`, `optimize_with_conditioned_utility()`
- `GP_model/model.py` — `GPModel` class (wrapper around the model dictionary)

---

## Table of Contents

1. [Overview: What the Active Learning Loop Does](#1-overview)
2. [Data Setup](#2-data-setup)
3. [Initial Model Fitting](#3-initial-model-fitting)
4. [The Loop: Standard Version (nd_utility)](#4-the-loop-standard-version)
5. [The Loop: Distribution-Aware Version](#5-the-loop-distribution-aware-version)
6. [Utility Functions — All Three Variants](#6-utility-functions)
7. [Key Internal Functions](#7-key-internal-functions)
8. [Model State and Data Structures](#8-model-state-and-data-structures)
9. [Differences Between Old and New Notebooks](#9-differences-between-old-and-new-notebooks)
10. [How This Fits in the Experimental Pipeline](#10-how-this-fits-in-the-experimental-pipeline)

---

## 1. Overview

The active learning loop implements **information-theoretic sequential experiment design** for neural response modeling. The goal: given a GP model of a retinal ganglion cell's response to visual stimuli, choose the next stimulus image that maximally reduces uncertainty about the model.

**High-level algorithm:**

```
1. Start with a small random set of image-response pairs (e.g., 50 or 500)
2. Fit a full variational GP on this initial set
3. LOOP (50-250 times):
   a. For every remaining (unseen) image, compute an information-theoretic utility
   b. Select the image with the highest utility
   c. Add that image + its known response to the training set
   d. Update kernel matrices incrementally (efficient column append)
   e. Refit the GP on the expanded training set
   f. Evaluate performance on a held-out test set
4. Track performance (R^2, log-likelihood) over iterations
```

The key insight: images that the GP is most uncertain about (in a specific information-theoretic sense) are the most informative to add next. This is more efficient than random selection — the model converges faster with fewer images.

---

## 2. Data Setup

### Dataset

The PNAS retinal ganglion cell dataset. Loaded from `PNAS_paper_sorted_data.npz`:

```python
X_train: (2910, 108, 108, 1)   # training images
X_val:   (250, 108, 108, 1)    # validation images
X_test:  (30, 108, 108, 1)     # test images (30 images x 30 repetitions x 42 cells)
R_train: (2910, 41)            # spike counts, 41 cells
R_val:   (250, 41)
R_test:  (30, 30, 42)          # 30 repetitions per image per cell
```

### Preprocessing

```python
# Merge train + val into one pool
X = cat(X_train, X_val)  # shape (3160, 108, 108, 1)
R = cat(R_train, R_val)  # shape (3160, 41)

# Flatten images to 1D
X = reshape(X, (3160, 11664))  # 108*108 = 11664 pixels

# Select a single cell
R = R[:, cellid]  # shape (3160,)
R_test = R_test[..., cellid]  # shape (30, 30) -> used differently
```

### Index Management

This is critical and subtle. The loop maintains several index tensors, all referring to positions in the full `X` array (shape 3160):

```python
# Fixed random permutation (seed=8)
all_idx_perm = randperm(3160)

# Held-out test set (last 1000 of permutation)
test_1000_idx = all_idx_perm[-1000:]

# Remove test indices from pool
all_idx_perm = all_idx_perm[~isin(all_idx_perm, test_1000_idx)]  # now 2160 entries

# Initial training set (first ntrain_start of remaining permutation)
start_idx = all_idx_perm[:ntrain_start]  # e.g., first 50 or 500

# Dynamic tracking:
in_use_idx    = start_idx.clone()         # grows each iteration
xtilde_idx    = in_use_idx.clone()         # inducing points = in_use images (grows too)
remaining_idx = all_idx_perm[~isin(all_idx_perm, in_use_idx)]  # shrinks each iteration
```

**Important**: `in_use_idx` and `xtilde_idx` are always the same — every training image is also an inducing point. This means `ntilde = ntrain` at all times, which simplifies the math (K = K_tilde when ntilde = nt).

### Test Sets

Two test sets are used:
1. **X_test (30 images)**: The original test split with 30 repetitions per image. Used for R^2 evaluation via `utils.test()`.
2. **X_test_1000 (1000 images)**: A larger held-out set from the training pool (via `test_1000_idx`). Used for log-likelihood evaluation only (in the old notebook).

---

## 3. Initial Model Fitting

Before the loop starts, a full GP is fitted on the initial training set.

### Hyperparameters

```python
# Kernel hyperparameters (arc-cosine kernel)
theta = {
    'sigma_0':      1.0,      # Signal variance
    'Amp':          1.0,      # Amplitude
    'eps_0x':       0.0001,   # RF center x (starts near 0 = image center)
    'eps_0y':       0.0001,   # RF center y
    '-2log2beta':   4.8069,   # RF width (derived from logbetasam=5.5)
    '-log2rho2':    4.3069,   # Smoothness (derived from logrhosam=5.0)
}
# All have requires_grad_() for M-step optimization

# Firing rate parameters
f_params = {
    'logA':    log(0.01),   # log-amplitude of f = exp(A*lambda + lambda0)
    'lambda0': 1.0,         # offset (new notebook) or 0.31 (old notebook)
}
# logA has requires_grad_()
```

### Fitting Parameters

```python
fit_parameters = {
    'ntilde':      ntrain_start,  # inducing points = training points
    'maxiter':     40,            # outer iterations of E-M
    'nMstep':      15,            # M-step iterations per outer iter
    'nEstep':      15,            # E-step iterations per outer iter
    'nFparamstep': 15,            # F-param step iterations per outer iter
    'kernfun':     'acosker',     # arc-cosine kernel
    'cellid':      8,
    'n_px_side':   108,
}
```

### The `varGP()` Call

```python
fit_model, err_dict = GP_utils.varGP(X_in_use, R_in_use, **init_model)
```

`varGP()` (in `utils.py:5293`) is the core variational GP fitting function. It runs an EM-like loop:

**For each outer iteration (maxiter times):**

1. **Kernel computation + eigendecomposition**:
   - Compute `K_tilde = acosker(theta, xtilde, xtilde)` — shape (ntilde, ntilde)
   - Eigendecompose: `eigvals, eigvecs = eigh(K_tilde)`
   - Keep eigenvectors with `eigval > max(eigvals.max() * EIGVAL_TOL, EIGVAL_TOL)`
   - Project everything into this eigenspace via `B = eigvecs[:, ikeep]`:
     - `K_tilde_b = diag(eigvals[ikeep])` — diagonal in eigenspace
     - `K_b = K @ B`
     - `K_tilde_inv_b = diag(1/eigvals[ikeep])`
     - `m_b = B^T @ m`, `V_b = B^T @ V @ B`

2. **E-step** (update variational parameters m_b, V_b):
   - Compute lambda moments: `lambda_m, lambda_var = lambda_moments(...)`
   - Compute firing rate mean: `f_mean = exp(A*lambda_m + 0.5*A^2*lambda_var + lambda0)`
   - Inner loop (nEstep iterations): call `Estep()` to update m_b, V_b
   - Early stopping if f_mean converges

3. **F-param step** (update logA, lambda0):
   - LBFGS optimization of log-likelihood w.r.t. logA
   - lambda0 is estimated analytically from logA

4. **M-step** (update kernel hyperparameters theta):
   - LBFGS optimization of log-marginal = loglikelihood - KL_divergence
   - Computes kernel gradients w.r.t. theta
   - Bounds checking on hyperparameters

**Returns**: A model dictionary (or GPModel) with all fitted parameters, kernel matrices, values_track (history), etc.

---

## 4. The Loop: Standard Version (nd_utility)

This is the loop in `one_cell_active_training.ipynb`, Cell 20. It uses dictionary-based model state (pre-GPModel class).

### Per-Iteration Steps

#### Step 1: Extract model state

```python
# From the fitted model dictionary:
mask          = active_model['mask']           # pixel mask from localker (receptive field)
C             = active_model['C']              # local kernel component
B             = active_model['B']              # eigenvector projection matrix
K_tilde_b     = active_model['K_tilde_b']      # projected kernel (diagonal)
K_tilde_inv_b = active_model['K_tilde_inv_b']  # projected inverse kernel (diagonal)
m_b           = active_model['m_b']            # variational mean in eigenspace
V_b           = active_model['V_b']            # variational covariance in eigenspace
theta         = active_model['hyperparams_tuple'][0]
A             = exp(f_params['logA'])
lambda0       = f_params['lambda0']            # or exp(f_params['loglambda0'])
```

#### Step 2: (Optional) Evaluate on 1000-image test set

```python
# Compute kernel between test images and inducing points
Kvec_test = acosker(theta, X_test_1000[:,mask], x2=None, C=C, diag=True)
K_test    = acosker(theta, X_test_1000[:,mask], x2=xtilde[:,mask], C=C, diag=False)
K_test_b  = K_test @ B

# Get predictive moments
lambda_m_t, lambda_var_t = lambda_moments(X_test_1000[:,mask], K_tilde_b, K_test_b @ K_tilde_inv_b, ...)

# Compute log-likelihood
f_mean = mean_f_given_lambda_moments(f_params, lambda_m_t, lambda_var_t)
loglik = compute_loglikelihood(R_test_1000, f_mean, lambda_m_t, lambda_var_t, f_params)
```

#### Step 3: Compute utility for ALL remaining images

This is the core selection step. For every image not yet in the training set:

```python
xstar = X[remaining_idx]  # all unseen images

# Kernel between remaining images and inducing points
Kvec_star = acosker(theta, xstar[:,mask], x2=None, C=C, diag=True)   # (n_remaining,)
K_star    = acosker(theta, xstar[:,mask], x2=xtilde[:,mask], C=C, diag=False)  # (n_remaining, ntilde)
K_star_b  = K_star @ B  # project into eigenspace

# Predictive moments of lambda at each remaining image
lambda_m_t, lambda_var_t = lambda_moments(
    xstar[:,mask], K_tilde_b, K_star_b @ K_tilde_inv_b, Kvec_star, K_star_b, C, m_b, V_b, theta
)
# lambda_m_t:   shape (n_remaining,) — posterior mean of lambda(x*)
# lambda_var_t: shape (n_remaining,) — posterior variance of lambda(x*)

# Transform to log-firing-rate moments
logf_mean = A * lambda_m_t + lambda0   # shape (n_remaining,)
logf_var  = A**2 * lambda_var_t        # shape (n_remaining,)

# Compute utility for each remaining image
r_masked = arange(0, 100)  # spike count range for summation
u2d = nd_utility(logf_var, logf_mean, r_masked)  # shape (n_remaining,)

# Select best
i_best     = u2d.argmax()
x_idx_best = remaining_idx[i_best]
```

**What `nd_utility` computes**: See [Section 6.1](#61-nd_utility-the-standard-pnas-utility) below.

#### Step 4: Update indices

```python
in_use_idx = cat(in_use_idx, x_idx_best[None])
xtilde_idx = in_use_idx  # inducing points = all training images
ntilde     = xtilde_idx.shape[0]
```

#### Step 5: Update variational parameters to new dimensionality

When we add one image, `ntilde` grows by 1. The variational parameters m_b and V_b are in the eigenspace of the old K_tilde (size ntilde-1). We need to expand them to size ntilde:

```python
# Back-project from eigenspace to full space
V = B @ V_b @ B.T    # shape (ntilde-1, ntilde-1)
V = 0.5*(V + V.T)    # ensure symmetry
m = B @ m_b           # shape (ntilde-1,)

# Expand to new size (ntilde) by appending a new row/column
V_new = eye(ntilde)
V_new[:ntilde-1, :ntilde-1] = V
active_model['V'] = V_new

active_model['m'] = cat(m, m.mean()[None])  # new inducing point gets mean of existing means
```

#### Step 6: Incrementally update kernel matrices

Instead of recomputing the full K_tilde from scratch, only the new column/row is computed:

```python
# Compute only the new column of K_tilde
K_tilde_column = acosker(theta, xtilde_updated[:,mask], xtilde_updated[-1,mask][None], C=C, diag=False)
# shape: (ntilde, 1)

# Append to existing K_tilde
K_tilde = cat(K_tilde_reduced, K_tilde_column[:-1], dim=1)  # add column
K_tilde = cat(K_tilde, K_tilde_column.T, dim=0)              # add row

# Since ntilde == ntrain: K = K_tilde
K = K_tilde

# Recompute diagonal (full, because mask might have changed)
Kvec = acosker(theta, X_in_use[:,mask], x2=None, C=C, diag=True)

# Re-eigendecompose
eigvals, eigvecs = eigh(K_tilde)
ikeep = eigvals > max(eigvals.max() * EIGVAL_TOL, EIGVAL_TOL)
B = eigvecs[:, ikeep]
K_tilde_b = diag(eigvals[ikeep])
K_b = K @ B
K_tilde_inv_b = diag(1/eigvals[ikeep])
```

**Why this is efficient**: Computing one column of K_tilde is O(ntilde * n_masked_pixels) instead of O(ntilde^2 * n_masked_pixels) for the full matrix. The eigendecomposition is still O(ntilde^3) but unavoidable.

#### Step 7: Refit the GP

```python
active_model, err_dict = varGP(X_in_use, R_in_use, **active_model)
```

This runs the full E-M loop again, but starting from the warm-started variational parameters and kernel matrices. The `init_kernel` dict passed via `active_model` tells `varGP` to skip the initial kernel computation and use the pre-computed matrices.

#### Step 8: Evaluate on test set

```python
spk_count_test, spk_count_pred, r2, sigma_r2 = test(X_test, R_test, at_iteration=None, **active_model)
```

The `test()` function (in `utils.py:4424`):
- For each test image, computes `lambda_moments_star()` to get predictive mean and variance
- Predicts firing rate: `rate_star = exp(A*mu_star + 0.5*A^2*sigma_star2 + lambda0)`
- Computes R^2 (explained variance) between predicted and observed (averaged over 30 repetitions)

---

## 5. The Loop: Distribution-Aware Version

This is the loop in `one_cell_active_training_distribution_aware.ipynb`, Cell 15. Key differences from the standard version:

### Structural Differences

| Aspect | Standard (old) | Distribution-aware (new) |
|--------|---------------|-------------------------|
| Model state | Dictionary-based | `GPModel` class |
| ntrain_start | 50 | 500 |
| n_iterations | 250 | 50 |
| Utility method | `nd_utility()` | `batch_utility_w_grad()` + `conditioned_utility_clean()` |
| Kernel update | Manual column append | `generate_new_active_model()` |
| Variational update | Manual expand | Handled by `generate_new_active_model()` |
| Test evaluation | Every iteration | Commented out |

### Per-Iteration Steps

#### Step 1: Compute utility with `batch_utility_w_grad()`

```python
result_rms = GP_utility.batch_utility_w_grad(
    active_model,               # GPModel object
    X,                          # full image dataset
    remaining_idx,              # indices of unseen images
    max_r_cap=r_cutoff,         # r_cutoff = 100
    max_iter=10,                # L-BFGS iterations for optimization
    test_rms_constraint=True,   # optimize with RMS constraint
    lr=1,                       # learning rate
    return_logf_moments=True,
    verbose=False
)
x_idx_best = result_rms['img_idx'].item()
```

This function:
1. Computes `nd_utility` for ALL remaining images (batch)
2. Selects the image with highest utility
3. Optionally optimizes that image's pixels (with RMS constraint to keep mean/std fixed)
4. Returns the best image index and utility values

#### Step 2: Compute conditioned utility (FOR COMPARISON ONLY)

```python
with torch.enable_grad():
    result_cond = GP_utility.optimize_with_conditioned_utility(
        model=active_model,
        imgs_train=X,
        remaining_idx=remaining_idx,
        start_img_idx=x_idx_best,     # starts from the same image selected in step 1
        N=1,                           # single MC sample (DEBUG_SINGLE_IMAGE mode)
        lambda_samples=100,
        n_iterations=20,
        r_cutoff=r_cutoff,
        return_logf_moments=True,
        DEBUG_dict={
            'DEBUG_SINGLE_IMAGE': True,
            'DEBUG_FULL_FIELD': True,
            'verbose': False}
    )
```

This runs `conditioned_utility_clean()` which computes a more sophisticated utility that accounts for the joint information between x* and other potential images. It uses **augmented kernel matrices** to capture correlations.

**This step is purely for analysis/comparison** — the actual image selection was already done in step 1.

#### Step 3: Save comparison plots

```python
GP_utils.single_image_sampling_plot(result_cond, X, save_path=..., shared_scale=True, initial_img=...)
GP_utils.update_optimization_comparison(result_dict={'RMS': result_rms, 'cond_utility': result_cond}, ...)
```

#### Step 4: Update model with ORIGINAL image

**Critical detail**: The optimization in steps 1-2 may modify the image pixels (gradient-based optimization). But the model is trained on the **original** image, because in a real experiment we have neural responses to the original images only.

```python
new_spike = R[x_idx_best]
current_spikes = cat(current_spikes, new_spike[None])

active_model = GP_utils.generate_new_active_model(
    current_model=active_model,
    x_idx_chosen=tensor([x_idx_best]),
    img_train=X,
    new_spike_counts=current_spikes
)
```

`generate_new_active_model()` (in `utils.py:502-554`):
1. Creates a new `GPModel` copying hyperparameters and settings
2. Calls `set_new_model_idxs()` to update in_use_idx and xtilde_idx
3. Calls `set_new_model_variational_params()` to back-project m_b, V_b to full space and expand
4. Calls `get_new_model_kernels()` to incrementally compute new kernel matrices
5. Calls `set_new_model_f_params_and_theta()` to copy firing rate and kernel params

#### Step 5: Refit GP

```python
model_dict, err_dict = GP_utils.varGP(
    X[active_model.in_use_idx],
    current_spikes,
    **active_model.to_dict(),  # converts GPModel back to dictionary for varGP
    verbose=False
)
active_model = GPModel(model_dict=model_dict)
```

---

## 6. Utility Functions

### 6.1 `nd_utility` — The Standard PNAS Utility

**Location**: `utils.py:2114-2140`

**Mathematical formula** (PNAS Paper, Eq. 27):

```
U(x*) = H(r | x*, D) - E_f[ H(r | f, x*) ]
```

Where:
- `H(r | x*, D)` = entropy of the predicted spike count distribution (how uncertain we are about r given the data so far)
- `E_f[ H(r | f, x*) ]` = expected noise entropy (irreducible Poisson noise)
- The difference = **mutual information** between r and f given x* and D

**Computation steps**:

1. **Input**: `mu` (logf_mean), `sigma2` (logf_var) for each candidate image, `r_masked` = [0, 1, ..., 99]

2. **Laplace approximation of p(r | x*, D)** via `nd_p_r_given_xD()`:
   - For each spike count r and each candidate image, find the mode of log p(r, lambda | x*, D) using Newton's method (Lambert W function)
   - Compute Laplace approximation around the mode
   - Result: `p_response` of shape (r_max, n_candidates)

3. **Marginal entropy** `H(r | x*, D)`:
   ```
   H = -sum_r p(r) * log p(r)
   ```

4. **Expected noise entropy** `E[H(r | f, x*)]` via `nd_mean_noise_entropy()`:
   ```
   E[H] = -exp(mu + sigma2/2) * (mu + sigma2 - 1) + sum_r p(r) * log(r!)
   ```

5. **Utility**: `U = H - E[H]`, shape (n_candidates,)

**Improved version**: `nd_utility_new()` in `utility.py:816` fixes an overflow bug in the Laplace approximation by working in log-space for the Lambert W computation. The old `nd_utility()` can produce incorrect probabilities (sum >> 1) when variance is high.

### 6.2 `batch_utility_w_grad` — Batch Utility + Image Optimization

**Location**: `utility.py:4673-6606` (very large function, ~1900 lines)

This function wraps `nd_utility` with:

1. **Batch computation**: Computes utility for ALL remaining images at once (same math as `nd_utility`)
2. **Image selection**: Finds the argmax
3. **Optional gradient-based optimization**: After selecting the best image, optimizes its pixels to further increase utility

The optimization part supports many constraint modes (RMS, L2, sigmoid, etc.). In the distribution-aware notebook, it uses `test_rms_constraint=True` which:
- Keeps the image's mean and standard deviation fixed (to match the dataset distribution)
- Optimizes pixel values within that constraint using L-BFGS
- This is relevant for designing NEW stimuli, not for selecting existing ones

**For pure active learning on existing images** (no optimization), the function still works — the optimization step is skipped when no constraint flag is set, and it just returns the batch utility + best image index.

### 6.3 `conditioned_utility_clean` — Distribution-Aware Utility

**Location**: `utility.py:1992-2196`

This computes a **conditional** utility that accounts for correlations between x* and other potential future observations:

```
U(x*) = H(r | x*, D) - E_{x_i}[ E_{lambda(x_i)}[ H(r | x*, lambda(x_i), D) ] ]
```

The key difference from `nd_utility`: the second term averages over **sampled** future observations (x_i, lambda(x_i)), asking "how much would observing a random other image reduce our uncertainty about r at x*?"

**Computation**:

1. Compute marginal moments of lambda(x*) given D → `logf_mean`, `logf_var`
2. Compute marginal entropy `H(r | x*, D)` via Laplace approximation
3. Sample N random images x_i from remaining pool
4. For each x_i:
   - Compute lambda moments at x_i
   - Sample lambda(x_i) from its posterior
   - Build **augmented kernel matrix** incorporating x_i as if it were observed
   - Compute **updated posterior** mean m' and variance V' (Eq. 92 from the derivation)
   - Compute conditional predictive moments at x* given the observation at x_i
   - Compute conditional entropy `H(r | x*, lambda(x_i), D)`
5. Average conditional entropy over samples
6. Utility = marginal entropy - averaged conditional entropy

This is more expensive (requires N kernel evaluations per candidate) but theoretically more informative because it considers the correlations between images.

---

## 7. Key Internal Functions

### 7.1 `lambda_moments()` — Predictive Moments of the Latent Function

**Location**: `utils.py:3906-3956`

Computes the posterior mean and variance of lambda(x) under the variational approximation.

```
lambda_mean(x) = u(x)^T @ m_b
lambda_var(x)  = k(x,x) - k(x,xtilde)^T @ K_tilde_inv @ k(x,xtilde)
                 + u(x)^T @ V_b @ u(x)
```

Where `u(x) = K_tilde_inv_b @ k_b(x, xtilde)` is the projection vector.

**Inputs** (all in eigenspace, suffix _b):
- `K_tilde_b`: (n_eigen, n_eigen) — diagonal
- `KKtilde_inv_b`: (n_query, n_eigen) — precomputed u vectors
- `Kvec`: (n_query,) — k(x, x) diagonal
- `K_b`: (n_query, n_eigen) — k(x, xtilde) @ B
- `m_b`: (n_eigen,) — variational mean
- `V_b`: (n_eigen, n_eigen) — variational covariance

**Outputs**:
- `lambda_m`: (n_query,) — posterior mean
- `lambda_var`: (n_query,) — posterior variance (diagonal of covariance)

### 7.2 The Log-Firing-Rate Transform

The GP models lambda(x), but neural responses depend on the firing rate f(x). The link:

```
log f(x) = A * lambda(x) + lambda0
```

So:
```
logf_mean = A * lambda_m + lambda0
logf_var  = A^2 * lambda_var
```

And the expected firing rate (for prediction, not utility) is:
```
E[f(x)] = exp(logf_mean + 0.5 * logf_var)
         = exp(A * lambda_m + 0.5 * A^2 * lambda_var + lambda0)
```

This is used in `test()` for predicting spike counts.

### 7.3 The Arc-Cosine Kernel (`acosker`)

The kernel function used throughout. It's a non-stationary kernel that naturally incorporates a receptive field structure:

```
k(x, x') = sigma_0^2 * arccos_kernel(C @ x_masked, C @ x_masked')
```

Where:
- `mask` selects pixels within the receptive field (from `localker()`)
- `C` is a local kernel matrix that weights pixels by their distance from the RF center
- `theta` parameters control RF center (eps_0x, eps_0y), width (beta), and smoothness (rho)

### 7.4 The Eigenspace Projection

A recurring pattern: K_tilde (ntilde x ntilde) is eigendecomposed and projected into the subspace of large eigenvalues:

```python
eigvals, eigvecs = eigh(K_tilde)
ikeep = eigvals > max(eigvals.max() * EIGVAL_TOL, EIGVAL_TOL)
B = eigvecs[:, ikeep]  # (ntilde, n_eigen)
```

This serves two purposes:
1. **Numerical stability**: Small eigenvalues cause ill-conditioning
2. **Efficiency**: n_eigen << ntilde, so all operations in eigenspace are cheaper

All matrices with suffix `_b` are in this projected eigenspace.

### 7.5 `test()` — Performance Evaluation

**Location**: `utils.py:4424-4508`

For each test image:
1. Compute kernel between test image and inducing points
2. Get `lambda_moments_star()` (single-point version of `lambda_moments`)
3. Predict: `rate_star = exp(A*mu_star + 0.5*A^2*sigma_star2 + lambda0)`
4. Compute explained variance (Pearson R^2) against observed responses

Returns: `(R_test_observed, R_predicted, r2, sigma_r2)`

---

## 8. Model State and Data Structures

### Dictionary-based (old notebook)

The old notebook uses plain dictionaries. After `varGP()` returns:

```python
active_model = {
    # Fit parameters
    'fit_parameters': {
        'ntilde': int,
        'maxiter': int,
        'nMstep': int,
        'nEstep': int,
        'nFparamstep': int,
        'kernfun': str or function,
        'cellid': int,
        'n_px_side': int,
        'in_use_idx': tensor,      # indices of training images in X
        'xtilde_idx': tensor,      # indices of inducing points in X (= in_use_idx)
        'start_idx': tensor,       # original starting indices
        'eigval_tol': float,       # eigenvalue tolerance
    },

    # Inducing points
    'xtilde': tensor,              # (ntilde, n_pixels)

    # Kernel matrices (in eigenspace)
    'mask': tensor,                # boolean pixel mask
    'C': tensor,                   # local kernel matrix
    'B': tensor,                   # eigenvector projection (ntilde, n_eigen)
    'K_tilde_b': tensor,           # diagonal kernel in eigenspace
    'K_tilde_inv_b': tensor,       # diagonal inverse
    'K_b': tensor,                 # projected kernel
    'Kvec': tensor,                # diagonal k(x,x)
    'm_b': tensor,                 # variational mean (n_eigen,)
    'V_b': tensor,                 # variational covariance (n_eigen, n_eigen)

    # Hyperparameters
    'hyperparams_tuple': (theta_dict, theta_lower_lims, theta_higher_lims),

    # Firing rate parameters
    'f_params': {'logA': tensor, 'lambda0': tensor},

    # Full-space kernel (for incremental updates)
    'final_kernel': {
        'C': tensor,
        'mask': tensor,
        'K_tilde': tensor,         # full-space K_tilde (ntilde, ntilde)
        'K': tensor,               # full-space K
        'eigvecs': tensor,         # full eigenvector matrix
    },

    # Training history
    'values_track': {
        'loss_track': {'logmarginal': [...], 'loglikelihood': [...], 'KL': [...]},
        'theta_track': {key: [...] for key in theta},
        'variation_par_track': {'m_b': [...], 'V_b': [...]},
        'f_par_track': {'logA': [...], 'lambda0': [...]},
    },
}
```

### GPModel class (new notebook)

The `GPModel` class (`GP_model/model.py`) wraps the same data with attribute access:

```python
model.ntilde           # = fit_parameters['ntilde']
model.in_use_idx       # = fit_parameters['in_use_idx']
model.xtilde           # inducing points
model.m_b, model.V_b   # variational parameters
model.theta            # = hyperparams_tuple[0]
model.f_params         # firing rate parameters
model.C, model.mask    # kernel components
model.B                # eigenvector projection
model.K_tilde_b        # projected kernel
model.K_tilde_inv_b    # projected inverse
model.to_dict()        # convert back to dictionary for varGP()
```

Key methods:
- `GPModel(model_dict=...)` — construct from varGP output dictionary
- `GPModel(old_model=...)` — construct from another GPModel, copying non-kernel params
- `model.to_dict()` — convert to dictionary for passing to `varGP()`
- `model.sync_kernel_components()` — update individual attributes from kernel values object

---

## 9. Differences Between Old and New Notebooks

### Parameter differences

| Parameter | Old notebook | New notebook |
|-----------|-------------|-------------|
| `ntrain_start` | 50 | 500 |
| `n_iterations` | 250 | 50 |
| `maxiter` | 40 | 40 |
| `nEstep` | 15 | 15 |
| `nMstep` | 15 | 15 |
| `nFparamstep` | 15 | 15 |
| `lambda0` init | 0.31 | 1.0 |
| `logA` init | log(0.01) | log(0.01) |
| `r_masked` range | 0..100 | 0..100 |
| `seed` for data split | 8 | 8 |
| Random loop | Cell 22 (10 random seeds, 250 iter each) | Not included |

### Architectural differences

| Aspect | Old | New |
|--------|-----|-----|
| Model representation | Plain dict | `GPModel` class |
| Kernel update | Manual column append + eigendecomp | `generate_new_active_model()` |
| Variational param expansion | Manual (B @ V_b @ B.T, then pad) | `set_new_model_variational_params()` |
| Utility computation | `nd_utility()` directly | `batch_utility_w_grad()` wrapping `nd_utility` |
| Test evaluation | Every iteration (R^2 + loglik) | Commented out |
| Image optimization | Not present | Yes (RMS constraint + conditioned utility) |
| Random baseline | Included (Cell 22) | Not included |

### What the new notebook adds

The new notebook is fundamentally a **comparison study** of utility methods, not a production active learning loop. It:
1. Uses `batch_utility_w_grad()` for selection (which internally still uses `nd_utility` for the batch step)
2. Also computes `conditioned_utility_clean()` on the same selected image
3. Saves detailed comparison plots to disk
4. Does NOT track R^2 during the loop (evaluation is commented out)

---

## 10. How This Fits in the Experimental Pipeline

### Offline (Notebook) Setting

The notebooks operate on a **pre-recorded dataset** (PNAS paper data). The "active learning" is simulated: we have all 3160 image-response pairs already, and we simulate the sequential selection process to demonstrate that active selection converges faster than random selection.

**The flow**:
1. Load full dataset
2. Pretend we only have 50 (or 500) images
3. Run active loop, "revealing" responses one at a time
4. Compare convergence curve to random selection baseline

### Online (Real Experiment) Setting

In a real closed-loop experiment (the ClosedLoopProject's main purpose), the active learning loop would be:

1. Present initial random images to retina via DMD (Digital Micromirror Device)
2. Record neural responses via MEA (Multi-Electrode Array)
3. Fit GP on initial data
4. **LOOP**:
   a. Compute utility for candidate images (from a pre-generated pool, or optimize new ones)
   b. Present the highest-utility image via DMD
   c. Record the neural response
   d. Update model with new data point
   e. Refit GP
5. Continue until performance plateaus or budget exhausted

The key connection: the notebook code demonstrates the **ML side** (steps 3-4.e), which in the full system runs on the Linux machine. The hardware communication (DMD control, MEA recording) runs on the Windows machine via ZMQ, handled by the code in `src/TCP/` and `src/Win_side/`.

### Which utility to use in production

For a real experiment:
- **`nd_utility`** (or `nd_utility_new`): The proven, simple choice. Fast enough for online use (computes utility for ~2000 images in seconds). This is what was validated in the PNAS paper.
- **`batch_utility_w_grad`**: Use if you want to also **optimize** the selected image's pixels (design a new stimulus, not just select from existing ones). Adds computational cost for the optimization step.
- **`conditioned_utility_clean`**: Theoretically better but much slower (requires MC sampling). Not validated for production use. The new notebook was comparing it to the standard approach.

### Connection to standalone branches

The standalone branches (`standalone-generate`, `standalone-gpytorch`) contain the cleaned-up experimental code that actually talks to hardware. The GP fitting and utility computation would be called from `standalone_linux/main_loop.py`, which manages the ZMQ communication and experiment flow. The active learning logic from these notebooks needs to be integrated into that main loop.

---

## Appendix A: Quick Reference — Running the Standard Active Learning Loop

Minimal code to run the standard active learning (no distribution-aware comparison):

```python
# After fitting initial model:
active_model = copy.deepcopy(start_model)

for j in range(n_iterations):
    # 1. Get remaining images
    in_use_idx = active_model['fit_parameters']['in_use_idx']
    remaining_idx = all_idx_perm[~isin(all_idx_perm, in_use_idx)]

    # 2. Extract model state
    theta, mask, C, B = ...  # from active_model
    K_tilde_b, K_tilde_inv_b, m_b, V_b = ...  # from active_model
    A, lambda0 = ...  # from f_params

    # 3. Compute utility for all remaining images
    xstar = X[remaining_idx]
    Kvec = acosker(theta, xstar[:,mask], x2=None, C=C, diag=True)
    K    = acosker(theta, xstar[:,mask], x2=xtilde[:,mask], C=C, diag=False)
    K_b  = K @ B
    lambda_m, lambda_var = lambda_moments(xstar[:,mask], K_tilde_b, K_b @ K_tilde_inv_b, Kvec, K_b, C, m_b, V_b, theta)
    logf_mean = A * lambda_m + lambda0
    logf_var  = A**2 * lambda_var
    u = nd_utility(logf_var, logf_mean, arange(0, 100))

    # 4. Select best image
    i_best = u.argmax()
    x_idx_best = remaining_idx[i_best]

    # 5. Update model (indices, variational params, kernel matrices)
    # ... (see Section 4, Steps 4-6)

    # 6. Refit
    active_model, err_dict = varGP(X_in_use, R_in_use, **active_model)

    # 7. Evaluate
    _, _, r2, _ = test(X_test, R_test, at_iteration=None, **active_model)
```

## Appendix B: Function Location Quick Reference

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `nd_utility()` | utils.py | 2114 | Standard utility (PNAS Eq. 27) |
| `nd_utility_new()` | utility.py | 816 | Fixed Laplace approximation |
| `batch_utility_w_grad()` | utility.py | 4673 | Batch utility + image optimization |
| `conditioned_utility_clean()` | utility.py | 1992 | Distribution-aware utility |
| `optimize_with_conditioned_utility()` | utility.py | 2198 | Adam optimizer wrapper for conditioned utility |
| `lambda_moments()` | utils.py | 3906 | Predictive mean/variance of lambda |
| `varGP()` | utils.py | 5293 | Full variational GP fitting |
| `generate_new_active_model()` | utils.py | 502 | Incremental model update |
| `test()` | utils.py | 4424 | R^2 evaluation on test set |
| `nd_p_r_given_xD()` | utils.py | 2078 | Laplace approximation of p(r|x,D) |
| `nd_mean_noise_entropy()` | utils.py | ~1884 | Expected Poisson noise entropy |
| `acosker()` | utils.py | (search) | Arc-cosine kernel function |
| `localker()` | utils.py | (search) | Receptive field mask + C matrix |
| `Estep()` | utils.py | (search) | Variational E-step update |
| `GPModel` | GP_model/model.py | 13 | Model class wrapper |

---

*Generated: 2026-04-02*
*Source: Analysis of one_cell_active_training.ipynb and one_cell_active_training_distribution_aware.ipynb*
