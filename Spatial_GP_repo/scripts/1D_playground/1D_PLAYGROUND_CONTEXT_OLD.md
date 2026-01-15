# GP Utility Playground - Session Context

## Purpose

This playground was created to **validate intuitions about the acquisition/utility function** used in our active learning system for neural response modeling. The main codebase (`Spatial_GP_repo/`) fits Gaussian Processes to neural responses to high-dimensional natural images. Before experimenting with complex modifications, we needed a simpler testbed.

**Core question we wanted to answer:**
> In a bounded 1D input domain with uniform stimulus distribution p(x), does the utility function select points that evenly space the domain?

**Answer: Yes.** The utility peaks in gaps between existing training points, which naturally leads to even spacing as points are added iteratively.

## Background

### The Utility Function Formula - Called ND_UTILITY in its non distribution aware form.

The utility (mutual information between spike count r and latent λ) is:

```
U(x*) = H_marg - H_cond
```

Where:
- **H_marg** = H[r|x*, D] = -Σ_r p(r|D) log p(r|D) — marginal entropy of spike count
- **H_cond** = E_λ[H[r|λ]] — expected conditional entropy (Poisson entropy averaged over posterior)

This measures how much observing r at x* would reduce uncertainty about λ.

---

## Utility Computation Methods

All functions are in `Spatial_GP_repo/utility.py`. They differ in how H_marg and H_cond are computed.

### Summary Table

| Function | H_marg | H_cond | Status | Typical Error vs NUMERICAL |
|----------|--------|--------|--------|---------------------------|
| `nd_utility_new` | Laplace | Laplace | ✅ RECOMMENDED | ~3.6% |
| `nd_utility_NUMERICAL` | Gauss-Hermite | Gauss-Hermite | ✅ GROUND TRUTH | 0% (reference) |
| `nd_utility_hybrid` | Laplace | MC (exact per sample) | ✅ NEW | ~3.5% |
| `nd_utility_MC_batched` | MC | Analytical+MC (inconsistent!) | ⚠️ ISSUES | 45-50% |
| `nd_utility_MC` | MC | Analytical+MC | ⚠️ ISSUES | Similar to batched |
| `nd_utility` | Laplace (old) | Laplace (old) | ❌ DEPRECATED | Overflow bugs |

---

<!-- ========== nd_utility_new ========== -->
### `nd_utility_new()` — Laplace Approximation (RECOMMENDED)

**Status**: ✅ RECOMMENDED for production use

**Location**: `utility.py`

**Signature**:
```python
def nd_utility_new(mu, sigma2, r_max=100, A=1.0, lambda0=0.0)
```

**Method**:
- **H_marg**: Uses `laplace_approximations_new()` to compute p(r|D) via Laplace approximation around the mode, then H = -Σ p log p
- **H_cond**: Analytical formula using Laplace-approximated E[log(r!)]

**Pros**:
- Numerically stable (uses log-space Lambert W)
- Fast (no sampling)
- Properly normalized (Σp ≈ 1.0 always)

**Cons**:
- Approximation error ~3.6% vs numerical integration

**Test Results** (from `test_nd_utility_MC.py`):
```
Laplace vs NUMERICAL error: 3.57%
```

---

<!-- ========== nd_utility_NUMERICAL ========== -->
### `nd_utility_NUMERICAL()` — Gauss-Hermite Quadrature (GROUND TRUTH)

**Status**: ✅ GROUND TRUTH for validation

**Location**: `utility.py`

**Signature**:
```python
def nd_utility_NUMERICAL(mu, sigma2, r_max=500, n_quadrature=100)
```

**Method**:
- **H_marg**: Gauss-Hermite quadrature to compute p(r|D) = ∫ Poisson(r|exp(λ)) N(λ|μ,σ²) dλ
- **H_cond**: Gauss-Hermite quadrature to compute E_λ[H[Poisson(exp(λ))]]

**Pros**:
- Most accurate (true numerical integration)
- Internally consistent
- Use as reference for validating other methods

**Cons**:
- Slower than Laplace
- Requires sufficient quadrature points for high variance

**Test Results**:
```
This is the reference. Other methods are compared against it.
```

---

<!-- ========== nd_utility_hybrid ========== -->
### `nd_utility_hybrid()` — Laplace H_marg + MC H_cond (NEW)

**Status**: ✅ NEW - good for research/comparison

**Location**: `utility.py`

**Signature**:
```python
def nd_utility_hybrid(mu, sigma2, r_max=500, n_samples=10000, batch_size=500)
```

**Method**:
- **H_marg**: Laplace approximation via `laplace_approximations_new()` (same as nd_utility_new)
- **H_cond**: Monte Carlo — for each sampled λ, compute exact Poisson entropy, then average

```python
# H_cond computation (simplified):
for each batch:
    λ_samples ~ N(μ, σ²)
    f_samples = exp(λ_samples)
    H_batch = -Σ_r Poisson(r|f) * log(Poisson(r|f))  # Exact Poisson entropy
H_cond = mean(H_batch)
```

**Pros**:
- Internally consistent (no analytical/MC mismatch)
- H_marg from stable Laplace, H_cond from unbiased MC
- Converges to correct answer with enough samples

**Cons**:
- Slower than pure Laplace
- Still affected by r_max truncation for very high firing rates

**Test Results** (from `test_nd_utility_MC.py`, 100k samples):
```
Hybrid vs NUMERICAL error: 3.53%
Hybrid vs Laplace error:   6.63%
```

---

<!-- ========== nd_utility_MC_batched ========== -->
### `nd_utility_MC_batched()` — Batched Monte Carlo (HAS ISSUES)

**Status**: ⚠️ HAS FUNDAMENTAL ISSUES — use `nd_utility_hybrid` instead

**Location**: `utility.py`

**Signature**:
```python
def nd_utility_MC_batched(mu, sigma2, r_max=500, n_samples=100000, batch_size=10000)
```

**Method**:
- **H_marg**: MC — average Poisson probs over λ samples, then compute entropy
- **H_cond**: MIXED — analytical term1 + MC E[log(r!)]

```python
# H_cond decomposition:
# H[Poisson(f)] = f(1-λ) + E_r[log(r!)]
term1 = exp(μ + σ²/2) * (1 - μ - σ²)  # ANALYTICAL (exact)
E_log_r_fact = MC_estimate             # MC (truncated at r_max)
H_cond = term1 + E_log_r_fact          # INCONSISTENT MIX!
```

**THE PROBLEM — Internal Inconsistency**:

| Component | Computation | Affected by r_max truncation? |
|-----------|-------------|-------------------------------|
| term1 | Analytical | **NO** — includes all λ values |
| E_log_r_fact | MC | **YES** — only r < r_max |
| H_marg | MC | **YES** — only r < r_max |

When high-variance posteriors produce λ samples where f = exp(λ) ≈ r_max:
- term1 includes these samples (can be large negative number)
- E_log_r_fact misses them (truncated out)
- Result: **H_cond is miscalculated, can even be negative!**

**Test Results** (from `test_nd_utility_MC.py`, 100k samples):
```
nd_utility_MC range: [-0.166659, 0.188180]  # NEGATIVE VALUES! (impossible for MI)
Numerical vs MC error: 47.41%
```

**Why negative utility is impossible**: U = H_marg - H_cond is mutual information, which is always ≥ 0.

---

<!-- ========== nd_utility_MC ========== -->
### `nd_utility_MC()` — Basic Monte Carlo (HAS ISSUES)

**Status**: ⚠️ SAME ISSUES as `nd_utility_MC_batched`

**Location**: `utility.py`

**Signature**:
```python
def nd_utility_MC(mu, sigma2, r_max=100, n_samples=5000)
```

**Method**: Same as `nd_utility_MC_batched` but without batching (higher memory usage).

**Issues**: Same internal inconsistency problem. Also can run out of GPU memory for large sample counts.

---

<!-- ========== nd_utility (DEPRECATED) ========== -->
### `nd_utility()` — Old Laplace (DEPRECATED)

**Status**: ❌ DEPRECATED — use `nd_utility_new` instead

**Location**: `utility.py`

**Issues**: Uses `argmax_g_old()` which has catastrophic overflow handling:
- In high-uncertainty regions, assigns constant p=0.27 to all large r values
- Causes Σp >> 1 (normalization failure)
- See `OVERFLOW_BUG_ANALYSIS.md` for details

---

<!-- ========== distribution_aware_utility_gpytorch ========== -->
### `distribution_aware_utility_gpytorch()` — p(x)-Aware Acquisition

**Status**: ✅ For distribution-aware active learning

**Location**: `utility.py`

**Signature**:
```python
def distribution_aware_utility_gpytorch(
    model, x_star, r_max=100, n_samples=50, n_lambda_samples=1,
    batch_size=1000, p_x_type="uniform", x_min=0.0, x_max=1.0,
    gaussian_mean=0.5, gaussian_std=0.2, A=1.0, lambda0=0.0, x_samples=None
)
```

**Method**:
```
U(x*) = H_marg(x*) - E_{x~p(x)} E_{λ(x)|D}[ H(r | x*, λ(x), D) ]
```

This utility asks: "How much will querying x* reduce uncertainty about predictions at locations x ~ p(x)?"

**Key insight**: Unlike `nd_utility_new` which only considers uncertainty at x*, this function accounts for *where predictions matter* by weighting by p(x).

---

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | GPyTorch model | Trained variational GP model |
| `x_star` | Tensor (K,) | Candidate query points to evaluate |
| `r_max` | int | Maximum spike count for summation |
| `n_samples` | int | Number of x samples from p(x) (ignored if `x_samples` provided) |
| `n_lambda_samples` | int | Lambda samples per x for MC estimation of H_cond |
| `batch_size` | int | Batch size for MC sampling (memory management) |
| `p_x_type` | str | `"uniform"` or `"gaussian"` — stimulus distribution type |
| `x_min`, `x_max` | float | Bounds for uniform distribution |
| `gaussian_mean`, `gaussian_std` | float | Parameters for Gaussian distribution |
| `A`, `lambda0` | float | Link function parameters: f = exp(A*λ + λ₀) |
| `x_samples` | Tensor or None | **If provided, uses these x values instead of sampling from p(x)** |

---

#### Usage Modes

**Mode 1: Uniform p(x)**
```python
u = distribution_aware_utility_gpytorch(
    model, x_star, p_x_type="uniform", x_min=0.0, x_max=1.0, n_samples=100
)
```
Samples x uniformly from [x_min, x_max]. Good for domains where all locations are equally important.

**Mode 2: Gaussian p(x)**
```python
u = distribution_aware_utility_gpytorch(
    model, x_star, p_x_type="gaussian",
    gaussian_mean=0.5, gaussian_std=0.2, n_samples=100
)
```
Samples x from N(mean, std²). Good when predictions near certain locations matter more.

**Mode 3: Dirac Delta p(x) = δ(x - x*)**
```python
u = distribution_aware_utility_gpytorch(
    model, x_star,
    x_samples=x_star,  # KEY: x_samples = x_star triggers Dirac delta mode
    n_lambda_samples=100000,
    batch_size=5000
)
```
When `x_samples = x_star`, computes p(x) = δ(x - x*). This **should collapse exactly to `nd_utility_new`** because we're only asking about uncertainty reduction at x* itself.

---

#### Dirac Delta Collapse — Mathematical Justification

When p(x) = δ(x - x*), the conditional distribution simplifies:

```
λ(x*) | λ(x) = λ(x*) | λ(x*)
```

Since we're conditioning on the same point:
- **μ_cond** = λ (the sampled value itself)
- **σ²_cond** = 0 (no uncertainty when conditioning on itself)

Therefore H_cond becomes:
```
H_cond = E_λ[H[r | λ]]  — same as nd_utility!
```

The function detects this case automatically when `x_samples = x_star` and bypasses cross-covariance computation.

---

#### Test Results — Dirac Delta Collapse Verification

From `test_dirac_delta_collapse.py` (100k lambda samples, 50 training points):

**Utility ranges**:
```
nd_utility (Laplace) range:   [0.012964, 1.005132]
nd_utility (NUMERICAL) range: [0.013145, 0.854418]
Dirac delta range:            [0.012742, 0.854260]
```

**Error comparison**:
```
Dirac vs NUMERICAL: 0.82% mean error  ✓ Excellent
Dirac vs Laplace:   4.20% mean error  ✓ Good (within Laplace approximation error)
Laplace vs NUMERICAL: 3.75% mean error (baseline)
```

The Dirac delta MC matches NUMERICAL better than Laplace does, confirming the implementation is correct.

---

#### Implementation Notes

1. **H_marg**: Always computed via Laplace approximation (`laplace_approximations_new`)

2. **H_cond for general p(x)**:
   - Sample x ~ p(x)
   - Sample λ(x) from GP posterior at x
   - Compute conditional moments μ_cond, σ²_cond at x* given λ(x) using cross-covariance
   - Compute Poisson entropy with these conditional moments
   - Average over samples

3. **H_cond for Dirac delta** (optimized path):
   - Detect `x_samples = x_star`
   - Skip cross-covariance (σ²_cond = 0 exactly)
   - Sample λ directly from GP posterior at x*
   - Compute exact Poisson entropy for each sample
   - Average over samples (batched for memory efficiency)

**Mathematical derivation**: See:
- `~/IDV_code/Papers/latex_summaries/active_learning_pietro_corrected.tex`
- `~/IDV_code/Papers/latex_summaries/predictive_distribution_derivation.tex`

---

## Test Scripts for Utility Methods

| Script | Purpose |
|--------|---------|
| `test_nd_utility_MC.py` | Compares Laplace, MC, Numerical, and Hybrid methods |
| `test_dirac_delta_collapse.py` | Verifies distribution_aware_utility collapses to nd_utility when p(x) = δ(x-x*) |
| `test_distribution_aware_utility.py` | Compares nd_utility with distribution_aware for uniform/Gaussian p(x) |

---

## Detailed Test Results (from `test_nd_utility_MC.py`)

**Test configuration**:
- 100 training points, 200 candidate points
- r_max = 1000, 100k MC samples
- GP posterior: max μ = 5.70, max σ = 0.41, max E[f] = 299.8

**Utility ranges**:
```
nd_utility_new (Laplace):   [0.000344, 0.057191]
nd_utility_MC_batched:      [-0.166659, 0.188180]  # NEGATIVE!
nd_utility_NUMERICAL:       [0.000353, 0.063976]
nd_utility_hybrid:          [0.000281, 0.064026]
```

**Error comparison (mean relative error)**:
```
Laplace vs NUMERICAL:  3.57%   ✓ Good
Hybrid vs NUMERICAL:   3.53%   ✓ Good
Hybrid vs Laplace:     6.63%   ✓ Acceptable
MC_batched vs NUMERICAL: 47.41%  ✗ Unacceptable (inconsistency bug)
```

---

## Recommendations

1. **For production**: Use `nd_utility_new()` — fast, stable, ~3.5% error
2. **For validation**: Use `nd_utility_NUMERICAL()` — ground truth reference
3. **For research**: Use `nd_utility_hybrid()` — good accuracy with MC for H_cond
4. **For p(x)-aware**: Use `distribution_aware_utility_gpytorch()`
5. **AVOID**: `nd_utility_MC`, `nd_utility_MC_batched` (inconsistency bug), `nd_utility` (overflow bug)

### Why a Playground?

The main codebase has:
- High-dimensional inputs (natural images)
- Complex arc-cosine kernel with structured covariance matrix
- Many hyperparameters and dependencies

The playground strips this down to:
- **1D inputs** in [0, 1]
- **Simple RBF kernel**
- **Same utility function** (`nd_utility_new` — see detailed comparison above)
- **GPyTorch** (user wanted to learn this library)

## Implementation Details

### Files

| File | Purpose |
|------|---------|
| `scripts/1D_playground/gp_utility_playground.py` | Main playground script (single-shot GP fit + utility) |
| `scripts/1D_playground/active_learning_loop.py` | Active learning loop (iterative acquisition) |
| `scripts/1D_playground/test_nd_utility_MC.py` | **Compares Laplace, MC, Numerical, Hybrid utility methods** |
| `scripts/1D_playground/test_dirac_delta_collapse.py` | **Verifies distribution_aware collapses to nd_utility for Dirac delta** |
| `scripts/1D_playground/test_distribution_aware_utility.py` | Compares nd_utility with distribution_aware for uniform/Gaussian p(x) |
| `scripts/1D_playground/gp_utility_playground_result.png` | Single-shot output |
| `scripts/1D_playground/active_learning_result.png` | Active learning output |
| `scripts/1D_playground/test_nd_utility_MC_result.png` | Utility method comparison plot |
| `scripts/1D_playground/test_dirac_delta_collapse_result.png` | Dirac delta collapse verification plot |
| `scripts/1D_playground/test_distribution_aware_result.png` | Distribution-aware utility comparison plot |
| `scripts/1D_playground/GP_PLAYGROUND_CONTEXT.md` | This context file |

### Key Technical Decisions

1. **Poisson Likelihood**: We implemented a custom `PoissonLikelihood` class because GPyTorch doesn't have one built-in. This was necessary because `nd_utility()` assumes Poisson observations - using Gaussian likelihood would give incorrect posterior moments.

2. **Variational Inference**: Poisson likelihood requires approximate inference (not exact GP). We use GPyTorch's `ApproximateGP` with `VariationalStrategy`.

3. **Manual ELBO**: We compute the variational ELBO explicitly rather than using GPyTorch's `VariationalELBO` abstraction. This is more educational and avoids compatibility issues with custom likelihoods.

4. **Gradient Context**: The `utility.py` module sets `torch.set_grad_enabled(False)` globally, so training requires explicit `torch.enable_grad()` context.

### Model Structure

```python
# Poisson likelihood with log-link: y ~ Poisson(exp(f))
# For variational inference, expected log-likelihood has closed form:
# E[log p(y|f)] = y*μ - exp(μ + σ²/2) - log(y!)

class PoissonLikelihood(gpytorch.likelihoods.Likelihood):
    def expected_log_prob(self, target, input):
        mean, var = input.mean, input.variance
        return (target * mean - torch.exp(mean + var/2)).sum(-1)

# Variational GP with RBF kernel
class VariationalGP(gpytorch.models.ApproximateGP):
    # Uses CholeskyVariationalDistribution + VariationalStrategy
    # Kernel: ScaleKernel(RBFKernel())
```

### Configuration

```python
X_MIN, X_MAX = 0.0, 1.0  # Input domain
N_INDUCING = 20          # Inducing points for variational GP
MAX_R = 50               # Max spike count for utility computation
DTYPE = torch.float64    # Numerical precision
```

### Ground Truth Functions

Two options available in `gp_utility_playground.py`:

**1. Sinusoidal (default)**
```python
def lambda_true(x):
    # Periodic, firing rates roughly in [0.5, 7]
    return torch.sin(2 * np.pi * x) + 1.0
```

**2. Asymmetric bump (optional)**
```python
def lambda_true_asymmetric(x):
    """
    Single peak at x=0.3, steep left side, gradual right decay.
    Not periodic - looks like a skewed distribution.
    """
    peak = 0.3
    sigma_left = 0.1    # Steep rise
    sigma_right = 0.35  # Gradual decay
    amplitude = 6.0     # Peak firing rate ~ exp(6) ≈ 400
    baseline = 0.5      # Minimum firing rate ~ exp(0.5) ≈ 1.6

    left_mask = x < peak
    z = torch.zeros_like(x)
    z[left_mask] = ((x[left_mask] - peak) / sigma_left) ** 2
    z[~left_mask] = ((x[~left_mask] - peak) / sigma_right) ** 2

    return baseline + (amplitude - baseline) * torch.exp(-0.5 * z)
```

### Active Learning Loop

`active_learning_loop.py` implements iterative acquisition:

```python
def run_active_learning_loop(n_initial, n_iterations, lambda_fn):
    # 1. Start with n_initial random points
    # 2. For each iteration:
    #    - Train fresh GP on current data
    #    - Evaluate utility on candidate grid
    #    - Select max-utility point
    #    - Observe Poisson response
    #    - Add to training set
    # 3. Return history for visualization
```

**Configuration:**
- `N_INITIAL = 2-3`: Starting points
- `N_ITERATIONS = 5-17`: Points to add
- Creates fresh model each iteration (simpler than updating)

**Visualization:**
- Subplot 1: Acquisition order (numbered points on domain)
- Subplot 2: Final GP fit in firing rate space
- Subplot 3: Final utility landscape

## Results Summary

With training points at x = [0.1, 0.15, 0.8]:
- **Max utility at x ≈ 0.39** (in the large gap between 0.15 and 0.8)
- Utility is LOW near existing training points
- Utility is HIGH where GP uncertainty is highest

This confirms the expected behavior: the utility function drives exploration of unexplored regions.

## Things to Experiment With

### Completed

- [x] **Active Learning Loop**: Implemented in `active_learning_loop.py`. Confirms gap-filling behavior.
- [x] **Asymmetric Ground Truth**: Added `lambda_true_asymmetric()` with single peak at x=0.3.

### Observed: Boundary Effects

The active learning loop revealed that **boundaries (x=0 and x=1) get selected repeatedly**. This is expected because:
- GP has no information beyond domain edges
- Uncertainty remains high at boundaries
- Utility stays elevated there

This is a known issue in active learning - could be addressed by:
- Excluding boundary points from candidates
- Adding prior observations at boundaries
- Using a different kernel that handles boundaries

### Still to Explore

1. **Effect of Firing Rate Magnitude**
   - Very low rates (high Poisson noise relative to mean)
   - Very high rates (Poisson approaches Gaussian)

2. **Comparison with Standard Acquisition Functions**
   - Pure variance-based selection (max σ²)
   - Expected Improvement / Upper Confidence Bound

### Completed

- [x] **Non-Uniform p(x)**: Implemented `distribution_aware_utility_gpytorch()` which supports:
  - Uniform p(x) over domain
  - Gaussian p(x) with configurable mean and std
  - Test script: `test_distribution_aware_utility.py`
  - The utility now asks "how informative is x* for predicting at locations x ~ p(x)?"

## Connection to Main Codebase

The playground uses:
- `nd_utility_new()` from `Spatial_GP_repo/utility.py` (RECOMMENDED — see utility methods section above)
- `distribution_aware_utility_gpytorch()` for p(x)-aware acquisition (GPyTorch version)
- Same Poisson likelihood assumption
- Same A=1, λ₀=0 simplification (in main code these are learnable)

The playground does NOT use:
- Arc-cosine kernel (uses RBF instead)
- Image inputs (uses 1D scalars)
- Complex kernel hyperparameters (β, ρ, ξ₀)

**Note**: The main codebase has `conditioned_utility_clean()` which implements distribution-aware utility for the custom GP model. The playground's `distribution_aware_utility_gpytorch()` is the GPyTorch equivalent.

## Session Notes

- Created 2024-12-16
- Environment: `pytorch_gpytorch` (cloned from `pytorch_py312` + gpytorch)
- The playground successfully validated the intuition about even spacing
- Added active learning loop demonstrating iterative acquisition
- Observed boundary effects (x=0, x=1 get selected repeatedly)
- Added asymmetric ground truth function option for non-periodic testing
- **2024-12-18**: Added `distribution_aware_utility_gpytorch()` - a p(x)-aware acquisition function that conditions on samples from the stimulus distribution. Supports uniform and Gaussian p(x). Test script: `test_distribution_aware_utility.py`
- **2024-12-22**: Comprehensive utility method comparison and analysis:
  - Added `nd_utility_hybrid()` — Laplace for H_marg, MC for H_cond
  - Added `test_nd_utility_MC.py` — compares all utility computation methods
  - Added `test_dirac_delta_collapse.py` — verifies distribution_aware collapses correctly
  - **Discovered critical bug in `nd_utility_MC_batched`**: Internal inconsistency between analytical term1 and MC-truncated E_log_r_fact causes 45-50% errors and impossible negative utilities
  - Documented all utility methods with test results and recommendations

## How to Run

**Single-shot playground:**
```bash
conda activate pytorch_gpytorch
cd /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/1D_playground
python gp_utility_playground.py
```

**Active learning loop:**
```bash
python active_learning_loop.py
```

**Utility method comparison (Laplace vs MC vs Numerical vs Hybrid):**
```bash
python test_nd_utility_MC.py
```

**Dirac delta collapse test:**
```bash
python test_dirac_delta_collapse.py
```

**Distribution-aware utility comparison:**
```bash
python test_distribution_aware_utility.py
```
This script compares the original `nd_utility` with `distribution_aware_utility_gpytorch` using both uniform and Gaussian p(x). Outputs a 2-panel figure showing GP fit quality and utility comparison.
