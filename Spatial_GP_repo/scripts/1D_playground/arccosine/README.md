# Arc-Cosine Kernel 1D Playground

Created: 2026-01-26

## Purpose

Compare the **Arc-Cosine kernel** (non-stationary, from infinite-width ReLU neural networks) with the **RBF kernel** (stationary) on a 1D GP playground with Poisson likelihood.

## Session Summary

### Goal
Implement a 1D playground script that:
1. Uses the same data/utilities as `gp_utility_playground.py`
2. Swaps the RBF kernel for Arc-Cosine kernel from `gpytorch_porting/kernels.py`
3. Minimizes code duplication by importing from existing scripts
4. Compares both kernels side-by-side

### Key Design Decisions

**Import Strategy**:
- Import from `gp_utility_playground.py`: `VariationalGP`, `PoissonLikelihood`, `compute_elbo`, utility functions, data generation
- Import from `gpytorch_porting/kernels.py`: `ArcCosineKernel` (with strict path validation)
- **No new GP model class needed** - just swap `model.covar_module` after creating `VariationalGP`

**Arc-Cosine Kernel Configuration**:
- `C=None` → Stage 1 (identity matrix, appropriate for 1D scalar inputs)
- Parameters: `sigma_0` (bias variance), `Amp` (amplitude scaling)
- Non-stationary: `k(x,x) = x² + σ₀²` (varies with input!)
- Hyperparameter clamping: `Amp` clamped at max 1000 via `clamp_hyperparameters()`

## Files

### `gp_arccosine_playground.py` (~386 lines)

**Main script** that trains both kernels and creates 2x2 comparison plot.

**Structure**:
```
Row 1: Arc-Cosine Kernel
  ├── Subplot 1: Latent function λ(x) with k(x,x) curve (purple, secondary axis)
  └── Subplot 2: Utility landscape (standard + distribution-aware)

Row 2: RBF Kernel
  ├── Subplot 3: Latent function λ(x) with k(x,x) curve (flat line)
  └── Subplot 4: Utility landscape (standard + distribution-aware)
```

**Key Components**:
- `train_arccosine_gp()`: Trains Arc-Cosine, prints σ₀/Amp, calls `clamp_hyperparameters()`
- `train_rbf_gp()`: Trains RBF, prints lengthscale/outputscale
- `check_kernel_health()`: Monitors condition number (warns if > 1e12)
- `plot_comparison()`: Creates 2x2 figure with both kernels

**Imports**:
```python
from gp_utility_playground import (
    VariationalGP, PoissonLikelihood, compute_elbo,
    lambda_true, generate_poisson_data,
    evaluate_nd_utility_new, evaluate_distribution_aware_utility,
    DEVICE, DTYPE, X_MIN, X_MAX, ...
)
from kernels import ArcCosineKernel  # from gpytorch_porting/
```

**Usage**:
```bash
cd /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/1D_playground/arccosine
python gp_arccosine_playground.py
```

## Results (Seed 42)

| Metric | Arc-Cosine | RBF |
|--------|-----------|-----|
| Final ELBO | 37.36 | **61.44** (better) |
| Condition # | 3.13e+06 | **80** (much better) |
| σ₀/ℓ | 0.487 | 0.216 |
| Amp/σ² | 1.000 | 1.155 |
| Mean | 2.327 | 0.542 |
| Utility max | x=0.18 | x=0.42 |

**Observation**: RBF achieves higher ELBO and dramatically better conditioning on this 1D problem. Arc-Cosine kernel's non-stationarity may be more useful in higher-dimensional or structured input spaces (images).

## Output Files

- `kernel_comparison.png` (307K) - Main 2x2 comparison plot
- `gp_arccosine_playground_result.png` (285K) - Old single-kernel plot (deprecated)

## Code Reuse Philosophy

**Why no separate `ArcCosineGP` class?**

Initial attempt created a custom class, but this was unnecessary. The `gpytorch_porting/` folder shows the pattern:
```python
# run_single_mode.py approach
kernel = ArcCosineKernel(sigma_0=1.0, Amp=1.0, C=None)
model = VariationalGPModel(inducing_points, kernel, jitter=0, ...)
```

We adapted this by:
```python
model = VariationalGP(inducing_points, jitter=0)  # from playground
model.covar_module = ArcCosineKernel(...)         # swap kernel
```

This minimizes duplication and ensures compatibility with all existing utility functions.

## Path Validation

The script includes **strict path checking** to ensure `ArcCosineKernel` is imported from the correct location:

```python
EXPECTED_KERNEL_PATH = '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting'
```

Raises `ImportError` if the kernel is imported from anywhere else.

## For Future Sessions

### To extend this work:

1. **Different kernels**: Import other kernels (Matérn, Spectral Mixture) and add to comparison
2. **Active learning loop**: Iterate training + utility-based acquisition
3. **2D inputs**: Test on synthetic 2D data to better showcase non-stationarity
4. **Stage 2 Arc-Cosine**: Use structured C matrix (requires 2D+ inputs and RF parameters)

### To debug:

- Check `[Kernel Health]` prints for condition number warnings
- Compare ELBO trajectories between kernels
- Verify learned hyperparameters make sense for the data scale

### Dependencies:

- Requires `gp_utility_playground.py` in parent directory
- Requires `gpytorch_porting/kernels.py` with `ArcCosineKernel` class
- Assumes `utility.py` has `nd_utility_new()`, `evaluate_distribution_aware_utility()`

## References (EXACT PATHS)

### Math Derivations
- **Arc-Cosine kernel**: `/home/idv-eqs8-pza/IDV_code/Papers/latex_summaries/acosker_kernel_def_and_gradients.tex`
- **Distribution-aware utility**: `/home/idv-eqs8-pza/IDV_code/Papers/latex_summaries/distribution_aware_utility_pietro.tex`

### Code References
- 1D playground context: `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/1D_playground/.claude/CLAUDE.md`
- gpytorch_porting kernel implementation: `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting/kernels.py`
- Original playground (RBF): `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/1D_playground/gp_utility_playground.py`


# FINAL PIETRO NOTES:

One weird thing was that when running the diagnosis scripts to see how much conditioning changed things we observed very different mean and variances for the predictive lambdas distributions EVEN FAR FROM THE OBSERVED/CONDITIONED POINT. this is due to the kernel itself, with therbf kernel we would have returned to the mean with constant variance if far away.

I was therefore expecting the Utility to explode at borders , since the variance was so different. But i still observed the utility going to zero, even  close to  p(x) if this was was put far enough from the parts where the predicted mean absolute value is low.

WHY? cause the entropy H calculated with laplace goes to zero when predicted mean lambda grows in ABSOLUTE value. check the heatmap for values of lambda mean above or below 10 , -10 and youll se H is zero. so even if U=H(mu,sigma2)-H(mu_cond,sigma2_cond) should be different from zero, they are both zero.m
mm