# Arc-Cosine Kernel 2D Playground

Created: 2026-01-26

## Purpose

Compare the **Arc-Cosine kernel** (non-stationary, from infinite-width ReLU neural networks) with the **RBF kernel** (stationary) on a 2D GP playground with Poisson likelihood.

Extension of the 1D arccosine playground to 2D inputs to better demonstrate non-stationarity effects.

## Key Differences from 1D

- **Input space**: 2D (x, y) instead of 1D
- **Visualization**: Contour plots instead of line plots
- **Non-stationarity more visible**: k(x,x) = ||x||² + σ₀² varies across 2D domain
- **Distribution-aware utility**: More interesting behavior in 2D (radial decay from p(x) center)

## Session Summary

### Goal
Implement a 2D playground script that:
1. Uses the same utilities/training as `utility_2d_base.py`
2. Swaps RBF kernel for Arc-Cosine kernel from `gpytorch_porting/kernels.py`
3. Minimizes code duplication by importing from existing scripts
4. Compares both kernels side-by-side

### Key Design Decisions

**Import Strategy**:
- Import from `utility_2d_base.py`: `VariationalGP`, `PoissonLikelihood`, `compute_elbo`, utility functions, 2D data generation
- Import from `gpytorch_porting/kernels.py`: `ArcCosineKernel` (with strict path validation)
- Reuse `get_conditional_moments_nd()` wrapper for dimension-agnostic conditioning
- **No new GP model class needed** - just swap `model.covar_module` after creating `VariationalGP`

**Arc-Cosine Kernel Configuration**:
- `C=None` → Stage 1 (identity matrix, appropriate for generic 2D inputs)
- Parameters: `sigma_0` (bias variance), `Amp` (amplitude scaling)
- Non-stationary: `k(x,x) = ||x||² + σ₀²` (varies with distance from origin!)
- Hyperparameter clamping: `Amp` clamped at max 1000 via `clamp_hyperparameters()`

## Files

### `gp_arccosine_playground_2d.py` (~400 lines)

**Main script** that trains both kernels and creates 2x2 comparison plot.

**Structure**:
```
Row 1: Arc-Cosine Kernel
  ├── Subplot 1: Latent function λ(x,y) with training points
  └── Subplot 2: Utility landscape (standard + distribution-aware)

Row 2: RBF Kernel
  ├── Subplot 3: Latent function λ(x,y) with training points
  └── Subplot 4: Utility landscape (standard + distribution-aware)
```

**Key Components**:
- `train_arccosine_gp()`: Trains Arc-Cosine, prints σ₀/Amp, calls `clamp_hyperparameters()`
- `train_rbf_gp()`: Trains RBF, prints lengthscale/outputscale
- `check_kernel_health()`: Monitors condition number (warns if > 1e12)
- `plot_comparison()`: Creates 2x2 figure with both kernels

**Imports**:
```python
from gp_utility_playground_2d import (
    VariationalGP, PoissonLikelihood, compute_elbo,
    lambda_true_2d, create_2d_grid, generate_poisson_data,
    evaluate_nd_utility_new, evaluate_distribution_aware_utility_2d,
    get_conditional_moments_nd,
    DEVICE, DTYPE, X_MIN, X_MAX, ...
)
from kernels import ArcCosineKernel  # from gpytorch_porting/
```

**Usage**:
```bash
cd /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/2D_playground/arccosine
python gp_arccosine_playground_2d.py
```

## Expected Results

The 2D setting should make the non-stationarity more visible:
- **Arc-Cosine k(x,x)**: Increases with distance from origin (parabolic)
- **RBF k(x,x)**: Constant across domain
- **Distribution-aware utility**: Should show radial structure in 2D
- **Kernel conditioning**: Arc-Cosine may have higher condition numbers

## Output Files

- `kernel_comparison_2d.png` - Main 2x2 comparison plot with contours

## Code Reuse Philosophy

**Why no separate `ArcCosineGP` class?**

Following the pattern from 1D playground and `gpytorch_porting/`:
```python
model = VariationalGP(inducing_points, jitter=1e-4)  # from playground
model.covar_module = ArcCosineKernel(...)            # swap kernel
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

1. **Stage 2 Arc-Cosine**: Use structured C matrix with RF parameters (requires actual image inputs, not synthetic 2D)
2. **Different kernels**: Import other kernels (Matérn, Spectral Mixture) and add to comparison
3. **Active learning loop**: Iterate training + utility-based acquisition in 2D
4. **Higher dimensions**: Test on 3D or actual image data

### To debug:

- Check `[Kernel Health]` prints for condition number warnings
- Compare ELBO trajectories between kernels
- Verify learned hyperparameters make sense for the data scale
- Compare 2D results with 1D to see if non-stationarity effects are more pronounced

### Dependencies:

- Requires `utility_2d_base.py` in parent directory
- Requires `gpytorch_porting/kernels.py` with `ArcCosineKernel` class
- Uses `get_conditional_moments_nd()` wrapper for 2D conditioning

## References

- Arc-cosine kernel math: `~/IDV_code/Papers/latex_summaries/acosker_kernel_def_and_gradients.tex`
- Distribution-aware utility: `~/IDV_code/Papers/latex_summaries/distribution_aware_utility_pietro.tex`
- 2D playground context: `../2D_PLAYGROUND_CONTEXT.md`
- 1D arccosine implementation: `../../1D_playground/arccosine/README.md`
- gpytorch_porting implementation: `../../gpytorch_porting/kernels.py`
