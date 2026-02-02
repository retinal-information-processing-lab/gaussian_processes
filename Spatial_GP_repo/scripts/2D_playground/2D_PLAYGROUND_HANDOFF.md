# 2D GP Utility Playground - Handoff Document

**Last Updated**: 2026-01-28
**Purpose**: Complete context for a new Claude Code session working on the 2D GP utility playground.

---

## 1. Quick Start

```bash
# Navigate to 2D playground
cd /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/2D_playground

# Train RBF model (creates checkpoint)
python train_rbf_2d.py

# Visualize RBF results (loads checkpoint)
python utility_2d_rbf_base.py

# Train Arc-Cosine model (creates checkpoint)
cd arccosine
python train_acos_2d.py

# Compare Arc-Cosine vs RBF (loads both checkpoints)
python utility_acos_2d_base.py

# Diagnostic: before/after conditioning analysis (2D version of 1D plot)
cd ..
python diagnose_rbf_2d_conditioning.py
```

---

## 2. File Structure

```
scripts/2D_playground/
├── utility_2d_rbf_base.py         # Config + visualization (loads checkpoint)
├── train_rbf_2d.py                     # RBF training → checkpoint
├── trained_rbf_2d_checkpoint.pt        # RBF model checkpoint
├── utility_2d_rbf_base.png # RBF visualization output
├── diagnose_rbf_2d_conditioning.py # Diagnostic: before/after conditioning analysis
├── diagnose_rbf_2d_conditioning.png # Diagnostic output (3-panel heatmap)
├── 2D_PLAYGROUND_HANDOFF.md            # THIS FILE
├── 2D_PLAYGROUND_CONTEXT.md            # Original context (outdated)
│
└── arccosine/
    ├── utility_acos_2d_base.py      # Comparison script (loads both checkpoints)
    ├── train_acos_2d.py              # Arc-Cosine training → checkpoint
    ├── trained_arccosine_2d_checkpoint.pt # Arc-Cosine model checkpoint
    ├── diagnose_acos_2d_utility.py   # Diagnostic script
    ├── kernel_comparison_2d.png           # Comparison visualization output
    └── README.md
```

---

## 3. How the Code Works

### Training Scripts (run once to create checkpoints)

**`train_rbf_2d.py`** - Trains RBF GP on 5×5 grid, saves checkpoint:
- Imports config from `utility_2d_rbf_base.py`
- Uses `train_gp()` from 1D playground
- Saves `trained_rbf_2d_checkpoint.pt`
- Expected ELBO: ~118

**`arccosine/train_acos_2d.py`** - Trains Arc-Cosine GP:
- Uses `ArcCosineKernel` and `PoissonLikelihood` from `gpytorch_porting/`
- Calls `model.covar_module.clamp_hyperparameters()` after each optimizer step
- Saves `trained_arccosine_2d_checkpoint.pt`
- Expected ELBO: Very negative (~-860,000) due to kernel non-stationarity

### Visualization Scripts (load checkpoints, generate plots)

**`utility_2d_rbf_base.py`** - RBF-only visualization:
- Loads RBF checkpoint via `load_rbf_2d_checkpoint()`
- Evaluates standard utility and distribution-aware utility
- Generates 2×3 plot (latent function, standard utility, distribution-aware utility)
- **Also serves as config reference** - exports constants used by other scripts

**`arccosine/utility_acos_2d_base.py`** - Arc-Cosine vs RBF comparison:
- Loads BOTH checkpoints (RBF from parent, Arc-Cosine from current folder)
- Evaluates utilities for both kernels
- Clips Arc-Cosine utilities to 99th percentile (handles extreme values at corners)
- Generates 2×3 comparison plot (Arc-Cosine row, RBF row)

### Diagnostic Scripts

**`diagnose_rbf_2d_conditioning.py`** - Before/after conditioning analysis:
- 2D version of `1D_playground/diagnose_1d_utility_w_fixed_ntrain.py`
- Loads RBF checkpoint, samples x ~ p(x), computes MC-averaged statistics
- Generates 3-panel heatmap:
  1. **Δμ** = E[μ_cond] - μ_marg (mean shift after conditioning)
  2. **Δσ²** = σ²_marg - E[σ²_cond] (variance reduction after conditioning)
  3. **Utility** = H_marg - E[H_cond] (information gain)
- Key insight: All metrics peak at p(x) center, confirming distribution-aware utility works correctly

---

## 4. Import Hierarchy

```
1D_playground/gp_utility_playground.py (universal basics)
    ├── DEVICE, DTYPE, MAX_R
    ├── VariationalGP, PoissonLikelihood (simple version)
    ├── train_gp(), compute_elbo(), evaluate_nd_utility_new()
    └── generate_poisson_data()

2D_playground/utility_2d_rbf_base.py (2D config + functions)
    ├── X_MIN, X_MAX, Y_MIN, Y_MAX
    ├── DEFAULT_P_X_MEAN_2D, DEFAULT_P_X_STD_2D
    ├── N_TRAIN_X_DEFAULT, N_TRAIN_Y_DEFAULT, etc.
    ├── lambda_true_2d(), create_2d_grid()
    ├── evaluate_distribution_aware_utility_2d()
    ├── get_conditional_moments_nd()
    └── load_rbf_2d_checkpoint(), save_rbf_2d_checkpoint()

gpytorch_porting/ (Arc-Cosine specific)
    ├── kernels.py → ArcCosineKernel (with clamp_hyperparameters())
    └── likelihoods.py → PoissonLikelihood (with A, lambda0 parameters)
```

**Important**: There are TWO different `PoissonLikelihood` classes:
- `1D_playground` version: Simple (no A, no lambda0) - used by RBF
- `gpytorch_porting` version: Full (with A, lambda0) - used by Arc-Cosine

---

## 5. Checkpoint Format

### RBF Checkpoint (`trained_rbf_2d_checkpoint.pt`)
```python
{
    'model_state_dict': ...,
    'likelihood_state_dict': ...,
    'inducing_points': tensor (25, 2),
    'train_x': tensor (25, 2),
    'train_y': tensor (25,),
    'hyperparameters': {
        'lengthscale': 2.889,
        'outputscale': 1.220
    },
    'training_config': {...}
}
```

### Arc-Cosine Checkpoint (`trained_arccosine_2d_checkpoint.pt`)
```python
{
    'model_state_dict': ...,
    'likelihood_state_dict': ...,
    'inducing_points': tensor (15, 2),  # Subset of training
    'train_x': tensor (25, 2),
    'train_y': tensor (25,),
    'kernel_params': {
        'sigma_0': 0.678,
        'Amp': 1.0
    },
    'likelihood_params': {
        'A': 0.01,
        'lambda_0': 1.0
    },
    'training_config': {...}
}
```

**Critical**: When loading Arc-Cosine checkpoint, kernel must be swapped BEFORE `load_state_dict()`:
```python
model = VariationalGP(inducing_points, jitter=1e-4)
model.covar_module = ArcCosineKernel(sigma_0=..., Amp=..., C=None)  # SWAP FIRST
model.load_state_dict(checkpoint['model_state_dict'])  # THEN LOAD
```

---

## 6. Key Results

### RBF Kernel (ELBO ~118, good fit)
- Standard utility peaks at edges (far from p(x) center)
- Distribution-aware utility peaks near p(x) center (as expected)
- Confirms hypothesis: distribution-aware utility respects p(x)

### Arc-Cosine Kernel (ELBO ~-860,000, poor fit)
- k(x,x) = ||x||² + σ₀² is NON-STATIONARY
- At (0,0): k ≈ 0.46
- At (5,5): k ≈ 50.46 (100× difference!)
- This causes extreme utility values at corners (up to millions)
- Visualization uses 99th percentile clipping to show meaningful structure

---

## 7. Mathematical References

- Distribution-aware utility: `/home/idv-eqs8-pza/IDV_code/Papers/latex_summaries/distribution_aware_utility_pietro.tex`
- Arc-Cosine kernel: `/home/idv-eqs8-pza/IDV_code/Papers/latex_summaries/acosker_kernel_def_and_gradients.tex`
- Conditional GP moments: `/home/idv-eqs8-pza/IDV_code/Papers/latex_summaries/predictive_distribution_conditioned_on_observation.tex`
- 1D playground context: `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/1D_playground/.claude/CLAUDE.md`

---

## 8. Files to Read for Full Context

**Start here:**
1. This file: `2D_playground/2D_PLAYGROUND_HANDOFF.md`
2. `2D_playground/utility_2d_rbf_base.py` - Main config + checkpoint functions
3. `2D_playground/arccosine/utility_acos_2d_base.py` - Comparison script

**For deeper understanding:**
4. `1D_playground/gp_utility_playground.py` - Core GP implementation
5. `1D_playground/.claude/CLAUDE.md` - Full mathematical derivation
6. `gpytorch_porting/kernels.py` - Arc-Cosine kernel implementation

---

## 9. Common Tasks

### Retrain models with different settings
1. Edit config in `utility_2d_rbf_base.py` (e.g., `N_TRAIN_X_DEFAULT`)
2. Run `train_rbf_2d.py` and `arccosine/train_acos_2d.py`
3. Run visualization scripts to see updated results

### Change p(x) distribution
Edit in `utility_2d_rbf_base.py`:
```python
DEFAULT_P_X_MEAN_2D = torch.tensor([0.0, 0.0], ...)
DEFAULT_P_X_STD_2D = torch.tensor([2.0, 2.0], ...)
```

### Change visualization domain
Local overrides in each script:
```python
X_MIN, X_MAX = -5.0, 5.0
Y_MIN, Y_MAX = -5.0, 5.0
```

---

## 10. Quick Reference: Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `load_rbf_2d_checkpoint()` | utility_2d_rbf_base.py | Load RBF model from checkpoint |
| `load_arccosine_2d_checkpoint()` | arccosine/utility_acos_2d_base.py | Load Arc-Cosine (with kernel swap) |
| `evaluate_distribution_aware_utility_2d()` | utility_2d_rbf_base.py | Compute U(x*) = H_marg - E[H_cond] |
| `get_conditional_moments_nd()` | utility_2d_rbf_base.py | GP conditioning for 2D inputs |
| `compute_mc_diagnostics_2d()` | diagnose_rbf_2d_conditioning.py | MC-averaged Δμ, Δσ², utility |
| `evaluate_nd_utility_new()` | 1D_playground/gp_utility_playground.py | Standard utility (no p(x)) |
| `train_gp()` | 1D_playground/gp_utility_playground.py | Adam training loop |
| `clamp_hyperparameters()` | gpytorch_porting/kernels.py | Arc-Cosine constraint enforcement |
