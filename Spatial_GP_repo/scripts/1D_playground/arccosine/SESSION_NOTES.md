# Session Notes: Arc-Cosine Kernel Implementation
**Date**: 2026-01-26
**Session ID**: multiple_bugs_batch3 branch

## Difference from README.md

**README.md** = Reference manual for future users
- What the code does
- How to use it
- Results and outputs
- Technical specifications

**SESSION_NOTES.md** = Conversational record for future Claude sessions
- What we tried and why
- Design decisions and iterations
- Things that didn't work
- Context that led to the final solution

---

## Initial Request

User wanted to test the Arc-Cosine kernel (from `gpytorch_porting/kernels.py`) in the 1D playground, comparing it against RBF.

**Key constraint**: "Change only the kernel and reuse as many code parts from the original `gp_utility_playground.py` as possible. This reduces potential for bugs."

---

## Understanding Phase (Read & Ask Questions)

### Files Read:
1. `/home/idv-eqs8-pza/IDV_code/Papers/latex_summaries/distribution_aware_utility_pietro.tex` - Math for utility functions
2. `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/1D_playground/gp_utility_playground.py` - The RBF baseline script
3. `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting/kernels.py` - The ArcCosineKernel implementation
4. `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting/model.py` - How VariationalGPModel works
5. `/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting/run_single_mode.py` - Usage pattern

### Math References (EXACT PATHS - READ THESE FOR UNDERSTANDING):
- **Arc-Cosine kernel definition**: `/home/idv-eqs8-pza/IDV_code/Papers/latex_summaries/acosker_kernel_def_and_gradients.tex`
- **Distribution-aware utility**: `/home/idv-eqs8-pza/IDV_code/Papers/latex_summaries/distribution_aware_utility_pietro.tex`

### Key Questions Asked:
1. **What kernel?** → Arc-Cosine kernel (C=I, Stage 1)
2. **1D or high-dim?** → 1D (GPyTorch, not custom varGP)
3. **Eigenspace projections?** → Out of scope for 1D
4. **Can kernel integrate?** → Yes, already GPyTorch-compatible

### Mathematical Understanding:
- Arc-Cosine: `K(x,x') = (1/π) · √(v_x·v_x') · J(θ)` where `v_x = x²·C·x + σ₀²`
- For C=I (Stage 1): `v_x = x² + σ₀²` → **non-stationary**, self-kernel varies with x
- RBF: `K(x,x') = σ² · exp(-||x-x'||²/(2ℓ²))` → **stationary**, self-kernel constant

---

## Implementation Evolution

### Attempt 1: Create New ArcCosineGP Class ❌
**What we did**: Created a custom `ArcCosineGP` class similar to `VariationalGP`

**Why it was wrong**: Unnecessary duplication. User pointed out: *"Why do we need a new ArcCosineGP class, there was none in the gpytorch_porting folder?"*

**Key insight**: Look at `run_single_mode.py` - it just passes the kernel to `VariationalGPModel`:
```python
kernel = ArcCosineKernel(sigma_0=1.0, Amp=1.0, C=None)
model = VariationalGPModel(inducing_points, kernel, ...)
```

### Attempt 2: Use VariationalGPModel from gpytorch_porting ❌
**What we did**: Imported `VariationalGPModel` from `gpytorch_porting/model.py`

**Why it failed**: `VariationalGPModel` uses `ZeroMean`, but the playground needs `ConstantMean` for better fit. Crashed with:
```
AttributeError: 'ZeroMean' object has no attribute 'constant'
```

**Results were different**: ELBO = 33.37 vs 37.36 expected

### Attempt 3: Import VariationalGP + Swap Kernel ✓
**Final solution**:
```python
from gp_utility_playground import VariationalGP, PoissonLikelihood, compute_elbo, ...

model = VariationalGP(inducing_points, jitter=0)  # Has ConstantMean + RBF by default
model.covar_module = ArcCosineKernel(sigma_0=1.0, Amp=1.0, C=None)  # Swap kernel
```

**Why it works**:
- Minimal change (2 lines in main)
- Reuses ALL functions from playground
- No new classes needed
- Results match expectations

---

## Key Design Decisions

### 1. Import Strategy
**Decision**: Import from `gp_utility_playground.py` NOT duplicate
- `VariationalGP`, `PoissonLikelihood`, `compute_elbo`
- `lambda_true`, `generate_poisson_data`
- `evaluate_nd_utility_new`, `evaluate_distribution_aware_utility`
- All config constants

**Rationale**: User emphasized "reuse to reduce bugs"

### 2. Kernel Path Validation
**Decision**: Strict path checking for `ArcCosineKernel` import

```python
EXPECTED_KERNEL_PATH = '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting'
```

**Rationale**: User insisted: *"make sure you are importing the right kernel. throw error if you are importing kernel from outside of [this path]"*

### 3. Condition Number Monitoring
**Decision**: Add `check_kernel_health()` to track condition numbers

**Rationale**: Arc-Cosine is non-stationary and might have numerical issues. Need to monitor.

**Result**: Arc-Cosine has cond ≈ 3e6, RBF has cond ≈ 80. Both acceptable but RBF is much better conditioned.

### 4. Hyperparameter Clamping
**Decision**: Call `model.covar_module.clamp_hyperparameters()` after each optimizer step

**Rationale**: Arc-Cosine kernel has `Amp` clamped at max 1000. Must enforce this during training.

---

## Evolution to Comparison Script

### User Request:
*"I want to compare the arccosine and the rbf kernels. Remove the firing rate/spike count subplot. First 2 subplots should be latent function lambda + K and the utility landscape. Subplot 3 and 4 should be the same, but with the rbf kernel."*

### Layout Change:
- **Before**: 3 subplots (latent, firing rate, utility) for one kernel
- **After**: 2×2 grid comparing both kernels

```
┌─────────────────┬─────────────────┐
│ Arc-Cosine λ(x) │ Arc-Cosine Util │  ← Row 1
│  + k(x,x) curve │                 │
├─────────────────┼─────────────────┤
│ RBF λ(x)        │ RBF Util        │  ← Row 2
│  + k(x,x) curve │                 │
└─────────────────┴─────────────────┘
```

### Key Visualization Features:
- k(x,x) on secondary y-axis (purple dotted line)
  - Arc-Cosine: parabola (x² + σ₀²)
  - RBF: flat line (outputscale)
- Both standard (green) and distribution-aware (blue) utilities
- Condition numbers in subplot titles
- Hyperparameters displayed

---

## What We Learned

### 1. Arc-Cosine on 1D is Interesting but Not Better
- **ELBO**: RBF wins (61.44 vs 37.36)
- **Conditioning**: RBF much better (80 vs 3e6)
- **Utility**: Different peaks, neither clearly superior

**Hypothesis**: Arc-Cosine's non-stationarity might shine in higher dimensions or structured inputs (images), not simple 1D.

### 2. Code Reuse is King
- Started with ~700 lines of duplicated code
- Ended with ~386 lines, importing most functions
- Fewer bugs, easier maintenance

### 3. Kernel Swapping Pattern
The pattern `model.covar_module = NewKernel(...)` is powerful:
- Works with any GPyTorch kernel
- No need for new model classes
- Easy to experiment

---

## For Next Session

### If Continuing This Work:

1. **Try 2D synthetic data**: Arc-Cosine non-stationarity might be more visible
2. **Test other kernels**: Matérn, Spectral Mixture, etc.
3. **Active learning loop**: Iterate training + utility-based acquisition
4. **Stage 2 Arc-Cosine**: Use structured C matrix (needs image inputs)

### If Debugging:

**Common issues**:
- Wrong kernel path → Check import validation prints
- Wrong ELBO → Compare hyperparameters with RBF baseline
- Ill-conditioned kernel → Check `[Kernel Health]` warnings

**Expected outputs**:
- Arc-Cosine: ELBO ≈ 37, σ₀ ≈ 0.49, Amp ≈ 1.0, cond ≈ 3e6
- RBF: ELBO ≈ 61, ℓ ≈ 0.22, σ² ≈ 1.2, cond ≈ 80

### Code Structure:
```
arccosine/
├── gp_arccosine_playground.py  # Main comparison script
├── kernel_comparison.png       # 2×2 output plot
├── README.md                   # Reference manual
└── SESSION_NOTES.md           # This file (conversation history)
```

---

## Files Modified/Created

### Created:
- `arccosine/gp_arccosine_playground.py` (386 lines)
- `arccosine/kernel_comparison.png` (307K)
- `arccosine/README.md`
- `arccosine/SESSION_NOTES.md`

### Not Modified:
- `gp_utility_playground.py` (source of imports)
- `gpytorch_porting/kernels.py` (source of ArcCosineKernel)

---

## Timeline

1. **Understanding phase** (15 min) - Read math docs, existing code
2. **First attempt** (10 min) - Custom ArcCosineGP class → rejected
3. **Second attempt** (10 min) - Use VariationalGPModel → wrong mean module
4. **Third attempt** (5 min) - Kernel swap pattern → success!
5. **Add comparison** (15 min) - 2×2 layout with both kernels
6. **Documentation** (10 min) - README + SESSION_NOTES

**Total**: ~65 minutes from concept to working comparison

---

## Key Quotes from User

*"This is ok but the goal of this script is to change only the kernel and reuse as many code parts from the original gp_utility_playground as possible. This reduces potential for bugs."*

*"Make sure you are importing/reusing stuff like the PoissonLikelihood."*

*"Why do we need a new ArcCosineGP class, there was none in the gpytorch_porting folder?"*

These guided us toward the minimal-change solution.

---

## Session 2: No-Training + Checkpoint System (2026-01-27)

### User Request:
*"Remove the model training. Fix the hyperparams independently of the domain. Use arange not linspace so widening domain adds points, doesn't change them."*

### Key Problem Discovered:

**Setting hyperparameters ≠ having a fit!**

A variational GP has TWO sets of parameters:
1. **Hyperparameters**: lengthscale, outputscale, mean (kernel/prior structure)
2. **Variational parameters**: m (mean vector), V (covariance matrix) - encode the actual fit

Initially I only set hyperparameters, leaving variational params at defaults. This gives a flat prior, not a fit. User caught this: *"The fit is shit. What is different?"*

### Solution: Checkpoint System

1. **Train once** in `gp_utility_playground.py`, save FULL model state
2. **Load checkpoint** in `gp_arccosine_playground.py` with explicit metadata

**Checkpoint format** (`trained_rbf_checkpoint.pt`):
```python
{
    'model_state_dict': ...,  # Full state including variational params
    'config': {
        'seed': 42,
        'n_train': 20,
        'n_iterations': 500,
        'x_min': -2.0, 'x_max': 2.0,
        'ground_truth': 'lambda_true_asymmetric',
    },
    'inducing_points': tensor([...]),  # Critical: must match when loading
    'hyperparameters': {
        'lengthscale': 0.1945,
        'outputscale': 0.7107,
        'mean_constant': 0.8938,
    },
    'created': '2026-01-27T...',
    'description': 'RBF GP trained on 1D asymmetric bump, Poisson likelihood',
}
```

### What the script now does:

1. **RBF**: Loads trained model (hyperparams + variational params) → shows actual fit
2. **Arc-Cosine**: Uses same inducing points and prior mean as RBF, but default variational params → shows prior

This is NOT a "which kernel fits better" comparison. It shows:
- How RBF posterior looks after training
- How Arc-Cosine prior looks with same mean
- How utility differs between trained posterior vs untrained prior

### Key Design Decisions:

1. **Same inducing points**: Arc-Cosine uses exact same inducing points as RBF (loaded from checkpoint)
2. **Same prior mean**: Arc-Cosine prior mean = RBF trained mean (0.8938)
3. **Default variational params for Arc-Cosine**: m=0, V=K (prior)
4. **Self-documenting checkpoint**: Loading prints all metadata so future sessions understand what's loaded

### Things to investigate (deferred):

1. **Arc-Cosine condition number ~3e6**: High due to k(x,x) = x² + σ₀². Could cause numerical issues for large domains.

2. **Arc-Cosine σ₀=0.5 is arbitrary**: Not trained. For fair kernel comparison, would need to train Arc-Cosine too.

3. **Domain extrapolation**: RBF trained on [-2,2] but script can evaluate on wider domains (currently [-15,15]). Outside training domain, posterior reverts to prior.

4. **Different PoissonLikelihood implementations**:
   - This script: simple `y ~ Poisson(exp(f))`
   - gpytorch_porting: has A, λ₀ params `y ~ Poisson(exp(A*f + λ₀))`

### Files changed:
- `gp_arccosine_playground.py`: Removed training, added checkpoint loading
- `trained_rbf_checkpoint.pt`: New file with full model state + metadata

### To regenerate checkpoint:
```bash
cd 1D_playground/
python -c "
import torch, numpy as np
from datetime import datetime
torch.manual_seed(42); np.random.seed(42)

from gp_utility_playground import *
train_x = torch.linspace(X_MIN, X_MAX, 20, dtype=DTYPE, device=DEVICE)
train_y = generate_poisson_data(train_x, lambda_true)
model = VariationalGP(train_x.clone(), jitter=0).to(DEVICE)
likelihood = PoissonLikelihood().to(DEVICE)
model, _ = train_gp(model, likelihood, train_x, train_y, n_iterations=500)

torch.save({
    'model_state_dict': model.state_dict(),
    'config': {'seed': 42, 'n_train': 20, 'n_iterations': 500,
               'x_min': X_MIN, 'x_max': X_MAX, 'dtype': str(DTYPE),
               'ground_truth': 'lambda_true_asymmetric'},
    'inducing_points': train_x.cpu(),
    'hyperparameters': {
        'lengthscale': model.covar_module.base_kernel.lengthscale.item(),
        'outputscale': model.covar_module.outputscale.item(),
        'mean_constant': model.mean_module.constant.item()},
    'created': datetime.now().isoformat(),
    'description': 'RBF GP trained on 1D asymmetric bump, Poisson likelihood',
}, 'arccosine/trained_rbf_checkpoint.pt')
"
```

### Confirmed: RBF mean IS trained

User asked: *"Is the mean the prior or posterior?"* and *"Confirm that the mean for RBF has been trained."*

Verified in `gp_utility_playground.py`:
- Line 147: `self.mean_module = gpytorch.means.ConstantMean()` (learnable)
- Line 193: `optimizer = torch.optim.Adam(model.parameters(), ...)` trains ALL params
- Mean goes from default (~0) to 0.8938 after training

---

## Session 3: Arc-Cosine Training Script (2026-01-28)

### User Request:
*"Write an acosker_training.py script to train and save an acosker model. Training parameters should be the same as RBF. Also investigate saving training points for plotting."*

### Created `acosker_training.py`

Training script that mirrors RBF training exactly:
- Same seed (42), domain ([-2, 2]), n_train (20), iterations (500)
- Saves checkpoint with full metadata + training data (train_x, train_y)

### Training Results Comparison

| Metric | RBF | Arc-Cosine |
|--------|-----|------------|
| ELBO | 47.37 | 37.36 |
| Condition | 3.12e+01 | 3.13e+06 |
| Mean | 0.8938 | 2.3274 |
| Lengthscale/σ₀ | 0.1945 | 0.4873 |
| Outputscale/Amp | 0.7107 | 1.0 (clamped) |

**Key observation**: Arc-Cosine has much higher mean (2.33 vs 0.89) and worse ELBO. The kernel's non-stationarity doesn't help on this simple 1D task.

### Checkpoint Format (both kernels)

```python
{
    'model_state_dict': ...,
    'config': {'seed': 42, 'n_train': 20, ...},
    'inducing_points': tensor([...]),
    'train_x': tensor([...]),  # NEW: for plotting
    'train_y': tensor([...]),  # NEW: for plotting
    'hyperparameters': {...},
    'condition_number': ...,
    'created': '...',
    'description': '...',
}
```

### Updated `gp_arccosine_playground.py`

Now loads BOTH trained models from checkpoints:
1. `trained_rbf_checkpoint.pt` - RBF trained model
2. `trained_acos_checkpoint.pt` - Arc-Cosine trained model

Training points are plotted as red dots (log(y) to match λ scale).

### Files in arccosine/

```
arccosine/
├── gp_arccosine_playground.py   # Main comparison (loads both checkpoints)
├── acosker_training.py          # Arc-Cosine training script
├── trained_rbf_checkpoint.pt    # RBF checkpoint with train data
├── trained_acos_checkpoint.pt   # Arc-Cosine checkpoint with train data
├── kernel_comparison.png        # Output plot
├── README.md
└── SESSION_NOTES.md
```

### To regenerate both checkpoints

```bash
# Activate environment
conda activate pytorch_gpytorch
cd 1D_playground/

# RBF (run from 1D_playground/)
python -c "
import torch, numpy as np
from datetime import datetime
torch.manual_seed(42); np.random.seed(42)
from gp_utility_playground import *

train_x = torch.linspace(X_MIN, X_MAX, 20, dtype=DTYPE, device=DEVICE)
train_y = generate_poisson_data(train_x, lambda_true)
model = VariationalGP(train_x.clone(), jitter=0).to(DEVICE)
likelihood = PoissonLikelihood().to(DEVICE)
model, _ = train_gp(model, likelihood, train_x, train_y, n_iterations=500)

with torch.no_grad():
    cond = torch.linalg.cond(model.covar_module(train_x).evaluate()).item()

torch.save({
    'model_state_dict': model.state_dict(),
    'config': {'seed': 42, 'n_train': 20, 'n_iterations': 500, 'lr': 0.1,
               'x_min': X_MIN, 'x_max': X_MAX, 'dtype': str(DTYPE),
               'ground_truth': 'lambda_true_asymmetric', 'kernel': 'RBF'},
    'inducing_points': train_x.cpu(),
    'train_x': train_x.cpu(),
    'train_y': train_y.cpu(),
    'hyperparameters': {
        'lengthscale': model.covar_module.base_kernel.lengthscale.item(),
        'outputscale': model.covar_module.outputscale.item(),
        'mean_constant': model.mean_module.constant.item()},
    'condition_number': cond,
    'created': datetime.now().isoformat(),
    'description': 'RBF GP trained on 1D asymmetric bump, Poisson likelihood',
}, 'arccosine/trained_rbf_checkpoint.pt')
"

# Arc-Cosine
python arccosine/acosker_training.py
```
