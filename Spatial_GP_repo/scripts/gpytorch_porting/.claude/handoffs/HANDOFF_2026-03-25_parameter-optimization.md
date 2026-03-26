# Handoff: Parameter Optimization to Match Paper Performance

**Created**: 2026-03-25
**Branch**: pietro/workingbranch
**Working dir**: `.../gpytorch_porting/`

---

## Goal

Systematically close the performance gap between our GP fits and Goldin et al. 2023 PNAS (the reference paper). The paper reports adjusted R² > 0.8 for 36/41 cells. Our best result is adjusted R² = 0.720 avg (13/41 cells > 0.8). The gap is real and significant.

**Primary metric**: Adjusted R² (Goldin et al. 2023 Eq. 5), implemented in `metrics.py:compute_adjusted_r_squared()`. This is a variance metric: `(mean_accuracy / sqrt(reliability))²`. It's already wired through `run_single_mode.py` and `run_experiment.py` JSONL output.

**Reference paper**: `papers/PNAS goldin 2023 scalable gaussian process inference of neural responses to natural images.pdf`. Read pages 2-4 carefully for their training procedure, kernel configuration, and reported numbers.

---

## Current Best Results (baseline to beat)

Best so far: **64x64, vargp_direct, M=2910, ground-truth RF init, 3 seeds averaged**:
- avg test_r = 0.823
- avg adjusted_r2 = 0.720
- Cells with adjusted_r2 > 0.8: 13/41

Results: `experiments/2026-03-24_ground_truth_init_64/results.jsonl`

The paper claims 36/41 cells > 0.8. We're at 13/41. That's the gap.

---

## What's Fixed (do NOT change)

- **Kernel type**: Arc-cosine kernel (`arc_cosine`). This is the paper's kernel.
- **GP architecture**: Variational GP with inducing points, Poisson likelihood with exp link.
- **Dataset**: PNAS retinal ganglion cell data, 41 cells, 2910 training images.
- **Test evaluation**: 30 test images x 30 repeats, even/odd split for adjusted R².
- **Training mode**: `vargp_direct` is our best mode (eigenspace projection). `default_gpy` consistently underperforms.

---

## What CAN Be Tuned (parameter space)

### Image resolution
- 108x108: full resolution, BUT 6 cells have STA edge artifact (use ground-truth RF init from `datasets/rf_centers_ground_truth.npz`)
- 64x64: center crop, avoids edge artifact, current best dataset
- 48x48: center crop, all RFs well within bounds, slightly worse than 64x64
- **Note**: beta=0.1 corresponds to different physical RF sizes on different image sizes (7.6px on 108, 4.5px on 64, 3.3px on 48). The paper likely uses a specific resolution — check.

### Kernel initialization
Current defaults in `default_params.json`:
```
beta = 0.1        (RF width, natural units → sigma_rf = beta * sqrt(2))
rho = 0.1         (smoothness)
Amp = 1.0          (amplitude)
sigma_0 = 1.0      (bias variance)
```
These may not match the paper. The paper might use different initial values or different parameterization.

### Link function
```
A_init = 0.01      (gain)
lambda0_init = 1.0  (bias, lambda0 is computed analytically during F-step)
```

### Training schedule
```
n_iterations = 50   (EM iterations)
n_estep = 10        (Newton steps per E-step)
n_fstep = 10        (LBFGS steps for A optimization)
n_mstep = 10        (LBFGS steps for kernel optimization)
lr = 0.1            (LBFGS learning rate for M-step and F-step)
```
The paper might use more iterations or a different schedule.

### Early stopping
```
enabled = true
window = 20
threshold = 0.005
min_iterations = 10
```
Early stopping might be cutting training short. Try disabling or loosening.

### Inducing points
```
M = 300 or 2910
selection = pivoted Cholesky
```
M=300 vs M=2910 gives marginal difference. The paper uses M inducing points — check what M they use.

### RF initialization
- STA-based (current default): noisy for low-firing cells
- Ground-truth from `datasets/rf_centers_ground_truth.npz`: fixes cells 6, 39
- Usage: `rf = np.load('datasets/rf_centers_ground_truth.npz'); eps_x, eps_y = rf['norm_64'][cell_id]`

### Numerical settings
```
eigval_tol = 1e-4   (eigenvalue cutoff for eigenspace truncation)
jitter = 1e-4
lambda_var_clamp = 1e-6
```

---

## How to Run Experiments

### Quick single-cell test
```bash
python run_single_mode.py --mode vargp_direct --ntilde 300 --n-train 2910 --seed 1 --cell 8 \
  --data-path datasets/PNAS_64x64_center_crop_no_renorm.npz
```

### With ground-truth RF init
```python
import numpy as np
rf = np.load('datasets/rf_centers_ground_truth.npz')
eps_x, eps_y = rf['norm_64'][cell_id]
# Pass via CLI: --eps-0x {eps_x} --eps-0y {eps_y}
```

### Batch experiment via subprocess (for multi-cell runs)
See `investigations/additional_dataset/run_lsta_init_64.py` for the pattern:
```python
from run_single_mode import build_config_from_defaults
config = build_config_from_defaults(mode='vargp_direct', M=300, n_train=2910,
    seed=1, cell=cell_id, data_path='datasets/PNAS_64x64_center_crop_no_renorm.npz',
    eps_0x=float(eps_x), eps_0y=float(eps_y),
    n_iterations=100,  # override any parameter
)
# Run as subprocess for GPU memory isolation:
# Write config to temp JSON, run: python run_single_mode.py --from-config temp.json
# Parse RESULT_JSON: line from stdout
```

### IMPORTANT: GPU memory isolation
Each fit MUST run as a separate subprocess (not in-process loop). GPU memory leaks if you call `run_single_config()` in a loop. Use the subprocess pattern from `run_experiment.py`.

---

## Investigation Strategy

### Phase 1: Read the paper carefully
Read pages 2-4 of the PNAS paper. Extract:
- What image resolution do they use?
- What M (inducing points) do they use?
- How many training iterations?
- What kernel hyperparameter initialization?
- Do they use early stopping?
- What training/test split?
- Any data preprocessing differences?

### Phase 2: Match paper configuration exactly
Before tuning, reproduce the paper's configuration as closely as possible. Any difference from the paper is a potential source of the gap.

### Phase 3: Systematic parameter sweep
If Phase 2 doesn't close the gap, sweep one parameter at a time:
1. n_iterations (50, 100, 200, 500) — are we undertrained?
2. n_estep (1, 5, 10, 20) — E-step convergence
3. n_mstep (5, 10, 20, 50) — M-step convergence
4. M (100, 300, 500, 1000) — inducing point count
5. beta init (0.05, 0.1, 0.2, 0.5) — RF width
6. Early stopping off vs on
7. eigval_tol (1e-3, 1e-4, 1e-5) — eigenspace dimension

Use a representative subset of cells (e.g., 5 cells spanning the performance range: cells 1, 8, 12, 30, 39) with 1 seed for the sweep. Full 41-cell x 3-seed runs only for the best configuration.

### Phase 4: Report honestly
- Every result must include adjusted_r2 (the paper's metric)
- Compare against the baseline (avg adj_r2 = 0.720, 13/41 > 0.8)
- Report number of cells > 0.8 (paper target: 36/41)
- If a change helps some cells but hurts others, report both
- If we can't match the paper, document why honestly

---

## Critical Rules

1. **Do NOT modify `default_params.json` or YAML configs** — these are the reproducibility baseline. Use CLI overrides or `build_config_from_defaults(**overrides)`.
2. **Do NOT change the kernel type** — must stay `arc_cosine`.
3. **Do NOT change the GP architecture** — variational GP with inducing points.
4. **All investigation scripts go in `investigations/`** — use a dedicated subfolder (e.g., `investigations/parameter_optimization/`).
5. **Report adjusted_r2**, not just test_r. The paper uses adjusted R².
6. **Use ground-truth RF init** (`datasets/rf_centers_ground_truth.npz`) for all runs — this is our best initialization.
7. **Any code change must be justified** — check CLAUDE.md and DECISION_LOG.md for why things are the way they are before modifying.
8. **Flag surprising results** to the user — good or bad.
9. **GPU memory**: Use subprocess isolation for batch runs. Check `nvidia-smi` between runs.
10. **The conda environment `pytorch_gpytorch` is already active** — just use `python` directly.

---

## Files You'll Need

| File | Purpose |
|------|---------|
| `run_single_mode.py` | Single-cell training (`build_config_from_defaults`, `run_single_config`, `--from-config`) |
| `metrics.py` | `compute_adjusted_r_squared()`, `compute_explained_variance()` |
| `default_params.json` | Current parameter defaults (DO NOT EDIT) |
| `datasets/rf_centers_ground_truth.npz` | Ground-truth RF centers (`norm_64`, `norm_108`, `norm_48`) |
| `datasets/PNAS_64x64_center_crop_no_renorm.npz` | Current best dataset |
| `experiments/2026-03-24_ground_truth_init_64/results.jsonl` | Baseline results to beat |
| `papers/PNAS goldin 2023...pdf` | Reference paper (read pages 2-4) |
| `.claude/CLAUDE.md` | Project documentation, known issues, decisions |

---

## Continuation Prompt

```
I'm investigating why our GP model fits don't match the performance reported in Goldin et al. 2023 PNAS. Read the handoff at .claude/handoffs/HANDOFF_2026-03-25_parameter-optimization.md for full context. Then:

1. Read pages 2-4 of the PNAS paper (papers/ folder) to extract their exact configuration
2. Compare with our current defaults (default_params.json)
3. Propose a systematic investigation plan
4. Run experiments, report adjusted_r2, compare against baseline (avg=0.720, 13/41 cells > 0.8)

Target: adjusted R² > 0.8 for 36/41 cells (matching the paper).
```
