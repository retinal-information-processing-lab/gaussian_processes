# Active Learning Loop — Design Document

## Why this script exists

`run_active_loop.py` is an evaluation harness for comparing utility optimization
algorithms on the PNAS retinal ganglion cell dataset. It runs a simulated active
learning loop where ground-truth spike counts are looked up from pre-recorded data,
removing all hardware dependencies (ZMQ, MEA, DMD) from the equation.

The motivation: the user is investigating alternative utility optimization methods
(gradient ascent, subspace-constrained optimization, distribution-aware utility) on
branches `pietro/utility_optimization` and `investigating_paper_gap`. These methods
need to be evaluated in the context of a full active learning loop — not just
one-shot utility computation on a fixed model — because the model evolves as new
data points are added.

---

## Scope: vargp_direct ONLY

This script uses the eigenspace-based variational GP (`DirectVGPModel` from
`eigenspace_model.py`). The `default_gpy` mode (standard GPyTorch variational
inference) is explicitly out of scope because:

1. The user's primary investigation branches use `vargp_direct`
2. `distribution_aware_utility()` in `acquisition.py` requires `default_gpy`
   (needs `model(X).covariance_matrix`), which is a separate investigation track
3. Mixing modes in one script adds complexity without value for the current goal

---

## Design decisions

### 1. Test set: 30-image PNAS test set (30 repetitions)

**Choice**: Use the held-out 30 test images with 30 repetitions each.
**Alternative rejected**: Reserve 1000 images from the 3160 pool as an internal test set
(as the original notebook does).

**Rationale**: The 30-image set has 30 repetitions per image, enabling reliability-
normalized metrics (adjusted R^2, explained variance). Reserving 1000 from the pool
would reduce the candidate set from 3160 to 2160, significantly shrinking the active
learning search space. The 30-image evaluation is the standard protocol used by
`run_single_config()` and all experiment scripts.

### 2. Phase 1: Delegate to run_single_config()

**Choice**: Call `build_config_from_defaults(mode='vargp_direct', M=50, n_train=50)`
then `run_single_config(config)`.

**Rationale**: `run_single_config()` handles data loading, STA-based RF center
initialization, pivoted Cholesky inducing point selection, kernel creation, and
training — over 200 lines of validated code. Reimplementing any of this would be
error-prone and violate DRY.

### 3. Phase 2 parameters from default_params.json

**Choice**: Added `active_learning` section to `default_params.json`.
**Source**: Phase 2 values (n_iterations=5, n_estep=10, etc.) match the production
closed-loop experiment configuration in `config/config.py:191-198`.

**Rationale**: Follows the project's "no hidden hardcoded parameters" rule. Any script
that reads `default_params.json` can see these defaults.

### 4. M == n_train enforced

**Choice**: Require that the number of inducing points equals the number of training
points at all times.

**Rationale**: `extend_model_with_new_point()` creates a new model with
`X_tilde_new = [X_tilde; x_new]` and passes it as both X_train and X_tilde to the
`DirectVGPModel` constructor. This is the same convention used in the notebook reference
and the standalone experiment code.

### 5. r_max: Fixed 100

**Choice**: Use `r_max=100` from `default_params.json` utility section.
**Alternative rejected**: Adaptive r_max.

**Rationale**: r_max=100 has been used throughout the project (notebooks, experiments,
acquisition.py validation). Adaptive r_max adds complexity and has not been validated
in the active loop context.

### 6. Inducing point selection: Pivoted Cholesky

**Choice**: Use `ip_selection='pivoted'` (the project default).
**Alternative**: Random selection (notebook convention).

**Rationale**: Pivoted Cholesky selects kernel-diverse inducing points, giving a better
initial coverage of the image space. This is the default in `default_params.json` and
`run_single_config()`.

### 7. Early stopping disabled for Phase 2

**Choice**: `early_stop=False` for Phase 2 retraining.

**Rationale**: Phase 2 uses only 5 EM iterations. The early stopping mechanism requires
`min_iterations=10` before it can trigger, so it would never fire. Disabling it avoids
the overhead of tracking the loss window.

### 8. GPU memory management

**Choice**: `del model` before creating the new model each iteration.

**Rationale**: Each iteration creates a new `DirectVGPModel` with M+i points. The old
model's kernel matrices (M+i-1 x M+i-1) should be freed before allocating new ones.
At typical sizes (M=50 to 300), memory is not a concern, but this is good practice.

### 9. Strategy: argmax only (v1)

**Choice**: Only implement `standard_utility` argmax selection.

**Rationale**: This is the simplest, most validated selection strategy. The script
structure (compute_utility_and_select function) is designed so that alternative strategies
can be added via if/elif on a strategy parameter. Future strategies:
- `random`: Random selection (baseline, trivial to add)
- `gradient`: Argmax + L-BFGS pixel optimization
- `subspace`: Argmax + PCA/C-eigenspace constrained optimization

---

## What was excluded (and why)

| Feature | Reason for exclusion |
|---------|---------------------|
| `default_gpy` mode | Out of scope per user instruction |
| `distribution_aware_utility` | Requires `covariance_matrix` (default_gpy only) |
| Image pixel optimization | Future strategy — needs gradient flow through utility |
| Multi-cell support | Single cell is cleaner for algorithm comparison |
| Threading / parallelism | Sequential is correct for offline simulation |
| STA snapshots / live plots | Not needed for algorithm evaluation |
| Save/load intermediate models | Can be added later if checkpointing is needed |
| Random baseline strategy | Trivial to add later (5 lines) |
| 1000-image internal test set | Reduces candidate pool; 30-image PNAS test is sufficient |

---

## Existing functions reused

| Function | File | Purpose in this script |
|----------|------|----------------------|
| `build_config_from_defaults()` | `run_single_mode.py:242` | Build Phase 1 config from defaults |
| `run_single_config()` | `run_single_mode.py:391` | Complete Phase 1 training |
| `load_pnas_data()` | `run_single_mode.py:78` | Load raw PNAS arrays |
| `set_reproducible_seed()` | `run_single_mode.py` | Reproducible random state |
| `standard_utility()` | `acquisition.py:41` | Compute utility for candidates |
| `extend_model_with_new_point()` | `rank1_update.py:25` | Rank-1 model extension |
| `train_eigenspace()` | `eigenspace_training.py:85` | Phase 2 retraining |
| `predict_eigenspace()` | `eigenspace_training.py:302` | Test set prediction |
| `compute_pearson_correlation()` | `metrics.py:44` | Test Pearson r |
| `compute_adjusted_r_squared()` | `metrics.py:117` | Adjusted R^2 |
| `compute_explained_variance()` | `metrics.py:74` | Explained variance + reliability |

---

## How to extend with new strategies

Add a `strategy` parameter to `compute_utility_and_select()`:

```python
def compute_utility_and_select(model, likelihood, X_candidates, r_max, strategy='argmax'):
    if strategy == 'argmax':
        result = standard_utility(model, likelihood, X_candidates, ...)
        best = result['utility'].argmax().item()
        return best, result['utility'][best].item()
    elif strategy == 'random':
        best = torch.randint(X_candidates.shape[0], (1,)).item()
        return best, 0.0
    elif strategy == 'gradient':
        # Select argmax, then optimize pixels via L-BFGS
        ...
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
```

The loop body does not change — only the selection function.
