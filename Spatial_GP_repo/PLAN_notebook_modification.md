# Implementation Plan: Modify Active Learning Loop in Notebook

## Overview

Modify `one_cell_active_training_distribution_aware.ipynb` cell 15 to:
1. Use `batch_utility_w_grad()` for efficient image selection from ALL remaining images
2. Use `optimize_with_conditioned_utility()` for conditioned utility optimization
3. **Train model with ORIGINAL images only** (optimization is just for analysis/snapshot)
4. Add plotting functions for visualization

## Design Principles

- **Script-compatible**: Code should work both in Jupyter and as a standalone .py script
- **No interactive displays**: All plots saved to files, not shown with `plt.show()`
- **Exception**: `utils.plot_loss_and_theta_notebook()` can remain interactive
- Use standard `tqdm` (not notebook-specific)

---

## Changes Required

### 1. Add New Cell Before Active Learning Loop (after cell 14)

Add a cell to:
- Define `update_optimization_comparison()` function (copied from experiment_re_run.py)
- Create save directories for plots

```python
# === Setup for optimization comparison tracking ===
import pickle
from pathlib import Path

# Create save directory for optimization comparison plots
save_dir = Path('data/optimization_comparison')
save_dir.mkdir(parents=True, exist_ok=True)

def update_optimization_comparison(result_dict, imgs_train, n_images, save_dir, model, window=10):
    """
    Update and save optimization constraint comparison figure.
    Completely self-contained - handles all tracking internally via pickle file.

    [Full function from experiment_re_run.py lines 103-312]
    """
    # ... (copy full function from experiment_re_run.py)
```

---

### 2. Replace Cell 15 (Active Learning Loop)

**BEFORE (current slow approach):**
```python
# Loop over each image - SLOW
for idx in tqdm(remaining_idx[:10], desc=f'Iter {j} utility'):
    utility = GP_utility.conditioned_utility_clean(x_star, model, remaining_imgs, N, r_cutoff)
    utilities.append(utility.item())
utilities = torch.tensor(utilities)
i_best = utilities.argmax()
x_idx_best = remaining_idx[i_best]
```

**AFTER (efficient batch approach):**

```python
torch.set_grad_enabled(False)

# Convert initial model to GPModel
active_model = GPModel(model_dict=start_model)
current_spikes = R_in_use.clone()

active_model.kernfun = GP_utils.acosker

# Active learning parameters
n_iterations = 50  # Number of images to add
r_cutoff = 100

# Tracking lists
utilities_track = []
r2_track = []
optimization_results_track = []

# Save directory for plots
save_dir = Path('data/optimization_comparison')
save_dir.mkdir(parents=True, exist_ok=True)

for j in range(n_iterations):
    print(f'=== Iteration {j} ===')

    # 1. Get remaining images (ALL of them, not just 10)
    in_use_idx = active_model.in_use_idx
    remaining_idx = all_idx_perm[~torch.isin(all_idx_perm, in_use_idx)]

    print(f'  Remaining images: {len(remaining_idx)}')

    # ==========================================================================
    # 2. SELECT & OPTIMIZE using batch_utility_w_grad (RMS constraint)
    #    This efficiently computes utility for ALL remaining images and optimizes
    # ==========================================================================
    result_rms = GP_utility.batch_utility_w_grad(
        active_model,
        X,  # Full image dataset
        remaining_idx,
        max_r_cap=r_cutoff,
        max_iter=10,
        test_rms_constraint=True,
        lr=1,
        return_logf_moments=True,
    )

    x_idx_best = result_rms['img_idx'].item()
    print(f'  Selected image idx: {x_idx_best}')
    print(f'  RMS Utility: {result_rms["U_initial"]:.4f} -> {result_rms["U_final"]:.4f}')

    # ==========================================================================
    # 3. OPTIMIZE using conditioned utility (for comparison/snapshot)
    #    This provides a different optimization approach using conditioned utility
    # ==========================================================================
    with torch.enable_grad():
        result_cond = GP_utility.optimize_with_conditioned_utility(
            model=active_model,
            imgs_train=X,
            remaining_idx=remaining_idx,
            start_img_idx=x_idx_best,
            N=1,  # Single sample for DEBUG_SINGLE_IMAGE compatibility
            lambda_samples=100,
            n_iterations=50,
            r_cutoff=r_cutoff,
            return_logf_moments=True,
            DEBUG_dict={'DEBUG_SINGLE_IMAGE': True},
        )

    print(f'  Cond Utility: {result_cond["U_initial"]:.4f} -> {result_cond["U_final"]:.4f}')

    # ==========================================================================
    # 4. SAVE PLOTS (not shown in notebook)
    # ==========================================================================

    # Single image sampling plot
    GP_utility.single_image_sampling_plot(
        result_cond,
        X,
        save_path=save_dir / 'single_img_samples' / f'iter_{j:03d}_idx_{x_idx_best}.png',
        shared_scale=False
    )

    # Update optimization comparison statistics
    update_optimization_comparison(
        result_dict={'RMS': result_rms, 'cond_utility': result_cond},
        imgs_train=X,
        n_images=len(in_use_idx) + 1,  # Current number of images after adding this one
        save_dir=save_dir,
        model=active_model
    )

    # Track results
    utilities_track.append({
        'rms_U_initial': result_rms['U_initial'],
        'rms_U_final': result_rms['U_final'],
        'cond_U_initial': result_cond['U_initial'],
        'cond_U_final': result_cond['U_final'],
        'img_idx': x_idx_best,
    })

    # ==========================================================================
    # 5. UPDATE MODEL WITH ORIGINAL IMAGE AND RESPONSE
    #    IMPORTANT: We use the ORIGINAL image for training, NOT the optimized one.
    #    The optimization above is just a snapshot of what WOULD happen if we
    #    optimized - but since we have real neural responses to original images,
    #    we must train on originals.
    # ==========================================================================
    new_spike = R[x_idx_best]
    current_spikes = torch.cat((current_spikes, new_spike[None]))

    active_model = utils.generate_new_active_model(
        current_model=active_model,
        x_idx_chosen=torch.tensor([x_idx_best], device=device),
        img_train=X,
        new_spike_counts=current_spikes
    )

    # ==========================================================================
    # 6. FIT MODEL (on original images)
    # ==========================================================================
    model_dict, err_dict = utils.varGP(
        X[active_model.in_use_idx],
        current_spikes,
        **active_model.to_dict(),
        verbose=False
    )

    if err_dict['is_error']:
        print(f'  Error in fit: {err_dict["error"]}')
        if not isinstance(err_dict['error'], GP_utils.LossStagnationError):
            raise err_dict['error']
        # Continue with current model if just stagnation
    else:
        active_model = GPModel(model_dict=model_dict)

    # ==========================================================================
    # 7. EVALUATE ON TEST SET
    # ==========================================================================
    _, _, r2, _ = utils.test(X_test, R_test, at_iteration=None, **active_model.to_dict())
    r2_track.append(r2.item())
    print(f'  R^2: {r2.item():.4f}')

print('\nActive learning loop completed!')
print(f'Final R^2: {r2_track[-1]:.4f}')
print(f'Plots saved to: {save_dir}')
```

---

### 3. Update Visualization Cell (cell 17)

**BEFORE:**
```python
# Plot utilities
ax1.plot(utilities_track, 'b-o', markersize=3)
```

**AFTER:**
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Extract data from new tracking structure
rms_U_initial = [u['rms_U_initial'] for u in utilities_track]
rms_U_final = [u['rms_U_final'] for u in utilities_track]
cond_U_initial = [u['cond_U_initial'] for u in utilities_track]
cond_U_final = [u['cond_U_final'] for u in utilities_track]

# Plot 1: RMS Utility (initial vs final)
ax1 = axes[0, 0]
ax1.plot(rms_U_initial, 'b--o', markersize=3, label='RMS Initial', alpha=0.7)
ax1.plot(rms_U_final, 'b-o', markersize=3, label='RMS Final')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Utility')
ax1.set_title('RMS-Constrained Optimization')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Conditioned Utility (initial vs final)
ax2 = axes[0, 1]
ax2.plot(cond_U_initial, 'g--o', markersize=3, label='Cond Initial', alpha=0.7)
ax2.plot(cond_U_final, 'g-o', markersize=3, label='Cond Final')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Utility')
ax2.set_title('Conditioned Utility Optimization')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Utility Improvement Comparison
ax3 = axes[1, 0]
rms_improvement = [f - i for f, i in zip(rms_U_final, rms_U_initial)]
cond_improvement = [f - i for f, i in zip(cond_U_final, cond_U_initial)]
ax3.plot(rms_improvement, 'b-o', markersize=3, label='RMS')
ax3.plot(cond_improvement, 'g-o', markersize=3, label='Conditioned')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Utility Improvement')
ax3.set_title('Utility Improvement per Iteration')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(0, color='k', linestyle='--', alpha=0.3)

# Plot 4: R^2 over iterations
ax4 = axes[1, 1]
ax4.plot(r2_track, 'r-o', markersize=3)
ax4.set_xlabel('Iteration')
ax4.set_ylabel('R^2')
ax4.set_title('Test R^2 Over Iterations')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(save_dir / 'learning_curves.png', dpi=150, bbox_inches='tight')
plt.close(fig)  # Close figure instead of showing - script compatible

print(f'Initial R^2: {r2_track[0]:.4f}')
print(f'Final R^2: {r2_track[-1]:.4f}')
print(f'Improvement: {r2_track[-1] - r2_track[0]:.4f}')
print(f'Learning curves saved to: {save_dir / "learning_curves.png"}')
```

---

## Summary of Key Changes

| Aspect | Before | After |
|--------|--------|-------|
| Image selection | Loop over 10 images with `conditioned_utility_clean()` | `batch_utility_w_grad()` on ALL remaining images |
| RMS optimization | None | `batch_utility_w_grad(max_iter=10, test_rms_constraint=True)` |
| Conditioned optimization | Used for selection (slow) | `optimize_with_conditioned_utility()` for snapshot |
| Training images | Original | Original (unchanged, with explicit comment) |
| Plotting | Basic utility + R^2 | + `single_image_sampling_plot()` + `update_optimization_comparison()` |
| Plot storage | Shown in notebook | Saved to `data/optimization_comparison/` |

---

## Files to Modify

1. `one_cell_active_training_distribution_aware.ipynb`:
   - Add new cell with `update_optimization_comparison()` function
   - Replace cell 15 (active learning loop)
   - Update cell 17 (visualization)

## Dependencies

The following imports are already present or need to be added:
- `GP_utility.batch_utility_w_grad`
- `GP_utility.optimize_with_conditioned_utility`
- `GP_utility.single_image_sampling_plot`
- `pickle` (for tracking data)
- `Path` from `pathlib`
