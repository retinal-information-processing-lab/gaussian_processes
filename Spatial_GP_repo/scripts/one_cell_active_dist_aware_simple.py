"""
One Cell Active Training - Simplified Version
Created by Claude

Purpose: Test that image optimization via conditioned_utility converges from
different gray starting images toward a sample gray image. This is a minimal
test case for understanding the optimization behavior.

What this script does:
1. Load data and pre-trained GP model
2. Evaluate model quality on test set (R²)
3. Run optimize_with_conditioned_utility with DEBUG_FULL_FIELD mode
   for two different starting gray levels
4. Verify both starting points converge toward the sample gray level
5. Print results and save visualization

What's removed from the full version:
- Full active learning loop (only optimization test here)
- Utility evaluation for all remaining images
- Model fitting (uses pre-trained model)
- Random vs active comparison
- Multiple initial conditions
"""

import torch
torch.set_grad_enabled(False)

import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# Add project path
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject')

from gaussian_processes.Spatial_GP_repo import utils as GP_utils
from gaussian_processes.Spatial_GP_repo import utility as GP_utility
from gaussian_processes.Spatial_GP_repo.model import GPModel

# =============================================================================
# Configuration - Adjust these parameters as needed
# =============================================================================
CELLID = 8
STARTING_GRAY_LEVELS = [0.3, 0.7]  # Two starting points to test (fraction of pixel range)
SAMPLE_GRAY_LEVEL = 0.5            # Sample image gray level (fraction of pixel range)
R_CUTOFF = 100                     # Max spike count for utility computation
N_ITERATIONS = 50                 # Optimization steps
LR = 0.01                         # Learning rate (lowered to prevent overshooting)
LAMBDA_SAMPLES = 500              # Lambda samples per x for conditioned utility

# =============================================================================
# Setup
# =============================================================================
TORCH_DTYPE = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(TORCH_DTYPE)
torch.set_default_device(device)

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

print(f"Device: {device}")
print(f"=" * 60)

# =============================================================================
# 1. Load Data
# =============================================================================
print("\n[1] Loading Data")
print("-" * 40)

data_path = Path(__file__).parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
data = np.load(data_path)

X_train = torch.tensor(data['images_train'], dtype=TORCH_DTYPE)
X_val = torch.tensor(data['images_val'], dtype=TORCH_DTYPE)
X_test_4d = torch.tensor(data['images_test'], dtype=TORCH_DTYPE)  # Keep 4D for test()

R_train = torch.tensor(data['responses_train'], dtype=TORCH_DTYPE)
R_val = torch.tensor(data['responses_val'], dtype=TORCH_DTYPE)
R_test = torch.tensor(data['responses_test'], dtype=TORCH_DTYPE)

# Combine train and validation
X = torch.cat((X_train, X_val), axis=0)
R = torch.cat((R_train, R_val), axis=0)

n_px_side = X.shape[1]

# Reshape images to 1D and select cell
X = X.reshape(X.shape[0], -1)  # (n_images, n_pixels)
R = R[:, CELLID]  # (n_images,)
R_test_cell = R_test[..., CELLID]  # For test evaluation

print(f"  Dataset: {X.shape[0]} images, {X.shape[1]} pixels")
print(f"  Cell ID: {CELLID}")
print(f"  Pixel range: [{X.min().item():.4f}, {X.max().item():.4f}]")

# =============================================================================
# 2. Create Train/Test Split (same as training)
# =============================================================================
print("\n[2] Creating Train/Test Split")
print("-" * 40)

ntrain_start = 500

all_idx = torch.arange(X.shape[0])
torch.manual_seed(8)
all_idx_perm = torch.randperm(len(all_idx))

test_1000_idx = all_idx_perm[-1000:]
all_idx_perm = all_idx_perm[~torch.isin(all_idx_perm, test_1000_idx)]
train_idx = all_idx_perm[:ntrain_start]
remaining_idx = all_idx_perm[ntrain_start:]

print(f"  Training images: {len(train_idx)}")
print(f"  Remaining images: {len(remaining_idx)}")

# =============================================================================
# 3. Load Pre-trained Model
# =============================================================================
print("\n[3] Loading Pre-trained Model")
print("-" * 40)

model_path = Path(__file__).parent.parent / 'notebooks' / 'data' / 'models' / f'model_cell:{CELLID}'

if not model_path.exists():
    raise FileNotFoundError(
        f"Pre-trained model not found at {model_path}\n"
        "Please run the full training notebook first to create the model."
    )

with open(model_path, 'rb') as f:
    model_dict = pickle.load(f)

model = GPModel(model_dict=model_dict)
model.kernfun = GP_utils.acosker

print(f"  Model loaded from: {model_path}")
print(f"  Inducing points: {len(model.in_use_idx)}")

# =============================================================================
# 4. Evaluate Model Quality on Test Set
# =============================================================================
print("\n[4] Evaluating Model Quality")
print("-" * 40)

_, _, r2, sigma_r2 = GP_utils.test(
    X_test_4d, R_test_cell,
    at_iteration=None,
    print_expl_var=False,
    **model.to_dict()
)

print(f"  Test set R²: {r2:.4f} +/- {sigma_r2:.4f}")

# =============================================================================
# 5. Run Image Optimization for Each Starting Gray Level
# =============================================================================
print("\n[5] Running Image Optimization")
print("-" * 40)
print(f"  Sample gray level: {SAMPLE_GRAY_LEVEL*100:.0f}%")
print(f"  Starting gray levels: {[f'{g*100:.0f}%' for g in STARTING_GRAY_LEVELS]}")
print(f"  Optimization iterations: {N_ITERATIONS}")
print(f"  Learning rate: {LR}")

# Pick a reference image index (not actually used in DEBUG_FULL_FIELD, but required by API)
start_img_idx = remaining_idx[0].item()

results = {}

for start_gray in STARTING_GRAY_LEVELS:
    print(f"\n  --- Starting gray: {start_gray*100:.0f}% ---")

    with torch.enable_grad():
        result = GP_utility.optimize_with_conditioned_utility(
            model=model,
            imgs_train=X,
            remaining_idx=remaining_idx,
            start_img_idx=start_img_idx,
            N=1,  # Single gray sample
            lambda_samples=LAMBDA_SAMPLES,
            n_iterations=N_ITERATIONS,
            lr=LR,
            r_cutoff=R_CUTOFF,
            return_logf_moments=True,
            DEBUG_dict={
                'DEBUG_SINGLE_IMAGE': True,
                'DEBUG_FULL_FIELD': True,
                'start_gray': start_gray,
                'sample_gray': SAMPLE_GRAY_LEVEL,
                'verbose': True
            }
        )

    results[start_gray] = result

    # Print summary
    delta_u = result['U_final'] - result['U_initial']
    print(f"      Utility: {result['U_initial']:.6f} -> {result['U_final']:.6f} (delta: {delta_u:+.6f})")

# =============================================================================
# 6. Analyze Convergence (using MASKED region only)
# =============================================================================
print("\n[6] Convergence Analysis")
print("-" * 40)

pixel_min = X.min()
pixel_max = X.max()
pixel_range = pixel_max - pixel_min
sample_pixel_value = pixel_min + pixel_range * SAMPLE_GRAY_LEVEL

# IMPORTANT: the utility/optimizer uses the mask stored in model['final_kernel']['mask']
# via GP_utils.get_final_K_vals(model). Use that exact mask here to avoid mismatches.
_, mask_utility, _, _, _ = GP_utils.get_final_K_vals(model)
mask_attr = getattr(model, 'mask', None)

mask = mask_utility

print(f"  Utility mask size: {mask.sum().item()} pixels out of {len(mask)} total")
if mask_attr is None:
    print("  model.mask attribute: None")
else:
    same_shape = (mask_attr.shape == mask.shape)
    same_values = bool(same_shape and torch.equal(mask_attr, mask))
    print(f"  model.mask attribute size: {mask_attr.sum().item()} pixels")
    print(f"  Mask shapes match: {same_shape}")
    print(f"  Masks identical: {same_values}")
print(f"  Sample pixel value: {sample_pixel_value.item():.4f}")

# Store analysis results for plotting
analysis_results = {}

for start_gray, result in results.items():
    initial_img = result.get('initial_img')
    final_img = result['optimized_img']

    # Compute starting pixel value
    start_pixel_val = (pixel_min + pixel_range * start_gray).item()

    # Diagnose whether any pixels OUTSIDE the utility mask changed
    if initial_img is not None:
        delta_img = final_img - initial_img
        max_abs_delta_outside = delta_img[~mask].abs().max().item()
        max_abs_delta_inside = delta_img[mask].abs().max().item()
    else:
        max_abs_delta_outside = float('nan')
        max_abs_delta_inside = float('nan')

    # Get values in MASKED region only
    if initial_img is not None:
        initial_masked_mean = initial_img[mask].mean().item()
    else:
        initial_masked_mean = start_pixel_val

    final_masked_mean = final_img[mask].mean().item()
    sample_val = sample_pixel_value.item()

    # Full-image means are NOT the right metric (utility only depends on masked pixels).
    # Keep them only as a diagnostic sanity check.
    if initial_img is not None:
        initial_full_mean = initial_img.mean().item()
    else:
        initial_full_mean = start_pixel_val
    final_full_mean = final_img.mean().item()

    # Check if it moved toward the sample (in masked region)
    initial_dist = abs(initial_masked_mean - sample_val)
    final_dist = abs(final_masked_mean - sample_val)
    converged = final_dist < initial_dist

    # Store for plotting
    analysis_results[start_gray] = {
        'start_pixel_val': start_pixel_val,
        'initial_masked_mean': initial_masked_mean,
        'final_masked_mean': final_masked_mean,
        'sample_val': sample_val,
        'initial_full_mean': initial_full_mean,
        'final_full_mean': final_full_mean,
        'converged': converged,
        'max_abs_delta_inside_mask': max_abs_delta_inside,
        'max_abs_delta_outside_mask': max_abs_delta_outside
    }

    print(f"\n  Starting {start_gray*100:.0f}% gray (diagnostic start pixel val: {start_pixel_val:.4f}):")
    print(f"    MASKED region mean (THIS is what matters for the utility):")
    print(f"      Initial: {initial_masked_mean:.4f}")
    print(f"      Final:   {final_masked_mean:.4f}")
    print(f"      Sample:  {sample_val:.4f}")
    print(f"      Distance to sample (masked): {initial_dist:.4f} -> {final_dist:.4f}")
    print(f"    FULL image mean (diagnostic only; expect small changes due to small mask):")
    print(f"      Initial: {initial_full_mean:.4f}")
    print(f"      Final:   {final_full_mean:.4f}")
    print(f"    Converged toward sample: {'YES' if converged else 'NO'}")
    if initial_img is not None:
        print(f"    Max |Δpixel| inside mask:  {max_abs_delta_inside:.6e}")
        print(f"    Max |Δpixel| outside mask: {max_abs_delta_outside:.6e}")

# =============================================================================
# 7. Visualization
# =============================================================================
print("\n[7] Creating Visualization")
print("-" * 40)

fig, axes = plt.subplots(len(STARTING_GRAY_LEVELS), 4, figsize=(18, 5 * len(STARTING_GRAY_LEVELS)))

# Ensure axes is 2D even with single row
if len(STARTING_GRAY_LEVELS) == 1:
    axes = axes[np.newaxis, :]

# Create 2D mask for visualization
mask_2d = mask.cpu().reshape(n_px_side, n_px_side)

# Use a single shared grayscale scale for all image panels
vmin, vmax = pixel_min.cpu().item(), pixel_max.cpu().item()

# Use a single shared diverging scale for all diff panels (across all starting grays)
global_max_abs_diff = 0.0
for start_gray, result in results.items():
    initial_img = result.get('initial_img')
    final_img = result['optimized_img']
    if initial_img is None:
        start_val = pixel_min + pixel_range * start_gray
        initial_img = torch.full_like(final_img, start_val.item())
    diff = (final_img - initial_img).detach()
    global_max_abs_diff = max(global_max_abs_diff, diff.abs().max().item())
global_max_abs_diff = max(global_max_abs_diff, 1e-12)

for i, (start_gray, result) in enumerate(results.items()):
    # Get images
    initial_img = result.get('initial_img')
    final_img = result['optimized_img']

    if initial_img is not None:
        initial_2d = initial_img.cpu().reshape(n_px_side, n_px_side)
    else:
        start_val = pixel_min + pixel_range * start_gray
        initial_2d = torch.full((n_px_side, n_px_side), start_val.item())

    final_2d = final_img.cpu().reshape(n_px_side, n_px_side)
    diff_2d = final_2d - initial_2d

    # Sample image (uniform gray) - create on CPU
    sample_2d = torch.full((n_px_side, n_px_side), sample_pixel_value.cpu().item(), device='cpu')

    # Get analysis values for this starting gray
    ar = analysis_results[start_gray]

    # Plot initial
    im0 = axes[i, 0].imshow(initial_2d.numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    axes[i, 0].set_title(
        f'Initial ({start_gray*100:.0f}% gray)\nmask mean={ar["initial_masked_mean"]:.3f}'
    )
    axes[i, 0].axis('off')
    plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)

    # Plot sample (target)
    im1 = axes[i, 1].imshow(sample_2d.numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    axes[i, 1].set_title(
        f'Sample ({SAMPLE_GRAY_LEVEL*100:.0f}% gray)\nmask mean={ar["sample_val"]:.3f}'
    )
    axes[i, 1].axis('off')
    plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

    # Plot final (optimized) with masked region mean
    im2 = axes[i, 2].imshow(final_2d.numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    axes[i, 2].set_title(f'Optimized (U={result["U_final"]:.4f})\nmask mean={ar["final_masked_mean"]:.3f}')
    axes[i, 2].axis('off')
    plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

    # Plot difference with min/max in masked region
    diff_masked = diff_2d[mask_2d]
    diff_masked_min = diff_masked.min().item()
    diff_masked_max = diff_masked.max().item()
    diff_masked_mean = diff_masked.mean().item()

    im3 = axes[i, 3].imshow(diff_2d.numpy(), cmap='RdBu_r', vmin=-global_max_abs_diff, vmax=global_max_abs_diff)
    axes[i, 3].set_title(f'Difference (Final - Initial)\nmask: [{diff_masked_min:.2f}, {diff_masked_max:.2f}], mean={diff_masked_mean:.3f}')
    axes[i, 3].axis('off')
    plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)

    # Add row label on the left
    axes[i, 0].set_ylabel(f'Start: {start_gray*100:.0f}%', fontsize=12, rotation=0, labelpad=50, va='center')

plt.suptitle(
    f'Image Optimization (masked-region objective): Gray Levels → {SAMPLE_GRAY_LEVEL*100:.0f}% Sample (masked mean={sample_pixel_value.item():.3f})\n'
    f'Model R² = {r2:.4f}, {N_ITERATIONS} iterations, lr={LR}, mask={mask.sum().item()} pixels',
    fontsize=14
)
plt.tight_layout()

# Save figure
save_path = Path(__file__).parent / 'optimization_gray_test.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"  Figure saved to: {save_path}")

plt.close()  # Close figure instead of showing (avoids blocking in headless environment)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Model R²: {r2:.4f} +/- {sigma_r2:.4f}")
print(f"Sample gray: {SAMPLE_GRAY_LEVEL*100:.0f}% (pixel={sample_pixel_value.item():.4f})")
print(f"Mask size: {mask.sum().item()} pixels")
print("\nOptimization results (masked region):")
for start_gray, result in results.items():
    ar = analysis_results[start_gray]
    delta_u = result['U_final'] - result['U_initial']
    print(f"\n  {start_gray*100:.0f}% gray (pixel={ar['start_pixel_val']:.4f}):")
    print(f"    Utility: {result['U_initial']:.6f} -> {result['U_final']:.6f} ({delta_u:+.6f})")
    print(f"    Masked mean: {ar['initial_masked_mean']:.4f} -> {ar['final_masked_mean']:.4f}")
    print(f"    Distance to sample: {abs(ar['initial_masked_mean'] - ar['sample_val']):.4f} -> {abs(ar['final_masked_mean'] - ar['sample_val']):.4f}")
    print(f"    Converged: {'YES' if ar['converged'] else 'NO'}")
print("\n" + "=" * 60)
