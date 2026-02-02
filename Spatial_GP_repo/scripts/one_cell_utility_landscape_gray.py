"""
One Cell Utility Landscape Visualization - Gray Full-Field Stimuli
Created by Claude

Purpose: Visualize how the distribution-aware utility varies across all possible
gray levels (from pixel_min to pixel_max) when conditioning on a fixed 50% gray sample.

Analogy: Like the third subplot in 1D_playground/gp_utility_playground.py which shows
utility across the 1D domain x, but here the effective domain is the gray level
(since we use uniform gray images, the high-dimensional image space collapses to 1D).

What this script does:
1. Load data and pre-trained GP model
2. Evaluate model quality on test set (R^2)
3. Sweep query gray levels from 0% to 100%
4. For each query gray level, compute conditioned_utility (conditioning on 50% gray)
5. Plot the utility landscape and GP moments vs gray level

This helps understand:
- Where utility is highest/lowest across gray levels
- How GP moments (mean, variance) change with gray level
- The relationship between uncertainty and utility
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
SAMPLE_GRAY_LEVEL = 0.5           # Fixed conditioning point (50% gray)
N_QUERY_POINTS = 50               # Number of gray levels to evaluate across the domain
R_CUTOFF = 100                    # Max spike count for utility computation
LAMBDA_SAMPLES = 2000             # Lambda samples per x for conditioned utility (higher = smoother)

# Use posterior mean instead of MC sampling (deterministic, no noise)
# When True: uses λ_i = μ_i instead of sampling λ_i ~ N(μ_i, σ²_i)
# This gives a perfectly smooth curve but ignores posterior uncertainty
USE_POSTERIOR_MEAN = True

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

torch.manual_seed(8)

# =============================================================================
# 3. Load Pre-trained Model
# =============================================================================
print("\n[3] Loading Pre-trained Model")
print("-" * 40)

# Old model path (from active training, ntilde=500):
# model_path = Path(__file__).parent.parent / 'notebooks' / 'data' / 'models' / f'model_cell:{CELLID}'

# Previous model path (from one_cell_fit.py with ntilde=250, ntrain=2500):
# NTILDE = 250
# NTRAIN = 2500
# model_path = Path(__file__).parent.parent / 'data' / 'models' / f'model_cell:{CELLID}_ntrain:{NTRAIN}'

# New model path (from one_cell_fit.py with ntilde=50, ntrain=100):
NTILDE = 40
NTRAIN = 40
model_path = Path(__file__).parent.parent / 'data' / 'models' / f'model_cell:{CELLID}_ntilde:{NTILDE}_ntrain:{NTRAIN}'

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

# Get training data from model indices
X_train_used = X[model.in_use_idx]
R_train_used = R[model.in_use_idx]

# Compute mean pixel value per training image (proxy for "gray level")
train_mean_pixels = X_train_used.mean(dim=1)  # Shape: (ntrain,)
print(f"  Training images: {len(model.in_use_idx)}")

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
# 5. Compute Utility Landscape Across Gray Levels
# =============================================================================
print("\n[5] Computing Utility Landscape")
print("-" * 40)

# Get pixel range from dataset
pixel_min = X.min()
pixel_max = X.max()
pixel_range = pixel_max - pixel_min
n_pixels = X.shape[1]

# Create query gray levels from 0% to 100%
query_gray_fracs = torch.linspace(0.0, 1.0, N_QUERY_POINTS)
query_gray_values = pixel_min + pixel_range * query_gray_fracs

# Create single sample image at 50% gray (the conditioning point)
sample_gray_value = pixel_min + pixel_range * SAMPLE_GRAY_LEVEL
sample_img = torch.full((1, n_pixels), sample_gray_value.item())  # Shape: (1, n_pixels)

print(f"  Sample gray level: {SAMPLE_GRAY_LEVEL*100:.0f}% (pixel value: {sample_gray_value.item():.4f})")
print(f"  Query gray levels: {N_QUERY_POINTS} points from 0% to 100%")
if USE_POSTERIOR_MEAN:
    print(f"  Mode: DETERMINISTIC (using posterior mean, no MC sampling)")
else:
    print(f"  Mode: MC SAMPLING ({LAMBDA_SAMPLES} lambda samples per query)")
print(f"  R cutoff: {R_CUTOFF}")

# Evaluate utility at each query gray level
utilities = []
logf_means = []
logf_vars = []

print(f"\n  Evaluating utility at {N_QUERY_POINTS} gray levels...")
for i, g_val in enumerate(query_gray_values):
    # Create query image (uniform gray at this level)
    x_star = torch.full((1, n_pixels), g_val.item())  # Shape: (1, n_pixels)

    # Compute conditioned utility
    # DEBUG_FIX_LAMBDA_i=True uses posterior mean instead of sampling
    U, logf_m, logf_v = GP_utility.conditioned_utility_clean(
        x_star=x_star,
        model=model,
        remaining_imgs=sample_img,  # Single 50% gray sample
        N=1,
        r_cutoff=R_CUTOFF,
        lambda_samples_per_x=LAMBDA_SAMPLES,
        DEBUG_FIX_LAMBDA_i=USE_POSTERIOR_MEAN,
        return_logf_moments=True
    )

    utilities.append(U.item())
    logf_means.append(logf_m.item())
    logf_vars.append(logf_v.item())

    # Progress indicator
    if (i + 1) % 10 == 0 or i == 0:
        print(f"    [{i+1:3d}/{N_QUERY_POINTS}] gray={query_gray_fracs[i].item()*100:5.1f}% -> U={U.item():.6f}")

# Convert to tensors for easier manipulation
utilities = torch.tensor(utilities)
logf_means = torch.tensor(logf_means)
logf_vars = torch.tensor(logf_vars)
logf_stds = torch.sqrt(logf_vars)

# Find max utility point
max_idx = torch.argmax(utilities)
max_gray_frac = query_gray_fracs[max_idx].item()
max_utility = utilities[max_idx].item()

print(f"\n  Max utility: {max_utility:.6f} at gray level {max_gray_frac*100:.1f}%")

# =============================================================================
# 6. Get Mask for Reference
# =============================================================================
print("\n[6] Getting Model Mask")
print("-" * 40)

# Get the mask from the model (for reference in visualization)
_, mask, _, _, _ = GP_utils.get_final_K_vals(model)
print(f"  Utility mask size: {mask.sum().item()} pixels out of {len(mask)} total")

# =============================================================================
# 7. Visualization - Utility Landscape
# =============================================================================
print("\n[7] Creating Utility Landscape Visualization")
print("-" * 40)

# Convert to numpy for plotting
query_gray_values_np = query_gray_values.cpu().numpy()  # Pixel values for x-axis
utilities_np = utilities.cpu().numpy()
logf_means_np = logf_means.cpu().numpy()
logf_stds_np = logf_stds.cpu().numpy()

# Convert log-firing rate to firing rate
# If log(f) ~ N(μ, σ²), the ±2σ bounds in log-space become multiplicative bounds in f-space
f_means_np = np.exp(logf_means_np)
f_upper_np = np.exp(logf_means_np + 2 * logf_stds_np)
f_lower_np = np.exp(logf_means_np - 2 * logf_stds_np)

# Get pixel value for sample gray level
sample_pixel_value = sample_gray_value.item()
max_gray_pixel_value = query_gray_values[max_idx].item()

# Create figure with 2 subplots + space for parameters
fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

# ---------- Subplot 1: Firing Rate Prediction ----------
ax1 = axes[0]
ax1.plot(query_gray_values_np, f_means_np, 'b-', linewidth=2, label='Predicted firing rate')
ax1.fill_between(
    query_gray_values_np,
    f_lower_np,
    f_upper_np,
    alpha=0.3, color='blue', label='±2σ (in f-space)'
)
# Mark sample gray level
ax1.axvline(x=sample_pixel_value, color='red', linestyle='--', linewidth=2,
            label=f'Sample ({SAMPLE_GRAY_LEVEL*100:.0f}% gray)')
# Scatter plot of training data (mean pixel value → observed response)
ax1.scatter(train_mean_pixels.cpu().numpy(), R_train_used.cpu().numpy(),
            c='black', s=30, alpha=0.6, zorder=3, label=f'Training data ({len(R_train_used)} imgs)')
ax1.set_ylabel('Firing Rate (spikes/frame)')
ax1.set_title('GP Firing Rate Prediction vs Pixel Value')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# ---------- Subplot 2: Utility Landscape ----------
ax2 = axes[1]
ax2.plot(query_gray_values_np, utilities_np, 'g-', linewidth=2, label='Utility U(g)')
ax2.fill_between(query_gray_values_np, 0, utilities_np, alpha=0.2, color='green')
# Mark sample gray level
ax2.axvline(x=sample_pixel_value, color='red', linestyle='--', linewidth=2,
            label=f'Sample ({SAMPLE_GRAY_LEVEL*100:.0f}% gray)')
# Mark maximum utility point
ax2.scatter([max_gray_pixel_value], [max_utility], c='darkgreen', s=150, zorder=5,
            marker='*', label=f'Max at {max_gray_frac*100:.1f}% gray')
ax2.set_ylabel('Utility')
ax2.set_title(f'Utility Landscape (conditioning on {SAMPLE_GRAY_LEVEL*100:.0f}% gray sample)')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel(f'Pixel Value (range: [{pixel_min.item():.2f}, {pixel_max.item():.2f}])')

# ---------- Parameter Info Box ----------
mode_str = 'Deterministic (posterior mean)' if USE_POSTERIOR_MEAN else f'MC ({LAMBDA_SAMPLES} samples)'
param_text = (
    f"MODEL: Cell {CELLID}, ntilde={NTILDE}, ntrain={NTRAIN}, R²={r2:.3f}\n"
    f"QUERY: {N_QUERY_POINTS} gray levels, r_cutoff={R_CUTOFF}, mode={mode_str}\n"
    f"DOMAIN: pixel ∈ [{pixel_min.item():.2f}, {pixel_max.item():.2f}], "
    f"sample at {sample_pixel_value:.2f} ({SAMPLE_GRAY_LEVEL*100:.0f}% gray)"
)
fig.text(0.5, 0.02, param_text, ha='center', va='bottom', fontsize=10,
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle(f'Utility Landscape for Gray Full-Field Stimuli (Cell {CELLID})', fontsize=14)
plt.tight_layout(rect=[0, 0.08, 1, 0.96])  # Leave space for parameter box at bottom

# Save figure
save_path = Path(__file__).parent / 'utility_landscape_gray.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"  Figure saved to: {save_path}")

plt.close()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Model R²: {r2:.4f} +/- {sigma_r2:.4f}")
print(f"Sample gray: {SAMPLE_GRAY_LEVEL*100:.0f}% (pixel value: {sample_gray_value.item():.4f})")
print(f"Query points: {N_QUERY_POINTS} gray levels from 0% to 100%")
print(f"Mode: {'DETERMINISTIC (posterior mean)' if USE_POSTERIOR_MEAN else f'MC SAMPLING ({LAMBDA_SAMPLES} samples)'}")
print(f"Mask size: {mask.sum().item()} pixels")
print(f"\nUtility landscape:")
print(f"  Max utility: {max_utility:.6f} at {max_gray_frac*100:.1f}% gray")
print(f"  Min utility: {utilities.min().item():.6f} at {query_gray_fracs[torch.argmin(utilities)].item()*100:.1f}% gray")
print(f"  Utility at sample gray ({SAMPLE_GRAY_LEVEL*100:.0f}%): {utilities[int(SAMPLE_GRAY_LEVEL * (N_QUERY_POINTS-1))].item():.6f}")
print("\n" + "=" * 60)
