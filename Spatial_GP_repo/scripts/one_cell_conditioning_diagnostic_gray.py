"""
One Cell Conditioning Diagnostic - Gray Full-Field Stimuli
Created by Claude

Purpose: Visualize how the GP posterior over λ (latent firing rate) changes when
we observe λ at a sample point. This diagnostic shows "before" (marginal) and
"after" (conditional) bands, demonstrating that variance always decreases after
conditioning.

What this script shows:
- X-axis: Gray level (pixel value from pixel_min to pixel_max)
- Y-axis: λ (latent firing rate)
- Blue band: GP posterior λ(gray) BEFORE conditioning (marginal)
- Red band: GP posterior λ(gray) AFTER conditioning on λ(gray_sample) = λ_obs
- Green star: The observation point (gray_sample, λ_obs)
- Black dots: Training data (mean pixel value → observed response)

Expected behavior:
1. Variance always decreases after conditioning
2. Effect is strongest near the sample point, decays with distance
3. Mean shifts based on "innovation" (λ_obs - prior mean)

Output: conditioning_diagnostic_gray.png
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
from gaussian_processes.Spatial_GP_repo.model import GPModel

# =============================================================================
# Configuration - Adjust these parameters as needed
# =============================================================================
CELLID = 8
NTILDE = 40
NTRAIN = 40
N_QUERY_POINTS = 25               # Number of gray levels to evaluate across the domain
SAMPLE_GRAY_LEVEL = 0.3
USE_POSTERIOR_MEAN = True         # If False, sample λ_obs from posterior
RANDOM_SEED = 42                  # For reproducibility when sampling

# =============================================================================
# Setup
# =============================================================================
TORCH_DTYPE = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(TORCH_DTYPE)
torch.set_default_device(device)

# Reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

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
X_test_4d = torch.tensor(data['images_test'], dtype=TORCH_DTYPE)

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
R_test_cell = R_test[..., CELLID]

print(f"  Dataset: {X.shape[0]} images, {X.shape[1]} pixels")
print(f"  Cell ID: {CELLID}")
print(f"  Pixel range: [{X.min().item():.4f}, {X.max().item():.4f}]")

# =============================================================================
# 2. Load Pre-trained Model
# =============================================================================
print("\n[2] Loading Pre-trained Model")
print("-" * 40)

model_path = Path(__file__).parent.parent / 'data' / 'models' / f'model_cell:{CELLID}_ntilde:{NTILDE}_ntrain:{NTRAIN}'

if not model_path.exists():
    raise FileNotFoundError(
        f"Pre-trained model not found at {model_path}\n"
        "Please run one_cell_fit.py first to create the model."
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
# 3. Evaluate Model Quality on Test Set
# =============================================================================
print("\n[3] Evaluating Model Quality")
print("-" * 40)

_, _, r2, sigma_r2 = GP_utils.test(
    X_test_4d, R_test_cell,
    at_iteration=None,
    print_expl_var=False,
    **model.to_dict()
)

print(f"  Test set R²: {r2:.4f} +/- {sigma_r2:.4f}")

# =============================================================================
# 4. Get Mask and Eigenspace
# =============================================================================
print("\n[4] Getting Mask and Eigenspace")
print("-" * 40)

C, mask, K_tilde_b, K_tilde_inv_b, B = GP_utils.get_final_K_vals(model)

# Extract model parameters
theta = model.theta
xtilde = model.xtilde
m_b = model.m_b  # Already projected to eigenspace
V_b = model.V_b  # Already projected to eigenspace
kernfun = model.kernfun
n_pixels = X.shape[1]
n_b = B.shape[1]

print(f"  Mask size: {mask.sum().item()} pixels out of {len(mask)} total")
print(f"  Eigenspace dimension: {n_b}")

# =============================================================================
# 5. Define Gray Domain
# =============================================================================
print("\n[5] Setting Up Gray Domain")
print("-" * 40)

# Restrict to reasonable gray range around mean (0)
PLOT_PIXEL_MIN = -2.48
PLOT_PIXEL_MAX = 2.48
# PLOT_PIXEL_MIN = -2.48
# PLOT_PIXEL_MAX = 2.48

# Create query gray levels in the restricted range
query_gray_values = torch.linspace(PLOT_PIXEL_MIN, PLOT_PIXEL_MAX, N_QUERY_POINTS)

print(f"  Query gray levels: {N_QUERY_POINTS} points")
print(f"  Pixel range: [{PLOT_PIXEL_MIN:.4f}, {PLOT_PIXEL_MAX:.4f}]")

# =============================================================================
# 6. Compute BEFORE (Marginal) Moments at Each Gray Level
# =============================================================================
print("\n[6] Computing BEFORE (Marginal) Moments")
print("-" * 40)

lambda_m_before = []
lambda_var_before = []
u_g_b_list = []  # Store for use in AFTER computation
k_g_b_list = []  # Store for computing cross-kernel in AFTER
x_g_masked_list = []  # Store for computing k(x_g, x_sample)

for i, g_val in enumerate(query_gray_values):
    # Create uniform gray image at this level
    x_g = torch.full((1, n_pixels), g_val.item())
    x_g_masked = x_g[:, mask]

    # Compute kernel quantities
    k0_g = kernfun(theta, x_g_masked, None, C=C, diag=True)
    k_g = kernfun(theta, x_g_masked, xtilde[:, mask], C=C, diag=False)
    k_g_b = k_g @ B
    u_g_b = k_g_b * K_tilde_inv_b.diag()  # Shape: (1, n_b)

    # Marginal moments using lambda_moments formula
    # E[λ(x)] = u'·m
    lambda_m = (u_g_b @ m_b).item()

    # Var[λ(x)] = k0 + u'·(V - K)·u = k0 - u'·K·u + u'·V·u
    # In eigenspace: k0 - k_b·u_b + u_b·V_b·u_b
    lambda_var = (k0_g - (k_g_b * u_g_b).sum() + u_g_b @ V_b @ u_g_b.T).squeeze().item()

    lambda_m_before.append(lambda_m)
    lambda_var_before.append(lambda_var)
    u_g_b_list.append(u_g_b)
    k_g_b_list.append(k_g_b)
    x_g_masked_list.append(x_g_masked)

lambda_m_before = np.array(lambda_m_before)
lambda_var_before = np.array(lambda_var_before)
lambda_std_before = np.sqrt(np.maximum(lambda_var_before, 1e-10))

print(f"  Computed marginal moments at {N_QUERY_POINTS} gray levels")
print(f"  λ mean range: [{lambda_m_before.min():.4f}, {lambda_m_before.max():.4f}]")
print(f"  λ variance range: [{lambda_var_before.min():.6f}, {lambda_var_before.max():.6f}]")

# =============================================================================
# 7. Pick Sample Point and Sample λ_obs
# =============================================================================
print("\n[7] Setting Up Sample Point")
print("-" * 40)

# Sample at a point ON the query grid (snap to nearest grid point)
sample_idx = int(round(SAMPLE_GRAY_LEVEL * (N_QUERY_POINTS - 1)))
sample_pixel_value = query_gray_values[sample_idx].item()
print(f"  Sample index on grid: {sample_idx} (of {N_QUERY_POINTS-1})")

# Create sample image
x_sample = torch.full((1, n_pixels), sample_pixel_value)
x_sample_masked = x_sample[:, mask]

# Compute marginal moments at sample point
k0_sample = kernfun(theta, x_sample_masked, None, C=C, diag=True)
k_sample = kernfun(theta, x_sample_masked, xtilde[:, mask], C=C, diag=False)
k_sample_b = k_sample @ B
u_sample_b = k_sample_b * K_tilde_inv_b.diag()  # Shape: (1, n_b)

lambda_m_sample = (u_sample_b @ m_b).item()
lambda_var_sample = (k0_sample - (k_sample_b * u_sample_b).sum() + u_sample_b @ V_b @ u_sample_b.T).squeeze().item()
lambda_std_sample = np.sqrt(max(lambda_var_sample, 1e-10))

# Sample λ_obs (or use mean for deterministic version)
if USE_POSTERIOR_MEAN:
    lambda_obs = lambda_m_sample
    print(f"  Mode: DETERMINISTIC (using posterior mean)")
else:
    lambda_obs = lambda_m_sample + lambda_std_sample * np.random.randn()
    print(f"  Mode: SAMPLED (from posterior distribution)")

print(f"  Sample gray level: {SAMPLE_GRAY_LEVEL*100:.0f}% (pixel value: {sample_pixel_value:.4f})")
print(f"  Marginal λ at sample: mean={lambda_m_sample:.4f}, std={lambda_std_sample:.4f}")
print(f"  Observed λ_obs: {lambda_obs:.4f}")

# =============================================================================
# 8. Compute m' Update (Once, for Fixed Sample Point)
# =============================================================================
print("\n[8] Computing m' Update")
print("-" * 40)

# Schur complement: s = k(x,x) - k'·K⁻¹·k
s = k0_sample - (k_sample_b * u_sample_b).sum()
s = max(s.item(), 1e-6)  # Numerical safety
print(f"  Schur complement s: {s:.6f}")

# Vu for update: V @ u
Vu_sample = V_b @ u_sample_b.T  # Shape: (n_b, 1)

# Total marginal variance: denom = s + u'·V·u
uVu = (u_sample_b @ V_b @ u_sample_b.T).squeeze().item()
denom = s + uVu
print(f"  Denominator (s + u'Vu): {denom:.6f}")

# Innovation: δ = λ_obs - u'·m
delta = lambda_obs - lambda_m_sample
print(f"  Innovation δ: {delta:.6f}")

# Updated mean: m' = m + Vu·δ/denom
m_prime_b = m_b + (Vu_sample.squeeze() * delta / denom)  # Shape: (n_b,)

print(f"  m' update norm: {torch.norm(m_prime_b - m_b).item():.6f}")

# =============================================================================
# 9. Compute AFTER (Conditional) Moments at Each Gray Level
# =============================================================================
print("\n[9] Computing AFTER (Conditional) Moments")
print("-" * 40)

lambda_m_after = []
lambda_var_after = []
debug_cov_terms = []  # Store for debugging

for i, u_g_b in enumerate(u_g_b_list):
    # Conditional mean using m': λ_m_cond = u'·m'
    lambda_m_cond = (u_g_b @ m_prime_b).item()

    # Conditional variance: var_after = var_before - Cov(λ_g, λ_sample)² / Var(λ_sample)
    #
    # The FULL covariance in a variational GP is:
    #   Cov(λ_g, λ_sample) = k(x_g, x_sample) - k_g·u_sample + u_g'·V·u_sample
    #                        \_____________/   \___________/   \_____________/
    #                         prior kernel    sparse GP term   variational term
    #
    # When x_g = x_sample, this equals Var(λ_sample), so variance collapses to 0.

    # Compute cross-kernel k(x_g, x_sample)
    x_g_masked = x_g_masked_list[i]
    k_cross = kernfun(theta, x_g_masked, x_sample_masked, C=C, diag=False).item()  # scalar

    # Sparse GP term: k_g_b · u_sample_b (using stored k_g_b)
    k_g_b = k_g_b_list[i]
    sparse_term = (k_g_b * u_sample_b).sum().item()

    # Variational term: u_g' · V · u_sample = u_g' · Vu_sample
    var_term = (u_g_b @ Vu_sample).item()

    # Full covariance
    cov_full = k_cross - sparse_term + var_term

    # Store debug info
    debug_cov_terms.append({
        'pixel': query_gray_values[i].item(),
        'k_cross': k_cross,
        'sparse_term': sparse_term,
        'var_term': var_term,
        'cov_full': cov_full,
        'var_before': lambda_var_before[i]
    })

    lambda_var_cond = lambda_var_before[i] - (cov_full ** 2) / denom
    lambda_var_cond = max(lambda_var_cond, 1e-10)  # Numerical safety

    lambda_m_after.append(lambda_m_cond)
    lambda_var_after.append(lambda_var_cond)

# Print debug info for first, sample, and last points
print("\n  DEBUG: Covariance breakdown at key points:")
for idx in [0, sample_idx, N_QUERY_POINTS - 1]:
    d = debug_cov_terms[idx]
    marker = " <-- SAMPLE" if idx == sample_idx else ""
    print(f"    pixel={d['pixel']:.4f}: k_cross={d['k_cross']:.6f}, sparse={d['sparse_term']:.6f}, "
          f"var_term={d['var_term']:.6f} => cov_full={d['cov_full']:.6f}{marker}")

lambda_m_after = np.array(lambda_m_after)
lambda_var_after = np.array(lambda_var_after)
lambda_std_after = np.sqrt(lambda_var_after)

print(f"  Computed conditional moments at {N_QUERY_POINTS} gray levels")
print(f"  λ mean range: [{lambda_m_after.min():.4f}, {lambda_m_after.max():.4f}]")
print(f"  λ variance range: [{lambda_var_after.min():.6f}, {lambda_var_after.max():.6f}]")

# =============================================================================
# 10. Sanity Checks
# =============================================================================
print("\n[10] Sanity Checks")
print("-" * 40)

# Check 1: Variance must decrease everywhere
var_decreased = np.all(lambda_var_after <= lambda_var_before + 1e-8)
print(f"  [{'PASS' if var_decreased else 'FAIL'}] Variance decreased at all points")

# Check 2: Variance at sample point (sample_idx was set in Step 7)
var_reduction_at_sample = lambda_var_before[sample_idx] - lambda_var_after[sample_idx]
print(f"  Variance reduction at sample: {var_reduction_at_sample:.6f}")
print(f"  Variance at sample (before): {lambda_var_before[sample_idx]:.6f}")
print(f"  Variance at sample (after): {lambda_var_after[sample_idx]:.6f}")

# Check 3: Effect decays with distance
var_reduction = lambda_var_before - lambda_var_after
max_reduction_idx = np.argmax(var_reduction)
print(f"  Max variance reduction at pixel value: {query_gray_values[max_reduction_idx].item():.4f}")

# =============================================================================
# 11. Create Visualization
# =============================================================================
print("\n[11] Creating Visualization")
print("-" * 40)

# Convert to numpy for plotting
query_gray_values_np = query_gray_values.cpu().numpy()

# Create figure with 2 subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# ---------- Add faint dotted lines at query gray levels (both subplots) ----------
for ax in axes:
    for g_val in query_gray_values_np:
        ax.axvline(x=g_val, color='gray', linestyle=':', alpha=0.15, linewidth=0.5, zorder=0)

# ---------- Subplot 1: λ (latent) Before vs After ----------
ax1 = axes[0]

# BEFORE (blue)
ax1.plot(query_gray_values_np, lambda_m_before, 'b-', linewidth=2, label='Before: λ mean')
ax1.fill_between(query_gray_values_np,
                  lambda_m_before - 2*lambda_std_before,
                  lambda_m_before + 2*lambda_std_before,
                  alpha=0.25, color='blue', label='Before: ±2σ')

# AFTER (red)
ax1.plot(query_gray_values_np, lambda_m_after, 'r-', linewidth=2, label='After: λ mean')
ax1.fill_between(query_gray_values_np,
                  lambda_m_after - 2*lambda_std_after,
                  lambda_m_after + 2*lambda_std_after,
                  alpha=0.25, color='red', label='After: ±2σ')

# Observation point (green star)
ax1.scatter([sample_pixel_value], [lambda_obs], c='green', s=200, marker='*',
            zorder=5, edgecolors='darkgreen', linewidths=2,
            label=f'Observed λ = {lambda_obs:.3f}')
ax1.axvline(x=sample_pixel_value, color='green', linestyle='--', alpha=0.5)

# Training data (black dots)
ax1.scatter(train_mean_pixels.cpu().numpy(), R_train_used.cpu().numpy(),
            c='black', s=30, alpha=0.6, zorder=3, label=f'Training data ({len(R_train_used)} imgs)')

ax1.set_ylabel('λ (latent firing rate)')
ax1.set_title('GP Posterior: Before vs After Conditioning')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# ---------- Subplot 2: Variance reduction ----------
ax2 = axes[1]
ax2.plot(query_gray_values_np, var_reduction, 'purple', linewidth=2, label='Variance reduction')
ax2.fill_between(query_gray_values_np, 0, var_reduction, alpha=0.3, color='purple')
ax2.axvline(x=sample_pixel_value, color='green', linestyle='--', alpha=0.5,
            label=f'Sample at pixel={sample_pixel_value:.3f}')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_xlabel('Pixel Value')
ax2.set_ylabel('Var(λ)_before - Var(λ)_after')
ax2.set_title('Variance Reduction from Conditioning')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# ---------- Parameter Info Box ----------
mode_str = 'Deterministic (posterior mean)' if USE_POSTERIOR_MEAN else 'Sampled'
param_text = (
    f"MODEL: Cell {CELLID}, ntilde={NTILDE}, ntrain={NTRAIN}, R²={r2:.3f}\n"
    f"SAMPLE: pixel={sample_pixel_value:.3f}, "
    f"λ_obs={lambda_obs:.3f}, mode={mode_str}\n"
    f"DOMAIN: pixel ∈ [{PLOT_PIXEL_MIN:.2f}, {PLOT_PIXEL_MAX:.2f}], "
    f"{N_QUERY_POINTS} query points"
)
fig.text(0.5, 0.02, param_text, ha='center', va='bottom', fontsize=10,
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle(f'Conditioning Diagnostic for Gray Full-Field Stimuli (Cell {CELLID})', fontsize=14)
plt.tight_layout(rect=[0, 0.08, 1, 0.96])

# Save figure
save_path = Path(__file__).parent / 'conditioning_diagnostic_gray.png'
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
print(f"Sample pixel value: {sample_pixel_value:.4f}")
print(f"Observed λ: {lambda_obs:.4f} (mode: {mode_str})")
print(f"Query points: {N_QUERY_POINTS} gray levels in [{PLOT_PIXEL_MIN}, {PLOT_PIXEL_MAX}]")
print(f"\nVariance reduction statistics:")
print(f"  Max reduction: {var_reduction.max():.6f} at pixel={query_gray_values[np.argmax(var_reduction)].item():.4f}")
print(f"  Min reduction: {var_reduction.min():.6f} at pixel={query_gray_values[np.argmin(var_reduction)].item():.4f}")
print(f"  Reduction at sample (pixel={sample_pixel_value:.4f}): {var_reduction_at_sample:.6f}")
print(f"\nMean shift statistics:")
mean_shift = lambda_m_after - lambda_m_before
print(f"  Max mean shift: {mean_shift.max():.6f}")
print(f"  Min mean shift: {mean_shift.min():.6f}")
print(f"  Mean shift at sample: {mean_shift[sample_idx]:.6f}")
print("\n" + "=" * 60)
