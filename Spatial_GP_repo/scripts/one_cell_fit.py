#!/usr/bin/env python3
"""
One Cell Fit Script - Created by Claude

This script fits a Gaussian Process model to neural response data from a single cell
using variational inference. It replicates the functionality of the notebook:
notebooks/one_cell_fit.ipynb

User request: Create a standalone script that performs the same operations as the
one_cell_fit.ipynb notebook - loading data, fitting a variational GP model to
predict neural responses from images, testing on held-out data, and visualizing results.

Key differences from notebook:
- Uses float32 dtype to match config.config (fixes dtype mismatch bug in notebook)
- Saves plots to files instead of displaying interactively
- No Jupyter magic commands
"""

import sys
import time
import random
import pickle
import numpy as np

import torch
torch.set_grad_enabled(False)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script
import matplotlib.pyplot as plt

from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject')
# Add torchlambertw library path (required by utils.py)
sys.path.insert(0, '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/torchlambertw')
from gaussian_processes.Spatial_GP_repo import utils as GP_utils

# =============================================================================
# Configuration
# =============================================================================

# IMPORTANT: Use float32 to match config.config settings
# The notebook used float64 which caused dtype mismatch with localker()
# which uses TORCH_DTYPE from config (float32)
TORCH_DTYPE = torch.float32
torch.set_default_dtype(TORCH_DTYPE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print(f'Device: {device}')

# Set seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================================================================
# Training Parameters
# =============================================================================

rand_xtilde = True  # If True, xtilde (inducing points) are chosen randomly

cellid = 8               # Choose cell
ntilde = 40              # Number of inducing points (xtilde)
additional_x_number = 0  # Additional x points to be added
kernfun = GP_utils.acosker  # Choose kernel function (must be the actual function, not a string)

nEstep = 10              # Total number of E-steps iterations
nFparamstep = 10         # Number of iterations for f params update per E-step
nMstep = 0               # Total number of M-steps iterations
maxiter = 10             # Iterations of the optimization algorithm

ntrain_start = 40       # Number of training images (can differ from ntilde)


# =============================================================================
# Import Dataset and Preprocess
# =============================================================================

print('Loading dataset...')

# Load the .npz dataset file (converted from original pickle format)
# The original notebook used a pickle file with a custom Dataset class,
# but we use the .npz format for better portability
data_path = Path(__file__).parent.parent / 'notebooks' / 'PNAS_paper_sorted_data.npz'
data = np.load(data_path)

# Convert to tensors with proper dtype and device
X_train = torch.tensor(data['images_train'], dtype=TORCH_DTYPE, device=device)
X_val = torch.tensor(data['images_val'], dtype=TORCH_DTYPE, device=device)
X_test = torch.tensor(data['images_test'], dtype=TORCH_DTYPE, device=device)

R_train = torch.tensor(data['responses_train'], dtype=TORCH_DTYPE, device=device)
R_val = torch.tensor(data['responses_val'], dtype=TORCH_DTYPE, device=device)
R_test = torch.tensor(data['responses_test'], dtype=TORCH_DTYPE, device=device)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'R_train shape: {R_train.shape}')
print(f'R_test shape: {R_test.shape}')

# Create the complete dataset
X = torch.cat((X_train, X_val), axis=0)
R = torch.cat((R_train, R_val), axis=0)

n_px_side = X.shape[1]

# Reshape images to 1D vector and choose a cell
X = torch.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))
R = R[:, cellid]  # shape (nt,) where nt is the number of trials
R_test = R_test[:, :, cellid]  # shape (repetitions, nimages) for test

# Choose a random subset of the data and save the idx
all_idx = torch.arange(0, X.shape[0])
torch.manual_seed(0)
torch.cuda.manual_seed(0)
all_idx_perm = torch.randperm(all_idx.shape[0])

test_1000_idx = all_idx_perm[-1000:]
all_idx_perm = all_idx_perm[~torch.isin(all_idx_perm, test_1000_idx)]
rndm_idx = all_idx_perm[:ntrain_start]

start_idx = rndm_idx
in_use_idx = start_idx
# Inducing points are a subset of training data (first ntilde points)
xtilde_idx = in_use_idx[:ntilde]
remaining_idx = all_idx_perm[~torch.isin(all_idx_perm, in_use_idx)]

# Set the starting set
xtilde_start = X[xtilde_idx, :]  # Shape: (ntilde, n_pixels)
X_in_use = X[in_use_idx, :]
X_remaining = X[remaining_idx, :]
X_test_1000 = X[test_1000_idx, :]

R_remaining = R[remaining_idx]
R_in_use = R[in_use_idx]
R_test_1000 = R[test_1000_idx]

# Estimate memory usage
X_memory = X.element_size() * X.nelement()
r_memory = R.element_size() * R.nelement()
total_memory_bytes = X_memory + r_memory
total_memory_MB = total_memory_bytes / (1024 ** 2)
print(f'Total dataset memory on GPU: {total_memory_MB:.2f} MB')


# =============================================================================
# Preprocessing Diagnostic
# =============================================================================
print("\n[Preprocessing Diagnostic]")
print("-" * 40)

# Full dataset statistics
print(f"  Full dataset (train+val):")
print(f"    Pixel range: [{X.min().item():.4f}, {X.max().item():.4f}]")
print(f"    Mean: {X.mean().item():.4f}, Std: {X.std().item():.4f}")

# Check if z-scored
is_zscored = abs(X.mean().item()) < 0.1 and 0.9 < X.std().item() < 1.1
print(f"    Z-scored: {'YES' if is_zscored else 'NO'}")

# Training subset statistics
X_train_subset = X[in_use_idx]
print(f"\n  Training subset ({len(in_use_idx)} images):")
print(f"    Pixel range: [{X_train_subset.min().item():.4f}, {X_train_subset.max().item():.4f}]")
print(f"    Mean: {X_train_subset.mean().item():.4f}, Std: {X_train_subset.std().item():.4f}")

# Range coverage
full_range = X.max().item() - X.min().item()
subset_range = X_train_subset.max().item() - X_train_subset.min().item()
coverage = (subset_range / full_range) * 100
print(f"    Range coverage: {coverage:.1f}% of full dataset range")

# Gray level reference
print(f"\n  Gray level reference (in z-scored space):")
print(f"    Black (0%): {X.min().item():.4f}")
print(f"    Mid-gray (50%): {(X.min().item() + X.max().item()) / 2:.4f}")
print(f"    White (100%): {X.max().item():.4f}")


# =============================================================================
# Choose Starting Values of Parameters
# =============================================================================

torch.set_grad_enabled(False)

# Set beta and rho directly
beta = torch.tensor(0.1)
rho = torch.tensor(0.1)
logbetaexpr = -2 * torch.log(2 * beta)
logrhoexpr = -torch.log(2 * rho * rho)

sigma_0 = torch.tensor(1.)
Amp = torch.tensor(1.0)

# Center of receptive field
eps_0x = torch.tensor(0.0001)
eps_0y = torch.tensor(0.0001)

# Hyperparameters dictionary
theta = {
    'sigma_0': sigma_0,
    'Amp': Amp,
    'eps_0x': eps_0x,
    'eps_0y': eps_0y,
    '-2log2beta': logbetaexpr,
    '-log2rho2': logrhoexpr
}

# Set the gradient of the hyperparameters to be updateable
for key, value in theta.items():
    theta[key] = value.requires_grad_()

# Generate the hyperparameters tuple
hyperparams_tuple = GP_utils.generate_theta(
    x=X_in_use, r=R_in_use, n_px_side=n_px_side, display_hyper=False, **theta
)

# Link function parameters
A = torch.tensor(0.01)
logA = torch.log(A)
lambda0 = torch.tensor(1.)
f_params = {'logA': logA, 'lambda0': lambda0}
f_params['logA'] = f_params['logA'].requires_grad_()

# Fit parameters
fit_parameters = {
    'ntilde': ntilde,
    'maxiter': maxiter,
    'nMstep': nMstep,
    'nEstep': nEstep,
    'nFparamstep': nFparamstep,
    'kernfun': kernfun,
    'cellid': cellid,
    'n_px_side': n_px_side,
}

args = {
    'fit_parameters': fit_parameters,
    'xtilde': xtilde_start,
    'hyperparams_tuple': hyperparams_tuple,
    'f_params': f_params,
    'm': torch.zeros((ntilde))
}


# =============================================================================
# Fit the Model
# =============================================================================

print('\n' + '=' * 60)
print('Fitting the model...')
print('=' * 60)

torch.set_grad_enabled(False)

start_time = time.time()
fit_model, err_dict = GP_utils.varGP(X_in_use, R_in_use, **args)
elapsed_time = time.time() - start_time

print(f'\nFitting completed in {elapsed_time:.2f} seconds')

if err_dict['is_error']:
    print(f'Error during fitting: {err_dict["error"]}')
    raise err_dict['error']


# =============================================================================
# Test the Model
# =============================================================================

print('\n' + '=' * 60)
print('Testing the model...')
print('=' * 60)

spk_count_test, spk_count_pred, r2, sigma_r2 = GP_utils.test(
    X_test, R_test, X_train=X, at_iteration=None, **fit_model
)

print(f'\nResults:')
print(f'  R^2 = {r2:.4f} +/- {sigma_r2:.4f}')
print(f'  Cell: {cellid}')
print(f'  ntilde: {ntilde}')


# =============================================================================
# Visualization
# =============================================================================

print('\n' + '=' * 60)
print('Creating visualization...')
print('=' * 60)

# Create output directory
output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(parents=True, exist_ok=True)

# Plot loss and hyperparameters
fig = plt.figure(figsize=(15, 10))
GP_utils.plot_loss_and_theta_notebook(fit_model, figsize=(15, 10), marker='o')

# Save the figure
output_path = output_dir / f'one_cell_fit_cell{cellid}_loss_and_theta.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Plot saved to: {output_path}')


# =============================================================================
# Plot Test Results
# =============================================================================

# Convert to numpy for plotting
spk_count_test_np = spk_count_test.cpu().numpy()
spk_count_pred_np = spk_count_pred.cpu().numpy()
r2_np = r2.cpu().numpy() if hasattr(r2, 'cpu') else r2
sigma_r2_np = sigma_r2.cpu().numpy() if hasattr(sigma_r2, 'cpu') else sigma_r2

spk_count_test_mean = np.mean(spk_count_test_np, axis=0)

# Convert to firing rate
predict_window_ms = 320
f_rate_test_Hz = spk_count_test_mean / (predict_window_ms / 1000)
f_rate_pred_Hz = spk_count_pred_np / (predict_window_ms / 1000)

dt = predict_window_ms / 1000
time_values = dt * np.arange(len(spk_count_pred_np))

# Create the plot
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(5, 5, left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.7)
ax = fig.add_subplot(gs[3:, :])

ax.plot(time_values, f_rate_test_Hz, 'o-', linewidth=1, label='Observed')
ax.plot(time_values, f_rate_pred_Hz, 'o-', color='red', label='Predicted (GP)')

txt = f'R^2 = {r2_np:.2f} +/- {sigma_r2_np:.2f} | Cell: {cellid} | ntilde: {ntilde}'
ax.set_title(txt)
ax.set_ylabel('Firing rate (Hz)')
ax.set_xlabel('Time (s)')
ax.grid(axis='both', alpha=0.3)
ax.legend()

# Save the test results plot
output_path = output_dir / f'one_cell_fit_cell{cellid}_test_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Test results plot saved to: {output_path}')

print('\n' + '=' * 60)
print('Done!')
print('=' * 60)


# =============================================================================
# Save Model (optional)
# =============================================================================

SAVE_MODEL = True  # Set to True to save the fitted model

if SAVE_MODEL:
    print('\n' + '=' * 60)
    print('Saving model...')
    print('=' * 60)

    # Save to the data/models directory
    models_dir = Path(__file__).parent.parent / 'data' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    # Model filename includes cell ID, ntilde and ntrain for easy identification
    model_name = f'model_cell:{cellid}_ntilde:{ntilde}_ntrain:{ntrain_start}'
    model_path = models_dir / model_name

    # Add indices to fit_parameters so GPModel.from_dict() can find them
    fit_model['fit_parameters']['in_use_idx'] = in_use_idx
    fit_model['fit_parameters']['xtilde_idx'] = xtilde_idx
    fit_model['fit_parameters']['remaining_idx'] = remaining_idx

    # Save using GP_utils.save_model (takes directory and name separately)
    additional_desc = f'r2={r2_np:.4f}, ntilde={ntilde}, maxiter={maxiter}'
    GP_utils.save_model(fit_model, models_dir, model_name, additional_description=additional_desc)

    print(f'  Model saved to: {model_path}')
    print(f'  Description: {additional_desc}')


# =============================================================================
# Summary
# =============================================================================

print(f'\nSummary:')
print(f'  Model fitted on {ntrain_start} training images')
print(f'  Number of inducing points: {ntilde}')
print(f'  E-steps: {nEstep}, M-steps: {nMstep}, F-param steps: {nFparamstep}')
print(f'  Max iterations: {maxiter}')
print(f'  Test R^2: {r2_np:.4f} +/- {sigma_r2_np:.4f}')
print(f'  Outputs saved to: {output_dir}')
if SAVE_MODEL:
    print(f'  Model saved to: {model_path}')
