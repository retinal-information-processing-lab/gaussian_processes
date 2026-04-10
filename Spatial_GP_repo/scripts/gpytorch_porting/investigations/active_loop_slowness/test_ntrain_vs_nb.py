"""
Investigation: does n_train affect n_b when inducing points are held fixed?

Apples-to-apples comparison:
  - SAME 300 inducing points (from active loop cell 8 seed 42 checkpoint)
  - SAME init kernel params (beta=0.1, rho=0.1 from default_params.json)
  - SAME cell, seed, dataset
  - VARY ONLY n_train: 300 (= M, active loop regime) vs 3160 (full pool)

If n_b differs at convergence: the M-step converges to different kernel params
depending on n_train, which in turn changes the eigenvalue spectrum.
If n_b is the same: something else is causing the active loop's n_b ≈ M behavior.
"""
import sys
import json
import torch
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_single_mode import build_config_from_defaults, load_pnas_data
from eigenspace_checkpoint import load_pool_indices
from eigenspace_training import train_eigenspace, predict_eigenspace
from eigenspace_model import DirectVGPModel
from kernels import create_kernel
from utils import apply_rf_center_bounds
from likelihoods import PoissonLikelihood
from utils import compute_rf_center_from_sta
from tests.test_utils import set_reproducible_seed
from metrics import compute_pearson_correlation as compute_pearson_r

# --- Config from default_params.json ---
config = build_config_from_defaults(
    mode='vargp_direct',
    cell=8,
    seed=42,
    M=300,
    data_path='datasets/PNAS_108x108_original.npz',
)
device = 'cuda'
dtype = torch.float32
seed = config['seed']
cell = config['cell']

# --- Load data ---
data_path = Path(__file__).resolve().parents[2] / config['data_path']
data = load_pnas_data(data_path, dtype=dtype)
X_pool = torch.cat([data['X_train'], data['X_val']], dim=0)
X_pool = X_pool.reshape(X_pool.shape[0], -1).to(device)
R_pool = torch.cat([data['R_train'], data['R_val']], dim=0).to(device)
r_pool = R_pool[:, cell]
X_test = data['X_test'].reshape(data['X_test'].shape[0], -1).to(device)
r_test = data['R_test'][:, :, cell].to(device)
r_test_mean = r_test.float().mean(dim=0)

n_px_side = 108
config['n_px_side'] = n_px_side

# --- Load the SAME 300 inducing point indices from the active loop checkpoint ---
ckpt_path = Path(__file__).resolve().parents[2] / \
    'results/active_loop/2026-04-08_cell8_seed42_M50_n250/argmax/checkpoints/iter_250.pt'
tol = config['pool_sum_tolerance']
ip_indices = load_pool_indices(ckpt_path, X_pool, tol).to(device)
print(f"Loaded {ip_indices.shape[0]} inducing point indices from active loop checkpoint")
inducing_points = X_pool[ip_indices]

# --- RF center from ground truth ---
import numpy as np
rf = np.load(Path(__file__).resolve().parents[2] / 'datasets/rf_centers_ground_truth.npz')
eps_0x = float(rf['norm_108'][cell][0])
eps_0y = float(rf['norm_108'][cell][1])
print(f"RF center (ground truth): ({eps_0x:.4f}, {eps_0y:.4f})")

# --- Build the n_train=1500 subset: the 300 IPs + 1200 random extras ---
# Use an isolated generator so this selection is reproducible and doesn't
# interfere with the training seed.
gen_extras = torch.Generator(device=device)
gen_extras.manual_seed(seed)
non_ip_mask = torch.ones(X_pool.shape[0], dtype=torch.bool, device=device)
non_ip_mask[ip_indices] = False
non_ip_indices = torch.nonzero(non_ip_mask, as_tuple=True)[0]
perm = torch.randperm(non_ip_indices.shape[0], generator=gen_extras, device=device)[:1200]
extra_indices = non_ip_indices[perm]
mid_indices = torch.cat([ip_indices, extra_indices])  # 300 IPs + 1200 extras = 1500
X_mid = X_pool[mid_indices]
r_mid = r_pool[mid_indices]

# --- Run three fits: same IPs, different n_train ---
for n_train_label, X_train, r_train in [
    ("n_train=300 (M==n_train)", X_pool[ip_indices], r_pool[ip_indices]),
    ("n_train=1500 (IPs + 1200 extras)", X_mid, r_mid),
    ("n_train=3160 (full pool)", X_pool, r_pool),
]:
    print(f"\n{'='*60}")
    print(f"  {n_train_label}")
    print(f"  Inducing points: {inducing_points.shape[0]} (SAME in both runs)")
    print(f"  Training points: {X_train.shape[0]}")
    print(f"{'='*60}")

    set_reproducible_seed(seed, device=device)

    # Create kernel and likelihood with SAME init params
    kernel = create_kernel(config, n_px_side, eps_0x, eps_0y).to(device=device, dtype=dtype)
    apply_rf_center_bounds(kernel, eps_0x, eps_0y, config)
    likelihood = PoissonLikelihood(
        A_init=config['A_init'],
        lambda0_init=config['lambda0_init'],
    ).to(device=device, dtype=dtype)

    # Create model with the SAME inducing points
    model = DirectVGPModel(
        kernel, likelihood, X_train, inducing_points,
        eigval_tol=config['eigval_tol'],
    )
    n_b_init = len(model.state.eigvals_b)
    print(f"  n_b at init: {n_b_init} / M={inducing_points.shape[0]} "
          f"(ratio={n_b_init/inducing_points.shape[0]:.3f})")

    # Train (eigval_tol, jitter, cholesky_max_tries, lambda_var_clamp are
    # read from the model or from _constants.py, not passed to train_eigenspace)
    result = train_eigenspace(
        model, r_train,
        n_iterations=config['n_iterations'],
        n_estep=config['n_estep'],
        n_fstep=config['n_fstep'],
        n_mstep=config['n_mstep'],
        lr_f=config['lr'],
        lr_m=config['lr'],
        f_mean_max_threshold=config['f_mean_max_threshold'],
        f_mean_mean_threshold=config['f_mean_mean_threshold'],
        early_stop=config['early_stop'],
        patience=config['patience'],
        min_delta_rel=config['min_delta_rel'],
        min_iterations=config['min_iterations'],
        restore_best=config['restore_best'],
        es_metric=config['es_metric'],
        print_every=10,
        verbose=False,
    )
    model = result['model']
    n_b_final = len(model.state.eigvals_b)

    # Predict on test set (jitter/cholesky/clamp read from _constants.py)
    with torch.no_grad():
        pred = predict_eigenspace(model, X_test)
    test_r = compute_pearson_r(pred['f_pred'], r_test_mean)

    k = model.kernel
    print(f"\n  RESULT:")
    print(f"    n_b_final: {n_b_final} / M=300 (ratio={n_b_final/300:.3f})")
    print(f"    beta:      {k.beta.item():.4f}")
    print(f"    rho:       {k.rho.item():.4f}")
    print(f"    sigma_0:   {k.sigma_0.item():.4f}")
    print(f"    Amp:       {k.Amp.item():.4f}")
    print(f"    A:         {model.likelihood.A.item():.4f}")
    print(f"    lambda0:   {model.likelihood.lambda0.item():.4f}")
    print(f"    test_r:    {test_r:.4f}")
    print(f"    stopped:   iter {result.get('best_iteration', '?')}")