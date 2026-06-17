"""
default_gpy model training + data loading for the lucent useful-image experiment.

Why default_gpy: the distribution-aware utility (acquisition.py) requires
model(X).covariance_matrix for Gaussian conditioning, which only the default_gpy
(GPyTorch VariationalGPModel) mode exposes. vargp_direct's EigenspacePosterior
returns mean/variance only.

All model/training params trace to default_params.json via build_config_from_defaults.
Experimental overrides (documented):
  - n_train  -> the model-quality axis (the user's chosen ladder).
  - M (inducing points) = n_train AT ALL TIMES: every training point is an inducing
    point => the full (non-sparse) variational GP at each training size. This is the
    standing convention for this investigation; do not use a fixed low M here.
  - seed = 42 (the default_params default).
"""
import os
import sys
import contextlib
import numpy as np
import torch

GP_PORT = "/home/idv-eqs8-pza/IDV_code/ClosedLoop-standalone-analysis_april26/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting"
if GP_PORT not in sys.path:
    sys.path.insert(0, GP_PORT)

# The configured dataset path (datasets/PNAS_108x108_original.npz) is a symlink that
# is absent in THIS worktree. Point at the canonical file in the sibling repo.
DATA = "/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/notebooks/PNAS_paper_sorted_data.npz"

from run_single_mode import build_config_from_defaults, run_single_config  # noqa: E402

IMG_SIDE = 108
N_PIX = IMG_SIDE * IMG_SIDE


def load_pool(device="cuda"):
    """Load the train+val image pool exactly as run_single_config builds X.

    Returns:
        X_all: (3160, 11664) float32 on `device` -- the natural-image pool.
        gp_min, gp_max: global min/max over X_all (affine-map endpoints).
    """
    d = np.load(DATA)
    X_all = torch.cat(
        [torch.tensor(d["images_train"], dtype=torch.float32),
         torch.tensor(d["images_val"], dtype=torch.float32)],
        dim=0,
    ).reshape(-1, N_PIX).to(device)
    gp_min = float(X_all.min())
    gp_max = float(X_all.max())
    return X_all, gp_min, gp_max


def pool_complement(X_all, idx_train):
    """Return the held-out pool = all images NOT used for training."""
    n = X_all.shape[0]
    if torch.is_tensor(idx_train):
        idx_train = idx_train.detach().cpu().numpy().tolist()
    pool_idx = sorted(set(range(n)) - set(int(i) for i in idx_train))
    return X_all[pool_idx], torch.tensor(pool_idx, dtype=torch.long)


def train_default_gpy(cell_id, n_train, M=None, seed=42, quiet=True):
    """Train a default_gpy model. Returns (model, likelihood, idx_train, test_r).

    M defaults to n_train (the standing convention: full inducing set, M = n_train).
    Pass an explicit M only for a deliberate sparse-GP experiment.
    model is in eval mode and on cuda; idx_train indexes into the (3160,) pool.
    """
    if M is None:
        M = n_train
    config = build_config_from_defaults(
        mode="default_gpy", cell=cell_id, n_train=n_train, M=M, seed=seed,
        data_path=DATA,
    )
    ctx = contextlib.redirect_stdout(open(os.devnull, "w")) if quiet else contextlib.nullcontext()
    with ctx:
        res = run_single_config(config)
    if res is None or res.get("status") != "success":
        status = None if res is None else res.get("status")
        raise RuntimeError(f"training failed cell={cell_id} n_train={n_train} M={M}: status={status}")
    model = res["_model"]
    likelihood = res["_likelihood"]
    model.eval()
    return model, likelihood, res["_indices_train"], float(res["test_r"])


if __name__ == "__main__":
    import time
    X_all, gp_min, gp_max = load_pool()
    print(f"pool {tuple(X_all.shape)}  gp_min={gp_min:.6f}  gp_max={gp_max:.6f}")
    t0 = time.time()
    model, lik, idx, test_r = train_default_gpy(cell_id=8, n_train=300, seed=42)  # M=n_train=300
    print(f"cell 8 n_train=300 M=n_train(300): test_r={test_r:.4f}  "
          f"A={float(lik.A):.4f} lambda0={float(lik.lambda0):.4f}  "
          f"({time.time()-t0:.1f}s)")
    X_pool, pool_idx = pool_complement(X_all, idx)
    print(f"X_pool {tuple(X_pool.shape)}")
