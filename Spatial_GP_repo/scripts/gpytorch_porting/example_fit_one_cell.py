#!/usr/bin/env python3
"""
example_fit_one_cell.py - Fit the GP to ONE cell and score it on held-out test images.

A short, linear walk through the engine on the Goldin et al. (2023) dataset:

    load the data -> RF centre from the STA -> choose the inducing points and the
    training set -> build kernel, likelihood and model -> train (EM) -> predict the
    test images -> score

It is the `vargp_direct` path of run_single_mode.py with everything else removed. For
the same settings it gives the same numbers as
    python run_single_mode.py --mode vargp_direct
run_single_mode.py remains the full runner (plots, the other training modes, every
option on the command line).

Settings: every value is read from default_params.json. The command-line options
override the few a first run usually changes; their defaults are the JSON values.

Data: datasets/PNAS_108x108_original.npz, not in git (see PAPER_ENGINE.md at the
repository root).

Usage:
    python example_fit_one_cell.py
    python example_fit_one_cell.py --cell 12 --n-train 1000 --ntilde 200
    python example_fit_one_cell.py --device cpu

Notation (Goldin et al. 2023):
    X        images, one row per image, shape (N, n_pixels)
    r        spike counts of the cell, one per image, shape (N,)
    X_tilde  inducing points, a subset of the training images, shape (ntilde, n_pixels)
    f_pred   predicted mean spike count for each test image, shape (n_test,)
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from kernels import create_kernel
from likelihoods import PoissonLikelihood
from eigenspace_model import DirectVGPModel
from eigenspace_training import train_eigenspace, predict_eigenspace
from metrics import compute_pearson_correlation, compute_adjusted_r_squared
from utils import (compute_rf_center_from_sta, select_inducing_points_pivoted,
                   apply_rf_center_bounds)
from tests.test_utils import set_reproducible_seed

HERE = Path(__file__).resolve().parent


def main(argv=None):
    # ------------------------------------------------------------------
    # Settings: default_params.json, plus the few command-line overrides
    # ------------------------------------------------------------------
    with open(HERE / 'default_params.json', 'r') as f:
        defaults = json.load(f)
    ker = defaults['kernel']
    lik = defaults['link_function']
    trn = defaults['training']
    es = defaults['early_stopping']
    mod = defaults['model']
    dat = defaults['data']
    ind = defaults['inducing']

    parser = argparse.ArgumentParser(description='Fit the GP to one cell (Goldin et al. 2023 data)')
    parser.add_argument('--data-path', type=str, default=dat['path'],
                        help='npz file; a relative path is resolved against this folder')
    parser.add_argument('--cell', type=int, default=dat['cellid'], help='index of the cell to fit')
    parser.add_argument('--n-train', type=int, default=dat['n_train'], help='number of training images')
    parser.add_argument('--ntilde', type=int, default=dat['ntilde'], help='number of inducing points')
    parser.add_argument('--ip-selection', type=str, default=ind['selection_method'],
                        choices=['pivoted', 'random'],
                        help='how the inducing points are chosen among the images')
    parser.add_argument('--seed', type=int, default=dat['seed'])
    parser.add_argument('--device', type=str, default=defaults['run']['device'], help='cuda or cpu')
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    dtype = torch.float32 if defaults['run']['dtype'] == 'float32' else torch.float64
    set_reproducible_seed(args.seed, device=device)

    # ------------------------------------------------------------------
    # 1. Data. The file holds a training set and a validation set of images shown
    #    once each, and 30 test images shown 30 times each.
    # ------------------------------------------------------------------
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = HERE / data_path
    data = np.load(data_path)

    X = torch.cat([torch.tensor(data['images_train'], dtype=dtype),
                   torch.tensor(data['images_val'], dtype=dtype)], dim=0)
    R = torch.cat([torch.tensor(data['responses_train'], dtype=dtype),
                   torch.tensor(data['responses_val'], dtype=dtype)], dim=0)
    n_px_side = X.shape[1]                              # images are (N, n_px_side, n_px_side, 1)
    X = X.reshape(X.shape[0], -1).to(device)            # (N, n_pixels)
    R = R.to(device)                                    # (N, n_cells)
    X_test = torch.tensor(data['images_test'], dtype=dtype)
    X_test = X_test.reshape(X_test.shape[0], -1).to(device)             # (n_test, n_pixels)
    R_test = torch.tensor(data['responses_test'], dtype=dtype).to(device)  # (n_repeats, n_test, n_cells)

    r = R[:, args.cell]                                 # (N,) spike counts of this cell
    r_test = R_test[:, :, args.cell]                    # (n_repeats, n_test)
    print(f"Data: {X.shape[0]} images of {n_px_side}x{n_px_side} px, {R.shape[1]} cells; "
          f"fitting cell {args.cell} on {device}")

    # ------------------------------------------------------------------
    # 2. Initial RF centre (eps_0x, eps_0y), in normalized image coordinates [-1, 1]:
    #    the peak of the smoothed spike-triggered average over all the images.
    # ------------------------------------------------------------------
    eps_0x, eps_0y = compute_rf_center_from_sta(X, r, n_px_side, zscore=True)
    print(f"RF centre from the STA: ({eps_0x:.4f}, {eps_0y:.4f})")

    # Kernel settings, in the flat form create_kernel() and apply_rf_center_bounds() read
    kernel_config = {
        'kernel_type': ker['type'], 'sigma_0': ker['sigma_0'], 'Amp': ker['Amp'],
        'beta': ker['beta'], 'rho': ker['rho'], 'use_mask': ker['use_mask'],
        'gradient_mode': ker['gradient_mode'], 'lengthscale': ker['lengthscale'],
        'bound_rf_center': ker['bound_rf_center'], 'n_sigma_rf_bounds': ker['n_sigma_rf_bounds'],
        'n_samples_sta': ind['n_samples_sta'],
    }

    # ------------------------------------------------------------------
    # 3. Inducing points X_tilde: ntilde of the images.
    #    'pivoted' picks the images that are most different under the kernel
    #    (pivoted Cholesky); 'random' picks a random subset.
    # ------------------------------------------------------------------
    n_train = min(args.n_train, X.shape[0])
    if args.ip_selection == 'pivoted':
        temp_kernel = create_kernel(kernel_config, n_px_side, eps_0x, eps_0y).to(device=device, dtype=dtype)
        ntilde = min(args.ntilde, X.shape[0])
        X_tilde, indices_inducing = select_inducing_points_pivoted(
            X, temp_kernel, ntilde,
            n_candidates=ind['n_candidates'], seed=args.seed, jitter=mod['jitter'])
        del temp_kernel
    else:
        ntilde = min(args.ntilde, n_train)
        indices_inducing = torch.randperm(X.shape[0], device=device)[:ntilde]
        X_tilde = X[indices_inducing].clone()

    # ------------------------------------------------------------------
    # 4. Training set: the inducing points, plus random other images up to n_train.
    #    With --ntilde equal to --n-train the inducing set IS the training set.
    # ------------------------------------------------------------------
    if n_train > ntilde:
        inducing_set = set(indices_inducing.cpu().numpy().tolist())
        remaining = torch.tensor([i for i in range(X.shape[0]) if i not in inducing_set], device=device)
        extra = remaining[torch.randperm(remaining.shape[0], device=device)[:n_train - ntilde]]
        indices_train = torch.cat([indices_inducing, extra])
    else:
        indices_train = indices_inducing
        n_train = ntilde
    X_train = X[indices_train]
    r_train = r[indices_train]
    print(f"Training images: {n_train}; inducing points: {ntilde} ({args.ip_selection})")

    # ------------------------------------------------------------------
    # 5. Model: kernel (arc-cosine with a localized receptive field), Poisson
    #    likelihood with an exponential link  f = exp(A * lambda + lambda0),
    #    and the variational GP that owns both.
    # ------------------------------------------------------------------
    kernel = create_kernel(kernel_config, n_px_side, eps_0x, eps_0y)
    apply_rf_center_bounds(kernel, eps_0x, eps_0y, kernel_config)   # keep the centre near the STA peak
    kernel = kernel.to(dtype=dtype, device=device)
    likelihood = PoissonLikelihood(A_init=lik['A_init'], lambda0_init=lik['lambda0_init'])
    likelihood = likelihood.to(dtype=dtype, device=device)
    model = DirectVGPModel(kernel, likelihood, X_train, X_tilde, mod['eigval_tol'],
                           lambda_var_clamp=mod['lambda_var_clamp'])

    # ------------------------------------------------------------------
    # 6. Training: EM. Each iteration runs an E-step (variational posterior), an
    #    F-step (link parameters A, lambda0) and an M-step (kernel hyperparameters).
    #    A step that crashes is skipped with a warning and the fit carries on from the
    #    values before that step, so the warnings are collected and reported below.
    # ------------------------------------------------------------------
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with torch.enable_grad():
            result = train_eigenspace(
                model, r_train,
                n_iterations=trn['n_iterations'],
                n_estep=trn['n_estep'], n_fstep=trn['n_fstep'], n_mstep=trn['n_mstep'],
                lr_f=trn['lr'], lr_m=trn['lr'],
                print_every=max(1, trn['n_iterations'] // 5),
                use_analytical_mstep=False,
                early_stop=es['enabled'], stop_window=es['window'],
                stop_thresh=es['threshold'], min_iterations=es['min_iterations'],
                stability_threshold=mod['stability_threshold'],
            )
    crashed_steps = [str(w.message) for w in caught if 'crashed' in str(w.message)]
    for w in caught:
        print(f"WARNING during training: {w.message}")

    # ------------------------------------------------------------------
    # 7. Prediction and score on the test images.
    #    test r        Pearson correlation between f_pred and the mean response over repeats
    #    adjusted R^2  Goldin et al. 2023, Eq. 5 (corrects for the cell's trial-to-trial noise)
    # ------------------------------------------------------------------
    f_pred = predict_eigenspace(model, X_test)['f_pred']
    test_r = compute_pearson_correlation(r_test.mean(dim=0), f_pred)
    adjusted_r2 = compute_adjusted_r_squared(r_test, f_pred)
    train_r = compute_pearson_correlation(r_train, predict_eigenspace(model, X_train)['f_pred'])

    print(f"\nEM iterations run: {result['final_iteration']}"
          f"{' (stopped early)' if result['stopped_early'] else ''}; "
          f"kept eigen-directions n_b = {len(model.state.eigvals_b)} of ntilde = {ntilde}")
    print("Fitted parameters:")
    print(f"  link     A = {likelihood.A.item():.4f}, lambda0 = {likelihood.lambda0.item():.4f}")
    print(f"  kernel   sigma_0 = {kernel.sigma_0.item():.4f}, Amp = {kernel.Amp.item():.4f}, "
          f"beta = {kernel.beta.item():.4f}, rho = {kernel.rho.item():.4f}, "
          f"centre = ({kernel.eps_0x.item():.4f}, {kernel.eps_0y.item():.4f})")
    print("Score:")
    print(f"  train Pearson r = {float(train_r):.4f}")
    print(f"  test Pearson r  = {float(test_r):.4f}")
    print(f"  adjusted R^2    = {float(adjusted_r2):.4f}")
    if crashed_steps:
        print(f"\n{len(crashed_steps)} optimizer step(s) crashed and were skipped. Do not trust this "
              f"fit before finding out why (a GPU out-of-memory error is the usual cause).")

    return {'model': model, 'f_pred': f_pred, 'indices_train': indices_train,
            'test_r': float(test_r), 'adjusted_r2': float(adjusted_r2), 'train_r': float(train_r),
            'losses': result['losses'], 'crashed_steps': crashed_steps}


if __name__ == '__main__':
    main()
