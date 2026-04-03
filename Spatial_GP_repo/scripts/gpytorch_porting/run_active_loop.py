#!/usr/bin/env python3
"""
run_active_loop.py - Simulated active learning loop on PNAS neural data.

Evaluation harness for comparing utility optimization algorithms using the
vargp_direct (eigenspace-based DirectVGPModel) training mode.

Algorithm:
  1. Train initial GP on a small random set (Phase 1)
  2. LOOP (N iterations):
     a. Compute utility for all remaining images
     b. Select the image with highest utility
     c. Look up its ground-truth spike count (PNAS data)
     d. Extend model with new point (rank-1 warm-start)
     e. Retrain the GP (Phase 2: short fine-tuning)
     f. Evaluate on held-out test set
  3. Write per-iteration metrics to JSONL file

Usage:
    python run_active_loop.py                          # All defaults
    python run_active_loop.py --n-active 100 --cell 8  # Override iterations and cell
    python run_active_loop.py --phase1-M 50 --seed 42  # Override Phase 1 size and seed

All parameters trace to default_params.json (active_learning + utility sections).
"""

import sys
import copy
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add repo parent to sys.path for `from gaussian_processes.Spatial_GP_repo import ...`
_repo_root = next(p for p in Path(__file__).resolve().parents if (p / 'Spatial_GP_repo').is_dir())
sys.path.insert(0, str(_repo_root.parent))

import torch
import numpy as np

from run_single_mode import (
    build_config_from_defaults,
    run_single_config,
    load_pnas_data,
    set_reproducible_seed,
)
from eigenspace_training import train_eigenspace, predict_eigenspace
from rank1_update import extend_model_with_new_point
from metrics import (
    compute_pearson_correlation,
    compute_explained_variance,
    compute_adjusted_r_squared,
)

# acquisition.py uses importlib to avoid sys.modules shadowing — import it directly
import importlib.util
_acq_path = Path(__file__).resolve().parent / 'acquisition.py'
_spec = importlib.util.spec_from_file_location("acquisition", str(_acq_path))
_acq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acq)
standard_utility = _acq.standard_utility


# =============================================================================
# Data loading
# =============================================================================

def load_pool_and_responses(data_path, cell, device, dtype):
    """Load PNAS dataset, combine train+val into pool, extract single cell.

    Args:
        data_path: Path to PNAS .npz file
        cell: Cell index (0-40)
        device: torch device
        dtype: torch dtype

    Returns:
        X_pool: (3160, n_pixels) all training+val images, flattened
        R_pool: (3160,) spike counts for selected cell
        X_test: (30, n_pixels) test images, flattened
        r_test: (30, 30) test responses [repeats, images] for selected cell
    """
    data = load_pnas_data(data_path, dtype=dtype)

    # Combine train + val into single pool (3160 images)
    X_pool = torch.cat([data['X_train'], data['X_val']], dim=0)  # (3160, 108, 108, 1)
    R_pool = torch.cat([data['R_train'], data['R_val']], dim=0)  # (3160, 41)

    # Flatten images
    X_pool = X_pool.reshape(X_pool.shape[0], -1).to(device)  # (3160, 11664)

    # Select single cell
    R_pool = R_pool[:, cell].to(device)  # (3160,)

    # Test set
    X_test = data['X_test'].reshape(data['X_test'].shape[0], -1).to(device)  # (30, 11664)
    r_test = data['R_test'][:, :, cell].to(device)  # (30, 30)

    return X_pool, R_pool, X_test, r_test


# =============================================================================
# Phase 1: Initial model training
# =============================================================================

def run_phase1(config, R_pool):
    """Train initial model using run_single_config.

    Args:
        config: Flat config dict from build_config_from_defaults()
        R_pool: (3160,) full response vector (to extract spike_counts for training indices)

    Returns:
        model: Trained DirectVGPModel
        indices_train: (M,) int tensor, indices into 3160-pool
        spike_counts: (M,) float tensor, responses for training images
        phase1_metrics: dict with test_r, final_loss, train_time
    """
    result = run_single_config(config)

    model = result['_model']
    likelihood = result['_likelihood']
    indices_train = result['_indices_train']

    # Extract spike counts from pool using training indices
    spike_counts = R_pool[indices_train]

    phase1_metrics = {
        'test_r': result.get('test_r'),
        'final_loss': result.get('final_loss'),
        'train_time': result.get('train_time'),
    }

    return model, indices_train, spike_counts, phase1_metrics


# =============================================================================
# Selection
# =============================================================================

def compute_utility_and_select(model, likelihood, X_candidates, r_max, adaptive_r_max):
    """Compute utility for all candidates and select argmax.

    Args:
        model: Trained DirectVGPModel
        likelihood: PoissonLikelihood
        X_candidates: (N_cand, n_pixels) candidate images
        r_max: int, max spike count for Laplace approximation
        adaptive_r_max: bool, whether to use adaptive r_max

    Returns:
        best_local_idx: int, index into X_candidates (0-based)
        utility_at_best: float, utility value at selected point
    """
    result = standard_utility(model, likelihood, X_candidates, r_max=r_max,
                              adaptive_r_max=adaptive_r_max)
    utility = result['utility']
    best_local_idx = utility.argmax().item()
    utility_at_best = utility[best_local_idx].item()

    return best_local_idx, utility_at_best


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model, X_test, r_test):
    """Evaluate model on 30-image PNAS test set.

    Args:
        model: Trained DirectVGPModel
        X_test: (30, n_pixels) test images
        r_test: (30, 30) test responses [repeats, images]

    Returns:
        dict with test_r, adjusted_r2, explained_var, reliability
    """
    pred = predict_eigenspace(model, X_test)
    f_pred = pred['f_pred']

    # r_test shape is (30, 30) = (n_repeats, n_images) from PNAS loading
    # metrics expect (n_repeats, n_images) — no transpose needed
    r_test_mean = r_test.mean(dim=0)  # (30,) average over repeats
    test_r = compute_pearson_correlation(r_test_mean, f_pred)
    adjusted_r2 = compute_adjusted_r_squared(r_test, f_pred)
    explained_var, reliability = compute_explained_variance(r_test, f_pred)

    return {
        'test_r': test_r,
        'adjusted_r2': adjusted_r2,
        'explained_var': explained_var,
        'reliability': reliability,
    }


# =============================================================================
# JSONL logging
# =============================================================================

def write_iteration_record(output_path, iteration, n_training, selected_idx,
                           spike_count, utility_value, eval_metrics,
                           train_loss, wall_time):
    """Append one iteration record to JSONL file."""
    record = {
        'iteration': iteration,
        'n_training': n_training,
        'selected_idx': selected_idx,
        'spike_count': spike_count,
        'utility': utility_value,
        'train_loss': train_loss,
        'wall_time_s': round(wall_time, 2) if wall_time is not None else None,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }
    record.update(eval_metrics)

    with open(output_path, 'a') as f:
        f.write(json.dumps(record) + '\n')


# =============================================================================
# Main active learning loop
# =============================================================================

def run_active_loop(config, al_config, output_path):
    """Run the full active learning loop.

    Args:
        config: Flat config dict for Phase 1 (from build_config_from_defaults)
        al_config: Active learning config dict (from default_params.json)
        output_path: Path to JSONL output file
    """
    device = torch.device(config['device'])
    dtype = torch.float32 if config['dtype'] == 'float32' else torch.float64
    seed = config['seed']
    eigval_tol = config['eigval_tol']
    r_max = config['r_max']
    adaptive_r_max = config['adaptive_r_max']
    n_active = al_config['n_active_iterations']
    strategy = al_config['strategy']

    phase2_kwargs = {
        'n_iterations': al_config['phase2_n_iterations'],
        'n_estep': al_config['phase2_n_estep'],
        'n_fstep': al_config['phase2_n_fstep'],
        'n_mstep': al_config['phase2_n_mstep'],
        'lr_f': al_config['phase2_lr'],
        'lr_m': al_config['phase2_lr'],
        'early_stop': al_config['phase2_early_stop'],
        'print_every': al_config['phase2_print_every'],
        'verbose': al_config['phase2_verbose'],
        'stability_threshold': config['stability_threshold'],
    }

    # --- Load data ---
    print(f"Loading PNAS data (cell {config['cell']})...")
    data_path = config['data_path']
    # Resolve relative path (default_params.json uses relative path from script dir)
    if not Path(data_path).is_absolute():
        data_path = str(Path(__file__).resolve().parent / data_path)

    X_pool, R_pool, X_test, r_test = load_pool_and_responses(
        data_path, config['cell'], device, dtype
    )
    print(f"  Pool: {X_pool.shape[0]} images, Test: {X_test.shape[0]} images")

    # --- Phase 1: Initial training ---
    print(f"\n=== Phase 1: Training initial model (M={config['M']}, n_train={config['n_train']}) ===")
    t0 = time.time()
    model, indices_train, spike_counts, phase1_metrics = run_phase1(config, R_pool)
    phase1_time = time.time() - t0

    # Validate M == n_train (required by extend_model_with_new_point)
    assert model.X_train.shape[0] == model.X_tilde.shape[0], \
        f"Active loop requires M == n_train, got X_train={model.X_train.shape[0]}, X_tilde={model.X_tilde.shape[0]}"

    print(f"  Phase 1 complete: test_r={phase1_metrics['test_r']:.4f}, time={phase1_time:.1f}s")

    # --- Index management ---
    all_pool_idx = torch.arange(X_pool.shape[0], device=device)
    in_use_idx = indices_train.clone()

    # --- Write Phase 1 baseline (iteration 0) ---
    eval_metrics = evaluate_model(model, X_test, r_test)
    write_iteration_record(
        output_path, iteration=0, n_training=in_use_idx.shape[0],
        selected_idx=None, spike_count=None, utility_value=None,
        eval_metrics=eval_metrics, train_loss=phase1_metrics['final_loss'],
        wall_time=phase1_time,
    )
    print(f"\n=== Phase 2: Active learning ({n_active} iterations, strategy={strategy}) ===")

    # Reset seed before Phase 2 so the random sequence is reproducible
    # independently of any RNG state changes inside Phase 1.
    set_reproducible_seed(seed, device=device)

    # --- Active loop ---
    for iteration in range(1, n_active + 1):
        t_start = time.time()

        # 1. Remaining candidates
        remaining_idx = all_pool_idx[~torch.isin(all_pool_idx, in_use_idx)]
        X_candidates = X_pool[remaining_idx]

        if X_candidates.shape[0] == 0:
            print(f"  Iter {iteration}: No remaining candidates. Stopping.")
            break

        # 2. Select image
        if strategy == 'random':
            best_local_idx = torch.randint(X_candidates.shape[0], (1,), device=device).item()
            utility_at_best = None
        elif strategy == 'argmax':
            with torch.no_grad():
                best_local_idx, utility_at_best = compute_utility_and_select(
                    model, model.likelihood, X_candidates, r_max, adaptive_r_max
                )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        selected_pool_idx = remaining_idx[best_local_idx].item()
        x_new = X_pool[selected_pool_idx]

        # 3. Observe (ground truth lookup)
        r_new = R_pool[selected_pool_idx]

        # 4. Update tracking
        in_use_idx = torch.cat([in_use_idx, torch.tensor([selected_pool_idx], device=device)])
        spike_counts = torch.cat([spike_counts, r_new.unsqueeze(0)])

        # 5. Extend model (rank-1 warm-start)
        kernel_copy = copy.deepcopy(model.kernel)
        likelihood_copy = copy.deepcopy(model.likelihood)
        new_model = extend_model_with_new_point(
            model, x_new, kernel_copy, likelihood_copy, eigval_tol
        )
        del model  # free old model's GPU memory

        # 6. Retrain (Phase 2: light fine-tuning)
        train_result = train_eigenspace(new_model, spike_counts, **phase2_kwargs)
        model = train_result['model']

        train_loss = train_result['losses'][-1] if train_result['losses'] else None

        # 7. Evaluate
        eval_metrics = evaluate_model(model, X_test, r_test)

        wall_time = time.time() - t_start

        # 8. Log
        write_iteration_record(
            output_path, iteration=iteration, n_training=in_use_idx.shape[0],
            selected_idx=selected_pool_idx, spike_count=r_new.item(),
            utility_value=utility_at_best, eval_metrics=eval_metrics,
            train_loss=train_loss, wall_time=wall_time,
        )

        # 9. Print progress
        u_str = f"{utility_at_best:.4f}" if utility_at_best is not None else "n/a"
        print(f"  Iter {iteration}/{n_active}: "
              f"idx={selected_pool_idx}, r={r_new.item():.1f}, "
              f"U={u_str}, test_r={eval_metrics['test_r']:.4f}, "
              f"time={wall_time:.1f}s")

    print(f"\nDone. Results written to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    # Load defaults from default_params.json
    defaults_path = Path(__file__).parent / 'default_params.json'
    with open(defaults_path, 'r') as f:
        defaults = json.load(f)
    al = defaults['active_learning']
    utl = defaults['utility']
    dat = defaults['data']

    parser = argparse.ArgumentParser(
        description='Simulated active learning loop on PNAS data (vargp_direct mode)')

    # Phase 1
    parser.add_argument('--phase1-M', type=int, default=al['phase1_M'],
                        help=f'Initial inducing/training points (default: {al["phase1_M"]})')
    parser.add_argument('--cell', type=int, default=dat['cellid'],
                        help=f'Cell ID (default: {dat["cellid"]})')
    parser.add_argument('--seed', type=int, default=dat['seed'],
                        help=f'Random seed (default: {dat["seed"]})')

    # Phase 2
    parser.add_argument('--n-active', type=int, default=al['n_active_iterations'],
                        help=f'Number of active iterations (default: {al["n_active_iterations"]})')
    parser.add_argument('--phase2-n-iterations', type=int, default=al['phase2_n_iterations'],
                        help=f'EM iterations per active step (default: {al["phase2_n_iterations"]})')
    parser.add_argument('--strategy', type=str, default=al['strategy'],
                        choices=['argmax', 'random'],
                        help=f'Selection strategy (default: {al["strategy"]})')

    # Output
    parser.add_argument('--output', type=str, default='active_loop_results.jsonl',
                        help='JSONL output file (default: active_loop_results.jsonl)')

    args = parser.parse_args()

    # Read mode from config and validate
    mode = al['mode']
    if mode != 'vargp_direct':
        raise ValueError(
            f"run_active_loop.py only supports vargp_direct mode, "
            f"got mode='{mode}' from default_params.json active_learning.mode"
        )

    # Build Phase 1 config — M == n_train enforced
    config = build_config_from_defaults(
        mode=mode,
        M=args.phase1_M,
        n_train=args.phase1_M,  # M == n_train enforced
        seed=args.seed,
        cell=args.cell,
    )

    # Build active learning config (merge CLI overrides)
    al_config = dict(al)  # copy defaults
    al_config['n_active_iterations'] = args.n_active
    al_config['phase2_n_iterations'] = args.phase2_n_iterations
    al_config['strategy'] = args.strategy

    # Print config summary
    print("Active Learning Loop Configuration:")
    print(f"  Cell: {args.cell}, Seed: {args.seed}")
    print(f"  Phase 1: M={args.phase1_M} (pivoted Cholesky)")
    print(f"  Phase 2: {args.n_active} iterations, "
          f"{al_config['phase2_n_iterations']} EM iters/step")
    print(f"  r_max: {config['r_max']}, Strategy: {args.strategy}")
    print(f"  Output: {args.output}")

    run_active_loop(config, al_config, args.output)


if __name__ == '__main__':
    main()
