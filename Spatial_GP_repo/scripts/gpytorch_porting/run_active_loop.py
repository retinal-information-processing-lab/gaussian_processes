#!/usr/bin/env python3
"""
run_active_loop.py - Simulated active learning loop on PNAS neural data.

Evaluation harness for comparing utility optimization algorithms using the
vargp_direct (eigenspace-based DirectVGPModel) training mode.

Algorithm:
  1. Train initial GP on a small random set (Phase 1)
  2. LOOP (N iterations):
     a. Compute utility for all remaining images (or pick randomly)
     b. Select the image
     c. Look up its ground-truth spike count (PNAS data)
     d. Extend model with new point (rank-1 warm-start)
     e. Retrain the GP (Phase 2: light fine-tuning)
     f. Evaluate on held-out test set
  3. Write per-iteration metrics and save model checkpoint

Output layout (one directory per run):
    <output-dir>/
        results.jsonl          # one line per active iteration (0..N)
        curves.jsonl           # one line: Phase 1 full training curves
        config.json            # full config + CLI args + git commit + integrity tags
        checkpoints/
            iter_000.pt        # Phase 1 model
            iter_NNN.pt        # after each active addition

Usage:
    python run_active_loop.py                                # All defaults
    python run_active_loop.py --n-active 50 --cell 8         # Override iters and cell
    python run_active_loop.py --strategy random --seed 42    # Random baseline

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
    get_git_commit,
)
from eigenspace_training import train_eigenspace, predict_eigenspace
from eigenspace_checkpoint import save_eigenspace_checkpoint
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
# Helpers: curves extraction + JSON serialization
# =============================================================================

# Keys expected in train_eigenspace result['curves'] that we want the final
# value of. Excludes 'iter_time' (semantically odd as a "final") and 'val_*'
# (only populated when n_val_split > 0, not our case).
_CURVE_TRAINING_KEYS = ('train_log_lik', 'train_kl', 'train_r')
_CURVE_HYPER_KEYS = (
    'A', 'lambda0', 'beta', 'rho', 'sigma_0', 'eps_0x', 'eps_0y', 'Amp',
)


def _extract_final_curve_values(curves):
    """Extract last value of each training metric and hyperparameter curve.

    Args:
        curves: dict returned by train_eigenspace() / run_single_config() under
                the 'curves' key. Must be non-None and contain every key in
                _CURVE_TRAINING_KEYS and _CURVE_HYPER_KEYS, each mapping to a
                non-empty list. Missing keys or empty series raise loudly —
                project rule: no silent fallback to None (working_guidelines
                section 3.7b).

    Returns:
        dict with keys train_log_lik, train_kl, train_r (flat) and keys
        final_A, final_lambda0, final_beta, ..., final_Amp (prefixed).
    """
    out = {}
    for key in _CURVE_TRAINING_KEYS:
        out[key] = curves[key][-1]
    for key in _CURVE_HYPER_KEYS:
        out[f'final_{key}'] = curves[key][-1]
    return out


def _to_json_safe(obj):
    """Recursively coerce an object to a JSON-serializable form.

    Handles: dict, list, tuple, Path -> str, numpy scalars -> python scalars,
    primitives pass through. Raises TypeError on anything else (fail loud,
    don't silently drop).
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        # Tensors don't belong in a config snapshot; fail loud so we catch
        # accidental tensor insertion during development.
        raise TypeError(
            f"_to_json_safe refuses to serialize a torch.Tensor of shape "
            f"{tuple(obj.shape)}. Config must contain only primitives."
        )
    raise TypeError(f"_to_json_safe cannot serialize type {type(obj).__name__}")


# =============================================================================
# Phase 1: Initial model training
# =============================================================================

def run_phase1(config, R_pool):
    """Train initial model using run_single_config.

    Returns the raw result dict from run_single_config so the caller has
    access to curves, final loss, and all model components.

    Args:
        config: Flat config dict from build_config_from_defaults()
        R_pool: (3160,) full response vector (to extract spike_counts for
                training indices)

    Returns:
        (result, spike_counts) where:
            result: dict returned by run_single_config (contains '_model',
                    '_likelihood', '_indices_train', 'curves', 'test_r',
                    'final_loss', etc.)
            spike_counts: (M,) float tensor, responses for training images
    """
    result = run_single_config(config)
    spike_counts = R_pool[result['_indices_train']]
    return result, spike_counts


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

def write_iteration_record(results_path, iteration, n_training, n_b,
                           selected_idx, spike_count, utility_value,
                           train_loss, curve_finals, eval_metrics, wall_time):
    """Append one iteration record to results.jsonl.

    Args:
        results_path: Path to results.jsonl
        iteration: int, active iteration number (0 for Phase 1 baseline)
        n_training: int, number of training points after this iteration
        n_b: int, eigenspace dimension (kept eigenvalues)
        selected_idx: int or None, pool index selected this iteration
        spike_count: float or None, ground-truth spike count for selected image
        utility_value: float or None, utility at selected image (None for random)
        train_loss: float, -ELBO at end of training (or None if losses empty)
        curve_finals: dict from _extract_final_curve_values(); contains
                      train_log_lik, train_kl, train_r and final_* hypers
        eval_metrics: dict from evaluate_model(); contains test_r, adjusted_r2,
                      explained_var, reliability
        wall_time: float, iteration wall time in seconds
    """
    record = {
        'iteration': iteration,
        'n_training': n_training,
        'n_b': n_b,
        'selected_idx': selected_idx,
        'spike_count': spike_count,
        'utility': utility_value,
        'train_loss': train_loss,
        'wall_time_s': round(wall_time, 2) if wall_time is not None else None,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }
    record.update(curve_finals)
    record.update(eval_metrics)

    with open(results_path, 'a') as f:
        f.write(json.dumps(record) + '\n')


def write_phase1_curves(curves_path, iteration, n_training, curves,
                        final_iteration, best_iteration, stopped_early):
    """Append one record to curves.jsonl.

    Currently called only for iteration 0 (Phase 1). Phase 2 retrains are
    too short (only phase2_n_iterations EM steps) to produce meaningful
    trajectories; their final values end up in results.jsonl instead.

    Args:
        curves_path: Path to curves.jsonl
        iteration: int, the active iteration this curve belongs to (0 = Phase 1)
        n_training: int, number of training points during this training run
        curves: full dict from result['curves']
        final_iteration, best_iteration, stopped_early: ES metadata from result
    """
    record = {
        'iteration': iteration,
        'n_training': n_training,
        'final_iteration': final_iteration,
        'best_iteration': best_iteration,
        'stopped_early': stopped_early,
        'curves': curves,
    }
    with open(curves_path, 'a') as f:
        f.write(json.dumps(record) + '\n')


# =============================================================================
# Main active learning loop
# =============================================================================

def run_active_loop(config, al_config, cli_args, output_dir):
    """Run the full active learning loop.

    Args:
        config: Flat config dict for Phase 1 (from build_config_from_defaults)
        al_config: Active learning config dict (from default_params.json)
        cli_args: Namespace from argparse (recorded in config.json)
        output_dir: Path to output directory. Will contain:
                    results.jsonl, curves.jsonl, config.json, checkpoints/
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
        'f_mean_max_threshold': config['f_mean_max_threshold'],
        'f_mean_mean_threshold': config['f_mean_mean_threshold'],
    }

    # --- Output layout ---
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    results_path = output_dir / 'results.jsonl'
    curves_path = output_dir / 'curves.jsonl'
    config_path = output_dir / 'config.json'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"Loading PNAS data (cell {config['cell']})...")
    data_path = config['data_path']
    # Resolve relative path (default_params.json uses a path relative to script dir).
    # Mutate config in place so config.json and every saved checkpoint see the same
    # absolute path — no resolved/unresolved inconsistency between outputs.
    if not Path(data_path).is_absolute():
        data_path = str(Path(__file__).resolve().parent / data_path)
        config['data_path'] = data_path

    X_pool, R_pool, X_test, r_test = load_pool_and_responses(
        data_path, config['cell'], device, dtype
    )
    print(f"  Pool: {X_pool.shape[0]} images, Test: {X_test.shape[0]} images")

    # --- Integrity tags (saved in every checkpoint + config.json) ---
    pool_shape = list(X_pool.shape)
    pool_sum = X_pool.sum().item()

    # --- Write config snapshot ---
    config_snapshot = {
        'phase1_config': _to_json_safe(config),
        'al_config': _to_json_safe(al_config),
        'cli_args': _to_json_safe(vars(cli_args)),
        'git_commit': get_git_commit(),
        'data_path': str(data_path),
        'pool_shape': pool_shape,
        'pool_sum': pool_sum,
        'saved_at': datetime.now().isoformat(timespec='seconds'),
    }
    with open(config_path, 'w') as f:
        json.dump(config_snapshot, f, indent=2)

    # --- Phase 1: Initial training ---
    print(f"\n=== Phase 1: Training initial model "
          f"(M={config['M']}, n_train={config['n_train']}) ===")
    t0 = time.time()
    phase1_result, spike_counts = run_phase1(config, R_pool)
    phase1_time = time.time() - t0

    model = phase1_result['_model']
    indices_train = phase1_result['_indices_train']
    phase1_curves = phase1_result['curves']
    phase1_final_loss = phase1_result['final_loss']

    # Validate M == n_train (required by extend_model_with_new_point)
    assert model.X_train.shape[0] == model.X_tilde.shape[0], (
        f"Active loop requires M == n_train, got "
        f"X_train={model.X_train.shape[0]}, X_tilde={model.X_tilde.shape[0]}"
    )

    # test_r can legitimately be None (NaN correlation on test set); format accordingly.
    phase1_test_r = phase1_result['test_r']
    phase1_test_r_str = f"{phase1_test_r:.4f}" if phase1_test_r is not None else "N/A"
    print(f"  Phase 1 complete: test_r={phase1_test_r_str}, "
          f"time={phase1_time:.1f}s")

    # --- Index management ---
    all_pool_idx = torch.arange(X_pool.shape[0], device=device)
    in_use_idx = indices_train.clone()

    # --- Write Phase 1 baseline (iteration 0): results row, curves row, checkpoint ---
    eval_metrics = evaluate_model(model, X_test, r_test)
    phase1_final_values = _extract_final_curve_values(phase1_curves)
    n_b_phase1 = len(model.state.eigvals_b)

    write_iteration_record(
        results_path,
        iteration=0, n_training=in_use_idx.shape[0], n_b=n_b_phase1,
        selected_idx=None, spike_count=None, utility_value=None,
        train_loss=phase1_final_loss,
        curve_finals=phase1_final_values,
        eval_metrics=eval_metrics,
        wall_time=phase1_time,
    )

    # Phase 1 curves are the only ones saved to curves.jsonl (50-iter training).
    # Phase 2 retrains are too short (phase2_n_iterations EM steps) to be useful.
    if phase1_curves:
        write_phase1_curves(
            curves_path,
            iteration=0, n_training=in_use_idx.shape[0],
            curves=phase1_curves,
            final_iteration=phase1_result['n_iterations_run'],
            best_iteration=phase1_result['best_iteration'],
            stopped_early=phase1_result['stopped_early'],
        )

    save_eigenspace_checkpoint(
        model=model,
        config=config,
        metrics={**eval_metrics, **phase1_final_values,
                 'train_loss': phase1_final_loss},
        pool_indices=in_use_idx,
        pool_shape=pool_shape,
        pool_sum=pool_sum,
        checkpoint_path=checkpoint_dir / 'iter_000.pt',
    )

    print(f"\n=== Phase 2: Active learning "
          f"({n_active} iterations, strategy={strategy}) ===")

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
            best_local_idx = torch.randint(
                X_candidates.shape[0], (1,), device=device
            ).item()
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
        in_use_idx = torch.cat([
            in_use_idx,
            torch.tensor([selected_pool_idx], device=device),
        ])
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
        phase2_curves = train_result['curves']
        phase2_final_values = _extract_final_curve_values(phase2_curves)

        # 7. Evaluate
        eval_metrics = evaluate_model(model, X_test, r_test)
        n_b_current = len(model.state.eigvals_b)

        wall_time = time.time() - t_start

        # 8. Log results row
        write_iteration_record(
            results_path,
            iteration=iteration, n_training=in_use_idx.shape[0], n_b=n_b_current,
            selected_idx=selected_pool_idx, spike_count=r_new.item(),
            utility_value=utility_at_best,
            train_loss=train_loss,
            curve_finals=phase2_final_values,
            eval_metrics=eval_metrics,
            wall_time=wall_time,
        )

        # 9. Save checkpoint
        save_eigenspace_checkpoint(
            model=model,
            config=config,
            metrics={**eval_metrics, **phase2_final_values,
                     'train_loss': train_loss},
            pool_indices=in_use_idx,
            pool_shape=pool_shape,
            pool_sum=pool_sum,
            checkpoint_path=checkpoint_dir / f'iter_{iteration:03d}.pt',
        )

        # 10. Print progress
        u_str = f"{utility_at_best:.4f}" if utility_at_best is not None else "n/a"
        print(f"  Iter {iteration}/{n_active}: "
              f"idx={selected_pool_idx}, r={r_new.item():.1f}, "
              f"U={u_str}, test_r={eval_metrics['test_r']:.4f}, "
              f"time={wall_time:.1f}s")

    print(f"\nDone. Results written to {output_dir}")


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
    parser.add_argument('--output-dir', type=str, default=al['output_dir_default'],
                        help=f'Output directory (default: {al["output_dir_default"]}). '
                             f'Relative paths resolved against the script directory.')

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

    # Resolve output directory (relative to script dir if not absolute)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir

    # Print config summary
    print("Active Learning Loop Configuration:")
    print(f"  Cell: {args.cell}, Seed: {args.seed}")
    print(f"  Phase 1: M={args.phase1_M} (pivoted Cholesky)")
    print(f"  Phase 2: {args.n_active} iterations, "
          f"{al_config['phase2_n_iterations']} EM iters/step")
    print(f"  r_max: {config['r_max']}, Strategy: {args.strategy}")
    print(f"  Output: {output_dir}")

    run_active_loop(config, al_config, args, output_dir)


if __name__ == '__main__':
    main()
