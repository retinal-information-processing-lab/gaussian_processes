"""
Validation sweep for the hyperparameter-prior + adaptive A_init fix.

Configuration (matching experiments/2026-04-13_M_sweep_64x64 baseline in every
non-fix knob):
  - vargp_direct, 64x64 PNAS, interleave_fstep=True, fix_Amp=True
  - lambda0_init=-1.0, n_estep=50, n_mstep=20, n_iterations=80
  - ip_selection='random', rf_init='ground_truth', n_val_split=0
  - ELBO ES, patience=15

Fix knobs (the reason this sweep exists):
  - A_init_mode='adaptive'                   (replaces fixed A_init=1e-4)
  - A_init_T_safe=0.01
  - hyperparam_prior_enabled=True
  - A_prior_mu=-3.0, A_prior_sigma=0.5

Grid: 11 cells x {50, 300, 1500} x 3 seeds = 99 runs.
  Degraders:  [39, 35, 16, 13, 10, 33, 15, 14, 27]
  Controls:   [1 (flat/ceiling), 8 (improver)]

Outputs:
  - results.jsonl: one JSON record per run with metrics + curves.
  - checkpoints/cell{cell}_M{M}_seed{seed}.pt: trained DirectVGPModel state
    saved via save_eigenspace_checkpoint (sufficient for inference).

Resume-safe: skips any (cell, M, seed) already present in results.jsonl.
Run from gpytorch_porting/:
  python experiments/2026-04-14_hyperparam_prior_validation/run_validation.py
"""
import sys
import os
import json
import time
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from run_single_mode import build_config_from_defaults, run_single_config
from eigenspace_checkpoint import save_eigenspace_checkpoint

# =============================================================================
# Config — identical to the 2026-04-13 sweep except for the fix knobs below.
# =============================================================================
BASELINE_CONFIG = dict(
    lambda0_init=-1.0,
    n_estep=50,
    n_mstep=20,
    n_iterations=80,
    interleave_fstep=True,
    fix_Amp=True,
)

FIX_CONFIG = dict(
    # Adaptive A_init replaces the hardcoded 1e-4 (see
    # investigations/M_degradation/REGULARIZATION_PROPOSAL.md Section beta1).
    A_init_mode='adaptive',
    A_init_T_safe=0.01,
    # Weakly-informative log-normal prior on A (see same doc).
    hyperparam_prior_enabled=True,
    A_prior_mu=-3.0,
    A_prior_sigma=0.5,
)

# Grid
DEGRADERS = [39, 35, 16, 13, 10, 33, 15, 14, 27]
CONTROLS = [1, 8]
CELLS = DEGRADERS + CONTROLS
M_VALUES = [50, 300, 1500]
SEEDS = [0, 1, 2]

N_TRAIN = 3160
DATA_64 = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'

EXP_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = EXP_DIR / 'results.jsonl'
CKPT_DIR = EXP_DIR / 'checkpoints'


# =============================================================================
# Helpers
# =============================================================================

def load_completed(results_file):
    completed = set()
    if not os.path.exists(results_file):
        return completed
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                completed.add((r['cell'], r['M'], r['seed']))
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


def _load_X_pool(data_path):
    """Load the full PNAS pool (train+val) matching run_single_config's loader."""
    npz = np.load(os.path.join(PROJ, data_path))
    X_train_np = npz['images_train']
    X_val_np = npz['images_val']
    X = np.concatenate([X_train_np, X_val_np], axis=0).astype('float32')
    # Flatten to (N, n_pixels) — matches how run_single_mode loads images.
    if X.ndim == 4:
        X = X.reshape(X.shape[0], -1)
    elif X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    return torch.from_numpy(X)


def extract_record(result, config, ckpt_path):
    if result is None:
        return {
            'cell': config['cell'], 'M': config['M'], 'seed': config['seed'],
            'status': 'failed',
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }
    model = result.get('_model')
    n_b = None; eigval_min = None; eigval_max = None
    if model is not None and hasattr(model, 'state') and model.state is not None:
        try:
            eigvals = model.state.eigvals_b
            n_b = int(len(eigvals))
            eigval_min = float(eigvals.min().item())
            eigval_max = float(eigvals.max().item())
        except Exception:
            pass
    curves = result.get('curves') or {}
    ll_curve = curves.get('train_log_lik') or []
    final_train_log_lik = float(ll_curve[-1]) if ll_curve else None

    M = config['M']
    return {
        'cell': config['cell'],
        'M': M,
        'seed': config['seed'],
        'n_train': result.get('n_train'),
        'config': 'prior_adaptive_intl_fixAmp',
        # Performance
        'test_r': result.get('test_r'),
        'train_r': result.get('train_r'),
        'adjusted_r2': result.get('adjusted_r2'),
        'final_loss': result.get('final_loss'),
        'final_train_log_lik': final_train_log_lik,
        # Optimization
        'n_iterations_run': result.get('n_iterations_run'),
        'stopped_early': result.get('stopped_early'),
        'best_iteration': result.get('best_iteration'),
        'train_time': result.get('train_time'),
        # Final hyperparameters
        'final_A': result.get('final_A'),
        'final_lambda0': result.get('final_lambda0'),
        'final_beta': result.get('final_beta'),
        'final_rho': result.get('final_rho'),
        'final_sigma_0': result.get('final_sigma_0'),
        'final_eps_0x': result.get('final_eps_0x'),
        'final_eps_0y': result.get('final_eps_0y'),
        # Eigenspace
        'n_b': n_b,
        'n_b_over_M': (n_b / M) if (n_b is not None and M > 0) else None,
        'eigval_min': eigval_min,
        'eigval_max': eigval_max,
        # Prior/A_init knobs (for audit)
        'A_init_mode': config.get('A_init_mode'),
        'A_init_T_safe': config.get('A_init_T_safe'),
        'hyperparam_prior_enabled': config.get('hyperparam_prior_enabled'),
        'A_prior_mu': config.get('A_prior_mu'),
        'A_prior_sigma': config.get('A_prior_sigma'),
        # Curves
        'curves_A': curves.get('A'),
        'curves_beta': curves.get('beta'),
        'curves_train_r': curves.get('train_r'),
        'curves_train_loss': curves.get('train_loss'),
        'curves_train_log_lik': curves.get('train_log_lik'),
        # Checkpoint path (relative to experiment folder)
        'checkpoint_path': str(Path(ckpt_path).relative_to(EXP_DIR))
                           if ckpt_path else None,
        # Metadata
        'status': result.get('status', 'success'),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    data_full = os.path.join(PROJ, DATA_64)
    if not os.path.exists(data_full):
        raise FileNotFoundError(f"Dataset not found: {data_full}")

    # Load X_pool once — checkpoint integrity tags are derived from it.
    X_pool = _load_X_pool(DATA_64)

    runs = [(cell, M, seed) for cell in CELLS for M in M_VALUES for seed in SEEDS]
    completed = load_completed(RESULTS_FILE)
    total = len(runs)
    n_skip = sum(1 for r in runs if r in completed)

    print(f"Hyperparam prior validation sweep")
    print(f"{'='*72}")
    print(f"  Cells: {CELLS}  (degraders + 2 controls)")
    print(f"  M: {M_VALUES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Fix: A_init_mode=adaptive (T_safe=0.01), prior on A "
          f"mu={FIX_CONFIG['A_prior_mu']}, sigma={FIX_CONFIG['A_prior_sigma']}")
    print(f"  Baseline non-fix knobs match experiments/2026-04-13_M_sweep_64x64")
    print(f"  {total} runs total, {n_skip} already done, {total - n_skip} to run")
    print(f"  Results JSONL: {RESULTS_FILE}")
    print(f"  Checkpoints:   {CKPT_DIR}")
    print(f"{'='*72}", flush=True)

    n_run = 0
    for cell, M, seed in runs:
        key = (cell, M, seed)
        if key in completed:
            continue
        n_run += 1
        print(f"\n  [{n_run}/{total - n_skip}] cell={cell} M={M} seed={seed}", flush=True)

        config = build_config_from_defaults(
            mode='vargp_direct',
            M=M,
            n_train=N_TRAIN,
            seed=seed,
            cell=cell,
            data_path=DATA_64,
            **BASELINE_CONFIG,
            **FIX_CONFIG,
        )

        ckpt_path = None
        try:
            result = run_single_config(config)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            record = {
                'cell': cell, 'M': M, 'seed': seed,
                'config': 'prior_adaptive_intl_fixAmp',
                'n_train': N_TRAIN,
                'status': 'error', 'error': str(e),
                'timestamp': datetime.now().isoformat(timespec='seconds'),
            }
            with open(RESULTS_FILE, 'a') as f:
                f.write(json.dumps(record) + '\n')
            completed.add(key)
            continue

        # Save checkpoint (model state sufficient for inference).
        # save_eigenspace_checkpoint saves kernel+likelihood+m_b+V_b+pool_indices
        # (training indices). For our M<<n_train sweep, we additionally inject
        # `inducing_indices` so a loader can reconstruct X_tilde from X_pool
        # without rerunning IP selection. The library's load_eigenspace_checkpoint
        # uses the active-loop invariant X_tilde==X_train; an inference loader
        # for this sweep should read 'inducing_indices' and set X_tilde = X_pool[inducing_indices].
        if result is not None and result.get('_model') is not None:
            ckpt_path = CKPT_DIR / f'cell{cell}_M{M}_seed{seed}.pt'
            try:
                metrics = {
                    'test_r': result.get('test_r'),
                    'train_r': result.get('train_r'),
                    'adjusted_r2': result.get('adjusted_r2'),
                    'final_loss': result.get('final_loss'),
                    'train_time': result.get('train_time'),
                }
                pool_indices = result['_indices_train'].detach().cpu().long()
                save_eigenspace_checkpoint(
                    model=result['_model'],
                    config=config,
                    metrics=metrics,
                    pool_indices=pool_indices,
                    X_pool=X_pool,
                    checkpoint_path=str(ckpt_path),
                )
                # Augment the saved .pt with inducing indices. The save function
                # does not store them (active-loop invariant M==n_train) but for
                # M<n_train sweeps they are needed to reconstruct X_tilde.
                saved = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
                ind_ind = result.get('_indices_inducing')
                if ind_ind is not None:
                    saved['inducing_indices'] = ind_ind.detach().cpu().long()
                saved['metadata']['n_train'] = config['n_train']
                saved['metadata']['M'] = config['M']
                torch.save(saved, str(ckpt_path))
            except Exception as e:
                print(f"  WARNING: checkpoint save failed: {e}", flush=True)
                ckpt_path = None

        record = extract_record(result, config, ckpt_path)

        if record.get('test_r') is not None:
            print(
                f"  test_r={record['test_r']:.4f}  train_r={record['train_r']:.4f}"
                f"  n_b={record['n_b']}  A={record['final_A']:.5f}  "
                f"beta={record['final_beta']:.4f}"
                f"  ES={record['stopped_early']}  time={record['train_time']:.0f}s",
                flush=True
            )
        else:
            print(f"  FAILED (no test_r)  time={record.get('train_time', 0):.0f}s",
                  flush=True)

        with open(RESULTS_FILE, 'a') as f:
            f.write(json.dumps(record) + '\n')
        completed.add(key)

        del result
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"\nDone. Results: {RESULTS_FILE}", flush=True)


if __name__ == '__main__':
    main()
