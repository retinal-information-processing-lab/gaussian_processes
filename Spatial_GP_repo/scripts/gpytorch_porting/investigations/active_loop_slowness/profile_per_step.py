#!/usr/bin/env python3
"""
DEBUG: Per-step GPU memory profiling within the active loop.

Runs a short active loop (30 iters) with memory snapshots at each step
within each iteration to identify WHERE memory accumulates.

Usage:
    python investigations/active_loop_slowness/profile_per_step.py
"""
import gc
import sys
import copy
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from run_single_mode import (
    build_config_from_defaults, run_single_config,
    load_pnas_data, set_reproducible_seed,
)
from eigenspace_training import train_eigenspace
from rank1_update import extend_model_with_new_point
from run_active_loop import (
    load_pool_and_responses, compute_utility_and_select, evaluate_model,
    save_eigenspace_checkpoint,
)
from eigenspace_checkpoint import save_eigenspace_checkpoint

device = torch.device('cuda')
N_ITERS = 30


def mem_gb():
    return torch.cuda.memory_allocated(device) / 1024**3


def mem_gb_clean():
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated(device) / 1024**3


# ── Phase 1 ──
config = build_config_from_defaults(
    mode='vargp_direct',
    data_path='datasets/PNAS_64x64_center_crop_no_renorm.npz',
    M=50, n_train=50, seed=42, cell=8,
)
result = run_single_config(config)
model = result['_model']
indices_train = result['_indices_train']

X_pool, R_pool, X_test, r_test = load_pool_and_responses(
    config['data_path'], config['cell'], device, torch.float32,
)
spike_counts = R_pool[indices_train]

n_pool = X_pool.shape[0]
remaining_mask = torch.ones(n_pool, dtype=torch.bool, device=device)
remaining_mask[indices_train] = False
in_use_idx = indices_train.clone()

eigval_tol = config['eigval_tol']
r_max = config['r_max']
adaptive_r_max = config['adaptive_r_max']

phase2_kwargs = {
    'n_iterations': 5,
    'n_estep': 10, 'n_fstep': 10, 'n_mstep': 10,
    'lr_f': 0.1, 'lr_m': 0.1,
    'early_stop': False, 'print_every': 999, 'verbose': False,
    'f_mean_max_threshold': config['f_mean_max_threshold'],
    'f_mean_mean_threshold': config['f_mean_mean_threshold'],
}

set_reproducible_seed(42, device=device)

print(f"After Phase 1: {mem_gb_clean():.3f} GB")
print()
print(f"{'iter':>4} {'pre':>7} {'util':>7} {'ext':>7} {'train':>7} {'eval':>7} {'ckpt':>7} {'clean':>7}")

checkpoint_dir = Path('/tmp/profile_per_step/checkpoints')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

for iteration in range(1, N_ITERS + 1):
    mem_pre = mem_gb()

    # 1. Utility
    remaining_idx = torch.nonzero(remaining_mask, as_tuple=True)[0]
    X_candidates = X_pool[remaining_idx]
    with torch.no_grad():
        best_local_idx, utility_at_best, _ = compute_utility_and_select(
            model, model.likelihood, X_candidates, r_max, adaptive_r_max
        )
    mem_util = mem_gb()

    selected_pool_idx = remaining_idx[best_local_idx].item()
    x_new = X_pool[selected_pool_idx]
    r_new = R_pool[selected_pool_idx]
    remaining_mask[selected_pool_idx] = False
    in_use_idx = torch.cat([in_use_idx, torch.tensor([selected_pool_idx], device=device)])
    spike_counts = torch.cat([spike_counts, r_new.unsqueeze(0)])

    # 2. Extend
    kernel_copy = copy.deepcopy(model.kernel)
    likelihood_copy = copy.deepcopy(model.likelihood)
    new_model = extend_model_with_new_point(model, x_new, kernel_copy, likelihood_copy, eigval_tol)
    del model
    mem_ext = mem_gb()

    # 3. Train
    train_result = train_eigenspace(new_model, spike_counts, **phase2_kwargs)
    model = train_result['model']
    mem_train = mem_gb()

    # 4. Evaluate
    eval_metrics = evaluate_model(model, X_test, r_test)
    mem_eval = mem_gb()

    # 5. Checkpoint save
    config['M'] = in_use_idx.shape[0]
    config['n_train'] = in_use_idx.shape[0]
    save_eigenspace_checkpoint(
        model=model, config=config,
        metrics={'test_r': eval_metrics['test_r'], 'train_loss': 0.0},
        pool_indices=in_use_idx, X_pool=X_pool,
        checkpoint_path=checkpoint_dir / f'iter_{iteration:03d}.pt',
    )
    mem_ckpt = mem_gb()

    # 6. Clean
    mem_clean = mem_gb_clean()

    print(f"{iteration:>4} {mem_pre:>7.3f} {mem_util:>7.3f} {mem_ext:>7.3f} {mem_train:>7.3f} {mem_eval:>7.3f} {mem_ckpt:>7.3f} {mem_clean:>7.3f}")

print()
print(f"Final clean: {mem_gb_clean():.3f} GB")
print(f"Single fit at M=419: 0.087 GB")
