#!/usr/bin/env python3
"""
DEBUG: Count GPU tensors after each active loop iteration to find the leak.

Runs 50 iters, counts all CUDA tensors reachable from gc at key points.
"""
import gc
import sys
import copy
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from run_single_mode import (
    build_config_from_defaults, run_single_config,
    set_reproducible_seed,
)
from eigenspace_training import train_eigenspace
from rank1_update import extend_model_with_new_point
from run_active_loop import (
    load_pool_and_responses, compute_utility_and_select, evaluate_model,
)
from eigenspace_checkpoint import save_eigenspace_checkpoint

device = torch.device('cuda')


def count_cuda_tensors():
    """Count all CUDA tensors tracked by gc."""
    gc.collect()
    count = 0
    total_bytes = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                count += 1
                total_bytes += obj.nelement() * obj.element_size()
        except Exception:
            pass
    return count, total_bytes


# ── Phase 1 ──
config = build_config_from_defaults(
    mode='vargp_direct',
    data_path='datasets/PNAS_64x64_center_crop_no_renorm.npz',
    M=50, n_train=50, seed=42, cell=8,
)
result = run_single_config(config)
model = result['_model']
indices_train = result['_indices_train']
# Drop result reference — we only need model and indices
del result

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

checkpoint_dir = Path('/tmp/profile_tensor_count/checkpoints')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

n_tensors, tensor_bytes = count_cuda_tensors()
mem = torch.cuda.memory_allocated(device) / 1024**3
print(f"After Phase 1: {mem:.3f} GB, {n_tensors} CUDA tensors, {tensor_bytes/1024**2:.1f} MB in tensors")
print()
print(f"{'it':>3} {'mem_GB':>7} {'#tens':>6} {'tens_MB':>8}")

for iteration in range(1, 51):
    # Utility
    remaining_idx = torch.nonzero(remaining_mask, as_tuple=True)[0]
    X_candidates = X_pool[remaining_idx]
    with torch.no_grad():
        best_local_idx, utility_at_best, _ = compute_utility_and_select(
            model, model.likelihood, X_candidates, r_max, adaptive_r_max
        )

    selected_pool_idx = remaining_idx[best_local_idx].item()
    x_new = X_pool[selected_pool_idx]
    r_new = R_pool[selected_pool_idx]
    remaining_mask[selected_pool_idx] = False
    in_use_idx = torch.cat([in_use_idx, torch.tensor([selected_pool_idx], device=device)])
    spike_counts = torch.cat([spike_counts, r_new.unsqueeze(0)])

    # Extend
    kernel_copy = copy.deepcopy(model.kernel)
    likelihood_copy = copy.deepcopy(model.likelihood)
    new_model = extend_model_with_new_point(model, x_new, kernel_copy, likelihood_copy, eigval_tol)
    del model

    # Train
    train_result = train_eigenspace(new_model, spike_counts, **phase2_kwargs)
    model = train_result['model']
    del train_result  # explicit cleanup

    # Evaluate
    eval_metrics = evaluate_model(model, X_test, r_test)

    # Checkpoint
    config['M'] = in_use_idx.shape[0]
    config['n_train'] = in_use_idx.shape[0]
    save_eigenspace_checkpoint(
        model=model, config=config,
        metrics={'test_r': eval_metrics['test_r'], 'train_loss': 0.0},
        pool_indices=in_use_idx, X_pool=X_pool,
        checkpoint_path=checkpoint_dir / f'iter_{iteration:03d}.pt',
    )

    # Count
    gc.collect()
    torch.cuda.empty_cache()
    n_tensors, tensor_bytes = count_cuda_tensors()
    mem = torch.cuda.memory_allocated(device) / 1024**3
    print(f"{iteration:>3} {mem:>7.3f} {n_tensors:>6} {tensor_bytes/1024**2:>8.1f}")
