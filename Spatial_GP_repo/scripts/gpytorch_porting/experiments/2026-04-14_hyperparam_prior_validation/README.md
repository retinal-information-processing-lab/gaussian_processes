# Hyperparameter Prior Validation — 64x64

**Date**: 2026-04-14
**Branch**: `pietro/investigate-M-degradation`
**Scope**: 11 cells × 3 M × 3 seeds = 99 runs on 64x64 PNAS, validating the
hyperparameter-prior + adaptive-A_init fix proposed in
`investigations/M_degradation/REGULARIZATION_PROPOSAL.md`.

## Context

`experiments/2026-04-13_M_sweep_64x64/` characterized M-degradation for 9/41
cells under the `vargp_direct + interleave_fstep + fix_Amp` config. Root
cause: the ELBO has a KL regularizer on `q(λ̃)` but **no prior on
hyperparameters**, so at large M the optimizer uses the extra capacity to
fit training noise (train_r rises while test_r falls).

This sweep tests the proposed fix:

1. **Adaptive A_init** (stability-derived): `A_init = sqrt(T_safe / sum(r^2))`
   with `T_safe = 0.01` — replaces the hardcoded `A_init = 1e-4`.
2. **Log-normal prior on A** (MAP instead of MLE): penalty
   `0.5 * (log A - A_mu)^2 / A_sigma^2` added to the F-step objective.
   `A_mu = -3.0`, `A_sigma = 0.5`.

Prior rationale: empirical distribution of `log(A_opt@M=50)` across 41 cells
has mean −2.59 and std 0.46; A_mu=−3.0 rounds the empirical mean to a
universal value. A_sigma=0.5 was picked from a smoke test on Cell 35 at
M=1500: sigma=1.0 too loose (no effect), sigma=0.3 regresses Cell 8
(improver), sigma=0.5 moderate and safe. See
`investigations/M_degradation/REGULARIZATION_PROPOSAL.md` Sections 3-4.

## Training configuration

All 99 runs use identical settings except (cell, M, seed):

```python
mode = 'vargp_direct'
interleave_fstep = True
fix_Amp = True

# ES-sweep overrides (from experiments/2026-04-13_M_sweep_64x64)
lambda0_init = -1.0
n_estep = 50
n_mstep = 20
n_iterations = 80

# FIX knobs — the reason this sweep exists
A_init_mode = 'adaptive'        # replaces A_init=1e-4
A_init_T_safe = 0.01
hyperparam_prior_enabled = True
A_prior_mu = -3.0
A_prior_sigma = 0.5

# Other knobs (from default_params.json)
kernel_type = 'arc_cosine'
rf_init = 'ground_truth'
ip_selection = 'random'
n_val_split = 0
early_stop = True
es_metric = 'elbo'
patience = 15

data_path = 'datasets/PNAS_64x64_center_crop_no_renorm.npz'
n_train = 3160
```

## Grid

| Dimension | Values |
|---|---|
| Cells (11) | 39, 35, 16, 13, 10, 33, 15, 14, 27 (degraders) + 1 (flat/ceiling), 8 (improver) |
| M (3) | 50, 300, 1500 |
| Seeds (3) | 0, 1, 2 |
| **Total** | **99 runs** |

## Outputs

| File | Purpose |
|---|---|
| `results.jsonl` | One JSON record per run (metrics + curves + checkpoint path + prior/A_init knobs). |
| `checkpoints/cellC_MN_seedS.pt` | Inference-ready DirectVGPModel state (kernel + likelihood + m_b + V_b + pool_indices + inducing_indices). |
| `sweep.log` | Console log from the sweep run. |
| `run_validation.py` | The sweep driver. Resume-safe on re-run. |

## Loading a checkpoint for inference (M < n_train)

Because `load_eigenspace_checkpoint` assumes the active-loop invariant
M==n_train, and this sweep uses M<n_train, a loader must use
`inducing_indices` to reconstruct `X_tilde` separately from `pool_indices`
(training subset). The sweep's `run_validation.py` augments the standard
save with the `inducing_indices` field for this reason.

```python
import torch
from kernels import create_kernel
from likelihoods import PoissonLikelihood
from eigenspace_model import DirectVGPModel
from eigenspace_training import predict_eigenspace

ckpt = torch.load('checkpoints/cell35_M50_seed0.pt', weights_only=False)
# X_pool: full PNAS train+val pool loaded from datasets/PNAS_64x64_*.npz
X_train = X_pool[ckpt['pool_indices']].to(dtype=ckpt['m_b'].dtype)
X_tilde = X_pool[ckpt['inducing_indices']].to(dtype=ckpt['m_b'].dtype)
kernel = create_kernel(ckpt['config'], 64,
                       eps_0x=ckpt['hyperparams']['eps_0x'],
                       eps_0y=ckpt['hyperparams']['eps_0y'])
kernel.load_state_dict(ckpt['kernel_state_dict'])
likelihood = PoissonLikelihood(
    A_init=ckpt['config']['A_init'],
    lambda0_init=ckpt['config']['lambda0_init'],
)
likelihood.load_state_dict(ckpt['likelihood_state_dict'])
model = DirectVGPModel(
    kernel, likelihood, X_train, X_tilde,
    eigval_tol=ckpt['metadata']['eigval_tol'],
    lambda_var_clamp=ckpt['metadata']['lambda_var_clamp'],
)
model.update_variational_params(ckpt['m_b'], ckpt['V_b'])
preds = predict_eigenspace(model, X_test)
```

Verified round-trip: reloaded test_r matches saved test_r to 1e-4.

## Reproduction

From `gpytorch_porting/`:

```bash
python experiments/2026-04-14_hyperparam_prior_validation/run_validation.py
```

Resume-safe (skips any `(cell, M, seed)` already in `results.jsonl`).
Estimated runtime: ~2-3 hours on a single GPU.

## Results

(To be filled in after the sweep completes. Reference baseline from
`experiments/2026-04-13_M_sweep_64x64/M_sweep_results.jsonl`.)

### Baseline reference (no prior, A_init=1e-4)

| Cell | M=50 | M=300 | M=1500 |
|------|------|-------|--------|
| 1 | 0.981 | 0.983 | 0.982 |
| 8 | 0.750 | 0.862 | 0.871 |
| 10 | 0.905 | 0.897 | 0.890 |
| 13 | 0.951 | 0.925 | 0.934 |
| 14 | 0.909 | 0.908 | 0.896 |
| 15 | 0.704 | 0.696 | 0.691 |
| 16 | 0.955 | 0.956 | 0.927 |
| 27 | 0.905 | 0.879 | 0.898 |
| 33 | 0.969 | 0.961 | 0.955 |
| 35 | 0.806 | 0.779 | 0.749 |
| 39 | 0.699 | 0.649 | 0.501 |

### Fix results (prior on, adaptive A_init)

TBD.

### Delta (fix − baseline)

TBD.

### Success criteria

- All 9 degraders trend flat or improving from M=50 to M=1500 (|delta| < 0.005),
  OR at minimum Cell 39 and Cell 35 fully saved.
- Cell 1 at M=1500 within 0.005 of baseline.
- Cell 8 at M=1500 within 0.005 of baseline.

## Related

- `experiments/2026-04-13_M_sweep_64x64/` — 984-run baseline sweep (this
  experiment's reference).
- `investigations/M_degradation/FINDINGS.md` — investigation story.
- `investigations/M_degradation/REGULARIZATION_PROPOSAL.md` — full proposal
  including the σ sensitivity smoke tests that drove the σ=0.5 choice.
