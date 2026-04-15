# Pre-trained variational GP models for retinal cell responses (M=300)

## What this is

Pre-trained variational Gaussian process models that predict the firing-rate
response of 41 retinal ganglion cells to 64x64 natural-image stimuli
(PNAS dataset). One model per cell, trained independently with M=300 inducing
points using the `vargp_direct` mode of the gpytorch_porting project (an
eigenspace-projected variational GP with Poisson likelihood, arc-cosine
kernel, and a learned receptive-field structure). Training was performed
once with seed=0 and the configuration that produced the best results in the
April 2026 ES-config sweep (interleaved F-step + fixed Amp).

The package is self-contained: the data, the trained models, the minimum
inference code, and a conda environment specification are all bundled.
You do NOT need the upstream `gaussian_processes` repository to run inference.

## Folder layout

```
2026-04-15_vargp_direct_M300/
|-- README.md                                     # this file
|-- environment.yml                               # conda env spec (name: gp_neural)
|-- data/
|   |-- PNAS_64x64_center_crop_no_renorm.npz      # natural-image dataset (~53 MB)
|   |-- rf_centers_ground_truth.npz               # ground-truth RF centers (one per cell)
|-- checkpoints/
|   |-- cell_00.pt ... cell_40.pt                 # 41 eigenspace checkpoints, one per cell
|   |-- ceiling_results.json                      # per-cell summary of metrics + final hyperparameters
|-- code/
|   |-- inference.py                              # entry point for inference (see below)
|   |-- default_params.json                       # config defaults read by _constants.py
|   |-- _constants.py                             # numerical constants imported by library code
|   |-- kernels.py                                # arc-cosine, arc-sine, RBF kernels (with RF structure)
|   |-- likelihoods.py                            # PoissonLikelihood (A, lambda0)
|   |-- metrics.py                                # Pearson r, explained variance, adjusted R^2
|   |-- utils.py                                  # math helpers (lambert W, RF center from STA, ...)
|   |-- eigenspace_model.py                       # DirectVGPModel, EigenspacePosterior
|   |-- eigenspace_utils.py                       # low-level eigenspace projection helpers
|   |-- eigenspace_estep.py                       # E-step Newton update (in eigenspace)
|   |-- eigenspace_fstep.py                       # F-step (interleaved damped Newton on A, lambda0)
|   |-- eigenspace_mstep.py                       # M-step kernel-hyperparameter update (LBFGS)
|   |-- eigenspace_training.py                    # train_eigenspace, predict_eigenspace
|   |-- eigenspace_gradients.py                   # analytical eigenspace M-step gradient kernels
|   |-- eigenspace_checkpoint.py                  # save_eigenspace_checkpoint / load_eigenspace_checkpoint
|   |-- analytical_gradients.py                   # Jacobian-based kernel grads (lazy import)
|   |-- analytical_gradients_vjp.py               # VJP-based kernel grads (lazy import)
|-- results/                                      # populated by inference.py (created on first run)
    |-- summary.csv                               # per-cell test_r, explained_var, adj_R^2, hyperparameters
    |-- plots/cell_XX.png                         # per-cell actual vs predicted firing-rate plot
```

## Setup

The conda environment is named `gp_neural`. Create it from the bundled spec
(see `environment.yml`; PyTorch is installed via pip per platform).

```
conda env create -f environment.yml
conda activate gp_neural
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
```

A CUDA-capable GPU is strongly recommended. CPU inference works but is much slower.

## Run inference

From the deliverable folder root, with `gp_neural` activated:

```
python code/inference.py \
    --data data/PNAS_64x64_center_crop_no_renorm.npz \
    --checkpoints checkpoints \
    --output results
```

Optional flags:

- `--cells 0 1 8` evaluate only the listed cell ids (default: all 41).
- `--no-plots` skip per-cell plot generation.
- `--device cuda|cpu` override device selection.

Outputs:

- `results/summary.csv` -- one row per cell with `test_r`, `explained_var`,
  `adjusted_r2`, `reliability`, `M`, and the final fitted hyperparameters
  (A, lambda0, sigma_0, Amp, beta, rho, eps_0x, eps_0y).
- `results/plots/cell_XX.png` -- two-panel diagnostic plot per cell
  (actual vs predicted firing rates in original order and sorted by
  actual rate, with metrics overlaid).

## Model summary

Per-cell metrics for the bundled checkpoints. Full table is in
`checkpoints/ceiling_results.json` (and reproduced row-by-row in
`results/summary.csv` after running inference).

<!-- BEGIN MODEL SUMMARY TABLE (auto-generated, do not edit by hand) -->
**Top 10 cells by test_r:**

| cell | test_r | explained_var | adj_r^2 | reliability | M   | A      | beta   | rho    |
|------|--------|---------------|---------|-------------|-----|--------|--------|--------|
|  1   | 0.9850 | 1.0066 | 0.9845 | 0.9717 | 300 | 0.0985 | 0.0433 | 0.0413 |
| 31   | 0.9706 | 0.9932 | 0.9542 | 0.9674 | 300 | 0.1159 | 0.0489 | 0.0428 |
| 11   | 0.9672 | 1.0101 | 0.9625 | 0.9434 | 300 | 0.0344 | 0.0621 | 0.1090 |
| 33   | 0.9658 | 0.9876 | 0.9471 | 0.9711 | 300 | 0.0622 | 0.0667 | 0.0585 |
| 12   | 0.9553 | 0.9694 | 0.9215 | 0.9807 | 300 | 0.0648 | 0.0793 | 0.0433 |
| 16   | 0.9494 | 0.9656 | 0.9109 | 0.9770 | 300 | 0.0901 | 0.0420 | 0.0797 |
|  3   | 0.9463 | 0.9957 | 0.9257 | 0.9336 | 300 | 0.0932 | 0.0395 | 0.0761 |
| 25   | 0.9445 | 1.0013 | 0.9280 | 0.9255 | 300 | 0.1443 | 0.0426 | 0.0502 |
| 13   | 0.9408 | 0.9799 | 0.9091 | 0.9468 | 300 | 0.0724 | 0.0598 | 0.0745 |
| 24   | 0.9346 | 0.9612 | 0.8903 | 0.9636 | 300 | 0.1303 | 0.0492 | 0.0634 |

**Lowest 5 cells by test_r (harder-to-fit cells):**

| cell | test_r | explained_var | adj_r^2 | reliability | M   | A      | beta   | rho    |
|------|--------|---------------|---------|-------------|-----|--------|--------|--------|
|  0   | 0.5926 | 0.6260 | 0.3631 | 0.9265 | 300 | 0.1072 | 0.0707 | 0.0541 |
| 39   | 0.6038 | 0.6293 | 0.3747 | 0.9461 | 300 | 0.0536 | 0.0903 | 0.0552 |
|  5   | 0.6297 | 0.6796 | 0.4191 | 0.9074 | 300 | 0.0617 | 0.0814 | 0.0727 |
|  7   | 0.6775 | 0.7293 | 0.4824 | 0.9070 | 300 | 0.1269 | 0.0489 | 0.0548 |
| 28   | 0.6916 | 0.7941 | 0.5258 | 0.8338 | 300 | 0.0955 | 0.0500 | 0.0356 |

**Mean across all 41 cells**: test_r = 0.8402, explained_var = 0.9020, adjusted_r^2 = 0.7506. M = 300 for all cells.

Full per-cell data (including final hyperparameters A, lambda0, sigma_0, Amp, beta, rho, eps_0x, eps_0y) is in `checkpoints/ceiling_results.json`, and is also reproduced in `results/summary.csv` after running inference.
<!-- END MODEL SUMMARY TABLE -->

## Scientific context

Each model predicts the **expected firing rate** (Poisson rate parameter) of
one retinal ganglion cell in response to a natural-image stimulus. The
training pool consists of **3160 natural images** (2910 originally labelled
"train" + 250 originally labelled "val", combined into a single pool because
the original validation split has a biased response distribution). The held-out
test set contains **30 natural images** that were each shown to the cell **30
times** (repeats), so the response statistics on test images are well estimated.

`test_r` is the **Pearson correlation between the predicted firing rate and
the mean over the 30 repeats of the actual response** for the 30 test images.
`reliability` is the noise ceiling (correlation between even and odd repeat
halves), and `explained_var` is `mean_accuracy / reliability` -- the fraction
of the explainable signal that the model captures. `adjusted_r2` is the
squared, Spearman-Brown-corrected version (Goldin et al. 2023, Eq. 5).

## Limitations and caveats

- Trained on **64x64 center crops** of PNAS natural images. Inputs at a
  different resolution, with different statistics (e.g. white noise,
  out-of-distribution natural images), or shown at a different mean
  luminance / contrast may give unreliable predictions.
- A handful of cells are noticeably harder to fit and have lower test_r
  than the population median. For this M=300 seed=0 training run the five
  lowest are cells 0, 39, 5, 7, 28 (test_r 0.59-0.69); see the "Lowest 5"
  table above.
- The models are **frozen**. Changing any hyperparameter would require
  re-training from scratch, which is out of scope for this package.
- The integrity of the input dataset is checked at load time: the
  checkpoint records `pool_shape`, `pool_dtype`, and `pool_sum` of the
  original training pool, and the loader re-derives these from the
  user-provided data and asserts they match (within a tight tolerance for
  float32 reduction noise). If you swap the dataset for a different file,
  the loader will refuse to proceed.

## Technical details (appendix)

**Training configuration** (baked into `train_and_save_M300.py` upstream):

| Setting              | Value           | Notes |
|----------------------|-----------------|-------|
| mode                 | vargp_direct    | eigenspace-projected variational GP |
| M (inducing points)  | 300             | population plateau from the 984-run M sweep |
| seed                 | 0               | one seed for a clean per-cell deliverable |
| n_train              | 3160            | full train+val pool, no val carving |
| ip_selection         | random          | inducing-point subset selection |
| rf_init              | ground_truth    | RF centers from white-noise/checkerboard fits |
| kernel               | arc_cosine      | with RF locality + smoothness structure |
| A_init               | 1e-4            | small init avoids overshoot in interleaved F-step |
| lambda0_init         | -1.0            | matches the paper's convention |
| n_estep              | 50              | inner E-step iterations per outer step |
| n_mstep              | 20              | inner M-step LBFGS iterations |
| n_iterations         | 80              | outer EM iterations |
| interleave_fstep     | True            | damped Newton F-step inside E-step (paper algorithm) |
| fix_Amp              | True            | freeze kernel Amp at 1.0 |
| early_stopping       | ELBO, p=15      | patience=15, min_delta_rel=0.001, restore_best=True |

**Checkpoint format**. Each `cell_XX.pt` is produced by
`eigenspace_checkpoint.save_eigenspace_checkpoint()` and contains the kernel
and likelihood `state_dict`s, the eigenspace variational parameters
(`m_b`, `V_b`), the inducing-point pool indices, integrity tags
(`pool_shape`, `pool_dtype`, `pool_sum`), the full training config dict, the
per-cell metrics, and metadata (cell id, n_px_side, eigval_tol,
lambda_var_clamp, save timestamp).

**Loading API**:

```python
from eigenspace_checkpoint import load_eigenspace_checkpoint
bundle = load_eigenspace_checkpoint(
    checkpoint_path='checkpoints/cell_08.pt',
    X_pool=X_pool,                  # tensor of shape (3160, 4096), float32
    pool_sum_tolerance=1e-3,        # default_params.json value
    device='cuda',
)
model = bundle['model']             # DirectVGPModel ready for inference
```

**Inference API**:

```python
from eigenspace_training import predict_eigenspace
out = predict_eigenspace(model, X_test)   # X_test: (n_test, 4096), float32
f_pred = out['f_pred']                    # predicted firing rate, shape (n_test,)
```

**Integrity check**. The dataset MUST be bit-identical to what was used in
training -- the loader will refuse to instantiate a model otherwise. If
something downstream fails an integrity assertion, double-check that the
.npz file has not been re-encoded or re-cropped.

**Note on the bundled `eigenspace_checkpoint.py`**. The bundled copy of the
loader has one small difference from the upstream library version: it reads
`metadata['M']` to set `X_tilde = X_train[:M]` instead of assuming the
active-loop invariant `X_tilde = X_train`. This is the correct behaviour for
M=300 / n_train=3160 single-run checkpoints (which by construction have
the M inducing points as the first M rows of `pool_indices`). Upstream will
be fixed to match separately.
