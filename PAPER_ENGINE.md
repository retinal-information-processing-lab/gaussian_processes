# `paper-engine`: the GP code used for the closed-loop retina paper

This branch is the Gaussian-process model code exactly as it was used for every fit of the
closed-loop paper (commit `d0db7f5`), plus the four small changes listed at the end. None of
them changes the result of a fit.

The repository's default branch (`pietro/workingbranch`) holds a different, later version
of the model. To get the paper's version, use this branch:

    git clone -b paper-engine git@github.com:retinal-information-processing-lab/gaussian_processes.git

The model is the one of Goldin, Virgili & Chalk (2023), "Scalable Gaussian process inference
of neural responses to natural images", PNAS 120(34) e2301150120.

## Where the code is

Everything needed is in `Spatial_GP_repo/scripts/gpytorch_porting/`:

| file | what it holds |
|---|---|
| `example_fit_one_cell.py` | start here: one fit of one cell, a short script read top to bottom |
| `kernels.py` | the kernel (arc-cosine with a localized receptive field) and its bounds |
| `likelihoods.py` | Poisson likelihood with the exponential link |
| `eigenspace_model.py` | the variational GP model and its fitted state |
| `eigenspace_training.py` | the training loop and the prediction function |
| `eigenspace_estep.py`, `_fstep.py`, `_mstep.py` | the three steps of one training iteration |
| `rank1_update.py` | add one image to a fitted model without refitting from scratch |
| `acquisition.py`, `utils.py` | utility of a candidate image (active selection), helpers |
| `default_params.json` | every default setting |
| `run_single_mode.py` | the full one-fit runner: plots, other training modes, every option |

The rest of the repository (`investigations/`, `archive/`, `old_results/`, the notebooks, the
older code directly under `Spatial_GP_repo/`) is lab-notebook history. It is not needed, it
is not maintained, and many scripts there contain paths of the machine they were written on.

## Setup

Python packages, with the versions the paper's fits ran on:
python 3.12.11, torch 2.5.1 (CUDA 12.1), gpytorch 1.14.3, linear_operator 0.6, numpy 2.3.3,
matplotlib 3.10.6, pyyaml 6.0.3. The last two are needed by `run_single_mode.py` only.

Data: the Goldin et al. (2023) file, 143 MB, is not in git. Copy it to

    Spatial_GP_repo/scripts/gpytorch_porting/datasets/PNAS_108x108_original.npz

sha256 `cb8f764fb9bfc55771e5f29aa0ab5699df6c1257ff03aba3f0f1b348c2e85b06`. It holds 3160
natural images of 108x108 pixels shown once each, 30 test images shown 30 times each, and the
spike counts of 41 cells.

## Fit one cell

    cd Spatial_GP_repo/scripts/gpytorch_porting
    python example_fit_one_cell.py                          # cell 8, 500 training images, 100 inducing points
    python example_fit_one_cell.py --ip-selection random    # same, inducing points drawn at random
    python example_fit_one_cell.py --cell 12 --n-train 1000 --ntilde 200
    python example_fit_one_cell.py --device cpu

A fit takes about one minute on a GPU. It prints the fitted parameters and the correlation
between predicted and measured responses on the 30 test images. On the machine the branch
was checked on, the first command gives test Pearson r = 0.508 and the second 0.812.

The example and `run_single_mode.py --mode vargp_direct` give identical results for the same
settings. `run_single_mode.py --help` lists every option.

The quality of a small fit varies a lot with the cell, the training images and the way the
inducing points are chosen. With 500 training images and 100 inducing points, cell 8 gave
test r between 0.51 and 0.86 over two seeds and the two selection methods, and cells 6 and
15 gave test r below 0.2 in every case. Judge a fit by its test score, not by whether it
finished.

## Stability checks in this version

- **A step that makes things worse is undone.** The training loop goes back to the values
  from before a step when the predicted firing rate explodes or a parameter overflows, and
  prints `reverted`.
- **A step that leaves the allowed parameter range is rejected**, including a parameter that
  becomes NaN or infinite (`params_in_bounds` in `kernels.py`).
- **A step that crashes is skipped, with a warning**: `M-step LBFGS crashed: ... Keeping
  pre-step parameters` (same for the F-step). The fit continues and finishes, so the warning
  is the only sign. Running out of GPU memory produces exactly this, and the resulting fit is
  wrong. Search every log for `crashed`; the example prints a notice when it happens.
- **Nothing checks that a fit is good.** A fit can finish with no warning at all and still
  predict the test images badly.

## Changes relative to `d0db7f5`

1. `kernels.py`: `params_in_bounds` returns False when a kernel parameter is NaN or
   infinite. Before, a NaN passed the bounds test, and on a GPU the next matrix solve never
   returned once the model had 1537 or more kept eigen-directions. Three fits were compared
   before and after the change and are bit-for-bit identical.
2. `example_fit_one_cell.py`: new.
3. `run_single_mode.py`, `analytical_gradients.py`: three paths that pointed into a folder
   of one particular machine now point inside this repository.
4. This file.

## Known issues at this version

- `python kernels.py` prints `FAIL: Kernels don't match!`. That self-test compares a kernel
  with a localized receptive field against a reference without one.
- `tests/test_direct_vgp_model.py` and `tests/test_mask_validation.py` refer to files that no
  longer exist, and test 5 of `tests/test_vargp_direct_match.py` cannot read the output of
  the script it calls. The other tests pass.
- `run_single_mode.py --mode vargp_old` (the older reference implementation) crashes when it
  collects its results. It also needs the external `torchlambertw` repository cloned at the
  repository root, and the repository folder to be named `gaussian_processes`.
