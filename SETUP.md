# Environment Setup

How to set up this repo on a new machine. Written for the NVIDIA DGX Spark (aarch64 + Blackwell GPU) but works on any Linux with an NVIDIA GPU.

## Prerequisites

- NVIDIA GPU with recent drivers (CUDA 12.1+ driver compatibility)
- conda or miniconda installed
- git

## 1. Clone the repo

```bash
git clone <repo-url> gaussian_processes
cd gaussian_processes
git checkout pietro/utility_optimization
```

## 2. Create conda environment and install PyTorch (platform-specific)

PyTorch wheels are platform-specific: an x86 wheel won't work on ARM, and a CPU-only wheel won't use your GPU. This step installs PyTorch separately so you can pick the right wheel for your hardware.

### DGX Spark (aarch64 + CUDA 12.x)

The DGX Spark has an ARM CPU (Grace) and a Blackwell GPU. Standard x86 PyTorch wheels will not work.

```bash
conda create -n gp_neural python=3.12 numpy scipy matplotlib pyyaml -c conda-forge -y
conda activate gp_neural

# Install PyTorch for aarch64 + CUDA.
# Try default PyPI first (has aarch64 CUDA wheels for PyTorch 2.5+):
pip install "torch>=2.5,<2.7"

# If that installs a CPU-only build (check with step below), try NVIDIA's index instead:
# pip install "torch>=2.5,<2.7" --extra-index-url https://developer.download.nvidia.com/compute/redist
```

**Verify GPU access immediately** (do not proceed if this fails):
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
```

Expected output: `CUDA: True, device: <your GPU name>`. If it prints `CUDA: False`, see Troubleshooting.

### x86_64 workstation (CUDA 12.1)

```bash
conda create -n gp_neural python=3.12 numpy scipy matplotlib pyyaml -c conda-forge -y
conda activate gp_neural
pip install "torch>=2.5,<2.7" --extra-index-url https://download.pytorch.org/whl/cu121
```

## 3. Install remaining dependencies

From the repo root (`gaussian_processes/`):

```bash
pip install "gpytorch>=1.14,<1.16" "linear-operator>=0.5,<0.7"
pip install -e ./torchlambertw
pip install pytest  # optional, for running tests
```

`torchlambertw` is a vendored copy of a small library (Lambert W function for PyTorch). The `-e` flag installs it in editable mode from the local directory. pip will automatically fetch its build dependency (`poetry-core`) during install.

## 4. Transfer data files

Data files (`.npz`) are not tracked by git. Copy them from an existing machine.

**Required files** (minimum to run gpytorch_porting scripts):

| File | Size | Destination |
|------|------|-------------|
| `PNAS_paper_sorted_data.npz` | 144 MB | `Spatial_GP_repo/notebooks/` |
| `rf_centers_ground_truth.npz` | 8 KB | `Spatial_GP_repo/scripts/gpytorch_porting/datasets/` |

Transfer example (run on DGX Spark):
```bash
scp user@workstation:/path/to/gaussian_processes/Spatial_GP_repo/notebooks/PNAS_paper_sorted_data.npz \
    Spatial_GP_repo/notebooks/
scp user@workstation:/path/to/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting/datasets/rf_centers_ground_truth.npz \
    Spatial_GP_repo/scripts/gpytorch_porting/datasets/
```

**Verify the main data file loaded correctly:**
```bash
python -c "
import numpy as np
d = np.load('Spatial_GP_repo/notebooks/PNAS_paper_sorted_data.npz')
print('Keys:', list(d.keys()))
print('images_train:', d['images_train'].shape)  # expect (N, 108, 108, 1)
print('responses_train:', d['responses_train'].shape)  # expect (N, 41)
"
```

**Create the 108x108 symlink and generate cropped datasets:**
```bash
cd Spatial_GP_repo/scripts/gpytorch_porting/datasets

# Symlink (uses realpath so it survives directory moves)
ln -s "$(realpath ../../../notebooks/PNAS_paper_sorted_data.npz)" PNAS_108x108_original.npz

# Generate 64x64 and 48x48 center crops (only needs numpy)
python create_cropped_dataset.py
cd ../../..
```

## 5. Verify the setup

```bash
cd Spatial_GP_repo/scripts/gpytorch_porting

# Quick smoke test: train one cell
python run_single_mode.py --mode default_gpy --seed 123

# Expected: completes in ~10s, prints test_r around 0.84
```

## Troubleshooting

### `CUDA available: False` after installing PyTorch

1. Check driver: `nvidia-smi` should show driver version and CUDA version.
2. Check architecture: `uname -m` should show `aarch64` (DGX Spark) or `x86_64`.
3. Check the installed torch build: `python -c "import torch; print(torch.__version__)"`. If it says `+cpu`, you got a CPU-only wheel. Reinstall with the correct index URL (see step 2).
4. On DGX Spark, if neither PyPI nor NVIDIA's index has a compatible wheel, use NVIDIA's NGC PyTorch container as a last resort: `docker pull nvcr.io/nvidia/pytorch:XX.XX-py3`.

### Import errors for `torchlambertw`

This is a vendored package in the repo root. It must be pip-installed in editable mode:
```bash
cd gaussian_processes   # repo root
pip install -e ./torchlambertw
```

### `ModuleNotFoundError` for project modules (eigenspace_model, kernels, etc.)

Scripts in `gpytorch_porting/` import each other as flat modules. You must `cd` into that directory before running them:
```bash
cd Spatial_GP_repo/scripts/gpytorch_porting
python run_single_mode.py ...
```

## What the environment contains

| Package | Version constraint | Why |
|---------|--------------------|-----|
| python | 3.12 | Tested version, matches torchlambertw requirement |
| torch | >=2.5, <2.7 | Core ML framework. CUDA build required. |
| gpytorch | >=1.14, <1.16 | Gaussian process framework built on PyTorch |
| linear-operator | >=0.5, <0.7 | GPyTorch's linear algebra backend |
| numpy | (latest) | Numerical computing, data loading |
| scipy | (latest) | Used for image filtering (gaussian_filter) in investigations |
| matplotlib | (latest) | Plotting results and diagnostics |
| pyyaml | (latest) | YAML config parsing for experiment system |
| torchlambertw | local (0.0.4) | Lambert W function, used by parent codebase (utils.py) |
| pytest | (latest) | Optional, for running test suite |

## Notes

- **GPU is required**. CPU execution is too slow for this codebase.
- **Float32 is the default**. All training runs in float32 (float64 is 10x slower).
- The `package/` subdirectory in `Spatial_GP_repo/scripts/gpytorch_porting/` has a separate, minimal `environment.yml` for inference-only use (no torchlambertw, no pytest, no experiment system).
- Pre-trained checkpoints (`.pt` files) are also not git-tracked and need separate transfer if you want to skip retraining.
