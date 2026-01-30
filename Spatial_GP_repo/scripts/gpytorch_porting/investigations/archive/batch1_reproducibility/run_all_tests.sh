#!/bin/bash
# Run all investigation tests and save output for comparison
# Each test runs in a fresh Python process

cd /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/gpytorch_porting

echo "Running all batch1 reproducibility tests..."
echo "Each test runs in a fresh Python process."
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpytorch

echo "===== Test A: Baseline ====="
python investigations/batch1_reproducibility/test_A_baseline.py
echo ""

echo "===== Test B: With torch.pi line ====="
python investigations/batch1_reproducibility/test_B_with_torch_pi.py
echo ""

echo "===== Test C: Just torch.zeros(1) ====="
python investigations/batch1_reproducibility/test_C_just_zeros.py
echo ""

echo "===== Test D: With cuda.init() ====="
python investigations/batch1_reproducibility/test_D_with_cuda_init.py
echo ""

echo "===== Test E: set_reproducible_seed(42) - no device ====="
python investigations/batch1_reproducibility/test_E_seed_no_device.py
echo ""

echo "===== Test F: set_reproducible_seed(42, device='cuda') ====="
python investigations/batch1_reproducibility/test_F_seed_with_device.py
echo ""

echo "===== DONE ====="
echo "Compare the checksums above to see which tests produce different results."
