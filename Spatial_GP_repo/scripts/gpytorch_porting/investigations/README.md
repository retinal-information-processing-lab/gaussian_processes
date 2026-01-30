# Investigation Archives

This directory contains archived debugging artifacts from the GPyTorch porting project. All investigations have been resolved.

## Summary of Past Investigations

### Whitening Issues (January 2025)
- **Problem**: Performance collapse at M=75 with whitening
- **Root cause**: Jitter mismatch between GPyTorch internal L_K (1e-4) and our conversions (1e-6)
- **Fix**: All jitter values now match `model.jitter`
- **Files**: `archive/diagnose_whitening_collapse.py`, `archive/WHITENING_DIVERGENCE_ANALYSIS.md`

### Reproducibility Issues (January 2025)
- **Problem**: Different random sequences with/without device parameter in seed function
- **Root cause**: `torch.pi` assignment side-effect (still mysterious)
- **Workaround**: `test_utils.py:set_reproducible_seed()` includes torch.pi workaround
- **Files**: `archive/batch1_reproducibility/`

### Kernel Instability (January 2025)
- **Problem**: Kernel matrix losing positive-definiteness during M-step
- **Root cause**: Hyperparameter bounds too permissive
- **Fix**: Added clamping and bounds checking in kernel
- **Files**: `archive/H3_KERNEL_INSTABILITY_CONFIRMED.md`

### Firing Rate Instability (January 2025)
- **Problem**: `f_mean` exploding during training
- **Root cause**: A and lambda0 interactions
- **Fix**: Stability checks in F-step, analytical lambda0
- **Files**: `archive/FIRING_RATE_INSTABILITY_FINDINGS.md`

### vargp_direct Eigenvalue Bug (January 2025)
- **Problem**: Some cells (6, 15) completely failed with vargp_direct
- **Root cause**: M-step used stale eigenvalues from E-step
- **Fix**: Use `torch.linalg.solve()` for K_tilde_inv_b (commit fa817dc)
- **Files**: `archive/vargp_direct_match/`

## Where to Find Conclusions

All investigation conclusions have been incorporated into the main documentation:
- **CLAUDE.md**: Known limitations section
- **DECISION_LOG.md**: Design decisions that resulted from investigations
- **test_utils.py**: Workarounds implemented in code

## Archive Contents

```
archive/
├── batch1_reproducibility/   # Seed/device reproducibility tests
├── batch2_correctness/       # Kernel matrix correctness tests
├── batch3_silent_failures/   # NaN/warning detection tests
├── amp_investigation/        # Amplitude parameter analysis
├── vargp_direct_match/       # vargp_direct vs vargp_old comparison
├── *.md                      # Investigation findings documents
└── *.py                      # Investigation test scripts
```
