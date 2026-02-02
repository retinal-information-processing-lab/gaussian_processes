# Deprecated Code - vargp_style Mode

This folder contains code for the **vargp_style** training mode, which has been deprecated in favor of the two maintained implementations:

1. **eigenspace** (vargp_direct) - Eigenspace projection with custom E-step
2. **default_gpy** - Standard GPyTorch variational inference

## Why Deprecated?

The vargp_style mode attempted to combine:
- Custom E-step (Newton updates)
- GPyTorch's VariationalStrategy
- Whitening parameter conversions

This hybrid approach had several issues:
- Complex whitening conversions between natural and whitened parameters
- Performance not better than either eigenspace or default_gpy
- Added maintenance burden without clear benefits

## What's Archived Here

- `vargp_style_training.py` - train_varGP_style() function
- `vargp_style_estep.py` - Custom E-step with kernel caching
- `vargp_style_fstep.py` - F-step variants (Adam, LBFGS)
- `vargp_style_mstep.py` - M-step for GPyTorch models
- `vargp_style_whitening.py` - Parameter conversion utilities
- `vargp_style_model.py` - VariationalGPModel (same as gpy_model.py)
- `vargp_style_run.py` - Snapshot of run_single_mode.py

## Migration Path

**If you were using vargp_style**, switch to:

- **For speed and exact varGP match**: Use `--mode vargp_direct` (eigenspace)
- **For standard GPyTorch**: Use `--mode default_gpy`

Both modes achieve similar or better performance than vargp_style.

## Last Known Working Version

- **Commit**: 23c3856 (Checkpoint before codebase reorganization)
- **Date**: 2025-02-02
- **Status**: Working but unmaintained

## Running Deprecated Code

The archived code should still work if needed:

```bash
# From deprecated/ folder
python vargp_style_run.py --mode vargp_style --float32 --cell 8 --ntilde 50
```

However, this is **not recommended** - use eigenspace or default_gpy instead.

---

*Archived during codebase reorganization: 2025-02-02*
