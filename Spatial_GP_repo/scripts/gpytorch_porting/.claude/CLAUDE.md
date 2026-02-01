# GPyTorch Porting Project

**Goal**: Port the custom variational GP (`utils.py:varGP()`) to GPyTorch with cleaner structure.

---

## Quick Start

| Item | Value |
|------|-------|
| **Conda environment** | `pytorch_gpytorch` - ALWAYS use this |
| **Run canonical test** | `python run_canonical_tests.py --seed 123` |
| **Run single mode** | `python run_single_mode.py --mode vargp_style --explicit-unwhitening` |
| **Query results** | `python query_benchmark.py --mode vargp_style --M 100` |
| **GPU REQUIRED** | Scripts default to CUDA. CPU is too slow. |
| **Read first** | WORKING_GUIDELINES.md (process), then this file |

**Current Status**:
| Component | Status |
|-----------|--------|
| Stage 1 (C=I kernel) | COMPLETE |
| Stage 2 (RF structure) | COMPLETE |
| Stage 3 (Custom E-step) | DEFERRED |
| Stage 4 (Analytical gradients) | COMPLETE |
| vargp_direct mode | COMPLETE |
| Pixel masking | COMPLETE |
| Utility functions | OUT OF SCOPE |

---

## CRITICAL RULES (MUST FOLLOW)

1. **NEVER use nMstep=0 or n_mstep=0** - Disables kernel learning. Always use nMstep >= 10.

2. **Parameters MUST match between GPyTorch and varGP** - See Parameter Matching Table below.

3. **Use --float32** - float64 is 10x slower and the reference old code used float32

4. **All jitter values MUST match model.jitter** (default 1e-4) - Mismatch causes whitening failures.

5. **Avoid .data parameter** - Use `torch.no_grad() + copy()` instead.

---

## Known Limitations

### torch.pi Workaround (HACKY)
`tests/test_utils.py:set_reproducible_seed()` contains:
```python
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # WHY DOES THIS MATTER?!
```
This replicates a side effect from `GP_utils.py` line 49. We don't understand why it works.

### Jitter Consistency
All jitter values MUST match `model.jitter`. The fix: all `estep.py` functions now default to `model.jitter`.

### set_reproducible_seed Device Parameter
`set_reproducible_seed(seed, device=device)` produces DIFFERENT random sequences than `set_reproducible_seed(seed)`. Root cause unknown.

### Whitening Seed Sensitivity
Whitened mode can be sensitive to random seed in some configurations.

### RF Center Initialization
RF center (eps_0x, eps_0y) needs reasonable init near image center. Won't learn from bad init.

---

## Parameter Matching Table (PREVENTS BUGS)

| Parameter | varGP | vargp_style | default_gpy |
|-----------|-------|-------------|-------------|
| A_init | 0.01 | 0.01 | 0.01 |
| lambda0_init | 1.0 | 1.0 | 1.0 |
| lr_f (F-step) | 0.1 (LBFGS) | 0.1 (LBFGS) | 0.01 (Adam) |
| lr_m (M-step) | 0.1 | 0.1 | 0.01 |
| F-step optimizer | LBFGS | LBFGS | Adam |
| M-step optimizer | LBFGS | **Adam** | Adam |

All modes load defaults from `default_params.json`.

---

## Variational Strategy and Whitening

**Two distinct concepts**:

1. **standard_variational_distribution** (model.py):
   - `True` (default): Use `VariationalStrategy` (whitened params)
   - `False`: Use `UnwhitenedVariationalStrategy` (natural params)
   - CLI: `--unwhitened-variational-dist`

2. **explicit_unwhitening** (train.py/estep.py):
   - Controls L_K whitening conversions in E-step
   - **REQUIRED** for `vargp_style` with standard distribution
   - CLI: `--explicit-unwhitening` or `--no-explicit-unwhitening`

**CLI examples**:
```bash
# Standard distribution with explicit unwhitening (most common)
python run_single_mode.py --mode vargp_style --explicit-unwhitening

# Unwhitened strategy
python run_single_mode.py --mode vargp_style --unwhitened-variational-dist --no-explicit-unwhitening
```

---

## Gradient Mode Selection

Use `--gradient-mode MODE` in CLI:
- `autograd` (default): PyTorch automatic differentiation
- `vjp`: VJP analytical - same speed as autograd, explicit formulas
- `jacobian`: Slow but matches original varGP exactly

---

## Training Modes

| Mode | Description |
|------|-------------|
| `vargp_old` | Original varGP implementation (reference baseline) |
| `default_gpy` | Standard GPyTorch variational inference |
| `vargp_style` | Matches original varGP structure (LBFGS F-step, analytical lambda0) |
| `vargp_direct` | Eigenspace projection matching varGP (use --float32) |

---

## File Map

| File | Purpose |
|------|---------|
| `kernels.py` | ArcCosineKernel with RF structure, masking, gradient modes |
| `likelihoods.py` | PoissonLikelihood with A, lambda0 |
| `model.py` | VariationalGPModel (whitened/unwhitened) for vargp_style |
| `eigenspace_model.py` | DirectVGPModel for vargp_direct (eigenspace projection) |
| `eigenspace.py` | Low-level eigenspace utilities (mainly for tests) |
| `train.py` | Training loops + evaluation utilities |
| `estep.py` | E-step with kernel caching |
| `fstep.py` | F-step: LBFGS for A, analytical lambda0 |
| `mstep.py` | M-step: Adam/LBFGS for kernel hyperparameters |
| `whitening.py` | Natural <-> whitened param conversions |
| `direct_vargp.py` | Analytical gradient functions for eigenspace M-step |
| `analytical_gradients.py` | Jacobian-based gradients (slow, reference) |
| `analytical_gradients_vjp.py` | VJP-based gradients (fast) |
| `default_params.json` | Centralized defaults for all modes |
| `run_single_mode.py` | Main test script |
| `run_canonical_tests.py` | 12-config benchmark matrix |
| `query_benchmark.py` | Query benchmark results |

**Test files** (in `tests/`):
- `test_kernel_cache.py`, `test_m_whitening.py`, `test_mask_validation.py`
- `test_reference_comparison.py`, `test_analytical_gradients.py`
- `test_vargp_direct_match.py`, `test_mstep_analytical.py`, `test_direct_vgp_model.py`

---

## Deferred Items (DO NOT IMPLEMENT UNLESS ASKED)

### Utility Functions - OUT OF SCOPE
`utility.py` functions are NOT part of this porting effort:
- `nd_utility_new()`, `distribution_aware_utility_gpytorch()`, `conditioned_utility_clean()`

### Custom E-step - DEFERRED
vargp_style is stable. See `results/BENCHMARK_LOG.md` for performance comparison.

### Multi-Cell Validation - DEFERRED
Cell 8 validation sufficient for initial implementation.

---

## vargp_direct Mode

**Key characteristics**:
- Stores m_b, V_b in reduced eigenspace (EIGVAL_TOL=1e-4)
- K_tilde_b is DIAGONAL (trivial inverse)
- Matches vargp_old E-step formulas exactly
- **IMPORTANT**: Use `--float32` for performance

**Usage**:
```bash
python run_single_mode.py --mode vargp_direct --float32 --ntilde 50 --n-iterations 50 --seed 123
```

**Performance** (M=50, 50 iterations):
| Mode | Dtype | Total Time | Test r |
|------|-------|------------|--------|
| vargp_old | float32 | 6.3s | 0.84 |
| vargp_direct | float32 | 5.6s | 0.84 |
| vargp_direct | float64 | 18.9s | 0.81 |

See `VARGP_DIRECT_REFERENCE.md` for full implementation details.

---

## Reference Code

### Original varGP (utils.py)
| Function | Location | Purpose |
|----------|----------|---------|
| `varGP()` | utils.py:5291 | Main training function |
| `Estep()` | utils.py:4215 | Newton update for (m, V) |
| `localker()` | utils.py | Compute C matrix |
| `acosker()` | utils.py | Arc-cosine kernel |
| `lambda_moments()` | utils.py | Posterior mean/var |

### Codebase Structure
```
Spatial_GP_repo/
├── utils.py           - Main GP: varGP(), Estep(), acosker()
├── utility.py         - Active learning (OUT OF SCOPE)
├── kernels/kernels.py - Clean kernel implementations
├── notebooks/PNAS_paper_sorted_data.npz - Dataset
└── scripts/gpytorch_porting/  - THIS PROJECT
```

---

## Authoritative Sources (Single Source of Truth)

| If you need... | The authoritative doc is... |
|----------------|----------------------------|
| Status, rules, parameter tables | THIS FILE (CLAUDE.md) |
| Math formulas | MATH_REFERENCE.md |
| "Why was X designed this way?" | DECISION_LOG.md |
| Performance numbers | results/BENCHMARK_LOG.md |
| How to work on this project | WORKING_GUIDELINES.md |
| vargp_direct implementation | VARGP_DIRECT_REFERENCE.md |
| Analytical gradients | ANALYTICAL_GRADIENTS_REFERENCE.md |
| GPyTorch code patterns | PATTERNS_REFERENCE.md |
| Data format/preprocessing | DATA_REFERENCE.md |

**DO NOT trust outdated information in other files if it conflicts with the authoritative source.**

---

## When to Read Other Docs

| If you're working on... | READ THIS FIRST |
|-------------------------|-----------------|
| Math formulas, E-step derivations | MATH_REFERENCE.md |
| Design rationale (Q1-Q25) | DECISION_LOG.md |
| vargp_direct mode | VARGP_DIRECT_REFERENCE.md |
| Analytical kernel gradients | ANALYTICAL_GRADIENTS_REFERENCE.md |
| GPyTorch patterns | PATTERNS_REFERENCE.md |
| Data loading/preprocessing | DATA_REFERENCE.md |

---

## Session Wrap-up (MUST FOLLOW)

When user says "wrap up", "done for now", or "session end":

1. Summarize what was accomplished (3-5 bullets)
2. List uncommitted changes (if any)
3. Update CLAUDE.md Quick Start if status changed
4. **CRITICAL**: If any conflicting information was found between docs during this session, RAISE IT TO THE USER immediately
5. Add brief entry to SESSION_LOG.md

---

## Conflict Detection Rule

During session wrap-up, Claude MUST check for conflicting information between docs:
- If conflict found: RAISE IT TO THE USER immediately, do not silently resolve
- User decides which version is correct, then authoritative doc is updated

---

*Last updated: February 2025*
*API cleanup: GPyTorch-like model(X) pattern, eigenspace_model.py reorganization*
