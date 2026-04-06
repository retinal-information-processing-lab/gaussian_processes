# GPyTorch Porting Project

**Goal**: Port the custom variational GP (`utils.py:varGP()`) to GPyTorch with cleaner structure.

---

## Working Guidelines (Auto-Loaded)

**`.claude/rules/working_guidelines.md`** should be loaded automatically as an always-on rule.

It defines:
- Core philosophy (scientist-developer balance, simplicity first)
- Communication rules (push back, ask questions)
- Development process (staged implementation, validation layers)
- Code style and documentation requirements
- Git hygiene for new users

**BLOCKING CHECK**: Before any work, verify `working_guidelines.md` appears in your context. If NOT loaded, **immediately alert the user** - this indicates a rules configuration problem that must be fixed before proceeding.

---

## Quick Start

| Item | Value |
|------|-------|
| **Conda environment** | `pytorch_gpytorch` - ALWAYS use this |
| **Run canonical experiment** | `python create_experiment.py --name baseline --desc "..."` then `python run_experiment.py --exp baseline` |
| **Run quick exploratory** | `python run_experiment.py --quick test_lr --mode vargp_direct` |
| **Analyze results** | `python analyze_experiment.py --exp baseline` or `--list` |
| **Quick dev test** | `python run_single_mode.py --mode vargp_direct` |
| **GPU REQUIRED** | Scripts default to CUDA. CPU is too slow. |

Experiments (`run_experiment.py`) use YAML configs (`configs/canonical.yaml`, `configs/quick.yaml`) — all params tracked.
Dev tests (`run_single_mode.py`) use `default_params.json` + CLI flags — faster iteration, less reproducibility tracking.

**Current Status**:
| Component | Status |
|-----------|--------|
| ArcCosine kernel with RF structure | COMPLETE |
| Custom E-step (eigenspace projection) | COMPLETE (vargp_direct) |
| Analytical gradients (VJP & Jacobian) | COMPLETE |
| vargp_direct mode | COMPLETE |
| default_gpy mode | COMPLETE |
| Pixel masking | COMPLETE |
| YAML experiment system | COMPLETE |
| Acquisition functions (default_gpy) | IN PROGRESS — `acquisition.py` |

---

## CRITICAL RULES (MUST FOLLOW)

1. **NEVER use nMstep=0 or n_mstep=0** - Disables kernel learning. Always use nMstep >= 10.

2. **Parameters MUST match between GPyTorch and varGP** - See Parameter Matching Table below.

3. **Float32 is the default** - float64 is 10x slower and the reference old code used float32. No need to pass `--float32` explicitly.

4. **LBFGS closure defense (3 layers)** — All LBFGS closures have three guard layers: (1) `params_in_bounds()` rejects trial steps with out-of-bounds parameters before any computation, (2) NaN/exception guards catch numerical failures even within bounds, (3) `clamp_*()` after `optimizer.step()` is a safety net for drift. Each closure calls `params_in_bounds()` only for the parameters it optimizes: kernel (M-step), likelihood (F-step), or both (default_gpy). See `kernels.py:params_in_bounds()`, `likelihoods.py:params_in_bounds()`. Jitter details in `.claude/rules/jitter.md`.

5. **Avoid .data parameter** - Use `torch.no_grad() + copy()` instead.

6. **Conda environment hook** - If a Bash command fails with `CONDA_ENV_WRONG`, immediately run `conda activate pytorch_gpytorch` and retry the command.

7. **No hidden hardcoded parameters** - Scripts (including investigations) must read defaults from `default_params.json`, not hardcode literals like `seed=123, M=50`. Use `build_config_from_defaults()` helper in `run_single_mode.py`. Explicit overrides are fine but must be visible and justified.

8. **No silent pixel clipping in plots** - When plotting images (especially optimized or synthetic ones), every subplot must check if pixel values exceed the dataset global range [min, max] and flag OOB with a red title. Use fixed vmin/vmax = dataset global range, never adaptive scaling. See `.claude/rules/critical_short_rules.md` "Image Pixel Range and Plotting" for full rule.

---

## Known Issues & Debugging

See `.claude/rules/debugging.md` (auto-loads for test files) or use `/debug` skill.

Key issues: torch.pi workaround, Cholesky jitter architecture (see `.claude/rules/jitter.md`), seed sensitivity, RF init, performance degradation, early stopping.

**Known bug**: `_validate_model_params()` in `run_single_mode.py` crashes for `vargp_direct` mode with `AttributeError: 'DirectVGPModel' object has no attribute 'covar_module'`. Training and evaluation complete fine — only the post-training parameter check fails. Needs fixing.

**vargp_old IP selection confound**: `run_single_mode.py` line 672 has `if ip_selection == 'pivoted' and mode != 'vargp_old'` — vargp_old ALWAYS gets random IPs regardless of config. Use `ip_selection='random'` for fair mode comparisons.

**Import side effects**: Importing from old codebase (1D/2D playgrounds, utility.py) can change global state (e.g., `torch.set_default_dtype`). Always guard with save/restore pattern. See `acquisition.py` for example.

**Note on whitening**: GPyTorch's `VariationalStrategy` uses whitened parameterization internally. The deprecated `vargp_style` mode attempted to combine custom E-step with GPyTorch's whitened params, but this caused instability. `vargp_direct` bypasses GPyTorch's `VariationalDistribution` entirely, storing (m, V) directly in eigenspace.

**sigma_0 parameterization**: Uses direct (identity) transform. Changed from exp transform during the paper gap investigation (closed a 0.004 gap). sigma_0 enters the kernel squared (v_x = x^T C x + sigma_0^2), so exp or log-space optimization could be more principled. A controlled re-evaluation (exp vs direct vs optimize sigma_0^2 directly) is planned in `investigations/optimization/possible_optimizations.md`.

**STA edge artifact (108x108)**: For 6/41 cells (0, 5, 6, 15, 22, 39), the z-scored STA on 108x108 images picks a spurious peak at the image edge due to natural image correlation leakage. These cells have near-zero test_r on 108x108. Center crops (48x48, 64x64) avoid this because edge pixels are excluded from the STA computation. Investigation and diagnostics in `investigations/sta_edge_artifact/`.

**PNAS datasets** (all have identical train/val/test splits, same images at different resolutions):

| Dataset | Path | .npz train | .npz val | pool | test |
|---------|------|------------|----------|------|------|
| 108x108 | `datasets/PNAS_108x108_original.npz` | 2910 | 250 | 3160 | 30 |
| 64x64 | `datasets/PNAS_64x64_center_crop_no_renorm.npz` | 2910 | 250 | 3160 | 30 |
| 48x48 | `datasets/PNAS_48x48_center_crop_no_renorm.npz` | 2910 | 250 | 3160 | 30 |

**Data loading (run_single_mode.py)**: The .npz files store a pre-baked train/val split, but that split has a **biased response distribution** (80% zeros in val vs 56% in train). The code always:
1. Combines `images_train` + `images_val` into a single pool of 3160
2. Carves 250 validation images via seeded random permutation (seed = model fitting seed)
3. Remaining 2910 images form the training pool (capped to `n_train` if smaller)

This means **effective max training size is 2910**, not 3160. The 250-image validation holdout costs ~8% of total data but ensures: (a) unbiased val distribution, (b) consistent splits across ES and no-ES runs, (c) seed-reproducible comparisons. Validation is always carved even with `early_stop=False` (curves are logged for post-hoc analysis).

Use `n_train=3160` in configs — it gets capped to 2910 after carving. The `n_val_split` parameter (default 250) controls the carve size.

**Ground-truth RF centers**: `datasets/rf_centers_ground_truth.npz` contains RF centers for all 41 cells from white noise/checkerboard ellipse fits. Source: `ellipses` array in `datasets/samuele_data/lsta_ref.npz`. Coordinate mapping: 72x72 grid → 108x108 via scale factor 1.5. File contains pixel coords (72, 108, 64, 48) and normalized [-1,1] coords (108, 64, 48). Use `rf['norm_64'][cell_id]` for eps_0x/eps_0y initialization. See `datasets/README.md`.

---

## Parameter Matching Table (PREVENTS BUGS)

| Parameter | varGP | vargp_direct | default_gpy |
|-----------|-------|--------------|-------------|
| A_init | 0.01 | 0.01 |  0.01 |
| lambda0_init | 1.0 | 1.0 | 1.0 |
| lr_f (F-step) | 0.1 (LBFGS) | 0.1 (LBFGS) |  0.1 (LBFGS) |
| lr_m (M-step) | 0.1 | 0.1 | 0.1 |
| F-step optimizer | LBFGS | LBFGS |  LBFGS |
| M-step optimizer | LBFGS | LBFGS |  LBFGS |
| **Dtype** | float32 |float32 |  float32 |

All modes load defaults from `default_params.json` (CLI) or YAML configs (experiments).

---

## Gradient Mode Selection

Use `--gradient-mode MODE` in CLI:
- `autograd` (default): PyTorch automatic differentiation
- `vjp`: VJP analytical - same speed as autograd, explicit formulas
- `jacobian`: Slow but matches original varGP exactly

---

## Early Stopping & Validation

**Validation data is ALWAYS held out** (250 images carved from combined train+val pool via seeded permutation, never in training set). When `n_train=3160` is requested, effective training size is 2910. See "Data loading" section above for details on why the .npz pre-baked val split is not used.

**Mechanism**: Patience-based stopping on validation expected log-likelihood:
1. At each outer EM iteration, compute val_ll on held-out 250 images
2. Track best val_ll seen so far
3. If val_ll improves by > `min_delta_rel` (0.1% relative), reset patience counter
4. If patience counter reaches `patience` (15) and `iteration >= min_iterations` (10), stop
5. On stop, restore model state from best-validation iteration

**Validation metric formula**: Expected log-likelihood (same as ELBO's log-lik term):
```
val_ll = sum(r_val * (A*mu + lambda0) - f_mean)
where f_mean = exp(A*mu + 0.5*A^2*var + lambda0)
```

Alternative not used: plug-in Poisson log-prob `sum(r*log(f_pred) - f_pred)`, which
additionally penalizes high posterior variance via an extra `r * 0.5*A^2*var` term.
Both track the same direction during training; the expected log-lik was chosen for
direct comparability with the training ELBO.

**Config** (`default_params.json` / YAML):
```json
"early_stopping": {
    "enabled": true,
    "patience": 15,
    "min_delta_rel": 0.001,
    "min_iterations": 10,
    "restore_best": true
}
```

**Curve logging**: Every training run logs per-iteration curves in `result['curves']`:
`train_loss`, `train_log_lik`, `train_kl`, `val_log_lik`, `A`, `lambda0`, `beta`,
`rho`, `sigma_0`, `eps_0x`, `eps_0y`, `Amp`, `iter_time`. Curves are logged even
with `early_stop=False` (useful for post-hoc convergence analysis).

**Tests**: `tests/test_early_stopping.py` (10 strict tests, ~17s on GPU).

---

## Training Modes

| Mode | Description |
|------|-------------|
| `vargp_old` | utils.py varGP() -- our approximation of the paper (NOT identical to paper's code, see Reference Code section) |
| `vargp_direct` | Eigenspace reimplementation, same algorithm as vargp_old but with GPyTorch patterns |
| `default_gpy` | Standard GPyTorch variational inference with LBFGS (use --float32)|

**Note**: `vargp_style` mode has been deprecated and moved to `deprecated/` folder. Use `vargp_direct` or `default_gpy` instead.

---

## Kernel Selection

Use `--kernel-type TYPE` in CLI (default: `arc_cosine` from `default_params.json`):

| Type | Class | Key property |
|------|-------|-------------|
| `arc_cosine` | `ArcCosineKernel` | K(x,x) ~ ||x||^2. Default, best test_r. |
| `arc_sine` | `ArcSineKernel` | K(x,x) saturates at 1 (erf activation). |
| `rbf` | `LocalRBFKernel` | K(x,x) = 1 (stationary). Extra `--lengthscale` param. |

All kernels share the same RF structure (C matrix with beta, rho, eps_0). Factory: `create_kernel()` in `kernels.py`.

Config: `default_params.json` -> `kernel.type`, `kernel.lengthscale` (RBF only).

**Restrictions**: `vargp_old` mode requires `arc_cosine`. Analytical gradients (`vjp`, `jacobian`) require `arc_cosine`.

---

## File Map

### Shared Components (used by both implementations)
| File | Purpose |
|------|---------|
| `kernels.py` | ArcCosineKernel with RF structure, masking, gradient modes. ArcCosineKernelNormalized, ArcSineKernel, LocalRBFKernel. `create_kernel()` factory, `KERNEL_TYPES`. `params_in_bounds()` + `clamp_hyperparameters()` for LBFGS defense. |
| `likelihoods.py` | PoissonLikelihood with A, lambda0. `params_in_bounds()` + `clamp_params()` for LBFGS defense. Bounds: A_MAX=10, lambda0 in [-50, 50]. |
| `metrics.py` | Evaluation functions (Pearson r, explained variance, adjusted R² per Goldin et al. 2023 Eq. 5) |
| `utils.py` | Shared utilities (lambda0_given_A, compute_f_mean, STA-based RF center) |
| `analytical_gradients.py` | Jacobian-based gradients (slow, reference) |
| `analytical_gradients_vjp.py` | VJP-based gradients (fast) |
| `default_params.json` | Centralized defaults for all modes |
| `acquisition.py` | Acquisition functions: `standard_utility()`, `distribution_aware_utility()`. Zero playground imports — all Laplace/entropy code local in `utils.py`. Both functions return `mu_g_marg` (log-firing rate) for f_max guard. Currently default_gpy only. |

### Eigenspace Implementation (vargp_direct mode)
| File | Purpose |
|------|---------|
| `eigenspace_model.py` | DirectVGPModel, DirectVariationalState, EigenspacePosterior |
| `eigenspace_utils.py` | Low-level eigenspace projection utilities |
| `eigenspace_gradients.py` | Analytical gradient functions for eigenspace M-step |
| `eigenspace_training.py` | train_eigenspace(), predict_eigenspace(), compute_elbo_eigenspace() |
| `eigenspace_estep.py` | E-step: Newton update in eigenspace |
| `eigenspace_fstep.py` | F-step: LBFGS for A with analytical lambda0; `damped_newton_update_A_lambda0()` for interleaved F-step |
| `eigenspace_mstep.py` | M-step: LBFGS for kernel (autograd & analytical) |

### GPyTorch Implementation (default_gpy mode)
| File | Purpose |
|------|---------|
| `gpy_model.py` | VariationalGPModel (standard GPyTorch) |
| `gpy_training.py` | train_gpy_default(), predict() |
| `checkpoint.py` | Save/load trained models as .pt files. `save_checkpoint()` stores model+likelihood state, config, metrics, hyperparams. `load_checkpoint()` reconstructs full model from .pt file. |

### Experiment System (YAML-based)
| File | Purpose |
|------|---------|
| `configs/canonical.yaml` | Full test matrix template — all 35 params, HARDCODED/WIRED annotated |
| `configs/quick.yaml` | Single-point defaults for exploratory runs |
| `create_experiment.py` | Create canonical experiment folder (freezes config + metadata) |
| `run_experiment.py` | Run canonical (`--exp`) or exploratory (`--quick`) experiments |
| `analyze_experiment.py` | Summarize (`--exp`), compare (`--compare`), list (`--list`) experiments |
| `experiments/` | Canonical experiment folders (date-prefixed) |
| `experiments/exploratory/` | Quick exploratory experiment folders |

### Entry Points
| File | Purpose |
|------|---------|
| `run_single_mode.py` | Quick dev test (reads `default_params.json`, full CLI control, no experiment tracking). Exports `run_single_config(config: dict) -> dict` (core training function), `build_config_from_defaults(mode, **overrides) -> dict` (MANDATORY for standalone scripts — reads `default_params.json`), and `flatten_yaml_config(yaml_config, mode, M, n_train, seed, cell) -> dict` (YAML experiment system bridge). |
| `run_experiment.py` | Structured experiments (reads YAML configs, frozen config, full tracking) |
| `run_inference.py` | Load pre-trained checkpoints, run prediction on test set, generate per-cell summary plots (STA+RF, scatter, sorted comparison), save summary CSV + hyperparameters JSON. Primary entry point for sharing results. |
| `train_all_cells.py` | Batch training: trains all 41 cells for specified datasets (108x108, 64x64), saves .pt checkpoints. Uses `build_config_from_defaults()` + `run_single_config()`. |

### Plotting (`plotting/`)
| File | Purpose |
|------|---------|
| `plotting/plot_training.py` | Per-cell training curve visualization (3 rows: log-lik, likelihood params, kernel params). Supports `--ylim-json` for fixed y-axis limits across cells. |
| `plotting/visualize_experiment.py` | Summary visualizations for all-cells experiments (STA galleries, RF overlays, performance plots). |
| `plotting/compute_param_ranges.py` | Compute y-axis ranges from sweep JSONL for consistent cross-cell plotting. Outputs `param_ranges_*.json`. |

### Archived data
| Path | Purpose |
|------|---------|
| `old_results/` | Archived flat JSONL results from development phase (220 records) |

### Deprecated Code (archived, self-contained)
| Folder | Purpose |
|---------|---------|
| `deprecated/` | Archived vargp_style mode, orphaned whitening/test files (self-contained, unmaintained) |

### Investigation Artifacts
| Path | Purpose |
|------|---------|
| `investigations/utility/` | Unified utility investigation. Scripts: `workbench.py` (shared setup + helpers), `gradient_ascent.py` (LBFGS pixel-space gradient ascent), `entropy_landscape.py` (entropy heatmap), `test_compute_H_MC.py`, `subspace_optimization.py` (PCA/C-eigen/combined subspace methods), `test_subspace_optimization.py` (52 tests). Docs in `docs/`: `da_utility_theory.md`, `subspace_operations.md`, `subspace_theory.md`, `entropy_landscape.md`, 3 proof `.tex` files. |
| `investigations/paper_gap/` | Paper performance gap investigation (RESOLVED). Gap A closed (Finding 19), Gap B explained by metric mismatch (Finding 21). 21 findings, 738-run sweep. Key docs: `INVESTIGATION_LOG.md`, `METRICS_COMPARISON.md`. |
| `investigations/optimization/` | Training loop optimization investigations. See `possible_optimizations.md` for the full list (Amp removal, ELBO-based ES, E-step convergence, F-step comparison). |

**Test files** (in `tests/`):
- `test_mask_validation.py`, `test_analytical_gradients.py`, `test_utils.py`
- `test_vargp_direct_match.py`, `test_mstep_analytical.py`, `test_direct_vgp_model.py`
- `test_acquisition.py` — validates `acquisition.py` against manual computation and 1D playground

---

## Deferred Items (DO NOT IMPLEMENT UNLESS ASKED)

### Acquisition Functions - IN PROGRESS
`acquisition.py` implements `standard_utility()` and `distribution_aware_utility()` for `default_gpy` mode **ONLY**.

**Why default_gpy only:**
- `distribution_aware_utility()` requires `model(X).covariance_matrix` for Gaussian conditioning
- `vargp_direct` uses `EigenspacePosterior` which doesn't expose full covariance (only mean/variance)
- `standard_utility()` works with both modes (only needs mean/variance) but kept consistent for now

**Current implementation:**
- All dependencies are local (in `utils.py`, no playground imports)
- Fully differentiable (gradient flow from x* through kernel into utility)
- Works with `ArcCosineKernel`, `ArcCosineKernelNormalized`, `ArcSineKernel`, and `LocalRBFKernel`
- Both functions return `mu_g_marg` (log-firing rate) for f_max firing rate guard
- `f_max` parameter (default 100.0) wired through `default_params.json` and YAML configs
- LBFGS gradient-based x* optimization in unified `investigations/utility/gradient_ascent.py` (supports all kernel types via `--kernel-type`)

**Deferred (acquisition functions)**:
- vargp_direct support (needs augmented matrix approach for distribution-aware utility)
- Scalability for large candidate pools

### Normalized Arc-Cosine Kernel - INVESTIGATED (Feb 2026)
`ArcCosineKernelNormalized` class in `kernels.py` implements K_bar(x,y) = J(theta)/pi with constant diagonal = 1.0. NOT included in `KERNEL_TYPES` or `create_kernel()` — deprecated for active use.

**Investigation folder deleted** (Feb 2026, retrievable from git). Key findings preserved:
- test_r drops ~25% (0.79 -> 0.59) — image norm is genuinely informative for neural encoding
- Eliminates norm-driven utility divergence (scaling by 5x keeps utility stable)
- Not a practical solution due to accuracy loss

### Multi-Cell Validation - DEFERRED
Cell 8 and 10 validation sufficient for initial implementation.

### LBFGS Tolerance Investigation - DEFERRED
The default LBFGS `strong_wolfe` line search uses internal tolerance ~1e-9. Since all code runs with `--float32` (per guidelines), this tolerance may be meaningless. Future investigation: consider whether to expose/adjust tolerance or validate that float32 precision is sufficient.

### YAML Experiment System - Open Items
- **Remaining hardcoded params**: kernel_bounds, lbfgs_tolerance/history_size, lambda_var_clamp, stability_threshold are documented in YAML with `HARDCODED` tags and file:line refs but NOT yet wired through code. Changing their YAML values has no effect.
- **Canonical/quick YAML sync hook**: Non-experiment sections of `canonical.yaml` and `quick.yaml` should stay in sync. A Claude Code hook is planned but not yet implemented.
- **`run_single_mode.py` early stopping defaults**: Now read from `default_params.json`.

---

## vargp_direct Mode

**Key characteristics**:
- Stores m_b, V_b in reduced eigenspace (EIGVAL_TOL=1e-4)
- K_tilde_b is DIAGONAL (trivial inverse)
- Matches vargp_old E-step formulas exactly
- **IMPORTANT**: Use `--float32` for performance

**Usage**:
```bash
python run_single_mode.py --mode vargp_direct
```

**Performance** (M=50, 50 iterations):
| Mode | Dtype | Total Time | Test r |
|------|-------|------------|--------|
| vargp_old | float32 | 6.3s | 0.84 |
| vargp_direct | float32 | 5.6s | 0.84 |
| vargp_direct | float64 | 18.9s | 0.81 |

See `EIGENSPACE_REFERENCE.md` for full implementation details.

---

## Reference Code

### Three Codebases (IMPORTANT DISTINCTION)

There are three distinct implementations. They are NOT equivalent:

1. **Paper's actual code** (Goldin et al. 2023 GitHub notebook): `Variational GP-Single change-GPU-ver2.0.ipynb`. We cannot run it but extracted its structure. Key unique features: F-step interleaved inside E-step (damped Newton), no Amp parameter, no eigenspace projection, scipy L-BFGS-B, float64.

2. **vargp_old** (`utils.py:varGP()`): Our approximation of the paper's approach. Has modifications NOT in the paper: Amp parameter, eigenspace projection. Does NOT interleave F-step (hardcoded `for i_estep in range(1)` loop). Uses torch LBFGS, float32.

3. **vargp_direct** (`eigenspace_*.py`): GPyTorch-based reimplementation. Proven equivalent to vargp_old within 0.002 avg adj_r2 when confounds are controlled (Finding 19). New features: `interleave_fstep` (damped Newton F-step inside E-step), `fix_Amp` (freeze Amp at 1.0).

See `investigations/paper_gap/INVESTIGATION_LOG.md` Finding 18 for the full three-way comparison table.

**Paper init** (verified parameterization mapping, both use [-1,1] coordinates):
- beta_nat=0.0452, rho_nat=0.0821, A=1e-4, lambda0=-1, sigma_0=1.0, NO Amp
- Paper: `0.5*exp(theta[4])` in locality exponent = our `exp(raw_m2log2beta)` = `1/(4*beta_nat^2)`

**Metric note**: Paper reports "adjusted R^2 > 0.8 for 36/41 cells" but likely computed the unsquared metric (mean_accuracy / reliability, our `explained_var`). Our best config matches 36/41 on that metric. See `investigations/paper_gap/METRICS_COMPARISON.md` and Finding 21.

### Original varGP (utils.py) -- vargp_old
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
| Math formulas | `.claude/rules/math.md` (auto-loads, or `/math` skill) |
| "Why was X designed this way?" | DECISION_LOG.md |
| Performance numbers | `analyze_experiment.py --exp <name>` (old: `old_results/BENCHMARK_LOG.md`) |
| How to work on this project | .claude/rules/working_guidelines.md (auto-loaded) |
| vargp_direct implementation | EIGENSPACE_REFERENCE.md |
| Analytical gradients | `.claude/rules/gradients.md` (auto-loads, or `/gradients` skill) |
| Acquisition functions | `.claude/rules/acquisition.md` (auto-loads, or `/acquisition` skill) |
| Jitter & Cholesky stability | `.claude/rules/jitter.md` |
| Session handoff (implementation) | `/handoff-plan` skill |
| Session handoff (investigation) | `/handoff-investigation` skill |
| GPyTorch code patterns | PATTERNS_REFERENCE.md |
| Data format/preprocessing | DATA_REFERENCE.md |

**DO NOT trust outdated information in other files if it conflicts with the authoritative source.**

---

## When to Read Other Docs

| If you're working on... | READ THIS FIRST |
|-------------------------|-----------------|
| Math formulas, E-step derivations | `.claude/rules/math.md` |
| Design rationale (Q1-Q25) | DECISION_LOG.md |
| vargp_direct mode | EIGENSPACE_REFERENCE.md |
| Analytical kernel gradients | `.claude/rules/gradients.md` |
| Acquisition functions, utility | `.claude/rules/acquisition.md` |
| Jitter, Cholesky, numerical stability | `.claude/rules/jitter.md` |
| Handing off to next session | `/handoff-plan` or `/handoff-investigation` skill |
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

*Last updated: April 2026*
*Paper gap investigation: RESOLVED (April 2026). Gap A closed, Gap B explained by metric mismatch. See investigations/paper_gap/INVESTIGATION_LOG.md.*
*Next phase: training loop optimization (investigations/optimization/possible_optimizations.md)*
