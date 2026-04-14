# Critical Short Rules

**Purpose**: Project-specific gotchas that have caused bugs in past sessions. Always loaded. Each rule exists because of a real mistake.

---

## Coordinates
- Always use **normalized coordinates** (0-1 range), not pixel coordinates, unless the user explicitly says otherwise.

## Variational Parameters
- Variational parameters (inducing points, variational distribution m/V) are **critical model state** — never omit them when reproducing or comparing models.

## Save the Final Trained Model for Expensive Sweeps

**Any sweep or experiment that (a) takes more than ~30 minutes of compute AND
(b) could plausibly be a reference for later work MUST save the final
trained model (one .pt per run) — not just summary metrics.** This includes
all multi-cell × multi-M × multi-seed sweeps.

- **One file per run, the final trained state.** NOT per-iteration
  snapshots, NOT intermediate checkpoints during training. Just the model
  at the end of training (`train_eigenspace` return).
- A JSONL with `test_r`, `final_A`, `final_beta`, etc. is **not enough** to
  use a trained model later. The variational parameters `m_b`, `V_b`, the
  eigenspace basis, and the inducing point indices are not serializable
  from summary numbers — the model is irretrievable once the Python
  process exits.
- Use `eigenspace_checkpoint.save_eigenspace_checkpoint()` for
  `vargp_direct` runs. See `train_ceiling_models.py` for the canonical
  pattern.
- Naming convention: `<experiment_dir>/models/cell_XX_M<M>_seed<S>.pt`
  (one `.pt` per run, final state only). Store alongside the JSONL.
- Exception: quick debug runs, single-seed single-cell smoke tests, or
  experiments explicitly flagged as "throwaway". If unsure, save.

**Historical context**: the 2026-04-13 M-sweep (984 runs, ~12h of compute)
wrote only the summary JSONL. Every fitted model was lost when the
processes exited. When the user later wanted to hand "the best fits" to a
colleague for inference, re-training was required. This rule exists so
that never happens again.

## Import Side Effects
- Importing from old codebase (1D/2D playgrounds, utility.py) can change `torch.default_dtype` and `sys.path`. Always save and restore both before/after import. See `acquisition.py` for the pattern.

## Eigenspace Conventions
- `K_tilde_b` is DIAGONAL in eigenspace — use element-wise operations, not matrix solves.
- `V_b` is NOT diagonal — it is the full projected covariance.

## Float32
- All training and evaluation uses `--float32`. Float64 is 10x slower and the reference code used float32.

## Numerical Constants: Use `_constants.py`
- `EIGVAL_TOL` and `LAMBDA_VAR_CLAMP` (and any future numerical defaults) live in `_constants.py`, which reads from `default_params.json['model']` at import time.
- **Never write `= 1e-4` or `= 1e-6`** (or any numeric literal) as a default argument in a library function. Import the constant from `_constants.py` instead.
- A pre-commit hook enforces this for module-level `UPPERCASE = <scientific notation>` in root-level `.py` files.

## Config API: Use the Right Builder
- **Standalone scripts** (investigations, one-offs): use `build_config_from_defaults(mode, **overrides)` from `run_single_mode.py`. It reads `default_params.json` and returns a complete config.
- **Experiment matrix iteration**: use `flatten_yaml_config(yaml_config, mode, M, n_train, seed, cell)`. This is for `run_experiment.py`'s `itertools.product()` loop — not for standalone scripts.
- Do NOT use `flatten_yaml_config` and fill in "reasonable" values for mode/M/seed — that's hardcoding with extra steps.

## Verify Config Keys Before Using Them
- Before setting a config key (e.g., `config['float32'] = True`), verify what key the consumer function actually reads. The correct key might be `config['dtype']`, not `config['float32']`. Check the source, don't assume.

## Investigation Scripts Are Not Throwaway Code
Investigation scripts that **fit models or produce metrics** follow the same parameter discipline as production scripts — parameters must trace to config files, use `build_config_from_defaults()`, etc. The dividing line: if the output will be used to make a decision or get documented, parameter discipline applies.

Quick inline checks (printing shapes, inspecting config keys, verifying a value) are fine without config traceability — they're debugging, not results.

## Image Pixel Range and Plotting (No Silent Clipping)
The DMD projector has a finite physical range [0, 255]. In our normalized data, this maps to **[dataset_global_min, dataset_global_max]** computed across ALL available images (train + pool, not just training subset). These are the hard physical limits — nothing brighter or darker can be displayed to the neuron.

**Plotting rules:**
- `imshow` must always use **fixed vmin/vmax = dataset global min/max**. Never adaptive, never per-image scaling.
- **Every subplot must independently check** if its pixel values exceed [vmin, vmax]. If any pixel is out of bounds, the title must turn red and report the OOB percentage. No exceptions — this applies to initial images, intermediate images, and final optimized images alike.
- `imshow` naturally clips values outside [vmin, vmax] to the colorbar endpoints. This clipping is acceptable **only when flagged visually** (red title). Silent clipping — where the plot looks fine but values are actually out of range — is a bug.
- This is the last checkpoint before an image would be shown to a real neuron. Treat it accordingly.

**Why this matters:** Unconstrained optimization (or even noisy initial conditions) can produce pixel values outside physical limits. The optimization itself is free to explore wherever utility is high — but we must always know whether the result is physically realizable. Hiding OOB behind silent clipping defeats the entire purpose of this experimental work.

## Numerical Debugging Checklist
When diagnosing numerical issues, always check these in order:
1. Kernel initialization (bounds, lengthscale)
2. Coordinate normalization (pixel vs normalized)
3. Parameter bounds (A > 0, lambda0 constraints)
4. Jitter architecture (see `.claude/rules/jitter.md` — model.jitter feeds both jitter_val and cholesky_jitter)
5. Gradient flow (no accidental detach or no_grad)
