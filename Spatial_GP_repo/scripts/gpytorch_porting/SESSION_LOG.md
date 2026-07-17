# Session Log

## 2026-03-24: All-cells fitting, STA edge artifact investigation, stability fixes

**Experiments run**: 1476 fits across 108x108, 64x64, 48x48 (41 cells x 3 seeds x 2 modes x 2 M values per dataset). Plus 123-run LSTA-init experiment on 64x64.

**STA edge artifact**: 6/41 cells (0,5,6,15,22,39) get spurious STA peaks at image edges on 108x108 due to natural image correlation leakage. Confirmed via 32x32 crop STA, whitening analysis, and controlled re-initialization. Center crops (48x48, 64x64) avoid the artifact. Ground-truth RF centers from white noise ellipses (`datasets/rf_centers_ground_truth.npz`) fix initialization for all cells. Investigation: `investigations/sta_edge_artifact/`.

**Stability fixes**: Subprocess isolation for GPU memory in `run_experiment.py`. F-step lambda0 overflow revert in `eigenspace_fstep.py` (root cause: LBFGS pushes A too high for low-firing cells). E-step f_mean revert. LBFGS crash guards. NaN graceful handling. `status: diverged` tracking. Working guidelines updated (rule 3.14: GPU smoke test discipline).

**New dataset**: `PNAS_48x48_center_crop_no_renorm.npz` created. `rf_centers_ground_truth.npz` with RF centers from white noise/checkerboard ellipses for all 41 cells (72x72 LSTA grid, scale factor 1.5 to 108x108).

**Metrics**: Added `compute_adjusted_r_squared` to `metrics.py` — Goldin et al. 2023 PNAS Eq. 5. Documented distinction from existing `compute_explained_variance`. Both tracked in results.jsonl.

**Bug fixes**: `--eps-0x 0.0` sentinel bug (couldn't distinguish from "not provided"). Missing `n_px_side` in result dict for vargp_direct/default_gpy. `visualize_experiment.py` 48x48 support.

**Pending for next session**: (1) Rerun 108x108, 64x64, 48x48 experiments with ground-truth RF init (only 64x64 done so far). (2) Training parameter exploration to maximize test_r (deferred). (3) The 108/64/48 experiments in `experiments/2026-03-24_massive_allcells_*` used STA init — compare with ground-truth init to quantify full benefit.

## 2026-03-10: Synthetic image generator — scoping finalized, handed off to diffusion worktree
**Handoff (this repo)**: `.claude/handoffs/HANDOFF_2026-03-09_synthetic-image-generator-package.md`
**Handoff (diffusion worktree)**: `../gpytorch_imagenet_diffusion/.../gpytorch_porting/.claude/handoffs/HANDOFF_2026-03-10_synthetic-image-generator-package.md`
**Plan**: `.claude/plans/giggly-riding-cook.md`
**Status**: Continuing in new session (diffusion worktree)

Finalized all decisions for diffusion image generator package: separate tarball, NPZ output with selectable pixel range (--pixel-range {uint8, pnas}), finetuned model only, synthetic dataset script deferred. Explored diffusion worktree structure, read existing generate_samples.py and model docs. Decided new session should work directly in diffusion worktree (../gpytorch_imagenet_diffusion/). Created comprehensive technical handoff there with all decisions, pixel conventions, environment.yml pattern, draft implementation steps, and verification checklist.

## 2026-03-09: Inference package validated
**Status**: GP package complete (~/gp_neural_fitting.tar.gz, 329 MB)

Validated GP inference package end-to-end: fixed environment.yml (conda-only failed, switched to pip for torch/gpytorch/linear_operator), tested conda env creation + inference from scratch, rebuilt tarball. Cleaned up test env.

## 2026-03-03: Diffusion model investigation planning
**Handoff**: `.claude/handoffs/HANDOFF_2026-03-03_diffusion-model-training-investigation.md`
**Plan**: `investigations/diffusion/PLAN_diffusion_model_training.md`
**Status**: Handed off for implementation

Explored diffusion model approach for natural image generation. Read motivation doc, diffusion intro (LaTeX), utility REFERENCE.md, subspace analysis. Corrected assumptions in motivation doc (108x108 not 30x30, 3,160 not 10,000 images). Decided: full-image training (cell-agnostic), tiny U-Net (~1-2M params), pure PyTorch, cosine schedule, 4x flip augmentation. Self-contained in investigations/diffusion/ (3 files). GP integration deferred to future investigation.

## 2026-03-03: Merge pca-utility-optimization + cleanup
**Handoff**: `investigations/HANDOFF_consolidate_utility_investigations.md`
**Status**: Continuing — next session consolidates utility/ and utility_decompositions/ into one folder

Merged `pietro/pca-utility-optimization` (10 commits) into `pietro/workingbranch`. Clean merge, no conflicts. Cleanup: deleted `understanding_utility/` (obsolete, 2 orphan PNGs), deleted superseded handoffs (HANDOFF_PCA.md, HANDOFF_C_EIGEN.md), added .gitignore for PNGs. Fixed test_subspace_optimization.py for unified offset convention (all 52 tests pass). Committed TARGET_INDEX revert. Input warping is docs/planning only — no code in kernels.py.

## 2026-02-24: Input warping investigation and planning
**Handoff**: `.claude/handoffs/HANDOFF_2026-02-24_input-warping-bounded-pixels.md`
**Plan**: `.claude/plans/playful-wandering-pine.md`
**Status**: Handed off for implementation

Literature survey on input/output warping for bounded-domain GPs. Created `investigations/input_warping/INPUT_WARPING_REFERENCE.md` (comprehensive reference: Beta CDF, Kumaraswamy, compositional warping, constrained GPs, Jacobian analysis, 10 references). Decided on fixed scaled tanh (zero learnable params, simplest option). Planned implementation: warping at top of kernel.forward(), togglable via config, disabled by default, both modes, all gradient modes. Branch `pietro/input-warping` to be created from `pietro/workingbranch`.

## 2026-02-23b: Unified subspace script + combined approach handoff

**Branch**: `pietro/pca-utility-optimization`
**Handoff**: `investigations/utility_decompositions/HANDOFF_COMBINED_SUBSPACE.md`
**Status**: Continuing — next session implements COMBINED PCA+C-eigenspace optimization

Created `subspace_optimization.py` unifying PCA and C-eigenspace as separate `--method` alternatives. Shared gradient_ascent, z_to_image, plotting. Both methods tested (PCA RBF + C-eigen arc_cosine). Brought c_eigen_optimization.py to this branch as reference. User clarified goal: COMBINE both decompositions in a single optimization (e.g., eigendecompose V_K^T @ C @ V_K), not just run them separately.

## 2026-02-23: PCA-constrained utility optimization (continuing)

**Branch**: `pietro/pca-utility-optimization`
**Handoff**: `investigations/utility_decompositions/HANDOFF_PCA.md`
**Status**: Continuing in next session

PCA optimization of x*=mu+V_K@z with LBFGS. Iterated through: Adam vs LBFGS, training-only vs full-dataset PCA, noise starts vs mean start, arc_cosine vs RBF, multiple var_thresholds. RBF+LBFGS+M=50 is the working configuration. Key insight: var_threshold < 1.0 is a genuine naturalness constraint (not just reparameterization).

## 2026-02-21: C-eigenvalue utility optimization (separate branch)

**Branch**: `pietro/c-eigen-utility-optimization`
**Handoff**: `investigations/utility_decompositions/HANDOFF_C_EIGEN.md`

C-eigenspace optimization (x*=U_K@z from kernel C matrix). Confirmed theoretical prediction: equivalent to pixel-space optimization, just better conditioning. K=91 of 1725 dims retained.

## 2026-02-20c: Complete Phases 3-4, consolidate utility folder
**Branch**: `pietro/workingbranch`
**Status**: All 4 phases complete

Phase 3: unified gradient.py (verified all 3 kernels). Phase 4: user deleted per-kernel folders. Moved understanding_utility/ reference files into utility/, deleted the folder. All utility investigation code now in investigations/utility/.

## 2026-02-20b: Implement Phases 1-2, handoff Phase 3
**Branch**: `pietro/workingbranch`
**Handoff**: `.claude/handoffs/HANDOFF_2026-02-20_unify-kernel-phase3-gradient.md`
**Plan**: `~/.claude/plans/cozy-humming-wadler.md`
**Status**: Phases 1-2 complete, Phase 3 handed off

Phase 1 (committed 8d54cf1): kernel selection in run_single_mode.py -- create_kernel() factory in kernels.py, --kernel-type CLI, config wiring, validation guards. Phase 2 (uncommitted): unified investigations/utility/explore_utility.py with auto-detect right panel, setup(kernel_type), all 3 kernels verified. Phase 3 (gradient.py) and Phase 4 (cleanup) remaining.

## 2026-02-20: Plan for unifying kernel selection and utility investigation
**Branch**: `pietro/workingbranch`
**Handoff**: `.claude/handoffs/HANDOFF_2026-02-20_unify-kernel-selection-utility.md`
**Plan**: `~/.claude/plans/cozy-humming-wadler.md`
**Status**: Handed off for implementation

Planning-only session. Explored codebase to understand duplication across 4 kernel investigation folders. Designed 4-phase plan: (1) kernel selection in run_single_mode.py, (2) unified explore_utility.py, (3) unified gradient.py, (4) cleanup. Key decisions: exclude normalized kernel (deprecated), no MC sampling in gradient script, sigmoid bounds with 2 modes only, single set of hardcoded hyperparams (not kernel-specific). No code written.

## 2026-02-12: Multi-image DA optimization, per-sample backward, OOM fixes
**Branch**: `pietro/rbf-kernel`
**Handoff**: `investigations/rbf_kernel/HANDOFF.md`
**Status**: Continuing

Created `distribution_gradient.py` (multi-image DA utility optimization). Added BOUNDS_MODE to gradient_rbf.py. Fixed OOM in acquisition.py (no_grad for conditioning images). Wired adaptive r_max into run_rbf.py. Implemented per-sample backward approach for O(1) memory — discovered and diagnosed two bugs: (1) x_query graph freed by H_marg.backward() (fixed: recompute per sample), (2) A_val/lam0_val retain grad_fn from squeeze() (fix identified: .detach(), NOT YET APPLIED).

## 2026-02-11: RBF utility exploration + LBFGS gradient ascent scripts
**Branch**: `pietro/rbf-kernel`
**Handoff**: `investigations/rbf_kernel/HANDOFF.md`
**Status**: Continuing

Created `explore_utility_rbf.py` and `gradient_rbf.py` in `investigations/rbf_kernel/`. Initial gradient script used stale pre-LBFGS template from worktree; merged workingbranch (f3508be), then rewrote gradient_rbf.py from the correct 585-line LBFGS template (sigmoid bounds, RF-only optimization, f_max guard). User began experimenting with parameters (USE_SYNTHETIC=False, LR=0.1, SIGMA_SMOOTH=10.0).

## 2026-02-11c: Arc-sine investigation paused
**Handoff**: `investigations/arcsine_kernel/HANDOFF.md`
**Status**: Paused — branch `pietro/arcsine-kernel` parked, nothing urgent to merge to workingbranch.

## 2026-02-11b: LBFGS upgrade, f_max guard, cross-kernel parity
Implemented LBFGS optimizer in gradient_arcsine.py (replaces plain GD). Investigated norm-driven utility divergence: optimizer exploits firing rate growth (0.49→5.82 spikes), confirmed with DEBUG diagnostics. Added f_max=100.0 firing rate guard to LBFGS closure (returns +inf to reject high-rate steps). Wired f_max through default_params.json, both YAMLs, and config builders. Upgraded gradient_unnormalized.py and gradient_normalized.py with same features (LBFGS, f_max, RF metrics, diagnostics). Fixed explore_utility.py output filename. All three kernel investigation folders now consistent.

## 2026-02-11: LocalRBFKernel implementation
**Branch**: `pietro/rbf-kernel` (worktree at `gpytorch_porting_rbf_kernel`)
**Handoff**: `investigations/rbf_kernel/HANDOFF.md`
**Status**: Continuing

Implemented `LocalRBFKernel` in `kernels.py` — stationary RBF kernel using same C matrix as arc-cosine but with log-space lengthscale instead of Amp. Two commits: initial kernel (9ae87b7), lengthscale reparametrization (565717b). test_r=0.7785 on cell 8, M=50 (vs arc-cosine ~0.84). Next: multi-cell validation, M=100 test, RF parameter comparison.

## 2026-02-11: Arc-sine utility exploration scripts + LBFGS plan
**Handoff**: `.claude/handoffs/HANDOFF_2026-02-11_arcsine-utility-exploration-lbfgs.md`
**Plan**: `.claude/plans/merry-nibbling-sutton.md`
**Status**: Handed off for continuation

Created `explore_utility_arcsine.py` and `gradient_arcsine.py` in `investigations/arcsine_kernel/`. Both verified working. Arc-sine saturation confirmed: sqrt(K(x,x)) clusters near 1.0. User changed N_TRAIN to 300 for better model. LBFGS upgrade for gradient ascent planned and approved but not yet implemented.

## 2026-02-10: Fix beta viz bug, smoothed argmax RF center, port viz to run_single_mode
**Branch**: `pietro/arcsine-kernel`

**Accomplished:**
- Corrected previous session's wrong beta analysis: sigma=7.6px not 120px (parameterization confusion)
- Fixed RF sigma formula in visualization: `beta*sqrt(2)` not `sqrt(1/(2*beta))`
- Replaced CoM with smoothed argmax in `utils.py:compute_rf_center_from_sta()` (12.9px error → 0px)
- Removed DEBUG closure counters from gpy_training.py
- Ported STA+RF 2x2 visualization to `run_single_mode.py` (both default_gpy and vargp_direct)

**Documentation updated:** HANDOFF.md (corrected beta section), MEMORY.md (added beta parameterization notes)

**Known issues:** Arc-sine M=100 test_r regression (0.765→0.545), LBFGS stuck pattern at M=50

## 2026-02-10: params_in_bounds, LBFGS diagnostics, RF visualization
**Handoff**: `investigations/arcsine_kernel/HANDOFF.md`
**Status**: Continuing

Added `params_in_bounds()` to kernel and likelihood, updated all LBFGS closures. STA+RF visualization in run_arcsine.py. Found: initial beta=0.1 covers entire image (sigma=120px — CORRECTED in next session: actually 7.6px), STA CoM is 12.9px off peak. Temp debug counters in gpy_training.py need removal.

## 2026-02-10: Arc-sine kernel implementation + LBFGS NaN guard fix
**Handoff**: `investigations/arcsine_kernel/HANDOFF.md`
**Status**: Continuing

Implemented ArcSineKernel (kernels.py) and sandboxed runner (run_arcsine.py). Arc-sine at M=100: test_r=0.7653 (only 3% below arc-cosine baseline). At M=50: NaN crash — diagnosed as missing NaN guard in `gpy_training.py` LBFGS closure (same bug as normalized kernel M=50). Root cause: LBFGS line search pushes parameters to extreme values, raw_m2log2beta overflows float32 (exp(186)=inf), kernel degenerates, loss=NaN, gradients=NaN, all params become NaN. Fixed with two-layer guard matching eigenspace_mstep.py pattern: (1) return inf on NaN loss/exception in closure, (2) clamp_hyperparameters() after step with warning. Arc-sine M=50 now works (test_r=0.4623). Arc-cosine M=50 no regression (0.6815 vs 0.6819). Needs full regression test and commit.

## 2026-02-10: Arc-sine kernel planning session
**Handoff**: `.claude/handoffs/HANDOFF_2026-02-10_arcsine-kernel-implementation.md`
**Plan**: `.claude/plans/merry-nibbling-sutton.md`
**Status**: Handed off for implementation

Math-focused planning session for a new kernel based on Williams (1998) arc-sine / erf-network kernel. Same C matrix and RF structure as arc-cosine, but K(x,x) saturates at 1 instead of growing quadratically. Discussed origin (single hidden layer erf network, same depth as arc-cosine), parameter roles (Amp controls saturation rate, not ceiling), saturation concern (v_x >> 1 for PNAS data with current params), and initialization strategy (default Amp=1.0, evaluate empirically). Implementation sandboxed in `investigations/arcsine_kernel/`, no mainline changes.

## 2026-02-10: DA utility gradient ascent investigation (normalized kernel)
**Handoff**: `investigations/normalized_kernel/HANDOFF_GRADIENT.md`
**Status**: Continuing

Created `gradient_normalized.py` -- gradient ascent investigation for DA utility with normalized kernel. Tested 3 approaches: additive noise, Gaussian smoothing, synthetic bipartite. Key finding: interpolation monotonicity confirmed in all valid cases, but gradient ascent consistently diverges from target (exploits H_marg via high posterior-variance regions). Next: reproduce with unnormalized kernel in `investigations/understanding_utility/`.

## 2026-02-09: Normalized kernel utility exploration
**Handoff**: `investigations/normalized_kernel/HANDOFF.md`
**Status**: Continuing

Created `explore_utility_normalized.py` -- copy of `explore_utility.py` trained with `ArcCosineKernelNormalized`. Confirmed normalized kernel eliminates utility divergence: scaling images 5x causes <5% utility change (vs 38-56x for unnormalized). All norms constant at 1.0. Added 20 random pool images (square markers) to both scripts' landscape plots. M=50 causes all-NaN Cholesky with normalized kernel (use M=100).

## 2026-02-09: Normalized kernel implementation + training validation
**Handoff**: `investigations/normalized_kernel/HANDOFF.md`
**Status**: Continuing

Implemented `ArcCosineKernelNormalized` in kernels.py (17/17 validation tests pass). Training on PNAS cell 8 shows 25% test_r drop (0.79 -> 0.59) -- image norm is genuinely informative for neural encoding.

## 2026-02-09: Clean up explore_utility.py + add DA landscape visualization

**Branch**: `pietro/acquisition-functions`

**Accomplished:**
- Cleaned up `explore_utility.py`: removed local utility wrappers, uses `acquisition.py` functions with adaptive r_max
- Added `eval_da_conditioned()`: DA utility table for training images conditioned on one image
- Added 2-panel figure: H(R) landscape heatmap + U_DA vs norm colored by angle
- Updated `HANDOFF.md`

**Documentation updated:** `investigations/understanding_utility/HANDOFF.md`

**Known issues:** none

## 2026-02-08: Plan 2D Playground Import Cleanup
**Handoff**: `.claude/handoffs/HANDOFF_2026-02-08_cleanup-2d-playground-imports.md`
**Plan**: `.claude/plans/partitioned-scribbling-hanrahan.md`
**Status**: Handed off for implementation

Explored 2D playground import chain (5+ levels deep), identified duplicated functions and inconsistent kernel/likelihood usage. Agreed on approach: gpytorch_porting as single source of truth for math, add SimpleArcCosineKernel, move adaptive_r_max, consolidate DA utility into acquisition.py. Keep 1D's VariationalGP and training functions.

## 2026-02-08: Import Cleanup Implementation + Test Fix + r_max Audit
**Handoff**: `.claude/handoffs/HANDOFF_2026-02-08_enforce-explicit-rmax.md`
**Plan**: `.claude/plans/partitioned-scribbling-hanrahan.md` (overwritten with r_max plan)
**Status**: Handed off for implementation

User implemented the 2D import cleanup plan. Fixed test_acquisition.py (updated imports from old utility.py/utility_2d_rbf_base to gpytorch_porting/utils.py via importlib.util pattern). All 6 tests pass. Deep audit confirmed 2D playground imports are clean. Found r_max hardcoded defaults in compute_H, nd_utility_new, standard_utility, distribution_aware_utility, compute_mc_diagnostics_2d. Planned enforcement: remove all silent defaults, require explicit r_max or adaptive_r_max=True.

## 2026-06-17: Lucent useful-image generation (solved the overblow problem)
**Folder**: `investigations/lucent_useful_images/`
**Status**: Complete. COMMITTED on branch `pietro/lucent-useful-images` (off 75b207a, the
superrepo-pinned submodule commit). Figures are gitignored (regenerable): current deliverable
in `out/panels/`; early M=50 figures archived in `early_exploration_fixed_M50/`.

Retried "most useful image" synthesis with the **lucent** library. Parameterize the image
with lucent's Fourier 1/f prior + sigmoid, affine-mapped to the dataset range [-2.40,+2.48]
→ **bounded by construction** (no clipping/rescaling). Maximize the **distribution-aware
utility** (`acquisition.py`, default_gpy). Optimized images stay within the pixel range
everywhere (saturation ≤7%, mostly <2%) — the long-standing contrast/border blow-up is
gone. Key insight: the utility intrinsically favors UNnatural images; naturalness comes
from the parameterization (+ natural-image start), not the objective. Headline science: as
n_train ↑, the optimum shifts from a broad epistemic RF probe to a compact localized RF.
Figures: 3 grids (5 cells × n_train: natural / preferred-stimulus / RF-difference) +
per-cell heroes. Gotcha: default_gpy unstable at low n_train for most cells (memory
`default-gpy-low-ntrain-unstable`); screened the full ladder → cells 13,3,1,11,12.

## 2026-06-17: LUT-backed standard utility — handed off to a new session
**Handoff**: `investigations/lucent_useful_images/standard_utility/HANDOFF_lut_standard_utility.md`
**Plan**: `investigations/lucent_useful_images/standard_utility/PLAN_lut_standard_utility.md`
**Status**: Handed off for continuation (new session)

Make the precomputed utility LUT (`analysis/figures/utility_landscape/`) usable inside the
lucent optimization as a torch-differentiable, **standalone (vendored)** module, to fix the
standard-utility `r_max=100` blow-up (295,800 nats / firing 26,605 in the standard_utility
control). Standard utility only, DA deferred. ADDITIVE files under
`investigations/lucent_useful_images/` ONLY — no engine edits (keeps the submodule
byte-identical to the superrepo-pinned 75b207a, which the `analysis/` pipeline uses).
Promotion into `acquisition.py` deferred to a branch off `pietro/workingbranch`. See the
handoff for the full rationale + the `*.npz` gitignore gotcha for the vendored `lut.npz`.
(DONE 2026-06-18 — see `investigations/lucent_useful_images/standard_utility/LUT_IMPLEMENTATION_REPORT.md`;
verified independently: 295,800 -> 3.724 nats, no engine file touched.)

## 2026-06-18: PNAS cell screening for image-pipeline testbed — handed off to a new session
**Handoff**: `investigations/lucent_useful_images/cell_screening/HANDOFF_cell_screening.md`
**Plan**: `investigations/lucent_useful_images/cell_screening/PLAN_cell_screening.md`
**Status**: Handed off for continuation (new session)

Screen the PNAS cells (0-40, default_gpy, M=n_train) for a few **testbed cells with an easy
monotonic test_r increase vs dataset size**, so the lucent image pipeline can be demonstrated
without the noisy-fit confound. Constraints: n_train = M = {50,100,150,200,250,300} (step 50),
ONE seed (42). The new session must FIRST discuss the cell-quality scoring methodology with the
user, then extend the existing `screen_ladder.py`. ADDITIVE files under `cell_screening/` ONLY,
no engine edits (stay on the pinned 75b207a engine). NOT the closed-loop analysis cells — PNAS
only. Known confound flagged in the handoff: 75b207a is 106 commits behind workingbranch's
fit-stability fixes.
(DONE 2026-06-18 — `cell_screening/FINDINGS.md`, commit `228418a`. Testbed cells: **3, 13, 36**
(PNAS, default_gpy, M=n_train). Honest finding: no PNAS cell has an ideal gradual 0.3->0.9 climb.)

## 2026-06-18: Lucent investigation — continuation handoff (next session, with testbed cells)
**Handoff**: `investigations/lucent_useful_images/HANDOFF.md` (the top-level "start here" for the
whole investigation; sub-handoffs in `standard_utility/` and `cell_screening/` are DONE)
**Status**: Handed off for continuation (exploratory)

Continue the lucent "most-useful image" investigation on the chosen testbed cells 3, 13, 36:
re-run the DA-utility image optimization across the n_train ladder and look at whether the
optimized image sharpens as the model improves, then explore the open knobs (DA vs standard_lut,
sample_lambda=True, an STA-correlation "is-this-the-RF" metric, decay_power). Full journey +
findings + what-to-read + what-NOT-to-try in HANDOFF.md. ADDITIVE files under the investigation
folder ONLY; no engine edits (byte-identical to pinned 75b207a). Still exploratory.

## 2026-07-04: Lucent gray-start crystallization on testbed cells 3, 13, 36
**Findings**: `investigations/lucent_useful_images/FINDINGS_gray_start.md`
**Status**: Continuing (exploratory) — next: `sample_lambda=True` robustness run

Ran the DA-utility image optimization on the testbed trio (M=n_train, seed 42, sample_lambda=False).
Gray-start view shows all three cells crystallize diffuse->compact center-surround RF as test_r
climbs (cell 13 sharpest @0.96, cell 3 @0.83, cell 36 @0.73). The natural-start concentration
metric is confounded by the scene background (corrected mid-session; gray-start is the clean view).
fr~0 for cells 3/36 explained by small response gain A~0.087 (vs cell 13 A~0.24) — their optima are
epistemic probes the cell barely fires to; cell 13's is a genuine high-response stimulus. lucent's
FFT param changes the whole image (global ripple), so "only the RF changes" is too strong. New
additive scripts: `diff_panels.py`, `rf_localization.py`, `gray_panels.py`, `firing_diagnostic.py`.
No engine edits (pinned 75b207a). Next: rerun with `sample_lambda=True` (unbiased DA).

**Update (same day):** `sample_lambda=True` DONE — crystallization is robust (True ≈ False in the
mean; biased mean-λ is a fine deterministic proxy). Single unbiased realization is noisier on the
low-gain cells 3 & 36. n_mc=48 cell-dependent: cell 36 needs ≥96 (spread 0.14→0.02), cell 3 already
stable, cell 13's residual is structural (not fixed by n_mc). New additive scripts:
`compare_sl_panels.py`, `noise_probe.py`, and a `--sample-lambda` flag on `gray_panels.py`. Figures:
`cell{N}_gray_sl`, `cell{N}_sl_compare`, `sl_divergence`, `cell{N}_sl_noise`, `sl_noise_summary`.
See `FINDINGS_gray_start.md` "sample_lambda robustness".

## 2026-07-04: Next-step design — in-silico oracle loop (planned, no code)
**Handoff**: `investigations/lucent_useful_images/oracle_loop/HANDOFF.md`
**Status**: Planned (design only)

Captured the design for a future session: use the full-data ceiling GP as an "oracle cell" and the
lucent generator as an acquisition strategy in a simulated closed loop. Two-part split — Part 1
reproduce `analysis/cross_sessions/results/v0.7/pooled_explained_variance_no_greedy.svg` in silico
(oracle as cell, active/random only) to validate the machinery, then Part 2 add the generated-image
arm. Locked decisions + the dataset/engine bridge (ceilings=closed-loop/analysis engine vs
generator=PNAS/default_gpy) and branch straddle (analysis/april26 vs pietro/lucent-useful-images) to
resolve with the user first are all in the handoff.
