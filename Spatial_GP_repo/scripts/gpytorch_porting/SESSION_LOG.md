# Session Log

## 2026-03-04: Diffusion-GP integration implementation
**Handoff**: `investigations/diffusion/HANDOFF_guided_optimization.md`
**Status**: Continuing -- optimization tuning needed

Implemented guided_optimization.py with both stages. Stage 1 (setup_gp) complete: test_r=0.7734 at 64x64, works for any square crop size. Stage 2 (combined optimization) works end-to-end but optimization dynamics need tuning. Key findings: LBFGS fails with combined objective (strong_wolfe line search can't handle competing forces), Adam normalizes away lambda_diff, SGD with momentum is the right optimizer. Tweedie L2 pull prevents structural degradation but not norm amplification. Next: try SGD lr=0.01, add pixel bounds.

## 2026-03-03: Diffusion-GP integration planning session
**Handoff**: `.claude/handoffs/HANDOFF_2026-03-03_diffusion-gp-integration-plan.md`
**Plan**: `investigations/diffusion/PLAN_guided_optimization.md`
**Status**: Handed off for implementation

Planning-only session. Explored both the GP utility optimization pipeline (gradient.py, explore_utility.py, acquisition.py, run_single_mode.py) and the diffusion model (diffusion_model.py, train.py, sample.py). Identified 6 key tricky parts for integration, most critically that run_single_config() hardcodes the data path so we must replicate the pipeline manually. Designed two-stage plan: (1) reusable setup_gp(crop_size) function that trains GP on center-cropped images of any square size (key deliverable), (2) combined LBFGS optimization with utility gradient + Tweedie denoising direction from diffusion model. No code written.

## 2026-03-03: Diffusion model — training, T-1 sampling fix, generation working
**Handoff**: `investigations/diffusion/HANDOFF.md`
**Reference**: `investigations/diffusion/REFERENCE.md`
**Status**: Continuing — unconditional generation works, next step is GP integration

Ran 500 and 1000 epoch training. Discovered generated images were all black due to cosine schedule instability at t=T=1000 (31.6x amplification in reverse formula). Fixed by starting reverse loop from T-1. Generation quality good after fix — pixel histogram matches real data. Created REFERENCE.md as single entry point for future sessions.

## 2026-03-03: Diffusion model investigation — code complete, training pending
**Handoff**: `investigations/diffusion/HANDOFF.md` (superseded by above)
**Plan**: `investigations/diffusion/PLAN_diffusion_model_training.md`
**Status**: Continuing — next session runs full training and evaluates generation quality

Built self-contained DDPM for 64x64 natural image generation. Decided on 64x64 (not 108x108) for clean U-Net architecture and massive random-crop augmentation. Three Python files: diffusion_model.py (2.16M param U-Net, cosine schedule), train.py (data pipeline with D4 augmentation), sample.py (generation + evaluation plots). All smoke-tested on CUDA. No real training run yet — that is the next step (500 epochs, ~10 min). Branch: `pietro/diffusion-investigation` in worktree at `gpytorch_porting_diffusion/`.

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
