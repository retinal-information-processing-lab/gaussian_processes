# Decision Log - GPyTorch Porting Project

This document records design decisions made during the porting effort.
Consult this when something about the implementation seems confusing or when facing similar choices.

For project status and quick reference, see `CLAUDE.md`.

---

## Session 1: Initial Planning (January 2025)

**Q1: How close do results need to match?**
> A: Qualitative match is sufficient. Exact numerical match not required.

**Q2: Should GPyTorch handle M-step (hyperparameter optimization)?**
> A: Yes, for initial implementation. Let GPyTorch's Adam optimize the ELBO w.r.t. hyperparameters. Custom analytical gradients deferred to later.

**Q3: What test data to use?**
> A: PNAS data from `one_cell_fit.py` - same dataset as current implementation.

**Q4: Should utility functions be ported?**
> A: NO. Utility functions (active learning, information gain) are explicitly OUT OF SCOPE for this effort.

**Q5: Should custom E-step be implemented now?**
> A: Deferred to later. Start with GPyTorch's variational inference.
>
> **Rationale**:
> - Custom E-step is a closed-form Newton update (faster than iterative)
> - But adding it later is straightforward (just replace optimizer loop)
> - Starting simple validates model structure first
> - GPyTorch's natural gradient optimizer is decent baseline

**Q6: Is hybrid approach practical to add later?**
> A: Yes, very practical. GPyTorch exposes `variational_mean` and `chol_variational_covar`. We can:
> 1. Read current parameters from GPyTorch model
> 2. Apply custom E-step update
> 3. Write back to GPyTorch model
>
> Only complexity: convert between Cholesky (GPyTorch) and full V (custom).

**Q7: Should we start with RBF kernel or arc-cosine with C=I?**
> A: **Arc-cosine with C=I** (identity covariance matrix, no RF structure).
>
> **Rationale**: RBF kernel is stationary (depends on distance), while arc-cosine is non-stationary (depends on actual input values via xᵀCx'). If RBF works but arc-cosine fails, we wouldn't know if the issue is:
> 1. The kernel implementation
> 2. The GPyTorch integration
> 3. Something inherent to arc-cosine with high-dimensional data
>
> Starting with arc-cosine (C=I) keeps the kernel math the same while removing RF complexity. This is a better stepping stone than RBF.

## Session 2: Stage 1 Validation and Debugging (January 2025)

**Q8: How to add amplitude scaling to the kernel?**
> A: Use GPyTorch's `ScaleKernel` wrapper rather than building amplitude into ArcCosineKernel.
>
> **Rationale**:
> - `ScaleKernel` is GPyTorch's standard pattern: `K_scaled = outputscale × K_base`
> - Equivalent to `Amp` parameter in original implementation
> - Keeps ArcCosineKernel simple and matching reference implementation
> - Can verify kernel correctness independent of scaling
>
> **Alternative considered**: Add `Amp` parameter directly to ArcCosineKernel
> - Rejected because: complicates kernel unit test (must match reference exactly)

**Q9: What precision (dtype) is required?**
> A: **float64 is required** for numerical stability.
>
> **Problem**: Arc-cosine kernel values are ~10,000 for PNAS data (11,664 pixels per image). With float32:
> - Cholesky decomposition fails with NaN
> - Precision loss in kernel matrix computations
>
> **Solution**: Use `model.double()` and load data with `dtype=torch.float64`
>
> **Note**: Order matters - call `.double()` BEFORE `.to(device)`

**Q10: Why does arc-cosine with C=I perform poorly on PNAS data?**
> A: **C=I is fundamentally unsuited** for image data without receptive field structure.
>
> **Experimental findings**:
> | Test | RBF Kernel | Arc-Cosine (C=I) |
> |------|-----------|------------------|
> | Synthetic (linear) | r = 0.94 | r = 0.43 |
> | PNAS real data | (not tested) | r ≈ 0.2 |
>
> **Root cause**: Arc-cosine with C=I captures only:
> - Input norms: `||x||²`
> - Angles between inputs: `x·x' / (||x|| ||x'||)`
>
> For random Gaussian inputs (or images with similar total energy), all points have similar norms → similar kernel values → constant predictions.
>
> **The original implementation works because C matrix encodes**:
> - Locality (β, eps_0x, eps_0y): which pixels matter
> - Smoothness (ρ): how nearby pixels correlate
>
> **Conclusion**: Stage 2 (structured C) is essential for PNAS data, not optional.

**Q11**: GPyTorch deprecation warnings? → **No action** - cosmetic, from `linear_operator` internals.

**Q12**: Rename `kernels.py` to avoid ambiguity? → **Deferred**. Two files share name but Python's search order works.

**Q13: Will C=I in Stage 2 reproduce Stage 1 results?**
> A: **Yes, exactly.** This is a key validation check.
>
> Mathematically:
> - C=None: `V = ||x||² + σ₀²`, cross = `x·x' + σ₀²`
> - C=I: `V = xᵀIx + σ₀² = ||x||² + σ₀²`, cross = `xᵀIx' + σ₀² = x·x' + σ₀²`
>
> Only difference: C=None avoids unnecessary `x @ I` matrix multiply (efficiency).
> Kernel values are identical.

## Session 3: Stage 2 Planning (January 2025)

**Q14**: Pixel masking in Stage 2? → **Deferred initially** (adds complexity). Later implemented - see Q22.

**Q15**: C matrix in separate class? → **No** - integrate into ArcCosineKernel. Keeps kernel logic in one place, avoids over-engineering.

**Q16**: Amplitude handling? → **Keep using ScaleKernel wrapper** (GPyTorch standard pattern, consistent with Stage 1). Alternative rejected: adding Amp directly to ArcCosineKernel would break Stage 1 compatibility.

**Q17**: Stage 2 validation approach? → **Two-step**: (1) C=I equivalence test (large β,ρ → C≈I), (2) Performance test (proper RF params → r > 0.5).

## Session 4: Validation Tests (January 2025)

**Q18: How does GPyTorch compare to reference implementation?**
> A: **Excellent match.** Test A ran both implementations on same data with proper hyperparameter learning.
>
> | Metric | Reference (varGP) | GPyTorch | Diff |
> |--------|-------------------|----------|------|
> | Pearson r | **0.8697** | **0.8433** | 0.0264 |
> | Final beta | 0.0606 | 0.1038 | 0.0432 |
> | Final rho | 0.0655 | 0.0797 | 0.0142 |
> | Final A | 0.0169 | 0.9489 | 0.9320 |
>
> **Settings**: ntilde=200, n_train=2000, maxiter=300 (ref) / iterations=500 (GPyTorch)
>
> **Success criterion**: Pearson r within 0.1 ✓

**Q19: Does A initialization matter?**
> A: **NO - model is robust to A initialization.**
>
> | A_init | Pearson r | Final beta | Final rho |
> |--------|-----------|------------|-----------|
> | 1.0 | 0.8714 | 0.1001 | 0.0851 |
> | 0.01 | 0.8715 | 0.0781 | 0.0696 |
>
> Both converge to same Pearson r (diff = 0.0001).
> Lower A_init (0.01) achieves better ELBO (1577 vs 1641) and learns smaller RF (beta closer to reference).

**Q20: Can RF center learn from bad initialization?**
> A: **NO - this is a limitation.** RF center barely moves from bad initialization.
>
> | Init eps_0 | Final eps_0 | Pearson r |
> |------------|-------------|-----------|
> | (0.0, 0.0) | (0.11, -0.04) | **0.87** |
> | (0.5, 0.5) | (0.53, 0.43) | **0.35** |
>
> **Root cause**: Adam struggles to move RF center far from initial position. The loss landscape may have local minima.
>
> **Practical implication**: Start with eps_0 near (0,0) or use prior knowledge about RF location.

## Session 5: Evaluation Metrics (January 2025)

**Q21: Which evaluation metric should we use?**
> A: **Explained variance** (Pearson r / reliability), matching `utils.py:explained_variance()`.
> Added `compute_explained_variance()` to `train.py` and `--plot`/`--save-plot` to `test_fit.py`.

## Session 6: Pixel Masking (January 2025)

**Q22: How should pixel masking be implemented?**
> A: Match reference implementation in `kernels/kernels.py:localker_clean()`.
>
> **Key design choices:**
> - Mask computed with **detached** theta parameters (structural stability during backprop)
> - Mask applied internally in `forward()` - user passes full images, kernel handles masking
> - `use_mask=True` by default when `n_px_side` is set
>
> **Hard-coded values:**
> - `MASK_THRESHOLD = 0.001` - pixels with locality weight α >= 0.001 included (matches reference)
> - Typical mask size: ~2400-2500 pixels (out of 11664) for beta=0.1, eps_0=(0,0)
>
> **Validation test tolerances** (`tests/test_mask_validation.py`):
> - Mask equivalence: exact match required
> - C matrix equivalence: max diff < 1e-6 (absolute)
> - Kernel equivalence: relative diff < 1e-5
> - End-to-end fit: Pearson r difference < 0.05 between masked and full
>
> **Memory reduction**: 11664×11664 (~1GB) → ~2480×2480 (~50MB) = ~20x reduction
>
> **Session notes**: `.claude/archive/ARCHIVE_2026-01-14_pixel_masking_session_notes.md`

## Session 7: Training Loop Comparison (January 2025)

**Q23: Training mode comparison?**
> A: `efm` mode (E-F-M loop) outperforms pure `adam` for M≤50.
>
> **Canonical test**: `python tests/test_estep_comparison.py`
>
> This script compares varGP (reference), GPyTorch efm, and GPyTorch adam with frozen parameters from `one_cell_fit.py`.

**Q24: Should we use softplus or exp/log for A parameterization?**
> A: **exp/log** (A = exp(raw_A), raw_A = logA).
>
> **Rationale**:
> - Matches original varGP's logA parameterization exactly
> - Simpler code (no inverse softplus computation in f_step_lbfgs)
> - Testing showed equivalent performance between transforms
> - Multiplicative gradient scaling (same relative change at any A value)

## Session 8: E-step Kernel Caching (January 2025)

**Q25: How to fix 4.9x E-step slowdown in `vargp_style` vs original varGP?**
> A: **Cache kernel matrices K and K̃** and reuse across Newton iterations within E-step.
>
> **Root cause**: Profiling revealed 35 kernel calls per E-step loop (10 Newton steps) vs ideal 2.
> Each `model(X)` call and explicit kernel computation was redundant since kernel params don't change during E-step.
>
> **Solution**: New functions in `estep.py`:
> - `compute_kernel_cache()`: Compute K, K̃, k0 once before E-step loop
> - `compute_moments_from_kernel_cache()`: Compute λ_m, λ_var from cached matrices (bypasses GPyTorch `model(X)`)
> - `e_step_with_kernel_cache()`: Newton update using cached kernels (bypasses GPyTorch)
>
> **Performance results** (M=50, N=500):
> | Path | Test r | E-step Time | Total Time |
> |------|--------|-------------|------------|
> | varGP (reference) | 0.8141 | 1.1s | 5.2s |
> | GPyTorch cached | 0.7752 | 1.0s | 6.4s |
> | GPyTorch non-cached | 0.7870 | 8.8s | 16.1s |
>
> **Key achievement**: GPyTorch E-step now **faster than original varGP** (1.0s vs 1.1s).
>
> **Testing**: Use `--no-cache` flag to test non-cached fallback path:
> ```bash
> python test_estep_pnas.py --mode vargp_style --ntilde 50 --no-cache
> ```
>
> **Note**: Small performance difference (r=0.7752 vs r=0.7870) between cached and non-cached paths may warrant investigation. Reference commit for original non-cached code: `44d9227`.
>
> **Documentation**: See `.claude/archive/ARCHIVE_2026-01-18_kernel_caching_and_whitening.md` and `results/PROFILING_2026-01-18.md` for details.

## Session 9: Whitening Investigation (January 2025)

**Q26: Does GPyTorch auto-adjust whitened variational parameters when kernel changes?**
> A: **NO.** GPyTorch does NOT automatically re-whiten or adjust variational parameters when kernel hyperparameters change.
>
> **Investigation**: See `TECHNICAL_ANALYSIS_2026-01-20_whitening_LK_mismatch.md` for full analysis.
>
> **Core Finding**: GPyTorch's whitening design assumes joint gradient optimization where autograd handles the coupling between L_K (Cholesky of inducing kernel) and whitened parameters implicitly. EM-style optimization with closed-form E-step bypasses autograd and creates an inconsistency.
>
> **The L_K Mismatch Problem**:
> 1. E-step stores whitened params: `m_stored = L_K_old^{-1} @ m_natural`
> 2. M-step changes kernel → `L_K` becomes `L_K_new`
> 3. Next E-step reads: `m_corrupted = L_K_new @ m_stored = L_K_new @ L_K_old^{-1} @ m_natural ≠ m_natural`
> 4. Corruption factor: `L_K_new @ L_K_old^{-1}` causes ~8x errors in λ_m
>
> **Why Joint Optimization Works**: Autograd differentiates through L_K, so gradients for both kernel and variational params account for the coupling. No explicit re-whitening needed.
>
> **Why EM Fails**: Our Newton E-step is closed-form (no autograd involvement). Stored whitened params become "stale" after M-step changes kernel.
>
> **Sources examined**:
> - GPyTorch source: `variational_strategy.py` lines 212-216 (mean formula), 238-272 (auto-whitening)
> - GitHub issues: #1308, #1754, #1556 (cache issues), PR #903 (whitening redesign)
> - Academic: Matthews 2017 (whitening derivation), Salimbeni 2018 (natural gradients)

**Q27: What is the solution for EM-style optimization with GPyTorch?**
> A: **Use `UnwhitenedVariationalStrategy`** instead of the default `VariationalStrategy`.
>
> **Why it solves the problem**:
> - Stores natural (unwhitened) params directly: m_natural, V_natural
> - No L_K transformation involved in storage or retrieval
> - Prior is actual N(μ_Z, K_ZZ), not transformed N(0, I)
> - Predictive mean uses `K_XZ @ K_ZZ^{-1} @ m` (standard SVGP formula)
> - When kernel changes, interpretation of stored params is unchanged
>
> **Comparison**:
> | Aspect | VariationalStrategy | UnwhitenedVariationalStrategy |
> |--------|--------------------|-----------------------------|
> | Stored params | whitened (L_K dependent) | natural (L_K independent) |
> | EM compatible? | NO (L_K mismatch) | YES |
> | Prior | N(0, I) | N(μ_Z, K_ZZ) |
>
> **Trade-offs**:
> - Cons: Slightly worse numerical conditioning, ~5% slower for M=50-200
> - Pros: Eliminates ALL whitening issues, simpler code, EM-compatible
>
> **Migration**: Change one import in `model.py`, remove whitening conversion functions from `estep.py`.
>
> **Documentation**: `TECHNICAL_ANALYSIS_2026-01-20_whitening_LK_mismatch.md` Section 9

**Q28: How to use UnwhitenedVariationalStrategy in the codebase?**
> A: Added as an **alternative** via `standard_variational_distribution` parameter (default `True`).
>
> **Usage**:
> ```python
> # In code:
> model = VariationalGPModel(inducing_points, kernel, standard_variational_distribution=False)
>
> # CLI:
> python run_single_mode.py --mode vargp_style --unwhitened-variational-dist --no-explicit-unwhitening
> ```
>
> **Implementation details** (naming updated 2026-01-23):
> - `model.py`: `standard_variational_distribution` parameter selects strategy type
> - `estep.py`: `explicit_unwhitening` parameter (required, no auto-detection) controls L_K conversions
> - `compute_kernel_cache()`: Skips L_K computation when `model.standard_variational_distribution=False`
>
> **Performance comparison** (M=50, N=500):
> | Strategy | Test r | Time | Notes |
> |----------|--------|------|-------|
> | Whitened (default) | 0.80 | 7.6s | Better optimization dynamics |
> | Unwhitened | 0.65 | 30.4s | Simpler math, slower |
>
> **When to use unwhitened**:
> - Investigating EM-style optimization behavior
> - Debugging whitening conversion issues
> - Research/comparison purposes
>
> **Files modified**:
> - `model.py`: Added `whitening` parameter, conditional strategy selection
> - `estep.py`: Auto-detect whitening, conditional L_K computation
> - `test_estep_pnas.py`: Added `--unwhitened` flag
> - `tests/test_estep_comparison.py`: Added `--unwhitened` flag

**Q29: Why does UnwhitenedVariationalStrategy achieve worse accuracy?**
> A: **Unknown - tentative hypothesis is KL divergence gradient instability.**
>
> **Observed behavior** (2026-01-20, `tests/diagnose_unwhitened_performance.py`):
>
> K̃ condition number: 1.02e+04. Gradient evolution:
> | Iter | Whitened KL | Unwhitened KL | Grad Ratio |
> |------|-------------|---------------|------------|
> | 0 | 0.00 | 0.16 | 0.64x |
> | 4 | 0.02 | **423.26** | **159x** |
>
> **Tentative hypothesis**: The unwhitened KL term `mᵀ K̃⁻¹ m` may amplify gradients.
> In whitened space, prior is N(0, I) so ∂KL/∂m_w ≈ m_w (no K̃⁻¹).
>
> **Status**: 16% accuracy gap (0.6878 vs 0.8381) is observed but **root cause not fully validated**.
> - E-step Newton update appears correct
> - M-step optimization behavior differs significantly
> - Further investigation needed to confirm hypothesis
>
> **Documentation**: `.claude/archive/ARCHIVE_2026-01-20_unwhitened_investigation.md`

---

## YAML Experiment System (February 2025)

**Q26: Bug fix — vargp_direct ignored `--lr` CLI flag**
> The old `run_single_mode.py` line 481 used `defaults['training']['lr']` directly instead of `args.lr` for vargp_direct mode. This meant `--lr` CLI overrides were silently ignored. Fixed during the `run_single_config()` refactor — `lr` now comes from the config dict in all modes.
