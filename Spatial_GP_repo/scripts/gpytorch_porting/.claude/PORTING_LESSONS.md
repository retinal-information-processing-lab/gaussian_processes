# Porting Lessons & Dead Ends

**Compiled**: April 2026
**Scope**: Complete porting effort from varGP (utils.py) to GPyTorch, January-March 2026
**Audience**: Future sessions inheriting this codebase. On-demand only (not auto-loaded).

The port achieved functional parity (test_r ~ 0.84 on reference cell 8, M=50) with two implementations: **vargp_direct** (eigenspace-based, exact match to original) and **default_gpy** (standard GPyTorch). The journey revealed critical incompatibilities between EM-style optimization and GPyTorch's whitening, parametrization pitfalls, and subtle numerical conditioning issues.

---

## 1. Dead Ends (What Was Tried & Why It Failed)

### 1.1 vargp_style Mode (Whitened GPyTorch + Custom E-step)

**Status**: DEPRECATED (code in `deprecated/`)
**Attempt**: Combine GPyTorch's `VariationalStrategy` (whitened params) with custom Newton E-step.

**Why it failed** (DECISION_LOG Q26-Q29):
- **Fundamental incompatibility**: GPyTorch's whitening assumes joint autograd optimization where kernel and variational params update together in one backward pass
- **EM breaks this coupling**: Custom E-step stores `m_w = L_K^{-1} @ m_natural` with kernel theta_old. M-step changes theta -> new L_K. Next E-step reads corrupted `m = L_K_new @ (L_K_old^{-1} @ m_natural) != m_natural`
- **Mismatch factor**: `L_K_new @ L_K_old^{-1}` causes ~8x errors in lambda_m predictions
- **UnwhitenedVariationalStrategy** eliminates L_K mismatch but suffers 30% accuracy loss + 4.6x slowdown (0.80 -> 0.65 test_r, 7.6s -> 30.4s at M=50)

**Why NOT to revisit**: Not a bug or limitation — it's an architectural choice for joint optimization. EM requires either explicit re-whitening after M-step (O(M^3) per iteration), natural parameter storage, or bypassing VariationalStrategy entirely (our vargp_direct solution).

**Docs**: `investigations/archive/INVESTIGATION_gpytorch_whitening_kernel_mismatch.md`, DECISION_LOG Q26-Q29

### 1.2 LogInterval Constraint for A Parameter

**Status**: ATTEMPTED, FAILED
**Attempt**: Bound A in [0.001, 0.15] using sigmoid-based LogInterval constraint to prevent explosion.

**Why it failed**:
- Prevented explosion (good) but caused stickiness: A collapses to lower bound 0.001 for all seeds
- test_r degraded from 0.80 -> 0.70
- **Root cause**: Sigmoid saturation near bounds. Gradient dA/draw -> 0 at extremes. Small A -> poor predictions -> optimizer reduces A further -> stuck at boundary

**Resolution**: Used clamped exponential + LBFGS closure defense instead (3-layer architecture: params_in_bounds, NaN guard, clamp_after_step).

**Docs**: `investigations/archive/LOGINTERVAL_PARAMETERIZATION_ATTEMPT.md`

### 1.3 Normalized Arc-Cosine Kernel (K_bar with diagonal = 1.0)

**Status**: INVESTIGATED Feb 2026, DEPRECATED
**Attempt**: K_bar(x,y) = J(theta)/pi with constant diagonal to eliminate norm-driven utility divergence.

**Why it failed**:
- test_r drops ~25%: 0.79 -> 0.59
- Image norm **is genuinely informative** for neural firing rate prediction. Normalizing it away loses real predictive signal.

**Why NOT to revisit**: The norm confound in utility optimization should be addressed at the optimization stage (pixel bounds, f_max guard, norm-invariant metrics), not by destroying information in the kernel.

### 1.4 Arc-Cosine with C=I (Identity Covariance)

**Status**: Stage 1 validation tool, failed on real data
**Attempt**: Skip C matrix structure, use identity. Validates kernel + GPyTorch integration in isolation.

**Results**:
- Synthetic (linear): r = 0.94 (looks good)
- PNAS real data: r ~ 0.2 (useless)

**Root cause**: C=I captures only input norms and angles. Random images with similar total energy -> similar kernel values -> near-constant predictions. The structured C (locality via beta, smoothness via rho) is essential for real neural data.

**Key learning**: Always validate on real data early. Synthetic data masks structural inadequacies.

### 1.5 Pivoted Inducing Point Selection

**Status**: IMPLEMENTED but not default (on branch `pietro/investigate-n50-failure`)
**Attempt**: Use gpytorch.functions.pivoted_cholesky to select inducing points that maximize coverage.

**Results**: Moderate test_r improvement but doesn't prevent Cholesky crashes alone. Not merged because the benefit didn't justify the added complexity for the standard workflow.

---

## 2. Key Debugging Discoveries

### 2.1 Whitening Collapse is SEED-DEPENDENT, Not M-Dependent

**Initial hypothesis**: M=75 is numerically problematic.
**Actual finding**:
- Seed 42: M=75 collapses (test_r=0.076), M=50 OK (0.845)
- Seed 123: M=75 OK (0.832), no collapse at any M
- Seed 456: M=50 collapses (test_r=-0.000), M=75 OK (0.798)

**Root cause**: Certain random inducing point selections create ill-conditioned K matrices that amplify whitening transformation errors.

**Rule**: Always test multiple seeds before diagnosing structural problems. Seed-dependent failures = numerical conditioning issues, not algorithmic bugs.

### 2.2 A Parameter Explodes via Positive Feedback Loop

**Trajectory** (M=75, seed=42):

| Phase | Iters | A value | Firing rates | Loss |
|-------|-------|---------|-------------|------|
| Growth | 1-37 | 0.01 -> 0.11 | Normal (0.2-8.0) | ~410-447 |
| Explosion | 38-43 | 0.12 -> 0.16 | Erratic | ~430-531 |
| Collapse | 44-50 | 0.16 -> 0.0004 | Constant 1.0 | ~500-510 |

**Mechanism**: Bad inducing points -> ill-conditioned E-step -> unstable lambda predictions -> LBFGS overcompensates A -> firing rates saturate -> zero gradient signal -> degenerate basin.

**Diagnostic**: Watch for loss plateaus with constant predictions.

**Docs**: `investigations/archive/FIRING_RATE_INSTABILITY_FINDINGS.md`

### 2.3 Hyperparameters Explode Without Bounds

**Finding**: GPyTorch port initially had NO bounds on beta, rho, eps_0. Original varGP enforces bounds via LBFGS closure.

| Seed | Beta range | Rho range | C condition |
|------|-----------|-----------|-------------|
| 123 (good) | 20 -> 61 | 9 -> 13 | 10^20-10^23 |
| 456 (bad) | 25 -> 116 | 8 -> 125 | 10^22 to 10^6 |

**Root cause**: exp parameterization prevents negative values but allows explosion to infinity. beta=116 means RF locality 1160x stronger than intended -> nearly all pixels masked out.

**Resolution**: Explicit bounds — eps_0 in [-1,1], beta_nat in [0.01,1.0], rho_nat in [0.01,0.5]. Implemented as LBFGS closure defense (params_in_bounds) + post-step clamping.

**Rule**: Log parameterization != bounds. If the original had bounds, port them.

**Docs**: `investigations/archive/HYPERPARAMETER_BOUNDS_ANALYSIS.md`

### 2.4 Whitening Conversions Compound Numerical Errors

| Metric | Whitened (Iter 0->2) | Unwhitened (Iter 0->2) |
|--------|---------------------|----------------------|
| KL | 0.00 -> 1.30 | 1501.82 -> 93.68 |
| cond(V) | 2.15e4 -> 1.36e5 | 1.00 -> 17888 |

Whitened mode starts correctly at the prior (KL=0) but condition number EXPLODES during E-step. Each forward-backward whitening conversion (V_natural = L_K @ V_stored @ L_K^T and inverse) compounds rounding errors.

**Key learning**: This is why eigenspace projection (reducing to ~10 principal dimensions) is essential — it dramatically improves conditioning.

### 2.5 torch.pi Side Effect from Old Codebase Import

`GP_utils.py` line 49 redefines `torch.pi = torch.acos(torch.zeros(1)).item() * 2`. Importing from old codebase silently changes global state. Replicated in `tests/test_utils.py:set_reproducible_seed()`.

### 2.6 Import Side Effects (torch.set_default_dtype)

Importing from 1D/2D playgrounds or utility.py can change `torch.default_dtype` to float64. Pattern in `acquisition.py`: save dtype before import, restore after. Use `importlib.util` for local utils.py to avoid sys.modules collision with repo-root utils.

### 2.7 Beta Parameterization Trap

Got this wrong in a session. The mapping:
- Config stores **natural** beta (e.g., 0.1)
- Code init: `raw_m2log2beta = -2 * log(2 * beta_nat)`
- In C matrix: `beta_code = exp(raw) = 1/(4*beta_nat^2)`
- RF sigma = `beta_nat * sqrt(2)` in normalized coords (NOT `sqrt(1/(2*beta_nat))`)
- For beta_nat=0.1: sigma = 0.1414 normalized = 7.6 pixels (NOT 120px)

The `kernel.beta` property returns natural beta. The formula `sqrt(1/(2*beta))` needs exp(raw), not natural beta.

### 2.8 set_reproducible_seed Device Parameter

`set_reproducible_seed(seed, device=device)` produces DIFFERENT random sequences than `set_reproducible_seed(seed)`. Root cause unknown. Rule: be consistent — always use same device parameter.

---

## 3. Paper Gap Investigation (March 2026)

**Observation**: Paper claims 36/41 cells with adjusted_r2 > 0.8. Our best: 15/41.

**Three codebases diverge**: paper (GitHub notebook), vargp_old (utils.py), vargp_direct (eigenspace_*.py). Paper != vargp_old: paper has no Amp, interleaves F-step, no eigenspace, uses float64 and scipy L-BFGS-B.

**Paper init values** (verified parameterization mapping): beta_nat=0.0452, rho_nat=0.0821, A=1e-4, lambda0=-1.

**Three confounds identified and controlled**:
1. **IP selection**: Paper uses interleaved F-step (damped Newton). We use separate M/F steps.
2. **Amp freedom**: Paper fixes Amp=1 (no A parameter). We learn it.
3. **Sigma_0 parameterization**: Paper uses direct. We use log.

**Results**:
- **Gap A CLOSED**: vargp_direct == vargp_old within 0.002 (mean diff 0.0008) when controlling all confounds
- **Gap B partially closed**: Best config avg adjusted_r2 = 0.730 (broad+interleave), 15/41 > 0.8

**Remaining gap likely due to**: scipy L-BFGS-B vs PyTorch LBFGS (different convergence), float64 in paper vs float32, possibly custom initialization.

**Code changes committed on branch `pietro/investigate-paper-gap`**: sigma_0 direct parameterization, LBFGS frozen-param filter, f_mean thresholds (mean>100, max>500), fix_Amp flag, interleave_fstep flag (damped Newton), vargp_old model bug fix.

**Investigation rules**: ip_selection='random' for mode comparisons; fix_Amp=True for paper comparisons.

**Sweep data**: `investigations/paper_gap/sweep_results.jsonl` (738 runs, 6 configs x 41 cells x 3 seeds)

**Docs**: `investigations/paper_gap/INVESTIGATION_LOG.md` (Findings 1-20)

---

## 4. Performance Pitfalls

### 4.1 Degradation with Large ntrain + M

Performance **decreases** with more training data for M >= 100:

| Mode | M | ntrain=500 | ntrain=2000 | Loss |
|------|---|-----------|------------|------|
| vargp_old | 100 | 0.869 | 0.767 | -11.8% |
| vargp_old | 200 | 0.814 | 0.735 | -9.7% |
| vargp_direct | 100 | 0.868 | 0.765 | -11.9% |
| vargp_direct | 200 | 0.815 | 0.731 | -10.2% |

Hypotheses (ranked): optimization failure (LBFGS local minima), eigenvalue rank loss (EIGVAL_TOL=1e-4 excludes info), underfitting with fixed iterations, hyperparameter mismatch.

**Status**: Partially investigated, all hypotheses remain open.

**Docs**: `investigations/performance_loss_ntrain_M/FINDINGS.md`

### 4.2 RF Center Initialization Sensitivity

| Init eps_0 | Final eps_0 | Test r |
|-----------|------------|--------|
| (0.0, 0.0) | (0.11, -0.04) | 0.87 |
| (0.5, 0.5) | (0.53, 0.43) | 0.35 |

LBFGS gets trapped in local minima. Use `rf_centers_ground_truth.npz` (white noise/checkerboard ellipse fits) or STA-derived centers.

### 4.3 STA Edge Artifact (108x108)

Cells 0, 5, 6, 15, 22, 39 show spurious STA peak at image edge on 108x108. Root cause: natural image correlation leakage. Center crops (48x48, 64x64) avoid this. On 108x108, these cells have near-zero test_r.

**Docs**: `investigations/sta_edge_artifact/`

### 4.4 default_gpy Fails on Cell 10 (ntrain=2000)

test_r ~ -0.05 to 0.11 vs vargp_direct ~ 0.90. Root cause: Adam optimizer cannot solve this problem. Needs LBFGS or better initialization. Status: known, not fixed.

### 4.5 Float64 is 10x Slower

M=50, 50 iterations: float32 = 5.6s, float64 = 18.9s. No meaningful accuracy benefit. Original varGP used float32. Only use float64 for debugging ill-conditioning.

---

## 5. Design Decisions (Porting-Relevant)

Full design decision log in `.claude/DECISION_LOG.md` (Q1-Q29). Key porting decisions summarized here:

| Decision | Choice | Rationale | Ref |
|----------|--------|-----------|-----|
| Stage 1 validation | Arc-cosine C=I | Isolates kernel+GPyTorch from RF structure | Q7, Q13 |
| Amplitude parameter | ScaleKernel wrapper | Standard GPyTorch pattern, clean separation | Q8, Q16 |
| Default dtype | float32 | Matches original, 10x faster | Q9 |
| Primary mode | vargp_direct (eigenspace) | Exact match to varGP, bypasses whitening issues | Q27-Q28 |
| Lambda0 computation | Analytical (closed-form) | Stable, fast, removes one parameter from optimization | Q24 |
| Kernel caching in E-step | Pre-compute K, K_tilde, k0 | 8.8x speedup (8.8s -> 1.0s per E-step) | Q25 |
| Pixel masking | Detached mask, MASK_THRESHOLD=0.001 | 20x memory reduction (1GB -> 50MB), structural stability | Q22 |

---

## 6. Open Questions (Not Yet Resolved)

1. **Performance degradation with large ntrain+M** — root cause unclear (Section 4.1)
2. **default_gpy failure on cell 10** — needs LBFGS or better init (Section 4.4)
3. **LBFGS tolerance vs float32 precision** — is 1e-9 tolerance meaningful for float32?
4. **Paper gap remaining 0.07** — 15/41 vs 36/41 cells > 0.8 (Section 3)
5. **torch.pi workaround** — root cause still unexplained (Section 2.5)

---

## 7. Input Warping (Deferred, Not Dead)

`investigations/input_warping/INPUT_WARPING_REFERENCE.md` documents a potential approach to constrain utility optimization to physically realizable images by warping kernel inputs through a bounding function. Not implemented — filed as reference for future consideration if pixel-space bounds prove insufficient.

---

*Last updated: April 2026*
*Sources: DECISION_LOG.md, investigations/archive/, investigations/paper_gap/INVESTIGATION_LOG.md, investigations/performance_loss_ntrain_M/FINDINGS.md, SESSION_LOG.md, MEMORY.md*
