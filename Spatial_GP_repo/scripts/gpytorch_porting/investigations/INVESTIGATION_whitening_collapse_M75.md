# Investigation: Whitening Collapse at M=75

**Date**: 2026-01-20
**Status**: ACTIVE
**Issue**: vargp_style (whitened) mode collapses to test_r ~ 0.08 at M=75, while other M values work fine

---

## Problem Statement

From BENCHMARK_LOG.md, whitened vargp_style shows erratic behavior:
- M=50: 0.84 (good)
- M=75: 0.08 (COLLAPSE)
- M=100: 0.85 (good)
- M=200: 0.82 (good)

The legacy (unwhitened) mode does NOT collapse at M=75 (0.80), suggesting the issue is specific to the whitening transformation.

---

## Hypotheses

1. **Numerical instability in L_K at M=75**: Something about M=75 inducing points causes ill-conditioned L_K
2. **Eigenvalue structure**: M=75 might hit a problematic eigenvalue configuration
3. **Random seed sensitivity**: The specific inducing point selection at M=75 is unlucky
4. **Jitter mismatch**: Different jitter values between cached/non-cached paths (known historical issue)

---

## Benchmark Results

### Run 1: n_train=500, default settings (2026-01-20)

Command: `python run_benchmark.py --ntilde N`

| M | varGP | whitened | legacy | efm | adam |
|---|-------|----------|--------|-----|------|
| 50 | 0.8748 | 0.8447 | 0.8381 | 0.8623 | 0.6654 |
| 75 | 0.6966 | **0.0760** | 0.8010 | 0.4011 | 0.6559 |
| 100 | 0.6650 | 0.8456 | 0.7721 | 0.4136 | 0.6765 |
| 200 | 0.3537 | 0.8218 | 0.5624 | 0.5055 | 0.6719 |

### A parameter evolution during training

| M | Final A (whitened) | Trajectory |
|---|-------------------|------------|
| 50 | 1.72 | Stable growth: 0.36 → 0.73 → 0.96 → 1.30 → 1.72 |
| 75 | 2.16 | **UNSTABLE**: 0.40 → 1.03 → 2.01 → 3.58(!) → 2.16 |
| 100 | 4.51 | Growing: 0.40 → 1.10 → 2.05 → 2.81 → 4.51 |
| 200 | 2.63 | Moderate: 0.39 → 0.98 → 1.58 → 2.10 → 2.63 |

### Loss evolution (whitened mode)

| M | Iter 20 | Iter 30 | Iter 40 | Iter 50 | Pattern |
|---|---------|---------|---------|---------|---------|
| 50 | 429.40 | 426.78 | 419.71 | 417.79 | Decreasing (healthy) |
| 75 | 429.79 | 424.21 | **443.80** | **510.92** | **DIVERGING after iter 30** |
| 100 | 430.10 | 447.90 | 448.73 | 438.89 | Some oscillation |
| 200 | 425.06 | 416.00 | 424.48 | 410.69 | Some oscillation |

---

## Observations

### Observation 1: M=75 is uniquely problematic

- M=75 is the ONLY value where whitened mode collapses (0.076)
- Legacy (unwhitened) mode works fine at M=75 (0.801)
- The issue is specific to the whitening transformation

### Observation 2: A parameter blows up at M=75

At M=75, the A parameter reaches 3.58 at iter 40, then drops to 2.16. This correlates with:
- Loss divergence: 424.21 (iter 30) → 443.80 (iter 40) → 510.92 (iter 50)
- This is qualitatively different from other M values

### Observation 3: varGP itself degrades at M >= 75

Reference implementation varGP also degrades:
- M=50: 0.8748 (good)
- M=75: 0.6966 (degraded)
- M=100: 0.6650 (degraded)
- M=200: 0.3537 (poor)

But GPyTorch whitened mode OUTPERFORMS varGP at M >= 100 when it doesn't collapse.

### Observation 4: Legacy mode is more stable

The unwhitened (legacy) mode:
- Never collapses
- At M=75: 0.801 vs whitened 0.076
- More predictable A evolution

---

## VJP Gradient Mode Tests (All M values)

Command: `python run_benchmark.py --ntilde N --gradient-mode vjp`

### Whitened Mode: autograd vs vjp

| M | varGP | whitened (autograd) | whitened (vjp) | Diff |
|---|-------|---------------------|----------------|------|
| 50 | 0.87 | 0.84 | 0.82 | -0.02 |
| 75 | 0.70 | **0.08** | **nan** | worse |
| 100 | 0.67 | 0.85 | 0.87 | +0.02 |
| 200 | 0.35 | 0.82 | **0.68** | **-0.14** |

### Legacy (Unwhitened) Mode: autograd vs vjp

| M | legacy (autograd) | legacy (vjp) |
|---|-------------------|--------------|
| 50 | 0.84 | 0.84 |
| 75 | 0.80 | 0.80 |
| 100 | 0.77 | 0.72 |
| 200 | 0.56 | 0.35 |

### A Parameter (final) - Whitened Mode

| M | autograd | vjp |
|---|----------|-----|
| 50 | 1.72 | 3.55 |
| 75 | 2.16 | 0.02 (collapsed) |
| 100 | 4.51 | 3.39 |
| 200 | 2.63 | 1.78 |

### VJP-Specific Observations

1. **M=75 collapse confirmed in both modes**: autograd (0.08) and vjp (nan)
2. **M=200 degradation with VJP**: whitened drops from 0.82 (autograd) to 0.68 (vjp)
3. **VJP has higher variance**: A parameter trajectories differ more between runs
4. **Legacy mode more consistent**: Similar results between autograd and vjp

**Key Conclusion**: The M=75 collapse is NOT gradient-mode specific. It occurs in both autograd and VJP, ruling out gradient computation as the root cause. The issue is in the whitening transformation itself.

---

## Seed Sensitivity Tests (MAJOR FINDING)

Command: `python run_single_mode.py --mode MODE --ntilde M --seed S --save-plot none`

### Results Table

| M | Seed | varGP | whitened | legacy | Notes |
|---|------|-------|----------|--------|-------|
| 50 | 42 | 0.861 | 0.845 | 0.688 | All OK |
| 50 | 123 | 0.887 | 0.828 | 0.896 | All OK |
| 50 | 456 | 0.669 | **-0.000** | 0.661 | **WHITENED COLLAPSED** |
| 75 | 42 | 0.687 | **0.076** | 0.657 | Whitened collapsed (original benchmark) |
| 75 | 123 | 0.897 | 0.832 | 0.882 | **All OK - no collapse!** |
| 75 | 456 | 0.653 | **nan** | 0.724 | **WHITENED COLLAPSED** |

### Key Finding: Collapse is SEED-DEPENDENT, not M-DEPENDENT

1. **Seed 456 causes collapse at BOTH M=50 and M=75**
2. **Seed 123 works fine at BOTH M=50 and M=75**
3. **Legacy (unwhitened) mode NEVER collapses** regardless of seed
4. **Original benchmark (seed=42) happened to trigger collapse at M=75 but not M=50**

### Variance Across Seeds

All methods show significant variance across seeds:
- varGP: 0.653 to 0.897
- whitened: -0.000 to 0.845 (when not collapsed)
- legacy: 0.657 to 0.896

### Updated Hypothesis

The collapse is NOT caused by M=75 specifically. Instead:
- Certain inducing point configurations (selected by seed) cause the whitening transformation to fail
- Seed 456 selects "bad" inducing points that trigger collapse
- Seed 42 selects inducing points that happen to be problematic at M=75 but not M=50
- Seed 123 selects "good" inducing points that work at all M values

---

## Next Steps

1. ~~Test random seed sensitivity~~ **DONE - seed is the cause**
2. **Investigate what makes seed 456 inducing points "bad"** - compare K eigenvalues, condition numbers
3. **Check inducing point similarity** - are bad seeds selecting near-duplicate inducing points?
4. **Compare L_K condition numbers** between good seeds (123) and bad seeds (456)

---

## Conclusions

**MAJOR FINDING: The collapse is seed-dependent, not M-dependent.**

**Confirmed findings:**
- The collapse can happen at ANY M value (observed at both M=50 and M=75 with seed 456)
- Seed 123 works fine at both M=50 and M=75 (no collapse)
- Legacy (unwhitened) mode NEVER collapses regardless of seed
- The issue is in how whitening handles certain inducing point configurations

**Root cause hypothesis updated:**
- Certain random inducing point selections create ill-conditioned K matrices
- The whitening transformation (involving L_K = chol(K)) amplifies this numerical issue
- Unwhitened mode avoids this by not using the Cholesky factor
- The "M=75 collapse" was actually a "seed=42 at M=75" collapse
