# Benchmark History

Track performance across development milestones. Update after significant changes.

---

## REQUIRED FORMAT FOR NEW ENTRIES

**All new benchmark entries MUST follow this template to ensure reproducibility.**

### Template

```markdown
## Benchmark: YYYY-MM-DD (Title)

### Environment
| Field | Value |
|-------|-------|
| **Commit hash** | `abc1234def5678` (min 7-char) |
| **Branch** | `pietro/workingbranch` |
| **Conda env** | `pytorch_gpytorch` |
| **GPU** | NVIDIA RTX 3090 |

### Training Parameters
| Parameter | Value |
|-----------|-------|
| **n_train** | 500 |
| **M (ntilde)** | 50 |
| **n_iter** | 50 |
| **n_estep** | 10 |
| **n_mstep** | 10 |
| **n_fstep** | 10 |
| **cell_id** | 8 |
| **seed** | 123 |

### Exact Command
```
python run_single_mode.py --mode vargp_style --ntilde 50 --seed 123
```

### Results

| Mode | M | Seed | Options | Test r | Expl Var | Time | Loss |
|------|---|------|---------|--------|----------|------|------|
| vargp_style | 50 | 123 | whitened | 0.7992 | 0.84 | 7.1s | 419.99 |

### Notes
- Any observations or anomalies
```

**Why this matters**: Previous benchmarks lack parameter details (n_train, n_iter, etc.), making results non-reproducible and comparisons invalid.

---

## Legacy Results (pre-template)

The following results were logged before this template was established. Parameters may vary and are not fully documented.

---

## Results Table (Explained Variance)

| Date | M | varGP | vargp_style | efm | adam | Notes |
|------|---|-------|-------------|-----|------|-------|
| 2025-01-17 | 50 | **0.87** | 0.83 | 0.86 | 0.67 | Baseline after adding vargp_style |
| 2025-01-17 | 75 | 0.70 ↓ | **0.83** | 0.40 ↓ | 0.66 | vargp_style now best! |
| 2025-01-17 | 100 | 0.66 ↓ | **0.87** | 0.41 ↓ | 0.68 | vargp_style outperforms varGP |
| 2025-01-17 | 200 | 0.35 ↓↓ | **0.81** | - | 0.67 | varGP collapses at M=200 |

---

## Notes

### 2025-01-17 (updated)
- **Surprising finding**: varGP (original) also degrades at M>50!
  - M=50: 0.87, M=75: 0.70, M=100: 0.66, M=200: 0.35
- **vargp_style outperforms varGP for M≥75** - this is unexpected
- efm degrades severely at M>50 (0.86 → 0.40)
- adam is consistent but weak (~0.67 regardless of M)
- Best overall: vargp_style at M=100 (0.87 explained var)

---

## Benchmark: 2026-01-18 (Gradient Mode Comparison)

**Command**: `python run_single_mode.py --mode MODE --gradient-mode GRAD --ntilde M`

| Mode        | Gradient | M=50 (ExplVar/Time) | M=75         | M=100            | M=200 |
|------       |----------|---------------------|------        |-------           |-------|
| vargp_old   | n/a      | **0.86** / 5.4s     | 0.69 / 6.0s  | 0.66 / 6.3s      | 0.35 / 8.6s |
| vargp_style | autograd | 0.83 / 15.7s        | 0.82 / 17.9s | **0.87** / 44.2s | 0.81 / 17.2s |
| vargp_style | vjp      | 0.85 / 17.4s        | 0.80 / 19.9s | 0.83 / 51.6s     | **0.84** / 18.5s |
| adam        | autograd | 0.67 / 1.9s         | 0.66 / 2.0s  | 0.68 / 2.1s      | 0.67 / 2.5s |
| adam        | vjp      | 0.67 / 2.5s         | 0.66 / 2.5s  | 0.68 / 2.8s      | 0.67 / 3.1s |

### Key Findings

1. **vargp_old degrades at M>50** (0.86 → 0.35 at M=200) - confirms previous finding
2. **vargp_style is stable across M** - best at M=100 (0.87) and M=200 (0.84)
3. **Gradient mode has minimal impact on accuracy** - autograd vs vjp produce similar results
4. **vjp is slightly slower** (~20-30% overhead) - autograd recommended for speed
5. **adam is fastest but weakest** (~0.67 regardless of M or gradient mode)

---

## Benchmark: 2026-01-20 (Seed Sensitivity - MAJOR FINDING)

**Command**: `python run_single_mode.py --mode MODE --ntilde M --seed S`

| M | Seed | varGP | whitened | legacy |
|---|------|-------|----------|--------|
| 50 | 42 | 0.86 | 0.84 | 0.69 |
| 50 | 123 | 0.89 | 0.83 | 0.90 |
| 50 | 456 | 0.67 | **-0.00** | 0.66 |
| 75 | 42 | 0.69 | **0.08** | 0.66 |
| 75 | 123 | 0.90 | 0.83 | 0.88 |
| 75 | 456 | 0.65 | **nan** | 0.72 |

### Key Finding

**Whitened mode collapse is SEED-DEPENDENT, not M-dependent.**
- Seed 456 causes collapse at both M=50 and M=75
- Seed 123 works fine at all M values
- Legacy (unwhitened) mode never collapses
- Original "M=75 collapse" was actually bad inducing point selection at seed=42

See `investigations/INVESTIGATION_whitening_collapse_M75.md` for details.
6. **vargp_style M=100 autograd achieves best overall** (0.87 ExplVar)

### Timing Anomalies
- M=100 runs (~44-51s) much slower than M=200 (~17-18s) for vargp_style
- Likely due to convergence differences or GPU warmup effects

---

## Benchmark: 2026-01-18 (E-step Kernel Caching Optimization)

**Problem**: vargp_style E-step was 4.9x slower than original varGP (8.8s vs 1.8s).

**Root cause**: 35 kernel calls per E-step loop vs ideal 2 (17.5x overhead).

**Solution**: Cache K, K̃ matrices and reuse across Newton iterations.

**Command**: `python run_single_mode.py --mode vargp_style --ntilde 50 [--no-cache]`

### Performance Results (M=50, N=500)

| Path | Test r | E-step Time | Total Time | Kernel Calls |
|------|--------|-------------|------------|--------------|
| varGP (reference) | 0.8141 | 1.1s | 5.2s | ~2 |
| GPyTorch cached | 0.7752 | **1.0s** | 6.4s | 3 |
| GPyTorch non-cached | 0.7870 | 8.8s | 16.1s | 35 |

### Key Findings

1. **E-step speedup**: 8.8s → 1.0s = **8.8x faster**
2. **GPyTorch E-step now faster than varGP** (1.0s vs 1.1s)
3. **Model quality preserved**: test r = 0.77-0.79 (vs 0.81 reference)
4. **Small accuracy difference** between cached (0.7752) and non-cached (0.7870) - may warrant investigation

### Reference

- Decision Log: Q25 in CLAUDE.md
- Details: `HANDOFF_2026-01-18.md`, `results/PROFILING_2026-01-18.md`
- Original non-cached commit: `44d9227`

---

## Benchmark: 2026-01-20 (Whitening + Caching Comparison)

**Commit**: `7f27b01`
**Command**: `python run_benchmark.py --ntilde M`

Tests whitened+cached (new default) vs legacy (no-whitening+cached) implementations.

### Results Table (Explained Variance)

| M | varGP | vargp_style (whitened) | vargp_style (legacy) | efm | adam |
|---|-------|------------------------|----------------------|-----|------|
| 50 | **0.87** | 0.84 | 0.84 | 0.86 | 0.67 |
| 75 | 0.70 | ⚠️ 0.08 | **0.80** | 0.40 | 0.66 |
| 100 | 0.67 | **0.85** | 0.77 | 0.41 | 0.68 |
| 200 | 0.35 | **0.82** | 0.56 | 0.51 | 0.67 |

### Timing Breakdown (seconds)

| M | varGP (E/M/Total) | vargp_style whitened (E/M/Total) | vargp_style legacy (E/M/Total) |
|---|-------------------|----------------------------------|-------------------------------|
| 50 | 1.2 / 4.1 / 5.6 | 0.7 / 6.2 / 7.3 | 0.7 / 5.6 / 6.7 |
| 75 | 1.2 / 4.8 / 6.3 | 1.0 / 7.2 / 8.7 | 0.8 / 5.8 / 7.0 |
| 100 | 1.2 / 5.1 / 6.6 | 1.1 / 8.0 / 9.6 | 0.9 / 6.2 / 7.5 |
| 200 | 1.2 / 6.9 / 8.6 | 2.1 / 17.7 / 20.9 | 1.6 / 10.2 / 12.4 |

### Key Findings

1. **E-step caching works well**: GPyTorch E-step (0.7-2.1s) is faster than varGP (1.2s) for M≤100

2. **Whitening is unstable at M=75**: Collapsed to 0.08 explained variance
   - This specific M value triggers numerical issues
   - Legacy (no-whitening) works fine at M=75 (0.80)

3. **Whitening helps at M≥100**: Outperforms both varGP and legacy
   - M=100: whitened 0.85 vs legacy 0.77 vs varGP 0.67
   - M=200: whitened 0.82 vs legacy 0.56 vs varGP 0.35

4. **varGP degrades severely at M>50**: Confirmed again (0.87 → 0.35)

5. **adam is stable but weak**: ~0.67 regardless of M (no kernel learning benefit)

### Recommendations

- **For M=50**: Use varGP or any GPyTorch mode (all perform well)
- **For M=75**: Use legacy (no-whitening) - whitening has numerical issues
- **For M≥100**: Use whitened vargp_style - significantly better than varGP

### ⚠️ Known Issue: M=75 Whitening Collapse

The whitened implementation collapses at M=75 specifically. This warrants investigation:
- May be related to L_K mismatch when kernel params change (see HANDOFF Section 14)
- The loss trajectory shows instability: 435 → 510 (diverging)
- A values grow large: 0.40 → 3.58 → 2.16 (oscillating)

---

## Benchmark: 2026-01-20 (n_train=2000 Comparison)

**Commit**: `39b04c9`
**Command**: `python run_benchmark.py --ntilde M --n-train 2000`

Tests with 4x more training data (2000 vs 500) to see if varGP degradation at high M persists.

### Results Table (Explained Variance, n_train=2000)

| M | varGP | vargp_style (whitened) | vargp_style (legacy) | efm | adam |
|---|-------|------------------------|----------------------|-----|------|
| 50 | **0.91** | 0.82 | 0.78 | 0.86 | 0.70 |
| 75 | **0.94** | ⚠️ 0.26 | 0.78 | 0.86 | 0.69 |
| 100 | **0.94** | ⚠️ 0.14 | 0.64 | 0.86 | 0.74 |
| 200 | **0.95** | 0.56 | 0.58 | 0.88 | 0.75 |

### Timing Breakdown (seconds, n_train=2000)

| M | varGP (E/M/Total) | vargp_style whitened (E/M/Total) | vargp_style legacy (E/M/Total) |
|---|-------------------|----------------------------------|-------------------------------|
| 50 | 2.8 / 5.3 / 8.4 | 2.0 / 29.8 / 33.4 | 1.1 / 13.7 / 15.6 |
| 75 | 2.8 / 6.3 / 9.3 | 2.2 / 31.0 / 34.8 | 1.2 / 13.0 / 15.0 |
| 100 | 3.1 / 8.5 / 12.0 | 7.3 / 53.5 / 64.3 | 11.0 / 44.2 / 58.7 |
| 200 | 7.3 / 40.5 / 51.0 | 9.1 / 78.1 / 92.0 | 8.3 / 42.4 / 53.8 |

### Key Findings

1. **varGP NO LONGER degrades at high M with more data!**
   - n_train=500: 0.87 → 0.70 → 0.67 → 0.35 (severe degradation)
   - n_train=2000: 0.91 → 0.94 → 0.94 → 0.95 (stable, excellent)
   - **Root cause of M>50 degradation was insufficient training data, not the algorithm**

2. **Whitening collapse is WORSE with more data**
   - M=75: 0.08 → 0.26 (still bad)
   - M=100: 0.85 → 0.14 (now collapsed, was good with n_train=500!)
   - Whitening issues are exacerbated, not helped, by more data

3. **efm is remarkably stable**: ~0.86-0.88 regardless of M or n_train

4. **adam improves slightly**: 0.67 → 0.70-0.75 with more data

5. **Legacy (no-whitening) degrades at M≥100 with more data**: 0.77 → 0.64

### Comparison: n_train=500 vs n_train=2000

| M | varGP (500) | varGP (2000) | Δ |
|---|-------------|--------------|---|
| 50 | 0.87 | 0.91 | +0.04 |
| 75 | 0.70 | **0.94** | **+0.24** |
| 100 | 0.67 | **0.94** | **+0.27** |
| 200 | 0.35 | **0.95** | **+0.60** |

### Recommendations (Updated)

- **For production use**: Use varGP with n_train≥2000 for best results at any M
- **If limited data (n_train~500)**:
  - M=50: varGP or efm
  - M>50: efm is most stable (0.86), varGP degrades
- **Avoid whitened vargp_style**: Unstable across M values, especially with more data
- **efm is a reliable fallback**: Consistent ~0.86 regardless of settings

---

## Benchmark: 2026-01-21 (Hyperparameter Bounds Post-Merge Verification)

### Environment

| Field | Value |
|-------|-------|
| **Commit hash** | `8855fc9` |
| **Branch** | `pietro/workingbranch` |
| **Conda env** | `pytorch_gpytorch` |
| **GPU** | NVIDIA GeForce RTX 4090 |

### Training Parameters

| Parameter | Value |
|-----------|-------|
| **n_train** | 500 |
| **n_iter** | 50 |
| **n_estep** | 10 |
| **n_mstep** | 10 |
| **n_fstep** | 10 |
| **cell_id** | 8 |

### Exact Commands

```
python run_single_mode.py --mode vargp_style --ntilde M --seed S
python run_single_mode.py --mode vargp_style --ntilde M --seed S --unwhitened
python run_single_mode.py --mode efm --ntilde M --seed S
python run_single_mode.py --mode adam --ntilde M --seed S
```

### Results: vargp_style (whitened, default)

| Mode | M | Seed | Options | Test r | Expl Var | Time | Loss |
|------|---|------|---------|--------|----------|------|------|
| vargp_style | 50 | 42 | whitened | 0.7845 | 0.83 | 7.1s | 421.54 |
| vargp_style | 50 | 123 | whitened | 0.7992 | 0.84 | 7.1s | 419.99 |
| vargp_style | 50 | 456 | whitened | nan | nan | 5.7s | 496.25 |
| vargp_style | 75 | 42 | whitened | 0.3317 | 0.35 | 8.2s | 514.41 |
| vargp_style | 75 | 123 | whitened | 0.7928 | 0.84 | 7.8s | 414.19 |
| vargp_style | 75 | 456 | whitened | -0.0000 | 0.00 | 6.4s | 496.25 |
| vargp_style | 100 | 123 | whitened | 0.7941 | 0.84 | 8.1s | 422.17 |
| vargp_style | 200 | 123 | whitened | 0.7763 | 0.82 | 11.4s | 418.40 |

### Results: vargp_style (unwhitened/legacy)

| Mode | M | Seed | Options | Test r | Expl Var | Time | Loss |
|------|---|------|---------|--------|----------|------|------|
| vargp_style | 50 | 42 | --unwhitened | 0.6490 | 0.69 | 30.9s | 410.62 |
| vargp_style | 50 | 123 | --unwhitened | 0.8511 | 0.90 | 47.0s | 419.52 |
| vargp_style | 50 | 456 | --unwhitened | 0.6319 | 0.66 | 54.0s | 430.57 |

### Results: Other Modes

| Mode | M | Seed | Options | Test r | Expl Var | Time | Loss |
|------|---|------|---------|--------|----------|------|------|
| efm | 50 | 123 | - | 0.8012 | 0.85 | 33.2s | 505.29 |
| efm | 75 | 123 | - | 0.8176 | 0.86 | 37.5s | 553.16 |
| adam | 50 | 123 | - | 0.6858 | 0.72 | 2.0s | 493.02 |
| vargp_style | 50 | 123 | --gradient-mode vjp | 0.7114 | 0.75 | 8.0s | 427.47 |

### Comparison with Previous Results

| Config | OLD | NEW | Delta |
|--------|-----|-----|-------|
| M=50 seed=42 whitened | 0.84 | 0.83 | -0.01 |
| M=50 seed=123 whitened | 0.83 | 0.84 | +0.01 |
| M=75 seed=42 whitened | 0.08 | 0.35 | **+0.27** |
| M=75 seed=123 whitened | 0.83 | 0.84 | +0.01 |
| M=50 seed=123 legacy | 0.90 | 0.90 | 0.00 |
| M=50 seed=456 legacy | 0.66 | 0.66 | 0.00 |

### Notes

1. **No regression**: Whitened mode unchanged for good seeds (within ±0.01)
2. **M=75 seed=42 improved**: 0.08 → 0.35 (bounds helped partial collapse case)
3. **Collapse seeds unchanged**: Seed 456 still collapses in whitened mode
4. **Legacy mode stable**: Never collapses, handles all seeds correctly
5. **Pending issue**: Amp placement differs from varGP (ScaleKernel vs inside C)

