# Benchmark History

Track performance across development milestones. Update after significant changes.

**How to update**: Run `python tests/test_estep_comparison.py` and copy results here.

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

**Command**: `python test_estep_pnas.py --mode MODE --gradient-mode GRAD --ntilde M`

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
6. **vargp_style M=100 autograd achieves best overall** (0.87 ExplVar)

### Timing Anomalies
- M=100 runs (~44-51s) much slower than M=200 (~17-18s) for vargp_style
- Likely due to convergence differences or GPU warmup effects
