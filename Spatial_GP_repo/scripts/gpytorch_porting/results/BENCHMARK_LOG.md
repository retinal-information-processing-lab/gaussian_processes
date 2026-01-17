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
