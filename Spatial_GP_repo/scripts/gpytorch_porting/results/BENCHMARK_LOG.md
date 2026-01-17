# Benchmark History

Track performance across development milestones. Update after significant changes.

**How to update**: Run `python tests/test_estep_comparison.py` and copy results here.

---

## Results Table (Explained Variance)

| Date | M | varGP | vargp_style | efm | adam | Notes |
|------|---|-------|-------------|-----|------|-------|
| 2025-01-17 | 50 | 0.87 | 0.83 | 0.86 | 0.67 | Baseline after adding vargp_style |
| 2025-01-17 | 75 | - | 0.83 | 0.40 ↓ | 0.66 | efm degrades at M>50 |
| 2025-01-17 | 100 | - | 0.87 | 0.41 ↓ | 0.68 | vargp_style stable, efm broken |
| 2025-01-17 | 200 | - | 0.81 ↓ | - | 0.67 | vargp_style drops slightly at M=200 |

---

## Notes

### 2025-01-17
- Added `vargp_style` mode to `test_estep_comparison.py`
- **Key finding**: efm degrades severely at M>50 (0.86 → 0.40)
- **Key finding**: vargp_style remains stable across M and improves at M=100
- vargp_style at M=100 achieves best GPyTorch result (0.87 explained var)
- adam is consistent but weak (~0.67 regardless of M)
