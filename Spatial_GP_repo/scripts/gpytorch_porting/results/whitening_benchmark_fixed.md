## Benchmark Results: Training Modes Comparison

Generated: 2026-01-20 01:11:55

### Configuration Legend

| Config | Mode | Cache | Whitening | Description |
|--------|------|-------|-----------|-------------|
| REF | vargp_old | - | - | Original varGP (reference) |
| ADAM | adam | - | - | Pure Adam optimization |
| A | vargp_style | ✓ | ✓ | cached+whitening (default) |
| B | vargp_style | ✓ | ✗ | cached+no-whitening |
| C | vargp_style | ✗ | ✓ | non-cached+whitening |
| D | vargp_style | ✗ | ✗ | non-cached+no-whitening |

### Test Correlation (Pearson r)

| M | REF | ADAM | A | B | C | D |
|---|------|------|------|------|------|------|
| 50 | 0.8141 | 0.6327 | 0.8001 | 0.7929 | 0.7936 | 0.0200 |
| 75 | 0.6482 | 0.6235 | 0.0708 | 0.7582 | 0.7271 | 0.7234 |
| 100 | 0.6187 | 0.6431 | 0.8009 | 0.7304 | 0.7608 | 0.7845 |

### Training Time (seconds)

| M | REF | ADAM | A | B | C | D |
|---|------|------|------|------|------|------|
| 50 | 5.2 | 1.6 | 6.8 | 6.3 | 9.8 | 123.1 |
| 75 | 5.7 | 1.7 | 8.0 | 6.5 | 12.0 | 17.9 |
| 100 | 6.0 | 1.8 | 9.1 | 7.3 | 13.3 | 50.6 |

### Key Comparisons

| M | A vs REF | A vs C | ADAM vs REF |
|---|----------|--------|-------------|
| 50 | -0.0140 | ~ | -0.1814 |
| 75 | -0.5774 | 0.6563 | -0.0247 |
| 100 | +0.1823 | 0.0401 | +0.0245 |

**Interpretation:**
- A vs REF: Positive = GPyTorch better, Negative = varGP better
- A vs C: Should be ~ (equivalent whitening paths)
- ADAM vs REF: Compares pure optimization to E-step approach

### Full Results

| M | Config | test_r | time (s) | A | λ₀ |
|---|--------|--------|----------|---|----|
| 50 | REF | 0.8141 | 5.2 | 0.0144 | -0.2797 |
| 50 | ADAM | 0.6327 | 1.6 | 0.8375 | 0.0243 |
| 50 | A | 0.8001 | 6.8 | 1.7162 | -2.4673 |
| 50 | B | 0.7929 | 6.3 | 0.9333 | -0.6588 |
| 50 | C | 0.7936 | 9.8 | 2.1485 | -3.1927 |
| 50 | D | 0.0200 | 123.1 | 0.0011 | 0.7277 |
| 75 | REF | 0.6482 | 5.7 | 0.0136 | -0.2667 |
| 75 | ADAM | 0.6235 | 1.7 | 0.8406 | 0.0288 |
| 75 | A | 0.0708 | 8.0 | 2.1646 | -2.9008 |
| 75 | B | 0.7582 | 6.5 | 0.9944 | -0.6973 |
| 75 | C | 0.7271 | 12.0 | 3.9946 | -5.1992 |
| 75 | D | 0.7234 | 17.9 | 1.2421 | -1.2391 |
| 100 | REF | 0.6187 | 6.0 | 0.0144 | -0.2795 |
| 100 | ADAM | 0.6431 | 1.8 | 0.8424 | 0.0314 |
| 100 | A | 0.8009 | 9.1 | 4.5103 | -6.3887 |
| 100 | B | 0.7304 | 7.3 | 1.5522 | -1.4834 |
| 100 | C | 0.7608 | 13.3 | 4.1682 | -5.7418 |
| 100 | D | 0.7845 | 50.6 | 0.4655 | -0.6808 |