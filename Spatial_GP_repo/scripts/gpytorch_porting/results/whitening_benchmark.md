## Benchmark Results: Training Modes Comparison

Generated: 2026-01-19 23:49:10

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
| 50 | 0.8141 | 0.6325 | 0.7676 | 0.7752 | 0.2111 | 0.7870 |
| 75 | 0.6482 | 0.6229 | 0.7994 | 0.7708 | 0.7940 | 0.7729 |
| 100 | 0.6187 | 0.6424 | 0.7323 | 0.7640 | 0.7914 | 0.8260 |

### Training Time (seconds)

| M | REF | ADAM | A | B | C | D |
|---|------|------|------|------|------|------|
| 50 | 5.2 | 1.6 | 7.0 | 5.9 | 10.3 | 15.6 |
| 75 | 5.5 | 1.7 | 8.6 | 6.5 | 11.9 | 17.8 |
| 100 | 6.0 | 1.8 | 11.2 | 6.8 | 15.4 | 43.8 |

### Key Comparisons

| M | A vs REF | A vs C | ADAM vs REF |
|---|----------|--------|-------------|
| 50 | -0.0465 | 0.5565 | -0.1816 |
| 75 | +0.1513 | ~ | -0.0253 |
| 100 | +0.1136 | 0.0591 | +0.0238 |

**Interpretation:**
- A vs REF: Positive = GPyTorch better, Negative = varGP better
- A vs C: Should be ~ (equivalent whitening paths)
- ADAM vs REF: Compares pure optimization to E-step approach

### Full Results

| M | Config | test_r | time (s) | A | λ₀ |
|---|--------|--------|----------|---|----|
| 50 | REF | 0.8141 | 5.2 | 0.0144 | -0.2797 |
| 50 | ADAM | 0.6325 | 1.6 | 0.8375 | 0.0243 |
| 50 | A | 0.7676 | 7.0 | 3.9488 | -5.5035 |
| 50 | B | 0.7752 | 5.9 | 0.8801 | -0.6544 |
| 50 | C | 0.2111 | 10.3 | 1.9571 | -2.7652 |
| 50 | D | 0.7870 | 15.6 | 0.5083 | -1.0364 |
| 75 | REF | 0.6482 | 5.5 | 0.0136 | -0.2667 |
| 75 | ADAM | 0.6229 | 1.7 | 0.8406 | 0.0288 |
| 75 | A | 0.7994 | 8.6 | 4.0369 | -5.5000 |
| 75 | B | 0.7708 | 6.5 | 1.0379 | -0.8010 |
| 75 | C | 0.7940 | 11.9 | 3.0500 | -3.9998 |
| 75 | D | 0.7729 | 17.8 | 1.1664 | -1.1240 |
| 100 | REF | 0.6187 | 6.0 | 0.0144 | -0.2795 |
| 100 | ADAM | 0.6424 | 1.8 | 0.8424 | 0.0313 |
| 100 | A | 0.7323 | 11.2 | 3.0608 | -4.6613 |
| 100 | B | 0.7640 | 6.8 | 1.3797 | -1.2803 |
| 100 | C | 0.7914 | 15.4 | 2.6875 | -4.0297 |
| 100 | D | 0.8260 | 43.8 | 0.9292 | -0.7567 |