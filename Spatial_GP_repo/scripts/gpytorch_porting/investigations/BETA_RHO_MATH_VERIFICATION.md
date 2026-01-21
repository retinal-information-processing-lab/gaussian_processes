# Beta and Rho Parameterization: Mathematical Verification

**Date**: January 2025
**Purpose**: Verify the mathematical relationships for beta and rho parameterization in the kernel implementation.

---

## Table of Contents

1. [Summary of Findings](#summary-of-findings)
2. [Beta Parameterization](#beta-parameterization)
3. [Rho Parameterization](#rho-parameterization)
4. [Physical Interpretations](#physical-interpretations)
5. [Parameter Bounds](#parameter-bounds)
6. [Concerns and Edge Cases](#concerns-and-edge-cases)

---

## 1. Summary of Findings

| Parameter | True Value | Raw Parameter | Usage in Code | Physical Meaning |
|-----------|------------|---------------|---------------|------------------|
| beta_true | 0.1 | raw = -2 log(2 * beta_true) = 3.22 | alpha = exp(-dist^2 / (4 * beta_true^2)) | RF width = 2 * beta_true |
| rho_true | 0.1 | raw = -log(2 * rho_true^2) = 3.91 | C_smooth = exp(-dist^2 / (2 * rho_true^2)) | Smoothness length scale |

**Key insight**: The parameterization creates Gaussian-like kernels where:
- **beta_true** is the **half-width** of the RF (full width = 4 * beta_true for ~99% of mass)
- **rho_true** is the **standard deviation** of the smoothness kernel

---

## 2. Beta Parameterization

### 2.1 Initialization (kernels.py line 146)

```python
raw_m2log2beta = -2 * np.log(2 * beta)
```

For beta_true = 0.1:
```
raw = -2 * log(2 * 0.1)
    = -2 * log(0.2)
    = -2 * (-1.6094)
    = 3.2189
```

### 2.2 Transform to "Used Value" (line 247)

```python
beta = torch.exp(self.raw_m2log2beta)  # This is NOT beta_true!
```

Let's call this **beta_code**:
```
beta_code = exp(raw)
          = exp(-2 * log(2 * beta_true))
          = exp(log((2 * beta_true)^(-2)))
          = (2 * beta_true)^(-2)
          = 1 / (4 * beta_true^2)
```

For beta_true = 0.1:
```
beta_code = 1 / (4 * 0.01) = 1 / 0.04 = 25
```

### 2.3 Usage in Locality Weights (line 263)

```python
logalpha = -beta * dist_sq_center
alpha = torch.exp(logalpha)
```

Substituting:
```
logalpha = -beta_code * dist^2
         = -(1 / (4 * beta_true^2)) * dist^2
         = -dist^2 / (4 * beta_true^2)

alpha = exp(-dist^2 / (4 * beta_true^2))
```

### 2.4 Final Formula for Alpha (Locality Weight)

```
alpha(dist) = exp(-dist^2 / (4 * beta_true^2))
```

This is a **Gaussian** with variance sigma^2 = 2 * beta_true^2, or equivalently:
```
alpha(dist) = exp(-(dist / (2 * beta_true))^2)
```

**Interpretation**: The locality weight alpha reaches e^(-1) at dist = 2 * beta_true.

### 2.5 Verification via Inverse Transform

From `utils.py` line 3405-3409:
```python
def logbetaexpr_to_beta(logbetaexpr):
    beta_paper = torch.exp(-0.5*logbetaexpr) * torch.tensor(0.5)
    return beta_paper
```

Let's verify:
```
beta_paper = exp(-0.5 * raw) * 0.5
           = exp(-0.5 * (-2 * log(2 * beta_true))) * 0.5
           = exp(log(2 * beta_true)) * 0.5
           = (2 * beta_true) * 0.5
           = beta_true  ✓
```

---

## 3. Rho Parameterization

### 3.1 Initialization (kernels.py line 147)

```python
raw_mlog2rho2 = -np.log(2 * rho**2)
```

For rho_true = 0.1:
```
raw = -log(2 * 0.01)
    = -log(0.02)
    = 3.9120
```

### 3.2 Transform to "Used Value" (line 248)

```python
rho2 = torch.exp(self.raw_mlog2rho2)  # This is NOT rho_true^2!
```

Let's call this **rho2_code**:
```
rho2_code = exp(raw)
          = exp(-log(2 * rho_true^2))
          = 1 / (2 * rho_true^2)
```

For rho_true = 0.1:
```
rho2_code = 1 / (2 * 0.01) = 1 / 0.02 = 50
```

### 3.3 Usage in Smoothness Kernel (line 270)

```python
C_smooth = torch.exp(-rho2 * dist_sq_pairwise)
```

Substituting:
```
C_smooth = exp(-rho2_code * dist^2)
         = exp(-(1 / (2 * rho_true^2)) * dist^2)
         = exp(-dist^2 / (2 * rho_true^2))
```

### 3.4 Final Formula for C_smooth (Smoothness Kernel)

```
C_smooth(dist) = exp(-dist^2 / (2 * rho_true^2))
```

This is a **Gaussian kernel** (RBF) with variance sigma^2 = rho_true^2, i.e., **rho_true is the standard deviation (length scale)**.

### 3.5 Verification via Inverse Transform

From `utils.py` line 3411-3416:
```python
def logrhoexpr_to_rho(logrhoexpr):
    rho_paper = torch.exp(-0.5*logrhoexpr) / torch.sqrt(torch.tensor(2))
    return rho_paper
```

Let's verify:
```
rho_paper = exp(-0.5 * raw) / sqrt(2)
          = exp(-0.5 * (-log(2 * rho_true^2))) / sqrt(2)
          = exp(0.5 * log(2 * rho_true^2)) / sqrt(2)
          = sqrt(2 * rho_true^2) / sqrt(2)
          = sqrt(2) * rho_true / sqrt(2)
          = rho_true  ✓
```

---

## 4. Physical Interpretations

### 4.1 Beta (Receptive Field Size Parameter)

**Formula**: alpha(dist) = exp(-dist^2 / (4 * beta_true^2))

| Characteristic | Formula | For beta_true = 0.1 |
|---------------|---------|---------------------|
| 1-sigma distance | 2 * beta_true | 0.2 |
| e^(-1) point | 2 * beta_true | 0.2 |
| Half-max point | 2 * beta_true * sqrt(ln(2)) = 1.665 * beta_true | 0.167 |
| 99% mass radius | 2 * beta_true * sqrt(2 * ln(10)) = 4.29 * beta_true | 0.429 |

**Physical meaning**:
- **beta_true** is the **half-width** of the RF at the e^(-1) level
- The RF has significant weight (alpha > 0.001) within a radius of approximately 4.8 * beta_true
- For beta_true = 0.1 on a [-1, 1] grid, the RF covers about ±0.2 from center (20% of the total span)

**IMPORTANT**: The variable named `beta` in `_compute_C_matrix()` is NOT beta_true! It is:
```
beta (in code) = 1 / (4 * beta_true^2)
```

### 4.2 Rho (Smoothness Length Scale)

**Formula**: C_smooth(dist) = exp(-dist^2 / (2 * rho_true^2))

| Characteristic | Formula | For rho_true = 0.1 |
|---------------|---------|---------------------|
| Standard deviation | rho_true | 0.1 |
| e^(-1) point | sqrt(2) * rho_true = 1.41 * rho_true | 0.141 |
| Half-max point | sqrt(2 * ln(2)) * rho_true = 1.18 * rho_true | 0.118 |
| 99% mass radius | sqrt(2 * 2.3) * rho_true = 2.15 * rho_true | 0.215 |

**Physical meaning**:
- **rho_true** is the **standard deviation** (length scale) of the smoothness kernel
- Pixels closer than rho_true apart are highly correlated (C_smooth > 0.6)
- Pixels more than 3 * rho_true apart are nearly uncorrelated (C_smooth < 0.01)

**IMPORTANT**: The variable named `rho2` in `_compute_C_matrix()` is NOT rho_true^2! It is:
```
rho2 (in code) = 1 / (2 * rho_true^2)
```

### 4.3 Combined Effect on C Matrix

The full C matrix is:
```
C[i,j] = alpha[i] * C_smooth[i,j] * alpha[j]
       = exp(-||xi - center||^2 / (4*beta^2)) * exp(-||xi - xj||^2 / (2*rho^2)) * exp(-||xj - center||^2 / (4*beta^2))
```

This creates:
1. A **localized** structure (via alpha) that weights pixels by distance from RF center
2. A **smooth** correlation structure (via C_smooth) that makes nearby pixels correlated

---

## 5. Parameter Bounds

### 5.1 Bounds for beta_true

**Context**: Pixel coordinates are on [-1, 1], so max distance is 2*sqrt(2) = 2.83.

| beta_true | Physical Interpretation | Suitability |
|-----------|------------------------|-------------|
| 0.01 | RF half-width = 0.02, only ~2 pixels active | Too small |
| 0.05 | RF half-width = 0.1, ~10% of image | Small but viable |
| 0.1 | RF half-width = 0.2, ~20% of image | **Typical** |
| 0.3 | RF half-width = 0.6, ~60% of image | Large RF |
| 1.0 | RF half-width = 2.0, covers entire image | Very large |

**Recommended range**: beta_true in [0.05, 0.5]

### 5.2 Corresponding Raw Parameter Bounds for beta

```
raw_m2log2beta = -2 * log(2 * beta_true)
```

| beta_true | raw_m2log2beta |
|-----------|----------------|
| 0.01 | -2 * log(0.02) = 7.82 |
| 0.05 | -2 * log(0.1) = 4.61 |
| 0.1 | -2 * log(0.2) = 3.22 |
| 0.3 | -2 * log(0.6) = 1.02 |
| 0.5 | -2 * log(1.0) = 0.00 |
| 1.0 | -2 * log(2.0) = -1.39 |

**Raw parameter range**: For beta_true in [0.01, 1.0], raw in [-1.39, 7.82]

### 5.3 Bounds for rho_true

**Context**: Typical pixel spacing on 108x108 grid with [-1,1] coords is 2/107 = 0.019.

| rho_true | Physical Interpretation | Suitability |
|----------|------------------------|-------------|
| 0.01 | ~0.5 pixel correlation length | Very rough |
| 0.02 | ~1 pixel correlation length | Minimal smoothing |
| 0.05 | ~2-3 pixel correlation length | Slight smoothing |
| 0.1 | ~5 pixel correlation length | **Typical** |
| 0.3 | ~16 pixel correlation length | Heavy smoothing |
| 0.5 | ~27 pixel correlation length | Very smooth |

**Recommended range**: rho_true in [0.02, 0.5]

### 5.4 Corresponding Raw Parameter Bounds for rho

```
raw_mlog2rho2 = -log(2 * rho_true^2)
```

| rho_true | raw_mlog2rho2 |
|----------|---------------|
| 0.01 | -log(0.0002) = 8.52 |
| 0.02 | -log(0.0008) = 7.13 |
| 0.05 | -log(0.005) = 5.30 |
| 0.1 | -log(0.02) = 3.91 |
| 0.3 | -log(0.18) = 1.71 |
| 0.5 | -log(0.5) = 0.69 |

**Raw parameter range**: For rho_true in [0.01, 0.5], raw in [0.69, 8.52]

---

## 6. Concerns and Edge Cases

### 6.1 Naming Confusion (CRITICAL)

**The code uses confusing variable names!**

In `_compute_C_matrix()`:
```python
beta = torch.exp(self.raw_m2log2beta)   # This is 1/(4*beta_true^2), NOT beta_true!
rho2 = torch.exp(self.raw_mlog2rho2)    # This is 1/(2*rho_true^2), NOT rho_true^2!
```

**Recommendation**: Consider renaming these variables for clarity:
```python
inv_4beta_sq = torch.exp(self.raw_m2log2beta)  # = 1/(4*beta_true^2)
inv_2rho_sq = torch.exp(self.raw_mlog2rho2)    # = 1/(2*rho_true^2)
```

### 6.2 Numerical Edge Cases

**Small beta_true (< 0.01)**:
- raw_m2log2beta > 7.8
- beta_code = exp(raw) > 2400
- alpha decays very rapidly (nearly all pixels masked out)
- Risk: only a few pixels remain, C becomes nearly diagonal

**Small rho_true (< 0.01)**:
- raw_mlog2rho2 > 8.5
- rho2_code = exp(raw) > 5000
- C_smooth becomes nearly diagonal (no smoothing)
- Risk: C has no off-diagonal structure, defeats purpose of smoothness

**Large beta_true (> 1.0)**:
- raw_m2log2beta < -1.4
- beta_code < 0.25
- alpha nearly constant (all pixels equally weighted)
- Risk: no localization, RF covers entire image

**Large rho_true (> 0.5)**:
- raw_mlog2rho2 < 0.7
- rho2_code < 2
- C_smooth nearly constant (all pixels equally correlated)
- Risk: very slow decay, C nearly rank-1

### 6.3 Mask Threshold Interaction

The mask threshold (0.001) interacts with beta:
```
alpha >= 0.001
exp(-dist^2 / (4*beta^2)) >= 0.001
dist^2 / (4*beta^2) <= ln(1000) = 6.91
dist <= sqrt(6.91 * 4 * beta^2) = 5.25 * beta
```

For beta_true = 0.1: max_dist = 0.525, mask radius ~28 pixels (on 108x108)

**If beta_true is too small**, the mask may select too few pixels and C becomes ill-conditioned.

### 6.4 Gradient Flow

Both parameterizations ensure smooth gradient flow because:
1. exp() has well-behaved gradients everywhere
2. The raw parameters are unconstrained
3. Chain rule applies cleanly: dL/d(beta_true) = dL/d(raw) * d(raw)/d(beta_true)

The gradient of raw w.r.t. beta_true:
```
d(raw)/d(beta_true) = d(-2*log(2*beta_true))/d(beta_true) = -2/beta_true
```

This becomes large for small beta_true, which could cause instability. However, the mask is computed with detached parameters, so this only affects the C matrix computation, not the mask structure.

---

## 7. Summary Table

| Quantity | Symbol | Formula | Default (beta=rho=0.1) |
|----------|--------|---------|------------------------|
| True beta | beta_true | User input | 0.1 |
| Raw beta param | raw_m2log2beta | -2 * log(2 * beta_true) | 3.22 |
| Code beta | beta_code | exp(raw) = 1/(4*beta_true^2) | 25 |
| Alpha formula | alpha | exp(-beta_code * dist^2) = exp(-dist^2/(4*beta_true^2)) | - |
| | | | |
| True rho | rho_true | User input | 0.1 |
| Raw rho param | raw_mlog2rho2 | -log(2 * rho_true^2) | 3.91 |
| Code rho2 | rho2_code | exp(raw) = 1/(2*rho_true^2) | 50 |
| C_smooth formula | C_smooth | exp(-rho2_code * dist^2) = exp(-dist^2/(2*rho_true^2)) | - |

---

## 8. Verification Code

```python
import numpy as np

def verify_beta_rho_math():
    """Verify the parameterization math."""

    # Test values
    beta_true = 0.1
    rho_true = 0.1

    # Forward transform (init)
    raw_beta = -2 * np.log(2 * beta_true)
    raw_rho = -np.log(2 * rho_true**2)

    print(f"beta_true = {beta_true}")
    print(f"  raw_m2log2beta = {raw_beta:.4f}")
    print(f"  beta_code = exp(raw) = {np.exp(raw_beta):.4f}")
    print(f"  Expected: 1/(4*beta_true^2) = {1/(4*beta_true**2):.4f}")
    print()

    print(f"rho_true = {rho_true}")
    print(f"  raw_mlog2rho2 = {raw_rho:.4f}")
    print(f"  rho2_code = exp(raw) = {np.exp(raw_rho):.4f}")
    print(f"  Expected: 1/(2*rho_true^2) = {1/(2*rho_true**2):.4f}")
    print()

    # Inverse transform (from utils.py)
    beta_recovered = np.exp(-0.5 * raw_beta) * 0.5
    rho_recovered = np.exp(-0.5 * raw_rho) / np.sqrt(2)

    print(f"Inverse transform verification:")
    print(f"  beta_recovered = {beta_recovered:.6f} (expected {beta_true})")
    print(f"  rho_recovered = {rho_recovered:.6f} (expected {rho_true})")

    # Verify formulas at specific distance
    dist = 0.2  # = 2 * beta_true

    beta_code = np.exp(raw_beta)
    rho2_code = np.exp(raw_rho)

    alpha = np.exp(-beta_code * dist**2)
    alpha_expected = np.exp(-dist**2 / (4 * beta_true**2))

    c_smooth = np.exp(-rho2_code * dist**2)
    c_smooth_expected = np.exp(-dist**2 / (2 * rho_true**2))

    print()
    print(f"At dist = {dist}:")
    print(f"  alpha = {alpha:.6f}, expected = {alpha_expected:.6f}")
    print(f"  c_smooth = {c_smooth:.6f}, expected = {c_smooth_expected:.6f}")

if __name__ == "__main__":
    verify_beta_rho_math()
```

Expected output:
```
beta_true = 0.1
  raw_m2log2beta = 3.2189
  beta_code = exp(raw) = 25.0000
  Expected: 1/(4*beta_true^2) = 25.0000

rho_true = 0.1
  raw_mlog2rho2 = 3.9120
  rho2_code = exp(raw) = 50.0000
  Expected: 1/(2*rho_true^2) = 50.0000

Inverse transform verification:
  beta_recovered = 0.100000 (expected 0.1)
  rho_recovered = 0.100000 (expected 0.1)

At dist = 0.2:
  alpha = 0.367879, expected = 0.367879
  c_smooth = 0.135335, expected = 0.135335
```

---

*End of verification document*
