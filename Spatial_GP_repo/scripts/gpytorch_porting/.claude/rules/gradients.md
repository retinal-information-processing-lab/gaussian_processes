---
paths:
  - "*gradient*.py"
---

# Analytical Gradients Reference

**Authoritative source for**: Kernel gradient formulas, VJP implementation, M-step integration

This document consolidates all analytical gradient documentation for the arc-cosine kernel.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Notation](#2-mathematical-notation)
3. [C Matrix Definition and Gradients](#3-c-matrix-definition-and-gradients)
4. [K Matrix Gradients](#4-k-matrix-gradients)
5. [VJP Implementation](#5-vjp-implementation)
6. [M-step Integration](#6-m-step-integration)
7. [Numerical Stability](#7-numerical-stability)

---

## 1. Overview

### Problem Statement

The kernel gradient computation has three implementations:
1. **autograd** (default): PyTorch automatic differentiation
2. **vjp**: VJP-based analytical - same speed as autograd, explicit formulas
3. **jacobian**: Jacobian materialization - slow reference, matches original varGP

Usage: `kernel = ArcCosineKernel(..., gradient_mode='vjp')`

### Why Multiple Implementations?

The Jacobian approach materializes 5 dC matrices and 5 dK matrices (15 large matrix multiplies). VJP computes dL/dC ONCE in backward, then chains to each hyperparameter via element-wise ops.

| Mode | Time per call | Use case |
|------|---------------|----------|
| autograd | 16.2 ms | Default, simplest |
| vjp | 16.2 ms | Explicit control, no speed penalty |
| jacobian | 73.1 ms | Exact match with original varGP |

---

## 2. Mathematical Notation

### Hyperparameters

| Symbol | Code name | Description |
|--------|-----------|-------------|
| sigma_0 | `sigma_0` | Bias variance |
| Amp | `Amp` | Amplitude (internal to kernel) |
| beta | `raw_m2log2beta` | RF size, parameterized as `-2log(2beta)` |
| rho | `raw_mlog2rho2` | Smoothness, parameterized as `-log(2rho^2)` |
| eps_0 | `eps_0x`, `eps_0y` | RF center position |

### Log-space Parameterization

```
theta_beta = -2log(2*beta)  ->  beta = exp(-theta_beta/2) / 2
theta_rho = -log(2*rho^2)   ->  rho^2 = exp(-theta_rho) / 2
```

### Intermediate Quantities

For inputs x1, x2:
```
v1 = x1.T @ C @ x1 + sigma_0^2    (variance)
v2 = x2.T @ C @ x2 + sigma_0^2
c12 = x1.T @ C @ x2 + sigma_0^2   (cross-term)

X1 = sqrt(v1)
X2 = sqrt(v2)
X1X2 = X1 * X2                    (magnitude)

cos(delta) = c12 / X1X2           (normalized inner product)
delta = arccos(cos(delta))        (angle)
sin(delta) = sqrt(1 - cos(delta)^2)

J(delta) = sin(delta) + (pi - delta)*cos(delta)   (angular term)

K(x1, x2) = (1/pi) * X1X2 * J(delta)
```

---

## 3. C Matrix Definition and Gradients

### C Matrix Formula

```
C[i,j] = Amp * alpha[i] * C_smooth[i,j] * alpha[j]
```

Where:
```
alpha[i] = exp(-beta_factor * dist_center[i])
dist_center[i] = (xcord[i] - eps_0x)^2 + (ycord[i] - eps_0y)^2
beta_factor = exp(raw_m2log2beta) = 1/(4*beta^2)

C_smooth[i,j] = exp(-rho_factor * dist_pairwise[i,j])
dist_pairwise[i,j] = (xcord[i] - xcord[j])^2 + (ycord[i] - ycord[j])^2
rho_factor = exp(raw_mlog2rho2) = 1/(2*rho^2)
```

### C Matrix Gradients

```
dC/d(Amp) = C / Amp

dC/d(eps_0x) = 2 * beta_factor * C * (xcord[:, None] + xcord[None, :] - 2*eps_0x)

dC/d(eps_0y) = 2 * beta_factor * C * (ycord[:, None] + ycord[None, :] - 2*eps_0y)

dC/d(raw_m2log2beta) = C * (logalpha[:, None] + logalpha[None, :])

dC/d(raw_mlog2rho2) = C * logCsmooth
```

---

## 4. K Matrix Gradients

### Gradient w.r.t. sigma_0

```python
dX1X2_sigma = sigma_0^2 * (X2/X1 + X1/X2)
dcos_sigma = (2*sigma_0^2 - cos_delta * dX1X2_sigma) / X1X2
dJ_sigma = -(delta - pi) * dcos_sigma / pi

dK['sigma_0'] = (X1X2 * dJ_sigma + dX1X2_sigma * J) / sigma_0
```

### Gradient w.r.t. C-dependent Parameters

For any parameter theta affecting K through C:

```python
# Given dC[theta] matrix
dX1_theta = 0.5 * (x1 * (dC[theta] @ x1)).sum(dim=0) / X1
dX2_theta = 0.5 * (x2 * (dC[theta] @ x2)).sum(dim=0) / X2

dX1X2_theta = dX1_theta[:, None] * X2 + X1[:, None] * dX2_theta

dc12_theta = x1.T @ dC[theta] @ x2
dcos_theta = (dc12_theta - cos_delta * dX1X2_theta) / X1X2
dJ_theta = -(delta - pi) * dcos_theta / pi

dK[theta] = dX1X2_theta * J + X1X2 * dJ_theta
```

---

## 5. VJP Implementation

### Concept

VJP computes gradients using Vector-Jacobian Products without materializing intermediate Jacobians. Given G = dL/dK, we chain backward through K -> J -> cos(delta) -> C -> hyperparameters.

### Backward Pass Chain

```
1. dL/dK -> dL/dJ, dL/dX1X2
   dL/dJ = G * X1X2
   dL/dX1X2 = G * J

2. dL/dJ -> dL/dcosdelta
   dJ/dcosdelta = (pi - delta) / pi
   dL/dcosdelta = dL/dJ * (pi - delta) / pi

3. dL/dcosdelta -> dL/dx1Cx2, additional dL/dX1X2
   dL/dx1Cx2 = dL/dcosdelta / X1X2
   dL/dX1X2_total = G * J - dL/dcosdelta * cosdelta / X1X2

4. dL/dX1X2 -> dL/dX1, dL/dX2
   dL/dX1 = dL/dX1X2 @ X2
   dL/dX2 = dL/dX1X2.T @ X1

5. dL/dX1, dL/dX2 -> dL/dV1, dL/dV2
   dL/dV1 = dL/dX1 / (2 * X1)
   dL/dV2 = dL/dX2 / (2 * X2)

6. dL/dV1, dL/dV2, dL/dx1Cx2 -> dL/dC
   dL/dC = x1 @ diag(dL/dV1) @ x1.T
         + x2 @ diag(dL/dV2) @ x2.T
         + x1 @ dL/dx1Cx2 @ x2.T

7. dL/dC -> dL/d(hyperparameters)
   dL/dAmp = (dL/dC * C).sum() / Amp
   dL/dbeta = -(dL/dalpha * alpha * dist_center).sum()
   dL/drho = -(dL/dS * S * dist_pairwise).sum()
   dL/dsigma_0 = 2*sigma_0 * [dL/dV1.sum() + dL/dV2.sum() + dL/dx1Cx2.sum()]
```

### Complexity Comparison

| Approach | Operations |
|----------|------------|
| Jacobian | 15 * O(nx^2 * n) - materialize 5 dC and 5 dK |
| VJP | O(nx^2 * n) + O(nx^2) - one backward pass |

---

## 6. M-step Integration

### Why VJP Cannot Be Used for LBFGS M-step

The VJP approach requires dL/dK at backward time, which changes every LBFGS line search evaluation. For M-step with LBFGS, we need:

```python
# Compute dK/dtheta matrices ONCE at start
K, dK = compute_kernel_with_grads(...)

# LBFGS closure reuses cached dK:
for key in dK:
    grad[key] = (dL_dK * dK[key]).sum()  # Fast, no recomputation
```

### M-step Analytical Gradient Formulas

**dlambda_m, dlambda_var:**
```python
da[key] = (dK[key] - a @ dK_tilde[key]) @ K_tilde_inv
dlambda_m[key] = da[key] @ m
dlambda_var[key] = (dKvec[key]
    + torch.einsum('ij,ji->i', 2*da[key], V @ a.T)
    - torch.einsum('ij,ij->i', dK[key], a)
    - torch.einsum('ij,ij->i', K, da[key]))
```

**dKL:**
```python
c = V @ K_tilde_inv
b = K_tilde_inv @ m
B = dK_tilde[key] @ K_tilde_inv
dKL[key] = 0.5*trace(B) - 0.5*trace(c@B) - 0.5*b.T@(B@m)
```

### Parameter Transform Correction

GPyTorch uses softplus for sigma_0 and Amp:
```python
# param = softplus(raw_param)
# Gradient chain: d/d(raw) = d/d(param) * sigmoid(raw)
kernel.raw_sigma_0.grad = dL['sigma_0'] * torch.sigmoid(kernel.raw_sigma_0)
kernel.raw_Amp.grad = dL['Amp'] * torch.sigmoid(kernel.raw_Amp)
```

---

## 7. Numerical Stability

1. **Clamp cos(delta)** to [-1+1e-6, 1-1e-6] before arccos
2. **Division by X1X2** needs jitter: `X1X2 + 1e-7`
3. **Symmetrize K_tilde**: `(K + K.T) / 2`
4. **Eigenvalue threshold**: `eigvals > max(eigvals.max() * 1e-4, 1e-4)`
5. **Use float64** throughout (kernel values can reach ~10,000)

---

## Files

- **Implementation**: `analytical_gradients.py` (Jacobian), `analytical_gradients_vjp.py` (VJP)
- **Kernel integration**: `kernels.py` (gradient_mode parameter)
- **M-step analytical**: `direct_vargp.py:mstep_lbfgs_analytical()`
- **Reference**: `utils.py:acosker()` lines 3663-3813

---

*Consolidated: January 2025*
*Sources: VJP_ANALYTICAL_GRADIENTS.md, MSTEP_ANALYTICAL_HANDOFF.md, ANALYTICAL_GRADIENTS_MATH.md*
