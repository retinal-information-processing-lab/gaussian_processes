# Subspace Optimization for GP Utility: PCA, C-Eigenspace, and Fourier Analysis

**Status**: Ongoing investigation — analysis of candidate approaches, no implementation yet.
**Context**: Optimizing images x* to maximize a GP-based utility function U(x*), where gradient
ascent in pixel space produces spatially smooth outputs due to the kernel's C matrix acting as
a low-pass filter.

---

## 1. Setup and Notation

We have d pixels in the masked image region (d ~ 900 for a 30x30 mask).
Images are vectors x in R^d. The GP kernel involves a d x d positive semi-definite matrix C:

    C = diag(alpha) * C_smooth * diag(alpha)

where C_smooth[i,j] = exp(-rho * ||xi_i - xi_j||^2) encodes spatial pixel correlations,
and alpha_i = exp(-beta * ||xi_i - xi_0||^2) is the locality mask.

We also have a dataset of N natural images {x_1, ..., x_N} with empirical mean mu and
empirical covariance Sigma_data = (1/N) * X_centered^T * X_centered.

The goal: find x* maximizing U(x*) while constraining x* to resemble a natural image.

---

## 2. Principal Component Analysis

### 2.1 Definition

PCA computes the eigendecomposition of the data covariance:

    Sigma_data = V * Lambda * V^T

where V = [v_1, ..., v_d] are orthonormal eigenvectors (principal components),
and Lambda = diag(lambda_1, ..., lambda_d) with lambda_1 >= lambda_2 >= ... >= 0.

Any image x can be decomposed as:

    x = mu + sum_k z_k * v_k,    z_k = v_k^T * (x - mu)

Truncation to K components gives the best rank-K approximation in MSE:

    x ~ mu + sum_{k=1}^{K} z_k * v_k

### 2.2 What PCA captures

PCA diagonalizes the data covariance. The eigenvectors are the directions of maximum
variance in the dataset. The eigenvalues measure how much variance each direction carries.

Critically, PCA captures **second-order statistics only**: the mean mu and the covariance
Sigma_data. It is blind to all higher-order statistics of the image distribution, including:

- Phase correlations (which create coherent edges and contours)
- Sparsity structure (natural images are sparse in wavelet bases)
- Non-Gaussian marginals (pixel differences have heavy-tailed distributions)

The Gaussian distribution N(mu, Sigma_data) has exactly the same PCA as the true natural
image distribution. However, samples from this Gaussian do not look like natural images —
they appear as spatially correlated noise with the correct power spectrum but random phases.

### 2.3 PCA-space optimization

Writing x* = mu + V_K * z and optimizing U(mu + V_K * z) over z in R^K restricts the
image to a K-dimensional affine subspace. The gradient is:

    nabla_z U = V_K^T * nabla_x U

This projects the pixel-space gradient onto the K principal directions.

This restricts which **directions** the optimizer can move in, but does not constrain
**how far** it moves. Any z in R^K is allowed. Values of z far from the data distribution
produce images in the PCA subspace that do not resemble natural images.

PCA-space optimization is therefore a **subspace constraint**, not a **distributional
constraint**. It restricts the optimization to directions where natural images vary, but
does not enforce membership in the natural image distribution.

---

## 3. Eigenspace of C

### 3.1 Definition

The kernel matrix C has eigendecomposition:

    C = U * Gamma * U^T

where U = [u_1, ..., u_d] are orthonormal eigenvectors and
Gamma = diag(gamma_1, ..., gamma_d) with gamma_1 >= ... >= gamma_d >= 0.

The eigenvectors of C represent the modes to which the kernel is most sensitive.
In all kernel evaluations, x enters through products like x^T C x, which in the
eigenbasis become sum_i gamma_i * (u_i^T x)^2. Directions with large eigenvalues
contribute strongly; directions with small eigenvalues are nearly invisible to the kernel.

### 3.2 Implicit eigenspace optimization under gradient ascent

For the RBF kernel, the gradient of a single kernel evaluation is:

    nabla_{x*} k(x*, z_m) = -k(x*, z_m) * C * (x* - z_m) / l^2

The multiplication by C means the gradient component along eigenvector u_i is scaled
by gamma_i. Directions with small eigenvalues receive proportionally small gradients.

Standard gradient ascent therefore **already concentrates movement along the top
eigenvectors of C**. The optimizer implicitly avoids directions the kernel cannot see.

### 3.3 Explicit eigenspace projection

Restricting x* = U_K * z (keeping eigenvectors with gamma_i > threshold) makes this
filtering explicit. This is analogous to the EIGVAL_TOL threshold used in the
vargp_direct mode for K_tilde.

The effect is primarily on convergence: flat directions are removed, the optimization
landscape is better conditioned, and the effective dimensionality is reduced. The
solution should not change significantly, because the removed directions already had
near-zero gradients.

### 3.4 What C-eigenspace optimization does not provide

The eigenvectors of C are determined by kernel geometry (the spatial Gaussian C_smooth
and the locality mask alpha), not by the image dataset. An image in the C-eigenspace
has the right structure for the kernel to evaluate it, but has no reason to resemble
a natural image.

C-eigenspace optimization is a **convergence aid**, not a naturalness constraint.

---

## 4. PCA and Fourier Equivalence Under Stationarity

### 4.1 Stationary processes and Toeplitz structure

A second-order stationary process on a regular grid has covariance:

    Sigma[i,j] = f(|i - j|)

for some function f. This makes Sigma a Toeplitz matrix (1D) or a block-Toeplitz
matrix with Toeplitz blocks (2D).

### 4.2 Szego's theorem

For Toeplitz matrices, as d -> infinity, the eigenvectors converge to Fourier modes
and the eigenvalues converge to the power spectral density S(omega) = sum_k f(k) e^{-i omega k}.

In 2D, the analogous result holds for block-Toeplitz matrices: eigenvectors converge
to 2D Fourier modes, eigenvalues to the 2D power spectrum S(omega_x, omega_y).

### 4.3 Application to natural images

Natural images have approximately stationary second-order statistics. Their power
spectrum follows approximately S(omega) ~ 1/|omega|^2 (the well-known 1/f^2 law).
This means:

- The eigenvectors of Sigma_data approximately equal the Fourier basis
- The eigenvalues of Sigma_data approximately follow 1/k^2 ordering
- Truncating to K principal components is approximately equivalent to
  retaining the K lowest-frequency Fourier modes

Therefore: **PCA-space optimization on natural images is approximately equivalent
to optimizing in the low-frequency Fourier subspace.**

### 4.4 Caveats

This equivalence is approximate. It weakens in several situations relevant to our problem:

1. **Irregular masked region**: The Fourier basis is defined on regular grids.
   PCA handles arbitrary pixel geometry. The equivalence is exact only on
   rectangular domains.

2. **Non-stationarity due to RF mask**: The locality mask alpha introduces
   non-stationarity — pixels near the RF center have higher variance than
   peripheral pixels. This breaks the Toeplitz structure of Sigma_data.

3. **Finite dataset**: With N ~ 10,000 images in d ~ 900 dimensions,
   Sigma_data is estimated from a moderate sample. Eigenvectors may differ
   from the population Fourier modes, especially for higher-order components.

4. **Non-stationarity in natural images**: Natural image statistics are not
   perfectly stationary (e.g., brightness gradients, horizon effects).
   However, for small image patches (30x30), approximate stationarity
   is a reasonable assumption.

Despite these caveats, the conceptual equivalence holds: PCA on natural images
captures their power spectrum structure, which is the same information that
Fourier analysis provides. Neither captures higher-order statistics.

---

## 5. Comparison of the Three Subspace Approaches

| Property | PCA (Sigma_data) | C-eigenspace | Fourier |
|----------|------------------|--------------|---------|
| Basis source | Data covariance | Kernel matrix | Fixed (sinusoids) |
| Data-dependent? | Yes | No (model-dependent) | No |
| Captures second-order stats? | Yes | No (kernel geometry) | Yes |
| Captures higher-order stats? | No | No | No |
| Constrains to natural distribution? | No | No | No |
| Handles irregular mask? | Yes | Yes | Poorly |
| Effect on optimization | Subspace constraint | Convergence aid | Subspace constraint |
| Relationship to gradient | Introduces new info | Reorganizes existing gradient | Introduces new info |

### 5.1 PCA vs C-eigenspace

These address different concerns:

- PCA restricts the image to directions where natural images vary most (data-driven).
  This introduces information the GP does not have.
- C-eigenspace restricts the image to directions the kernel is most sensitive to
  (model-driven). This makes explicit what gradient ascent already does implicitly.

The overlap between the two eigenspaces depends on how well C_smooth approximates
the data covariance. Both capture distance-dependent spatial correlations, so low-order
eigenvectors likely overlap. Higher-order eigenvectors may diverge: PCA captures
empirical correlation patterns (including texture and edge statistics at second order),
while C captures a parametric Gaussian spatial kernel.

### 5.2 PCA vs Fourier

Under approximate stationarity of natural image statistics (Section 4), PCA and Fourier
are approximately equivalent. PCA has practical advantages (handles irregular geometry,
adapts to non-stationarity, provides data-driven truncation criterion) but captures the
same information: the power spectrum of the image distribution.

---

## 6. Implications for Distribution-Constrained Optimization

The three subspace approaches (PCA, C-eigenspace, Fourier) all provide subspace
constraints on the optimization, not distributional constraints. None of them ensure
that the optimized image x* is a plausible sample from the natural image distribution.

The distinction matters:

- **Subspace constraint**: x* lies in a K-dimensional subspace. This restricts which
  directions the optimizer can explore. Images with the right power spectrum but wrong
  phase structure (i.e., correlated noise) satisfy this constraint.

- **Distributional constraint**: x* is a plausible sample from p(x). This requires
  matching all statistics of the distribution, including higher-order structure
  (phases, edges, sparsity). Methods that enforce this include generative models
  that learn the full distribution (diffusion models, GANs, normalizing flows).

A subspace approach may still be useful as a diagnostic: it can reveal whether
restricting to the right power spectrum meaningfully changes utility values, and
whether the GP utility function itself operates primarily at the level of second-order
statistics. But it does not address the full distributional constraint.

---

## 7. Open Questions

1. **Empirical overlap**: How much do the top eigenvectors of Sigma_data and C
   overlap in practice? This determines whether C-eigenspace optimization
   incidentally captures some data structure, or operates in a different subspace entirely.

2. **Stationarity check**: How well does the Toeplitz approximation hold for the
   masked image region and the specific dataset? A direct comparison of PCA eigenvectors
   and 2D Fourier modes on the masked region would quantify this.

3. **Utility sensitivity to higher-order statistics**: Does the GP utility function
   assign different values to a natural image and its PCA reconstruction (same power
   spectrum, degraded phases)? If the utility is insensitive to phase structure, then
   subspace constraints may be sufficient in practice, even though they do not enforce
   distributional membership.

4. **Practical sufficiency at low resolution**: At 30x30 pixels, the amount of
   higher-order structure in natural images is limited. Whether second-order constraints
   are "close enough" to distributional constraints at this resolution is an empirical
   question.
