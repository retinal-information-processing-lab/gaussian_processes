# Why Diffusion-Guided Optimization for Stimulus Design

## The Problem

We optimize images x* to maximize a GP-based utility function U(x*) via gradient ascent.
The gradient flows through the GP kernel, which involves a matrix C:

    nabla_{x*} k(x*, z_m) = -k(x*, z_m) * C * (x* - z_m) / l^2

The C matrix contains C_smooth, a Gaussian kernel on pixel locations:

    C_smooth[i,j] = exp(-rho * ||xi_i - xi_j||^2)

Multiplication by C_smooth is equivalent to spatial convolution with a Gaussian.
This acts as a low-pass filter: the utility gradient is inherently smooth, regardless
of the number of inducing points, the training data, or the utility definition.

The result: gradient-ascent-optimized images lack the high-frequency structure
(edges, textures, local contrast) present in natural images. They also tend to
have pixel values outside the physical display range.

## Why Subspace Approaches Are Insufficient

PCA-space optimization (restricting x* to the top eigenvectors of the data covariance)
constrains the image to have the right power spectrum. But PCA captures only second-order
statistics. Under approximate stationarity of natural image statistics, PCA eigenvectors
converge to Fourier modes (Szego's theorem), so PCA-space optimization is approximately
equivalent to Fourier low-pass filtering. Neither enforces higher-order structure
(phase coherence, edge sparsity) that makes images look natural.

Similarly, optimizing in the eigenspace of C restricts the image to directions the kernel
is sensitive to. But gradient ascent already implicitly moves along these directions
(gradient component proportional to eigenvalue). This is a convergence aid, not a
naturalness constraint.

Since the optimized stimuli will be presented to real neurons, they must be
indistinguishable from natural images — requiring the full distribution, not
just second-order statistics.

## The Decomposition

The problem has two objectives:

1. **Informativeness**: x* should maximize utility (reduce uncertainty about neural responses)
2. **Naturalness**: x* should be a plausible sample from the natural image distribution

Current gradient ascent handles (1) but not (2). The utility gradient provides a smooth
directional signal about which region of image space is informative. What is missing is
a complementary signal about what natural images look like at the fine-grained level.

## Why Diffusion Models

A diffusion model trained on natural images learns the score function:

    s(x, t) = nabla_x log p_t(x)

This is the gradient of the log-probability of the data distribution at noise level t.
At low noise levels, it encodes the full statistical structure of natural images,
including higher-order statistics that PCA and Fourier analysis miss.

Guided generation modifies the reverse diffusion process:

    score_guided = s_theta(x_t, t) + w * nabla_{x_t} U(x_hat_0)

The score model provides high-frequency naturalness (edges, textures, local contrast).
The utility gradient provides low-frequency direction (which region of image space is
informative). The guidance strength w controls the tradeoff.

The smoothness of the utility gradient, which was the problem in direct optimization,
becomes acceptable in this framework: it only needs to bias generation toward informative
regions, not prescribe pixel values.

## Feasibility

- Image dimensions: 30x30 grayscale (~900 pixels) — far simpler than typical diffusion applications
- Dataset: ~10,000 natural images — moderate but workable at this resolution
- Architecture: small MLP or tiny U-Net, training in minutes on GPU
- Generation: 50-200 denoising steps, each requiring one score evaluation + one GP evaluation

## Status

This is an ongoing investigation. No implementation exists yet.
See `subspace_optimization_pca_ceigen_fourier.md` (in `investigations/utility/`)
for the analysis that motivated this direction.
