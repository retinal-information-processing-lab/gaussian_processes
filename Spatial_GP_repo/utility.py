"""
Slim version of batch_utility_w_grad with single optimizer.step() call (correct usage)
"""

from tabnanny import verbose
import torch
import matplotlib.pyplot as plt
import numpy as np
# from gaussian_processes.Spatial_GP_repo.utils import (
    # TORCH_DTYPE, DEVICE)

from torchlambertw.special import lambertw as torch_lambertw


from gaussian_processes.Spatial_GP_repo import utils as GP_utils
from gaussian_processes.Spatial_GP_repo.utils import TORCH_DTYPE, DEVICE, TINY, utility

from gaussian_processes.Spatial_GP_repo import analysis_utils as GP_analysis_utils


# ========================= CUSTOM LAMBERT W WITH STABLE GRADIENTS =========================

class LambertWLogFunction(torch.autograd.Function):
    """
    Compute W_0(exp(y)) without computing exp(y).
    Solves w + log(w) = y via Newton iteration.

    Does not calculate W for z < 0 ( cannot write it as exp(y) )
    """

    @staticmethod
    def forward(ctx, y):

        # Clamping low values to avoid log(small). Will be discarded by where(log)
        safe_log_input = y.clamp(min=1.0) 

        # Piecewise initial guess:
        #   y >= 2:    asymptotic W(e^y) ≈ y - log(y)
        #   0 <= y < 2: linear interp (W(1)=0.567, W(e^2)=1.56)
        #   y < 0:     small-z asymptotic W(z) ≈ z = e^y
        w = torch.where( y >= 2.0, y - torch.log(safe_log_input), # "safe" avoids large negative vals
            torch.where( y >= 0.0, 0.567 + 0.5 * y,               # Linear interpolation between W(e^0)=0.567 and W(e^2)=1.56
                        torch.exp(y)))                # Small z : W(z) ≈ z = e^y 
        

        # --- Newton Iterations ---
        # Solve f(w) = w + log(w) - y = 0
        # Update: w_new = w - (w + log(w) - y) / (1 + 1/w)
        # Simplified: w_new = w - w * (w + log(w) - y) / (1 + w)
        for _ in range(10):
            w = w.clamp(min=TINY)
            log_w = torch.log(w)
            w = w - w * (w + log_w - y) / (w + 1.0)

        # For y < underflow threshold, W(e^y) ≈ 0``
        w = w.clamp(min=TINY)

        ctx.save_for_backward(w)
        return w

    @staticmethod
    def backward(ctx, grad_output):
        w, = ctx.saved_tensors
        # dW/dy = W / (1 + W)  — the e^y terms cancel
        return grad_output * w / (1.0 + w)

class LambertWFunction(torch.autograd.Function):
    """
    Custom autograd function for Lambert W with numerically stable gradient.

    Forward pass: Uses torchlambertw.special.lambertw (iterative solver)
    Backward pass: Uses analytical formula dW/dz = W / (z * (1 + W))

    The analytical gradient is more stable than differentiating through the
    iterative solver, especially for edge cases where z→0 or W→-1.
    """

    @staticmethod
    def forward(ctx, z, k=0):
        """
        Forward pass: Compute Lambert W function.

        Args:
            z: Input tensor
            k: Branch (0 for principal, -1 for non-principal)

        Returns:
            w: Lambert W function of z
        """
        w = torch_lambertw(z, k=k)
        ctx.save_for_backward(z, w)
        ctx.k = k
        return w

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute gradient using analytical formula.

        The derivative of Lambert W is: dW/dz = W / (z * (1 + W))

        This can be numerically unstable when:
        - z ≈ 0 (division by zero)
        - W ≈ -1 (division by zero, occurs at z = -1/e)

        We handle these cases by clamping the denominator and filtering non-finite values.
        """
        z, w = ctx.saved_tensors

        # Analytical gradient: dW/dz = W / (z * (1 + W))
        denominator = z * (1.0 + w)

        # Avoid division by zero - clamp denominator away from zero
        eps = 1e-10
        safe_denom = torch.where(
            torch.abs(denominator) > eps,
            denominator,
            torch.sign(denominator) * eps + (denominator == 0).float() * eps
        )

        grad_z = grad_output * w / safe_denom

        # Set gradient to zero for invalid values (inf, nan)
        grad_z = torch.where(
            torch.isfinite(grad_z),
            grad_z,
            torch.zeros_like(grad_z)
        )

        return grad_z, None  # None for k (not differentiable)

def lambertw_stable(z, k=0):
    """
    Lambert W function with stable gradients for backpropagation.

    Uses custom autograd implementation that computes gradients via the
    analytical formula dW/dz = W/(z(1+W)) rather than differentiating
    through the iterative solver.

    Args:
        z: Input tensor (real-valued)
        k: Branch selection (0 for principal, -1 for non-principal)

    Returns:
        W(z): Lambert W function of z
    """
    return LambertWFunction.apply(z, k)

def lambertw0_log(y):
    """
    Lambert W function of exp(y) using only y, 0-th branch.

    Uses custom autograd implementation that computes gradients via the
    analytical formula dW/dz = W/(z(1+W)) rather than differentiating
    through the iterative solver.

    Args:
        y: Input tensor (real-valued)

    Returns:
        W(z): Lambert W function of z
    """
    return LambertWLogFunction.apply(y)

# ======================== Unconditioned Utility Functions ========================

def argmax_g_old(r, sigma2, mu):
    """
    PyTorch-native version using torchlambertw with stable gradients.

    Computes the argmax(g) from 0 to r_cutoff
    For laplace approximation of logp(r|x,D) [eq 34 Paper PNAS].
    In the paper is Lambda cause there where no A and lambda_0. We call it g here.
    This version uses lambertw_stable() which provides numerically stable gradients via
    analytical formula dW/dz = W/(z(1+W)) rather than differentiating through the iterative solver.

    Args:
        r: tensor of values from 0 to r_cutoff (max number for the sum in eq 29 Paper PNAS)
        sigma2: tensor of shape (nstar) : Moments of g = Alambda(x) + lambda_0
        mu: tensor of shape (nstar)

        nstar: n of inputs we are calculating argmax_g for. For conditioned utility should be 1.
    Returns:
        lambW: tensor of shape (r, nstar) - the computed lambda values
        sum_mask: tensor of shape (r, nstar) - mask indicating valid (non-infinite) values
    """
    # Ensure all inputs are on the same device
    device = r.device
    eps = 1e-10  # Small epsilon for numerical stability # BUG : set to machine epsilon

    rsigma2 = torch.outer(r, sigma2)                          # shape (r, nstar)

    # Prevent overflow in exp() by clamping the exponent
    # exp(88) ≈ 1.65e38 (close to float32 max)
    # exp(709) ≈ 8.22e307 (close to float64 max)
    # We use 85 to be safe and allow some margin
    max_exp = 85.0
    exponent = rsigma2 + mu
    exponent_clamped = torch.clamp(exponent, max=max_exp)

    # Compute z with clamped exponent to prevent inf
    z = sigma2.unsqueeze(0) * torch.exp(exponent_clamped)     # shape (r, nstar)

    # Track which values would have overflowed (for masking in caller)
    # Note: We still compute with clamped values to maintain gradient flow
    would_overflow = exponent > max_exp
    sum_mask = ~would_overflow                                 # shape (r, nstar)

    # For values that would overflow, set them to a small value
    # This prevents numerical issues in Lambert W
    z = torch.where(sum_mask, z, torch.full_like(z, eps))

    # Also mask rsigma2 for consistency
    rsigma2 = torch.where(sum_mask, rsigma2, torch.tensor(0., device=device))  # shape (r, nstar)

    # Use stable Lambert W with custom gradient implementation
    lambertW_result = lambertw_stable(z, k=0)
    argmax_g = rsigma2 + mu - lambertW_result.real

    return argmax_g, sum_mask

def argmax_g(r, sigma2, mu):
    """
    
    Computes r * sigma2 + mu - W_0( sigma2 * exp( r * sigma2 + mu ) )

    Args:
        r: tensor of values from 0 to r_cutoff=R shape [R]
        sigma2: tensor of shape (N) : Moments of g = Alambda(x) + lambda_0 , shape [N]
        mu: tensor of shape (N)

        N: n of inputs we are calculating argmax_g for. For conditioned utility should be 1.
    Returns:
        lambW: tensor of shape (r, nstar) - the computed lambda values
        sum_mask: tensor of shape (r, nstar) - mask indicating valid (non-infinite) values
    """
    # Ensure all inputs are on the same device
    device = r.device

    rsigma2 = sigma2[:,None] * r[None,:] # shape (N, R)

    # e^y = sigma2 * exp( rsigma2 + mu )
    y = torch.log(sigma2.clamp(min=TINY))[:,None] + rsigma2 + mu[:,None]  # shape (N, R)

    lambert_W_result = lambertw0_log( y )

    # DEBUG
    # import time
    # torch.autograd.grad(  lambertw0_log(y)[0,40] , y, retain_graph=True)
    # torch.cuda.synchronize() 

    # t0 = time.perf_counter()
    # torch.autograd.grad(  lambertw0_log(y)[0,40] , y, retain_graph=True)
    # torch.cuda.synchronize()
    # t1 = time.perf_counter()

    # print(f" Gradient time lambertw_log: {(t1 - t0)*1000:.6f} ms ")


    argmax_g = rsigma2 + mu[:,None] - lambert_W_result

    return argmax_g

def laplace_approximations_old(r, sigma2, mu):
    # Computes p( r|x,D ) [eq 31 Paper PNAS]. It's the first term of I(r;f|x,D) [eq:27] 

    # Calculating lambda mean for different values of r
    # Note that the sum over r was already reduced by a r_cutoff set in the utility function
    # If despite this cutoff, the exponential still goes to infinity for certain r values, they are removed from the sum
    # lambda_mean, sum_mask = nd_lambda_r_mean(r, sigma2, mu)          # shape (r, nstar) , (r, nstar)
    
    g_bar, sum_mask = argmax_g_old(r, sigma2, mu)    # shape (r, nstar) , (r, nstar)

    # Mask g_bar to prevent overflow values from affecting probability calculations
    g_bar = torch.where(sum_mask, g_bar, torch.tensor(0., device=g_bar.device)) # QUESTION : will this break gradients?

    exp_g_bar = torch.exp(g_bar)                          # shape (r, nstar)

    # To calculate log(r!) we use the fact that the Gamma function of an integer is the factorial of that integer -1
    # G(r+1) = r! and torch. provides exactly the logarithm of this
    # torch.lgamma(r+1) = log( G(r+1) ) = log( r! )
    log_r_fact = torch.lgamma(r+1) # BUG: Should we also check for overflows here?

    # r needs to be a 2D tensor, and so does log_r_fact
    r = r.unsqueeze(1)                                                # shape (r) -> shape (r, 1)
    r = r.repeat(1, sigma2.shape[0])                                  # shape (r, 1) -> shape (r, nstar)
    log_r_fact = log_r_fact.unsqueeze(1)                              # shape (r) -> shape (r, 1) ( equivalent to [:,None])
    log_r_fact = log_r_fact.repeat(1, sigma2.shape[0])                # shape (r, 1) -> shape (r, nstar)

    log_r_fact = torch.where(sum_mask, log_r_fact, torch.tensor(0., device=log_r_fact.device))  # shape (r, nstar)
    r_masked          = torch.where(sum_mask, r, torch.tensor(0., device=r.device))           # shape (r, nstar)

    # We woulnd't need to unsqueeze / add an empty first dimension to sigma2. It's just to show that we are dividing each column of g_bar by the corresponding sigma2
    log_p = g_bar*r_masked - exp_g_bar - ((g_bar-mu)**2)/(2*sigma2.unsqueeze(0)) - 0.5*GP_utils.safe_log( exp_g_bar*sigma2 + 1) - log_r_fact # TODO: is this factorial too slow?

    return torch.exp(log_p), log_p, r_masked, log_r_fact

def laplace_approximations_new(mu, sigma2, r):
    """
    Computes p(r|x,D) using Laplace approximation (Eq. 31 Paper PNAS).

    For samples with very small variance (sigma2 < 1e-6), falls back to
    exact Poisson log-likelihood to avoid numerical instability from
    division by near-zero sigma2.

    Mathematical justification: As sigma2 -> 0, the Laplace approximation
    reduces to exact Poisson. The mode g_bar -> mu, and the correction terms
    (g_bar - mu)^2/(2*sigma2) and 0.5*log(1 + sigma2*exp(g_bar)) both -> 0.

    Args:
        mu: (N,) mean of log-firing rate g = A*lambda + lambda0
        sigma2: (N,) variance of log-firing rate
        r: (R,) spike count values from 0 to r_cutoff

    Returns:
        p_r: (N, R) probabilities p(r|x,D)
        log_p_r: (N, R) log probabilities
    """
    N = mu.shape[0]
    R = r.shape[0]
    log_r_fact = torch.lgamma(r + 1)  # (R,)

    # Element-wise check for small variance
    small_var_mask = sigma2 < 1.e-6  # (N,) boolean

    # Initialize output
    log_p = torch.empty(N, R, device=mu.device, dtype=mu.dtype)

    # Get indices for each case
    small_idx = small_var_mask.nonzero(as_tuple=True)[0]
    normal_idx = (~small_var_mask).nonzero(as_tuple=True)[0]

    # Compute exact Poisson ONLY for small variance samples
    if small_idx.numel() > 0:
        mu_small = mu[small_idx]
        g_bar_small = mu_small[:, None].expand(-1, R)
        log_p[small_idx] = g_bar_small * r - torch.exp(g_bar_small) - log_r_fact

    # Compute Laplace ONLY for normal variance samples
    if normal_idx.numel() > 0:
        mu_normal = mu[normal_idx]
        sigma2_normal = sigma2[normal_idx]
        g_bar = argmax_g(r, sigma2_normal, mu_normal)
        exp_g_bar = torch.exp(g_bar)
        log_p[normal_idx] = g_bar * r \
                            - exp_g_bar \
                            - ((g_bar - mu_normal[:, None])**2) / (2 * sigma2_normal[:, None]) \
                            - 0.5 * GP_utils.safe_log(1 + sigma2_normal[:, None] * exp_g_bar) \
                            - log_r_fact

    return torch.exp(log_p), log_p

def nd_mean_noise_entropy(p_response, log_r2d_fact, sigma2, mu ):
    # Computes the conditional noise entropy < H( r|f,x ) >_p(f|D) [eq 33 Paper PNAS]
    # INPUTS:
    # sigma2: 
    #     - [nstar]
    # mu:
    #     - [nstar]
    # p_response: set of probabilities p(r|x,D) for r that goes from 0 to a low number, set in utility(). Should go to infninty but the mean responses are low
    #     - [r, nstar]
    # log_r2d_fact: log(r!) argument of the sum, remember gamma(r+1) = r!
    #     - [r, nstar]   
    # In the nstar=1 case it was:  p_times_logr_sum = p_response@torch.lgamma(r+1) shape (1)
    
    p_times_logr_sum = torch.sum( p_response*log_r2d_fact, dim=0 ) # shape (nstar)

    # TODO: check this formula. In the paper
    H_mean = -torch.exp(mu + 0.5*sigma2)*(mu + sigma2 - 1) + p_times_logr_sum

    return H_mean

def nd_utility_old( mu, sigma2, r_masked):
    # Computes the utility function [eq 27 Paper PNAS]

    # mu, sigma2 are moments of log(f) with f(x) = exp(g(x)) = exp( A*lambda(x) + lambda_0 ) firing rate
    # They are not the mean and variance of the lambda that we learn with the GP.
    # And they are not even the log of the mean and variance of firing rate. ( look at (37) notes)
    # sigma2: shape (nstar)
    # mu:     shape (nstar)

    # Generates the tensor of the probability of the responses p(r|x,D)
    # r_cutoff: for r that goes from 0 to a low number set by r_cutoff, 
    # r_tensor_masked: this will as well be reduced if the exponential in the lambda_r_mean function goes to infinity

    if sigma2.ndim == 0:
        sigma2 = sigma2[None]
        mu     = mu[None]
    # Returns the the Laplace approximations of p(r|x,D) for many different r values. 
    # Also the masked r tensor to know which r where used for which inputs ( nstar )
    conditioned_p_r, log_conditioned_p_r, r_masked_2d, log_r2d_fact_masked = laplace_approximations_old( r=r_masked, sigma2=sigma2, mu=mu, )  # shape (r, nstar), shape (r, nstar), shape (r, nstar)

    
    # Response entropy H(r|x_star,D) [eq 28 Paper PNAS]
    H_r = -torch.sum( conditioned_p_r*log_conditioned_p_r, dim=0 )  # shape (1, nstar)

    E_H_r_f = nd_mean_noise_entropy( conditioned_p_r, log_r2d_fact_masked, sigma2, mu,  )
    U = H_r - E_H_r_f


    # print(f'Sigma2 = {sigma2.item():.4f}, H(r|x,D) = {a.item():.4f}, <H(r|f,x)> = {b.item():.4f}, U = {U.item():.4f}')
    return U


def nd_utility_MC(mu, sigma2, r_max=500, n_samples=5000):
    """
    Compute utility using Monte Carlo sampling (no Laplace approximation).

    This computes the TRUE information gain by replacing the Laplace approximation
    with Monte Carlo estimates of the required integrals:

        U = H_marg - H_cond

    where:
        H_marg = H(R | D) = -Σ_r p_true(r) log p_true(r)
            p_true(r) = E_λ[Poisson(r|exp(λ))] estimated via MC sampling

        H_cond = E_λ[H(Poisson(exp(λ)))]
               = exp(μ+σ²/2)(1-μ-σ²) + E_λ[Σ_r Poisson(r|exp(λ)) log(r!)]
            First term is EXACT (via MGF), second term is estimated via MC

    Key difference from nd_utility():
        - nd_utility uses p_Lap(r) from Laplace approximation
        - This function uses p_true(r) from Monte Carlo sampling
        - The difference measures the Laplace approximation error

    Args:
        mu: (N,) posterior mean of log-firing rate at query points
        sigma2: (N,) posterior variance at query points
        r_max: maximum spike count (truncation)
        n_samples: number of MC samples per query point

    Returns:
        utility: (N,) utility values at each query point
    """
    if mu.ndim == 0:
        mu = mu[None]
        sigma2 = sigma2[None]

    N = mu.shape[0]
    device = mu.device
    dtype = mu.dtype

    r_values = torch.arange(0, r_max, dtype=dtype, device=device)  # (R,)
    log_r_fact = torch.lgamma(r_values + 1)  # log(r!) for r = 0, 1, ..., R-1

    # =========================================================================
    # Step 1: Sample λ from the posterior N(μ, σ²)
    # =========================================================================
    # Shape: (N, n_samples)
    eps = torch.randn(N, n_samples, dtype=dtype, device=device)
    lambda_samples = mu[:, None] + torch.sqrt(sigma2[:, None]) * eps

    # =========================================================================
    # Step 2: Compute Poisson(r | exp(λ)) for all samples and r values
    # =========================================================================
    # log Poisson(r|f) = r·log(f) - f - log(r!)
    #                  = r·λ - exp(λ) - log(r!)   since f = exp(λ)
    # Shape: (N, n_samples, R)
    f_samples = torch.exp(lambda_samples)  # firing rates (N, n_samples)
    log_poisson = (lambda_samples[:, :, None] * r_values[None, None, :]
                   - f_samples[:, :, None]
                   - log_r_fact[None, None, :])
    poisson_probs = torch.exp(log_poisson)  # (N, n_samples, R)

    # =========================================================================
    # Step 3: H_marg = -Σ p_true(r) log p_true(r)
    # =========================================================================
    # Estimate p_true(r) = E_λ[Poisson(r|exp(λ))] by averaging over samples
    p_true = poisson_probs.mean(dim=1)  # (N, R)

    # Compute entropy of the marginal distribution
    # For r values far from the mean, p_true(r) ≈ 0
    # By L'Hôpital's rule: lim_{p→0} p·log(p) = 0
    # So we use the convention 0·log(0) = 0 in entropy calculations
    # Implementation: use torch.xlogy(p, p) which computes p*log(p) with xlogy(0,0)=0
    H_marg = -torch.sum(torch.xlogy(p_true, p_true), dim=1)  # (N,)

    # =========================================================================
    # Step 4: H_cond = E[exp(λ)(1-λ)] + E[Σ_r Poisson(r|exp(λ)) log(r!)]
    # =========================================================================
    # Term 1: EXACT via moment generating function of log-normal
    # E[exp(λ)(1-λ)] = E[exp(λ)] - E[λ·exp(λ)]
    #                = exp(μ+σ²/2) - (μ+σ²)·exp(μ+σ²/2)
    #                = exp(μ+σ²/2)·(1 - μ - σ²)
    # No Monte Carlo needed here — this is analytically exact.
    term1 = torch.exp(mu + 0.5 * sigma2) * (1 - mu - sigma2)  # (N,)

    # Term 2: MC estimate of E_λ[Σ_r Poisson(r|exp(λ)) log(r!)]
    # For each sample j: compute Σ_r Poisson(r|exp(λ_j)) log(r!)
    E_log_r_fact_per_sample = torch.sum(
        poisson_probs * log_r_fact[None, None, :], dim=2
    )  # (N, n_samples)
    # Average over samples
    term2 = E_log_r_fact_per_sample.mean(dim=1)  # (N,)

    H_cond = term1 + term2  # (N,)

    # =========================================================================
    # Step 5: Utility = H_marg - H_cond
    # =========================================================================
    utility = H_marg - H_cond

    return utility


def nd_utility_MC_batched(mu, sigma2, r_max=500, n_samples=100000, batch_size=10000):
    """
    Compute utility using Monte Carlo sampling with batched processing.

    This is a memory-efficient version of nd_utility_MC that processes samples
    in fixed-size batches and accumulates running means. This allows for very
    large sample counts (100k-1M+) without exhausting GPU memory.

    Memory usage:
        - nd_utility_MC: O(N × n_samples × R) - ~3.6GB for N=200, n_samples=45k, R=100
        - nd_utility_MC_batched: O(N × batch_size × R) - ~800MB for batch_size=10k (constant)

    Mathematical formula:
        U = H_marg - H_cond

    where:
        H_marg = H(R | D) = -Σ_r p_true(r) log p_true(r)
            p_true(r) = E_λ[Poisson(r|exp(λ))] estimated via MC sampling

        H_cond = E_λ[H(Poisson(exp(λ)))]
               = exp(μ+σ²/2)(1-μ-σ²) + E_λ[Σ_r Poisson(r|exp(λ)) log(r!)]
            First term is EXACT (via MGF), second term is estimated via MC

    Key insight for batching:
        Both quantities can be computed as running averages:
        - p_true(r) = (1/n_batches) Σ_b [batch_mean of Poisson probs]
        - E[Σ_r P(r)log(r!)] = (1/n_batches) Σ_b [batch_mean of weighted sum]

    Args:
        mu: (N,) posterior mean of log-firing rate at query points
        sigma2: (N,) posterior variance at query points
        r_max: maximum spike count (truncation)
        n_samples: total number of MC samples per query point
        batch_size: number of samples to process at once (memory-speed tradeoff)

    Returns:
        utility: (N,) utility values at each query point
    """
    if mu.ndim == 0:
        mu = mu[None]
        sigma2 = sigma2[None]

    N = mu.shape[0]
    device = mu.device
    dtype = mu.dtype

    n_batches = n_samples // batch_size
    if n_batches == 0:
        # Fall back to non-batched version for small sample counts
        return nd_utility_MC(mu, sigma2, r_max=r_max, n_samples=n_samples)

    # Precompute constants
    r_values = torch.arange(0, r_max, dtype=dtype, device=device)  # (R,)
    log_r_fact = torch.lgamma(r_values + 1)  # log(r!) for r = 0, 1, ..., R-1
    sigma = torch.sqrt(sigma2)  # (N,)

    # Accumulators for running means
    p_true_sum = torch.zeros(N, r_max, dtype=dtype, device=device)  # (N, R)
    E_log_r_fact_sum = torch.zeros(N, dtype=dtype, device=device)  # (N,)

    # Process samples in batches
    for _ in range(n_batches):
        # Sample λ from the posterior N(μ, σ²)
        eps = torch.randn(N, batch_size, dtype=dtype, device=device)
        lambda_samples = mu[:, None] + sigma[:, None] * eps  # (N, batch_size)
        f_samples = torch.exp(lambda_samples)  # (N, batch_size)

        # Compute Poisson(r | exp(λ)) for all samples and r values
        # log Poisson(r|f) = r·λ - exp(λ) - log(r!)
        # Shape: (N, batch_size, R)
        log_poisson = (lambda_samples[:, :, None] * r_values[None, None, :]
                       - f_samples[:, :, None]
                       - log_r_fact[None, None, :])
        poisson_probs = torch.exp(log_poisson)  # (N, batch_size, R)

        # Accumulate batch means for p_true
        p_true_sum += poisson_probs.mean(dim=1)  # (N, R)

        # Accumulate batch means for E[Σ_r P(r)log(r!)]
        E_log_r_fact_sum += (poisson_probs * log_r_fact[None, None, :]).sum(dim=2).mean(dim=1)  # (N,)

    # Average over batches to get final estimates
    p_true = p_true_sum / n_batches  # (N, R)
    E_log_r_fact = E_log_r_fact_sum / n_batches  # (N,)

    # Compute H_marg = -Σ p_true(r) log p_true(r)
    H_marg = -torch.sum(torch.xlogy(p_true, p_true), dim=1)  # (N,)

    # Compute H_cond = E[exp(λ)(1-λ)] + E[Σ_r Poisson(r|exp(λ)) log(r!)]
    # Term 1: EXACT via moment generating function
    term1 = torch.exp(mu + 0.5 * sigma2) * (1 - mu - sigma2)  # (N,)

    H_cond = term1 + E_log_r_fact  # (N,)

    # Utility = H_marg - H_cond
    utility = H_marg - H_cond

    return utility


def nd_utility_hybrid(mu, sigma2, r_max=500, n_samples=10000, batch_size=500):
    """
    Compute utility with Laplace approximation for H_marg and MC sampling for H_cond.

    This hybrid approach combines the numerical stability of Laplace for the marginal
    entropy with the accuracy of Monte Carlo for the conditional entropy.

    Mathematical formula:
        U = H_marg - H_cond

    where:
        H_marg = -Σ_r p_Laplace(r) log p_Laplace(r)
            Computed via laplace_approximations_new (numerically stable)

        H_cond = E_λ[H[r|λ]] = E_λ[-Σ_r p(r|exp(λ)) log p(r|exp(λ))]
            Computed via MC sampling (exact Poisson entropy per sample)

    Why this works:
        - Laplace approximation is accurate for H_marg (properly normalized p(r))
        - For each MC sample λ, the Poisson entropy is computed EXACTLY
        - MC averaging gives unbiased estimate of E[H_cond]

    Args:
        mu: (N,) posterior mean of log-firing rate at query points
        sigma2: (N,) posterior variance at query points
        r_max: maximum spike count (truncation)
        n_samples: number of MC samples for H_cond estimation
        batch_size: samples per batch (memory-speed tradeoff)

    Returns:
        utility: (N,) utility values at each query point
    """
    if mu.ndim == 0:
        mu = mu[None]
        sigma2 = sigma2[None]

    N = mu.shape[0]
    device = mu.device
    dtype = mu.dtype
    sigma = torch.sqrt(sigma2)

    # Precompute constants
    r_values = torch.arange(0, r_max, dtype=dtype, device=device)  # (R,)
    log_r_fact = torch.lgamma(r_values + 1)  # log(r!) for r = 0, 1, ..., R-1

    # =========================================================================
    # H_marg: Laplace approximation (numerically stable)
    # =========================================================================
    p_r_laplace, log_p_r_laplace = laplace_approximations_new(mu, sigma2, r_values)
    H_marg = -torch.sum(torch.xlogy(p_r_laplace, p_r_laplace), dim=1)  # (N,)

    # =========================================================================
    # H_cond: Monte Carlo estimate of E_λ[H[r|λ]]
    # For each sampled λ, compute exact Poisson entropy, then average
    # =========================================================================
    n_batches = (n_samples + batch_size - 1) // batch_size  # ceiling division
    H_cond_sum = torch.zeros(N, dtype=dtype, device=device)
    total_samples = 0

    for _ in range(n_batches):
        # Determine actual batch size (last batch may be smaller)
        current_batch = min(batch_size, n_samples - total_samples)
        total_samples += current_batch

        # Sample λ from posterior N(μ, σ²)
        eps = torch.randn(N, current_batch, dtype=dtype, device=device)
        lambda_samples = mu[:, None] + sigma[:, None] * eps  # (N, batch)

        # Compute f = exp(λ) for Poisson rate
        f_samples = torch.exp(lambda_samples)  # (N, batch)

        # Compute log p(r|f) = r*log(f) - f - log(r!) for all r
        # Shape: (N, batch, R)
        log_p_r_given_f = (
            r_values[None, None, :] * lambda_samples[:, :, None]
            - f_samples[:, :, None]
            - log_r_fact[None, None, :]
        )

        # Convert to probabilities
        p_r_given_f = torch.exp(log_p_r_given_f)  # (N, batch, R)

        # Compute exact Poisson entropy for each sample: H = -Σ_r p(r) log p(r)
        # Using xlogy for numerical stability (handles p=0 correctly)
        H_batch = -torch.sum(torch.xlogy(p_r_given_f, p_r_given_f), dim=2)  # (N, batch)

        # Accumulate sum across samples
        H_cond_sum += H_batch.sum(dim=1)  # (N,)

        # Free memory
        del eps, lambda_samples, f_samples, log_p_r_given_f, p_r_given_f, H_batch

    # Average to get E[H_cond]
    H_cond = H_cond_sum / total_samples  # (N,)

    # =========================================================================
    # Utility = H_marg - H_cond
    # =========================================================================
    return H_marg - H_cond


def nd_utility_NUMERICAL(mu, sigma2, r_max=500, n_quadrature=100):
    """
    Compute utility using Gauss-Hermite numerical integration (no Laplace approximation).

    This computes the TRUE information gain by evaluating the integral:
        p(r|D) = ∫ Poisson(r|exp(g)) · N(g|μ,σ²) dg

    using Gauss-Hermite quadrature, which is optimal for integrals against Gaussian.

    The substitution z = (g - μ) / (σ√2) transforms the integral to:
        p(r|D) = (1/√π) ∫ Poisson(r|exp(μ + σ√2·z)) · exp(-z²) dz

    This is exact Gauss-Hermite form with weight function exp(-z²).

    Key difference from nd_utility():
        - nd_utility uses Laplace approximation (fails when Σp(r) ≠ 1)
        - This function uses numerical integration (guaranteed normalized if enough points)

    Args:
        mu: (N,) posterior mean of log-firing rate at query points
        sigma2: (N,) posterior variance at query points
        r_max: maximum spike count (truncation)
        n_quadrature: number of Gauss-Hermite quadrature points (default 100)

    Returns:
        utility: (N,) utility values at each query point
    """
    if mu.ndim == 0:
        mu = mu[None]
        sigma2 = sigma2[None]

    N = mu.shape[0]
    device = mu.device
    dtype = mu.dtype

    r_values = torch.arange(0, r_max, dtype=dtype, device=device)  # (R,)
    log_r_fact = torch.lgamma(r_values + 1)  # log(r!) for r = 0, 1, ..., R-1
    R = r_values.shape[0]

    # =========================================================================
    # Step 1: Get Gauss-Hermite quadrature nodes and weights
    # =========================================================================
    # numpy.polynomial.hermite.hermgauss gives nodes z_i and weights w_i such that
    # ∫ f(z) exp(-z²) dz ≈ Σ w_i f(z_i)
    z_nodes, weights = np.polynomial.hermite.hermgauss(n_quadrature)
    z_nodes = torch.tensor(z_nodes, dtype=dtype, device=device)  # (Q,)
    weights = torch.tensor(weights, dtype=dtype, device=device)  # (Q,)

    # Our integral is:
    # p(r|D) = (1/√π) ∫ Poisson(r|exp(μ + σ√2·z)) exp(-z²) dz
    # Using quadrature: p(r|D) ≈ (1/√π) Σ w_i Poisson(r|exp(μ + σ√2·z_i))
    weights_normalized = weights / np.sqrt(np.pi)  # (Q,)

    # =========================================================================
    # Step 2: Compute g = μ + σ√2·z for all query points and quadrature nodes
    # =========================================================================
    sigma = torch.sqrt(sigma2)  # (N,)
    # g[i, j] = μ[i] + σ[i] * √2 * z[j]
    g_values = mu[:, None] + sigma[:, None] * np.sqrt(2) * z_nodes[None, :]  # (N, Q)
    f_values = torch.exp(g_values)  # (N, Q)

    # =========================================================================
    # Step 3: Compute Poisson(r | f) for all (query point, quadrature node, r)
    # =========================================================================
    # log Poisson(r|f) = r·log(f) - f - log(r!) = r·g - exp(g) - log(r!)
    # Shape: (N, Q, R)
    log_poisson = (g_values[:, :, None] * r_values[None, None, :]
                   - f_values[:, :, None]
                   - log_r_fact[None, None, :])
    poisson_probs = torch.exp(log_poisson)  # (N, Q, R)

    # =========================================================================
    # Step 4: Compute p_true(r) = (1/√π) Σ w_i Poisson(r|exp(g_i))
    # =========================================================================
    # Weighted sum over quadrature nodes
    # weights_normalized: (Q,), poisson_probs: (N, Q, R)
    p_true = torch.einsum('q,nqr->nr', weights_normalized, poisson_probs)  # (N, R)

    # =========================================================================
    # Step 5: H_marg = -Σ p_true(r) log p_true(r)
    # =========================================================================
    # Use torch.xlogy for numerical stability (handles 0*log(0)=0)
    H_marg = -torch.sum(torch.xlogy(p_true, p_true), dim=1)  # (N,)

    # =========================================================================
    # Step 6: H_cond = E[exp(λ)(1-λ)] + E[Σ_r Poisson(r|exp(λ)) log(r!)]
    # =========================================================================
    # Term 1: EXACT via moment generating function of log-normal
    term1 = torch.exp(mu + 0.5 * sigma2) * (1 - mu - sigma2)  # (N,)

    # Term 2: Numerical integration of E[Σ_r Poisson(r|exp(g)) log(r!)]
    # For each quadrature node, compute Σ_r Poisson(r|f) log(r!)
    # Shape: (N, Q)
    weighted_log_r_fact = torch.sum(poisson_probs * log_r_fact[None, None, :], dim=2)  # (N, Q)
    # Integrate over quadrature nodes
    term2 = torch.einsum('q,nq->n', weights_normalized, weighted_log_r_fact)  # (N,)

    H_cond = term1 + term2  # (N,)

    # =========================================================================
    # Step 7: Utility = H_marg - H_cond
    # =========================================================================
    utility = H_marg - H_cond

    return utility


def nd_utility_new(mu, sigma2, r_max=500):
    """
    Compute utility using the FIXED Laplace approximation (no overflow bug).

    This function is equivalent to nd_utility() but uses laplace_approximations_new()
    instead of laplace_approximations_old(), which fixes the catastrophic normalization
    failure caused by overflow handling.

    Mathematical formula:
        U = H_marg - H_cond

    where:
        H_marg = H(R | D) = -Σ_r p_Lap(r) log p_Lap(r)
            p_Lap(r) from Laplace approximation (Eq. 31 PNAS)

        H_cond = E[H(Poisson(exp(λ)))]
               = exp(μ+σ²/2)(1-μ-σ²) + Σ_r p_Lap(r) log(r!)

    KEY DIFFERENCE from nd_utility():
        - Uses laplace_approximations_new() which works in LOG SPACE
        - Avoids overflow by computing W₀(e^y) without computing e^y
        - No overflow clamping → correct probabilities for large r
        - Σp_Lap(r) ≈ 1.0 even for high uncertainty

    NUMERICAL CONSIDERATIONS:

    1. **Log-space Lambert W** (lines 243-245 in argmax_g):
       - Computes y = log(σ²) + r·σ² + μ (never overflows)
       - Solves w + log(w) = y via Newton iteration
       - No explicit exp() computation → handles arbitrary large r

    2. **Normalization**:
       - OLD (buggy): Σp(r) can be >>1 (e.g., 108 for high uncertainty)
       - NEW (fixed): Σp(r) ≈ 1.0 always (within 1% typically)
       - Remaining error is from:
           a) r_max truncation (use larger r_max if Σp < 0.99)
           b) Numerical precision in Newton iteration (~10⁻⁸)

    3. **Entropy computation** (line H_r):
       - Uses p*log(p) which can be unstable for very small p
       - For p < 1e-30: contribution ≈ 0 (safe to ignore)
       - torch.xlogy() would be more stable but not used here for
         consistency with original nd_utility()

    4. **Potential issues**:
       a) **Small σ² (σ² < 1e-6)**:
          - laplace_approximations_new handles this via fallback to exact Poisson
          - See lines 323-337: uses g_bar = μ when σ² is tiny

       b) **Very large r_max (>10000)**:
          - Memory: (N, r_max) can be large
          - Computation: Newton iteration for each (n, r) pair
          - Solution: Adaptive r_max based on σ² (see note below)

       c) **Negative utility**:
          - Can occur if p_Lap is poorly normalized (Σp ≠ 1)
          - With new implementation, should not happen
          - If U < -0.1, check normalization via p.sum(dim=1)

    5. **Adaptive r_max** (optional improvement):
       - Current: Fixed r_max = 500 for all query points
       - Better: r_max = max(500, int(10 * sqrt(E[f²])))
       - Rationale: For f ~ LogNormal(μ, σ²), need r_max ~ E[f] + 5*σ_f
       - This saves computation for low-uncertainty regions

    Args:
        mu: (N,) posterior mean of log-firing rate at query points
        sigma2: (N,) posterior variance at query points
        r_max: maximum spike count (default 500)
            NOTE: For high uncertainty (σ² > 1), may need r_max > 1000

    Returns:
        utility: (N,) utility values

    Example:
        >>> mu = torch.tensor([0.5, -0.5])
        >>> sigma2 = torch.tensor([0.1, 0.8])
        >>> U = nd_utility_new(mu, sigma2, r_max=500)
        >>> # Check normalization (should be close to 1.0)
        >>> p, _ = laplace_approximations_new(mu, sigma2, torch.arange(0, 500))
        >>> print(f"Normalization: {p.sum(dim=1)}")  # Should be [1.00, 1.00]

    See also:
        - nd_utility(): Original (buggy) version
        - nd_utility_MC(): Monte Carlo ground truth
        - nd_utility_NUMERICAL(): Gauss-Hermite ground truth
        - laplace_approximations_new(): Fixed Laplace approximation
    """
    # Handle scalar inputs
    if sigma2.ndim == 0:
        sigma2 = sigma2[None]
        mu = mu[None]

    N = mu.shape[0]

    # Generate r values
    r = torch.arange(0, r_max + 1, dtype=mu.dtype, device=mu.device)  # (R,)
    R = r.shape[0]

    # =========================================================================
    # Step 1: Compute p(r|x,D) using FIXED Laplace approximation
    # =========================================================================
    # Shape: (N, R)
    p_r, log_p_r = laplace_approximations_new(mu=mu, sigma2=sigma2, r=r)

    # =========================================================================
    # Step 2: Response entropy H(r|x,D) [Eq. 28 PNAS]
    # =========================================================================
    # H_marg = -Σ_r p(r) log p(r)
    # NOTE: Using p*log_p instead of xlogy for consistency with nd_utility()
    #       This can be unstable for very small p, but laplace_approximations_new
    #       ensures p > 0 for all r in [0, r_max]
    H_r = -torch.sum(p_r * log_p_r, dim=1)  # (N,)

    # =========================================================================
    # Step 3: Conditional entropy E[H(r|f,x)] [Eq. 33 PNAS]
    # =========================================================================
    # Need to compute: Σ_r p(r|x,D) log(r!)
    log_r_fact = torch.lgamma(r + 1)  # (R,)

    # Prepare inputs for nd_mean_noise_entropy
    # IMPORTANT: nd_mean_noise_entropy expects shapes (R, N), not (N, R)
    p_r_transposed = p_r.T  # (R, N)
    log_r_fact_2d = log_r_fact[:, None].expand(R, N)  # (R, N)

    # Compute E[H(r|f,x)] = -exp(μ+σ²/2)(μ+σ²-1) + Σ_r p(r) log(r!)
    E_H_r_f = nd_mean_noise_entropy(
        p_response=p_r_transposed,
        log_r2d_fact=log_r_fact_2d,
        sigma2=sigma2,
        mu=mu
    )  # (N,)

    # =========================================================================
    # Step 4: Utility = H_marg - H_cond
    # =========================================================================
    utility = H_r - E_H_r_f  # (N,)

    # =========================================================================
    # Optional: Normalization check (can be removed in production)
    # =========================================================================
    # Uncomment for debugging:
    # p_sum = p_r.sum(dim=1)
    # if (p_sum < 0.95).any() or (p_sum > 1.05).any():
    #     print(f"WARNING: Poor normalization detected!")
    #     print(f"  Σp(r) range: [{p_sum.min():.4f}, {p_sum.max():.4f}]")
    #     print(f"  Consider increasing r_max (current: {r_max})")

    return utility


# ======================= Distribution-Aware Utility for GPyTorch =====================
#
# This function implements the corrected utility formula from:
# ~/IDV_code/Papers/latex_summaries/active_learning_pietro_corrected.tex
#
# Key insight: The cross-covariance Sigma_x* includes BOTH:
#   1. Residual correlation: k(x,x*) - u^T @ K @ u_*  (direct kernel correlation)
#   2. Epistemic correlation: u^T @ V @ u_*           (through inducing points)
#
# The original nd_utility only uses the epistemic term, missing the residual.
# =====================================================================================

def distribution_aware_utility_gpytorch_old(
    model,
    x_star,
    r_max=100,
    n_samples=50,
    n_lambda_samples=1,
    batch_size=1000,
    p_x_type="uniform",
    x_min=0.0,
    x_max=1.0,
    gaussian_mean=0.5,
    gaussian_std=0.2,
    A=1.0,
    lambda0=0.0,
    x_samples=None
):
    """
    BUG: This funcion does not unwhiten the m and V saved in gpytorch.


    Compute distribution-aware utility U(x*) for GPyTorch VariationalGP models.

    This implements the corrected acquisition function that accounts for the
    stimulus distribution p(x) when computing the conditional entropy term.

    The utility formula is:
        U(x*) = H_marg(x*) - H_cond(x*)

    where:
        - H_marg(x*) = H(R | x*, D) is the marginal response entropy at query point
        - H_cond(x*) = E_{x ~ p(x)} E_{lambda(x)|D}[ H(R | x*, lambda(x), D) ]
                       is the expected conditional entropy averaging over the
                       stimulus distribution and lambda samples

    Mathematical derivation: See active_learning_pietro_corrected.tex

    Args:
        model: Trained GPyTorch VariationalGP model
               Expects model with:
               - model.variational_strategy.inducing_points: (M,) or (M,1) inducing points
               - model.variational_strategy.variational_distribution.mean: (M,) mean vector m
               - model.variational_strategy.variational_distribution.covariance_matrix: (M,M) cov V
               - model.covar_module: kernel function

        x_star: Query point(s) - tensor of shape (K,) for K query points

        r_max: Maximum spike count for Laplace approximation (default: 100)

        n_samples: Number of Monte Carlo samples from p(x) for H_cond (default: 50)
                   Ignored if x_samples is provided.

        n_lambda_samples: Number of lambda samples per x_sample (default: 1).
                          Use higher values (e.g., 5000+) for Dirac delta case.

        batch_size: Number of lambda samples to process at once (default: 1000).
                    Used for memory-efficient batched MC in Dirac delta case.
                    Allows n_lambda_samples=100k+ without exhausting GPU memory.

        p_x_type: Distribution type for p(x), either "uniform" or "gaussian"
                  Ignored if x_samples is provided.

        x_min, x_max: Domain bounds for uniform distribution (default: [0, 1])

        gaussian_mean, gaussian_std: Parameters for Gaussian p(x) if p_x_type="gaussian"

        A, lambda0: Firing rate parameters: f(x) = exp(A * lambda(x) + lambda0)
                    In pure GP case (no scaling), set A=1, lambda0=0

        x_samples: Optional tensor of shape (N,) with pre-specified x samples.
                   If provided, these are used directly instead of sampling from p(x).
                   For Dirac delta test, set x_samples=x_star to verify collapse to nd_utility.

    Returns:
        utility: (K,) tensor of utility values at each query point in x_star
    """
    # Ensure x_star is 1D tensor
    if x_star.ndim == 0:
        x_star = x_star.unsqueeze(0)  # scalar -> (1,)

    device = x_star.device
    dtype = x_star.dtype

    # Create r range for Laplace approximation
    r_values = torch.arange(0, r_max, dtype=dtype, device=device)

    # =========================================================================
    # STEP 1: Extract variational parameters from GPyTorch model
    # =========================================================================
    # GPyTorch stores variational posterior q(u) = N(m, V) over inducing point values
    # where u = [lambda(z_1), ..., lambda(z_M)] for M inducing points

    # Inducing points z: shape (M,) or (M, 1)
    z = model.variational_strategy.inducing_points
    if z.ndim == 2 and z.shape[1] == 1:
        z = z.squeeze(-1)  # (M, 1) -> (M,)
    M = z.shape[0]  # number of inducing points

    # Variational covariance V: shape (M, M)
    # (We don't need the variational mean m directly since we use model() for marginal moments)
    V = model.variational_strategy.variational_distribution.covariance_matrix

    # Compute inducing point kernel matrix K_zz: shape (M, M)
    # This is the prior covariance between inducing points
    K_zz = model.covar_module(z, z).evaluate()

    # Compute inverse of K_zz for projections
    # u = K_zz^{-1} @ k(x, z) is the projection vector
    K_zz_inv = torch.linalg.inv(K_zz + 1e-6 * torch.eye(M, device=device, dtype=dtype))

    # =========================================================================
    # STEP 2: Compute marginal entropy H_marg(x*) at each query point
    # =========================================================================
    # Use model(x_star) to get the posterior predictive distribution directly
    # This matches exactly what nd_utility does via evaluate_utility()

    K = len(x_star)  # number of query points

    # Get posterior distribution at query points from GPyTorch
    posterior_star = model(x_star)
    mu_star = posterior_star.mean  # (K,)
    Sigma_star_star = posterior_star.variance  # (K,)

    # Convert to log-firing rate moments: g = A*lambda + lambda0
    logf_mean_marg = A * mu_star + lambda0  # (K,)
    logf_var_marg = A**2 * Sigma_star_star  # (K,)

    # Compute marginal response probabilities using Laplace approximation
    p_r_marg, log_p_r_marg = laplace_approximations_new(
        mu=logf_mean_marg, sigma2=logf_var_marg, r=r_values
    )  # shapes: (K, r_max)

    # Marginal entropy: H_marg = -sum_r p(r) * log(p(r))
    H_marg = -torch.sum(p_r_marg * log_p_r_marg, dim=1)  # (K,)

    # =========================================================================
    # STEP 3: Sample x_i from p(x) and lambda(x_i) from posterior
    # =========================================================================

    # Check if x_samples provided directly (for Dirac delta case)
    if x_samples is not None:
        # Use provided x_samples directly
        x_samples_used = x_samples.to(device=device, dtype=dtype)
        if x_samples_used.ndim == 0:
            x_samples_used = x_samples_used.unsqueeze(0)
        N = len(x_samples_used)
        # Check if this is the Dirac case (x_samples == x_star)
        is_dirac_case = (N == K) and torch.allclose(x_samples_used, x_star, atol=1e-6)
    else:
        # Sample from p(x)
        if p_x_type == "uniform":
            # Uniform distribution on [x_min, x_max]
            x_samples_used = torch.rand(n_samples, dtype=dtype, device=device) * (x_max - x_min) + x_min
        elif p_x_type == "gaussian":
            # Gaussian distribution, truncated to [x_min, x_max]
            # Use rejection sampling for simplicity
            samples = []
            while len(samples) < n_samples:
                s = torch.randn(n_samples * 2, dtype=dtype, device=device) * gaussian_std + gaussian_mean
                valid = (s >= x_min) & (s <= x_max)
                samples.extend(s[valid].tolist())
            x_samples_used = torch.tensor(samples[:n_samples], dtype=dtype, device=device)
        else:
            raise ValueError(f"Unknown p_x_type: {p_x_type}. Use 'uniform' or 'gaussian'.")
        N = n_samples
        is_dirac_case = False

    # Get posterior moments at sample points using model() for consistency
    posterior_i = model(x_samples_used)
    mu_i = posterior_i.mean  # (N,)
    Sigma_ii = posterior_i.variance  # (N,)
    Sigma_ii = torch.clamp(Sigma_ii, min=1e-8)

    # =========================================================================
    # STEP 4 & 5: Compute H_cond - branching based on Dirac vs general case
    # =========================================================================
    # For Dirac case, we skip cross-covariance computation and use batched MC.
    # For general case, we compute full conditional moments.

    if is_dirac_case:
        # =====================================================================
        # DIRAC DELTA CASE: x_samples = x_star (with batched MC sampling)
        # =====================================================================
        # For query k, only use sample k (where x_sample[k] = x_star[k])
        # When conditioning on the same point, sigma2_cond = 0 exactly.
        # mu_cond = lambda_sample (the sampled value itself)
        #
        # We bypass the cross-covariance calculation here because the variational
        # approximation doesn't guarantee Sigma_i_star[k,k] = Sigma_star_star[k].
        # Instead, we directly use the lambda samples as the conditional mean.
        #
        # Uses batched MC sampling for memory efficiency with large n_lambda_samples.

        L = n_lambda_samples
        n_batches = L // batch_size
        if n_batches == 0:
            n_batches = 1
            actual_batch_size = L
        else:
            actual_batch_size = batch_size

        # Precompute constants
        sigma_i = torch.sqrt(Sigma_ii)  # (K,) - std dev at each query point
        logf_var_dirac = torch.full((K,), 1e-8, dtype=dtype, device=device)  # (K,)

        # Accumulator for running mean of H_cond
        H_cond_sum = torch.zeros(K, dtype=dtype, device=device)

        for _ in range(n_batches):
            # Sample batch of lambda values: shape (K, batch_size)
            eps = torch.randn(K, actual_batch_size, dtype=dtype, device=device)
            lambda_batch = mu_i[:, None] + sigma_i[:, None] * eps  # (K, batch_size)

            # Convert to log-firing rate
            logf_mean_batch = A * lambda_batch + lambda0  # (K, batch_size)

            # Compute H_cond for each sample in batch
            H_batch_sum = torch.zeros(K, dtype=dtype, device=device)

            for b in range(actual_batch_size):
                # Compute Poisson entropy at exp(g) where g = A*lambda + lambda0
                p_r_cond, log_p_r_cond = laplace_approximations_new(
                    mu=logf_mean_batch[:, b],  # (K,)
                    sigma2=logf_var_dirac,  # (K,) all 1e-8
                    r=r_values
                )  # shapes: (K, r_max)

                H_batch_sum += -torch.sum(p_r_cond * log_p_r_cond, dim=1)  # (K,)

            # Add batch mean to accumulator
            H_cond_sum += H_batch_sum / actual_batch_size

        # Average over batches
        H_cond = H_cond_sum / n_batches  # (K,)
    else:
        # =====================================================================
        # GENERAL CASE: Full cross-covariance computation
        # =====================================================================
        # Sample lambda(x_i) from posterior: lambda_i ~ N(mu_i, Sigma_ii)
        # Shape: (N, L) where L = n_lambda_samples
        L = n_lambda_samples
        eps = torch.randn(N, L, dtype=dtype, device=device)
        lambda_samples = mu_i[:, None] + torch.sqrt(Sigma_ii[:, None]) * eps  # (N, L)

        # Compute kernel matrices needed for cross-covariance
        # k(x*, z): kernel from query points to inducing points
        k_star_z = model.covar_module(x_star, z).evaluate()  # (K, M)
        # k(x_i, z): kernel from sample points to inducing points
        k_i_z = model.covar_module(x_samples_used, z).evaluate()  # (N, M)
        # k(x_i, x*): direct kernel between sample and query points
        k_i_star = model.covar_module(x_samples_used, x_star).evaluate()  # (N, K)

        # Projection vectors: u = K_zz^{-1} @ k(z, x)
        u_star = K_zz_inv @ k_star_z.T  # (M, K)
        u_i = K_zz_inv @ k_i_z.T  # (M, N)

        # CROSS-COVARIANCE
        # Sigma_i* = k(x_i, x*) + u_i^T @ (V - K_zz) @ u_*
        epistemic_cov = u_i.T @ V @ u_star  # (N, K)
        prior_cov_via_inducing = u_i.T @ K_zz @ u_star  # (N, K)
        residual_cov = k_i_star - prior_cov_via_inducing  # (N, K)
        Sigma_i_star = residual_cov + epistemic_cov  # (N, K)

        # Numerical stability: Clamp cross-covariance
        max_abs_cov = torch.sqrt(Sigma_ii[:, None] * Sigma_star_star[None, :]) * 0.999
        Sigma_i_star = torch.clamp(Sigma_i_star, min=-max_abs_cov, max=max_abs_cov)

        # Conditional moments via Gaussian conditioning
        innovation = lambda_samples - mu_i[:, None]  # (N, L)
        regression_coef = Sigma_i_star / Sigma_ii[:, None]  # (N, K)
        mu_cond = mu_star[None, None, :] + regression_coef[:, None, :] * innovation[:, :, None]  # (N, L, K)

        sigma2_cond = Sigma_star_star[None, :] - Sigma_i_star**2 / Sigma_ii[:, None]  # (N, K)
        sigma2_cond = torch.clamp(sigma2_cond, min=1e-8)

        # Convert to log-firing rate moments
        logf_mean_cond = A * mu_cond + lambda0  # (N, L, K)
        logf_var_cond = A**2 * sigma2_cond  # (N, K)

        # Compute conditional entropy for each (sample, query) pair
        H_cond_samples = torch.zeros(N, L, K, dtype=dtype, device=device)

        for n in range(N):
            for l in range(L):
                # Get p(r | x*, lambda(x_n), D) using Laplace approximation
                p_r_cond, log_p_r_cond = laplace_approximations_new(
                    mu=logf_mean_cond[n, l],  # (K,)
                    sigma2=logf_var_cond[n],  # (K,)
                    r=r_values
                )  # shapes: (K, r_max)

                # Conditional entropy for sample (n, l)
                H_cond_samples[n, l] = -torch.sum(p_r_cond * log_p_r_cond, dim=1)  # (K,)

        # Average conditional entropy over all samples: H_cond = E[H(R | x*, lambda(x), D)]
        H_cond = H_cond_samples.mean(dim=(0, 1))  # (K,)

    # =========================================================================
    # STEP 6: Final utility = H_marg - H_cond
    # =========================================================================
    utility = H_marg - H_cond  # (K,)

    return utility


def compute_utility_single_image(model_active, x_image, max_r_cap=100, return_logf_moments=False): # TO REMOVE
    """
    Compute utility for a single image with gradient support.

    This function is similar to most_useful_remaining_idx but operates on a single
    image tensor (which can have requires_grad=True for optimization).

    IMPORTANT: Uses torch.enable_grad() to override the global torch.set_grad_enabled(False)
    at line 2 of utils.py, allowing PyTorch autograd to build computational graphs.

    Parameters:
    -----------
    model_active : GPModel
        The current active learning GP model
    x_image : torch.Tensor
        Single image tensor of shape (nx,) where nx = n_px_side^2
        Can have requires_grad=True for gradient-based optimization
    max_r_cap : int
        Maximum spike count for utility computation (default: 100)

    Returns:
    --------
    utility : torch.Tensor
        Scalar utility value U(x) with gradient support
    """
    # CRITICAL: Enable gradients locally (utils.py has torch.set_grad_enabled(False) at module level)
    with torch.enable_grad():
        # Extract model parameters
        mask = model_active.mask
        theta = model_active.theta
        kernfun = model_active.kernfun
        xtilde = model_active.xtilde

        C = model_active.C
        B = model_active.B
        K_tilde_b = model_active.K_tilde_b
        K_tilde_inv_b = model_active.K_tilde_inv_b

        m_b = model_active.m_b
        V_b = model_active.V_b
        A = torch.exp(model_active.f_params['logA'])
        lambda0 = model_active.f_params['lambda0']

        # Compute kernel values for this image
        # x_image has shape (nx,), we need to index with mask and add batch dimension
        x_masked = x_image[mask].unsqueeze(0)  # Shape: (1, n_masked_pixels)

        Kvec = kernfun(theta, x_masked, x2=None, C=C, dC=None, diag=True)  # Shape: (1,)
        K = kernfun(theta, x_masked, xtilde[:, mask], C=C, dC=None, diag=False)  # Shape: (1, ntilde)
        K_b = K @ B  # Shape: (1, ntilde)
        KKtilde_inv_b = K_b @ K_tilde_inv_b  # Shape: (1, ntilde)

        # Compute lambda moments
        lambda_m, lambda_var = GP_utils.lambda_moments(
            x_masked, K_tilde_b, KKtilde_inv_b, Kvec, K_b, C, m_b, V_b, theta
        )

        # Convert to log-firing rate moments
        logf_mean = A * lambda_m + lambda0  # Shape: (1,)
        logf_var = A**2 * lambda_var  # Shape: (1,)

        # Compute utility
        r_masked = torch.arange(0, max_r_cap, dtype=TORCH_DTYPE, device=DEVICE)
        # U = nd_utility(logf_mean, logf_var, r_masked)  # Shape: (1,)

        U = nd_utility_new(mu=logf_mean, sigma2=logf_var, r_max=max_r_cap)

        if return_logf_moments:
            return U.squeeze(), logf_mean.squeeze(), logf_var.squeeze()  # Return scalar and moments
        else:
            return U.squeeze()  # Return scalar

# ======================= Conditioned utility funciton =====================

def compose_inverse_block_matrix(s, u, Kinv):
    """
    Batched block matrix inverse using Schur complement.

    Args:
        N is batch dimension

        s: (N,) - Schur complement for each N sample 
        u: (N, n) - projection vectors
        Kinv: (n, n) - inverse kernel matrix (shared)

    Returns:
        K_x_b_inv: (N, n+1, n+1) - batched inverse matrices
    """
    N, n = u.shape

    # Top-left: 1/s -> (N, 1, 1)
    top_left = (1/s).reshape(N, 1, 1)

    # Top-right: -u/s -> (N, 1, n)
    top_right = (-u / s[:,None])[:,None]  # (N, 1, n)

    # Bottom-left: -u/s -> (N, n, 1)
    bottom_left = top_right.transpose(-1, -2)  # (N, n, 1)

    # Bottom-right: K_tilde_inv + u @ u.T / s -> (N, n, n)
    # Kinv broadcasts: (n, n) + (N, n, n)
    bottom_right = Kinv[None,:] + torch.einsum('ni,nj->nij', u, u) / s.view(N, 1, 1)

    # Assemble block matrix: (N, n+1, n+1)
    K_x_b_inv = torch.zeros(N, n + 1, n + 1, device=s.device, dtype=s.dtype)
    K_x_b_inv[:, 0:1, 0:1 ]= top_left
    K_x_b_inv[:, 0:1, 1:]  = top_right
    K_x_b_inv[:, 1:,  0:1] = bottom_left
    K_x_b_inv[:, 1:,  1:]  = bottom_right

    return K_x_b_inv

def compute_U_star_direct(k_xi_star_b, u_b, s, K_tilde_inv_b):
    """
    Compute A_x = k_xi_star_b @ K_x^{-1} directly without building full inverse.

    Key insight: K_x^{-1} depends only on x_i (not x_star), so we don't need
    gradients through u_b, s, or K_tilde_inv_b.

    Args:
        k_xi_star_b: (N, n_b+1) - [k(x_i, x_star), k_star_b]  ← NEEDS gradient
        u_b: (N, n_b) - k_i_b @ K_tilde_inv_b.diag()          ← NO gradient needed
        s: (N,) - Schur complement k0_i - k_i_b @ u_b          ← NO gradient needed
        K_tilde_inv_b: (n_b, n_b) - inverse kernel matrix      ← NO gradient needed

    Returns:
        A_x: (N, n_b+1) - the result of k_xi_star_b @ K_x^{-1}

    K_x^{-1} has structure (from Schur complement):
        [[1/s,           -u^T/s        ],
         [-u/s,  K^{-1} + u*u^T/s]]

    Formula derivation:
        A_x[0] = k[0]/s - k[1:] @ u / s = (k[0] - k[1:] @ u) / s
        A_x[1:] = -k[0]*u/s + k[1:] @ K^{-1} + (k[1:] @ u) * u / s
                = k[1:] @ K^{-1} - (k[0] - k[1:] @ u) * u / s
                = k[1:] @ K^{-1} - alpha/s * u
    """
    # Detach quantities that don't need gradients (they don't depend on x_star)
    u_b_detached = u_b.detach()
    s_detached = s.detach()

    # Extract components from k_xi_star_b (which DOES need gradients)
    a = k_xi_star_b[:, 0]   # (N,) - k(x_i, x_star)
    b = k_xi_star_b[:, 1:]  # (N, n_b) - k_star_b expanded

    # Compute alpha = a - b @ u  (gradient flows through a and b)
    b_dot_u = torch.einsum('Nb,Nb->N', b, u_b_detached)  # (N,)
    alpha = a - b_dot_u  # (N,)

    # Compute alpha/s (s is detached, so gradient only flows through alpha)
    alpha_over_s = alpha / s_detached  # (N,)

    # Compute b @ K^{-1} (gradient flows through b)
    b_Kinv = b @ K_tilde_inv_b  # (N, n_b)

    # Build A_x
    A_x_0 = alpha_over_s  # (N,)
    A_x_rest = b_Kinv - alpha_over_s[:, None] * u_b_detached  # (N, n_b)

    A_x = torch.cat([A_x_0[:, None], A_x_rest], dim=1)  # (N, n_b+1)

    return A_x

def conditioned_utility( x_star, model, remaining_imgs, N, r_cutoff=100, DEBUG_SAME_X=False, DEBUG_FIXED_X=False, DEBUG_FIX_LAMBDA_i=True, return_logf_moments=False ):

    # ==== Compute first term: Marginal entropy ====

    # Using moments of lambda(x_star) conditioned on Dataset D

    # Get hyperparameters of the model and the Kernel values related to x_tilde ( as done in the beginning of batch_utility_w_grad)

    # xtilde kernel and params
        # m, V, K_tilde_inv 
    # xstar kernel values
        # K_vec = K(x_star,x_star) ( nstar,1 )
        # K ( = K(x_star, x_tilde)

    # Get mu_star and sigma_2_star, moments of the log firing rate

    # Sum of laplace approximations as in nd utility -> Marginal entropy

    # (We need access to the gradient, do not break it)

    # Ensure x_star is 2D (add batch dim if needed)
    if x_star.ndim == 1:
        x_star = x_star[None, :]  # [n_pixels] → [1, n_pixels]
    elif x_star.ndim != 2:
        raise ValueError(f"x_star must be 1D or 2D, got shape {x_star.shape}")


    r_masked = torch.arange(0, r_cutoff, dtype=TORCH_DTYPE, device=DEVICE)

    kernfun     = model.kernfun
    xtilde      = model.xtilde

    theta = model.theta
    m_b   = model.m_b
    V_b   = model.V_b

    A = torch.exp(model.f_params['logA'])
    lambda0 = model.f_params['lambda0']

    C, mask, K_tilde_b, K_tilde_inv_b, B = GP_utils.get_final_K_vals(model)

    x_star_masked = x_star[:,mask]  # [N, n_masked_pixels]

    k0_star = kernfun(theta,x_star_masked, x2=None, C=C, dC=None, diag=True)

    k_star   = kernfun(theta,x_star_masked, xtilde[:,mask], C=C, dC=None, diag=False)
    k_star_b = k_star @ B
    # k_starKtilde_inv_b = k_star_b @ K_tilde_inv_b
    k_starKtilde_inv_b = k_star_b * K_tilde_inv_b.diag()

    # === Marginal moments of lambda(x_star) | Dataset D ===
    lambda_m, lambda_var = GP_utils.lambda_moments(
       x_star_masked, K_tilde_b, k_starKtilde_inv_b, k0_star, k_star_b, C, m_b, V_b, theta)

    logf_mean = A*lambda_m + lambda0 # [N]
    logf_var  = A**2 * lambda_var    # [N]

    p_r, log_p_r = laplace_approximations_new( \
            mu=logf_mean, sigma2=logf_var, r=r_masked )  # shape (r, nstar), shape (r, nstar), shape (r, nstar)


    # Marginal Entropy  r | x_star, D ( first term )
    H_r = -torch.sum( p_r*log_p_r, dim=1 )  # shape (N)

    # ==== Conditional Entropy, second term ====

    # Sample N random images without replacement
    n_imgs = remaining_imgs.shape[0]

    perm = torch.randperm(n_imgs, device=remaining_imgs.device)[:N]

    x_i = remaining_imgs[perm]  # shape (N, n_pixels)

    if DEBUG_FIXED_X:
        x_i = remaining_imgs[:N]

    if DEBUG_SAME_X and N==1:
        # print(" DEBUG MODE: Using x_i identical to x_star for testing ")
        x_i = x_star.detach().clone().repeat(N, 1)  # shape (N, n_pixels)

    if DEBUG_SAME_X and N!=1:
        raise ValueError(" DEBUG_SAME_X MODE with N!=1 is not allowed, please set N=1 ")

    # check we didnt select any xi == x_star

    if not N==1:
        for j in range(N):
            if torch.all( x_i[j] == x_star ):
                raise ValueError(f" Sampled training image {j} is identical to x_star in conditioned utility ")

    # x_i = remaining_imgs[:N]
    # x_i = x_star.detach().clone().repeat(N, 1)  # shape (N, n_pixels)
    x_i_masked = x_i[:,mask]  # shape (N, n_masked_pixels)

    # Sample lambda(x_i)|x_i, D ~ N(mu_i, sigma_2_i) 
    # Moments of lambda(x_i) conditioned on Dataset D 
    k0_i = kernfun(theta,x_i_masked, x2=None, C=C, dC=None, diag=True)

    k_i   = kernfun(theta,x_i_masked, xtilde[:,mask], C=C, dC=None, diag=False) # [N, n_tilde]
    k_i_b = k_i @ B # (N, n_b)
    # k_iKtilde_inv_b = k_i_b @ K_tilde_inv_b
    k_iKtilde_inv_b = k_i_b * K_tilde_inv_b.diag()

    lambda_m_i, lambda_var_i = GP_utils.lambda_moments(
       x_i_masked, K_tilde_b, k_iKtilde_inv_b, k0_i, k_i_b, C, m_b, V_b, theta)

    # Sample from the Gaussian ( N independent samples with randn_like )
    if DEBUG_FIXED_X or DEBUG_SAME_X or DEBUG_FIX_LAMBDA_i:
        lambda_i_samples = lambda_m_i 
    else:    
        lambda_i_samples = lambda_m_i + torch.sqrt( lambda_var_i )*torch.randn_like(lambda_m_i) 


    # ===== Conditional entropy r|x, lambda(x), D , averaged with samples extracted above ====

    # Get kernel values related to x_i needed for for computing moments of (logf_i, logf_star)
    # notice the K_x^-1 can be obtained using K_tilde ^-1 as reported in latex document

    # k(x_i, x_star) # [N,1]
    k1 = kernfun(theta,x_i_masked,x_star_masked, C=C, dC=None, diag=False)

    # DEBUG: we find that sometimes the gradient of k1 wrt x_star is nan
    # k1_manual, dk1_manual = kernfun(theta, x_i_masked, x_star_masked, C=C, get_dK_x=True)
    # print(f" Gradient k1_manual norm: {dk1_manual[0].norm().item():.6f} ")
    # if DEBUG:
    #     k1_grad = torch.autograd.grad(  k1[1] , x_star_masked, retain_graph=True)[0]
    #     if torch.any( torch.isnan( k1_grad ) ):
    #         # DEBUG : checking gradient is not actually nan if we compute manually the kernel gradient
    #         print(f" Gradient k1 norm: {k1_grad.norm().item():.6f} ")
    #         raise ValueError(f" NaN gradient found in k1 ")


    # for j in range(k_star_b.shape[1]): # loop indicung points
    #     k_star_b_grad = torch.autograd.grad(  k_star_b[0][j] , x_star, retain_graph=True)[0]
    #     if torch.any( torch.isnan( k_star_b_grad ) ):
    #         print(f" Gradient mean k_star_b  norm: {k_star_b_grad.norm().item():.6f} ")
    #         raise ValueError(f" NaN gradient found in k_star_b for basis point {j} ")

    k_xi_star_b = torch.cat( 
        ( k1, k_star_b.expand(N, -1) ), 
        dim=1 )  # shape (N, nb+1)

    # Inverse of K_x on b basis

    # u_b = k_i_b @ K_tilde_inv_b
    u_b = k_i_b * K_tilde_inv_b.diag()
    s = k0_i - torch.einsum( 'Nb,Nb->N', k_i_b, u_b )

    s = torch.clamp( s, min=1e-6 )

    # OLD: Build full (N, n_b+1, n_b+1) inverse matrix - SLOW backward pass!
    # Kinv_x_b = compose_inverse_block_matrix( s=s, u=u_b, Kinv=K_tilde_inv_b )
    # A_x = torch.einsum('Ni, Nij->Nj', k_xi_star_b, Kinv_x_b.detach() )

    # NEW: Compute A_x directly without building full inverse 
    # Key insight: Kinv_x_b doesn't depend on x_star, so we can detach u_b and s
    A_x = compute_U_star_direct(k_xi_star_b, u_b, s, K_tilde_inv_b)

    # Moments of lambda conditioned on Dataset + x_i
    lambda_vec = torch.cat( ( lambda_i_samples[:,None], m_b.expand(N, -1) ), dim=1 )  # shape (N, nb+1)

    lambda_m_star = torch.einsum('Nj,Nj->N', A_x, lambda_vec )

    # print(f"lambda_m_star has NaN: {torch.any(torch.isnan(lambda_m_star))}")

    # === DEBUG: Test corrected V_cond formula === # This was not enough
    DEBUG_VCOND = True
    # DEBUG_VCOND = False
    if DEBUG_VCOND:
        # Correction: V_cond = V - VuuᵀV/(s + uᵀVu)
        # Instead of building V_cond, compute the correction term directly:
        # correction = (A_x[1:] @ V @ u)² / (s + uᵀVu)
        
        c = torch.einsum('Nj,jk,Nk->N', A_x[:,1:], V_b, u_b)  # A_x[1:] @ V @ u
        uVu = torch.einsum('Nj,jk,Nk->N', u_b, V_b, u_b)      # uᵀ @ V @ u
        denom = s + uVu  # = Var[λ(x_i)|D]
        correction = c**2 / denom
        
        lambda_var_star = k0_star - \
            torch.einsum('Nj, Nj -> N', A_x, k_xi_star_b) + \
            torch.einsum('Nj, jk, Nk->N', A_x[:,1:], V_b, A_x[:,1:]) - correction
    else:
        # Original formula (uses V instead of V_cond)
        lambda_var_star = k0_star - \
            torch.einsum('Nj, Nj -> N', A_x, k_xi_star_b) + \
            torch.einsum('Nj, jk, Nk->N', A_x[:,1:], V_b, A_x[:,1:])

    # Sum of laplace approximations as in nd utility -> Sample of conditional entropy
    logf_mean_star = A*lambda_m_star + lambda0 # [N]
    logf_var_star  = A**2 * lambda_var_star    # [N]

    # In case of very small variance it uses exact formula
    p_r_xi, log_p_r_xi = laplace_approximations_new( \
        mu=logf_mean_star, sigma2=logf_var_star, r=r_masked )
    
    # Marginal Entropy  r | x_star, D ( first term )
    H_r_xi_samples = -torch.sum( p_r_xi*log_p_r_xi, dim=1 )  # shape (N)
    # assert not torch.any(torch.isnan(H_r_xi_samples)), "NaN values found in conditional entropy samples"

    H_r_xi = torch.mean( H_r_xi_samples )  # Average over N samples

    # H_r_xi_grad = torch.autograd.grad(  H_r_xi , x_star, retain_graph=True)[0]

    # if torch.any( torch.isnan( H_r_xi_grad ) ):
    #     print(f" Gradient mean H_r_xi  norm: {H_r_xi_grad.norm().item():.6f} ")
    #     raise ValueError(f" NaN gradient found in H_r_xi ")


    # import time
    # torch.autograd.grad(  H_r_xi , x_star, retain_graph=True)
    # torch.cuda.synchronize() 

    # t0 = time.perf_counter()
    # H_r_xi_grad = torch.autograd.grad(  H_r_xi , x_star, retain_graph=True)[0]
    # torch.cuda.synchronize()
    # t1 = time.perf_counter()

    # print(f" Gradient time H_r_xi: {(t1 - t0)*1000:.6f} ms ")
    # print(f" Gradient mean H_r_xi norm: {H_r_xi_grad.norm().item():.6f} ")

    # Z_marg = torch.sum(p_r, dim=1)      # Should be ≈ 1
    # Z_cond = torch.sum(p_r_xi, dim=1)   # Should be ≈ 1
    # print(f"Z_marg: {Z_marg.item():.10f}")
    # print(f"Z_cond: {Z_cond.item():.10f}")
    # print(f"Z difference: {(Z_marg - Z_cond).item():.2e}")

    utility = H_r - H_r_xi

    # === DEBUG: Numerical integration instead of Laplace ===
    # DEBUG_NUMERICAL_ENTROPY = True
    DEBUG_NUMERICAL_ENTROPY = False

    if DEBUG_NUMERICAL_ENTROPY:
        import numpy as np
        
        def entropy_gauss_hermite(mu, sigma2, r_max=200, n_quad=50):
            """
            Compute entropy of Poisson-log-Normal using Gauss-Hermite quadrature.
            
            Args:
                mu: (N,) mean of log-firing rate
                sigma2: (N,) variance of log-firing rate
                r_max: maximum spike count to consider
                n_quad: number of quadrature points
            Returns:
                H: (N,) entropy for each sample
            """
            device = mu.device
            dtype = mu.dtype
            N = mu.shape[0]
            
            # Gauss-Hermite nodes and weights
            nodes_np, weights_np = np.polynomial.hermite.hermgauss(n_quad)
            nodes = torch.tensor(nodes_np, device=device, dtype=dtype)    # (n_quad,)
            weights = torch.tensor(weights_np, device=device, dtype=dtype)  # (n_quad,)
            
            # Transform nodes: g = mu + sqrt(2)*sigma*node
            sigma = torch.sqrt(sigma2)  # (N,)
            g_points = mu[:, None] + np.sqrt(2) * sigma[:, None] * nodes[None, :]  # (N, n_quad)
            
            # Compute rates at quadrature points
            rates = torch.exp(g_points)  # (N, n_quad)
            
            # Spike counts
            r_vals = torch.arange(r_max, device=device, dtype=dtype)  # (r_max,)
            log_r_fact = torch.lgamma(r_vals + 1)  # (r_max,)
            
            # Compute p(r) = (1/sqrt(pi)) * sum_i w_i * Poisson(r | rate_i)
            # log Poisson(r | λ) = r*log(λ) - λ - log(r!)
            # Shape: (N, n_quad, r_max)
            log_poisson = (r_vals[None, None, :] * torch.log(rates[:, :, None] + 1e-40) 
                        - rates[:, :, None] 
                        - log_r_fact[None, None, :])
            poisson_probs = torch.exp(log_poisson)  # (N, n_quad, r_max)
            
            # Weighted sum over quadrature points: (N, r_max)
            # p(r) = (1/sqrt(pi)) * sum_i w_i * Poisson(r | rate_i)
            p_r = torch.einsum('q,Nqr->Nr', weights, poisson_probs) / np.sqrt(np.pi)
            
            # Normalize to ensure sum = 1 (should be close already)
            p_r = p_r / p_r.sum(dim=1, keepdim=True)
            
            # Entropy: H = -sum p(r) log p(r)
            log_p_r = torch.log(p_r + 1e-40)
            H = -torch.sum(p_r * log_p_r, dim=1)  # (N,)
            
            return H
    
        # Compute entropies using numerical integration
        H_r_numerical = entropy_gauss_hermite(logf_mean, logf_var, r_max=200, n_quad=50)
        H_r_xi_numerical = entropy_gauss_hermite(logf_mean_star, logf_var_star, r_max=200, n_quad=50)
        
        # Compare with Laplace
        print(f"=== Entropy Comparison ===")
        print(f"Laplace  H_marg: {H_r.item():.8f}, H_cond: {H_r_xi.item():.8f}, U: {(H_r - H_r_xi).item():.8f}")
        print(f"Numerical H_marg: {H_r_numerical.item():.8f}, H_cond: {H_r_xi_numerical.item():.8f}, U: {(H_r_numerical - H_r_xi_numerical).item():.8f}")
        print(f"Laplace error marg: {(H_r - H_r_numerical).item():.2e}")
        print(f"Laplace error cond: {(H_r_xi - H_r_xi_numerical).item():.2e}")
        
        # Use numerical values for utility
        H_r = H_r_numerical
        H_r_xi = H_r_xi_numerical.mean()  # If N>1

    utility = H_r - H_r_xi

    # if utility < 0.0:
    #     if torch.abs(utility) > 1e-6:
    #         raise ValueError(f" Negative utility computed: U={utility.item():.8f} ")

    if return_logf_moments:
        return utility, logf_mean.squeeze(), logf_var.squeeze()
    else:
        return utility

def conditioned_utility_corrected( x_star, model, remaining_imgs, N, r_cutoff=100, DEBUG_SAME_X=False, DEBUG_FIXED_X=False, DEBUG_FIX_LAMBDA_i=False, return_logf_moments=False ):

    # THIS IMPLEMENTATION FOLLOWS THE AUGMENTED MATRIX APPROACH WITH V' CORRECTION
    # AS DERIVED IN: ~/Documents/Papers/predictive_distribution_conditioned_no_approx.tex
    #
    # Key differences from conditioned_utility():
    # 1. Uses augmented matrices to capture exact k(x_i, x*) correlation
    # 2. Applies V' correction (updated posterior covariance) instead of using V directly
    # This ensures variance → 0 when x_i = x* (the sparse GP approximation artifact is fixed)

    # ==== Compute first term: Marginal entropy ====

    # Using moments of lambda(x_star) conditioned on Dataset D

    # Get hyperparameters of the model and the Kernel values related to x_tilde ( as done in the beginning of batch_utility_w_grad)

    # xtilde kernel and params
        # m, V, K_tilde_inv 
    # xstar kernel values
        # K_vec = K(x_star,x_star) ( nstar,1 )
        # K ( = K(x_star, x_tilde)

    # Get mu_star and sigma_2_star, moments of the log firing rate

    # Sum of laplace approximations as in nd utility -> Marginal entropy

    # (We need access to the gradient, do not break it)

    # Ensure x_star is 2D (add batch dim if needed)
    if x_star.ndim == 1:
        x_star = x_star[None, :]  # [n_pixels] → [1, n_pixels]
    elif x_star.ndim != 2:
        raise ValueError(f"x_star must be 1D or 2D, got shape {x_star.shape}")


    r_masked = torch.arange(0, r_cutoff, dtype=TORCH_DTYPE, device=DEVICE)

    kernfun     = model.kernfun
    xtilde      = model.xtilde

    theta = model.theta
    m_b   = model.m_b
    V_b   = model.V_b

    A = torch.exp(model.f_params['logA'])
    lambda0 = model.f_params['lambda0']

    C, mask, K_tilde_b, K_tilde_inv_b, B = GP_utils.get_final_K_vals(model)

    x_star_masked = x_star[:,mask]  # [N, n_masked_pixels]

    k0_star = kernfun(theta,x_star_masked, x2=None, C=C, dC=None, diag=True)

    k_star   = kernfun(theta,x_star_masked, xtilde[:,mask], C=C, dC=None, diag=False)

    k_star_b = k_star @ B
    # k_starKtilde_inv_b = k_star_b @ K_tilde_inv_b
    k_starKtilde_inv_b = k_star_b * K_tilde_inv_b.diag() # this is u_star_b(only of inducing points)

    # k_star, k_star_grad = kernfun(theta,x_star_masked, xtilde[:,mask], C=C, dC=None, diag=False, get_dK_x=True)
    # for j in range(k_star_b.shape[1]): # loop indicung points

    #     if torch.any( torch.isnan( k_star_b_grad ) ):
    #         print(f" Gradient mean k_star_b[0][{j}]  norm: {k_star_b_grad.norm().item():.6f} ")
            # raise ValueError(f" NaN gradient found in k_star_b for basis point {j} ")


    # === Marginal moments of lambda(x_star) | Dataset D ===
    lambda_m, lambda_var = GP_utils.lambda_moments(
       x_star_masked, K_tilde_b, k_starKtilde_inv_b, k0_star, k_star_b, C, m_b, V_b, theta)

    logf_mean = A*lambda_m + lambda0 # [N]
    logf_var  = A**2 * lambda_var    # [N]

    p_r, log_p_r = laplace_approximations_new( \
            mu=logf_mean, sigma2=logf_var, r=r_masked )  # shape (r, nstar), shape (r, nstar), shape (r, nstar)


    # Marginal Entropy  r | x_star, D ( first term )
    H_r = -torch.sum( p_r*log_p_r, dim=1 )  # shape (N)

    # ==== Conditional Entropy, second term ====

    # Sample N random images without replacement
    n_imgs = remaining_imgs.shape[0]

    perm = torch.randperm(n_imgs, device=remaining_imgs.device)[:N]
    x_i = remaining_imgs[perm]  # shape (N, n_pixels)

    if DEBUG_FIXED_X:
        x_i = remaining_imgs[:N]

    if DEBUG_SAME_X and N==1:
        # print(" DEBUG MODE: Using x_i identical to x_star for testing ")
        x_i = x_star.detach().clone().repeat(N, 1)  # shape (N, n_pixels)

    if DEBUG_SAME_X and N!=1:
        raise ValueError(" DEBUG_SAME_X MODE with N!=1 is not allowed, please set N=1 ")

    # check we didnt select any xi == x_star

    if not N==1:
        for j in range(N):
            if torch.all( x_i[j] == x_star ):
                raise ValueError(f" Sampled training image {j} is identical to x_star in conditioned utility ")

    # x_i = remaining_imgs[:N]
    # x_i = x_star.detach().clone().repeat(N, 1)  # shape (N, n_pixels)
    x_i_masked = x_i[:,mask]  # shape (N, n_masked_pixels)

    # Sample lambda(x_i)|x_i, D ~ N(mu_i, sigma_2_i) 
    # Moments of lambda(x_i) conditioned on Dataset D 
    k0_i = kernfun(theta,x_i_masked, x2=None, C=C, dC=None, diag=True)

    k_i   = kernfun(theta,x_i_masked, xtilde[:,mask], C=C, dC=None, diag=False) # [N, n_tilde]
    k_i_b = k_i @ B # (N, n_b)
    # k_iKtilde_inv_b = k_i_b @ K_tilde_inv_b
    k_iKtilde_inv_b = k_i_b * K_tilde_inv_b.diag()

    lambda_m_i, lambda_var_i = GP_utils.lambda_moments(
       x_i_masked, K_tilde_b, k_iKtilde_inv_b, k0_i, k_i_b, C, m_b, V_b, theta)

    # Sample from the Gaussian ( N independent samples with randn_like )
    if DEBUG_FIXED_X or DEBUG_SAME_X or DEBUG_FIX_LAMBDA_i:
        lambda_i_samples = lambda_m_i 
    else:    
        lambda_i_samples = lambda_m_i + torch.sqrt( lambda_var_i )*torch.randn_like(lambda_m_i) 


    # ===== Conditional entropy r|x, lambda(x), D , averaged with samples extracted above ====
    #
    # This implementation uses AUGMENTED MATRICES to capture exact kernel correlations
    # between x_i and x_star, following ~/Documents/Papers/predictive_distribution.tex (Section 4)

    # Compute cross-covariance k(x_i, x*) 
    k1 = kernfun(theta, x_i_masked, x_star_masked, C=C, dC=None, diag=False)  # shape (N, 1)


    # k1_manual, dk1_manual = kernfun(theta, x_i_masked, x_star_masked, C=C, get_dK_x=True)
    # print(f" Gradient k1_manual norm: {dk1_manual[0].norm().item():.6f} ")

    # k1_grad = torch.autograd.grad(  k1[0] , x_star_masked, retain_graph=True)[0]
    # if torch.any( torch.isnan( k1_grad ) ):
    #     # DEBUG : checking gradient is not actually nan if we compute manually the kernel gradient
    #     print(f" Gradient k1 norm: {k1_grad.norm().item():.6f} ")
        # raise ValueError(f" NaN gradient found in k1 ")

    # Build augmented cross-covariance vector k_aug(x*) = [k(x_i,x*); k(x*)] 
    k_xi_star_b = torch.cat(
        (k1, k_star_b.expand(N, -1)),
        dim=1
    )  # shape (N, n_b+1)

    # u and s for point x_i 
    # u_b = K⁻¹k(x_i)
    u_b = k_i_b * K_tilde_inv_b.diag()  # shape (N, n_b)

    # s = k(x_i,x_i) - k(x_i)ᵀK⁻¹k(x_i) = prior Schur complement for x_i
    s = k0_i - torch.einsum('Nb,Nb->N', k_i_b, u_b)  # shape (N,)
    s = torch.clamp(s, min=1e-6)

    # === Augmented System ===
    # Compute U_star = k_aug(x*)ᵀ K_aug⁻¹ using efficient block inverse 
    U_star = compute_U_star_direct(k_xi_star_b, u_b, s, K_tilde_inv_b)  # shape (N, n_b+1)

    # ===== Compute Updated Posterior m' (Eq. 92 in predictive_distribution.tex) =====

    # δ = λ(x) - uᵀm  (innovation: how much sample differs from marginal mean)
    delta = lambda_i_samples - torch.einsum('Nb,b->N', u_b, m_b)  # shape (N,)

    # denom = s + uᵀVu  (marginal variance of λ(x_i) under posterior)
    uVu = torch.einsum('Nb,bj,Nj->N', u_b, V_b, u_b)  # shape (N,)
    denom = s + uVu  # shape (N,)

    # Vu = V @ u  (needed for m' update)
    Vu = torch.einsum('bj,Nj->Nb', V_b, u_b)  # shape (N, n_b)

    # m' = m + Vu · δ / denom  (Eq. 92)
    m_prime_b = m_b[None, :] + Vu * (delta / denom)[:, None]  # shape (N, n_b)

    # ===== Build Augmented Mean Vector [λ(x); m'] (Eq. 187-191) =====
    lambda_aug_mean = torch.cat(
        (lambda_i_samples[:, None], m_prime_b),
        dim=1
    )  # shape (N, n_b+1)

    # ===== Predictive Mean (Eq. 200-204) =====
    # μ_pred = U_star ᵀ [λ(x); m']
    lambda_m2 = torch.einsum('Nj,Nj->N', U_star, lambda_aug_mean)  # shape (N,)

    # ===== Predictive Variance (Eq. 208-217) =====
    # S_* = k(x*,x*) - U_star ᵀ k_aug(x*) = k(x*,x*) - U_star ᵀ K_aug U_star
    # We never computed K_aug explicitly
    S_star = k0_star - torch.einsum('Nj,Nj->N', U_star, k_xi_star_b)  # shape (N,)

    # U_star[1:]ᵀ V' U_star[1:] where V' = V - VuuᵀV/denom
    # Expanding: = U_star[1:]ᵀ V U_star[1:] - (U_star[1:]ᵀ Vu)² / denom
    U_star_V_U_star = torch.einsum('Nj,jk,Nk->N', U_star[:, 1:], V_b, U_star[:, 1:])  # shape (N,)
    c = torch.einsum('Nj,Nj->N', U_star[:, 1:], Vu)  # = U_star[1:]ᵀ V u, shape (N,)
    V_prime_term = U_star_V_U_star - c**2 / denom  # = U_star[1:]ᵀ V' U_star[1:]

    # σ²_pred = S_* + U_star[1:]ᵀ V' U_star[1:]  (Eq. 211)
    lambda_var2 = S_star + V_prime_term  # shape (N,)


    # Sum of laplace approximations as in nd utility -> Sample of conditional entropy
    logf_mean2 = A*lambda_m2 + lambda0 # [N]
    logf_var2  = A**2 * lambda_var2    # [N]

    # In case of very small variance it uses exact formula
    p_r_xi, log_p_r_xi = laplace_approximations_new( \
        mu=logf_mean2, sigma2=logf_var2, r=r_masked )
    
    # Marginal Entropy  r | x_star, D ( first term )
    H_r_xi_samples = -torch.sum( p_r_xi*log_p_r_xi, dim=1 )  # shape (N)
    # assert not torch.any(torch.isnan(H_r_xi_samples)), "NaN values found in conditional entropy samples"

    H_r_xi = torch.mean( H_r_xi_samples )  # Average over N samples

    utility = H_r - H_r_xi


    # if utility < 0.0:
    #     if torch.abs(utility) > 1e-6:
    #         raise ValueError(f" Negative utility computed: U={utility.item():.8f} ")

    if return_logf_moments:
        return utility, logf_mean.squeeze(), logf_var.squeeze()
    else:
        return utility

def conditioned_utility_clean( x_star, model, remaining_imgs, N, r_cutoff=100, lambda_samples_per_x=1, DEBUG_SAME_X=False, DEBUG_FIXED_X=False, DEBUG_FIX_LAMBDA_i=False, return_logf_moments=False ):
    """
    Same as conditioned_utility_corrected but uses acosker_clean kernel function.

    THIS IMPLEMENTATION FOLLOWS THE AUGMENTED MATRIX APPROACH WITH V' CORRECTION
    AS DERIVED IN: ~/Documents/Papers/predictive_distribution_conditioned_no_approx.tex

    Key differences from conditioned_utility_corrected():
    - Uses GP_utils.acosker_clean instead of model.kernfun (acosker)
    - acosker_clean has native PyTorch shapes (no internal transpose)
    """

    # Ensure x_star is 2D (add batch dim if needed)
    if x_star.ndim == 1:
        x_star = x_star[None, :]  # [n_pixels] → [1, n_pixels]
    elif x_star.ndim != 2:
        raise ValueError(f"x_star must be 1D or 2D, got shape {x_star.shape}")


    r_masked = torch.arange(0, r_cutoff, dtype=TORCH_DTYPE, device=DEVICE)

    xtilde      = model.xtilde

    kernfun = GP_utils.acosker_with_grad
    # kernfun = GP_utils.acosker_clean


    theta = model.theta
    m_b   = model.m_b
    V_b   = model.V_b

    A = torch.exp(model.f_params['logA'])
    lambda0 = model.f_params['lambda0']

    C, mask, K_tilde_b, K_tilde_inv_b, B = GP_utils.get_final_K_vals(model)

    x_star_masked = x_star[:,mask]  # [N, n_masked_pixels]

    k0_star = kernfun(theta, x_star_masked, None, C=C, diag=True)

    k_star = kernfun(theta, x_star_masked, xtilde[:,mask], C=C, diag=False)

    k_star_b = k_star @ B
    k_starKtilde_inv_b = k_star_b * K_tilde_inv_b.diag() # this is u_star_b(only of inducing points)

    # k_star_grad = torch.autograd.grad(  k_star[0][0] , x_star_masked, retain_graph=True)[0].mean()
    # print(f" Gradient mean k_star norm: {k_star_grad.norm().item():.6f} ")


    # === Marginal moments of lambda(x_star) | Dataset D ===
    lambda_m, lambda_var = GP_utils.lambda_moments(
       x_star_masked, K_tilde_b, k_starKtilde_inv_b, k0_star, k_star_b, C, m_b, V_b, theta,
       kernfun=kernfun)

    logf_mean = A*lambda_m + lambda0 # [N]
    logf_var  = A**2 * lambda_var    # [N]

    p_r, log_p_r = laplace_approximations_new( \
            mu=logf_mean, sigma2=logf_var, r=r_masked )  # shape (r, nstar), shape (r, nstar), shape (r, nstar)


    # Marginal Entropy  r | x_star, D ( first term )
    H_r = -torch.sum( p_r*log_p_r, dim=1 )  # shape (N)

    # ==== Conditional Entropy, second term ====

    # Sample N random images without replacement
    n_imgs = remaining_imgs.shape[0]

    perm = torch.randperm(n_imgs, device=remaining_imgs.device)[:N]
    x_i = remaining_imgs[perm]  # shape (N, n_pixels)

    if DEBUG_FIXED_X:
        x_i = remaining_imgs[:N]

    if DEBUG_SAME_X and N==1:
        x_i = x_star.detach().clone().repeat(N, 1)  # shape (N, n_pixels)

    if DEBUG_SAME_X and N!=1:
        raise ValueError(" DEBUG_SAME_X MODE with N!=1 is not allowed, please set N=1 ")

    # check we didnt select any xi == x_star

    if not DEBUG_SAME_X:
        for j in range(N):
            if torch.all( x_i[j] == x_star ):
                raise ValueError(f" Sampled training image {j} is identical to x_star in conditioned utility ")

    x_i_masked = x_i[:,mask]  # shape (N, n_masked_pixels)

    # Sample lambda(x_i)|x_i, D ~ N(mu_i, sigma_2_i)
    # Moments of lambda(x_i) conditioned on Dataset D
    k0_i = kernfun(theta, x_i_masked, None, C=C, diag=True)

    k_i = kernfun(theta, x_i_masked, xtilde[:,mask], C=C, diag=False) # [N, n_tilde]
    k_i_b = k_i @ B # (N, n_b)
    k_iKtilde_inv_b = k_i_b * K_tilde_inv_b.diag()

    lambda_m_i, lambda_var_i = GP_utils.lambda_moments(
       x_i_masked, K_tilde_b, k_iKtilde_inv_b, k0_i, k_i_b, C, m_b, V_b, theta,
       kernfun=kernfun)

    H_r_x_list = [] # Stores the MEANS of each H_r_xi over lambda samples

    # ===== Conditional entropy r|x, lambda(x), D , averaged with samples extracted above ====
    #
    # This implementation uses AUGMENTED MATRICES to capture exact kernel correlations
    # between x_i and x_star, following ~/Documents/Papers/predictive_distribution.tex (Section 4)

    # the fact that we allow more than one lambda sample per x_i makes the order of variables 
    # a bit messy, we are keeping everything not dependent on lambda_i_samples outside the loop


    # Compute cross-covariance k(x_i, x*)
    k1 = kernfun(theta, x_i_masked, x_star_masked, C=C, diag=False)  # shape (N, 1)

    # Build augmented cross-covariance vector k_aug(x*) = [k(x_i,x*); k(x*)]
    k_xi_star_b = torch.cat(
        (k1, k_star_b.expand(N, -1)),
        dim=1
    )  # shape (N, n_b+1)

    # u and s for point x_i
    # u_b = K⁻¹k(x_i)
    u_b = k_i_b * K_tilde_inv_b.diag()  # shape (N, n_b)

    # s = k(x_i,x_i) - k(x_i)ᵀK⁻¹k(x_i) = prior Schur complement for x_i
    s = k0_i - torch.einsum('Nb,Nb->N', k_i_b, u_b)  # shape (N,)
    s = torch.clamp(s, min=1e-6)

    # === Augmented System ===
    # Compute U_star = k_aug(x*)ᵀ K_aug⁻¹ using efficient block inverse
    U_star = compute_U_star_direct(k_xi_star_b, u_b, s, K_tilde_inv_b)  # shape (N, n_b+1)

    # ===== Compute Updated Posterior m' (Eq. 92 in predictive_distribution.tex) =====

    # Vu = V @ u  (needed for m' update)
    Vu = torch.einsum('bj,Nj->Nb', V_b, u_b)  # shape (N, n_b)

    # denom = s + uᵀVu  (marginal variance of λ(x_i) under posterior)
    uVu = torch.einsum('Nb,bj,Nj->N', u_b, V_b, u_b)  # shape (N,)
    denom = s + uVu  # shape (N,)

    # ===== Predictive Variance (Eq. 208-217) =====
    # S_* = k(x*,x*) - U_star ᵀ k_aug(x*) = k(x*,x*) - U_star ᵀ K_aug U_star
    # We never computed K_aug explicitly
    S_star = k0_star - torch.einsum('Nj,Nj->N', U_star, k_xi_star_b)  # shape (N,)

    # U_star[1:]ᵀ V' U_star[1:] where V' = V - VuuᵀV/denom
    # Expanding: = U_star[1:]ᵀ V U_star[1:] - (U_star[1:]ᵀ Vu)² / denom
    U_star_V_U_star = torch.einsum('Nj,jk,Nk->N', U_star[:, 1:], V_b, U_star[:, 1:])  # shape (N,)
    c = torch.einsum('Nj,Nj->N', U_star[:, 1:], Vu)  # = U_star[1:]ᵀ V u, shape (N,)
    V_prime_term = U_star_V_U_star - c**2 / denom  # = U_star[1:]ᵀ V' U_star[1:]

    for _ in range(lambda_samples_per_x):
        # Sample from the Gaussian ( N independent samples with randn_like )
        if DEBUG_SAME_X or DEBUG_FIX_LAMBDA_i:
            lambda_i_samples = lambda_m_i
        else:
            lambda_i_samples = lambda_m_i + torch.sqrt( lambda_var_i )*torch.randn_like(lambda_m_i)

        # δ = λ(x) - uᵀm  (innovation: how much sample differs from marginal mean)
        delta = lambda_i_samples - torch.einsum('Nb,b->N', u_b, m_b)  # shape (N,)

        # m' = m + Vu · δ / denom  (Eq. 92)
        m_prime_b = m_b[None, :] + Vu * (delta / denom)[:, None]  # shape (N, n_b)

        # ===== Build Augmented Mean Vector [λ(x); m'] (Eq. 187-191) =====
        lambda_aug_mean = torch.cat(
            (lambda_i_samples[:, None], m_prime_b),
            dim=1
        )  # shape (N, n_b+1)

        # ===== Predictive Mean (Eq. 200-204) =====
        # μ_pred = U_star ᵀ [λ(x); m']
        lambda_m2 = torch.einsum('Nj,Nj->N', U_star, lambda_aug_mean)  # shape (N,)

        # σ²_pred = S_* + U_star[1:]ᵀ V' U_star[1:]  (Eq. 211)
        lambda_var2 = S_star + V_prime_term  # shape (N,)

        # Sum of laplace approximations as in nd utility -> Sample of conditional entropy
        logf_mean2 = A*lambda_m2 + lambda0 # [N]
        logf_var2  = A**2 * lambda_var2    # [N]

        # In case of very small variance it uses exact formula
        p_r_xi, log_p_r_xi = laplace_approximations_new( \
            mu=logf_mean2, sigma2=logf_var2, r=r_masked )

        # Marginal Entropy  r | x_star, D ( first term )
        H_r_xi_samples = -torch.sum( p_r_xi*log_p_r_xi, dim=1 )  # shape (N)

        H_r_xi = torch.mean( H_r_xi_samples )  # Average over N samples

        H_r_x_list.append( H_r_xi )

    H_r_x = torch.mean( torch.stack( H_r_x_list ) )  # Mean over lambda samples

    utility = H_r - H_r_x
    # print(f"DEBUG: Utility is conditonal entropy only")
    # utility = -H_r_x

    if return_logf_moments:
        return utility, logf_mean.squeeze(), logf_var.squeeze()
    else:
        return utility

def optimize_with_conditioned_utility(
    model,
    imgs_train,
    remaining_idx,
    start_img_idx,
    N=50,
    lambda_samples=1, # lambda samples per x
    n_iterations=10,
    lr=1e-2,
    r_cutoff=100,
    return_logf_moments=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    use_sigmoid_constraint=False,
    pixel_min=None,
    pixel_max=None,

    DEBUG_dict={}
):
    """
    Optimize image using conditioned_utility with Adam optimizer.

    Optimizes only the masked region (receptive field) - this happens naturally
    because conditioned_utility() internally masks the image via x_star[:,mask],
    so gradients are zero for unmasked pixels.

    Optionally uses sigmoid reparameterization to enforce hard pixel bounds.

    Parameters
    ----------
    model : GPModel
        Current fitted GP model
    imgs_train : torch.Tensor
        Full training image dataset
    start_img_idx : int
        Index of starting image in imgs_train
    N : int
        Number of Monte Carlo samples for conditioned_utility (default 50)
    n_iterations : int
        Number of Adam optimization steps (default 10)
    lr : float
        Learning rate for Adam optimizer (default 0.01)
    r_cutoff : int
        Maximum spike count for utility computation (default 100)
    return_logf_moments : bool
        If True, return logf moments (not implemented for this method)
    betas : tuple of floats
        Adam beta parameters for momentum (default (0.9, 0.999))
    eps : float
        Adam epsilon for numerical stability (default 1e-8)
    use_sigmoid_constraint : bool
        If True, reparameterize pixels via sigmoid to enforce hard bounds (default False)
    pixel_min : float or None
        Minimum pixel value for sigmoid constraint (default: imgs_train.min())
    pixel_max : float or None
        Maximum pixel value for sigmoid constraint (default: imgs_train.max())

    Returns
    -------
    dict
        Dictionary with keys: optimized_img, img_idx, utility_batch, start_gradient,
        end_gradient, U_initial, U_final, lr, line_search_fn, max_iter, logf_mean_initial,
        logf_mean_final, logf_var_initial, logf_var_final
    """
    DEBUG_SINGLE_IMAGE = DEBUG_dict.get('DEBUG_SINGLE_IMAGE', False)
    DEBUG_FULL_FIELD = DEBUG_dict.get('DEBUG_FULL_FIELD', False)
    # New: use a small pool of full-field gray images for sampling
    DEBUG_FULL_FIELD_MULTI = DEBUG_dict.get('DEBUG_FULL_FIELD_MULTI', False)
    verbose = DEBUG_dict.get('verbose', True)

    # Filter out start_img_idx from remaining_idx (always done)
    samplable_idx = remaining_idx != start_img_idx
    samplable_idx = remaining_idx[samplable_idx]

    assert start_img_idx not in samplable_idx, \
        "Starting image index must not be in remaining images for conditioned utility."

    # Initialize starting image and sampled images based on debug mode
    if DEBUG_FULL_FIELD or DEBUG_FULL_FIELD_MULTI:
        # DEBUG MODE: Test model response to uniform full-field stimuli
        # Gray levels are configurable via DEBUG_dict:
        #   - start_gray: fraction of pixel range for starting image (default 0.7)
        # Single-sample mode (DEBUG_FULL_FIELD):
        #   - sample_gray: fraction of pixel range for single sample image (default 0.5)
        # Multi-sample mode (DEBUG_FULL_FIELD_MULTI):
        #   - center_gray: center fraction for the pool (default 0.5)
        #   - span: total span around the center (default 0.05)
        #   - n_full_field_samples: number of full-field samples in the pool (default 10)
        # NOTE: conditioned_utility_clean() will randomly sample N images from remaining_imgs.
        pixel_min_val = imgs_train.min()
        pixel_max_val = imgs_train.max()
        pixel_range = pixel_max_val - pixel_min_val

        start_gray_frac = DEBUG_dict.get('start_gray', 0.7)

        # Backward-compatibility: if user sets sample_gray in multi mode, treat it as the center.
        center_gray_frac = DEBUG_dict.get('center_gray', DEBUG_dict.get('sample_gray', 0.5))
        span = DEBUG_dict.get('span', 0.05)
        n_full_field_samples = int(DEBUG_dict.get('n_full_field_samples', 10))

        start_gray_value = pixel_min_val + pixel_range * start_gray_frac

        x = torch.full_like(imgs_train[start_img_idx], start_gray_value)
        initial_img = x.clone().detach()

        if DEBUG_FULL_FIELD_MULTI:
            if n_full_field_samples < 1:
                raise ValueError(f"n_full_field_samples must be >= 1, got {n_full_field_samples}")
            if N > n_full_field_samples:
                raise ValueError(
                    f"In DEBUG_FULL_FIELD_MULTI mode, N must be <= n_full_field_samples. Got N={N}, n_full_field_samples={n_full_field_samples}."
                )

            # Build the pool of full-field images with gray fractions in [0,1]
            gray_fracs = torch.linspace(
                center_gray_frac - span / 2,
                center_gray_frac + span / 2,
                steps=n_full_field_samples,
                device=imgs_train.device,
                dtype=imgs_train.dtype,
            )
            gray_fracs = torch.clamp(gray_fracs, 0.0, 1.0)
            gray_values = pixel_min_val + pixel_range * gray_fracs

            remaining_imgs = torch.stack(
                [torch.full_like(imgs_train[0], v) for v in gray_values],
                dim=0
            )  # Shape: (n_full_field_samples, n_pixels)

            if verbose:
                print(" DEBUG FULL FIELD MULTI MODE ACTIVE ")
                print(f"   Starting gray: {start_gray_frac*100:.2f}% -> pixel value {start_gray_value.item():.4f}")
                print(f"   Pool center gray: {center_gray_frac*100:.2f}%, span: {span:.4f}, n_samples: {n_full_field_samples}")
                # Keep prints short but explicit (10 samples)
                for j in range(n_full_field_samples):
                    print(f"     sample[{j:02d}]: frac={gray_fracs[j].item():.6f} -> pixel={gray_values[j].item():.4f}")
        else:
            sample_gray_frac = DEBUG_dict.get('sample_gray', 0.5)
            sample_gray_value = pixel_min_val + pixel_range * sample_gray_frac
            remaining_imgs = torch.full_like(imgs_train[0], sample_gray_value)[None]  # Shape: (1, n_pixels)

            if verbose:
                print(" DEBUG FULL FIELD MODE ACTIVE ")
                print(f"   Starting gray: {start_gray_frac*100:.0f}% -> pixel value {start_gray_value.item():.4f}")
                print(f"   Sample gray:   {sample_gray_frac*100:.0f}% -> pixel value {sample_gray_value.item():.4f}")

    elif DEBUG_SINGLE_IMAGE:
        if verbose:
            print(" DEBUG SINGLE IMAGE MODE ACTIVE ")
        x = imgs_train[start_img_idx].clone().detach()

        sampled_img_idx = samplable_idx[torch.randint(len(samplable_idx), (1,))]
        remaining_imgs = imgs_train[sampled_img_idx] 

    else:
        # Normal operation: use actual images from dataset
        x = imgs_train[start_img_idx].clone().detach()

        remaining_imgs = imgs_train[samplable_idx]

    # Setup sigmoid constraint if enabled
    theta = None  # Will hold unconstrained parameters if using sigmoid
    if use_sigmoid_constraint:
        # Get pixel bounds from dataset if not provided
        if pixel_min is None:
            pixel_min = imgs_train.min()
        if pixel_max is None:
            pixel_max = imgs_train.max()

        # Transform to unconstrained θ space via logit
        p = (x - pixel_min) / (pixel_max - pixel_min)
        p = torch.clamp(p, 1e-7, 1 - 1e-7)  # Numerical safety
        theta = torch.log(p / (1 - p)).requires_grad_(True)

        # For initial state computation, transform back to x
        x = pixel_min + (pixel_max - pixel_min) * torch.sigmoid(theta)
    else:
        x = x.requires_grad_(True)

    # Initial state
    if return_logf_moments:
        U_initial, logf_mean_init, logf_var_init = conditioned_utility_clean(
            x, model, remaining_imgs, N, r_cutoff, 
            lambda_samples_per_x=lambda_samples, 
            return_logf_moments=True
            )
    else:
        U_initial = conditioned_utility_clean(
            x, model, remaining_imgs, N, r_cutoff, 
            lambda_samples_per_x = lambda_samples
            )
        logf_mean_init, logf_var_init = None, None
    grad_initial = torch.autograd.grad(U_initial, x)[0].detach().clone()

    if verbose:
        print(f"\n=== Conditioned Utility Optimization (Adam) ===")
        print(f"Starting image index: {start_img_idx}")
        print(f"Parameters: N={N}, n_iterations={n_iterations}, lr={lr}, betas={betas}, eps={eps}")
        if use_sigmoid_constraint:
            print(f"Sigmoid constraint active: pixels ∈ [{pixel_min.item():.4f}, {pixel_max.item():.4f}]")
        print(f"Initial utility: {U_initial.item():.6f}")
        print(f"Initial gradient norm: {grad_initial.norm().item():.6e}")

    # Create Adam optimizer for gradient ascent
    if use_sigmoid_constraint:
        optimizer = torch.optim.Adam([theta], lr=lr, betas=betas, eps=eps)
    else:
        optimizer = torch.optim.Adam([x], lr=lr, betas=betas, eps=eps)

    # Adam optimization loop
    for i in range(n_iterations):

        # Zero gradients
        optimizer.zero_grad()

        # Transform θ → x if using sigmoid constraint
        if use_sigmoid_constraint:
            x = pixel_min + (pixel_max - pixel_min) * torch.sigmoid(theta)

        # Compute utility
        U = conditioned_utility_clean(x, model, remaining_imgs, N, r_cutoff, lambda_samples)

        # Negate utility for maximization (Adam minimizes)
        loss = -U
        loss.backward()

        # Check for NaN gradients (on the parameter being optimized)
        grad_param = theta.grad if use_sigmoid_constraint else x.grad
        if torch.any(torch.isnan(grad_param)):
            print(f"NaN gradient detected at iteration {i}")
            print(f"x stats: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
            raise ValueError(f"NaN values found in gradient at iteration {i}")

        # Store gradient norm before step (for logging)
        grad_norm = grad_param.norm().item()

        # Adam step
        optimizer.step()
        if verbose:
            print(f"  Iter {i}/{n_iterations}: U = {U.item():.6f}, ||grad|| = {grad_norm:.6e}")

    # Final state - transform θ → x if using sigmoid constraint
    if use_sigmoid_constraint:
        x = pixel_min + (pixel_max - pixel_min) * torch.sigmoid(theta)

    if return_logf_moments:
        U_final, logf_mean_final, logf_var_final = conditioned_utility_clean(
            x, model, remaining_imgs, N, r_cutoff, 
            lambda_samples_per_x = lambda_samples, 
            return_logf_moments  = True
            )
    else:
        U_final = conditioned_utility_clean(
            x, model, remaining_imgs, N, r_cutoff, 
            lambda_samples_per_x = lambda_samples
            )
        logf_mean_final, logf_var_final = None, None
    grad_final = torch.autograd.grad(U_final, x)[0]

    if verbose:
        print(f"\nFinal utility: {U_final.item():.6f}")
        print(f"Final gradient norm: {grad_final.norm().item():.6e}")
        print(f"Utility change: {U_final.item() - U_initial.item():+.6f} ({(U_final.item()/U_initial.item() - 1)*100:+.2f}%)")
        if use_sigmoid_constraint:
            print(f"Final pixel range: [{x.min().item():.4f}, {x.max().item():.4f}] (bounds: [{pixel_min.item():.4f}, {pixel_max.item():.4f}])")

    # Return compatible dict
    return {
        'optimized_img': x.detach(),
        'img_idx': torch.tensor([start_img_idx], device=x.device),
        'utility_batch': None,
        'start_gradient': grad_initial,
        'end_gradient': grad_final.detach(),
        'U_initial': U_initial.item(),
        'U_final': U_final.item(),
        'lr': lr,
        'line_search_fn': None,
        'max_iter': n_iterations,
        'logf_mean_initial': logf_mean_init.item() if logf_mean_init is not None else None,
        'logf_mean_final': logf_mean_final.item() if logf_mean_final is not None else None,
        'logf_var_initial': logf_var_init.item() if logf_var_init is not None else None,
        'logf_var_final': logf_var_final.item() if logf_var_final is not None else None,
        'sampled_img': remaining_imgs if N==1 else None,
        'sampled_img_pool': remaining_imgs.detach() if (DEBUG_FULL_FIELD or DEBUG_FULL_FIELD_MULTI) else None,
        'initial_img': initial_img if (DEBUG_FULL_FIELD or DEBUG_FULL_FIELD_MULTI) else None
    }

def soft_sigmoid_compress_pixels(x_masked, pixel_min, pixel_max, steepness=1.0,
                                 transition_width=0.95, verbose=True):
    """
    Apply piecewise soft compression to bring outliers into valid range while preserving valid pixels.

    Uses a piecewise function that:
    - PRESERVES pixels in the safe zone (identity function)
    - Smoothly COMPRESSES outliers near boundaries using exponential approach
    - ASYMPTOTICALLY approaches pixel_min/pixel_max for far outliers
    - Maintains monotonicity (ordering preserved)

    The transformation is piecewise:
    1. Safe zone [x_min_safe, x_max_safe]: f(x) = x (unchanged)
    2. Upper transition (x > x_max_safe): f(x) = pixel_max - margin * exp(-(x - x_max_safe)/steepness)
    3. Lower transition (x < x_min_safe): f(x) = pixel_min + margin * exp((x - x_min_safe)/steepness)

    Parameters:
    -----------
    x_masked : torch.Tensor
        Masked pixel values (n_masked_pixels,)
    pixel_min : float
        Minimum valid pixel value (dataset minimum)
    pixel_max : float
        Maximum valid pixel value (dataset maximum)
    steepness : float, default=1.0
        Controls compression rate (higher = gentler compression)
        - Higher values (e.g., 2.0): Outliers compressed slowly
        - Lower values (e.g., 0.5): Outliers compressed aggressively
        - Recommended range: [0.5, 2.0]
    transition_width : float, default=0.95
        Fraction of valid range that is "safe zone" (no compression)
        - 0.95 means inner 95% of range is safe
        - 0.90 means inner 90% of range is safe
        - Range: (0, 1) where higher = larger safe zone
    verbose : bool, default=True
        Print compression statistics

    Returns:
    --------
    x_compressed : torch.Tensor
        Compressed pixels with outliers brought into valid range

    Examples:
    ---------
    >>> # Valid range: [-2.68, 3.12], transition_width=0.95
    >>> # Safe zone: [-2.39, 2.83] (inner 95%)
    >>> # Pixel at 0.0 (in safe zone) -> 0.0 (unchanged)
    >>> # Pixel at 5.0 (outlier) -> ~3.08 (compressed toward pixel_max)
    """
    x_min_actual = x_masked.min().item()
    x_max_actual = x_masked.max().item()

    # Count outliers before compression
    n_outliers_low = (x_masked < pixel_min).sum().item()
    n_outliers_high = (x_masked > pixel_max).sum().item()
    n_total = len(x_masked)

    # Check if any compression is needed
    has_outliers = (n_outliers_low > 0) or (n_outliers_high > 0)

    if not has_outliers:
        if verbose:
            print(f"  [Compression] All pixels within valid range - no compression needed")
            print(f"    Range: [{x_min_actual:.4f}, {x_max_actual:.4f}]")
            print(f"    Valid: [{pixel_min:.4f}, {pixel_max:.4f}]")
        return x_masked

    # Define safe zone boundaries
    range_width = pixel_max - pixel_min
    margin = (1 - transition_width) * range_width / 2
    x_min_safe = pixel_min + margin
    x_max_safe = pixel_max - margin

    # Start with original values
    x_compressed = x_masked.clone()

    # Compress upper outliers: x > x_max_safe
    upper_region = x_masked > x_max_safe
    if upper_region.any():
        # Exponential approach to pixel_max
        # As x -> infinity, f(x) -> pixel_max
        # At x = x_max_safe, f(x) = x_max_safe (continuous)
        x_compressed[upper_region] = pixel_max - margin * torch.exp(
            -(x_masked[upper_region] - x_max_safe) / steepness
        )

    # Compress lower outliers: x < x_min_safe
    lower_region = x_masked < x_min_safe
    if lower_region.any():
        # Exponential approach to pixel_min
        # As x -> -infinity, f(x) -> pixel_min
        # At x = x_min_safe, f(x) = x_min_safe (continuous)
        x_compressed[lower_region] = pixel_min + margin * torch.exp(
            (x_masked[lower_region] - x_min_safe) / steepness
        )

    # Middle region: unchanged (identity function)
    # This is automatic since we cloned x_masked

    if verbose:
        n_unchanged = (~upper_region & ~lower_region).sum().item()
        n_compressed = (upper_region | lower_region).sum().item()

        print(f"  [Compression] Piecewise exponential (steepness={steepness}, safe_zone={transition_width*100:.0f}%)")
        print(f"    Original range: [{x_min_actual:.4f}, {x_max_actual:.4f}]")
        print(f"    Valid range:    [{pixel_min:.4f}, {pixel_max:.4f}]")
        print(f"    Safe zone:      [{x_min_safe:.4f}, {x_max_safe:.4f}]")
        print(f"    Compressed to:  [{x_compressed.min().item():.4f}, {x_compressed.max().item():.4f}]")
        print(f"    Pixels: {n_unchanged} unchanged ({100*n_unchanged/n_total:.1f}%), "
              f"{n_compressed} compressed ({100*n_compressed/n_total:.1f}%)")
        print(f"    Outliers before: {n_outliers_low} low, {n_outliers_high} high "
              f"({100*(n_outliers_low + n_outliers_high)/n_total:.1f}%)")

    return x_compressed

def plot_optimization_comparison_slim(initial_img, optimized_img, initial_gradient, final_gradient,
                                       U_initial, U_final, model_active, img_idx, n_images,
                                       save_path, constraint_name="RMS",
                                       lr=None, line_search_fn=None, max_iter=None,
                                       lr_attempts=None, success_attempt=None):
    """
    Create simplified comparison visualization of initial vs optimized image.

    Creates a 2x2 grid showing:
    - Row 1: Initial image | Optimized image
    - Row 2: Initial gradient ∂U/∂x | Final gradient ∂U/∂x

    All images are zoomed to show only the masked region with a small margin.

    Parameters:
    -----------
    initial_img : torch.Tensor
        Initial image (11664,) or (108*108,)
    optimized_img : torch.Tensor
        Optimized image (11664,) or (108*108,)
    initial_gradient : torch.Tensor
        Initial gradient ∂U/∂x for masked pixels only
    final_gradient : torch.Tensor
        Final gradient ∂U/∂x for masked pixels only
    U_initial : float
        Initial utility value
    U_final : float
        Final utility value
    model_active : GPModel
        Current fitted GP model
    img_idx : int
        Image index
    n_images : int
        Current training iteration
    save_path : Path
        Directory to save figures
    constraint_name : str
        Constraint name for filename (e.g., "RMS")
    lr : float, optional
        Learning rate used for optimization
    line_search_fn : str, optional
        Line search function used ('armijo', 'strong_wolfe', or None)
    max_iter : int, optional
        Maximum iterations used for optimization
    """
    import matplotlib.gridspec as gridspec

    n_px_side = 108

    # Prepare mask for plotting
    C, mask, K_tilde_b, K_tilde_inv_b, B = GP_utils.get_final_K_vals(model_active)

    mask_2d = torch.zeros((n_px_side * n_px_side), device=mask.device)
    mask_2d[mask] = 1.0
    mask_2d = mask_2d.reshape(n_px_side, n_px_side).cpu().numpy()

    # === COMPUTE ZOOM REGION FROM MASK ===
    # Find bounding box of masked region
    rows, cols = np.where(mask_2d == 1)
    if rows.size > 0 and cols.size > 0:
        ymin, ymax = np.min(rows), np.max(rows)
        xmin, xmax = np.min(cols), np.max(cols)

        # Add margin (10 pixels on each side, clamped to image bounds)
        margin = 10
        ymin_zoom = max(0, ymin - margin)
        ymax_zoom = min(n_px_side - 1, ymax + margin)
        xmin_zoom = max(0, xmin - margin)
        xmax_zoom = min(n_px_side - 1, xmax + margin)
    else:
        # Fallback: show entire image if mask is empty
        ymin_zoom, ymax_zoom = 0, n_px_side - 1
        xmin_zoom, xmax_zoom = 0, n_px_side - 1

    # Compute gradient L2 norms
    grad_norm_init = torch.norm(initial_gradient).item()
    grad_norm_final = torch.norm(final_gradient).item()

    # Find consistent gradient vmin/vmax (shared colormap limits)
    grad_vmax = max(torch.max(torch.abs(initial_gradient)).item(),
                    torch.max(torch.abs(final_gradient)).item())

    # Create figure with 2x2 grid (same height as original 4-row figure)
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.1, figure=fig)

    # Add figure title with hyperparameters
    hyperparam_str = ""
    if lr is not None and line_search_fn is not None and max_iter is not None:
        hyperparam_str = f" | LR={lr:.0e}, LS={line_search_fn}, MaxIter={max_iter}"

    fig.suptitle(
        f"Optimization Comparison: Image {img_idx} (n_images={n_images})\n"
        f"Constraint: {constraint_name}{hyperparam_str}",
        fontsize=14, fontweight='bold', y=0.98
    )

    # Row 1: Images
    ax_img_init = fig.add_subplot(gs[0, 0])
    ax_img_opt = fig.add_subplot(gs[0, 1])

    # Row 2: Gradients
    ax_grad_init = fig.add_subplot(gs[1, 0])
    ax_grad_opt = fig.add_subplot(gs[1, 1])

    # ===== Initial Image =====
    img_init_2d = initial_img.reshape(n_px_side, n_px_side).cpu().numpy()
    vmin_init = img_init_2d.min()
    vmax_init = img_init_2d.max()
    # ===== Optimized Image =====
    img_opt_2d = optimized_img.reshape(n_px_side, n_px_side).cpu().numpy()
    vmin_opt = img_opt_2d.min()
    vmax_opt = img_opt_2d.max()

    vmin_img = min(vmin_init, vmin_opt)
    vmax_img = max(vmax_init, vmax_opt)

    im_init = ax_img_init.imshow(img_init_2d, cmap='gray', vmin=vmin_img, vmax=vmax_img)
    ax_img_init.contour(mask_2d, levels=[0.5], colors='yellow', linewidths=1.5, alpha=0.7)
    ax_img_init.set_xlim(xmin_zoom, xmax_zoom)
    ax_img_init.set_ylim(ymax_zoom, ymin_zoom)
    ax_img_init.set_title(f"Initial Image\nU = {U_initial:.4f}", fontsize=12, fontweight='bold')
    ax_img_init.axis('off')


    im_opt = ax_img_opt.imshow(img_opt_2d, cmap='gray', vmin=vmin_img, vmax=vmax_img)
    ax_img_opt.contour(mask_2d, levels=[0.5], colors='yellow', linewidths=1.5, alpha=0.7)
    ax_img_opt.set_xlim(xmin_zoom, xmax_zoom)
    ax_img_opt.set_ylim(ymax_zoom, ymin_zoom)

    # Calculate improvement
    delta_U = U_final - U_initial
    pct_improvement = 100 * delta_U / U_initial if U_initial != 0 else 0

    ax_img_opt.set_title(f"Optimized Image\nU = {U_final:.4f}\nΔU = {delta_U:+.4f} ({pct_improvement:+.1f}%)",
                         fontsize=12, fontweight='bold')
    ax_img_opt.axis('off')

    # ===== Plot Initial Gradient =====
    # Reconstruct gradient image: create full image with zeros, fill masked region
    grad_init_full = torch.zeros((n_px_side * n_px_side), device=initial_gradient.device)
    grad_init_full[mask] = initial_gradient
    grad_init_2d = grad_init_full.reshape(n_px_side, n_px_side).cpu().numpy()

    im_grad_init = ax_grad_init.imshow(grad_init_2d, cmap='RdBu_r', vmin=-grad_vmax, vmax=grad_vmax)
    ax_grad_init.contour(mask_2d, levels=[0.5], colors='yellow', linewidths=1.5, alpha=0.7)
    ax_grad_init.set_xlim(xmin_zoom, xmax_zoom)
    ax_grad_init.set_ylim(ymax_zoom, ymin_zoom)
    ax_grad_init.set_title(f"Initial Gradient ∂U/∂x\n||∇U|| = {grad_norm_init:.2e}",
                          fontsize=12)
    ax_grad_init.axis('off')

    # Add colorbar for initial gradient
    plt.colorbar(im_grad_init, ax=ax_grad_init, fraction=0.046, pad=0.04)

    # ===== Plot Final Gradient =====
    grad_final_full = torch.zeros((n_px_side * n_px_side), device=final_gradient.device)
    grad_final_full[mask] = final_gradient
    grad_final_2d = grad_final_full.reshape(n_px_side, n_px_side).cpu().numpy()

    im_grad_opt = ax_grad_opt.imshow(grad_final_2d, cmap='RdBu_r', vmin=-grad_vmax, vmax=grad_vmax)
    ax_grad_opt.contour(mask_2d, levels=[0.5], colors='yellow', linewidths=1.5, alpha=0.7)
    ax_grad_opt.set_xlim(xmin_zoom, xmax_zoom)
    ax_grad_opt.set_ylim(ymax_zoom, ymin_zoom)

    # Calculate gradient reduction
    grad_reduction = grad_norm_final - grad_norm_init
    pct_grad_reduction = 100 * grad_reduction / grad_norm_init if grad_norm_init != 0 else 0

    ax_grad_opt.set_title(f"Final Gradient ∂U/∂x\n||∇U|| = {grad_norm_final:.2e}\nΔ||∇U|| = {grad_reduction:+.2e} ({pct_grad_reduction:+.1f}%)",
                         fontsize=12)
    ax_grad_opt.axis('off')

    # Add colorbar for final gradient
    plt.colorbar(im_grad_opt, ax=ax_grad_opt, fraction=0.046, pad=0.04)

    # ===== Add statistics text box =====
    stats_text = (
        f"Image {img_idx} (n_images={n_images})\n"
        f"Constraint: {constraint_name}\n\n"
        f"Initial U: {U_initial:.4f}\n"
        f"Final U: {U_final:.4f}\n"
        f"Improvement: ΔU = {delta_U:+.4f} ({pct_improvement:+.1f}%)\n\n"
        f"Initial ||∇U||: {grad_norm_init:.2e}\n"
        f"Final ||∇U||: {grad_norm_final:.2e}\n"
        f"Reduction: Δ||∇U|| = {grad_reduction:+.2e} ({pct_grad_reduction:+.1f}%)"
    )

    # Add lr information if available
    if lr_attempts is not None and len(lr_attempts) > 1:
        # Scheduled mode: show all attempts
        stats_text += f"\n\nLR Scheduling:"
        for i, lr_val in enumerate(lr_attempts):
            marker = "✓" if i == success_attempt else " "
            stats_text += f"\n  {marker} {lr_val:.1e}"
        stats_text += f"\nFinal: {lr:.1e}"
    elif lr is not None:
        # Standard mode: show single lr
        stats_text += f"\n\nLR: {lr:.1e}"

    # Add text box to figure
    fig.text(0.02, 0.98, stats_text, transform=fig.transFigure,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            family='monospace')

    # Suppress tight_layout warning
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    if lr is not None and line_search_fn is not None and max_iter is not None:
        lr_str = f"{lr:.0e}".replace('+', '')  # e.g., "1e+04" -> "1e04"
        save_name = (f'{constraint_name}_opt_comparison_n{n_images:03d}_'
                     f'ls{line_search_fn}_iter{max_iter}_lr{lr_str}_'
                     f'img{img_idx:04d}.png')
    else:
        save_name = f'{constraint_name}_opt_comparison_n{n_images:03d}_img{img_idx:04d}.png'

    fig.savefig(save_path / save_name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved optimization comparison plot: {save_name}")

def compute_pixel_penalty(x_masked, pixel_min, pixel_max, alpha, threshold_fraction=0.9):
    """
    Compute smooth-threshold penalty for pixel bounds with penalty-free safe zone.

    Implements a smooth, differentiable penalty that creates a "safe zone" in the
    inner threshold_fraction of the pixel bounds where penalty is zero. This allows
    pixels to move freely within the safe zone without fighting penalty gradients,
    while still providing smooth protection near actual boundaries.

    **Key Features:**
    - **Centered softplus:** Penalty is exactly zero at threshold boundaries
    - **Negative penalties inside safe zone:** Pixels deep in safe zone get "reward" (negative penalty)
    - **Positive penalties outside threshold:** Smooth push-back toward bounds
    - **True penalty-free zone:** No baseline penalty accumulation (unlike standard softplus)
    - **Smooth gradients:** No discontinuities anywhere (C^∞ differentiable)
    - **L-BFGS compatible:** Works with high learning rates

    **Design Philosophy:**
    Pixels should be free to explore useful values without constraint, but still
    be prevented from extreme violations. The threshold creates a buffer zone.

    Parameters:
    -----------
    x_masked : torch.Tensor
        Masked pixel values being optimized (shape: n_masked_pixels)
    pixel_min : float
        Dataset minimum pixel value (actual hard boundary)
    pixel_max : float
        Dataset maximum pixel value (actual hard boundary)
    alpha : float
        Penalty steepness parameter (higher = sharper transition)
        Recommended: 5.0
    threshold_fraction : float
        Fraction of pixel range considered "safe" (no penalty zone)
        Default: 0.9 (inner 90% is penalty-free)
        Range: (0, 1) where 0.9 = 90% safe, 0.95 = 95% safe, etc.

    Returns:
    --------
    penalty : torch.Tensor
        Scalar penalty value (differentiable w.r.t. x_masked)

    Mathematical formulation:
    -------------------------
    Safe zone boundaries:
        range_width = pixel_max - pixel_min
        margin = (1 - threshold_fraction) * range_width / 2
        x_max_thresh = pixel_max - margin
        x_min_thresh = pixel_min + margin

    Centered softplus penalty (zero at threshold, negative inside, positive outside):
        P(x) = Σᵢ [(softplus(α(xᵢ - x_max_thresh)) - log(2)) +
                   (softplus(α(x_min_thresh - xᵢ)) - log(2))]
        where softplus(z) = log(1 + exp(z))

    Penalty behavior:
        - At threshold boundary (xᵢ = x_max_thresh): P(xᵢ) = 0 (exactly zero!)
        - Inside safe zone (xᵢ < x_max_thresh): P(xᵢ) < 0 (negative = reward)
        - Outside threshold (xᵢ > x_max_thresh): P(xᵢ) > 0 (positive = penalty)
        - Deep in safe zone: P(xᵢ) → -log(2) ≈ -0.693 per boundary (maximum reward)

    Example with threshold_fraction=0.9, bounds [-2.683, 3.117]:
        range_width = 5.8
        margin = 0.29
        Safe zone: [-2.39, 2.83] → negative to zero penalty (reward to neutral)
        Threshold boundaries: x = -2.39 or x = 2.83 → exactly zero penalty
        Warning zones: [-2.683, -2.39] and [2.83, 3.117] → positive penalty (push back)

    Gradient (computed automatically by PyTorch):
        ∂P/∂xᵢ = α[σ(α(xᵢ - x_max_thresh)) - σ(α(x_min_thresh - xᵢ))]
        where σ(z) = 1/(1 + exp(-z)) is the sigmoid function
    """
    import torch.nn.functional as F

    # Calculate safe zone boundaries (inner threshold_fraction of range)
    range_width = pixel_max - pixel_min
    margin = (1 - threshold_fraction) * range_width / 2

    # Threshold boundaries where penalty starts (shifted inward from actual bounds)
    x_max_thresh = pixel_max - margin
    x_min_thresh = pixel_min + margin

    # Centered softplus penalties: subtract log(2) to make penalty exactly zero at threshold
    # This creates a true "penalty-free" safe zone where penalty can even be negative (rewarding)
    # Mathematical reasoning:
    #   softplus(0) = log(2) ≈ 0.693 (baseline at threshold)
    #   Subtracting log(2) makes: softplus(0) - log(2) = 0 (zero at threshold)
    #   For pixels inside safe zone: penalty becomes NEGATIVE (reward for staying inside)
    #   For pixels outside threshold: penalty is POSITIVE (pushes back toward bounds)

    import torch
    log_2 = torch.log(torch.tensor(2.0, dtype=x_masked.dtype, device=x_masked.device))

    upper_penalty = F.softplus(alpha * (x_masked - x_max_thresh)) - log_2
    lower_penalty = F.softplus(alpha * (x_min_thresh - x_masked)) - log_2

    # Sum over all pixels (can be negative if most pixels deep in safe zone)
    penalty = torch.sum(upper_penalty + lower_penalty)

    return penalty

def compute_log_barrier(x_masked, pixel_min_safe, pixel_max_safe, epsilon=1e-8):
    """
    Compute log barrier for interior point optimization: B(x) = -Σ[log(x - x_min) + log(x_max - x)]

    The log barrier enforces HARD constraints by creating an infinite penalty at boundaries.
    This is fundamentally different from softplus penalty:
    - Domain: Only defined for x_min < x < x_max (undefined outside)
    - Gradient: ∂B/∂x = -1/(x - x_min) + 1/(x_max - x) → ∞ as x → boundary
    - Behavior: Impossible to violate bounds (would require infinite step)

    Interior Point Method Requirements:
    - Initial point MUST be strictly feasible (inside bounds)
    - Barrier parameter 1/t controls strength (small t = strong barrier)
    - Must use barrier scheduling: gradually increase t to approach optimum

    Parameters:
    -----------
    x_masked : torch.Tensor
        Masked pixel values being optimized (shape: n_masked_pixels)
    pixel_min_safe : float
        Minimum allowed pixel value (with safety margin applied)
    pixel_max_safe : float
        Maximum allowed pixel value (with safety margin applied)
    epsilon : float
        Numerical safety term to prevent log(0) (default: 1e-8)
        Added to log arguments but does not affect feasibility check

    Returns:
    --------
    barrier : torch.Tensor or None
        Scalar barrier value (differentiable w.r.t. x_masked)
        Returns None if point is infeasible
    is_feasible : bool
        True if all pixels strictly inside bounds, False otherwise

    Mathematical Formulation:
    -------------------------
    For each pixel i:
        B_i = -[log(x_i - x_min) + log(x_max - x_i)]

    Total barrier:
        B(x) = Σ B_i

    Gradient (computed automatically by PyTorch autograd):
        ∂B/∂x_i = -1/(x_i - x_min) + 1/(x_max - x_i)

    Barrier behavior:
        - At center (x = (x_min + x_max)/2): ∂B/∂x = 0 (equilibrium)
        - Left of center: ∂B/∂x < 0 (pushes right)
        - Right of center: ∂B/∂x > 0 (pushes left)
        - Near boundaries: |∂B/∂x| → ∞ (strong repulsion)

    Usage in Interior Point:
    ------------------------
    loss = -U(x) + (1/t) * B(x)

    where t is barrier parameter:
        - Small t: barrier dominates, stay far from boundaries
        - Large t: utility dominates, approach boundaries
        - Schedule: t_k = μ^k * t_0 (exponential growth)

    Example:
    --------
    >>> x = torch.tensor([0.0, 1.0, 2.5])
    >>> B, feasible = compute_log_barrier(x, pixel_min_safe=-3.0, pixel_max_safe=3.0)
    >>> if feasible:
    >>>     print(f"Barrier: {B.item():.4f}")
    >>>     B.backward()  # Gradients computed automatically
    """
    import torch

    # Strict feasibility check (must be INSIDE bounds, not on boundaries)
    # This is critical for log barrier - log(0) or log(negative) = NaN
    if (x_masked <= pixel_min_safe).any() or (x_masked >= pixel_max_safe).any():
        # Point is infeasible - return None to signal rejection
        return None, False

    # Compute barrier terms
    # Add epsilon for numerical safety (prevents exact log(0) in edge cases)
    # Note: epsilon is small enough not to affect optimization dynamics
    lower_term = torch.log(x_masked - pixel_min_safe + epsilon)
    upper_term = torch.log(pixel_max_safe - x_masked + epsilon)

    # Barrier is negative sum of log terms
    # Minimizing barrier → maximizing sum of logs → staying centered in feasible region
    barrier = -torch.sum(lower_term + upper_term)

    return barrier, True

def shrink_to_safe_interior(x, pixel_min_safe, pixel_max_safe, shrink_factor=0.95):
    """
    Project pixels to safe interior while preserving image structure as much as possible.

    This function is needed when initial image has pixels at or outside the safe bounds,
    which would make it infeasible for interior point methods (log barrier undefined).

    Strategy:
    ---------
    1. Check if already inside bounds → return unchanged if safe
    2. If not, linearly scale image toward center of safe zone
    3. Preserve relative structure (don't just clamp pixels independently)
    4. Apply final safety clamp to ensure strict feasibility

    The scaling preserves image features while ensuring all pixels fall within
    the required bounds for interior point optimization.

    Parameters:
    -----------
    x : torch.Tensor
        Pixel values to project (shape: n_pixels)
    pixel_min_safe : float
        Minimum safe pixel value (after margin)
    pixel_max_safe : float
        Maximum safe pixel value (after margin)
    shrink_factor : float
        Fraction of safe range to target (default: 0.95)
        0.95 means use inner 95% of safe zone, leaving 2.5% buffer on each side

    Returns:
    --------
    x_shrunk : torch.Tensor
        Projected pixel values, guaranteed to satisfy:
        pixel_min_safe < x_shrunk[i] < pixel_max_safe for all i

    Algorithm:
    ----------
    1. Compute center of safe zone: c = (pixel_min_safe + pixel_max_safe) / 2
    2. Compute safe range: r_safe = (pixel_max_safe - pixel_min_safe) * shrink_factor
    3. Compute current range: r_current = x.max() - x.min()
    4. If r_current > r_safe: scale = r_safe / r_current, else scale = 1.0
    5. Center and scale: x_shrunk = c + (x - x.mean()) * scale
    6. Final safety clamp: ensure all pixels within [pixel_min_safe + ε, pixel_max_safe - ε]

    Example:
    --------
    >>> x = torch.tensor([-4.0, 0.0, 5.0])  # Outside bounds
    >>> x_safe = shrink_to_safe_interior(x, pixel_min_safe=-3.0, pixel_max_safe=3.0)
    >>> print(x_safe)  # All values now in (-3.0, 3.0)
    tensor([-2.4000,  0.0000,  2.4000])
    """
    import torch

    # Check current range
    x_min_current = x.min()
    x_max_current = x.max()

    # If already strictly inside safe bounds, no modification needed
    if x_min_current > pixel_min_safe and x_max_current < pixel_max_safe:
        return x.clone()

    # Center of safe zone (target equilibrium point)
    center = (pixel_min_safe + pixel_max_safe) / 2.0

    # Available safe range (with shrink factor for extra buffer)
    safe_range = (pixel_max_safe - pixel_min_safe) * shrink_factor

    # Current range of pixel values
    current_range = x_max_current - x_min_current

    # Determine scaling factor
    # If current range exceeds safe range, must compress
    # Otherwise, just recenter without scaling
    if current_range > safe_range:
        scale = safe_range / current_range
    else:
        scale = 1.0

    # Center the data (remove mean), scale if needed, then shift to safe center
    x_centered = x - x.mean()
    x_shrunk = center + x_centered * scale

    # Final safety clamp: ensure strict interior
    # Add small epsilon to guarantee strict inequalities (not boundary contact)
    epsilon = 1e-6
    x_shrunk = torch.clamp(x_shrunk,
                           pixel_min_safe + epsilon,
                           pixel_max_safe - epsilon)

    return x_shrunk

def _optimize_rms_constraint(
    model_active,
    imgs_train,
    x_idx_best,
    mask,
    u2d,
    max_r_cap,
    lr,
    max_iter,
    line_search_fn,
    pixel_min,
    pixel_max,
    apply_soft_compression,
    apply_hard_clipping,
    compression_steepness,
    compression_transition_width,
    return_logf_moments=False,
    verbose=True
):
    """
    Optimize image using RMS (mean + std) constraint with optional post-processing.

    This method fixes both the mean and standard deviation of masked pixels
    using reparameterization: x_masked = μ + σ * (θ - θ.mean()) / θ.std()

    Parameters
    ----------
    model_active : GPModel
        Current fitted GP model
    imgs_train : torch.Tensor
        Full training image dataset
    x_idx_best : torch.Tensor
        Index of best initial image
    mask : torch.Tensor
        Boolean mask for pixels to optimize
    u2d : torch.Tensor
        Utility batch for all remaining images
    max_r_cap : int
        Maximum spike count for utility computation
    lr : float
        Learning rate for L-BFGS optimizer
    max_iter : int
        Maximum number of L-BFGS iterations
    line_search_fn : str or None
        Line search function ('strong_wolfe', 'armijo', or None)
    pixel_min : float
        Minimum pixel value for post-processing bounds
    pixel_max : float
        Maximum pixel value for post-processing bounds
    apply_soft_compression : bool
        Whether to apply soft sigmoid compression post-processing
    apply_hard_clipping : bool
        Whether to apply hard clipping post-processing
    compression_steepness : float
        Steepness parameter for soft compression
    compression_transition_width : float
        Transition width parameter for soft compression

    Returns
    -------
    dict
        Dictionary with keys: optimized_img, img_idx, utility_batch, start_gradient,
        end_gradient, U_initial, U_final, lr, line_search_fn, max_iter
    """
    print("\n=== RMS-Constrained Optimization (Slim Version) ===")

    if return_logf_moments:
        logf_mean_init = None
        logf_var_init = None
        logf_mean_final = None
        logf_var_final = None


    # Setup
    initial_img = imgs_train[x_idx_best].clone()
    initial_unmasked = initial_img[~mask].clone()
    initial_masked = initial_img[mask].clone()

    # Target constraints from initial image
    μ_target = initial_masked.mean().item()
    σ_target = initial_masked.std().item()
    # print(f"Target constraints: μ = {μ_target:.6f}, σ = {σ_target:.6f}")
    # print(f"Initial L2 norm of masked region: {torch.linalg.norm(initial_masked).item():.6f}")

    # Optimization variable θ (unconstrained)
    θ = initial_masked.clone().detach().requires_grad_(True)
    θ_std_min = 0.01  # Numerical safety

    # Get initial state
    with torch.enable_grad():
        θ_mean_init = θ.mean()
        θ_std_init = θ.std()
        x_masked_init = μ_target + σ_target * (θ - θ_mean_init) / θ_std_init

        x_full_init = initial_img.clone()
        x_full_init[mask] = x_masked_init

        # Initial state: conditionally compute logf moments
        if return_logf_moments:
            U_init, logf_mean_init, logf_var_init = compute_utility_single_image(
                model_active, x_full_init, max_r_cap, return_logf_moments=True)
        else:
            U_init = compute_utility_single_image(model_active, x_full_init, max_r_cap)
            logf_mean_init, logf_var_init = None, None

        dU_dtheta_init, dU_dx_init = torch.autograd.grad(
            U_init,
            [θ, x_masked_init])

    # Create optimizer
    optimizer = torch.optim.LBFGS(
        [θ],
        lr=lr,
        max_iter=max_iter,  # Internal L-BFGS iterations
        history_size=500,
        line_search_fn=line_search_fn,
        armijo_c1=1e-3,
        armijo_rho=0.7,
        armijo_min_step_size=1e-5,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        verbose=False,  # Set True for debugging
    )

    # Track NaN gradient occurrences
    nan_grad_count = [0]
    max_nan_grad_allowed = 50

    # Closure
    def closure():
        """Closure for L-BFGS with RMS reparameterization"""
        # Numerical safety: clamp std
        with torch.no_grad():
            θ_std_current = θ.std()
            if θ_std_current < θ_std_min:
                if verbose:
                    print(f"    [Warning] Clamping θ std in closure")
                θ.data = (θ - θ.mean()) * (θ_std_min / θ_std_current) + θ.mean()

        optimizer.zero_grad()

        # Reparameterization: θ → x_masked (constraints satisfied automatically)
        with torch.enable_grad():
            θ_mean = θ.mean()
            θ_std = θ.std()
            x_masked_constrained = μ_target + σ_target * (θ - θ_mean) / θ_std

            # Reconstruct full image
            x_full = torch.empty_like(initial_img)
            x_full[~mask] = initial_unmasked
            x_full[mask] = x_masked_constrained

            # Compute utility
            U = compute_utility_single_image(model_active, x_full, max_r_cap)

        # Handle invalid values
        if torch.isnan(U) or torch.isinf(U):
            print(f"    [Closure] U is {'NaN' if torch.isnan(U) else 'Inf'}, rejecting step")
            return torch.tensor(1e10, dtype=U.dtype, device=U.device, requires_grad=True)

        # L-BFGS minimizes, so return negative
        loss = -U
        loss.backward()

        # Handle invalid gradients
        if θ.grad is not None:
            if torch.isnan(θ.grad).any() or torch.isinf(θ.grad).any():
                nan_grad_count[0] += 1
                if nan_grad_count[0] <= max_nan_grad_allowed:
                    print(f"    [Closure] Warning: Gradient is NaN/Inf (occurrence {nan_grad_count[0]}/{max_nan_grad_allowed}), clearing and rejecting step")
                    optimizer.zero_grad()
                    return torch.tensor(1e10, dtype=loss.dtype, device=loss.device, requires_grad=False)
                else:
                    raise RuntimeError(
                        f"Gradient contains NaN or Inf after backward pass (occurred {nan_grad_count[0]} times)! "
                        f"U={U.item():.6f}, loss={loss.item():.6f}. "
                    )
        else:
            raise ValueError("[Closure] θ.grad is None")

        return loss

    # === SINGLE OPTIMIZATION CALL ===
    print(f"Running L-BFGS with max_iter={max_iter}...")
    optimizer.step(closure)

    # Get final state
    with torch.enable_grad():
        θ_mean_final = θ.mean()
        θ_std_final = θ.std()
        x_masked_final = μ_target + σ_target * (θ - θ_mean_final) / θ_std_final

        if apply_soft_compression:
            # Get pixel bounds from dataset if not provided
            if pixel_min is None or pixel_max is None:
                pix_min = imgs_train.min().item()
                pix_max = imgs_train.max().item()
            else:
                pix_min = pixel_min
                pix_max = pixel_max

            # print(f"\n{'='*60}")
            # print(f"Applying soft sigmoid compression...")

            # Apply compression
            x_masked_final = soft_sigmoid_compress_pixels(
                x_masked_final.detach(),
                pixel_min=pix_min,
                pixel_max=pix_max,
                steepness=compression_steepness,
                transition_width=compression_transition_width,
                verbose=True
            )

            # Enable gradients for recomputation
            x_masked_final = x_masked_final.requires_grad_(True)

            # Report RMS constraint violations
            μ_compressed = x_masked_final.mean().item()
            σ_compressed = x_masked_final.std().item()
            μ_error = abs(μ_compressed - μ_target)
            σ_error = abs(σ_compressed - σ_target)

            # print(f"    RMS constraint violations after compression:")
            # print(f"      Mean: {μ_compressed:.6f} (target: {μ_target:.6f}, error: {μ_error:.6f})")
            # print(f"      Std:  {σ_compressed:.6f} (target: {σ_target:.6f}, error: {σ_error:.6f})")
            # print(f"{'='*60}\n")

        elif apply_hard_clipping:
            # Get pixel bounds from dataset if not provided
            if pixel_min is None or pixel_max is None:
                pix_min = imgs_train.min().item()
                pix_max = imgs_train.max().item()
            else:
                pix_min = pixel_min
                pix_max = pixel_max

            # Count outliers before clipping
            n_outliers = ((x_masked_final < pix_min) | (x_masked_final > pix_max)).sum().item()

            # Apply hard clipping
            x_masked_final = torch.clamp(x_masked_final.detach(), pix_min, pix_max)
            x_masked_final = x_masked_final.requires_grad_(True)

            # Report RMS violations
            μ_clipped = x_masked_final.mean().item()
            σ_clipped = x_masked_final.std().item()
            # print(f"  [Hard Clipping] {n_outliers} outliers clipped. RMS errors: Δμ={abs(μ_clipped - μ_target):.6f}, Δσ={abs(σ_clipped - σ_target):.6f}")

        x_full_final = initial_img.clone()
        x_full_final[mask] = x_masked_final

        # Final state: conditionally compute logf moments
        if return_logf_moments:
            U_final, logf_mean_final, logf_var_final = compute_utility_single_image(
                model_active, x_full_final, max_r_cap, return_logf_moments=True)
        else:
            U_final = compute_utility_single_image(model_active, x_full_final, max_r_cap)
            logf_mean_final, logf_var_final = None, None

        dU_dtheta_final, dU_dx_final = torch.autograd.grad(
            U_final,
            [θ, x_masked_final],
            allow_unused=True)

        # Handle None when compression breaks computational graph
        if dU_dtheta_final is None:
            dU_dtheta_final = torch.zeros_like(θ)

    if verbose:
        print(f"\nInitial: U = {U_init.item():.6f}, ||∂U/∂θ|| = {dU_dtheta_init.norm().item():.6e}, ||∂U/∂x_masked|| = {dU_dx_init.norm().item():.6e}")
        print(f"Final:   U = {U_final.item():.6f}, ||∂U/∂θ|| = {dU_dtheta_final.norm().item():.6e}, ||∂U/∂x_masked|| = {dU_dx_final.norm().item():.6e}")
        print(f"Change:  ΔU = {U_final.item() - U_init.item():+.6f} ({(U_final.item()/U_init.item() - 1)*100:+.2f}%)")
        print(f"         Δ||∂U/∂θ|| = {dU_dtheta_final.norm().item() - dU_dtheta_init.norm().item():+.6e} ({((dU_dtheta_final.norm().item()/dU_dtheta_init.norm().item()) - 1)*100:+.2f}%)")
        print(f"         Δ||∂U/∂x|| = {dU_dx_final.norm().item() - dU_dx_init.norm().item():+.6e} ({((dU_dx_final.norm().item()/dU_dx_init.norm().item()) - 1)*100:+.2f}%)")
    # Verify constraints
    μ_error = abs(x_masked_final.mean().item() - μ_target)
    σ_error = abs(x_masked_final.std().item() - σ_target)
    if verbose:
        if μ_error > 1e-4 or σ_error > 1e-4:
            print(f"    [Warning] Final constraints not satisfied within tolerance: |Δμ|={μ_error:.6e}, |Δσ|={σ_error:.6e}")

    # L-BFGS diagnostics
    state = optimizer.state[optimizer._params[0]]
    n_iters = state.get('n_iter', 0)
    n_evals = state.get('func_evals', 0)
    if verbose:
        print(f"L-BFGS iterations: {n_iters}")

    # Update gradient and optimized image

    optimized_img = x_full_final

    # Return dictionary matching original batch_utility_w_grad format (without trajectory)
    # Return dictionary matching original batch_utility_w_grad format (without trajectory)
    return {
        'optimized_img': optimized_img,
        'img_idx': x_idx_best[None] if x_idx_best.dim() == 0 else x_idx_best,
        'utility_batch': u2d,
        'start_gradient': dU_dx_init.squeeze(0) if dU_dx_init.dim() > 1 else dU_dx_init,
        'end_gradient': dU_dx_final.squeeze(0) if  dU_dx_final.dim() > 1 else dU_dx_final,
        'U_initial': U_init.item() if isinstance(U_init, torch.Tensor) else U_init,
        'U_final': U_final.item() if isinstance(U_final, torch.Tensor) else U_final,
        'lr': lr,
        'line_search_fn': line_search_fn,
        'max_iter': max_iter,
        'logf_mean_initial': logf_mean_init.item() if logf_mean_init is not None else None,
        'logf_mean_final': logf_mean_final.item() if logf_mean_final is not None else None,
        'logf_var_initial': logf_var_init.item() if logf_var_init is not None else None,
        'logf_var_final': logf_var_final.item() if logf_var_final is not None else None
    }

def _optimize_px_sigmoid_constraint(
    model_active,
    imgs_train,
    x_idx_best,
    mask,
    u2d,
    max_r_cap,
    lr,
    max_iter,
    line_search_fn,
    initialize_from_gray=False,
    return_logf_moments=False,
):
    """
    Optimize image using pixel-wise sigmoid constraint for hard bounds.

    This method enforces hard pixel bounds via sigmoid transformation:
    x = pixel_min + (pixel_max - pixel_min) * sigmoid(θ)

    Parameters
    ----------
    model_active : GPModel
        Current fitted GP model
    imgs_train : torch.Tensor
        Full training image dataset
    x_idx_best : torch.Tensor
        Index of best initial image
    mask : torch.Tensor
        Boolean mask for pixels to optimize
    u2d : torch.Tensor
        Utility batch for all remaining images
    max_r_cap : int
        Maximum spike count for utility computation
    lr : float
        Learning rate for L-BFGS optimizer
    max_iter : int
        Maximum number of L-BFGS iterations
    line_search_fn : str or None
        Line search function ('strong_wolfe', 'armijo', or None)
    initialize_from_gray : bool
        If True, initialize masked region as uniform gray (mean intensity).
        Tests de novo optimization from neutral baseline. (default: False)
    return_logf_moments : bool
        Whether to return logf_mean and logf_var

    Returns
    -------
    dict
        Dictionary with keys: optimized_img, img_idx, utility_batch, start_gradient,
        end_gradient, U_initial, U_final, lr, line_search_fn, max_iter
    """
    print("\n=== Pixel-wise Sigmoid Constraint ===")

    if return_logf_moments:
        logf_mean_init = None
        logf_var_init = None
        logf_mean_final = None
        logf_var_final = None

    # Setup (standard pattern)
    initial_img = imgs_train[x_idx_best].clone()
    initial_unmasked = initial_img[~mask].clone()
    initial_masked = initial_img[mask].clone()

    # Optionally initialize from gray
    if initialize_from_gray:
        gray_value = initial_masked.mean()
        initial_masked = torch.full_like(initial_masked, gray_value)
        print(f"Initializing from uniform gray value: {gray_value.item():.6f}")

    # Get dataset-wide pixel bounds
    pixel_min = imgs_train.min()
    pixel_max = imgs_train.max()
    print(f"Dataset pixel bounds: [{pixel_min.item():.6f}, {pixel_max.item():.6f}]")
    print(f"Initial masked pixel range: [{initial_masked.min().item():.6f}, {initial_masked.max().item():.6f}]")

    # Transform initial pixels to unconstrained space via logit
    p = (initial_masked - pixel_min) / (pixel_max - pixel_min)
    p = torch.clamp(p, 1e-7, 1 - 1e-7)  # Numerical safety: avoid log(0) or log(∞)
    θ = torch.log(p / (1 - p))  # logit transformation
    θ = θ.detach().requires_grad_(True)

    # Compute initial utility and gradients
    with torch.enable_grad():
        x_masked_init = pixel_min + (pixel_max - pixel_min) * torch.sigmoid(θ)

        x_full_init = initial_img.clone()
        x_full_init[mask] = x_masked_init

        # Initial state: conditionally compute logf moments
        if return_logf_moments:
            U_init, logf_mean_init, logf_var_init = compute_utility_single_image(
                model_active, x_full_init, max_r_cap, return_logf_moments=True)
        else:
            U_init = compute_utility_single_image(model_active, x_full_init, max_r_cap)
            logf_mean_init, logf_var_init = None, None

        dU_dtheta_init, dU_dx_init = torch.autograd.grad(
            U_init,
            [θ, x_masked_init])

    print(f"Initial utility: {U_init.item():.6f}")
    print(f"Initial ||∇U||: {torch.linalg.norm(dU_dx_init).item():.6f}")

    # Create optimizer (standard LBFGS configuration)
    optimizer = torch.optim.LBFGS(
        [θ],
        lr=lr,
        max_iter=max_iter,
        history_size=500,
        line_search_fn=line_search_fn,
        armijo_c1=1e-3,
        armijo_rho=0.7,
        armijo_min_step_size=1e-5,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        verbose=False
    )

    # NaN gradient tracking
    nan_grad_count = [0]
    max_nan_grad_allowed = 50

    # Define closure function
    def closure():
        optimizer.zero_grad()

        with torch.enable_grad():
            # Sigmoid transformation: θ → x_masked
            x_masked_constrained = pixel_min + (pixel_max - pixel_min) * torch.sigmoid(θ)

            # Reconstruct full image
            x_full = torch.empty_like(initial_img)
            x_full[~mask] = initial_unmasked
            x_full[mask] = x_masked_constrained

            # Compute utility
            U = compute_utility_single_image(model_active, x_full, max_r_cap)

        # Handle invalid utility values
        if torch.isnan(U) or torch.isinf(U):
            return torch.tensor(1e10, dtype=U.dtype, device=U.device, requires_grad=True)

        # Loss (LBFGS minimizes, so negate utility)
        loss = -U
        loss.backward()

        # Handle invalid gradients
        if θ.grad is not None:
            if torch.isnan(θ.grad).any() or torch.isinf(θ.grad).any():
                nan_grad_count[0] += 1
                if nan_grad_count[0] <= max_nan_grad_allowed:
                    optimizer.zero_grad()
                    return torch.tensor(1e10, dtype=loss.dtype, device=loss.device, requires_grad=False)
                else:
                    raise RuntimeError(f"[Closure] NaN/Inf gradient exceeded limit: {nan_grad_count[0]}")
        else:
            raise ValueError("[Closure] θ.grad is None after backward pass")

        return loss

    # Run optimization
    optimizer.step(closure)

    # Compute final state and gradients
    with torch.enable_grad():
        x_masked_final = pixel_min + (pixel_max - pixel_min) * torch.sigmoid(θ)

        x_full_final = initial_img.clone()
        x_full_final[mask] = x_masked_final

        # Final state: conditionally compute logf moments
        if return_logf_moments:
            U_final, logf_mean_final, logf_var_final = compute_utility_single_image(
                model_active, x_full_final, max_r_cap, return_logf_moments=True)
        else:
            U_final = compute_utility_single_image(model_active, x_full_final, max_r_cap)
            logf_mean_final, logf_var_final = None, None

        dU_dtheta_final, dU_dx_final = torch.autograd.grad(
            U_final,
            [θ, x_masked_final],
            allow_unused=True)

        if dU_dtheta_final is None:
            dU_dtheta_final = torch.zeros_like(θ)

    # Print diagnostics
    print(f"Final utility: {U_final.item():.6f}")
    print(f"Utility improvement: {(U_final.item() - U_init.item()):.6f} ({((U_final.item() - U_init.item()) / abs(U_init.item()) * 100):.2f}%)")
    print(f"Final ||∇U||: {torch.linalg.norm(dU_dx_final).item():.6f}")
    print(f"Final masked pixel range: [{x_masked_final.min().item():.6f}, {x_masked_final.max().item():.6f}]")
    print(f"Constraints satisfied: pixels ∈ [{pixel_min.item():.6f}, {pixel_max.item():.6f}]")
    if nan_grad_count[0] > 0:
        print(f"Warning: {nan_grad_count[0]} NaN/Inf gradient occurrences during optimization")

    # Prepare return dictionary (standard format)
    optimized_img = x_full_final

    return {
        'optimized_img': optimized_img,
        'img_idx': x_idx_best[None] if x_idx_best.dim() == 0 else x_idx_best,
        'utility_batch': u2d,
        'start_gradient': dU_dx_init.squeeze(0) if dU_dx_init.dim() > 1 else dU_dx_init,
        'end_gradient': dU_dx_final.squeeze(0) if dU_dx_final.dim() > 1 else dU_dx_final,
        'U_initial': U_init.item() if isinstance(U_init, torch.Tensor) else U_init,
        'U_final': U_final.item() if isinstance(U_final, torch.Tensor) else U_final,
        'lr': lr,
        'line_search_fn': line_search_fn,
        'max_iter': max_iter,
        'logf_mean_initial': logf_mean_init.item() if logf_mean_init is not None else None,
        'logf_mean_final': logf_mean_final.item() if logf_mean_final is not None else None,
        'logf_var_initial': logf_var_init.item() if logf_var_init is not None else None,
        'logf_var_final': logf_var_final.item() if logf_var_final is not None else None
    }

def _optimize_px_sigmoid_variance_only(
    model_active,
    imgs_train,
    x_idx_best,
    mask,
    u2d,
    max_r_cap,
    lr,
    max_iter,
    line_search_fn,
    mean_penalty_weight=100.0,
    return_logf_moments=False,
):
    """
    Optimize image using sigmoid constraint + variance-only optimization.

    This method combines:
    1. Hard pixel bounds via sigmoid: x = pixel_min + (pixel_max - pixel_min) * sigmoid(θ)
    2. Soft constraint on logf_mean via penalty: loss = -U + λ * (logf_mean - target)²

    The goal is to maximize utility by increasing logf_var (model uncertainty) while
    keeping logf_mean (predicted response) approximately constant.

    Parameters
    ----------
    model_active : GPModel
        Current fitted GP model
    imgs_train : torch.Tensor
        Full training image dataset
    x_idx_best : torch.Tensor
        Index of best initial image
    mask : torch.Tensor
        Boolean mask for pixels to optimize
    u2d : torch.Tensor
        Utility batch for all remaining images
    max_r_cap : int
        Maximum spike count for utility computation
    lr : float
        Learning rate for L-BFGS optimizer
    max_iter : int
        Maximum number of L-BFGS iterations
    line_search_fn : str or None
        Line search function ('strong_wolfe', 'armijo', or None)
    mean_penalty_weight : float
        Weight for logf_mean constraint penalty (default: 100.0)
    return_logf_moments : bool
        Whether to return logf_mean and logf_var

    Returns
    -------
    dict
        Dictionary with keys: optimized_img, img_idx, utility_batch, start_gradient,
        end_gradient, U_initial, U_final, lr, line_search_fn, max_iter,
        logf_mean_initial, logf_mean_final, logf_var_initial, logf_var_final,
        mean_penalty_initial, mean_penalty_final
    """
    print("\n=== Pixel-wise Sigmoid + Variance-Only Optimization ===")
    print(f"Mean penalty weight: {mean_penalty_weight}")

    # Setup (standard pattern)
    initial_img = imgs_train[x_idx_best].clone()
    initial_unmasked = initial_img[~mask].clone()
    initial_masked = initial_img[mask].clone()

    # Get dataset-wide pixel bounds
    pixel_min = imgs_train.min()
    pixel_max = imgs_train.max()
    print(f"Dataset pixel bounds: [{pixel_min.item():.6f}, {pixel_max.item():.6f}]")
    print(f"Initial masked pixel range: [{initial_masked.min().item():.6f}, {initial_masked.max().item():.6f}]")

    # Transform initial pixels to unconstrained space via logit
    p = (initial_masked - pixel_min) / (pixel_max - pixel_min)
    p = torch.clamp(p, 1e-7, 1 - 1e-7)  # Numerical safety: avoid log(0) or log(∞)
    θ = torch.log(p / (1 - p))  # logit transformation
    θ = θ.detach().requires_grad_(True)

    # Compute initial utility, gradients, and logf_mean target
    with torch.enable_grad():
        x_masked_init = pixel_min + (pixel_max - pixel_min) * torch.sigmoid(θ)

        x_full_init = initial_img.clone()
        x_full_init[mask] = x_masked_init

        # ALWAYS compute logf moments to get target logf_mean
        U_init, logf_mean_init, logf_var_init = compute_utility_single_image(
            model_active, x_full_init, max_r_cap, return_logf_moments=True)

        # Store target logf_mean (detach to prevent gradient flow)
        logf_mean_target = logf_mean_init.detach().clone()

        dU_dtheta_init, dU_dx_init = torch.autograd.grad(
            U_init,
            [θ, x_masked_init])

    print(f"Initial utility: {U_init.item():.6f}")
    print(f"Initial logf_mean (target): {logf_mean_target.item():.6f}")
    print(f"Initial logf_var: {logf_var_init.item():.6f}")
    print(f"Initial ||∇U||: {torch.linalg.norm(dU_dx_init).item():.6f}")

    # Create optimizer (standard LBFGS configuration)
    optimizer = torch.optim.LBFGS(
        [θ],
        lr=lr,
        max_iter=max_iter,
        history_size=500,
        line_search_fn=line_search_fn,
        armijo_c1=1e-3,
        armijo_rho=0.7,
        armijo_min_step_size=1e-5,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        verbose=False
    )

    # NaN gradient tracking
    nan_grad_count = [0]
    max_nan_grad_allowed = 50

    # Define closure function with variance-only penalty
    def closure():
        optimizer.zero_grad()

        with torch.enable_grad():
            # Sigmoid transformation: θ → x_masked
            x_masked_constrained = pixel_min + (pixel_max - pixel_min) * torch.sigmoid(θ)

            # Reconstruct full image
            x_full = torch.empty_like(initial_img)
            x_full[~mask] = initial_unmasked
            x_full[mask] = x_masked_constrained

            # Compute utility and logf moments
            U, logf_mean, logf_var = compute_utility_single_image(
                model_active, x_full, max_r_cap, return_logf_moments=True)

            # Compute penalty for logf_mean deviation
            mean_penalty = ((logf_mean - logf_mean_target) ** 2)

        # Handle invalid utility values
        if torch.isnan(U) or torch.isinf(U):
            return torch.tensor(1e10, dtype=U.dtype, device=U.device, requires_grad=True)

        # Loss: negate utility + penalty for mean deviation
        loss = -U + mean_penalty_weight * mean_penalty
        loss.backward()

        # Handle invalid gradients
        if θ.grad is not None:
            if torch.isnan(θ.grad).any() or torch.isinf(θ.grad).any():
                nan_grad_count[0] += 1
                if nan_grad_count[0] <= max_nan_grad_allowed:
                    optimizer.zero_grad()
                    return torch.tensor(1e10, dtype=loss.dtype, device=loss.device, requires_grad=False)
                else:
                    raise RuntimeError(f"[Closure] NaN/Inf gradient exceeded limit: {nan_grad_count[0]}")
        else:
            raise ValueError("[Closure] θ.grad is None after backward pass")

        return loss

    # Run optimization
    optimizer.step(closure)

    # Compute final state and gradients
    with torch.enable_grad():
        x_masked_final = pixel_min + (pixel_max - pixel_min) * torch.sigmoid(θ)

        x_full_final = initial_img.clone()
        x_full_final[mask] = x_masked_final

        # Final state: always compute logf moments
        U_final, logf_mean_final, logf_var_final = compute_utility_single_image(
            model_active, x_full_final, max_r_cap, return_logf_moments=True)

        # Compute final penalty
        mean_penalty_final = ((logf_mean_final - logf_mean_target) ** 2)

        dU_dtheta_final, dU_dx_final = torch.autograd.grad(
            U_final,
            [θ, x_masked_final],
            allow_unused=True)

        if dU_dtheta_final is None:
            dU_dtheta_final = torch.zeros_like(θ)

    # Print diagnostics
    print(f"\nFinal utility: {U_final.item():.6f}")
    print(f"Utility improvement: {(U_final.item() - U_init.item()):.6f} ({((U_final.item() - U_init.item()) / abs(U_init.item()) * 100):.2f}%)")
    print(f"Final ||∇U||: {torch.linalg.norm(dU_dx_final).item():.6f}")
    print(f"Final masked pixel range: [{x_masked_final.min().item():.6f}, {x_masked_final.max().item():.6f}]")
    print(f"\nlogf_mean: {logf_mean_init.item():.6f} → {logf_mean_final.item():.6f} (Δ = {(logf_mean_final - logf_mean_init).item():.6f})")
    print(f"logf_var:  {logf_var_init.item():.6f} → {logf_var_final.item():.6f} (Δ = {(logf_var_final - logf_var_init).item():.6f})")
    print(f"Mean penalty: {mean_penalty_final.item():.6e}")
    print(f"Constraints satisfied: pixels ∈ [{pixel_min.item():.6f}, {pixel_max.item():.6f}]")
    if nan_grad_count[0] > 0:
        print(f"Warning: {nan_grad_count[0]} NaN/Inf gradient occurrences during optimization")

    # Prepare return dictionary (standard format + logf moments)
    optimized_img = x_full_final

    return {
        'optimized_img': optimized_img,
        'img_idx': x_idx_best[None] if x_idx_best.dim() == 0 else x_idx_best,
        'utility_batch': u2d,
        'start_gradient': dU_dx_init.squeeze(0) if dU_dx_init.dim() > 1 else dU_dx_init,
        'end_gradient': dU_dx_final.squeeze(0) if dU_dx_final.dim() > 1 else dU_dx_final,
        'U_initial': U_init.item() if isinstance(U_init, torch.Tensor) else U_init,
        'U_final': U_final.item() if isinstance(U_final, torch.Tensor) else U_final,
        'lr': lr,
        'line_search_fn': line_search_fn,
        'max_iter': max_iter,
        'logf_mean_initial': logf_mean_init.item() if logf_mean_init is not None else None,
        'logf_mean_final': logf_mean_final.item() if logf_mean_final is not None else None,
        'logf_var_initial': logf_var_init.item() if logf_var_init is not None else None,
        'logf_var_final': logf_var_final.item() if logf_var_final is not None else None,
        'logf_mean_target': logf_mean_target.item(),
        'mean_penalty_weight': mean_penalty_weight,
        'mean_penalty_initial': 0.0,  # Initial penalty is zero by definition
        'mean_penalty_final': mean_penalty_final.item(),
    }

def _optimize_mean_only_gray_init(
    model_active,
    imgs_train,
    x_idx_best,
    mask,
    u2d,
    max_r_cap,
    lr,
    max_iter,
    line_search_fn,
    pixel_min,
    pixel_max,
    return_logf_moments=False,
):
    """
    Optimize image with mean-only constraint starting from uniform gray.

    This method:
    1. Initializes all masked pixels to gray (mean intensity)
    2. Constrains mean via reparameterization: x = μ_gray + (θ - θ.mean())
    3. Lets variance evolve freely (no std constraint)
    4. Applies hard clipping to [pixel_min, pixel_max] post-optimization

    Tests: "Can we create informative structure (variance) while maintaining mean brightness?"

    Parameters
    ----------
    model_active : GPModel
        Current fitted GP model
    imgs_train : torch.Tensor
        Full training image dataset
    x_idx_best : torch.Tensor
        Index of best initial image
    mask : torch.Tensor
        Boolean mask for pixels to optimize
    u2d : torch.Tensor
        Utility batch for all remaining images
    max_r_cap : int
        Maximum spike count for utility computation
    lr : float
        Learning rate for L-BFGS optimizer
    max_iter : int
        Maximum number of L-BFGS iterations
    line_search_fn : str or None
        Line search function ('strong_wolfe', 'armijo', or None)
    pixel_min : float
        Minimum pixel value for hard clipping
    pixel_max : float
        Maximum pixel value for hard clipping
    return_logf_moments : bool
        Whether to return logf_mean and logf_var

    Returns
    -------
    dict
        Dictionary with optimization results
    """
    print("\n=== Mean-Only Gray Initialization ===")

    # Setup
    initial_img = imgs_train[x_idx_best].clone()
    initial_unmasked = initial_img[~mask].clone()
    initial_masked = initial_img[mask].clone()

    # Compute gray value (target mean)
    μ_gray = initial_masked.mean().item()
    print(f"Target mean (gray): μ = {μ_gray:.6f}")

    # Initialize from uniform gray
    gray_masked = torch.full_like(initial_masked, μ_gray)
    θ = gray_masked.clone().detach().requires_grad_(True)
    print(f"Initial std: {gray_masked.std().item():.6f} (uniform)")

    # Get initial state
    with torch.enable_grad():
        θ_mean_init = θ.mean()
        x_masked_init = μ_gray + (θ - θ_mean_init)

        x_full_init = initial_img.clone()
        x_full_init[mask] = x_masked_init

        # Compute initial utility
        if return_logf_moments:
            U_init, logf_mean_init, logf_var_init = compute_utility_single_image(
                model_active, x_full_init, max_r_cap, return_logf_moments=True)
        else:
            U_init = compute_utility_single_image(model_active, x_full_init, max_r_cap)
            logf_mean_init, logf_var_init = None, None

        dU_dtheta_init, dU_dx_init = torch.autograd.grad(
            U_init,
            [θ, x_masked_init])

    print(f"Initial utility: {U_init.item():.6f}")
    print(f"Initial ||∇U||: {torch.linalg.norm(dU_dx_init).item():.6f}")

    # Create optimizer
    optimizer = torch.optim.LBFGS(
        [θ],
        lr=lr,
        max_iter=max_iter,
        history_size=500,
        line_search_fn=line_search_fn,
        armijo_c1=1e-3,
        armijo_rho=0.7,
        armijo_min_step_size=1e-5,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        verbose=False
    )

    # Track NaN gradients
    nan_grad_count = [0]
    max_nan_grad_allowed = 50

    # Closure function
    def closure():
        optimizer.zero_grad()

        with torch.enable_grad():
            # Mean-only reparameterization (NO std constraint)
            θ_mean = θ.mean()
            x_masked_constrained = μ_gray + (θ - θ_mean)

            # Reconstruct full image
            x_full = torch.empty_like(initial_img)
            x_full[~mask] = initial_unmasked
            x_full[mask] = x_masked_constrained

            # Compute utility
            U = compute_utility_single_image(model_active, x_full, max_r_cap)

        # Handle invalid values
        if torch.isnan(U) or torch.isinf(U):
            return torch.tensor(1e10, dtype=U.dtype, device=U.device, requires_grad=True)

        loss = -U
        loss.backward()

        # Handle invalid gradients
        if θ.grad is not None:
            if torch.isnan(θ.grad).any() or torch.isinf(θ.grad).any():
                nan_grad_count[0] += 1
                if nan_grad_count[0] <= max_nan_grad_allowed:
                    optimizer.zero_grad()
                    return torch.tensor(1e10, dtype=loss.dtype, device=loss.device, requires_grad=False)
                else:
                    raise RuntimeError(f"[Closure] NaN/Inf gradient exceeded limit: {nan_grad_count[0]}")
        else:
            raise ValueError("[Closure] θ.grad is None after backward pass")

        return loss

    # Run optimization
    print(f"Running L-BFGS with max_iter={max_iter}...")
    optimizer.step(closure)

    # Get final state BEFORE clipping
    with torch.no_grad():
        θ_mean_final = θ.mean()
        x_masked_final = μ_gray + (θ - θ_mean_final)

        # Check constraint satisfaction before clipping
        μ_before_clip = x_masked_final.mean().item()
        σ_before_clip = x_masked_final.std().item()

    # Apply hard clipping
    pix_min = pixel_min if pixel_min is not None else imgs_train.min().item()
    pix_max = pixel_max if pixel_max is not None else imgs_train.max().item()

    n_outliers = ((x_masked_final < pix_min) | (x_masked_final > pix_max)).sum().item()
    x_masked_final = torch.clamp(x_masked_final, pix_min, pix_max)
    x_masked_final = x_masked_final.requires_grad_(True)

    # Recompute utility and gradients AFTER clipping
    with torch.enable_grad():
        x_full_final = initial_img.clone()
        x_full_final[mask] = x_masked_final

        if return_logf_moments:
            U_final, logf_mean_final, logf_var_final = compute_utility_single_image(
                model_active, x_full_final, max_r_cap, return_logf_moments=True)
        else:
            U_final = compute_utility_single_image(model_active, x_full_final, max_r_cap)
            logf_mean_final, logf_var_final = None, None

        dU_dtheta_final, dU_dx_final = torch.autograd.grad(
            U_final,
            [θ, x_masked_final],
            allow_unused=True)

        if dU_dtheta_final is None:
            dU_dtheta_final = torch.zeros_like(θ)

    # Report results
    μ_after_clip = x_masked_final.mean().item()
    σ_after_clip = x_masked_final.std().item()

    print(f"\nFinal utility: {U_final.item():.6f}")
    print(f"Utility improvement: {(U_final.item() - U_init.item()):.6f} ({((U_final.item() - U_init.item()) / abs(U_init.item()) * 100):.2f}%)")
    print(f"Final ||∇U||: {torch.linalg.norm(dU_dx_final).item():.6f}")
    print(f"\nMean constraint (before clip): {μ_before_clip:.6f} (target: {μ_gray:.6f}, error: {abs(μ_before_clip - μ_gray):.6e})")
    print(f"Mean after clipping: {μ_after_clip:.6f} (error: {abs(μ_after_clip - μ_gray):.6e})")
    print(f"Std before clip: {σ_before_clip:.6f} (FREE - no constraint)")
    print(f"Std after clip: {σ_after_clip:.6f}")
    print(f"Pixels clipped: {n_outliers}/{x_masked_final.numel()} outside [{pix_min:.6f}, {pix_max:.6f}]")
    if nan_grad_count[0] > 0:
        print(f"Warning: {nan_grad_count[0]} NaN/Inf gradient occurrences during optimization")

    # Return results
    optimized_img = x_full_final

    return {
        'optimized_img': optimized_img,
        'img_idx': x_idx_best[None] if x_idx_best.dim() == 0 else x_idx_best,
        'utility_batch': u2d,
        'start_gradient': dU_dx_init.squeeze(0) if dU_dx_init.dim() > 1 else dU_dx_init,
        'end_gradient': dU_dx_final.squeeze(0) if dU_dx_final.dim() > 1 else dU_dx_final,
        'U_initial': U_init.item() if isinstance(U_init, torch.Tensor) else U_init,
        'U_final': U_final.item() if isinstance(U_final, torch.Tensor) else U_final,
        'lr': lr,
        'line_search_fn': line_search_fn,
        'max_iter': max_iter,
        'logf_mean_initial': logf_mean_init.item() if logf_mean_init is not None else None,
        'logf_mean_final': logf_mean_final.item() if logf_mean_final is not None else None,
        'logf_var_initial': logf_var_init.item() if logf_var_init is not None else None,
        'logf_var_final': logf_var_final.item() if logf_var_final is not None else None,
        'mean_target': μ_gray,
        'std_initial': 0.0,  # Gray initialization
        'std_final': σ_after_clip,
        'n_pixels_clipped': n_outliers,
    }

def _optimize_norm_constraint(
    norm_type,
    model_active,
    imgs_train,
    x_idx_best,
    mask,
    u2d,
    max_r_cap,
    lr,
    max_iter,
    line_search_fn,
    pixel_min,
    pixel_max,
    apply_soft_compression,
    apply_hard_clipping,
    compression_steepness,
    compression_transition_width,
    return_logf_moments=False
):
    """
    Optimize image using Lp norm constraint (supports L2 and L4).

    This method fixes the Lp norm of masked pixels using reparameterization:
    x_masked = (θ / ||θ||_p) * target_norm

    Parameters
    ----------
    norm_type : str
        Type of norm constraint: 'L2' or 'L4'
    model_active : GPModel
        Current fitted GP model
    imgs_train : torch.Tensor
        Full training image dataset
    x_idx_best : torch.Tensor
        Index of best initial image
    mask : torch.Tensor
        Boolean mask for pixels to optimize
    u2d : torch.Tensor
        Utility batch for all remaining images
    max_r_cap : int
        Maximum spike count for utility computation
    lr : float
        Learning rate for L-BFGS optimizer
    max_iter : int
        Maximum number of L-BFGS iterations
    line_search_fn : str or None
        Line search function ('strong_wolfe', 'armijo', or None)
    pixel_min : float or None
        Minimum pixel value for post-processing bounds
    pixel_max : float or None
        Maximum pixel value for post-processing bounds
    apply_soft_compression : bool
        Whether to apply soft sigmoid compression post-processing
    apply_hard_clipping : bool
        Whether to apply hard clipping post-processing
    compression_steepness : float
        Steepness parameter for soft compression
    compression_transition_width : float
        Transition width parameter for soft compression

    Returns
    -------
    dict
        Dictionary with keys: optimized_img, img_idx, utility_batch, start_gradient,
        end_gradient, U_initial, U_final, lr, line_search_fn, max_iter
    """
    # Select norm computation based on type
    if norm_type == 'L2':
        compute_norm = lambda x: torch.linalg.norm(x)
        norm_symbol = '₂'
        norm_name = 'L2'
    elif norm_type == 'L4':
        compute_norm = lambda x: torch.sum(x**4).pow(0.25)
        norm_symbol = '₄'
        norm_name = 'L4'
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. Supported: 'L2', 'L4'")

    if return_logf_moments:
        logf_mean_init = None
        logf_var_init = None
        logf_mean_final = None
        logf_var_final = None


    print(f"\n=== {norm_name} Norm Constrained Optimization (Slim Version) ===")

    # Setup
    initial_img = imgs_train[x_idx_best].clone()
    initial_unmasked = initial_img[~mask].clone()
    initial_masked = initial_img[mask].clone()

    # Target norm from initial masked region
    norm_target = compute_norm(initial_masked).item()
    print(f"Target {norm_name} norm: {norm_target:.6f} (from initial masked region)")

    # Optimization variable θ (unconstrained)
    θ = initial_masked.clone().detach().requires_grad_(True)
    θ_norm_min = 0.01  # Numerical safety

    # Compute x₀: the STARTING POINT for optimization
    # KEY: x₀ ≠ initial_masked! We normalize to target norm
    with torch.no_grad():
        initial_norm = compute_norm(initial_masked)
        x0_masked = (initial_masked / initial_norm) * norm_target

    print(f"||initial_masked||{norm_symbol} (original): {initial_norm:.6f}")
    print(f"||x₀||{norm_symbol} (starting point): {compute_norm(x0_masked).item():.6f} = {norm_target:.6f}")

    # === Compute gradients AT x₀ (starting point) ===
    with torch.enable_grad():
        # Apply norm reparameterization: θ → x₀
        θ_norm_init = compute_norm(θ)
        x0_from_theta = (θ / θ_norm_init) * norm_target

        # Reconstruct full image with x₀ in masked region
        x_full_at_x0 = initial_img.clone()
        x_full_at_x0[mask] = x0_from_theta

        # Compute utility at x₀ via torch
        if return_logf_moments:
            U_init, logf_mean_init, logf_var_init = compute_utility_single_image(
                model_active, x_full_at_x0, max_r_cap, return_logf_moments=True)
        else:
            U_init = compute_utility_single_image(model_active, x_full_at_x0, max_r_cap)
            logf_mean_init, logf_var_init = None, None

        # Get gradients: ∂U/∂θ and ∂U/∂x both at x₀
        dU_dtheta_init, dU_dx_init = torch.autograd.grad(
            U_init,
            [θ, x0_from_theta])

    print(f"\nInitial: U (at x₀) = {U_init.item():.6f}, ||∂U/∂θ|| = {dU_dtheta_init.norm().item():.6e}, ||∂U/∂x|| = {dU_dx_init.norm().item():.6e}")

    # Create optimizer
    optimizer = torch.optim.LBFGS(
        [θ],
        lr=lr,
        max_iter=max_iter,
        history_size=100,
        line_search_fn=line_search_fn,
        armijo_c1=1e-3,
        armijo_rho=0.7,
        armijo_min_step_size=1e-5,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        verbose=False,
    )

    # Track NaN gradient occurrences
    nan_grad_count = [0]
    max_nan_grad_allowed = 50

    # Closure
    def closure():
        f"""Closure for L-BFGS with {norm_name} norm reparameterization"""
        # Numerical safety: clamp ||θ||
        with torch.no_grad():
            θ_norm_current = compute_norm(θ)
            if θ_norm_current < θ_norm_min:
                print(f"    [Warning] Clamping ||θ|| in closure")
                θ.data = (θ / θ_norm_current) * θ_norm_min

        optimizer.zero_grad()

        # Reparameterization: θ → x_masked (constraint satisfied automatically)
        with torch.enable_grad():
            θ_norm = compute_norm(θ)
            x_masked_constrained = (θ / θ_norm) * norm_target

            # Reconstruct full image
            x_full = torch.empty_like(initial_img)
            x_full[~mask] = initial_unmasked
            x_full[mask] = x_masked_constrained

            # Compute utility
            U = compute_utility_single_image(model_active, x_full, max_r_cap)

        # Handle invalid values
        if torch.isnan(U) or torch.isinf(U):
            print(f"    [Closure] U is {'NaN' if torch.isnan(U) else 'Inf'}, rejecting step")
            return torch.tensor(1e10, dtype=U.dtype, device=U.device, requires_grad=True)

        # L-BFGS minimizes, so return negative
        loss = -U
        loss.backward()

        # Handle invalid gradients
        if θ.grad is not None:
            if torch.isnan(θ.grad).any() or torch.isinf(θ.grad).any():
                nan_grad_count[0] += 1
                if nan_grad_count[0] <= max_nan_grad_allowed:
                    print(f"    [Closure] Warning: Gradient is NaN/Inf (occurrence {nan_grad_count[0]}/{max_nan_grad_allowed}), clearing and rejecting step")
                    optimizer.zero_grad()
                    return torch.tensor(1e10, dtype=loss.dtype, device=loss.device, requires_grad=False)
                else:
                    raise RuntimeError(
                        f"Gradient contains NaN or Inf after backward pass (occurred {nan_grad_count[0]} times)! "
                        f"U={U.item():.6f}, loss={loss.item():.6f}. "
                    )
        else:
            raise ValueError("[Closure] θ.grad is None")

        return loss

    # === SINGLE OPTIMIZATION CALL ===
    print(f"Running L-BFGS with max_iter={max_iter}...")
    optimizer.step(closure)

    # Get final state
    with torch.enable_grad():
        θ_norm_final = compute_norm(θ)
        x_masked_final = (θ / θ_norm_final) * norm_target

        if apply_soft_compression:
            # Get pixel bounds from dataset if not provided
            if pixel_min is None or pixel_max is None:
                pix_min = imgs_train.min().item()
                pix_max = imgs_train.max().item()
            else:
                pix_min = pixel_min
                pix_max = pixel_max

            print(f"\n{'='*60}")
            print(f"Applying soft sigmoid compression...")

            # Apply compression
            x_masked_final = soft_sigmoid_compress_pixels(
                x_masked_final.detach(),
                pixel_min=pix_min,
                pixel_max=pix_max,
                steepness=compression_steepness,
                transition_width=compression_transition_width,
                verbose=True
            )

            # Enable gradients for recomputation
            x_masked_final = x_masked_final.requires_grad_(True)

            # Report norm constraint violations
            norm_compressed = compute_norm(x_masked_final).item()
            norm_error = abs(norm_compressed - norm_target)

            print(f"    {norm_name} constraint violation after compression:")
            print(f"      ||x||{norm_symbol}: {norm_compressed:.6f} (target: {norm_target:.6f}, error: {norm_error:.6f})")
            print(f"{'='*60}\n")

        elif apply_hard_clipping:
            # Get pixel bounds from dataset if not provided
            if pixel_min is None or pixel_max is None:
                pix_min = imgs_train.min().item()
                pix_max = imgs_train.max().item()
            else:
                pix_min = pixel_min
                pix_max = pixel_max

            # Count outliers before clipping
            n_outliers = ((x_masked_final < pix_min) | (x_masked_final > pix_max)).sum().item()

            # Apply hard clipping
            x_masked_final = torch.clamp(x_masked_final.detach(), pix_min, pix_max)
            x_masked_final = x_masked_final.requires_grad_(True)

            # Report norm constraint violation
            norm_clipped = compute_norm(x_masked_final).item()
            norm_error = abs(norm_clipped - norm_target)
            print(f"  [Hard Clipping] {n_outliers} outliers clipped. {norm_name} norm error: Δ||x||{norm_symbol}={norm_error:.6f} ({(norm_error/norm_target)*100:.2f}%)")

        x_full_final = initial_img.clone()
        x_full_final[mask] = x_masked_final

        # Final state: conditionally compute logf moments
        if return_logf_moments:
            U_final, logf_mean_final, logf_var_final = compute_utility_single_image(
                model_active, x_full_final, max_r_cap, return_logf_moments=True)
        else:
            U_final = compute_utility_single_image(model_active, x_full_final, max_r_cap)
            logf_mean_final, logf_var_final = None, None
        dU_dtheta_final, dU_dx_final = torch.autograd.grad(U_final, [θ, x_masked_final], allow_unused=True)

        # Handle None when compression/clipping breaks computational graph
        if dU_dtheta_final is None:
            dU_dtheta_final = torch.zeros_like(θ)

    print(f"Final:   U = {U_final.item():.6f}, ||∂U/∂θ|| = {dU_dtheta_final.norm().item():.6e}, ||∂U/∂x|| = {dU_dx_final.norm().item():.6e}")
    print(f"Change:  ΔU = {U_final.item() - U_init.item():+.6f} ({(U_final.item()/U_init.item() - 1)*100:+.2f}%)")
    print(f"         Δ||∂U/∂θ|| = {dU_dtheta_final.norm().item() - dU_dtheta_init.norm().item():+.6e} ({((dU_dtheta_final.norm().item()/dU_dtheta_init.norm().item()) - 1)*100:+.2f}%)")
    print(f"         Δ||∂U/∂x|| = {dU_dx_final.norm().item() - dU_dx_init.norm().item():+.6e} ({((dU_dx_final.norm().item()/dU_dx_init.norm().item()) - 1)*100:+.2f}%)")

    # Verify constraint
    norm_actual = compute_norm(x_masked_final).item()
    norm_error = abs(norm_actual - norm_target) / norm_target if norm_target > 1e-8 else 0.0
    if norm_error > 1e-4:
        print(f"    [Warning] {norm_name} norm error: ||x||={norm_actual:.6f} (target: {norm_target:.6f}, error: {norm_error*100:.4f}%)")

    optimized_img = x_full_final

    return {
        'optimized_img': optimized_img,
        'img_idx': x_idx_best[None] if x_idx_best.dim() == 0 else x_idx_best,
        'utility_batch': u2d,
        'start_gradient': dU_dx_init.squeeze(0) if dU_dx_init.dim() > 1 else dU_dx_init,
        'end_gradient': dU_dx_final.squeeze(0) if dU_dx_final.dim() > 1 else dU_dx_final,
        'U_initial': U_init.item() if isinstance(U_init, torch.Tensor) else U_init,
        'U_final': U_final.item() if isinstance(U_final, torch.Tensor) else U_final,
        'lr': lr,
        'line_search_fn': line_search_fn,
        'max_iter': max_iter,
        'logf_mean_initial': logf_mean_init.item() if logf_mean_init is not None else None,
        'logf_mean_final': logf_mean_final.item() if logf_mean_final is not None else None,
        'logf_var_initial': logf_var_init.item() if logf_var_init is not None else None,
        'logf_var_final': logf_var_final.item() if logf_var_final is not None else None

    }

def _test_simple_optimization(
    model_active,
    imgs_train,
    x_idx_best,
    mask,
    u2d,
    max_r_cap,
    lr,
    max_iter,
    line_search_fn,
    return_logf_moments=False   
):
    """
    Test method: Direct pixel optimization without any constraints.

    This is the simplest possible optimization - directly optimize pixel values
    using L-BFGS with no reparameterization, no constraints, nothing.
    Optimizes x_masked directly (no intermediate θ variable).

    Parameters
    ----------
    model_active : GPModel
        Current fitted GP model
    imgs_train : torch.Tensor
        Full training image dataset
    x_idx_best : torch.Tensor
        Index of best initial image
    mask : torch.Tensor
        Boolean mask for pixels to optimize
    u2d : torch.Tensor
        Utility batch for all remaining images
    max_r_cap : int
        Maximum spike count for utility computation
    lr : float
        Learning rate for L-BFGS optimizer
    max_iter : int
        Maximum number of L-BFGS iterations
    line_search_fn : str or None
        Line search function ('strong_wolfe', 'armijo', or None)

    Returns
    -------
    dict
        Dictionary with keys: optimized_img, img_idx, utility_batch, start_gradient,
        end_gradient, U_initial, U_final, lr, line_search_fn, max_iter
    """
    if return_logf_moments:
        logf_mean_init = None
        logf_var_init = None
        logf_mean_final = None
        logf_var_final = None

    print("\n=== Test Simple Optimization (No Constraints) ===")

    # Setup
    initial_img = imgs_train[x_idx_best].clone()
    initial_unmasked = initial_img[~mask].clone()
    initial_masked = initial_img[mask].clone()

    print(f"Initial masked pixel range: [{initial_masked.min().item():.6f}, {initial_masked.max().item():.6f}]")

    # Optimization variable: optimize x_masked directly (no intermediate θ!)
    x_masked = initial_masked.clone().detach().requires_grad_(True)

    # Compute initial utility and gradients
    with torch.enable_grad():
        x_full_init = initial_img.clone()
        x_full_init[mask] = x_masked

        # U_init = compute_utility_single_image(model_active, x_full_init, max_r_cap)

        U_init, logf_mean_init, logf_var_init = compute_utility_single_image(
            model_active, x_full_init, max_r_cap, return_logf_moments=return_logf_moments)

        # Only one gradient needed (no θ variable)
        dU_dx_init = torch.autograd.grad(U_init, x_masked)[0]

    print(f"Initial utility: {U_init.item():.6f}")
    print(f"Initial ||∇U||: {torch.linalg.norm(dU_dx_init).item():.6f}")

    # Create optimizer
    optimizer = torch.optim.LBFGS(
        [x_masked],
        lr=lr,
        max_iter=max_iter,
        history_size=500,
        line_search_fn=line_search_fn,
        armijo_c1=1e-3,
        armijo_rho=0.7,
        armijo_min_step_size=1e-5,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        verbose=False
    )

    # Track NaN gradient occurrences
    nan_grad_count = [0]
    max_nan_grad_allowed = 50

    # Closure
    def closure():
        optimizer.zero_grad()

        # Reconstruct full image with current x_masked
        x_full = torch.empty_like(initial_img)
        x_full[~mask] = initial_unmasked
        x_full[mask] = x_masked

        # Compute utility
        U = compute_utility_single_image(model_active, x_full, max_r_cap)



        # Handle invalid utility values
        if torch.isnan(U) or torch.isinf(U):
            return torch.tensor(1e10, dtype=U.dtype, device=U.device, requires_grad=True)

        # Loss (LBFGS minimizes, so negate utility)
        loss = -U
        loss.backward()

        # Handle invalid gradients
        if x_masked.grad is not None:
            if torch.isnan(x_masked.grad).any() or torch.isinf(x_masked.grad).any():
                nan_grad_count[0] += 1
                if nan_grad_count[0] <= max_nan_grad_allowed:
                    optimizer.zero_grad()
                    return torch.tensor(1e10, dtype=loss.dtype, device=loss.device, requires_grad=False)
                else:
                    raise RuntimeError(f"[Closure] NaN/Inf gradient exceeded limit: {nan_grad_count[0]}")
        else:
            raise ValueError("[Closure] x_masked.grad is None after backward pass")

        return loss

    # Run optimization
    optimizer.step(closure)

    # Compute final state and gradients
    with torch.enable_grad():
        x_full_final = initial_img.clone()
        x_full_final[mask] = x_masked

        # U_final = compute_utility_single_image(model_active, x_full_final, max_r_cap)

        U_final, logf_mean_final, logf_var_final = compute_utility_single_image(
            model_active, x_full_final, max_r_cap, return_logf_moments=return_logf_moments)

        # Only one gradient needed
        dU_dx_final = torch.autograd.grad(U_final, x_masked, allow_unused=True)[0]

        if dU_dx_final is None:
            dU_dx_final = torch.zeros_like(x_masked)

    # Print diagnostics
    print(f"Final utility: {U_final.item():.6f}")
    print(f"Utility improvement: {(U_final.item() - U_init.item()):.6f} ({((U_final.item() - U_init.item()) / abs(U_init.item()) * 100):.2f}%)")
    print(f"Final ||∇U||: {torch.linalg.norm(dU_dx_final).item():.6f}")
    print(f"Final masked pixel range: [{x_masked.min().item():.6f}, {x_masked.max().item():.6f}]")
    if nan_grad_count[0] > 0:
        print(f"Warning: {nan_grad_count[0]} NaN/Inf gradient occurrences during optimization")

    # Prepare optimized image
    optimized_img = x_full_final

    return {
        'optimized_img': optimized_img,
        'img_idx': x_idx_best[None] if x_idx_best.dim() == 0 else x_idx_best,
        'utility_batch': u2d,
        'start_gradient': dU_dx_init.squeeze(0) if dU_dx_init.dim() > 1 else dU_dx_init,
        'end_gradient': dU_dx_final.squeeze(0) if dU_dx_final.dim() > 1 else dU_dx_final,
        'U_initial': U_init.item() if isinstance(U_init, torch.Tensor) else U_init,
        'U_final': U_final.item() if isinstance(U_final, torch.Tensor) else U_final,
        'lr': lr,
        'line_search_fn': line_search_fn,
        'max_iter': max_iter,
        'logf_mean_initial': logf_mean_init.item() if logf_mean_init is not None else None,
        'logf_mean_final': logf_mean_final.item() if logf_mean_final is not None else None,
        'logf_var_initial': logf_var_init.item() if logf_var_init is not None else None,
        'logf_var_final': logf_var_final.item() if logf_var_final is not None else None

    }

def batch_utility_w_grad(
    model_active,
    imgs_train,
    remaining_active_idx,
    max_r_cap=100,
    test_rms_constraint=False,
    test_L2norm_constraint=False,
    test_L4norm_constraint=False,
    test_rms_and_px_constraint=False,
    test_mean_only_and_px_constraint=False,
    test_rms_constraint_scheduled=False,
    test_px_only_lbfgsb_constraint=False,
    test_px_sigmoid_constraint=False,
    test_px_sigmoid_variance_only=False,
    test_mean_only_gray_init=False,
    test_simple_optimization=False,
    max_iter=20,
    lr=1e4,
    line_search_fn='strong_wolfe', # or 'armijo' (implemented by Pietro)
    clip_pixel_values=False,
    pixel_min=None,
    pixel_max=None,
    penalty_lambda=0.001,
    penalty_alpha=5.0,
    threshold_fraction=0.99,
    interior_point_params=None,
    max_lr_increases=3,
    min_utility_increase_fraction=0.20,
    apply_soft_compression=False,
    compression_steepness=1.0,
    compression_transition_width=0.95,
    apply_hard_clipping=False,
    mean_penalty_weight=100.0,
    initialize_from_gray=False,
    return_logf_moments=False,
    verbose=True
):
    """
    Compute Utility of all images, best image and gradient of utility w.r.t. the best image.

    Slim version: Supports RMS and L2 norm constraint optimization with single optimizer.step() call.

    Parameters:
    -----------
    model_active : GPModel
        Current fitted GP model
    imgs_train : torch.Tensor
        Full training image dataset
    remaining_active_idx : torch.Tensor
        Indices of candidate images
    max_r_cap : int
        Maximum spike count for utility computation
    test_rms_constraint : bool
        If True, perform RMS-constrained optimization (fixes mean and std)
    test_L2norm_constraint : bool
        If True, perform L2 norm-constrained optimization (fixes ||x||₂)
    test_L4norm_constraint : bool
        If True, perform L4 norm-constrained optimization (fixes ||x||₄ = (Σxᵢ⁴)^(1/4))
    test_rms_and_px_constraint : bool
        If True, perform RMS-constrained optimization WITH soft pixel bounds penalty.
        Combines RMS reparameterization (fixes mean/std) with softplus penalty function
        to prevent individual pixels from exceeding dataset bounds. Requires pixel_min
        and pixel_max to be provided.
    test_mean_only_and_px_constraint : bool
        If True, perform MEAN-ONLY constrained optimization WITH soft pixel bounds penalty.
        Fixes only the mean (not std) via simplified reparameterization: x = μ + (θ - θ.mean()).
        Standard deviation is free to vary during optimization. Uses same softplus penalty
        as test_rms_and_px_constraint. Requires pixel_min and pixel_max to be provided.
    test_rms_constraint_scheduled : bool
        If True, perform RMS-constrained optimization with adaptive learning rate scheduler.
        Uses the same RMS reparameterization as test_rms_constraint but with automatic
        lr adjustment. Starts with initial lr, runs 5 L-BFGS iterations, checks if utility
        increased by min_utility_increase_fraction. If not, discards results, multiplies
        lr by 10, and retries from initial state. Continues up to max_lr_increases times.
        Returns the first successful result or the best result if all attempts fail.
    test_px_sigmoid_constraint : bool
        If True, perform pixel-wise sigmoid-constrained optimization. Uses sigmoid
        reparameterization to automatically enforce hard bounds on each pixel individually.
        Transformation: x = pixel_min + (pixel_max - pixel_min) * sigmoid(θ), where θ is
        the unconstrained optimization variable. This guarantees all pixels remain strictly
        within [pixel_min, pixel_max] throughout optimization without requiring penalties
        or projections. Uses dataset-wide pixel bounds (imgs_train.min(), imgs_train.max()).
    max_iter : int
        Number of L-BFGS internal iterations per step() call.
        Note: When test_rms_constraint_scheduled=True, this is fixed to 5 regardless of
        the value provided.
    lr : float
        Learning rate for L-BFGS optimizer
    line_search_fn : str or None
        Line search function for L-BFGS ('armijo', 'strong_wolfe', or None)
    pixel_min : float or None
        Dataset minimum pixel value (e.g., 1st percentile) for soft bounds constraint.
        If None, pixel bounds penalty is disabled. Used with pixel_max.
    pixel_max : float or None
        Dataset maximum pixel value (e.g., 99th percentile) for soft bounds constraint.
        If None, pixel bounds penalty is disabled. Used with pixel_min.
    penalty_lambda : float
        Weight of pixel bounds penalty in loss function (default: 0.001).
        Higher values enforce bounds more strictly. Should be tuned based on P/U ratio.
        Only used if pixel_min/max provided.
    penalty_alpha : float
        Steepness of softplus penalty transition (default: 5.0).
        Higher values create sharper penalty near boundaries. Only used if pixel_min/max provided.
    threshold_fraction : float
        Fraction of pixel range considered "safe zone" with zero penalty (default: 0.9).
        Creates a penalty-free region in inner threshold_fraction of bounds.
        Range: (0, 1) where 0.9 = 90% safe, 0.95 = 95% safe, etc.
        Only used if pixel_min/max provided.
    interior_point_params : dict or None
        Parameters for interior point (log barrier) constraint optimization.
        If provided, enables RMS-constrained optimization WITH interior point log barrier
        for HARD pixel bounds enforcement (no violations allowed).

        Required dictionary keys:
            'enabled': bool - If True, use interior point method (default: False)
            'n_outer': int - Number of outer barrier iterations (default: 5)
            't_init': float - Initial barrier parameter (default: 1.0)
            'mu': float - Barrier growth factor (default: 10.0)
            'margin_pct': float - Safety margin as % of pixel range (default: 0.01 = 1%)

        Example:
            interior_point_params = {
                'enabled': True,
                'n_outer': 5,
                't_init': 1.0,
                'mu': 10.0,
                'margin_pct': 0.01
            }

        Requires pixel_min and pixel_max to be provided.

        Interior Point Method Characteristics:
        - HARD constraints: Violations mathematically impossible
        - Nested optimization: Outer loop schedules barrier strength
        - Barrier schedule: t_k = t_init * mu^k
        - Initial point: Automatically shrunk to safe interior if needed
        - Gradient: Non-zero everywhere (guides toward constrained optimum)
        - RMS preserved: Infeasible RMS transform steps automatically rejected
    max_lr_increases : int
        Maximum number of times to increase learning rate (default: 3).
        Only used when test_rms_constraint_scheduled=True.
        Total attempts = max_lr_increases + 1 (e.g., if 3: tries lr, lr×10, lr×100, lr×1000).
        Each attempt runs 5 L-BFGS iterations and checks success criterion.
    min_utility_increase_fraction : float
        Minimum fractional utility increase required for success (default: 0.20 = 20%).
        Only used when test_rms_constraint_scheduled=True.
        Success criterion: (U_final - U_initial) / |U_initial| >= min_utility_increase_fraction.
        If not met after 5 iterations, lr is multiplied by 10 and optimization restarts
        from initial state.
    apply_soft_compression : bool
        If True, apply piecewise soft compression to optimized pixels AFTER optimization
        to bring outliers into valid range [pixel_min, pixel_max]. Only applies when
        test_rms_constraint_scheduled=True. Uses piecewise exponential compression that
        PRESERVES pixels in the safe zone (unchanged) while smoothly compressing outliers
        near boundaries. Note: This WILL violate the RMS constraints slightly (mean and
        std will change), but ensures all pixels are displayable. Pixel bounds are
        auto-detected from imgs_train if not provided. Default: False (no post-processing).
    compression_steepness : float
        Controls compression rate for outliers (default: 1.0).
        Only used if apply_soft_compression=True.
        - Higher values (e.g., 2.0): Gentler compression, outliers compressed slowly
        - Lower values (e.g., 0.5): Aggressive compression, outliers pulled in quickly
        - Recommended range: [0.5, 2.0]
        Exponential decay rate for approaching pixel bounds.
    compression_transition_width : float
        Fraction of valid range that is "safe zone" with no compression (default: 0.95).
        Only used if apply_soft_compression=True.
        - 0.95 means inner 95% of range is preserved unchanged
        - 0.90 means inner 90% of range is preserved unchanged
        - Range: (0, 1) where higher = larger safe zone
        Pixels in safe zone remain exactly as optimized, only outliers are compressed.
    apply_hard_clipping : bool
        If True, apply simple hard clipping (torch.clamp) to optimized pixels AFTER optimization
        to force them into valid range [pixel_min, pixel_max]. Only applies when
        test_rms_constraint_scheduled=True. This is the simplest approach: any pixel above
        pixel_max becomes pixel_max, any below pixel_min becomes pixel_min. Note: This WILL
        violate RMS constraints (mean and std will change) and destroys information about
        relative ordering among outliers (all become same value). Use for comparison with
        soft compression. Cannot be used simultaneously with apply_soft_compression.
        Default: False (no clipping).

    Returns:
    --------
    dict : Dictionary containing:
        - 'optimized_img': Final optimized image tensor (if optimization enabled)
        - 'img_idx': Index of best initial image
        - 'utility_batch': Utility values for all candidates
        - 'start_gradient': Initial gradient (if optimization enabled)
        - 'end_gradient': Final gradient (if optimization enabled)
        - 'U_initial': Initial utility value (if optimization enabled)
        - 'U_final': Final utility value (if optimization enabled)
        - 'lr': Learning rate used (if optimization enabled)
        - 'line_search_fn': Line search function used (if optimization enabled)
        - 'max_iter': Max iterations used (if optimization enabled)

        Additional fields when test_rms_constraint_scheduled=True:
        - 'lr_attempts': List of all lr values tried
        - 'U_attempts': List of final utilities at each attempt
        - 'grad_norm_attempts': List of gradient norms at each attempt
        - 'success_attempt': Index of successful attempt (or -1 if none)
        - 'best_attempt': Index of attempt with best utility
        - 'scheduler_params': Dict with max_lr_increases and min_utility_increase_fraction
    """

    # ==========================================================================
    # PART 1: Initial Utility Estimation (EXACT COPY from lines 2231-2275)
    # ==========================================================================

    r_masked = torch.arange(0, max_r_cap, dtype=TORCH_DTYPE, device=DEVICE)

    X_remaining = imgs_train[remaining_active_idx]

    kernfun     = model_active.kernfun
    xtilde      = model_active.xtilde

    theta = model_active.theta
    m_b   = model_active.m_b
    V_b   = model_active.V_b

    A = torch.exp(model_active.f_params['logA'])
    lambda0 = model_active.f_params['lambda0']

    C, mask, K_tilde_b, K_tilde_inv_b, B = GP_utils.get_final_K_vals(model_active)

    Kvec = kernfun(theta, X_remaining[:,mask], x2=None, C=C, dC=None, diag=True)

    K   = kernfun(theta, X_remaining[:,mask], xtilde[:,mask], C=C, dC=None, diag=False)
    K_b = K @ B
    KKtilde_inv_b = K_b @ K_tilde_inv_b

    # === Utility of all remaining images ===
    lambda_m_batch, lambda_var_batch = GP_utils.lambda_moments(
        X_remaining[:,mask], K_tilde_b, KKtilde_inv_b, Kvec, K_b, C, m_b, V_b, theta)

    # NOTE: These are mean and variance of log_f = A*lambda + lambda0
    #   - Not the log of mean and variance of f
    #   - Not the mean and variance of lambda
    logf_mean_batch = A*lambda_m_batch + lambda0
    logf_var_batch  = A**2 * lambda_var_batch

    # === Utility of all remaining images ===
    # Keep gradient tracking enabled (minimal overhead) for potential gradient computation later
    with torch.enable_grad():
        logf_mean_batch_grad = logf_mean_batch.clone().requires_grad_(True)
        logf_var_batch_grad  = logf_var_batch.clone().requires_grad_(True)

        u2d = nd_utility_old(logf_mean_batch_grad, logf_var_batch_grad, r_masked)
        print(f'[WARNING] Using old utility function')
        # U = nd_utility_new(mu=logf_mean, sigma2=logf_var, r_max=max_r_cap)

        i_best = u2d.argmax()  # Index of the best image in the utility vector
        x_idx_best = remaining_active_idx[i_best]  # Index of the best image in the original dataset

    # Initialize return variables (will be populated if test_rms_constraint=True)
    dU_dx_init = None

    # ==========================================================================
    # PART 2: RMS Optimization (Minimal Implementation)
    # ==========================================================================

    if test_rms_constraint:
        return _optimize_rms_constraint(
            model_active,
            imgs_train,
            x_idx_best,
            mask,
            u2d,
            max_r_cap,
            lr,
            max_iter,
            line_search_fn,
            pixel_min,
            pixel_max,
            apply_soft_compression,
            apply_hard_clipping,
            compression_steepness,
            compression_transition_width,
            return_logf_moments,
            verbose
        )

    elif test_px_sigmoid_constraint:
        return _optimize_px_sigmoid_constraint(
            model_active,
            imgs_train,
            x_idx_best,
            mask,
            u2d,
            max_r_cap,
            lr,
            max_iter,
            line_search_fn,
            initialize_from_gray,
            return_logf_moments
        )

    elif test_px_sigmoid_variance_only:
        return _optimize_px_sigmoid_variance_only(
            model_active,
            imgs_train,
            x_idx_best,
            mask,
            u2d,
            max_r_cap,
            lr,
            max_iter,
            line_search_fn,
            mean_penalty_weight,
            return_logf_moments
        )

    elif test_mean_only_gray_init:
        return _optimize_mean_only_gray_init(
            model_active,
            imgs_train,
            x_idx_best,
            mask,
            u2d,
            max_r_cap,
            lr,
            max_iter,
            line_search_fn,
            pixel_min,
            pixel_max,
            return_logf_moments
        )

    elif test_rms_constraint_scheduled:
        print("\n=== RMS-Constrained Optimization with LR Scheduler (Slim Version) ===")
        print(f"Scheduler parameters: max_lr_increases={max_lr_increases}, min_utility_increase={min_utility_increase_fraction*100:.0f}%")


        # Setup
        initial_img = imgs_train[x_idx_best].clone()
        initial_unmasked = initial_img[~mask].clone()
        initial_masked = initial_img[mask].clone()

        # Target constraints from initial image
        μ_target = initial_masked.mean().item()
        σ_target = initial_masked.std().item()
        print(f"Target constraints: μ = {μ_target:.6f}, σ = {σ_target:.6f}")
        print(f"Initial L2 norm of masked region: {torch.linalg.norm(initial_masked).item():.6f}")
        print(f"Max init masked pixel value: {initial_masked.max().item():.6f}, Min: {initial_masked.min().item():.6f}")

        # Numerical safety
        θ_std_min = 0.01

        # Get initial utility (before any optimization)
        θ_initial = initial_masked.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            θ_mean_init = θ_initial.mean()
            θ_std_init = θ_initial.std()
            x_masked_init = μ_target + σ_target * (θ_initial - θ_mean_init) / θ_std_init

            x_full_init = initial_img.clone()
            x_full_init[mask] = x_masked_init

            U_initial = compute_utility_single_image(model_active, x_full_init, max_r_cap)
            dU_dtheta_init, dU_dx_init = torch.autograd.grad(U_initial, [θ_initial, x_masked_init])

        print(f"\nInitial utility (before optimization): U = {U_initial.item():.6f}")
        print(f"Initial gradients: ||∂U/∂θ|| = {dU_dtheta_init.norm().item():.6e}, ||∂U/∂x_masked|| = {dU_dx_init.norm().item():.6e}")

        # Tracking variables
        lr_attempts = []
        U_attempts = []
        grad_norm_attempts = []
        best_result = None
        best_utility = -float('inf')
        success_attempt = -1

        # Scheduling loop
        for attempt in range(max_lr_increases + 1):
            # Calculate current learning rate
            lr_current = lr * (10 ** attempt)
            lr_attempts.append(lr_current)

            print(f"\n{'='*60}")
            print(f"Attempt {attempt}/{max_lr_increases}: lr = {lr_current:.2e}")
            print(f"{'='*60}")

            # Reset θ to initial state for this attempt
            θ = initial_masked.clone().detach().requires_grad_(True)

            # Create fresh optimizer for this attempt with max_iter=5
            optimizer = torch.optim.LBFGS(
                [θ],
                lr=lr_current,
                max_iter=10,  # Fixed to 5 iterations per scheduler requirement
                history_size=200,
                line_search_fn=line_search_fn,
                armijo_c1=1e-3,
                armijo_rho=0.7,
                armijo_min_step_size=1e-5,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                verbose=False,
            )

            # Track NaN gradient occurrences for this attempt
            nan_grad_count = [0]
            max_nan_grad_allowed = 50

            # Closure function
            def closure():
                """Closure for L-BFGS with RMS reparameterization"""
                # Numerical safety: clamp std
                with torch.no_grad():
                    θ_std_current = θ.std()
                    if θ_std_current < θ_std_min:
                        print(f"    [Warning] Clamping θ std in closure")
                        θ.data = (θ - θ.mean()) * (θ_std_min / θ_std_current) + θ.mean()

                optimizer.zero_grad()

                # Reparameterization: θ → x_masked (constraints satisfied automatically)
                with torch.enable_grad():
                    θ_mean = θ.mean()
                    θ_std = θ.std()
                    x_masked_constrained = μ_target + σ_target * (θ - θ_mean) / θ_std

                    # Reconstruct full image
                    x_full = torch.empty_like(initial_img)
                    x_full[~mask] = initial_unmasked
                    x_full[mask] = x_masked_constrained

                    # Compute utility
                    U = compute_utility_single_image(model_active, x_full, max_r_cap)

                # Handle invalid values
                if torch.isnan(U) or torch.isinf(U):
                    print(f"    [Closure] U is {'NaN' if torch.isnan(U) else 'Inf'}, rejecting step")
                    return torch.tensor(1e10, dtype=U.dtype, device=U.device, requires_grad=True)

                # L-BFGS minimizes, so return negative
                loss = -U
                loss.backward()

                # Handle invalid gradients
                if θ.grad is not None:
                    if torch.isnan(θ.grad).any() or torch.isinf(θ.grad).any():
                        nan_grad_count[0] += 1
                        if nan_grad_count[0] <= max_nan_grad_allowed:
                            print(f"    [Closure] Warning: Gradient is NaN/Inf (occurrence {nan_grad_count[0]}/{max_nan_grad_allowed}), clearing and rejecting step")
                            optimizer.zero_grad()
                            return torch.tensor(1e10, dtype=loss.dtype, device=loss.device, requires_grad=False)
                        else:
                            raise RuntimeError(
                                f"Gradient contains NaN or Inf after backward pass (occurred {nan_grad_count[0]} times)! "
                                f"U={U.item():.6f}, loss={loss.item():.6f}. "
                            )
                else:
                    raise ValueError("[Closure] θ.grad is None")

                return loss

            # Run optimization for this attempt
            print(f"Running L-BFGS with max_iter=5...")
            optimizer.step(closure)

            # Get final state after this attempt
            with torch.enable_grad():
                θ_mean_final = θ.mean()
                θ_std_final = θ.std()
                x_masked_final = μ_target + σ_target * (θ - θ_mean_final) / θ_std_final

                # Optional pixel renormalization to projector bounds
                if clip_pixel_values:
                    if pixel_min is None or pixel_max is None:
                        raise ValueError("clip_pixel_values=True requires pixel_min and pixel_max")

                    # pixel_min = initial_masked.min().item() 
                    # pixel_max = initial_masked.max().item()

                    current_min = x_masked_final.min()
                    current_max = x_masked_final.max()

                    # Linear rescaling to [pixel_min, pixel_max]
                    x_masked_final = pixel_min + (x_masked_final - current_min) * \
                                     (pixel_max - pixel_min) / (current_max - current_min)

                # Optional soft sigmoid compression
                if apply_soft_compression:
                    # Get pixel bounds from dataset if not provided
                    if pixel_min is None or pixel_max is None:
                        pix_min = imgs_train.min().item()
                        pix_max = imgs_train.max().item()
                    else:
                        pix_min = pixel_min
                        pix_max = pixel_max

                    print(f"\n{'='*60}")
                    
                    print(f"Applying soft sigmoid compression...")
                    # Apply compression
                    x_masked_final = soft_sigmoid_compress_pixels(
                        x_masked_final.detach(),
                        pixel_min=pix_min,
                        pixel_max=pix_max,
                        steepness=compression_steepness,
                        transition_width=compression_transition_width,
                        verbose=True
                    )

                    # Enable gradients for recomputation
                    x_masked_final = x_masked_final.requires_grad_(True)

                    # Report RMS constraint violations
                    μ_compressed = x_masked_final.mean().item()
                    σ_compressed = x_masked_final.std().item()
                    μ_error = abs(μ_compressed - μ_target)
                    σ_error = abs(σ_compressed - σ_target)

                    print(f"    RMS constraint violations after compression:")
                    print(f"      Mean: {μ_compressed:.6f} (target: {μ_target:.6f}, error: {μ_error:.6f})")
                    print(f"      Std:  {σ_compressed:.6f} (target: {σ_target:.6f}, error: {σ_error:.6f})")
                    print(f"{'='*60}\n")

                elif apply_hard_clipping:
                    # Get pixel bounds from dataset if not provided
                    if pixel_min is None or pixel_max is None:
                        pix_min = imgs_train.min().item()
                        pix_max = imgs_train.max().item()
                    else:
                        pix_min = pixel_min
                        pix_max = pixel_max

                    # Count outliers and apply hard clipping
                    n_outliers = ((x_masked_final < pix_min) | (x_masked_final > pix_max)).sum().item()
                    x_masked_final = torch.clamp(x_masked_final.detach(), pix_min, pix_max)
                    x_masked_final = x_masked_final.requires_grad_(True)

                    # Report RMS violations
                    μ_clipped = x_masked_final.mean().item()
                    σ_clipped = x_masked_final.std().item()
                    print(f"  [Hard Clipping] {n_outliers} outliers clipped. RMS errors: Δμ={abs(μ_clipped - μ_target):.6f}, Δσ={abs(σ_clipped - σ_target):.6f}")

                x_full_final = initial_img.clone()
                x_full_final[mask] = x_masked_final

                U_final = compute_utility_single_image(model_active, x_full_final, max_r_cap)
                dU_dtheta_final, dU_dx_final = torch.autograd.grad(
                    U_final,
                    [θ, x_masked_final],
                    allow_unused=True)

                # Handle None when compression breaks computational graph
                if dU_dtheta_final is None:
                    dU_dtheta_final = torch.zeros_like(θ)


            # Track this attempt's results
            U_attempts.append(U_final.item())
            grad_norm_attempts.append(dU_dx_final.norm().item())

            # Calculate utility improvement
            utility_increase = U_final.item() - U_initial.item()
            # Handle edge case where U_initial might be zero or negative
            denominator = abs(U_initial.item()) if abs(U_initial.item()) > 1e-10 else 1e-10
            utility_increase_fraction = utility_increase / denominator

            # Print diagnostics for this attempt
            print(f"\nAttempt {attempt} results:")
            print(f"  Initial: U = {U_initial.item():.6f}")
            print(f"  Final:   U = {U_final.item():.6f}")
            print(f"  Change:  ΔU = {utility_increase:+.6f} ({utility_increase_fraction*100:+.2f}%)")
            print(f"  Gradients: ||∂U/∂θ|| = {dU_dtheta_final.norm().item():.6e}, ||∂U/∂x|| = {dU_dx_final.norm().item():.6e}")

            # Verify constraints
            μ_error = abs(x_masked_final.mean().item() - μ_target)
            σ_error = abs(x_masked_final.std().item() - σ_target)
            if μ_error > 1e-4 or σ_error > 1e-4:
                print(f"  [Warning] Final constraints not satisfied within tolerance: |Δμ|={μ_error:.6e}, |Δσ|={σ_error:.6e}")

            # L-BFGS diagnostics
            state = optimizer.state[optimizer._params[0]]
            n_iters = state.get('n_iter', 0)
            print(f"  L-BFGS iterations completed: {n_iters}")

            # Store result if it's the best so far
            if U_final.item() > best_utility:
                best_utility = U_final.item()
                best_result = {
                    'optimized_img': x_full_final.clone(),
                    'U_final': U_final.item(),
                    'dU_dx_final': dU_dx_final.clone(),
                    'dU_dtheta_final': dU_dtheta_final.clone(),
                    'x_masked_final': x_masked_final.clone(),
                    'lr_used': lr_current,
                    'attempt': attempt
                }

            # Check success criterion
            if utility_increase_fraction >= min_utility_increase_fraction:
                print(f"\n✓ SUCCESS! Utility increased by {utility_increase_fraction*100:.2f}% (≥ {min_utility_increase_fraction*100:.0f}% threshold)")
                success_attempt = attempt
                # Use the result from this successful attempt
                best_result = {
                    'optimized_img': x_full_final.clone(),
                    'U_final': U_final.item(),
                    'dU_dx_final': dU_dx_final.clone(),
                    'dU_dtheta_final': dU_dtheta_final.clone(),
                    'x_masked_final': x_masked_final.clone(),
                    'lr_used': lr_current,
                    'attempt': attempt
                }
                break
            else:
                print(f"\n✗ FAILED: Utility increased by {utility_increase_fraction*100:.2f}% (< {min_utility_increase_fraction*100:.0f}% threshold)")
                if attempt < max_lr_increases:
                    print(f"  → Discarding result and trying with lr × 10 = {lr_current * 10:.2e}")

        # Final summary
        print(f"\n{'='*60}")
        print(f"SCHEDULING SUMMARY")
        print(f"{'='*60}")
        if success_attempt >= 0:
            print(f"✓ Optimization succeeded at attempt {success_attempt} with lr = {best_result['lr_used']:.2e}")
        else:
            print(f"✗ All attempts exhausted. Using best result from attempt {best_result['attempt']} with lr = {best_result['lr_used']:.2e}")

        print(f"\nAll attempts:")
        for i, (lr_val, U_val) in enumerate(zip(lr_attempts, U_attempts)):
            marker = "✓" if i == success_attempt else ("*" if i == best_result['attempt'] else " ")
            print(f"  {marker} Attempt {i}: lr = {lr_val:.2e}, U_final = {U_val:.6f}")

        print(f"\nFinal result:")
        print(f"  Initial: U = {U_initial.item():.6f}")
        print(f"  Final:   U = {best_result['U_final']:.6f}")
        print(f"  Change:  ΔU = {best_result['U_final'] - U_initial.item():+.6f} ({(best_result['U_final']/U_initial.item() - 1)*100:+.2f}%)")
        print(f"  Final masked pixel stats: μ={best_result['x_masked_final'].mean().item():.6f}, σ={best_result['x_masked_final'].std().item():.6f}")
        print(f"  Max final masked pixel value: {best_result['x_masked_final'].max().item():.6f}, Min: {best_result['x_masked_final'].min().item():.6f}")

        # Return dictionary matching the format of test_rms_constraint
        return {
            'optimized_img': best_result['optimized_img'],
            'img_idx': x_idx_best[None] if x_idx_best.dim() == 0 else x_idx_best,
            'utility_batch': u2d,
            'start_gradient': dU_dx_init.squeeze(0) if dU_dx_init.dim() > 1 else dU_dx_init,
            'end_gradient': best_result['dU_dx_final'].squeeze(0) if best_result['dU_dx_final'].dim() > 1 else best_result['dU_dx_final'],
            'U_initial': U_initial.item() if isinstance(U_initial, torch.Tensor) else U_initial,
            'U_final': best_result['U_final'],
            'lr': best_result['lr_used'],  # The lr that was actually used
            'line_search_fn': line_search_fn,
            'max_iter': 5,  # Always 5 per scheduler design
            # Additional scheduler-specific fields
            'lr_attempts': lr_attempts,
            'U_attempts': U_attempts,
            'grad_norm_attempts': grad_norm_attempts,
            'success_attempt': success_attempt,
            'best_attempt': best_result['attempt'],
            'scheduler_params': {
                'max_lr_increases': max_lr_increases,
                'min_utility_increase_fraction': min_utility_increase_fraction
            }
        }

    elif test_rms_and_px_constraint:
        print("\n=== RMS + Pixel Bounds Constrained Optimization (Slim Version) ===")

        # Validate pixel bounds are provided
        if pixel_min is None or pixel_max is None:
            raise ValueError("test_rms_and_px_constraint=True requires pixel_min and pixel_max to be provided")

        print(f"Pixel bounds: [{pixel_min:.6f}, {pixel_max:.6f}]")
        print(f"Penalty parameters: λ={penalty_lambda:.2f}, α={penalty_alpha:.2f}")

        # Calculate and display safe zone boundaries
        range_width = pixel_max - pixel_min
        margin = (1 - threshold_fraction) * range_width / 2
        x_max_thresh = pixel_max - margin
        x_min_thresh = pixel_min + margin
        print(f"Safe zone (threshold={threshold_fraction:.2f}): [{x_min_thresh:.6f}, {x_max_thresh:.6f}]")
        print(f"  → Inner {threshold_fraction*100:.0f}% of range is penalty-free")
        print(f"  → Outer {(1-threshold_fraction)*100:.0f}% provides smooth warning gradient")

        # Setup
        initial_img = imgs_train[x_idx_best].clone()
        initial_unmasked = initial_img[~mask].clone()
        initial_masked = initial_img[mask].clone()

        # Target constraints from initial image
        μ_target = initial_masked.mean().item()
        σ_target = initial_masked.std().item()
        print(f"Target constraints: μ = {μ_target:.6f}, σ = {σ_target:.6f}")
        #print(f"Initial L2 norm of masked region: {torch.linalg.norm(initial_masked).item():.6f}")
        # DEBUG
        #print(f"Max init masked pixel value: {initial_masked.max().item():.6f}, Min: {initial_masked.min().item():.6f}")

        # Check initial pixel bounds violations
        n_pixels_exceeding_bounds = ((initial_masked > pixel_max) | (initial_masked < pixel_min)).sum().item()
        if n_pixels_exceeding_bounds > 0:
            print(f"Initial violations: {n_pixels_exceeding_bounds}/{initial_masked.numel()} pixels exceed bounds")

        # Analyze initial pixel distribution relative to safe zones
        n_pixels_in_safe_zone = ((initial_masked >= x_min_thresh) & (initial_masked <= x_max_thresh)).sum().item()
        n_pixels_in_upper_warning = (initial_masked > x_max_thresh).sum().item()
        n_pixels_in_lower_warning = (initial_masked < x_min_thresh).sum().item()
        n_pixels_total = initial_masked.numel()

        #print(f"\n=== Initial Pixel Distribution vs Safe Zone ===")
        #print(f"Total masked pixels: {n_pixels_total}")
        #print(f"Pixels in safe zone [{x_min_thresh:.3f}, {x_max_thresh:.3f}]: {n_pixels_in_safe_zone}/{n_pixels_total} ({100*n_pixels_in_safe_zone/n_pixels_total:.1f}%)")
        #print(f"Pixels in upper warning zone (>{x_max_thresh:.3f}): {n_pixels_in_upper_warning}/{n_pixels_total} ({100*n_pixels_in_upper_warning/n_pixels_total:.1f}%)")
        #print(f"Pixels in lower warning zone (<{x_min_thresh:.3f}): {n_pixels_in_lower_warning}/{n_pixels_total} ({100*n_pixels_in_lower_warning/n_pixels_total:.1f}%)")

        if n_pixels_in_upper_warning > 0:
            max_distance_upper = (initial_masked[initial_masked > x_max_thresh] - x_max_thresh).max().item()
            avg_distance_upper = (initial_masked[initial_masked > x_max_thresh] - x_max_thresh).mean().item()
            print(f"  Upper warning zone: max intrusion={max_distance_upper:.3f}, avg intrusion={avg_distance_upper:.3f}")

        if n_pixels_in_lower_warning > 0:
            max_distance_lower = (x_min_thresh - initial_masked[initial_masked < x_min_thresh]).max().item()
            avg_distance_lower = (x_min_thresh - initial_masked[initial_masked < x_min_thresh]).mean().item()
            print(f"  Lower warning zone: max intrusion={max_distance_lower:.3f}, avg intrusion={avg_distance_lower:.3f}")

        # Warning if too many pixels in warning zones
        pct_in_warning = 100 * (n_pixels_in_upper_warning + n_pixels_in_lower_warning) / n_pixels_total
        if pct_in_warning > 5.0:
            print(f"\n  ⚠ WARNING: {pct_in_warning:.1f}% of pixels are in warning zones!")
            print(f"  This creates penalty even though pixels are within bounds.")
            print(f"  The optimizer will reduce penalty by sacrificing utility.")
            print(f"  Recommendations:")
            if threshold_fraction < 0.99:
                print(f"    → Increase threshold_fraction from {threshold_fraction} to 0.99 or 0.995")
            else:
                print(f"    → threshold_fraction is already {threshold_fraction} (very high)")
            print(f"    → Reduce penalty_lambda from {penalty_lambda} to {penalty_lambda/10:.6f}")
            print(f"    → Or accept utility decrease as tradeoff for bounded pixels\n")

        # Optimization variable θ (unconstrained)
        θ = initial_masked.clone().detach().requires_grad_(True)
        θ_std_min = 0.01  # Numerical safety

        # Get initial state
        with torch.enable_grad():
            θ_mean_init = θ.mean()
            θ_std_init = θ.std()
            x_masked_init = μ_target + σ_target * (θ - θ_mean_init) / θ_std_init

            x_full_init = initial_img.clone()
            x_full_init[mask] = x_masked_init

            U_init = compute_utility_single_image(model_active, x_full_init, max_r_cap)
            P_init = compute_pixel_penalty(x_masked_init, pixel_min, pixel_max, penalty_alpha, threshold_fraction)

            dU_dtheta_init, dU_dx_init = torch.autograd.grad(
                U_init,
                [θ, x_masked_init])

        print(f"Initial: U={U_init.item():.6f}, P={P_init.item():.6f}, Loss={(-U_init + penalty_lambda * P_init).item():.6f}")

        # Check for penalty dominance and warn user
        weighted_penalty_utility_ratio = np.abs((penalty_lambda * P_init.item()) / abs(U_init.item()))
        if weighted_penalty_utility_ratio > 100:
            print(f"    [WARNING] Penalty dominates utility!")
            print(f"    Weighted penalty/utility ratio: {weighted_penalty_utility_ratio:.1f}:1")
            print(f"    The optimizer may sacrifice utility to reduce penalty.")
            recommended_lambda = penalty_lambda / (weighted_penalty_utility_ratio / 10)
            print(f"    Consider reducing penalty_lambda from {penalty_lambda} to ~{recommended_lambda:.6f}")
        elif weighted_penalty_utility_ratio < 0.01:
            print(f"    [WARNING] Utility dominates penalty!")
            print(f"    Weighted penalty/utility ratio: {weighted_penalty_utility_ratio:.6f}:1")
            print(f"    Pixel bounds may not be enforced effectively.")
            recommended_lambda = penalty_lambda / (weighted_penalty_utility_ratio / 0.1)
            print(f"    Consider increasing penalty_lambda from {penalty_lambda} to ~{recommended_lambda:.6f}")

        # Create optimizer
        optimizer = torch.optim.LBFGS(
            [θ],
            lr=lr,
            max_iter=max_iter,  # Internal L-BFGS iterations
            history_size=500,
            line_search_fn=line_search_fn,
            armijo_c1=1e-3,
            armijo_rho=0.7,
            armijo_min_step_size=1e-5,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            verbose=False,  # Set True for debugging
        )

        # Track NaN gradient occurrences
        nan_grad_count = [0]
        max_nan_grad_allowed = 50

        # Closure
        def closure():
            """Closure for L-BFGS with RMS reparameterization + pixel bounds penalty"""
            # Numerical safety: clamp std
            with torch.no_grad():
                θ_std_current = θ.std()
                if θ_std_current < θ_std_min:
                    print(f"    [Warning] Clamping θ std in closure")
                    θ.data = (θ - θ.mean()) * (θ_std_min / θ_std_current) + θ.mean()

            optimizer.zero_grad()

            # Reparameterization: θ → x_masked (RMS constraints satisfied automatically)
            with torch.enable_grad():
                θ_mean = θ.mean()
                θ_std = θ.std()
                x_masked_constrained = μ_target + σ_target * (θ - θ_mean) / θ_std

                # Reconstruct full image
                x_full = torch.empty_like(initial_img)
                x_full[~mask] = initial_unmasked
                x_full[mask] = x_masked_constrained

                # Compute utility
                U = compute_utility_single_image(model_active, x_full, max_r_cap)

                # Compute pixel bounds penalty
                P = compute_pixel_penalty(x_masked_constrained, pixel_min, pixel_max, penalty_alpha, threshold_fraction)

            # Handle invalid values
            if torch.isnan(U) or torch.isinf(U):
                print(f"    [Closure] U is {'NaN' if torch.isnan(U) else 'Inf'}, rejecting step")
                return torch.tensor(1e10, dtype=U.dtype, device=U.device, requires_grad=True)

            if torch.isnan(P) or torch.isinf(P):
                print(f"    [Closure] Penalty P is {'NaN' if torch.isnan(P) else 'Inf'}, rejecting step")
                return torch.tensor(1e10, dtype=U.dtype, device=U.device, requires_grad=True)

            # L-BFGS minimizes: minimize -U + λ*P (maximize utility, minimize penalty)
            loss = -U + penalty_lambda * P
            loss.backward()

            # Handle invalid gradients
            if θ.grad is not None:
                if torch.isnan(θ.grad).any() or torch.isinf(θ.grad).any():
                    nan_grad_count[0] += 1
                    if nan_grad_count[0] <= max_nan_grad_allowed:
                        print(f"    [Closure] Warning: Gradient is NaN/Inf (occurrence {nan_grad_count[0]}/{max_nan_grad_allowed}), clearing and rejecting step")
                        optimizer.zero_grad()
                        return torch.tensor(1e10, dtype=loss.dtype, device=loss.device, requires_grad=False)
                    else:
                        raise RuntimeError(
                            f"Gradient contains NaN or Inf after backward pass (occurred {nan_grad_count[0]} times)! "
                            f"U={U.item():.6f}, P={P.item():.6f}, loss={loss.item():.6f}. "
                        )
            else:
                raise ValueError("[Closure] θ.grad is None")

            return loss

        # === SINGLE OPTIMIZATION CALL ===
        print(f"Running L-BFGS with max_iter={max_iter}...")
        optimizer.step(closure)

        # Get final state
        with torch.enable_grad():
            θ_mean_final = θ.mean()
            θ_std_final = θ.std()
            x_masked_final = μ_target + σ_target * (θ - θ_mean_final) / θ_std_final

            if apply_soft_compression:
                # Get pixel bounds from dataset if not provided
                if pixel_min is None or pixel_max is None:
                    pix_min = imgs_train.min().item()
                    pix_max = imgs_train.max().item()
                else:
                    pix_min = pixel_min
                    pix_max = pixel_max

                print(f"\n{'='*60}")
                print(f"Applying soft sigmoid compression...")

                # Apply compression
                x_masked_final = soft_sigmoid_compress_pixels(
                    x_masked_final.detach(),
                    pixel_min=pix_min,
                    pixel_max=pix_max,
                    steepness=compression_steepness,
                    transition_width=compression_transition_width,
                    verbose=True
                )

                # Enable gradients for recomputation
                x_masked_final = x_masked_final.requires_grad_(True)

                # Report RMS constraint violations
                μ_compressed = x_masked_final.mean().item()
                σ_compressed = x_masked_final.std().item()
                μ_error = abs(μ_compressed - μ_target)
                σ_error = abs(σ_compressed - σ_target)

                print(f"    RMS constraint violations after compression:")
                print(f"      Mean: {μ_compressed:.6f} (target: {μ_target:.6f}, error: {μ_error:.6f})")
                print(f"      Std:  {σ_compressed:.6f} (target: {σ_target:.6f}, error: {σ_error:.6f})")
                print(f"{'='*60}\n")

            elif apply_hard_clipping:
                # Get pixel bounds from dataset if not provided
                if pixel_min is None or pixel_max is None:
                    pix_min = imgs_train.min().item()
                    pix_max = imgs_train.max().item()
                else:
                    pix_min = pixel_min
                    pix_max = pixel_max

                # Count outliers before clipping
                n_outliers = ((x_masked_final < pix_min) | (x_masked_final > pix_max)).sum().item()

                # Apply hard clipping
                x_masked_final = torch.clamp(x_masked_final.detach(), pix_min, pix_max)
                x_masked_final = x_masked_final.requires_grad_(True)

                # Report RMS violations
                μ_clipped = x_masked_final.mean().item()
                σ_clipped = x_masked_final.std().item()
                print(f"  [Hard Clipping] {n_outliers} outliers clipped. RMS errors: Δμ={abs(μ_clipped - μ_target):.6f}, Δσ={abs(σ_clipped - σ_target):.6f}")

            x_full_final = initial_img.clone()
            x_full_final[mask] = x_masked_final

            U_final = compute_utility_single_image(model_active, x_full_final, max_r_cap)
            P_final = compute_pixel_penalty(x_masked_final, pixel_min, pixel_max, penalty_alpha, threshold_fraction)
            dU_dtheta_final, dU_dx_final = torch.autograd.grad(
                U_final, 
                [θ, x_masked_final],
                allow_unused=True)

            # Handle None when compression breaks computational graph
            if dU_dtheta_final is None:
                dU_dtheta_final = torch.zeros_like(θ)

        print(f"\nInitial: U = {U_init.item():.6f}, P = {P_init.item():.6f}, Loss = {(-U_init + penalty_lambda * P_init).item():.6f}")
        print(f"Final:   U = {U_final.item():.6f}, P = {P_final.item():.6f}, Loss = {(-U_final + penalty_lambda * P_final).item():.6f}")
        print(f"Change:  ΔU = {U_final.item() - U_init.item():+.6f} ({(U_final.item()/U_init.item() - 1)*100:+.2f}%)")
        print(f"         ΔP = {P_final.item() - P_init.item():+.6f} ({((P_final.item()/(P_init.item() + 1e-10)) - 1)*100:+.2f}%)")
        print(f"         ||∂U/∂θ||: {dU_dtheta_init.norm().item():.6e} → {dU_dtheta_final.norm().item():.6e}")
        print(f"         ||∂U/∂x||: {dU_dx_init.norm().item():.6e} → {dU_dx_final.norm().item():.6e}")

        # Verify RMS constraints
        μ_error = abs(x_masked_final.mean().item() - μ_target)
        σ_error = abs(x_masked_final.std().item() - σ_target)
        if μ_error > 1e-4 or σ_error > 1e-4:
            print(f"    [Warning] Final RMS constraints not satisfied within tolerance: |Δμ|={μ_error:.6e}, |Δσ|={σ_error:.6e}")

        # Pixel bounds diagnostics
        print(f"\n=== Pixel Bounds Diagnostics ===")
        print(f"Initial masked pixel stats: μ={μ_target:.6f}, σ={σ_target:.6f}")
        print(f"Final masked pixel stats: μ={x_masked_final.mean().item():.6f}, σ={x_masked_final.std().item():.6f}")
        #print(f"Final L2 norm of masked region: {torch.linalg.norm(x_masked_final).item():.6f}")
        print(f"Initial pixel range: [{x_masked_init.min().item():.6f}, {x_masked_init.max().item():.6f}]")
        print(f"Final pixel range: [{x_masked_final.min().item():.6f}, {x_masked_final.max().item():.6f}]")
        print(f"Dataset bounds:    [{pixel_min:.6f}, {pixel_max:.6f}]")

        # Count violations
        n_pixels_above = (x_masked_final > pixel_max).sum().item()
        n_pixels_below = (x_masked_final < pixel_min).sum().item()
        n_pixels_total = x_masked_final.numel()
        n_violations = n_pixels_above + n_pixels_below

        if n_violations > 0:
            print(f"Pixels exceeding upper bound: {n_pixels_above}/{n_pixels_total} ({100*n_pixels_above/n_pixels_total:.2f}%)")
            print(f"Pixels exceeding lower bound: {n_pixels_below}/{n_pixels_total} ({100*n_pixels_below/n_pixels_total:.2f}%)")
            print(f"Total violations: {n_violations}/{n_pixels_total} ({100*n_violations/n_pixels_total:.2f}%)")
            if n_pixels_above > 0:
                max_violation = (x_masked_final[x_masked_final > pixel_max] - pixel_max).max().item()
                print(f"Max upper violation: {max_violation:.6f}")
            if n_pixels_below > 0:
                max_violation = (pixel_min - x_masked_final[x_masked_final < pixel_min]).max().item()
                print(f"Max lower violation: {max_violation:.6f}")
        else:
            print(f"✓ All pixels within bounds! (0/{n_pixels_total} violations)")

        # Analyze final pixel distribution relative to safe zones
        n_pixels_in_safe_zone_final = ((x_masked_final >= x_min_thresh) & (x_masked_final <= x_max_thresh)).sum().item()
        n_pixels_in_upper_warning_final = (x_masked_final > x_max_thresh).sum().item()
        n_pixels_in_lower_warning_final = (x_masked_final < x_min_thresh).sum().item()

        print(f"\n=== Final Pixel Distribution vs Safe Zone ===")
        print(f"Pixels in safe zone [{x_min_thresh:.3f}, {x_max_thresh:.3f}]: {n_pixels_in_safe_zone_final}/{n_pixels_total} ({100*n_pixels_in_safe_zone_final/n_pixels_total:.1f}%)")
        print(f"Pixels in upper warning zone (>{x_max_thresh:.3f}): {n_pixels_in_upper_warning_final}/{n_pixels_total} ({100*n_pixels_in_upper_warning_final/n_pixels_total:.1f}%)")
        print(f"Pixels in lower warning zone (<{x_min_thresh:.3f}): {n_pixels_in_lower_warning_final}/{n_pixels_total} ({100*n_pixels_in_lower_warning_final/n_pixels_total:.1f}%)")

        if n_pixels_in_upper_warning_final > 0:
            max_distance_upper_final = (x_masked_final[x_masked_final > x_max_thresh] - x_max_thresh).max().item()
            avg_distance_upper_final = (x_masked_final[x_masked_final > x_max_thresh] - x_max_thresh).mean().item()
            print(f"  Upper warning zone: max intrusion={max_distance_upper_final:.3f}, avg intrusion={avg_distance_upper_final:.3f}")

        if n_pixels_in_lower_warning_final > 0:
            max_distance_lower_final = (x_min_thresh - x_masked_final[x_masked_final < x_min_thresh]).max().item()
            avg_distance_lower_final = (x_min_thresh - x_masked_final[x_masked_final < x_min_thresh]).mean().item()
            print(f"  Lower warning zone: max intrusion={max_distance_lower_final:.3f}, avg intrusion={avg_distance_lower_final:.3f}")

        # Show migration from warning zones to safe zone
        pixels_moved_to_safe = n_pixels_in_safe_zone_final - n_pixels_in_safe_zone
        if pixels_moved_to_safe > 0:
            print(f"\n✓ {pixels_moved_to_safe} pixels moved from warning zones into safe zone")
        elif pixels_moved_to_safe < 0:
            print(f"\n✗ {-pixels_moved_to_safe} pixels moved from safe zone into warning zones")
        else:
            print(f"\n→ No change in safe zone pixel count")

        # L-BFGS diagnostics
        state = optimizer.state[optimizer._params[0]]
        n_iters = state.get('n_iter', 0)
        n_evals = state.get('func_evals', 0)
        print(f"\nL-BFGS iterations: {n_iters}")

        # Update gradient and optimized image
        optimized_img = x_full_final

        # Return dictionary matching original batch_utility_w_grad format (without trajectory)
        return {
            'optimized_img': optimized_img,
            'img_idx': x_idx_best[None] if x_idx_best.dim() == 0 else x_idx_best,
            'utility_batch': u2d,
            'start_gradient': dU_dx_init.squeeze(0) if dU_dx_init.dim() > 1 else dU_dx_init,
            'end_gradient': dU_dx_final.squeeze(0) if  dU_dx_final.dim() > 1 else dU_dx_final,
            'U_initial': U_init.item() if isinstance(U_init, torch.Tensor) else U_init,
            'U_final': U_final.item() if isinstance(U_final, torch.Tensor) else U_final,
            'P_initial': P_init.item() if isinstance(P_init, torch.Tensor) else P_init,
            'P_final': P_final.item() if isinstance(P_final, torch.Tensor) else P_final,
            'pixel_violations': n_violations,
            'lr': lr,
            'line_search_fn': line_search_fn,
            'max_iter': max_iter,
            'penalty_lambda': penalty_lambda,
            'penalty_alpha': penalty_alpha
        }

    elif test_mean_only_and_px_constraint:
        print("\n=== Mean-Only + Pixel Bounds Constrained Optimization (Slim Version) ===")
        print("(Std is free to vary, only mean is constrained)")

        # Validate pixel bounds are provided
        if pixel_min is None or pixel_max is None:
            raise ValueError("test_mean_only_and_px_constraint=True requires pixel_min and pixel_max to be provided")

        print(f"Pixel bounds: [{pixel_min:.6f}, {pixel_max:.6f}]")
        print(f"Penalty parameters: λ={penalty_lambda:.2f}, α={penalty_alpha:.2f}")

        # Calculate and display safe zone boundaries
        range_width = pixel_max - pixel_min
        margin = (1 - threshold_fraction) * range_width / 2
        x_max_thresh = pixel_max - margin
        x_min_thresh = pixel_min + margin
        print(f"Safe zone (threshold={threshold_fraction:.2f}): [{x_min_thresh:.6f}, {x_max_thresh:.6f}]")
        print(f"  → Inner {threshold_fraction*100:.0f}% of range is penalty-free")
        print(f"  → Outer {(1-threshold_fraction)*100:.0f}% provides smooth warning gradient")

        # Setup
        initial_img = imgs_train[x_idx_best].clone()
        initial_unmasked = initial_img[~mask].clone()
        initial_masked = initial_img[mask].clone()

        # Target constraint: mean only (no std constraint!)
        μ_target = initial_masked.mean().item()
        σ_initial = initial_masked.std().item()  # For reporting only
        print(f"Target constraint: μ = {μ_target:.6f} (std free to vary, initial σ = {σ_initial:.6f})")

        # Check initial pixel bounds violations
        n_pixels_exceeding_bounds = ((initial_masked > pixel_max) | (initial_masked < pixel_min)).sum().item()
        if n_pixels_exceeding_bounds > 0:
            print(f"Initial violations: {n_pixels_exceeding_bounds}/{initial_masked.numel()} pixels exceed bounds")

        # Analyze initial pixel distribution relative to safe zones
        n_pixels_in_safe_zone = ((initial_masked >= x_min_thresh) & (initial_masked <= x_max_thresh)).sum().item()
        n_pixels_in_upper_warning = (initial_masked > x_max_thresh).sum().item()
        n_pixels_in_lower_warning = (initial_masked < x_min_thresh).sum().item()
        n_pixels_total = initial_masked.numel()

        if n_pixels_in_upper_warning > 0:
            max_distance_upper = (initial_masked[initial_masked > x_max_thresh] - x_max_thresh).max().item()
            avg_distance_upper = (initial_masked[initial_masked > x_max_thresh] - x_max_thresh).mean().item()
            print(f"  Upper warning zone: max intrusion={max_distance_upper:.3f}, avg intrusion={avg_distance_upper:.3f}")

        if n_pixels_in_lower_warning > 0:
            max_distance_lower = (x_min_thresh - initial_masked[initial_masked < x_min_thresh]).max().item()
            avg_distance_lower = (x_min_thresh - initial_masked[initial_masked < x_min_thresh]).mean().item()
            print(f"  Lower warning zone: max intrusion={max_distance_lower:.3f}, avg intrusion={avg_distance_lower:.3f}")

        # Warning if too many pixels in warning zones
        pct_in_warning = 100 * (n_pixels_in_upper_warning + n_pixels_in_lower_warning) / n_pixels_total
        if pct_in_warning > 5.0:
            print(f"\n  ⚠ WARNING: {pct_in_warning:.1f}% of pixels are in warning zones!")
            print(f"  This creates penalty even though pixels are within bounds.")
            print(f"  The optimizer will reduce penalty by sacrificing utility.")
            print(f"  Recommendations:")
            if threshold_fraction < 0.99:
                print(f"    → Increase threshold_fraction from {threshold_fraction} to 0.99 or 0.995")
            else:
                print(f"    → threshold_fraction is already {threshold_fraction} (very high)")
            print(f"    → Reduce penalty_lambda from {penalty_lambda} to {penalty_lambda/10:.6f}")
            print(f"    → Or accept utility decrease as tradeoff for bounded pixels\n")

        # Optimization variable θ (unconstrained)
        θ = initial_masked.clone().detach().requires_grad_(True)

        # Get initial state
        with torch.enable_grad():
            θ_mean_init = θ.mean()
            # SIMPLIFIED REPARAMETRIZATION: Mean-only (no std constraint!)
            x_masked_init = μ_target + (θ - θ_mean_init)

            x_full_init = initial_img.clone()
            x_full_init[mask] = x_masked_init
            x_full_init[~mask] = initial_unmasked

            U_init = compute_utility_single_image(model_active, x_full_init, max_r_cap)
            P_init = compute_pixel_penalty(x_masked_init, pixel_min, pixel_max, penalty_alpha, threshold_fraction)

            dU_dtheta_init, dU_dx_init = torch.autograd.grad(
                U_init,
                [θ, x_masked_init])

        print(f"Initial: U={U_init.item():.6f}, P={P_init.item():.6f}, Loss={(-U_init + penalty_lambda * P_init).item():.6f}")

        # Check for penalty dominance and warn user
        weighted_penalty_utility_ratio = np.abs((penalty_lambda * P_init.item()) / abs(U_init.item()))
        if weighted_penalty_utility_ratio > 100:
            print(f"    [WARNING] Penalty dominates utility!")
            print(f"    Weighted penalty/utility ratio: {weighted_penalty_utility_ratio:.1f}:1")
            print(f"    The optimizer may sacrifice utility to reduce penalty.")
            recommended_lambda = penalty_lambda / (weighted_penalty_utility_ratio / 10)
            print(f"    Consider reducing penalty_lambda from {penalty_lambda} to ~{recommended_lambda:.6f}")
        elif weighted_penalty_utility_ratio < 0.01:
            print(f"    [WARNING] Utility dominates penalty!")
            print(f"    Weighted penalty/utility ratio: {weighted_penalty_utility_ratio:.6f}:1")
            print(f"    Pixel bounds may not be enforced effectively.")
            recommended_lambda = penalty_lambda / (weighted_penalty_utility_ratio / 0.1)
            print(f"    Consider increasing penalty_lambda from {penalty_lambda} to ~{recommended_lambda:.6f}")

        # Create optimizer
        optimizer = torch.optim.LBFGS(
            [θ],
            lr=lr,
            max_iter=max_iter,  # Internal L-BFGS iterations
            history_size=500,
            line_search_fn=line_search_fn,
            armijo_c1=1e-3,
            armijo_rho=0.7,
            armijo_min_step_size=1e-5,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            verbose=False,  # Set True for debugging
        )

        # Track NaN gradient occurrences
        nan_grad_count = [0]
        max_nan_grad_allowed = 50

        # Closure
        def closure():
            """Closure for L-BFGS with mean-only reparameterization + pixel bounds penalty"""
            optimizer.zero_grad()

            # SIMPLIFIED REPARAMETERIZATION: θ → x_masked (mean constraint satisfied automatically)
            # NO std constraint, NO division by std - always numerically stable!
            with torch.enable_grad():
                θ_mean = θ.mean()
                x_masked_constrained = μ_target + (θ - θ_mean)

                # Reconstruct full image
                x_full = torch.empty_like(initial_img)
                x_full[~mask] = initial_unmasked
                x_full[mask] = x_masked_constrained

                # Compute utility
                U = compute_utility_single_image(model_active, x_full, max_r_cap)

                # Compute pixel bounds penalty
                P = compute_pixel_penalty(x_masked_constrained, pixel_min, pixel_max, penalty_alpha, threshold_fraction)

            # Handle invalid values
            if torch.isnan(U) or torch.isinf(U):
                print(f"    [Closure] U is {'NaN' if torch.isnan(U) else 'Inf'}, rejecting step")
                return torch.tensor(1e10, dtype=U.dtype, device=U.device, requires_grad=True)

            if torch.isnan(P) or torch.isinf(P):
                print(f"    [Closure] Penalty P is {'NaN' if torch.isnan(P) else 'Inf'}, rejecting step")
                return torch.tensor(1e10, dtype=U.dtype, device=U.device, requires_grad=True)

            # L-BFGS minimizes: minimize -U + λ*P (maximize utility, minimize penalty)
            loss = -U + penalty_lambda * P
            loss.backward()

            # Handle invalid gradients
            if θ.grad is not None:
                if torch.isnan(θ.grad).any() or torch.isinf(θ.grad).any():
                    nan_grad_count[0] += 1
                    if nan_grad_count[0] <= max_nan_grad_allowed:
                        print(f"    [Closure] Warning: Gradient is NaN/Inf (occurrence {nan_grad_count[0]}/{max_nan_grad_allowed}), clearing and rejecting step")
                        optimizer.zero_grad()
                        return torch.tensor(1e10, dtype=loss.dtype, device=loss.device, requires_grad=False)
                    else:
                        raise RuntimeError(
                            f"Gradient contains NaN or Inf after backward pass (occurred {nan_grad_count[0]} times)! "
                            f"U={U.item():.6f}, P={P.item():.6f}, loss={loss.item():.6f}. "
                        )
            else:
                raise ValueError("[Closure] θ.grad is None")

            return loss

        # === SINGLE OPTIMIZATION CALL ===
        print(f"Running L-BFGS with max_iter={max_iter}...")
        optimizer.step(closure)

        # Get final state
        with torch.enable_grad():
            θ_mean_final = θ.mean()
            # SIMPLIFIED REPARAMETRIZATION: Mean-only
            x_masked_final = μ_target + (θ - θ_mean_final)

            if apply_soft_compression:
                # Get pixel bounds from dataset if not provided
                if pixel_min is None or pixel_max is None:
                    pix_min = imgs_train.min().item()
                    pix_max = imgs_train.max().item()
                else:
                    pix_min = pixel_min
                    pix_max = pixel_max

                print(f"\n{'='*60}")
                print(f"Applying soft sigmoid compression...")

                # Apply compression
                x_masked_final = soft_sigmoid_compress_pixels(
                    x_masked_final.detach(),
                    pixel_min=pix_min,
                    pixel_max=pix_max,
                    steepness=compression_steepness,
                    transition_width=compression_transition_width,
                    verbose=True
                )

                # Enable gradients for recomputation
                x_masked_final = x_masked_final.requires_grad_(True)

                # Report mean constraint violation (std is informational only)
                μ_compressed = x_masked_final.mean().item()
                σ_compressed = x_masked_final.std().item()
                μ_error = abs(μ_compressed - μ_target)
                σ_change = σ_compressed - σ_initial

                print(f"    Constraint violations after compression:")
                print(f"      Mean: {μ_compressed:.6f} (target: {μ_target:.6f}, error: {μ_error:.6f})")
                print(f"      Std:  {σ_compressed:.6f} (initial: {σ_initial:.6f}, change: {σ_change:+.6f}) [unconstrained]")
                print(f"{'='*60}\n")

            elif apply_hard_clipping:
                # Get pixel bounds from dataset if not provided
                if pixel_min is None or pixel_max is None:
                    pix_min = imgs_train.min().item()
                    pix_max = imgs_train.max().item()
                else:
                    pix_min = pixel_min
                    pix_max = pixel_max

                # Count outliers before clipping
                n_outliers = ((x_masked_final < pix_min) | (x_masked_final > pix_max)).sum().item()

                # Apply hard clipping
                x_masked_final = torch.clamp(x_masked_final.detach(), pix_min, pix_max)
                x_masked_final = x_masked_final.requires_grad_(True)

                # Report mean violation and std change
                μ_clipped = x_masked_final.mean().item()
                σ_clipped = x_masked_final.std().item()
                print(f"  [Hard Clipping] {n_outliers} outliers clipped. Mean error: Δμ={abs(μ_clipped - μ_target):.6f}, Std change: Δσ={σ_clipped - σ_initial:+.6f} [unconstrained]")

            x_full_final = initial_img.clone()
            x_full_final[mask] = x_masked_final

            U_final = compute_utility_single_image(model_active, x_full_final, max_r_cap)
            P_final = compute_pixel_penalty(x_masked_final, pixel_min, pixel_max, penalty_alpha, threshold_fraction)
            dU_dtheta_final, dU_dx_final = torch.autograd.grad(
                U_final,
                [θ, x_masked_final],
                allow_unused=True)

            # Handle None when compression breaks computational graph
            if dU_dtheta_final is None:
                dU_dtheta_final = torch.zeros_like(θ)

        print(f"\nInitial: U = {U_init.item():.6f}, P = {P_init.item():.6f}, Loss = {(-U_init + penalty_lambda * P_init).item():.6f}")
        print(f"Final:   U = {U_final.item():.6f}, P = {P_final.item():.6f}, Loss = {(-U_final + penalty_lambda * P_final).item():.6f}")
        print(f"Change:  ΔU = {U_final.item() - U_init.item():+.6f} ({(U_final.item()/U_init.item() - 1)*100:+.2f}%)")
        print(f"         ΔP = {P_final.item() - P_init.item():+.6f} ({((P_final.item()/(P_init.item() + 1e-10)) - 1)*100:+.2f}%)")
        print(f"         ||∂U/∂θ||: {dU_dtheta_init.norm().item():.6e} → {dU_dtheta_final.norm().item():.6e}")
        print(f"         ||∂U/∂x||: {dU_dx_init.norm().item():.6e} → {dU_dx_final.norm().item():.6e}")

        # Verify mean constraint (std is informational only)
        μ_error = abs(x_masked_final.mean().item() - μ_target)
        if μ_error > 1e-4:
            print(f"    [Warning] Final mean constraint not satisfied within tolerance: |Δμ|={μ_error:.6e}")

        # Report std change (informational, not enforced)
        σ_final = x_masked_final.std().item()
        σ_change = σ_final - σ_initial
        print(f"    Std change: {σ_initial:.6f} → {σ_final:.6f} (Δσ = {σ_change:+.6f}) [unconstrained]")

        # Pixel bounds diagnostics
        print(f"\n=== Pixel Bounds Diagnostics ===")
        print(f"Initial masked pixel stats: μ={μ_target:.6f}, σ={σ_initial:.6f}")
        print(f"Final masked pixel stats: μ={x_masked_final.mean().item():.6f}, σ={σ_final:.6f}")
        print(f"Initial pixel range: [{x_masked_init.min().item():.6f}, {x_masked_init.max().item():.6f}]")
        print(f"Final pixel range: [{x_masked_final.min().item():.6f}, {x_masked_final.max().item():.6f}]")
        print(f"Dataset bounds:    [{pixel_min:.6f}, {pixel_max:.6f}]")

        # Count violations
        n_pixels_above = (x_masked_final > pixel_max).sum().item()
        n_pixels_below = (x_masked_final < pixel_min).sum().item()
        n_pixels_total = x_masked_final.numel()
        n_violations = n_pixels_above + n_pixels_below

        if n_violations > 0:
            print(f"Pixels exceeding upper bound: {n_pixels_above}/{n_pixels_total} ({100*n_pixels_above/n_pixels_total:.2f}%)")
            print(f"Pixels exceeding lower bound: {n_pixels_below}/{n_pixels_total} ({100*n_pixels_below/n_pixels_total:.2f}%)")
            print(f"Total violations: {n_violations}/{n_pixels_total} ({100*n_violations/n_pixels_total:.2f}%)")
            if n_pixels_above > 0:
                max_violation = (x_masked_final[x_masked_final > pixel_max] - pixel_max).max().item()
                print(f"Max upper violation: {max_violation:.6f}")
            if n_pixels_below > 0:
                max_violation = (pixel_min - x_masked_final[x_masked_final < pixel_min]).max().item()
                print(f"Max lower violation: {max_violation:.6f}")
        else:
            print(f"✓ All pixels within bounds! (0/{n_pixels_total} violations)")

        # Analyze final pixel distribution relative to safe zones
        n_pixels_in_safe_zone_final = ((x_masked_final >= x_min_thresh) & (x_masked_final <= x_max_thresh)).sum().item()
        n_pixels_in_upper_warning_final = (x_masked_final > x_max_thresh).sum().item()
        n_pixels_in_lower_warning_final = (x_masked_final < x_min_thresh).sum().item()

        print(f"\n=== Final Pixel Distribution vs Safe Zone ===")
        print(f"Pixels in safe zone [{x_min_thresh:.3f}, {x_max_thresh:.3f}]: {n_pixels_in_safe_zone_final}/{n_pixels_total} ({100*n_pixels_in_safe_zone_final/n_pixels_total:.1f}%)")
        print(f"Pixels in upper warning zone (>{x_max_thresh:.3f}): {n_pixels_in_upper_warning_final}/{n_pixels_total} ({100*n_pixels_in_upper_warning_final/n_pixels_total:.1f}%)")
        print(f"Pixels in lower warning zone (<{x_min_thresh:.3f}): {n_pixels_in_lower_warning_final}/{n_pixels_total} ({100*n_pixels_in_lower_warning_final/n_pixels_total:.1f}%)")

        if n_pixels_in_upper_warning_final > 0:
            max_distance_upper_final = (x_masked_final[x_masked_final > x_max_thresh] - x_max_thresh).max().item()
            avg_distance_upper_final = (x_masked_final[x_masked_final > x_max_thresh] - x_max_thresh).mean().item()
            print(f"  Upper warning zone: max intrusion={max_distance_upper_final:.3f}, avg intrusion={avg_distance_upper_final:.3f}")

        if n_pixels_in_lower_warning_final > 0:
            max_distance_lower_final = (x_min_thresh - x_masked_final[x_masked_final < x_min_thresh]).max().item()
            avg_distance_lower_final = (x_min_thresh - x_masked_final[x_masked_final < x_min_thresh]).mean().item()
            print(f"  Lower warning zone: max intrusion={max_distance_lower_final:.3f}, avg intrusion={avg_distance_lower_final:.3f}")

        # Show migration from warning zones to safe zone
        pixels_moved_to_safe = n_pixels_in_safe_zone_final - n_pixels_in_safe_zone
        if pixels_moved_to_safe > 0:
            print(f"\n✓ {pixels_moved_to_safe} pixels moved from warning zones into safe zone")
        elif pixels_moved_to_safe < 0:
            print(f"\n✗ {-pixels_moved_to_safe} pixels moved from safe zone into warning zones")
        else:
            print(f"\n→ No change in safe zone pixel count")

        # L-BFGS diagnostics
        state = optimizer.state[optimizer._params[0]]
        n_iters = state.get('n_iter', 0)
        n_evals = state.get('func_evals', 0)
        print(f"\nL-BFGS iterations: {n_iters}")

        # Update gradient and optimized image
        optimized_img = x_full_final

        # Return dictionary matching original batch_utility_w_grad format (without trajectory)
        return {
            'optimized_img': optimized_img,
            'img_idx': x_idx_best[None] if x_idx_best.dim() == 0 else x_idx_best,
            'utility_batch': u2d,
            'start_gradient': dU_dx_init.squeeze(0) if dU_dx_init.dim() > 1 else dU_dx_init,
            'end_gradient': dU_dx_final.squeeze(0) if  dU_dx_final.dim() > 1 else dU_dx_final,
            'U_initial': U_init.item() if isinstance(U_init, torch.Tensor) else U_init,
            'U_final': U_final.item() if isinstance(U_final, torch.Tensor) else U_final,
            'P_initial': P_init.item() if isinstance(P_init, torch.Tensor) else P_init,
            'P_final': P_final.item() if isinstance(P_final, torch.Tensor) else P_final,
            'pixel_violations': n_violations,
            'lr': lr,
            'line_search_fn': line_search_fn,
            'max_iter': max_iter,
            'penalty_lambda': penalty_lambda,
            'penalty_alpha': penalty_alpha
        }

    elif test_px_only_lbfgsb_constraint:
        print("\n=== LBFGS-B Pixel-Only Constrained Optimization ===")
        print("Using true L-BFGS-B algorithm with box constraints (no RMS constraints)")

        # Validate pixel bounds are provided
        if pixel_min is None or pixel_max is None:
            raise ValueError("test_px_only_lbfgsb_constraint requires pixel_min and pixel_max to be provided")

        # Import LBFGSB optimizer
        from torch.optim.lbfgsb import LBFGSB

        # Setup
        initial_img = imgs_train[x_idx_best].clone()
        initial_unmasked = initial_img[~mask].clone()
        initial_masked = initial_img[mask].clone()

        print(f"Pixel bounds: [{pixel_min:.6f}, {pixel_max:.6f}]")
        print(f"Initial masked pixel stats: μ={initial_masked.mean().item():.6f}, σ={initial_masked.std().item():.6f}")
        print(f"Initial masked pixel range: [{initial_masked.min().item():.6f}, {initial_masked.max().item():.6f}]")
        print(f"Number of masked pixels (RF): {initial_masked.numel()}")

        # Convert to float64 (LBFGSB requirement)
        x_masked_f64 = initial_masked.clone().double().requires_grad_(True)

        # Create bounds tensors (float64, same shape as x_masked_f64)
        n_masked_pixels = x_masked_f64.numel()
        lower_bound = torch.full((n_masked_pixels,), pixel_min, dtype=torch.float64, device=x_masked_f64.device)
        upper_bound = torch.full((n_masked_pixels,), pixel_max, dtype=torch.float64, device=x_masked_f64.device)

        # Validate and project initial point to feasible region if needed
        # Check if projection is needed (use detached copy for checking)
        needs_projection = (x_masked_f64 < pixel_min).any() or (x_masked_f64 > pixel_max).any()

        if needs_projection:
            print(f"    [Warning] Initial point infeasible, projecting to bounds...")
            with torch.no_grad():
                n_below = (x_masked_f64 < pixel_min).sum().item()
                n_above = (x_masked_f64 > pixel_max).sum().item()
                print(f"    Pixels below min: {n_below}, above max: {n_above}")

            # Clamp WITHOUT no_grad to preserve gradient tracking
            x_masked_f64.data.clamp_(pixel_min, pixel_max)
            print(f"    After projection: [{x_masked_f64.min().item():.6f}, {x_masked_f64.max().item():.6f}]")

        # Get initial state
        with torch.enable_grad():
            # Reconstruct full image (convert back to float32 if needed)
            x_full_init = initial_img.clone()
            if TORCH_DTYPE == torch.float32:
                x_full_init[mask] = x_masked_f64.float()
            else:
                x_full_init[mask] = x_masked_f64

            U_init = compute_utility_single_image(model_active, x_full_init, max_r_cap)

            # Compute initial gradient
            dU_dx_init = torch.autograd.grad(U_init, x_masked_f64, create_graph=False)[0]

        print(f"Initial utility: U = {U_init.item():.6f}")
        print(f"Initial gradient norm: ||∂U/∂x|| = {dU_dx_init.norm().item():.6e}")

        # Create LBFGSB optimizer
        optimizer = LBFGSB(
            [x_masked_f64],
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            max_iter=max_iter,
            tolerance_grad=1e-7,
            tolerance_change=1e-20,  # LBFGS-B default (more permissive for flat landscapes)
            history_size=7,  # Standard LBFGS value (3-20 recommended, does NOT scale with params)
            batch_mode=False,  # Full-batch optimization
            cost_use_gradient=False  # Standard case
        )

        print(f"LBFGSB optimizer configured: max_iter={max_iter}, history_size=7, tolerance_change=1e-20")

        # Track NaN gradient occurrences
        nan_grad_count = [0]
        max_nan_grad_allowed = 50

        # Closure
        def closure():
            """Closure for LBFGS-B with box constraints on pixels"""
            optimizer.zero_grad()

            with torch.enable_grad():
                # Reconstruct full image
                x_full = torch.empty_like(initial_img)
                x_full[~mask] = initial_unmasked
                # Convert masked pixels back to float32 if needed
                if TORCH_DTYPE == torch.float32:
                    x_full[mask] = x_masked_f64.float()
                else:
                    x_full[mask] = x_masked_f64

                # Compute utility
                U = compute_utility_single_image(model_active, x_full, max_r_cap)

                # Handle invalid values
                if torch.isnan(U) or torch.isinf(U):
                    print(f"    [Closure] U is {'NaN' if torch.isnan(U) else 'Inf'}, rejecting step")
                    return torch.tensor(1e10, dtype=torch.float64, device=U.device, requires_grad=True)

                # LBFGS-B minimizes, so negate utility
                loss = -U.double()  # Ensure float64
                loss.backward()

            # Handle invalid gradients (check AFTER backward, but outside enable_grad)
            if x_masked_f64.grad is not None:
                if torch.isnan(x_masked_f64.grad).any() or torch.isinf(x_masked_f64.grad).any():
                    nan_grad_count[0] += 1
                    if nan_grad_count[0] <= max_nan_grad_allowed:
                        print(f"    [Closure] Warning: Gradient is NaN/Inf (occurrence {nan_grad_count[0]}/{max_nan_grad_allowed}), clearing and rejecting step")
                        optimizer.zero_grad()
                        return torch.tensor(1e10, dtype=torch.float64, device=x_masked_f64.device, requires_grad=False)
                    else:
                        raise RuntimeError(
                            f"Gradient contains NaN or Inf after backward pass (occurred {nan_grad_count[0]} times)! "
                            f"U={U.item():.6f}, loss={loss.item():.6f}."
                        )
            else:
                raise ValueError("[Closure] x_masked_f64.grad is None")

            return loss

        # === SINGLE OPTIMIZATION CALL ===
        print(f"Running LBFGS-B optimization...")
        optimizer.step(closure)

        # Get final state
        with torch.enable_grad():
            # Extract optimized pixels
            optimized_masked = x_masked_f64.detach()

            # Reconstruct full optimized image
            x_full_final = initial_img.clone()
            if TORCH_DTYPE == torch.float32:
                x_full_final[mask] = optimized_masked.float()
            else:
                x_full_final[mask] = optimized_masked

            # Compute final utility and gradient
            x_masked_for_grad = x_masked_f64.clone().requires_grad_(True)
            x_full_for_grad = initial_img.clone()
            if TORCH_DTYPE == torch.float32:
                x_full_for_grad[mask] = x_masked_for_grad.float()
            else:
                x_full_for_grad[mask] = x_masked_for_grad

            U_final = compute_utility_single_image(model_active, x_full_for_grad, max_r_cap)
            dU_dx_final = torch.autograd.grad(U_final, x_masked_for_grad, create_graph=False)[0]

        # Verify bounds satisfied
        with torch.no_grad():
            n_below = (optimized_masked < pixel_min).sum().item()
            n_above = (optimized_masked > pixel_max).sum().item()
            if n_below > 0 or n_above > 0:
                print(f"    [Warning] Bounds violated! Pixels below min: {n_below}, above max: {n_above}")
                print(f"    This should not happen with L-BFGS-B!")
            else:
                print(f"    ✓ All pixels within bounds [{pixel_min:.6f}, {pixel_max:.6f}]")

        # Print diagnostics
        print(f"\nOptimization Results:")
        print(f"Initial: U = {U_init.item():.6f}, ||∂U/∂x|| = {dU_dx_init.norm().item():.6e}")
        print(f"Final:   U = {U_final.item():.6f}, ||∂U/∂x|| = {dU_dx_final.norm().item():.6e}")
        print(f"Change:  ΔU = {U_final.item() - U_init.item():+.6f} ({(U_final.item()/U_init.item() - 1)*100:+.2f}%)")
        print(f"         Δ||∂U/∂x|| = {dU_dx_final.norm().item() - dU_dx_init.norm().item():+.6e}")

        print(f"\nFinal masked pixel stats:")
        print(f"  μ = {optimized_masked.mean().item():.6f}, σ = {optimized_masked.std().item():.6f}")
        print(f"  Range: [{optimized_masked.min().item():.6f}, {optimized_masked.max().item():.6f}]")
        print(f"  L2 norm: {torch.linalg.norm(optimized_masked).item():.6f}")

        # Convert back to float32 for consistency with rest of codebase
        if TORCH_DTYPE == torch.float32:
            dU_dx_init_f32 = dU_dx_init.float()
            dU_dx_final_f32 = dU_dx_final.float()
        else:
            dU_dx_init_f32 = dU_dx_init
            dU_dx_final_f32 = dU_dx_final

        # Return dictionary
        return {
            'optimized_img': x_full_final,
            'img_idx': x_idx_best[None] if x_idx_best.dim() == 0 else x_idx_best,
            'utility_batch': u2d,
            'start_gradient': dU_dx_init_f32.squeeze(0) if dU_dx_init_f32.dim() > 1 else dU_dx_init_f32,
            'end_gradient': dU_dx_final_f32.squeeze(0) if dU_dx_final_f32.dim() > 1 else dU_dx_final_f32,
            'U_initial': U_init.item() if isinstance(U_init, torch.Tensor) else U_init,
            'U_final': U_final.item() if isinstance(U_final, torch.Tensor) else U_final,
            'lr': lr,  # Not used by LBFGSB but keep for consistency
            'line_search_fn': 'lbfgsb',  # Indicate which method used
            'max_iter': max_iter,
            'optimizer_type': 'LBFGSB',  # New field to identify algorithm
            'dtype_used': 'float64'  # New field to track precision
        }

    elif interior_point_params is not None and interior_point_params.get('enabled', False):
        print("\n=== Interior Point (Log Barrier) Constrained Optimization ===")

        # Validate pixel bounds are provided
        if pixel_min is None or pixel_max is None:
            raise ValueError("Interior point method requires pixel_min and pixel_max to be provided")

        # Extract parameters with defaults
        barrier_n_outer = interior_point_params.get('n_outer', 5)
        barrier_t_init = interior_point_params.get('t_init', 1.0)
        barrier_mu = interior_point_params.get('mu', 10.0)
        barrier_margin_pct = interior_point_params.get('margin_pct', 0.01)

        # Compute safe bounds (margin inward from dataset bounds)
        range_width = pixel_max - pixel_min
        margin = barrier_margin_pct * range_width
        pixel_min_safe = pixel_min + margin
        pixel_max_safe = pixel_max - margin

        print(f"Pixel bounds: [{pixel_min:.6f}, {pixel_max:.6f}]")
        print(f"Safe interior (margin={barrier_margin_pct*100:.1f}%): [{pixel_min_safe:.6f}, {pixel_max_safe:.6f}]")

        # Setup
        initial_img = imgs_train[x_idx_best].clone()
        initial_unmasked = initial_img[~mask].clone()
        initial_masked = initial_img[mask].clone()

        # Check if initial image needs shrinking to fit in safe zone
        needs_shrinking = (initial_masked <= pixel_min_safe).any() or (initial_masked >= pixel_max_safe).any()

        if needs_shrinking:
            print(f"Initial image has pixels outside safe zone - shrinking...")
            initial_masked_feasible = shrink_to_safe_interior(
                initial_masked, pixel_min_safe, pixel_max_safe
            )
            print(f"  Before: [{initial_masked.min():.4f}, {initial_masked.max():.4f}]")
            print(f"  After:  [{initial_masked_feasible.min():.4f}, {initial_masked_feasible.max():.4f}]")
        else:
            initial_masked_feasible = initial_masked.clone()
            print("Initial image already in safe zone ✓")

        # Target RMS constraints from FEASIBLE initial image
        μ_target = initial_masked_feasible.mean().item()
        σ_target = initial_masked_feasible.std().item()
        print(f"Target RMS: μ={μ_target:.6f}, σ={σ_target:.6f}")

        # Barrier schedule: t_k = t_init * mu^k
        t_schedule = [barrier_t_init * (barrier_mu ** k) for k in range(barrier_n_outer)]
        print(f"Barrier schedule: {t_schedule}")

        # Optimization variable θ (unconstrained)
        θ = initial_masked_feasible.clone().detach().requires_grad_(True)
        θ_std_min = 0.01  # Numerical safety

        # Get initial state
        with torch.enable_grad():
            θ_mean_init = θ.mean()
            θ_std_init = θ.std()
            x_masked_init = μ_target + σ_target * (θ - θ_mean_init) / θ_std_init

            # Check feasibility of initial RMS-transformed point
            if (x_masked_init <= pixel_min_safe).any() or (x_masked_init >= pixel_max_safe).any():
                print("[WARNING] RMS transform made initial point infeasible!")
                print(f"  This may cause optimization issues - consider adjusting σ_target or margin")

            x_full_init = initial_img.clone()
            x_full_init[mask] = x_masked_init

            U_init = compute_utility_single_image(model_active, x_full_init, max_r_cap)
            B_init, feasible_init = compute_log_barrier(x_masked_init, pixel_min_safe, pixel_max_safe)

            if not feasible_init:
                raise RuntimeError(
                    "Initial point infeasible after RMS transform! "
                    "Try: (1) increase margin_pct, (2) reduce σ_target, or (3) different initial image"
                )

            dU_dtheta_init, dU_dx_init = torch.autograd.grad(U_init, [θ, x_masked_init])

        print(f"Initial: U={U_init.item():.6f}, B={B_init.item():.6f}, Loss={(-U_init + (1/barrier_t_init)*B_init).item():.6f}")

        # Track history across barrier iterations
        U_history = [U_init.item()]
        B_history = [B_init.item()]

        # NESTED OPTIMIZATION LOOP (outer: barrier schedule, inner: L-BFGS)
        for outer_iter, t_current in enumerate(t_schedule):
            print(f"\n--- Barrier Iteration {outer_iter+1}/{barrier_n_outer}, t={t_current:.2f} ---")

            # Track rejections within this barrier iteration
            nan_grad_count = [0]
            infeasible_count = [0]
            max_nan_allowed = 50
            max_infeasible_allowed = 100

            # Create optimizer for this barrier level
            optimizer = torch.optim.LBFGS(
                [θ],
                lr=lr,
                max_iter=max_iter,
                history_size=500,
                line_search_fn=line_search_fn,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                verbose=False
            )

            # Closure for this barrier level
            def closure():
                """L-BFGS closure with RMS reparameterization + log barrier"""
                # Numerical safety: clamp θ std
                with torch.no_grad():
                    θ_std_current = θ.std()
                    if θ_std_current < θ_std_min:
                        θ.data = (θ - θ.mean()) * (θ_std_min / θ_std_current) + θ.mean()

                optimizer.zero_grad()

                # RMS reparameterization: θ → x_masked (RMS constraints satisfied)
                with torch.enable_grad():
                    θ_mean = θ.mean()
                    θ_std = θ.std()
                    x_masked_constrained = μ_target + σ_target * (θ - θ_mean) / θ_std

                    # FEASIBILITY CHECK (critical for log barrier!)
                    # If RMS transform produces infeasible point, reject this step
                    if (x_masked_constrained <= pixel_min_safe).any() or \
                       (x_masked_constrained >= pixel_max_safe).any():
                        infeasible_count[0] += 1
                        if infeasible_count[0] <= max_infeasible_allowed:
                            # Silently reject step (L-BFGS will reduce step size and retry)
                            return torch.tensor(1e10, dtype=θ.dtype, device=θ.device, requires_grad=False)
                        else:
                            raise RuntimeError(
                                f"Too many infeasible steps ({infeasible_count[0]})! "
                                f"RMS transform keeps producing points outside bounds. "
                                f"Recommendations: "
                                f"(1) increase margin_pct from {barrier_margin_pct} to {barrier_margin_pct*2}, "
                                f"(2) reduce σ_target from {σ_target:.4f}, "
                                f"(3) use different initial image"
                            )

                    # Reconstruct full image
                    x_full = torch.empty_like(initial_img)
                    x_full[~mask] = initial_unmasked
                    x_full[mask] = x_masked_constrained

                    # Compute utility
                    U = compute_utility_single_image(model_active, x_full, max_r_cap)

                    # Compute log barrier
                    B, feasible = compute_log_barrier(x_masked_constrained, pixel_min_safe, pixel_max_safe)

                    if not feasible:
                        # Should not happen (checked above), but safety catch
                        return torch.tensor(1e10, dtype=θ.dtype, device=θ.device, requires_grad=False)

                # Handle invalid utility
                if torch.isnan(U) or torch.isinf(U):
                    return torch.tensor(1e10, dtype=U.dtype, device=U.device, requires_grad=True)

                # Barrier loss: minimize -U + (1/t)*B
                # As t increases, (1/t) → 0, barrier influence weakens
                loss = -U + (1.0 / t_current) * B
                loss.backward()

                # Handle invalid gradients
                if θ.grad is not None:
                    if torch.isnan(θ.grad).any() or torch.isinf(θ.grad).any():
                        nan_grad_count[0] += 1
                        if nan_grad_count[0] <= max_nan_allowed:
                            optimizer.zero_grad()
                            return torch.tensor(1e10, dtype=loss.dtype, device=loss.device, requires_grad=False)
                        else:
                            raise RuntimeError(
                                f"Gradient NaN/Inf occurred {nan_grad_count[0]} times! "
                                f"U={U.item():.6f}, B={B.item():.6f}, t={t_current:.2f}"
                            )
                else:
                    raise ValueError("θ.grad is None after backward pass")

                return loss

            # Run L-BFGS for this barrier level
            optimizer.step(closure)

            # Get state after this barrier iteration
            with torch.enable_grad():
                θ_mean_current = θ.mean()
                θ_std_current = θ.std()
                x_masked_current = μ_target + σ_target * (θ - θ_mean_current) / θ_std_current

                x_full_current = initial_img.clone()
                x_full_current[mask] = x_masked_current

                U_current = compute_utility_single_image(model_active, x_full_current, max_r_cap)
                B_current, feasible_current = compute_log_barrier(x_masked_current, pixel_min_safe, pixel_max_safe)

            if not feasible_current:
                print(f"  [ERROR] Final point of iteration {outer_iter+1} is infeasible!")
                print(f"  This should not happen - stopping barrier iterations")
                break

            U_history.append(U_current.item())
            B_history.append(B_current.item())

            # Diagnostics for this barrier iteration
            loss_current = -U_current + (1.0 / t_current) * B_current
            print(f"  U={U_current.item():.6f}, B={B_current.item():.6f}, Loss={loss_current.item():.6f}")
            print(f"  Pixel range: [{x_masked_current.min():.6f}, {x_masked_current.max():.6f}]")
            print(f"  Infeasible rejections: {infeasible_count[0]}")

            # Show improvement from previous barrier iteration
            if outer_iter > 0:
                delta_U = U_current.item() - U_history[-2]
                delta_B = B_current.item() - B_history[-2]
                print(f"  ΔU = {delta_U:+.6f} ({100*delta_U/U_history[-2]:+.2f}%)")
                print(f"  ΔB = {delta_B:+.6f}")

        # Get final state
        with torch.enable_grad():
            θ_mean_final = θ.mean()
            θ_std_final = θ.std()
            x_masked_final = μ_target + σ_target * (θ - θ_mean_final) / θ_std_final

            x_full_final = initial_img.clone()
            x_full_final[mask] = x_masked_final

            U_final = GP_utils.compute_utility_single_image(model_active, x_full_final, max_r_cap)
            B_final, feasible_final = compute_log_barrier(x_masked_final, pixel_min_safe, pixel_max_safe)
            dU_dtheta_final, dU_dx_final = torch.autograd.grad(U_final, [θ, x_masked_final])

        print(f"\n=== Final Results ===")
        print(f"Initial: U={U_init.item():.6f}, B={B_init.item():.6f}")
        print(f"Final:   U={U_final.item():.6f}, B={B_final.item():.6f}")
        print(f"Improvement: ΔU = {U_final.item() - U_init.item():+.6f} ({100*(U_final.item()/U_init.item()-1):+.2f}%)")

        # RMS constraint verification
        μ_final = x_masked_final.mean().item()
        σ_final = x_masked_final.std().item()
        μ_error = abs(μ_final - μ_target)
        σ_error = abs(σ_final - σ_target)
        print(f"\n=== RMS Constraint Verification ===")
        print(f"Target: μ={μ_target:.6f}, σ={σ_target:.6f}")
        print(f"Final:  μ={μ_final:.6f}, σ={σ_final:.6f}")
        print(f"Error:  |Δμ|={μ_error:.6e}, |Δσ|={σ_error:.6e}")
        if μ_error > 1e-4 or σ_error > 1e-4:
            print(f"  [Warning] RMS constraint not satisfied within tolerance (1e-4)")

        # Pixel bounds verification
        print(f"\n=== Pixel Bounds Verification ===")
        print(f"Final pixel range: [{x_masked_final.min():.6f}, {x_masked_final.max():.6f}]")
        print(f"Safe zone:         [{pixel_min_safe:.6f}, {pixel_max_safe:.6f}]")
        print(f"Dataset bounds:    [{pixel_min:.6f}, {pixel_max:.6f}]")

        # Count violations (should be ZERO for interior point!)
        n_pixels_below_safe = (x_masked_final <= pixel_min_safe).sum().item()
        n_pixels_above_safe = (x_masked_final >= pixel_max_safe).sum().item()
        n_pixels_below_hard = (x_masked_final < pixel_min).sum().item()
        n_pixels_above_hard = (x_masked_final > pixel_max).sum().item()
        n_pixels_total = x_masked_final.numel()

        if n_pixels_below_safe > 0 or n_pixels_above_safe > 0:
            print(f"  [ERROR] Pixels outside safe zone: {n_pixels_below_safe + n_pixels_above_safe}/{n_pixels_total}")
            print(f"  This should NEVER happen with interior point method!")
        else:
            print(f"  ✓ All pixels within safe zone! (0/{n_pixels_total} violations)")

        if n_pixels_below_hard > 0 or n_pixels_above_hard > 0:
            print(f"  [ERROR] Pixels outside dataset bounds: {n_pixels_below_hard + n_pixels_above_hard}/{n_pixels_total}")
        else:
            print(f"  ✓ All pixels within dataset bounds!")

        # Gradient diagnostics
        print(f"\n=== Gradient Diagnostics ===")
        print(f"||∂U/∂θ||: {dU_dtheta_init.norm().item():.6e} → {dU_dtheta_final.norm().item():.6e}")
        print(f"||∂U/∂x||: {dU_dx_init.norm().item():.6e} → {dU_dx_final.norm().item():.6e}")

        optimized_img = x_full_final

        # Return dictionary (SAME structure as other constraint methods)
        return {
            'optimized_img': optimized_img,
            'img_idx': x_idx_best[None] if x_idx_best.dim() == 0 else x_idx_best,
            'utility_batch': u2d,
            'start_gradient': dU_dx_init.squeeze(0) if dU_dx_init.dim() > 1 else dU_dx_init,
            'end_gradient': dU_dx_final.squeeze(0) if dU_dx_final.dim() > 1 else dU_dx_final,
            'U_initial': U_init.item() if isinstance(U_init, torch.Tensor) else U_init,
            'U_final': U_final.item() if isinstance(U_final, torch.Tensor) else U_final,
            'lr': lr,
            'line_search_fn': line_search_fn,
            'max_iter': max_iter
        }

    elif test_L2norm_constraint:
        return _optimize_norm_constraint(
            'L2',
            model_active,
            imgs_train,
            x_idx_best,
            mask,
            u2d,
            max_r_cap,
            lr,
            max_iter,
            line_search_fn,
            pixel_min,
            pixel_max,
            apply_soft_compression,
            apply_hard_clipping,
            compression_steepness,
            compression_transition_width,
            return_logf_moments
        )

    elif test_L4norm_constraint:
        return _optimize_norm_constraint(
            'L4',
            model_active,
            imgs_train,
            x_idx_best,
            mask,
            u2d,
            max_r_cap,
            lr,
            max_iter,
            line_search_fn,
            pixel_min,
            pixel_max,
            apply_soft_compression,
            apply_hard_clipping,
            compression_steepness,
            compression_transition_width,
            return_logf_moments
        )

    elif test_simple_optimization:
        return _test_simple_optimization(
            model_active,
            imgs_train,
            x_idx_best,
            mask,
            u2d,
            max_r_cap,
            lr,
            max_iter,
            line_search_fn,
            return_logf_moments
        )

    else:
        return {
            'img_idx': x_idx_best[None] if x_idx_best.dim() == 0 else x_idx_best,
            'utility_batch': u2d,
        }

def compute_simple_loglikelihood_from_model(x, r, **kwargs):

    """
    This function takes a model dictionary and computes the log-likelihood.

    Its basically a copy of the first lines of varGP

    """
    kernfun     = kwargs['fit_parameters'].get('kernfun')
    xtilde      = kwargs.get('xtilde').to(DEVICE, dtype=TORCH_DTYPE)

    theta = kwargs.get('hyperparams_tuple')[0]
    m_b   = kwargs['m_b']
    V_b   = kwargs['V_b']
    f_params = kwargs['f_params']

    C, mask, K_tilde_b, K_tilde_inv_b, B = GP_utils.get_final_K_vals(kwargs,)

    # mask        = kwargs.get('mask')
    # theta       = kwargs.get('hyperparams_tuple')[0]
    # C           = kwargs.get('C')
    # m_b           = kwargs.get('m_b')
    # V_b           = kwargs.get('V_b')
    # B             = kwargs.get('B')
    # K_tilde_b     = kwargs.get('K_tilde_b')
    # print('WARNING using ktilde from model but not from final_kernel')
    # K_tilde_inv_b = kwargs.get('K_tilde_inv_b')
    # f_params      = kwargs.get('f_params')

    Kvec    = kernfun(theta, x[:,mask], x2=None, C=C, dC=None, diag=True)

    K   = kernfun(theta, x[:,mask], xtilde[:,mask], C=C, dC=None, diag=False)
    K_b       = K @ B
    KKtilde_inv_b = K_b @ K_tilde_inv_b

    lambda_m, lambda_var = GP_utils.lambda_moments( x[:,mask], K_tilde_b, KKtilde_inv_b, Kvec, 
                                                   K_b, C, m_b, V_b, theta, kernfun=kernfun)
    f_mean               = GP_utils.mean_f_given_lambda_moments(f_params, lambda_m, lambda_var)

    loglikelihood, _, _ = GP_utils.compute_loglikelihood(r, f_mean, lambda_m, lambda_var, f_params, compute_grad_for_f_params=False)


    return loglikelihood

def get_performance_and_likelihood_model( model, h_imgs_spikes, h_images_idxs, imgs_train, img_test_reshaped, spike_counts_test_tensor, get_performance=True, kwargs_bootstrap={} ): 

    '''
    h_images_idxs: indexes of the holdout images in the full test set
    h_imgs_spikes: spikes corresponding to the holdout images on which we test the likelihood
    '''

    full_loglikelihood = compute_simple_loglikelihood_from_model(
            imgs_train[h_images_idxs], 
            h_imgs_spikes,
            **model.to_dict()
            ); 

    per_img_full_loglikelihood = full_loglikelihood / h_images_idxs.shape[0] 

    if get_performance:

        # Performance on Test images
        spk_count_test, spk_count_pred, \
        expl_variance, std_expl_variance, \
            r, b_cor_images, b_std_images, \
                b_cor_reps, b_std_reps, \
                    b_nested_cor, b_nested_sdt = \
            GP_analysis_utils.test_and_plot(model=model.to_dict(), 
                            img_test_reshaped=img_test_reshaped, 
                            imgs_train=imgs_train[model.in_use_idx],
                            spike_counts_test_tensor=spike_counts_test_tensor,
                            spike_counts_train=model.spike_counts,
                            show=False,
                            silent=True,
                            **kwargs_bootstrap,
                            )
        model_performance = {
            'r': r,

            'expl_var': expl_variance,
            'std_exp_var': std_expl_variance,

            'b_r_images': b_cor_images,
            'b_std_images': b_std_images,
            'b_r_reps': b_cor_reps,
            'b_std_reps': b_std_reps,
            'nested_b_r': b_nested_cor,
            'nested_b_std': b_nested_sdt,}
        
    else:
        model_performance = {}

    return model_performance, per_img_full_loglikelihood

if __name__ == "__main__":

    print("\n=== Testing distribution aware utility function ====")

    import sys
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parents[2] / 'ClosedLoopProject'
    sys.path.insert(0, str(REPO_ROOT))

    import pickle
    import os
    from config import config
    from pathlib import Path
    from gaussian_processes.Spatial_GP_repo import utils as GP_utils
    from gaussian_processes.Spatial_GP_repo.model import GPModel
    from gaussian_processes.Spatial_GP_repo import analysis_utils as GP_analysis_utils
    from gaussian_processes.Spatial_GP_repo import utility as GP_utility

    from src import main_utils
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np
    import torch
    import copy
    import time
    cmap = 'RdBu_r'
    TORCH_DTYPE = config.TORCH_DTYPE
    DEVICE = config.DEVICE

    import warnings
    warnings.filterwarnings("ignore", "UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor)")
    # set seeds 
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # May raise warnings for operations without deterministic implementations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Upload dataset used to train the model
    imgs_train, imgs_test = main_utils.upload_natural_image_dataset(
            dataset_path=config.img_dataset_path, astensor=True)

    img_test_reshaped = imgs_test.reshape(imgs_test.shape[0], 108, 108)

    init_model_name = 'active_orig_model_checkpoint_110_ch_133_ses_251120_ch133_rep0'
    init_model_path = '/home/idv-eqs8-pza/IDV_code/Experiments analysis and data/20251120/local_data_final_longrun/full_models/checkpoint_models/active_orig_model_checkpoint_110_ch_133_ses_251120_ch133_rep0.pkl'

    with open(init_model_path, "rb") as f:
        init_model_rep = pickle.load(f)

    model = GPModel(model_dict=init_model_rep)

    # x_star = imgs_train[ 0 ]
    # x_star = imgs_train[ 0:3 ]


    U_xstar1_counter, U_xstar2_counter, dU_xstar1_counter, dU_dxstar2_counter = 0,0,0,0

    for i in range(2000):
    # for i in [0]:
    # for i in [1370]:
    # for i in [90]:        
    # for i in [6]: #gradient nan
    # for i in [27]:        
        # print(f'=== Original Utility ===')

        # i = 7

        print(f'=== Utility i: {i}===')
        id = torch.tensor([i])

        result = GP_utility.batch_utility_w_grad(
            model,
            imgs_train, 
            id,
            max_r_cap=100,
            test_rms_constraint=False,            
            max_iter=0
        )

        U_orig = result['utility_batch'][0]

        # print(f'U_orig : {U_orig.item():.6f}')

        start_img_idx = i
        remaining_idx = torch.arange(imgs_train.shape[0], device=DEVICE)

        samplable_idx = remaining_idx != start_img_idx

        samplable_idx = remaining_idx[samplable_idx]

        remaining_imgs = imgs_train[samplable_idx]

        # print("==== Conditioned Utility ====")
        with torch.enable_grad():

            # Before calling conditioned_utility
            x_star = imgs_train[i].clone().requires_grad_(True)

            N = 1
            # N = 3000

            # U_xstar1 = conditioned_utility( 
            #     x_star, model, remaining_imgs, N=N, r_cutoff=100, 
            #     DEBUG_FIXED_X = False,
            #     DEBUG_SAME_X = False,
            #     DEBUG_FIX_LAMBDA_i=True
            #         )

            # dU_dxstar1 = torch.autograd.grad( U_xstar1, x_star )[0]

            # print(f"U_xstar1: {U_xstar1.item():.8f}")
            # print(f"dU_dxstar norm: {dU_dxstar1.norm().item():.6e}")

            # U_xstar2 = conditioned_utility_corrected( 
            #     x_star, model, remaining_imgs, N=N, r_cutoff=100, 
            #     DEBUG_FIXED_X = False,
            #     DEBUG_SAME_X = False,
            #     DEBUG_FIX_LAMBDA_i=True
            #     )
            
            # dU_dxstar2 = torch.autograd.grad( U_xstar2, x_star )[0]

            # print(f"U_xstar2: {U_xstar2.item():.8f}")
            # print(f"dU_dxstar2 norm: {dU_dxstar2.norm().item():.6e}")

            U_xstar = conditioned_utility_clean( 
                x_star, model, remaining_imgs, N=N, r_cutoff=100, 
                DEBUG_FIXED_X = True,
                DEBUG_SAME_X = False,
                DEBUG_FIX_LAMBDA_i=False,
                lambda_samples_per_x=10000

                )

            # dU_dxstar = torch.autograd.grad( U_xstar, x_star )[0]

            print(f"U_xstar:   {U_xstar.item():.8f}")
            # print(f"dU_dxstar norm: {dU_dxstar.norm().item():.6e}")

            U_xstar2 = conditioned_utility_clean( 
                x_star, model, remaining_imgs, N=N, r_cutoff=100, 
                DEBUG_FIXED_X = True,
                DEBUG_SAME_X = False,
                DEBUG_FIX_LAMBDA_i=False,
                lambda_samples_per_x=10000
                )

            print(f"U_xstar2:  {U_xstar2.item():.8f}")


            if (U_xstar2 - U_xstar).abs().item() > 1e-4:
                raise ValueError(f"U mismatch between two calls to conditioned_utility_clean: U_xstar2={U_xstar2.item():.6f}, U_xstar={U_xstar.item():.6f}")













            # if torch.any(torch.isnan(dU_dxstar1)):
            #     # raise ValueError("NaN in dU_dxstar1!")
            #     print("=====Warning: NaN in dU_dxstar1!")
            #     dU_xstar1_counter += 1
            # if torch.any(torch.isnan(dU_dxstar2)):
            #     # raise ValueError("NaN in dU_dxstar2!")
            #     print("======Warning: NaN in dU_dxstar2!")
            #     dU_dxstar2_counter += 1
            # if torch.any(torch.isnan(dU_dxstar)):
            #     raise ValueError("NaN in dU_dxstar!")
            

            # if U_xstar1 < -1.e-6:
            #     print("=====Warning: U_star1 negative!")
            #     U_xstar1_counter += 1
            # if U_xstar2 < -1.e-6:
            #     print("=====Warning: U_star2 negative!")
            #     U_xstar2_counter += 1

            if U_xstar < -1.e-6:
                raise ValueError("U_star negative!")



            # compare
            # if (U_xstar2 - U_xstar).abs().item() > 1e-3:
            #     raise ValueError(f"U mismatch between conditioned_utility and conditioned_utility_corrected: U_xstar2={U_xstar2.item():.6f}, U_xstar={U_xstar.item():.6f}")

    print(f"\nSummary after 2000 images:")
    print(f"U_xstar1 negative count: {U_xstar1_counter}")
    print(f"U_xstar2 negative count: {U_xstar2_counter}")
    print(f"dU_dxstar1 NaN count: {dU_xstar1_counter}")
    print(f"dU_dxstar2 NaN count: {dU_dxstar2_counter}")


    # print(f"dU_dxstar norm: {dU_dxstar.norm().item():.6e}")
    # ==== For timing ====

    # Warm-up run (first run is slower due to CUDA kernel compilation)

    # N=1
    # print("==== Conditioned Utility ====")

    # with torch.enable_grad():
    #     x_star = imgs_train[0].clone().requires_grad_(True)
    #     _ = conditioned_utility(x_star, model, imgs_train, N=N, r_cutoff=100, DEBUG=True)
    # torch.cuda.synchronize()

    # # Time the utility computation
    # with torch.enable_grad():
    #     x_star = imgs_train[0].clone().requires_grad_(True)
        
    #     torch.cuda.synchronize()  # Ensure previous ops complete
    #     t0 = time.perf_counter()
        
    #     U_xstar = conditioned_utility(x_star, model, imgs_train, N=N, r_cutoff=100, DEBUG=True)
        
    #     torch.cuda.synchronize()  # Wait for GPU to finish
    #     t1 = time.perf_counter()
        
    #     print(f"Utility forward pass: {(t1-t0)*1000:.2f} ms")
        
    #     # Time the gradient computation
    #     torch.cuda.synchronize()
    #     t2 = time.perf_counter()
        
    #     grad = torch.autograd.grad(U_xstar, x_star)[0]
        
    #     torch.cuda.synchronize()
    #     t3 = time.perf_counter()
        
    #     print(f"Gradient backward pass: {(t3-t2)*1000:.2f} ms")
    #     print(f"Total: {(t3-t0)*1000:.2f} ms")

    # # ==== Original Utility (compute_utility_single_image) for comparison ====
    # print("\n==== Original Utility (compute_utility_single_image) ====")

    # # Warm-up run
    # with torch.enable_grad():
    #     x_star_orig = imgs_train[0].clone().requires_grad_(True)
    #     _ = compute_utility_single_image(model, x_star_orig, max_r_cap=100)
    # torch.cuda.synchronize()

    # # Time the utility computation
    # with torch.enable_grad():
    #     x_star_orig = imgs_train[0].clone().requires_grad_(True)

    #     torch.cuda.synchronize()
    #     t0_orig = time.perf_counter()

    #     U_orig = compute_utility_single_image(model, x_star_orig, max_r_cap=100)

    #     torch.cuda.synchronize()
    #     t1_orig = time.perf_counter()

    #     print(f"Utility forward pass: {(t1_orig-t0_orig)*1000:.2f} ms")

    #     # Time the gradient computation
    #     torch.cuda.synchronize()
    #     t2_orig = time.perf_counter()

    #     grad_orig = torch.autograd.grad(U_orig, x_star_orig)[0]

    #     torch.cuda.synchronize()
    #     t3_orig = time.perf_counter()

    #     print(f"Gradient backward pass: {(t3_orig-t2_orig)*1000:.2f} ms")
    #     print(f"Total: {(t3_orig-t0_orig)*1000:.2f} ms")

    # # ==== Comparison Summary ====
    # print("\n==== Timing Comparison Summary ====")
    # print(f"                        Forward (ms)    Backward (ms)    Total (ms)")
    # print(f"conditioned_utility:        {(t1-t0)*1000:8.2f}          {(t3-t2)*1000:8.2f}        {(t3-t0)*1000:8.2f}")
    # print(f"compute_utility_single:     {(t1_orig-t0_orig)*1000:8.2f}          {(t3_orig-t2_orig)*1000:8.2f}        {(t3_orig-t0_orig)*1000:8.2f}")

    # # Value comparison
    # print(f"\n==== Value Comparison ====")
    # print(f"conditioned_utility:    U = {U_xstar.item():.6f}")
    # print(f"compute_utility_single: U = {U_orig.item():.6f}")
    # print(f"Difference: {abs(U_xstar.item() - U_orig.item()):.6e}")

    # # Gradient comparison
    # grad_diff = torch.abs(grad - grad_orig).max()
    # print(f"\nGradient max diff: {grad_diff.item():.6e}")
    # print(f"Gradient norms: conditioned={grad.norm().item():.4f}, original={grad_orig.norm().item():.4f}")

