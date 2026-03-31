"""
Diffusion model core module: noise schedule, U-Net architecture, sampling.

Self-contained DDPM implementation for 64x64 grayscale images.
No imports from the GP codebase.

Architecture:
    Tiny U-Net with 3 downsampling levels [1->32->64->128 channels]
    Spatial dims: 64->32->16->8 (bottleneck) ->16->32->64
    Time conditioning via sinusoidal embedding + MLP
    Skip connections at each resolution level

Noise schedule:
    Cosine schedule (Nichol & Dhariwal, 2021), T=1000

Usage:
    from diffusion_model import UNet, cosine_schedule, q_sample, p_sample, sample

References:
    Ho et al. (2020) - Denoising Diffusion Probabilistic Models
    Nichol & Dhariwal (2021) - Improved Denoising Diffusion Probabilistic Models
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Noise Schedule
# =============================================================================

def cosine_schedule(T, s=0.008):
    """
    Cosine noise schedule from Nichol & Dhariwal (2021).

    Defines alpha_bar_t directly:
        f(t) = cos^2((t/T + s) / (1 + s) * pi/2)
        alpha_bar_t = f(t) / f(0)

    Args:
        T: number of diffusion timesteps
        s: small offset to prevent beta_t from being too small near t=0

    Returns:
        dict with precomputed schedule constants, all shape (T+1,) or (T,)
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]

    # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}, clipped to max 0.999
    # beta[0] is unused (t starts at 1), but we keep the indexing clean
    beta = torch.zeros(T + 1, dtype=torch.float64)
    beta[1:] = 1 - alpha_bar[1:] / alpha_bar[:-1]
    beta = beta.clamp(max=0.999)

    # Recompute alpha_bar from clipped betas for consistency
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    # Precompute all constants needed for training and sampling
    schedule = {
        'T': T,
        'beta': beta.float(),                                    # (T+1,)
        'alpha': alpha.float(),                                  # (T+1,)
        'alpha_bar': alpha_bar.float(),                          # (T+1,)
        'sqrt_alpha_bar': alpha_bar.sqrt().float(),              # (T+1,)
        'sqrt_one_minus_alpha_bar': (1 - alpha_bar).sqrt().float(),  # (T+1,)
        'sqrt_recip_alpha': (1 / alpha).sqrt().float(),          # (T+1,)
        'beta_over_sqrt_one_minus_alpha_bar': (                  # (T+1,)
            beta / (1 - alpha_bar).sqrt()
        ).float(),
    }

    # Handle t=0 edge case (alpha_bar[0] = 1, so 1-alpha_bar[0] = 0)
    schedule['beta_over_sqrt_one_minus_alpha_bar'][0] = 0.0

    return schedule


# =============================================================================
# Forward Process
# =============================================================================

def q_sample(x_0, t, schedule, noise=None):
    """
    Forward process: add noise to clean image at timestep t.

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

    Args:
        x_0: clean images, shape (B, 1, H, W)
        t: timesteps, shape (B,) with values in {0, ..., T}
        schedule: dict from cosine_schedule()
        noise: optional pre-sampled noise, shape (B, 1, H, W)

    Returns:
        x_t: noisy images, shape (B, 1, H, W)
        noise: the noise that was added
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    # Index schedule on CPU (schedule tensors are always CPU), then move to device
    t_cpu = t.cpu()
    sqrt_ab = schedule['sqrt_alpha_bar'][t_cpu].view(-1, 1, 1, 1).to(x_0.device)
    sqrt_1m_ab = schedule['sqrt_one_minus_alpha_bar'][t_cpu].view(-1, 1, 1, 1).to(x_0.device)

    x_t = sqrt_ab * x_0 + sqrt_1m_ab * noise
    return x_t, noise


# =============================================================================
# Sinusoidal Time Embedding
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps scalar timestep t to a d_emb-dimensional vector using sinusoidal
    position encoding, then passes through a small MLP.

    emb(t)_{2k}   = sin(t / 10000^{2k/d_emb})
    emb(t)_{2k+1} = cos(t / 10000^{2k/d_emb})
    """

    def __init__(self, d_emb):
        super().__init__()
        self.d_emb = d_emb
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, d_emb * 4),
            nn.SiLU(),
            nn.Linear(d_emb * 4, d_emb),
        )

    def forward(self, t):
        """
        Args:
            t: timesteps, shape (B,), integer or float

        Returns:
            embedding, shape (B, d_emb)
        """
        half = self.d_emb // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, d_emb)
        return self.mlp(emb)


# =============================================================================
# U-Net Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """
    Two convolutions with GroupNorm and SiLU activation.
    Conv2d(3x3, pad=1) -> GroupNorm -> SiLU -> Conv2d(3x3, pad=1) -> GroupNorm -> SiLU

    Time embedding is added after the first normalization+activation.
    """

    def __init__(self, in_ch, out_ch, d_emb, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.time_proj = nn.Linear(d_emb, out_ch)

        # Residual connection if channel dims differ
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        """
        Args:
            x: input features, shape (B, in_ch, H, W)
            t_emb: time embedding, shape (B, d_emb)

        Returns:
            output features, shape (B, out_ch, H, W)
        """
        h = F.silu(self.norm1(self.conv1(x)))
        # Add time embedding (broadcast over spatial dims)
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.residual(x)


class Downsample(nn.Module):
    """Halve spatial dimensions with stride-2 convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Double spatial dimensions with nearest-neighbor + convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# =============================================================================
# U-Net
# =============================================================================

class UNet(nn.Module):
    """
    Tiny U-Net for DDPM noise prediction on 64x64 grayscale images.

    Architecture:
        Input:  (B, 1, 64, 64)
        Encoder: 3 levels [32, 64, 128 channels], spatial: 64->32->16->8
        Middle:  128 channels at 8x8
        Decoder: 3 levels [128, 64, 32 channels], spatial: 8->16->32->64
        Output: (B, 1, 64, 64) predicted noise

    Time conditioning: sinusoidal embedding (d_emb=128) -> MLP -> added to features.
    Skip connections: concatenation at each resolution level.
    """

    def __init__(self, in_channels=1, channels=(32, 64, 128), d_emb=128):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(d_emb)

        # Initial projection: 1 -> channels[0]
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch_in = channels[0]
        for ch_out in channels:
            self.down_blocks.append(ConvBlock(ch_in, ch_out, d_emb))
            self.downsamples.append(Downsample(ch_out))
            ch_in = ch_out

        # Middle (bottleneck at 8x8)
        self.mid_block = ConvBlock(channels[-1], channels[-1], d_emb)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, ch_out in enumerate(reversed(channels)):
            # Input: upsampled features (ch_in) + skip features (ch_out)
            ch_skip = ch_out  # channels from corresponding encoder level
            self.upsamples.append(Upsample(ch_in))
            self.up_blocks.append(ConvBlock(ch_in + ch_skip, ch_out, d_emb))
            ch_in = ch_out

        # Final projection: channels[0] -> 1
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, 1),
        )

    def forward(self, x, t):
        """
        Predict noise from noisy image and timestep.

        Args:
            x: noisy image, shape (B, 1, 64, 64)
            t: timestep, shape (B,), integer values in {0, ..., T}

        Returns:
            predicted noise, shape (B, 1, 64, 64)
        """
        t_emb = self.time_emb(t)

        # Initial projection
        h = self.init_conv(x)

        # Encoder: save skip connections
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        # Middle
        h = self.mid_block(h, t_emb)

        # Decoder: use skip connections (reverse order)
        for block, up, skip in zip(self.up_blocks, self.upsamples, reversed(skips)):
            h = up(h)
            h = torch.cat([h, skip], dim=1)  # concatenate along channel axis
            h = block(h, t_emb)

        return self.final_conv(h)


# =============================================================================
# Reverse Process (Sampling)
# =============================================================================

@torch.no_grad()
def p_sample(model, x_t, t, schedule):
    """
    One step of the reverse process: denoise x_t to x_{t-1}.

    mu_t = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_hat)
    x_{t-1} = mu_t + sqrt(beta_t) * z   (z=0 if t=1)

    Args:
        model: trained UNet
        x_t: current noisy image, shape (B, 1, H, W)
        t: current timestep (scalar integer)
        schedule: dict from cosine_schedule()

    Returns:
        x_{t-1}: slightly less noisy image
    """
    B = x_t.shape[0]
    t_batch = torch.full((B,), t, device=x_t.device, dtype=torch.long)

    # Predict noise
    eps_hat = model(x_t, t_batch)

    # Compute reverse mean
    sqrt_recip_alpha = schedule['sqrt_recip_alpha'][t].to(x_t.device)
    beta_coeff = schedule['beta_over_sqrt_one_minus_alpha_bar'][t].to(x_t.device)
    mu = sqrt_recip_alpha * (x_t - beta_coeff * eps_hat)

    if t > 1:
        beta_t = schedule['beta'][t].to(x_t.device)
        noise = torch.randn_like(x_t)
        return mu + beta_t.sqrt() * noise
    else:
        return mu


@torch.no_grad()
def sample(model, schedule, n_samples=1, image_shape=(1, 64, 64), device='cuda'):
    """
    Generate images from pure noise using the full reverse process.

    Args:
        model: trained UNet
        schedule: dict from cosine_schedule()
        n_samples: number of images to generate
        image_shape: (C, H, W) shape of each image
        device: torch device

    Returns:
        generated images, shape (n_samples, C, H, W)
    """
    model.eval()
    T = schedule['T']

    # Start from pure noise
    x = torch.randn(n_samples, *image_shape, device=device)

    # Reverse process: t = T-1, T-2, ..., 1
    # Skip t=T because alpha_bar[T] ≈ 0 makes alpha[T] ≈ 0.001, so
    # sqrt(1/alpha[T]) ≈ 31.6 -- the reverse formula amplifies any prediction
    # error by 31x at that single step, causing divergence. Starting from T-1
    # (where sqrt(1/alpha) ≈ 2.0) is standard practice with cosine schedules.
    for t in range(T - 1, 0, -1):
        x = p_sample(model, x, t, schedule)

    return x


# =============================================================================
# Utilities
# =============================================================================

def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
