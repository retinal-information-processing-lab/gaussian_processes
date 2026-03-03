"""
Generate and evaluate samples from a trained DDPM.

Loads a checkpoint, generates images via the reverse process, and compares
to real training images. Self-contained -- no GP codebase imports.

Usage:
    python sample.py --checkpoint checkpoints/ddpm_epoch0500.pt
    python sample.py --checkpoint checkpoints/ddpm_epoch0500.pt --n-samples 64
    python sample.py --checkpoint checkpoints/ddpm_epoch0500.pt --save-dir samples/
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion_model import UNet, cosine_schedule, sample


# Same default data path as train.py
DEFAULT_DATA_PATH = os.path.join(
    os.path.expanduser('~'),
    'IDV_code', 'ClosedLoopProject', 'gaussian_processes',
    'Spatial_GP_repo', 'notebooks', 'PNAS_paper_sorted_data.npz'
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_checkpoint(checkpoint_path, device):
    """Load model and config from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = UNet().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    config = ckpt['config']
    scale_factor = ckpt['scale_factor']
    loss_history = ckpt.get('loss_history', [])
    epoch = ckpt.get('epoch', -1)

    return model, config, scale_factor, loss_history, epoch


def generate_samples(model, schedule, n_samples, device, scale_factor):
    """
    Generate images and rescale from [-1, 1] back to original pixel range.

    Returns:
        generated: numpy array (n_samples, 64, 64), original pixel scale
    """
    raw = sample(model, schedule, n_samples=n_samples,
                 image_shape=(1, 64, 64), device=device)
    # Rescale from [-1, 1] back to original data range
    raw = raw * scale_factor
    # (n, 1, 64, 64) -> (n, 64, 64) numpy
    return raw.squeeze(1).cpu().numpy()


def load_real_images(data_path, scale_factor):
    """Load real images for comparison. Returns center-cropped 64x64 patches."""
    data = np.load(data_path)
    all_images = np.concatenate([data['images_train'], data['images_val']], axis=0)
    # Center crop 108x108 -> 64x64
    offset = (108 - 64) // 2  # = 22
    cropped = all_images[:, offset:offset+64, offset:offset+64, 0]  # (N, 64, 64)
    return cropped


def plot_image_grid(images, n_rows, n_cols, title, save_path, vmin, vmax):
    """Plot a grid of grayscale images."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    fig.suptitle(title, fontsize=14)

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            ax = axes[i, j] if n_rows > 1 else axes[j]
            if idx < len(images):
                ax.imshow(images[idx], cmap='gray', vmin=vmin, vmax=vmax)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_comparison(generated, real, save_path, vmin, vmax):
    """Side-by-side grid: generated (left) vs real (right)."""
    n_show = min(16, len(generated), len(real))
    n_cols = 8
    n_rows = 4  # 2 rows generated, 2 rows real

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5 + 0.5))
    fig.suptitle('Generated (top 2 rows) vs Real (bottom 2 rows)', fontsize=12)

    for i in range(n_show):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        ax.imshow(generated[i], cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')

    # Select random real images for comparison
    rng = np.random.RandomState(42)
    real_idx = rng.choice(len(real), n_show, replace=False)
    for i in range(n_show):
        row, col = divmod(i, n_cols)
        ax = axes[row + 2, col]
        ax.imshow(real[real_idx[i]], cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_pixel_histogram(generated, real, save_path):
    """Compare pixel value distributions of generated vs real images."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.hist(real.ravel(), bins=100, alpha=0.5, density=True, label='Real', color='blue')
    ax.hist(generated.ravel(), bins=100, alpha=0.5, density=True, label='Generated', color='orange')
    ax.set_xlabel('Pixel value')
    ax.set_ylabel('Density')
    ax.set_title('Pixel value distribution: Generated vs Real')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_power_spectrum(generated, real, save_path):
    """
    Compare azimuthally-averaged power spectra.
    Natural images should follow ~1/f^2 (linear with slope -2 on log-log).
    """
    def azimuthal_average_psd(images):
        """Compute azimuthally averaged power spectral density."""
        H, W = images.shape[1], images.shape[2]
        # Compute 2D FFT for each image, average power
        fft_images = np.fft.fft2(images, axes=(1, 2))
        power = np.abs(fft_images) ** 2
        avg_power = power.mean(axis=0)

        # Azimuthal average
        cy, cx = H // 2, W // 2
        Y, X = np.ogrid[:H, :W]
        R = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
        max_r = min(cx, cy)

        # Shift to center
        avg_power_shifted = np.fft.fftshift(avg_power)

        radial_profile = np.zeros(max_r)
        for r in range(max_r):
            mask = R == r
            if mask.any():
                radial_profile[r] = avg_power_shifted[mask].mean()

        return radial_profile

    psd_gen = azimuthal_average_psd(generated)
    psd_real = azimuthal_average_psd(real[:len(generated)])

    freqs = np.arange(len(psd_gen))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.loglog(freqs[1:], psd_real[1:], 'b-', alpha=0.8, label='Real', linewidth=2)
    ax.loglog(freqs[1:], psd_gen[1:], 'r--', alpha=0.8, label='Generated', linewidth=2)

    # Reference 1/f^2 line
    ref = psd_real[2] * (freqs[1:] / 2.0) ** (-2)
    ax.loglog(freqs[1:], ref, 'k:', alpha=0.3, label='1/f^2 reference')

    ax.set_xlabel('Spatial frequency')
    ax.set_ylabel('Power')
    ax.set_title('Azimuthally-averaged power spectrum')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_loss_curve(loss_history, save_path):
    """Plot training loss over epochs."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def print_statistics(generated, real):
    """Print summary statistics for comparison."""
    print('\n' + '=' * 60)
    print('Image Statistics Comparison')
    print('=' * 60)
    print(f'{"Statistic":<25} {"Generated":>12} {"Real":>12}')
    print('-' * 50)
    print(f'{"Mean":<25} {generated.mean():>12.4f} {real.mean():>12.4f}')
    print(f'{"Std":<25} {generated.std():>12.4f} {real.std():>12.4f}')
    print(f'{"Min":<25} {generated.min():>12.4f} {real.min():>12.4f}')
    print(f'{"Max":<25} {generated.max():>12.4f} {real.max():>12.4f}')
    print(f'{"Median":<25} {np.median(generated):>12.4f} {np.median(real):>12.4f}')
    print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='Generate and evaluate DDPM samples')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint (.pt)')
    parser.add_argument('--n-samples', type=int, default=64,
                        help='Number of images to generate (default: 64)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory for output plots (default: same as checkpoint)')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA_PATH,
                        help='Path to PNAS .npz data file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Set seed for reproducible generation
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load model ----
    print(f'Loading checkpoint: {args.checkpoint}')
    model, config, scale_factor, loss_history, epoch = load_checkpoint(
        args.checkpoint, device
    )
    print(f'  Epoch: {epoch + 1}')
    print(f'  Config: {config}')
    print(f'  Scale factor: {scale_factor:.4f}')

    # ---- Setup output directory ----
    if args.save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, 'samples')
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # ---- Generate samples ----
    schedule = cosine_schedule(config['T'])
    print(f'\nGenerating {args.n_samples} images...')
    import time
    t0 = time.time()
    generated = generate_samples(model, schedule, args.n_samples, device, scale_factor)
    gen_time = time.time() - t0
    print(f'  Generated {args.n_samples} images in {gen_time:.1f}s '
          f'({gen_time/args.n_samples:.2f}s per image)')
    print(f'  Shape: {generated.shape}')

    # ---- Load real images for comparison ----
    real = load_real_images(args.data, scale_factor)
    print(f'  Real images loaded: {real.shape}')

    # ---- Pixel range from real data (for consistent plotting) ----
    vmin = real.min()
    vmax = real.max()
    print(f'  Plot range: [{vmin:.4f}, {vmax:.4f}]')

    # ---- Statistics ----
    print_statistics(generated, real)

    # ---- Plots ----
    prefix = f'epoch{epoch+1:04d}'

    # Generated grid
    plot_image_grid(
        generated[:min(64, args.n_samples)], 8, 8,
        f'Generated Images (epoch {epoch+1})',
        os.path.join(save_dir, f'{prefix}_generated_grid.png'),
        vmin, vmax
    )

    # Real grid (for comparison)
    rng = np.random.RandomState(42)
    real_subset = real[rng.choice(len(real), 64, replace=False)]
    plot_image_grid(
        real_subset, 8, 8,
        'Real Images (center crops)',
        os.path.join(save_dir, f'{prefix}_real_grid.png'),
        vmin, vmax
    )

    # Side-by-side comparison
    plot_comparison(
        generated, real,
        os.path.join(save_dir, f'{prefix}_comparison.png'),
        vmin, vmax
    )

    # Pixel histogram
    plot_pixel_histogram(
        generated, real,
        os.path.join(save_dir, f'{prefix}_pixel_histogram.png')
    )

    # Power spectrum
    plot_power_spectrum(
        generated, real,
        os.path.join(save_dir, f'{prefix}_power_spectrum.png')
    )

    # Loss curve
    if loss_history:
        plot_loss_curve(
            loss_history,
            os.path.join(save_dir, f'{prefix}_loss_curve.png')
        )

    print(f'\nAll plots saved to: {save_dir}')


if __name__ == '__main__':
    main()
