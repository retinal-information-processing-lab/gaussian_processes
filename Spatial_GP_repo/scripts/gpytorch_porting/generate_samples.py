"""Generate samples from a DDPM pipeline (pretrained or fine-tuned).

Usage:
    python generate_samples.py                           # 16 images, full 1000 steps
    python generate_samples.py --num-images 4 --num-steps 50  # quick test
    python generate_samples.py --model-path ddpm-pnas-finetuned  # from fine-tuned
    python generate_samples.py --no-comparison            # skip PNAS comparison row
    python generate_samples.py --save-individual          # save each image separately
"""

import argparse
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline

# --- Shared constants (imported by finetune.py and generate_samples_finetuned.py) ---

# Verified: max(abs(all_images.min()), abs(all_images.max())) across all 3190 PNAS images
PNAS_ABS_MAX = 2.478047

# Absolute path — worktree-safe (relative traversal breaks in git worktrees)
PNAS_DATA_PATH = '/home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/notebooks/PNAS_paper_sorted_data.npz'

DEFAULT_PRETRAINED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        'ddpm-imagenet-grayscale')
DEFAULT_FINETUNED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'ddpm-pnas-finetuned')


def create_parser():
    """Create argparse parser. Returned so generate_samples_finetuned.py can reuse it."""
    parser = argparse.ArgumentParser(description='Generate samples from a DDPM pipeline')
    parser.add_argument('--model-path', type=str, default=DEFAULT_PRETRAINED_PATH,
                        help='Path to diffusers pipeline directory')
    parser.add_argument('--num-images', type=int, default=16,
                        help='Number of images to generate')
    parser.add_argument('--num-steps', type=int, default=1000,
                        help='Number of denoising steps (1000=full, 50=fast preview)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='generated_samples',
                        help='Directory to save outputs')
    parser.add_argument('--save-individual', action='store_true',
                        help='Save each generated image as a separate PNG')
    parser.add_argument('--no-comparison', action='store_true',
                        help='Skip PNAS comparison row in the grid')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    return parser


def load_pnas_comparison(n, seed):
    """Load n random PNAS images, center-crop to 64x64, convert to [0,1] for display.

    Returns:
        np.ndarray of shape (n, 64, 64), values in [0, 1]
    """
    data = np.load(PNAS_DATA_PATH)
    all_imgs = np.concatenate([data['images_train'], data['images_val'],
                               data['images_test']], axis=0)
    # all_imgs shape: (3190, 108, 108, 1)

    rng = np.random.default_rng(seed)
    indices = rng.choice(all_imgs.shape[0], size=n, replace=False)
    selected = all_imgs[indices]  # (n, 108, 108, 1)

    # Center crop to 64x64
    offset = (108 - 64) // 2  # = 22
    cropped = selected[:, offset:offset+64, offset:offset+64, 0]  # (n, 64, 64)

    # Convert from PNAS range to [0, 1] display space
    # Same mapping as pipeline: model_space = pnas / PNAS_ABS_MAX, display = (model + 1) / 2
    display = (cropped / PNAS_ABS_MAX + 1.0) / 2.0
    display = np.clip(display, 0.0, 1.0)

    return display


def generate_images(pipeline, num_images, num_steps, seed, device):
    """Generate images from the pipeline in sub-batches of 16.

    Returns:
        list of PIL Images
    """
    pipeline = pipeline.to(device)

    all_images = []
    batch_size = min(16, num_images)
    num_batches = (num_images + batch_size - 1) // batch_size

    generator = torch.Generator(device=device).manual_seed(seed)

    for i in range(num_batches):
        current_batch = min(batch_size, num_images - len(all_images))
        output = pipeline(
            batch_size=current_batch,
            num_inference_steps=num_steps,
            generator=generator,
        )
        all_images.extend(output.images)

    return all_images[:num_images]


def make_grid(generated_images, real_images, save_path):
    """Create a comparison grid: generated images on top rows, real PNAS on bottom row.

    Args:
        generated_images: list of PIL Images (grayscale, 64x64)
        real_images: np.ndarray (n, 64, 64) in [0,1], or None if no comparison
        save_path: where to save the PNG
    """
    n_gen = len(generated_images)
    n_cols = min(8, n_gen)
    n_gen_rows = (n_gen + n_cols - 1) // n_cols
    n_real_rows = 0 if real_images is None else 1
    n_rows = n_gen_rows + n_real_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Plot generated images
    for i in range(n_gen_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            ax = axes[i, j]
            if idx < n_gen:
                img = np.array(generated_images[idx])
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
                if i == 0 and j == 0:
                    ax.set_title('Generated', fontsize=9)
            ax.axis('off')

    # Plot real PNAS images (bottom row)
    if real_images is not None:
        for j in range(n_cols):
            ax = axes[n_gen_rows, j]
            if j < len(real_images):
                ax.imshow(real_images[j], cmap='gray', vmin=0.0, vmax=1.0)
                if j == 0:
                    ax.set_title('PNAS (real)', fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grid saved to {save_path}")


def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    print(f"Model path: {args.model_path}")
    print(f"Generating {args.num_images} images with {args.num_steps} steps, seed={args.seed}")

    # Load pipeline
    t0 = time.time()
    pipeline = DDPMPipeline.from_pretrained(args.model_path)
    print(f"Pipeline loaded in {time.time() - t0:.1f}s")

    # Generate
    t0 = time.time()
    images = generate_images(pipeline, args.num_images, args.num_steps, args.seed, args.device)
    print(f"Generated {len(images)} images in {time.time() - t0:.1f}s")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save individual images if requested
    if args.save_individual:
        for i, img in enumerate(images):
            img.save(os.path.join(args.output_dir, f"sample_{i:04d}.png"))
        print(f"Saved {len(images)} individual PNGs to {args.output_dir}/")

    # Load PNAS comparison images
    real_images = None
    if not args.no_comparison:
        n_compare = min(8, args.num_images)
        real_images = load_pnas_comparison(n_compare, seed=args.seed)

    # Make grid
    grid_path = os.path.join(args.output_dir, 'sample_grid.png')
    make_grid(images, real_images, grid_path)


if __name__ == '__main__':
    main()
