"""Fine-tune a pretrained DDPM on PNAS natural images.

The pretrained model (64x64 grayscale, trained on ImageNet) is fine-tuned on
~3190 PNAS images (108x108) using random crops to 64x64 for data augmentation.

Usage:
    python finetune.py                                    # 50 epochs, default settings
    python finetune.py --num-epochs 2 --checkpoint-interval 1  # quick test
    python finetune.py --lr 5e-6 --batch-size 32          # custom hyperparameters

Output:
    ddpm-pnas-finetuned/                  # final fine-tuned pipeline
    ddpm-pnas-finetuned/checkpoint_epoch_N/  # intermediate checkpoints
    ddpm-pnas-finetuned/loss_curve.png    # training loss over epochs
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDPMScheduler

from generate_samples import PNAS_ABS_MAX, PNAS_DATA_PATH, DEFAULT_PRETRAINED_PATH


class PNASDataset(Dataset):
    """PNAS natural images with random crop and flip augmentation.

    Loads all splits (train + val + test = 3190 images).
    Each __getitem__ returns a tensor in ~[-1, 1] (divided by PNAS_ABS_MAX).
    Random crop 108->64 provides ~2025 crops per image.
    """

    def __init__(self, seed=42):
        data = np.load(PNAS_DATA_PATH)
        all_imgs = np.concatenate([data['images_train'], data['images_val'],
                                   data['images_test']], axis=0)
        # (3190, 108, 108, 1) -> (3190, 108, 108)
        self.images = all_imgs[:, :, :, 0].astype(np.float32)
        self.rng = np.random.default_rng(seed)
        print(f"Loaded {len(self.images)} PNAS images, shape {self.images.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (108, 108)

        # Random crop to 64x64
        max_offset = 108 - 64  # = 44
        y = self.rng.integers(0, max_offset + 1)
        x = self.rng.integers(0, max_offset + 1)
        img = img[y:y+64, x:x+64]

        # Random horizontal flip
        if self.rng.random() < 0.5:
            img = img[:, ::-1]

        # Random vertical flip
        if self.rng.random() < 0.5:
            img = img[::-1, :]

        # Normalize to ~[-1, 1]
        img = img / PNAS_ABS_MAX

        # To tensor: (64, 64) -> (1, 64, 64)
        tensor = torch.from_numpy(img.copy()).unsqueeze(0)
        return tensor


def train_one_epoch(unet, scheduler, dataloader, optimizer, device, gradient_accumulation=1):
    """Standard diffusion training: sample t, add noise, predict noise, MSE loss.

    Returns:
        float: average loss for the epoch
    """
    unet.train()
    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()
    for batch_idx, images in enumerate(dataloader):
        images = images.to(device)  # (B, 1, 64, 64)
        batch_size = images.shape[0]

        # Sample random timesteps for each image
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                  (batch_size,), device=device).long()

        # Sample noise
        noise = torch.randn_like(images)

        # Add noise to images (forward diffusion)
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        # Predict noise
        noise_pred = unet(noisy_images, timesteps).sample

        # MSE loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        loss = loss / gradient_accumulation
        loss.backward()

        if (batch_idx + 1) % gradient_accumulation == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation
        n_batches += 1

    return total_loss / n_batches


def generate_checkpoint_samples(pipeline, n, seed, epoch, save_dir):
    """Generate a quick sample grid at a checkpoint (50 steps for speed)."""
    pipeline.to(next(pipeline.unet.parameters()).device)
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    with torch.no_grad():
        output = pipeline(batch_size=n, num_inference_steps=50, generator=generator)

    images = output.images

    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            ax = axes[i, j]
            if idx < n:
                ax.imshow(np.array(images[idx]), cmap='gray', vmin=0, vmax=255)
            ax.axis('off')

    plt.suptitle(f'Epoch {epoch}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, f'samples_epoch_{epoch:03d}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Checkpoint samples saved to {path}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune DDPM on PNAS images')
    parser.add_argument('--model-path', type=str, default=DEFAULT_PRETRAINED_PATH,
                        help='Path to pretrained diffusers pipeline')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (low to avoid catastrophic forgetting)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--save-dir', type=str, default='ddpm-pnas-finetuned',
                        help='Directory to save fine-tuned model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Save checkpoint every N epochs (0=no checkpoints)')
    parser.add_argument('--num-sample-images', type=int, default=8,
                        help='Number of sample images at checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                        help='Gradient accumulation steps')
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Fine-tuning DDPM")
    print(f"  Model: {args.model_path}")
    print(f"  Epochs: {args.num_epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"  Save to: {args.save_dir}")
    print(f"  PNAS_ABS_MAX: {PNAS_ABS_MAX}")

    # Load pipeline
    pipeline = DDPMPipeline.from_pretrained(args.model_path)
    unet = pipeline.unet.to(args.device)
    scheduler = pipeline.scheduler

    # Dataset and dataloader
    dataset = PNASDataset(seed=args.seed)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=False)
    print(f"  Dataloader: {len(dataloader)} batches per epoch")

    # Optimizer
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)

    # Output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop
    losses = []
    t_total = time.time()

    for epoch in range(args.num_epochs):
        t_epoch = time.time()
        avg_loss = train_one_epoch(unet, scheduler, dataloader, optimizer,
                                   args.device, args.gradient_accumulation)
        elapsed = time.time() - t_epoch
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{args.num_epochs}: loss={avg_loss:.6f} ({elapsed:.1f}s)")

        # Checkpoint
        if args.checkpoint_interval > 0 and (epoch + 1) % args.checkpoint_interval == 0:
            ckpt_dir = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1:03d}')
            pipeline.save_pretrained(ckpt_dir)
            print(f"  Checkpoint saved to {ckpt_dir}")
            generate_checkpoint_samples(pipeline, args.num_sample_images,
                                        args.seed, epoch + 1, args.save_dir)

    total_time = time.time() - t_total
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Best loss: {min(losses):.6f} (epoch {losses.index(min(losses))+1})")

    # Save final model
    pipeline.save_pretrained(args.save_dir)
    print(f"Final model saved to {args.save_dir}")

    # Save loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses)+1), losses, 'b-o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Fine-tuning Loss')
    ax.grid(True, alpha=0.3)
    loss_path = os.path.join(args.save_dir, 'loss_curve.png')
    plt.tight_layout()
    plt.savefig(loss_path, dpi=100)
    plt.close()
    print(f"Loss curve saved to {loss_path}")


if __name__ == '__main__':
    main()
