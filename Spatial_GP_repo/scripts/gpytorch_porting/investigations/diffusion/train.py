"""
Train a DDPM on 64x64 grayscale natural image patches.

Self-contained training script. No imports from the GP codebase.

Data pipeline:
    1. Load PNAS 108x108 images (train + val)
    2. Random 64x64 crop (different each epoch)
    3. Random D4 symmetry transform (8 variants)
    4. Normalize to [-1, 1] range

Usage:
    python train.py                          # train with defaults
    python train.py --epochs 1000            # more epochs
    python train.py --batch-size 64          # larger batch
    python train.py --lr 2e-4               # different learning rate
    python train.py --resume checkpoint.pt   # resume from checkpoint
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from diffusion_model import UNet, cosine_schedule, q_sample, count_parameters


# =============================================================================
# Dataset
# =============================================================================

# Path to PNAS dataset — canonical absolute path (npz is gitignored, not in worktrees)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(
    os.path.expanduser('~'),
    'IDV_code', 'ClosedLoopProject', 'gaussian_processes',
    'Spatial_GP_repo', 'notebooks', 'PNAS_paper_sorted_data.npz'
)


class NaturalImageDataset(Dataset):
    """
    Dataset of 108x108 natural image patches with random crop + D4 augmentation.

    Each __getitem__ returns a random 64x64 crop with a random D4 transformation
    applied. Since the crop and transform are sampled fresh each call, every
    epoch sees different augmented versions.
    """

    def __init__(self, images_108, crop_size=64, scale_factor=None):
        """
        Args:
            images_108: numpy array of shape (N, 108, 108, 1), float32
            crop_size: output spatial size (default 64)
            scale_factor: divide images by this to normalize to ~[-1, 1].
                         If None, computed from the data.
        """
        # Store as (N, 1, 108, 108) for PyTorch convention
        self.images = torch.from_numpy(images_108).permute(0, 3, 1, 2).float()
        self.crop_size = crop_size
        self.max_offset = self.images.shape[-1] - crop_size  # 108 - 64 = 44

        if scale_factor is None:
            self.scale_factor = self.images.abs().max().item()
        else:
            self.scale_factor = scale_factor

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]  # (1, 108, 108)

        # Random 64x64 crop
        top = torch.randint(0, self.max_offset + 1, (1,)).item()
        left = torch.randint(0, self.max_offset + 1, (1,)).item()
        img = img[:, top:top + self.crop_size, left:left + self.crop_size]

        # Random D4 transformation (8 elements of dihedral group)
        img = random_d4(img)

        # Normalize to [-1, 1]
        img = img / self.scale_factor

        return img


def random_d4(img):
    """
    Apply a random element of the D4 symmetry group.

    D4 = {identity, rot90, rot180, rot270} x {identity, horizontal_flip}
    = 8 transformations total.

    Args:
        img: tensor of shape (C, H, W)

    Returns:
        transformed image, same shape
    """
    # Random rotation: 0, 90, 180, or 270 degrees
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        img = torch.rot90(img, k, dims=(1, 2))

    # Random horizontal flip
    if torch.rand(1).item() > 0.5:
        img = torch.flip(img, dims=(2,))

    return img


# =============================================================================
# Training
# =============================================================================

def train(args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- Load data ----
    data_path = os.path.normpath(args.data)
    print(f'Loading data from: {data_path}')
    data = np.load(data_path)

    # Combine train + val for diffusion training (no neural response labels needed)
    images_train = data['images_train']  # (2910, 108, 108, 1)
    images_val = data['images_val']      # (250, 108, 108, 1)
    all_images = np.concatenate([images_train, images_val], axis=0)
    print(f'Total images: {all_images.shape[0]} (train={images_train.shape[0]}, val={images_val.shape[0]})')
    print(f'Image shape: {all_images.shape[1:]}')
    print(f'Pixel range: [{all_images.min():.4f}, {all_images.max():.4f}]')

    # Compute normalization scale from data
    scale_factor = float(np.abs(all_images).max())
    print(f'Scale factor: {scale_factor:.4f} (dividing by this normalizes to ~[-1, 1])')

    dataset = NaturalImageDataset(all_images, crop_size=64, scale_factor=scale_factor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Create model ----
    model = UNet().to(device)
    n_params = count_parameters(model)
    print(f'UNet parameters: {n_params:,} ({n_params/1e6:.2f}M)')

    # ---- Noise schedule ----
    schedule = cosine_schedule(args.T)
    print(f'Noise schedule: cosine, T={args.T}')

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'Optimizer: Adam, lr={args.lr}')

    # ---- Resume from checkpoint ----
    start_epoch = 0
    loss_history = []

    if args.resume:
        print(f'Resuming from: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        loss_history = ckpt.get('loss_history', [])
        print(f'Resumed at epoch {start_epoch}')

    # ---- Training loop ----
    print(f'\nStarting training: epochs={args.epochs}, batch_size={args.batch_size}')
    print(f'Batches per epoch: {len(loader)}')
    print('-' * 60)

    checkpoint_dir = os.path.join(SCRIPT_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    T = args.T
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            x_0 = batch.to(device)  # (B, 1, 64, 64), normalized to [-1, 1]

            # Sample random timesteps for each image in the batch
            t = torch.randint(1, T + 1, (x_0.shape[0],), device=device)

            # Forward diffusion: add noise
            x_t, noise = q_sample(x_0, t, schedule)

            # Predict noise
            noise_pred = model(x_t, t)

            # MSE loss
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        # Print progress
        elapsed = time.time() - t_start
        if (epoch + 1) % args.print_every == 0 or epoch == start_epoch:
            print(f'Epoch {epoch+1:4d}/{args.epochs}  loss={avg_loss:.6f}  '
                  f'elapsed={elapsed:.1f}s')

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(checkpoint_dir, f'ddpm_epoch{epoch+1:04d}.pt')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_history': loss_history,
                'scale_factor': scale_factor,
                'config': {
                    'T': args.T,
                    'lr': args.lr,
                    'batch_size': args.batch_size,
                    'crop_size': 64,
                    'n_images': all_images.shape[0],
                },
            }, ckpt_path)
            print(f'  -> Saved checkpoint: {ckpt_path}')

    total_time = time.time() - t_start
    print('-' * 60)
    print(f'Training complete. Total time: {total_time:.1f}s ({total_time/60:.1f} min)')
    print(f'Final loss: {loss_history[-1]:.6f}')

    # Save loss curve
    loss_path = os.path.join(checkpoint_dir, 'loss_history.npy')
    np.save(loss_path, np.array(loss_history))
    print(f'Loss history saved to: {loss_path}')


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train DDPM on natural image patches')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--T', type=int, default=1000,
                        help='Number of diffusion timesteps (default: 1000)')
    parser.add_argument('--print-every', type=int, default=10,
                        help='Print loss every N epochs (default: 10)')
    parser.add_argument('--save-every', type=int, default=100,
                        help='Save checkpoint every N epochs (default: 100)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA_PATH,
                        help='Path to PNAS .npz data file')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
