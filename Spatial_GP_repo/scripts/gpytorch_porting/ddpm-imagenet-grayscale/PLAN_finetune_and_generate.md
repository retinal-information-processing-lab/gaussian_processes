# Plan: DDPM Fine-tuning and Generation Scripts

## Context

We have a pretrained DDPM (99.5M params, 64x64 grayscale, diffusers format) trained on ImageNet, sitting at `ddpm-imagenet-grayscale/`. We want to (1) generate samples from it, (2) fine-tune it on the PNAS dataset (~3190 images, 108x108), and (3) generate samples from the fine-tuned model. Three scripts, all in the `gpytorch_porting/` directory.

---

## Normalization Strategy

PNAS pixels are in [-2.40, 2.48] (z-scored). DDPM model works in [-1, 1].

**Mapping**: `x_model = x_pnas / PNAS_ABS_MAX` where `PNAS_ABS_MAX = 2.478047`

- Preserves zero-mean (0 maps to 0)
- Nearly symmetric: maps to [-0.969, 1.0]
- Simple, invertible
- Inverse: `x_pnas = x_model * PNAS_ABS_MAX`

For display, the pipeline internally converts [-1,1] -> [0,1] via `(x/2 + 0.5).clamp(0,1)`.

---

## Script 1: `generate_samples.py`

**Generate images from any DDPM pipeline (pretrained or fine-tuned).**

Structure:
- `create_parser()` — returns argparse parser (so Script 3 can reuse it)
- `load_pnas_comparison(n, seed)` — load random PNAS images, crop to 64x64, convert to [0,1] for display
- `generate_images(pipeline, num_images, num_steps, seed, device)` — generate in sub-batches of 16
- `make_grid(images, real_images, save_path)` — matplotlib grid, generated on top, real PNAS on bottom
- `main(args)` — parse, load, generate, plot, save

Args: `--model-path` (default: `ddpm-imagenet-grayscale`), `--num-images` (16), `--num-steps` (1000), `--seed` (42), `--output-dir` (`generated_samples`), `--save-individual`, `--no-comparison`, `--device` (cuda)

---

## Script 2: `finetune.py`

**Fine-tune the pretrained DDPM on PNAS images.**

Structure:
- `PNASDataset(Dataset)` class:
  - Loads all images from npz (train + val + test = 3190)
  - `__getitem__`: transpose to (1,108,108) -> random crop 64x64 -> random H-flip -> random V-flip -> divide by PNAS_ABS_MAX -> tensor in ~[-1,1]
- `train_one_epoch(unet, scheduler, dataloader, optimizer, device)` — standard diffusion loop: sample t, add noise, predict noise, MSE loss
- `generate_checkpoint_samples(pipeline, n, seed, epoch, save_dir)` — quick sample grid at checkpoints (50 steps for speed)
- `main()` — load pipeline, extract unet/scheduler, create dataset/dataloader, train loop, save pipeline at end

Args: `--model-path` (`ddpm-imagenet-grayscale`), `--num-epochs` (50), `--lr` (1e-5), `--batch-size` (16), `--save-dir` (`ddpm-pnas-finetuned`), `--seed` (42), `--checkpoint-interval` (10), `--num-sample-images` (8), `--device` (cuda), `--gradient-accumulation` (1)

Key details:
- Optimizer: Adam, lr=1e-5 (low to avoid catastrophic forgetting)
- Checkpoints saved as full diffusers pipelines in `{save_dir}/checkpoint_epoch_{N}/`
- Final model saved at `{save_dir}/` (loadable by Script 3)
- Loss curve saved as `{save_dir}/loss_curve.png`
- Print loss and timing per epoch

---

## Script 3: `generate_samples_finetuned.py`

**Thin wrapper around Script 1 with fine-tuned defaults.**

~20 lines. Imports `create_parser` and `main` from `generate_samples.py`, overrides defaults:
- `--model-path` -> `ddpm-pnas-finetuned`
- `--output-dir` -> `generated_samples_finetuned`

---

## File Layout

```
gpytorch_porting/
    ddpm-imagenet-grayscale/           # pretrained (existing, untouched)
    generate_samples.py                # Script 1
    finetune.py                        # Script 2
    generate_samples_finetuned.py      # Script 3
    ddpm-pnas-finetuned/               # created by finetune.py (output)
    generated_samples/                 # created by generate_samples.py (output)
    generated_samples_finetuned/       # created by Script 3 (output)
```

---

## Shared Constants (defined in generate_samples.py, imported by others)

```python
PNAS_ABS_MAX = 2.478047
PNAS_DATA_PATH = '<absolute path to PNAS_paper_sorted_data.npz>'
DEFAULT_PRETRAINED_PATH = 'ddpm-imagenet-grayscale'
DEFAULT_FINETUNED_PATH = 'ddpm-pnas-finetuned'
```

---

## Implementation Order

1. `generate_samples.py` — test with `python generate_samples.py --num-images 4 --num-steps 50`
2. `finetune.py` — test with `python finetune.py --num-epochs 2 --checkpoint-interval 1`
3. `generate_samples_finetuned.py` — test with `python generate_samples_finetuned.py --num-images 4`

---

## Verification

1. **Script 1**: generates a grid PNG with 16 images + PNAS comparison row. Images should look like grayscale natural scenes.
2. **Script 2**: loss decreases over epochs. Checkpoint directories are valid diffusers pipelines (loadable by `DDPMPipeline.from_pretrained()`). Sample grids at checkpoints show visual progression.
3. **Script 3**: generates from fine-tuned model, output looks more PNAS-like than pretrained.
