# Diffusion Model Investigation Log

## 2026-03-03: Initial planning

**Decision**: 64x64 resolution (not 108x108)
- Power-of-2 for clean U-Net downsampling (64->32->16->8)
- Random crop augmentation from 108x108 originals (effectively unlimited data)
- RF region (~28px radius) fits comfortably in 64x64
- Faster training and inference

**Decision**: D4 augmentation (8 variants per crop)
- 4 rotations (0, 90, 180, 270) x 2 flips = 8 transformations
- Natural scene patches are symmetric under all D4 elements
- Combined with random crops: massive data augmentation

**Decision**: Pure PyTorch, no external diffusion libraries
- No pretrained models for our data distribution (108x108 grayscale PNAS patches)
- diffusers/HuggingFace is overkill for this scale
- Total code: ~300-400 lines across 3 files

**Decision**: Cosine noise schedule, T=1000, DDPM sampling
- Cosine preserves more structure at low noise (where natural textures live)
- DDPM sampling is simplest and fast enough at 64x64

**Architecture**: Tiny U-Net (~0.5-1M params)
- 3 down-blocks [1->32->64->128 channels], spatial: 64->32->16->8
- Middle block at 8x8
- 3 up-blocks with skip connections [128->64->32->1]
- Sinusoidal time embedding

**Deferred**:
- GP utility guidance (separate investigation after unconditional generation works)
- DDIM fast sampling (DDPM is fast enough)
- Classifier-free guidance, EMA, attention layers
