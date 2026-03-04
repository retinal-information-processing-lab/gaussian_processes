#!/usr/bin/env python
"""Create a center-cropped version of an image dataset.

Center-crops all images from a source NPZ file and saves a new NPZ
with the same key structure. Response arrays are copied unchanged.
Pixel values are NOT re-normalized after cropping (see README.md).

Usage:
    python create_cropped_dataset.py --crop-size 64
    python create_cropped_dataset.py --crop-size 80 --source path/to/other.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    # Read default data path from default_params.json (single source of truth)
    script_dir = Path(__file__).parent
    gpytorch_dir = script_dir.parent
    defaults_path = gpytorch_dir / 'default_params.json'
    with open(defaults_path, 'r') as f:
        defaults = json.load(f)

    default_source = defaults['data']['path']
    # Resolve relative to gpytorch_porting/
    if not Path(default_source).is_absolute():
        default_source = str(gpytorch_dir / default_source)

    parser = argparse.ArgumentParser(
        description='Create a center-cropped dataset from an existing NPZ file.'
    )
    parser.add_argument('--crop-size', type=int, required=True,
                        help='Target side length (e.g., 64 for 64x64 crop)')
    parser.add_argument('--source', type=str, default=default_source,
                        help=f'Source NPZ file (default: from default_params.json = {default_source})')
    parser.add_argument('--output', type=str, default=None,
                        help='Output NPZ path (default: datasets/PNAS_{crop_size}x{crop_size}_center_crop_no_renorm.npz)')
    args = parser.parse_args()

    crop_size = args.crop_size
    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = gpytorch_dir / source_path

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = script_dir / f'PNAS_{crop_size}x{crop_size}_center_crop_no_renorm.npz'

    # Load source
    print(f"Loading source: {source_path}")
    data = np.load(source_path)

    # Detect original dimensions
    original_side = data['images_train'].shape[1]
    assert data['images_train'].shape[1] == data['images_train'].shape[2], \
        f"Expected square images, got {data['images_train'].shape[1]}x{data['images_train'].shape[2]}"

    print(f"Original image size: {original_side}x{original_side}")
    print(f"Target crop size: {crop_size}x{crop_size}")

    # Validate
    if crop_size >= original_side:
        raise ValueError(
            f"Crop size ({crop_size}) must be smaller than original ({original_side})"
        )
    margin = original_side - crop_size
    if margin % 2 != 0:
        raise ValueError(
            f"Cannot center-crop {original_side} -> {crop_size}: "
            f"difference ({margin}) is odd. Use an even crop size or even original."
        )

    offset = margin // 2
    print(f"Crop offset: {offset} pixels from each edge")

    # Center-crop all image arrays
    image_keys = ['images_train', 'images_val', 'images_test']
    response_keys = ['responses_train', 'responses_val', 'responses_test']

    result = {}

    for key in image_keys:
        original = data[key]
        cropped = original[:, offset:offset + crop_size, offset:offset + crop_size, :]
        result[key] = cropped
        print(f"  {key}: {original.shape} -> {cropped.shape}")

    for key in response_keys:
        result[key] = data[key]
        print(f"  {key}: {data[key].shape} (unchanged)")

    # Pixel range check
    for key in image_keys:
        orig_min, orig_max = data[key].min(), data[key].max()
        crop_min, crop_max = result[key].min(), result[key].max()
        print(f"\n  {key} pixel range:")
        print(f"    Original: [{orig_min:.4f}, {orig_max:.4f}]")
        print(f"    Cropped:  [{crop_min:.4f}, {crop_max:.4f}]")

    # Center pixel spot-check: center pixel of cropped should match center of original
    center_orig = original_side // 2
    center_crop = crop_size // 2
    for key in image_keys:
        orig_val = data[key][0, center_orig, center_orig, 0]
        crop_val = result[key][0, center_crop, center_crop, 0]
        match = "OK" if orig_val == crop_val else "MISMATCH"
        print(f"  {key} center pixel check: original[{center_orig},{center_orig}]={orig_val:.6f} "
              f"== cropped[{center_crop},{center_crop}]={crop_val:.6f} [{match}]")

    # Save
    print(f"\nSaving to: {output_path}")
    np.savez(output_path, **result)
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("Done. NOT re-normalized (original pixel values preserved).")


if __name__ == '__main__':
    main()
