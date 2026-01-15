"""
Visualization utilities for GP model analysis and optimization comparison.

This module contains plotting functions for:
- Optimization constraint comparisons
- Single image sampling plots
"""

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch


def update_optimization_comparison(result_dict, imgs_train, n_images, save_dir, model, window=10):
    """
    Update and save optimization constraint comparison figure.

    Completely self-contained - handles all tracking internally via pickle file.

    Args:
        result_dict: Dict mapping method names to batch_utility_w_grad() results
                     e.g., {'RMS': result1, 'L2': result2, 'L4': result3}
        imgs_train: Original image tensor
        n_images: Current iteration number
        save_dir: Path to save outputs
        model: GP model (used to extract mask for correlation computation)
        window: Window size for running average (default 10)
    """
    from gaussian_processes.Spatial_GP_repo import utils as GP_utils

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pickle_path = save_dir / 'comparison_tracking_data.pkl'

    # Load existing tracking data or initialize
    if pickle_path.exists():
        with open(pickle_path, 'rb') as f:
            tracking_data = pickle.load(f)
    else:
        tracking_data = {}

    # Get method names from result_dict keys (flexible!)
    methods = list(result_dict.keys())

    # Initialize tracking for new methods
    for method in methods:
        if method not in tracking_data:
            tracking_data[method] = {
                'n_images': [],
                'U_initial': [],
                'U_final': [],
                'U_improvement': [],
                'corr_init_opt': [],
                'pct_increase': []
            }

    # Extract mask from model for correlation computation (only on optimized region)
    _, mask, _, _, _ = GP_utils.get_final_K_vals(model)

    # Extract metrics and compute correlations for each method
    for method, result in result_dict.items():
        tracking_data[method]['n_images'].append(n_images)
        tracking_data[method]['U_initial'].append(result['U_initial'])
        tracking_data[method]['U_final'].append(result['U_final'])
        tracking_data[method]['U_improvement'].append(
            result['U_final'] - result['U_initial']
        )

        # Compute percentage increase in utility relative to ORIGINAL IMAGE
        # NOTE: Uses first method's U_initial (e.g., 'RMS') as baseline for ALL methods
        # This ensures fair comparison - all methods measured against same original image utility
        first_method = list(result_dict.keys())[0]
        U_original = result_dict[first_method]['U_initial']
        pct_increase = 100 * (result['U_final'] - U_original) / abs(U_original)
        tracking_data[method]['pct_increase'].append(pct_increase)

        # Compute pixel correlation between initial and optimized image (MASKED REGION ONLY)
        initial_img = imgs_train[result['img_idx'].item()]
        optimized_img = result['optimized_img']

        # Use only masked pixels for correlation (the region that was actually optimized)
        initial_masked = initial_img[mask]
        optimized_masked = optimized_img[mask]

        corr = torch.corrcoef(torch.stack([
            initial_masked.flatten(),
            optimized_masked.flatten()
        ]))[0, 1].item()

        tracking_data[method]['corr_init_opt'].append(corr)

    # Create figure (one column per method, 3 rows)
    n_methods = len(methods)
    fig, axes = plt.subplots(3, n_methods, figsize=(6*n_methods, 12))
    if n_methods == 1:
        axes = axes[:, np.newaxis]  # Ensure 2D array
    fig.suptitle('Optimization Constraint Comparison', fontsize=16)

    # Plot each method's data
    for col, method in enumerate(methods):
        data = tracking_data[method]
        n_imgs = np.array(data['n_images'])

        # Row 0: Initial vs Optimized Utility with dual y-axis
        ax1 = axes[0, col]
        ax2 = ax1.twinx()

        line1 = ax1.plot(n_imgs, data['U_initial'], 'b-o', label='Initial', markersize=3)
        line2 = ax2.plot(n_imgs, data['U_final'], 'r-s', label='Optimized', markersize=3)

        ax1.set_ylabel('Initial Utility', fontsize=10, color='b')
        ax2.set_ylabel('Optimized Utility', fontsize=10, color='r')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')

        # Calculate and display average optimized utility (excluding outliers via IQR method)
        U_final_array = np.array(data['U_final'])
        Q1 = np.percentile(U_final_array, 25)
        Q3 = np.percentile(U_final_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_mask = (U_final_array >= lower_bound) & (U_final_array <= upper_bound)
        avg_U_final = np.mean(U_final_array[iqr_mask])
        n_outliers = (~iqr_mask).sum()

        title_suffix = f' (excl. {n_outliers} outliers)' if n_outliers > 0 else ''
        ax1.set_title(f'{method} - Utility (Avg Opt: {avg_U_final:.2f}{title_suffix})', fontsize=11, fontweight='bold')

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, fontsize=9, loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Row 1: Initial vs Optimized Correlation
        axes[1, col].plot(n_imgs, data['corr_init_opt'], 'b-o', markersize=3)
        axes[1, col].set_ylabel('Correlation', fontsize=10)
        axes[1, col].set_xlabel('Number of Images', fontsize=10)
        axes[1, col].set_title(f'{method} - Init vs Opt Correlation', fontsize=11, fontweight='bold')
        axes[1, col].set_ylim([0, 1])
        axes[1, col].grid(True, alpha=0.3)

        # Row 2: Running Average of Percentage Increase
        ax = axes[2, col]

        pct_data = np.array(data['pct_increase'])
        n_points = len(pct_data)

        # Compute running average with outlier exclusion using IQR method
        running_avgs = []

        for i in range(1, n_points + 1):
            subset = pct_data[:i]

            # IQR outlier detection
            if len(subset) >= 4:
                Q1 = np.percentile(subset, 25)
                Q3 = np.percentile(subset, 75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                iqr_mask_subset = (subset >= lower) & (subset <= upper)

                # Running average excluding outliers
                if iqr_mask_subset.any():
                    running_avg = np.mean(subset[iqr_mask_subset])
                else:
                    running_avg = np.mean(subset)
            else:
                running_avg = np.mean(subset)

            running_avgs.append(running_avg)

        # Identify outliers in full dataset for marking
        if n_points >= 4:
            Q1 = np.percentile(pct_data, 25)
            Q3 = np.percentile(pct_data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (pct_data < lower_bound) | (pct_data > upper_bound)
        else:
            outlier_mask = np.zeros(n_points, dtype=bool)

        # Plot individual data points
        ax.scatter(n_imgs[~outlier_mask], pct_data[~outlier_mask],
                  alpha=0.5, s=30, color='gray', label='Data', zorder=2)

        # Mark outliers in red
        if outlier_mask.any():
            ax.scatter(n_imgs[outlier_mask], pct_data[outlier_mask],
                      color='red', marker='x', s=50, label='Outliers', zorder=3)

        # Plot running average line
        ax.plot(n_imgs, running_avgs, linewidth=2.5, color='blue',
                label='Running Avg', zorder=4)

        # Final mean (excluding outliers)
        final_mean = np.mean(pct_data[~outlier_mask]) if (~outlier_mask).any() else np.mean(pct_data)
        ax.axhline(final_mean, linestyle='--', color='blue', alpha=0.7,
                  linewidth=1.5, label=f'Final: {final_mean:.1f}%', zorder=1)

        # Add horizontal line at 0% for reference
        ax.axhline(0, linestyle=':', color='black', alpha=0.3, linewidth=1)

        # Formatting
        n_outliers_total = outlier_mask.sum()
        title_suffix = f' (excl. {n_outliers_total} outliers)' if n_outliers_total > 0 else ''
        ax.set_title(f'{method} - Running Avg: {final_mean:.1f}%{title_suffix}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Number of Images', fontsize=10)
        ax.set_ylabel('% Utility Increase', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_dir / 'optimization_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save updated tracking data
    with open(pickle_path, 'wb') as f:
        pickle.dump(tracking_data, f)


def single_image_sampling_plot(result, imgs_train, save_path=None, show=False, shared_scale=True,
                               dataset_vmin=None, dataset_vmax=None, **kwargs):
    """
    Plot 3-panel comparison for conditioned utility optimization (N=1 case).

    Left: Initial image
    Middle: Optimized image
    Right: Sampled image

    Args:
        result: dict from optimize_with_conditioned_utility() with N=1
        imgs_train: full image dataset tensor
        save_path: Optional path to save figure
        show: Whether to display the figure
        shared_scale: If True, use same vmin/vmax for all images. If False, each image uses its own scale.
        dataset_vmin: If provided, use this as the minimum value for the colorscale (overrides shared_scale).
        dataset_vmax: If provided, use this as the maximum value for the colorscale (overrides shared_scale).
    """
    if result.get('sampled_img') is None:
        raise ValueError("result['sampled_img'] is None. This function requires N=1 in optimize_with_conditioned_utility().")

    # Extract images
    if kwargs['initial_img'] is None:
        initial_img = imgs_train[result['img_idx'].item()].cpu().numpy().reshape(108, 108)
    else:
        initial_img = kwargs['initial_img'].cpu().numpy().reshape(108, 108)
    optimized_img = result['optimized_img'].cpu().numpy().reshape(108, 108)
    sampled_img = result['sampled_img'][0].cpu().numpy().reshape(108, 108)

    all_imgs = [initial_img, optimized_img, sampled_img]

    if dataset_vmin is not None and dataset_vmax is not None:
        # Use dataset bounds for all images
        vmins = [dataset_vmin, dataset_vmin, dataset_vmin]
        vmaxs = [dataset_vmax, dataset_vmax, dataset_vmax]
    elif shared_scale:
        # Compute shared colorscale
        vmin = min(img.min() for img in all_imgs)
        vmax = max(img.max() for img in all_imgs)
        vmins = [vmin, vmin, vmin]
        vmaxs = [vmax, vmax, vmax]
    else:
        # Each image uses its own scale
        vmins = [img.min() for img in all_imgs]
        vmaxs = [img.max() for img in all_imgs]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    titles = ['Initial Image', 'Optimized Image', 'Sampled Image (N=1)']
    
    for i, (img, ax, title, vmin, vmax) in enumerate(zip(all_imgs, axes, titles, vmins, vmaxs)):
        im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Add actual min/max values to colorbar
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f'{vmin:.3f}', f'{vmax:.3f}'])
        
        # Add utility info below optimized image
        if i == 1:
            U_init = result['U_initial']
            U_final = result['U_final']
            pct_change = 100 * (U_final - U_init) / abs(U_init)
            ax.text(0.5, -0.1, f"U: {U_init:.3f} → {U_final:.3f} ({pct_change:+.1f}%)",
                    transform=ax.transAxes, fontsize=10,
                    ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved conditioned utility comparison to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)