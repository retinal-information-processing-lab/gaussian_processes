
import torch
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from gaussian_processes.Spatial_GP_repo import utils as GP_utils

import warnings
warnings.filterwarnings("ignore", "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor)")


# ========= Build spikes dicts from image history ==========
def load_training_data_optim(
        session_data_path,
        session_name,
        attempted_reps, 
        active_kind,
        chosen_electrode,
        exclude_reps_train=None,
        ):
    
    """
    Equivalent to load_training_data but only uploads data for active_kind='opt'
    """

    if active_kind != 'opt':
        raise ValueError("load_training_data_optim only supports active_kind='opt'.")

    if exclude_reps_train is None:
        exclude_reps_train = []

    # numbers for reps that went to the end successfully
    successful_reps = []     

    train_idx_spikes_dict = {}    # { img_idx: tensor([spikes from different reps]) }
    successful_reps_sequences = {}
    final_active_models = {}

    for repetition_number in attempted_reps:

        print(f"\n=== Repetition {repetition_number:<2} ===", end=' ')
        session_repetition_path = session_data_path / f"repetition_{repetition_number}"
        model_path              = session_repetition_path / 'models'
        
        final_model_name = f'final_active_model_{active_kind}_ch_{chosen_electrode}_ses_{session_name}_rep{repetition_number}.pkl'
        final_model_path = model_path / f'ch_{chosen_electrode}' / final_model_name

        history_name = f'image_history_ses_{session_name}_rep{repetition_number}.pkl'
        history_path = os.path.join(session_repetition_path, history_name)
            
        # ---- Load history ----
        if not os.path.exists(history_path):
            print(f" Skip, History not found")
            raise FileNotFoundError(f"History file not found for repetition {repetition_number} of electrode {chosen_electrode}.", end=' ')
        with open(history_path, "rb") as f:
            image_history = pickle.load(f)

        print(f'History length: {len(image_history):<4}', end=' ')
        if len(image_history) < 10:
            print(f" Skip History less than 10.", end=' ')
            continue

        # ---- Load final model ----
        try:
            with open(final_model_path, "rb") as f:
                final_model = pickle.load(f)
                final_active_models[repetition_number] = final_model
                modelok = True
        except Exception as e:
            modelok = False
            # print(f" Skip, Final model not found")
            # print(f" No final model " , end=' ')
            # continue
        print(f"Model {'found' if modelok else 'not found':<13}", end=' ')
        
        active_spikes_dict_rep, active_idx_rep = \
            build_spikes_dict_from_history(image_history, f'active_{active_kind}_{chosen_electrode}', ch_ids_local=[chosen_electrode], sorted_data=False)

        active_spikes_rep = active_spikes_dict_rep[chosen_electrode]

        print(f'Active Opt {active_spikes_rep.shape[0]:<4}', end=' ')

        if active_idx_rep.shape[0] != torch.unique(active_idx_rep).shape[0]:
            print(f' [Warning] duplicate indices in active opt', end=' ')
            continue

        # Pad active and random spikes to 300 with NaNs
        if 300 - active_spikes_rep.shape[0] > 0:
            print(f" [Warning] active spikes less than 300, skip", end =' ')
            continue
            # print(f' Padding active of rep {repetition_number} by {300 - active_spikes_rep.shape[0]} ', end =' ')
            # active_spikes_rep =  F.pad(active_spikes_rep, (0, 300 - active_spikes_rep.shape[0]), value=float('nan'))
            
        elif not modelok:
            print(f" [Warning] no final model, skip", end =' ')
            continue
        else:
            successful_reps.append(repetition_number)

        # assert no Nans in active and random spikes
        if torch.isnan(active_spikes_rep).any():
            raise ValueError(f"NaN values found in spikes for repetition {repetition_number}.")


        # All successful reps sequences, even if excluded from training dataset
        successful_reps_sequences[repetition_number] = {
            'active_indices': active_idx_rep,
            'active_spikes' : active_spikes_rep,
        }

        if not repetition_number in exclude_reps_train:
            # Fill train_idx_spikes_dict with active and random spikes
            for act_img_idx, act_spk in zip(active_idx_rep, active_spikes_rep):
                if act_img_idx.item() not in train_idx_spikes_dict:
                    train_idx_spikes_dict[act_img_idx.item()] = torch.empty(0, device=GP_utils.DEVICE, dtype=GP_utils.TORCH_DTYPE)
                train_idx_spikes_dict[act_img_idx.item()] = torch.cat( (train_idx_spikes_dict[act_img_idx.item()], act_spk[None]) )
        else:
            print(f" Excluded from training set", end=' ')  
            
    print(f"\nTotal successful repetitions that got to the end: {len(successful_reps)} -> {successful_reps}")

    # sort train_idx_spikes_dict by keys
    train_idx_spikes_dict = dict(sorted(train_idx_spikes_dict.items()))

    return train_idx_spikes_dict, successful_reps, successful_reps_sequences, final_active_models

def load_training_data(
        session_data_path,
        session_name,
        attempted_reps, 
        active_kind,
        chosen_electrode,
        exclude_reps_train=None,
        ):
    
    """
    Load training data from all successful repetitions, it also determines ifa repetitoins has been successful, 
    by checking number of img indexes"""

    if active_kind != 'orig':
        # TODO: remove references to 'opt' active kind. Optimized images reponses will never
        # contribute to the general training data on which we train the full models
        raise ValueError("load_training_data only supports active_kind='orig'.")

    if exclude_reps_train is None:
        exclude_reps_train = []

    # numbers for reps that went to the end successfully
    successful_reps = []     

    train_idx_spikes_dict = {}    # { img_idx: tensor([spikes from different reps]) }
    successful_reps_sequences = {}
    final_active_models = {}

    for repetition_number in attempted_reps:

        print(f"\n=== Repetition {repetition_number:<2} ===", end=' ')
        session_repetition_path = session_data_path / f"repetition_{repetition_number}"
        model_path              = session_repetition_path / 'models'
        
        final_model_name = f'final_active_model_{active_kind}_ch_{chosen_electrode}_ses_{session_name}_rep{repetition_number}.pkl'
        final_model_path = model_path / f'ch_{chosen_electrode}' / final_model_name

        history_name = f'image_history_ses_{session_name}_rep{repetition_number}.pkl'
        history_path = os.path.join(session_repetition_path, history_name)
            
        # ---- Load history ----
        if not os.path.exists(history_path):
            print(f" Skip, History not found")
            raise FileNotFoundError(f"History file not found for repetition {repetition_number} of electrode {chosen_electrode}.", end=' ')
        with open(history_path, "rb") as f:
            image_history = pickle.load(f)

        print(f'History length: {len(image_history):<4}', end=' ')
        if len(image_history) < 10:
            print(f" Skip History less than 10.", end=' ')
            continue

        # ---- Load final model ----
        try:
            with open(final_model_path, "rb") as f:
                final_model = pickle.load(f)
                final_active_models[repetition_number] = final_model
                modelok = True
        except Exception as e:
            modelok = False
            # print(f" Skip, Final model not found")
            # print(f" No final model " , end=' ')
            # continue
        print(f"Model {'found' if modelok else 'not found':<13}", end=' ')

        random_spikes_dict_rep, random_idx_rep = \
            build_spikes_dict_from_history(image_history, 'random', ch_ids_local=[chosen_electrode], sorted_data=False)
        
        active_spikes_dict_rep, active_idx_rep = \
            build_spikes_dict_from_history(image_history, f'active_{active_kind}_{chosen_electrode}', ch_ids_local=[chosen_electrode], sorted_data=False)

        random_spikes_rep = random_spikes_dict_rep[chosen_electrode]
        active_spikes_rep = active_spikes_dict_rep[chosen_electrode]
        # test_spikes_rep   = test_spikes_dict_rep[chosen_electrode]


        print(f'Active {active_spikes_rep.shape[0]:<4}  Random {random_spikes_rep.shape[0]:<4}', end=' ')

        if active_idx_rep.shape[0] != torch.unique(active_idx_rep).shape[0] or \
            random_idx_rep.shape[0] != torch.unique(random_idx_rep).shape[0]:
            print(f' [Warning] duplicate indices in active or random', end=' ')
            continue

        # Pad active and random spikes to 300 with NaNs
        if 300 - active_spikes_rep.shape[0] > 0:
            print(f" [Warning] active spikes less than 300, skip", end =' ')
            continue
            # print(f' Padding active of rep {repetition_number} by {300 - active_spikes_rep.shape[0]} ', end =' ')
            # active_spikes_rep =  F.pad(active_spikes_rep, (0, 300 - active_spikes_rep.shape[0]), value=float('nan'))
            
        elif 300 - random_spikes_rep.shape[0] > 0:
            print(f" [Warning] random spikes less than 300, skip", end =' ')
            continue
            # print(f' Padding random of rep {repetition_number} ', end =' ')
            # random_spikes_rep =  F.pad(random_spikes_rep, (0, 300 - random_spikes_rep.shape[0]), value=float('nan'))
        elif not modelok:
            print(f" [Warning] no final model, skip", end =' ')
            continue
        elif repetition_number in exclude_reps_train:
            print(f" Excluded from training set", end=' ')
            continue
        else:
            successful_reps.append(repetition_number)

        # assert no Nans in active and random spikes
        if torch.isnan(active_spikes_rep).any() or torch.isnan(random_spikes_rep).any():
            raise ValueError(f"NaN values found in spikes for repetition {repetition_number}.")


        # All successful reps sequences, even if excluded from training dataset
        successful_reps_sequences[repetition_number] = {
            'active_indices': active_idx_rep,
            'active_spikes' : active_spikes_rep,
            'random_indices': random_idx_rep,
            'random_spikes' : random_spikes_rep,
        }

        if not repetition_number in exclude_reps_train:
            # Fill train_idx_spikes_dict with active and random spikes
            for act_img_idx, act_spk in zip(active_idx_rep, active_spikes_rep):
                if act_img_idx.item() not in train_idx_spikes_dict:
                    train_idx_spikes_dict[act_img_idx.item()] = torch.empty(0, device=GP_utils.DEVICE, dtype=GP_utils.TORCH_DTYPE)
                train_idx_spikes_dict[act_img_idx.item()] = torch.cat( (train_idx_spikes_dict[act_img_idx.item()], act_spk[None]) )

            for rand_img_idx, rand_spk in zip(random_idx_rep, random_spikes_rep):
                if rand_img_idx.item() not in train_idx_spikes_dict:
                    train_idx_spikes_dict[rand_img_idx.item()] = torch.empty(0, device=GP_utils.DEVICE, dtype=GP_utils.TORCH_DTYPE)
                train_idx_spikes_dict[rand_img_idx.item()] = torch.cat( (train_idx_spikes_dict[rand_img_idx.item()], rand_spk[None]) )
        else:
            print(f" Excluded from training set", end=' ')  
            
    print(f"\nTotal successful repetitions that got to the end: {len(successful_reps)} -> {successful_reps}")

    # sort train_idx_spikes_dict by keys
    train_idx_spikes_dict = dict(sorted(train_idx_spikes_dict.items()))

    return train_idx_spikes_dict, successful_reps, successful_reps_sequences, final_active_models

def build_spikes_dict_from_history(image_history, typ, ch_ids_local=None, sorted_data=False, device=GP_utils.DEVICE, dtype=GP_utils.TORCH_DTYPE):
    """
    Build dictionary of spike counts from image history for multiple electrodes.
    
    Returns:
        out: Dictionary { ch:int -> tensor([counts in time], dtype=dtype, device=device) }
        imgs_idx_tensor: Tensor of image indices in temporal order
    
    Args:
        image_history: List of image records from experiment
        typ: String, image kind to filter by ('random', 'active', 'test', etc.)
        ch_ids_local: List of electrode IDs to extract data for
        sorted_data: If True, use sorted spike counts instead of raw counts
        device: Torch device to use
        dtype: Torch data type to use
    """
    chs = list(ch_ids_local)
    per_ch = {ch: [] for ch in chs}
    imgs_idx = []
    
    # Keep temporal order identical to acquisition/appends
    sorted_hist = sorted(image_history, key=lambda r: r.get('start_frame', 0))

    for rec in sorted_hist:
        kind = rec.get('kind')
        if kind not in ('Initial', typ):
            continue
        if not rec.get('processed', False):
            continue

        # Get image index for this record
        img_idx = rec.get('image_number')
        has_data = False
        
        # Process each requested channel
        for ch in chs:
            if sorted_data:
                # Get spike counts from sorted data if available
                if 'sorted_spike_count' in rec and ch in rec['sorted_spike_count']:
                    per_ch[ch].append(int(rec['sorted_spike_count'][ch]))
                    has_data = True
                else:
                    # Skip this record if sorted data not available for this electrode
                    per_ch[ch].append(0)  # Add 0 to maintain alignment with other channels
                    print(f"Warning: Sorted data not available for channel {ch} in record with image_number {img_idx}")
            else:
                # Get spike counts from unsorted data
                if 'spike_counts' in rec and ch in rec.get('spike_counts', {}):
                    per_ch[ch].append(int(rec['spike_counts'][ch]))
                    has_data = True
                else:
                    # Skip this record if unsorted data not available for this electrode
                    per_ch[ch].append(0)  # Add 0 to maintain alignment with other channels
        
        # Only add image index if at least one channel had data
        if has_data:
            imgs_idx.append(img_idx)

    # Convert to tensors on the requested device/dtype
    out = {}
    for ch in chs:
        if len(per_ch[ch]) == len(imgs_idx):  # Ensure alignment
            out[ch] = torch.tensor(per_ch[ch], device=device, dtype=dtype)
    
    imgs_idx_tensor = torch.tensor(imgs_idx, device=device, dtype=torch.int64)

    return out, imgs_idx_tensor

def build_uniform_test_dict(reps_numbers, session_data_path, session_name, chosen_electrode,
                           sorted_data=False, device=GP_utils.DEVICE, dtype=GP_utils.TORCH_DTYPE):
    """
    Build a uniform test spikes dictionary where all test images have the same number of repetitions.
    The function pools repetitions across all experiment runs to maximize available data.
    
    Args:
        reps_numbers: List of repetition numbers to include
        session_data_path: Path to the session data directory
        session_name: Session name
        chosen_electrode: Electrode ID to extract data for
        ch_ids: List of all electrode IDs
        sorted_data: If True, use sorted spike counts instead of raw counts
        device: Torch device to use
        dtype: Torch data type to use
    
    Returns:
        test_idx_spikes_dict: Dictionary {img_idx: tensor([rep_spikes])} with uniform repetition counts
    """
    # Step 1: Collect all test images and their repetitions across all runs
    all_test_images = {}  # {img_idx: [list of all spike tensors from different repetitions]}
    all_image_indices = set()
    
    print(f"Collecting {'sorted' if sorted_data else 'unsorted'} test spikes from all repetitions...")
    for repetition_number in reps_numbers:
        # Load image history for this repetition
        history_path = session_data_path / f"repetition_{repetition_number}" / f'image_history_ses_{session_name}_rep{repetition_number}.pkl'
        if not os.path.exists(history_path):
            print(f"Rep {repetition_number}: History not found, skipping")
            continue
        
        try:
            with open(history_path, "rb") as f:
                image_history = pickle.load(f)
            
            # Extract test spikes for this repetition
            test_spikes_dict = {}
            test_idx_set = set()
            
            # Process each record in the history
            for rec in image_history:
                # Skip non-test images or unprocessed images
                if rec.get('kind') != 'test' or not rec.get('processed', False):
                    continue
                
                img_idx = rec.get('image_number')
                
                if sorted_data:
                    # Use sorted spike counts if available
                    if 'sorted_spike_count' in rec and chosen_electrode in rec['sorted_spike_count']:
                        spike_count = int(rec['sorted_spike_count'][chosen_electrode])
                    else:
                        continue  # Skip if sorted data not available for this electrode
                else:
                    # Use regular spike counts
                    if 'spike_counts' in rec and chosen_electrode in rec.get('spike_counts', {}):
                        spike_count = int(rec['spike_counts'][chosen_electrode])
                    else:
                        continue  # Skip if unsorted data not available
                
                # Initialize dictionary entry if needed
                if img_idx not in test_spikes_dict:
                    test_spikes_dict[img_idx] = []
                    test_idx_set.add(img_idx)
                
                # Add this spike count
                test_spikes_dict[img_idx].append(spike_count)
            
            # Convert lists to tensors
            for img_idx in test_spikes_dict:
                test_spikes_dict[img_idx] = torch.tensor(test_spikes_dict[img_idx], 
                                                       device=device, dtype=dtype)
            
            if not test_spikes_dict:
                print(f"Rep {repetition_number}: No {'sorted' if sorted_data else 'unsorted'} test data found")
                continue
                
            print(f"Rep {repetition_number}: Found test data for {len(test_spikes_dict)} images")
            
            # Track all image indices we've seen
            all_image_indices.update(test_spikes_dict.keys())
            
            # Store all repetitions by image index
            for img_idx, spikes in test_spikes_dict.items():
                if img_idx not in all_test_images:
                    all_test_images[img_idx] = []
                all_test_images[img_idx].append(spikes)
                
        except Exception as e:
            print(f"Rep {repetition_number}: Error processing - {str(e)}")
    
    # Step 2: Find images that have repetitions
    common_images = set()
    for img_idx, spike_tensors in all_test_images.items():
        if len(spike_tensors) > 0:  # Image has at least some repetitions
            common_images.add(img_idx)
    
    print(f"Found {len(common_images)} test images with at least one repetition")
    
    if not common_images:
        print("No common test images found across repetitions!")
        return {}, torch.empty(0, 0, device=device, dtype=dtype)
    
    # Step 3: Calculate how many repetitions we can have per image
    # First concatenate all repetitions for each image
    combined_test_spikes = {}
    for img_idx in common_images:
        combined_test_spikes[img_idx] = torch.cat(all_test_images[img_idx])
    
    # Find the minimum number of available repetitions
    min_reps = min(tensor.shape[0] for tensor in combined_test_spikes.values())
    
    print(f"Maximum uniform repetitions possible: {min_reps}")
    
    # Step 4: Build the final uniform dictionary
    uniform_test_dict = {}
    for img_idx, all_spikes in combined_test_spikes.items():
        uniform_test_dict[img_idx] = all_spikes[:min_reps]
    
    # Verify all images have the same number of repetitions
    for img_idx, spikes in uniform_test_dict.items():
        assert spikes.shape[0] == min_reps, f"Image {img_idx} has {spikes.shape[0]} repetitions, expected {min_reps}"
    
    print(f"Successfully built uniform test dictionary with {len(uniform_test_dict)} images, each with exactly {min_reps} repetitions")
    
    # Step 5: Create spike counts tensor (imgs x reps)
    spike_counts_test_tensor = torch.stack([
        uniform_test_dict[idx] for idx in sorted(uniform_test_dict.keys())
    ], dim=0).to(GP_utils.DEVICE, dtype=GP_utils.TORCH_DTYPE)
    
    # Transpose to get (reps × images) format as expected by testing functions
    spike_counts_test_tensor = spike_counts_test_tensor.T

    return uniform_test_dict, spike_counts_test_tensor

# ========= Reliabilities ============== 

def bootstrap_split_half_reliability_over_images(test_idx_spikes_dict,
                                                 n_bootstrap=5000,
                                                 split_mode='even-odd',   # 'even-odd' or 'random'
                                                 apply_spearman_brown=True,
                                                 random_state=None):
    """
    Simple, botstrap across images
    Computes split-half reliability when repetitions per image are non-uniform.
    Bootstraps along images:
      1) For each boot, resample images with replacement.
      2) For each selected image, split its repetitions into two halves
         (even-odd or random), take means, and build two vectors across images.
      3) Correlate the two vectors, optionally apply Spearman–Brown.
    Returns: mean_r, std_r, r_samples (array of length n_bootstrap)
    """
    rng = np.random.default_rng(random_state)
    items = list(test_idx_spikes_dict.items())
    n_images_total = len(items)
    r_samples = np.full(n_bootstrap, np.nan, dtype=np.float64)

    def split_means(spks_1d):
        # Remove NaNs and require at least 2 reps
        x = spks_1d.detach().cpu().numpy()
        x = x[np.isfinite(x)]
        if x.size < 2:
            return np.nan, np.nan
        if split_mode == 'even-odd':
            a = x[0::2]
            b = x[1::2]
        else:
            perm = rng.permutation(x.size)
            k = x.size // 2
            a = x[perm[:k]]
            b = x[perm[k:]]
        if a.size == 0 or b.size == 0:
            return np.nan, np.nan
        return float(np.mean(a)), float(np.mean(b))

    for b in range(n_bootstrap):
        # Resample images with replacement
        idxs = rng.integers(0, n_images_total, size=n_images_total)
        A, B = [], []
        for i in idxs:
            _, spks = items[i]
            ma, mb = split_means(spks)
            if np.isfinite(ma) and np.isfinite(mb):
                A.append(ma); B.append(mb)
        if len(A) >= 3:
            r = np.corrcoef(A, B)[0, 1]
            if apply_spearman_brown:
                r = (2 * r) / (1 + r) if r < 1 else 1.0
            r_samples[b] = np.clip(r, -1, 1)

    mean_r = np.nanmean(r_samples)
    std_r  = np.nanstd(r_samples)
    return mean_r, std_r, r_samples

def get_reliability(spike_counts_test_tensor):
    '''
    Computes the reliability of the test spikes by averaging even and odd repetitions
    and calculating the correlation between them.
    
    Args:
        spike_counts_test_tensor: Tensor of shape (n_reps, n_imgs) with test spikes.
        
    Returns:
        reliability: Correlation coefficient between even and odd repetitions.
    '''
    # Even and odd repetitions of the same image, mean response. First index is repetitions
    reven = torch.mean(spike_counts_test_tensor[0::2,:], axis=0)
    rodd  = torch.mean(spike_counts_test_tensor[1::2,:], axis=0)
    reliability = torch.abs(torch.corrcoef( torch.stack((reven,rodd))))[0,1]
    return reliability

def calculate_fano_factors(test_idx_spikes_dict):
    """
    Calculate Fano factor (variance/mean) for each test image and the average.
    
    Args:
        test_idx_spikes_dict: Dictionary {img_idx: tensor([rep_spikes])} containing
                             spike counts for each image across repetitions
        
    Returns:
        fano_factors_dict: Dictionary {img_idx: fano_factor} for each image
        mean_fano: Average Fano factor across all images
        std_fano: Standard deviation of Fano factors
    """
    fano_factors_dict = {}
    fano_factors_list = []
    
    for img_idx, spike_counts in test_idx_spikes_dict.items():
        # Convert to CPU numpy for calculation if it's a tensor
        if isinstance(spike_counts, torch.Tensor):
            spike_counts = spike_counts.detach().cpu().numpy()
        
        # Remove any NaN values that might be present from padding
        spike_counts = spike_counts[~np.isnan(spike_counts)]
        
        # Calculate mean and variance
        if len(spike_counts) > 1:  # Need at least 2 repetitions
            mean = np.mean(spike_counts)
            variance = np.var(spike_counts, ddof=1)  # Using sample variance (ddof=1)
            
            # Avoid division by zero
            if mean > 0:
                fano = variance / mean
                fano_factors_dict[img_idx] = fano
                fano_factors_list.append(fano)
    
    # Calculate average Fano factor and std
    fano_factors_array = np.array(fano_factors_list)
    mean_fano = np.mean(fano_factors_array) if len(fano_factors_array) > 0 else np.nan
    std_fano = np.std(fano_factors_array) if len(fano_factors_array) > 0 else np.nan
    
    return fano_factors_dict, mean_fano, std_fano
# ========= Plotting utilities ==========
def plot_mean_activity_across_repetitions(
    reps_sequences,
    session_name,
    active_label='Active Images',
    random_label='Random Images',
    figsize=(8, 6),
    active_color_idx=3,
    random_color_idx=0,
    label_offset_dy=0.5,
    label_offset_dx=1,
    save_path=None
):
    """
    Plot mean spike counts for active and random images across repetitions.
    
    Args:
        reps_sequences: Dictionary mapping repetition numbers to sequence data.
                       Each entry should have 'active_spikes' and 'random_spikes' keys.
        session_name: Name of the experimental session for the title.
        active_label: Label for active images line (default: 'Active Images').
        random_label: Label for random images line (default: 'Random Images').
        figsize: Figure size as (width, height) tuple (default: (8, 6)).
        active_color_idx: Color index from tab10 colormap for active line (default: 3).
        random_color_idx: Color index from tab10 colormap for random line (default: 0).
        label_offset_dy: Vertical offset for inline labels (default: 0.5).
        label_offset_dx: Horizontal offset for inline labels (default: 1).
        save_path: Optional path to save the figure. If None, figure is not saved.
    
    Returns:
        fig: The matplotlib figure object.
        ax: The matplotlib axes object.
        stats: Dictionary with summary statistics.
    """
    # Extract repetition numbers and compute means
    repetition_numbers = np.array(list(reps_sequences.keys()))
    
    means_active_spikes = np.array([
        torch.nanmean(reps_sequences[rep]['active_spikes']).cpu().numpy() 
        for rep in repetition_numbers
    ])
    
    means_random_spikes = np.array([
        torch.nanmean(reps_sequences[rep]['random_spikes']).cpu().numpy() 
        for rep in repetition_numbers
    ])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    colors = [plt.cm.tab10(i) for i in range(10)]
    
    # Plot lines
    ln1, = ax.plot(repetition_numbers, means_active_spikes, '-o',
                   color=colors[active_color_idx], linewidth=2, markersize=8)
    ln2, = ax.plot(repetition_numbers, means_random_spikes, '-o',
                   color=colors[random_color_idx], linewidth=2, markersize=8)
    
    # Add inline labels
    ax.text(repetition_numbers[0] - label_offset_dx * 1.1, 
            means_active_spikes[0] + label_offset_dy, 
            active_label,
            color=colors[active_color_idx], fontsize=12,
            va='center', ha='left',
            backgroundcolor='white', alpha=0.7)
    
    ax.text(repetition_numbers[0] - label_offset_dx * 1.1, 
            means_random_spikes[0] - label_offset_dy, 
            random_label,
            color=colors[random_color_idx], fontsize=12,
            va='center', ha='left',
            backgroundcolor='white', alpha=0.7)
    
    # Set axis limits and labels
    y_max = np.max([np.nanmax(means_active_spikes), np.nanmax(means_random_spikes)]) * 1.1
    ax.set_xlim(repetition_numbers[0] - 1.5, repetition_numbers[-1] + 0.5)
    ax.set_ylim(0, y_max)
    ax.set_xlabel('Repetition Number', fontsize=14)
    ax.set_ylabel('Mean spike count', fontsize=14)
    ax.set_title(f'Mean Activity Across Repetitions - Exp {session_name}', fontsize=16)
    ax.set_xticks(repetition_numbers)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Compute summary statistics
    stats = {
        'active_mean': np.nanmean(means_active_spikes),
        'active_std': np.nanstd(means_active_spikes),
        'random_mean': np.nanmean(means_random_spikes),
        'random_std': np.nanstd(means_random_spikes),
        'repetition_numbers': repetition_numbers,
        'means_active': means_active_spikes,
        'means_random': means_random_spikes
    }
    
    # Print summary
    print(f"{active_label} - Mean: {stats['active_mean']:.2f}, Std: {stats['active_std']:.2f}")
    print(f"{random_label} - Mean: {stats['random_mean']:.2f}, Std: {stats['random_std']:.2f}")
    
    return fig, ax, stats

def plot_image_presentation_histogram(
    train_idx_spikes_dict,
    figsize=(12, 6),
    hist_color='skyblue',
    hist_alpha=0.7,
    bin_step=1,
    xtick_step=10,
    stats_text_position=(0.7, 0.8),
    save_path=None
):
    """
    Plot histogram of image presentation counts from training data.
    
    Args:
        train_idx_spikes_dict: Dictionary mapping image indices to spike count lists.
                              Each value is a list/tensor of spike counts for that image.
        figsize: Figure size as (width, height) tuple (default: (12, 6)).
        hist_color: Color for histogram bars (default: 'skyblue').
        hist_alpha: Transparency of histogram bars (default: 0.7).
        bin_step: Step size for histogram bins (default: 1).
        xtick_step: Step size for x-axis tick marks (default: 10).
        stats_text_position: Position for statistics text box as (x, y) in axes coordinates (default: (0.7, 0.8)).
        save_path: Optional path to save the figure. If None, figure is not saved.
    
    Returns:
        fig: The matplotlib figure object.
        ax: The matplotlib axes object.
        stats: Dictionary with summary statistics.
    """
    # Extract spike counts per image from train_idx_spikes dictionary
    image_indices = []
    spike_counts_per_image = []
    
    for img_idx, spike_list in train_idx_spikes_dict.items():
        image_indices.append(img_idx)
        spike_counts_per_image.append(len(spike_list))  # Number of times this image was shown
    
    # Convert to numpy arrays for easier plotting
    image_indices = np.array(image_indices)
    spike_counts_per_image = np.array(spike_counts_per_image)
    
    # Create histogram
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(spike_counts_per_image, 
            bins=range(1, max(spike_counts_per_image) + 2, bin_step), 
            alpha=hist_alpha, 
            edgecolor='black', 
            color=hist_color)
    
    ax.set_xlabel('Number of presentations per image', fontsize=12)
    ax.set_ylabel('Number of images', fontsize=12)
    ax.set_xticks(range(1, max(spike_counts_per_image) + 1, xtick_step))
    ax.set_title(f'Distribution of Image Presentations\n(Total: {len(train_idx_spikes_dict)} unique images)', 
                fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Calculate statistics
    mean_presentations = np.mean(spike_counts_per_image)
    std_presentations = np.std(spike_counts_per_image)
    
    # Add statistics text
    ax.text(stats_text_position[0], stats_text_position[1], 
            f'Mean: {mean_presentations:.1f}\nStd: {std_presentations:.1f}', 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Compile summary statistics
    stats = {
        'total_unique_images': len(train_idx_spikes_dict),
        'mean_presentations': mean_presentations,
        'std_presentations': std_presentations,
        'min_presentations': int(min(spike_counts_per_image)),
        'max_presentations': int(max(spike_counts_per_image)),
        'images_shown_once': int(sum(spike_counts_per_image == 1)),
        'images_shown_more_than_10': int(sum(spike_counts_per_image > 10)),
        'image_indices': image_indices,
        'spike_counts_per_image': spike_counts_per_image
    }
    
    # Print summary statistics
    print(f"Summary of image presentations:")
    print(f"Total unique images: {stats['total_unique_images']}")
    print(f"Mean presentations per image: {stats['mean_presentations']:.2f}")
    print(f"Standard deviation: {stats['std_presentations']:.2f}")
    print(f"Min presentations: {stats['min_presentations']}")
    print(f"Max presentations: {stats['max_presentations']}")
    print(f"Images shown only once: {stats['images_shown_once']}")
    print(f"Images shown more than 10 times: {stats['images_shown_more_than_10']}")
    
    return fig, ax, stats

def plot_train_test_spike_statistics(
    train_idx_spikes_dict,
    spike_counts_test_tensor,
    test_idx_spikes_dict,
    min_presentations=2,
    figsize=(16, 6),
    train_color='steelblue',
    train_edge_color='lightsteelblue',
    test_color='darkorange',
    test_edge_color='moccasin',
    gap=2,
    mean_line_color='red',
    divider_alpha=0.5,
    stats_box_position=(0.15, 0.75),
    save_path=None
):
    """
    Plot spike count statistics for training and test images, sorted by response strength.
    
    Args:
        train_idx_spikes_dict: Dictionary mapping image indices to spike count tensors for training.
        spike_counts_test_tensor: Tensor of test spike counts (reps × images).
        test_idx_spikes_dict: Dictionary mapping test image indices to spike counts.
        min_presentations: Minimum number of presentations to include training image (default: 2).
        figsize: Figure size as (width, height) tuple (default: (16, 6)).
        train_color: Color for training data points (default: 'steelblue').
        train_edge_color: Color for training error bars (default: 'lightsteelblue').
        test_color: Color for test data points (default: 'darkorange').
        test_edge_color: Color for test error bars (default: 'moccasin').
        gap: Spacing between train and test sections (default: 2).
        mean_line_color: Color for mean reference line (default: 'red').
        divider_alpha: Transparency of divider line (default: 0.5).
        stats_box_position: Position of statistics text box as (x, y) (default: (0.15, 0.75)).
        save_path: Optional path to save the figure. If None, figure is not saved.
    
    Returns:
        fig: The matplotlib figure object.
        ax: The matplotlib axes object.
        stats: Dictionary with summary statistics.
    """
    # Filter training images with more than specified presentations
    filtered_images = {}
    for img_idx, spike_list in train_idx_spikes_dict.items():
        if len(spike_list) >= min_presentations:
            # Convert to numpy if tensor
            if hasattr(spike_list, 'cpu'):
                filtered_images[img_idx] = spike_list.cpu().numpy()
            else:
                filtered_images[img_idx] = np.array(spike_list)
    
    # Calculate statistics for training images
    image_data = []
    for idx, spike_list in filtered_images.items():
        mean_spike = np.nanmean(spike_list)
        std_spike = np.nanstd(spike_list)
        image_data.append((idx, mean_spike, std_spike))
    
    # Sort by mean spike count
    sorted_by_mean = sorted(image_data, key=lambda x: x[1])
    
    # Extract sorted training data
    train_image_indices = np.array([item[0] for item in sorted_by_mean])
    train_spike_means = np.array([item[1] for item in sorted_by_mean])
    train_spike_stds = np.array([item[2] for item in sorted_by_mean])
    
    # Calculate and sort test image data
    if hasattr(spike_counts_test_tensor, 'cpu'):
        test_data_np = spike_counts_test_tensor.cpu().numpy()
    else:
        test_data_np = np.array(spike_counts_test_tensor)
    
    test_spike_means_np = np.nanmean(test_data_np, axis=0)
    test_spike_stds_np = np.nanstd(test_data_np, axis=0)
    
    # Create tuples for sorting test images
    test_data = []
    for i, idx in enumerate(test_idx_spikes_dict.keys()):
        test_data.append((idx, test_spike_means_np[i], test_spike_stds_np[i]))
    
    # Sort test images by mean
    sorted_test_data = sorted(test_data, key=lambda x: x[1])
    
    test_image_indices = np.array([item[0] for item in sorted_test_data])
    test_spike_means = np.array([item[1] for item in sorted_test_data])
    test_spike_stds = np.array([item[2] for item in sorted_test_data])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training images
    n_train = len(train_image_indices)
    train_positions = np.arange(n_train)
    ax.errorbar(train_positions, train_spike_means, yerr=train_spike_stds, fmt='o',
                color=train_color, ecolor=train_edge_color, elinewidth=1,
                capsize=2, alpha=0.7, markersize=4, label='Training images')
    
    # Plot test images (with gap)
    n_test = len(test_image_indices)
    test_positions = np.arange(n_train + gap, n_train + gap + n_test)
    ax.errorbar(test_positions, test_spike_means, yerr=test_spike_stds, fmt='o',
                color=test_color, ecolor=test_edge_color, elinewidth=1,
                capsize=2, alpha=0.7, markersize=4, label='Test images (sorted)')
    
    # Add divider line
    ax.axvline(x=n_train + gap/2, color='black', linestyle='--', alpha=divider_alpha)
    ax.text(n_train + gap/2, ax.get_ylim()[1]*0.9, 'Test images →',
            ha='center', va='center', rotation=90, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    # Add mean reference line
    overall_train_mean = np.nanmean(train_spike_means)
    ax.axhline(y=overall_train_mean, color=mean_line_color, linestyle='--',
               label=f'Training Mean: {overall_train_mean:.2f}')
    
    # Labels and title
    ax.set_xlabel('Images ordered by response strength', fontsize=12)
    ax.set_ylabel('Mean spike count (with std)', fontsize=12)
    ax.set_title(f'Mean Spike Count per Image\n'
                 f'({n_train} training images + {n_test} test images, both sorted)',
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Calculate statistics
    mean_of_stds_train = np.nanmean(train_spike_stds)
    mean_of_stds_test = np.nanmean(test_spike_stds)
    
    # Add statistics text box
    ax.text(stats_box_position[0], stats_box_position[1],
            f"Training images: {n_train}\n"
            f"- Mean response: {overall_train_mean:.2f}\n"
            f"- Mean variability: {mean_of_stds_train:.2f}\n\n"
            f"Test images: {n_test}\n"
            f"- Mean response: {np.nanmean(test_spike_means):.2f}\n"
            f"- Mean variability: {mean_of_stds_test:.2f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Adjust x-ticks based on number of images
    if n_train + n_test > 50:
        ax.set_xticks([n_train/2, n_train + gap + n_test/2])
        ax.set_xticklabels(['Training', 'Test'])
    else:
        tick_positions = (list(range(0, n_train, max(1, n_train//5))) +
                         list(range(n_train + gap, n_train + gap + n_test, max(1, n_test//3))))
        ax.set_xticks(tick_positions)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Compile statistics
    stats = {
        'n_train': n_train,
        'n_test': n_test,
        'train_mean_response': overall_train_mean,
        'train_mean_std': mean_of_stds_train,
        'train_response_range': (float(np.min(train_spike_means)), float(np.max(train_spike_means))),
        'test_mean_response': float(np.nanmean(test_spike_means)),
        'test_mean_std': mean_of_stds_test,
        'test_response_range': (float(np.min(test_spike_means)), float(np.max(test_spike_means))),
        'train_image_indices': train_image_indices,
        'test_image_indices': test_image_indices,
        'train_spike_means': train_spike_means,
        'train_spike_stds': train_spike_stds,
        'test_spike_means': test_spike_means,
        'test_spike_stds': test_spike_stds
    }
    
    # Print summary
    print(f"Training images - {n_train} images with {min_presentations}+ presentations")
    print(f"- Average spike count: {overall_train_mean:.2f}")
    print(f"- Average std: {mean_of_stds_train:.2f}")
    print(f"- Range: {stats['train_response_range'][0]:.2f} to {stats['train_response_range'][1]:.2f}")
    print(f"\nTest images - {n_test} images with {test_data_np.shape[0]} repetitions each")
    print(f"- Average spike count: {stats['test_mean_response']:.2f}")
    print(f"- Average std: {mean_of_stds_test:.2f}")
    print(f"- Range: {stats['test_response_range'][0]:.2f} to {stats['test_response_range'][1]:.2f}")
    
    return fig, ax, stats

# ========= Testing ============

# ========== Testing and plotting =============

# def test_and_plot(model, img_test_reshaped, imgs_train, spike_counts_test_tensor, spike_counts_train, show=True):
    
#     # rtst <- spike_counts_test_tensor needs to be ( reps x imgs )
#     # GP_utils.test, explained_variance and plot_fit want r_test to be ( nreps, nimgs )

#     spk_count_test, spk_count_pred, expl_variance, std_expl_variance =GP_utils.test(
#         img_test_reshaped[...,None], 
#         spike_counts_test_tensor, 
#         X_train = None, 
#         at_iteration=None, 
#         print_expl_var=False,
#         **model )
    
#     # GP_utils.test, explained_variance and plot_fit want r_test to be ( nreps, nimgs )
#     # fig_test = GP_utils.plot_fit(spk_count_pred, spike_counts_test_tensor, expl_variance, sigma_expl_variance, 0)
#     # plt.show()



#     # Plot sta and hyp
#     n_px_side = 108
#     eps_0x  = model['hyperparams_tuple'][0]['eps_0x']
#     eps_0y  = model['hyperparams_tuple'][0]['eps_0y']
#     theta   = model['hyperparams_tuple'][0]

#     STA = GP_utils.get_cell_STA(imgs_train, spike_counts_train, zscore=True, show=False)

#     # ======== Plot RF on STA =========
#     # Eps_0 : Center of the receptive field
#     center_idxs = torch.tensor([(n_px_side-1)/2, (n_px_side-1)/2])
#     eps_idxs    = torch.tensor( [
#         center_idxs[0]*(1+eps_0x), 
#         center_idxs[1]*(1+eps_0y)
#         ])

#     # Beta : Width of the receptive field - Implemented by the "alpha_local" part of the C covariance matrix
#     ycord, xcord = torch.meshgrid( torch.linspace(-1, 1, n_px_side), torch.linspace(-1, 1, n_px_side), indexing='ij') # a grid of 108x108 points between -1 and 1
#     xcord = xcord.flatten()
#     ycord = ycord.flatten()
#     logalpha    = -torch.exp( theta['-2log2beta']     )*((xcord - eps_0x)**2+(ycord - eps_0y)**2  )
#     alpha_local =  torch.exp(logalpha)    # aplha_local in the paper

#     # Levels of the contour plot for distances [1sigma, 2sigma, 3sigma]
#     # (x**2 + y**2) = n*sigma -> alpha_local = exp( - (n*sigma)^2 / (2*sigma^2) )
#     levels = torch.tensor( [np.exp(-4.5), np.exp(-2), np.exp(-1/2) ])
#     if show:
#         initial_STA_fig, ax = plt.subplots(1, 1, figsize=(5,5)) 
#         ax.contour( alpha_local.reshape(n_px_side,n_px_side).cpu(), levels=levels.cpu(), colors='red', alpha=0.5)
#         ax.scatter( eps_idxs[0].cpu(),eps_idxs[1].cpu(), color='white', s=30, marker="o", label='Initial center guess', )
#         # ax.imshow( alpha_local.reshape(n_px_side,n_px_side).cpu(),)
#         ax.imshow(STA, origin='lower', cmap='bwr', vmax=STA.max(), vmin=-STA.max())
#         ax.legend(loc='upper right')
#         initial_STA_fig.suptitle(f'STA from nat imgs')
#         plt.show()

    

#     # ========  Standard correlation coefficient =========
#     # Even and odd repetitions of the same image, mean response. First index is repetitions
#     reven = torch.mean(spike_counts_test_tensor[0::2,:], axis=0)
#     rodd  = torch.mean(spike_counts_test_tensor[1::2,:], axis=0)
#     reliability = torch.abs(torch.corrcoef( torch.stack((reven,rodd))))[0,1]

#     corr = np.corrcoef(spk_count_pred.cpu().numpy(), np.nanmean(spike_counts_test_tensor.cpu().numpy(), axis=0))[0,1]

#     # ======== Correlation using bootstrapping ========

#     b_cor_reps, b_std_reps, _, _ = \
#         bootstrap_along_repetitions(
#             predictions=spk_count_pred.cpu().numpy(),
#             observations=spike_counts_test_tensor.cpu().numpy(),
#             n_bootstrap=10000,
#             plot_histogram=False  # Set to False if you don't want the plot in some cases
#     )

#     # Image to image variability: How much is my correlation affected by the choice of the images?
#     b_cor_images, b_std_images,\
#         = bootstrap_correlation(
#         predictions=spk_count_pred.cpu().numpy(),
#         observations_1D=np.mean(spike_counts_test_tensor.cpu().numpy(), axis=0),
#         n_bootstrap=10000,
#         plot_histogram=False  # Set to False if you don't want the plot in some cases
#     )

#     nested_b_cor, nested_var_cor, _, _ = \
#         bootstrap_correlation_images_samples_torch(
#             predictions=spk_count_pred,
#             observations=spike_counts_test_tensor,
#             n_bootstrap_reps=10000,
#             n_bootstrap_images=10000
#     )

#     print(f" r              = {corr:.2f}")
#     print(f" Bootstrapped r = {b_cor_images:.2f} ± {b_std_images:.2f} across images")
#     print(f' Bootstrapped r = {b_cor_reps:.2f} ± {b_std_reps:.2f} across reps')
#     print(f" Bootstrapped r = {nested_b_cor:.2f} ± {np.sqrt(nested_var_cor):.2f} across images and reps")

#     print(f" Explained var = {expl_variance:.2f} ± {std_expl_variance:.2f}")
#     # print(f" Bootstrap r²  = {mean_corr_squared:.2f} ± {std_corr_squared:.2f}  across images")
#     # print(f" Bootstrap r²  = {mean_corr_squared1:.2f} ± {np.sqrt(var_corr_squared1):.2f}  across images and reps")

#     if show:
#         plt.scatter( spk_count_pred.cpu().numpy(), np.mean(spk_count_test.cpu().numpy(), axis=0), 
#                     color='k',)
#         plt.xlim(-0.5, 12)
#         plt.ylim(-0.5, 12)
#         plt.xticks(np.arange(0, 13, 2), fontsize=22)
#         plt.yticks(np.arange(0, 13, 2), fontsize=22)
#         # Add text directly to the plot
#         plt.title(f'r = {corr:.2f}  Reliability = {reliability:.2f}', fontsize=22)
#     # # endregion

#     return spk_count_test, spk_count_pred, \
#         expl_variance, std_expl_variance, \
#         corr, \
#         b_cor_images, b_std_images, \
#         b_cor_reps, b_std_reps, \
#         nested_b_cor, np.sqrt(nested_var_cor), 
        

def test_and_plot(model, img_test_reshaped, imgs_train, spike_counts_test_tensor, spike_counts_train, silent=False, show=True, get_sta=False, **kwargs):
    

    bootstrap_images = kwargs.get('bootstrap_images', True)
    bootstrap_repetitions = kwargs.get('bootstrap_repetitions', True)
    boostrap_nested = kwargs.get('boostrap_nested', True)

    # rtst <- spike_counts_test_tensor needs to be ( reps x imgs )
    # GP_utils.test, explained_variance and plot_fit want r_test to be ( nreps, nimgs )

    spk_count_test, spk_count_pred, expl_variance, std_expl_variance = GP_utils.test(
        img_test_reshaped[...,None], 
        spike_counts_test_tensor, 
        X_train = None, 
        at_iteration=None, 
        print_expl_var=False,
        **model )
    
    # GP_utils.test, explained_variance and plot_fit want r_test to be ( nreps, nimgs )
    # fig_test = GP_utils.plot_fit(spk_count_pred, spike_counts_test_tensor, expl_variance, sigma_expl_variance, 0)
    # plt.show()

    if get_sta:
        # Plot sta and hyp
        n_px_side = 108
        eps_0x  = model['hyperparams_tuple'][0]['eps_0x']
        eps_0y  = model['hyperparams_tuple'][0]['eps_0y']
        theta   = model['hyperparams_tuple'][0]

        STA = GP_utils.get_cell_STA(imgs_train, spike_counts_train, zscore=True, show=False)

        # ======== Plot RF on STA =========
        # Eps_0 : Center of the receptive field
        center_idxs = torch.tensor([(n_px_side-1)/2, (n_px_side-1)/2])
        eps_idxs    = torch.tensor( [
            center_idxs[0]*(1+eps_0x), 
            center_idxs[1]*(1+eps_0y)
            ])

        # Beta : Width of the receptive field - Implemented by the "alpha_local" part of the C covariance matrix
        ycord, xcord = torch.meshgrid( torch.linspace(-1, 1, n_px_side), torch.linspace(-1, 1, n_px_side), indexing='ij') # a grid of 108x108 points between -1 and 1
        xcord = xcord.flatten()
        ycord = ycord.flatten()
        logalpha    = -torch.exp( theta['-2log2beta']     )*((xcord - eps_0x)**2+(ycord - eps_0y)**2  )
        alpha_local =  torch.exp(logalpha)    # aplha_local in the paper

        # Levels of the contour plot for distances [1sigma, 2sigma, 3sigma]
        # (x**2 + y**2) = n*sigma -> alpha_local = exp( - (n*sigma)^2 / (2*sigma^2) )
        levels = torch.tensor( [np.exp(-4.5), np.exp(-2), np.exp(-1/2) ])
    
    if show and get_sta:
        initial_STA_fig, ax = plt.subplots(1, 1, figsize=(5,5)) 
        ax.contour( alpha_local.reshape(n_px_side,n_px_side).cpu(), levels=levels.cpu(), colors='red', alpha=0.5)
        ax.scatter( eps_idxs[0].cpu(),eps_idxs[1].cpu(), color='white', s=30, marker="o", label='Initial center guess', )
        # ax.imshow( alpha_local.reshape(n_px_side,n_px_side).cpu(),)
        ax.imshow(STA, origin='lower', cmap='bwr', vmax=STA.max(), vmin=-STA.max())
        ax.legend(loc='upper right')
        initial_STA_fig.suptitle(f'STA from nat imgs')
        plt.show()

    # ========  Standard correlation coefficient =========
    # Even and odd repetitions of the same image, mean response. First index is repetitions
    reven = torch.mean(spike_counts_test_tensor[0::2,:], axis=0)
    rodd  = torch.mean(spike_counts_test_tensor[1::2,:], axis=0)
    reliability = torch.abs(torch.corrcoef( torch.stack((reven,rodd))))[0,1]

    corr = np.corrcoef(spk_count_pred.cpu().numpy(), np.mean(spike_counts_test_tensor.cpu().numpy(), axis=0))[0,1]

    # ======== Correlation using bootstrapping ========

    if bootstrap_repetitions:
        b_cor_reps, b_std_reps, _, _ = \
            bootstrap_along_repetitions(
                predictions=spk_count_pred.cpu().numpy(),
                observations=spike_counts_test_tensor.cpu().numpy(),
                n_bootstrap=10000,
                plot_histogram=False  # Set to False if you don't want the plot in some cases
        )
    else:
        b_cor_reps, b_std_reps = np.nan, np.nan

    if bootstrap_images:
        # Image to image variability: How much is my correlation affected by the choice of the images?
        b_cor_images, b_std_images,\
            = bootstrap_correlation(
            predictions=spk_count_pred.cpu().numpy(),
            observations_1D=np.mean(spike_counts_test_tensor.cpu().numpy(), axis=0),
            n_bootstrap=10000,
            plot_histogram=False  # Set to False if you don't want the plot in some cases
        )
    else:
        b_cor_images, b_std_images = np.nan, np.nan

    if boostrap_nested:
        nested_b_cor, nested_var_cor, _, _ = \
            bootstrap_correlation_images_samples_torch(
            predictions=spk_count_pred,
            observations=spike_counts_test_tensor,
            n_bootstrap_reps=10000,
            n_bootstrap_images=10000
    )
    else:
        nested_b_cor, nested_var_cor = np.nan, np.nan

    if not silent:
        print(f" r              = {corr:.2f}")
        print(f" Bootstrapped r = {b_cor_images:.2f} ± {b_std_images:.2f} across images")
        print(f' Bootstrapped r = {b_cor_reps:.2f} ± {b_std_reps:.2f} across reps')
        print(f" Bootstrapped r = {nested_b_cor:.2f} ± {np.sqrt(nested_var_cor):.2f} across images and reps")

        print(f" Explained var = {expl_variance:.2f} ± {std_expl_variance:.2f}")
        # print(f" Bootstrap r²  = {mean_corr_squared:.2f} ± {std_corr_squared:.2f}  across images")
        # print(f" Bootstrap r²  = {mean_corr_squared1:.2f} ± {np.sqrt(var_corr_squared1):.2f}  across images and reps")

    if show:
        plt.scatter( spk_count_pred.cpu().numpy(), np.mean(spk_count_test.cpu().numpy(), axis=0), 
                    color='k',)
        plt.xlim(-0.5, 12)
        plt.ylim(-0.5, 12)
        plt.xticks(np.arange(0, 13, 2), fontsize=22)
        plt.yticks(np.arange(0, 13, 2), fontsize=22)
        # Add text directly to the plot
        plt.title(f'r = {corr:.2f}  Reliability = {reliability:.2f}', fontsize=22)
    # # endregion

    return spk_count_test, spk_count_pred, \
        expl_variance, std_expl_variance, \
        corr, \
        b_cor_images, b_std_images, \
        b_cor_reps, b_std_reps, \
        nested_b_cor, np.sqrt(nested_var_cor), 

# ========== Bootstrapping test set ===========

def bootstrap_correlation(predictions, observations_1D, n_bootstrap=1000, plot_histogram=True):
    '''

    You have predictions and observations arrays of the same shape (nsamples).

    What is the correlation between these two if samples where chosen differently 
    ( 3 timese sample 1, 2 times sample 2, zero times sample 3, etc..)?

    Bootstrapping estimates this by resampling the 2 arrays with replacement.

    for i in 1000:
        correlation_i = corrcoeff(predictions[indices], observations[indices])

    mean_correlation = mean([correlation_i])
    std_correlation  = std([correlation_i])

    This code also gets the mean and std for r_squared ( not the explained variance, r_squared )
    
    Args:
        observations: shape = (nimgs) . its already averaged along the repetitions for each image

        plot_histogram: bool, whether to plot the histogram of correlations obtained 
                              from each bootstrap sample
    
    '''
    n_samples = len(predictions)
    
    correlations = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_corr = np.corrcoef(predictions[indices], observations_1D[indices])[0,1]

        correlations[i] = boot_corr

    mean_r = np.mean(correlations)
    std_r  = np.std(correlations)

    if plot_histogram:
        plt.figure(figsize=(8, 6))
        
        # Plot histogram of correlation values (not squared)
        plt.hist(correlations, bins=30, alpha=0.7, color='steelblue', 
                edgecolor='black', label=f'Bootstrap Samples (n={n_bootstrap})')
        
        # Add vertical line for the mean
        plt.axvline(np.mean(correlations), color='red', linestyle='--', linewidth=2,
                label=f'Mean r: {mean_r:.2f} ± {std_r:.2f}')
        
        # Plot details
        plt.xlabel('Correlation Coefficient (r)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Bootstrap Distribution of Correlation Coefficients', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return mean_r, std_r

def bootstrap_along_repetitions(predictions, observations, n_bootstrap=1000, plot_histogram=True):
    '''
    Bootstraps correlation by resampling repetitions for each image rather than
    resampling images. This better captures trial-to-trial variability.
    
    Args:
        predictions: shape (n_images) - model predictions
        observations: shape (n_repetitions, n_images) - neural responses
        n_bootstrap: number of bootstrap iterations
        plot_histogram: whether to plot the distribution of correlations
    
    Returns:
        mean_r, std_r, mean_r_squared, std_r_squared
    '''
    n_reps, n_images = observations.shape
    correlations = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Create a bootstrapped average response for each image
        bootstrap_means = np.zeros(n_images)
        
        for img_idx in range(n_images):
            # Sample repetitions with replacement for this image
            rep_indices = np.random.choice(n_reps, size=n_reps, replace=True)
            bootstrap_means[img_idx] = np.mean(observations[rep_indices, img_idx])
        
        # Calculate correlation with model predictions
        correlations[i] = np.corrcoef(predictions, bootstrap_means)[0,1]
    
    # Calculate statistics
    mean_r = np.mean(correlations)
    std_r = np.std(correlations)
    mean_r_squared = np.mean(correlations**2)
    std_r_squared = np.std(correlations**2)
    
    if plot_histogram:
        plt.figure(figsize=(8, 6))
        plt.hist(correlations, bins=30, alpha=0.7, color='steelblue', 
                edgecolor='black', label=f'Bootstrap Samples (n={n_bootstrap})')
        plt.axvline(mean_r, color='red', linestyle='--', linewidth=2,
                label=f'Mean r: {mean_r:.2f} ± {std_r:.2f}')
        plt.xlabel('Correlation Coefficient (r)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Bootstrap Distribution (Sampling Repetitions)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    return mean_r, std_r, mean_r_squared, std_r_squared

def bootstrap_correlation_images_samples_torch(
        predictions,
        observations,                  # shape (n_reps, n_images)
        n_bootstrap_reps=1000,
        n_bootstrap_images=1000,
        outer_batch_size=1000,
        device=None):
    """
    Torch nested bootstrap with variance decomposition.
    Outer: resample repetitions per image -> synthetic mean response vector.
    Inner: bootstrap images (vectorized) -> Var(r | reps).
    Returns:
        mean_r,
        total_variance,
        mean_r_per_outer (tensor length n_bootstrap_reps),
        var_r_given_reps_per_outer (tensor length n_bootstrap_reps)
    """
    if device is None:
        device = observations.device if isinstance(observations, torch.Tensor) else 'cpu'

    if not isinstance(observations, torch.Tensor):
        observations_t = torch.as_tensor(observations, dtype=torch.float32, device=device)
    else:
        observations_t = observations.to(device=device, dtype=torch.float32)

    if not isinstance(predictions, torch.Tensor):
        predictions_t = torch.as_tensor(predictions, dtype=torch.float32, device=device)
    else:
        predictions_t = predictions.to(device=device, dtype=torch.float32)

    n_reps, n_images = observations_t.shape

    mean_r_outer = torch.empty(n_bootstrap_reps, device=device)
    var_r_given_reps = torch.empty(n_bootstrap_reps, device=device)

    # Pre-sample image bootstrap indices once (shared across outers) for efficiency.
    img_idx = torch.randint(0, n_images, (n_bootstrap_images, n_images), device=device)

    # Process outer bootstraps in batches to reduce Python overhead.
    for start in range(0, n_bootstrap_reps, outer_batch_size):
        end = min(start + outer_batch_size, n_bootstrap_reps)
        batch_size = end - start

        # Sample repetition indices: shape (batch_size, n_images, n_reps)
        # We generate (batch_size, n_images, n_reps) indices then gather and mean over reps.
        rep_idx = torch.randint(0, n_reps, (batch_size, n_images, n_reps), device=device)
        # Expand observations for gather: (1, n_reps, n_images) -> (batch_size, n_reps, n_images)
        obs_exp = observations_t.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_reps, n_images)
        # Gather along repetitions dim=1 using rep_idx permuted to (batch, n_reps, n_images)
        rep_idx_perm = rep_idx.permute(0, 2, 1)  # (batch, n_reps, n_images)
        sampled_reps = obs_exp.gather(1, rep_idx_perm)  # (batch, n_reps, n_images)
        b_observations = sampled_reps.mean(dim=1)       # (batch, n_images)

        # Inner image bootstrap (vectorized per outer sample)
        # For each outer sample, we need correlations for n_bootstrap_images sets of indices.
        # We reuse img_idx for all batch members: broadcast and gather.
        # Shapes:
        # predictions: (n_images) -> (n_bootstrap_images, n_images)
        pred_boot = predictions_t.expand(n_bootstrap_images, n_images).gather(1, img_idx)

        # For observations: need (batch_size, n_bootstrap_images, n_images)
        b_obs_expanded = b_observations.unsqueeze(0).expand(n_bootstrap_images, batch_size, n_images)
        b_obs_boot = b_obs_expanded.gather(2, img_idx.unsqueeze(1).expand(-1, batch_size, -1))
        # Rearrange to (batch_size, n_bootstrap_images, n_images)
        b_obs_boot = b_obs_boot.permute(1, 0, 2)
        pred_boot_expanded = pred_boot.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute per-bootstrap correlation for each outer sample:
        # Center along last dim
        pred_mean = pred_boot_expanded.mean(dim=2, keepdim=True)
        obs_mean  = b_obs_boot.mean(dim=2, keepdim=True)
        pred_c = pred_boot_expanded - pred_mean
        obs_c  = b_obs_boot - obs_mean
        cov = (pred_c * obs_c).sum(dim=2)
        var_p = (pred_c.pow(2)).sum(dim=2)
        var_o = (obs_c.pow(2)).sum(dim=2)
        denom = torch.sqrt(var_p * var_o).clamp_min(1e-12)
        r_inner = (cov / denom).clamp(-1, 1)  # (batch_size, n_bootstrap_images)

        mean_r_inner = r_inner.mean(dim=1)                # E[r | reps] for each outer
        var_r_inner  = r_inner.var(dim=1, unbiased=False) # Var(r | reps)

        mean_r_outer[start:end] = mean_r_inner
        var_r_given_reps[start:end] = var_r_inner

    # Decomposition
    var_r_stimulus = var_r_given_reps.mean()              # E[ Var(r | reps) ]
    var_neural_noise = mean_r_outer.var(unbiased=False)   # Var( E[r | reps] )
    total_variance = var_r_stimulus + var_neural_noise

    return (mean_r_outer.mean().item(),
            total_variance.item(),
            mean_r_outer.detach().cpu(),
            var_r_given_reps.detach().cpu())





















