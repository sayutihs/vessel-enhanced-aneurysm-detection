import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class AneurysmDataset(Dataset):
    """
    A PyTorch Dataset designed to load real NIfTI (.nii or .nii.gz) scans,
    normalize them, cache them to the fast local SSD as raw .npy files,
    apply dynamic padding for smaller volumes, and extract 3D patches with augmentation.
    """
    def __init__(self, image_dir=None, mask_dir=None, file_list=None, patch_size=(64, 64, 64), num_patches_per_volume=4, transform=None, cache_dir='/content/cache_npy', augment=False):
        """
        Args:
            image_dir (str, optional): Folder containing raw 3D scans.
            mask_dir (str, optional): Folder containing corresponding binary masks.
            file_list (list, optional): Pre-existing list of dicts [{'image': path, 'label': path}, ...].
            patch_size (tuple): Size of the 3D sub-volumes to extract (D, H, W).
            num_patches_per_volume (int): Number of patches to crop from each 3D volume during one epoch.
            transform (callable, optional): Optional transform.
            cache_dir (str): Location on local SSD to store uncompressed .npy cache.
            augment (bool): Set to True for the training dataset to enable random 3D flips/rotations.
        """
        self.patch_size = patch_size
        self.num_patches_per_volume = num_patches_per_volume
        self.transform = transform
        self.cache_dir = cache_dir
        self.augment = augment
        self.file_pairs = []

        if file_list is not None:
            self.file_pairs = [dict(d) for d in file_list]
        elif image_dir is not None and mask_dir is not None:
            image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii*")))
            mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.nii*")))

            if len(image_paths) == 0:
                raise FileNotFoundError(f"No NIfTI (.nii or .nii.gz) images found in directory: {image_dir}")
            
            for img_path in image_paths:
                base_name = os.path.basename(img_path)
                matching_mask = None
                for m_path in mask_paths:
                    m_base = os.path.basename(m_path)
                    raw_img_name = os.path.splitext(os.path.splitext(base_name)[0])[0]
                    if raw_img_name in m_base:
                        matching_mask = m_path
                        break
                
                if matching_mask is not None:
                    self.file_pairs.append({'image': img_path, 'label': matching_mask})
                else:
                    print(f"Warning: No matching mask found for image {img_path}")
        else:
            raise ValueError("You must provide either a 'file_list' or both 'image_dir' and 'mask_dir'.")

        print(f"Dataset initialized: Found {len(self.file_pairs)} valid image-mask pairs.")
        
        # Setup and run pre-caching
        os.makedirs(self.cache_dir, exist_ok=True)
        self._pre_cache_files()

    def _pre_cache_files(self):
        import nibabel as nib
        print(f"\n---> Checking/Pre-caching dataset to '{self.cache_dir}' (this happens once)...")
        
        cached_count = 0
        for i, pair in enumerate(self.file_pairs):
            patient_id = pair.get('patient_id')
            if not patient_id:
                patient_id = os.path.basename(os.path.dirname(pair['image']))
                pair['patient_id'] = patient_id
                
            pair['cached_img'] = os.path.join(self.cache_dir, f"img_{patient_id}.npy")
            pair['cached_lbl'] = os.path.join(self.cache_dir, f"lbl_{patient_id}.npy")
            
            # Cache image and label if not already done
            if not (os.path.exists(pair['cached_img']) and os.path.exists(pair['cached_lbl'])):
                img_nii = nib.load(pair['image'])
                image_data = img_nii.get_fdata().astype(np.float32)
                
                # Z-Score Intensity Normalization (applied to original high-res volumes)
                mean = np.mean(image_data)
                std = np.std(image_data)
                if std > 0:
                    image_data = (image_data - mean) / std
                
                lbl_nii = nib.load(pair['label'])
                label_raw = lbl_nii.get_fdata().astype(np.float32)
                label_data = (np.abs(label_raw - 1.0) < 0.5).astype(np.float32)
                
                np.save(pair['cached_img'], image_data)
                np.save(pair['cached_lbl'], label_data)
                cached_count += 1
                
            # Pre-calculate positive voxel coordinates (cached in memory before dataloader workers are spawned)
            lbl_full = np.load(pair['cached_lbl'])
            pair['positive_voxels'] = np.argwhere(lbl_full > 0.5)
                
            if (i + 1) % 10 == 0 or (i + 1) == len(self.file_pairs):
                print(f"  Processed {i + 1}/{len(self.file_pairs)} patients...")
        
        if cached_count > 0:
            print(f"---> Cached {cached_count} new patients to local SSD. Ready!")
        else:
            print("---> All patient files already cached. Loading instantly!")

    def __len__(self):
        return len(self.file_pairs) * self.num_patches_per_volume

    def __getitem__(self, idx):
        volume_idx = idx // self.num_patches_per_volume
        pair = self.file_pairs[volume_idx]

        # Memory-map the cached arrays (extremely fast, zero memory overhead)
        image_data = np.load(pair['cached_img'], mmap_mode='r')
        label_data = np.load(pair['cached_lbl'], mmap_mode='r')

        D, H, W = image_data.shape
        p_d, p_h, p_w = self.patch_size

        # Dynamic padding if scan dimension is smaller than patch size (e.g. depth < 96)
        if D < p_d or H < p_h or W < p_w:
            # Load fully to memory for padding
            image_data = np.load(pair['cached_img'])
            label_data = np.load(pair['cached_lbl'])
            
            pad_d = max(0, p_d - D)
            pad_h = max(0, p_h - H)
            pad_w = max(0, p_w - W)
            
            image_data = np.pad(image_data, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            label_data = np.pad(label_data, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            D, H, W = image_data.shape

        # Find positive voxel coordinates (cached in memory per patient to avoid np.argwhere recalculations)
        if 'positive_voxels' not in pair:
            lbl_full = np.load(pair['cached_lbl'])
            pair['positive_voxels'] = np.argwhere(lbl_full > 0.5)

        positive_voxels = pair['positive_voxels']

        # 70% chance to center patch around a random aneurysm voxel, 30% chance completely random
        if len(positive_voxels) > 0 and np.random.rand() < 0.7:
            center_voxel = positive_voxels[np.random.choice(len(positive_voxels))]
            z_c, y_c, x_c = center_voxel
            
            z_start = int(np.clip(z_c - p_d // 2, 0, D - p_d))
            y_start = int(np.clip(y_c - p_h // 2, 0, H - p_h))
            x_start = int(np.clip(x_c - p_w // 2, 0, W - p_w))
        else:
            z_start = np.random.randint(0, D - p_d + 1)
            y_start = np.random.randint(0, H - p_h + 1)
            x_start = np.random.randint(0, W - p_w + 1)

        # Slice sub-volume (only triggers disk read of this small sub-volume)
        image_patch = np.array(image_data[z_start:z_start+p_d, y_start:y_start+p_h, x_start:x_start+p_w])
        label_patch = np.array(label_data[z_start:z_start+p_d, y_start:y_start+p_h, x_start:x_start+p_w])

        # Apply 3D Spatial Data Augmentation (Only for training)
        if self.augment:
            # 1. Random Flips along Depth, Height, or Width (50% chance each)
            if np.random.rand() < 0.5:
                image_patch = np.flip(image_patch, axis=0) # flip Z
                label_patch = np.flip(label_patch, axis=0)
            if np.random.rand() < 0.5:
                image_patch = np.flip(image_patch, axis=1) # flip Y
                label_patch = np.flip(label_patch, axis=1)
            if np.random.rand() < 0.5:
                image_patch = np.flip(image_patch, axis=2) # flip X
                label_patch = np.flip(label_patch, axis=2)

            # 2. Random 90-degree rotations in 3D planes (50% chance)
            if np.random.rand() < 0.5:
                axes = np.random.choice([0, 1, 2], size=2, replace=False)
                k = np.random.choice([1, 2, 3])
                image_patch = np.rot90(image_patch, k, axes)
                label_patch = np.rot90(label_patch, k, axes)

        # Binarize label patch
        label_patch = (label_patch > 0.5).astype(np.float32)

        # Convert to PyTorch tensors and add channel dimension: [C, D, H, W]
        image_tensor = torch.tensor(image_patch.copy(), dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label_patch.copy(), dtype=torch.float32).unsqueeze(0)

        if self.transform:
            image_tensor, label_tensor = self.transform(image_tensor, label_tensor)

        return image_tensor, label_tensor


def split_dataset_by_test_patients(file_list, test_patient_ids, val_ratio=0.15, random_seed=42):
    """
    Splits the dataset using a pre-defined list of test patient IDs.
    The remaining patients are shuffled and split into Train and Validation sets.
    
    Args:
        file_list (list): List of dicts containing image and label paths.
        test_patient_ids (list): List of patient ID strings/numbers designated for testing.
        val_ratio (float): Ratio of remaining patients to assign to validation.
        random_seed (int): Random seed for reproducibility.
    
    Returns:
        train_files, val_files, test_files
    """
    import random
    
    test_files = []
    remaining_files = []
    
    # Convert test_patient_ids elements to string for comparison
    test_ids_set = {str(pid).strip().upper() for pid in test_patient_ids}
    
    for pair in file_list:
        pid = str(pair.get('patient_id', '')).strip().upper()
        if pid in test_ids_set:
            test_files.append(pair)
        else:
            remaining_files.append(pair)
            
    # Shuffle remaining files using the random seed for reproducible train/val split
    random.seed(random_seed)
    random.shuffle(remaining_files)
    
    num_remaining = len(remaining_files)
    total_n = len(file_list)
    val_size = max(1, int(val_ratio * total_n))
    
    val_files = remaining_files[:val_size]
    train_files = remaining_files[val_size:]
    
    return train_files, val_files, test_files
