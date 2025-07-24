# File: src/data_loader.py

import os
import h5py
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Callable
import logging
from pathlib import Path

from utils import fft2c, ifft2c, complex_to_real, apply_mask, root_sum_of_squares


class FastMRIDataset(Dataset):
    """
    Dataset class for FastMRI multi-coil data.
    
    This dataset handles loading of multi-coil k-space data, applies undersampling masks,
    and provides both k-space and image domain data for training neural operators.
    """
    
    def __init__(
        self,
        data_path: str,
        acceleration: int = 4,
        center_fraction: float = 0.08,
        transform: Optional[Callable] = None,
        use_seed: bool = True,
        seed: int = 42
    ):
        """
        Initialize FastMRI dataset.
        
        Args:
            data_path: Path to directory containing .h5 files
            acceleration: Acceleration factor for undersampling
            center_fraction: Fraction of center k-space lines to retain
            transform: Optional transform to apply to samples
            use_seed: Whether to use seed for reproducible masks
            seed: Random seed for mask generation
        """
        self.data_path = Path(data_path)
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.transform = transform
        self.use_seed = use_seed
        self.seed = seed
        
        # Find all .h5 files
        self.files = list(self.data_path.glob("*.h5"))
        if not self.files:
            raise ValueError(f"No .h5 files found in {data_path}")
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Found {len(self.files)} files in {data_path}")
        
        # Cache file metadata
        self._cache_metadata()
    
    def _cache_metadata(self) -> None:
        """Cache metadata about the dataset files."""
        self.file_metadata = []
        
        for file_path in self.files:
            try:
                with h5py.File(file_path, 'r') as f:
                    kspace = f['kspace']
                    num_slices = kspace.shape[0]
                    
                    for slice_idx in range(num_slices):
                        self.file_metadata.append({
                            'file_path': file_path,
                            'slice_idx': slice_idx,
                            'shape': kspace.shape[1:]  # (coils, height, width)
                        })
            except Exception as e:
                self.logger.warning(f"Error reading {file_path}: {e}")
                continue
        
        self.logger.info(f"Total slices: {len(self.file_metadata)}")
    
    def __len__(self) -> int:
        """Return the total number of slices in the dataset."""
        return len(self.file_metadata)
    
    # def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    #     """
    #     Get a sample from the dataset.
        
    #     Args:
    #         idx: Index of the sample
            
    #     Returns:
    #         Dictionary containing:
    #             - 'kspace_full': Fully sampled k-space data
    #             - 'kspace_masked': Undersampled k-space data  
    #             - 'mask': Undersampling mask
    #             - 'target': Target fully sampled image
    #             - 'image_masked': Zero-filled reconstruction from masked k-space
    #             - 'metadata': Additional metadata
    #     """
    #     metadata = self.file_metadata[idx]
        
    #     try:
    #         with h5py.File(metadata['file_path'], 'r') as f:
    #             # Load k-space data for specific slice
    #             kspace_full = f['kspace'][metadata['slice_idx']]  # Shape: (coils, height, width)
                
    #             # Convert to torch tensor and make complex
    #             kspace_full = torch.from_numpy(kspace_full)
    #             if not torch.is_complex(kspace_full):
    #                 # Assume real/imaginary are stored as separate array elements
    #                 kspace_full = torch.view_as_complex(kspace_full.contiguous())
                
    #             # Generate undersampling mask
    #             mask = self._generate_mask(kspace_full.shape[-2:], idx if self.use_seed else None)
                
    #             # Apply mask to k-space
    #             kspace_masked = apply_mask(kspace_full, mask)
                
    #             # Reconstruct images
    #             image_full = self._reconstruct_image(kspace_full)
    #             image_masked = self._reconstruct_image(kspace_masked)
                
    #             # Prepare sample
    #             sample = {
    #                 'kspace_full': complex_to_real(kspace_full),
    #                 'kspace_masked': complex_to_real(kspace_masked),
    #                 'mask': mask,
    #                 'target': torch.abs(image_full),
    #                 'image_masked': torch.abs(image_masked),
    #                 'metadata': {
    #                     'file_path': str(metadata['file_path']),
    #                     'slice_idx': metadata['slice_idx'],
    #                     'acceleration': self.acceleration,
    #                     'center_fraction': self.center_fraction
    #                 }
    #             }
                
    #             # Apply transforms if provided
    #             if self.transform:
    #                 sample = self.transform(sample)
                
    #             return sample
                
    #     except Exception as e:
    #         self.logger.error(f"Error loading sample {idx}: {e}")
    #         # Return a dummy sample in case of error
    #         return self._get_dummy_sample()
    
    def _pad_to_common_size(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Pad tensor to common FastMRI size.
        Most FastMRI images are 640x368, but some vary slightly.
        """
        # Target size - slightly larger to accommodate all variations
        target_h, target_w = 640, 372  # Use the larger width we've seen
        
        # Get current size
        *batch_dims, h, w = tensor.shape
        
        # Pad height if needed
        if h < target_h:
            pad_h = target_h - h
            tensor = F.pad(tensor, (0, 0, 0, pad_h))
        
        # Pad width if needed  
        if w < target_w:
            pad_w = target_w - w
            tensor = F.pad(tensor, (0, pad_w, 0, 0))
        
        return tensor


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        metadata = self.file_metadata[idx]
        target_size = (640, 368)  # (height, width)
        try:
            with h5py.File(metadata['file_path'], 'r') as f:
                # Load k-space data for specific slice
                kspace_full = f['kspace'][metadata['slice_idx']]  # Shape: (coils, height, width), complex64
                
                # Convert to torch tensor
                kspace_full = torch.from_numpy(kspace_full)
                self.logger.debug(f"kspace_full shape: {kspace_full.shape}, dtype: {kspace_full.dtype}")
                
                # Ensure kspace_full is complex
                if not torch.is_complex(kspace_full):
                    self.logger.error(f"kspace_full is not complex, shape: {kspace_full.shape}")
                    raise ValueError(f"Expected complex k-space data, got shape {kspace_full.shape}")
                
                # Generate undersampling mask
                mask = self._generate_mask(kspace_full.shape[-2:], idx if self.use_seed else None)
                
                # Apply mask to k-space
                kspace_masked = apply_mask(kspace_full, mask)
                
                kspace_full = self._pad_to_common_size(kspace_full)
                kspace_masked = self._pad_to_common_size(kspace_masked)

                # Reconstruct images
                image_full = self._reconstruct_image(kspace_full)
                image_masked = self._reconstruct_image(kspace_masked)
          
                
                # Prepare sample
                sample = {
                    'kspace_full': complex_to_real(kspace_full),
                    'kspace_masked': complex_to_real(kspace_masked),
                    'mask': mask,
                    'target': torch.abs(image_full),
                    'image_masked': torch.abs(image_masked),
                    'metadata': {
                        'file_path': str(metadata['file_path']),
                        'slice_idx': metadata['slice_idx'],
                        'acceleration': self.acceleration,
                        'center_fraction': self.center_fraction
                    }
                }
                self.logger.debug(f"sample['kspace_full'] shape: {sample['kspace_full'].shape}")
                
                # Apply transforms if provided
                if self.transform:
                    sample = self.transform(sample)
                
                return sample
                
        except Exception as e:
            self.logger.error(f"Error loading sample {idx}: {e}")
            return self._get_dummy_sample()
    
    def _generate_mask(self, shape: Tuple[int, int], seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate random undersampling mask.
        
        Args:
            shape: Shape of the k-space (height, width)
            seed: Optional seed for reproducible masks
            
        Returns:
            Binary mask tensor
        """
        height, width = shape
        
        # Set seed for reproducible masks
        if seed is not None:
            torch.manual_seed(self.seed + seed)
        
        # Calculate number of center lines to keep
        num_center_lines = int(self.center_fraction * width)
        center_start = width // 2 - num_center_lines // 2
        center_end = center_start + num_center_lines
        
        # Calculate total lines to keep
        total_lines = width // self.acceleration
        remaining_lines = total_lines - num_center_lines
        
        # Create mask
        mask = torch.zeros(width)
        
        # Always keep center lines
        mask[center_start:center_end] = 1
        
        # Randomly select remaining lines
        available_indices = list(range(width))
        for i in range(center_start, center_end):
            if i in available_indices:
                available_indices.remove(i)
        
        if remaining_lines > 0 and available_indices:
            selected_indices = torch.randperm(len(available_indices))[:remaining_lines]
            for idx in selected_indices:
                mask[available_indices[idx]] = 1
        
        # Expand to full k-space shape
        mask = mask.unsqueeze(0).expand(height, -1)
        
        return mask
    
    def _reconstruct_image(self, kspace: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct image from k-space using IFFT and coil combination.
        
        Args:
            kspace: K-space data (coils, height, width)
            
        Returns:
            Combined image (height, width)
        """
        # Apply inverse FFT to each coil
        coil_images = ifft2c(kspace)
        
        # Combine coils using root sum of squares
        combined_image = root_sum_of_squares(coil_images, dim=0).squeeze()
        
        return combined_image
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return a dummy sample in case of loading errors."""
        dummy_shape = (8, 320, 320)  # Typical multi-coil shape
        
        return {
            'kspace_full': torch.zeros(*dummy_shape, 2),
            'kspace_masked': torch.zeros(*dummy_shape, 2),
            'mask': torch.zeros(dummy_shape[-2:]),
            'target': torch.zeros(dummy_shape[-2:]),
            'image_masked': torch.zeros(dummy_shape[-2:]),
            'metadata': {
                'file_path': 'dummy',
                'slice_idx': 0,
                'acceleration': self.acceleration,
                'center_fraction': self.center_fraction
            }
        }


class NormalizationTransform:
    """Transform to normalize data to zero mean and unit variance."""
    
    def __init__(self, keys: List[str] = ['target', 'image_masked']):
        """
        Initialize normalization transform.
        
        Args:
            keys: Keys in sample dict to normalize
        """
        self.keys = keys
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply normalization to specified keys."""
        for key in self.keys:
            if key in sample:
                tensor = sample[key]
                mean = torch.mean(tensor)
                std = torch.std(tensor)
                sample[key] = (tensor - mean) / (std + 1e-8)
                sample[f'{key}_mean'] = mean
                sample[f'{key}_std'] = std
        
        return sample


def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int = 1,
    num_workers: int = 4,
    acceleration: int = 4,
    center_fraction: float = 0.08,
    use_normalization: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        acceleration: Acceleration factor
        center_fraction: Center fraction for masks
        use_normalization: Whether to apply normalization transform
        **kwargs: Additional arguments for dataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Setup transforms
    transform = NormalizationTransform() if use_normalization else None
    
    # Create datasets
    train_dataset = FastMRIDataset(
        data_path=train_path,
        acceleration=acceleration,
        center_fraction=center_fraction,
        transform=transform,
        use_seed=True,
        **kwargs
    )
    
    val_dataset = FastMRIDataset(
        data_path=val_path,
        acceleration=acceleration,
        center_fraction=center_fraction,
        transform=transform,
        use_seed=True,
        **kwargs
    )
    
    test_dataset = FastMRIDataset(
        data_path=test_path,
        acceleration=acceleration,
        center_fraction=center_fraction,
        transform=transform,
        use_seed=True,
        **kwargs
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader