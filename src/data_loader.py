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
        seed: int = 42,
        target_size: Optional[Tuple[int, int]] = None,  # Auto-detect if None
        use_rss_target: bool = True  # Use RSS reconstruction as target
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
            target_size: Target size for all images (height, width). Auto-detect if None
            use_rss_target: Whether to use provided RSS reconstruction as target
        """
        self.data_path = Path(data_path)
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.transform = transform
        self.use_seed = use_seed
        self.seed = seed
        self.use_rss_target = use_rss_target
        
        # Find all .h5 files
        self.files = list(self.data_path.glob("*.h5"))
        if not self.files:
            raise ValueError(f"No .h5 files found in {data_path}")
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Found {len(self.files)} files in {data_path}")
        
        # Auto-detect target size if not provided
        if target_size is None:
            self.target_size = self._detect_target_size()
        else:
            self.target_size = target_size
        
        self.logger.info(f"Using target size: {self.target_size}")
        
        # Cache file metadata
        self._cache_metadata()
    
    def _detect_target_size(self) -> Tuple[int, int]:
        """Auto-detect appropriate target size from data."""
        try:
            with h5py.File(self.files[0], 'r') as f:
                kspace_shape = f['kspace'].shape
                # Use k-space spatial dimensions
                return kspace_shape[-2:]  # (height, width)
        except Exception as e:
            self.logger.warning(f"Could not auto-detect target size: {e}")
            return (640, 368)  # Default fallback
    
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
                            'kspace_shape': kspace.shape[1:],  # (coils, height, width)
                            'has_rss': 'reconstruction_rss' in f
                        })
            except Exception as e:
                self.logger.warning(f"Error reading {file_path}: {e}")
                continue
        
        self.logger.info(f"Total slices: {len(self.file_metadata)}")
    
    def __len__(self) -> int:
        """Return the total number of slices in the dataset."""
        return len(self.file_metadata)
    
    def _resize_to_target_size(self, tensor: torch.Tensor, use_interpolation: bool = False) -> torch.Tensor:
        """
        Resize tensor to target size using appropriate method.
        
        Args:
            tensor: Input tensor with shape (..., height, width)
            use_interpolation: Whether to use interpolation instead of padding
            
        Returns:
            Resized tensor with target size
        """
        target_h, target_w = self.target_size
        current_shape = tensor.shape
        current_h, current_w = current_shape[-2], current_shape[-1]
        
        if current_h == target_h and current_w == target_w:
            return tensor
        
        # Use interpolation for image data that needs resizing
        if use_interpolation or (current_h > target_h or current_w > target_w):
            # Need to add batch dimension for interpolation
            needs_batch_dim = tensor.dim() == 2
            if needs_batch_dim:
                tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            elif tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)  # Add batch dim
            
            if torch.is_complex(tensor):
                # Handle complex tensors by processing real and imaginary parts
                real_part = F.interpolate(tensor.real, size=self.target_size, 
                                        mode='bilinear', align_corners=False)
                imag_part = F.interpolate(tensor.imag, size=self.target_size, 
                                        mode='bilinear', align_corners=False)
                result = torch.complex(real_part, imag_part)
            else:
                result = F.interpolate(tensor, size=self.target_size, 
                                     mode='bilinear', align_corners=False)
            
            # Remove added dimensions
            if needs_batch_dim:
                result = result.squeeze(0).squeeze(0)
            elif tensor.dim() == 4:  # Had 3 dims originally, added 1
                result = result.squeeze(0)
            
            return result
        
        else:
            # Pad to target size (for k-space data)
            pad_h = target_h - current_h
            pad_w = target_w - current_w
            
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            
            return F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), 
                        mode='constant', value=0)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset with consistent tensor shapes.
        """
        metadata = self.file_metadata[idx]
        
        try:
            with h5py.File(metadata['file_path'], 'r') as f:
                # Load k-space data for specific slice
                kspace_full = f['kspace'][metadata['slice_idx']]  # Shape: (coils, height, width)
                
                # Convert to torch tensor
                kspace_full = torch.from_numpy(kspace_full.copy())
                
                # Ensure complex type
                if not torch.is_complex(kspace_full):
                    self.logger.error(f"Expected complex k-space data, got {kspace_full.dtype}")
                    return self._get_dummy_sample()
                
                # Resize k-space to target size (use padding, not interpolation)
                kspace_full = self._resize_to_target_size(kspace_full, use_interpolation=False)
                
                # Generate mask for target size
                mask = self._generate_mask(self.target_size, idx if self.use_seed else None)
                
                # Apply mask to k-space
                kspace_masked = apply_mask(kspace_full, mask)
                
                # Get target image
                if self.use_rss_target and metadata['has_rss']:
                    # Use provided RSS reconstruction
                    target_rss = f['reconstruction_rss'][metadata['slice_idx']]
                    target_image = torch.from_numpy(target_rss.copy())
                    
                    # Resize RSS to match our target size using interpolation
                    target_image = self._resize_to_target_size(target_image, use_interpolation=True)
                else:
                    # Compute RSS from full k-space
                    image_full = self._reconstruct_image(kspace_full)
                    target_image = torch.abs(image_full)
                
                # Reconstruct zero-filled image
                image_masked = self._reconstruct_image(kspace_masked)
                
                # Convert k-space to real format for network
                kspace_full_real = complex_to_real(kspace_full)
                kspace_masked_real = complex_to_real(kspace_masked)
                
                # Ensure all tensors have exactly the target size
                assert kspace_full_real.shape[-2:] == self.target_size, f"kspace_full shape mismatch: {kspace_full_real.shape}"
                assert kspace_masked_real.shape[-2:] == self.target_size, f"kspace_masked shape mismatch: {kspace_masked_real.shape}"
                assert mask.shape == self.target_size, f"mask shape mismatch: {mask.shape}"
                assert target_image.shape == self.target_size, f"target shape mismatch: {target_image.shape}"
                assert torch.abs(image_masked).shape == self.target_size, f"image_masked shape mismatch: {torch.abs(image_masked).shape}"
                
                # Prepare sample
                sample = {
                    'kspace_full': kspace_full_real,
                    'kspace_masked': kspace_masked_real,
                    'mask': mask,
                    'target': target_image,
                    'image_masked': torch.abs(image_masked),
                    'metadata': {
                        'file_path': str(metadata['file_path']),
                        'slice_idx': metadata['slice_idx'],
                        'acceleration': self.acceleration,
                        'center_fraction': self.center_fraction,
                        'original_kspace_shape': metadata['kspace_shape'],
                        'target_shape': self.target_size,
                        'has_rss': metadata['has_rss'],
                        'used_rss_target': self.use_rss_target and metadata['has_rss']
                    }
                }
                
                # Apply transforms if provided
                if self.transform:
                    sample = self.transform(sample)
                
                return sample
                
        except Exception as e:
            self.logger.error(f"Error loading sample {idx}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_dummy_sample()
    
    def _generate_mask(self, shape: Tuple[int, int], seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate random undersampling mask for target shape.
        
        Args:
            shape: Target shape (height, width)
            seed: Optional seed for reproducible masks
            
        Returns:
            Binary mask tensor with exact target shape
        """
        height, width = shape
        
        # Set seed for reproducible masks
        if seed is not None:
            torch.manual_seed(self.seed + seed)
        
        # Calculate number of center lines to keep
        num_center_lines = max(1, int(self.center_fraction * width))
        center_start = width // 2 - num_center_lines // 2
        center_end = center_start + num_center_lines
        
        # Calculate total lines to keep
        total_lines = max(num_center_lines, width // self.acceleration)
        remaining_lines = max(0, total_lines - num_center_lines)
        
        # Create mask
        mask = torch.zeros(width, dtype=torch.float32)
        
        # Always keep center lines
        mask[center_start:center_end] = 1
        
        # Randomly select remaining lines
        if remaining_lines > 0:
            available_indices = list(range(width))
            # Remove center indices
            for i in range(center_start, center_end):
                if i in available_indices:
                    available_indices.remove(i)
            
            if available_indices:
                num_to_select = min(remaining_lines, len(available_indices))
                selected_indices = torch.randperm(len(available_indices))[:num_to_select]
                for idx in selected_indices:
                    mask[available_indices[idx]] = 1
        
        # Expand to full target shape
        mask = mask.unsqueeze(0).expand(height, -1).clone()
        
        # Ensure exact target shape
        assert mask.shape == shape, f"Mask shape {mask.shape} doesn't match target {shape}"
        
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
        coil_images = ifft2c(kspace * 1e6)
        
        # Combine coils using root sum of squares
        combined_image = root_sum_of_squares(coil_images, dim=0).squeeze()
        
        # Ensure 2D output
        if combined_image.dim() > 2:
            combined_image = combined_image.squeeze()
        
        return combined_image
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return a dummy sample with consistent shapes."""
        num_coils = 15  # Typical FastMRI coil count
        height, width = self.target_size
        
        return {
            'kspace_full': torch.zeros(num_coils, 2, height, width, dtype=torch.float32),
            'kspace_masked': torch.zeros(num_coils, 2, height, width, dtype=torch.float32),
            'mask': torch.zeros(height, width, dtype=torch.float32),
            'target': torch.zeros(height, width, dtype=torch.float32),
            'image_masked': torch.zeros(height, width, dtype=torch.float32),
            'metadata': {
                'file_path': 'dummy',
                'slice_idx': 0,
                'acceleration': self.acceleration,
                'center_fraction': self.center_fraction,
                'original_kspace_shape': (num_coils, height, width),
                'target_shape': self.target_size,
                'has_rss': False,
                'used_rss_target': False
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
                
                # Avoid division by zero
                if std > 1e-8:
                    sample[key] = (tensor - mean) / std
                else:
                    sample[key] = tensor - mean
                
                sample[f'{key}_mean'] = mean
                sample[f'{key}_std'] = std
        
        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle potential remaining size mismatches.
    """
    try:
        # Standard collation for most keys
        collated = {}
        
        for key in batch[0].keys():
            if key == 'metadata':
                # Handle metadata separately
                collated[key] = [item[key] for item in batch]
            else:
                # Stack tensors
                tensors = [item[key] for item in batch]
                
                # Check that all tensors have the same shape
                shapes = [t.shape for t in tensors]
                if not all(shape == shapes[0] for shape in shapes):
                    # Log the mismatch for debugging
                    logging.warning(f"Shape mismatch for key {key}: {shapes}")
                    # Try to pad to largest size
                    max_shape = tuple(max(dim) for dim in zip(*shapes))
                    padded_tensors = []
                    for t in tensors:
                        padding_needed = [(0, max_shape[i] - t.shape[i]) for i in range(len(max_shape))]
                        # Flatten padding for F.pad (expects [left, right, top, bottom, ...])
                        pad_values = []
                        for pad in reversed(padding_needed):
                            pad_values.extend(pad)
                        if any(p > 0 for p in pad_values):
                            t = F.pad(t, pad_values)
                        padded_tensors.append(t)
                    tensors = padded_tensors
                
                collated[key] = torch.stack(tensors)
        
        return collated
        
    except Exception as e:
        logging.error(f"Error in collate_fn: {e}")
        # Fallback to default collation
        return torch.utils.data.dataloader.default_collate(batch)


def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int = 1,
    num_workers: int = 4,
    acceleration: int = 4,
    center_fraction: float = 0.08,
    use_normalization: bool = True,
    target_size: Optional[Tuple[int, int]] = None,  # Auto-detect if None
    use_rss_target: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders with consistent tensor shapes.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        acceleration: Acceleration factor
        center_fraction: Center fraction for masks
        use_normalization: Whether to apply normalization transform
        target_size: Target size for all images (height, width). Auto-detect if None
        use_rss_target: Whether to use provided RSS reconstruction as target
        **kwargs: Additional arguments for dataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Setup transforms
    transform = NormalizationTransform() if use_normalization else None
    
    # Create datasets with consistent target size
    train_dataset = FastMRIDataset(
        data_path=train_path,
        acceleration=acceleration,
        center_fraction=center_fraction,
        transform=transform,
        use_seed=True,
        target_size=target_size,
        use_rss_target=use_rss_target,
        **kwargs
    )
    
    val_dataset = FastMRIDataset(
        data_path=val_path,
        acceleration=acceleration,
        center_fraction=center_fraction,
        transform=transform,
        use_seed=True,
        target_size=target_size,
        use_rss_target=use_rss_target,
        **kwargs
    )
    
    test_dataset = FastMRIDataset(
        data_path=test_path,
        acceleration=acceleration,
        center_fraction=center_fraction,
        transform=transform,
        use_seed=True,
        target_size=target_size,
        use_rss_target=use_rss_target,
        **kwargs
    )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader, test_loader