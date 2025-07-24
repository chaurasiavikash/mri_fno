# File: tests/test_data_loader.py

import pytest
import torch
import numpy as np
import tempfile
import h5py
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import FastMRIDataset, NormalizationTransform, create_data_loaders


class TestFastMRIDataset:
    
    @pytest.fixture
    def mock_data_dir(self):
        """Create mock data directory with synthetic .h5 files."""
        temp_dir = tempfile.mkdtemp()
        
        # Create synthetic multi-coil k-space data
        num_files = 3
        num_slices = 5
        num_coils = 8
        height, width = 320, 320
        
        for file_idx in range(num_files):
            file_path = os.path.join(temp_dir, f"test_file_{file_idx:03d}.h5")
            
            with h5py.File(file_path, 'w') as f:
                # Create complex k-space data
                kspace_real = np.random.randn(num_slices, num_coils, height, width).astype(np.float32)
                kspace_imag = np.random.randn(num_slices, num_coils, height, width).astype(np.float32)
                
                # Store as complex array (h5py format)
                kspace_complex = kspace_real + 1j * kspace_imag
                f.create_dataset('kspace', data=kspace_complex)
        
        yield temp_dir
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_dataset_initialization(self, mock_data_dir):
        """Test dataset initialization."""
        dataset = FastMRIDataset(
            data_path=mock_data_dir,
            acceleration=4,
            center_fraction=0.08
        )
        
        assert len(dataset) > 0
        assert dataset.acceleration == 4
        assert dataset.center_fraction == 0.08
    
    def test_dataset_getitem(self, mock_data_dir):
        """Test getting items from dataset."""
        dataset = FastMRIDataset(
            data_path=mock_data_dir,
            acceleration=4,
            center_fraction=0.08
        )
        
        sample = dataset[0]
        
        # Check required keys
        required_keys = ['kspace_full', 'kspace_masked', 'mask', 'target', 'image_masked', 'metadata']
        for key in required_keys:
            assert key in sample
        
        # Check tensor shapes and types
        assert isinstance(sample['kspace_full'], torch.Tensor)
        assert isinstance(sample['kspace_masked'], torch.Tensor)
        assert isinstance(sample['mask'], torch.Tensor)
        assert isinstance(sample['target'], torch.Tensor)
        assert isinstance(sample['image_masked'], torch.Tensor)
        
        # Check that k-space has real/imaginary channels
        assert sample['kspace_full'].shape[-1] == 2  # Real and imaginary channels
        assert sample['kspace_masked'].shape[-1] == 2
        
        # Check mask properties
        mask = sample['mask']
        assert torch.all((mask == 0) | (mask == 1))  # Binary mask
        assert torch.sum(mask) > 0  # Some lines should be sampled
    
    def test_mask_generation(self, mock_data_dir):
        """Test undersampling mask generation."""
        dataset = FastMRIDataset(
            data_path=mock_data_dir,
            acceleration=4,
            center_fraction=0.08,
            use_seed=True,
            seed=42
        )
        
        # Generate mask
        shape = (320, 320)
        mask1 = dataset._generate_mask(shape, seed=0)
        mask2 = dataset._generate_mask(shape, seed=0)  # Same seed
        mask3 = dataset._generate_mask(shape, seed=1)  # Different seed
        
        # Check reproducibility with same seed
        assert torch.allclose(mask1, mask2)
        
        # Check difference with different seed
        assert not torch.allclose(mask1, mask3)
        
        # Check mask properties
        assert mask1.shape == shape
        assert torch.all((mask1 == 0) | (mask1 == 1))
        
        # Check center fraction
        center_lines = int(0.08 * shape[1])
        center_start = shape[1] // 2 - center_lines // 2
        center_end = center_start + center_lines
        assert torch.all(mask1[:, center_start:center_end] == 1)
    
    def test_image_reconstruction(self, mock_data_dir):
        """Test image reconstruction from k-space."""
        dataset = FastMRIDataset(
            data_path=mock_data_dir,
            acceleration=4,
            center_fraction=0.08
        )
        
        # Create synthetic k-space data
        num_coils = 8
        height, width = 64, 64
        kspace = torch.randn(num_coils, height, width, dtype=torch.complex64)
        
        # Reconstruct image
        image = dataset._reconstruct_image(kspace)
        
        # Check output properties
        assert image.shape == (height, width)
        assert torch.all(image >= 0)  # Magnitude should be non-negative
        assert torch.is_floating_point(image)
    
    def test_normalization_transform(self):
        """Test normalization transform."""
        transform = NormalizationTransform(keys=['target', 'image_masked'])
        
        # Create sample with known statistics
        sample = {
            'target': torch.ones(10, 10) * 5,  # Mean=5, std=0
            'image_masked': torch.randn(10, 10) * 2 + 3,  # Mean~3, std~2
            'other_key': torch.randn(5, 5)
        }
        
        transformed = transform(sample)
        
        # Check normalization of target (constant tensor)
        assert 'target_mean' in transformed
        assert 'target_std' in transformed
        assert torch.allclose(transformed['target_mean'], torch.tensor(5.0))
        
        # Check that other keys are unchanged
        assert torch.allclose(transformed['other_key'], sample['other_key'])
    
    def test_empty_directory_error(self):
        """Test error handling for empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="No .h5 files found"):
                FastMRIDataset(data_path=temp_dir)
    
    def test_create_data_loaders(self, mock_data_dir):
        """Test data loader creation."""
        # Create three directories (train, val, test) with same data
        train_dir = mock_data_dir
        val_dir = mock_data_dir 
        test_dir = mock_data_dir
        
        train_loader, val_loader, test_loader = create_data_loaders(
            train_path=train_dir,
            val_path=val_dir,
            test_path=test_dir,
            batch_size=2,
            num_workers=0,  # Use 0 workers for testing
            acceleration=4,
            center_fraction=0.08,
            use_normalization=True
        )
        
        # Check loader properties
        assert train_loader.batch_size == 2
        assert val_loader.batch_size == 2
        assert test_loader.batch_size == 2
        
        # Check that we can iterate through loaders
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))
        
        # Check batch structure
        for batch in [train_batch, val_batch, test_batch]:
            assert 'kspace_full' in batch
            assert 'target' in batch
            assert batch['kspace_full'].shape[0] <= 2  # Batch size


if __name__ == "__main__":
    pytest.main([__file__])