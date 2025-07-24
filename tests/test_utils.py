# File: tests/test_utils.py

import pytest
import torch
import numpy as np
import tempfile
import os
import yaml
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    load_config, set_seed, get_device, complex_to_real, real_to_complex,
    fft2c, ifft2c, apply_mask, root_sum_of_squares, normalize_tensor,
    denormalize_tensor, count_parameters, create_directory
)


class TestUtils:
    
    def test_load_config(self):
        """Test configuration loading."""
        # Create temporary config file
        config_data = {
            'model': {'name': 'test_model'},
            'training': {'epochs': 10}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            loaded_config = load_config(config_path)
            assert loaded_config['model']['name'] == 'test_model'
            assert loaded_config['training']['epochs'] == 10
        finally:
            os.unlink(config_path)
    
    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        set_seed(42)
        x1 = torch.randn(10)
        
        set_seed(42)
        x2 = torch.randn(10)
        
        assert torch.allclose(x1, x2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("cpu")
        assert device.type == "cpu"
        
        device_auto = get_device("auto")
        assert device_auto.type in ["cpu", "cuda"]
    
    def test_complex_to_real_conversion(self):
        """Test complex to real tensor conversion."""
        # Create complex tensor
        real_part = torch.randn(2, 3, 4)
        imag_part = torch.randn(2, 3, 4)
        complex_tensor = torch.complex(real_part, imag_part)
        
        # Convert to real
        real_tensor = complex_to_real(complex_tensor)
        
        # Check shape and values
        assert real_tensor.shape == (2, 2, 3, 4)
        assert torch.allclose(real_tensor[:, 0, :, :], real_part)
        assert torch.allclose(real_tensor[:, 1, :, :], imag_part)
        
        # Test with already real tensor
        already_real = torch.randn(2, 2, 3, 4)
        real_output = complex_to_real(already_real)
        assert torch.allclose(real_output, already_real)
    
    def test_real_to_complex_conversion(self):
        """Test real to complex tensor conversion."""
        # Create real tensor with 2 channels
        real_tensor = torch.randn(2, 2, 3, 4)
        
        # Convert to complex
        complex_tensor = real_to_complex(real_tensor)
        
        # Check shape and round-trip conversion
        assert complex_tensor.shape == (2, 3, 4)
        assert torch.allclose(complex_tensor.real, real_tensor[:, 0, :, :])
        assert torch.allclose(complex_tensor.imag, real_tensor[:, 1, :, :])
        
        # Test with already complex tensor
        already_complex = torch.complex(torch.randn(2, 3, 4), torch.randn(2, 3, 4))
        complex_output = real_to_complex(already_complex)
        assert torch.allclose(complex_output, already_complex)
        
        # Test error case
        with pytest.raises(ValueError):
            real_to_complex(torch.randn(2, 3, 3, 4))  # Wrong channel dimension
    
    def test_fft_ifft_consistency(self):
        """Test FFT and IFFT are inverse operations."""
        x = torch.randn(10, 10, dtype=torch.complex64)
        
        # Forward and inverse should recover original
        x_recovered = ifft2c(fft2c(x))
        
        assert torch.allclose(x, x_recovered, atol=1e-6)
    
    def test_apply_mask(self):
        """Test mask application."""
        kspace = torch.randn(5, 5, dtype=torch.complex64)
        mask = torch.zeros(5, 5)
        mask[2, :] = 1  # Only keep center line
        
        masked_kspace = apply_mask(kspace, mask)
        
        # Check that only center line is preserved
        assert torch.allclose(masked_kspace[2, :], kspace[2, :])
        assert torch.allclose(masked_kspace[0, :], torch.zeros(5, dtype=torch.complex64))
    
    def test_root_sum_of_squares(self):
        """Test RSS computation."""
        x = torch.tensor([[[3.0, 4.0]], [[0.0, 0.0]]])  # Shape: (2, 1, 2)
        
        rss = root_sum_of_squares(x, dim=0)
        
        expected = torch.sqrt(torch.tensor([[[9.0, 16.0]]]))  # sqrt(3^2 + 0^2), sqrt(4^2 + 0^2)
        assert torch.allclose(rss, expected)
    
    def test_normalize_denormalize(self):
        """Test tensor normalization and denormalization."""
        x = torch.randn(100, 100) * 5 + 10  # Mean ~10, std ~5
        
        normalized, mean, std = normalize_tensor(x)
        denormalized = denormalize_tensor(normalized, mean, std)
        
        # Check normalization properties
        assert torch.abs(torch.mean(normalized)) < 1e-6
        assert torch.abs(torch.std(normalized) - 1.0) < 1e-6
        
        # Check round-trip
        assert torch.allclose(x, denormalized, atol=1e-6)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)  # 10*5 + 5 = 55 parameters
        
        param_count = count_parameters(model)
        
        assert param_count == 55
    
    def test_create_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "test", "nested", "directory")
            
            create_directory(test_path)
            
            assert os.path.exists(test_path)
            assert os.path.isdir(test_path)


if __name__ == "__main__":
    pytest.main([__file__])