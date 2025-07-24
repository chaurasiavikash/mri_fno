# File: tests/test_model.py

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import MRIReconstructionModel, DataConsistencyLayer, ReconstructionLoss
from utils import complex_to_real


class TestMRIReconstructionModel:
    
    @pytest.fixture
    def model_config(self):
        """Default model configuration for testing."""
        return {
            'in_channels': 2,
            'out_channels': 2,
            'hidden_channels': 16,
            'num_layers': 2,
            'modes': 4,
            'dropout': 0.1,
            'use_residual': True,
            'activation': 'gelu'
        }
    
    def test_model_initialization(self, model_config):
        """Test model initialization."""
        model = MRIReconstructionModel(
            neural_operator_config=model_config,
            use_data_consistency=True,
            dc_weight=1.0,
            num_dc_iterations=3
        )
        
        assert model.use_data_consistency == True
        assert model.dc_weight == 1.0
        assert model.num_dc_iterations == 3
        assert len(model.dc_layers) == 3
    
    def test_model_forward_pass(self, model_config):
        """Test model forward pass."""
        model = MRIReconstructionModel(
            neural_operator_config=model_config,
            use_data_consistency=True,
            num_dc_iterations=2
        )
        
        batch_size, num_coils, height, width = 2, 8, 64, 64
        
        # Create synthetic input data
        kspace_masked = torch.randn(batch_size, num_coils, 2, height, width)
        mask = torch.randint(0, 2, (height, width)).float()
        
        # Forward pass
        output = model(kspace_masked, mask)
        
        # Check output structure
        assert 'output' in output
        assert 'kspace_pred' in output
        assert 'image_complex' in output
        assert 'intermediate' in output
        
        # Check output shapes
        assert output['output'].shape == (batch_size, height, width)
        assert output['kspace_pred'].shape == kspace_masked.shape
        assert output['image_complex'].shape == (batch_size, height, width)
    
    def test_model_without_data_consistency(self, model_config):
        """Test model without data consistency layers."""
        model = MRIReconstructionModel(
            neural_operator_config=model_config,
            use_data_consistency=False
        )
        
        assert model.use_data_consistency == False
        assert not hasattr(model, 'dc_layers') or len(model.dc_layers) == 0
        
        # Test forward pass
        batch_size, num_coils, height, width = 1, 4, 32, 32
        kspace_masked = torch.randn(batch_size, num_coils, 2, height, width)
        mask = torch.randint(0, 2, (height, width)).float()
        
        output = model(kspace_masked, mask)
        assert output['output'].shape == (batch_size, height, width)
    
    def test_mask_broadcasting(self, model_config):
        """Test different mask shapes and broadcasting."""
        model = MRIReconstructionModel(neural_operator_config=model_config)
        
        batch_size, num_coils, height, width = 2, 4, 32, 32
        kspace_masked = torch.randn(batch_size, num_coils, 2, height, width)
        
        # Test different mask shapes
        mask_shapes = [
            (height, width),  # 2D mask
            (batch_size, height, width),  # 3D mask with batch
            (batch_size, 1, height, width)  # 4D mask
        ]
        
        for mask_shape in mask_shapes:
            mask = torch.randint(0, 2, mask_shape).float()
            output = model(kspace_masked, mask)
            assert output['output'].shape == (batch_size, height, width)
    
    def test_coil_combination(self, model_config):
        """Test coil combination functionality."""
        model = MRIReconstructionModel(neural_operator_config=model_config)
        
        # Test different coil image formats
        batch_size, num_coils, height, width = 1, 8, 32, 32
        
        # Complex format
        coil_images_complex = torch.randn(batch_size, num_coils, height, width, dtype=torch.complex64)
        combined_complex = model._combine_coils(coil_images_complex)
        assert combined_complex.shape == (batch_size, height, width)
        assert torch.all(combined_complex >= 0)  # RSS should be non-negative
        
        # Real/imaginary format
        coil_images_real = torch.randn(batch_size, num_coils, 2, height, width)
        combined_real = model._combine_coils(coil_images_real)
        assert combined_real.shape == (batch_size, height, width)
        assert torch.all(combined_real >= 0)


class TestDataConsistencyLayer:
    
    def test_initialization(self):
        """Test DataConsistencyLayer initialization."""
        layer = DataConsistencyLayer(weight=0.5)
        assert layer.weight == 0.5
    
    def test_forward_pass(self):
        """Test DataConsistencyLayer forward pass."""
        layer = DataConsistencyLayer(weight=1.0)
        
        batch_size, num_coils, height, width = 2, 4, 32, 32
        
        # Create test data
        predicted_kspace = torch.randn(batch_size, num_coils, 2, height, width)
        measured_kspace = torch.randn(batch_size, num_coils, 2, height, width)
        mask = torch.randint(0, 2, (height, width)).float()
        
        # Apply data consistency
        output = layer(predicted_kspace, measured_kspace, mask)
        
        assert output.shape == predicted_kspace.shape
    
    def test_data_consistency_enforcement(self):
        """Test that data consistency properly enforces measured data."""
        layer = DataConsistencyLayer(weight=1.0)
        
        batch_size, num_coils, height, width = 1, 2, 4, 4
        
        # Create known data
        predicted_kspace = torch.ones(batch_size, num_coils, 2, height, width)
        measured_kspace = torch.zeros(batch_size, num_coils, 2, height, width)
        
        # Create mask that samples half the k-space
        mask = torch.zeros(height, width)
        mask[:, :width//2] = 1  # Sample left half
        
        output = layer(predicted_kspace, measured_kspace, mask)
        
        # Check that sampled regions match measured data
        sampled_regions = output[:, :, :, :, :width//2]
        expected_sampled = measured_kspace[:, :, :, :, :width//2]
        
        assert torch.allclose(sampled_regions, expected_sampled, atol=1e-6)
    
    def test_mask_broadcasting(self):
        """Test mask broadcasting for different shapes."""
        layer = DataConsistencyLayer(weight=1.0)
        
        batch_size, num_coils, height, width = 2, 4, 8, 8
        predicted_kspace = torch.randn(batch_size, num_coils, 2, height, width)
        measured_kspace = torch.randn(batch_size, num_coils, 2, height, width)
        
        # Test different mask shapes
        mask_shapes = [
            (height, width),
            (batch_size, height, width),
            (1, height, width)
        ]
        
        for mask_shape in mask_shapes:
            mask = torch.randint(0, 2, mask_shape).float()
            output = layer(predicted_kspace, measured_kspace, mask)
            assert output.shape == predicted_kspace.shape


class TestReconstructionLoss:
    
    def test_initialization(self):
        """Test ReconstructionLoss initialization."""
        loss_fn = ReconstructionLoss(
            l1_weight=1.0,
            ssim_weight=0.1,
            perceptual_weight=0.01,
            data_consistency_weight=1.0
        )
        
        assert loss_fn.l1_weight == 1.0
        assert loss_fn.ssim_weight == 0.1
        assert loss_fn.perceptual_weight == 0.01
        assert loss_fn.data_consistency_weight == 1.0
    
    def test_basic_loss_computation(self):
        """Test basic loss computation."""
        loss_fn = ReconstructionLoss(
            l1_weight=1.0,
            ssim_weight=0.1,
            data_consistency_weight=0.0  # Disable for simple test
        )
        
        batch_size, height, width = 2, 32, 32
        prediction = torch.randn(batch_size, height, width)
        target = torch.randn(batch_size, height, width)
        
        losses = loss_fn(prediction, target)
        
        # Check that required loss components are present
        assert 'l1' in losses
        assert 'ssim' in losses
        assert 'total' in losses
        
        # Check that losses are scalars
        for loss_name, loss_value in losses.items():
            assert loss_value.dim() == 0  # Scalar
            assert loss_value >= 0  # Non-negative
    
    def test_data_consistency_loss(self):
        """Test data consistency loss computation."""
        loss_fn = ReconstructionLoss(
            l1_weight=0.0,
            ssim_weight=0.0,
            data_consistency_weight=1.0
        )
        
        batch_size, num_coils, height, width = 1, 4, 16, 16
        prediction = torch.randn(batch_size, height, width)
        target = torch.randn(batch_size, height, width)
        
        # Create k-space data
        predicted_kspace = torch.randn(batch_size, num_coils, 2, height, width)
        target_kspace = torch.randn(batch_size, num_coils, 2, height, width)
        mask = torch.randint(0, 2, (height, width)).float()
        
        losses = loss_fn(
            prediction, target,
            predicted_kspace=predicted_kspace,
            target_kspace=target_kspace,
            mask=mask
        )
        
        assert 'data_consistency' in losses
        assert 'total' in losses
        assert losses['data_consistency'] >= 0
    
    def test_ssim_loss_computation(self):
        """Test SSIM loss computation."""
        loss_fn = ReconstructionLoss(
            l1_weight=0.0,
            ssim_weight=1.0,
            data_consistency_weight=0.0
        )
        
        # Create identical images (should have low SSIM loss)
        batch_size, height, width = 1, 32, 32
        image = torch.randn(batch_size, height, width)
        
        losses_identical = loss_fn(image, image)
        
        # Create different images (should have higher SSIM loss)
        different_image = torch.randn(batch_size, height, width)
        losses_different = loss_fn(image, different_image)
        
        # SSIM loss should be lower for identical images
        assert losses_identical['ssim'] < losses_different['ssim']
    
    def test_loss_weighting(self):
        """Test that loss weights are properly applied."""
        # Create two loss functions with different weights
        loss_fn_1 = ReconstructionLoss(l1_weight=1.0, ssim_weight=0.0)
        loss_fn_2 = ReconstructionLoss(l1_weight=2.0, ssim_weight=0.0)
        
        batch_size, height, width = 1, 16, 16
        prediction = torch.randn(batch_size, height, width)
        target = torch.randn(batch_size, height, width)
        
        losses_1 = loss_fn_1(prediction, target)
        losses_2 = loss_fn_2(prediction, target)
        
        # L1 loss should be doubled in second case
        assert torch.allclose(losses_2['l1'], 2 * losses_1['l1'], atol=1e-6)
    
    def test_loss_components_disable(self):
        """Test disabling loss components by setting weight to 0."""
        loss_fn = ReconstructionLoss(
            l1_weight=1.0,
            ssim_weight=0.0,  # Disabled
            data_consistency_weight=0.0  # Disabled
        )
        
        batch_size, height, width = 1, 16, 16
        prediction = torch.randn(batch_size, height, width)
        target = torch.randn(batch_size, height, width)
        
        losses = loss_fn(prediction, target)
        
        # Should only have L1 loss and total
        assert 'l1' in losses
        assert 'total' in losses
        assert 'ssim' not in losses or losses['ssim'] == 0
        assert 'data_consistency' not in losses
        
        # Total should equal L1 loss
        assert torch.allclose(losses['total'], losses['l1'])


class TestModelIntegration:
    """Integration tests for the complete model pipeline."""
    
    @pytest.fixture
    def model_config(self):
        """Small model configuration for integration testing."""
        return {
            'in_channels': 2,
            'out_channels': 2,
            'hidden_channels': 8,
            'num_layers': 2,
            'modes': 2,
            'dropout': 0.0,
            'use_residual': False,
            'activation': 'relu'
        }
    
    def test_full_training_step(self, model_config):
        """Test a complete training step with model and loss."""
        # Initialize model and loss
        model = MRIReconstructionModel(
            neural_operator_config=model_config,
            use_data_consistency=True,
            num_dc_iterations=1
        )
        loss_fn = ReconstructionLoss(
            l1_weight=1.0,
            ssim_weight=0.1,
            data_consistency_weight=0.5
        )
        
        # Create synthetic training data
        batch_size, num_coils, height, width = 1, 4, 32, 32
        kspace_full = torch.randn(batch_size, num_coils, 2, height, width)
        kspace_masked = torch.randn(batch_size, num_coils, 2, height, width)
        mask = torch.randint(0, 2, (height, width)).float()
        target_image = torch.randn(batch_size, height, width)
        
        # Forward pass
        output = model(kspace_masked, mask, kspace_full)
        
        # Compute loss
        losses = loss_fn(
            prediction=output['output'],
            target=target_image,
            predicted_kspace=output['kspace_pred'],
            target_kspace=kspace_full,
            mask=mask
        )
        
        # Check that we can backpropagate
        total_loss = losses['total']
        total_loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_eval_mode(self, model_config):
        """Test model in evaluation mode."""
        model = MRIReconstructionModel(
            neural_operator_config=model_config,
            use_data_consistency=False
        )
        
        # Set to eval mode
        model.eval()
        
        # Test forward pass
        batch_size, num_coils, height, width = 1, 2, 16, 16
        kspace_masked = torch.randn(batch_size, num_coils, 2, height, width)
        mask = torch.randint(0, 2, (height, width)).float()
        
        with torch.no_grad():
            output = model(kspace_masked, mask)
        
        assert output['output'].shape == (batch_size, height, width)
    
    def test_model_device_transfer(self, model_config):
        """Test transferring model to different devices."""
        model = MRIReconstructionModel(neural_operator_config=model_config)
        
        # Test CPU
        batch_size, num_coils, height, width = 1, 2, 8, 8
        kspace_cpu = torch.randn(batch_size, num_coils, 2, height, width)
        mask_cpu = torch.randint(0, 2, (height, width)).float()
        
        output_cpu = model(kspace_cpu, mask_cpu)
        assert output_cpu['output'].device == kspace_cpu.device
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            kspace_cuda = kspace_cpu.cuda()
            mask_cuda = mask_cpu.cuda()
            
            output_cuda = model_cuda(kspace_cuda, mask_cuda)
            assert output_cuda['output'].device == kspace_cuda.device


if __name__ == "__main__":
    pytest.main([__file__])