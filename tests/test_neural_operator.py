# File: tests/test_neural_operator.py

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_operator import (
    SpectralConv2d, FeedForward, NeuralOperatorBlock, DISCONeuralOperator
)


class TestSpectralConv2d:
    
    def test_initialization(self):
        """Test SpectralConv2d initialization."""
        layer = SpectralConv2d(
            in_channels=64,
            out_channels=128,
            modes1=12,
            modes2=12,
            activation="gelu"
        )
        
        assert layer.in_channels == 64
        assert layer.out_channels == 128
        assert layer.modes1 == 12
        assert layer.modes2 == 12
        assert layer.weights1.shape == (64, 128, 12, 12)
        assert layer.weights2.shape == (64, 128, 12, 12)
        assert layer.weights1.dtype == torch.cfloat
        assert layer.weights2.dtype == torch.cfloat
    
    def test_forward_pass(self):
        """Test SpectralConv2d forward pass."""
        batch_size, in_channels, height, width = 2, 32, 64, 64
        modes = 8
        
        layer = SpectralConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            modes1=modes,
            modes2=modes
        )
        
        x = torch.randn(batch_size, in_channels, height, width)
        output = layer(x)
        
        assert output.shape == (batch_size, in_channels, height, width)
        assert output.dtype == torch.float32
    
    def test_complex_multiplication(self):
        """Test complex multiplication function."""
        layer = SpectralConv2d(4, 8, 6, 6)
        
        input_tensor = torch.randn(2, 4, 6, 6, dtype=torch.cfloat)
        weights = torch.randn(4, 8, 6, 6, dtype=torch.cfloat)
        
        result = layer.compl_mul2d(input_tensor, weights)
        
        assert result.shape == (2, 8, 6, 6)
        assert result.dtype == torch.cfloat


class TestFeedForward:
    
    def test_initialization(self):
        """Test FeedForward initialization."""
        ffn = FeedForward(dim=64, hidden_dim=256, dropout=0.1, activation="gelu")
        
        # Check that network has correct structure
        assert len(ffn.net) == 5  # Conv -> Activation -> Dropout -> Conv -> Dropout
        assert isinstance(ffn.net[0], nn.Conv2d)
        assert isinstance(ffn.net[3], nn.Conv2d)
    
    def test_forward_pass(self):
        """Test FeedForward forward pass."""
        batch_size, channels, height, width = 2, 64, 32, 32
        
        ffn = FeedForward(dim=channels, dropout=0.1)
        x = torch.randn(batch_size, channels, height, width)
        output = ffn(x)
        
        assert output.shape == x.shape
    
    def test_different_activations(self):
        """Test different activation functions."""
        activations = ["gelu", "relu", "swish"]
        
        for activation in activations:
            ffn = FeedForward(dim=32, activation=activation)
            x = torch.randn(1, 32, 16, 16)
            output = ffn(x)
            assert output.shape == x.shape


class TestNeuralOperatorBlock:
    
    def test_initialization(self):
        """Test NeuralOperatorBlock initialization."""
        block = NeuralOperatorBlock(
            dim=64,
            modes1=12,
            modes2=12,
            dropout=0.1,
            activation="gelu"
        )
        
        assert isinstance(block.spectral_conv, SpectralConv2d)
        assert isinstance(block.conv, nn.Conv2d)
        assert isinstance(block.norm1, nn.LayerNorm)
        assert isinstance(block.norm2, nn.LayerNorm)
        assert isinstance(block.ffn, FeedForward)
    
    def test_forward_pass(self):
        """Test NeuralOperatorBlock forward pass."""
        batch_size, channels, height, width = 2, 64, 32, 32
        
        block = NeuralOperatorBlock(
            dim=channels,
            modes1=8,
            modes2=8,
            dropout=0.1
        )
        
        x = torch.randn(batch_size, channels, height, width)
        output = block(x)
        
        assert output.shape == x.shape
    
    def test_residual_connection(self):
        """Test that residual connections work properly."""
        batch_size, channels, height, width = 1, 32, 16, 16
        
        block = NeuralOperatorBlock(
            dim=channels,
            modes1=4,
            modes2=4,
            dropout=0.0  # No dropout for testing
        )
        
        x = torch.randn(batch_size, channels, height, width)
        
        # Set block to eval mode to disable dropout
        block.eval()
        output = block(x)
        
        # Output should be different from input due to transformations
        assert not torch.allclose(output, x, atol=1e-6)
        assert output.shape == x.shape


class TestDISCONeuralOperator:
    
    def test_initialization(self):
        """Test DISCONeuralOperator initialization."""
        model = DISCONeuralOperator(
            in_channels=2,
            out_channels=2,
            hidden_channels=64,
            num_layers=4,
            modes=12,
            dropout=0.1,
            use_residual=True,
            activation="gelu"
        )
        
        assert model.in_channels == 2
        assert model.out_channels == 2
        assert model.hidden_channels == 64
        assert model.num_layers == 4
        assert model.modes == 12
        assert model.use_residual == True
        
        # Check that layers are properly initialized
        assert isinstance(model.input_proj, nn.Conv2d)
        assert isinstance(model.output_proj, nn.Conv2d)
        assert len(model.encoder_layers) == 2  # num_layers // 2
        assert len(model.decoder_layers) == 2
        assert len(model.downsample_layers) == 1  # num_layers // 2 - 1
        assert len(model.upsample_layers) == 2
    
    def test_forward_pass(self):
        """Test DISCONeuralOperator forward pass."""
        batch_size, in_channels, height, width = 2, 2, 64, 64
        
        model = DISCONeuralOperator(
            in_channels=in_channels,
            out_channels=2,
            hidden_channels=32,
            num_layers=4,
            modes=8,
            dropout=0.1
        )
        
        x = torch.randn(batch_size, in_channels, height, width)
        output = model(x)
        
        assert output.shape == (batch_size, 2, height, width)
    
    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        model = DISCONeuralOperator(
            in_channels=2,
            out_channels=2,
            hidden_channels=32,
            num_layers=4,
            modes=6
        )
        
        # Test different sizes
        sizes = [(32, 32), (64, 64), (128, 128)]
        
        for height, width in sizes:
            x = torch.randn(1, 2, height, width)
            output = model(x)
            assert output.shape == (1, 2, height, width)
    
    def test_residual_connection(self):
        """Test residual connection functionality."""
        model_with_residual = DISCONeuralOperator(
            in_channels=2,
            out_channels=2,
            hidden_channels=16,
            num_layers=2,
            modes=4,
            use_residual=True,
            dropout=0.0
        )
        
        model_without_residual = DISCONeuralOperator(
            in_channels=2,
            out_channels=2,
            hidden_channels=16,
            num_layers=2,
            modes=4,
            use_residual=False,
            dropout=0.0
        )
        
        x = torch.randn(1, 2, 32, 32)
        
        # Set models to eval mode
        model_with_residual.eval()
        model_without_residual.eval()
        
        output_with = model_with_residual(x)
        output_without = model_without_residual(x)
        
        # Outputs should be different
        assert not torch.allclose(output_with, output_without, atol=1e-6)
        assert output_with.shape == output_without.shape == x.shape
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = DISCONeuralOperator(
            in_channels=2,
            out_channels=2,
            hidden_channels=16,
            num_layers=2,
            modes=4
        )
        
        x = torch.randn(1, 2, 32, 32, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_parameters_count(self):
        """Test parameter counting."""
        model = DISCONeuralOperator(
            in_channels=2,
            out_channels=2,
            hidden_channels=32,
            num_layers=4,
            modes=8
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All params should be trainable by default
        assert total_params > 1000  # Should have a reasonable number of parameters
    
    def test_different_modes(self):
        """Test model with different numbers of Fourier modes."""
        modes_list = [4, 8, 12, 16]
        
        for modes in modes_list:
            model = DISCONeuralOperator(
                in_channels=2,
                out_channels=2,
                hidden_channels=16,
                num_layers=2,
                modes=modes
            )
            
            x = torch.randn(1, 2, 32, 32)
            output = model(x)
            assert output.shape == x.shape
    
    def test_model_device_compatibility(self):
        """Test model works on different devices."""
        model = DISCONeuralOperator(
            in_channels=2,
            out_channels=2,
            hidden_channels=16,
            num_layers=2,
            modes=4
        )
        
        # Test CPU
        x_cpu = torch.randn(1, 2, 32, 32)
        output_cpu = model(x_cpu)
        assert output_cpu.device == x_cpu.device
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            x_cuda = x_cpu.cuda()
            output_cuda = model_cuda(x_cuda)
            assert output_cuda.device == x_cuda.device


if __name__ == "__main__":
    pytest.main([__file__])