# File: src/neural_operator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math
from einops import rearrange


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution layer for Neural Operators.
    
    This layer performs convolution in the Fourier domain by learning
    spectral weights for a subset of Fourier modes.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        activation: str = "gelu"
    ):
        """
        Initialize spectral convolution layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes in first dimension
            modes2: Number of Fourier modes in second dimension
            activation: Activation function to use
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        # Initialize spectral weights
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        
        # Activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = F.gelu
    
    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication in Fourier space.
        
        Args:
            input: Input tensor in Fourier domain
            weights: Spectral weights
            
        Returns:
            Result of complex multiplication
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spectral convolution.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Output tensor after spectral convolution
        """
        batch_size = x.shape[0]
        
        # Store original dtype for mixed precision compatibility
        original_dtype = x.dtype
        
        # Apply FFT to get frequency domain representation
        # Convert to float32 for complex operations if needed
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        
        x_ft = torch.fft.rfft2(x)
        
        # Initialize output in frequency domain
        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-2), x_ft.size(-1),
            dtype=torch.cfloat, device=x.device  # Always use ComplexFloat
        )
        
        # Apply spectral convolution for selected modes
        # Lower modes
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        
        # Upper modes (exploiting symmetry)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        
        # Apply inverse FFT to get back to spatial domain
        x_out = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        # Convert back to original dtype if needed
        if original_dtype == torch.float16:
            x_out = x_out.to(torch.float16)
        
        return x_out


class FeedForward(nn.Module):
    """Feed-forward network for neural operator."""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu"
    ):
        """
        Initialize feed-forward network.
        
        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (defaults to 4*dim)
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        hidden_dim = hidden_dim or 4 * dim
        
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == "gelu":
            return nn.GELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            return nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class NeuralOperatorBlock(nn.Module):
    """
    Single block of the Neural Operator.
    
    Combines spectral convolution with feed-forward network and residual connections.
    """
    
    def __init__(
        self,
        dim: int,
        modes1: int,
        modes2: int,
        dropout: float = 0.0,
        activation: str = "gelu"
    ):
        """
        Initialize neural operator block.
        
        Args:
            dim: Channel dimension
            modes1: Number of modes in first spatial dimension
            modes2: Number of modes in second spatial dimension
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        self.spectral_conv = SpectralConv2d(dim, dim, modes1, modes2, activation)
        self.conv = nn.Conv2d(dim, dim, 1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout, activation=activation)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = F.gelu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of neural operator block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Spectral convolution with residual connection
        x1 = self.spectral_conv(x)
        x2 = self.conv(x)
        x = x + self.dropout(self.activation(x1 + x2))
        
        # Apply layer norm (need to permute for LayerNorm)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm1(x)
        x = rearrange(x, 'b h w c -> b c h w')
        
        # Feed-forward network with residual connection
        x = x + self.dropout(self.ffn(x))
        
        # Apply layer norm again
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm2(x)
        x = rearrange(x, 'b h w c -> b c h w')
        
        return x


class DISCONeuralOperator(nn.Module):
    """
    DISCO (Deep Image Structure and Correlation Optimization) Neural Operator
    for MRI reconstruction.
    
    This is a U-shaped neural operator that processes k-space and image domain
    data for MRI reconstruction tasks.
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_channels: int = 64,
        num_layers: int = 4,
        modes: int = 12,
        width: int = 64,
        dropout: float = 0.1,
        use_residual: bool = True,
        activation: str = "gelu"
    ):
        """
        Initialize DISCO Neural Operator.
        
        Args:
            in_channels: Number of input channels (2 for complex: real, imag)
            out_channels: Number of output channels
            hidden_channels: Number of hidden channels
            num_layers: Number of neural operator layers
            modes: Number of Fourier modes to use
            width: Width multiplier for hidden dimensions
            dropout: Dropout rate
            use_residual: Whether to use residual connections
            activation: Activation function
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.modes = modes
        self.use_residual = use_residual
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, hidden_channels, 1)
        
        # Calculate number of downsampling levels
        self.num_down = num_layers // 2
        
        # Encoder layers (downsampling path)
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i in range(self.num_down):
            current_channels = hidden_channels * (2 ** i)
            
            self.encoder_layers.append(
                NeuralOperatorBlock(
                    dim=current_channels,
                    modes1=modes,
                    modes2=modes,
                    dropout=dropout,
                    activation=activation
                )
            )
            
            # Add downsampling layer except for the last encoder
            if i < self.num_down - 1:
                next_channels = hidden_channels * (2 ** (i + 1))
                self.downsample_layers.append(
                    nn.Conv2d(current_channels, next_channels, kernel_size=3, stride=2, padding=1)
                )
        
        # Bottleneck
        bottleneck_channels = hidden_channels * (2 ** (self.num_down - 1))
        self.bottleneck = NeuralOperatorBlock(
            dim=bottleneck_channels,
            modes1=modes,
            modes2=modes,
            dropout=dropout,
            activation=activation
        )
        
        # Decoder layers (upsampling path)
        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for i in range(self.num_down - 1, -1, -1):
            current_channels = hidden_channels * (2 ** i)
            
            # Upsampling layer
            if i == self.num_down - 1:
                # First upsampling from bottleneck
                upsample_in = bottleneck_channels
            else:
                # Subsequent upsampling
                upsample_in = hidden_channels * (2 ** (i + 1))
            
            self.upsample_layers.append(
                nn.ConvTranspose2d(
                    upsample_in, current_channels,
                    kernel_size=3, stride=2, padding=1, output_padding=1
                )
            )
            
            # Skip connection processing
            self.skip_convs.append(
                nn.Conv2d(current_channels * 2, current_channels, 1)  # Reduce concatenated channels
            )
            
            # Decoder block
            self.decoder_layers.append(
                NeuralOperatorBlock(
                    dim=current_channels,
                    modes1=modes,
                    modes2=modes,
                    dropout=dropout,
                    activation=activation
                )
            )
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_channels, out_channels, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DISCO Neural Operator.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Reconstructed output tensor
        """
        # Store input for residual connection
        input_x = x
        input_size = x.shape[-2:]
        
        # Input projection
        x = self.input_proj(x)
        
        # Encoder path with skip connections
        skip_connections = []
        
        for i, encoder_layer in enumerate(self.encoder_layers):
            x = encoder_layer(x)
            skip_connections.append(x)
            
            # Downsampling (except for last layer)
            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i, (upsample, skip_conv, decoder_layer) in enumerate(zip(
            self.upsample_layers, self.skip_convs, self.decoder_layers
        )):
            # Upsampling
            x = upsample(x)
            
            # Get corresponding skip connection
            skip_idx = len(skip_connections) - 1 - i
            skip = skip_connections[skip_idx]
            
            # Ensure spatial dimensions match
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
            # Concatenate and reduce channels
            x = torch.cat([x, skip], dim=1)
            x = skip_conv(x)
            
            # Apply decoder block
            x = decoder_layer(x)
        
        # Ensure output matches input size
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        # Output projection
        x = self.output_proj(x)
        
        # Residual connection with input
        if self.use_residual and x.shape == input_x.shape:
            x = x + input_x
        
        return x

    def _setup_model(self):
        # ...existing code...
        if self.model_has_complex_params():
            self.scaler = None  # Disable mixed precision
        elif self.config['training']['mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def model_has_complex_params(self):
        # Returns True if any parameter is complex
        return any(p.is_complex() for p in self.model.parameters())

def train_one_epoch(trainer):
    # ...existing code...
    if any(p.is_complex() for p in trainer.model.parameters()):
        assert trainer.scaler is None
        return  # Mixed precision is not supported for complex models
    else:
        assert trainer.scaler is not None
        # Continue with training epoch and checks