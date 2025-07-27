# File: src/unet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

from utils import complex_to_real, real_to_complex, fft2c, ifft2c, apply_mask


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Traditional U-Net implementation for MRI reconstruction."""
    
    def __init__(self, n_channels=2, n_classes=2, features=64, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor)
        self.up1 = Up(features * 16, features * 8 // factor, bilinear)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear)
        self.up4 = Up(features * 2, features, bilinear)
        self.outc = OutConv(features, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class MRIUNetModel(nn.Module):
    """
    Complete MRI Reconstruction Model using U-Net.
    
    Similar interface to DISCO model for fair comparison.
    """
    
    def __init__(
        self,
        unet_config: Dict[str, Any],
        use_data_consistency: bool = True,
        dc_weight: float = 1.0,
        num_dc_iterations: int = 5
    ):
        """
        Initialize MRI U-Net Model.
        
        Args:
            unet_config: Configuration for U-Net
            use_data_consistency: Whether to use data consistency layers
            dc_weight: Weight for data consistency term
            num_dc_iterations: Number of data consistency iterations
        """
        super().__init__()
        
        self.use_data_consistency = use_data_consistency
        self.dc_weight = dc_weight
        self.num_dc_iterations = num_dc_iterations
        
        # U-Net for image domain reconstruction
        self.unet = UNet(
            n_channels=unet_config['in_channels'],
            n_classes=unet_config['out_channels'],
            features=unet_config['features'],
            bilinear=True
        )
        
        # Data consistency layers (reuse from DISCO model)
        if use_data_consistency:
            from model import DataConsistencyLayer
            self.dc_layers = nn.ModuleList([
                DataConsistencyLayer(dc_weight) for _ in range(num_dc_iterations)
            ])
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def forward(
        self,
        kspace_masked: torch.Tensor,
        mask: torch.Tensor,
        kspace_full: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the U-Net reconstruction model.
        Robustly handles all k-space and coil-combination scenarios, matching MRIReconstructionModel.
        """
        batch_size = kspace_masked.shape[0]
        # Ensure mask has correct dimensions
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        elif mask.dim() == 3 and mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1)

        # Convert to image domain for U-Net processing
        current_image = self._kspace_to_image(kspace_masked)
        # Combine coils robustly (see _combine_coils)
        combined_image = self._combine_coils(current_image)

        # Prepare input for U-Net: always (batch, 2, H, W)
        if combined_image.dim() == 3:
            # (batch, H, W) -> (batch, 2, H, W) with zeros in 2nd channel
            image_input = torch.stack([combined_image, torch.zeros_like(combined_image)], dim=1)
        elif combined_image.dim() == 4 and combined_image.shape[1] == 2:
            # Already (batch, 2, H, W)
            image_input = combined_image
        else:
            raise ValueError(f"Unexpected combined_image shape for U-Net input: {combined_image.shape}")

        # U-Net prediction
        predicted_image_channels = self.unet(image_input)

        # Extract magnitude prediction (use first channel)
        predicted_magnitude = predicted_image_channels[:, 0, :, :]
        final_magnitude = predicted_magnitude
        dummy_kspace = torch.zeros_like(kspace_masked)
        return {
            'output': torch.abs(final_magnitude),
            'kspace_pred': dummy_kspace,
            'image_complex': final_magnitude,
            'intermediate': {
                'unet_pred': torch.abs(predicted_magnitude),
                'zero_filled': combined_image if combined_image.dim() == 3 else torch.abs(combined_image)
            }
        }
    
    def _kspace_to_image(self, kspace: torch.Tensor) -> torch.Tensor:
        """Convert k-space to image domain using IFFT."""
        # Handle real/imaginary channels
        if kspace.shape[-1] == 2:  # Real/imaginary format (batch, coils, 2, height, width)
            kspace_complex = real_to_complex(kspace)
        else:
            kspace_complex = kspace
        
        return ifft2c(kspace_complex)
    
    def _image_to_kspace(self, image: torch.Tensor) -> torch.Tensor:
        """Convert image to k-space domain using FFT."""
        return fft2c(image)
    
    def _combine_coils(self, coil_images: torch.Tensor) -> torch.Tensor:
        """Combine multi-coil images using root sum of squares. Robust to all input shapes/formats."""
        # Complex format: (batch, coils, height, width)
        if torch.is_complex(coil_images):
            combined = torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=1, keepdim=False))
        # Real/imag format: (batch, coils, 2, height, width)
        elif coil_images.dim() == 5 and coil_images.shape[-3] == 2:
            complex_images = real_to_complex(coil_images)
            combined = torch.sqrt(torch.sum(torch.abs(complex_images) ** 2, dim=1, keepdim=False))
        # Real format: (batch, coils, height, width)
        elif coil_images.dim() == 4:
            combined = torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=1, keepdim=False))
        # Already combined: (batch, height, width)
        elif coil_images.dim() == 3:
            combined = coil_images
        else:
            # Fallback - assume single coil or already combined
            combined = coil_images.squeeze(1) if coil_images.dim() > 3 else coil_images
        # Ensure output is 3D: (batch, height, width)
        if combined.dim() == 4:
            combined = combined.squeeze(1)
        return combined