# File: src/simple_unet.py (PROPERLY FIXED)

import torch
import torch.nn as nn
import torch.nn.functional as F


def fft2c(x):
    """Centered 2D FFT"""
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=[-2, -1])), dim=[-2, -1])


def ifft2c(x):
    """Centered 2D IFFT"""
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-2, -1])), dim=[-2, -1])


def complex_to_real(x):
    """Convert complex tensor to real with separate channels"""
    return torch.stack([x.real, x.imag], dim=-1)


def real_to_complex(x):
    """Convert real tensor with real/imag in dim=2 to complex"""
    # For shape (batch, coils, 2, height, width) -> (batch, coils, height, width)
    #print(f"real_to_complex input: {x.shape}")
    if x.shape[2] == 2:  # Check if dim=2 has real/imag
        result = torch.complex(x[:, :, 0, :, :], x[:, :, 1, :, :])
    else:
        result = torch.complex(x[..., 0], x[..., 1])  # Fallback for other formats
    #print(f"real_to_complex output: {result.shape}")
    return result


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
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        # Safety check for pooling
        _, _, h, w = x.shape
        if h >= 2 and w >= 2:
            x = self.maxpool(x)
        #else:
            #print(f"Warning: Skipping maxpool for small tensor {x.shape}")
            
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatches
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AdaptiveUNet(nn.Module):
    """U-Net that adapts to input size and avoids pooling issues"""
    
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(AdaptiveUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Ensure minimum dimensions and handle padding
        original_shape = x.shape
        b, c, h, w = x.shape
        
        # Minimum size for 4 pooling operations: 16x16
        min_size = 16
        pad_h = max(0, min_size - h)
        pad_w = max(0, min_size - w)
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
        
        # Ensure even dimensions for pooling
        _, _, h, w = x.shape
        pad_h_even = h % 2
        pad_w_even = w % 2
        
        if pad_h_even or pad_w_even:
            x = F.pad(x, (0, pad_w_even, 0, pad_h_even))

        # Forward pass
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
        
        # Crop back to original size if we padded
        target_h = original_shape[2]
        target_w = original_shape[3]
        if logits.shape[2] != target_h or logits.shape[3] != target_w:
            logits = logits[:, :, :target_h, :target_w]
        
        return logits


class SimpleMRIUNet(nn.Module):
    """
    PROPERLY FIXED - Simple MRI reconstruction model using U-Net.
    
    Fixed the real_to_complex conversion that was dropping dimensions.
    """
    
    def __init__(self, n_channels=2, n_classes=1, dc_weight=1.0, num_dc_iterations=5):
        super().__init__()
        
        # U-Net for image reconstruction (1 channel in, 1 channel out)
        self.unet = AdaptiveUNet(n_channels=1, n_classes=1, bilinear=True)
        
        # Store parameters
        self.dc_weight = dc_weight
        self.num_dc_iterations = num_dc_iterations
        
    def _kspace_to_image(self, kspace_tensor):
        """
        Convert k-space tensor to image domain.
        
        PROPERLY FIXED: Real/imag to complex conversion preserves all dimensions.
        """
        #print(f"K-space input shape: {kspace_tensor.shape}")
        
        # Input should be (batch, coils, 2, height, width)
        #print(f"Checking if kspace_tensor.shape[-1] == 2: {kspace_tensor.shape[-1]} == 2 -> {kspace_tensor.shape[-1] == 2}")
        #print(f"Checking if kspace_tensor.shape[2] == 2: {kspace_tensor.shape[2]} == 2 -> {kspace_tensor.shape[2] == 2}")
        
        if kspace_tensor.shape[2] == 2:  # FIXED: Check dimension 2, not last dimension
            # CRITICAL FIX: Use proper indexing - dim=2 is the real/imag dimension
            #print(f"Converting real/imag to complex...")
            #print(f"Real part: {kspace_tensor[:, :, 0, :, :].shape}")
            #print(f"Imag part: {kspace_tensor[:, :, 1, :, :].shape}")
            
            kspace_complex = torch.complex(
                kspace_tensor[:, :, 0, :, :],  # Real part
                kspace_tensor[:, :, 1, :, :]   # Imag part
            )
        else:
            kspace_complex = kspace_tensor
        
        #print(f"K-space complex shape: {kspace_complex.shape}")
        
        # Verify we didn't lose dimensions
        # Input: (batch, coils, 2, height, width) -> Output: (batch, coils, height, width)
        expected_shape = (kspace_tensor.shape[0], kspace_tensor.shape[1], kspace_tensor.shape[3], kspace_tensor.shape[4])
        if kspace_complex.shape != expected_shape:
            raise ValueError(f"Complex conversion failed: expected {expected_shape}, got {kspace_complex.shape}")
        
        print(f"✅ Complex conversion successful!")
        
        # IFFT to image domain: (batch, coils, height, width) -> (batch, coils, height, width)
        image_coils = ifft2c(kspace_complex)
        #print(f"Image coils shape: {image_coils.shape}")
        
        # Root Sum of Squares to combine coils: (batch, coils, height, width) -> (batch, height, width)
        image_magnitude = torch.sqrt(torch.sum(torch.abs(image_coils) ** 2, dim=1))
        #print(f"Combined image shape: {image_magnitude.shape}")
        
        # Final verification
        expected_final_shape = (kspace_tensor.shape[0], kspace_tensor.shape[3], kspace_tensor.shape[4])
        if image_magnitude.shape != expected_final_shape:
            raise ValueError(f"Final shape error: expected {expected_final_shape}, got {image_magnitude.shape}")
        
        return image_magnitude
    
    def forward(self, kspace_masked, mask, kspace_full=None):
        """
        Forward pass for MRI reconstruction.
        
        PROPERLY FIXED: All dimension issues resolved.
        """
        #print(f"\n=== SimpleMRIUNet Forward ===")
        #print(f"kspace_masked shape: {kspace_masked.shape}")
        #print(f"mask shape: {mask.shape}")
        
        # Convert k-space to image domain
        image_input = self._kspace_to_image(kspace_masked)
        batch_size = image_input.shape[0]
        
        #print(f"Image after coil combination: {image_input.shape}")
        
        # Normalize each image individually
        normalized_images = []
        for i in range(batch_size):
            img = image_input[i]
            img_min, img_max = torch.min(img), torch.max(img)
            if img_max > img_min:
                img_norm = (img - img_min) / (img_max - img_min)
            else:
                img_norm = img
            normalized_images.append(img_norm)
        
        image_input = torch.stack(normalized_images, dim=0)
        
        # Add channel dimension for U-Net: (batch, height, width) -> (batch, 1, height, width)
        if len(image_input.shape) == 3:
            image_input = image_input.unsqueeze(1)
        
        #print(f"Final input to UNet: {image_input.shape}")
        
        # Final shape verification before U-Net
        if image_input.shape[1] != 1:
            raise ValueError(f"U-Net expects 1 channel, got {image_input.shape[1]} channels")
        
        # Pass through U-Net
        reconstructed = self.unet(image_input)
        
        # Remove channel dimension if present
        if reconstructed.shape[1] == 1:
            reconstructed = reconstructed.squeeze(1)
        
        #print(f"UNet output shape: {reconstructed.shape}")
        
        return {
            'output': reconstructed,
            'image_input': image_input.squeeze(1) if image_input.shape[1] == 1 else image_input
        }


# Backward compatibility
MRIReconstructionUNet = SimpleMRIUNet