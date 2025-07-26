#!/usr/bin/env python3
"""
Simple UNet inference script that matches the trained model exactly.
File: scripts/simple_unet_inference.py
"""

import os
import sys
import torch
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import complex_to_real, real_to_complex, fft2c, ifft2c, root_sum_of_squares

# Simple UNet definition (matches your trained model)
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
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
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
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

class SimpleUNetModel(nn.Module):
    """Wrapper to match the checkpoint structure"""
    def __init__(self, in_channels=2, out_channels=2, features=64):
        super().__init__()
        self.unet = UNet(in_channels, out_channels, bilinear=False)
    
    def forward(self, x, mask=None):
        return self.unet(x)

def compute_metrics(prediction, target):
    """Compute evaluation metrics."""
    pred_np = prediction.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    
    # PSNR
    mse = np.mean((pred_np - target_np) ** 2)
    if mse > 0:
        max_val = np.max(target_np)
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # Simple SSIM
    mu1, mu2 = np.mean(pred_np), np.mean(target_np)
    sigma1, sigma2 = np.var(pred_np), np.var(target_np)
    sigma12 = np.mean((pred_np - mu1) * (target_np - mu2))
    
    c1, c2 = (0.01 * np.max(target_np)) ** 2, (0.03 * np.max(target_np)) ** 2
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    
    # NMSE
    nmse = mse / np.mean(target_np ** 2)
    
    # MAE
    mae = np.mean(np.abs(pred_np - target_np))
    
    return {
        'psnr': float(psnr),
        'ssim': float(ssim),
        'nmse': float(nmse),
        'mae': float(mae)
    }

def process_single_file(file_path, model, device, max_slices=5):
    """Process a single .h5 file."""
    results = []
    
    with h5py.File(file_path, 'r') as f:
        kspace_data = f['kspace']
        num_slices = min(max_slices, kspace_data.shape[0])
        
        for slice_idx in range(num_slices):
            # Get k-space for this slice
            kspace_slice = kspace_data[slice_idx]  # Shape: (coils, height, width)
            kspace_tensor = torch.from_numpy(kspace_slice)
            
            # Create undersampling mask (4x acceleration, 8% center)
            height, width = kspace_slice.shape[-2:]
            mask = torch.zeros(width)
            
            # Center 8%
            center_lines = int(0.08 * width)
            center_start = width // 2 - center_lines // 2
            center_end = center_start + center_lines
            mask[center_start:center_end] = 1
            
            # Random additional lines for 4x acceleration
            total_lines = width // 4
            remaining_lines = total_lines - center_lines
            available_indices = list(range(width))
            for i in range(center_start, center_end):
                if i in available_indices:
                    available_indices.remove(i)
            
            if remaining_lines > 0 and available_indices:
                selected_indices = torch.randperm(len(available_indices))[:remaining_lines]
                for idx in selected_indices:
                    mask[available_indices[idx]] = 1
            
            mask = mask.unsqueeze(0).expand(height, -1)
            
            # Apply mask
            kspace_masked = kspace_tensor * mask.unsqueeze(0)
            
            # Convert to image domain
            coil_images = ifft2c(kspace_masked)
            combined_image = root_sum_of_squares(coil_images, dim=0)
            target_image = torch.abs(combined_image)
            
            # Create zero-filled reconstruction
            zero_filled = torch.abs(ifft2c(kspace_masked).sum(dim=0))
            
            # Prepare UNet input (real/imaginary of zero-filled)
            zero_filled_complex = ifft2c(kspace_masked).sum(dim=0)
            unet_input = torch.stack([zero_filled_complex.real, zero_filled_complex.imag], dim=0)
            unet_input = unet_input.unsqueeze(0).to(device)
            
            # Forward pass
            start_time = time.time()
            with torch.no_grad():
                unet_output = model(unet_input)
                # Convert back to magnitude
                output_complex = torch.complex(unet_output[0, 0], unet_output[0, 1])
                reconstruction = torch.abs(output_complex).cpu()
            
            inference_time = time.time() - start_time
            
            # Compute metrics
            metrics = compute_metrics(reconstruction, target_image)
            metrics['inference_time'] = inference_time
            
            results.append({
                'file': file_path.name,
                'slice': slice_idx,
                'metrics': metrics,
                'reconstruction': reconstruction,
                'target': target_image,
                'zero_filled': torch.abs(zero_filled)
            })
    
    return results

def main():
    """Main UNet inference."""
    print("=== SIMPLE UNET INFERENCE ===")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "/scratch/vchaurasia/organized_models/unet_epoch20.pth"
    test_data_path = "/scratch/vchaurasia/fastmri_data/test"
    output_dir = "/scratch/vchaurasia/organized_models/inference_results/unet_simple"
    
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading UNet model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = SimpleUNetModel(in_channels=2, out_channels=2, features=64).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Get test files
    test_files = list(Path(test_data_path).glob("*.h5"))[:10]  # Process 10 files
    print(f"Processing {len(test_files)} files...")
    
    # Process files
    all_metrics = {'psnr': [], 'ssim': [], 'nmse': [], 'mae': [], 'inference_time': []}
    
    for file_path in tqdm(test_files, desc="Processing files"):
        try:
            file_results = process_single_file(file_path, model, device, max_slices=2)
            
            for result in file_results:
                metrics = result['metrics']
                for metric_name, metric_value in metrics.items():
                    if metric_name in all_metrics:
                        all_metrics[metric_name].append(metric_value)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Save summary
    print("\n=== RESULTS SUMMARY ===")
    summary_file = Path(output_dir) / "summary_metrics.txt"
    
    with open(summary_file, 'w') as f:
        f.write("Simple UNet Inference Summary\n")
        f.write("=" * 40 + "\n\n")
        
        for metric in ['psnr', 'ssim', 'nmse', 'mae', 'inference_time']:
            if metric in all_metrics and all_metrics[metric]:
                values = all_metrics[metric]
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                f.write(f"{metric.upper()}:\n")
                f.write(f"  Mean: {mean_val:.6f}\n")
                f.write(f"  Std:  {std_val:.6f}\n")
                f.write(f"  Count: {len(values)}\n\n")
                
                print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save raw metrics
    np.savez_compressed(Path(output_dir) / "all_metrics.npz", **all_metrics)
    
    print(f"\n✅ UNet inference completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Processed {len(all_metrics['psnr'])} samples total")

if __name__ == "__main__":
    main()