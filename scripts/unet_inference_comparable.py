#!/usr/bin/env python3
"""
UNet inference that matches DISCO evaluation exactly.
Same parameters, same number of samples, same metrics.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path
import time
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import complex_to_real, real_to_complex, fft2c, ifft2c, apply_mask

class SimpleUNet(nn.Module):
    """Simple UNet for MRI reconstruction."""
    
    def __init__(self, in_channels=2, out_channels=2, features=64, depth=4, dropout=0.1):
        super().__init__()
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        current_features = features
        for i in range(depth):
            in_ch = in_channels if i == 0 else current_features // 2
            self.encoder_layers.append(self._double_conv(in_ch, current_features, dropout))
            if i < depth - 1:
                self.pool_layers.append(nn.MaxPool2d(2))
            current_features *= 2
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()
        
        for i in range(depth - 1):
            self.upconv_layers.append(nn.ConvTranspose2d(current_features, current_features // 2, 2, 2))
            self.decoder_layers.append(self._double_conv(current_features, current_features // 2, dropout))
            current_features //= 2
        
        # Final layer
        self.final = nn.Conv2d(features, out_channels, 1)
    
    def _double_conv(self, in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
    
    def forward(self, x):
        # Encoder
        skip_connections = []
        for i, (encoder, pool) in enumerate(zip(self.encoder_layers[:-1], self.pool_layers)):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.encoder_layers[-1](x)
        
        # Decoder
        for i, (upconv, decoder) in enumerate(zip(self.upconv_layers, self.decoder_layers)):
            x = upconv(x)
            skip = skip_connections[-(i + 1)]
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        return self.final(x)

def load_unet_model(checkpoint_path, device):
    """Load UNet model from checkpoint."""
    print(f"Loading UNet checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']['model']
    
    # Create model with exact same config as training
    model = SimpleUNet(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'], 
        features=config['features'],
        depth=config['depth'],
        dropout=config['dropout']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ UNet model loaded successfully")
    print(f"Model config: {config}")
    
    return model, config

def compute_metrics(prediction, target):
    """Compute evaluation metrics."""
    # Convert to numpy
    pred_np = prediction.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    
    # PSNR
    mse = np.mean((pred_np - target_np) ** 2)
    if mse > 0:
        max_val = np.max(target_np)
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # SSIM (simplified)
    mean_pred = np.mean(pred_np)
    mean_target = np.mean(target_np)
    var_pred = np.var(pred_np)
    var_target = np.var(target_np)
    cov = np.mean((pred_np - mean_pred) * (target_np - mean_target))
    
    c1 = (0.01 * np.max(target_np)) ** 2
    c2 = (0.03 * np.max(target_np)) ** 2
    
    ssim = ((2 * mean_pred * mean_target + c1) * (2 * cov + c2)) / \
           ((mean_pred**2 + mean_target**2 + c1) * (var_pred + var_target + c2))
    
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

def process_file(model, file_path, device, max_slices=5):
    """Process a single h5 file."""
    results = []
    
    with h5py.File(file_path, 'r') as f:
        kspace_data = f['kspace'][:]
        num_slices = min(max_slices, kspace_data.shape[0])
        
        for slice_idx in range(num_slices):
            kspace_slice = torch.from_numpy(kspace_data[slice_idx]).to(device)
            
            # Generate mask (4x acceleration, 8% center)
            height, width = kspace_slice.shape[-2:]
            mask = torch.zeros(width, device=device)
            
            # Center fraction
            center_lines = int(0.08 * width)
            center_start = width // 2 - center_lines // 2
            center_end = center_start + center_lines
            mask[center_start:center_end] = 1
            
            # Random lines for 4x acceleration
            total_lines = width // 4
            remaining_lines = total_lines - center_lines
            
            if remaining_lines > 0:
                available_indices = [i for i in range(width) if mask[i] == 0]
                if len(available_indices) >= remaining_lines:
                    selected = torch.randperm(len(available_indices))[:remaining_lines]
                    for idx in selected:
                        mask[available_indices[idx]] = 1
            
            mask = mask.unsqueeze(0).expand(height, -1)
            
            # Apply mask
            kspace_masked = apply_mask(kspace_slice, mask)
            
            # Convert to image space for UNet input
            image_masked = ifft2c(kspace_masked)
            image_full = ifft2c(kspace_slice)
            
            # Root sum of squares for multi-coil combination
            image_masked_rss = torch.sqrt(torch.sum(torch.abs(image_masked) ** 2, dim=0))
            image_full_rss = torch.sqrt(torch.sum(torch.abs(image_full) ** 2, dim=0))
            
            # Prepare input for UNet (real/imaginary channels)
            input_tensor = complex_to_real(image_masked_rss.unsqueeze(0).unsqueeze(0))
            target_tensor = torch.abs(image_full_rss)
            
            # Forward pass
            start_time = time.time()
            with torch.no_grad():
                prediction_tensor = model(input_tensor)
                # Convert back to magnitude
                prediction_complex = real_to_complex(prediction_tensor)
                prediction = torch.abs(prediction_complex.squeeze())
            
            inference_time = time.time() - start_time
            
            # Compute metrics
            metrics = compute_metrics(prediction, target_tensor)
            metrics['inference_time'] = inference_time
            
            results.append(metrics)
            
            print(f"  Slice {slice_idx}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
    
    return results

def main():
    print("=== UNET INFERENCE (COMPARABLE TO DISCO) ===")
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Paths
    model_path = "/scratch/vchaurasia/organized_models/unet_epoch20.pth"
    test_data_path = "/scratch/vchaurasia/fastmri_data/test"
    output_dir = "/scratch/vchaurasia/organized_models/inference_results/unet_comparable"
    
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, model_config = load_unet_model(model_path, device)
    
    # Get test files
    test_files = sorted(list(Path(test_data_path).glob("*.h5")))
    max_files = 100  # Same as DISCO
    test_files = test_files[:max_files]
    
    print(f"Processing {len(test_files)} files...")
    
    # Process files
    all_results = []
    
    for i, file_path in enumerate(test_files):
        print(f"Processing: {file_path.name} ({i+1}/{len(test_files)})")
        
        try:
            file_results = process_file(model, file_path, device)
            all_results.extend(file_results)
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            continue
    
    # Compute summary statistics
    if all_results:
        metrics_summary = {}
        for metric in ['psnr', 'ssim', 'nmse', 'mae', 'inference_time']:
            values = [r[metric] for r in all_results if metric in r]
            if values:
                metrics_summary[f'{metric}_mean'] = np.mean(values)
                metrics_summary[f'{metric}_std'] = np.std(values)
                metrics_summary[f'{metric}_median'] = np.median(values)
                metrics_summary[f'{metric}_min'] = np.min(values)
                metrics_summary[f'{metric}_max'] = np.max(values)
        
        # Print summary
        print("\n=== RESULTS SUMMARY ===")
        print(f"PSNR: {metrics_summary.get('psnr_mean', 0):.4f} ± {metrics_summary.get('psnr_std', 0):.4f}")
        print(f"SSIM: {metrics_summary.get('ssim_mean', 0):.4f} ± {metrics_summary.get('ssim_std', 0):.4f}")
        print(f"NMSE: {metrics_summary.get('nmse_mean', 0):.4f} ± {metrics_summary.get('nmse_std', 0):.4f}")
        print(f"MAE: {metrics_summary.get('mae_mean', 0):.4f} ± {metrics_summary.get('mae_std', 0):.4f}")
        print(f"INFERENCE_TIME: {metrics_summary.get('inference_time_mean', 0):.4f} ± {metrics_summary.get('inference_time_std', 0):.4f}")
        
        # Save results
        results_data = {
            'summary': metrics_summary,
            'all_results': all_results,
            'config': {
                'model_path': model_path,
                'num_files': len(test_files),
                'total_samples': len(all_results),
                'model_config': model_config
            }
        }
        
        # Save as JSON
        with open(os.path.join(output_dir, 'unet_results.json'), 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save summary text
        with open(os.path.join(output_dir, 'summary_metrics.txt'), 'w') as f:
            f.write("UNet Reconstruction Results Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Files processed: {len(test_files)}\n")
            f.write(f"Total samples: {len(all_results)}\n\n")
            f.write("Summary Statistics:\n")
            f.write("-" * 20 + "\n")
            for metric in ['psnr', 'ssim', 'nmse', 'mae', 'inference_time']:
                f.write(f"{metric.upper()}:\n")
                f.write(f"  Mean: {metrics_summary.get(f'{metric}_mean', 0):.6f}\n")
                f.write(f"  Std:  {metrics_summary.get(f'{metric}_std', 0):.6f}\n")
                f.write(f"  Min:  {metrics_summary.get(f'{metric}_min', 0):.6f}\n")
                f.write(f"  Max:  {metrics_summary.get(f'{metric}_max', 0):.6f}\n")
                f.write(f"  Median: {metrics_summary.get(f'{metric}_median', 0):.6f}\n\n")
        
        print(f"\n✅ UNet inference completed!")
        print(f"Results saved to: {output_dir}")
        print(f"Processed {len(all_results)} samples total")
    
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main()