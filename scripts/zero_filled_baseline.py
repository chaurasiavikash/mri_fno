#!/usr/bin/env python3
"""
Zero-filled baseline using EXACTLY the same pipeline as UNet inference.
This ensures fair comparison by using identical data processing.
File: scripts/zerofilled_like_unet.py
"""

import os
import sys
import torch
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import (
    complex_to_real, real_to_complex, fft2c, ifft2c, 
    root_sum_of_squares, apply_mask
)

def compute_metrics(prediction, target):
    """Compute evaluation metrics - EXACT COPY from UNet inference."""
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

def create_undersampling_mask(shape, acceleration=4, center_fraction=0.08, seed=None):
    """Create undersampling mask - EXACT COPY from UNet inference."""
    height, width = shape
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Calculate number of center lines to keep
    num_center_lines = int(center_fraction * width)
    center_start = width // 2 - num_center_lines // 2
    center_end = center_start + num_center_lines
    
    # Calculate total lines to keep
    total_lines = width // acceleration
    remaining_lines = total_lines - num_center_lines
    
    # Create mask
    mask = torch.zeros(width)
    
    # Always keep center lines
    mask[center_start:center_end] = 1
    
    # Randomly select remaining lines
    available_indices = list(range(width))
    for i in range(center_start, center_end):
        if i in available_indices:
            available_indices.remove(i)
    
    if remaining_lines > 0 and available_indices:
        selected_indices = torch.randperm(len(available_indices))[:remaining_lines]
        for idx in selected_indices:
            mask[available_indices[idx]] = 1
    
    # Expand to full k-space shape
    mask = mask.unsqueeze(0).expand(height, -1)
    
    return mask

def process_single_file(file_path, max_slices=5):
    """Process file using EXACT same pipeline as UNet, but no neural network."""
    results = []
    
    with h5py.File(file_path, 'r') as f:
        kspace_data = f['kspace']
        num_slices = min(max_slices, kspace_data.shape[0])
        
        for slice_idx in range(num_slices):
            # EXACT COPY of UNet data processing pipeline
            kspace_full = torch.from_numpy(kspace_data[slice_idx])  # Shape: (coils, height, width)
            
            # Create undersampling mask (same as UNet)
            height, width = kspace_full.shape[-2:]
            mask = create_undersampling_mask(
                (height, width), 
                acceleration=4, 
                center_fraction=0.08,
                seed=slice_idx  # Reproducible masks
            )
            
            # Apply mask to get undersampled k-space
            kspace_masked = apply_mask(kspace_full, mask)
            
            # Create target image (EXACT same as UNet)
            image_full = ifft2c(kspace_full)
            target_image = root_sum_of_squares(image_full, dim=0)
            target_image = torch.abs(target_image)
            
            mean = target_image.mean()
            std = target_image.std()
            target_image = (target_image - mean) / (std + 1e-8)
            
            # ZERO-FILLED RECONSTRUCTION (instead of neural network)
            image_masked = ifft2c(kspace_masked)  # Just IFFT of masked k-space
            zero_filled_reconstruction = root_sum_of_squares(image_masked, dim=0)
            zero_filled_reconstruction = torch.abs(zero_filled_reconstruction)
            
            # Apply SAME normalization as target (this is the key!)
            zero_filled_reconstruction = (zero_filled_reconstruction - mean) / (std + 1e-8)
            
            # Compute metrics using exact same function as UNet
            metrics = compute_metrics(zero_filled_reconstruction, target_image)
            
            results.append({
                'file': file_path.name,
                'slice': slice_idx,
                'metrics': metrics,
                'reconstruction': zero_filled_reconstruction,
                'target': target_image
            })
            
            print(f"  Slice {slice_idx}: Zero-filled PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
    
    return results

def main():
    """Main zero-filled baseline using UNet's exact pipeline."""
    print("=== ZERO-FILLED BASELINE (USING UNET'S EXACT PIPELINE) ===")
    
    test_data_path = "/scratch/vchaurasia/fastmri_data/test"
    output_dir = "/scratch/vchaurasia/organized_models/inference_results/zerofilled_like_unet"
    
    print(f"Test data: {test_data_path}")
    print(f"Output: {output_dir}")
    print("Using EXACTLY the same data processing as UNet inference")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test files - EXACT same 100 files as UNet
    test_files = list(Path(test_data_path).glob("*.h5"))[:100]  # Process 100 files
    print(f"Processing {len(test_files)} files...")
    
    # Process files
    all_metrics = {'psnr': [], 'ssim': [], 'nmse': [], 'mae': []}
    
    for file_path in tqdm(test_files, desc="Processing files"):
        try:
            print(f"\nProcessing: {file_path.name}")
            file_results = process_single_file(file_path, max_slices=5)
            
            for result in file_results:
                metrics = result['metrics']
                for metric_name, metric_value in metrics.items():
                    if metric_name in all_metrics:
                        all_metrics[metric_name].append(metric_value)
                        
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue
    
    # Save summary
    print("\n=== ZERO-FILLED BASELINE RESULTS ===")
    summary_file = Path(output_dir) / "summary_metrics.txt"
    
    with open(summary_file, 'w') as f:
        f.write("Zero-filled Baseline (Using UNet's Exact Pipeline)\n")
        f.write("=" * 50 + "\n\n")
        f.write("Method: Simple IFFT of undersampled k-space\n")
        f.write("Data processing: IDENTICAL to UNet inference\n")
        f.write("Normalization: Same target statistics as UNet uses\n")
        f.write("Acceleration: 4x\n")
        f.write("Center fraction: 8%\n\n")
        
        for metric in ['psnr', 'ssim', 'nmse', 'mae']:
            if metric in all_metrics and all_metrics[metric]:
                values = all_metrics[metric]
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                f.write(f"{metric.upper()}:\n")
                f.write(f"  Mean: {mean_val:.6f}\n")
                f.write(f"  Std:  {std_val:.6f}\n")
                f.write(f"  Count: {len(values)}\n\n")
                
                print(f"Zero-filled {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save raw metrics
    np.savez_compressed(Path(output_dir) / "all_metrics.npz", **all_metrics)
    
    print(f"\n✅ Zero-filled baseline completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Processed {len(all_metrics['psnr'])} samples total")
    
    print("\n📊 COMPARISON WITH UNET:")
    print("UNet PSNR: 18.71 ± 3.42")
    print(f"Zero-filled PSNR: {np.mean(all_metrics['psnr']):.2f} ± {np.std(all_metrics['psnr']):.2f}")
    improvement = np.mean(all_metrics['psnr']) - 18.71
    print(f"UNet improvement: {-improvement:.2f} dB better than zero-filled")

if __name__ == "__main__":
    main()