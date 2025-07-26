#!/usr/bin/env python3
"""
Correct UNet inference using the EXACT same setup as training.
File: scripts/correct_unet_inference.py
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

# Import the EXACT same model class used in training
from unet_model import MRIUNetModel
from utils import (
    complex_to_real, real_to_complex, fft2c, ifft2c, 
    root_sum_of_squares, apply_mask
)

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

def create_undersampling_mask(shape, acceleration=4, center_fraction=0.08, seed=None):
    """Create undersampling mask exactly like in training."""
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

def process_single_file(file_path, model, device, max_slices=10):
    """Process a single .h5 file using the EXACT same pipeline as training."""
    results = []
    
    with h5py.File(file_path, 'r') as f:
        kspace_data = f['kspace']
        num_slices = min(max_slices, kspace_data.shape[0])
        
        for slice_idx in range(num_slices):
            # Get k-space for this slice (exactly like training data loader)
            kspace_full = torch.from_numpy(kspace_data[slice_idx])  # Shape: (coils, height, width)
            
            # Create undersampling mask (same as training)
            height, width = kspace_full.shape[-2:]
            mask = create_undersampling_mask(
                (height, width), 
                acceleration=4, 
                center_fraction=0.08,
                seed=slice_idx  # Reproducible masks
            )
            
            # Apply mask to get undersampled k-space
            kspace_masked = apply_mask(kspace_full, mask)
            
            # Convert to real format (same as training)
            kspace_full_real = complex_to_real(kspace_full.unsqueeze(0))    # Add batch dim
            kspace_masked_real = complex_to_real(kspace_masked.unsqueeze(0)) # Add batch dim
            
            # Create target image (same as training)
            image_full = ifft2c(kspace_full)
            target_image = root_sum_of_squares(image_full, dim=0)
            target_image = torch.abs(target_image)
            
            # Move to device
            kspace_masked_real = kspace_masked_real.to(device)
            mask_batch = mask.unsqueeze(0).to(device)
            kspace_full_real = kspace_full_real.to(device)
            
            # Forward pass (EXACT same call as training)
            start_time = time.time()
            with torch.no_grad():
                output = model(kspace_masked_real, mask_batch, kspace_full_real)
                
                # Extract reconstruction (should be a dict with 'output' key)
                if isinstance(output, dict) and 'output' in output:
                    reconstruction = output['output'].squeeze().cpu()
                else:
                    reconstruction = output.squeeze().cpu()
            
            inference_time = time.time() - start_time
            
            # Compute metrics
            metrics = compute_metrics(reconstruction, target_image)
            metrics['inference_time'] = inference_time
            
            results.append({
                'file': file_path.name,
                'slice': slice_idx,
                'metrics': metrics,
                'reconstruction': reconstruction,
                'target': target_image
            })
            
            print(f"  Slice {slice_idx}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
    
    return results

def main():
    """Main UNet inference using EXACT training setup."""
    print("=== CORRECT UNET INFERENCE (EXACT TRAINING SETUP) ===")
    
    # Setup (same as training)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "/scratch/vchaurasia/organized_models/unet_epoch20.pth"
    test_data_path = "/scratch/vchaurasia/fastmri_data/test"
    output_dir = "/scratch/vchaurasia/organized_models/inference_results/unet_correct"
    
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model checkpoint
    print("Loading UNet checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get the EXACT config used during training
    if 'config' in checkpoint:
        model_config = checkpoint['config']['model']
        loss_config = checkpoint['config']['loss']
        
        print("Model config from training:")
        for key, value in model_config.items():
            print(f"  {key}: {value}")
        
        # Create model using EXACT same parameters as training
        model = MRIUNetModel(
            unet_config={
                'in_channels': model_config['in_channels'],      # 2
                'out_channels': model_config['out_channels'],    # 2  
                'features': model_config['features']             # 64
            },
            use_data_consistency=True,
            dc_weight=loss_config['data_consistency_weight'],   # Same as training
            num_dc_iterations=5                                 # Same as training
        ).to(device)
        
        print("✅ Model created with EXACT training configuration")
        
    else:
        print("❌ No config found in checkpoint - cannot recreate exact model")
        return
    
    # Load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model weights loaded successfully")
    except Exception as e:
        print(f"❌ Error loading state dict: {e}")
        import traceback
        traceback.print_exc()
        return
    
    model.eval()
    
    # Get test files
    test_files = list(Path(test_data_path).glob("*.h5"))[:10]  # Process 10 files
    print(f"Processing {len(test_files)} files...")
    
    # Process files
    all_metrics = {'psnr': [], 'ssim': [], 'nmse': [], 'mae': [], 'inference_time': []}
    
    for file_path in tqdm(test_files, desc="Processing files"):
        try:
            print(f"\nProcessing: {file_path.name}")
            file_results = process_single_file(file_path, model, device, max_slices=5)
            
            for result in file_results:
                metrics = result['metrics']
                for metric_name, metric_value in metrics.items():
                    if metric_name in all_metrics:
                        all_metrics[metric_name].append(metric_value)
                        
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue
    
    # Save summary
    print("\n=== RESULTS SUMMARY ===")
    summary_file = Path(output_dir) / "summary_metrics.txt"
    
    with open(summary_file, 'w') as f:
        f.write("Correct UNet Inference Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write("Model configuration (from training):\n")
        if 'config' in checkpoint:
            for key, value in checkpoint['config']['model'].items():
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
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
    
    # Print training info for reference
    if 'config' in checkpoint:
        print(f"\nModel was trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown')}")

if __name__ == "__main__":
    main()