#!/usr/bin/env python3
"""
Debug a single sample to see exact values and find the PSNR issue.
"""

import os
import sys
import torch
import numpy as np
import h5py
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from unet_model import MRIUNetModel
from utils import (
    complex_to_real, real_to_complex, fft2c, ifft2c, 
    root_sum_of_squares, apply_mask
)

def compute_metrics_debug(prediction, target):
    """Compute metrics with debug info."""
    print(f"\n=== METRIC COMPUTATION DEBUG ===")
    
    # Convert to numpy
    pred_np = prediction.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    
    print(f"Prediction numpy shape: {pred_np.shape}")
    print(f"Target numpy shape: {target_np.shape}")
    print(f"Prediction range: [{pred_np.min():.6f}, {pred_np.max():.6f}]")
    print(f"Target range: [{target_np.min():.6f}, {target_np.max():.6f}]")
    print(f"Prediction mean: {pred_np.mean():.6f}")
    print(f"Target mean: {target_np.mean():.6f}")
    
    # PSNR calculation step by step
    mse = np.mean((pred_np - target_np) ** 2)
    print(f"MSE: {mse:.6f}")
    
    if mse > 0:
        max_val = np.max(target_np)
        print(f"Max target value: {max_val:.6f}")
        
        if max_val > 0:
            psnr = 20 * np.log10(max_val / np.sqrt(mse))
            print(f"PSNR calculation: 20 * log10({max_val:.6f} / sqrt({mse:.6f})) = {psnr:.6f}")
        else:
            print("ERROR: Max target value is 0!")
            psnr = float('-inf')
    else:
        print("MSE is 0 - perfect match!")
        psnr = float('inf')
    
    return {'psnr': float(psnr), 'mse': float(mse)}

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

def debug_single_sample():
    """Debug processing of a single sample."""
    print("=== DEBUGGING SINGLE SAMPLE ===")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "/scratch/vchaurasia/organized_models/unet_epoch20.pth"
    test_data_path = "/scratch/vchaurasia/fastmri_data/test"
    
    print(f"Device: {device}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['config']['model']
    loss_config = checkpoint['config']['loss']
    
    model = MRIUNetModel(
        unet_config={
            'in_channels': model_config['in_channels'],
            'out_channels': model_config['out_channels'], 
            'features': model_config['features']
        },
        use_data_consistency=True,
        dc_weight=loss_config['data_consistency_weight'],
        num_dc_iterations=5
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded")
    
    # Get first test file
    test_files = list(Path(test_data_path).glob("*.h5"))
    file_path = test_files[0]
    print(f"Processing: {file_path.name}")
    
    # Load data
    with h5py.File(file_path, 'r') as f:
        kspace_data = f['kspace']
        print(f"K-space data shape: {kspace_data.shape}")
        
        # Get first slice
        kspace_full = torch.from_numpy(kspace_data[0])  # Shape: (coils, height, width)
        print(f"K-space slice shape: {kspace_full.shape}")
        print(f"K-space is complex: {torch.is_complex(kspace_full)}")
        print(f"K-space range: [{torch.abs(kspace_full).min():.6f}, {torch.abs(kspace_full).max():.6f}]")
        
        # Create mask
        height, width = kspace_full.shape[-2:]
        mask = create_undersampling_mask((height, width), seed=0)
        print(f"Mask shape: {mask.shape}")
        print(f"Mask sampling ratio: {mask.sum().item() / mask.numel():.4f}")
        
        # Apply mask
        kspace_masked = apply_mask(kspace_full, mask)
        print(f"Masked k-space range: [{torch.abs(kspace_masked).min():.6f}, {torch.abs(kspace_masked).max():.6f}]")
        
        # Convert to real format for model input
        kspace_full_real = complex_to_real(kspace_full.unsqueeze(0))    # Add batch dim
        kspace_masked_real = complex_to_real(kspace_masked.unsqueeze(0)) # Add batch dim
        
        print(f"K-space real format shape: {kspace_masked_real.shape}")
        print(f"K-space real format range: [{kspace_masked_real.min():.6f}, {kspace_masked_real.max():.6f}]")
        
        # Create target image
        image_full = ifft2c(kspace_full)
        print(f"Full image shape: {image_full.shape}")
        print(f"Full image range: [{torch.abs(image_full).min():.6f}, {torch.abs(image_full).max():.6f}]")
        
        target_image = root_sum_of_squares(image_full, dim=0)
        target_image = torch.abs(target_image)
        print(f"Target image shape: {target_image.shape}")
        print(f"Target image range: [{target_image.min():.6f}, {target_image.max():.6f}]")
        
        # Move to device and run model
        kspace_masked_real = kspace_masked_real.to(device)
        mask_batch = mask.unsqueeze(0).to(device)
        kspace_full_real = kspace_full_real.to(device)
        
        print(f"\n=== MODEL FORWARD PASS ===")
        with torch.no_grad():
            output = model(kspace_masked_real, mask_batch, kspace_full_real)
            
            if isinstance(output, dict) and 'output' in output:
                reconstruction = output['output'].squeeze().cpu()
                print(f"Model output keys: {list(output.keys())}")
            else:
                reconstruction = output.squeeze().cpu()
                print(f"Model output type: {type(output)}")
            
            print(f"Reconstruction shape: {reconstruction.shape}")
            print(f"Reconstruction range: [{reconstruction.min():.6f}, {reconstruction.max():.6f}]")
            print(f"Reconstruction mean: {reconstruction.mean():.6f}")
            print(f"Reconstruction has NaN: {torch.isnan(reconstruction).any()}")
            print(f"Reconstruction has Inf: {torch.isinf(reconstruction).any()}")
        
        # Compute metrics with debug
        metrics = compute_metrics_debug(reconstruction, target_image)
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"PSNR: {metrics['psnr']:.6f}")
        print(f"MSE: {metrics['mse']:.6f}")

if __name__ == "__main__":

        # Test the conversion
    import torch
    from src.utils import complex_to_real, real_to_complex

    # Test with simple complex number
    test_complex = torch.tensor(3.0 + 4.0j)
    test_real = complex_to_real(test_complex)
    print(f"Test complex: {test_complex}")
    print(f"Test real: {test_real}")
    print(f"Test real shape: {test_real.shape}")
    #debug_single_sample()