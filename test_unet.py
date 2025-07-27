#!/usr/bin/env python3
"""
Fixed UNet Inference Script
Based on debug findings - using forward normalization FFT and proper scaling
"""

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append('/home/vchaurasia/projects/mri_fno/src')

from simple_unet import SimpleUNet

def fft_forward_norm(kspace):
    """Best FFT implementation from debug - forward normalization"""
    return torch.fft.ifft2(kspace, norm='forward')

def apply_scaling(image, scale_factor=1e6):
    """Apply scaling found during debug"""
    return image * scale_factor

def root_sum_of_squares(tensor, dim=0):
    """RSS coil combination"""
    return torch.sqrt(torch.sum(torch.abs(tensor) ** 2, dim=dim))

def load_and_preprocess_data(file_path, slice_idx=17):
    """Load k-space data for SimpleMRIUNet (expects k-space input, not images)"""
    print(f"🔄 Loading data from {file_path}, slice {slice_idx}")
    
    with h5py.File(file_path, 'r') as f:
        # Load k-space data
        kspace = torch.from_numpy(f['kspace'][slice_idx])  # (15, 640, 368)
        mask = torch.from_numpy(f['mask'][:])  # (368,)
        
        print(f"  📊 K-space shape: {kspace.shape}")
        print(f"  📊 Mask shape: {mask.shape}")
        print(f"  📊 K-space max: {torch.abs(kspace).max():.6f}")
        
        # Apply mask to k-space
        mask_2d = mask.unsqueeze(0).expand(kspace.shape[-2], -1)  # (640, 368)
        kspace_masked = kspace * mask_2d.unsqueeze(0)  # (15, 640, 368)
        
        print(f"  📊 Masked k-space max: {torch.abs(kspace_masked).max():.6f}")
        
        # Convert k-space to real/imaginary format for SimpleMRIUNet
        # SimpleMRIUNet expects (batch, coils, 2, height, width)
        kspace_real_imag = torch.stack([kspace_masked.real, kspace_masked.imag], dim=1)  # (15, 2, 640, 368)
        kspace_real_imag = kspace_real_imag.unsqueeze(0)  # (1, 15, 2, 640, 368)
        
        # Expand mask to 2D
        mask_2d = mask.unsqueeze(0).expand(kspace.shape[-2], -1)  # (640, 368)
        
        print(f"  📊 K-space real/imag shape: {kspace_real_imag.shape}")
        print(f"  📊 Mask 2D shape: {mask_2d.shape}")
        
        # Also create reference image using debug method
        coil_images = fft_forward_norm(kspace_masked)
        coil_images = apply_scaling(coil_images, scale_factor=1e6)
        rss_image = root_sum_of_squares(coil_images, dim=0)
        
        return {
            'kspace_masked': kspace_real_imag,  # For SimpleMRIUNet input
            'mask': mask_2d,                   # 2D mask
            'reference_image': rss_image,      # For comparison
            'kspace_full': kspace_real_imag    # Same as masked for now
        }

def run_unet_inference():
    """Run UNet inference with fixed preprocessing"""
    print("🚀 RUNNING FIXED UNET INFERENCE")
    print("=" * 60)
    
    # Paths
    model_path = "/scratch/vchaurasia/organized_models/unet_epoch20.pth"
    data_file = "/scratch/vchaurasia/fastmri_data/test/file1000998.h5"
    output_dir = Path("/scratch/vchaurasia/data_debug")
    
    # Load model
    print("📋 Loading UNet model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract model config
    config = checkpoint['config']['model']
    print(f"  📊 Model config: {config}")
    
    # Initialize model with correct arguments
    model = SimpleMRIUNet(
        n_channels=config.get('in_channels', 2),
        n_classes=config.get('out_channels', 1),
        dc_weight=1.0,
        num_dc_iterations=5
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  ✅ Model loaded from epoch {checkpoint['epoch']}")
    
    # Load and preprocess data
    data = load_and_preprocess_data(data_file)
    
    # Run inference with SimpleMRIUNet (takes k-space input)
    print("🧠 Running SimpleMRIUNet inference...")
    with torch.no_grad():
        output = model(
            kspace_masked=data['kspace_masked'],
            mask=data['mask'],
            kspace_full=data['kspace_full']
        )
        
        prediction = output['output'].squeeze()  # Remove batch dimension
        
        print(f"  📊 Prediction shape: {prediction.shape}")
        print(f"  📊 Prediction range: [{prediction.min():.6f}, {prediction.max():.6f}]")
    
    # Create visualization
    print("📊 Creating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Reference image (zero-filled)
    axes[0].imshow(data['reference_image'].numpy(), cmap='gray')
    axes[0].set_title('Zero-filled Reference')
    axes[0].axis('off')
    
    # SimpleMRIUNet reconstruction
    axes[1].imshow(prediction.numpy(), cmap='gray')
    axes[1].set_title('SimpleMRIUNet Output')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save results
    output_file = output_dir / "simplemriuNet_inference_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Results saved to: {output_file}")
    
    # Save numerical results
    results = {
        'prediction': prediction.numpy(),
        'reference_image': data['reference_image'].numpy(),
    }
    
    np.savez_compressed(output_dir / "simplemriuNet_inference_data.npz", **results)
    print(f"✅ Numerical data saved to: {output_dir}/simplemriuNet_inference_data.npz")
    
    # Print summary statistics
    print("\n📊 INFERENCE SUMMARY:")
    print("=" * 40)
    print(f"Reference image max:  {data['reference_image'].max():.6f}")
    print(f"Reference image mean: {data['reference_image'].mean():.6f}")
    print(f"Prediction max:       {prediction.max():.6f}")
    print(f"Prediction mean:      {prediction.mean():.6f}")
    print(f"Difference max:       {(prediction - data['reference_image']).abs().max():.6f}")
    
    # Check if prediction makes sense
    if prediction.max() > 0.001:
        print("✅ SUCCESS: SimpleMRIUNet is producing reasonable outputs!")
    else:
        print("❌ WARNING: SimpleMRIUNet outputs are still very small")
    
    return True

if __name__ == "__main__":
    try:
        success = run_unet_inference()
        if success:
            print("\n🎉 UNet inference completed successfully!")
            print("Check the generated images to verify quality.")
        else:
            print("\n❌ UNet inference failed!")
    except Exception as e:
        print(f"\n💥 Error during inference: {e}")
        import traceback
        traceback.print_exc()