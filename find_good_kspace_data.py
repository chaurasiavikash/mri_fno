# File: find_good_kspace_data.py
"""
Find files with actual k-space data (not corrupted/empty).
Learn from the working U-Net implementation.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def check_multiple_files():
    """Check multiple files to find good k-space data."""
    print("🔍 CHECKING MULTIPLE FILES FOR GOOD K-SPACE DATA")
    print("=" * 60)
    
    data_path = Path("/scratch/vchaurasia/fastmri_data/test")
    test_files = list(data_path.glob("*.h5"))[:10]  # Check first 10 files
    
    good_files = []
    
    for i, test_file in enumerate(test_files):
        print(f"\n📁 File {i+1}: {test_file.name}")
        
        try:
            with h5py.File(test_file, 'r') as f:
                print(f"   Keys: {list(f.keys())}")
                
                kspace = f['kspace']
                print(f"   K-space shape: {kspace.shape}")
                
                # Check multiple slices
                for slice_idx in [0, kspace.shape[0]//2, -1]:  # First, middle, last
                    if slice_idx < 0:
                        slice_idx = kspace.shape[0] - 1
                    
                    kspace_slice = kspace[slice_idx]
                    magnitude = np.abs(kspace_slice)
                    
                    max_val = np.max(magnitude)
                    mean_val = np.mean(magnitude)
                    
                    print(f"   Slice {slice_idx}: max={max_val:.6f}, mean={mean_val:.6f}")
                    
                    if max_val > 1e-4:  # Reasonable k-space values
                        good_files.append({
                            'file': test_file,
                            'slice': slice_idx,
                            'max_val': max_val,
                            'mean_val': mean_val
                        })
                        print(f"   ✅ Good data found!")
                        break
                else:
                    print(f"   ❌ All slices have very low values")
                    
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Checked {len(test_files)} files")
    print(f"   Found {len(good_files)} files with good k-space data")
    
    if good_files:
        # Sort by max value (strongest signal)
        good_files.sort(key=lambda x: x['max_val'], reverse=True)
        
        print(f"\n🏆 BEST FILES:")
        for i, file_info in enumerate(good_files[:3]):
            print(f"   {i+1}. {file_info['file'].name}")
            print(f"      Slice {file_info['slice']}: max={file_info['max_val']:.6f}")
    
    return good_files

def test_good_file(file_info):
    """Test reconstruction with a good file."""
    print(f"\n🧪 TESTING RECONSTRUCTION WITH GOOD FILE")
    print("=" * 60)
    
    file_path = file_info['file']
    slice_idx = file_info['slice']
    
    print(f"File: {file_path.name}")
    print(f"Slice: {slice_idx}")
    
    with h5py.File(file_path, 'r') as f:
        # Load k-space
        kspace_data = f['kspace'][slice_idx]
        print(f"K-space shape: {kspace_data.shape}")
        print(f"K-space dtype: {kspace_data.dtype}")
        
        # Convert to torch
        kspace_torch = torch.from_numpy(kspace_data)
        
        # Check statistics
        magnitude = torch.abs(kspace_torch)
        print(f"K-space magnitude stats:")
        print(f"  Mean: {torch.mean(magnitude):.6f}")
        print(f"  Std:  {torch.std(magnitude):.6f}")
        print(f"  Max:  {torch.max(magnitude):.6f}")
        print(f"  Min:  {torch.min(magnitude):.6f}")
        
        # Convert to image
        from utils import ifft2c
        
        # Apply IFFT
        coil_images = ifft2c(kspace_torch)
        print(f"Coil images shape: {coil_images.shape}")
        
        # Root sum of squares
        rss_image = torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=0))
        print(f"RSS image shape: {rss_image.shape}")
        
        print(f"RSS image stats:")
        print(f"  Mean: {torch.mean(rss_image):.6f}")
        print(f"  Std:  {torch.std(rss_image):.6f}")
        print(f"  Max:  {torch.max(rss_image):.6f}")
        print(f"  Min:  {torch.min(rss_image):.6f}")
        
        # Create visualization
        output_dir = Path("/scratch/vchaurasia/data_debug")
        output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # K-space (first coil, log scale)
        kspace_log = np.log(np.abs(kspace_data[0]) + 1e-8)
        axes[0, 0].imshow(kspace_log, cmap='viridis')
        axes[0, 0].set_title('K-space (log, coil 0)')
        axes[0, 0].axis('off')
        
        # Individual coil images
        coil_0 = torch.abs(coil_images[0]).numpy()
        coil_7 = torch.abs(coil_images[7]).numpy()
        
        axes[0, 1].imshow(coil_0, cmap='gray')
        axes[0, 1].set_title('Coil 0 Image')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(coil_7, cmap='gray')
        axes[0, 2].set_title('Coil 7 Image')
        axes[0, 2].axis('off')
        
        # RSS image
        rss_np = rss_image.numpy()
        axes[1, 0].imshow(rss_np, cmap='gray')
        axes[1, 0].set_title('RSS Combined Image')
        axes[1, 0].axis('off')
        
        # Center crop (common in MRI)
        center_h, center_w = rss_np.shape[0]//2, rss_np.shape[1]//2
        crop_size = 256
        h_start = max(0, center_h - crop_size//2)
        h_end = min(rss_np.shape[0], center_h + crop_size//2)
        w_start = max(0, center_w - crop_size//2)
        w_end = min(rss_np.shape[1], center_w + crop_size//2)
        
        cropped = rss_np[h_start:h_end, w_start:w_end]
        axes[1, 1].imshow(cropped, cmap='gray')
        axes[1, 1].set_title('Center Crop')
        axes[1, 1].axis('off')
        
        # Histogram
        axes[1, 2].hist(rss_np.flatten(), bins=50, alpha=0.7)
        axes[1, 2].set_title('Intensity Distribution')
        axes[1, 2].set_xlabel('Intensity')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / f"good_reconstruction_{file_path.stem}_slice{slice_idx}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {plot_path}")
        
        return {
            'kspace': kspace_torch,
            'rss_image': rss_image,
            'file': file_path,
            'slice': slice_idx
        }

def compare_with_unet_data_loading():
    """Compare with how U-Net loads data to understand the difference."""
    print(f"\n📋 COMPARING WITH U-NET DATA LOADING")
    print("=" * 60)
    
    # Let's examine the U-Net data loader
    try:
        sys.path.append("src")
        # Try to import the working data loader
        from data_loader import FastMRIDataset
        
        print("✅ Found FastMRIDataset class")
        
        # Create dataset
        dataset = FastMRIDataset(
            data_path="/scratch/vchaurasia/fastmri_data/test",
            acceleration=4,
            center_fraction=0.08
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Get a sample
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                if value.numel() > 0:
                    print(f"  Stats: mean={torch.mean(value.float()):.6f}, max={torch.max(value.float()):.6f}")
        
        return sample
        
    except Exception as e:
        print(f"❌ Could not load U-Net data loader: {e}")
        return None

def create_corrected_disco_test():
    """Create a test with corrected data loading for DISCO."""
    print(f"\n🔧 CREATING CORRECTED DISCO TEST")
    print("=" * 60)
    
    # Find good files
    good_files = check_multiple_files()
    
    if not good_files:
        print("❌ No good files found!")
        return None
    
    # Test with best file
    best_file = good_files[0]
    result = test_good_file(best_file)
    
    # Compare with U-Net loading
    unet_sample = compare_with_unet_data_loading()
    
    if result and unet_sample:
        print(f"\n🎯 COMPARISON RESULTS:")
        print(f"   Direct loading RSS max: {torch.max(result['rss_image']):.6f}")
        if 'target' in unet_sample:
            print(f"   U-Net target max: {torch.max(unet_sample['target']):.6f}")
        
        # Check if they're similar
        if 'target' in unet_sample and result['rss_image'].shape == unet_sample['target'].shape:
            diff = torch.mean(torch.abs(result['rss_image'] - unet_sample['target']))
            print(f"   Difference: {diff:.6f}")
    
    return result

def main():
    """Main function."""
    print("🔍 FINDING GOOD K-SPACE DATA FOR DISCO")
    print("=" * 70)
    print("This script will find files with actual MRI data (not empty/corrupted).")
    print()
    
    try:
        result = create_corrected_disco_test()
        
        if result:
            print(f"\n✅ SUCCESS!")
            print(f"   Found good k-space data in: {result['file'].name}")
            print(f"   Slice {result['slice']} has meaningful image data")
            print(f"   RSS image max value: {torch.max(result['rss_image']):.6f}")
            
            print(f"\n🎯 NEXT STEPS:")
            print(f"   1. Use files with good k-space data (max > 1e-4)")
            print(f"   2. Check why first file had empty k-space")
            print(f"   3. Update DISCO data loading to use proper files")
            print(f"   4. Retrain DISCO with corrected data")
        else:
            print(f"\n❌ No good data found")
            print(f"   All files seem to have very low k-space values")
            print(f"   Check if the dataset download was correct")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())