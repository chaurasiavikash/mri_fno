#!/usr/bin/env python3
"""
Demonstrate k-space to image conversion using your actual FastMRI data.
File: scripts/demonstrate_kspace_conversion.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def demonstrate_conversion():
    """Show how k-space converts to images using your data."""
    
    # Use your actual test file
    test_file = "/scratch/vchaurasia/fastmri_data/test/file1000998.h5"
    
    print("=== K-SPACE TO IMAGE CONVERSION DEMO ===")
    print(f"Using file: {Path(test_file).name}")
    print()
    
    with h5py.File(test_file, 'r') as f:
        print("📊 File contents:")
        for key in f.keys():
            print(f"  - {key}: {f[key].shape}")
        print()
        
        # Get k-space data (first slice)
        kspace_data = f['kspace'][0]  # Shape: (15, 640, 368) - 15 coils
        print(f"🔍 K-space data for slice 0:")
        print(f"  Shape: {kspace_data.shape}")
        print(f"  Data type: {kspace_data.dtype}")
        print(f"  Is complex: {np.iscomplexobj(kspace_data)}")
        print()
        
        # Step 1: Apply IFFT to each coil
        print("⚡ Step 1: Converting k-space to image domain (IFFT)")
        coil_images = np.fft.ifftshift(
            np.fft.ifft2(
                np.fft.fftshift(kspace_data, axes=(-2, -1)), 
                axes=(-2, -1)
            ), 
            axes=(-2, -1)
        )
        print(f"  Coil images shape: {coil_images.shape}")
        print(f"  Each coil image is complex: {np.iscomplexobj(coil_images)}")
        print()
        
        # Step 2: Root Sum of Squares coil combination
        print("🔗 Step 2: Combining coils (Root Sum of Squares)")
        combined_image = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))
        print(f"  Combined image shape: {combined_image.shape}")
        print(f"  Combined image is real: {not np.iscomplexobj(combined_image)}")
        print()
        
        # Compare with provided reconstruction
        if 'reconstruction_rss' in f:
            reference_image = f['reconstruction_rss'][0]  # Ground truth
            print("✅ Comparison with reference reconstruction:")
            print(f"  Reference shape: {reference_image.shape}")
            
            # Calculate difference
            diff = np.abs(combined_image - reference_image)
            relative_error = np.mean(diff) / np.mean(reference_image)
            print(f"  Mean relative error: {relative_error:.6f}")
            print(f"  Our reconstruction matches reference: {'✅' if relative_error < 0.01 else '❌'}")
        print()
        
        # Show what's happening in k-space
        print("🎯 K-space characteristics:")
        center_region = kspace_data[0, 300:340, 160:200]  # Center of k-space
        edge_region = kspace_data[0, 0:40, 0:40]  # Edge of k-space
        
        center_magnitude = np.mean(np.abs(center_region))
        edge_magnitude = np.mean(np.abs(edge_region))
        
        print(f"  Center region magnitude: {center_magnitude:.2e}")
        print(f"  Edge region magnitude: {edge_magnitude:.2e}")
        print(f"  Center/Edge ratio: {center_magnitude/edge_magnitude:.1f}x")
        print("  → Center has most energy (low frequencies = contrast)")
        print("  → Edges have fine details (high frequencies)")
        print()
        
        # Explain undersampling
        if 'mask' in f:
            mask_data = f['mask'][:]
            print("🎭 Undersampling mask analysis:")
            print(f"  Mask shape: {mask_data.shape}")
            print(f"  Sampling ratio: {np.mean(mask_data):.1%}")
            print(f"  Acceleration factor: ~{1/np.mean(mask_data):.1f}x")
        
        return combined_image, reference_image if 'reconstruction_rss' in f else None

def create_visualization():
    """Create a visualization showing the conversion process."""
    
    print("📊 Creating visualization...")
    
    # This would create plots showing:
    # 1. K-space magnitude (log scale)
    # 2. Individual coil images  
    # 3. Combined RSS image
    # 4. Comparison with reference
    
    print("  (Visualization code would go here)")
    print("  Shows: K-space → Coil Images → Combined Image")

def main():
    """Main demonstration."""
    try:
        image, reference = demonstrate_conversion()
        create_visualization()
        
        print("=== KEY TAKEAWAYS ===")
        print("✅ Your data contains RAW k-space measurements")
        print("✅ IFFT converts frequency → spatial domain")
        print("✅ RSS combines multiple coil data")
        print("✅ This process creates the 'ground truth' images")
        print("✅ Neural networks learn to do this from undersampled data")
        print()
        print("🎯 The Challenge:")
        print("Given incomplete k-space (undersampled), can neural networks")
        print("reconstruct images as good as the fully-sampled version?")
        
    except FileNotFoundError:
        print("❌ Test file not found. Check your data path.")
        print("Available files:")
        test_dir = Path("/scratch/vchaurasia/fastmri_data/test")
        if test_dir.exists():
            for file in list(test_dir.glob("*.h5"))[:5]:
                print(f"  - {file.name}")

if __name__ == "__main__":
    main()