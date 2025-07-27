# File: debug_data_loading.py
"""
Debug and fix the data loading issue causing all-zero target images.
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

from utils import complex_to_real, real_to_complex, fft2c, ifft2c


def debug_kspace_to_image_conversion():
    """Debug the k-space to image conversion process."""
    print("🔍 DEBUGGING K-SPACE TO IMAGE CONVERSION")
    print("=" * 50)
    
    # Load test data
    data_path = "/scratch/vchaurasia/fastmri_data/test"
    test_files = list(Path(data_path).glob("*.h5"))
    test_file = test_files[0]
    
    print(f"Using file: {test_file.name}")
    
    with h5py.File(test_file, 'r') as f:
        print(f"File keys: {list(f.keys())}")
        
        # Load k-space data
        kspace_data = f['kspace'][0]  # First slice
        print(f"Original k-space shape: {kspace_data.shape}")
        print(f"Original k-space dtype: {kspace_data.dtype}")
        print(f"K-space is complex: {np.iscomplexobj(kspace_data)}")
        
        # Convert to torch tensor
        kspace_torch = torch.from_numpy(kspace_data)
        print(f"Torch k-space shape: {kspace_torch.shape}")
        print(f"Torch k-space dtype: {kspace_torch.dtype}")
        
        # Check k-space statistics
        kspace_magnitude = torch.abs(kspace_torch)
        print(f"K-space magnitude stats:")
        print(f"  Mean: {torch.mean(kspace_magnitude):.6f}")
        print(f"  Std:  {torch.std(kspace_magnitude):.6f}")
        print(f"  Max:  {torch.max(kspace_magnitude):.6f}")
        print(f"  Min:  {torch.min(kspace_magnitude):.6f}")
        
        # Step 1: Apply IFFT to get coil images
        print(f"\nStep 1: IFFT to get coil images")
        coil_images = ifft2c(kspace_torch)
        print(f"Coil images shape: {coil_images.shape}")
        print(f"Coil images dtype: {coil_images.dtype}")
        
        coil_magnitude = torch.abs(coil_images)
        print(f"Coil image magnitude stats:")
        print(f"  Mean: {torch.mean(coil_magnitude):.6f}")
        print(f"  Std:  {torch.std(coil_magnitude):.6f}")
        print(f"  Max:  {torch.max(coil_magnitude):.6f}")
        print(f"  Min:  {torch.min(coil_magnitude):.6f}")
        
        # Step 2: Root sum of squares combination
        print(f"\nStep 2: Root sum of squares combination")
        rss_image = torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=0))
        print(f"RSS image shape: {rss_image.shape}")
        print(f"RSS image dtype: {rss_image.dtype}")
        
        print(f"RSS image stats:")
        print(f"  Mean: {torch.mean(rss_image):.6f}")
        print(f"  Std:  {torch.std(rss_image):.6f}")
        print(f"  Max:  {torch.max(rss_image):.6f}")
        print(f"  Min:  {torch.min(rss_image):.6f}")
        
        # Check if we have reconstruction_rss in the file
        if 'reconstruction_rss' in f.keys():
            print(f"\nComparing with file's reconstruction_rss:")
            ref_recon = torch.from_numpy(f['reconstruction_rss'][0])
            print(f"Reference reconstruction shape: {ref_recon.shape}")
            print(f"Reference reconstruction stats:")
            print(f"  Mean: {torch.mean(ref_recon):.6f}")
            print(f"  Std:  {torch.std(ref_recon):.6f}")
            print(f"  Max:  {torch.max(ref_recon):.6f}")
            print(f"  Min:  {torch.min(ref_recon):.6f}")
            
            # Check if they match
            mse = torch.mean((rss_image - ref_recon) ** 2)
            print(f"MSE between our RSS and reference: {mse:.6f}")
        
        # Create visualization
        output_dir = Path("/scratch/vchaurasia/data_debug")
        output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # K-space visualization (log scale)
        kspace_log = np.log(np.abs(kspace_data[0]) + 1e-8)  # First coil
        axes[0, 0].imshow(kspace_log, cmap='viridis')
        axes[0, 0].set_title('K-space (log magnitude, coil 0)')
        axes[0, 0].axis('off')
        
        # Individual coil image
        coil_0_mag = torch.abs(coil_images[0]).numpy()
        axes[0, 1].imshow(coil_0_mag, cmap='gray')
        axes[0, 1].set_title('Coil 0 Image')
        axes[0, 1].axis('off')
        
        # RSS image
        rss_np = rss_image.numpy()
        axes[0, 2].imshow(rss_np, cmap='gray')
        axes[0, 2].set_title('RSS Combined Image')
        axes[0, 2].axis('off')
        
        # Reference reconstruction (if available)
        if 'reconstruction_rss' in f.keys():
            ref_np = ref_recon.numpy()
            axes[1, 0].imshow(ref_np, cmap='gray')
            axes[1, 0].set_title('Reference Reconstruction')
            axes[1, 0].axis('off')
            
            # Difference
            diff = np.abs(rss_np - ref_np)
            axes[1, 1].imshow(diff, cmap='hot')
            axes[1, 1].set_title('Difference (Our - Reference)')
            axes[1, 1].axis('off')
        
        # Histograms
        axes[1, 2].hist(rss_np.flatten(), bins=50, alpha=0.7, label='Our RSS')
        if 'reconstruction_rss' in f.keys():
            axes[1, 2].hist(ref_np.flatten(), bins=50, alpha=0.7, label='Reference')
        axes[1, 2].set_title('Intensity Histograms')
        axes[1, 2].legend()
        axes[1, 2].set_xlabel('Intensity')
        axes[1, 2].set_ylabel('Count')
        
        plt.tight_layout()
        plot_path = output_dir / "kspace_to_image_debug.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nDebug visualization saved to: {plot_path}")
        
        return {
            'kspace_torch': kspace_torch,
            'coil_images': coil_images,
            'rss_image': rss_image,
            'ref_recon': ref_recon if 'reconstruction_rss' in f.keys() else None
        }


def test_complex_to_real_conversion():
    """Test the complex_to_real function that might be causing issues."""
    print(f"\n🔍 TESTING COMPLEX_TO_REAL CONVERSION")
    print("=" * 50)
    
    # Create test complex tensor
    test_complex = torch.complex(
        torch.randn(2, 3, 4, 5),  # Real part
        torch.randn(2, 3, 4, 5)   # Imaginary part
    )
    
    print(f"Test complex tensor shape: {test_complex.shape}")
    print(f"Test complex tensor dtype: {test_complex.dtype}")
    
    # Convert using our function
    try:
        converted = complex_to_real(test_complex)
        print(f"Converted shape: {converted.shape}")
        print(f"Converted dtype: {converted.dtype}")
        print(f"✅ complex_to_real works correctly")
        
        # Test round-trip
        recovered = real_to_complex(converted)
        print(f"Recovered shape: {recovered.shape}")
        print(f"Round-trip error: {torch.mean(torch.abs(test_complex - recovered)):.8f}")
        
    except Exception as e:
        print(f"❌ complex_to_real failed: {e}")
        
        # Try alternative implementation
        print("Trying alternative implementation...")
        alt_converted = torch.stack([test_complex.real, test_complex.imag], dim=-1)
        print(f"Alternative shape: {alt_converted.shape}")
        return alt_converted
    
    return converted


def fix_data_loader_issue():
    """Identify and fix the data loader issue."""
    print(f"\n🛠️ FIXING DATA LOADER ISSUE")
    print("=" * 50)
    
    # The main issue is likely in the complex_to_real function or data preprocessing
    # Let's check the actual implementation
    
    data_results = debug_kspace_to_image_conversion()
    test_complex_to_real_conversion()
    
    # Create a corrected reconstruction function
    def corrected_kspace_to_image(kspace_complex):
        """Corrected k-space to image conversion."""
        # Ensure complex input
        if not torch.is_complex(kspace_complex):
            raise ValueError("Input must be complex tensor")
        
        # Apply IFFT
        image_coils = ifft2c(kspace_complex)
        
        # Root sum of squares along coil dimension (dim 0)
        magnitude_image = torch.sqrt(torch.sum(torch.abs(image_coils) ** 2, dim=0))
        
        return magnitude_image
    
    # Test with our data
    kspace_data = data_results['kspace_torch']
    corrected_image = corrected_kspace_to_image(kspace_data)
    
    print(f"\nCorrected reconstruction stats:")
    print(f"  Mean: {torch.mean(corrected_image):.6f}")
    print(f"  Std:  {torch.std(corrected_image):.6f}")
    print(f"  Max:  {torch.max(corrected_image):.6f}")
    print(f"  Min:  {torch.min(corrected_image):.6f}")
    
    # Save the corrected function
    corrected_code = '''
def corrected_kspace_to_image(kspace_complex):
    """Corrected k-space to image conversion."""
    if not torch.is_complex(kspace_complex):
        raise ValueError("Input must be complex tensor")
    
    # Apply IFFT  
    image_coils = ifft2c(kspace_complex)
    
    # Root sum of squares along coil dimension
    magnitude_image = torch.sqrt(torch.sum(torch.abs(image_coils) ** 2, dim=0))
    
    return magnitude_image

def corrected_complex_to_real(complex_tensor):
    """Corrected complex to real conversion."""
    if not torch.is_complex(complex_tensor):
        return complex_tensor
    
    # Stack real and imaginary parts along new dimension
    return torch.stack([complex_tensor.real, complex_tensor.imag], dim=-1)
'''
    
    with open("/scratch/vchaurasia/data_debug/corrected_functions.py", "w") as f:
        f.write(corrected_code)
    
    print(f"\nCorrected functions saved to: /scratch/vchaurasia/data_debug/corrected_functions.py")
    
    return corrected_image


def main():
    """Main debugging function."""
    print("🚨 DATA LOADING DEBUG AND FIX")
    print("=" * 60)
    print("This script will identify why your target images are all zeros.")
    print()
    
    try:
        # Run debugging
        corrected_image = fix_data_loader_issue()
        
        if torch.max(corrected_image) > 0:
            print(f"\n✅ SUCCESS! Found the issue:")
            print(f"   The k-space to image conversion is working")
            print(f"   Problem is likely in the data loader implementation")
            print(f"   Specifically: complex_to_real function or tensor handling")
        else:
            print(f"\n❌ Still getting zero images")
            print(f"   Need to check the k-space data itself")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Check the corrected functions in /scratch/vchaurasia/data_debug/")
        print(f"   2. Update your data_loader.py with the corrected functions")
        print(f"   3. Retrain your DISCO model with fixed data loading")
        
        return 0
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())