# File: deep_fft_debug.py
"""
Deep debug of FFT functions - test all possible implementations and scaling.
The k-space values are small (max=0.000166) but images are still zero.
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


def test_all_fft_implementations():
    """Test all possible FFT implementations and scaling."""
    print("🔍 TESTING ALL FFT IMPLEMENTATIONS")
    print("=" * 60)
    
    # Load the k-space data with "good" values
    data_path = "/scratch/vchaurasia/fastmri_data/test/file1000998.h5"
    
    with h5py.File(data_path, 'r') as f:
        kspace_data = f['kspace'][0]  # First slice
        
        print(f"Raw k-space shape: {kspace_data.shape}")
        print(f"Raw k-space dtype: {kspace_data.dtype}")
        print(f"Raw k-space max: {np.max(np.abs(kspace_data)):.10f}")
        print(f"Raw k-space mean: {np.mean(np.abs(kspace_data)):.10f}")
        print(f"Raw k-space std: {np.std(np.abs(kspace_data)):.10f}")
        
        # Check if any values are non-zero
        non_zero_count = np.count_nonzero(kspace_data)
        total_count = kspace_data.size
        print(f"Non-zero elements: {non_zero_count}/{total_count} ({100*non_zero_count/total_count:.2f}%)")
        
        if non_zero_count == 0:
            print("❌ K-space is completely zero - data is corrupted")
            return test_with_synthetic_data()
        
        # Convert to torch
        kspace_torch = torch.from_numpy(kspace_data)
        
        # Test different FFT implementations
        fft_implementations = []
        
        # 1. Basic torch.fft.ifft2
        def fft_basic(x):
            return torch.fft.ifft2(x)
        
        # 2. Centered IFFT (current implementation)
        def fft_centered(x):
            return torch.fft.fftshift(
                torch.fft.ifft2(
                    torch.fft.ifftshift(x, dim=(-2, -1)), 
                    dim=(-2, -1)
                ), 
                dim=(-2, -1)
            )
        
        # 3. With ortho normalization
        def fft_ortho(x):
            return torch.fft.ifft2(x, norm='ortho')
        
        # 4. Centered + ortho
        def fft_centered_ortho(x):
            return torch.fft.fftshift(
                torch.fft.ifft2(
                    torch.fft.ifftshift(x, dim=(-2, -1)), 
                    dim=(-2, -1),
                    norm='ortho'
                ), 
                dim=(-2, -1)
            )
        
        # 5. Forward normalization
        def fft_forward_norm(x):
            return torch.fft.ifft2(x, norm='forward')
        
        # 6. Different shift order
        def fft_shift_first(x):
            return torch.fft.ifft2(
                torch.fft.fftshift(x, dim=(-2, -1)), 
                dim=(-2, -1)
            )
        
        implementations = [
            ("Basic IFFT", fft_basic),
            ("Centered IFFT", fft_centered),
            ("Ortho normalization", fft_ortho),
            ("Centered + Ortho", fft_centered_ortho),
            ("Forward normalization", fft_forward_norm),
            ("Shift first", fft_shift_first)
        ]
        
        print(f"\n📊 TESTING {len(implementations)} FFT IMPLEMENTATIONS:")
        print("=" * 60)
        
        best_max = 0
        best_impl = None
        best_image = None
        
        for name, fft_func in implementations:
            print(f"\n{name}:")
            
            try:
                # Apply IFFT to each coil
                image_coils = fft_func(kspace_torch)
                
                print(f"  Coil images shape: {image_coils.shape}")
                print(f"  Coil images max: {torch.max(torch.abs(image_coils)):.10f}")
                print(f"  Coil images mean: {torch.mean(torch.abs(image_coils)):.10f}")
                
                # Root sum of squares
                rss_image = torch.sqrt(torch.sum(torch.abs(image_coils) ** 2, dim=0))
                
                print(f"  RSS image max: {torch.max(rss_image):.10f}")
                print(f"  RSS image mean: {torch.mean(rss_image):.10f}")
                
                max_val = torch.max(rss_image).item()
                if max_val > best_max:
                    best_max = max_val
                    best_impl = name
                    best_image = rss_image.clone()
                
                # Check for NaN/Inf
                if torch.isnan(rss_image).any():
                    print(f"  ❌ Contains NaN values")
                elif torch.isinf(rss_image).any():
                    print(f"  ❌ Contains Inf values")
                elif max_val > 1e-10:
                    print(f"  ✅ Has non-zero values")
                else:
                    print(f"  ⚠️  Very small values")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print(f"\n🏆 BEST IMPLEMENTATION: {best_impl}")
        print(f"   Best max value: {best_max:.10f}")
        
        if best_max > 1e-10:
            print("✅ Found working FFT implementation!")
            test_with_scaling(kspace_torch, best_image, best_impl)
        else:
            print("❌ All FFT implementations produce near-zero values")
            test_with_synthetic_data()


def test_with_scaling(kspace_torch, best_image, best_impl_name):
    """Test different scaling approaches."""
    print(f"\n🔧 TESTING SCALING APPROACHES")
    print("=" * 50)
    
    original_max = torch.max(best_image).item()
    
    # Test different scaling factors
    scaling_factors = [1, 1e3, 1e6, 1e9, 1e12]
    
    for scale in scaling_factors:
        scaled_kspace = kspace_torch * scale
        
        # Use the best FFT implementation
        if "Centered + Ortho" in best_impl_name:
            image_coils = torch.fft.fftshift(
                torch.fft.ifft2(
                    torch.fft.ifftshift(scaled_kspace, dim=(-2, -1)), 
                    dim=(-2, -1),
                    norm='ortho'
                ), 
                dim=(-2, -1)
            )
        else:
            image_coils = torch.fft.ifft2(scaled_kspace)
        
        rss_image = torch.sqrt(torch.sum(torch.abs(image_coils) ** 2, dim=0))
        max_val = torch.max(rss_image).item()
        
        print(f"Scale x{scale:.0e}: max = {max_val:.10f}")
        
        if max_val > 0.01:  # Reasonable image values
            print(f"✅ Found good scaling factor: {scale:.0e}")
            
            # Create visualization
            create_scaling_visualization(scaled_kspace, rss_image, scale)
            return scale
    
    print("❌ No scaling factor produces reasonable image values")
    return None


def test_with_synthetic_data():
    """Test FFT with synthetic data to verify implementation."""
    print(f"\n🧪 TESTING WITH SYNTHETIC DATA")
    print("=" * 50)
    
    # Create synthetic k-space data
    height, width = 64, 64
    
    # Create a simple test image
    test_image = torch.zeros(height, width, dtype=torch.complex64)
    test_image[height//4:3*height//4, width//4:3*width//4] = 1.0
    
    # Forward FFT to get k-space
    kspace_synthetic = torch.fft.fft2(test_image)
    
    print(f"Synthetic k-space max: {torch.max(torch.abs(kspace_synthetic)):.6f}")
    
    # Test IFFT implementations
    implementations = [
        ("Basic IFFT", lambda x: torch.fft.ifft2(x)),
        ("Centered IFFT", lambda x: torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))),
        ("Ortho IFFT", lambda x: torch.fft.ifft2(x, norm='ortho')),
    ]
    
    for name, fft_func in implementations:
        recovered = fft_func(kspace_synthetic)
        error = torch.mean(torch.abs(test_image - recovered)).item()
        
        print(f"{name}:")
        print(f"  Recovered max: {torch.max(torch.abs(recovered)):.6f}")
        print(f"  Error: {error:.10f}")
        
        if error < 1e-6:
            print(f"  ✅ Round-trip successful")
        else:
            print(f"  ❌ Round-trip failed")


def create_scaling_visualization(kspace, image, scale):
    """Create visualization of successful scaling."""
    output_dir = Path("/scratch/vchaurasia/data_debug")
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # K-space
    kspace_log = torch.log(torch.abs(kspace[0]) + 1e-12).numpy()
    axes[0].imshow(kspace_log, cmap='viridis')
    axes[0].set_title(f'K-space (log)\nScale: {scale:.0e}')
    axes[0].axis('off')
    
    # Image
    image_np = image.numpy()
    axes[1].imshow(image_np, cmap='gray')
    axes[1].set_title(f'Reconstructed Image\nMax: {torch.max(image):.6f}')
    axes[1].axis('off')
    
    # Histogram
    axes[2].hist(image_np.flatten(), bins=50, alpha=0.7)
    axes[2].set_title('Image Histogram')
    axes[2].set_xlabel('Intensity')
    axes[2].set_ylabel('Count')
    
    plt.tight_layout()
    
    plot_path = output_dir / f"successful_reconstruction_scale_{scale:.0e}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")


def check_fastmri_reference():
    """Check if there's a reference reconstruction in the file."""
    print(f"\n📋 CHECKING FOR REFERENCE RECONSTRUCTION")
    print("=" * 50)
    
    data_path = "/scratch/vchaurasia/fastmri_data/test/file1000998.h5"
    
    with h5py.File(data_path, 'r') as f:
        print(f"File keys: {list(f.keys())}")
        
        # Check if there's a reference reconstruction
        if 'reconstruction_rss' in f.keys():
            ref_recon = f['reconstruction_rss'][0]
            print(f"Reference reconstruction shape: {ref_recon.shape}")
            print(f"Reference reconstruction max: {np.max(ref_recon):.6f}")
            
            if np.max(ref_recon) > 0:
                print("✅ Reference reconstruction has real values!")
                
                # Save reference image
                output_dir = Path("/scratch/vchaurasia/data_debug")
                output_dir.mkdir(exist_ok=True)
                
                plt.figure(figsize=(8, 8))
                plt.imshow(ref_recon, cmap='gray')
                plt.title(f'Reference Reconstruction\nMax: {np.max(ref_recon):.6f}')
                plt.axis('off')
                plt.savefig(output_dir / "reference_reconstruction.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Reference saved to: {output_dir / 'reference_reconstruction.png'}")
                return ref_recon
            else:
                print("❌ Reference reconstruction is also zero")
        else:
            print("❌ No reference reconstruction in file")
        
        # Check attributes for scaling info
        if hasattr(f, 'attrs'):
            print(f"File attributes: {dict(f.attrs)}")
        
        # Check k-space attributes
        kspace = f['kspace']
        if hasattr(kspace, 'attrs'):
            print(f"K-space attributes: {dict(kspace.attrs)}")
    
    return None


def main():
    """Main function for deep FFT debugging."""
    print("🔍 DEEP FFT DEBUG - FIND THE REAL ISSUE")
    print("=" * 70)
    print("K-space has small values (max=0.000166) but images are still zero")
    print("Testing all possible FFT implementations and scaling...")
    print()
    
    try:
        # Check for reference reconstruction first
        ref_recon = check_fastmri_reference()
        
        # Test all FFT implementations
        test_all_fft_implementations()
        
        print(f"\n🎯 SUMMARY:")
        print("If all FFT implementations fail, the issue is likely:")
        print("1. K-space data is actually corrupted/empty")
        print("2. Need different scaling/normalization")
        print("3. Wrong interpretation of the data format")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())