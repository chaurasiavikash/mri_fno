# File: debug_final_fix.py
import torch
import h5py

def debug_indexing_issue():
    print("=== DEBUGGING INDEXING ISSUE ===")
    
    # Load real data
    with h5py.File('/scratch/vchaurasia/fastmri_data/train/file1001434.h5', 'r') as f:
        kspace = torch.from_numpy(f['kspace'][0])  # First slice: (15, 640, 368)
        
    print(f"Raw k-space: {kspace.shape}, complex: {torch.is_complex(kspace)}")
    
    # Add batch dimension and convert to real/imag format
    kspace_batch = kspace.unsqueeze(0)  # (1, 15, 640, 368)
    kspace_real_imag = torch.stack([kspace_batch.real, kspace_batch.imag], dim=2)
    print(f"Real/imag format: {kspace_real_imag.shape}")  # Should be (1, 15, 2, 640, 368)
    
    # PROBLEM: Let's see what [..., 0] actually gives us
    print(f"\n=== DEBUGGING INDEXING ===")
    print(f"Full tensor shape: {kspace_real_imag.shape}")
    print(f"kspace_real_imag[..., 0] shape: {kspace_real_imag[..., 0].shape}")
    print(f"kspace_real_imag[..., 1] shape: {kspace_real_imag[..., 1].shape}")
    
    # This is wrong! [..., 0] takes the LAST dimension
    # We want the real/imag dimension which is dim=2
    
    print(f"\n=== CORRECT INDEXING ===")
    print(f"kspace_real_imag[:, :, 0, :, :] shape: {kspace_real_imag[:, :, 0, :, :].shape}")
    print(f"kspace_real_imag[:, :, 1, :, :] shape: {kspace_real_imag[:, :, 1, :, :].shape}")
    
    # CORRECT conversion
    kspace_complex_correct = torch.complex(
        kspace_real_imag[:, :, 0, :, :],  # Real part: (1, 15, 640, 368)
        kspace_real_imag[:, :, 1, :, :]   # Imag part: (1, 15, 640, 368)
    )
    print(f"Correctly converted complex: {kspace_complex_correct.shape}")
    
    if kspace_complex_correct.shape == (1, 15, 640, 368):
        print("✅ CORRECT! Complex conversion works!")
        
        # Now test RSS
        image_coils = torch.fft.ifftshift(
            torch.fft.ifft2(torch.fft.fftshift(kspace_complex_correct, dim=[-2, -1])), 
            dim=[-2, -1]
        )
        print(f"Image coils: {image_coils.shape}")
        
        # RSS over coil dimension (dim=1)
        image_magnitude = torch.sqrt(torch.sum(torch.abs(image_coils) ** 2, dim=1))
        print(f"RSS result: {image_magnitude.shape}")
        
        if image_magnitude.shape == (1, 640, 368):
            print("✅ PERFECT! Everything works!")
        else:
            print("❌ RSS still wrong")
    
    else:
        print("❌ Still wrong")

if __name__ == "__main__":
    debug_indexing_issue()