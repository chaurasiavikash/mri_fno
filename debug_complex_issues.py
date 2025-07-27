# File: debug_complex_issue.py
import torch

def test_complex_conversion():
    print("=== DEBUGGING COMPLEX CONVERSION ===")
    
    # Simulate your data
    kspace_tensor = torch.randn(1, 15, 2, 640, 368)
    print(f"Input tensor: {kspace_tensor.shape}")
    
    # Test indexing
    real_part = kspace_tensor[:, :, 0, :, :]
    imag_part = kspace_tensor[:, :, 1, :, :]
    
    print(f"Real part: {real_part.shape}")
    print(f"Imag part: {imag_part.shape}")
    
    # Test complex conversion
    complex_result = torch.complex(real_part, imag_part)
    print(f"Complex result: {complex_result.shape}")
    
    if complex_result.shape == (1, 15, 640, 368):
        print("✅ Correct shape!")
    else:
        print("❌ Wrong shape!")
    
    # Test what your current code is doing
    print(f"\n=== TESTING CURRENT CODE ===")
    print(f"kspace_tensor[:, :, 0, :, :]: {kspace_tensor[:, :, 0, :, :].shape}")
    print(f"kspace_tensor[:, :, 1, :, :]: {kspace_tensor[:, :, 1, :, :].shape}")
    
    # This should work
    test_complex = torch.complex(
        kspace_tensor[:, :, 0, :, :],
        kspace_tensor[:, :, 1, :, :]
    )
    print(f"Test complex: {test_complex.shape}")

if __name__ == "__main__":
    test_complex_conversion()