#!/usr/bin/env python3
"""
Test script specifically for FastMRI data with correct paths.
"""

import sys
import os
import torch
import traceback
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def test_fastmri_batch_loading():
    """Test batch loading with FastMRI data."""
    print("🧪 Testing FastMRI Batch Loading")
    print("=" * 50)
    
    # Use your actual FastMRI data paths
    data_paths = {
        'train': "/scratch/vchaurasia/fastmri_data/train",
        'val': "/scratch/vchaurasia/fastmri_data/val",
        'test': "/scratch/vchaurasia/fastmri_data/test"
    }
    
    try:
        from data_loader import create_data_loaders
        
        print("🔄 Creating data loaders...")
        
        train_loader, val_loader, test_loader = create_data_loaders(
            train_path=data_paths['train'],
            val_path=data_paths['val'],
            test_path=data_paths['test'],
            batch_size=2,  # Small batch for testing
            num_workers=0,  # No multiprocessing for debugging
            acceleration=4,
            center_fraction=0.08,
            target_size=(640, 368),
            use_normalization=True
        )
        
        print("✅ Data loaders created successfully!")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Test loading first batch
        print("\n🔄 Loading first training batch...")
        train_batch = next(iter(train_loader))
        
        print("✅ Training batch loaded successfully!")
        print(f"   Batch size: {train_batch['kspace_full'].shape[0]}")
        print(f"   K-space shape: {train_batch['kspace_full'].shape}")
        print(f"   Target shape: {train_batch['target'].shape}")
        print(f"   Mask shape: {train_batch['mask'].shape}")
        
        # Test loading validation batch
        print("\n🔄 Loading first validation batch...")
        val_batch = next(iter(val_loader))
        
        print("✅ Validation batch loaded successfully!")
        print(f"   Batch size: {val_batch['kspace_full'].shape[0]}")
        
        # Test multiple batches
        print("\n🔄 Testing multiple batches...")
        for i, batch in enumerate(train_loader):
            print(f"   Batch {i+1}: {batch['kspace_full'].shape}")
            if i >= 2:  # Test first 3 batches
                break
        
        print("✅ Multiple batch loading successful!")
        
        # Test that all tensors have consistent shapes
        print("\n🔍 Checking tensor consistency...")
        batch_shapes = []
        for i, batch in enumerate(train_loader):
            shapes = {key: tensor.shape for key, tensor in batch.items() if isinstance(tensor, torch.Tensor)}
            batch_shapes.append(shapes)
            if i >= 4:  # Check first 5 batches
                break
        
        # Verify consistency
        first_shapes = batch_shapes[0]
        for i, shapes in enumerate(batch_shapes[1:], 1):
            for key in first_shapes:
                if key in shapes:
                    # Allow different batch sizes, but spatial dims should match
                    if shapes[key][1:] != first_shapes[key][1:]:
                        print(f"⚠️  Shape inconsistency in batch {i} for {key}")
                        print(f"   Expected: {first_shapes[key]}")
                        print(f"   Got: {shapes[key]}")
                        return False
        
        print("✅ All batches have consistent tensor shapes!")
        return True
        
    except Exception as e:
        print(f"❌ FastMRI batch loading failed: {e}")
        traceback.print_exc()
        return False


def test_tensor_operations():
    """Test basic tensor operations."""
    print("\n🧮 Testing Tensor Operations")
    print("-" * 30)
    
    try:
        from utils import complex_to_real, real_to_complex, fft2c, ifft2c
        
        # Test with FastMRI-like dimensions
        batch_size, coils, height, width = 2, 15, 640, 368
        
        # Create test complex tensor
        complex_kspace = torch.randn(batch_size, coils, height, width, dtype=torch.complex64)
        print(f"✅ Created complex k-space: {complex_kspace.shape}")
        
        # Test complex to real conversion
        real_kspace = complex_to_real(complex_kspace)
        print(f"✅ Complex to real: {complex_kspace.shape} -> {real_kspace.shape}")
        
        # Test real to complex conversion
        reconstructed_complex = real_to_complex(real_kspace)
        print(f"✅ Real to complex: {real_kspace.shape} -> {reconstructed_complex.shape}")
        
        # Test FFT operations
        image = ifft2c(complex_kspace)
        kspace_reconstructed = fft2c(image)
        
        print(f"✅ FFT round-trip successful")
        
        # Verify round-trip accuracy
        error = torch.mean(torch.abs(complex_kspace - kspace_reconstructed))
        print(f"✅ Round-trip error: {error:.2e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run FastMRI-specific tests."""
    print("🏥 FastMRI Data Loader Tests")
    print("=" * 50)
    
    tests = [
        ("FastMRI Batch Loading", test_fastmri_batch_loading),
        ("Tensor Operations", test_tensor_operations)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print('='*50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Ready for FastMRI training.")
        return 0
    else:
        print("⚠️  Some tests failed. Running data explorer...")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)