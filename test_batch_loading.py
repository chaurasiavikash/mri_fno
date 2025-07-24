#!/usr/bin/env python3
"""
Test script to verify batch loading works with variable image sizes.
"""

import sys
sys.path.append('src')

import torch
from data_loader import create_data_loaders

def test_batch_loading():
    """Test that we can load batches without size errors."""
    print("=== Testing Batch Loading ===")
    
    try:
        # Create data loaders with batch size > 1
        train_loader, val_loader, test_loader = create_data_loaders(
            train_path="/scratch/vchaurasia/fastmri_data/test",  # Use test data for quick test
            val_path="/scratch/vchaurasia/fastmri_data/test",
            test_path="/scratch/vchaurasia/fastmri_data/test",
            batch_size=4,  # Test with multiple samples
            num_workers=2,  # Test with workers
            acceleration=4,
            center_fraction=0.08,
            use_normalization=True
        )
        
        print(f"✅ Data loaders created successfully")
        print(f"   Train batches: {len(train_loader)}")
        
        # Test loading first batch
        print("=== Testing First Batch ===")
        batch = next(iter(train_loader))
        
        print("✅ Batch loaded successfully!")
        print(f"   Batch size: {batch['kspace_full'].shape[0]}")
        print(f"   K-space shape: {batch['kspace_full'].shape}")
        print(f"   Target shape: {batch['target'].shape}")
        print(f"   Mask shape: {batch['mask'].shape}")
        
        # Test multiple batches
        print("=== Testing Multiple Batches ===")
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Test first 3 batches
                break
            print(f"   Batch {i}: {batch['kspace_full'].shape}")
        
        print("🎉 All batch loading tests passed!")
        
    except Exception as e:
        print(f"❌ Batch loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_batch_loading()
    if success:
        print("\n✅ Ready for full training!")
    else:
        print("\n❌ Fix issues before submitting job")