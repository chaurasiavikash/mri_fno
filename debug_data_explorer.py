#!/usr/bin/env python3
"""
Data explorer to understand the FastMRI data structure and debug loading issues.
"""

import os
import sys
import h5py
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def explore_data_directory(data_path: str, max_files: int = 5):
    """Explore the structure of the data directory."""
    print(f"🔍 Exploring data directory: {data_path}")
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"❌ Directory does not exist: {data_path}")
        return False
    
    # Find all files
    all_files = list(data_path.rglob("*"))
    h5_files = list(data_path.glob("*.h5"))
    
    print(f"📁 Total files in directory: {len(all_files)}")
    print(f"📄 .h5 files found: {len(h5_files)}")
    
    if len(h5_files) == 0:
        print("❌ No .h5 files found in the directory")
        print("📋 First 10 files in directory:")
        for i, file_path in enumerate(all_files[:10]):
            print(f"   {i+1}. {file_path.name} ({'dir' if file_path.is_dir() else 'file'})")
        return False
    
    print(f"📋 First {min(max_files, len(h5_files))} .h5 files:")
    for i, file_path in enumerate(h5_files[:max_files]):
        print(f"   {i+1}. {file_path.name}")
    
    return True


def examine_h5_file(file_path: str):
    """Examine the structure of an HDF5 file."""
    print(f"\n🔬 Examining file: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print("📊 File structure:")
            
            def print_structure(name, obj):
                indent = "  " * (name.count('/'))
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}📈 {name}: {obj.shape}, {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"{indent}📁 {name}/")
            
            f.visititems(print_structure)
            
            # Check for common FastMRI keys
            common_keys = ['kspace', 'reconstruction_rss', 'ismrmrd_header']
            
            print("\n🔑 Key information:")
            for key in common_keys:
                if key in f:
                    dataset = f[key]
                    print(f"   ✅ {key}: {dataset.shape}, {dataset.dtype}")
                    
                    if key == 'kspace':
                        # Additional info about k-space
                        print(f"      Slices: {dataset.shape[0]}")
                        if len(dataset.shape) >= 3:
                            print(f"      Coils: {dataset.shape[1] if len(dataset.shape) > 3 else 'N/A'}")
                        print(f"      Spatial: {dataset.shape[-2:] if len(dataset.shape) >= 2 else 'N/A'}")
                        
                        # Check if complex
                        if np.iscomplexobj(dataset[0]):
                            print(f"      ✅ Complex data detected")
                        else:
                            print(f"      ⚠️  Real data - might need special handling")
                else:
                    print(f"   ❌ {key}: Not found")
            
            return True
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False


def test_data_loading(data_path: str):
    """Test basic data loading with our dataset."""
    print(f"\n🧪 Testing data loading from: {data_path}")
    
    try:
        from data_loader import FastMRIDataset
        
        # Create dataset
        dataset = FastMRIDataset(
            data_path=data_path,
            acceleration=4,
            center_fraction=0.08,
            target_size=(640, 368),
            use_seed=True,
            seed=42
        )
        
        print(f"✅ Dataset created successfully!")
        print(f"   Total samples: {len(dataset)}")
        
        # Try to load first sample
        print("\n🔄 Loading first sample...")
        sample = dataset[0]
        
        print("✅ Sample loaded successfully!")
        print(f"   Keys: {list(sample.keys())}")
        
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}, {value.dtype}")
            elif isinstance(value, dict):
                print(f"   {key}: {type(value)} with keys {list(value.keys())}")
            else:
                print(f"   {key}: {type(value)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_creation(data_path: str):
    """Test batch creation."""
    print(f"\n🗂️  Testing batch creation from: {data_path}")
    
    try:
        from data_loader import create_data_loaders
        
        # Create data loaders with small batch
        train_loader, _, _ = create_data_loaders(
            train_path=data_path,
            val_path=data_path,  # Use same path for testing
            test_path=data_path,
            batch_size=2,
            num_workers=0,  # No multiprocessing for debugging
            acceleration=4,
            center_fraction=0.08,
            target_size=(640, 368),
            use_normalization=False  # Disable for simplicity
        )
        
        print(f"✅ DataLoader created successfully!")
        print(f"   Number of batches: {len(train_loader)}")
        
        # Try to load first batch
        print("\n🔄 Loading first batch...")
        batch = next(iter(train_loader))
        
        print("✅ Batch loaded successfully!")
        print(f"   Batch size: {batch['kspace_full'].shape[0]}")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}, {value.dtype}")
            elif isinstance(value, list):
                print(f"   {key}: list of {len(value)} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main exploration function."""
    print("🕵️  FastMRI Data Explorer and Debugger")
    print("=" * 50)
    
    # Data paths to try
    data_paths = [
        "/scratch/vchaurasia/fastmri_data/train",
        "/scratch/vchaurasia/fastmri_data/val",
        "/scratch/vchaurasia/fastmri_data/test"
    ]
    
    for data_path in data_paths:
        print(f"\n{'='*60}")
        print(f"Checking: {data_path}")
        print('='*60)
        
        # Explore directory
        if not explore_data_directory(data_path, max_files=3):
            continue
        
        # Examine first h5 file
        h5_files = list(Path(data_path).glob("*.h5"))
        if h5_files:
            examine_h5_file(str(h5_files[0]))
            
            # Test data loading
            test_data_loading(data_path)
            
            # Test batch creation
            test_batch_creation(data_path)
            
            # Only test first valid path
            break
    
    print(f"\n{'='*60}")
    print("🏁 Exploration Complete!")
    print('='*60)


if __name__ == "__main__":
    main()