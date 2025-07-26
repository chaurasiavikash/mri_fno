#!/usr/bin/env python3
"""
Check the actual FastMRI data structure and contents.
File: scripts/check_fastmri_data.py
"""

import os
import h5py
import numpy as np
from pathlib import Path

def check_fastmri_data():
    """Check the structure and contents of FastMRI data."""
    
    base_path = Path("/scratch/vchaurasia/fastmri_data")
    
    print("=== FASTMRI DATA STRUCTURE CHECK ===")
    print(f"Base path: {base_path}")
    print()
    
    # Check if base directory exists
    if not base_path.exists():
        print(f"❌ Base directory not found: {base_path}")
        return
    
    print("✅ Base directory exists")
    print()
    
    # Check subdirectories
    subdirs = ['train', 'val', 'test']
    
    for subdir in subdirs:
        subdir_path = base_path / subdir
        print(f"📁 Checking {subdir}/ directory:")
        
        if subdir_path.exists():
            print(f"  ✅ Directory exists: {subdir_path}")
            
            # Count .h5 files
            h5_files = list(subdir_path.glob("*.h5"))
            print(f"  📊 Number of .h5 files: {len(h5_files)}")
            
            if h5_files:
                print(f"  📋 First few files:")
                for i, file in enumerate(h5_files[:5]):
                    size = file.stat().st_size / (1024**3)  # GB
                    print(f"    {i+1}. {file.name} ({size:.2f} GB)")
                
                if len(h5_files) > 5:
                    print(f"    ... and {len(h5_files) - 5} more files")
                
                # Try to examine the first file
                print(f"  🔍 Examining first file: {h5_files[0].name}")
                try:
                    examine_h5_file(h5_files[0])
                except Exception as e:
                    print(f"    ❌ Error reading file: {e}")
            else:
                print("  ❌ No .h5 files found")
        else:
            print(f"  ❌ Directory not found: {subdir_path}")
        
        print()

def examine_h5_file(file_path):
    """Examine the contents of an H5 file."""
    print(f"    📂 File structure:")
    
    with h5py.File(file_path, 'r') as f:
        print(f"      Keys: {list(f.keys())}")
        
        # Check k-space data
        if 'kspace' in f:
            kspace = f['kspace']
            print(f"      K-space shape: {kspace.shape}")
            print(f"      K-space dtype: {kspace.dtype}")
            print(f"      K-space is complex: {np.iscomplexobj(kspace[0])}")
            
            # Show some statistics
            sample_slice = kspace[0]  # First slice
            print(f"      Sample slice shape: {sample_slice.shape}")
            
        # Check reconstruction data
        if 'reconstruction' in f:
            recon = f['reconstruction']
            print(f"      Reconstruction shape: {recon.shape}")
            print(f"      Reconstruction dtype: {recon.dtype}")
        
        # Check attributes/metadata
        if f.attrs:
            print(f"      Attributes: {dict(f.attrs)}")

def main():
    """Main function."""
    check_fastmri_data()
    
    print("=== SUMMARY ===")
    print("This shows you the actual structure of your FastMRI data.")
    print("Look for:")
    print("  - 'kspace' dataset: Raw frequency domain data")
    print("  - 'reconstruction' dataset: Pre-computed images (optional)")
    print("  - Complex data types: K-space should be complex64/128")
    print()
    print("If you see k-space data, that's the raw MRI scanner measurements!")

if __name__ == "__main__":
    main()