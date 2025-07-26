#!/usr/bin/env python3
"""
Inspect the UNet model checkpoint to understand its structure.
File: scripts/inspect_unet_model.py
"""

import torch
import sys
from pathlib import Path

def inspect_unet_checkpoint():
    """Inspect the UNet model checkpoint."""
    
    model_path = "/scratch/vchaurasia/organized_models/unet_epoch20.pth"
    
    print("=== UNET MODEL INSPECTION ===")
    print(f"Model: {model_path}")
    print()
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("📦 Checkpoint keys:")
        for key in checkpoint.keys():
            print(f"  - {key}")
        print()
        
        # Check config if available
        if 'config' in checkpoint:
            print("⚙️ Model config from checkpoint:")
            config = checkpoint['config']
            if 'model' in config:
                for key, value in config['model'].items():
                    print(f"  {key}: {value}")
            print()
        
        # Check model state dict structure
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("🧠 Model architecture (state dict keys):")
            
            # Group keys by component
            unet_keys = []
            other_keys = []
            
            for key in state_dict.keys():
                if 'unet' in key.lower() or 'conv' in key.lower() or 'down' in key.lower() or 'up' in key.lower():
                    unet_keys.append(key)
                else:
                    other_keys.append(key)
            
            print("  UNet-related layers:")
            for key in sorted(unet_keys)[:10]:  # Show first 10
                shape = state_dict[key].shape
                print(f"    {key}: {shape}")
            
            if len(unet_keys) > 10:
                print(f"    ... and {len(unet_keys) - 10} more UNet layers")
            
            if other_keys:
                print("  Other components:")
                for key in sorted(other_keys)[:5]:
                    shape = state_dict[key].shape  
                    print(f"    {key}: {shape}")
            
            print(f"  Total parameters: {len(state_dict)} layers")
        
        # Check training info
        if 'epoch' in checkpoint:
            print(f"📊 Training info:")
            print(f"  Epoch: {checkpoint['epoch']}")
            
        if 'best_val_loss' in checkpoint:
            print(f"  Best validation loss: {checkpoint['best_val_loss']}")
            
        print()
        
        # Try to determine the model architecture
        print("🔍 Architecture analysis:")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Check for specific layer patterns
            has_down_layers = any('down' in key for key in state_dict.keys())
            has_up_layers = any('up' in key for key in state_dict.keys())
            has_bottleneck = any('bottleneck' in key for key in state_dict.keys())
            
            print(f"  Has down layers: {has_down_layers}")
            print(f"  Has up layers: {has_up_layers}")
            print(f"  Has bottleneck: {has_bottleneck}")
            
            # Check input/output channels
            first_layer_key = list(state_dict.keys())[0]
            if 'weight' in first_layer_key:
                first_weight = state_dict[first_layer_key]
                if len(first_weight.shape) == 4:  # Conv layer
                    in_channels = first_weight.shape[1]
                    out_channels = first_weight.shape[0]
                    print(f"  First layer: {in_channels} → {out_channels} channels")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False
    
    return True

def main():
    """Main inspection."""
    success = inspect_unet_checkpoint()
    
    if success:
        print("✅ Checkpoint inspected successfully!")
        print()
        print("💡 Next steps:")
        print("1. Use this info to create correct UNet inference script")
        print("2. Match the model architecture from training")
        print("3. Focus on DISCO results first (it's running successfully)")
    else:
        print("❌ Could not inspect checkpoint")

if __name__ == "__main__":
    main()