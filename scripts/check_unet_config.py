#!/usr/bin/env python3
"""
Check exactly what config was saved in the UNet checkpoint.
"""

import torch
import yaml

def check_unet_config():
    """Check the UNet config saved in checkpoint."""
    
    model_path = "/scratch/vchaurasia/organized_models/unet_epoch20.pth"
    
    print("=== CHECKING UNET CONFIG FROM CHECKPOINT ===")
    print(f"Model: {model_path}")
    print()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'config' in checkpoint:
        print("✅ Config found in checkpoint!")
        config = checkpoint['config']
        
        print("\n📋 FULL CONFIG FROM CHECKPOINT:")
        print("-" * 50)
        
        # Print the full config
        def print_config(data, indent=0):
            for key, value in data.items():
                spaces = "  " * indent
                if isinstance(value, dict):
                    print(f"{spaces}{key}:")
                    print_config(value, indent + 1)
                else:
                    print(f"{spaces}{key}: {value}")
        
        print_config(config)
        
        print("\n🎯 KEY MODEL CONFIG:")
        print("-" * 30)
        if 'model' in config:
            for key, value in config['model'].items():
                print(f"  {key}: {value}")
        
        print("\n🔧 KEY LOSS CONFIG:")
        print("-" * 30)
        if 'loss' in config:
            for key, value in config['loss'].items():
                print(f"  {key}: {value}")
                
    else:
        print("❌ No config found in checkpoint")
        print("Available keys:", list(checkpoint.keys()))

def compare_with_current_config():
    """Compare with the current unet_baseline_config.yaml."""
    
    config_file = "configs/unet_baseline_config.yaml"
    
    print(f"\n=== COMPARING WITH CURRENT CONFIG FILE ===")
    print(f"File: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            current_config = yaml.safe_load(f)
        
        print("\n📋 CURRENT CONFIG FILE:")
        print("-" * 30)
        
        def print_config(data, indent=0):
            for key, value in data.items():
                spaces = "  " * indent
                if isinstance(value, dict):
                    print(f"{spaces}{key}:")
                    print_config(value, indent + 1)
                else:
                    print(f"{spaces}{key}: {value}")
        
        print_config(current_config)
        
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_file}")
    except Exception as e:
        print(f"❌ Error reading config: {e}")

def main():
    """Main function."""
    check_unet_config()
    compare_with_current_config()
    
    print("\n=== RECOMMENDATION ===")
    print("✅ Use the config from the checkpoint (guaranteed correct)")
    print("⚠️  Don't rely on current config files (might have changed)")

if __name__ == "__main__":
    main()