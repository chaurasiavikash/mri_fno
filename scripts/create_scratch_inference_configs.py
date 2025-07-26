#!/usr/bin/env python3
"""
Create inference configuration files for DISCO and UNet models.
All paths point to scratch directory to save space.
File: scripts/create_scratch_inference_configs.py
"""

import yaml
from pathlib import Path

def create_disco_inference_config():
    """Create inference config for DISCO model (scratch-only)."""
    config = {
        'data': {
            'train_path': "/scratch/vchaurasia/fastmri_data/train",
            'val_path': "/scratch/vchaurasia/fastmri_data/val", 
            'test_path': "/scratch/vchaurasia/fastmri_data/test",
            'acceleration': 4,
            'center_fraction': 0.08,
            'num_workers': 4,
            'batch_size': 1,  # Inference typically uses batch size 1
            'mask_type': "random"
        },
        'model': {
            'name': "disco_neural_operator",
            'in_channels': 2,
            'out_channels': 2,
            'hidden_channels': 64,
            'num_layers': 4,
            'modes': 12,
            'width': 64,
            'dropout': 0.1,
            'use_residual': True,
            'activation': "gelu"
        },
        'loss': {
            'type': "combined",
            'l1_weight': 1.0,
            'ssim_weight': 0.1,
            'perceptual_weight': 0.0,
            'data_consistency_weight': 0.5
        },
        'evaluation': {
            'metrics': ["psnr", "ssim", "nmse", "mae", "nrmse"],
            'save_images': True,
            'save_raw_data': True,
            'num_test_images': 10
        },
        'system': {
            'device': "cuda",
            'seed': 42,
            'num_threads': 8,
            'log_level': "INFO"
        },
        'logging': {
            'use_wandb': False,
            'project_name': "disco_inference",
            'log_dir': "/scratch/vchaurasia/organized_models/inference_logs",
            'save_model_dir': "/scratch/vchaurasia/organized_models",
            'results_dir': "/scratch/vchaurasia/organized_models/inference_results"
        }
    }
    
    return config

def create_unet_inference_config():
    """Create inference config for UNet model (scratch-only)."""
    config = {
        'data': {
            'train_path': "/scratch/vchaurasia/fastmri_data/train",
            'val_path': "/scratch/vchaurasia/fastmri_data/val",
            'test_path': "/scratch/vchaurasia/fastmri_data/test", 
            'acceleration': 4,
            'center_fraction': 0.08,
            'num_workers': 4,
            'batch_size': 1,
            'mask_type': "random"
        },
        'model': {
            'name': "unet_cnn",
            'in_channels': 2,
            'out_channels': 2,
            'features': 64,
            'depth': 4,
            'dropout': 0.1,
            'use_residual': False,
            'activation': "relu"
        },
        'loss': {
            'type': "combined",
            'l1_weight': 1.0,
            'ssim_weight': 0.1,
            'perceptual_weight': 0.0,
            'data_consistency_weight': 0.5
        },
        'evaluation': {
            'metrics': ["psnr", "ssim", "nmse", "mae", "nrmse"],
            'save_images': True,
            'save_raw_data': True,
            'num_test_images': 10
        },
        'system': {
            'device': "cuda",
            'seed': 42,
            'num_threads': 8,
            'log_level': "INFO"
        },
        'logging': {
            'use_wandb': False,
            'project_name': "unet_inference",
            'log_dir': "/scratch/vchaurasia/organized_models/inference_logs",
            'save_model_dir': "/scratch/vchaurasia/organized_models", 
            'results_dir': "/scratch/vchaurasia/organized_models/inference_results"
        }
    }
    
    return config

def main():
    """Create inference config files for scratch directory."""
    print("Creating scratch-only inference configurations...")
    
    # Ensure configs directory exists
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Create DISCO inference config
    disco_config = create_disco_inference_config()
    disco_config_path = configs_dir / "disco_inference_scratch.yaml"
    with open(disco_config_path, 'w') as f:
        yaml.dump(disco_config, f, default_flow_style=False, indent=2)
    
    print(f"✓ Created {disco_config_path}")
    
    # Create UNet inference config
    unet_config = create_unet_inference_config()
    unet_config_path = configs_dir / "unet_inference_scratch.yaml"
    with open(unet_config_path, 'w') as f:
        yaml.dump(unet_config, f, default_flow_style=False, indent=2)
    
    print(f"✓ Created {unet_config_path}")
    
    print("\n✅ Scratch-only inference configs created successfully!")
    print("\nAll outputs will be saved to:")
    print("  📁 Models: /scratch/vchaurasia/organized_models/")
    print("  📊 Results: /scratch/vchaurasia/organized_models/inference_results/")
    print("  📝 Logs: /scratch/vchaurasia/organized_models/inference_logs/")
    
    print("\nNext steps:")
    print("1. Submit DISCO inference job")
    print("2. Submit UNet inference job")
    print("3. Run comparison analysis")

if __name__ == "__main__":
    main()