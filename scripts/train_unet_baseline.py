# File: scripts/train_unet_baseline.py

#!/usr/bin/env python3
"""
Training script for U-Net baseline comparison.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import training pipeline but use U-Net model
from train import MRITrainer
from unet_model import MRIUNetModel
from model import ReconstructionLoss
from utils import load_config, get_device, set_seed, setup_logging, create_directory


class UNetTrainer(MRITrainer):
    """Modified trainer for U-Net baseline."""
    
    def _setup_model(self):
        """Setup U-Net model instead of DISCO."""
        model_config = self.config['model']
        
        self.model = MRIUNetModel(
            unet_config={
                'in_channels': model_config['in_channels'],
                'out_channels': model_config['out_channels'],
                'features': model_config['features']
            },
            use_data_consistency=True,
            dc_weight=self.config['loss']['data_consistency_weight'],
            num_dc_iterations=5
        ).to(self.device)
        
        # Enable mixed precision if requested
        if self.config['training']['mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None


def main():
    """Main U-Net training function."""
    parser = argparse.ArgumentParser(description="Train U-Net Baseline Model")
    parser.add_argument('--config', type=str, default='configs/unet_baseline_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override epochs if specified
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Setup device and seed
    device = get_device(args.device)
    set_seed(config['system']['seed'])
    
    # Setup logging
    setup_logging(
        config['logging']['log_dir'],
        config['system']['log_level']
    )
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Starting U-Net baseline training on {device}")
    logger.info(f"Configuration: {args.config}")
    
    # Create trainer
    trainer = UNetTrainer(config, device)
    
    # Start training
    try:
        trainer.train()
        logger.info("U-Net training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint("unet_checkpoint_interrupted.pth")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        trainer.save_checkpoint("unet_checkpoint_error.pth")
        raise


if __name__ == "__main__":
    main()