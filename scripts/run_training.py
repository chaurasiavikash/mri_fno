# File: scripts/run_training.py

#!/usr/bin/env python3
"""
Convenient script to run MRI reconstruction training.

This script provides a simple interface to start training with different
configurations and handles common setup tasks.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from train import main as train_main


def setup_training_environment():
    """Setup training environment and check requirements."""
    # Check if CUDA is available
    import torch
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA not available, using CPU")
    
    # Check memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {gpu_memory:.1f} GB")


def main():
    """Main function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train MRI Reconstruction Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python scripts/run_training.py

  # Training with custom config
  python scripts/run_training.py --config configs/my_config.yaml

  # Resume from checkpoint
  python scripts/run_training.py --resume outputs/models/checkpoint_epoch_10.pth

  # Training on CPU
  python scripts/run_training.py --device cpu

  # Custom data paths
  python scripts/run_training.py \\
    --train-data /path/to/train \\
    --val-data /path/to/val \\
    --test-data /path/to/test
        """
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    
    # Model and training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    
    # Data paths (optional overrides)
    parser.add_argument('--train-data', type=str, default=None,
                       help='Path to training data (overrides config)')
    parser.add_argument('--val-data', type=str, default=None,
                       help='Path to validation data (overrides config)')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Path to test data (overrides config)')
    
    # Output directories
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Base output directory (overrides config)')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name for organizing outputs')
    
    # Debugging and development
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (smaller dataset, more logging)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - setup only, no training')
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling')
    
    args = parser.parse_args()
    
    # Setup environment
    print("Setting up training environment...")
    setup_training_environment()
    
    # Modify config file if needed
    if any([args.epochs, args.batch_size, args.learning_rate, 
           args.train_data, args.val_data, args.test_data, args.output_dir]):
        
        from utils import load_config
        import yaml
        import tempfile
        
        # Load original config
        config = load_config(args.config)
        
        # Apply overrides
        if args.epochs:
            config['training']['epochs'] = args.epochs
        if args.batch_size:
            config['data']['batch_size'] = args.batch_size
        if args.learning_rate:
            config['training']['learning_rate'] = args.learning_rate
        if args.train_data:
            config['data']['train_path'] = args.train_data
        if args.val_data:
            config['data']['val_path'] = args.val_data
        if args.test_data:
            config['data']['test_path'] = args.test_data
        
        # Handle output directory and experiment name
        if args.output_dir or args.experiment_name:
            base_dir = args.output_dir or "outputs"
            if args.experiment_name:
                base_dir = os.path.join(base_dir, args.experiment_name)
            
            config['logging']['log_dir'] = os.path.join(base_dir, "logs")
            config['logging']['save_model_dir'] = os.path.join(base_dir, "models")
            config['logging']['results_dir'] = os.path.join(base_dir, "results")
        
        # Debug mode adjustments
        if args.debug:
            config['training']['epochs'] = min(config['training']['epochs'], 5)
            config['data']['batch_size'] = min(config['data']['batch_size'], 2)
            config['system']['log_level'] = 'DEBUG'
            config['training']['validate_every'] = 1
            config['training']['save_every'] = 1
        
        # Save modified config to temporary file
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config, temp_config, default_flow_style=False)
        temp_config.close()
        
        # Update args to use temporary config
        args.config = temp_config.name
    
    # Prepare arguments for train.py
    train_args = [
        '--config', args.config,
        '--device', args.device
    ]
    
    if args.resume:
        train_args.extend(['--resume', args.resume])
    
    # Set up sys.argv for the train script
    original_argv = sys.argv.copy()
    sys.argv = ['train.py'] + train_args
    
    try:
        if args.dry_run:
            print("Dry run mode - would start training with:")
            print(f"  Config: {args.config}")
            print(f"  Device: {args.device}")
            if args.resume:
                print(f"  Resume: {args.resume}")
            print("Dry run completed successfully!")
        else:
            # Start training
            if args.profile:
                import cProfile
                import pstats
                
                pr = cProfile.Profile()
                pr.enable()
                
                train_main()
                
                pr.disable()
                
                # Save profiling results
                profile_dir = "outputs/profiling"
                os.makedirs(profile_dir, exist_ok=True)
                
                pr.dump_stats(os.path.join(profile_dir, "training_profile.prof"))
                
                # Print top functions
                stats = pstats.Stats(pr)
                stats.sort_stats('cumulative')
                stats.print_stats(20)
            else:
                train_main()
    
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
        
        # Clean up temporary config file
        if hasattr(args, 'config') and args.config.endswith('.yaml') and 'tmp' in args.config:
            try:
                os.unlink(args.config)
            except:
                pass


if __name__ == "__main__":
    main()