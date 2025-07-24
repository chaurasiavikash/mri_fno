# File: scripts/run_inference.py

#!/usr/bin/env python3
"""
Convenient script to run MRI reconstruction inference.

This script provides a simple interface to run inference on trained models
with various options for output and visualization.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from infer import main as infer_main


def find_best_model(model_dir: str) -> str:
    """
    Find the best model checkpoint in a directory.
    
    Args:
        model_dir: Directory containing model checkpoints
        
    Returns:
        Path to the best model checkpoint
    """
    model_dir = Path(model_dir)
    
    # Look for best_model.pth first
    best_model = model_dir / "best_model.pth"
    if best_model.exists():
        return str(best_model)
    
    # Look for latest checkpoint
    checkpoints = list(model_dir.glob("checkpoint_epoch_*.pth"))
    if checkpoints:
        # Sort by epoch number
        def extract_epoch(path):
            try:
                return int(path.stem.split('_')[-1])
            except:
                return -1
        
        latest_checkpoint = max(checkpoints, key=extract_epoch)
        return str(latest_checkpoint)
    
    raise FileNotFoundError(f"No model checkpoints found in {model_dir}")


def main():
    """Main function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run MRI Reconstruction Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference with best model
  python scripts/run_inference.py --model outputs/models/best_model.pth

  # Inference with auto-detection of best model
  python scripts/run_inference.py --model-dir outputs/models

  # Inference with custom test data
  python scripts/run_inference.py \\
    --model outputs/models/best_model.pth \\
    --test-data /path/to/test/data

  # Quick inference on limited samples
  python scripts/run_inference.py \\
    --model outputs/models/best_model.pth \\
    --max-samples 10 \\
    --visualize 0 1 2

  # Inference with custom output directory
  python scripts/run_inference.py \\
    --model outputs/models/best_model.pth \\
    --output outputs/inference_results
        """
    )
    
    # Model specification
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model', type=str,
                           help='Path to trained model checkpoint')
    model_group.add_argument('--model-dir', type=str,
                           help='Directory containing model checkpoints (auto-select best)')
    
    # Configuration and data
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Path to test data (overrides config)')
    
    # Inference options
    parser.add_argument('--output', type=str, default='outputs/inference',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference')
    
    # Output options
    parser.add_argument('--save-all', action='store_true', default=True,
                       help='Save all reconstruction results')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save individual results (only metrics)')
    parser.add_argument('--save-raw', action='store_true',
                       help='Save raw numpy arrays in addition to images')
    
    # Visualization options
    parser.add_argument('--visualize', type=int, nargs='*', default=[0, 1, 2],
                       help='Sample indices to visualize (default: first 3)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization generation')
    
    # Advanced options
    parser.add_argument('--acceleration', type=int, default=None,
                       help='Acceleration factor (overrides config)')
    parser.add_argument('--center-fraction', type=float, default=None,
                       help='Center fraction (overrides config)')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of data loading workers (overrides config)')
    
    # Performance and debugging
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run inference benchmark (timing analysis)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Resolve model path
    if args.model_dir:
        try:
            model_path = find_best_model(args.model_dir)
            print(f"Auto-selected model: {model_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        model_path = args.model
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found: {model_path}")
        sys.exit(1)
    
    # Modify config if needed
    config_overrides = {}
    if args.test_data:
        config_overrides['test_path'] = args.test_data
    if args.acceleration:
        config_overrides['acceleration'] = args.acceleration
    if args.center_fraction:
        config_overrides['center_fraction'] = args.center_fraction
    if args.num_workers:
        config_overrides['num_workers'] = args.num_workers
    if args.batch_size != 1:
        config_overrides['batch_size'] = args.batch_size
    
    # Apply config overrides
    if config_overrides:
        from utils import load_config
        import yaml
        import tempfile
        
        config = load_config(args.config)
        
        # Apply data overrides
        for key, value in config_overrides.items():
            if key in ['test_path', 'acceleration', 'center_fraction', 'num_workers', 'batch_size']:
                config['data'][key] = value
        
        # Enable verbose logging if requested
        if args.verbose:
            config['system']['log_level'] = 'DEBUG'
        
        # Save raw data option
        if args.save_raw:
            config['evaluation']['save_raw_data'] = True
        
        # Save modified config to temporary file
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config, temp_config, default_flow_style=False)
        temp_config.close()
        
        config_path = temp_config.name
    else:
        config_path = args.config
    
    # Prepare arguments for infer.py
    infer_args = [
        '--config', config_path,
        '--model', model_path,
        '--output', args.output,
        '--device', args.device
    ]
    
    if args.max_samples:
        infer_args.extend(['--max-samples', str(args.max_samples)])
    
    if args.no_save:
        infer_args.append('--no-save-all')
    elif args.save_all:
        infer_args.append('--save-all')
    
    if not args.no_visualize and args.visualize is not None:
        infer_args.extend(['--visualize'] + [str(x) for x in args.visualize])
    
    # Set up sys.argv for the inference script
    original_argv = sys.argv.copy()
    sys.argv = ['infer.py'] + infer_args
    
    try:
        print("Starting inference...")
        print(f"Model: {model_path}")
        print(f"Output: {args.output}")
        if args.max_samples:
            print(f"Max samples: {args.max_samples}")
        
        if args.profile:
            import cProfile
            import pstats
            
            pr = cProfile.Profile()
            pr.enable()
            
            infer_main()
            
            pr.disable()
            
            # Save profiling results
            profile_dir = os.path.join(args.output, "profiling")
            os.makedirs(profile_dir, exist_ok=True)
            
            pr.dump_stats(os.path.join(profile_dir, "inference_profile.prof"))
            
            # Print top functions
            stats = pstats.Stats(pr)
            stats.sort_stats('cumulative')
            print("\nTop 20 functions by cumulative time:")
            stats.print_stats(20)
            
        elif args.benchmark:
            import time
            
            start_time = time.time()
            infer_main()
            end_time = time.time()
            
            total_time = end_time - start_time
            print(f"\nBenchmark Results:")
            print(f"Total inference time: {total_time:.2f} seconds")
            
            # Calculate throughput if max_samples is known
            if args.max_samples:
                throughput = args.max_samples / total_time
                print(f"Throughput: {throughput:.2f} samples/second")
        
        else:
            infer_main()
    
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
        
        # Clean up temporary config file
        if config_path != args.config:
            try:
                os.unlink(config_path)
            except:
                pass


if __name__ == "__main__":
    main()