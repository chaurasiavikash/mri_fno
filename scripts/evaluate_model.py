# File: scripts/evaluate_model.py

#!/usr/bin/env python3
"""
Convenient script to run comprehensive MRI reconstruction evaluation.

This script provides a simple interface to evaluate and compare multiple
trained models with comprehensive analysis and reporting.
"""

import os
import sys
import argparse
import glob
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluate import main as evaluate_main


def find_models_in_directory(directory: str, pattern: str = "best_model.pth") -> list:
    """
    Find model checkpoints in directory structure.
    
    Args:
        directory: Base directory to search
        pattern: Pattern to match for model files
        
    Returns:
        List of found model paths
    """
    directory = Path(directory)
    
    # Look for pattern in subdirectories
    model_paths = []
    
    # Direct pattern match
    direct_matches = list(directory.glob(pattern))
    model_paths.extend(direct_matches)
    
    # Search in subdirectories
    for subdir in directory.iterdir():
        if subdir.is_dir():
            subdir_matches = list(subdir.glob(pattern))
            model_paths.extend(subdir_matches)
    
    return [str(path) for path in model_paths]


def auto_generate_model_names(model_paths: list) -> list:
    """
    Auto-generate meaningful names for models based on their paths.
    
    Args:
        model_paths: List of model file paths
        
    Returns:
        List of generated model names
    """
    names = []
    
    for path in model_paths:
        path_obj = Path(path)
        
        # Try to extract meaningful name from path
        if 'best_model' in path_obj.name:
            # Use parent directory name
            name = path_obj.parent.name
        else:
            # Use filename without extension
            name = path_obj.stem
        
        # Clean up name
        name = name.replace('_', ' ').replace('-', ' ').title()
        
        # Ensure unique names
        if name in names:
            counter = 2
            original_name = name
            while name in names:
                name = f"{original_name} {counter}"
                counter += 1
        
        names.append(name)
    
    return names


def main():
    """Main function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description="Comprehensive MRI Reconstruction Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate multiple specific models
  python scripts/evaluate_model.py \\
    --models outputs/exp1/models/best_model.pth outputs/exp2/models/best_model.pth \\
    --names "Experiment 1" "Experiment 2"

  # Auto-discover and evaluate all models in a directory
  python scripts/evaluate_model.py \\
    --model-dir outputs \\
    --auto-names

  # Evaluate with custom test data
  python scripts/evaluate_model.py \\
    --models outputs/models/best_model.pth \\
    --names "My Model" \\
    --test-data /path/to/test/data

  # Quick evaluation on limited samples
  python scripts/evaluate_model.py \\
    --models outputs/models/best_model.pth \\
    --names "My Model" \\
    --max-samples 20

  # Compare with baseline methods
  python scripts/evaluate_model.py \\
    --models outputs/models/best_model.pth \\
    --names "Neural Operator" \\
    --baseline-results baseline_results.json
        """
    )
    
    # Model specification
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--models', type=str, nargs='+',
                           help='Paths to trained model checkpoints')
    model_group.add_argument('--model-dir', type=str,
                           help='Directory to search for models (auto-discovery)')
    
    # Model naming
    parser.add_argument('--names', type=str, nargs='+',
                       help='Names for the models (must match number of models)')
    parser.add_argument('--auto-names', action='store_true',
                       help='Auto-generate model names from paths')
    
    # Configuration and data
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Path to test data (overrides config)')
    
    # Evaluation options
    parser.add_argument('--output', type=str, default='outputs/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    
    # Comparison options
    parser.add_argument('--baseline-results', type=str, default=None,
                       help='Path to baseline results file (JSON format)')
    
    # Advanced options
    parser.add_argument('--acceleration', type=int, default=None,
                       help='Acceleration factor (overrides config)')
    parser.add_argument('--center-fraction', type=float, default=None,
                       help='Center fraction (overrides config)')
    
    # Output control
    parser.add_argument('--no-html-report', action='store_true',
                       help='Skip HTML report generation')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    # Model discovery options
    parser.add_argument('--model-pattern', type=str, default='best_model.pth',
                       help='Pattern to match when auto-discovering models')
    parser.add_argument('--include-checkpoints', action='store_true',
                       help='Include epoch checkpoints in auto-discovery')
    
    args = parser.parse_args()
    
    # Resolve model paths
    if args.model_dir:
        print(f"Auto-discovering models in: {args.model_dir}")
        
        patterns = [args.model_pattern]
        if args.include_checkpoints:
            patterns.extend(['checkpoint_epoch_*.pth', 'model_*.pth'])
        
        model_paths = []
        for pattern in patterns:
            found_models = find_models_in_directory(args.model_dir, pattern)
            model_paths.extend(found_models)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_models = []
        for model in model_paths:
            if model not in seen:
                seen.add(model)
                unique_models.append(model)
        
        model_paths = unique_models
        
        if not model_paths:
            print(f"Error: No models found in {args.model_dir}")
            print(f"Searched for patterns: {patterns}")
            sys.exit(1)
        
        print(f"Found {len(model_paths)} models:")
        for i, path in enumerate(model_paths, 1):
            print(f"  {i}. {path}")
    
    else:
        model_paths = args.models
    
    # Resolve model names
    if args.auto_names:
        model_names = auto_generate_model_names(model_paths)
        print("Auto-generated model names:")
        for name, path in zip(model_names, model_paths):
            print(f"  {name}: {path}")
    
    elif args.names:
        if len(args.names) != len(model_paths):
            print(f"Error: Number of names ({len(args.names)}) must match number of models ({len(model_paths)})")
            sys.exit(1)
        model_names = args.names
    
    else:
        # Generate simple names
        if len(model_paths) == 1:
            model_names = ["Model"]
        else:
            model_names = [f"Model {i+1}" for i in range(len(model_paths))]
    
    # Verify all models exist
    for path in model_paths:
        if not os.path.exists(path):
            print(f"Error: Model checkpoint not found: {path}")
            sys.exit(1)
    
    # Modify config if needed
    config_overrides = {}
    if args.test_data:
        config_overrides['test_path'] = args.test_data
    if args.acceleration:
        config_overrides['acceleration'] = args.acceleration
    if args.center_fraction:
        config_overrides['center_fraction'] = args.center_fraction
    
    # Apply config overrides
    if config_overrides:
        from utils import load_config
        import yaml
        import tempfile
        
        config = load_config(args.config)
        
        for key, value in config_overrides.items():
            config['data'][key] = value
        
        if args.verbose:
            config['system']['log_level'] = 'DEBUG'
        
        # Save modified config
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config, temp_config, default_flow_style=False)
        temp_config.close()
        
        config_path = temp_config.name
    else:
        config_path = args.config
    
    # Prepare arguments for evaluate.py
    eval_args = [
        '--config', config_path,
        '--models'] + model_paths + [
        '--names'] + model_names + [
        '--output', args.output,
        '--device', args.device
    ]
    
    if args.max_samples:
        eval_args.extend(['--max-samples', str(args.max_samples)])
    
    if args.baseline_results:
        eval_args.extend(['--baseline-results', args.baseline_results])
    
    # Set up sys.argv for the evaluation script
    original_argv = sys.argv.copy()
    sys.argv = ['evaluate.py'] + eval_args
    
    try:
        print("\nStarting comprehensive evaluation...")
        print(f"Models to evaluate: {len(model_paths)}")
        for name, path in zip(model_names, model_paths):
            print(f"  - {name}: {path}")
        print(f"Output directory: {args.output}")
        if args.max_samples:
            print(f"Max samples: {args.max_samples}")
        
        evaluate_main()
        
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {args.output}")
        
        # Show key result files
        output_path = Path(args.output)
        key_files = [
            "performance_report.html",
            "evaluation_results/all_metrics.json",
            "plots/metric_boxplots.png",
            "comparisons/model_comparison.txt"
        ]
        
        print("\nKey result files:")
        for file_name in key_files:
            file_path = output_path / file_name
            if file_path.exists():
                print(f"  - {file_path}")
            else:
                print(f"  - {file_path} (not found)")
    
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        raise
    
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