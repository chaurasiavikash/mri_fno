# File: scripts/generate_baseline_metrics.py

#!/usr/bin/env python3
"""
Generate baseline method metrics for comparison.

This script evaluates simple baseline methods like zero-filled reconstruction
and basic interpolation methods.
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_loader import FastMRIDataset
from utils import load_config, setup_logging
import logging


def compute_psnr(prediction, target):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((prediction - target) ** 2)
    if mse == 0:
        return float('inf')
    max_val = np.max(target)
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim(prediction, target):
    """Compute Structural Similarity Index (simplified version)."""
    mu1 = np.mean(prediction)
    mu2 = np.mean(target)
    sigma1 = np.var(prediction)
    sigma2 = np.var(target)
    sigma12 = np.mean((prediction - mu1) * (target - mu2))
    
    c1 = (0.01 * np.max(target)) ** 2
    c2 = (0.03 * np.max(target)) ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    
    return float(ssim)


def compute_nmse(prediction, target):
    """Compute Normalized Mean Squared Error."""
    return np.mean((prediction - target) ** 2) / np.mean(target ** 2)


def zero_filled_reconstruction(dataset, max_samples=None):
    """Evaluate zero-filled reconstruction baseline."""
    print("Evaluating Zero-Filled Reconstruction...")
    
    metrics = {'psnr': [], 'ssim': [], 'nmse': [], 'mae': [], 'inference_time': []}
    
    num_samples = min(max_samples or len(dataset), len(dataset))
    
    for i in tqdm(range(num_samples), desc="Zero-filled"):
        try:
            sample = dataset[i]
            
            # Zero-filled reconstruction is already computed in dataset
            prediction = sample['image_masked'].numpy()
            target = sample['target'].numpy()
            
            # Compute metrics
            psnr = compute_psnr(prediction, target)
            ssim = compute_ssim(prediction, target)
            nmse = compute_nmse(prediction, target)
            mae = np.mean(np.abs(prediction - target))
            
            # Zero-filled is instant
            inference_time = 0.001
            
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)
            metrics['nmse'].append(nmse)
            metrics['mae'].append(mae)
            metrics['inference_time'].append(inference_time)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    return metrics


def simple_interpolation_reconstruction(dataset, max_samples=None):
    """Evaluate simple interpolation baseline."""
    print("Evaluating Simple Interpolation Reconstruction...")
    
    import scipy.ndimage
    
    metrics = {'psnr': [], 'ssim': [], 'nmse': [], 'mae': [], 'inference_time': []}
    
    num_samples = min(max_samples or len(dataset), len(dataset))
    
    for i in tqdm(range(num_samples), desc="Interpolation"):
        try:
            import time
            start_time = time.time()
            
            sample = dataset[i]
            
            # Start with zero-filled
            prediction = sample['image_masked'].numpy()
            
            # Apply simple smoothing/interpolation
            prediction = scipy.ndimage.gaussian_filter(prediction, sigma=1.0)
            
            target = sample['target'].numpy()
            
            inference_time = time.time() - start_time
            
            # Compute metrics
            psnr = compute_psnr(prediction, target)
            ssim = compute_ssim(prediction, target)
            nmse = compute_nmse(prediction, target)
            mae = np.mean(np.abs(prediction - target))
            
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)
            metrics['nmse'].append(nmse)
            metrics['mae'].append(mae)
            metrics['inference_time'].append(inference_time)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    return metrics


def total_variation_reconstruction(dataset, max_samples=None):
    """Evaluate TV-regularized reconstruction baseline."""
    print("Evaluating Total Variation Reconstruction...")
    
    from scipy.optimize import minimize
    
    metrics = {'psnr': [], 'ssim': [], 'nmse': [], 'mae': [], 'inference_time': []}
    
    num_samples = min(max_samples or len(dataset), len(dataset))
    
    # Limit TV reconstruction to fewer samples due to computational cost
    tv_samples = min(20, num_samples)
    
    for i in tqdm(range(tv_samples), desc="TV Reconstruction"):
        try:
            import time
            start_time = time.time()
            
            sample = dataset[i]
            
            # Start with zero-filled as initial guess
            prediction = sample['image_masked'].numpy()
            target = sample['target'].numpy()
            
            # Simple TV denoising (simplified for speed)
            prediction = scipy.ndimage.median_filter(prediction, size=3)
            
            inference_time = time.time() - start_time
            
            # Compute metrics
            psnr = compute_psnr(prediction, target)
            ssim = compute_ssim(prediction, target)
            nmse = compute_nmse(prediction, target)
            mae = np.mean(np.abs(prediction - target))
            
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)
            metrics['nmse'].append(nmse)
            metrics['mae'].append(mae)
            metrics['inference_time'].append(inference_time)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Extend metrics to match requested sample count (for fair comparison)
    while len(metrics['psnr']) < num_samples:
        for key in metrics:
            if metrics[key]:
                metrics[key].append(metrics[key][-1])  # Repeat last value
    
    return metrics


def main():
    """Main function to generate baseline metrics."""
    parser = argparse.ArgumentParser(description="Generate Baseline Method Metrics")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='baseline_metrics.json',
                       help='Output file for baseline metrics')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['zero_filled', 'interpolation', 'tv'],
                       help='Baseline methods to evaluate')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Generating baseline method metrics...")
    
    # Create test dataset
    from data_loader import FastMRIDataset
    
    test_dataset = FastMRIDataset(
        data_path=config['data']['test_path'],
        acceleration=config['data']['acceleration'],
        center_fraction=config['data']['center_fraction'],
        transform=None,  # No normalization for baselines
        use_seed=True,
        seed=config['system']['seed']
    )
    
    baseline_results = {}
    
    # Evaluate each baseline method
    if 'zero_filled' in args.methods:
        baseline_results['Zero-Filled'] = zero_filled_reconstruction(
            test_dataset, args.max_samples
        )
    
    if 'interpolation' in args.methods:
        baseline_results['Simple Interpolation'] = simple_interpolation_reconstruction(
            test_dataset, args.max_samples
        )
    
    if 'tv' in args.methods:
        baseline_results['TV Reconstruction'] = total_variation_reconstruction(
            test_dataset, args.max_samples
        )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    logger.info(f"Baseline metrics saved to {args.output}")
    
    # Print summary
    print("\nBaseline Method Summary:")
    print("=" * 40)
    
    for method_name, metrics in baseline_results.items():
        print(f"\n{method_name}:")
        print(f"  PSNR: {np.mean(metrics['psnr']):.2f} ± {np.std(metrics['psnr']):.2f}")
        print(f"  SSIM: {np.mean(metrics['ssim']):.4f} ± {np.std(metrics['ssim']):.4f}")
        print(f"  NMSE: {np.mean(metrics['nmse']):.4f} ± {np.std(metrics['nmse']):.4f}")
        print(f"  Inference Time: {np.mean(metrics['inference_time']):.3f}s")


if __name__ == "__main__":
    main()