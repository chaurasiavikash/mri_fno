#!/usr/bin/env python3
"""
Fixed U-Net specific inference script.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np

import torch
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_loader import FastMRIDataset
from unet_model import MRIUNetModel
from utils import (
    load_config, set_seed, setup_logging, get_device,
    create_directory, save_tensor_as_image, real_to_complex, ifft2c, root_sum_of_squares
)


def compute_metrics(prediction, target):
    """Compute evaluation metrics."""
    pred_np = prediction.squeeze().numpy()
    target_np = target.squeeze().numpy()
    
    # PSNR
    mse = np.mean((pred_np - target_np) ** 2)
    if mse > 0:
        max_val = np.max(target_np)
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # SSIM (simplified)
    mu1, mu2 = np.mean(pred_np), np.mean(target_np)
    sigma1, sigma2 = np.var(pred_np), np.var(target_np)
    sigma12 = np.mean((pred_np - mu1) * (target_np - mu2))
    
    c1, c2 = (0.01 * np.max(target_np)) ** 2, (0.03 * np.max(target_np)) ** 2
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    
    # NMSE
    nmse = mse / np.mean(target_np ** 2)
    
    # MAE
    mae = np.mean(np.abs(pred_np - target_np))
    
    return {
        'psnr': float(psnr),
        'ssim': float(ssim),
        'nmse': float(nmse),
        'mae': float(mae)
    }


def main():
    """Main U-Net inference function."""
    parser = argparse.ArgumentParser(description="Run U-Net Inference")
    parser.add_argument('--model', type=str, required=True, help='Path to U-Net model')
    parser.add_argument('--config', type=str, default='configs/single_test.yaml')
    parser.add_argument('--output', type=str, default='outputs/unet_inference')
    parser.add_argument('--max-samples', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Load config and setup
    config = load_config(args.config)
    device = get_device(args.device)
    set_seed(config['system']['seed'])
    
    # Setup logging
    create_directory(args.output)
    setup_logging(os.path.join(args.output, 'logs'), config['system']['log_level'])
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting U-Net inference on {device}")
    
    # Load U-Net model
    logger.info("Loading U-Net model...")
    checkpoint = torch.load(args.model, map_location=device)
    
    # Create U-Net config
    unet_config = {
        'in_channels': 2,
        'out_channels': 2,  # Fixed: should be 2 based on debug output
        'features': 64,
        'bilinear': False
    }
    
    logger.info(f"Using U-Net config: {unet_config}")
    
    # Create model
    model = MRIUNetModel(
        unet_config=unet_config,
        use_data_consistency=False,
        dc_weight=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("U-Net model loaded successfully")
    
    # Load dataset
    data_config = config['data']
    dataset = FastMRIDataset(
        data_path=data_config['test_path'],
        acceleration=data_config['acceleration'],
        center_fraction=data_config['center_fraction'],
        transform=None,
        use_seed=True,
        seed=config['system']['seed']
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Setup output directories
    results_dir = Path(args.output) / "reconstructions"
    metrics_dir = Path(args.output) / "metrics"
    create_directory(results_dir)
    create_directory(metrics_dir)
    
    # Run inference
    all_metrics = {'psnr': [], 'ssim': [], 'nmse': [], 'mae': [], 'inference_time': []}
    
    num_samples = min(args.max_samples, len(dataset))
    
    logger.info(f"Starting inference on {num_samples} samples...")
    
    for i in tqdm(range(num_samples), desc="U-Net Inference"):
        sample = dataset[i]
        
        start_time = time.time()
        
        # Prepare U-Net input
        kspace_masked = sample['kspace_masked']
        mask = sample['mask']
        target = sample['target']
        
        # Convert to image domain
        kspace_complex = real_to_complex(kspace_masked.unsqueeze(0))
        coil_images = ifft2c(kspace_complex)
        combined_image = root_sum_of_squares(coil_images, dim=1).squeeze(1)
        
        # Create U-Net input (magnitude and phase)
        magnitude = torch.abs(combined_image).unsqueeze(0).unsqueeze(0)
        phase = torch.angle(combined_image).unsqueeze(0).unsqueeze(0)
        unet_input = torch.cat([magnitude, phase], dim=1).to(device)
        
        # Prepare mask
        unet_mask = mask.unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            output_dict = model(unet_input, unet_mask)
            # Extract the main reconstruction from the dictionary
            reconstruction = output_dict['output'].squeeze().cpu()
        
        inference_time = time.time() - start_time
        
        # Compute metrics
        metrics = compute_metrics(reconstruction, target)
        metrics['inference_time'] = inference_time
        
        # Collect metrics
        for metric_name, metric_value in metrics.items():
            if metric_name in all_metrics:
                all_metrics[metric_name].append(metric_value)
        
        # Save results
        save_tensor_as_image(
            reconstruction,
            results_dir / f"unet_reconstruction_{i:04d}.png",
            f"U-Net Reconstruction {i}"
        )
        
        save_tensor_as_image(
            target,
            results_dir / f"target_{i:04d}.png",
            f"Target {i}"
        )
        
        # Save zero-filled for comparison
        zero_filled = torch.abs(combined_image).cpu()
        save_tensor_as_image(
            zero_filled,
            results_dir / f"zero_filled_{i:04d}.png",
            f"Zero-filled {i}"
        )
        
        # Save metrics
        metrics_file = metrics_dir / f"metrics_{i:04d}.txt"
        with open(metrics_file, 'w') as f:
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name.upper()}: {metric_value:.6f}\n")
        
        logger.info(f"Sample {i}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, Time={metrics['inference_time']:.3f}s")
    
    # Save summary
    summary_file = Path(args.output) / "summary_metrics.txt"
    with open(summary_file, 'w') as f:
        f.write("U-Net Inference Summary\n")
        f.write("=" * 30 + "\n\n")
        
        for metric in ['psnr', 'ssim', 'nmse', 'mae', 'inference_time']:
            if metric in all_metrics and all_metrics[metric]:
                values = all_metrics[metric]
                f.write(f"{metric.upper()}:\n")
                f.write(f"  Mean: {np.mean(values):.6f}\n")
                f.write(f"  Std:  {np.std(values):.6f}\n")
                f.write(f"  Min:  {np.min(values):.6f}\n")
                f.write(f"  Max:  {np.max(values):.6f}\n\n")
    
    # Save raw metrics
    np.savez_compressed(
        Path(args.output) / "all_metrics.npz",
        **all_metrics
    )
    
    logger.info("U-Net inference completed!")
    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
