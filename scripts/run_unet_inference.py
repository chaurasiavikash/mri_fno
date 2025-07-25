# File: scripts/run_unet_inference.py

#!/usr/bin/env python3
"""
U-Net specific inference script.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

import torch
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_loader import FastMRIDataset
from utils import (
    load_config, set_seed, setup_logging, get_device,
    create_directory, save_tensor_as_image
)


class UNetInferenceEngine:
    """Inference engine specifically for U-Net models."""
    
    def __init__(self, config, model_path, device, output_dir):
        self.config = config
        self.model_path = model_path
        self.device = device
        self.output_dir = Path(output_dir)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup directories
        self.results_dir = self.output_dir / "reconstructions"
        self.metrics_dir = self.output_dir / "metrics"
        self.visualizations_dir = self.output_dir / "visualizations"
        
        create_directory(self.results_dir)
        create_directory(self.metrics_dir)
        create_directory(self.visualizations_dir)
        
        # Load model
        self._load_unet_model()
        
        # Setup data loader
        self._setup_data_loader()
        
        self.logger.info("U-Net inference engine initialized")
    
    def _load_unet_model(self):
        """Load U-Net model from checkpoint."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Import U-Net model
        from model_unet import UNetBaseline
        
        # Get model config from checkpoint or use defaults
        if 'config' in checkpoint and 'model' in checkpoint['config']:
            model_config = checkpoint['config']['model']
        else:
            # Fallback defaults
            model_config = {
                'in_channels': 2,
                'out_channels': 1,
                'base_channels': 32,
                'num_pool_layers': 4,
                'dropout': 0.0,
                'use_residual': True,
                'activation': 'relu'
            }
        
        # Create model
        self.model = UNetBaseline(
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            base_channels=model_config['base_channels'],
            num_pool_layers=model_config['num_pool_layers'],
            dropout=model_config['dropout'],
            use_residual=model_config['use_residual'],
            activation=model_config['activation']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.logger.info(f"U-Net model loaded from {self.model_path}")
        
        if 'epoch' in checkpoint:
            self.logger.info(f"Model trained for {checkpoint['epoch']} epochs")
    
    def _setup_data_loader(self):
        """Setup data loader."""
        data_config = self.config['data']
        
        self.test_dataset = FastMRIDataset(
            data_path=data_config['test_path'],
            acceleration=data_config['acceleration'],
            center_fraction=data_config['center_fraction'],
            transform=None,
            use_seed=True,
            seed=self.config['system']['seed']
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        self.logger.info(f"Test dataset loaded: {len(self.test_dataset)} samples")
    
    def infer_single_sample(self, kspace_masked, mask, kspace_full=None):
        """Perform inference on single sample."""
        with torch.no_grad():
            # Move to device
            kspace_masked = kspace_masked.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            
            start_time = time.time()
            
            # Convert to image domain for U-Net input
            from utils import real_to_complex, ifft2c, root_sum_of_squares, complex_to_real
            
            # Convert k-space to image
            kspace_complex = real_to_complex(kspace_masked)
            coil_images = ifft2c(kspace_complex)
            combined_image = root_sum_of_squares(coil_images, dim=1).squeeze(1)
            
            # Prepare U-Net input (magnitude and phase or real/imag)
            input_magnitude = torch.abs(combined_image).unsqueeze(1)
            input_phase = torch.angle(combined_image).unsqueeze(1)
            unet_input = torch.cat([input_magnitude, input_phase], dim=1)
            
            # U-Net forward pass
            output_magnitude = self.model(unet_input)
            
            inference_time = time.time() - start_time
            
            return {
                'reconstruction': output_magnitude.squeeze(1).cpu(),
                'inference_time': inference_time,
                'zero_filled': torch.abs(combined_image).cpu(),
                'input_magnitude': input_magnitude.squeeze(1).cpu()
            }
    
    def compute_metrics(self, prediction, target):
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
    
    def run_inference(self, max_samples=None, save_all=True):
        """Run inference on test dataset."""
        self.logger.info("Starting U-Net inference...")
        
        all_metrics = {
            'psnr': [], 'ssim': [], 'nmse': [], 'mae': [], 'inference_time': []
        }
        
        num_samples = min(max_samples or len(self.test_loader), len(self.test_loader))
        
        pbar = tqdm(enumerate(self.test_loader), total=num_samples, desc="U-Net Inference")
        
        for sample_idx, batch in pbar:
            if max_samples and sample_idx >= max_samples:
                break
            
            # Extract data
            kspace_masked = batch['kspace_masked']
            mask = batch['mask']
            target = batch['target']
            
            # Run inference
            results = self.infer_single_sample(kspace_masked, mask)
            
            # Compute metrics
            metrics = self.compute_metrics(results['reconstruction'], target)
            metrics['inference_time'] = results['inference_time']
            
            # Collect metrics
            for metric_name, metric_value in metrics.items():
                if metric_name in all_metrics:
                    all_metrics[metric_name].append(metric_value)
            
            # Save results
            if save_all:
                reconstruction = results['reconstruction'].squeeze()
                save_tensor_as_image(
                    reconstruction,
                    self.results_dir / f"unet_reconstruction_{sample_idx:04d}.png",
                    f"U-Net Reconstruction {sample_idx}"
                )
                
                # Save metrics
                metrics_file = self.metrics_dir / f"metrics_{sample_idx:04d}.txt"
                with open(metrics_file, 'w') as f:
                    for metric_name, metric_value in metrics.items():
                        f.write(f"{metric_name.upper()}: {metric_value:.6f}\n")
            
            # Update progress
            pbar.set_postfix({
                'PSNR': f"{metrics['psnr']:.2f}",
                'SSIM': f"{metrics['ssim']:.4f}",
                'Time': f"{metrics['inference_time']:.3f}s"
            })
        
        # Save summary
        self._save_summary_metrics(all_metrics)
        
        self.logger.info("U-Net inference completed!")
        return all_metrics
    
    def _save_summary_metrics(self, all_metrics):
        """Save summary metrics."""
        summary_file = self.output_dir / "summary_metrics.txt"
        
        with open(summary_file, 'w') as f:
            f.write("U-Net Baseline Inference Summary\n")
            f.write("=" * 40 + "\n\n")
            
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
            self.output_dir / "all_metrics.npz",
            **all_metrics
        )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run U-Net Inference")
    parser.add_argument('--model', type=str, required=True, help='Path to U-Net model')
    parser.add_argument('--config', type=str, default='configs/unet_baseline_config.yaml')
    parser.add_argument('--output', type=str, default='outputs/unet_inference')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='auto')
    
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
    
    # Create inference engine
    engine = UNetInferenceEngine(config, args.model, device, args.output)
    
    # Run inference
    engine.run_inference(max_samples=args.max_samples, save_all=True)


if __name__ == "__main__":
    main()