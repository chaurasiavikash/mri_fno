# File: src/infer.py

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

import torch
import numpy as np
from tqdm import tqdm
import h5py

# Local imports
from model import MRIReconstructionModel
from data_loader import FastMRIDataset, create_data_loaders
from utils import (
    load_config, set_seed, setup_logging, get_device,
    create_directory, save_tensor_as_image, complex_to_real
)


class MRIInferenceEngine:
    """
    Inference engine for MRI reconstruction using trained neural operators.
    
    Handles loading trained models, processing test data, and generating
    reconstructed images with evaluation metrics.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model_path: str,
        device: torch.device,
        output_dir: str
    ):
        """
        Initialize inference engine.
        
        Args:
            config: Configuration dictionary
            model_path: Path to trained model checkpoint
            device: Device to use for inference
            output_dir: Directory to save outputs
        """
        self.config = config
        self.model_path = model_path
        self.device = device
        self.output_dir = Path(output_dir)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup output directories
        self._setup_directories()
        
        # Load model
        self._load_model()
        
        # Setup data loader
        self._setup_data_loader()
        
        self.logger.info("Inference engine initialized")
    
    def _setup_directories(self):
        """Setup output directories."""
        self.results_dir = self.output_dir / "reconstructions"
        self.metrics_dir = self.output_dir / "metrics"
        self.visualizations_dir = self.output_dir / "visualizations"
        
        create_directory(self.results_dir)
        create_directory(self.metrics_dir)
        create_directory(self.visualizations_dir)
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model configuration from checkpoint
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
        else:
            # Fallback to config file if not in checkpoint
            model_config = self.config['model']
        
        # Initialize model
        self.model = MRIReconstructionModel(
            neural_operator_config={
                'in_channels': model_config['in_channels'],
                'out_channels': model_config['out_channels'],
                'hidden_channels': model_config['hidden_channels'],
                'num_layers': model_config['num_layers'],
                'modes': model_config['modes'],
                'width': model_config.get('width', 64),
                'dropout': model_config['dropout'],
                'use_residual': model_config['use_residual'],
                'activation': model_config['activation']
            },
            use_data_consistency=True,
            dc_weight=self.config['loss']['data_consistency_weight'],
            num_dc_iterations=5
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.logger.info(f"Model loaded from {self.model_path}")
        
        # Log model info
        if 'config' in checkpoint:
            epoch = checkpoint.get('epoch', 'Unknown')
            val_loss = checkpoint.get('best_val_loss', 'Unknown')
            self.logger.info(f"Model trained for {epoch} epochs, best validation loss: {val_loss}")
    
    def _setup_data_loader(self):
        """Setup data loader for inference."""
        data_config = self.config['data']
        
        # Create test dataset
        self.test_dataset = FastMRIDataset(
            data_path=data_config['test_path'],
            acceleration=data_config['acceleration'],
            center_fraction=data_config['center_fraction'],
            transform=None,  # No normalization for inference
            use_seed=True,
            seed=self.config['system']['seed']
        )
        
        # Create data loader
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,  # Process one sample at a time for inference
            shuffle=False,
            num_workers=data_config.get('num_workers', 0),
            pin_memory=torch.cuda.is_available()
        )
        
        self.logger.info(f"Test dataset loaded: {len(self.test_dataset)} samples")
    
    def infer_single_sample(
        self,
        kspace_masked: torch.Tensor,
        mask: torch.Tensor,
        kspace_full: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform inference on a single sample.
        
        Args:
            kspace_masked: Masked k-space data
            mask: Undersampling mask
            kspace_full: Full k-space data (for reference)
            
        Returns:
            Dictionary containing reconstruction results
        """
        with torch.no_grad():
            # Move to device
            kspace_masked = kspace_masked.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            
            if kspace_full is not None:
                kspace_full = kspace_full.to(self.device, non_blocking=True)
            
            # Start timing
            start_time = time.time()
            
            # Forward pass
            output = self.model(kspace_masked, mask, kspace_full)
            
            # End timing
            inference_time = time.time() - start_time
            
            # Prepare results
            results = {
                'reconstruction': output['output'].cpu(),
                'kspace_pred': output['kspace_pred'].cpu(),
                'image_complex': output['image_complex'].cpu(),
                'inference_time': inference_time
            }
            
            # Add intermediate results if available
            if 'intermediate' in output:
                results['intermediate'] = {
                    k: v.cpu() for k, v in output['intermediate'].items()
                }
            
            return results
    
    def compute_metrics(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            prediction: Predicted image
            target: Target image
            mask: Optional mask for region-specific metrics
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Convert to numpy for metric computation
        pred_np = prediction.squeeze().numpy()
        target_np = target.squeeze().numpy()
        
        # Peak Signal-to-Noise Ratio (PSNR)
        mse = np.mean((pred_np - target_np) ** 2)
        if mse > 0:
            max_val = np.max(target_np)
            metrics['psnr'] = 20 * np.log10(max_val / np.sqrt(mse))
        else:
            metrics['psnr'] = float('inf')
        
        # Structural Similarity Index (SSIM)
        metrics['ssim'] = self._compute_ssim(pred_np, target_np)
        
        # Normalized Mean Squared Error (NMSE)
        metrics['nmse'] = mse / np.mean(target_np ** 2)
        
        # Mean Absolute Error (MAE)
        metrics['mae'] = np.mean(np.abs(pred_np - target_np))
        
        # Normalized Root Mean Squared Error (NRMSE)
        metrics['nrmse'] = np.sqrt(mse) / np.mean(target_np)
        
        return metrics
    
    def _compute_ssim(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        data_range: Optional[float] = None
    ) -> float:
        """
        Compute Structural Similarity Index (SSIM).
        
        Args:
            prediction: Predicted image
            target: Target image
            data_range: Data range for SSIM computation
            
        Returns:
            SSIM value
        """
        if data_range is None:
            data_range = np.max(target) - np.min(target)
        
        # Constants for SSIM
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        
        # Compute means
        mu1 = np.mean(prediction)
        mu2 = np.mean(target)
        
        # Compute variances and covariance
        sigma1_sq = np.var(prediction)
        sigma2_sq = np.var(target)
        sigma12 = np.mean((prediction - mu1) * (target - mu2))
        
        # Compute SSIM
        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
        
        ssim = numerator / denominator
        
        return float(ssim)
    
    def save_results(
        self,
        results: Dict[str, torch.Tensor],
        metrics: Dict[str, float],
        sample_idx: int,
        metadata: Dict[str, Any]
    ):
        """
        Save reconstruction results and metrics.
        
        Args:
            results: Inference results
            metrics: Computed metrics
            sample_idx: Sample index
            metadata: Sample metadata
        """
        # Save reconstructed image
        reconstruction = results['reconstruction'].squeeze()
        save_tensor_as_image(
            reconstruction,
            self.results_dir / f"reconstruction_{sample_idx:04d}.png",
            f"Reconstruction {sample_idx}"
        )
        
        # Save intermediate results if available
        if 'intermediate' in results:
            for name, image in results['intermediate'].items():
                save_tensor_as_image(
                    image.squeeze(),
                    self.visualizations_dir / f"{name}_{sample_idx:04d}.png",
                    f"{name.title()} {sample_idx}"
                )
        
        # Save metrics
        metrics_file = self.metrics_dir / f"metrics_{sample_idx:04d}.txt"
        with open(metrics_file, 'w') as f:
            f.write(f"Sample {sample_idx} Metrics\n")
            f.write("=" * 30 + "\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name.upper()}: {metric_value:.6f}\n")
            f.write(f"\nInference time: {results['inference_time']:.4f} seconds\n")
            f.write(f"Metadata: {metadata}\n")
        
        # Save raw data (optional)
        if self.config['evaluation'].get('save_raw_data', False):
            raw_data_file = self.results_dir / f"raw_data_{sample_idx:04d}.npz"
            np.savez_compressed(
                raw_data_file,
                reconstruction=reconstruction.numpy(),
                inference_time=results['inference_time'],
                metrics=metrics,
                metadata=metadata
            )
    
    def run_inference(
        self,
        max_samples: Optional[int] = None,
        save_all: bool = True
    ) -> Dict[str, List[float]]:
        """
        Run inference on test dataset.
        
        Args:
            max_samples: Maximum number of samples to process (None for all)
            save_all: Whether to save all results
            
        Returns:
            Dictionary of collected metrics
        """
        self.logger.info("Starting inference...")
        
        # Initialize metric collectors
        all_metrics = {
            'psnr': [],
            'ssim': [],
            'nmse': [],
            'mae': [],
            'nrmse': [],
            'inference_time': []
        }
        
        # Process samples
        num_samples = min(max_samples or len(self.test_loader), len(self.test_loader))
        
        pbar = tqdm(enumerate(self.test_loader), total=num_samples, desc="Inference")
        
        for sample_idx, batch in pbar:
            if max_samples and sample_idx >= max_samples:
                break
            
            # Extract data
            kspace_masked = batch['kspace_masked']
            mask = batch['mask']
            target = batch['target']
            kspace_full = batch.get('kspace_full')
            metadata = batch['metadata']
            
            # Run inference
            results = self.infer_single_sample(kspace_masked, mask, kspace_full)
            
            # Compute metrics
            metrics = self.compute_metrics(results['reconstruction'], target)
            metrics['inference_time'] = results['inference_time']
            
            # Collect metrics
            for metric_name, metric_value in metrics.items():
                if metric_name in all_metrics:
                    all_metrics[metric_name].append(metric_value)
            
            # Save results
            if save_all:
                # Convert metadata from batch format
                sample_metadata = {
                    k: v[0] if isinstance(v, (list, torch.Tensor)) else v
                    for k, v in metadata.items()
                }
                self.save_results(results, metrics, sample_idx, sample_metadata)
            
            # Update progress bar
            pbar.set_postfix({
                'PSNR': f"{metrics['psnr']:.2f}",
                'SSIM': f"{metrics['ssim']:.4f}",
                'Time': f"{metrics['inference_time']:.3f}s"
            })
        
        # Compute summary statistics
        summary_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:  # Check if list is not empty
                summary_metrics[f'{metric_name}_mean'] = np.mean(values)
                summary_metrics[f'{metric_name}_std'] = np.std(values)
                summary_metrics[f'{metric_name}_median'] = np.median(values)
                summary_metrics[f'{metric_name}_min'] = np.min(values)
                summary_metrics[f'{metric_name}_max'] = np.max(values)
        
        # Save summary metrics
        self._save_summary_metrics(summary_metrics, all_metrics)
        
        # Log summary
        self.logger.info("Inference completed!")
        self.logger.info(f"Processed {len(all_metrics['psnr'])} samples")
        self.logger.info(f"Average PSNR: {summary_metrics.get('psnr_mean', 0):.2f} ± {summary_metrics.get('psnr_std', 0):.2f}")
        self.logger.info(f"Average SSIM: {summary_metrics.get('ssim_mean', 0):.4f} ± {summary_metrics.get('ssim_std', 0):.4f}")
        self.logger.info(f"Average inference time: {summary_metrics.get('inference_time_mean', 0):.3f}s")
        
        return all_metrics
    
    def _save_summary_metrics(
        self,
        summary_metrics: Dict[str, float],
        all_metrics: Dict[str, List[float]]
    ):
        """Save summary metrics to file."""
        summary_file = self.output_dir / "summary_metrics.txt"
        
        with open(summary_file, 'w') as f:
            f.write("MRI Reconstruction Inference Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Test samples: {len(all_metrics['psnr'])}\n")
            f.write(f"Acceleration factor: {self.config['data']['acceleration']}\n")
            f.write(f"Center fraction: {self.config['data']['center_fraction']}\n\n")
            
            f.write("Summary Statistics:\n")
            f.write("-" * 20 + "\n")
            
            metrics_order = ['psnr', 'ssim', 'nmse', 'mae', 'nrmse', 'inference_time']
            
            for metric in metrics_order:
                if f'{metric}_mean' in summary_metrics:
                    f.write(f"{metric.upper()}:\n")
                    f.write(f"  Mean: {summary_metrics[f'{metric}_mean']:.6f}\n")
                    f.write(f"  Std:  {summary_metrics[f'{metric}_std']:.6f}\n")
                    f.write(f"  Min:  {summary_metrics[f'{metric}_min']:.6f}\n")
                    f.write(f"  Max:  {summary_metrics[f'{metric}_max']:.6f}\n")
                    f.write(f"  Median: {summary_metrics[f'{metric}_median']:.6f}\n\n")
        
        # Save raw metrics as numpy file
        metrics_npz = self.output_dir / "all_metrics.npz"
        np.savez_compressed(metrics_npz, **all_metrics)
        
        self.logger.info(f"Summary metrics saved to {summary_file}")
    
    def create_comparison_visualization(
        self,
        sample_indices: List[int],
        save_path: Optional[str] = None
    ):
        """
        Create comparison visualization for selected samples.
        
        Args:
            sample_indices: List of sample indices to visualize
            save_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt
        
        if save_path is None:
            save_path = self.visualizations_dir / "comparison.png"
        
        # Process selected samples
        samples_data = []
        
        for idx in sample_indices:
            if idx >= len(self.test_dataset):
                self.logger.warning(f"Sample index {idx} out of range, skipping")
                continue
            
            # Get sample
            sample = self.test_dataset[idx]
            
            # Run inference
            kspace_masked = sample['kspace_masked'].unsqueeze(0)
            mask = sample['mask']
            target = sample['target']
            
            results = self.infer_single_sample(kspace_masked, mask)
            
            samples_data.append({
                'target': target.squeeze().numpy(),
                'prediction': results['reconstruction'].squeeze().numpy(),
                'zero_filled': sample['image_masked'].numpy(),
                'metrics': self.compute_metrics(results['reconstruction'], target.unsqueeze(0))
            })
        
        # Create visualization
        fig, axes = plt.subplots(len(samples_data), 3, figsize=(12, 4 * len(samples_data)))
        
        if len(samples_data) == 1:
            axes = axes.reshape(1, -1)
        
        for i, data in enumerate(samples_data):
            # Zero-filled reconstruction
            axes[i, 0].imshow(data['zero_filled'], cmap='gray')
            axes[i, 0].set_title('Zero-filled')
            axes[i, 0].axis('off')
            
            # Neural operator reconstruction
            axes[i, 1].imshow(data['prediction'], cmap='gray')
            axes[i, 1].set_title(f'Neural Operator\nPSNR: {data["metrics"]["psnr"]:.2f}')
            axes[i, 1].axis('off')
            
            # Ground truth
            axes[i, 2].imshow(data['target'], cmap='gray')
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparison visualization saved to {save_path}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run MRI Reconstruction Inference")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='outputs/inference',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process')
    parser.add_argument('--visualize', type=int, nargs='+', default=[0, 1, 2],
                       help='Sample indices to visualize')
    parser.add_argument('--save-all', action='store_true', default=True,
                       help='Save all reconstruction results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device and seed
    device = get_device(args.device)
    set_seed(config['system']['seed'])
    
    # Setup logging
    create_directory(args.output)
    setup_logging(
        os.path.join(args.output, 'logs'),
        config['system']['log_level']
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting MRI reconstruction inference on {device}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output directory: {args.output}")
    
    # Create inference engine
    inference_engine = MRIInferenceEngine(
        config=config,
        model_path=args.model,
        device=device,
        output_dir=args.output
    )
    
    # Run inference
    try:
        all_metrics = inference_engine.run_inference(
            max_samples=args.max_samples,
            save_all=args.save_all
        )
        
        # Create comparison visualization
        if args.visualize:
            inference_engine.create_comparison_visualization(args.visualize)
        
        logger.info("Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Inference failed with error: {e}")
        raise


if __name__ == "__main__":
    main()