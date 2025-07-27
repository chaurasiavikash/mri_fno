# File: diagnose_disco_standalone.py
"""
Self-contained comprehensive diagnostic tool for DISCO neural operator issues.
Analyzes trained model, reconstructions, and identifies problems.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from model import MRIReconstructionModel
from neural_operator import DISCONeuralOperator
from utils import complex_to_real, real_to_complex, fft2c, ifft2c


class DISCODiagnostics:
    """Comprehensive diagnostics for DISCO neural operator issues."""
    
    def __init__(self):
        # Hard-coded paths - modify these if needed
        self.model_path = "/scratch/vchaurasia/organized_models/disco_epoch20.pth"
        self.config_path = "configs/config.yaml"
        self.data_path = "/scratch/vchaurasia/fastmri_data/test"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory in scratch
        self.output_dir = Path("/scratch/vchaurasia/disco_diagnostics")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🔍 DISCO Diagnostics initialized")
        print(f"   Model: {self.model_path}")
        print(f"   Config: {self.config_path}")
        print(f"   Data: {self.data_path}")
        print(f"   Device: {self.device}")
        print(f"   Output: {self.output_dir}")
        
        # Load model and config
        self.config = self._load_config()
        self.model = self._load_model()
    
    def _load_config(self):
        """Load configuration from YAML file or create default."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Config loaded from {self.config_path}")
            return config
        except Exception as e:
            print(f"⚠️  Could not load config from {self.config_path}: {e}")
            print("   Using default config...")
            
            # Default config for DISCO
            return {
                'model': {
                    'name': 'disco_neural_operator',
                    'in_channels': 2,
                    'out_channels': 2,
                    'hidden_channels': 64,
                    'num_layers': 4,
                    'modes': 12,
                    'width': 64,
                    'dropout': 0.1,
                    'use_residual': True,
                    'activation': 'gelu'
                },
                'loss': {
                    'data_consistency_weight': 1.0
                },
                'data': {
                    'acceleration': 4,
                    'center_fraction': 0.08
                }
            }
    
    def _load_model(self):
        """Load the trained DISCO model."""
        print(f"📥 Loading model from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model config
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
            print("   Using config from checkpoint")
        else:
            model_config = self.config['model']
            print("   Using config from file")
        
        # Create neural operator config
        neural_operator_config = {
            'in_channels': model_config['in_channels'],
            'out_channels': model_config['out_channels'], 
            'hidden_channels': model_config['hidden_channels'],
            'num_layers': model_config['num_layers'],
            'modes': model_config['modes'],
            'width': model_config.get('width', 64),
            'dropout': model_config['dropout'],
            'use_residual': model_config['use_residual'],
            'activation': model_config['activation']
        }
        
        # Print config for verification
        print("   Model configuration:")
        for key, value in neural_operator_config.items():
            print(f"     {key}: {value}")
        
        # Create model
        model = MRIReconstructionModel(
            neural_operator_config=neural_operator_config,
            use_data_consistency=True,
            dc_weight=self.config['loss']['data_consistency_weight'],
            num_dc_iterations=5
        ).to(self.device)
        
        # Load weights
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model weights loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model weights: {e}")
            # Try to load with strict=False
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("   Loaded with strict=False (some weights may be missing)")
        
        model.eval()
        
        # Print checkpoint info
        if 'epoch' in checkpoint:
            print(f"   Trained for {checkpoint['epoch']} epochs")
        if 'best_val_loss' in checkpoint:
            print(f"   Best validation loss: {checkpoint['best_val_loss']:.6f}")
        
        return model
    
    def load_test_data(self):
        """Load a test sample for analysis."""
        print(f"📊 Loading test data from {self.data_path}")
        
        # Find first .h5 file
        data_files = list(Path(self.data_path).glob("*.h5"))
        if not data_files:
            raise FileNotFoundError(f"No .h5 files found in {self.data_path}")
        
        test_file = data_files[0]
        print(f"   Using file: {test_file.name}")
        
        with h5py.File(test_file, 'r') as f:
            print(f"   File keys: {list(f.keys())}")
            
            # Load first slice
            kspace_full = torch.from_numpy(f['kspace'][0])  # (15, 640, 368)
            print(f"   K-space shape: {kspace_full.shape}")
            print(f"   K-space dtype: {kspace_full.dtype}")
            print(f"   K-space is complex: {torch.is_complex(kspace_full)}")
            
            # Create mask
            acceleration = self.config['data']['acceleration']
            center_fraction = self.config['data']['center_fraction']
            mask = self._create_mask(kspace_full.shape[-2:], acceleration, center_fraction)
            print(f"   Mask shape: {mask.shape}")
            print(f"   Acceleration: {acceleration}x, Center fraction: {center_fraction}")
            
            # Apply mask
            kspace_masked = kspace_full * mask.unsqueeze(0)
            
            # Add batch dimension
            kspace_full = kspace_full.unsqueeze(0)  # (1, 15, 640, 368)
            kspace_masked = kspace_masked.unsqueeze(0)  # (1, 15, 640, 368)
            
            # Convert to real/imag format for neural network
            kspace_full_ri = complex_to_real(kspace_full)  # Should be (1, 15, 2, 640, 368)
            kspace_masked_ri = complex_to_real(kspace_masked)  # Should be (1, 15, 2, 640, 368)
            
            print(f"   K-space real/imag format: {kspace_full_ri.shape}")
            
            # Create target image (ground truth)
            target_image = self._kspace_to_magnitude_image(kspace_full)
            print(f"   Target image shape: {target_image.shape}")
            
            # Create zero-filled reconstruction for comparison
            zero_filled = self._kspace_to_magnitude_image(kspace_masked)
            print(f"   Zero-filled shape: {zero_filled.shape}")
            
            return {
                'kspace_full': kspace_full_ri.to(self.device),
                'kspace_masked': kspace_masked_ri.to(self.device),
                'mask': mask.unsqueeze(0).to(self.device),
                'target': target_image.to(self.device),
                'zero_filled': zero_filled.to(self.device),
                'kspace_full_complex': kspace_full.to(self.device),
                'filename': test_file.name
            }
    
    def _create_mask(self, shape, acceleration=4, center_fraction=0.08):
        """Create undersampling mask."""
        height, width = shape
        
        # Calculate center lines
        num_center_lines = int(center_fraction * width)
        center_start = width // 2 - num_center_lines // 2
        center_end = center_start + num_center_lines
        
        # Calculate total lines
        total_lines = width // acceleration
        remaining_lines = max(0, total_lines - num_center_lines)
        
        # Create mask
        mask = torch.zeros(width)
        mask[center_start:center_end] = 1
        
        # Randomly select remaining lines
        available_indices = [i for i in range(width) if i < center_start or i >= center_end]
        if remaining_lines > 0 and available_indices:
            selected_indices = torch.randperm(len(available_indices))[:remaining_lines]
            for idx in selected_indices:
                mask[available_indices[idx]] = 1
        
        # Expand to full shape
        return mask.unsqueeze(0).expand(height, -1)
    
    def _kspace_to_magnitude_image(self, kspace_complex):
        """Convert k-space to magnitude image."""
        # IFFT
        image_coils = ifft2c(kspace_complex)
        # Root sum of squares
        magnitude = torch.sqrt(torch.sum(torch.abs(image_coils) ** 2, dim=1))
        return magnitude
    
    def diagnose_model_architecture(self):
        """Analyze the neural operator architecture."""
        print(f"\n🏗️ ANALYZING MODEL ARCHITECTURE")
        print("=" * 50)
        
        # Get neural operator
        neural_op = self.model.neural_operator
        
        print(f"Neural Operator Type: {type(neural_op).__name__}")
        print(f"Input channels: {neural_op.in_channels}")
        print(f"Output channels: {neural_op.out_channels}")
        print(f"Hidden channels: {neural_op.hidden_channels}")
        print(f"Number of layers: {neural_op.num_layers}")
        print(f"Fourier modes: {neural_op.modes}")
        print(f"Use residual: {neural_op.use_residual}")
        
        # Check if model has proper spectral layers
        has_spectral = False
        spectral_layers = []
        for name, module in neural_op.named_modules():
            if 'spectral' in name.lower() or 'SpectralConv' in str(type(module)):
                has_spectral = True
                spectral_layers.append(name)
                print(f"✅ Found spectral layer: {name}")
        
        if not has_spectral:
            print("❌ WARNING: No spectral convolution layers found!")
        else:
            print(f"✅ Found {len(spectral_layers)} spectral convolution layers")
        
        # Count parameters
        total_params = sum(p.numel() for p in neural_op.parameters())
        trainable_params = sum(p.numel() for p in neural_op.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Check data consistency layers
        if hasattr(self.model, 'dc_layers'):
            print(f"Data consistency layers: {len(self.model.dc_layers)}")
        else:
            print("❌ No data consistency layers found")
        
        return has_spectral
    
    def diagnose_reconstruction_quality(self, data):
        """Analyze reconstruction quality issues."""
        print(f"\n🔍 ANALYZING RECONSTRUCTION QUALITY")
        print("=" * 50)
        
        with torch.no_grad():
            # Get model output
            print("   Running forward pass...")
            output = self.model(data['kspace_masked'], data['mask'], data['kspace_full'])
            
            # Extract results
            reconstruction = output['output']
            target = data['target']
            zero_filled = data['zero_filled']
            
            print(f"   Reconstruction shape: {reconstruction.shape}")
            print(f"   Target shape: {target.shape}")
            print(f"   Output keys: {list(output.keys())}")
            
            # Ensure same shape for comparison
            if reconstruction.shape != target.shape:
                print(f"   ⚠️  Shape mismatch! Squeezing tensors...")
                reconstruction = reconstruction.squeeze()
                target = target.squeeze()
                zero_filled = zero_filled.squeeze()
            
            # Compute metrics vs target
            mse = torch.mean((reconstruction - target) ** 2).item()
            mae = torch.mean(torch.abs(reconstruction - target)).item()
            
            # PSNR
            max_val = torch.max(target).item()
            psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float('inf')
            
            # SSIM approximation
            def simple_ssim(x, y):
                mu_x = torch.mean(x)
                mu_y = torch.mean(y)
                sigma_x = torch.std(x)
                sigma_y = torch.std(y)
                sigma_xy = torch.mean((x - mu_x) * (y - mu_y))
                
                c1 = 0.01 ** 2
                c2 = 0.03 ** 2
                
                ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                       ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
                return ssim.item()
            
            ssim = simple_ssim(reconstruction, target)
            
            # Compute same metrics for zero-filled (baseline)
            mse_zf = torch.mean((zero_filled - target) ** 2).item()
            psnr_zf = 20 * np.log10(max_val / np.sqrt(mse_zf)) if mse_zf > 0 else float('inf')
            ssim_zf = simple_ssim(zero_filled, target)
            
            # Statistical analysis
            recon_stats = {
                'mean': torch.mean(reconstruction).item(),
                'std': torch.std(reconstruction).item(),
                'min': torch.min(reconstruction).item(),
                'max': torch.max(reconstruction).item()
            }
            
            target_stats = {
                'mean': torch.mean(target).item(),
                'std': torch.std(target).item(),
                'min': torch.min(target).item(),
                'max': torch.max(target).item()
            }
            
            zf_stats = {
                'mean': torch.mean(zero_filled).item(),
                'std': torch.std(zero_filled).item(),
                'min': torch.min(zero_filled).item(),
                'max': torch.max(zero_filled).item()
            }
            
            print(f"\n📊 RECONSTRUCTION METRICS:")
            print(f"   DISCO     - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
            print(f"   Zero-fill - PSNR: {psnr_zf:.2f} dB, SSIM: {ssim_zf:.4f}")
            print(f"   Improvement - PSNR: {psnr - psnr_zf:+.2f} dB, SSIM: {ssim - ssim_zf:+.4f}")
            
            print(f"\n📈 STATISTICS COMPARISON:")
            print(f"   Target        - Mean: {target_stats['mean']:.4f}, Std: {target_stats['std']:.4f}, Range: [{target_stats['min']:.4f}, {target_stats['max']:.4f}]")
            print(f"   DISCO         - Mean: {recon_stats['mean']:.4f}, Std: {recon_stats['std']:.4f}, Range: [{recon_stats['min']:.4f}, {recon_stats['max']:.4f}]")
            print(f"   Zero-fill     - Mean: {zf_stats['mean']:.4f}, Std: {zf_stats['std']:.4f}, Range: [{zf_stats['min']:.4f}, {zf_stats['max']:.4f}]")
            
            # Check for common issues
            issues = []
            
            if psnr < 15:
                issues.append("❌ Very poor PSNR (< 15 dB)")
            elif psnr < 20:
                issues.append("⚠️  Low PSNR (< 20 dB)")
            elif psnr < 25:
                issues.append("⚠️  Moderate PSNR (< 25 dB)")
            else:
                issues.append("✅ Good PSNR (>= 25 dB)")
            
            if psnr <= psnr_zf:
                issues.append("❌ DISCO not better than zero-filling!")
            else:
                issues.append(f"✅ DISCO improves over zero-filling by {psnr - psnr_zf:.2f} dB")
            
            if abs(recon_stats['mean'] - target_stats['mean']) > 0.2 * target_stats['mean']:
                issues.append("❌ Large mean difference (> 20%)")
            
            if recon_stats['max'] > 2 * target_stats['max']:
                issues.append("❌ Reconstruction values too high")
            
            if recon_stats['std'] > 3 * target_stats['std']:
                issues.append("❌ Reconstruction too noisy")
            
            if recon_stats['std'] < 0.1 * target_stats['std']:
                issues.append("❌ Reconstruction too smooth/blurry")
            
            # Check for NaN/Inf
            if torch.isnan(reconstruction).any():
                issues.append("❌ NaN values in reconstruction")
            
            if torch.isinf(reconstruction).any():
                issues.append("❌ Infinite values in reconstruction")
            
            print(f"\n🔍 QUALITY ASSESSMENT:")
            for issue in issues:
                print(f"   {issue}")
            
            return {
                'reconstruction': reconstruction,
                'target': target,
                'zero_filled': zero_filled,
                'metrics': {
                    'psnr': psnr, 'ssim': ssim, 'mse': mse, 'mae': mae,
                    'psnr_zf': psnr_zf, 'ssim_zf': ssim_zf
                },
                'stats': {'recon': recon_stats, 'target': target_stats, 'zf': zf_stats},
                'issues': issues
            }
    
    def create_visual_diagnosis(self, data, quality_results):
        """Create visual diagnosis plots."""
        print(f"\n📊 CREATING VISUAL DIAGNOSIS")
        print("=" * 50)
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # Convert to numpy for plotting
        target = quality_results['target'].cpu().numpy()
        reconstruction = quality_results['reconstruction'].cpu().numpy()
        zero_filled = quality_results['zero_filled'].cpu().numpy()
        
        # Ensure 2D
        if target.ndim > 2:
            target = target.squeeze()
        if reconstruction.ndim > 2:
            reconstruction = reconstruction.squeeze()
        if zero_filled.ndim > 2:
            zero_filled = zero_filled.squeeze()
        
        vmax = np.max(target)
        
        # Row 1: Image domain comparison
        axes[0, 0].imshow(target, cmap='gray', vmin=0, vmax=vmax)
        axes[0, 0].set_title('Ground Truth')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(zero_filled, cmap='gray', vmin=0, vmax=vmax)
        axes[0, 1].set_title(f'Zero-filled\nPSNR: {quality_results["metrics"]["psnr_zf"]:.1f} dB')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(reconstruction, cmap='gray', vmin=0, vmax=vmax)
        axes[0, 2].set_title(f'DISCO Reconstruction\nPSNR: {quality_results["metrics"]["psnr"]:.1f} dB')
        axes[0, 2].axis('off')
        
        # Error map
        error = np.abs(reconstruction - target)
        im = axes[0, 3].imshow(error, cmap='hot', vmin=0, vmax=np.max(error))
        axes[0, 3].set_title('Error Map (DISCO)')
        axes[0, 3].axis('off')
        plt.colorbar(im, ax=axes[0, 3], shrink=0.8)
        
        # Row 2: Error comparisons
        error_zf = np.abs(zero_filled - target)
        
        im1 = axes[1, 0].imshow(error_zf, cmap='hot', vmin=0, vmax=np.max(error_zf))
        axes[1, 0].set_title('Error Map (Zero-fill)')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], shrink=0.8)
        
        # Difference in errors
        error_diff = error_zf - error
        im2 = axes[1, 1].imshow(error_diff, cmap='RdBu_r', vmin=-np.max(np.abs(error_diff)), vmax=np.max(np.abs(error_diff)))
        axes[1, 1].set_title('Error Improvement\n(Red = DISCO better)')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], shrink=0.8)
        
        # Central profiles
        center_row = target.shape[0] // 2
        axes[1, 2].plot(target[center_row, :], 'k-', label='Target', linewidth=2)
        axes[1, 2].plot(zero_filled[center_row, :], 'b--', label='Zero-fill', alpha=0.7)
        axes[1, 2].plot(reconstruction[center_row, :], 'r-', label='DISCO', alpha=0.8)
        axes[1, 2].set_title('Central Row Profile')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Histogram comparison
        axes[1, 3].hist(target.flatten(), bins=50, alpha=0.7, label='Target', density=True, color='black')
        axes[1, 3].hist(zero_filled.flatten(), bins=50, alpha=0.5, label='Zero-fill', density=True, color='blue')
        axes[1, 3].hist(reconstruction.flatten(), bins=50, alpha=0.5, label='DISCO', density=True, color='red')
        axes[1, 3].set_title('Intensity Distribution')
        axes[1, 3].legend()
        axes[1, 3].set_xlabel('Intensity')
        axes[1, 3].set_ylabel('Density')
        
        # Row 3: Frequency domain analysis
        target_fft = np.fft.fftshift(np.fft.fft2(target))
        recon_fft = np.fft.fftshift(np.fft.fft2(reconstruction))
        zf_fft = np.fft.fftshift(np.fft.fft2(zero_filled))
        
        log_scale = lambda x: np.log(np.abs(x) + 1e-8)
        
        axes[2, 0].imshow(log_scale(target_fft), cmap='viridis')
        axes[2, 0].set_title('Target K-space (log)')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(log_scale(zf_fft), cmap='viridis')
        axes[2, 1].set_title('Zero-fill K-space (log)')
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(log_scale(recon_fft), cmap='viridis')
        axes[2, 2].set_title('DISCO K-space (log)')
        axes[2, 2].axis('off')
        
        # K-space error
        freq_error = np.abs(recon_fft - target_fft)
        im3 = axes[2, 3].imshow(log_scale(freq_error), cmap='hot')
        axes[2, 3].set_title('K-space Error (log)')
        axes[2, 3].axis('off')
        plt.colorbar(im3, ax=axes[2, 3], shrink=0.8)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "visual_diagnosis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Visual diagnosis saved to: {plot_path}")
        
        return plot_path
    
    def generate_diagnosis_report(self, data, quality_results):
        """Generate comprehensive diagnosis report."""
        report_path = self.output_dir / "diagnosis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("DISCO NEURAL OPERATOR DIAGNOSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Config: {self.config_path}\n")
            f.write(f"Test file: {data['filename']}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            f.write("RECONSTRUCTION METRICS:\n")
            f.write("-" * 30 + "\n")
            metrics = quality_results['metrics']
            f.write(f"DISCO Neural Operator:\n")
            f.write(f"  PSNR: {metrics['psnr']:.2f} dB\n")
            f.write(f"  SSIM: {metrics['ssim']:.4f}\n")
            f.write(f"  MSE:  {metrics['mse']:.6f}\n")
            f.write(f"  MAE:  {metrics['mae']:.6f}\n\n")
            
            f.write(f"Zero-filled Baseline:\n")
            f.write(f"  PSNR: {metrics['psnr_zf']:.2f} dB\n")
            f.write(f"  SSIM: {metrics['ssim_zf']:.4f}\n\n")
            
            f.write(f"Improvement over baseline:\n")
            f.write(f"  PSNR: {metrics['psnr'] - metrics['psnr_zf']:+.2f} dB\n")
            f.write(f"  SSIM: {metrics['ssim'] - metrics['ssim_zf']:+.4f}\n\n")
            
            f.write("STATISTICAL ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            stats = quality_results['stats']
            f.write("Target image:\n")
            f.write(f"  Mean: {stats['target']['mean']:.6f}\n")
            f.write(f"  Std:  {stats['target']['std']:.6f}\n")
            f.write(f"  Min:  {stats['target']['min']:.6f}\n")
            f.write(f"  Max:  {stats['target']['max']:.6f}\n\n")
            
            f.write("DISCO reconstruction:\n")
            f.write(f"  Mean: {stats['recon']['mean']:.6f}\n")
            f.write(f"  Std:  {stats['recon']['std']:.6f}\n")
            f.write(f"  Min:  {stats['recon']['min']:.6f}\n")
            f.write(f"  Max:  {stats['recon']['max']:.6f}\n\n")
            
            f.write("IDENTIFIED ISSUES:\n")
            f.write("-" * 30 + "\n")
            if quality_results['issues']:
                for issue in quality_results['issues']:
                    f.write(f"  {issue}\n")
            else:
                f.write("  No major issues detected\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            
            psnr = metrics['psnr']
            psnr_improvement = metrics['psnr'] - metrics['psnr_zf']
            
            if psnr < 15:
                f.write("  CRITICAL ISSUES:\n")
                f.write("  - Very poor reconstruction quality\n")
                f.write("  - Check model architecture and training\n")
                f.write("  - Verify spectral convolution implementation\n")
                f.write("  - Examine loss function and data consistency\n")
            elif psnr < 20:
                f.write("  MODERATE ISSUES:\n")
                f.write("  - Below expected performance for MRI reconstruction\n")
                f.write("  - Consider longer training or learning rate adjustment\n")
                f.write("  - Verify data preprocessing and normalization\n")
            elif psnr < 25:
                f.write("  MINOR ISSUES:\n")
                f.write("  - Acceptable but could be improved\n")
                f.write("  - Fine-tune hyperparameters\n")
                f.write("  - Consider additional training epochs\n")
            else:
                f.write("  GOOD PERFORMANCE:\n")
                f.write("  - Reconstruction quality is acceptable\n")
                f.write("  - Consider optimizations for inference speed\n")
            
            if psnr_improvement <= 0:
                f.write("  - CRITICAL: Model not improving over zero-filling!\n")
                f.write("  - Check model weights and training convergence\n")
                f.write("  - Verify data consistency implementation\n")
            elif psnr_improvement < 2:
                f.write("  - Limited improvement over baseline\n")
                f.write("  - Consider architectural improvements\n")
            else:
                f.write("  - Good improvement over baseline\n")
            
            # Model-specific recommendations
            if 'NaN values' in str(quality_results['issues']):
                f.write("  - NaN DETECTED: Check for numerical instability\n")
                f.write("  - Verify gradient clipping and learning rates\n")
                f.write("  - Check for division by zero in loss functions\n")
            
            if any('too high' in issue for issue in quality_results['issues']):
                f.write("  - High values detected: Check output scaling\n")
                f.write("  - Verify activation functions and normalization\n")
            
            f.write("\nNEXT STEPS:\n")
            f.write("-" * 30 + "\n")
            f.write("1. Review visual diagnosis plots\n")
            f.write("2. Check training logs for convergence issues\n")
            f.write("3. Compare with U-Net baseline results\n")
            f.write("4. Consider hyperparameter tuning if needed\n")
            f.write("5. Validate on additional test samples\n")
        
        print(f"   Diagnosis report saved to: {report_path}")
        return report_path
    
    def run_full_diagnosis(self):
        """Run complete diagnosis pipeline."""
        print(f"🚀 STARTING FULL DISCO DIAGNOSIS")
        print("=" * 60)
        
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                print("   Available models in organized_models:")
                models_dir = Path("/scratch/vchaurasia/organized_models")
                if models_dir.exists():
                    for model_file in models_dir.glob("*.pth"):
                        print(f"     {model_file.name}")
                return None
            
            # Load test data
            data = self.load_test_data()
            
            # Run architecture diagnostics
            has_spectral = self.diagnose_model_architecture()
            
            # Run quality diagnostics
            quality_results = self.diagnose_reconstruction_quality(data)
            
            # Create visualizations
            plot_path = self.create_visual_diagnosis(data, quality_results)
            
            # Generate report
            report_path = self.generate_diagnosis_report(data, quality_results)
            
            # Summary
            print(f"\n✅ DIAGNOSIS COMPLETE!")
            print(f"   Results saved to: {self.output_dir}")
            print(f"   Visual: {plot_path}")
            print(f"   Report: {report_path}")
            
            # Key findings summary
            psnr = quality_results['metrics']['psnr']
            psnr_improvement = quality_results['metrics']['psnr'] - quality_results['metrics']['psnr_zf']
            
            print(f"\n🎯 KEY FINDINGS:")
            print(f"   PSNR: {psnr:.2f} dB (improvement: {psnr_improvement:+.2f} dB)")
            print(f"   SSIM: {quality_results['metrics']['ssim']:.4f}")
            
            if psnr > 25 and psnr_improvement > 2:
                print(f"   ✅ DISCO is working well!")
            elif psnr > 20 and psnr_improvement > 0:
                print(f"   ⚠️  DISCO is working but could be improved")
            else:
                print(f"   ❌ DISCO has significant issues")
            
            # Check for critical issues
            critical_issues = [issue for issue in quality_results['issues'] if '❌' in issue]
            if critical_issues:
                print(f"   🚨 Critical issues found: {len(critical_issues)}")
            
            return {
                'quality': quality_results,
                'data': data,
                'has_spectral': has_spectral
            }
            
        except Exception as e:
            print(f"❌ DIAGNOSIS FAILED: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to provide helpful error messages
            if "No module named" in str(e):
                print("\n💡 TIP: Make sure you're running from the correct directory with src/ folder")
            elif "CUDA out of memory" in str(e):
                print("\n💡 TIP: Try running on CPU by editing device = 'cpu' in the script")
            elif "File not found" in str(e):
                print(f"\n💡 TIP: Check if the file paths in the script are correct")
            
            return None


def main():
    """Main function to run DISCO diagnostics."""
    print("🔬 DISCO Neural Operator Diagnostic Tool")
    print("=" * 50)
    print("This tool will analyze your trained DISCO model and identify issues.")
    print("Results will be saved to /scratch/vchaurasia/disco_diagnostics/")
    print()
    
    # Run diagnostics
    diagnostics = DISCODiagnostics()
    results = diagnostics.run_full_diagnosis()
    
    if results:
        print(f"\n🎉 SUCCESS! Check the diagnosis results in:")
        print(f"   {diagnostics.output_dir}")
        print(f"\n📁 Generated files:")
        print(f"   - visual_diagnosis.png  (visual comparison)")
        print(f"   - diagnosis_report.txt  (detailed analysis)")
        
        # Provide actionable next steps
        psnr = results['quality']['metrics']['psnr']
        if psnr < 20:
            print(f"\n⚡ NEXT STEPS (Poor Performance):")
            print(f"   1. Check if U-Net baseline is working better")
            print(f"   2. Review training logs for convergence issues") 
            print(f"   3. Consider retraining with different hyperparameters")
        elif psnr < 25:
            print(f"\n⚡ NEXT STEPS (Moderate Performance):")
            print(f"   1. Compare with U-Net baseline")
            print(f"   2. Consider fine-tuning hyperparameters")
            print(f"   3. Evaluate on more test samples")
        else:
            print(f"\n⚡ NEXT STEPS (Good Performance):")
            print(f"   1. Run comprehensive evaluation vs U-Net")
            print(f"   2. Generate final comparison report")
            print(f"   3. Prepare results for presentation")
        
        return 0
    else:
        print(f"\n💥 DIAGNOSIS FAILED")
        print(f"   Check the error messages above")
        print(f"   Common issues:")
        print(f"   - Wrong file paths")
        print(f"   - Missing dependencies")
        print(f"   - CUDA memory issues")
        return 1


if __name__ == "__main__":
    exit(main())