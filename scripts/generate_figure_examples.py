#!/usr/bin/env python3
"""
Generate input/output examples for UNet and DISCO visualization.
This creates the data needed for your presentation figure.
File: scripts/generate_figure_examples.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import with error handling
try:
    from utils import (
        complex_to_real, real_to_complex, fft2c, ifft2c, 
        root_sum_of_squares, apply_mask
    )
    print("✅ Utils imported successfully")
except ImportError as e:
    print(f"❌ Error importing utils: {e}")
    sys.exit(1)

try:
    # Try different import paths for UNet model
    try:
        from unet_model import MRIUNetModel
        print("✅ UNet model imported successfully")
    except ImportError:
        # Alternative path
        sys.path.append("/home/vchaurasia/projects/mri_fno/src")
        from unet_model import MRIUNetModel
        print("✅ UNet model imported successfully (alternative path)")
except ImportError as e:
    print(f"❌ Error importing UNet model: {e}")
    print("Available files in src:")
    src_path = Path(__file__).parent.parent / "src"
    for f in src_path.glob("*.py"):
        print(f"  - {f.name}")
    sys.exit(1)

try:
    from model import MRIReconstructionModel
    print("✅ DISCO model imported successfully")
except ImportError as e:
    print(f"❌ Error importing DISCO model: {e}")
    sys.exit(1)

def compute_metrics(prediction, target):
    """Compute evaluation metrics."""
    pred_np = prediction.squeeze().cpu().numpy() if hasattr(prediction, 'cpu') else prediction.squeeze()
    target_np = target.squeeze().cpu().numpy() if hasattr(target, 'cpu') else target.squeeze()
    
    # PSNR
    mse = np.mean((pred_np - target_np) ** 2)
    if mse > 0:
        max_val = np.max(target_np)
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # Simple SSIM
    mu1, mu2 = np.mean(pred_np), np.mean(target_np)
    sigma1, sigma2 = np.var(pred_np), np.var(target_np)
    sigma12 = np.mean((pred_np - mu1) * (target_np - mu2))
    
    c1, c2 = (0.01 * np.max(target_np)) ** 2, (0.03 * np.max(target_np)) ** 2
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    
    return {
        'psnr': float(psnr),
        'ssim': float(ssim)
    }

def create_undersampling_mask(shape, acceleration=4, center_fraction=0.08, seed=42):
    """Create undersampling mask."""
    height, width = shape
    torch.manual_seed(seed)
    
    # Calculate number of center lines to keep
    num_center_lines = int(center_fraction * width)
    center_start = width // 2 - num_center_lines // 2
    center_end = center_start + num_center_lines
    
    # Calculate total lines to keep
    total_lines = width // acceleration
    remaining_lines = total_lines - num_center_lines
    
    # Create mask
    mask = torch.zeros(width)
    mask[center_start:center_end] = 1
    
    # Randomly select remaining lines
    available_indices = list(range(width))
    for i in range(center_start, center_end):
        if i in available_indices:
            available_indices.remove(i)
    
    if remaining_lines > 0 and available_indices:
        selected_indices = torch.randperm(len(available_indices))[:remaining_lines]
        for idx in selected_indices:
            mask[available_indices[idx]] = 1
    
    # Expand to full k-space shape
    mask = mask.unsqueeze(0).expand(height, -1)
    return mask

def process_example_data():
    """Generate example input/output data."""
    
    # Load one test file
    test_data_path = "/scratch/vchaurasia/fastmri_data/test"
    test_files = list(Path(test_data_path).glob("*.h5"))
    file_path = test_files[0]  # Use first file
    
    print(f"Processing: {file_path.name}")
    
    with h5py.File(file_path, 'r') as f:
        kspace_data = f['kspace']
        slice_idx = 15  # Pick a middle slice
        
        # Get k-space for this slice
        kspace_full = torch.from_numpy(kspace_data[slice_idx])
        height, width = kspace_full.shape[-2:]
        
        # Create undersampling mask
        mask = create_undersampling_mask((height, width), acceleration=4, center_fraction=0.08, seed=42)
        
        # Apply mask to get undersampled k-space
        kspace_masked = apply_mask(kspace_full, mask)
        
        # Create ground truth image
        image_full = ifft2c(kspace_full)
        ground_truth = root_sum_of_squares(image_full, dim=0)
        ground_truth = torch.abs(ground_truth)
        
        # Create zero-filled reconstruction (THE INPUT for both models)
        image_zerofilled = ifft2c(kspace_masked)
        zero_filled_input = root_sum_of_squares(image_zerofilled, dim=0)
        zero_filled_input = torch.abs(zero_filled_input)
        
        # Normalize the input (same as models expect)
        mean = ground_truth.mean()
        std = ground_truth.std()
        ground_truth_norm = (ground_truth - mean) / (std + 1e-8)
        zero_filled_norm = (zero_filled_input - mean) / (std + 1e-8)
        
        return {
            'ground_truth': ground_truth_norm,
            'zero_filled_input': zero_filled_norm,  # This is the INPUT to both models
            'kspace_full': kspace_full,
            'kspace_masked': kspace_masked,
            'mask': mask,
            'mean': mean,
            'std': std,
            'file_info': file_path.name
        }

def load_models_and_infer(data):
    """Load both models and generate outputs."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare input data
    kspace_masked_real = complex_to_real(data['kspace_masked'].unsqueeze(0))
    mask_batch = data['mask'].unsqueeze(0)
    kspace_full_real = complex_to_real(data['kspace_full'].unsqueeze(0))
    
    # Move to device
    kspace_masked_real = kspace_masked_real.to(device)
    mask_batch = mask_batch.to(device)
    kspace_full_real = kspace_full_real.to(device)
    
    results = {}
    
    # Load and run UNet
    print("Loading UNet model...")
    unet_checkpoint = torch.load("/scratch/vchaurasia/organized_models/unet/best_model.pth", map_location=device)
    unet_config = unet_checkpoint['config']['model']
    unet_loss_config = unet_checkpoint['config']['loss']
    
    unet_model = MRIUNetModel(
        unet_config={
            'in_channels': unet_config['in_channels'],
            'out_channels': unet_config['out_channels'],
            'features': unet_config['features']
        },
        use_data_consistency=True,
        dc_weight=unet_loss_config['data_consistency_weight'],
        num_dc_iterations=5
    ).to(device)
    
    unet_model.load_state_dict(unet_checkpoint['model_state_dict'])
    unet_model.eval()
    
    with torch.no_grad():
        unet_output = unet_model(kspace_masked_real, mask_batch, kspace_full_real)
        if isinstance(unet_output, dict) and 'output' in unet_output:
            results['unet_output'] = unet_output['output'].squeeze().cpu()
        else:
            results['unet_output'] = unet_output.squeeze().cpu()
    
    # Load and run DISCO
    print("Loading DISCO model...")
    disco_checkpoint = torch.load("/scratch/vchaurasia/organized_models/disco/best_model.pth", map_location=device)
    disco_config = disco_checkpoint['config']['model']
    disco_loss_config = disco_checkpoint['config']['loss']
    
    disco_model = MRIReconstructionModel(
        neural_operator_config={
            'in_channels': disco_config['in_channels'],
            'out_channels': disco_config['out_channels'],
            'hidden_channels': disco_config['hidden_channels'],
            'num_layers': disco_config['num_layers'],
            'modes': disco_config['modes'],
            'width': disco_config.get('width', 64),
            'dropout': disco_config['dropout'],
            'use_residual': disco_config['use_residual'],
            'activation': disco_config['activation']
        },
        use_data_consistency=True,
        dc_weight=disco_loss_config['data_consistency_weight'],
        num_dc_iterations=5
    ).to(device)
    
    disco_model.load_state_dict(disco_checkpoint['model_state_dict'])
    disco_model.eval()
    
    with torch.no_grad():
        disco_output = disco_model(kspace_masked_real, mask_batch, kspace_full_real)
        if isinstance(disco_output, dict) and 'output' in disco_output:
            results['disco_output'] = disco_output['output'].squeeze().cpu()
        else:
            results['disco_output'] = disco_output.squeeze().cpu()
    
    return results

def create_visualization_figure(data, model_outputs):
    """Create the visualization figure."""
    
    # Extract images
    ground_truth = data['ground_truth'].numpy()
    zero_filled = data['zero_filled_input'].numpy()
    unet_output = model_outputs['unet_output'].numpy()
    disco_output = model_outputs['disco_output'].numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Pipeline overview
    axes[0, 0].imshow(ground_truth, cmap='gray')
    axes[0, 0].set_title('Ground Truth\n(Fully Sampled)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(zero_filled, cmap='gray')
    axes[0, 1].set_title('Zero-filled Input\n(4x Undersampled)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(unet_output, cmap='gray')
    axes[0, 2].set_title('U-Net Output\n(CNN Reconstruction)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(disco_output, cmap='gray')
    axes[0, 3].set_title('DISCO Output\n(Neural Operator)', fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')
    
    # Row 2: Error maps
    unet_error = np.abs(unet_output - ground_truth)
    disco_error = np.abs(disco_output - ground_truth)
    zero_error = np.abs(zero_filled - ground_truth)
    
    im1 = axes[1, 0].imshow(zero_error, cmap='hot', vmin=0, vmax=np.max(zero_error))
    axes[1, 0].set_title('Zero-filled Error', fontsize=12)
    axes[1, 0].axis('off')
    
    im2 = axes[1, 1].imshow(unet_error, cmap='hot', vmin=0, vmax=np.max(zero_error))
    axes[1, 1].set_title('U-Net Error', fontsize=12)
    axes[1, 1].axis('off')
    
    im3 = axes[1, 2].imshow(disco_error, cmap='hot', vmin=0, vmax=np.max(zero_error))
    axes[1, 2].set_title('DISCO Error', fontsize=12)
    axes[1, 2].axis('off')
    
    # Compute metrics for comparison
     
    unet_metrics = compute_metrics(torch.tensor(unet_output), torch.tensor(ground_truth))
    disco_metrics = compute_metrics(torch.tensor(disco_output), torch.tensor(ground_truth))
    zero_metrics = compute_metrics(torch.tensor(zero_filled), torch.tensor(ground_truth))
    
    # Text summary
    axes[1, 3].text(0.1, 0.8, 'PSNR Comparison:', fontsize=14, fontweight='bold', transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.1, 0.65, f'Zero-filled: {zero_metrics["psnr"]:.2f} dB', fontsize=12, transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.1, 0.5, f'U-Net: {unet_metrics["psnr"]:.2f} dB', fontsize=12, transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.1, 0.35, f'DISCO: {disco_metrics["psnr"]:.2f} dB', fontsize=12, transform=axes[1, 3].transAxes)
    
    axes[1, 3].text(0.1, 0.15, 'Both neural networks\ntake zero-filled as input\nand produce clean output', 
                    fontsize=10, style='italic', transform=axes[1, 3].transAxes)
    axes[1, 3].axis('off')
    
    # Add arrows showing pipeline
    fig.text(0.22, 0.65, '→', fontsize=30, ha='center')
    fig.text(0.47, 0.65, '→', fontsize=30, ha='center')
    fig.text(0.72, 0.65, '→', fontsize=30, ha='center')
    
    # Add method labels
    fig.text(0.47, 0.85, 'U-Net Pipeline', fontsize=14, ha='center', fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    fig.text(0.72, 0.85, 'DISCO Pipeline', fontsize=14, ha='center', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)
    
    # Save figure
    output_dir = "/scratch/vchaurasia/organized_models/presentation_figures"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(f"{output_dir}/unet_vs_disco_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/unet_vs_disco_comparison.pdf", bbox_inches='tight')
    
    print(f"Figure saved to: {output_dir}/unet_vs_disco_comparison.png")
    
    # Also save individual images for slides
    individual_dir = f"{output_dir}/individual_images"
    os.makedirs(individual_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(ground_truth, cmap='gray')
    plt.title('Ground Truth (Fully Sampled)', fontsize=16)
    plt.axis('off')
    plt.savefig(f"{individual_dir}/ground_truth.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(zero_filled, cmap='gray')
    plt.title('Zero-filled Input (4x Undersampled)', fontsize=16)
    plt.axis('off')
    plt.savefig(f"{individual_dir}/zero_filled_input.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(unet_output, cmap='gray')
    plt.title('U-Net Output', fontsize=16)
    plt.axis('off')
    plt.savefig(f"{individual_dir}/unet_output.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(disco_output, cmap='gray')
    plt.title('DISCO Output', fontsize=16)
    plt.axis('off')
    plt.savefig(f"{individual_dir}/disco_output.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Individual images saved to: {individual_dir}/")

def main():
    """Generate visualization data and create figure."""
    print("=== GENERATING PRESENTATION FIGURES ===")
    
    # Process example data
    print("1. Processing example data...")
    data = process_example_data()
    
    # Generate model outputs
    print("2. Running inference with both models...")
    model_outputs = load_models_and_infer(data)
    
    # Create visualization
    print("3. Creating visualization figure...")
    create_visualization_figure(data, model_outputs)
    
    print("\n✅ Figure generation completed!")
    print("Files created:")
    print("  - unet_vs_disco_comparison.png (main figure)")
    print("  - unet_vs_disco_comparison.pdf (for presentations)")
    print("  - individual_images/ (separate images for slides)")

if __name__ == "__main__":
    main()