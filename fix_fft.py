# File: fix_fft_functions.py
"""
Debug and fix the FFT functions that are causing zero reconstructions.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def test_fft_functions():
    """Test the current FFT implementations."""
    print("🔍 TESTING CURRENT FFT IMPLEMENTATIONS")
    print("=" * 50)
    
    try:
        from utils import fft2c, ifft2c
        print("✅ Imported current FFT functions")
        
        # Create test data
        test_image = torch.randn(64, 64, dtype=torch.complex64)
        print(f"Test image shape: {test_image.shape}")
        print(f"Test image max: {torch.max(torch.abs(test_image)):.6f}")
        
        # Test forward FFT
        kspace = fft2c(test_image)
        print(f"FFT result shape: {kspace.shape}")
        print(f"FFT result max: {torch.max(torch.abs(kspace)):.6f}")
        
        # Test inverse FFT
        recovered = ifft2c(kspace)
        print(f"IFFT result shape: {recovered.shape}")
        print(f"IFFT result max: {torch.max(torch.abs(recovered)):.6f}")
        
        # Check round-trip error
        error = torch.mean(torch.abs(test_image - recovered))
        print(f"Round-trip error: {error:.8f}")
        
        if error < 1e-6:
            print("✅ FFT functions work correctly")
            return True
        else:
            print("❌ FFT functions have issues")
            return False
            
    except Exception as e:
        print(f"❌ Error testing FFT functions: {e}")
        return False


def test_corrected_fft():
    """Test corrected FFT implementation."""
    print(f"\n🔧 TESTING CORRECTED FFT IMPLEMENTATION")
    print("=" * 50)
    
    def corrected_fft2c(x):
        """Corrected centered 2D FFT."""
        return torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.ifftshift(x, dim=(-2, -1)), 
                dim=(-2, -1), 
                norm='ortho'  # This is important!
            ), 
            dim=(-2, -1)
        )
    
    def corrected_ifft2c(x):
        """Corrected centered 2D IFFT."""
        return torch.fft.fftshift(
            torch.fft.ifft2(
                torch.fft.ifftshift(x, dim=(-2, -1)), 
                dim=(-2, -1),
                norm='ortho'  # This is important!
            ), 
            dim=(-2, -1)
        )
    
    # Test with various data
    test_cases = [
        torch.randn(64, 64, dtype=torch.complex64),
        torch.randn(128, 128, dtype=torch.complex64) * 100,
        torch.ones(32, 32, dtype=torch.complex64),
    ]
    
    all_good = True
    
    for i, test_data in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"  Input max: {torch.max(torch.abs(test_data)):.6f}")
        
        # Forward FFT
        kspace = corrected_fft2c(test_data)
        print(f"  FFT max: {torch.max(torch.abs(kspace)):.6f}")
        
        # Inverse FFT
        recovered = corrected_ifft2c(kspace)
        print(f"  IFFT max: {torch.max(torch.abs(recovered)):.6f}")
        
        # Error
        error = torch.mean(torch.abs(test_data - recovered))
        print(f"  Error: {error:.8f}")
        
        if error > 1e-6:
            print(f"  ❌ High error!")
            all_good = False
        else:
            print(f"  ✅ Good")
    
    if all_good:
        print(f"\n✅ Corrected FFT functions work!")
    
    return corrected_fft2c, corrected_ifft2c


def test_with_real_data(corrected_fft2c, corrected_ifft2c):
    """Test with real k-space data."""
    print(f"\n🧪 TESTING WITH REAL K-SPACE DATA")
    print("=" * 50)
    
    # Load the best file we found
    data_path = "/scratch/vchaurasia/fastmri_data/test/file1000186.h5"
    
    with h5py.File(data_path, 'r') as f:
        # Load k-space with meaningful data
        kspace_data = f['kspace'][17]  # Slice 17 had max=0.004203
        
        print(f"K-space shape: {kspace_data.shape}")
        print(f"K-space max: {np.max(np.abs(kspace_data)):.6f}")
        
        # Convert to torch
        kspace_torch = torch.from_numpy(kspace_data)
        
        # Test current implementation
        try:
            from utils import ifft2c
            coil_images_old = ifft2c(kspace_torch)
            rss_old = torch.sqrt(torch.sum(torch.abs(coil_images_old) ** 2, dim=0))
            print(f"Current IFFT result max: {torch.max(rss_old):.6f}")
        except Exception as e:
            print(f"Current IFFT failed: {e}")
            rss_old = torch.zeros(1)
        
        # Test corrected implementation
        coil_images_new = corrected_ifft2c(kspace_torch)
        rss_new = torch.sqrt(torch.sum(torch.abs(coil_images_new) ** 2, dim=0))
        print(f"Corrected IFFT result max: {torch.max(rss_new):.6f}")
        
        # Create visualization
        output_dir = Path("/scratch/vchaurasia/data_debug")
        output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # K-space (first coil)
        kspace_log = np.log(np.abs(kspace_data[0]) + 1e-8)
        axes[0, 0].imshow(kspace_log, cmap='viridis')
        axes[0, 0].set_title('K-space (log)')
        axes[0, 0].axis('off')
        
        # Old reconstruction
        if torch.max(rss_old) > 0:
            axes[0, 1].imshow(rss_old.numpy(), cmap='gray')
            axes[0, 1].set_title(f'Old IFFT\nMax: {torch.max(rss_old):.6f}')
        else:
            axes[0, 1].text(0.5, 0.5, 'All Zeros', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Old IFFT (Failed)')
        axes[0, 1].axis('off')
        
        # New reconstruction
        rss_new_np = rss_new.numpy()
        if torch.max(rss_new) > 0:
            axes[0, 2].imshow(rss_new_np, cmap='gray')
            axes[0, 2].set_title(f'Corrected IFFT\nMax: {torch.max(rss_new):.6f}')
        else:
            axes[0, 2].text(0.5, 0.5, 'Still Zeros', ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Corrected IFFT (Still Failed)')
        axes[0, 2].axis('off')
        
        # Individual coil (new method)
        coil_0_new = torch.abs(coil_images_new[0]).numpy()
        axes[0, 3].imshow(coil_0_new, cmap='gray')
        axes[0, 3].set_title(f'Coil 0\nMax: {np.max(coil_0_new):.6f}')
        axes[0, 3].axis('off')
        
        # Test different normalizations
        scales = [1.0, 1e3, 1e6, 1e9]
        for i, scale in enumerate(scales):
            scaled_kspace = kspace_torch * scale
            coil_images_scaled = corrected_ifft2c(scaled_kspace)
            rss_scaled = torch.sqrt(torch.sum(torch.abs(coil_images_scaled) ** 2, dim=0))
            
            if torch.max(rss_scaled) > 0:
                axes[1, i].imshow(rss_scaled.numpy(), cmap='gray')
            else:
                axes[1, i].text(0.5, 0.5, 'Zeros', ha='center', va='center', transform=axes[1, i].transAxes)
            
            axes[1, i].set_title(f'Scale x{scale:.0e}\nMax: {torch.max(rss_scaled):.6f}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plot_path = output_dir / "fft_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison saved to: {plot_path}")
        
        return torch.max(rss_new).item() > 0


def create_corrected_utils():
    """Create corrected utils.py with working FFT functions."""
    print(f"\n📝 CREATING CORRECTED UTILS.PY")
    print("=" * 50)
    
    corrected_code = '''
import torch
import numpy as np
import random
import yaml
import logging
import os
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device)


def complex_to_real(tensor: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor to real tensor with separate real/imaginary channels."""
    if not torch.is_complex(tensor):
        return tensor
    
    # Stack real and imaginary parts along a new dimension
    return torch.stack([tensor.real, tensor.imag], dim=-1)


def real_to_complex(tensor: torch.Tensor) -> torch.Tensor:
    """Convert real tensor with separate real/imaginary channels to complex tensor."""
    if torch.is_complex(tensor):
        return tensor
    
    # Assume last dimension contains real and imaginary parts
    if tensor.shape[-1] != 2:
        raise ValueError(f"Last dimension must be 2 for real/imag, got {tensor.shape[-1]}")
    
    return torch.complex(tensor[..., 0], tensor[..., 1])


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """
    Centered 2D Fast Fourier Transform.
    
    Args:
        x: Input tensor (complex)
        
    Returns:
        FFT of input tensor
    """
    return torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(x, dim=(-2, -1)), 
            dim=(-2, -1),
            norm='ortho'
        ), 
        dim=(-2, -1)
    )


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """
    Centered 2D Inverse Fast Fourier Transform.
    
    Args:
        x: Input tensor in frequency domain (complex)
        
    Returns:
        IFFT of input tensor
    """
    return torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(x, dim=(-2, -1)), 
            dim=(-2, -1),
            norm='ortho'
        ), 
        dim=(-2, -1)
    )


def apply_mask(kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply undersampling mask to k-space data."""
    # Ensure mask has the right shape
    while mask.dim() < kspace.dim():
        mask = mask.unsqueeze(0)
    
    # Broadcast mask to match k-space
    mask = mask.expand_as(kspace)
    
    return kspace * mask


def root_sum_of_squares(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Compute root sum of squares along specified dimension."""
    return torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=dim, keepdim=True))


def normalize_tensor(x: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize tensor to zero mean and unit variance."""
    mean = torch.mean(x)
    std = torch.std(x)
    normalized = (x - mean) / (std + eps)
    return normalized, mean, std


def denormalize_tensor(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor using provided mean and std."""
    return x * std + mean


def save_tensor_as_image(tensor: torch.Tensor, filepath: str, title: str = None) -> None:
    """Save tensor as image file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert to numpy and handle complex tensors
    if torch.is_complex(tensor):
        image = torch.abs(tensor).cpu().numpy()
    else:
        image = tensor.cpu().numpy()
    
    # Squeeze to 2D if needed
    while image.ndim > 2:
        image = image.squeeze()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()


def create_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
'''
    
    # Save corrected utils
    output_path = "/scratch/vchaurasia/data_debug/corrected_utils.py"
    with open(output_path, 'w') as f:
        f.write(corrected_code)
    
    print(f"Corrected utils.py saved to: {output_path}")
    
    return output_path


def main():
    """Main function."""
    print("🔧 FFT FUNCTIONS DEBUG AND FIX")
    print("=" * 60)
    print("This script will identify and fix the FFT issues causing zero reconstructions.")
    print()
    
    try:
        # Test current FFT functions
        current_works = test_fft_functions()
        
        # Test corrected FFT functions
        corrected_fft2c, corrected_ifft2c = test_corrected_fft()
        
        # Test with real data
        real_data_works = test_with_real_data(corrected_fft2c, corrected_ifft2c)
        
        # Create corrected utils
        utils_path = create_corrected_utils()
        
        print(f"\n🎯 RESULTS:")
        print(f"   Current FFT functions work: {'✅' if current_works else '❌'}")
        print(f"   Real data reconstruction works: {'✅' if real_data_works else '❌'}")
        
        if real_data_works:
            print(f"\n✅ SUCCESS! FFT functions are working.")
            print(f"   The issue might be elsewhere in the data pipeline.")
            print(f"   Check the visualization in /scratch/vchaurasia/data_debug/")
        else:
            print(f"\n⚠️ FFT functions fixed but still getting zeros.")
            print(f"   The issue might be:")
            print(f"   1. K-space data scaling/normalization")
            print(f"   2. Complex number handling")
            print(f"   3. Data format issues")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Replace src/utils.py with: {utils_path}")
        print(f"   2. Check the visualization plots")
        print(f"   3. Test with the corrected functions")
        print(f"   4. Retrain both DISCO and U-Net")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())