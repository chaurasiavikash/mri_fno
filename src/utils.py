# File: src/utils.py

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
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
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
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cuda", "cpu")
        
    Returns:
        PyTorch device object
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device)


def complex_to_real(x: torch.Tensor) -> torch.Tensor:
    """
    Convert complex tensor to real tensor with separate real/imaginary channels.
    
    Args:
        x: Complex tensor of shape (..., H, W)
        
    Returns:
        Real tensor of shape (..., 2, H, W) where channel 0 is real, channel 1 is imaginary
    """
    if not torch.is_complex(x):
        raise ValueError(f"Input tensor must be complex, got {x.dtype}")
    
    # Stack real and imaginary parts along new dimension
    real_part = torch.real(x)
    imag_part = torch.imag(x)
    
    # Insert the channel dimension at the second-to-last position
    # For shape (..., H, W) -> (..., 2, H, W)
    return torch.stack([real_part, imag_part], dim=-3)


def real_to_complex(x: torch.Tensor) -> torch.Tensor:
    """
    Convert real tensor with separate real/imaginary channels to complex tensor.
    
    Args:
        x: Real tensor of shape (..., 2, H, W) where channel 0 is real, channel 1 is imaginary
        
    Returns:
        Complex tensor of shape (..., H, W)
    """
    if x.shape[-3] != 2:
        raise ValueError(f"Expected 2 channels for real/imaginary, got shape {x.shape}")
    
    real_part = x[..., 0, :, :]
    imag_part = x[..., 1, :, :]
    
    return torch.complex(real_part, imag_part)


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """
    Centered 2D Fast Fourier Transform.
    
    Args:
        x: Input tensor
        
    Returns:
        FFT of input tensor
    """
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """
    Centered 2D Inverse Fast Fourier Transform.
    
    Args:
        x: Input tensor in frequency domain
        
    Returns:
        IFFT of input tensor
    """
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))


def apply_mask(kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply undersampling mask to k-space data.
    
    Args:
        kspace: K-space data tensor
        mask: Undersampling mask
        
    Returns:
        Masked k-space data
    """
    # Ensure mask has the same number of dimensions as kspace
    while mask.dim() < kspace.dim():
        mask = mask.unsqueeze(0)
    
    # Broadcast mask to match kspace shape
    mask = mask.expand_as(kspace)
    
    return kspace * mask


def root_sum_of_squares(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute root sum of squares along specified dimension.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute RSS
        
    Returns:
        RSS tensor
    """
    return torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=dim, keepdim=False))


def normalize_tensor(x: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize tensor to zero mean and unit variance.
    
    Args:
        x: Input tensor
        eps: Small value to avoid division by zero
        
    Returns:
        Tuple of (normalized tensor, mean, std)
    """
    mean = torch.mean(x)
    std = torch.std(x)
    normalized = (x - mean) / (std + eps)
    return normalized, mean, std


def denormalize_tensor(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Denormalize tensor using provided mean and std.
    
    Args:
        x: Normalized tensor
        mean: Mean value used for normalization
        std: Standard deviation used for normalization
        
    Returns:
        Denormalized tensor
    """
    return x * std + mean


def save_tensor_as_image(tensor: torch.Tensor, filepath: str, title: str = None) -> None:
    """
    Save tensor as image file.
    
    Args:
        tensor: Tensor to save (assumed to be 2D)
        filepath: Path to save the image
        title: Optional title for the image
    """
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
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)