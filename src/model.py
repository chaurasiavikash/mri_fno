# File: src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import logging

from neural_operator import DISCONeuralOperator
from utils import complex_to_real, real_to_complex, fft2c, ifft2c, apply_mask


class MRIReconstructionModel(nn.Module):
    """
    Complete MRI Reconstruction Model using Neural Operators.
    
    This model combines the DISCO Neural Operator with data consistency layers
    and handles the complete reconstruction pipeline from masked k-space to images.
    """
    
    def __init__(
        self,
        neural_operator_config: Dict[str, Any],
        use_data_consistency: bool = True,
        dc_weight: float = 1.0,
        num_dc_iterations: int = 5
    ):
        """
        Initialize MRI Reconstruction Model.
        
        Args:
            neural_operator_config: Configuration for neural operator
            use_data_consistency: Whether to use data consistency layers
            dc_weight: Weight for data consistency term
            num_dc_iterations: Number of data consistency iterations
        """
        super().__init__()
        
        self.use_data_consistency = use_data_consistency
        self.dc_weight = dc_weight
        self.num_dc_iterations = num_dc_iterations
        
        # Neural operator for k-space to image reconstruction
        self.neural_operator = DISCONeuralOperator(**neural_operator_config)
        
        # Data consistency layers
        if use_data_consistency:
            self.dc_layers = nn.ModuleList([
                DataConsistencyLayer(dc_weight) for _ in range(num_dc_iterations)
            ])
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def forward(
    self,
    kspace_masked: torch.Tensor,
    mask: torch.Tensor,
    kspace_full: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the reconstruction model.
        
        Args:
            kspace_masked: Masked k-space data (batch, coils, 2, height, width)
            mask: Undersampling mask (batch, height, width) or (height, width)
            kspace_full: Full k-space data for training (optional)
            
        Returns:
            Dictionary containing:
                - 'output': Final reconstructed image
                - 'kspace_pred': Predicted k-space
                - 'intermediate': Intermediate results (if applicable)
        """
        batch_size = kspace_masked.shape[0]
        
        # Ensure mask has correct dimensions for data consistency layers
        # The mask should be (batch, height, width) for data consistency
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        elif mask.dim() == 3 and mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1)
        
        # Initialize with zero-filled reconstruction
        current_kspace = kspace_masked.clone()
        
        # Apply neural operator
        # First, convert to image domain for processing
        current_image = self._kspace_to_image(current_kspace)
        
        # Process with neural operator (operates on combined coil data)
        combined_image = self._combine_coils(current_image)
        
        # Prepare input for neural operator: convert magnitude image to 2-channel format
        # For neural operator, we create a 2-channel representation
        if combined_image.dim() == 3:  # (batch, height, width)
            # Create a 2-channel version: magnitude + zeros
            combined_image_2ch = torch.stack([combined_image, torch.zeros_like(combined_image)], dim=1)
        else:
            combined_image_2ch = combined_image
        
        # Neural operator prediction
        predicted_image_2ch = self.neural_operator(combined_image_2ch)
        
        # Ensure predicted image has correct format
        if predicted_image_2ch.shape[1] == 2:
            # Convert predicted image back to complex for k-space conversion
            predicted_image = real_to_complex(predicted_image_2ch)
        else:
            # If neural operator outputs single channel, make it complex
            predicted_image = predicted_image_2ch.squeeze(1)
            predicted_image = torch.complex(predicted_image, torch.zeros_like(predicted_image))
        
        # Convert back to k-space (this will be single-coil k-space)
        predicted_kspace_single = self._image_to_kspace(predicted_image)
        
        # Expand to multi-coil format by replicating across coils
        num_coils = current_kspace.shape[1]
        predicted_kspace = predicted_kspace_single.unsqueeze(1).expand(-1, num_coils, -1, -1)
        
        # Convert to real/imaginary format to match input
        predicted_kspace = complex_to_real(predicted_kspace)
        
        # Apply data consistency if enabled
        if self.use_data_consistency:
            current_kspace = predicted_kspace
            for dc_layer in self.dc_layers:
                current_kspace = dc_layer(current_kspace, kspace_masked, mask)
        else:
            current_kspace = predicted_kspace
        
        # Final image reconstruction
        final_image = self._kspace_to_image(current_kspace)
        output_image = self._combine_coils(final_image)

        # Ensure output is single-channel magnitude image
        if torch.is_complex(output_image):
            output_magnitude = torch.abs(output_image)
        elif output_image.dim() == 4 and output_image.shape[1] == 2:
            # If output has 2 channels, take the magnitude
            output_magnitude = torch.sqrt(output_image[:, 0] ** 2 + output_image[:, 1] ** 2)
        elif output_image.dim() == 4:
            # If output has unexpected channels, take the first one
            output_magnitude = output_image[:, 0]
        else:
            output_magnitude = torch.abs(output_image)

        # Ensure final output is (batch, height, width)
        if output_magnitude.dim() == 4:
            output_magnitude = output_magnitude.squeeze(1)

        # Always return image_complex as (batch, height, width)
        if output_image.dim() == 4 and output_image.shape[1] == 2:
            image_complex_out = torch.sqrt(output_image[:, 0] ** 2 + output_image[:, 1] ** 2)
        elif output_image.dim() == 4:
            image_complex_out = output_image[:, 0]
        elif torch.is_complex(output_image):
            image_complex_out = torch.abs(output_image)
        else:
            image_complex_out = output_image

        if image_complex_out.dim() == 4:
            image_complex_out = image_complex_out.squeeze(1)

        return {
            'output': output_magnitude,
            'kspace_pred': current_kspace,
            'image_complex': image_complex_out,
            'intermediate': {
                'no_pred': torch.abs(predicted_image),
                'zero_filled': combined_image if combined_image.dim() == 3 else torch.abs(combined_image)
            }
        }
    
    def _kspace_to_image(self, kspace: torch.Tensor) -> torch.Tensor:
        """Convert k-space to image domain using IFFT."""
        # Handle real/imaginary channels
        if kspace.shape[-1] == 2:  # Real/imaginary format
            kspace_complex = real_to_complex(kspace)
        else:
            kspace_complex = kspace
        
        return ifft2c(kspace_complex)
    
    def _image_to_kspace(self, image: torch.Tensor) -> torch.Tensor:
        """Convert image to k-space domain using FFT."""
        return fft2c(image)
    
    def _combine_coils(self, coil_images: torch.Tensor) -> torch.Tensor:
        """Combine multi-coil images using root sum of squares."""
        if torch.is_complex(coil_images):
            # Complex format: (batch, coils, height, width)
            combined = torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=1, keepdim=False))
        elif coil_images.dim() == 4:
            # Real format: (batch, coils, height, width) - treat as magnitude
            combined = torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=1, keepdim=False))
        elif coil_images.dim() == 5 and coil_images.shape[-3] == 2:
            # Real/imaginary format: (batch, coils, 2, height, width)
            complex_images = real_to_complex(coil_images)
            combined = torch.sqrt(torch.sum(torch.abs(complex_images) ** 2, dim=1, keepdim=False))
        else:
            # Fallback - assume single coil or already combined
            combined = coil_images.squeeze(1) if coil_images.dim() > 3 else coil_images
        
        # Ensure output is 3D: (batch, height, width)
        if combined.dim() == 4:
            combined = combined.squeeze(1)
        
        return combined


class DataConsistencyLayer(nn.Module):
    """
    Data Consistency Layer for enforcing consistency with acquired k-space data.
    
    This layer ensures that the predicted k-space data matches the acquired data
    in the sampled regions while allowing the network to fill in missing data.
    """
    
    def __init__(self, weight: float = 1.0):
        """
        Initialize data consistency layer.
        
        Args:
            weight: Weight for data consistency term
        """
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        predicted_kspace: torch.Tensor,
        measured_kspace: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply data consistency.
        
        Args:
            predicted_kspace: Predicted k-space data
            measured_kspace: Measured k-space data
            mask: Sampling mask
            
        Returns:
            Data-consistent k-space
        """
        # Convert to complex if needed
        if predicted_kspace.shape[-1] == 2:
            pred_complex = real_to_complex(predicted_kspace)
            meas_complex = real_to_complex(measured_kspace)
        else:
            pred_complex = predicted_kspace
            meas_complex = measured_kspace
        
        # Handle mask broadcasting properly
        # pred_complex shape: (batch, coils, height, width)
        # mask can be (height, width), (batch, height, width), etc.

        if mask.dim() == 2:  # (height, width)
            # Expand to match (batch, coils, height, width)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.expand(pred_complex.shape[0], pred_complex.shape[1], -1, -1)
        elif mask.dim() == 3:  # (batch, height, width)
            # Expand to (batch, coils, height, width)
            mask = mask.unsqueeze(1)
            mask = mask.expand(-1, pred_complex.shape[1], -1, -1)
        elif mask.dim() == 4:  # (batch, 1, height, width) or similar
            # Expand to (batch, coils, height, width)
            mask = mask.expand(-1, pred_complex.shape[1], -1, -1)

        # If input has a channel dimension (e.g., real/imag), expand mask accordingly
        if predicted_kspace.dim() == 5 and mask.dim() == 4:
            # (batch, coils, 1, height, width)
            mask = mask.unsqueeze(2)

        # Apply data consistency: use measured data where available, predicted elsewhere
        dc_kspace = pred_complex * (1 - mask) + meas_complex * mask * self.weight + pred_complex * mask * (1 - self.weight)
        
        # Convert back to real/imaginary format if needed
        if predicted_kspace.shape[-1] == 2:
            return complex_to_real(dc_kspace)
        else:
            return dc_kspace


class ReconstructionLoss(nn.Module):
    """
    Combined loss function for MRI reconstruction.
    
    Combines multiple loss terms including L1, SSIM, perceptual loss, and data consistency.
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.1,
        perceptual_weight: float = 0.01,
        data_consistency_weight: float = 1.0
    ):
        """
        Initialize reconstruction loss.
        
        Args:
            l1_weight: Weight for L1 loss
            ssim_weight: Weight for SSIM loss
            perceptual_weight: Weight for perceptual loss
            data_consistency_weight: Weight for data consistency loss
        """
        super().__init__()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.data_consistency_weight = data_consistency_weight
        
        # Loss components
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        predicted_kspace: Optional[torch.Tensor] = None,
        target_kspace: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            prediction: Predicted image
            target: Target image
            predicted_kspace: Predicted k-space (optional)
            target_kspace: Target k-space (optional)
            mask: Sampling mask (optional)
            
        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}
        
        # L1 loss (primary image loss)
        losses['l1'] = self.l1_loss(prediction, target) * self.l1_weight
        
        # SSIM loss
        if self.ssim_weight > 0:
            losses['ssim'] = self._ssim_loss(prediction, target) * self.ssim_weight
        
        # Data consistency loss
        if (self.data_consistency_weight > 0 and 
            predicted_kspace is not None and 
            target_kspace is not None and 
            mask is not None):
            losses['data_consistency'] = self._data_consistency_loss(
                predicted_kspace, target_kspace, mask
            ) * self.data_consistency_weight
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def _ssim_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss."""
        # Simple SSIM approximation using local statistics
        mu_pred = F.avg_pool2d(prediction, 3, 1, 1)
        mu_target = F.avg_pool2d(target, 3, 1, 1)
        
        mu_pred_sq = mu_pred * mu_pred
        mu_target_sq = mu_target * mu_target
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool2d(prediction * prediction, 3, 1, 1) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target * target, 3, 1, 1) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(prediction * target, 3, 1, 1) - mu_pred_target
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / \
                   ((mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2))
        
        return 1 - ssim_map.mean()
    
    def _data_consistency_loss(
        self,
        predicted_kspace: torch.Tensor,
        target_kspace: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute data consistency loss."""
        # Handle mask broadcasting properly
        # predicted_kspace shape: (batch, coils, 2, height, width)
        # mask can be (height, width), (batch, height, width), etc.
        
        if mask.dim() == 2:  # (height, width)
            # Expand to match predicted_kspace: (1, 1, 1, height, width)
            mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:  # (batch, height, width)
            # Expand to (batch, 1, 1, height, width)
            mask = mask.unsqueeze(1).unsqueeze(1)
        elif mask.dim() == 4:  # (batch, 1, height, width)
            # Expand to (batch, 1, 1, height, width)
            mask = mask.unsqueeze(2)
        
        # Expand mask to match predicted_kspace dimensions
        mask = mask.expand_as(predicted_kspace)
        
        # Compute loss only on sampled k-space locations
        masked_pred = predicted_kspace * mask
        masked_target = target_kspace * mask
        
        return self.mse_loss(masked_pred, masked_target)