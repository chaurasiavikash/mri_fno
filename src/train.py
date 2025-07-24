# File: src/train.py

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Local imports
from model import MRIReconstructionModel, ReconstructionLoss
from data_loader import create_data_loaders
from utils import (
    load_config, set_seed, setup_logging, get_device, 
    create_directory, count_parameters, save_tensor_as_image
)


class MRITrainer:
    """
    Trainer class for MRI reconstruction using neural operators.
    
    Handles the complete training pipeline including data loading, model training,
    validation, checkpointing, and logging.
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            device: Device to use for training
        """
        self.config = config
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self._setup_directories()
        self._setup_data_loaders()
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss()
        self._setup_logging()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.logger.info(f"Trainer initialized with {count_parameters(self.model)} parameters")
    
    def _setup_directories(self):
        """Setup output directories."""
        self.log_dir = Path(self.config['logging']['log_dir'])
        self.model_dir = Path(self.config['logging']['save_model_dir'])
        self.results_dir = Path(self.config['logging']['results_dir'])
        
        create_directory(self.log_dir)
        create_directory(self.model_dir)
        create_directory(self.results_dir)
    
    def _setup_data_loaders(self):
        """Setup data loaders."""
        data_config = self.config['data']
        
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            train_path=data_config['train_path'],
            val_path=data_config['val_path'],
            test_path=data_config['test_path'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            acceleration=data_config['acceleration'],
            center_fraction=data_config['center_fraction'],
            use_normalization=True
        )
        
        self.logger.info(f"Data loaders created: "
                        f"Train: {len(self.train_loader)}, "
                        f"Val: {len(self.val_loader)}, "
                        f"Test: {len(self.test_loader)}")
    
    def _setup_model(self):
        """Setup model."""
        model_config = self.config['model']
        
        self.model = MRIReconstructionModel(
            neural_operator_config={
                'in_channels': model_config['in_channels'],
                'out_channels': model_config['out_channels'],
                'hidden_channels': model_config['hidden_channels'],
                'num_layers': model_config['num_layers'],
                'modes': model_config['modes'],
                'width': model_config['width'],
                'dropout': model_config['dropout'],
                'use_residual': model_config['use_residual'],
                'activation': model_config['activation']
            },
            use_data_consistency=True,
            dc_weight=self.config['loss']['data_consistency_weight'],
            num_dc_iterations=5
        ).to(self.device)
        
        # Enable mixed precision only if model does not have complex parameters
        if any(p.is_complex() for p in self.model.parameters()):
            self.scaler = None  # Disable mixed precision for complex models
        elif self.config['training']['mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        optimizer_config = self.config['optimizer']
        training_config = self.config['training']
        
        if optimizer_config['name'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                betas=optimizer_config['betas'],
                eps=optimizer_config['eps'],
                amsgrad=optimizer_config['amsgrad']
            )
        elif optimizer_config['name'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                betas=optimizer_config['betas'],
                eps=optimizer_config['eps'],
                amsgrad=optimizer_config['amsgrad']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        training_config = self.config['training']
        
        if training_config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['epochs'],
                eta_min=training_config['learning_rate'] * 0.01
            )
        elif training_config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=training_config['epochs'] // 3,
                gamma=0.1
            )
        elif training_config['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            self.scheduler = None
        
        # Warmup scheduler
        if training_config.get('warmup_epochs', 0) > 0:
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=training_config['warmup_epochs']
            )
        else:
            self.warmup_scheduler = None
    
    def _setup_loss(self):
        """Setup loss function."""
        loss_config = self.config['loss']
        
        self.criterion = ReconstructionLoss(
            l1_weight=loss_config['l1_weight'],
            ssim_weight=loss_config['ssim_weight'],
            perceptual_weight=loss_config.get('perceptual_weight', 0.0),
            data_consistency_weight=loss_config['data_consistency_weight']
        )
    
    def _setup_logging(self):
        """Setup tensorboard logging."""
        if self.config['logging'].get('use_wandb', False):
            try:
                import wandb
                wandb.init(
                    project=self.config['logging']['project_name'],
                    config=self.config
                )
                self.use_wandb = True
            except ImportError:
                self.logger.warning("wandb not available, using tensorboard only")
                self.use_wandb = False
        else:
            self.use_wandb = False
        
        self.writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            kspace_full = batch['kspace_full'].to(self.device, non_blocking=True)
            kspace_masked = batch['kspace_masked'].to(self.device, non_blocking=True)
            mask = batch['mask'].to(self.device, non_blocking=True)
            target = batch['target'].to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.model(kspace_masked, mask, kspace_full)
                    losses = self.criterion(
                        prediction=output['output'],
                        target=target,
                        predicted_kspace=output['kspace_pred'],
                        target_kspace=kspace_full,
                        mask=mask
                    )
                    loss = losses['total']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(kspace_masked, mask, kspace_full)
                losses = self.criterion(
                    prediction=output['output'],
                    target=target,
                    predicted_kspace=output['kspace_pred'],
                    target_kspace=kspace_full,
                    mask=mask
                )
                loss = losses['total']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Update learning rate (warmup)
            if self.warmup_scheduler is not None and self.epoch < self.config['training'].get('warmup_epochs', 0):
                self.warmup_scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to tensorboard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                # Log individual loss components
                for loss_name, loss_value in losses.items():
                    if loss_name != 'total':
                        self.writer.add_scalar(f'train/{loss_name}', loss_value.item(), self.global_step)
            
            self.global_step += 1
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        return {'train_loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                # Move data to device
                kspace_full = batch['kspace_full'].to(self.device, non_blocking=True)
                kspace_masked = batch['kspace_masked'].to(self.device, non_blocking=True)
                mask = batch['mask'].to(self.device, non_blocking=True)
                target = batch['target'].to(self.device, non_blocking=True)
                
                # Forward pass
                output = self.model(kspace_masked, mask, kspace_full)
                losses = self.criterion(
                    prediction=output['output'],
                    target=target,
                    predicted_kspace=output['kspace_pred'],
                    target_kspace=kspace_full,
                    mask=mask
                )
                
                total_loss += losses['total'].item()
                
                # Save sample images for first batch
                if batch_idx == 0 and self.epoch % 10 == 0:
                    self._save_sample_images(batch, output, prefix=f"val_epoch_{self.epoch}")
        
        avg_loss = total_loss / num_batches
        
        return {'val_loss': avg_loss}
    
    def _save_sample_images(self, batch: Dict, output: Dict, prefix: str = "sample"):
        """Save sample images for visualization."""
        # Save first sample in batch
        target_img = batch['target'][0].cpu()
        pred_img = output['output'][0].cpu()
        input_img = batch['image_masked'][0].cpu()
        
        # Save images
        save_dir = self.results_dir / "samples"
        create_directory(save_dir)
        
        save_tensor_as_image(target_img, save_dir / f"{prefix}_target.png", "Target")
        save_tensor_as_image(pred_img, save_dir / f"{prefix}_prediction.png", "Prediction")
        save_tensor_as_image(input_img, save_dir / f"{prefix}_input.png", "Zero-filled")
        
        # Log to tensorboard
        self.writer.add_image(f'{prefix}/target', target_img.unsqueeze(0), self.epoch)
        self.writer.add_image(f'{prefix}/prediction', pred_img.unsqueeze(0), self.epoch)
        self.writer.add_image(f'{prefix}/input', input_img.unsqueeze(0), self.epoch)
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, self.model_dir / filename)
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.model_dir / 'best_model.pth')
        
        self.logger.info(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.model_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Update training state
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Checkpoint loaded: {filename} (epoch {self.epoch})")
    
    def train(self, num_epochs: Optional[int] = None):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train (uses config if None)
        """
        if num_epochs is None:
            num_epochs = self.config['training']['epochs']
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        initial_epoch = self.epoch
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            if epoch % self.config['training']['validate_every'] == 0:
                val_metrics = self.validate()
            else:
                val_metrics = {}
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if 'val_loss' in val_metrics:
                        self.scheduler.step(val_metrics['val_loss'])
                else:
                    if self.epoch >= self.config['training'].get('warmup_epochs', 0):
                        self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics.get('val_loss', 'N/A')}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Log to tensorboard
            self.writer.add_scalar('train/epoch_loss', train_metrics['train_loss'], epoch)
            if val_metrics:
                self.writer.add_scalar('val/epoch_loss', val_metrics['val_loss'], epoch)
            
            # Log to wandb
            if self.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['train_loss'],
                    'val_loss': val_metrics.get('val_loss'),
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time
                })
            
            # Save checkpoint
            if epoch % self.config['training']['save_every'] == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
            
            # Save best model
            if val_metrics and val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth", is_best=True)
        
        self.logger.info("Training completed!")
        
        # Close tensorboard writer
        self.writer.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train MRI Reconstruction Model")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device and seed
    device = get_device(args.device)
    set_seed(config['system']['seed'])
    
    # Setup logging
    setup_logging(
        config['logging']['log_dir'],
        config['system']['log_level']
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting MRI reconstruction training on {device}")
    logger.info(f"Configuration: {args.config}")
    
    # Create trainer
    trainer = MRITrainer(config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint("checkpoint_interrupted.pth")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        trainer.save_checkpoint("checkpoint_error.pth")
        raise


if __name__ == "__main__":
    main()

