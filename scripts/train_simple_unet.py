# File: scripts/train_simple_unet.py
"""
Simple training script based on your working repository pattern.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from simple_unet import SimpleMRIUNet
from data_loader import FastMRIDataset


class SimpleLoss(nn.Module):
    """Simple loss function - just L1 loss."""
    
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, prediction, target):
        return self.l1_loss(prediction, target)


def create_simple_dataloaders(train_path, val_path, batch_size=1, num_workers=4):
    """Create simple data loaders."""
    
    # Simple datasets without complex transforms
    train_dataset = FastMRIDataset(
        data_path=train_path,
        acceleration=4,
        center_fraction=0.08,
        transform=None,
        use_seed=True,
        seed=42
    )
    
    val_dataset = FastMRIDataset(
        data_path=val_path,
        acceleration=4,
        center_fraction=0.08,
        transform=None,
        use_seed=True,
        seed=42
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0  # FIXED: Initialize total_loss
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # DEBUG: Add shape checking
            if batch_idx == 0:  # Only debug first batch
                print(f"\n=== BATCH {batch_idx} DEBUG ===")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"{key}: {value.shape}")
            
            # Get data
            kspace_masked = batch['kspace_masked'].to(device)
            kspace_full = batch['kspace_full'].to(device)
            mask = batch['mask'].to(device)
            target = batch['target'].to(device)
            
            # DEBUG: Check shapes before model call
            if batch_idx == 0:
                print(f"Before model call:")
                print(f"  kspace_masked: {kspace_masked.shape}")
                print(f"  mask: {mask.shape}")
                print(f"  target: {target.shape}")
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(kspace_masked, mask, kspace_full)
            
            # Compute loss
            loss = criterion(output['output'], target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()  # FIXED: Accumulate total loss
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{total_loss/num_batches:.4f}'  # Show running average
            })
            
        except Exception as e:
            print(f"\n❌ ERROR in batch {batch_idx}: {e}")
            print(f"Error type: {type(e).__name__}")
            
            # Print debug info
            if 'batch' in locals():
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")
            
            raise e  # Re-raise to stop training
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, val_loader, criterion, device):  # FIXED: criterion, not optimizer
    """Validate the model."""
    model.eval()
    total_loss = 0.0  # FIXED: Initialize total_loss
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        
        for batch in pbar:
            try:
                # Get data
                kspace_masked = batch['kspace_masked'].to(device)
                kspace_full = batch['kspace_full'].to(device)
                mask = batch['mask'].to(device)
                target = batch['target'].to(device)
                
                # Forward pass
                output = model(kspace_masked, mask, kspace_full)
                
                # Compute loss
                loss = criterion(output['output'], target)
                
                total_loss += loss.item()  # FIXED: Accumulate total loss
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg': f'{total_loss/num_batches:.4f}'  # Show running average
                })
                
            except Exception as e:
                print(f"\n❌ ERROR in validation: {e}")
                raise e
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, output_dir, filename):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'timestamp': time.time()
    }
    
    filepath = os.path.join(output_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Simple U-Net")
    parser.add_argument('--train-data', type=str, default='/scratch/vchaurasia/fastmri_data/train')
    parser.add_argument('--val-data', type=str, default='/scratch/vchaurasia/fastmri_data/val')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', type=str, default='/scratch/vchaurasia/simple_unet_models')
    parser.add_argument('--num-workers', type=int, default=2)  # Reduced for stability
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_simple_dataloaders(
        args.train_data, args.val_data, args.batch_size, args.num_workers
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = SimpleMRIUNet(
        n_channels=2,
        n_classes=1,
        dc_weight=1.0,
        num_dc_iterations=5
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    criterion = SimpleLoss()
    
    # Training loop
    best_val_loss = float('inf')
    training_log = []
    
    print("Starting training...")
    print("=" * 50)
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)
        
        try:
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss = validate(model, val_loader, criterion, device)
            
            epoch_time = time.time() - epoch_start_time
            
            # Log results
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch_time': epoch_time,
                'lr': optimizer.param_groups[0]['lr']
            }
            training_log.append(log_entry)
            
            # Print results
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss:   {val_loss:.6f}")
            print(f"Time:       {epoch_time:.2f}s")
            print(f"LR:         {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoints
            # Always save latest
            save_checkpoint(
                model, optimizer, epoch + 1, train_loss, val_loss,
                args.output_dir, 'latest_checkpoint.pth'
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch + 1, train_loss, val_loss,
                    args.output_dir, 'best_model.pth'
                )
                print(f"🎉 New best model! Val loss: {val_loss:.6f}")
            
            # Save periodic checkpoints
            if (epoch + 1) % 5 == 0:
                save_checkpoint(
                    model, optimizer, epoch + 1, train_loss, val_loss,
                    os.path.join(args.output_dir, 'checkpoints'),
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
            
            # Save training log
            log_file = os.path.join(args.output_dir, 'logs', 'training_log.txt')
            with open(log_file, 'w') as f:
                f.write("Epoch,Train_Loss,Val_Loss,Time,LR\n")
                for entry in training_log:
                    f.write(f"{entry['epoch']},{entry['train_loss']:.6f},"
                           f"{entry['val_loss']:.6f},{entry['epoch_time']:.2f},"
                           f"{entry['lr']:.2e}\n")
            
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
            save_checkpoint(
                model, optimizer, epoch + 1, 0.0, 0.0,
                args.output_dir, 'interrupted_checkpoint.pth'
            )
            break
            
        except Exception as e:
            print(f"\n❌ Training failed at epoch {epoch+1}: {e}")
            save_checkpoint(
                model, optimizer, epoch + 1, 0.0, 0.0,
                args.output_dir, 'error_checkpoint.pth'
            )
            raise e
    
    print("\n" + "=" * 50)
    print("🎯 Training Summary:")
    print(f"📈 Best validation loss: {best_val_loss:.6f}")
    print(f"📁 Models saved to: {args.output_dir}")
    print(f"📊 Training log: {os.path.join(args.output_dir, 'logs', 'training_log.txt')}")
    
    # Show final model info
    if training_log:
        final_epoch = training_log[-1]
        print(f"📋 Final epoch ({final_epoch['epoch']}):")
        print(f"   Train loss: {final_epoch['train_loss']:.6f}")
        print(f"   Val loss:   {final_epoch['val_loss']:.6f}")


if __name__ == "__main__":
    main()