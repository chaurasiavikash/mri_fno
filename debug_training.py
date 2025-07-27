# File: debug_training.py
"""
Complete debug script for training pipeline.
Tests all components before submitting the job.
"""

import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from simple_unet import SimpleMRIUNet
from data_loader import FastMRIDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


class SimpleLoss(nn.Module):
    """Simple loss function - just L1 loss."""
    
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, prediction, target):
        return self.l1_loss(prediction, target)


def test_data_loading():
    """Test data loading pipeline."""
    print("\n" + "="*50)
    print("🔍 TESTING DATA LOADING")
    print("="*50)
    
    try:
        # Create dataset
        dataset = FastMRIDataset(
            data_path='/scratch/vchaurasia/fastmri_data/train',
            acceleration=4,
            center_fraction=0.08,
            transform=None,
            use_seed=True,
            seed=42
        )
        
        print(f"✅ Dataset created: {len(dataset)} samples")
        
        # Test single sample
        sample = dataset[0]
        print(f"✅ Single sample loaded")
        
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")
            else:
                print(f"   {key}: {type(value)}")
        
        # Test data loader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Use 0 for debugging
            pin_memory=False
        )
        
        print(f"✅ DataLoader created: {len(dataloader)} batches")
        
        # Test first batch
        batch = next(iter(dataloader))
        print(f"✅ First batch loaded")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")
        
        # Verify expected shapes
        expected_shapes = {
            'kspace_full': (1, 15, 2, 640, 368),
            'kspace_masked': (1, 15, 2, 640, 368),
            'mask': (1, 640, 368),
            'target': (1, 640, 368)
        }
        
        all_correct = True
        for key, expected_shape in expected_shapes.items():
            if key in batch:
                actual_shape = tuple(batch[key].shape)
                if actual_shape == expected_shape:
                    print(f"   ✅ {key}: shape correct")
                else:
                    print(f"   ❌ {key}: expected {expected_shape}, got {actual_shape}")
                    all_correct = False
        
        if all_correct:
            print("✅ ALL DATA SHAPES CORRECT!")
            return batch
        else:
            print("❌ DATA SHAPE ERRORS FOUND!")
            return None
            
    except Exception as e:
        print(f"❌ DATA LOADING FAILED: {e}")
        return None


def test_model_creation():
    """Test model creation and parameter counting."""
    print("\n" + "="*50)
    print("🔍 TESTING MODEL CREATION")
    print("="*50)
    
    try:
        model = SimpleMRIUNet(
            n_channels=2,
            n_classes=1,  # Should be 1, not 2
            dc_weight=1.0,
            num_dc_iterations=5
        )
        
        print("✅ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Test model on CPU first
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"❌ MODEL CREATION FAILED: {e}")
        return None


def test_model_forward(model, batch):
    """Test model forward pass."""
    print("\n" + "="*50)
    print("🔍 TESTING MODEL FORWARD PASS")
    print("="*50)
    
    if model is None or batch is None:
        print("❌ Skipping - model or batch is None")
        return None
        
    try:
        # Move data to model device
        device = next(model.parameters()).device
        kspace_masked = batch['kspace_masked'].to(device)
        kspace_full = batch['kspace_full'].to(device)
        mask = batch['mask'].to(device)
        target = batch['target'].to(device)
        
        print(f"✅ Data moved to device: {device}")
        
        # Forward pass (WITHOUT no_grad for gradient testing)
        output = model(kspace_masked, mask, kspace_full)
        
        print(f"✅ Forward pass successful")
        print(f"   Output keys: {list(output.keys())}")
        
        if 'output' in output:
            print(f"   Output shape: {output['output'].shape}")
            print(f"   Target shape: {target.shape}")
            
            # Check if shapes match for loss computation
            if output['output'].shape == target.shape:
                print("✅ Output and target shapes match!")
                return output
            else:
                print(f"❌ Shape mismatch: output {output['output'].shape} vs target {target.shape}")
                return None
        else:
            print("❌ No 'output' key in model output")
            return None
            
    except Exception as e:
        print(f"❌ MODEL FORWARD FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_loss_computation(output, target):
    """Test loss computation."""
    print("\n" + "="*50)
    print("🔍 TESTING LOSS COMPUTATION")
    print("="*50)
    
    if output is None:
        print("❌ Skipping - output is None")
        return None
        
    try:
        criterion = SimpleLoss()
        
        loss = criterion(output['output'], target)
        
        print(f"✅ Loss computed: {loss.item():.6f}")
        print(f"   Loss type: {type(loss)}")
        print(f"   Loss requires_grad: {loss.requires_grad}")
        
        return loss
        
    except Exception as e:
        print(f"❌ LOSS COMPUTATION FAILED: {e}")
        return None


def test_backward_pass(model, loss):
    """Test backward pass and optimization."""
    print("\n" + "="*50)
    print("🔍 TESTING BACKWARD PASS")
    print("="*50)
    
    if loss is None:
        print("❌ Skipping - loss is None")
        return False
        
    try:
        # Check if model is in training mode
        print(f"Model training mode: {model.training}")
        
        # Check if loss requires grad
        print(f"Loss requires_grad: {loss.requires_grad}")
        print(f"Loss grad_fn: {loss.grad_fn}")
        
        if not loss.requires_grad:
            print("❌ Loss doesn't require gradients! Model was probably in eval mode.")
            return False
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        print("✅ Backward pass successful")
        
        # Check if gradients exist
        has_gradients = False
        grad_norms = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    has_gradients = True
                else:
                    print(f"   ⚠️ No gradient for {name}")
        
        if has_gradients:
            print(f"✅ Gradients computed: avg norm = {sum(grad_norms)/len(grad_norms):.6f}")
        else:
            print("❌ No gradients found!")
            return False
        
        # Test optimizer step
        optimizer.step()
        print("✅ Optimizer step successful")
        
        return True
        
    except Exception as e:
        print(f"❌ BACKWARD PASS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_compatibility():
    """Test GPU compatibility."""
    print("\n" + "="*50)
    print("🔍 TESTING GPU COMPATIBILITY")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available - will run on CPU")
        return torch.device('cpu')
    
    try:
        device = torch.device('cuda')
        
        # Test basic CUDA operations
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = torch.mm(x, y)
        
        print(f"✅ CUDA operations work")
        print(f"   Device: {device}")
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        return device
        
    except Exception as e:
        print(f"❌ GPU TEST FAILED: {e}")
        print("⚠️ Falling back to CPU")
        return torch.device('cpu')


def test_full_training_step():
    """Test complete training step."""
    print("\n" + "="*50)
    print("🔍 TESTING COMPLETE TRAINING STEP")
    print("="*50)
    
    # Test GPU
    device = test_gpu_compatibility()
    
    # Test data loading
    batch = test_data_loading()
    if batch is None:
        return False
    
    # Test model creation
    model = test_model_creation()
    if model is None:
        return False
    
    # Move model to device
    model = model.to(device)
    print(f"✅ Model moved to {device}")
    
    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    # Test forward pass
    model.train()  # Set to training mode for gradient computation
    output = test_model_forward(model, batch)
    if output is None:
        return False
    
    # Test loss computation
    loss = test_loss_computation(output, batch['target'])
    if loss is None:
        return False
    
    # Test backward pass
    success = test_backward_pass(model, loss)
    if not success:
        return False
    
    print("\n" + "="*50)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Training pipeline is ready!")
    print("="*50)
    
    return True


def main():
    """Run all debugging tests."""
    print("🚀 MRI RECONSTRUCTION TRAINING DEBUG")
    print("Testing all components before job submission...")
    
    success = test_full_training_step()
    
    if success:
        print("\n🎯 READY TO SUBMIT JOB!")
        print("All components working correctly.")
        print("\nTo submit training job:")
        print("sbatch scripts/submit_unet_training.sh")
    else:
        print("\n❌ ISSUES FOUND!")
        print("Fix the problems above before submitting.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())