# File: tests/test_train.py

import pytest
import torch
import torch.nn as nn
import tempfile
import os
import sys
import yaml
import shutil
from pathlib import Path
import h5py
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import MRITrainer
from utils import get_device


class TestMRITrainer:
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with mock .h5 files."""
        temp_dir = tempfile.mkdtemp()
        
        # Create train, val, test directories
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(temp_dir, split)
            os.makedirs(split_dir)
            
            # Create a few mock .h5 files
            for i in range(2):
                file_path = os.path.join(split_dir, f"file_{i:03d}.h5")
                
                with h5py.File(file_path, 'w') as f:
                    # Create synthetic k-space data
                    num_slices, num_coils, height, width = 3, 4, 32, 32
                    kspace_data = (
                        np.random.randn(num_slices, num_coils, height, width).astype(np.float32) +
                        1j * np.random.randn(num_slices, num_coils, height, width).astype(np.float32)
                    )
                    f.create_dataset('kspace', data=kspace_data)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_config(self, temp_data_dir):
        """Create test configuration."""
        return {
            'data': {
                'train_path': os.path.join(temp_data_dir, 'train'),
                'val_path': os.path.join(temp_data_dir, 'val'),
                'test_path': os.path.join(temp_data_dir, 'test'),
                'acceleration': 4,
                'center_fraction': 0.08,
                'num_workers': 0,  # Use 0 workers for testing
                'batch_size': 1,
                'mask_type': 'random'
            },
            'model': {
                'name': 'disco_neural_operator',
                'in_channels': 2,
                'out_channels': 2,
                'hidden_channels': 8,
                'num_layers': 2,
                'modes': 4,
                'width': 8,
                'dropout': 0.0,
                'use_residual': False,
                'activation': 'relu'
            },
            'training': {
                'epochs': 2,
                'learning_rate': 1e-3,
                'weight_decay': 1e-6,
                'scheduler': 'cosine',
                'warmup_epochs': 0,
                'gradient_clip': 1.0,
                'mixed_precision': False,
                'save_every': 1,
                'validate_every': 1
            },
            'loss': {
                'type': 'combined',
                'l1_weight': 1.0,
                'ssim_weight': 0.1,
                'perceptual_weight': 0.0,
                'data_consistency_weight': 0.5
            },
            'optimizer': {
                'name': 'adamw',
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'amsgrad': False
            },
            'evaluation': {
                'metrics': ['psnr', 'ssim', 'nmse'],
                'save_images': True,
                'num_test_images': 5
            },
            'system': {
                'device': 'cpu',
                'seed': 42,
                'num_threads': 1,
                'log_level': 'INFO'
            },
            'logging': {
                'use_wandb': False,
                'project_name': 'test_mri_reconstruction',
                'log_dir': 'test_logs',
                'save_model_dir': 'test_models',
                'results_dir': 'test_results'
            }
        }
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_trainer_initialization(self, test_config, temp_output_dir):
        """Test trainer initialization."""
        # Update config with temp directory
        test_config['logging']['log_dir'] = os.path.join(temp_output_dir, 'logs')
        test_config['logging']['save_model_dir'] = os.path.join(temp_output_dir, 'models')
        test_config['logging']['results_dir'] = os.path.join(temp_output_dir, 'results')
        
        device = get_device('cpu')
        trainer = MRITrainer(test_config, device)
        
        # Check that trainer is properly initialized
        assert trainer.config == test_config
        assert trainer.device == device
        assert trainer.epoch == 0
        assert trainer.global_step == 0
        assert trainer.best_val_loss == float('inf')
        
        # Check that components are initialized
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
        assert trainer.train_loader is not None
        assert trainer.val_loader is not None
        assert trainer.test_loader is not None
    
    def test_data_loader_setup(self, test_config, temp_output_dir):
        """Test data loader setup."""
        test_config['logging']['log_dir'] = os.path.join(temp_output_dir, 'logs')
        test_config['logging']['save_model_dir'] = os.path.join(temp_output_dir, 'models')
        test_config['logging']['results_dir'] = os.path.join(temp_output_dir, 'results')
        
        device = get_device('cpu')
        trainer = MRITrainer(test_config, device)
        
        # Check data loaders
        assert len(trainer.train_loader) > 0
        assert len(trainer.val_loader) > 0
        assert len(trainer.test_loader) > 0
        
        # Test that we can iterate through data loaders
        train_batch = next(iter(trainer.train_loader))
        val_batch = next(iter(trainer.val_loader))
        
        # Check batch structure
        required_keys = ['kspace_full', 'kspace_masked', 'mask', 'target', 'image_masked']
        for key in required_keys:
            assert key in train_batch
            assert key in val_batch
    
    def test_model_setup(self, test_config, temp_output_dir):
        """Test model setup."""
        test_config['logging']['log_dir'] = os.path.join(temp_output_dir, 'logs')
        test_config['logging']['save_model_dir'] = os.path.join(temp_output_dir, 'models')
        test_config['logging']['results_dir'] = os.path.join(temp_output_dir, 'results')
        
        device = get_device('cpu')
        trainer = MRITrainer(test_config, device)
        
        # Check model properties
        assert trainer.model.neural_operator.in_channels == 2
        assert trainer.model.neural_operator.out_channels == 2
        assert trainer.model.use_data_consistency == True
        
        # Check that model is on correct device
        for param in trainer.model.parameters():
            assert param.device == device
    
    def test_optimizer_setup(self, test_config, temp_output_dir):
        """Test optimizer setup."""
        test_config['logging']['log_dir'] = os.path.join(temp_output_dir, 'logs')
        test_config['logging']['save_model_dir'] = os.path.join(temp_output_dir, 'models')
        test_config['logging']['results_dir'] = os.path.join(temp_output_dir, 'results')
        
        device = get_device('cpu')
        trainer = MRITrainer(test_config, device)
        
        # Check optimizer properties
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert trainer.optimizer.param_groups[0]['lr'] == test_config['training']['learning_rate']
        assert trainer.optimizer.param_groups[0]['weight_decay'] == test_config['training']['weight_decay']
    
    def test_scheduler_setup(self, test_config, temp_output_dir):
        """Test scheduler setup."""
        test_config['logging']['log_dir'] = os.path.join(temp_output_dir, 'logs')
        test_config['logging']['save_model_dir'] = os.path.join(temp_output_dir, 'models')
        test_config['logging']['results_dir'] = os.path.join(temp_output_dir, 'results')
        
        device = get_device('cpu')
        trainer = MRITrainer(test_config, device)
        
        # Check scheduler (cosine annealing)
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    
    def test_loss_setup(self, test_config, temp_output_dir):
        """Test loss function setup."""
        test_config['logging']['log_dir'] = os.path.join(temp_output_dir, 'logs')
        test_config['logging']['save_model_dir'] = os.path.join(temp_output_dir, 'models')
        test_config['logging']['results_dir'] = os.path.join(temp_output_dir, 'results')
        
        device = get_device('cpu')
        trainer = MRITrainer(test_config, device)
        
        # Check loss function properties
        assert trainer.criterion.l1_weight == test_config['loss']['l1_weight']
        assert trainer.criterion.ssim_weight == test_config['loss']['ssim_weight']
        assert trainer.criterion.data_consistency_weight == test_config['loss']['data_consistency_weight']
    
    def test_train_epoch(self, test_config, temp_output_dir):
        """Test training for one epoch."""
        test_config['logging']['log_dir'] = os.path.join(temp_output_dir, 'logs')
        test_config['logging']['save_model_dir'] = os.path.join(temp_output_dir, 'models')
        test_config['logging']['results_dir'] = os.path.join(temp_output_dir, 'results')
        test_config['training']['gradient_clip'] = 0  # Disable for testing
        
        device = get_device('cpu')
        trainer = MRITrainer(test_config, device)
        
        # Train for one epoch
        initial_step = trainer.global_step
        metrics = trainer.train_epoch()
        
        # Check that metrics are returned
        assert 'train_loss' in metrics
        assert isinstance(metrics['train_loss'], float)
        assert metrics['train_loss'] >= 0
        
        # Check that global step was updated
        assert trainer.global_step > initial_step
    
    def test_validation(self, test_config, temp_output_dir):
        """Test validation."""
        test_config['logging']['log_dir'] = os.path.join(temp_output_dir, 'logs')
        test_config['logging']['save_model_dir'] = os.path.join(temp_output_dir, 'models')
        test_config['logging']['results_dir'] = os.path.join(temp_output_dir, 'results')
        
        device = get_device('cpu')
        trainer = MRITrainer(test_config, device)
        
        # Run validation
        metrics = trainer.validate()
        
        # Check that metrics are returned
        assert 'val_loss' in metrics
        assert isinstance(metrics['val_loss'], float)
        assert metrics['val_loss'] >= 0
    
    def test_checkpoint_save_load(self, test_config, temp_output_dir):
        """Test checkpoint saving and loading."""
        test_config['logging']['log_dir'] = os.path.join(temp_output_dir, 'logs')
        test_config['logging']['save_model_dir'] = os.path.join(temp_output_dir, 'models')
        test_config['logging']['results_dir'] = os.path.join(temp_output_dir, 'results')
        
        device = get_device('cpu')
        trainer = MRITrainer(test_config, device)
        
        # Set some training state
        trainer.epoch = 5
        trainer.global_step = 100
        trainer.best_val_loss = 0.5
        
        # Save checkpoint
        checkpoint_name = "test_checkpoint.pth"
        trainer.save_checkpoint(checkpoint_name)
        
        # Check that checkpoint file exists
        checkpoint_path = trainer.model_dir / checkpoint_name
        assert checkpoint_path.exists()
        
        # Create new trainer and load checkpoint
        trainer2 = MRITrainer(test_config, device)
        trainer2.load_checkpoint(checkpoint_name)
        
        # Check that state was restored
        assert trainer2.epoch == 5
        assert trainer2.global_step == 100
        assert trainer2.best_val_loss == 0.5
    
    def test_full_training_loop(self, test_config, temp_output_dir):
        """Test complete training loop."""
        test_config['logging']['log_dir'] = os.path.join(temp_output_dir, 'logs')
        test_config['logging']['save_model_dir'] = os.path.join(temp_output_dir, 'models')
        test_config['logging']['results_dir'] = os.path.join(temp_output_dir, 'results')
        test_config['training']['epochs'] = 2  # Train for 2 epochs
        test_config['training']['gradient_clip'] = 0  # Disable for testing
        
        device = get_device('cpu')
        trainer = MRITrainer(test_config, device)
        
        # Run training
        initial_epoch = trainer.epoch
        trainer.train(num_epochs=2)
        
        # Check that training progressed
        assert trainer.epoch >= initial_epoch + 1
        
        # Check that checkpoints were saved
        assert any(trainer.model_dir.glob("checkpoint_epoch_*.pth"))
    
    def test_mixed_precision_training(self, test_config, temp_output_dir):
        """Test mixed precision training (if CUDA available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision testing")
        test_config['logging']['log_dir'] = os.path.join(temp_output_dir, 'logs')
        test_config['logging']['save_model_dir'] = os.path.join(temp_output_dir, 'models')
        test_config['logging']['results_dir'] = os.path.join(temp_output_dir, 'results')
        test_config['training']['mixed_precision'] = True
        test_config['system']['device'] = 'cuda'
        device = get_device('cuda')
        trainer = MRITrainer(test_config, device)
        # Check mixed precision support for complex models
        if any(p.is_complex() for p in trainer.model.parameters()):
            assert trainer.scaler is None
            return  # Mixed precision is not supported for complex models
        else:
            assert trainer.scaler is not None
            # Train for one epoch
            metrics = trainer.train_epoch()
            assert 'train_loss' in metrics
    
    def test_different_schedulers(self, test_config, temp_output_dir):
        """Test different learning rate schedulers."""
        test_config['logging']['log_dir'] = os.path.join(temp_output_dir, 'logs')
        test_config['logging']['save_model_dir'] = os.path.join(temp_output_dir, 'models')
        test_config['logging']['results_dir'] = os.path.join(temp_output_dir, 'results')
        
        schedulers = ['cosine', 'step', 'plateau', None]
        
        for scheduler_name in schedulers:
            test_config['training']['scheduler'] = scheduler_name
            device = get_device('cpu')
            trainer = MRITrainer(test_config, device)
            
            if scheduler_name is None:
                assert trainer.scheduler is None
            else:
                assert trainer.scheduler is not None
    
    def test_warmup_scheduler(self, test_config, temp_output_dir):
        """Test warmup scheduler."""
        test_config['logging']['log_dir'] = os.path.join(temp_output_dir, 'logs')
        test_config['logging']['save_model_dir'] = os.path.join(temp_output_dir, 'models')
        test_config['logging']['results_dir'] = os.path.join(temp_output_dir, 'results')
        test_config['training']['warmup_epochs'] = 1
        
        device = get_device('cpu')
        trainer = MRITrainer(test_config, device)
        
        # Check that warmup scheduler is initialized
        assert trainer.warmup_scheduler is not None
        
        # Train for one epoch (within warmup period)
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        trainer.train_epoch()
        
        # Learning rate should change during warmup
        # (Note: this is a basic check, actual behavior depends on implementation)
        #self.epoch += 1


if __name__ == "__main__":
    pytest.main([__file__])