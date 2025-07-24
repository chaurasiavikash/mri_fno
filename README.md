# MRI Reconstruction with Neural Operators

A complete, testable, and modular end-to-end Python pipeline for MRI reconstruction using neural operators, specifically implementing U-shaped DISCO (Deep Image Structure and Correlation Optimization) Neural Operators as described in recent CVPR research.

## 🚀 Features

- **State-of-the-art Neural Operators**: Implementation of DISCO Neural Operators with spectral convolutions
- **Complete Pipeline**: End-to-end training, inference, and evaluation
- **Modular Design**: Clean, maintainable code with comprehensive testing
- **Data Consistency**: Built-in data consistency layers for physics-informed reconstruction
- **Comprehensive Evaluation**: Statistical analysis, visualizations, and comparison tools
- **Production Ready**: Proper logging, checkpointing, and configuration management

## 📁 Project Structure

```
mri_reconstruction_project/
├── src/                          # Source code
│   ├── data_loader.py           # FastMRI data loading and preprocessing
│   ├── neural_operator.py       # Neural operator implementation
│   ├── model.py                 # Complete reconstruction model
│   ├── train.py                 # Training pipeline
│   ├── infer.py                 # Inference engine
│   ├── evaluate.py              # Comprehensive evaluation
│   └── utils.py                 # Utility functions
├── tests/                        # Test suite
│   ├── test_data_loader.py      # Data loading tests
│   ├── test_neural_operator.py  # Model architecture tests
│   ├── test_model.py            # Integration tests
│   ├── test_train.py            # Training tests
│   └── test_utils.py            # Utility tests
├── scripts/                      # Convenient run scripts
│   ├── run_training.py          # Enhanced training script
│   ├── run_inference.py         # Enhanced inference script
│   └── evaluate_model.py        # Enhanced evaluation script
├── configs/                      # Configuration files
│   └── config.yaml              # Main configuration
├── data/                         # Data directory
│   ├── raw/                     # Raw FastMRI data
│   └── processed/               # Processed data
├── outputs/                      # Output directory
│   ├── models/                  # Saved models
│   ├── results/                 # Results and visualizations
│   └── logs/                    # Training logs
└── README.md                    # This file
```

## 🛠️ Installation

### 1. Setup Project Structure

Run the setup script to create the project structure:

```bash
bash setup_project.sh
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Settings

Edit `configs/config.yaml` to match your data paths and training preferences:

```yaml
data:
  train_path: "data/raw/train"
  val_path: "data/raw/val"
  test_path: "data/raw/test"
  acceleration: 4
  center_fraction: 0.08

model:
  hidden_channels: 64
  num_layers: 4
  modes: 12

training:
  epochs: 100
  learning_rate: 1e-4
  batch_size: 1
```

## 🏃‍♂️ Quick Start

### Training

```bash
# Basic training
python scripts/run_training.py

# Custom configuration
python scripts/run_training.py --config configs/my_config.yaml

# Resume from checkpoint
python scripts/run_training.py --resume outputs/models/checkpoint_epoch_50.pth

# Debug mode (quick test)
python scripts/run_training.py --debug --epochs 2
```

### Inference

```bash
# Run inference with best model
python scripts/run_inference.py --model outputs/models/best_model.pth

# Auto-detect best model
python scripts/run_inference.py --model-dir outputs/models

# Limited samples with visualization
python scripts/run_inference.py \
  --model outputs/models/best_model.pth \
  --max-samples 10 \
  --visualize 0 1 2
```

### Evaluation

```bash
# Evaluate single model
python scripts/evaluate_model.py \
  --models outputs/models/best_model.pth \
  --names "DISCO Neural Operator"

# Compare multiple models
python scripts/evaluate_model.py \
  --models model1.pth model2.pth \
  --names "Model A" "Model B"

# Auto-discover and evaluate all models
python scripts/evaluate_model.py \
  --model-dir outputs \
  --auto-names
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_neural_operator.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Model Architecture

The DISCO Neural Operator consists of:

1. **Spectral Convolution Layers**: Learn in Fourier domain for global receptive field
2. **U-Net Architecture**: Encoder-decoder with skip connections
3. **Data Consistency Layers**: Enforce k-space data fidelity
4. **Multi-scale Processing**: Handle different image resolutions

### Key Components

- **SpectralConv2d**: 2D spectral convolution in Fourier domain
- **NeuralOperatorBlock**: Building block with spectral conv + feed-forward
- **DISCONeuralOperator**: Complete U-shaped neural operator
- **DataConsistencyLayer**: Physics-informed data consistency

## 📈 Performance Metrics

The evaluation provides comprehensive metrics:

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **NMSE**: Normalized Mean Squared Error
- **MAE**: Mean Absolute Error
- **Inference Time**: Computational efficiency

## 🔧 Configuration Options

### Model Configuration

```yaml
model:
  in_channels: 2              # Real + imaginary
  out_channels: 2
  hidden_channels: 64         # Base number of channels
  num_layers: 4               # Number of U-Net layers
  modes: 12                   # Fourier modes to learn
  dropout: 0.1
  use_residual: true
  activation: "gelu"
```

### Training Configuration

```yaml
training:
  epochs: 100
  learning_rate: 1e-4
  weight_decay: 1e-6
  scheduler: "cosine"
  gradient_clip: 1.0
  mixed_precision: true
  save_every: 10
  validate_every: 1
```

### Loss Configuration

```yaml
loss:
  l1_weight: 1.0
  ssim_weight: 0.1
  data_consistency_weight: 1.0
```

## 📋 Data Format

The pipeline expects FastMRI-format HDF5 files:

```
dataset.h5
├── kspace          # Multi-coil k-space data (slices, coils, height, width)
└── reconstruction  # Optional reference reconstructions
```

Data should be organized as:
```
data/
├── train/
│   ├── file1.h5
│   └── file2.h5
├── val/
│   └── file3.h5
└── test/
    └── file4.h5
```

## 🚨 Common Issues and Solutions

### CUDA Out of Memory
- Reduce batch size: `--batch-size 1`
- Use gradient checkpointing
- Enable mixed precision: set `mixed_precision: true`

### Slow Training
- Increase number of workers: `num_workers: 8`
- Use SSD for data storage
- Enable mixed precision training

### Poor Reconstruction Quality
- Increase model capacity: `hidden_channels: 128`
- Add more layers: `num_layers: 6`
- Tune data consistency weight

## 📚 Advanced Usage

### Custom Data Transforms

```python
from data_loader import FastMRIDataset

# Custom transform
def my_transform(sample):
    # Apply custom preprocessing
    return sample

dataset = FastMRIDataset(
    data_path="data/train",
    transform=my_transform
)
```

### Custom Loss Functions

```python
from model import ReconstructionLoss

class CustomLoss(ReconstructionLoss):
    def forward(self, prediction, target, **kwargs):
        # Custom loss implementation
        pass
```

### Model Ensembling

```python
# Load multiple models
models = [load_model(path) for path in model_paths]

# Ensemble prediction
predictions = [model(x) for model in models]
ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
```

## 📖 API Reference

### Core Classes

- `FastMRIDataset`: Multi-coil MRI data loader
- `DISCONeuralOperator`: Main neural operator model
- `MRIReconstructionModel`: Complete reconstruction pipeline
- `MRITrainer`: Training management
- `MRIInferenceEngine`: Inference and evaluation

### Key Functions

- `create_data_loaders()`: Setup train/val/test loaders
- `load_config()`: Load YAML configuration
- `set_seed()`: Set random seeds for reproducibility

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FastMRI dataset and evaluation metrics
- Neural operator research community
- PyTorch and scientific Python ecosystem

## 📞 Support

For questions and issues:

1. Check the documentation and examples
2. Review common issues section
3. Open an issue with detailed information
4. Include configuration and error messages

---

**Happy Reconstructing! 🧠⚡**