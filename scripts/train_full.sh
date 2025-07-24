#!/bin/bash
# File: scripts/train_full.sh
# Full training script that runs in background

echo "=== MRI DISCO Training Started ==="
echo "Start time: $(date)"
echo "PID: $$"

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate FastReg

# Navigate to project
cd /home/vchaurasia/projects/mri_fno

# Create output directories
mkdir -p outputs/{logs,models,results}

# Print system info
echo "=== System Info ==="
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Print data info
echo "=== Dataset Info ==="
echo "Train files: $(find /scratch/vchaurasia/fastmri_data/train -name "*.h5" | wc -l)"
echo "Val files: $(find /scratch/vchaurasia/fastmri_data/val -name "*.h5" | wc -l)"
echo "Test files: $(find /scratch/vchaurasia/fastmri_data/test -name "*.h5" | wc -l)"

# Run training with logging
echo "=== Starting Training ==="
python scripts/run_training.py \
    --config configs/full_training_config.yaml \
    --device cuda \
    --epochs 100 2>&1 | tee outputs/logs/training_$(date +%Y%m%d_%H%M%S).log

echo "=== Training Completed ==="
echo "End time: $(date)"