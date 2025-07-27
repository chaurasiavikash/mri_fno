#!/bin/bash
# File: scripts/setup_simple_unet.sh
# Quick setup script to prepare for Simple U-Net training

echo "=== SETTING UP SIMPLE U-NET (SCRATCH STORAGE) ==="

# Base paths
SCRATCH_BASE="/scratch/vchaurasia"
PROJECT_DIR="/home/vchaurasia/projects/mri_fno"

cd "$PROJECT_DIR"

echo "1. Creating scratch directories..."
mkdir -p "$SCRATCH_BASE/simple_unet_models"
mkdir -p "$SCRATCH_BASE/simple_unet_models/logs"
mkdir -p "$SCRATCH_BASE/simple_unet_models/checkpoints"

echo "2. Checking data availability..."
if [ -d "$SCRATCH_BASE/fastmri_data" ]; then
    echo "✅ FastMRI data found"
    echo "   Train files: $(ls $SCRATCH_BASE/fastmri_data/train/*.h5 2>/dev/null | wc -l)"
    echo "   Val files: $(ls $SCRATCH_BASE/fastmri_data/val/*.h5 2>/dev/null | wc -l)"
else
    echo "❌ FastMRI data not found at $SCRATCH_BASE/fastmri_data"
    echo "   Please ensure your data is in the correct location"
fi

echo "3. Checking disk space..."
echo "Available space in /scratch/vchaurasia:"
df -h /scratch/vchaurasia

echo "4. Copying simple U-Net files..."
# Copy the artifacts to the right locations
if [ ! -f "src/simple_unet.py" ]; then
    echo "⚠️  simple_unet.py not found in src/"
    echo "   Please copy it from the artifacts to src/simple_unet.py"
else
    echo "✅ simple_unet.py found"
fi

if [ ! -f "scripts/train_simple_unet.py" ]; then
    echo "⚠️  train_simple_unet.py not found in scripts/"
    echo "   Please copy it from the artifacts to scripts/train_simple_unet.py"
else
    echo "✅ train_simple_unet.py found"
fi

if [ ! -f "jobs/train_simple_unet.sh" ]; then
    echo "⚠️  train_simple_unet.sh not found in jobs/"
    echo "   Please copy it from the artifacts to jobs/train_simple_unet.sh"
else
    echo "✅ train_simple_unet.sh found"
fi

echo "5. Checking environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate FastReg
echo "   Python: $(which python)"
echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
echo "   CUDA: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Not available')"

echo ""
echo "=== SETUP COMPLETE ==="
echo ""
echo "To start training:"
echo "  sbatch jobs/train_simple_unet.sh"
echo ""
echo "To monitor progress:"
echo "  tail -f /scratch/vchaurasia/simple_unet_models/logs/training_*.out"
echo ""
echo "Storage locations:"
echo "  Models: /scratch/vchaurasia/simple_unet_models/"
echo "  Logs: /scratch/vchaurasia/simple_unet_models/logs/"