#!/bin/bash
#SBATCH --job-name=mri_unet_inference
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/vchaurasia/simple_unet_models/logs/training_%j.out
#SBATCH --error=/scratch/vchaurasia/simple_unet_models/logs/training_%j.err

echo "========================================"
echo "Simple U-Net Training (Scratch Storage)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo ""

# Setup paths - EVERYTHING in scratch
SCRATCH_BASE="/scratch/vchaurasia"
PROJECT_DIR="/home/vchaurasia/projects/mri_fno"
OUTPUT_DIR="$SCRATCH_BASE/simple_unet_models"
LOG_DIR="$OUTPUT_DIR/logs"
MODEL_DIR="$OUTPUT_DIR/checkpoints"

# Create all directories in scratch
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$MODEL_DIR"

echo "Storage locations:"
echo "  Output: $OUTPUT_DIR"
echo "  Logs: $LOG_DIR"
echo "  Models: $MODEL_DIR"
echo "  Data: $SCRATCH_BASE/fastmri_data"
echo ""

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate FastReg

# Go to project directory (code only, no data storage)
cd "$PROJECT_DIR"

echo "Environment:"
echo "  Python: $(which python)"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Check data availability
echo "Checking data availability:"
ls -la "$SCRATCH_BASE/fastmri_data/"
echo ""

# Copy the new model file to src (if not already there)
if [ ! -f "src/simple_unet.py" ]; then
    echo "Copying simple_unet.py to src/"
    cp scripts/simple_unet.py src/ 2>/dev/null || echo "simple_unet.py already in place"
fi

echo "Starting Simple U-Net training..."
echo "Command: python scripts/train_simple_unet.py"
echo "  --train-data $SCRATCH_BASE/fastmri_data/train"
echo "  --val-data $SCRATCH_BASE/fastmri_data/val" 
echo "  --epochs 30"
echo "  --batch-size 1"
echo "  --lr 1e-3"
echo "  --output-dir $OUTPUT_DIR"
echo ""

# Run training with all output to scratch
python scripts/train_simple_unet.py \
    --train-data "$SCRATCH_BASE/fastmri_data/train" \
    --val-data "$SCRATCH_BASE/fastmri_data/val" \
    --epochs 30 \
    --batch-size 1 \
    --lr 1e-3 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_DIR/training_detailed.log"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Simple U-Net training completed successfully!"
    echo ""
    echo "Results summary:"
    echo "  📁 Models saved to: $MODEL_DIR"
    echo "  📊 Logs saved to: $LOG_DIR"
    
    if [ -f "$OUTPUT_DIR/best_model.pth" ]; then
        echo "  🏆 Best model: $OUTPUT_DIR/best_model.pth"
        echo "  📏 Model size: $(du -h $OUTPUT_DIR/best_model.pth | cut -f1)"
    fi
    
    echo ""
    echo "  📈 Available checkpoints:"
    ls -la "$OUTPUT_DIR"/*.pth 2>/dev/null | head -5
    
    echo ""
    echo "  💾 Disk usage in scratch:"
    du -sh "$OUTPUT_DIR"
    
else
    echo ""
    echo "❌ Simple U-Net training failed!"
    echo "Check the error log for details: $LOG_DIR/training_*.err"
    exit 1
fi

echo ""
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"

# Final disk usage check
echo ""
echo "Final scratch disk usage:"
df -h /scratch/vchaurasia