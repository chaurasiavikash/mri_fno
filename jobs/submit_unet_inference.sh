#!/bin/bash
#SBATCH --job-name=unet_simple
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/vchaurasia/organized_models/logs/unet_simple_%j.out
#SBATCH --error=/scratch/vchaurasia/organized_models/logs/unet_simple_%j.err

# File: jobs/submit_simple_unet.sh
# SLURM job script for simple UNet inference

echo "========================================"
echo "Simple UNet Inference Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Skip problematic modules, use conda directly
echo "Using conda environment directly..."

# Activate miniconda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate FastReg

# Verify environment
echo "Python path: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Go to project directory
cd /home/vchaurasia/projects/mri_fno

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Print GPU info
echo "=== GPU Info ==="
nvidia-smi
echo ""

# Model and output paths
ORGANIZED_DIR="/scratch/vchaurasia/organized_models"
MODEL_PATH="$ORGANIZED_DIR/unet_epoch20.pth"
OUTPUT_DIR="$ORGANIZED_DIR/inference_results/unet_simple"

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Test data: /scratch/vchaurasia/fastmri_data/test"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: UNet model not found at $MODEL_PATH"
    echo "Available models:"
    ls -la "$ORGANIZED_DIR/"*.pth 2>/dev/null || echo "No .pth files found"
    exit 1
fi

echo "✅ Model file found: $(ls -lh $MODEL_PATH)"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$ORGANIZED_DIR/logs"

echo "Starting simple UNet inference..."
echo "Command: python scripts/simple_unet_inference.py"
echo ""

# Run inference
python scripts/simple_unet_inference.py

# Check exit status
INFERENCE_EXIT_CODE=$?

if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Simple UNet inference completed successfully!"
    echo ""
    echo "Results summary:"
    echo "  📁 Output directory: $OUTPUT_DIR"
    
    if [ -f "$OUTPUT_DIR/summary_metrics.txt" ]; then
        echo "  📊 Summary metrics:"
        cat "$OUTPUT_DIR/summary_metrics.txt"
    fi
    
    echo ""
    echo "  📈 Generated files:"
    find "$OUTPUT_DIR" -type f -name "*.txt" -o -name "*.npz" | head -5
    
    echo ""
    echo "🎯 Ready for comparison with DISCO results!"
    
else
    echo ""
    echo "❌ Simple UNet inference failed with exit code: $INFERENCE_EXIT_CODE"
    echo "Check the error log for details."
    exit 1
fi

echo ""
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"