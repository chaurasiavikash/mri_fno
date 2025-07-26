#!/bin/bash
#SBATCH --job-name=unet_comparable
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/vchaurasia/organized_models/logs/unet_comparable_%j.out
#SBATCH --error=/scratch/vchaurasia/organized_models/logs/unet_comparable_%j.err

echo "========================================"
echo "UNet Inference (Comparable to DISCO)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo ""

# Load modules (match DISCO setup)
module purge
module load 2023r1
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Activate same environment as DISCO
source ~/miniconda3/etc/profile.d/conda.sh
conda activate FastReg

# Go to project directory
cd /home/vchaurasia/projects/mri_fno

# Set environment variables (same as DISCO)
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Paths
MODEL_PATH="/scratch/vchaurasia/organized_models/unet_epoch20.pth"
OUTPUT_DIR="/scratch/vchaurasia/organized_models/inference_results/unet_comparable"

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Max samples: 100 (same as DISCO)"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: UNet model not found at $MODEL_PATH"
    exit 1
fi

echo "✅ Model file found: $(ls -lh $MODEL_PATH)"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "/scratch/vchaurasia/organized_models/logs"

echo "Starting UNet inference (comparable to DISCO)..."
echo "Command: python scripts/unet_inference_comparable.py"
echo ""

# Run inference
python scripts/unet_inference_comparable.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ UNet inference completed successfully!"
    echo ""
    echo "Results summary:"
    echo "  📁 Output directory: $OUTPUT_DIR"
    
    if [ -f "$OUTPUT_DIR/summary_metrics.txt" ]; then
        echo "  📊 Summary metrics:"
        head -15 "$OUTPUT_DIR/summary_metrics.txt"
    fi
    
    echo ""
    echo "  📈 Generated files:"
    ls -la "$OUTPUT_DIR/"
    
else
    echo ""
    echo "❌ UNet inference failed!"
    echo "Check the error log for details."
    exit 1
fi

echo ""
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"