#!/bin/bash
#SBATCH --job-name=disco_inference
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/vchaurasia/organized_models/logs/disco_inference_%j.out
#SBATCH --error=/scratch/vchaurasia/organized_models/logs/disco_inference_%j.err
# File: jobs/submit_disco_inference.sh
# SLURM job script for DISCO model inference
echo "========================================"
echo "DISCO Neural Operator Inference Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo ""
# Load modules (match your working setup)
module purge
module load 2023r1
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
# Activate miniconda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate FastReg
# Go to project directory
cd /home/vchaurasia/projects/mri_fno
# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
# Model and config paths
ORGANIZED_DIR="/scratch/vchaurasia/organized_models"
MODEL_PATH="$ORGANIZED_DIR/disco_epoch20.pth"
CONFIG_PATH="configs/disco_inference_scratch.yaml"
OUTPUT_DIR="$ORGANIZED_DIR/inference_results/disco_epoch20"
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Config: $CONFIG_PATH" 
echo "  Output: $OUTPUT_DIR"
echo ""
# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: DISCO model not found at $MODEL_PATH"
    echo "Available models in disco directory:"
    ls -la "$ORGANIZED_DIR/disco/" || echo "No disco directory found"
    echo ""
    echo "Available symlinks:"
    ls -la "$ORGANIZED_DIR/"*.pth || echo "No symlinks found"
    exit 1
fi
echo "✅ Model file found: $(ls -lh $MODEL_PATH)"
echo ""
# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$ORGANIZED_DIR/logs"
echo "Starting DISCO inference..."
echo "Command: python src/infer.py --config $CONFIG_PATH --model $MODEL_PATH --output $OUTPUT_DIR --max-samples 100 --save-all --visualize 0 1 2 3 4"
echo ""
# Run inference
python src/infer.py \
    --config "$CONFIG_PATH" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_DIR" \
    --device cuda \
    --max-samples 100 \
    --save-all \
    --visualize 0 1 2 3 4
# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ DISCO inference completed successfully!"
    echo ""
    echo "Results summary:"
    echo "  📁 Output directory: $OUTPUT_DIR"
    
    if [ -f "$OUTPUT_DIR/summary_metrics.txt" ]; then
        echo "  📊 Summary metrics:"
        head -10 "$OUTPUT_DIR/summary_metrics.txt"
    fi
    
    echo ""
    echo "  📈 Generated files:"
    find "$OUTPUT_DIR" -type f -name "*.png" -o -name "*.txt" -o -name "*.npz" | head -10
    
else
    echo ""
    echo "❌ DISCO inference failed!"
    echo "Check the error log for details."
    exit 1
fi
echo ""
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"