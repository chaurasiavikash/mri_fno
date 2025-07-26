#!/bin/bash
#SBATCH --job-name=unet_correct
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/vchaurasia/organized_models/logs/unet_correct_%j.out
#SBATCH --error=/scratch/vchaurasia/organized_models/logs/unet_correct_%j.err

echo "========================================"
echo "Correct UNet Inference (Using Training Setup)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Use same environment setup as training
source ~/miniconda3/etc/profile.d/conda.sh
conda activate FastReg

# Go to same project directory as training
cd /home/vchaurasia/projects/mri_fno

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

echo "Python path: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

echo "Starting correct UNet inference..."
python scripts/correct_unet_inference.py

INFERENCE_EXIT_CODE=$?

if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Correct UNet inference completed successfully!"
else
    echo ""
    echo "❌ UNet inference failed with exit code: $INFERENCE_EXIT_CODE"
    exit 1
fi

echo ""
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"