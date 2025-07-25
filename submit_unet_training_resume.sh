#!/bin/bash
#SBATCH --job-name=unet_baseline
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err

echo "=== SLURM Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# Load modules
module purge
module load 2023r1
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate FastReg

# Go to project
cd /home/vchaurasia/projects/mri_fno

# Print GPU info
echo "=== GPU Info ==="
nvidia-smi

echo "=== Starting U-Net Training ==="

# Handle resume argument
if [ "$1" = "--resume" ] && [ -n "$2" ]; then
    echo "Resuming U-Net from checkpoint: $2"
    python scripts/train_unet_baseline.py \
        --config configs/unet_baseline_config.yaml \
        --device cuda \
        --epochs 100 \
        --resume "$2"
else
    echo "Starting fresh U-Net training"
    python scripts/train_unet_baseline.py \
        --config configs/unet_baseline_config.yaml \
        --device cuda \
        --epochs 100
fi

echo "=== U-Net Training Completed ==="
echo "End time: $(date)"
