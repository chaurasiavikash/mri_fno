#!/bin/bash
#SBATCH --job-name=unet_baseline
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=unet_training_%j.out
#SBATCH --error=unet_training_%j.err

echo "=== U-Net Baseline Training Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# Load modules (skip the problematic ones - conda handles everything)
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate FastReg

# Go to project
cd /home/vchaurasia/projects/mri_fno

# Print GPU info
echo "=== GPU Info ==="
nvidia-smi

echo "=== Starting U-Net Baseline Training ==="
python scripts/train_unet_baseline.py \
    --config configs/unet_baseline_config.yaml \
    --device cuda \
    --epochs 100

echo "=== U-Net Training Completed ==="
echo "End time: $(date)"
