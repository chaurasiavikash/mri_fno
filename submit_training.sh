#!/bin/bash
#SBATCH --job-name=mri_disco_fixed
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/vchaurasia/disco_training_%j.out
#SBATCH --error=/scratch/vchaurasia/disco_training_%j.err

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

echo "=== Starting DISCO Training with Fixed Scaling ==="
echo "Training from scratch (no resume)"

# Start fresh DISCO training with fixed scaling
python scripts/run_training.py \
    --config configs/full_training_config.yaml \
    --device cuda \
    --epochs 40 \
    --experiment-name disco_fixed_scaling \
    --output-dir /scratch/vchaurasia/disco_models

echo "=== Training Completed ==="
echo "End time: $(date)"

# Show final results
echo "=== Final Model Location ==="
ls -la /scratch/vchaurasia/disco_models/disco_fixed_scaling/models/

echo "=== Training Log Summary ==="
tail -20 /scratch/vchaurasia/disco_models/disco_fixed_scaling/logs/training.log || echo "No training log found"