#!/bin/bash
#SBATCH --job-name=mri_disco
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

# Load modules (adjust if needed)
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

# Print data info  
echo "=== Dataset Info ==="
echo "Train files: $(find /scratch/vchaurasia/fastmri_data/train -name "*.h5" | wc -l)"
echo "Val files: $(find /scratch/vchaurasia/fastmri_data/val -name "*.h5" | wc -l)"
echo "Test files: $(find /scratch/vchaurasia/fastmri_data/test -name "*.h5" | wc -l)"

# Run training
echo "=== Starting Training ==="
python scripts/run_training.py \
    --config configs/full_training_config.yaml \
    --device cuda \
    --epochs 100

echo "=== Training Completed ==="
echo "End time: $(date)"