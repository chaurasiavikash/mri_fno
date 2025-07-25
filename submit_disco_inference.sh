#!/bin/bash
#SBATCH --job-name=disco_inference
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=inference_%j.out
#SBATCH --error=inference_%j.err

echo "=== DISCO Inference Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate FastReg

# Go to project
cd /home/vchaurasia/projects/mri_fno

# Print GPU info
echo "=== GPU Info ==="
nvidia-smi

echo "=== Starting DISCO Inference ==="
python scripts/run_inference.py \
    --model /scratch/vchaurasia/mri_models/disco/models/best_model.pth \
    --config configs/single_test.yaml \
    --max-samples 10 \
    --output /scratch/vchaurasia/mri_models/disco/inference_results \
    --device cuda \
    --save-all

echo "=== DISCO Inference Completed ==="
echo "End time: $(date)"
