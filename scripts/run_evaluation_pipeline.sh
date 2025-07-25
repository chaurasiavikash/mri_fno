#!/bin/bash
# File: scripts/run_evaluation_pipeline.sh

# Comprehensive evaluation pipeline for DISCO vs U-Net comparison

echo "=========================================="
echo "MRI Reconstruction Model Evaluation"
echo "=========================================="

# Configuration
DISCO_MODEL="outputs/models/best_model.pth"  # Adjust path to your best DISCO model
UNET_MODEL="outputs/unet_models/best_model.pth"  # Adjust path to your best U-Net model
OUTPUT_DIR="outputs/comparison_study"
MAX_SAMPLES=100  # Adjust based on your test set size

echo "Models to evaluate:"
echo "  DISCO Neural Operator: $DISCO_MODEL"
echo "  U-Net Baseline: $UNET_MODEL"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# 1. Individual Model Evaluation
echo "Step 1: Individual Model Evaluation"
echo "-----------------------------------"

# Evaluate DISCO Neural Operator
echo "Evaluating DISCO Neural Operator..."
python scripts/run_inference.py \
    --model $DISCO_MODEL \
    --output $OUTPUT_DIR/disco_results \
    --max-samples $MAX_SAMPLES \
    --save-all \
    --visualize 0 1 2 3 4

# Evaluate U-Net Baseline
echo "Evaluating U-Net Baseline..."
python scripts/run_inference.py \
    --model $UNET_MODEL \
    --output $OUTPUT_DIR/unet_results \
    --max-samples $MAX_SAMPLES \
    --save-all \
    --visualize 0 1 2 3 4

# 2. Comparative Analysis
echo ""
echo "Step 2: Comparative Analysis"
echo "----------------------------"

# Compare both models
python scripts/evaluate_model.py \
    --models $DISCO_MODEL $UNET_MODEL \
    --names "DISCO Neural Operator" "U-Net Baseline" \
    --output $OUTPUT_DIR/comparative_analysis \
    --max-samples $MAX_SAMPLES

# 3. Generate Baseline Comparisons
echo ""
echo "Step 3: Baseline Method Comparison"
echo "---------------------------------"

# Generate zero-filled reconstruction metrics for comparison
python scripts/generate_baseline_metrics.py \
    --output $OUTPUT_DIR/baseline_metrics.json \
    --max-samples $MAX_SAMPLES

# Re-run comparison including baseline methods
python scripts/evaluate_model.py \
    --models $DISCO_MODEL $UNET_MODEL \
    --names "DISCO Neural Operator" "U-Net Baseline" \
    --baseline-results $OUTPUT_DIR/baseline_metrics.json \
    --output $OUTPUT_DIR/full_comparison

# 4. Generate Final Report
echo ""
echo "Step 4: Generating Final Report"
echo "------------------------------"

python scripts/generate_final_report.py \
    --disco-results $OUTPUT_DIR/disco_results \
    --unet-results $OUTPUT_DIR/unet_results \
    --comparison-results $OUTPUT_DIR/full_comparison \
    --output $OUTPUT_DIR/final_report.html

echo ""
echo "Evaluation Complete!"
echo "Results available in: $OUTPUT_DIR"
echo "Main report: $OUTPUT_DIR/final_report.html"