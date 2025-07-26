#!/bin/bash

# Script to submit both inference jobs and monitor them
# File: scripts/submit_inference_jobs.sh

echo "=== SUBMITTING MRI INFERENCE JOBS ==="
echo "Date: $(date)"
echo ""

# Create directories
ORGANIZED_DIR="/scratch/vchaurasia/organized_models"
mkdir -p "$ORGANIZED_DIR/logs"
mkdir -p jobs

# Step 1: Create inference configs
echo "📝 Creating inference configurations..."
python scripts/create_scratch_inference_configs.py

echo ""
echo "🔍 Checking model availability..."

# Check DISCO model
if [ -f "$ORGANIZED_DIR/disco_epoch20.pth" ]; then
    echo "✅ DISCO model ready: $(ls -lh $ORGANIZED_DIR/disco_epoch20.pth)"
    DISCO_READY=true
else
    echo "❌ DISCO model not found"
    echo "Available DISCO models:"
    ls -la "$ORGANIZED_DIR/disco/"*.pth 2>/dev/null || echo "  No models found"
    DISCO_READY=false
fi

# Check UNet model
if [ -f "$ORGANIZED_DIR/unet_epoch20.pth" ]; then
    echo "✅ UNet model ready: $(ls -lh $ORGANIZED_DIR/unet_epoch20.pth)"
    UNET_READY=true
else
    echo "❌ UNet model not found"
    echo "Available UNet models:"
    ls -la "$ORGANIZED_DIR/unet/"*.pth 2>/dev/null || echo "  No models found"
    UNET_READY=false
fi

echo ""

# Submit jobs
JOB_IDS=()

if [ "$DISCO_READY" = true ]; then
    echo "🚀 Submitting DISCO inference job..."
    DISCO_JOB_ID=$(sbatch --parsable jobs/submit_disco_inference.sh)
    
    if [ $? -eq 0 ]; then
        echo "✅ DISCO job submitted: Job ID $DISCO_JOB_ID"
        JOB_IDS+=($DISCO_JOB_ID)
    else
        echo "❌ Failed to submit DISCO job"
    fi
else
    echo "⏭️  Skipping DISCO job submission (model not ready)"
fi

if [ "$UNET_READY" = true ]; then
    echo "🚀 Submitting UNet inference job..."
    UNET_JOB_ID=$(sbatch --parsable jobs/submit_unet_inference.sh)
    
    if [ $? -eq 0 ]; then
        echo "✅ UNet job submitted: Job ID $UNET_JOB_ID"
        JOB_IDS+=($UNET_JOB_ID)
    else
        echo "❌ Failed to submit UNet job"
    fi
else
    echo "⏭️  Skipping UNet job submission (model not ready)"
fi

echo ""

if [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo "📊 Job Status Summary:"
    echo "Submitted jobs: ${JOB_IDS[*]}"
    echo ""
    
    echo "🔍 Current queue status:"
    squeue -u $(whoami) --format="%.10i %.20j %.8T %.10M %.20S"
    
    echo ""
    echo "📝 Monitoring commands:"
    echo "  Check queue: squeue -u $(whoami)"
    echo "  Watch logs:  tail -f $ORGANIZED_DIR/logs/*_inference_*.out"
    echo "  Job details: scontrol show job <JOB_ID>"
    
    echo ""
    echo "📂 Output locations:"
    echo "  DISCO results: $ORGANIZED_DIR/inference_results/disco_epoch20/"
    echo "  UNet results:  $ORGANIZED_DIR/inference_results/unet_epoch20/"
    echo "  Job logs:      $ORGANIZED_DIR/logs/"
    
    echo ""
    echo "⏰ Estimated completion time: ~30-60 minutes per job"
    echo ""
    echo "🔔 To get notified when jobs complete:"
    echo "  watch -n 30 'squeue -u $(whoami)'"
    
    # Create a status check script
    cat << EOF > check_inference_status.sh
#!/bin/bash

echo "=== INFERENCE JOBS STATUS ==="
echo "Time: \$(date)"
echo ""

echo "🔍 Queue status:"
squeue -u \$(whoami) --format="%.10i %.20j %.8T %.10M %.6D %R"

echo ""
echo "📁 Output directories:"
for dir in disco_epoch20 unet_epoch20; do
    result_dir="$ORGANIZED_DIR/inference_results/\$dir"
    if [ -d "\$result_dir" ]; then
        echo "  \$dir: \$(find \$result_dir -name "*.txt" -o -name "*.npz" | wc -l) files"
        if [ -f "\$result_dir/summary_metrics.txt" ]; then
            echo "    ✅ Summary metrics available"
        else
            echo "    ⏳ Still processing..."
        fi
    else
        echo "  \$dir: Not started yet"
    fi
done

echo ""
echo "📊 Latest log entries:"
tail -5 $ORGANIZED_DIR/logs/*_inference_*.out 2>/dev/null | head -20

echo ""
echo "Run this script again to check status: bash check_inference_status.sh"
EOF
    
    chmod +x check_inference_status.sh
    echo "📋 Created status checker: ./check_inference_status.sh"
    
else
    echo "❌ No jobs were submitted. Please check your model setup:"
    echo "  1. Run: bash scripts/quick_check_organized.sh"
    echo "  2. Ensure epoch 20 models are available"
    echo "  3. Create symlinks manually if needed"
fi

echo ""
echo "=== NEXT STEPS ==="
echo "1. Monitor jobs with: ./check_inference_status.sh"
echo "2. When both jobs complete, run analysis:"
echo "   python scripts/error_analysis_comparison.py"
echo "3. Generate final report"