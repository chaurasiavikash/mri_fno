#!/bin/bash

# Debug inference issues
# File: scripts/debug_inference_issue.sh

echo "=== DEBUGGING INFERENCE ISSUES ==="
echo "Time: $(date)"
echo ""

ORGANIZED_DIR="/scratch/vchaurasia/organized_models"
LOGS_DIR="$ORGANIZED_DIR/logs"

echo "🔍 Checking error logs..."

# Find the latest error logs
DISCO_ERR=$(ls -t $LOGS_DIR/disco_inference_*.err 2>/dev/null | head -1)
UNET_ERR=$(ls -t $LOGS_DIR/unet_inference_*.err 2>/dev/null | head -1)

if [ -f "$UNET_ERR" ]; then
    echo ""
    echo "❌ UNet Error Log ($UNET_ERR):"
    echo "================================="
    cat "$UNET_ERR"
    echo ""
fi

if [ -f "$DISCO_ERR" ]; then
    echo ""
    echo "🔍 DISCO Error Log ($DISCO_ERR):"
    echo "================================="
    if [ -s "$DISCO_ERR" ]; then
        cat "$DISCO_ERR"
    else
        echo "(No errors logged yet)"
    fi
    echo ""
fi

echo "📊 Job Status:"
echo "=============="
squeue -u $(whoami) --format="%.10i %.20j %.8T %.10M %.6D %R"
echo ""

echo "🔍 Config File Check:"
echo "===================="
echo "DISCO config:"
if [ -f "configs/disco_inference_scratch.yaml" ]; then
    echo "✅ configs/disco_inference_scratch.yaml exists"
    echo "Model section:"
    grep -A 10 "^model:" configs/disco_inference_scratch.yaml
else
    echo "❌ configs/disco_inference_scratch.yaml missing"
fi

echo ""
echo "UNet config:"
if [ -f "configs/unet_inference_scratch.yaml" ]; then
    echo "✅ configs/unet_inference_scratch.yaml exists"
    echo "Model section:"
    grep -A 10 "^model:" configs/unet_inference_scratch.yaml
else
    echo "❌ configs/unet_inference_scratch.yaml missing"
fi

echo ""
echo "🔧 Model File Check:"
echo "==================="
echo "DISCO model:"
if [ -L "$ORGANIZED_DIR/disco_epoch20.pth" ]; then
    echo "✅ Symlink exists: $(readlink $ORGANIZED_DIR/disco_epoch20.pth)"
    if [ -f "$ORGANIZED_DIR/disco_epoch20.pth" ]; then
        echo "✅ Target file exists: $(ls -lh $ORGANIZED_DIR/disco_epoch20.pth)"
    else
        echo "❌ Symlink target missing"
    fi
else
    echo "❌ disco_epoch20.pth symlink missing"
fi

echo ""
echo "UNet model:"
if [ -L "$ORGANIZED_DIR/unet_epoch20.pth" ]; then
    echo "✅ Symlink exists: $(readlink $ORGANIZED_DIR/unet_epoch20.pth)"
    if [ -f "$ORGANIZED_DIR/unet_epoch20.pth" ]; then
        echo "✅ Target file exists: $(ls -lh $ORGANIZED_DIR/unet_epoch20.pth)"
    else
        echo "❌ Symlink target missing"
    fi
else
    echo "❌ unet_epoch20.pth symlink missing"
fi

echo ""
echo "📂 Directory Structure:"
echo "======================"
echo "Inference results:"
ls -la "$ORGANIZED_DIR/inference_results/" 2>/dev/null || echo "Directory doesn't exist yet"

echo ""
echo "Available models:"
echo "DISCO models:"
ls -la "$ORGANIZED_DIR/disco/"*.pth 2>/dev/null || echo "No DISCO models found"
echo "UNet models:"
ls -la "$ORGANIZED_DIR/unet/"*.pth 2>/dev/null || echo "No UNet models found"

echo ""
echo "🔧 Quick Fix Suggestions:"
echo "========================"
echo "If UNet failed due to model architecture mismatch:"
echo "1. Check if UNet model was trained with different config"
echo "2. Compare training config vs inference config"
echo "3. May need to update UNet inference config"
echo ""
echo "If DISCO is running slowly:"
echo "1. Check GPU utilization: nvidia-smi"
echo "2. Monitor progress in output log"
echo "3. Typical inference time: 30-60 minutes"