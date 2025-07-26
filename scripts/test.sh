#!/bin/bash

# Quick check of organized models directory
# File: scripts/quick_check_organized.sh

ORGANIZED_DIR="/scratch/vchaurasia/organized_models"

echo "=== QUICK CHECK: ORGANIZED MODELS ==="
echo ""

if [ -d "$ORGANIZED_DIR" ]; then
    echo "📁 Directory exists: $ORGANIZED_DIR"
    echo ""
    
    echo "📊 Total size: $(du -sh $ORGANIZED_DIR | cut -f1)"
    echo ""
    
    echo "🔍 Structure:"
    ls -la "$ORGANIZED_DIR"
    echo ""
    
    echo "🤖 DISCO models:"
    if [ -d "$ORGANIZED_DIR/disco" ]; then
        ls -lh "$ORGANIZED_DIR/disco/"*.pth 2>/dev/null || echo "  No models found"
    else
        echo "  disco/ directory not found"
    fi
    echo ""
    
    echo "🔬 UNet models:"
    if [ -d "$ORGANIZED_DIR/unet" ]; then
        ls -lh "$ORGANIZED_DIR/unet/"*.pth 2>/dev/null || echo "  No models found"
    else
        echo "  unet/ directory not found"
    fi
    echo ""
    
    echo "🔗 Symlinks:"
    if [ -L "$ORGANIZED_DIR/disco_epoch20.pth" ]; then
        echo "  ✓ disco_epoch20.pth -> $(readlink $ORGANIZED_DIR/disco_epoch20.pth)"
    else
        echo "  ✗ disco_epoch20.pth not found"
    fi
    
    if [ -L "$ORGANIZED_DIR/unet_epoch20.pth" ]; then
        echo "  ✓ unet_epoch20.pth -> $(readlink $ORGANIZED_DIR/unet_epoch20.pth)"
    else
        echo "  ✗ unet_epoch20.pth not found"
    fi
    echo ""
    
    echo "📋 Ready for inference?"
    if [ -f "$ORGANIZED_DIR/disco_epoch20.pth" ] && [ -f "$ORGANIZED_DIR/unet_epoch20.pth" ]; then
        echo "  ✅ YES! Both epoch 20 models ready"
        echo ""
        echo "🚀 Next steps:"
        echo "  1. python scripts/create_inference_configs.py"
        echo "  2. bash run_inference_organized.sh"
    else
        echo "  ❌ NO - Missing epoch 20 models"
        echo ""
        echo "🛠️  Manual fix needed:"
        echo "  Find closest epochs and create symlinks:"
        echo "  ln -sf disco/checkpoint_epoch_XX.pth $ORGANIZED_DIR/disco_epoch20.pth"
        echo "  ln -sf unet/checkpoint_epoch_YY.pth $ORGANIZED_DIR/unet_epoch20.pth"
    fi
    
else
    echo "❌ Directory not found: $ORGANIZED_DIR"
    echo ""
    echo "🛠️  Need to run: bash scripts/move_and_organize_models.sh"
fi