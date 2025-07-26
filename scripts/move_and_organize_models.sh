#!/bin/bash

# Script to MOVE (not copy) trained models to save space
# File: scripts/move_and_organize_models.sh

echo "=== MOVING AND ORGANIZING TRAINED MODELS (SPACE-EFFICIENT) ==="
echo "Date: $(date)"
echo ""

# Actual model locations from config files
DISCO_MODEL_DIR="/home/vchaurasia/projects/mri_fno/outputs/test_models"
DISCO_LOG_DIR="/home/vchaurasia/projects/mri_fno/outputs/test_logs"

UNET_MODEL_DIR="/scratch/vchaurasia/mri_models/unet/models"
UNET_LOG_DIR="/scratch/vchaurasia/mri_models/unet/logs"

# Target organized location
ORGANIZED_DIR="/scratch/vchaurasia/organized_models"

echo "⚠️  WARNING: This script will MOVE (not copy) your model files!"
echo "   Original locations will no longer contain the models."
echo "   Press Ctrl+C in the next 5 seconds to cancel..."
sleep 5

echo ""
echo "=== CHECKING DISCO MODELS ==="
echo "Looking in: $DISCO_MODEL_DIR"
if [ -d "$DISCO_MODEL_DIR" ]; then
    echo "✓ DISCO model directory exists"
    echo "DISCO model files:"
    find "$DISCO_MODEL_DIR" -name "*.pth" -exec ls -lh {} \; | while read line; do
        echo "  $line"
    done
    
    # Look for epoch 20 specifically
    disco_epoch20=$(find "$DISCO_MODEL_DIR" -name "*epoch_20*" -o -name "*epoch_020*" | head -1)
    if [ -n "$disco_epoch20" ]; then
        echo "✓ Found DISCO epoch 20: $disco_epoch20"
    else
        echo "⚠ DISCO epoch 20 not found, looking for nearby epochs..."
        find "$DISCO_MODEL_DIR" -name "*epoch_1*" -o -name "*epoch_2*" | sort
    fi
else
    echo "✗ DISCO model directory not found: $DISCO_MODEL_DIR"
fi
echo ""

echo "=== CHECKING UNET MODELS ==="
echo "Looking in: $UNET_MODEL_DIR"
if [ -d "$UNET_MODEL_DIR" ]; then
    echo "✓ UNet model directory exists"
    echo "UNet model files:"
    find "$UNET_MODEL_DIR" -name "*.pth" -exec ls -lh {} \; | while read line; do
        echo "  $line"
    done
    
    # Look for epoch 20 specifically
    unet_epoch20=$(find "$UNET_MODEL_DIR" -name "*epoch_20*" -o -name "*epoch_020*" | head -1)
    if [ -n "$unet_epoch20" ]; then
        echo "✓ Found UNet epoch 20: $unet_epoch20"
    else
        echo "⚠ UNet epoch 20 not found, looking for nearby epochs..."
        find "$UNET_MODEL_DIR" -name "*epoch_1*" -o -name "*epoch_2*" | sort
    fi
else
    echo "✗ UNet model directory not found: $UNET_MODEL_DIR"
fi
echo ""

echo "=== CREATING ORGANIZED STRUCTURE ==="
mkdir -p "$ORGANIZED_DIR/disco"
mkdir -p "$ORGANIZED_DIR/unet"
mkdir -p "$ORGANIZED_DIR/logs/disco"
mkdir -p "$ORGANIZED_DIR/logs/unet"
mkdir -p "$ORGANIZED_DIR/inference_results"

echo "Created organized directory structure at: $ORGANIZED_DIR"
echo ""

echo "=== MOVING MODELS TO ORGANIZED LOCATION ==="

# Move DISCO models
if [ -d "$DISCO_MODEL_DIR" ]; then
    echo "Moving DISCO models..."
    
    # Check if there are .pth files to move
    disco_pth_files=$(find "$DISCO_MODEL_DIR" -name "*.pth" | wc -l)
    if [ "$disco_pth_files" -gt 0 ]; then
        find "$DISCO_MODEL_DIR" -name "*.pth" -exec mv {} "$ORGANIZED_DIR/disco/" \;
        echo "  Moved $disco_pth_files DISCO model files"
    else
        echo "  No .pth files found in DISCO directory"
    fi
    
    # Move logs if they exist (optional, takes less space)
    if [ -d "$DISCO_LOG_DIR" ] && [ "$(ls -A $DISCO_LOG_DIR)" ]; then
        mv "$DISCO_LOG_DIR"/* "$ORGANIZED_DIR/logs/disco/" 2>/dev/null || echo "  No logs to move"
        echo "  Moved DISCO logs"
    fi
fi

# Move UNet models  
if [ -d "$UNET_MODEL_DIR" ]; then
    echo "Moving UNet models..."
    
    # Check if there are .pth files to move
    unet_pth_files=$(find "$UNET_MODEL_DIR" -name "*.pth" | wc -l)
    if [ "$unet_pth_files" -gt 0 ]; then
        find "$UNET_MODEL_DIR" -name "*.pth" -exec mv {} "$ORGANIZED_DIR/unet/" \;
        echo "  Moved $unet_pth_files UNet model files"
    else
        echo "  No .pth files found in UNet directory"
    fi
    
    # Move logs if they exist (optional, takes less space)
    if [ -d "$UNET_LOG_DIR" ] && [ "$(ls -A $UNET_LOG_DIR)" ]; then
        mv "$UNET_LOG_DIR"/* "$ORGANIZED_DIR/logs/unet/" 2>/dev/null || echo "  No logs to move"
        echo "  Moved UNet logs"
    fi
fi

echo ""
echo "=== FINAL ORGANIZATION CHECK ==="
echo "DISCO models in $ORGANIZED_DIR/disco/:"
ls -lh "$ORGANIZED_DIR/disco/"*.pth 2>/dev/null || echo "  No DISCO models found"

echo ""
echo "UNet models in $ORGANIZED_DIR/unet/:"
ls -lh "$ORGANIZED_DIR/unet/"*.pth 2>/dev/null || echo "  No UNet models found"

echo ""
echo "=== IDENTIFYING EPOCH 20 MODELS ==="

# Find epoch 20 models
disco_20=$(find "$ORGANIZED_DIR/disco" -name "*epoch_20*" -o -name "*epoch_020*" | head -1)
unet_20=$(find "$ORGANIZED_DIR/unet" -name "*epoch_20*" -o -name "*epoch_020*" | head -1)

if [ -n "$disco_20" ] && [ -n "$unet_20" ]; then
    echo "✓ Both epoch 20 models found:"
    echo "  DISCO: $disco_20"
    echo "  UNet:  $unet_20"
    
    # Create symlinks for easy access
    ln -sf "disco/$(basename $disco_20)" "$ORGANIZED_DIR/disco_epoch20.pth"
    ln -sf "unet/$(basename $unet_20)" "$ORGANIZED_DIR/unet_epoch20.pth"
    
    echo ""
    echo "✓ Created convenience symlinks:"
    echo "  $ORGANIZED_DIR/disco_epoch20.pth -> disco/$(basename $disco_20)"
    echo "  $ORGANIZED_DIR/unet_epoch20.pth -> unet/$(basename $unet_20)"
    
elif [ -n "$disco_20" ]; then
    echo "✓ DISCO epoch 20 found: $disco_20"
    echo "✗ UNet epoch 20 not found"
    ln -sf "disco/$(basename $disco_20)" "$ORGANIZED_DIR/disco_epoch20.pth"
    
    echo ""
    echo "Available UNet epochs:"
    find "$ORGANIZED_DIR/unet" -name "*epoch*" | sort
    echo "Please manually create symlink for closest UNet epoch to 20"
    
elif [ -n "$unet_20" ]; then
    echo "✗ DISCO epoch 20 not found"
    echo "✓ UNet epoch 20 found: $unet_20"
    ln -sf "unet/$(basename $unet_20)" "$ORGANIZED_DIR/unet_epoch20.pth"
    
    echo ""
    echo "Available DISCO epochs:"
    find "$ORGANIZED_DIR/disco" -name "*epoch*" | sort
    echo "Please manually create symlink for closest DISCO epoch to 20"
    
else
    echo "✗ Neither epoch 20 model found"
    echo ""
    echo "Available DISCO epochs:"
    find "$ORGANIZED_DIR/disco" -name "*epoch*" | sort
    echo ""
    echo "Available UNet epochs:"
    find "$ORGANIZED_DIR/unet" -name "*epoch*" | sort
    echo ""
    echo "Manual symlink creation needed. Examples:"
    echo "  ln -sf disco/checkpoint_epoch_XX.pth $ORGANIZED_DIR/disco_epoch20.pth"
    echo "  ln -sf unet/checkpoint_epoch_YY.pth $ORGANIZED_DIR/unet_epoch20.pth"
fi

echo ""
echo "=== SPACE SAVED ==="
original_disco_size=$(du -sh "$DISCO_MODEL_DIR" 2>/dev/null | cut -f1 || echo "0")
original_unet_size=$(du -sh "$UNET_MODEL_DIR" 2>/dev/null | cut -f1 || echo "0")
organized_size=$(du -sh "$ORGANIZED_DIR" 2>/dev/null | cut -f1)

echo "Models now organized in: $ORGANIZED_DIR ($organized_size)"
echo "Original locations freed up (DISCO: $original_disco_size, UNet: $original_unet_size)"

echo ""
echo "=== NEXT STEPS ==="
echo "1. Verify models are correctly organized in: $ORGANIZED_DIR"
echo "2. Check that symlinks point to the right epoch models"
echo "3. Run inference comparison:"
echo "   python scripts/create_inference_configs.py"
echo "   bash scripts/run_inference_organized.sh"

echo ""
echo "✓ Models successfully moved and organized!"
echo "✓ Space-efficient organization completed!"