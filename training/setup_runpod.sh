#!/bin/bash
# setup_runpod.sh — One-time setup for RunPod Pod (RTX 4090)
# Usage: bash training/setup_runpod.sh [--hf-token YOUR_TOKEN]
set -e

WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/voice"
DATASET_DIR="$WORKSPACE/dataset"

echo "=== RunPod Training Setup ==="
echo "Working directory: $REPO_DIR"

# Parse arguments
HF_TOKEN=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-token) HF_TOKEN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# 1. Install system dependencies
echo ""
echo "[1/5] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 > /dev/null 2>&1
echo "  Done."

# 2. Install Python dependencies
echo ""
echo "[2/5] Installing Python dependencies..."
cd "$REPO_DIR"
pip install -q -r training/requirements_train.txt
pip install -q f5-tts-th>=1.0.9 TTS>=0.22.0
pip install -q pyyaml librosa noisereduce
echo "  Done."

# 3. HuggingFace login (if token provided)
if [ -n "$HF_TOKEN" ]; then
    echo ""
    echo "[3/5] Logging into HuggingFace..."
    pip install -q huggingface_hub
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    echo "  Done."
else
    echo ""
    echo "[3/5] Skipping HuggingFace login (no --hf-token provided)"
fi

# 4. Download Porjai dataset
if [ -d "$DATASET_DIR" ] && [ "$(ls -A $DATASET_DIR 2>/dev/null)" ]; then
    echo ""
    echo "[4/5] Dataset already exists at $DATASET_DIR, skipping download."
else
    echo ""
    echo "[4/5] Downloading Porjai dataset from HuggingFace..."
    python -c "
from datasets import load_dataset
print('  Downloading CMKL/Porjai-Thai-voice-dataset-central...')
ds = load_dataset('CMKL/Porjai-Thai-voice-dataset-central', split='train')
ds.save_to_disk('$DATASET_DIR')
print('  Saved to $DATASET_DIR')
"
fi

# 5. Create output directories
echo ""
echo "[5/5] Creating output directories..."
mkdir -p "$WORKSPACE/training_data/f5/wavs"
mkdir -p "$WORKSPACE/training_data/xtts/wavs"
mkdir -p "$WORKSPACE/checkpoints/f5"
mkdir -p "$WORKSPACE/checkpoints/xtts"
echo "  Done."

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Prepare F5 dataset:"
echo "     python training/prepare_f5_dataset.py --input $DATASET_DIR --output $WORKSPACE/training_data/f5/"
echo ""
echo "  2. Prepare XTTS dataset:"
echo "     python training/prepare_xtts_dataset.py --input $DATASET_DIR --output $WORKSPACE/training_data/xtts/"
echo ""
echo "  3. Start F5 training:"
echo "     python training/train_f5.py --config training/config/f5_finetune.yaml"
echo ""
echo "  4. Start XTTS training:"
echo "     python training/train_xtts.py --config training/config/xtts_finetune.yaml"
