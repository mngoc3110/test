#!/bin/bash
# CAER Quick Training Script - For Testing Pipeline Speed & Stability
# Strategy: Subset Data (500 samples) | 5 Epochs | Fast Debug

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- PATH CONFIGURATION ---
DATASET_ROOT="./dataset/CAER"
ANNOTATION_ROOT="./dataset/CAER/annotations"
FULL_TRAIN="$ANNOTATION_ROOT/train.txt"
QUICK_TRAIN="$ANNOTATION_ROOT/train_quick.txt"
BOUNDING_BOX_ROOT="./dataset/CAER/bounding_box"

# 1. Create Quick Train List (First 1000 lines ~ 15% data)
echo "=> Creating quick training subset..."
if [ -f "$FULL_TRAIN" ]; then
    head -n 1000 "$FULL_TRAIN" > "$QUICK_TRAIN"
    echo "   Created $QUICK_TRAIN with $(wc -l < "$QUICK_TRAIN") samples."
else
    echo "   Error: $FULL_TRAIN not found."
    exit 1
fi

# 2. Start Training
echo "=> Starting Quick Training..."

python main.py \
  --mode train \
  --exper-name Train-CAER-Quick-Debug \
  --dataset CAER \
  --gpu 0 \
  --epochs 5 \
  --batch-size 8 \
  --optimizer AdamW \
  --lr 1e-4 \
  --lr-image-encoder 5e-6 \
  --lr-prompt-learner 5e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.01 \
  --milestones 2 4 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --temporal-pooling attn_pool \
  --num-segments 8 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 10 \
  --root-dir "$DATASET_ROOT" \
  --train-annotation "$QUICK_TRAIN" \
  --val-annotation "$ANNOTATION_ROOT/test.txt" \
  --test-annotation "$ANNOTATION_ROOT/test.txt" \
  --bounding-box-face "$BOUNDING_BOX_ROOT/face.json" \
  --bounding-box-body "$BOUNDING_BOX_ROOT/body.json" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --loss-type ldam \
  --ldam-s 30.0 \
  --ldam-max-m 0.35 \
  --lambda_dc 0.1 \
  --mi-warmup 1 \
  --dc-warmup 1 \
  --lambda_mi 0.1 \
  --use-amp \
  --use-weighted-sampler \
  --crop-body \
  --grad-clip 1.0 \
  --mixup-alpha 0.2
