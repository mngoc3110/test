#!/bin/bash
# ==============================================================================
# DAiSEE STANDALONE TRAINING SCRIPT FOR COLAB
# This script handles setup, data merging, and training in one go.
# ==============================================================================

# 1. Install Dependencies
echo "=> Installing dependencies..."
pip install ftfy regex tqdm -q

# 2. Path Configuration
# Consistent with your project structure: dataset/DAiSEE/...
DATASET_ROOT="./dataset/DAiSEE/DataSet"
# Pointing to the correct annotations folder you just listed
ANNOTATION_ROOT="./dataset/DAiSEE/annotations"
TRAIN_LIST="$ANNOTATION_ROOT/daisee_train_full.txt"

# 3. Automatic Data Merging (Train + Val -> Full Train)
echo "=> Merging training and validation sets..."
if [ -f "$ANNOTATION_ROOT/train.txt" ] && [ -f "$ANNOTATION_ROOT/validation.txt" ]; then
    cat "$ANNOTATION_ROOT/train.txt" "$ANNOTATION_ROOT/validation.txt" > "$TRAIN_LIST"
    echo "   Successfully created $TRAIN_LIST"
    echo "   Total samples in merged list: $(wc -l < "$TRAIN_LIST")"
else
    echo "   Error: Could not find train.txt or validation.txt in $ANNOTATION_ROOT"
    ls -F "$ANNOTATION_ROOT"
    exit 1
fi

# 4. Start Training
echo "=> Starting Training Pipeline (SOTA Max WAR Configuration)..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --mode train \
  --exper-name Train-DAiSEE-SOTA-Colab \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 20 \
  --batch-size 4 \
  --optimizer AdamW \
  --lr 1e-4 \
  --lr-image-encoder 1e-5 \
  --lr-prompt-learner 5e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.02 \
  --milestones 10 15 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --temporal-pooling attn_pool \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "$DATASET_ROOT" \
  --train-annotation "$TRAIN_LIST" \
  --val-annotation "$ANNOTATION_ROOT/test.txt" \
  --test-annotation "$ANNOTATION_ROOT/test.txt" \
  --bounding-box-face "" \
  --clip-path "ViT-B/16" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --loss-type ce \
  --lambda_dc 0.1 \
  --mi-warmup 5 \
  --dc-warmup 5 \
  --lambda_mi 0.1 \
  --use-amp \
  --grad-clip 1.0 \
  --mixup-alpha 0.2


