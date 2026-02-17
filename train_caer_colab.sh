#!/bin/bash
# ==============================================================================
# CAER STANDALONE TRAINING SCRIPT FOR COLAB
# All-in-one: Install, Merge Data, and Train (Benchmarking Config)
# ==============================================================================

# 1. Install Dependencies
echo "=> Installing dependencies..."
pip install ftfy regex tqdm -q

# 2. Path Configuration (Aligned with project structure)
DATASET_ROOT="./dataset/CAER"
ANNOTATION_ROOT="./dataset/CAER/annotations"
TRAIN_LIST="./dataset/CAER/annotations/caer_train_full.txt"
BOUNDING_BOX_ROOT="./dataset/CAER/bounding_box"

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

# 4. Start Training (Benchmarking Config: LDAM + Weighted Sampler + 8 Frames)
echo "=> Starting Training Pipeline (CAER SOTA Config)..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --mode train \
  --exper-name Train-CAER-SOTA-Colab \
  --dataset CAER \
  --gpu 0 \
  --epochs 40 \
  --batch-size 8 \
  --optimizer AdamW \
  --lr 1e-4 \
  --lr-image-encoder 5e-6 \
  --lr-prompt-learner 5e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.02 \
  --milestones 20 30 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --temporal-pooling attn_pool \
  --num-segments 8 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "$DATASET_ROOT" \
  --train-annotation "$TRAIN_LIST" \
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
  --lambda_mi 0.1 \
  --mi-warmup 5 \
  --dc-warmup 5 \
  --use-amp \
  --use-weighted-sampler \
  --crop-body \
  --grad-clip 1.0 \
  --mixup-alpha 0.2
