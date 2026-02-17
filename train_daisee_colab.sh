#!/bin/bash
# ==============================================================================
# DAiSEE STANDALONE TRAINING SCRIPT FOR COLAB
# All-in-one: Install, Merge Data, and Train (Benchmarking Config)
# ==============================================================================

# 1. Install Dependencies
echo "=> Installing dependencies..."
pip install ftfy regex tqdm -q

# 2. Path Configuration (Aligned with your project structure)
DATASET_ROOT="./dataset/DAiSEE/DataSet"
ANNOTATION_ROOT="./dataset/DAiSEE/annotations"
TRAIN_LIST="./dataset/DAiSEE/annotations/daisee_train_full.txt"

# 3. Automatic Data Merging (Train + Val -> Full Train)
echo "=> Merging training and validation sets..."
if [ -f "$ANNOTATION_ROOT/train.txt" ] && [ -f "$ANNOTATION_ROOT/validation.txt" ]; then
    cat "$ANNOTATION_ROOT/train.txt" "$ANNOTATION_ROOT/validation.txt" > "$TRAIN_LIST"
    echo "   Successfully created $TRAIN_LIST"
    echo "   Total samples in merged list: $(wc -l < "$TRAIN_LIST")"
else
    echo "   Error: Could not find train.txt or validation.txt in $ANNOTATION_ROOT"
    echo "   Available files in $ANNOTATION_ROOT:"
    ls -F "$ANNOTATION_ROOT"
    exit 1
fi

# 4. Start Training (Benchmarking Config: LDAM + Weighted Sampler + 16 Frames)
echo "=> Starting Training Pipeline..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --mode train \
  --exper-name Train-DAiSEE-Benchmarking-Colab \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 20 \
  --batch-size 4 \
  --optimizer AdamW \
  --lr 3e-5 \
  --lr-image-encoder 2e-6 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.01 \
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
  --loss-type ldam \
  --ldam-s 30.0 \
  --ldam-max-m 0.35 \
  --lambda_dc 0.1 \
  --lambda_mi 0.1 \
  --mi-warmup 5 \
  --dc-warmup 5 \
  --mi-ramp 10 \
  --dc-ramp 10 \
  --use-amp \
  --use-weighted-sampler \
  --grad-clip 1.0 \
  --mixup-alpha 0.2
