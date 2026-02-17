#!/bin/bash
# DAiSEE Training Script - Optimized for WAR (Accuracy)
# Configuration: Attention Pooling | LDAM (High Margin) | Weighted Sampler (Essential) | Stable LR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- PATH CONFIGURATION ---
# Detect environment and set paths
if [ -d "/kaggle/input" ]; then
    echo "=> Detected Kaggle Environment."
    DATASET_ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data/DataSet"
    ANNOTATION_ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
else
    echo "=> Detected Local Environment (Mac/Linux)."
    DATASET_ROOT="./dataset/DAiSEE/DataSet"
    ANNOTATION_ROOT="./dataset/DAiSEE"
fi

echo "Using DATASET_ROOT: $DATASET_ROOT"
echo "Using ANNOTATION_ROOT: $ANNOTATION_ROOT"

python main.py \
  --mode train \
  --exper-name Train-DAiSEE-WAR-Balanced \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 20 \
  --batch-size 8 \
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
  --num-segments 8 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "$DATASET_ROOT" \
  --train-annotation "$ANNOTATION_ROOT/daisee_train.txt" \
  --val-annotation "$ANNOTATION_ROOT/daisee_val.txt" \
  --test-annotation "$ANNOTATION_ROOT/daisee_test.txt" \
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
  --dc-warmup 5 \
  --dc-ramp 10 \
  --lambda_mi 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --use-amp \
  --use-weighted-sampler \
  --grad-clip 1.0 \
  --mixup-alpha 0.0
