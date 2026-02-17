#!/bin/bash
# DAiSEE Training Script - SOTA Configuration (Aiming for 70% WAR)
# Strategy: Full Data (Train+Val) | CE Loss (Max Accuracy) | Natural Distribution | 16 Frames

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- PATH CONFIGURATION ---
if [ -d "/kaggle/input" ]; then
    echo "=> Detected Kaggle Environment."
    DATASET_ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data/DataSet"
    ANNOTATION_ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
    # Create Full Train list (Train + Val) in a writable directory
    cat "$ANNOTATION_ROOT/daisee_train.txt" "$ANNOTATION_ROOT/daisee_val.txt" > daisee_train_full.txt
    TRAIN_LIST="daisee_train_full.txt"
else
    echo "=> Detected Local Environment."
    DATASET_ROOT="./dataset/DAiSEE/DataSet"
    ANNOTATION_ROOT="./dataset/DAiSEE"
    # Create Full Train list
    cat "$ANNOTATION_ROOT/daisee_train.txt" "$ANNOTATION_ROOT/daisee_val.txt" > dataset/DAiSEE/daisee_train_full.txt
    TRAIN_LIST="$ANNOTATION_ROOT/daisee_train_full.txt"
fi

echo "Using Full Training List: $TRAIN_LIST"

python main.py \
  --mode train \
  --exper-name Train-DAiSEE-SOTA-MaxWAR \
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
  --val-annotation "$ANNOTATION_ROOT/daisee_test.txt" \
  --test-annotation "$ANNOTATION_ROOT/daisee_test.txt" \
  --bounding-box-face "" \
  --clip-path "ViT-B/16" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --loss-type ce \
  --lambda_dc 0.1 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --lambda_mi 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --use-amp \
  --grad-clip 1.0 \
  --mixup-alpha 0.2
