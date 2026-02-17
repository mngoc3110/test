#!/bin/bash
# CAER Local Training Script - SOTA Configuration (Aiming for 80% Accuracy)
# Strategy: Combine Train+Val -> Train Full | Validate on Test | Mixup + More Frames

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- LOCAL PATHS ---
DATASET_ROOT="./dataset/CAER"
ANNOTATION_ROOT="./dataset/CAER/annotations"
BOUNDING_BOX_ROOT="./dataset/CAER/bounding_box"

python main.py \
  --mode train \
  --exper-name Train-CAER-SOTA-FullData \
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
  --num-segments 8 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "$DATASET_ROOT" \
  --train-annotation "$ANNOTATION_ROOT/train_val_full.txt" \
  --val-annotation "$ANNOTATION_ROOT/test.txt" \
  --test-annotation "$ANNOTATION_ROOT/test.txt" \
  --clip-path ViT-B/16 \
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
  --mi-warmup 5 \
  --dc-warmup 5 \
  --lambda_mi 0.1 \
  --use-amp \
  --use-weighted-sampler \
  --crop-body \
  --grad-clip 1.0 \
  --mixup-alpha 0.2