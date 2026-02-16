#!/bin/bash
# CAER Local Training Script - Optimized for WAR (Accuracy) using Advanced Pipeline
# Loss: LDAM (Stronger than CE) | Sampler: None (Natural distribution for WAR) | Prompt: Tuning (CoOp)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- LOCAL PATHS ---
DATASET_ROOT="./dataset/CAER"
ANNOTATION_ROOT="./dataset/CAER/annotations"
BOUNDING_BOX_ROOT="./dataset/CAER/bounding_box"

python main.py \
  --mode train \
  --exper-name Train-CAER-WAR-Advanced \
  --dataset CAER \
  --gpu 0 \
  --epochs 20 \
  --batch-size 4 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.005 \
  --milestones 10 15 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "$DATASET_ROOT" \
  --train-annotation "$ANNOTATION_ROOT/train.txt" \
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
  --ldam-max-m 0.5 \
  --lambda_dc 0.1 \
  --mi-warmup 5 \
  --dc-warmup 5 \
  --lambda_mi 0.1 \
  --use-amp \
  --crop-body \
  --grad-clip 1.0 \
  --mixup-alpha 0.0