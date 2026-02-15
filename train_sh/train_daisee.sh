#!/bin/bash
# DAiSEE Training Script - Engagement Only
# Configuration: Attention Pooling | With MI/DC Loss | LDAM | Weighted Sampler | No Mixup

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- LOCAL PATHS ---
DATASET_ROOT="./dataset/DAiSEE/DataSet"
ANNOTATION_ROOT="./dataset/DAiSEE/annotations"

python main.py \
  --mode train \
  --exper-name Train-DAiSEE-Engage-AttnPool \
  --dataset DAiSEE \
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
  --temporal-pooling attn_pool \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "${DATASET_ROOT}" \
  --train-annotation "${ANNOTATION_ROOT}/train.txt" \
  --val-annotation "${ANNOTATION_ROOT}/validation.txt" \
  --test-annotation "${ANNOTATION_ROOT}/test.txt" \
  --bounding-box-face "" \
  --clip-path "ViT-B/16" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --loss-type ldam \
  --ldam-s 30.0 \
  --ldam-max-m 0.5 \
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
