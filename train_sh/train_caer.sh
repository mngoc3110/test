#!/bin/bash
# CAER Training Script - Exact Replication of RAER Configuration
# Loss: LDAM | Sampler: Weighted | Mixup: 0.0 | Adapter: Learned | Prompt: Tuning (CoOp)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --mode train \
  --exper-name Train-CAER-Full-Config \
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
  --root-dir ./dataset/CAER \
  --train-annotation ./dataset/CAER/train.txt \
  --val-annotation ./dataset/CAER/test.txt \
  --test-annotation ./dataset/CAER/test.txt \
  --clip-path ViT-B/16 \
  --bounding-box-face ./dataset/CAER/bounding_box/face_boxes.json \
  --bounding-box-body ./dataset/CAER/bounding_box/body_boxes.json \
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
  --use-weighted-sampler \
  --crop-body \
  --grad-clip 1.0 \
  --mixup-alpha 0.0