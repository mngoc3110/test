#!/bin/bash
# CAER RESEARCH PIPELINE - TARGET UAR > 82%
# Chiến lược: Fix Scaling + Unfreeze Backbone + High Update Frequency

python main.py \
  --mode train \
  --dataset CAER \
  --exper-name CAER-Research-UAR82 \
  --gpu 0 \
  --workers 4 \
  --epochs 30 \
  --batch-size 8 \
  --accumulation-steps 2 \
  --optimizer AdamW \
  --scheduler cosine \
  --lr 5e-05 \
  --lr-image-encoder 1e-05 \
  --lr-prompt-learner 2e-04 \
  --lr-adapter 2e-04 \
  --weight-decay 0.0005 \
  --num-segments 8 \
  --temperature 1.0 \
  --loss-type ldam \
  --ldam-max-m 0.2 \
  --ldam-s 30.0 \
  --use-weighted-sampler True \
  --lambda_mi 0.05 \
  --mi-warmup 2 \
  --label-smoothing 0 \
  --root-dir /kaggle/input/processed-caer-video-dataset \
  --train-annotation /kaggle/working/test/CAER_Video/train_subset.txt \
  --val-annotation /kaggle/input/processed-caer-video-dataset/validation.txt \
  --clip-path ViT-B/32 \
  --bounding-box-face /kaggle/input/processed-caer-video-dataset/face_boxes_mediapipe.json
