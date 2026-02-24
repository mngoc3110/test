#!/bin/bash
# CAER BASELINE - CLEAN TEST
# Mục tiêu: Đạt WAR > 35% để verify pipeline.

python main.py \
  --mode train \
  --dataset CAER \
  --exper-name CAER-Baseline-Clean \
  --gpu 0 \
  --workers 4 \
  --epochs 10 \
  --batch-size 8 \
  --accumulation-steps 4 \
  --optimizer AdamW \
  --scheduler cosine \
  --lr 2e-05 \
  --lr-image-encoder 1e-06 \
  --lr-prompt-learner 1e-04 \
  --loss-type ce \
  --use-weighted-sampler False \
  --num-segments 8 \
  --temperature 0.07 \
  --lambda_mi 0 \
  --lambda_dc 0 \
  --label-smoothing 0 \
  --root-dir /kaggle/input/processed-caer-video-dataset \
  --train-annotation /kaggle/working/test/CAER_Video/train_subset.txt \
  --val-annotation /kaggle/input/processed-caer-video-dataset/validation.txt \
  --clip-path ViT-B/32 \
  --bounding-box-face /kaggle/input/processed-caer-video-dataset/face_boxes_mediapipe.json
