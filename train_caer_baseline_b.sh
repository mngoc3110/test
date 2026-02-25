#!/bin/bash
# CAER TRAINING - BASELINE B (No Prompt Learning)
# ViT-B/32 + CE Loss + Prompt Learner OFF + Text Ensemble + No MI/DC
# UPDATED: Reduced LR/WD, Added Weighted Sampler, Added Body BBox (Sparse)

python main.py \
  --mode train \
  --exper-name CAER-Baseline-B-Prompt-OFF-BodyBox \
  --dataset CAER \
  --gpu mps \
  --epochs 40 \
  --batch-size 8 \
  --accumulation-steps 4 \
  --optimizer AdamW \
  --scheduler cosine \
  --lr 5e-5 \
  --lr-image-encoder 5e-6 \
  --weight-decay 0.001 \
  --num-segments 8 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 100 \
  --root-dir ./CAER_Video \
  --train-annotation ./CAER_Video/train.txt \
  --val-annotation ./CAER_Video/validation.txt \
  --test-annotation ./CAER_Video/test.txt \
  --clip-path ViT-B/32 \
  --bounding-box-face ./CAER_Video/face_boxes_mediapipe.json \
  --bounding-box-body ./CAER_Video/body_boxes.json \
  --crop-body \
  --text-type prompt_ensemble \
  --load_and_tune_prompt_learner False \
  --loss-type ce \
  --lambda_dc 0.0 \
  --lambda_mi 0.0 \
  --use-weighted-sampler True \
  --label-smoothing 0.0 \
  --grad-clip 1.0 \
  --mixup-alpha 0.0