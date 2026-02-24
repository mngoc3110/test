#!/bin/bash
# CAER TRAINING - FAST CONVERGENCE & HIGH UAR (>82%)
# Phù hợp chạy trên Kaggle/Colab với 1 GPU.

python main.py \
  --mode train \
  --exper-name CAER-Fast-Convergence-UAR82 \
  --dataset CAER \
  --gpu 0 \
  --workers 4 \
  --epochs 30 \
  --batch-size 8 \
  --accumulation-steps 4 \
  --optimizer AdamW \
  --scheduler cosine \
  --lr 5e-05 \
  --lr-image-encoder 1e-05 \
  --lr-prompt-learner 0.0005 \
  --lr-adapter 0.0005 \
  --weight-decay 0.0005 \
  --temporal-layers 1 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 10 \
  --root-dir /kaggle/input/processed-caer-video-dataset \
  --train-annotation /kaggle/working/test/CAER_Video/train_subset.txt \
  --val-annotation /kaggle/input/processed-caer-video-dataset/validation.txt \
  --test-annotation /kaggle/input/processed-caer-video-dataset/test.txt \
  --clip-path ViT-B/32 \
  --bounding-box-face /kaggle/input/processed-caer-video-dataset/face_boxes_mediapipe.json \
  --bounding-box-body dummy_boxes.json \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --loss-type ldam \
  --ldam-max-m 0.25 \
  --ldam-s 30.0 \
  --lambda_dc 0.1 \
  --mi-warmup 5 \
  --dc-warmup 5 \
  --lambda_mi 0.1 \
  --use-weighted-sampler True \
  --crop-body True \
  --grad-clip 1.0 \
  --mixup-alpha 0.0 \
  --label-smoothing 0.05
