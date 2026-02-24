#!/bin/bash
# CAER RESEARCH PIPELINE - PRETRAINED SFER
# Mục tiêu: Đạt SOTA UAR trên CAER bằng cách mượn kiến thức từ SFER

# Trỏ đường dẫn đến file pretrained (Thay đổi nếu cần)
PRETRAINED_PATH="outputs/Train-SFER-Stable-[02-10]-[11:46]/model_best.pth"

python main.py \
  --mode train \
  --dataset CAER \
  --exper-name CAER-Pretrained-SFER \
  --gpu 0 \
  --workers 4 \
  --epochs 20 \
  --batch-size 8 \
  --accumulation-steps 2 \
  --optimizer AdamW \
  --scheduler cosine \
  --lr 2e-05 \
  --lr-image-encoder 5e-06 \
  --lr-prompt-learner 1e-04 \
  --lr-adapter 1e-04 \
  --weight-decay 0.0005 \
  --num-segments 8 \
  --temperature 1.0 \
  --loss-type ldam \
  --ldam-max-m 0.2 \
  --ldam-s 30.0 \
  --use-weighted-sampler True \
  --lambda_mi 0.05 \
  --mi-warmup 0 \
  --label-smoothing 0 \
  --root-dir /kaggle/input/processed-caer-video-dataset \
  --train-annotation /kaggle/working/test/CAER_Video/train_subset.txt \
  --val-annotation /kaggle/input/processed-caer-video-dataset/validation.txt \
  --clip-path ViT-B/32 \
  --bounding-box-face /kaggle/input/processed-caer-video-dataset/face_boxes_mediapipe.json \
  --resume $PRETRAINED_PATH
