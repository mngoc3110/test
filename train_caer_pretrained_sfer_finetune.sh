#!/bin/bash
# CAER RESEARCH PIPELINE - PRETRAINED SFER FINE-TUNING
# Mục tiêu: Fine-tune SFER model hiệu quả trên CAER bằng CE loss và đúng temp scale.

# Trỏ đường dẫn đến file pretrained (Thay đổi nếu cần)
PRETRAINED_PATH="/kaggle/input/datasets/bearmn/pretrain-fer6/Train-SFER-Stable-[02-10]-[11:46]/model_best.pth"

python main.py \
  --mode train \
  --dataset CAER \
  --exper-name CAER-Pretrained-SFER-FineTune \
  --gpu 0 \
  --workers 4 \
  --epochs 20 \
  --batch-size 8 \
  --accumulation-steps 2 \
  --optimizer AdamW \
  --scheduler cosine \
  --lr 5e-05 \
  --lr-image-encoder 1e-05 \
  --lr-prompt-learner 5e-05 \
  --lr-adapter 5e-05 \
  --weight-decay 0.0005 \
  --num-segments 8 \
  --temperature 0.07 \
  --loss-type ce \
  --use-weighted-sampler False \
  --lambda_mi 0.0 \
  --lambda_dc 0.0 \
  --label-smoothing 0.1 \
  --root-dir /kaggle/input/processed-caer-video-dataset \
  --train-annotation /kaggle/working/test/CAER_Video/train_subset.txt \
  --val-annotation /kaggle/input/processed-caer-video-dataset/validation.txt \
  --clip-path ViT-B/16 \
  --bounding-box-face /kaggle/input/processed-caer-video-dataset/face_boxes_mediapipe.json \
  --resume $PRETRAINED_PATH
