#!/bin/bash
# CAER TRAINING - SINGLE FRAME TEST (DISABLE TEMPORAL + CONTEXT)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFREPORT=file=/dev/null:level=24

python main.py \
  --mode train \
  --exper-name CAER-Test-FaceOnly-SingleFrame \
  --dataset CAER \
  --gpu 0 \
  --epochs 10 \
  --batch-size 16 \
  --accumulation-steps 1 \
  --optimizer AdamW \
  --scheduler multistep \
  --lr 1e-4 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 1e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.05 \
  --milestones 5 8 \
  --gamma 0.1 \
  --temporal-layers 0 \
  --num-segments 1 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir /kaggle/input/processed-caer-video-dataset \
  --train-annotation ./CAER_Video/train_subset.txt \
  --val-annotation /kaggle/input/processed-caer-video-dataset/validation.txt \
  --test-annotation /kaggle/input/processed-caer-video-dataset/test.txt \
  --clip-path ViT-B/16 \
  --bounding-box-face /kaggle/input/processed-caer-video-dataset/face_boxes_mediapipe.json \
  --bounding-box-body dummy_boxes.json \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner False \
  --loss-type ce \
  --use-amp \
  --grad-clip 1.0 \
  --mixup-alpha 0.0 \
  --label-smoothing 0.0 | grep -E "Epoch|UAR|WAR|Loss|BBOX" --line-buffered
