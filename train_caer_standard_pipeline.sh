#!/bin/bash
# CAER TRAINING - SUBSET (30% Data) for Fast Experimentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFREPORT=file=/dev/null:level=24

python main.py \
  --mode train \
  --exper-name CAER-Subset-Experiment \
  --dataset CAER \
  --gpu mps \
  --epochs 30 \
  --batch-size 4 \
  --accumulation-steps 8 \
  --optimizer AdamW \
  --scheduler multistep \
  --lr 3e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.05 \
  --milestones 15 25 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 200 \
  --root-dir ./CAER_Video \
  --train-annotation ./CAER_Video/train_subset.txt \
  --val-annotation ./CAER_Video/validation.txt \
  --test-annotation ./CAER_Video/test.txt \
  --clip-path ViT-B/16 \
  --bounding-box-face ./CAER_Video/face_boxes_mediapipe.json \
  --bounding-box-body dummy_boxes.json \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --loss-type ldam \
  --lambda_dc 0.1 \
  --mi-warmup 5 \
  --dc-warmup 5 \
  --lambda_mi 0.1 \
  --use-amp \
  --use-weighted-sampler \
  --crop-body \
  --grad-clip 1.0 \
  --mixup-alpha 0.0 \
  --label-smoothing 0.1 | grep -E "Epoch|UAR|WAR|Loss" --line-buffered
