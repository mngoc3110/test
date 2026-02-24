#!/bin/bash
# CAER TRAINING - KAGGLE PATHS + SUBSET (30% Data) - QUICK TEST B/32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFREPORT=file=/dev/null:level=24

# Thiết lập đường dẫn Kaggle (Dựa trên log của bạn)
DATA_ROOT="/kaggle/input/processed-caer-video-dataset"
SUBSET_PATH="./CAER_Video/train_subset.txt"

# Kiểm tra file subset có tồn tại không (do mới pull từ git)
if [ ! -f "$SUBSET_PATH" ]; then
    echo "Warning: Subset file not found at $SUBSET_PATH. Generating it now..."
    python create_subset.py
fi

python main.py \
  --mode train \
  --exper-name CAER-Kaggle-Subset-TestB32 \
  --dataset CAER \
  --gpu 0 \
  --epochs 2 \
  --batch-size 4 \
  --accumulation-steps 8 \
  --optimizer AdamW \
  --scheduler multistep \
  --lr 3e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.05 \
  --milestones 1 2 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "$DATA_ROOT" \
  --train-annotation "$SUBSET_PATH" \
  --val-annotation "$DATA_ROOT/validation.txt" \
  --test-annotation "$DATA_ROOT/test.txt" \
  --clip-path ViT-B/32 \
  --bounding-box-face "$DATA_ROOT/face_boxes_mediapipe.json" \
  --bounding-box-body dummy_boxes.json \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --loss-type ldam \
  --lambda_dc 0.1 \
  --mi-warmup 1 \
  --dc-warmup 1 \
  --lambda_mi 0.1 \
  --use-amp \
  --use-weighted-sampler \
  --crop-body \
  --grad-clip 1.0 \
  --mixup-alpha 0.0 \
  --label-smoothing 0.1 | grep -E "Epoch|UAR|WAR|Loss" --line-buffered
