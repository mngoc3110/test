#!/bin/bash
# CAER Training Script - SOTA Configuration (Aiming for 80% Accuracy)
# Strategy: Combine Train+Val -> Train Full | Validate on Test | Mixup + More Frames

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- PATH CONFIGURATION ---
if [ -d "/kaggle/input" ]; then
    echo "=> Detected Kaggle Environment."
    DATASET_ROOT="/kaggle/input/caer-video-dataset/CAER"
    ANNOTATION_ROOT="/kaggle/input/caer-annotations"
    # Create Full Train list (Train + Val) in a writable directory
    cat "$ANNOTATION_ROOT/train.txt" "$ANNOTATION_ROOT/validation.txt" > caer_train_full.txt
    TRAIN_LIST="caer_train_full.txt"
    VAL_LIST="$ANNOTATION_ROOT/test.txt"
    TEST_LIST="$ANNOTATION_ROOT/test.txt"
    BOUNDING_BOX_FACE="/kaggle/input/caer-bounding-box/face.json"
    BOUNDING_BOX_BODY="/kaggle/input/caer-bounding-box/body.json"
else
    echo "=> Detected Local Environment."
    DATASET_ROOT="./dataset/CAER"
    ANNOTATION_ROOT="./dataset/CAER/annotations"
    # Create Full Train list
    cat "$ANNOTATION_ROOT/train.txt" "$ANNOTATION_ROOT/validation.txt" > "$ANNOTATION_ROOT/caer_train_full.txt"
    TRAIN_LIST="$ANNOTATION_ROOT/caer_train_full.txt"
    VAL_LIST="$ANNOTATION_ROOT/test.txt"
    TEST_LIST="$ANNOTATION_ROOT/test.txt"
    BOUNDING_BOX_FACE="./dataset/CAER/bounding_box/face.json"
    BOUNDING_BOX_BODY="./dataset/CAER/bounding_box/body.json"
fi

echo "Using Full Training List: $TRAIN_LIST"

python main.py \
  --mode train \
  --exper-name Train-CAER-SOTA-MaxACC \
  --dataset CAER \
  --gpu 0 \
  --epochs 40 \
  --batch-size 8 \
  --optimizer AdamW \
  --lr 1e-4 \
  --lr-image-encoder 5e-6 \
  --lr-prompt-learner 5e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.02 \
  --milestones 20 30 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 8 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "$DATASET_ROOT" \
  --train-annotation "$TRAIN_LIST" \
  --val-annotation "$VAL_LIST" \
  --test-annotation "$TEST_LIST" \
  --clip-path ViT-B/16 \
  --bounding-box-face "$BOUNDING_BOX_FACE" \
  --bounding-box-body "$BOUNDING_BOX_BODY" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --loss-type ldam \
  --ldam-s 30.0 \
  --ldam-max-m 0.35 \
  --lambda_dc 0.1 \
  --mi-warmup 5 \
  --dc-warmup 5 \
  --lambda_mi 0.1 \
  --use-amp \
  --use-weighted-sampler \
  --crop-body \
  --grad-clip 1.0 \
  --mixup-alpha 0.2