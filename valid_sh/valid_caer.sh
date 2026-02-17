#!/bin/bash
# CAER Validation Script
# This script evaluates the best checkpoint from the 'Train-CAER-WAR-Advanced' experiment.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- PATH CONFIGURATION ---
DATASET_ROOT="./dataset/CAER"
ANNOTATION_ROOT="./dataset/CAER/annotations"
BOUNDING_BOX_ROOT="./dataset/CAER/bounding_box"

# Path to the best model checkpoint from your log
CHECKPOINT="outputs/Train-CAER-WAR-Advanced-[02-17]-[01:13]/model_best.pth"

python main.py \
  --mode eval \
  --eval-checkpoint "$CHECKPOINT" \
  --dataset CAER \
  --gpu 0 \
  --batch-size 16 \
  --num-segments 4 \
  --duration 1 \
  --image-size 224 \
  --root-dir "$DATASET_ROOT" \
  --test-annotation "$ANNOTATION_ROOT/test.txt" \
  --bounding-box-face "$BOUNDING_BOX_ROOT/face.json" \
  --bounding-box-body "$BOUNDING_BOX_ROOT/body.json" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --temporal-layers 1 \
  --temporal-pooling attn_pool \
  --crop-body
