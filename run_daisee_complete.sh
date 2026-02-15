#!/bin/bash
# DAiSEE Complete Pipeline: Data Prep + Training
# Engagement Only (4 Classes) | Attention Pooling | MI/DC Loss

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 1. CONFIGURATION ---
DATASET_ROOT="./dataset/DAiSEE"
LABELS_DIR="${DATASET_ROOT}/Labels"
VIDEO_DIR="${DATASET_ROOT}/DataSet"

TRAIN_TXT="${LABELS_DIR}/trainlist_generated.txt"
VAL_TXT="${LABELS_DIR}/validationlist_generated.txt"
TEST_TXT="${LABELS_DIR}/testlist_generated.txt"

# --- 2. DATA PREPARATION (CSV -> TXT) ---
echo "=> Generating label files from CSV..."
python3 -c "
import os
import pandas as pd

label_dir = '${LABELS_DIR}'
file_map = {
    'TrainLabels.csv': '${TRAIN_TXT}',
    'ValidationLabels.csv': '${VAL_TXT}',
    'TestLabels.csv': '${TEST_TXT}'
}

for csv_file, txt_file in file_map.items():
    csv_path = os.path.join(label_dir, csv_file)
    if os.path.exists(csv_path):
        print(f'Processing {csv_file}...')
        try:
            df = pd.read_csv(csv_path)
            with open(txt_file, 'w') as f:
                for _, row in df.iterrows():
                    clip_id = str(row['ClipID']).strip()
                    if not clip_id.endswith('.avi') and not clip_id.endswith('.mp4'):
                        clip_id += '.avi'
                    
                    # Get Engagement Label
                    if 'Engagement' in df.columns:
                        label = int(row['Engagement'])
                    else:
                        label = int(row.iloc[2]) # Fallback index
                    
                    # Write: filename num_frames(dummy) label
                    f.write(f'{clip_id} 300 {label}\n')
        except Exception as e:
            print(f'Error processing {csv_file}: {e}')
    else:
        print(f'Warning: {csv_file} not found at {csv_path}')
"

# --- 3. TRAINING COMMAND ---
echo "=> Starting Training..."

python main.py \
  --mode train \
  --exper-name Train-DAiSEE-Engage-AttnPool \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 20 \
  --batch-size 4 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.005 \
  --milestones 10 15 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --temporal-pooling attn_pool \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "${VIDEO_DIR}" \
  --train-annotation "${TRAIN_TXT}" \
  --val-annotation "${VAL_TXT}" \
  --test-annotation "${TEST_TXT}" \
  --bounding-box-face "" \
  --clip-path "ViT-B/16" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --loss-type ldam \
  --ldam-s 30.0 \
  --ldam-max-m 0.5 \
  --lambda_dc 0.1 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --lambda_mi 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --use-amp \
  --use-weighted-sampler \
  --grad-clip 1.0 \
  --mixup-alpha 0.0
