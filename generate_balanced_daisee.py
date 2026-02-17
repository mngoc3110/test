import os
import pandas as pd
import glob

# Cấu hình đường dẫn
DATASET_ROOT = './dataset/DAiSEE/DataSet'
LABELS_DIR = './dataset/DAiSEE/Labels'

# Map tên file CSV -> file TXT mới
file_map = {
    'TrainLabels_Balanced.csv': 'train_balanced.txt',
    'ValidationLabels_Balanced.csv': 'val_balanced.txt',
    'TestLabels_Balanced.csv': 'test_balanced.txt'
}

def build_video_index(root_dir):
    """Quét toàn bộ video và tạo index: ClipID -> Full Path"""
    print(f"Scanning videos in {root_dir}...")
    video_index = {}
    # Tìm tất cả file .avi, .mp4, .mov
    files = glob.glob(os.path.join(root_dir, '**', '*.*'), recursive=True)
    
    for f in files:
        if f.lower().endswith(('.avi', '.mp4', '.mov')):
            filename = os.path.basename(f)
            clip_id = os.path.splitext(filename)[0]
            video_index[clip_id] = f
            
    print(f"Found {len(video_index)} videos.")
    return video_index

def generate():
    # 1. Quét video
    video_index = build_video_index(DATASET_ROOT)
    
    # 2. Duyệt qua từng file CSV
    for csv_file, txt_file in file_map.items():
        csv_path = os.path.join(LABELS_DIR, csv_file)
        txt_path = os.path.join(LABELS_DIR, txt_file)
        
        if not os.path.exists(csv_path):
            print(f"Warning: CSV {csv_file} not found in {LABELS_DIR}.")
            continue
            
        print(f"Processing {csv_file}...")
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        found_count = 0
        missing_count = 0
        
        with open(txt_path, 'w') as f:
            for _, row in df.iterrows():
                clip_id = str(row['ClipID']).strip()
                if clip_id.endswith('.avi'):
                    clip_id = clip_id[:-4]
                
                if clip_id in video_index:
                    real_path = video_index[clip_id]
                    # Engagement là cột thứ 3 (index 2)
                    label = int(row['Engagement'])
                    f.write(f"{real_path} 300 {label}\n")
                    found_count += 1
                else:
                    missing_count += 1
        
        print(f"-> Created {txt_path}: Found {found_count}, Missing {missing_count}")

if __name__ == "__main__":
    generate()
