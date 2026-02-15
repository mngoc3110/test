import os
import pandas as pd
import glob

# Cấu hình đường dẫn (trên Colab)
DATASET_ROOT = './dataset/DAiSEE/DataSet'
LABELS_DIR = './dataset/DAiSEE/Labels'
OUTPUT_DIR = './dataset/DAiSEE/annotations'

# Map tên file CSV sang file TXT
file_map = {
    'TrainLabels.csv': 'train.txt',
    'ValidationLabels.csv': 'validation.txt',
    'TestLabels.csv': 'test.txt'
}

def build_video_index(root_dir):
    """Quét toàn bộ video và tạo index: ClipID -> Full Path"""
    print(f"Scanning videos in {root_dir}...")
    video_index = {}
    # Tìm tất cả file .avi và .mp4
    files = glob.glob(os.path.join(root_dir, '**', '*.*'), recursive=True)
    
    for f in files:
        if f.lower().endswith(('.avi', '.mp4', '.mov')):
            # ClipID thường là tên file (bỏ đuôi)
            filename = os.path.basename(f)
            clip_id = os.path.splitext(filename)[0]
            video_index[clip_id] = f
            
    print(f"Found {len(video_index)} videos.")
    return video_index

def fix_annotations():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Quét video trước
    video_index = build_video_index(DATASET_ROOT)
    
    # 2. Duyệt qua từng file CSV và map
    for csv_file, txt_file in file_map.items():
        csv_path = os.path.join(LABELS_DIR, csv_file)
        txt_path = os.path.join(OUTPUT_DIR, txt_file)
        
        if not os.path.exists(csv_path):
            print(f"Warning: CSV {csv_path} not found.")
            continue
            
        print(f"Processing {csv_file}...")
        df = pd.read_csv(csv_path)
        
        found_count = 0
        missing_count = 0
        
        with open(txt_path, 'w') as f:
            for _, row in df.iterrows():
                clip_id = str(row['ClipID']).strip()
                # Xử lý trường hợp ClipID trong CSV có đuôi .avi
                if clip_id.endswith('.avi'):
                    clip_id = clip_id[:-4]
                
                # Tìm đường dẫn thực tế
                if clip_id in video_index:
                    real_path = video_index[clip_id]
                    
                    # Lấy nhãn Engagement
                    if 'Engagement' in df.columns:
                        label = int(row['Engagement'])
                    else:
                        label = int(row.iloc[2])
                    
                    # Ghi file: path 300 label
                    f.write(f"{real_path} 300 {label}\n")
                    found_count += 1
                else:
                    missing_count += 1
                    # print(f"Missing video for ClipID: {clip_id}")
        
        print(f"-> Saved {txt_path}: Found {found_count}, Missing {missing_count}")

if __name__ == "__main__":
    fix_annotations()
