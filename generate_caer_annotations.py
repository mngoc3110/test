import os
import cv2
import glob

# Map Class -> Index (0-6)
class_map = {
    'Anger': 0,
    'Disgust': 1,
    'Fear': 2,
    'Happy': 3,
    'Neutral': 4,
    'Sad': 5,
    'Surprise': 6
}

# --- CẤU HÌNH ĐƯỜNG DẪN TRÊN KAGGLE ---
# Nơi chứa Video gốc (Read-only)
DATASET_ROOT = '/kaggle/input/caer-video-dataset/CAER' 
TRAIN_DIR = os.path.join(DATASET_ROOT, 'train')
TEST_DIR = os.path.join(DATASET_ROOT, 'test')

# Nơi lưu file list (Writeable)
OUTPUT_DIR = '/kaggle/working'

def generate_list(data_dir, output_file):
    print(f"Generating list for {data_dir} -> {output_file}...")
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found!")
        return

    with open(output_file, 'w') as f:
        for class_name, label in class_map.items():
            class_dir = os.path.join(data_dir, class_name)
            
            # Lấy tất cả video
            videos = glob.glob(os.path.join(class_dir, '*'))
            
            count = 0
            for video_path in videos:
                if not video_path.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                    continue
                
                # Mở video để đếm frame
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    continue
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # Ghi đường dẫn TUYỆT ĐỐI để dataloader đọc được từ bất kỳ đâu
                f.write(f"{video_path} {num_frames} {label}\n")
                count += 1
            
            print(f"  Processed class {class_name}: {count} videos")

# Tạo annotation file
os.makedirs(OUTPUT_DIR, exist_ok=True)

generate_list(TRAIN_DIR, os.path.join(OUTPUT_DIR, 'train_caer.txt'))
generate_list(TEST_DIR, os.path.join(OUTPUT_DIR, 'test_caer.txt'))

print(f"Done! Annotation files generated in {OUTPUT_DIR}")