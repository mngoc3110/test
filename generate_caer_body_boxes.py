import os
import cv2
import json
import torch
import glob
from tqdm import tqdm
from ultralytics import YOLO
import argparse

def generate_body_boxes(root_dir, output_file, model_type='yolov8n.pt', step=1):
    """
    Generates body bounding boxes for CAER dataset using YOLOv8. 
    
    Args:
        root_dir (str): Root directory of CAER_Video (containing train/test/val).
        output_file (str): Path to save the JSON output.
        model_type (str): YOLO model type (yolov8n.pt, yolov8s.pt, etc.).
        step (int): Frame stride (process every step-th frame). 1 = all frames.
    """
    
    print(f"Loading YOLO model: {model_type}...")
    model = YOLO(model_type)
    
    # Check if we can resume
    data = {}
    if os.path.exists(output_file):
        print(f"Found existing file {output_file}. Loading to resume...")
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} processed videos.")
        except:
            print("Could not load existing file. Starting fresh.")
    
    # Collect all video files
    print(f"Scanning videos in {root_dir}...")
    video_extensions = ['*.avi', '*.mp4']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
    
    video_files.sort()
    print(f"Found {len(video_files)} videos.")
    
    # Filter out already processed videos
    # Key format in JSON: 'train/Anger/0001' (relative path without extension)
    processed_keys = set(data.keys())
    videos_to_process = []
    
    for vid_path in video_files:
        rel_path = os.path.relpath(vid_path, root_dir)
        key = os.path.splitext(rel_path)[0]
        # Normalize slashes
        key = key.replace('\\', '/')
        
        if key not in processed_keys:
            videos_to_process.append(vid_path)
            
    print(f"Videos remaining to process: {len(videos_to_process)}")
    
    # Processing loop
    try:
        for vid_idx, vid_path in enumerate(tqdm(videos_to_process, desc="Detecting Bodies")):
            rel_path = os.path.relpath(vid_path, root_dir)
            key = os.path.splitext(rel_path)[0].replace('\\', '/')
            
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                print(f"Error opening {vid_path}")
                continue
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_boxes = {}
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every step-th frame to save time (if configured)
                if frame_idx % step == 0:
                    # Run YOLO inference
                    # classes=0 means only detect 'person'
                    results = model(frame, verbose=False, classes=0)
                    
                    best_box = None
                    max_area = 0
                    
                    # Iterate results
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            # Box format: [x1, y1, x2, y2]
                            b = box.xyxy[0].cpu().numpy().tolist()
                            conf = box.conf[0].item()
                            
                            if conf < 0.3: # Filter low confidence
                                continue
                                
                            x1, y1, x2, y2 = map(int, b)
                            w = x2 - x1
                            h = y2 - y1
                            area = w * h
                            
                            # Heuristic: Largest person is likely the subject
                            if area > max_area:
                                max_area = area
                                best_box = [x1, y1, x2, y2]
                    
                    if best_box:
                        # Save frame key as "frame_idx.jpg" to match VideoDataset expectation
                        # Or just integer frame_idx if your dataloader handles it.
                        # Looking at VideoDataset._face_detect/get:
                        # frame_key = f"{p}.jpg"
                        frame_key = f"{frame_idx}.jpg"
                        video_boxes[frame_key] = best_box
                
                frame_idx += 1
            
            cap.release()
            
            # Store results
            if video_boxes:
                data[key] = video_boxes
            
            # Save periodically (every 50 videos)
            if (vid_idx + 1) % 50 == 0:
                with open(output_file, 'w') as f:
                    json.dump(data, f)
                    
    except KeyboardInterrupt:
        print("\nInterrupted! Saving current progress...")
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(data, f)
    print(f"Done! Saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', default='./CAER_Video', help='Path to CAER video root')
    parser.add_argument('--output', default='./CAER_Video/body_boxes.json', help='Output JSON path')
    parser.add_argument('--step', type=int, default=1, help='Frame stride (1=every frame)')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model version')
    
    args = parser.parse_args()
    
    generate_body_boxes(args.root_dir, args.output, args.model, args.step)
