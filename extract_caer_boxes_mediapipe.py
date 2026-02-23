import os
import glob
import json
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Configuration
ROOT_DIR = './CAER_Video'
OUTPUT_FILE = './CAER_Video/face_boxes_mediapipe.json'

def get_center_weighted_face(detections, img_w, img_h):
    """Helper to select best face from MediaPipe detections."""
    img_cx, img_cy = img_w / 2, img_h / 2
    best_box = None
    max_score = -float('inf')
    
    for detection in detections:
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * img_w)
        y = int(bboxC.ymin * img_h)
        w = int(bboxC.width * img_w)
        h = int(bboxC.height * img_h)
        
        x = max(0, x)
        y = max(0, y)
        w = min(img_w - x, w)
        h = min(img_h - y, h)
        
        if w <= 0 or h <= 0: continue

        area = w * h
        face_cx = x + (w / 2)
        face_cy = y + (h / 2)
        
        dist = ((face_cx - img_cx)**2 + (face_cy - img_cy)**2)**0.5
        score = area - (dist * 15)
        
        if score > max_score:
            max_score = score
            best_box = [x, y, x+w, y+h]
            
    return best_box

def extract_boxes():
    print("Initializing MediaPipe Face Detection...")
    mp_face_detection = mp.solutions.face_detection
    
    all_boxes = {}
    
    # Check if we can resume (optional, but good for safety)
    if os.path.exists(OUTPUT_FILE):
        try:
            print(f"Found existing file {OUTPUT_FILE}, loading to resume/append...")
            with open(OUTPUT_FILE, 'r') as f:
                all_boxes = json.load(f)
        except:
            print("Existing file corrupted or empty, starting fresh.")
            all_boxes = {}

    search_pattern = os.path.join(ROOT_DIR, '*/*/*.avi')
    video_paths = glob.glob(search_pattern)
    video_paths.sort()
    
    print(f"Found {len(video_paths)} videos. Starting extraction...")
    
    # Use context manager for MediaPipe
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        for video_idx, video_path in enumerate(tqdm(video_paths)):
            
            # Generate key
            rel_path = os.path.relpath(video_path, ROOT_DIR)
            video_key = os.path.splitext(rel_path)[0]
            video_key = video_key.replace('\\', '/')
            
            # Skip if already processed (and valid)
            if video_key in all_boxes and len(all_boxes[video_key]) > 0:
                continue
            
            all_boxes[video_key] = {}
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            prev_box = None
            
            for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, frame = cap.read()
                if not ret: break
                
                # Convert to RGB for MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image_rgb)
                
                if results.detections:
                    # Get best face
                    detections = results.detections
                    img_h, img_w = frame.shape[:2]
                    
                    # Logic for best face selection
                    img_cx, img_cy = img_w / 2, img_h / 2
                    best_box = None
                    max_score = -float('inf')
                    
                    for detection in detections:
                        bboxC = detection.location_data.relative_bounding_box
                        x = int(bboxC.xmin * img_w)
                        y = int(bboxC.ymin * img_h)
                        w = int(bboxC.width * img_w)
                        h = int(bboxC.height * img_h)
                        
                        x = max(0, x)
                        y = max(0, y)
                        w = min(img_w - x, w)
                        h = min(img_h - y, h)
                        
                        if w <= 0 or h <= 0: continue

                        area = w * h
                        face_cx = x + (w / 2)
                        face_cy = y + (h / 2)
                        
                        dist = ((face_cx - img_cx)**2 + (face_cy - img_cy)**2)**0.5
                        score = area - (dist * 15)
                        
                        if score > max_score:
                            max_score = score
                            best_box = [x, y, x+w, y+h]
                    
                    current_box = best_box
                else:
                    current_box = None

                # Tracking / Smoothing Strategy
                if current_box is None:
                    if prev_box is not None:
                        current_box = prev_box # Holding (use previous box if lost)
                else:
                    if prev_box is not None:
                        # Smoothing
                        curr_cx = (current_box[0] + current_box[2]) / 2
                        prev_cx = (prev_box[0] + prev_box[2]) / 2
                        
                        # Only smooth if the jump is small (avoid smoothing scene cuts)
                        if abs(curr_cx - prev_cx) < frame.shape[1] * 0.3:
                            alpha = 0.7
                            current_box = [
                                int(alpha * current_box[0] + (1-alpha) * prev_box[0]),
                                int(alpha * current_box[1] + (1-alpha) * prev_box[1]),
                                int(alpha * current_box[2] + (1-alpha) * prev_box[2]),
                                int(alpha * current_box[3] + (1-alpha) * prev_box[3]),
                            ]
                    prev_box = current_box
                
                # Save box if valid
                if current_box is not None:
                    frame_key = f"{frame_idx}.jpg"
                    if video_key not in all_boxes:
                         all_boxes[video_key] = {}
                    all_boxes[video_key][frame_key] = current_box
            
            cap.release()
            
            # Save every 200 videos
            if (video_idx + 1) % 200 == 0:
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(all_boxes, f)
    
    # Final Save
    print(f"Saving final results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_boxes, f)
    print("Done!")

if __name__ == "__main__":
    extract_boxes()
