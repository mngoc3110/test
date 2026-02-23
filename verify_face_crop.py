import os
import cv2
import torch
import numpy as np
import random
import glob
import json
from PIL import Image
# from dataloader.video_dataloader import VideoDataset # Avoiding full import to keep it simple if possible, but let's stick to it for consistency
from torchvision import transforms

# Configuration
ROOT_DIR = './CAER_Video'
LIST_FILE = './CAER_Video/train.txt'
FACE_BOXES = './CAER_Video/face_boxes_mediapipe.json'
OUTPUT_DIR = 'verification_results'
NUM_SAMPLES = 20

# Minimal mock of VideoDataset to avoid import issues if any, or just import it.
# The original code imported it, so let's keep it.
from dataloader.video_dataloader import VideoDataset

def verify_crops():
    print(f"Loading dataset with boxes from {FACE_BOXES}...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize Dataset
    dataset = VideoDataset(
        root_dir=ROOT_DIR,
        list_file=LIST_FILE,
        num_segments=1, 
        duration=1,
        mode='test', 
        transform=transforms.Compose([]), 
        image_size=224,
        bounding_box_face=FACE_BOXES,
        bounding_box_body='dummy_boxes.json', # Ensure this exists
        crop_body=False
    )
    
    print(f"Dataset loaded with {len(dataset)} videos.")
    print(f"Selecting {NUM_SAMPLES} random samples to verify...")
    
    indices = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))
    
    success_count = 0
    
    for idx in indices:
        try:
            record = dataset.video_list[idx]
            
            # 1. Open Video
            cap = cv2.VideoCapture(record.path)
            if not cap.isOpened():
                print(f"Could not open {record.path}")
                continue
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mid_frame_idx = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if not ret: 
                print(f"Could not read frame {mid_frame_idx} from {record.path}")
                continue
                
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # 2. Get Box (Using Dataset Logic)
            # Re-implementing logic to see exactly what's happening
            rel_path = record.path.replace('./', '').replace('\\', '/')
            # dataset.boxs keys are like 'train/Anger/0001'
            # record.path is like './CAER_Video/train/Anger/0001.avi'
            
            # Normalize path to match keys
            # Key format in JSON: 'train/Anger/0001' (no extension)
            # Video path: 'CAER_Video/train/Anger/0001.avi'
            
            video_key = os.path.relpath(record.path, ROOT_DIR)
            video_key = os.path.splitext(video_key)[0]
            
            frame_key = f"{mid_frame_idx}.jpg"
            
            box = None
            if video_key in dataset.boxs:
                if frame_key in dataset.boxs[video_key]:
                    box = dataset.boxs[video_key][frame_key]
            
            # Fallback check (sometimes keys are simpler)
            if box is None:
                 # Check if key has prefix 'CAER_Video/'
                 pass 

            # 3. Draw Box on Original (Verification)
            img_vis = frame.copy()
            if box is not None:
                x1, y1, x2, y2 = [int(c) for c in box]
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                status = "Matched Box"
            else:
                status = "No Box Found (Will use Auto-Detect)"
                # Let's see what auto-detect does
                # dataset._face_detect would handle it
            
            # 4. Save
            # Resize for viewing
            h, w = img_vis.shape[:2]
            if w > 800:
                scale = 800 / w
                img_vis = cv2.resize(img_vis, (0,0), fx=scale, fy=scale)
                
            label_map = {1:'Anger', 2:'Disgust', 3:'Fear', 4:'Happy', 5:'Neutral', 6:'Sad', 7:'Surprise'}
            text = f"{label_map.get(record.label, 'Unknown')} | {status}"
            cv2.putText(img_vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img_vis, video_key, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            out_path = os.path.join(OUTPUT_DIR, f"verify_{idx}.jpg")
            cv2.imwrite(out_path, img_vis)
            print(f"Saved {out_path} [{status}]")
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {idx}: {e}")

    print(f"Verification complete. {success_count}/{NUM_SAMPLES} images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    verify_crops()
