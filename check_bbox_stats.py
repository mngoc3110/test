import os
import torch
import random
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
from dataloader.video_dataloader import VideoDataset
from dataloader.video_transform import GroupResize, Stack, ToTorchFormatTensor

# Setup arguments
parser = argparse.ArgumentParser(description='Check BBox Stats and Temporal Sampling')
parser.add_argument('--root-dir', type=str, default='./CAER_Video')
parser.add_argument('--annotation-file', type=str, default='./CAER_Video/train.txt')
parser.add_argument('--bounding-box-face', type=str, default='./CAER_Video/face_boxes_mediapipe.json')
parser.add_argument('--num-segments', type=int, default=8)
parser.add_argument('--duration', type=int, default=1)
args = parser.parse_args()

# Setup transform (simplified for visual check)
transform = transforms.Compose([
    GroupResize(224),
    Stack(),
    ToTorchFormatTensor()
])

# Initialize Dataset
print(f"Initializing VideoDataset from {args.root_dir}...")
dataset = VideoDataset(
    root_dir=args.root_dir,
    list_file=args.annotation_file,
    num_segments=args.num_segments,
    duration=args.duration,
    mode='train',
    transform=transform,
    image_size=224,
    bounding_box_face=args.bounding_box_face,
    bounding_box_body=None, # We want to check what happens without body box first
    crop_body=False,
    num_classes=7
)

print(f"Dataset length: {len(dataset)}")

# Create output directory
output_dir = "debug_bbox_check"
os.makedirs(output_dir, exist_ok=True)

# Sample 5 random indices
indices = random.sample(range(len(dataset)), 5)

print("\n--- Temporal Sampling Check ---")
for idx in indices:
    record = dataset.video_list[idx]
    # Access private method to check sampling logic
    segment_indices = dataset._get_train_indices(record)
    print(f"Video: {record.path}")
    print(f"  Total Frames: {record.num_frames}")
    print(f"  Sampled Indices: {segment_indices}")
    
    # Get items to visualize crops
    # We manually call get() to intercept the PIL images before transform if possible, 
    # but since dataset.get() returns transformed tensors, we will visualize the tensors.
    
    # Get the raw images first (simulating what get() does)
    images_face_tensor, images_body_tensor, label = dataset[idx]
    
    # Save face crops
    # Tensor is (T*C, H, W) or (T, C, H, W) depending on view. 
    # VideoDataset returns (T, C, H, W) for face and body.
    
    # Face
    T = images_face_tensor.size(0)
    for t in range(T):
        img_tensor = images_face_tensor[t]
        # Un-normalize not needed as ToTorchFormatTensor scales to [0, 1] but doesn't normalize with mean/std
        # It actually subtracts mean and divides by std in video_transform.py. Let's check video_transform.py later.
        # Assuming ToTorchFormatTensor just divides by 255.
        
        # Actually ToTorchFormatTensor does: div(255) then sub(mean).div(std).
        # We need to reverse this for visualization.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        save_path = os.path.join(output_dir, f"vid_{idx}_frame_{t}_face.png")
        torchvision.utils.save_image(img_tensor, save_path)

    # Body (Context)
    # Since we passed bounding_box_body=None, we expect full image or center crop
    for t in range(T):
        img_tensor = images_body_tensor[t]
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        save_path = os.path.join(output_dir, f"vid_{idx}_frame_{t}_body.png")
        torchvision.utils.save_image(img_tensor, save_path)

print(f"\nSaved debug images to {output_dir}")