import os
import glob
import json
import random
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torch.utils import data
from numpy.random import randint
from dataloader.video_transform import *
from dataloader.daisee_dataloader import daisee_train_data_loader, daisee_test_data_loader

def generate_caer_list(root_dir, output_file, mode):
    """
    Generates a list file for CAER dataset by scanning the directory structure.
    Structure: root_dir/mode/Class/video.avi
    Output format: relative_path num_frames label
    """
    if os.path.exists(output_file):
        print(f"List file {output_file} already exists. Skipping generation.")
        return

    print(f"Generating CAER list file: {output_file} from {os.path.join(root_dir, mode)}...")
    
    # Class mapping based on models/Text.py order:
    # ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    class_map = {
        'Anger': 1,
        'Disgust': 2,
        'Fear': 3,
        'Happy': 4,
        'Neutral': 5,
        'Sad': 6,
        'Surprise': 7
    }
    
    lines = []
    mode_dir = os.path.join(root_dir, mode)
    if not os.path.isdir(mode_dir):
        print(f"Error: Directory {mode_dir} not found. Cannot generate list.")
        return

    for class_name, label in class_map.items():
        class_dir = os.path.join(mode_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        videos = glob.glob(os.path.join(class_dir, '*.avi')) + glob.glob(os.path.join(class_dir, '*.mp4'))
        videos.sort()
        
        for video_path in videos:
            # Get relative path for the list file (relative to root_dir)
            # The VideoDataset joins root_dir + path.
            # Assuming root_dir is the dataset root (e.g., CAER_Video)
            rel_path = os.path.relpath(video_path, root_dir)
            
            # Count frames
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open {video_path}")
                continue
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if num_frames > 0:
                lines.append(f"{rel_path} {num_frames} {label}\n")

    # Ensure the directory for the output file exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.writelines(lines)
    print(f"Generated {len(lines)} entries in {output_file}")

# Custom Transform for List of Images (Group Transform)
class GroupRandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img_group):
        if random.random() < self.p:
            # Convert to Grayscale (L) then back to RGB to keep 3 channels
            return [img.convert('L').convert('RGB') for img in img_group]
        return img_group

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self): # 路径
        return self._data[0]

    @property       # 帧数
    def num_frames(self):
        return int(self._data[1])

    @property       # 标签
    def label(self):
        return int(self._data[2])

class VideoDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size,bounding_box_face,bounding_box_body, crop_body=False, root_dir="", num_classes=8):
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.bounding_box_face = bounding_box_face
        self.bounding_box_body = bounding_box_body
        self.crop_body = crop_body
        self.root_dir = root_dir
        
        # Bbox hit stats
        self.stats = {'face_box_hit': 0, 'haar_hit': 0, 'center_crop': 0, 'body_box_hit': 0, 'total': 0}
        
        # Initialize OpenCV Face Detector (Haar Cascade)
        # This is a fallback for datasets like CAER which don't have bounding boxes provided.
        # We store the path and load lazily to avoid pickling errors with DataLoader workers.
        self.cascade_path = 'haarcascade_frontalface_alt.xml'
        self.face_cascade = None 
        if not os.path.exists(self.cascade_path):
             print(f"Warning: Face detector {self.cascade_path} not found. Will use full image if boxes are missing.")

        # Debugging: Initialize for saving sample images
        self.debug_samples_path = 'debug_samples'
        os.makedirs(self.debug_samples_path, exist_ok=True)
        self._saved_samples = {i: 0 for i in range(num_classes)}
        
        self._read_sample()
        self._parse_list()
        self._read_boxs()
        if self.crop_body: # Only read body boxes if cropping is enabled
            self._read_body_boxes()

    def _read_boxs(self):
        with open(self.bounding_box_face, 'r') as f:
            self.boxs = json.load(f)


    
    def _read_body_boxes(self):
        if self.bounding_box_body:
            with open(self.bounding_box_body, 'r') as f:
                self.body_boxes = json.load(f)


    def _cv2pil(self,im_cv):
        cv_img_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        pillow_img = Image.fromarray(cv_img_rgb.astype('uint8'))
        return pillow_img

    def _pil2cv(self,im_pil):
        cv_img_rgb = np.array(im_pil)
        cv_img_bgr = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR)
        return cv_img_bgr

    def _resize_image(self,im, width, height):
        w, h = im.shape[1], im.shape[0]
        r = min(width / w, height / h)
        new_w, new_h = int(w * r), int(h * r)
        im = cv2.resize(im, (new_w, new_h))
        pw = (width - new_w) // 2
        ph = (height - new_h) // 2
        top, bottom = ph, ph
        left, right = pw, pw
        if top + bottom + new_h < height:
            bottom += 1
        if left + right + new_w < width:
            right += 1
        im = cv2.copyMakeBorder(im, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return im, r

    def _face_detect(self,img,box,margin,mode = 'face'):
        if box is None:
            if mode == 'face':
                # Try to detect face using OpenCV if available
                # Lazy loading to avoid pickle error
                if self.face_cascade is None and os.path.exists(self.cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
                
                if self.face_cascade is not None:
                    # Convert PIL to CV2 BGR (grayscale for detection)
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) > 0:
                        self.stats['haar_hit'] += 1
                        # Strategy: Center-Weighted Selection
                        # Prioritize faces that are Large AND Central.
                        # Score = Area - (Penalty * Distance_to_Center)
                        
                        img_cx, img_cy = img.width / 2, img.height / 2
                        best_face = faces[0]
                        max_score = -float('inf')
                        
                        for (x, y, w, h) in faces:
                            area = w * h
                            face_cx = x + (w / 2)
                            face_cy = y + (h / 2)
                            
                            # Euclidean distance to image center
                            dist = ((face_cx - img_cx)**2 + (face_cy - img_cy)**2)**0.5
                            
                            # Heuristic: Penalty factor. 
                            # If a face is 100px away from center, penalty is significant.
                            # We want to favor a slightly smaller face at center over a huge face at edge.
                            score = area - (dist * 20) # Weight 20 is empirical
                            
                            if score > max_score:
                                max_score = score
                                best_face = (x, y, w, h)
                        
                        (x, y, w, h) = best_face
                        # Apply margin
                        left = max(0, x - margin)
                        upper = max(0, y - margin)
                        right = min(img.width, x + w + margin)
                        lower = min(img.height, y + h + margin)
                        
                        return img.crop((left, upper, right, lower))
                
                # FALLBACK: Return center crop instead of full image if no face detected
                self.stats['center_crop'] += 1
                w, h = img.size
                s = int(min(w, h) * 0.6)
                return img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
            
            self.stats['center_crop'] += 1
            w, h = img.size
            s = int(min(w, h) * 0.6)
            return img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
        else:
            if mode == 'face':
                self.stats['face_box_hit'] += 1
            left, upper, right, lower = box
            left = int(left)
            upper = int(upper)
            right = int(right)
            lower = int(lower)

            # Heuristic: if right < left, assume it's xywh and convert to xyxy
            if right < left and right > 0:
                right = left + right
            if lower < upper and lower > 0:
                lower = upper + lower

            left = max(0, left - margin)
            upper = max(0, upper - margin)
            right = min(img.width, right + margin)
            lower = min(img.height, lower + margin)

            # Safety check to prevent crash
            if right <= left or lower <= upper:
                return img

            if mode == 'face':
                img = img.crop((left, upper, right, lower))
                return img
            elif mode == 'body':
                occluded_image = img.copy()
                draw = ImageDraw.Draw(occluded_image)
                draw.rectangle([left, upper, right, lower], fill=(0, 0, 0))
                return occluded_image
    
    def _read_sample(self):
        # tmp = [x.strip().split(' ') for x in open(self.list_file)]
        # self.sample_list = [item for item in tmp]
        
        self.sample_list = []
        with open(self.list_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) > 3:
                    # Path contains spaces, join all parts except the last two
                    path = ' '.join(parts[:-2])
                    num_frames = parts[-2]
                    label = parts[-1]
                    self.sample_list.append([path, num_frames, label])
                else:
                    self.sample_list.append(parts)


    def _parse_list(self):
        # 
        # Data Form: [video_id, num_frames, class_idx]
        # 
        self.video_list = [VideoRecord([os.path.join(self.root_dir, item[0])] + item[1:]) for item in self.sample_list]
        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record):
        # 
        # Split all frames into seg parts, then select frame in each part randomly
        # 
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def _get_test_indices(self, record):
        # 
        # Split all frames into seg parts, then select frame in the mid of each part
        # 
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        # Check if record.path is a directory (frames) or a file (video)
        if os.path.isdir(record.path):
            video_frames_path = glob.glob(os.path.join(record.path, '*'))
            video_frames_path.sort()
            num_real_frames = len(video_frames_path)
            is_video_file = False
        else:
            # Assume it's a video file
            is_video_file = True
            cap = cv2.VideoCapture(record.path)
            num_real_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Don't release cap yet, we need it to read frames
            if not cap.isOpened():
                 print(f"Warning: Could not open video file {record.path}, returning zeros.")
                 num_real_frames = 0

        if num_real_frames == 0:
            print(f"Warning: No frames found for video {record.path}, returning zeros.")
            dummy_shape = (self.num_segments * self.duration, 3, self.image_size, self.image_size)
            if is_video_file and 'cap' in locals(): cap.release()
            return torch.zeros(dummy_shape), torch.zeros(dummy_shape), record.label - 1

        # Clamp indices to be valid
        indices = np.clip(indices, 0, num_real_frames - 1)
        
        frames_dict = {}
        if is_video_file:
            wanted_indices = set(indices.tolist())
            max_wanted = max(wanted_indices) if wanted_indices else 0
            for idx in range(max_wanted + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                if idx in wanted_indices:
                    img_cv_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_dict[idx] = Image.fromarray(img_cv_rgb)
        
        images = list()
        images_face = list()
        
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                self.stats['total'] += 1
                if self.stats['total'] % 1000 == 0:
                    total = self.stats['total']
                    face_box = self.stats['face_box_hit']
                    haar = self.stats['haar_hit']
                    center = self.stats['center_crop']
                    body_box_count = self.stats['body_box_hit']
                    print(f"\n[BBOX STATS] Total: {total} | Face: {face_box} ({face_box/total*100:.1f}%) | Body: {body_box_count} ({body_box_count/total*100:.1f}%) | Haar: {haar} | Center: {center}")

                img_pil = None
                box = None
                
                # 1. Read Image
                if is_video_file:
                    if p in frames_dict:
                        img_pil = frames_dict[p].copy()
                    else:
                        if frames_dict:
                            nearest_p = min(frames_dict.keys(), key=lambda k: abs(k - p))
                            img_pil = frames_dict[nearest_p].copy()
                        else:
                            img_pil = Image.new('RGB', (self.image_size, self.image_size))
                else:
                    img_path = video_frames_path[p]
                    try:
                        img_pil = Image.open(img_path).convert('RGB')
                    except:
                        img_pil = Image.new('RGB', (self.image_size, self.image_size))

                # 2. Key Lookup Strategy for Bounding Box
                # Construct possible keys to look up in the JSON
                # Priority 1: Full relative path from dataset root (e.g., 'RAER/train/Neutral/001')
                # Priority 2: Parent dir + Filename (e.g., 'Neutral/001')
                
                # Normalize path separators to forward slash
                rel_path = record.path.replace('./', '').replace('\\', '/')
                # Remove extension
                video_key_full = os.path.splitext(rel_path)[0]
                
                frame_key = f"{p}.jpg" # Standard frame key format
                
                # Try finding the video key in boxes
                matched_video_key = None
                
                # Strategy A: Exact match
                if video_key_full in self.boxs:
                    matched_video_key = video_key_full
                
                # Strategy B: Suffix match (handle 'dataset/' prefix issues)
                if matched_video_key is None:
                    parts = video_key_full.split('/')
                    for idx in range(1, len(parts)):
                        sub_key = '/'.join(parts[idx:])
                        if sub_key in self.boxs:
                            matched_video_key = sub_key
                            break
                
                # 3. Retrieve Box
                if matched_video_key:
                    for fk in [f"{p}.jpg", f"{p-1}.jpg", f"{p+1}.jpg", f"{p-2}.jpg", f"{p+2}.jpg", f"{p:06d}.jpg", f"{p+1:06d}.jpg", f"{p-1:06d}.jpg"]:
                        if fk in self.boxs.get(matched_video_key, {}):
                            box = self.boxs[matched_video_key][fk]
                            break
                
                # Debug logging for missing boxes (only once per video to avoid spam)
                if box is None and i == 0 and p == indices[0]: 
                    # Only log if it's the first frame of the first segment
                    # print(f"[DEBUG] Missing Box: Video='{video_key_full}', Frame='{frame_key}'. MatchedKey='{matched_video_key}'")
                    pass

                # 4. Face Detection (Crop)
                # Reduce margin to 10 (Tight Crop) to zoom in on micro-expressions (eyebrows/eyes)
                # This helps separate Neutral vs Confusion
                # IMPORTANT FIX: If box is None, _face_detect will attempt auto-detection
                img_pil_face = self._face_detect(img_pil, box, margin=10, mode='face')

                # 5. Body Crop (Optional)
                img_pil_body = img_pil # Default to full image (Context)
                
                # If explicit body boxes exist, use them. Otherwise, full image is the best context.
                if self.crop_body:
                    body_box = None
                    if matched_video_key and matched_video_key in self.body_boxes:
                        # Try exact match
                        if frame_key in self.body_boxes[matched_video_key]:
                            body_box = self.body_boxes[matched_video_key][frame_key]
                        else:
                            # Nearest Neighbor Lookup for sparse JSON
                            # Keys are "0.jpg", "5.jpg"...
                            available_keys = list(self.body_boxes[matched_video_key].keys())
                            if available_keys:
                                try:
                                    # Extract frame indices
                                    avail_indices = [int(k.split('.')[0]) for k in available_keys]
                                    # Find nearest
                                    curr_idx = p
                                    nearest_idx = min(avail_indices, key=lambda x: abs(x - curr_idx))
                                    nearest_key = f"{nearest_idx}.jpg"
                                    body_box = self.body_boxes[matched_video_key][nearest_key]
                                except:
                                    pass # Fallback to None (Full Image)
                    
                    if body_box is not None:
                        self.stats['body_box_hit'] += 1
                        left, upper, right, lower = body_box
                        # Ensure coordinates are within image bounds
                        left = max(0, left); upper = max(0, upper)
                        right = min(img_pil.width, right); lower = min(img_pil.height, lower)
                        if right > left and lower > upper:
                            img_pil_body = img_pil.crop((left, upper, right, lower))
                    # else: Keep img_pil_body as full image (Context)

                # 6. Resize and Stack
                # Resize Body
                img_cv_body = self._pil2cv(img_pil_body)
                img_cv_body, _ = self._resize_image(img_cv_body, self.image_size, self.image_size)
                img_pil_body = self._cv2pil(img_cv_body)
                
                # Resize Face (Fix for GroupRandomSizedCrop assertion error)
                img_cv_face = self._pil2cv(img_pil_face)
                img_cv_face, _ = self._resize_image(img_cv_face, self.image_size, self.image_size)
                img_pil_face = self._cv2pil(img_cv_face)
                
                images.append(img_pil_body)
                images_face.append(img_pil_face)
                
                if p < num_real_frames - 1:
                    p += 1

        
        if is_video_file:
            cap.release()

        # Transforms take a list of PIL images
        # images_face: List[PIL.Image]
        # images: List[PIL.Image]
        
        # Apply transforms
        # Note: self.transform usually expects a list and returns a Tensor of shape (T*C, H, W)
        # ToTorchFormatTensor returns (C*T, H, W) because Stack concatenates on axis 2.
        
        process_data = self.transform(images) # (C*T, H, W)
        process_data_face = self.transform(images_face) # (C*T, H, W)

        # Reshape to (T, C, H, W)
        # Since Stack concatenated [img1, img2, ...] -> [R1,G1,B1, R2,G2,B2, ...]
        # Reshaping to (-1, 3, H, W) correctly separates frames.
        process_data = process_data.view(-1, 3, self.image_size, self.image_size)
        process_data_face = process_data_face.view(-1, 3, self.image_size, self.image_size)
        
        return process_data_face, process_data, record.label - 1

    def __len__(self):
        return len(self.video_list)


def train_data_loader(root_dir, list_file, num_segments, duration, image_size,dataset_name,bounding_box_face,bounding_box_body, crop_body=False, num_classes=8):
    if dataset_name == 'DAiSEE':
        print(f"=> Using DAiSEE smart dataloader...")
        return daisee_train_data_loader(root_dir, list_file, num_segments, duration, image_size, 
                                        bounding_box_face, bounding_box_body, crop_body, num_classes)
        
    if dataset_name == 'CAER':
        # Auto-generate list file if missing
        if not os.path.exists(list_file):
            # Infer mode from filename or assume 'train' if not obvious?
            # list_file path is usually something like '.../train.txt'
            # But let's check if 'train' is in the path
            mode = 'train' # Default
            if 'val' in list_file: mode = 'validation'
            elif 'test' in list_file: mode = 'test'
            
            generate_caer_list(root_dir, list_file, mode)

    if dataset_name == "RAER" or dataset_name == "CAER":
         train_transforms = torchvision.transforms.Compose([
            # Apply ColorJitter from video_transform (works on list of images)
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2), 
            GroupRandomGrayscale(p=0.2), # Custom transform for list of images
            RandomRotation(30),
            GroupRandomSizedCrop(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])
    else:
         # Default transforms for other datasets like CK+
         train_transforms = torchvision.transforms.Compose([
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])
            
    
    train_data = VideoDataset(root_dir=root_dir, list_file=list_file,
                              num_segments=num_segments, #16
                              duration=duration, #1
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size,
                              bounding_box_face=bounding_box_face,
                              bounding_box_body=bounding_box_body,
                              crop_body=crop_body,
                              num_classes=num_classes
                              )
    return train_data


def test_data_loader(root_dir, list_file, num_segments, duration, image_size,bounding_box_face,bounding_box_body, crop_body=False, num_classes=8, dataset_name=None):
    # Auto-generate list file for CAER if missing
    if dataset_name == 'CAER' and not os.path.exists(list_file):
        mode = 'test' # Default for test_loader, but could be validation
        if 'val' in list_file: mode = 'validation'
        elif 'train' in list_file: mode = 'train'
        
        generate_caer_list(root_dir, list_file, mode)

    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor()])
    
    test_data = VideoDataset(root_dir=root_dir, list_file=list_file,
                             num_segments=num_segments,
                             duration=duration,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size,
                             bounding_box_face=bounding_box_face,
                             bounding_box_body=bounding_box_body,
                             crop_body=crop_body,
                             num_classes=num_classes
                             )
    return test_data