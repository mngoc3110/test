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

# Custom Transform for List of Images (Group Transform)
class GroupRandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img_group):
        if random.random() < self.p:
            # Convert to Grayscale (L) then back to RGB to keep 3 channels
            return [img.convert('L').convert('RGB') for img in img_group]
        return img_group

class DAiSEERecord(object):
    def __init__(self, row, root_dir):
        self._data = row
        self._root_dir = root_dir
        self._resolved_path = None

    @property
    def path(self):
        if self._resolved_path:
            return self._resolved_path
        
        # Determine if the path in _data[0] is absolute or relative
        raw_path = self._data[0]
        if os.path.isabs(raw_path):
             full_path = raw_path
        else:
             full_path = os.path.join(self._root_dir, raw_path)

        if os.path.exists(full_path):
            self._resolved_path = full_path
            return full_path
            
        # Fallback for video extensions
        if not os.path.exists(full_path):
             for ext in ['.avi', '.mp4', '.mov', '.mkv']:
                 if os.path.exists(full_path + ext):
                     self._resolved_path = full_path + ext
                     return self._resolved_path
        
        return full_path

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class DAiSEEDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size, bounding_box_face, bounding_box_body, crop_body=False, root_dir="", num_classes=4):
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
        self.label_is_0_based = True # DAiSEE Engagement is 0, 1, 2, 3
        
        self.debug_samples_path = 'debug_samples_daisee'
        os.makedirs(self.debug_samples_path, exist_ok=True)
        self._saved_samples = {i: 0 for i in range(num_classes)}
        
        self.boxs = {}
        self.body_boxes = {}
        
        # Load bounding boxes if provided
        if self.bounding_box_face and os.path.exists(self.bounding_box_face):
            try:
                with open(self.bounding_box_face, 'r') as f:
                    self.boxs = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load face boxes from {self.bounding_box_face}: {e}")

        if self.crop_body and self.bounding_box_body and os.path.exists(self.bounding_box_body):
            try:
                with open(self.bounding_box_body, 'r') as f:
                    self.body_boxes = json.load(f)
            except Exception as e:
                 print(f"Warning: Failed to load body boxes from {self.bounding_box_body}: {e}")

        self._parse_list()

    def _cv2pil(self, im_cv):
        cv_img_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        pillow_img = Image.fromarray(cv_img_rgb.astype('uint8'))
        return pillow_img

    def _pil2cv(self, im_pil):
        cv_img_rgb = np.array(im_pil)
        cv_img_bgr = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR)
        return cv_img_bgr

    def _resize_image(self, im, width, height):
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

    def _face_detect(self, img, box, margin, mode='face'):
        if box is None:
            if mode == 'face':
                # FALLBACK: Return original image instead of black image if no face detected
                return img
            return img
        else:
            left, upper, right, lower = box
            left = int(left); upper = int(upper); right = int(right); lower = int(lower)
            left = max(0, left - margin)
            upper = max(0, upper - margin)
            right = min(img.width, right + margin)
            lower = min(img.height, lower + margin)
            if mode == 'face':
                img = img.crop((left, upper, right, lower))
                return img
            elif mode == 'body':
                occluded_image = img.copy()
                draw = ImageDraw.Draw(occluded_image)
                draw.rectangle([left, upper, right, lower], fill=(0, 0, 0))
                return occluded_image

    def _parse_list(self):
        self.video_list = []
        invalid_count = 0
        try:
            with open(self.list_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    print(f"DEBUG: First line of {self.list_file}: {lines[0].strip()}")
                
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    # Split by any whitespace (handles tabs and spaces)
                    parts = line.split() 
                    
                    if len(parts) >= 3:
                        # Path is everything except the last two elements
                        path = ' '.join(parts[:-2])
                        num_frames = parts[-2]
                        label = int(parts[-1]) # Convert label to int immediately
                        
                        # Handle invalid labels (e.g., label 4 in 4-class dataset)
                        if label >= 4: # Hardcoded for DAiSEE 4 classes
                            label = 3
                            invalid_count += 1
                        
                        # Add simple check to ensure path isn't empty
                        if path:
                            self.video_list.append(DAiSEERecord([path, num_frames, str(label)], self.root_dir))
                    else:
                        pass 
                        
        except FileNotFoundError:
            print(f"Error: List file not found: {self.list_file}")
            
        if invalid_count > 0:
            print(f"WARNING: Fixed {invalid_count} samples with invalid labels (>=4) by clamping to 3.")
            
        print(f'DAiSEE {self.mode} samples: {len(self.video_list)}')

    def _get_train_indices(self, record):
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def _get_test_indices(self, record):
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
        path = record.path
        is_video_file = os.path.isfile(path) and path.lower().endswith(('.avi', '.mp4', '.mov', '.mkv'))
        
        video_frames_path = []
        num_real_frames = 0
        cap = None

        if is_video_file:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                num_real_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                # print(f"Warning: Could not open video file {path}")
                pass
        elif os.path.isdir(path):
            video_frames_path = glob.glob(os.path.join(path, '*'))
            video_frames_path.sort()
            num_real_frames = len(video_frames_path)
        
        if num_real_frames == 0:
            dummy_shape = (self.num_segments * self.duration, 3, self.image_size, self.image_size)
            if cap: cap.release()
            return torch.zeros(dummy_shape), torch.zeros(dummy_shape), record.label # Label is usually 0-3 for DAiSEE

        indices = np.clip(indices, 0, num_real_frames - 1)
        
        images = []
        images_face = []
        
        # Determine video key for box lookup
        # DAiSEE structure: ClipID.avi or ClipID/frames
        # Key in JSON usually: ClipID
        video_key = os.path.basename(path)
        if '.' in video_key:
            video_key = os.path.splitext(video_key)[0]
        
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                img_pil = None
                
                if is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                    ret, frame = cap.read()
                    if ret:
                        img_cv_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_cv_rgb)
                else:
                    if p < len(video_frames_path):
                        try:
                            img_pil = Image.open(video_frames_path[p]).convert('RGB')
                        except:
                            pass

                if img_pil is None:
                    img_pil = Image.new('RGB', (self.image_size, self.image_size))

                # Face Detection Strategy
                box = None
                frame_key = f"{p + 1}" # DAiSEE JSON usually uses 1-based indexing for frames or filename
                # If JSON keys are just filenames "1.jpg", "2.jpg"
                frame_key_alt = f"{p + 1}.jpg"
                
                if video_key in self.boxs:
                    # Check different frame key formats
                    if frame_key in self.boxs[video_key]:
                        box = self.boxs[video_key][frame_key]
                    elif frame_key_alt in self.boxs[video_key]:
                        box = self.boxs[video_key][frame_key_alt]
                
                # Use 10 margin balanced crop
                img_pil_face = self._face_detect(img_pil, box, margin=10, mode='face')

                # Body Crop (Optional)
                img_pil_body = img_pil
                if self.crop_body:
                    body_box = None
                    if video_key in self.body_boxes:
                        if frame_key in self.body_boxes[video_key]:
                            body_box = self.body_boxes[video_key][frame_key]
                        elif frame_key_alt in self.body_boxes[video_key]:
                             body_box = self.body_boxes[video_key][frame_key_alt]
                    
                    if body_box:
                         img_pil_body = self._face_detect(img_pil, body_box, margin=0, mode='body_crop_internal') # Reuse crop logic
                         # Note: _face_detect doesn't implement 'body_crop_internal' specifically but crop logic is generic
                         # Let's just use manual crop here for safety or update _face_detect
                         left, upper, right, lower = body_box
                         img_pil_body = img_pil.crop((left, upper, right, lower))

                # Resize Body
                img_cv_body = self._pil2cv(img_pil_body)
                img_cv_body, _ = self._resize_image(img_cv_body, self.image_size, self.image_size)
                img_pil_body = self._cv2pil(img_cv_body)
                
                images.append(img_pil_body)
                images_face.append(img_pil_face)
                
                if p < num_real_frames - 1:
                    p += 1
        
        if cap:
            cap.release()

        # Apply Transforms
        process_data = self.transform(images) # (C*T, H, W)
        process_data_face = self.transform(images_face) # (C*T, H, W)

        # Reshape to (T, C, H, W)
        process_data = process_data.view(-1, 3, self.image_size, self.image_size)
        process_data_face = process_data_face.view(-1, 3, self.image_size, self.image_size)
        
        return process_data_face, process_data, record.label # DAiSEE labels are 0,1,2,3

    def __len__(self):
        return len(self.video_list)

def daisee_train_data_loader(root_dir, list_file, num_segments, duration, image_size, bounding_box_face, bounding_box_body, crop_body=False, num_classes=4):
    train_transforms = torchvision.transforms.Compose([
        # Use custom ColorJitter from video_transform which handles list of images
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        GroupRandomGrayscale(p=0.2),
        RandomRotation(4),
        GroupResize(image_size),
        GroupRandomHorizontalFlip(),
        Stack(),
        ToTorchFormatTensor()])
    
    return DAiSEEDataset(root_dir=root_dir, list_file=list_file,
                         num_segments=num_segments,
                         duration=duration,
                         mode='train',
                         transform=train_transforms,
                         image_size=image_size,
                         bounding_box_face=bounding_box_face,
                         bounding_box_body=bounding_box_body,
                         crop_body=crop_body,
                         num_classes=num_classes)

def daisee_test_data_loader(root_dir, list_file, num_segments, duration, image_size, bounding_box_face, bounding_box_body, crop_body=False, num_classes=4):
    test_transform = torchvision.transforms.Compose([
        GroupResize(image_size),
        Stack(),
        ToTorchFormatTensor()])
    
    return DAiSEEDataset(root_dir=root_dir, list_file=list_file,
                         num_segments=num_segments,
                         duration=duration,
                         mode='test',
                         transform=test_transform,
                         image_size=image_size,
                         bounding_box_face=bounding_box_face,
                         bounding_box_body=bounding_box_body,
                         crop_body=crop_body,
                         num_classes=num_classes)
