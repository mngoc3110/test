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
        
        # ASSUMPTION: Kaggle DAiSEE labels are 1-based (1,2,3,4) -> Need to subtract 1 to get (0,1,2,3)
        self.label_is_0_based = False 
        
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

    # ... (Keep existing methods: _cv2pil, _pil2cv, _resize_image, _face_detect) ...

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
        all_labels = []
        invalid_count = 0
        try:
            with open(self.list_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    print(f"DEBUG: First line of {self.list_file}: {lines[0].strip()}")
                
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    parts = line.split() 
                    
                    if len(parts) >= 3:
                        raw_path = ' '.join(parts[:-2])
                        num_frames = parts[-2]
                        
                        # SMART PATH CORRECTION
                        # If raw_path contains keywords like Train/Test/Validation, extract the relative path
                        # regardless of what prefix it has (e.g. /content/drive/.../Train/...)
                        keywords = ['/Train/', '/Test/', '/Validation/']
                        clean_path = raw_path
                        for kw in keywords:
                            if kw in raw_path:
                                # Split and take the part starting with the keyword (without the leading /)
                                # e.g. /data/DAiSEE/Train/Clip1 -> Train/Clip1
                                clean_path = raw_path.split(kw, 1)[1]
                                clean_path = kw.strip('/') + '/' + clean_path
                                break
                        
                        # If the path starts with the keyword but no leading slash (e.g. Train/Clip1)
                        # clean_path remains as is, which is correct.
                        
                        try:
                            label = int(parts[-1])
                            all_labels.append(label)
                            
                            # Store with clean path logic handled by DAiSEERecord or just pass clean path
                            # We pass clean_path to DAiSEERecord, but we need to ensure DAiSEERecord uses root_dir
                            # effectively. Let's pass the relative path.
                            if clean_path:
                                self.video_list.append([clean_path, num_frames, label])
                        except ValueError:
                            print(f"Warning: Invalid label format in line: {line}")
        except FileNotFoundError:
            print(f"Error: List file not found: {self.list_file}")
            
        # AUTO-DETECT LABEL RANGE
        if all_labels:
            min_label = min(all_labels)
            max_label = max(all_labels)
            print(f"DEBUG: Label range detected: [{min_label}, {max_label}]")
            
            if min_label >= 1 and max_label <= 4:
                print("=> Detected 1-based labels (1-4). Will convert to 0-based (0-3).")
                self.label_is_0_based = False
            elif min_label >= 0 and max_label <= 3:
                print("=> Detected 0-based labels (0-3). Keeping as is.")
                self.label_is_0_based = True
            else:
                print(f"WARNING: Unusual label range! Min={min_label}, Max={max_label}. Defaulting to 0-based check.")
                if max_label > 3:
                     self.label_is_0_based = False
                else:
                     self.label_is_0_based = True

            # NORMALIZE LABELS IN VIDEO LIST
            normalized_list = []
            for item in self.video_list:
                path, num, raw_lbl = item
                final_lbl = raw_lbl if self.label_is_0_based else raw_lbl - 1
                final_lbl = max(0, min(final_lbl, 3))
                
                # Here we pass the CLEAN relative path. DAiSEERecord will join it with self.root_dir
                normalized_list.append(DAiSEERecord([path, num, str(final_lbl)], self.root_dir))
            
            self.video_list = normalized_list
            self.label_is_0_based = True 
            
        print(f'DAiSEE {self.mode} samples: {len(self.video_list)}')

    # ... (Keep existing methods: _get_train_indices, _get_test_indices) ...
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
            # Handle label adjustment here if needed, but VideoDataset usually handles it.
            # BUT: DAiSEEDataset is used directly. We need to respect label_is_0_based logic.
            final_label = record.label if self.label_is_0_based else record.label - 1
            return torch.zeros(dummy_shape), torch.zeros(dummy_shape), final_label

        indices = np.clip(indices, 0, num_real_frames - 1)
        
        images = []
        images_face = []
        
        # Determine video key for box lookup
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
                frame_key = f"{p + 1}"
                frame_key_alt = f"{p + 1}.jpg"
                
                if video_key in self.boxs:
                    if frame_key in self.boxs[video_key]:
                        box = self.boxs[video_key][frame_key]
                    elif frame_key_alt in self.boxs[video_key]:
                        box = self.boxs[video_key][frame_key_alt]
                
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
        process_data = self.transform(images) 
        process_data_face = self.transform(images_face) 

        process_data = process_data.view(-1, 3, self.image_size, self.image_size)
        process_data_face = process_data_face.view(-1, 3, self.image_size, self.image_size)
        
        # CRITICAL FIX: Apply label adjustment based on label_is_0_based flag
        final_label = record.label if self.label_is_0_based else record.label - 1
        return process_data_face, process_data, final_label

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
