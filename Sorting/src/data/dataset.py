# data/dataset.py
# EEEM068 Action Recognition using  ViT
# Author: Prasanna Lamgade
# Group members: Ben Davison, Chris Gainullin, Saba Ali, Youssef Abdelrahim
# SC: python -models data.dataset "./HMDB_simp"

import os
from pathlib import Path
from PIL import Image # opening, manipulating image files

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import v2


class HMDBDataset(Dataset):
    def __init__(self, root: str, num_frames: int = 8, transform=None):
        self.root = Path(root)
        self.num_frames = num_frames
        self.transform = transform
        self.samples= []   # list of (frame_dir, class_idx)
        self.class_to_idx = {}
        self.classes = []

        self._discover()

    def _discover(self):
        # Get all subdirectories in root, sorted alphabetically
        class_dirs = []
        self.classes = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir():
                class_dirs.append(d) # list of folder paths
                self.classes.append(d.name) # list of class names

        # Map each class name to a unique integer index
        self.class_to_idx = {}
        for i, c in enumerate(self.classes):
            self.class_to_idx[c] = i

        # Walk each class folder and collect all sample subfolders
        for class_dir in class_dirs:
            label = self.class_to_idx[class_dir.name]
            for sample_dir in sorted(class_dir.iterdir()):
                if sample_dir.is_dir():
                    self.samples.append((sample_dir, label))

        print(f"Found {len(self.classes)} classes, {len(self.samples)} samples")

    def _load_frames(self, sample_dir: Path) -> torch.Tensor:
        # Collect and sort all JPEGs in sample_dir, sort preserves temporal order
        frame_paths = sorted(sample_dir.glob("*.jpg"))

        if len(frame_paths) == 0:
            raise ValueError(f"No files found in {sample_dir}")

        # Uniform sampling
        total = len(frame_paths)
        step = total / self.num_frames # sampling interval

        indices = []
        for i in range(self.num_frames):
            idx = int(i * step)
            idx = min(idx, total - 1) # safety net, to prevent oob
            indices.append(idx)

        frames = []
        for idx in indices:
            img = Image.open(frame_paths[idx]).convert("RGB") # safety net, if there is a greyscale image
            frames.append(img) # list of PIL image object to pass into transform pipeline

        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample_dir, label = self.samples[i]
        pil_frames = self._load_frames(sample_dir)  # List of PIL images

        # Convert list of PIL images to a 4D Tensor (T, C, H, W)
        # transforms.v2 is used now to support video properly
        frames = torch.stack([v2.functional.to_image(f) for f in pil_frames])

        # Apply the transform to the entire block at once
        if self.transform:
            frames = self.transform(frames)
        else:
            # Simple fallback to float32 [0, 1]
            frames = v2.functional.to_dtype(frames, torch.float32, scale=True)

        return frames, label
