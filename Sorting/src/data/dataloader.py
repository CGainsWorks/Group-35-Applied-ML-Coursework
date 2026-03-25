# data/dataloader.py
# EEEM068 Action Recognition using  ViT
# Author: Prasanna Lamgade
# Group members: Ben Davison, Chris Gainullin, Saba Ali, Youssef Abdelrahim
# SC: python -models data.dataloader "./HMDB_simp"

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

from src.data.dataset import HMDBDataset
import torchvision.transforms.v2 as v2

def get_transforms(resolution=224, horizontal_flip=False, colour_jitter=False, reversed_prob=False, random_crop=False):
    reversed_prob = 0.5 if reversed_prob else 0.0

    transforms: list[v2.Transform] = [v2.ToImage()]

    # Random resize crop
    if random_crop:
        transforms.append(v2.RandomResizedCrop(
            size=(resolution, resolution),
            scale=(0.5, 1.0),
            ratio=(0.75, 1.333),
            interpolation=v2.InterpolationMode.BICUBIC,
            # Bicubic is often preferred for VideoMAE
            antialias=True
        ))

    # Temporal Reversing
    if reversed_prob > 0:
        transforms.append(
            v2.RandomApply([v2.Lambda(lambda x: torch.flip(x, dims=[0]))],
                           p=reversed_prob))

    # Spatial Resize
    transforms.append(v2.Resize((resolution, resolution), antialias=True))

    # Horizontal Flip
    if horizontal_flip:
        transforms.append(v2.RandomHorizontalFlip(p=0.5))

    # Colour Jitter
    if colour_jitter:
        transforms.append(
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
                           hue=0.1))

    # Final conversion and normalise
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    transforms.append(
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return v2.Compose(transforms)

def build_splits(dataset, train_ratio=0.70, val_ratio=0.15, seed=42):
    # Extract all labels from the dataset, needed for stratified splitting
    label_list = []
    for i in range(len(dataset)):
        label_list.append(dataset.samples[i][1])
        # dataset.samples[i] is a (folder_path, label_int) tuple, so [1] gets the label
    labels = np.array(label_list) # class (int) for each position
    indices = np.arange(len(dataset)) # positions

    # Test set
    test_ratio = 1.0 - train_ratio - val_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(indices, labels))

    # Split remaining into train / val
    val_frac = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(sss2.split(trainval_idx, labels[trainval_idx]))

    # Remap local indices back to global dataset indices
    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    print(f"Split sizes Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

def build_dataloaders(root, cfg):
    use_pin_memory = torch.cuda.is_available()
    dataset = HMDBDataset(
        root=root,
        num_frames=cfg["num_frames"],
        transform=get_transforms(cfg["resolution"], cfg["horizontal_flip"], cfg["colour_jitter"], cfg["reversed"], cfg["random_crop"])
    )

    # Get split indices from the train dataset (all three share same sample ordering)
    train_idx, val_idx, test_idx = build_splits(
        dataset,
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        seed=cfg["seed"]
    )

    # Wrap each split as a Subset. No data is copied, just index references
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg["batch_size"], # calls `__getitem__` n times and stacks the results into `(4, 8, 3, 224, 224)`
        shuffle=True, # Randomises the order of samples each epoch. Train only
        num_workers=cfg["num_workers"], # how many parallel processes to use for loading
        pin_memory=use_pin_memory # pins loaded tensors for faster transfer to GPU
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=use_pin_memory
    )

    return train_loader, val_loader, test_loader
