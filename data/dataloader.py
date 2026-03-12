# data/dataloader.py
# EEEM068 Action Recognition using  ViT
# Author: Prasanna Lamgade
# Group members: Ben Davison, Chris Gainullin, Saba Ali, Youssef Abdelrahim
# SC: python -m data.dataloader "./HMDB_simp"

import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

from data.dataset import HMDBDataset
import torchvision.transforms as T

def get_transforms(resolution=224):
    return T.Compose([
        T.Resize((resolution, resolution)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], # Mean pixel values for RGB channels
                    std=[0.229, 0.224, 0.225]), # Std for RGB channels
    ])

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
    dataset = HMDBDataset(
        root=root,
        num_frames=cfg["num_frames"],
        transform=get_transforms(cfg["resolution"])
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
        pin_memory=True # pins loaded tensors to CPU memory for faster transfer to GPU
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

# Sanity check
if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "./HMDB_simp"

    cfg = {
        "num_frames": 8,
        "resolution": 224,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "seed": 42,
        "batch_size": 4,
        "num_workers": 0
    }

    train_loader, val_loader, test_loader = build_dataloaders(root, cfg)

    print(f"\nBatches\n Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # Pull one batch and check shapes
    frames, labels = next(iter(train_loader))
    print(f"\nTrain batch:")
    print(f"Frames shape: {frames.shape}") # expect torch.Size([4, 8, 3, 224, 224])
    print(f"Labels shape: {labels.shape}") # expect torch.Size([4])
    print(f"Labels: {labels.tolist()}")