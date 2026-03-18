import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from src.data.jhmdb_dataset import JHMDBDataset


def get_transforms(resolution=224):
    return T.Compose([
        T.Resize((resolution, resolution)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def build_jhmdb_dataloaders(root, cfg):
    use_pin_memory = torch.cuda.is_available()

    train_dataset = JHMDBDataset(
        root=root,
        split="train",
        split_id=cfg["split_id"],
        num_frames=cfg["num_frames"],
        transform=get_transforms(cfg["resolution"])
    )

    test_dataset = JHMDBDataset(
        root=root,
        split="test",
        split_id=cfg["split_id"],
        num_frames=cfg["num_frames"],
        transform=get_transforms(cfg["resolution"])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=use_pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=use_pin_memory
    )

    return train_loader, test_loader


if __name__ == "__main__":
    cfg = {
        "split_id": 1,
        "num_frames": 8,
        "resolution": 224,
        "batch_size": 2,
        "num_workers": 0,
    }

    root = "./JHMDB"

    train_loader, test_loader = build_jhmdb_dataloaders(root, cfg)

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    frames, boxes, labels, meta = next(iter(train_loader))
    print("Frames shape:", frames.shape)
    print("Boxes shape:", boxes.shape)
    print("Labels shape:", labels.shape)
    print("Meta sample:", meta)