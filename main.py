# main.py
# EEEM068 Action Recognition using ViT
# Author: Prasanna Lamgade
# Group members: Ben Davison, Chris Gainullin, Saba Ali, Youssef Abdelrahim
# Run: python main.py --model timesformer

import os
import argparse
import torch

from src.data.dataloader import build_dataloaders
from src.data.dataset import HMDBDataset
from src.training.train import train
from src.utils import seed_everything, count_parameters
from src.model.timesformer import load_timesformer
from src.model.videomae import load_videomae

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     type=str, required=True, choices=["timesformer", "videomae"])
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()

    CFG = {
        "num_frames": 8,
        "resolution": 224,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "seed": 42,
        "batch_size": 4,
        "num_workers": 0,
        "warmup_epochs": 3,
        "num_epochs": 10,
        "lr_head": 1e-3,
        "lr_backbone": 1e-5,
        "weight_decay": 1e-2,
        "grad_clip": 1.0,
        "early_stop_patience": 4,
    }

    CHECKPOINTS = {
        "timesformer": "facebook/timesformer-base-finetuned-k600",
        "videomae":  "MCG-NJU/videomae-base-finetuned-kinetics"
    }

    DATA_DIR = "./HMDB_simp"
    OUTPUT_DIR = "./outputs"
    CKPT_PATH = os.path.join(OUTPUT_DIR, f"{args.model}_best.pt")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Model: {args.model}")
    print(f"Device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    seed_everything(CFG["seed"])

    # Data setup
    train_loader, val_loader, test_loader = build_dataloaders(DATA_DIR, CFG)
    ds = HMDBDataset(root=DATA_DIR, num_frames=CFG["num_frames"])
    class_names = ds.classes

    # Model setup
    if args.model == "timesformer":
        model = load_timesformer(num_classes=25, checkpoint=CHECKPOINTS[args.model])
    elif args.model == "videomae":
        model = load_videomae(num_classes=25, checkpoint=CHECKPOINTS[args.model])
        # VideoMAE positional embeddings require clip/token layout to match pretrained config.
        CFG["num_frames"] = int(model.config.num_frames)
        image_size = model.config.image_size
        CFG["resolution"] = int(image_size[0] if isinstance(image_size, (tuple,
                                                                         list)) else image_size)
    else:
        raise ValueError(f"Invalid model: {args.model}")

    params = count_parameters(model)
    print(f"\nParameters:")
    print(f"Total: {params['total']:,}")
    print(f"Trainable : {params['trainable']:,}")

    if args.eval_only:
        if not os.path.exists(CKPT_PATH):
            raise FileNotFoundError(f"No checkpoint at {CKPT_PATH}")
        model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
        print(f"Loaded checkpoint from {CKPT_PATH}")
    else:
        model, history = train(
            model, train_loader, val_loader,
            cfg=CFG, device=DEVICE, output_dir=OUTPUT_DIR, model_arch=args.model
        )

        print(f"\nTraining history:")
        for k, v in history.items():
            print(f"{k}: {[round(x, 4) for x in v]}")

if __name__ == "__main__":
    main()