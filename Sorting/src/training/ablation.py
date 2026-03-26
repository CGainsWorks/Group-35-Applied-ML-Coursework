# ablation.py
# EEEM068 Action Recognition using ViT
# Author: Prasanna Lamgade
# Group members: Ben Davison, Chris Gainullin, Saba Ali, Youssef Abdelrahim
#
# Number of input frames (4, 8, 16)
# Learning rate schedule (cosine, step, linear_warmup_cosine)
#
#   python ablation.py

import os
import json
import copy
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

from src.data.dataloader import build_dataloaders
from src.model.videomae import load_videomae
from src.training.train import train
from src.utils import seed_everything, count_parameters

CHECKPOINT = "MCG-NJU/videomae-base-finetuned-kinetics"
NUM_CLASSES = 25
DATA_DIR    = "./HMDB_simp"
OUTPUT_DIR  = "./outputs/ablation"

# Single ablation run
def run_ablation(label, cfg, device, data_dir, output_dir):
    print(f"Ablation run: {label}")
    print(f"num_frames={cfg['num_frames']}")
    print(f"scheduler={cfg['scheduler_type']}")
    print(f"horizontal_flip={cfg['horizontal_flip']}")
    print(f"colour_jitter={cfg['colour_jitter']}")
    print(f"random_crop={cfg['random_crop']}")
    print(f"reversed={cfg['reversed']}")

    seed_everything(cfg["seed"])

    pretrained_num_frames = 16
    if cfg["num_frames"] != pretrained_num_frames:
        config = VideoMAEConfig.from_pretrained(CHECKPOINT)
        config.num_frames = cfg["num_frames"]
        config.num_labels = NUM_CLASSES
        model = VideoMAEForVideoClassification.from_pretrained(
            CHECKPOINT,
            ignore_mismatched_sizes=True,  # head + positional embeddings may differ
            config=config,
        )
    else:
        # num_frames matches pretrained default — use Ben's loader as-is
        model = load_videomae(num_classes=NUM_CLASSES, checkpoint=CHECKPOINT)

    params = count_parameters(model)
    print(f"Parameters Total: {params['total']:,}\n  Trainable: {params['trainable']:,}")

    # Build dataloaders with this run's num_frames
    train_loader, val_loader, _ = build_dataloaders(data_dir, cfg)

    # Each run gets its own subdirectory so checkpoints don't overwrite each other
    run_output_dir = os.path.join(output_dir, label)
    os.makedirs(run_output_dir, exist_ok=True)

    _, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        output_dir=run_output_dir,
        model_arch=f"videomae_{label}",
    )

    # Pull the best val accuracy seen during training (EarlyStopping tracks this)
    best_val_acc = max(history["val_acc"])
    best_val_f1  = max(history["val_f1_macro"])

    # Mean FLOPs per epoch (ignoring NaN placeholders)
    epoch_flops = history["total_epoch_flops"]
    if epoch_flops:
        mean_flops  = sum(epoch_flops) / len(epoch_flops)
    else:
        mean_flops = None

    summary = {
        "label": label,
        "num_frames": cfg["num_frames"],
        "scheduler_type": cfg["scheduler_type"],
        "horizontal_flip": cfg["horizontal_flip"],
        "colour_jitter": cfg["colour_jitter"],
        "random_crop": cfg["random_crop"],
        "reversed": cfg["reversed"],
        "best_val_acc": round(best_val_acc, 4),
        "best_val_f1": round(best_val_f1, 4),
        "epochs_run": len(history["val_acc"]),
        "mean_epoch_gflops": round(mean_flops / 1e9, 2) if mean_flops is not None else None,
    }

    # Save this run's summary alongside its checkpoint
    summary_path = os.path.join(run_output_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRun summary saved: {summary_path}")

    return summary