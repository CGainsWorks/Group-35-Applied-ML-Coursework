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

# Baseline config: Should look to create a config.py
BASE_CFG = {
    "num_frames": 16,
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
    "scheduler_type": "cosine",
    "horizontal_flip": False,
    "colour_jitter": False,
    "random_crop": False,
    "reversed": False,
}

CHECKPOINT = "MCG-NJU/videomae-base-finetuned-kinetics"
NUM_CLASSES = 25
DATA_DIR    = "./HMDB_simp"
OUTPUT_DIR  = "./outputs/ablation"

# Ablation definitions
ABLATIONS = {
    # Study 1: Number of input frames
    "frames": [
        ("frames_04", {"num_frames": 4}),
        ("frames_08", {"num_frames": 8}),
        ("frames_16", {"num_frames": 16}),
        ("frames_32", {"num_frames": 32}),
    ],
    # Study 2: learning rate schedule
    "scheduler": [
        ("sched_cosine", {"scheduler_type": "cosine"}),
        ("sched_step", {"scheduler_type": "step"}),
        ("sched_linear_warmup_cosine", {"scheduler_type": "linear_warmup_cosine"}),
    ],
    # Study 3: data augmentation
    "augmentation": [
        ("aug_flip", {"horizontal_flip": True}),
        ("aug_jitter", {"colour_jitter": True}),
        ("aug_crop", {"random_crop": True}),
        ("aug_reversed", {"reversed": True}),
    ]
}

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
        model = VideoMAEForVideoClassification.from_pretrained(
            CHECKPOINT,
            config=config,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,  # head + positional embeddings may differ
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_summaries = {}

    for study_name in ABLATIONS.keys():
        print(f"\nSTUDY: {study_name}")

        study_summaries = []
        for label, overrides in ABLATIONS[study_name]:
            # Build this run's config: start from base, apply overrides
            cfg = copy.deepcopy(BASE_CFG)
            cfg.update(overrides)

            summary = run_ablation(
                label=label,
                cfg=cfg,
                device=device,
                data_dir=DATA_DIR,
                output_dir=os.path.join(OUTPUT_DIR, study_name),
            )
            study_summaries.append(summary)

        all_summaries[study_name] = study_summaries

        # Print a compact results table for this study
        print(f"\n{study_name} ablation results")
        for s in study_summaries:
            if s["mean_epoch_gflops"] is not None:
                gflops = f"{s['mean_epoch_gflops']:.2f}"
            else:
                gflops = "N/A"
            print(
                f"label={s['label']} "
                f"num of frames={s['num_frames']} "
                f"scheduler={s['scheduler_type']} "
                f"horizontal_flip={s['horizontal_flip']} "
                f"colour_jitter={s['colour_jitter']} "
                f"random_crop={s['random_crop']} "
                f"reversed={s['reversed']} "
                f"val_acc={s['best_val_acc']} "
                f"val_f1={s['best_val_f1']} "
                f"gflops={gflops}"
            )

    results_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAblation results saved: {results_path}")

if __name__ == "__main__":
    main()