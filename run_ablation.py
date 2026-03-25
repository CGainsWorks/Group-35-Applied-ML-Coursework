import os
import json
import copy
import torch

from src.training.ablation import run_ablation

# Baseline config: Should look to create a config.py
BASE_CFG = {
    "num_frames": 16,
    "resolution": 224,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "seed": 42,
    "batch_size": 1,
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