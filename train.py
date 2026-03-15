import sys
import os
import torch

from src.data.dataloader import build_dataloaders
from src.utils import count_parameters
from src.training.train import train

"""
To run:
python train.py <path_to_data> <model_selection>
e.g. python train.py ./HMDB_simp timesformer
Valid model_selection options: timesformer, videomae
"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Argument parsing
root = sys.argv[1] if len(sys.argv) > 1 else "./HMDB_simp"
model_selection = sys.argv[2] if (
        len(sys.argv) > 2 and sys.argv[2] in ["timesformer",
                                              "videomae"]) else "timesformer"

cfg = {
    # Data
    "num_frames": 8,
    "resolution": 224,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "seed": 42,
    "batch_size": 4,
    "num_workers": 0,

    # Training
    "warmup_epochs": 3,
    "num_epochs": 5,
    "lr_head": 1e-3,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-2,
    "grad_clip": 1.0,
    "early_stop_patience": 4,
}

# Model setup
if model_selection == "videomae":
    from src.model.videomae import load_videomae

    model = load_videomae(num_classes=25,
                          checkpoint="MCG-NJU/videomae-base-finetuned-kinetics")
    # VideoMAE positional embeddings require clip/token layout to match pretrained config.
    cfg["num_frames"] = int(model.config.num_frames)
    image_size = model.config.image_size
    cfg["resolution"] = int(image_size[0] if isinstance(image_size, (tuple, list)) else image_size)
else:
    from src.model.timesformer import load_timesformer

    model = load_timesformer(num_classes=25,
                             checkpoint="facebook/timesformer-base-finetuned-k600")

# Data setup
train_loader, val_loader, _ = build_dataloaders(root, cfg)

# Starting training
print(f"Training {model_selection}")

params = count_parameters(model)
print(f"\nParameters:")
print(f"Total: {params['total']:,}")
print(f"Trainable : {params['trainable']:,}")

model, history = train(
    model, train_loader, val_loader,
    cfg=cfg, device=DEVICE, output_dir="./outputs", model_arch=model_selection
)

print(f"\nTraining history:")
for k, v in history.items():
    print(f"{k}: {[round(x, 4) for x in v]}")
