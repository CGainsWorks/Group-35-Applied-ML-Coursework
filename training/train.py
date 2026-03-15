# training/train.py
# EEEM068 Action Recognition using  ViT
# Author: Prasanna Lamgade
# Group members: Ben Davison, Chris Gainullin, Saba Ali, Youssef Abdelrahim
# SC: python -models training.train "./HMDB_simp"

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict


# Early Stopping
class EarlyStopping:
    def __init__(self, patience=4, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -np.inf
        self.counter = 0
        self.best_state = None

    def step(self, val_acc, model):
        if val_acc > self.best + self.min_delta:
            self.best = val_acc # Saves better val_acc
            self.counter = 0 # Resets counter
            # Store a copy of the best weights on CPU to save VRAM
            self.best_state = {}
            for k, v in model.state_dict().items():
                self.best_state[k] = v.cpu().clone()
            print(f"New best val acc: {self.best:.4f}. Saving weights")
        else:
            self.counter += 1
        print(f"No improvement ({self.counter}/{self.patience})")

        return self.counter >= self.patience # True = stop

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)
            print(f"Restored best weights (val acc {self.best:.4f})")

# Fine-tuning Strategy
def freeze_backbone(model):
    # Freeze all layers except the classification head.
    for name, param in model.named_parameters(): # returns (name,tensor)
        if "classifier" not in name:
            param.requires_grad = False # Frozen

    # Logging step after freezing step
    trainable = 0
    for p in model.parameters(): # returns just tensor
        if p.requires_grad:
            trainable += p.numel()
    print(f"Backbone frozen\n Trainable params: {trainable:,} (head only)") # Expect 19,225

def unfreeze_all(model, lr_backbone, lr_head, weight_decay):
    for param in model.parameters():
        param.requires_grad = True # All params set to True

    trainable = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable += p.numel()

    print(f"Backbone unfrozen\n Trainable params: {trainable:,}") #  expect 121,277,977

    # Separate backbone and head params for differential lr
    backbone_params = []
    for name, param in model.named_parameters():
        if "classifier" not in name:
            backbone_params.append(param)

    head_params = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            head_params.append(param)

    # Define new lr for each
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone}, # small, preserve pretrained
        {"params": head_params, "lr": lr_head}, # large, train from beginning
    ], weight_decay=weight_decay)

    return optimizer

# Single Epoch
def run_epoch(model, loader, optimizer, scheduler, grad_clip, device, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, correct, total = 0.0, 0, 0

    if train:
        context = torch.enable_grad() # track gradient
    else:
        context = torch.no_grad()

    with context:
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            # HuggingFace models compute loss internally when labels are passed
            # Forward PAss
            out = model(pixel_values=frames, labels=labels)
            loss = out.loss

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler:
                    scheduler.step()

            predictions = out.logits.argmax(-1)
            correct_mask = (predictions == labels)
            correct += correct_mask.sum().item()

            total += frames.size(0)

    return total_loss / total, correct / total # avg_loss, accuracy

# Main Training Function
def train(model, train_loader, val_loader, cfg, device, output_dir):
    model = model.to(device)
    os.makedirs(output_dir, exist_ok=True)

    # head only
    freeze_backbone(model)
    # Only pass trainable parameters to the optimiser (head only in warmup)
    trainable_params = []
    for p in model.parameters():
        if p.requires_grad:
            trainable_params.append(p)

    optimizer = optim.AdamW(
        trainable_params,
        lr=cfg["lr_head"],
        weight_decay=cfg["weight_decay"]
    )

    scheduler = None # no scheduler during warmup
    stopper = EarlyStopping(patience=cfg["early_stop_patience"])
    history = defaultdict(list)
    phase = 1

    for epoch in range(1, cfg["num_epochs"] + 1):
        # Switch to phase 2 after warmup
        if epoch == cfg["warmup_epochs"] + 1:
            optimizer = unfreeze_all(
                model,
                lr_backbone=cfg["lr_backbone"],
                lr_head=cfg["lr_head"],
                weight_decay=cfg["weight_decay"]
            )
            total_remaining = (cfg["num_epochs"] - cfg["warmup_epochs"]) * len(train_loader)
            # Remaining epochs * number of batches per epoch
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_remaining # steps to decay over
            )
            phase = 2

        print(f"\nEpoch {epoch:02d}/{cfg['num_epochs']} [Phase {phase}]")

        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, scheduler, cfg["grad_clip"], device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, None, None, cfg["grad_clip"], device, train=False
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if stopper.step(val_acc, model): # called every epoch
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    stopper.restore(model)

    # Save best checkpoint
    checkpoint_path = os.path.join(output_dir, "timesformer_best.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nSaved best checkpoint: {checkpoint_path}")

    return model, dict(history)

# Sanity check
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from model.timesformer import load_timesformer
    from data.dataloader import build_dataloaders

    root = sys.argv[1] if len(sys.argv) > 1 else "./HMDB_simp"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    train_loader, val_loader, _ = build_dataloaders(root, cfg)
    model = load_timesformer(num_classes=25, checkpoint="facebook/timesformer-base-finetuned-k600")

    model, history = train(
        model, train_loader, val_loader,
        cfg=cfg, device=DEVICE, output_dir="./outputs"
    )

    print(f"\nTraining history:")
    for k, v in history.items():
        print(f"{k}: {[round(x, 4) for x in v]}")