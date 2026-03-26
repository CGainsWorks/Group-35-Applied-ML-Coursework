# training/train.py
# EEEM068 Action Recognition using  ViT
# Author: Prasanna Lamgade
# Group members: Ben Davison, Chris Gainullin, Saba Ali, Youssef Abdelrahim
# SC: python -models training.train "./HMDB_simp"

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import tqdm

from src.eval.metrics import compute_classification_metrics, format_metrics_summary

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
            self.best = val_acc  # Saves better val_acc
            self.counter = 0  # Resets counter
            # Store a copy of the best weights on CPU to save VRAM
            self.best_state = {}
            for k, v in model.state_dict().items():
                self.best_state[k] = v.cpu().clone()
            print(f"New best val acc: {self.best:.4f}.")
        else:
            self.counter += 1
            print(f"No improvement ({self.counter}/{self.patience})")

        return self.counter >= self.patience  # True = stop

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)
            print(f"Restored best weights (val acc {self.best:.4f})")

# Fine-tuning Strategy
def freeze_backbone(model):
    # Freeze all layers except the classification head.
    for name, param in model.named_parameters():  # returns (name,tensor)
        if "classifier" not in name:
            param.requires_grad = False  # Frozen

    # Logging step after freezing step
    trainable = 0
    for p in model.parameters():  # returns just tensor
        if p.requires_grad:
            trainable += p.numel()
    print(
        f"Backbone frozen\n Trainable params: {trainable:,} (head only)")  # Expect 19,225

def unfreeze_all(model, lr_backbone, lr_head, weight_decay):
    for param in model.parameters():
        param.requires_grad = True  # All params set to True

    trainable = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable += p.numel()

    print(
        f"Backbone unfrozen\n Trainable params: {trainable:,}")  # expect 121,277,977

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
        {"params": backbone_params, "lr": lr_backbone},
        # small, preserve pretrained
        {"params": head_params, "lr": lr_head},  # large, train from beginning
    ], weight_decay=weight_decay)

    return optimizer

def build_scheduler(optimizer, scheduler_type: str, total_steps: int, warmup_steps: int):
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    elif scheduler_type == "step":
        step_size = max(1, total_steps // 3)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    elif scheduler_type == "linear_warmup_cosine":
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError

# Single Epoch
def run_epoch(model, loader, optimizer, scheduler, grad_clip, device, train=True, return_predictions=False):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    num_batches = 0

    if train:
        context = torch.enable_grad()  # track gradient
    else:
        context = torch.no_grad()

    with context:
        for frames, labels in tqdm.tqdm(loader):
            num_batches += 1
            frames = frames.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            # HuggingFace models compute loss internally when labels are passed
            # Forward PAss
            out = model(pixel_values=frames, labels=labels)
            loss = out.loss
            batch_size = frames.size(0)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler:
                    scheduler.step()

            predictions = out.logits.argmax(-1)
            correct_mask = (predictions == labels)
            correct += correct_mask.sum().item()
            total_loss += loss.item() * batch_size
            if return_predictions:
                all_preds.append(predictions.detach().cpu())
                all_labels.append(labels.detach().cpu())

            total += batch_size

    result = {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "num_batches": num_batches,
    }
    if return_predictions:
        result["predictions"] = torch.cat(all_preds).numpy()
        result["labels"] = torch.cat(all_labels).numpy()
    return result


def _extract_class_names(loader):
    dataset_obj = loader.dataset
    if hasattr(dataset_obj, "dataset"):
        dataset_obj = dataset_obj.dataset
    if hasattr(dataset_obj, "classes"):
        return list(dataset_obj.classes)
    return None


from thop import profile

def _estimate_batch_flops(model, frames, labels, train=True):
    try:
        macs, params = profile(model, inputs=(frames,), verbose=False)
        fwd_flops = macs * 2

        if fwd_flops > 0:
            return fwd_flops * (3.0 if train else 1.0)
    except Exception as e:
        print(f"Error calculating FLOPs with thop: {e}")
        pass
    return None


# Main Training Function
def train(model, train_loader, val_loader, cfg, device, output_dir, model_arch):
    model = model.to(device)
    os.makedirs(output_dir, exist_ok=True)
    class_names = _extract_class_names(train_loader)

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

    scheduler = None  # no scheduler during warmup
    stopper = EarlyStopping(patience=cfg["early_stop_patience"])
    history = defaultdict(list)
    phase = 1

    train_batch_flops = None
    val_batch_flops = None
    try:
        train_frames, train_labels = next(iter(train_loader))
        train_batch_flops = _estimate_batch_flops(
            model,
            train_frames.to(device),
            train_labels.to(device),
            train=True,
        )
    except StopIteration:
        raise ValueError("train_loader is empty")

    try:
        val_frames, val_labels = next(iter(val_loader))
        val_batch_flops = _estimate_batch_flops(
            model,
            val_frames.to(device),
            val_labels.to(device),
            train=False,
        )
    except StopIteration:
        raise ValueError("val_loader is empty")

    for epoch in range(1, cfg["num_epochs"] + 1):
        # Switch to phase 2 after warmup
        if epoch == cfg["warmup_epochs"] + 1:
            optimizer = unfreeze_all(
                model,
                lr_backbone=cfg["lr_backbone"],
                lr_head=cfg["lr_head"],
                weight_decay=cfg["weight_decay"]
            )
            total_remaining = (cfg["num_epochs"] - cfg["warmup_epochs"]) * len(
                train_loader)
            # Remaining epochs * number of batches per epoch
            scheduler_type = cfg.get("scheduler_type", "cosine")
            warmup_steps = len(train_loader)
            scheduler = build_scheduler(
                optimizer,
                scheduler_type=scheduler_type,
                total_steps=total_remaining,
                warmup_steps=warmup_steps,
            )
            phase = 2

        print(f"\nEpoch {epoch:02d}/{cfg['num_epochs']} [Phase {phase}]")

        train_stats = run_epoch(
            model, train_loader, optimizer, scheduler, cfg["grad_clip"], device,
            train=True
        )
        val_stats = run_epoch(
            model, val_loader, None, None, cfg["grad_clip"], device, train=False, return_predictions=True
        )
        val_metrics = compute_classification_metrics(
            y_true=val_stats["labels"],
            y_pred=val_stats["predictions"],
            class_names=class_names,
        )

        history["train_loss"].append(train_stats["loss"])
        history["train_acc"].append(train_stats["accuracy"])
        history["val_loss"].append(val_stats["loss"])
        history["val_acc"].append(val_stats["accuracy"])
        history["val_balanced_acc"].append(val_metrics["balanced_accuracy"])
        history["val_precision_macro"].append(val_metrics["precision_macro"])
        history["val_recall_macro"].append(val_metrics["recall_macro"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["val_f1_weighted"].append(val_metrics["f1_weighted"])
        train_epoch_flops = None if train_batch_flops is None else train_batch_flops * train_stats["num_batches"]
        val_epoch_flops = None if val_batch_flops is None else val_batch_flops * val_stats["num_batches"]
        total_epoch_flops = None if (train_epoch_flops is None or val_epoch_flops is None) else (train_epoch_flops + val_epoch_flops)
        history["train_epoch_flops"].append(float("nan") if train_epoch_flops is None else float(train_epoch_flops))
        history["val_epoch_flops"].append(float("nan") if val_epoch_flops is None else float(val_epoch_flops))
        history["total_epoch_flops"].append(float("nan") if total_epoch_flops is None else float(total_epoch_flops))

        print(f"Train Loss: {train_stats['loss']:.4f} Acc: {train_stats['accuracy']:.4f}")
        print(f"Val Loss: {val_stats['loss']:.4f} Acc: {val_stats['accuracy']:.4f}")
        print(f"Val Metrics: {format_metrics_summary(val_metrics)}")
        if total_epoch_flops is None:
            print("Epoch FLOPs (estimate): unavailable")
        else:
            print(
                "Epoch FLOPs (estimate): "
                f"train={train_epoch_flops / 1e9:.2f} GFLOPs | "
                f"val={val_epoch_flops / 1e9:.2f} GFLOPs | "
                f"total={total_epoch_flops / 1e9:.2f} GFLOPs"
            )

        if stopper.step(val_stats["accuracy"], model):  # called every epoch
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    stopper.restore(model)

    final_val_stats = run_epoch(
        model, val_loader, None, None, cfg["grad_clip"], device, train=False, return_predictions=True
    )
    final_metrics = compute_classification_metrics(
        y_true=final_val_stats["labels"],
        y_pred=final_val_stats["predictions"],
        class_names=class_names,
    )

    # Save best checkpoint
    checkpoint_path = os.path.join(output_dir, f"{model_arch}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nSaved best checkpoint: {checkpoint_path}")

    # Sve metrics
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    np.save(os.path.join(metrics_dir, f"{model_arch}_val_confusion_matrix.npy"), final_metrics["confusion_matrix"])
    with open(os.path.join(metrics_dir, f"{model_arch}_val_classification_report.json"), "w") as f:
        json.dump(final_metrics["classification_report"], f, indent=2)
    with open(os.path.join(metrics_dir, f"{model_arch}_val_metrics_summary.txt"), "w") as f:
        f.write(format_metrics_summary(final_metrics) + "\n")
    print(f"Saved classification metrics to: {metrics_dir}")

    return model, dict(history)
