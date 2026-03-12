# models/videomae.py
# EEEM068 Action Recognition using  ViT
# Author: Ben Davison, (adapted from code by Prasanna Lamgade)
# Group members: Ben Davison, Chris Gainullin, Saba Ali, Youssef Abdelrahim
# SC: python -m models.videomae "./HMDB_simp"

import torch
from transformers import VideoMAEForVideoClassification


def load_videomae(num_classes: int,
    checkpoint: str) -> VideoMAEForVideoClassification:
    model = VideoMAEForVideoClassification.from_pretrained(
        checkpoint,
        num_labels=num_classes,  # define the new head size
        ignore_mismatched_sizes=True  # allows head to be replaced without error
    )
    return model


def count_parameters(model) -> dict:
    total = 0
    for p in model.parameters():
        total = + p.numel()  # number of elements

    trainable = 0
    for p in model.parameters():
        if p.requires_grad:  # parameter will be updated during training
            trainable = + p.numel()
    return {"total": total, "trainable": trainable}


# Sanity check
if __name__ == "__main__":
    CHECKPOINT = "MCG-NJU/videomae-base-finetuned-kinetics"
    NUM_CLASSES = 25
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {DEVICE}")
    print(f"Loading {CHECKPOINT}...")

    model = load_videomae(NUM_CLASSES, CHECKPOINT)
    model = model.to(DEVICE)
    model.eval()

    # Parameter count
    params = count_parameters(model)
    print(f"\nParameters:")
    print(f"Total: {params['total']:,}")
    print(f"Trainable : {params['trainable']:,}")

    # Check the classification head has been swapped correctly
    print(
        f"\nClassifier output features: {model.classifier.out_features}")  # expect 25

    # Single forward pass with a dummy batch
    # Shape: (B, T, C, H, W) matches DataLoader output
    dummy = torch.zeros(2, 8, 3, 224, 224).to(DEVICE)
    print(f"\nRunning forward pass with dummy input {list(dummy.shape)}...")

    with torch.no_grad():
        out = model(pixel_values=dummy)

    print(f"Output logits shape: {out.logits.shape}")  # expect (2, 25)
    print(f"Logits sample: {out.logits[0].tolist()[:5]}...")  # first 5 logits
