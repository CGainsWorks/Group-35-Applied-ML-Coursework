# models/timesformer.py
# EEEM068 Action Recognition using  ViT
# Author: Prasanna Lamgade
# Group members: Ben Davison, Chris Gainullin, Saba Ali, Youssef Abdelrahim
# SC: python -m models.timesformer "./HMDB_simp"

import torch
from transformers import TimesformerForVideoClassification
from src.utils import count_parameters


def load_timesformer(num_classes: int,
    checkpoint: str) -> TimesformerForVideoClassification:
    model = TimesformerForVideoClassification.from_pretrained(
        checkpoint,
        num_labels=num_classes,  # define the new head size
        ignore_mismatched_sizes=True  # allows head to be replaced without error
    )
    return model


# Sanity check
if __name__ == "__main__":
    CHECKPOINT = "facebook/timesformer-base-finetuned-k600"
    NUM_CLASSES = 25
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {DEVICE}")
    print(f"Loading {CHECKPOINT}...")

    model = load_timesformer(NUM_CLASSES, CHECKPOINT)
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
