# models/videomae.py
# EEEM068 Action Recognition using  ViT
# Author: Ben Davison, (adapted from code by Prasanna Lamgade)
# Group members: Ben Davison, Chris Gainullin, Saba Ali, Youssef Abdelrahim
# SC: python -m models.videomae "./HMDB_simp"

import torch
from transformers import VideoMAEForVideoClassification


def load_videomae(num_classes: int,
    checkpoint: str) -> VideoMAEForVideoClassification:
    model_inst = VideoMAEForVideoClassification.from_pretrained(
        checkpoint,
        num_labels=num_classes,  # define the new head size
        ignore_mismatched_sizes=True  # allows head to be replaced without error
    )
    return model_inst


def count_parameters(model_inst) -> dict:
    total = 0
    for p in model_inst.parameters():
        total += p.numel()  # number of elements

    trainable = 0
    for p in model_inst.parameters():
        if p.requires_grad:  # parameter will be updated during training
            trainable += p.numel()
    return {"total": total, "trainable": trainable}