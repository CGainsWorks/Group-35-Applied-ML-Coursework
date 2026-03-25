# utils.py

import os
import random
import numpy as np
import torch


def seed_everything(seed: int):
    """
    Fix all random seeds for reproducibility.
    Call this once at the start of m.py before any data loading or model init.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)

def count_parameters(model_inst) -> dict:
    total = 0
    for p in model_inst.parameters():
        total += p.numel()  # number of elements

    trainable = 0
    for p in model_inst.parameters():
        if p.requires_grad:  # parameter will be updated during training
            trainable += p.numel()
    return {"total": total, "trainable": trainable}