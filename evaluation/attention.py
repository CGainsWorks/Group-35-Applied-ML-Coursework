# evaluation/attention.py
# EEEM068 Action Recognition using  ViT
# Author: Prasanna Lamgade
# Group members: Ben Davison, Chris Gainullin, Saba Ali, Youssef Abdelrahim

import os
import math
import numpy as np
import torch

# Attention Extraction
@torch.no_grad()
def extract_attentions(model, frames, device):
    model.eval()
    frames = frames.to(device)
    out = model(pixel_values=frames, output_attentions=True)
    return out.attentions

def get_spatial_attention_map(attentions, layer_idx, num_frames, patch_grid=14):
    attn = attentions[layer_idx] # (1, attn. heads, seq_len, seq_len)
    attn = attn[0].mean(0) # one averaged over heads matrix -> (seq_len, seq_len)

    # CLS token is at position 0
    # Attention over all other tokens
    cls_attn = attn[0, 1:] # (seq_len - 1,), exclude CLS attending to itself

    num_patches = patch_grid * patch_grid # 14 * 14 = 196 spatial patches per frame
    total_tokens = cls_attn.shape[0] # total number of tokens in the sequence after removing the CLS token

    if total_tokens >= num_frames * num_patches:
        # Reshape to (T, patch_grid, patch_grid) and average over time
        spatial = cls_attn[:num_frames * num_patches]
        spatial = spatial.reshape(num_frames, patch_grid, patch_grid)
        attn_map = spatial.mean(0).cpu().numpy()
    else:
        # Fallback
        side = int(math.isqrt(total_tokens))
        attn_map = cls_attn[:side * side].reshape(side, side).cpu().numpy()

    # Normalise to [0, 1]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return attn_map

def get_temporal_attention(attentions, layer_idx, num_frames, patch_grid=14):
    attn = attentions[layer_idx]
    attn = attn[0].mean(0)
    cls_attn = attn[0, 1:]

    total_tokens = cls_attn.shape[0]
    num_patches  = patch_grid * patch_grid

    if total_tokens >= num_frames * num_patches:
        spatial = cls_attn[:num_frames * num_patches]
        spatial = spatial.reshape(num_frames, num_patches)
        temporal_weights = spatial.mean(1).cpu().numpy()
    elif total_tokens == num_frames:
        temporal_weights = cls_attn.cpu().numpy()
    else:
        chunk = total_tokens // num_frames
        if chunk > 0:
            temporal_weights = np.array([
                cls_attn[i * chunk:(i + 1) * chunk].mean().item()
                for i in range(num_frames)
            ])
        else:
            temporal_weights = np.ones(num_frames) / num_frames

    # Normalise to [0, 1]
    rng = temporal_weights.max() - temporal_weights.min()
    if rng > 1e-8:
        temporal_weights = (temporal_weights - temporal_weights.min()) / rng
    else:
        temporal_weights = np.ones(num_frames) / num_frames

    return temporal_weights
