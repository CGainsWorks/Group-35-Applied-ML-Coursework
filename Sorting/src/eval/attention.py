# evaluation/attention.py
# EEEM068 Action Recognition using  ViT
# Author: Prasanna Lamgade
# Group members: Ben Davison, Chris Gainullin, Saba Ali, Youssef Abdelrahim

import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

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

# Visualisation
def denormalise_frame(frame_tensor):
    # Denormalise frames for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) # ImageNet Mean and reshapes
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) # Imagenet std
    frame = (frame_tensor.cpu() * std + mean).clamp(min=0, max=1)

    return frame.permute(1, 2, 0).cpu().numpy() # reorder (H, W, 3)

def attention_on_frame(frame_tensor, attn_map, alpha=0.5):
    frame_np = denormalise_frame(frame_tensor)
    H, W = frame_np.shape[:2]

    attn_pil = Image.fromarray((attn_map * 255).astype(np.uint8))
    attn_resized = np.array(attn_pil.resize((W, H), Image.BILINEAR)) / 255.0 # PIL takes (W, H) not (H, W)

    # Heatmap
    heatmap = cm.jet(attn_resized)[:,:,:3] #drop alpha

    # blend frame and heat map
    blended = (1 - alpha) * frame_np + alpha * heatmap
    return np.clip(blended, 0, 1)

def visualise_spatial_attention(model,frames, class_names, class_idx, device,
                                output_dir, num_layers=4, model_name="timesformer"):
    attentions = extract_attentions(model, frames, device)
    num_frames = frames.shape[1]

    # Use the middle frame as the background
    mid_frame = frames[0, num_frames // 2]  # (C, H, W)

    total_layers = len(attentions)
    layers_to_show = np.linspace(0, total_layers - 1, num_layers, dtype=int)

    fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 4))
    if num_layers == 1:
        axes = [axes]

    for ax, layer_idx in zip(axes, layers_to_show):
        attn_map = get_spatial_attention_map(attentions, layer_idx, num_frames)
        blended = attention_on_frame(mid_frame, attn_map, alpha=0.5)
        ax.imshow(blended)
        ax.set_title(f"Layer {layer_idx + 1}", fontsize=10)
        ax.axis("off")

    if class_names:
        label_name = class_names[class_idx]
    else:
        label_name = str(class_idx)

    fig.suptitle(f"Spatial Attention: '{label_name}'", fontsize=13)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}_spatial_attention_{label_name}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved spatial attention: {path}")

def visualise_temporal_attention(model, frames, class_names, class_idx,
                                 device, output_dir, layer_idx=-1, model_name="timesformer"):
    attentions = extract_attentions(model, frames, device)
    num_frames = frames.shape[1]
    layer_idx = layer_idx % len(attentions)  # handle negative index

    temporal_weights = get_temporal_attention(attentions, layer_idx, num_frames)

    fig, axes = plt.subplots(2, num_frames, figsize=(2.5 * num_frames, 5))

    for t in range(num_frames):
        frame_np = denormalise_frame(frames[0, t])

        # Top row: frames
        axes[0, t].imshow(frame_np)
        axes[0, t].set_title(f"t={t + 1}", fontsize=8)
        axes[0, t].axis("off")

        # Bottom row: bar showing attention weight for this frame
        color = plt.cm.Reds(temporal_weights[t])
        axes[1, t].bar(0, temporal_weights[t], color=color, width=0.6)
        axes[1, t].set_ylim(0, 1)
        axes[1, t].set_xlim(-0.5, 0.5)
        axes[1, t].set_xticks([])
        axes[1, t].set_ylabel("Attn" if t == 0 else "", fontsize=7)
        axes[1, t].tick_params(axis="y", labelsize=6)

    if class_names:
        label_name = class_names[class_idx]
    else:
        label_name =  str(class_idx)

    fig.suptitle(f"Temporal Attention: '{label_name}' (Layer {layer_idx + 1})", fontsize=13)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}_temporal_attention_{label_name}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved temporal attention: {path}")
