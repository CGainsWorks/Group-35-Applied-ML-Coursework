import os
import cv2
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from src.data.jhmdb_dataloader import build_jhmdb_dataloaders
from src.model.localizer import VideoMAELocalizer
from src.data.jhmdb_dataset import JHMDBDataset


def draw_box(img, box, color, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    img = img.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def get_class_color(class_idx):
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
    ]
    return colors[class_idx % len(colors)]


def denormalize_box(box, width, height):
    box = box.clone()
    box[[0, 2]] *= width
    box[[1, 3]] *= height
    return box


def denormalize_frame(frame_tensor):
   
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=frame_tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=frame_tensor.dtype).view(3, 1, 1)
    frame_tensor = frame_tensor * std + mean
    frame_tensor = torch.clamp(frame_tensor, 0.0, 1.0)
    return frame_tensor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = {
        "split_id": 1,
        "num_frames": 16,
        "resolution": 224,
        "batch_size": 1,
        "num_workers": 0,
    }

    root = "./JHMDB"

    _, test_loader = build_jhmdb_dataloaders(root, cfg)
    dataset = JHMDBDataset(root=root, split="test", split_id=1, num_frames=16)
    class_names = dataset.classes

    model = VideoMAELocalizer(
        checkpoint="MCG-NJU/videomae-base-finetuned-kinetics",
        num_classes=21,
        num_frames=None
    ).to(device)

    model.load_state_dict(
        torch.load("outputs/localizer_videomae.pt", map_location=device)
    )
    model.eval()

    os.makedirs("outputs/localisation_vis", exist_ok=True)

    for sample_idx, batch in enumerate(test_loader):
        if sample_idx >= 3:
            break

        frames, gt_boxes, labels, meta = batch

        frames = frames.to(device)
        gt_boxes = gt_boxes.to(device)

        with torch.no_grad():
            class_logits, pred_boxes = model(frames)
            pred_label = class_logits.argmax(dim=1)[0].item()
            gt_label = labels[0].item()

        frames = frames[0].cpu()
        gt_boxes = gt_boxes[0].cpu()
        pred_boxes = pred_boxes[0].cpu()

        original_imgs = []
        overlay_imgs = []
        to_pil = T.ToPILImage()

        for frame_idx in range(frames.shape[0]):
            # Convert normalized tensor back to natural-looking RGB image
            frame_vis = denormalize_frame(frames[frame_idx])

            img = to_pil(frame_vis)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            h, w = img.shape[:2]

            gt = denormalize_box(gt_boxes[frame_idx], w, h)
            pred = denormalize_box(pred_boxes[frame_idx], w, h)

            # Top row: original frame
            orig_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
            original_imgs.append(orig_rgb)

            # Bottom row: overlayed result
            overlay = img.copy()
            overlay = draw_box(overlay, gt, (0, 255, 0), 2)  # Green = GT
            pred_color = get_class_color(pred_label)         # Colored = prediction class
            overlay = draw_box(overlay, pred, pred_color, 2)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            overlay_imgs.append(overlay_rgb)

        fig, axes = plt.subplots(2, len(original_imgs), figsize=(3 * len(original_imgs), 6))

        for i in range(len(original_imgs)):
            axes[0, i].imshow(original_imgs[i])
            axes[0, i].axis("off")
            axes[0, i].set_title(f"F{i+1}", fontsize=8)

            axes[1, i].imshow(overlay_imgs[i])
            axes[1, i].axis("off")

        gt_name = class_names[gt_label]
        pred_name = class_names[pred_label]
        video_name = meta["video_name"][0]

        title = (
            f"C2 Qualitative Localisation - Sample {sample_idx+1}\n"
            f"Video: {video_name}\n"
            f"GT Class: {gt_name} | Predicted Class: {pred_name} | "
            f"Green = GT Box | Colored = Predicted Box"
        )
        fig.suptitle(title, fontsize=11)

        plt.tight_layout(rect=[0, 0, 1, 0.88])

        save_path = (
            f"outputs/localisation_vis/"
            f"C2_sample_{sample_idx+1}_{pred_name}_qualitative.png"
        )
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved visualisation to {save_path}")


if __name__ == "__main__":
    main()