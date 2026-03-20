import os
import cv2
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from src.data.jhmdb_dataloader import build_jhmdb_dataloaders
from src.model.localizer import VideoMAELocalizer


def draw_box(img, box, color, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    img = img.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def denormalize_box(box, width, height):
    box = box.clone()
    box[[0, 2]] *= width
    box[[1, 3]] *= height
    return box


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
            _, pred_boxes = model(frames)

        frames = frames[0].cpu()
        gt_boxes = gt_boxes[0].cpu()
        pred_boxes = pred_boxes[0].cpu()

        imgs = []
        to_pil = T.ToPILImage()

        for frame_idx in range(frames.shape[0]):
            img = to_pil(frames[frame_idx])
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            h, w = img.shape[:2]

            gt = denormalize_box(gt_boxes[frame_idx], w, h)
            pred = denormalize_box(pred_boxes[frame_idx], w, h)

            img = draw_box(img, gt, (0, 255, 0), 2)
            img = draw_box(img, pred, (0, 0, 255), 2)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        fig, axes = plt.subplots(1, len(imgs), figsize=(3 * len(imgs), 3))
        for ax, img in zip(axes, imgs):
            ax.imshow(img)
            ax.axis("off")

        title = f"{meta['class_name'][0]} | {meta['video_name'][0]}"
        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(f"outputs/localisation_vis/sample_{sample_idx+1}.png")
        plt.close()

        print(f"Saved visualisation to outputs/localisation_vis/sample_{sample_idx+1}.png")


if __name__ == "__main__":
    main()