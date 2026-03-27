import numpy as np
import torch
import os
from sklearn.metrics import average_precision_score
from src.data.jhmdb_dataloader import build_jhmdb_dataloaders
from src.data.jhmdb_dataset import JHMDBDataset
from src.model.localizer import VideoMAELocalizer
from src.eval.iou import box_iou


def compute_video_map(model, loader, device, num_classes, iou_threshold=0.3):
    model.eval()

    class_scores = {c: [] for c in range(num_classes)}
    class_targets = {c: [] for c in range(num_classes)}

    with torch.no_grad():
        for frames, boxes, labels, _ in loader:
            frames = frames.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            class_logits, preds = model(frames)

            probs = torch.softmax(class_logits, dim=1)      # [B, C]
            pred_labels = class_logits.argmax(dim=1)        # [B]
            frame_ious = box_iou(preds, boxes)              # [B, T]

            # simple tube IoU = mean IoU across sampled frames
            tube_ious = frame_ious.mean(dim=1)              # [B]

            B = tube_ious.shape[0]

            for b in range(B):
                true_class = labels[b].item()
                pred_class = pred_labels[b].item()

                for c in range(num_classes):
                    score = probs[b, c].item()

                    is_tp = int(
                        c == true_class and
                        pred_class == true_class and
                        tube_ious[b].item() >= iou_threshold
                    )

                    class_scores[c].append(score)
                    class_targets[c].append(is_tp)

    ap_per_class = {}

    for c in range(num_classes):
        y_true = np.array(class_targets[c])
        y_score = np.array(class_scores[c])

        if y_true.sum() == 0:
            ap_per_class[c] = float("nan")
        else:
            ap_per_class[c] = average_precision_score(y_true, y_score)

    valid_aps = [v for v in ap_per_class.values() if not np.isnan(v)]
    video_map = float(np.mean(valid_aps)) if valid_aps else float("nan")

    return video_map, ap_per_class


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = {
        "split_id": 1,
        "num_frames": 16,
        "resolution": 224,
        "batch_size": 2,
        "num_workers": 0,
    }

    root = "./JHMDB"
    num_classes = 21

    _, test_loader = build_jhmdb_dataloaders(root, cfg)

    model = VideoMAELocalizer(
        checkpoint="MCG-NJU/videomae-base-finetuned-kinetics",
        num_classes=num_classes,
        num_frames=None
    ).to(device)

    model.load_state_dict(torch.load("outputs/localizer_videomae.pt", map_location=device))
    model.eval()

    video_map, ap_per_class = compute_video_map(
        model, test_loader, device, num_classes=num_classes, iou_threshold=0.5
    )

    print(f"Video-level mAP@0.5 = {video_map:.4f}")

    dataset = JHMDBDataset(root=root, split="test", split_id=1, num_frames=16)
    class_names = dataset.classes

    items = [(class_names[c], ap) for c, ap in ap_per_class.items() if not np.isnan(ap)]
    items.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 5 video-AP classes:")
    for name, ap in items[:5]:
        print(f"{name}: {ap:.4f}")

    print("\nBottom 5 video-AP classes:")
    for name, ap in items[-5:]:
        print(f"{name}: {ap:.4f}")

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/C2_video_map_results.txt", "w") as f:
        f.write(f"Video-level mAP@0.5 = {video_map:.4f}\n\n")

        f.write("Top 5 video-AP classes:\n")
        for name, ap in items[:5]:
            f.write(f"{name}: {ap:.4f}\n")

        f.write("\nBottom 5 video-AP classes:\n")
        for name, ap in items[-5:]:
            f.write(f"{name}: {ap:.4f}\n")

if __name__ == "__main__":
    main()