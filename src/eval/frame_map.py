import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.data.jhmdb_dataloader import build_jhmdb_dataloaders
from src.model.localizer import VideoMAELocalizer
from src.eval.iou import box_iou
from src.data.jhmdb_dataset import JHMDBDataset

def compute_frame_map(model, loader, device, num_classes, iou_threshold=0.5):
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
            ious = box_iou(preds, boxes)                    # [B, T]

            B, T = ious.shape

            for b in range(B):
                true_class = labels[b].item()

                for t in range(T):
                    for c in range(num_classes):
                        # confidence score for class c
                        score = probs[b, c].item()

                        # positive only if:
                        # - class c is the true class
                        # - predicted class is correct
                        # - IoU threshold met
                        is_tp = int(
                            c == true_class and
                            pred_labels[b].item() == true_class and
                            ious[b, t].item() >= iou_threshold
                        )

                        class_scores[c].append(score)
                        class_targets[c].append(is_tp)

    ap_per_class = {}

    for c in range(num_classes):
        y_true = np.array(class_targets[c])
        y_score = np.array(class_scores[c])

        # average_precision_score needs at least one positive
        if y_true.sum() == 0:
            ap_per_class[c] = float("nan")
        else:
            ap_per_class[c] = average_precision_score(y_true, y_score)

    valid_aps = [v for v in ap_per_class.values() if not np.isnan(v)]
    frame_map = float(np.mean(valid_aps)) if valid_aps else float("nan")

    return frame_map, ap_per_class


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

    frame_map, ap_per_class = compute_frame_map(
        model, test_loader, device, num_classes=num_classes, iou_threshold=0.5
    )

    print(f"Frame-level mAP@0.5 = {frame_map:.4f}")

    for c, ap in ap_per_class.items():
        print(f"Class {c}: AP = {ap:.4f}")

    dataset = JHMDBDataset(root=root, split="test", split_id=1, num_frames=16)
    class_names = dataset.classes

    items = [(class_names[c], ap) for c, ap in ap_per_class.items() if not np.isnan(ap)]
    items.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 5 classes:")
    for name, ap in items[:5]:
        print(f"{name}: {ap:.4f}")

    print("\nBottom 5 classes:")
    for name, ap in items[-5:]:
        print(f"{name}: {ap:.4f}")

if __name__ == "__main__":
    main()