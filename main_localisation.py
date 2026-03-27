import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.data.jhmdb_dataloader import build_jhmdb_dataloaders
from src.model.localizer import VideoMAELocalizer
from src.eval.iou import box_iou

print("Running localisation script...")


def train_localizer(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for frames, boxes, labels, _ in loader:
        frames = frames.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        class_logits, preds = model(frames)

        cls_loss = nn.functional.cross_entropy(class_logits, labels)
        box_loss = nn.functional.mse_loss(preds, boxes)
        loss = cls_loss + box_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate_localizer(model, loader, device):
    model.eval()
    total_iou = 0.0
    total_frames = 0

    with torch.no_grad():
        for frames, boxes, labels, _ in loader:
            frames = frames.to(device)
            boxes = boxes.to(device)

            _, preds = model(frames)

            ious = box_iou(preds, boxes)  # [B, T]
            total_iou += ious.sum().item()
            total_frames += ious.numel()

    return total_iou / total_frames
def detection_accuracy(model, loader, device, threshold=0.5):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, boxes, labels, _ in loader:
            frames = frames.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            class_logits, preds = model(frames)

            pred_labels = class_logits.argmax(dim=1)   # [B]
            ious = box_iou(preds, boxes)               # [B, T]

            # class must be correct for the whole clip
            class_correct = pred_labels.eq(labels).unsqueeze(1).expand_as(ious)

            detections = (ious >= threshold) & class_correct

            correct += detections.sum().item()
            total += detections.numel()

    return correct / total

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = "./JHMDB"

    cfg = {
        "split_id": 1,
        "num_frames": 8,      # temporary
        "resolution": 224,    # temporary
        "batch_size": 2,
        "num_workers": 0,
    }

    model = VideoMAELocalizer(
        checkpoint="MCG-NJU/videomae-base-finetuned-kinetics",
        num_classes=21,
        num_frames=None
    ).to(device)

    cfg["num_frames"] = int(model.backbone.config.num_frames)
    image_size = model.backbone.config.image_size
    cfg["resolution"] = int(image_size[0] if isinstance(image_size, (tuple, list)) else image_size)

    train_loader, test_loader = build_jhmdb_dataloaders(root, cfg)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("Starting training...")
    print(f"Using num_frames={cfg['num_frames']}, resolution={cfg['resolution']}")

    for epoch in range(5):
        loss = train_localizer(model, train_loader, optimizer, device)
        test_iou = evaluate_localizer(model, test_loader, device)
        det_acc = detection_accuracy(model, test_loader, device, threshold=0.5)

        print(
            f"Epoch {epoch+1}: "
            f"Loss = {loss:.4f} | "
            f"Test IoU = {test_iou:.4f} | "
            f"Det Acc@0.5 = {det_acc:.4f}"
        )

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/localizer_videomae.pt")
    print("Saved trained localizer to outputs/localizer_videomae.pt")

    print("Done!")

if __name__ == "__main__":
    main()