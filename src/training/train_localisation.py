import os
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


def train_localizer(model, train_loader, test_loader, cfg, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)

    cls_criterion = nn.CrossEntropyLoss()
    box_criterion = nn.SmoothL1Loss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    history = {
        "train_loss": [],
        "train_cls_loss": [],
        "train_box_loss": [],
        "test_loss": [],
        "test_cls_loss": [],
        "test_box_loss": [],
    }

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        total_loss, total_cls_loss, total_box_loss = 0.0, 0.0, 0.0
        num_batches = 0

        for frames, boxes, labels, meta in tqdm.tqdm(train_loader):
            frames = frames.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            class_logits, pred_boxes = model(frames)

            cls_loss = cls_criterion(class_logits, labels)
            box_loss = box_criterion(pred_boxes, boxes)
            loss = cls_loss + cfg["box_loss_weight"] * box_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_box_loss += box_loss.item()
            num_batches += 1

        history["train_loss"].append(total_loss / num_batches)
        history["train_cls_loss"].append(total_cls_loss / num_batches)
        history["train_box_loss"].append(total_box_loss / num_batches)

        model.eval()
        total_loss, total_cls_loss, total_box_loss = 0.0, 0.0, 0.0
        num_batches = 0

        with torch.no_grad():
            for frames, boxes, labels, meta in tqdm.tqdm(test_loader):
                frames = frames.to(device)
                boxes = boxes.to(device)
                labels = labels.to(device)

                class_logits, pred_boxes = model(frames)

                cls_loss = cls_criterion(class_logits, labels)
                box_loss = box_criterion(pred_boxes, boxes)
                loss = cls_loss + cfg["box_loss_weight"] * box_loss

                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_box_loss += box_loss.item()
                num_batches += 1

        history["test_loss"].append(total_loss / num_batches)
        history["test_cls_loss"].append(total_cls_loss / num_batches)
        history["test_box_loss"].append(total_box_loss / num_batches)

        print(f"\nEpoch {epoch}/{cfg['num_epochs']}")
        print(
            f"Train Loss: {history['train_loss'][-1]:.4f} | "
            f"Cls: {history['train_cls_loss'][-1]:.4f} | "
            f"Box: {history['train_box_loss'][-1]:.4f}"
        )
        print(
            f"Test Loss: {history['test_loss'][-1]:.4f} | "
            f"Cls: {history['test_cls_loss'][-1]:.4f} | "
            f"Box: {history['test_box_loss'][-1]:.4f}"
        )

    return model, history