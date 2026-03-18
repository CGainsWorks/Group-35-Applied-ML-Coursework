import torch
import torch.nn as nn
from transformers import VideoMAEModel


class VideoMAELocalizer(nn.Module):
    def __init__(self, checkpoint: str, num_classes: int, num_frames=None):
        super().__init__()

        self.backbone = VideoMAEModel.from_pretrained(checkpoint)
        self.num_frames = int(self.backbone.config.num_frames) if num_frames is None else num_frames

        hidden_size = self.backbone.config.hidden_size

        self.class_head = nn.Linear(hidden_size, num_classes)
        self.box_head = nn.Linear(hidden_size, self.num_frames * 4)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        features = outputs.last_hidden_state.mean(dim=1)

        class_logits = self.class_head(features)

        batch_size = pixel_values.size(0)
        pred_boxes = self.box_head(features).view(batch_size, self.num_frames, 4)

        return class_logits, pred_boxes