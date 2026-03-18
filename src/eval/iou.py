import torch

def box_iou(boxes1, boxes2):
    """
    boxes1, boxes2: (..., 4) with [x1, y1, x2, y2]
    returns: (...) IoU
    """
    x1 = torch.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.minimum(boxes1[..., 3], boxes2[..., 3])

    inter_w = torch.clamp(x2 - x1, min=0)
    inter_h = torch.clamp(y2 - y1, min=0)
    inter = inter_w * inter_h

    area1 = torch.clamp(boxes1[..., 2] - boxes1[..., 0], min=0) * torch.clamp(boxes1[..., 3] - boxes1[..., 1], min=0)
    area2 = torch.clamp(boxes2[..., 2] - boxes2[..., 0], min=0) * torch.clamp(boxes2[..., 3] - boxes2[..., 1], min=0)

    union = area1 + area2 - inter + 1e-8
    return inter / union