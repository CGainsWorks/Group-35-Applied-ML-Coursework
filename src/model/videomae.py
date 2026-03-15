from transformers import VideoMAEForVideoClassification


def load_videomae(num_classes: int,
    checkpoint: str) -> VideoMAEForVideoClassification:
    model_inst = VideoMAEForVideoClassification.from_pretrained(
        checkpoint,
        num_labels=num_classes,  # define the new head size
        ignore_mismatched_sizes=True  # allows head to be replaced without error
    )
    return model_inst
