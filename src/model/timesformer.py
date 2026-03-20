# models/timesformer.py
# EEEM068 Action Recognition using  ViT
# Author: Prasanna Lamgade
# Group members: Ben Davison, Chris Gainullin, Saba Ali, Youssef Abdelrahim
# SC: python -m models.timesformer "./HMDB_simp"

from transformers import TimesformerForVideoClassification


def load_timesformer(num_classes: int,
    checkpoint: str) -> TimesformerForVideoClassification:
    model = TimesformerForVideoClassification.from_pretrained(
        checkpoint,
        num_labels=num_classes,  # define the new head size
        ignore_mismatched_sizes=True,  # allows head to be replaced without error
        output_attentions=True
    )
    return model
