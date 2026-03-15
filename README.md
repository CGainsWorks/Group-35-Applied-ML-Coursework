# Project Structure

```text
.
├── data/              # Data loading and dataset definitions
│   ├── dataloader.py
│   └── dataset.py
├── eval/              # Evaluation scripts and utilities
│   └── attention.py
├── HMDB_simp/         # HMDB needs to be here
├── model/             # Model architectures
│   ├── __init__.py
│   ├── timesformer.py
│   └── videomae.py
├── outputs/           # Outputs can go here
├── training/          # Training and ablation scripts
│   ├── ablation.py
│   └── train.py
├── train.py            # Main entry point training
├── utils.py           # Utility functions
```