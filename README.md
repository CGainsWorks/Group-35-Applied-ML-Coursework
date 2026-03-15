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

To download the datasets:
```bash
wget https://zenodo.org/records/7718549/files/HMDB_simp.zip                                                                                                 
unzip HMDB_simp.zip

wget http://files.is.tue.mpg.de/jhmdb/JHMDB_video.zip
wget  http://files.is.tue.mpg.de/jhmdb/splits.zip
wget http://files.is.tue.mpg.de/jhmdb/sub_splits.zip
wget http://files.is.tue.mpg.de/jhmdb/joint_positions.zip
unzip yougethepoint.zip
```