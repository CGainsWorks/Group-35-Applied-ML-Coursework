import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
args = parser.parse_args()

model = args.model

cm = np.load(f"outputs/metrics/{model}_val_confusion_matrix.npy")

plt.figure(figsize=(10,8))
sns.heatmap(cm, cmap="Blues")

plt.title(f"{model} Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

os.makedirs("outputs", exist_ok=True)

output_file = f"outputs/{model}_confusion_matrix_heatmap.png"
plt.savefig(output_file)
plt.close()

print("Saved:", output_file)