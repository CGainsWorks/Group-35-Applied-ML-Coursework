import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load confusion matrix
cm = np.load("outputs/metrics/timesformer_val_confusion_matrix.npy")

# Create heatmap
plt.figure(figsize=(10,8))
sns.heatmap(cm, cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Save figure
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/confusion_matrix_heatmap.png")
plt.close()

print("Confusion matrix heatmap saved to outputs/")