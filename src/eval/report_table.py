import json
import csv
import os
# per-class table
input_path = "outputs/metrics/timesformer_val_classification_report.json"
output_path = "outputs/metrics/timesformer_per_class_metrics.csv"

with open(input_path, "r") as f:
    report = json.load(f)

rows = []
for class_name, metrics in report.items():
    if class_name in ["accuracy", "macro avg", "weighted avg"]:
        continue
    rows.append([
        class_name,
        metrics.get("precision", ""),
        metrics.get("recall", ""),
        metrics.get("f1-score", ""),
        metrics.get("support", "")
    ])

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Precision", "Recall", "F1-score", "Support"])
    writer.writerows(rows)

print(f"Saved table to {output_path}")