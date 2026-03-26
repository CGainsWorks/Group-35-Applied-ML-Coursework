import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
args = parser.parse_args()

model = args.model

input_path = f"outputs/metrics/{model}_val_classification_report.json"
output_path = f"outputs/C1_{model}_per_class_metrics.csv"

with open(input_path, "r") as f:
    report = json.load(f)

rows = []

for class_name, metrics in report.items():
    if class_name in ["accuracy", "macro avg", "weighted avg"]:
        continue

    rows.append({
        "Class": class_name,
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1-score": metrics["f1-score"],
        "Support": metrics["support"]
    })

df = pd.DataFrame(rows)
df.to_csv(output_path, index=False)

print("Saved per-class metrics table to:", output_path)
print(df)