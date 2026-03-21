import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def _to_numpy_1d(values):
    arr = np.asarray(values)
    return arr.reshape(-1)


def compute_classification_metrics(y_true, y_pred, class_names=None):
    y_true = _to_numpy_1d(y_true).astype(int)
    y_pred = _to_numpy_1d(y_pred).astype(int)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    if class_names is not None:
        labels = list(range(len(class_names)))
    else:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
        class_names = [str(v) for v in labels]

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "classification_report": report,
    }


def format_metrics_summary(metrics: dict) -> str:
    return (
        f"Top1: {metrics['accuracy']:.4f} | "
        f"Top5: {metrics['top5_accuracy']:.4f} | "
        f"Bal Acc: {metrics['balanced_accuracy']:.4f} | "
        f"P/R/F1 macro: {metrics['precision_macro']:.4f}/"
        f"{metrics['recall_macro']:.4f}/"
        f"{metrics['f1_macro']:.4f} | "
        f"F1 weighted: {metrics['f1_weighted']:.4f}"
    )
