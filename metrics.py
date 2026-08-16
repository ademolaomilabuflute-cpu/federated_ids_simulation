"""
Evaluation helpers for the held-out global test set — produces the
per-class precision/recall/F1 breakdown the paper's Table 4 reports
(no prior version of this repository computed this; only an aggregate
weighted score was produced before).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader


def evaluate_global(model, test_loader: DataLoader, device: torch.device, class_names: list[str] | None = None):
    """
    Runs `model` (assumed already loaded with trained weights) over the
    full global test set and returns:
        - a dict of aggregate metrics (accuracy, macro/weighted P/R/F1, avg latency ms/flow)
        - a per-class table (list of dicts: class, precision, recall, f1, support)
    """
    model.eval()
    all_preds, all_targets = [], []

    start = time.perf_counter()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
    elapsed = time.perf_counter() - start

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    n = len(y_true)
    avg_latency_ms = (elapsed / n) * 1000 if n else 0.0
    accuracy = float((y_true == y_pred).mean()) if n else 0.0

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class_table = []
    for i, label in enumerate(labels):
        name = class_names[label] if class_names and label < len(class_names) else f"class_{label}"
        per_class_table.append({
            "class": name,
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
            "support": int(support[i]),
        })

    aggregate = {
        "accuracy": accuracy,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "avg_latency_ms": avg_latency_ms,
    }
    return aggregate, per_class_table


def print_report(aggregate: dict, per_class_table: list[dict]) -> None:
    print("\nPer-class results:")
    print(f"{'Class':<25}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    for row in per_class_table:
        print(f"{row['class']:<25}{row['precision']*100:>9.2f}%{row['recall']*100:>9.2f}%"
              f"{row['f1']*100:>9.2f}%{row['support']:>10d}")

    print("\nAggregate results:")
    print(f"  Accuracy         : {aggregate['accuracy']*100:.2f}%")
    print(f"  Macro F1         : {aggregate['macro_f1']*100:.2f}%")
    print(f"  Weighted F1      : {aggregate['weighted_f1']*100:.2f}%")
    print(f"  Avg latency/flow : {aggregate['avg_latency_ms']:.3f} ms")
