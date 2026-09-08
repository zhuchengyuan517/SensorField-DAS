from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - sklearn is optional in small smoke envs
    roc_auc_score = None


@dataclass(frozen=True)
class TaskMetricBundle:
    acc: float
    macro_f1: float
    auc: float
    far: float
    task_score: float
    confusion: np.ndarray
    per_class: list[dict[str, Any]]
    auc_defined: bool


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        target_idx = int(target)
        prediction_idx = int(prediction)
        if 0 <= target_idx < num_classes and 0 <= prediction_idx < num_classes:
            matrix[target_idx, prediction_idx] += 1
    return matrix


def _macro_auc(targets: np.ndarray, probabilities: np.ndarray | None, num_classes: int) -> tuple[float, bool]:
    if probabilities is None or probabilities.size == 0 or roc_auc_score is None:
        return 0.5, False
    if probabilities.ndim != 2 or probabilities.shape[1] != num_classes:
        return 0.5, False

    auc_values: list[float] = []
    for class_idx in range(num_classes):
        binary_targets = (targets == class_idx).astype(np.int32)
        if np.unique(binary_targets).size < 2:
            continue
        try:
            auc_values.append(float(roc_auc_score(binary_targets, probabilities[:, class_idx])))
        except Exception:
            continue
    if not auc_values:
        return 0.5, False
    return float(np.mean(auc_values)), True


def classification_metrics(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    probabilities: list[list[float]] | np.ndarray | None,
    class_names: list[str],
) -> TaskMetricBundle:
    targets_np = np.asarray(targets, dtype=np.int64)
    predictions_np = np.asarray(predictions, dtype=np.int64)
    probabilities_np = None if probabilities is None else np.asarray(probabilities, dtype=np.float64)
    num_classes = len(class_names)

    if targets_np.size == 0:
        matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        per_class = [
            {
                "class_index": idx,
                "class_name": name,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "far": 0.0,
                "support": 0,
            }
            for idx, name in enumerate(class_names)
        ]
        return TaskMetricBundle(
            acc=0.0,
            macro_f1=0.0,
            auc=0.5,
            far=0.0,
            task_score=0.375,
            confusion=matrix,
            per_class=per_class,
            auc_defined=False,
        )

    matrix = confusion_matrix(targets_np, predictions_np, num_classes)
    total = float(matrix.sum())
    acc = float(np.trace(matrix) / max(total, 1.0))
    per_class = []
    f1_values = []
    far_values = []
    for idx, class_name in enumerate(class_names):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - tp)
        fn = float(matrix[idx, :].sum() - tp)
        tn = float(total - tp - fp - fn)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        far = fp / max(fp + tn, 1.0)
        f1_values.append(f1)
        far_values.append(far)
        per_class.append(
            {
                "class_index": idx,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "far": far,
                "support": int(matrix[idx, :].sum()),
            }
        )

    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    far = float(np.mean(far_values)) if far_values else 0.0
    auc, auc_defined = _macro_auc(targets_np, probabilities_np, num_classes)
    task_score = float((acc + macro_f1 + auc + (1.0 - far)) / 4.0)
    if not math.isfinite(task_score):
        task_score = 0.0
    return TaskMetricBundle(
        acc=acc,
        macro_f1=macro_f1,
        auc=auc,
        far=far,
        task_score=task_score,
        confusion=matrix,
        per_class=per_class,
        auc_defined=auc_defined,
    )


def attach_task_metrics(
    metrics: dict[str, Any],
    *,
    event_classes: list[str],
    distance_classes: list[str],
) -> dict[str, Any]:
    event_bundle = classification_metrics(
        metrics.get("event_targets", []),
        metrics.get("event_predictions", []),
        metrics.get("event_probabilities", None),
        event_classes,
    )
    location_bundle = classification_metrics(
        metrics.get("location_targets", []),
        metrics.get("location_predictions", []),
        metrics.get("location_probabilities", None),
        distance_classes,
    )
    metrics.update(
        {
            "event_acc": event_bundle.acc,
            "event_macro_f1": event_bundle.macro_f1,
            "event_auc": event_bundle.auc,
            "event_far": event_bundle.far,
            "event_task_score": event_bundle.task_score,
            "event_auc_defined": event_bundle.auc_defined,
            "location_acc": location_bundle.acc,
            "location_macro_f1": location_bundle.macro_f1,
            "location_auc": location_bundle.auc,
            "location_far": location_bundle.far,
            "location_task_score": location_bundle.task_score,
            "location_auc_defined": location_bundle.auc_defined,
            "mtl_score": float((event_bundle.task_score + location_bundle.task_score) / 2.0),
        }
    )
    # Backward-compatible name used by existing runners; new code should prefer mtl_score.
    metrics["score"] = metrics["mtl_score"]
    metrics["_event_metric_bundle"] = event_bundle
    metrics["_location_metric_bundle"] = location_bundle
    return metrics


def metric_rows(prefix: str, bundle: TaskMetricBundle) -> list[dict[str, Any]]:
    rows = []
    for row in bundle.per_class:
        item = dict(row)
        item["task"] = prefix
        rows.append(item)
    return rows
