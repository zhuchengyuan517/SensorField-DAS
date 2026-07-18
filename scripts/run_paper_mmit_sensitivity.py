from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - runtime dependent
    raise RuntimeError("matplotlib is required to render sensitivity plots.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBMTL_ROOT = PROJECT_ROOT / "libmtl_das_patch"
EXAMPLE_ROOT = LIBMTL_ROOT / "examples" / "das_csv"
if str(LIBMTL_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBMTL_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from create_dataset_mmit import mmit_dataloader
from LibMTL.model import DASMultiModalNet, FocalLoss, compute_total_loss


EVENT_CLASSES = ("walking", "excavator", "driving", "background")


@dataclass
class SweepConfig:
    sweep_name: str
    value_name: str
    value: float
    lambda_value: float
    gamma_value: float
    alpha_power: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-aligned MMIT sensitivity sweeps for lambda/alpha/gamma.")
    parser.add_argument("--dataset_root", default=str(PROJECT_ROOT / "converted_csv" / "MTL43"), type=str)
    parser.add_argument("--output_root", default=str(PROJECT_ROOT / "_tmp_paper_mmit_sensitivity"), type=str)
    parser.add_argument("--train_ratio", default=0.25, type=float)
    parser.add_argument("--val_ratio", default=0.50, type=float)
    parser.add_argument("--test_ratio", default=0.50, type=float)
    parser.add_argument("--seed", default=20260613, type=int)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=96, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--raw_length", default=4096, type=int)
    parser.add_argument("--stft_size", default=128, type=int)
    parser.add_argument("--gaf_size", default=128, type=int)
    parser.add_argument("--d_model", default=128, type=int)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--gaf_patch_size", default=8, type=int)
    parser.add_argument("--gaf_depth", default=2, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--step_size", default=10, type=int)
    parser.add_argument("--gamma_decay", default=0.5, type=float)
    parser.add_argument("--lambda_values", default="0.0,0.05,0.1,0.2", type=str)
    parser.add_argument("--gamma_values", default="0.5,1.0,2.0,3.0,4.0", type=str)
    parser.add_argument("--alpha_powers", default="0.0,0.5,1.0,1.5", type=str)
    parser.add_argument("--alpha_mode", default="power", choices=["power", "class_weight"])
    parser.add_argument("--alpha_total", default=10800.0, type=float)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_float_list(payload: str) -> list[float]:
    if payload.strip().lower() in {"", "none", "skip"}:
        return []
    return [float(item.strip()) for item in payload.split(",") if item.strip()]


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stratified_subset(rows: list[dict[str, str]], ratio: float, seed: int) -> list[dict[str, str]]:
    if ratio >= 0.9999:
        return list(rows)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["event_label"]].append(row)
    rng = random.Random(seed)
    subset: list[dict[str, str]] = []
    for label in EVENT_CLASSES:
        label_rows = list(grouped[label])
        rng.shuffle(label_rows)
        keep = max(1, int(math.ceil(len(label_rows) * ratio)))
        subset.extend(label_rows[:keep])
    rng.shuffle(subset)
    return subset


def ensure_subset_manifests(
    dataset_root: Path,
    subset_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    subset_root.mkdir(parents=True, exist_ok=True)
    for split, ratio, split_seed in (
        ("train", train_ratio, seed + 11),
        ("val", val_ratio, seed + 17),
        ("test", test_ratio, seed + 23),
    ):
        src_rows = load_manifest(dataset_root / f"{split}.csv")
        dst_rows = stratified_subset(src_rows, ratio=ratio, seed=split_seed)
        write_manifest(subset_root / f"{split}.csv", dst_rows)


def compute_alpha_from_manifest(train_manifest: Path, alpha_power: float) -> torch.Tensor:
    rows = load_manifest(train_manifest)
    counts = np.zeros(len(EVENT_CLASSES), dtype=np.float64)
    label_to_idx = {label: idx for idx, label in enumerate(EVENT_CLASSES)}
    for row in rows:
        counts[label_to_idx[row["event_label"]]] += 1.0
    counts = np.maximum(counts, 1.0)
    if alpha_power <= 1e-12:
        weights = np.ones_like(counts)
    else:
        weights = np.power(1.0 / counts, alpha_power)
    weights = weights / max(weights.mean(), 1e-12)
    return torch.tensor(weights, dtype=torch.float32)


def compute_class_weight_alpha(dataset_root: Path, alpha_total: float) -> torch.Tensor:
    counts = np.zeros(len(EVENT_CLASSES), dtype=np.float64)
    label_to_idx = {label: idx for idx, label in enumerate(EVENT_CLASSES)}
    summary_path = dataset_root / "summary.csv"
    if summary_path.is_file():
        for row in load_manifest(summary_path):
            group = row.get("group", "").split("/", 1)[0]
            if group in label_to_idx:
                counts[label_to_idx[group]] += float(row["count"])
    else:
        manifest_path = dataset_root / "manifest.csv"
        for row in load_manifest(manifest_path):
            label = row.get("event_class", row.get("event_label", ""))
            if label in label_to_idx:
                counts[label_to_idx[label]] += 1.0
    counts = np.maximum(counts, 1.0)
    weights = float(alpha_total) / (len(EVENT_CLASSES) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_alpha_tensor(args: argparse.Namespace, subset_root: Path, alpha_power: float) -> torch.Tensor:
    if args.alpha_mode == "class_weight":
        return compute_class_weight_alpha(Path(args.dataset_root).expanduser().resolve(), args.alpha_total)
    return compute_alpha_from_manifest(subset_root / "train.csv", alpha_power)


def build_confusion_matrix(targets: list[int], predictions: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def classification_metrics(matrix: np.ndarray) -> dict[str, float]:
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    for idx in range(matrix.shape[0]):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - tp)
        fn = float(matrix[idx, :].sum() - tp)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)
    return {
        "accuracy": correct / max(total, 1),
        "macro_precision": float(np.mean(per_class_precision)) if per_class_precision else 0.0,
        "macro_recall": float(np.mean(per_class_recall)) if per_class_recall else 0.0,
        "macro_f1": float(np.mean(per_class_f1)) if per_class_f1 else 0.0,
    }


def run_epoch(
    model: DASMultiModalNet,
    loader,
    criterion: FocalLoss,
    lambda_value: float,
    optimizer: AdamW | None,
    device: torch.device,
    collect_predictions: bool,
) -> dict[str, float | list[int]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_cls = 0.0
    total_dec = 0.0
    total_cons = 0.0
    total_samples = 0
    targets_all: list[int] = []
    predictions_all: list[int] = []

    for batch_inputs, batch_labels in loader:
        raw = batch_inputs["raw"].to(device)
        stft = batch_inputs["stft"].to(device)
        gaf = batch_inputs["gaf"].to(device)
        targets = batch_labels["event_type"].to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(raw, stft, gaf)
            loss_out = compute_total_loss(outputs, targets, criterion, lambda1=lambda_value, lambda2=lambda_value)
            loss = loss_out.total_loss
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        predictions = outputs["logits"].argmax(dim=1)
        batch_size = raw.size(0)
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_cls += float(loss_out.cls_loss.item()) * batch_size
        total_dec += float(loss_out.disentanglement_loss.item()) * batch_size
        total_cons += float(loss_out.consistency_loss.item()) * batch_size
        if collect_predictions:
            targets_all.extend(targets.detach().cpu().tolist())
            predictions_all.extend(predictions.detach().cpu().tolist())

    metrics: dict[str, float | list[int]] = {
        "loss": total_loss / max(total_samples, 1),
        "cls_loss": total_cls / max(total_samples, 1),
        "dec_loss": total_dec / max(total_samples, 1),
        "cons_loss": total_cons / max(total_samples, 1),
    }
    if collect_predictions:
        matrix = build_confusion_matrix(targets_all, predictions_all, len(EVENT_CLASSES))
        metrics.update(classification_metrics(matrix))
        metrics["targets"] = targets_all
        metrics["predictions"] = predictions_all
    return metrics


def train_one_sweep(
    subset_root: Path,
    output_root: Path,
    sweep: SweepConfig,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    set_seed(args.seed)
    device = torch.device("cpu")
    alpha_tensor = compute_alpha_tensor(args, subset_root, sweep.alpha_power)
    loaders = mmit_dataloader(
        dataset_path=subset_root,
        batch_size=args.batch_size,
        event_classes=list(EVENT_CLASSES),
        raw_length=args.raw_length,
        stft_size=args.stft_size,
        gaf_size=args.gaf_size,
        sample_level="manifest",
        normalize="sample",
        num_workers=args.num_workers,
        train_sampler="event_balanced",
        augment=False,
        stft_n_fft=256,
        stft_hop_length=128,
        stft_win_length=256,
    )
    model = DASMultiModalNet(
        num_classes=len(EVENT_CLASSES),
        d_model=args.d_model,
        temporal_branch_dim=args.d_model,
        tf_branch_dim=args.d_model,
        gaf_patch_size=args.gaf_patch_size,
        gaf_depth=args.gaf_depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    criterion = FocalLoss(gamma=sweep.gamma_value, alpha=alpha_tensor)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma_decay)

    run_dir = output_root / sweep.sweep_name / sweep.value_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history_rows = []
    best_epoch = 0
    best_val_f1 = -1.0
    best_val_metrics: dict[str, float] | None = None
    best_test_metrics: dict[str, float] | None = None
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            lambda_value=sweep.lambda_value,
            optimizer=optimizer,
            device=device,
            collect_predictions=False,
        )
        val_metrics = run_epoch(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            lambda_value=sweep.lambda_value,
            optimizer=None,
            device=device,
            collect_predictions=True,
        )
        scheduler.step()

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history_rows.append(epoch_row)

        current_f1 = float(val_metrics["macro_f1"])
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_epoch = epoch
            best_val_metrics = {
                "loss": float(val_metrics["loss"]),
                "accuracy": float(val_metrics["accuracy"]),
                "macro_precision": float(val_metrics["macro_precision"]),
                "macro_recall": float(val_metrics["macro_recall"]),
                "macro_f1": float(val_metrics["macro_f1"]),
            }
            best_test_epoch_metrics = run_epoch(
                model=model,
                loader=loaders["test"],
                criterion=criterion,
                lambda_value=sweep.lambda_value,
                optimizer=None,
                device=device,
                collect_predictions=True,
            )
            best_test_metrics = {
                "loss": float(best_test_epoch_metrics["loss"]),
                "accuracy": float(best_test_epoch_metrics["accuracy"]),
                "macro_precision": float(best_test_epoch_metrics["macro_precision"]),
                "macro_recall": float(best_test_epoch_metrics["macro_recall"]),
                "macro_f1": float(best_test_epoch_metrics["macro_f1"]),
            }

    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps(history_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "sweep_name": sweep.sweep_name,
        "value_name": sweep.value_name,
        "value": sweep.value,
        "lambda_value": sweep.lambda_value,
        "gamma_value": sweep.gamma_value,
        "alpha_power": sweep.alpha_power,
        "alpha_mode": args.alpha_mode,
        "alpha_total": args.alpha_total,
        "alpha_vector": [round(float(item), 6) for item in alpha_tensor.tolist()],
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    result_row = {
        "sweep_name": sweep.sweep_name,
        "value_name": sweep.value_name,
        "value": sweep.value,
        "lambda_value": sweep.lambda_value,
        "gamma_value": sweep.gamma_value,
        "alpha_power": sweep.alpha_power,
        "alpha_mode": args.alpha_mode,
        "alpha_total": args.alpha_total,
        "best_epoch": best_epoch,
        "val_accuracy": best_val_metrics["accuracy"] if best_val_metrics else 0.0,
        "val_macro_precision": best_val_metrics["macro_precision"] if best_val_metrics else 0.0,
        "val_macro_recall": best_val_metrics["macro_recall"] if best_val_metrics else 0.0,
        "val_macro_f1": best_val_metrics["macro_f1"] if best_val_metrics else 0.0,
        "test_accuracy": best_test_metrics["accuracy"] if best_test_metrics else 0.0,
        "test_macro_precision": best_test_metrics["macro_precision"] if best_test_metrics else 0.0,
        "test_macro_recall": best_test_metrics["macro_recall"] if best_test_metrics else 0.0,
        "test_macro_f1": best_test_metrics["macro_f1"] if best_test_metrics else 0.0,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    print(
        f"[{sweep.sweep_name}] {sweep.value_name}: "
        f"val_f1={result_row['val_macro_f1']:.4f}, "
        f"test_f1={result_row['test_macro_f1']:.4f}, "
        f"time={result_row['elapsed_seconds']:.1f}s"
    )
    return result_row


def build_sweeps(args: argparse.Namespace) -> list[SweepConfig]:
    sweeps: list[SweepConfig] = []
    for value in parse_float_list(args.lambda_values):
        sweeps.append(
            SweepConfig(
                sweep_name="lambda",
                value_name=f"lambda_{str(value).replace('.', 'p')}",
                value=value,
                lambda_value=value,
                gamma_value=2.0,
                alpha_power=1.0,
            )
        )
    for value in parse_float_list(args.gamma_values):
        sweeps.append(
            SweepConfig(
                sweep_name="gamma",
                value_name=f"gamma_{str(value).replace('.', 'p')}",
                value=value,
                lambda_value=0.1,
                gamma_value=value,
                alpha_power=1.0,
            )
        )
    for value in parse_float_list(args.alpha_powers):
        sweeps.append(
            SweepConfig(
                sweep_name="alpha_power",
                value_name=f"alpha_{str(value).replace('.', 'p')}",
                value=value,
                lambda_value=0.1,
                gamma_value=2.0,
                alpha_power=value,
            )
        )
    return sweeps


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key, value in row.items():
                if isinstance(value, float):
                    formatted[key] = f"{value:.10f}"
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def render_sensitivity_plots(rows: list[dict[str, float | str]], output_root: Path) -> list[Path]:
    plot_paths: list[Path] = []
    for sweep_name in ("lambda", "gamma", "alpha_power"):
        sweep_rows = [row for row in rows if row["sweep_name"] == sweep_name]
        sweep_rows.sort(key=lambda item: float(item["value"]))
        x = [float(row["value"]) for row in sweep_rows]
        precision = [float(row["val_macro_precision"]) for row in sweep_rows]
        recall = [float(row["val_macro_recall"]) for row in sweep_rows]
        f1 = [float(row["val_macro_f1"]) for row in sweep_rows]

        plt.figure(figsize=(8, 5))
        plt.plot(x, precision, marker="o", linewidth=2, label="Macro Precision")
        plt.plot(x, recall, marker="s", linewidth=2, label="Macro Recall")
        plt.plot(x, f1, marker="^", linewidth=2, label="Macro F1")
        plt.xlabel(sweep_name)
        plt.ylabel("Validation metric")
        plt.title(f"Paper-model sensitivity: {sweep_name}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot_path = output_root / f"{sweep_name}_sensitivity.png"
        plt.savefig(plot_path, dpi=220)
        plt.close()
        plot_paths.append(plot_path)
    return plot_paths


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    subset_root = output_root / "subset_manifests"
    ensure_subset_manifests(
        dataset_root=dataset_root,
        subset_root=subset_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    meta = {
        "dataset_root": str(dataset_root),
        "subset_root": str(subset_root),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "event_classes": list(EVENT_CLASSES),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "experiment_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    sweeps = build_sweeps(args)
    results = []
    for sweep in sweeps:
        results.append(train_one_sweep(subset_root=subset_root, output_root=output_root / "runs", sweep=sweep, args=args))

    write_csv(output_root / "sensitivity_results.csv", results)
    (output_root / "sensitivity_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_paths = render_sensitivity_plots(results, output_root=output_root)
    (output_root / "plot_manifest.json").write_text(
        json.dumps({"plots": [str(path) for path in plot_paths]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved results to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
