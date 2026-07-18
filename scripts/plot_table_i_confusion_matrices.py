"""Plot Table-I-aligned confusion matrices for Event and Location tasks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_current8_model_tsne import (  # noqa: E402
    DEFAULT_RUNS,
    build_loaders,
    build_sensorfield_model,
    load_checkpoint,
    parse_label_list,
    resolve_device,
    to_device_hybrid,
)
from create_dataset import DISTANCE_IGNORE_INDEX  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "paper_assets" / "sensorfield_m3t_experiments" / "confusion_table_i"
DEFAULT_TABLE_CSV = Path(r"C:\Users\SUTD\Desktop\final.csv")

MODEL_ROWS = [
    ("ConvNeXt-Small", "ConvNeXt"),
    ("MultiModN", "MultiModN"),
    ("M4oE", "M4oE"),
    ("DAS-MAE + downstream fine-tuning head", "DAS-MAE"),
    ("PipelineADWinT", "PipelineADWinT"),
    ("Aligned-MTL", "Aligned-MTL"),
    ("MoCo-weighting", "MoCo-MTL"),
    ("SensorField-M3T", "SensorField-M3T"),
]

EVENT_CLASSES = ["walking", "excavator", "driving", "background"]
LOCATION_CLASSES = ["Alarm area", "Tracking area", "No-threat area"]
EVENT_SHORT = ["Walk", "Excav.", "Drive", "BG"]
LOCATION_SHORT = ["Alarm", "Track", "No-threat"]


def setup_style() -> None:
    candidates = ["Times New Roman", "Cambria", "Georgia", "STIXGeneral", "DejaVu Serif"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 13.0,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "figure.dpi": 240,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
        }
    )


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_confusion_csv(path: Path) -> tuple[list[str], np.ndarray]:
    labels = []
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        pred_labels = header[1:]
        for row in reader:
            labels.append(row[0])
            rows.append([int(float(value)) for value in row[1:]])
    if labels != pred_labels:
        # Keep the row order; some files use "label" vs "true/pred" only in header.
        pass
    return labels, np.asarray(rows, dtype=np.int64)


def write_confusion_csv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[int(value) for value in row]])


def build_confusion(targets: list[int], predictions: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(targets, predictions):
        matrix[int(target), int(pred)] += 1
    return matrix


def compute_sensorfield_test_confusions(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path]:
    run_dir = DEFAULT_RUNS["SensorField-M3T"].resolve()
    checkpoint = run_dir / "best.pt"
    config = load_json(run_dir / "run_config.json")
    selected = load_json(run_dir / "selected_best_blend_eval.json").get("selected_blend", {})
    if selected:
        config.update(selected)

    device = resolve_device(args.gpu_id)
    event_classes = parse_label_list(args.event_classes)
    distance_classes = parse_label_list(args.distance_classes)
    hybrid_loader, _, _ = build_loaders(args, event_classes, distance_classes)

    model = build_sensorfield_model(config, event_classes, distance_classes).to(device)
    load_checkpoint(model, checkpoint, device)
    if selected:
        model.image_event_expert_weight = float(selected.get("image_event_expert_weight", model.image_event_expert_weight))
        model.image_location_expert_weight = float(
            selected.get("image_location_expert_weight", model.image_location_expert_weight)
        )
    model.eval()

    event_targets, event_preds = [], []
    location_targets, location_preds = [], []
    with torch.no_grad():
        for batch_inputs, batch_labels in hybrid_loader:
            batch_inputs, batch_labels = to_device_hybrid(batch_inputs, batch_labels, device)
            outputs = model(batch_inputs)
            event_logits = outputs["event_type"]
            event_targets.extend(batch_labels["event_type"].detach().cpu().tolist())
            event_preds.extend(torch.argmax(event_logits, dim=1).detach().cpu().tolist())

            loc_targets = batch_labels["distance_cls"]
            valid_mask = loc_targets != DISTANCE_IGNORE_INDEX
            if valid_mask.any():
                loc_logits = outputs["distance_cls"][valid_mask]
                location_targets.extend(loc_targets[valid_mask].detach().cpu().tolist())
                location_preds.extend(torch.argmax(loc_logits, dim=1).detach().cpu().tolist())

    event_matrix = build_confusion(event_targets, event_preds, len(event_classes))
    location_matrix = build_confusion(location_targets, location_preds, len(distance_classes))
    event_path = output_dir / "sensorfield_m3t_best_test_event_confusion.csv"
    location_path = output_dir / "sensorfield_m3t_best_test_location_confusion.csv"
    write_confusion_csv(event_path, event_classes, event_matrix)
    write_confusion_csv(location_path, distance_classes, location_matrix)
    return event_path, location_path


def collect_confusions(args: argparse.Namespace, output_dir: Path) -> dict[str, dict[str, tuple[list[str], np.ndarray, Path]]]:
    sensor_event_path, sensor_location_path = compute_sensorfield_test_confusions(args, output_dir)
    confusions: dict[str, dict[str, tuple[list[str], np.ndarray, Path]]] = {}
    for model_key, _ in MODEL_ROWS:
        if model_key == "SensorField-M3T":
            event_path = sensor_event_path
            location_path = sensor_location_path
        else:
            history = DEFAULT_RUNS[model_key].resolve() / "history"
            event_path = history / "best_test_event_confusion.csv"
            location_path = history / "best_test_location_confusion.csv"
        event_labels, event_matrix = read_confusion_csv(event_path)
        location_labels, location_matrix = read_confusion_csv(location_path)
        confusions[model_key] = {
            "event": (event_labels, event_matrix, event_path),
            "location": (location_labels, location_matrix, location_path),
        }
    return confusions


def parse_metric_mean(value: object) -> float | None:
    """Parse metric cells such as '0.8733 ± 0.0305' into the mean value."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Â±", "±")
    mean_text = text.split("±", maxsplit=1)[0].strip()
    try:
        return float(mean_text)
    except ValueError:
        return None


def load_table_accs(table_csv: Path) -> dict[tuple[str, str], float]:
    if not table_csv.is_file():
        return {}
    rows: dict[tuple[str, str], float] = {}
    with table_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = str(row.get("Model", "")).strip()
            event_acc = parse_metric_mean(row.get("Task1 ACC"))
            location_acc = parse_metric_mean(row.get("Task2 ACC"))
            if event_acc is not None:
                rows[(model, "event")] = event_acc
            if location_acc is not None:
                rows[(model, "location")] = location_acc
    return rows


def annotate_matrix(ax, matrix_count: np.ndarray, max_count: int) -> None:
    threshold = max_count * 0.52
    for row_idx in range(matrix_count.shape[0]):
        for col_idx in range(matrix_count.shape[1]):
            value = int(matrix_count[row_idx, col_idx])
            color = "white" if value >= threshold else "black"
            ax.text(col_idx, row_idx, f"{value}", ha="center", va="center", color=color, fontsize=8.8)


def plot_confusion_grid(
    confusions: dict[str, dict[str, tuple[list[str], np.ndarray, Path]]],
    task: str,
    output_dir: Path,
) -> Path:
    if task == "event":
        class_labels = EVENT_SHORT
        title = "Event-type task confusion matrices (sample counts)"
        output_name = "fig_table_i_event_confusion_counts"
    elif task == "location":
        class_labels = LOCATION_SHORT
        title = "Location task confusion matrices (sample counts)"
        output_name = "fig_table_i_location_confusion_counts"
    else:
        raise ValueError(f"Unsupported task: {task}")

    fig, axes = plt.subplots(2, 4, figsize=(18.2, 9.0))
    axes = axes.reshape(2, 4)
    image_ref = None
    max_count = max(int(confusions[model_key][task][1].max()) for model_key, _ in MODEL_ROWS)
    for idx, (model_key, label) in enumerate(MODEL_ROWS):
        ax = axes[idx // 4, idx % 4]
        _, matrix, _ = confusions[model_key][task]
        image_ref = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max_count)
        annotate_matrix(ax, matrix, max_count)
        ax.set_title(f"({chr(ord('a') + idx)}) {label}", fontsize=13, fontweight="semibold", pad=8)
        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels, rotation=28, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(class_labels)
        ax.set_xlabel("Predicted label", fontsize=11.2)
        ax.set_ylabel("True label", fontsize=11.2)
        if label == "SensorField-M3T":
            for spine in ax.spines.values():
                spine.set_edgecolor("#B3322C")
                spine.set_linewidth(1.8)
        else:
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

    fig.suptitle(title, fontsize=18, fontweight="semibold", y=0.985)
    fig.subplots_adjust(left=0.045, right=0.91, top=0.90, bottom=0.09, wspace=0.34, hspace=0.58)
    if image_ref is not None:
        cbar_ax = fig.add_axes([0.93, 0.18, 0.015, 0.64])
        cbar = fig.colorbar(image_ref, cax=cbar_ax)
        cbar.set_label("Sample count", fontsize=11.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_name}.png"
    pdf_path = output_dir / f"{output_name}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def write_source_summary(
    output_dir: Path,
    confusions: dict[str, dict[str, tuple[list[str], np.ndarray, Path]]],
    table_csv: Path,
) -> None:
    table_accs = load_table_accs(table_csv)
    rows = []
    for model_key, label in MODEL_ROWS:
        for task in ("event", "location"):
            _, matrix, path = confusions[model_key][task]
            matrix_acc = float(np.trace(matrix) / max(matrix.sum(), 1))
            table_acc = table_accs.get((label, task))
            if table_acc is None:
                table_acc = table_accs.get((model_key, task))
            rows.append(
                {
                    "model": label,
                    "task": task,
                    "source_csv": str(path),
                    "samples": int(matrix.sum()),
                    "diagonal": int(np.trace(matrix)),
                    "accuracy_from_matrix": matrix_acc,
                    "table_acc_mean": table_acc,
                    "abs_acc_gap": None if table_acc is None else abs(matrix_acc - table_acc),
                    "matches_table_acc": None if table_acc is None else abs(matrix_acc - table_acc) <= 0.01,
                }
            )
    with (output_dir / "confusion_source_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    mismatches = [row for row in rows if row["matches_table_acc"] is False]
    if mismatches:
        print("WARNING: Some confusion matrices do not match the reference table ACC within 0.01.")
        for row in mismatches:
            print(
                "  "
                f"{row['model']} {row['task']}: "
                f"matrix_acc={row['accuracy_from_matrix']:.4f}, "
                f"table_acc={row['table_acc_mean']:.4f}, "
                f"gap={row['abs_acc_gap']:.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Table-I-aligned confusion matrices.")
    parser.add_argument("--dataset_path", default=str(ROOT / "converted_csv" / "MTL43_imagefork_dedup_clean"), type=str)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--save_path", default=str(DEFAULT_OUTPUT_DIR), type=str)
    parser.add_argument("--table_csv", default=str(DEFAULT_TABLE_CSV), type=str)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--csv_input_height", default=6, type=int)
    parser.add_argument("--csv_input_width", default=10000, type=int)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--raw_length", default=4096, type=int)
    parser.add_argument("--stft_size", default=128, type=int)
    parser.add_argument("--sota_gaf_size", default=128, type=int)
    parser.add_argument("--stft_n_fft", default=256, type=int)
    parser.add_argument("--stft_hop_length", default=128, type=int)
    parser.add_argument("--stft_win_length", default=256, type=int)
    parser.add_argument("--event_classes", default="walking,excavator,driving,background", type=str)
    parser.add_argument("--distance_classes", default="Alarm area,Tracking area,No-threat area", type=str)
    return parser.parse_args()


def main() -> None:
    setup_style()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.save_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    confusions = collect_confusions(args, output_dir)
    event_path = plot_confusion_grid(confusions, "event", output_dir)
    location_path = plot_confusion_grid(confusions, "location", output_dir)
    write_source_summary(output_dir, confusions, Path(args.table_csv).expanduser().resolve())
    print(f"Saved Event confusion grid: {event_path}")
    print(f"Saved Location confusion grid: {location_path}")
    print(f"Saved source summary: {output_dir / 'confusion_source_summary.csv'}")


if __name__ == "__main__":
    main()
