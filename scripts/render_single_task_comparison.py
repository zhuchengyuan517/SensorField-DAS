from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT / "_tmp_table3_resplit_seed123_cuda_converge" / "20260705_184607"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "paper_assets"
    / "sensorfield_m3t_experiments"
    / "single_task_comparison"
)

MODEL_ORDER = [
    ("resnet", "ResNet"),
    ("vgg", "VGG"),
    ("vit", "ViT"),
    ("proposed", "Proposed method"),
]


def read_summary(run_root: Path) -> dict[str, dict[str, str]]:
    summary_path = run_root / "table3_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["Model"]: row for row in csv.DictReader(handle)}


def read_overall_report(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing report file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["label"] == "overall":
                return {
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                    "support": float(row["support"]),
                    "accuracy": float(row["accuracy"]),
                }
    raise RuntimeError(f"Overall row not found in {path}")


def read_confusion_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing confusion file: {path}")
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            rows.append([int(float(value)) for value in row[1:]])
    return np.asarray(rows, dtype=np.int64)


def compute_macro_far(matrix: np.ndarray) -> float:
    far_values = []
    for idx in range(matrix.shape[0]):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - tp)
        fn = float(matrix[idx, :].sum() - tp)
        tn = float(matrix.sum() - tp - fp - fn)
        far_values.append(fp / max(fp + tn, 1.0))
    return float(np.mean(far_values)) if far_values else 0.0


def read_per_class_report(model_key: str, model_name: str, path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["label"] == "overall":
                continue
            rows.append(
                {
                    "model_key": model_key,
                    "model": model_name,
                    "class": row["label"],
                    "precision": f"{float(row['precision']):.4f}",
                    "recall": f"{float(row['recall']):.4f}",
                    "f1": f"{float(row['f1']):.4f}",
                    "support": row["support"],
                }
            )
    return rows


def build_records(run_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    summary = read_summary(run_root)
    records: list[dict[str, str]] = []
    per_class_rows: list[dict[str, str]] = []

    for model_key, model_name in MODEL_ORDER:
        report_path = run_root / model_key / "history" / "best_test_report.csv"
        confusion_path = run_root / model_key / "history" / "best_test_confusion.csv"
        overall = read_overall_report(report_path)
        matrix = read_confusion_matrix(confusion_path)
        far = compute_macro_far(matrix)
        compact_score = (overall["accuracy"] + overall["f1"] + (1.0 - far)) / 3.0
        per_class_rows.extend(read_per_class_report(model_key, model_name, report_path))
        summary_row = summary[model_key]
        records.append(
            {
                "model_key": model_key,
                "model": model_name,
                "best_epoch": summary_row["Best Epoch"],
                "test_acc": f"{overall['accuracy']:.4f}",
                "test_precision": f"{overall['precision']:.4f}",
                "test_recall": f"{overall['recall']:.4f}",
                "test_f1": f"{overall['f1']:.4f}",
                "test_far": f"{far:.4f}",
                "compact_score": f"{compact_score:.4f}",
                "support": str(int(overall["support"])),
                "source": str(run_root / model_key),
            }
        )
    return records, per_class_rows


def resolve_sensorfield_run_dir(path: Path) -> Path:
    """Accept either the model directory or its parent experiment directory."""
    if (path / "history").is_dir():
        return path
    candidate = path / "sensorfield_m3t"
    if (candidate / "history").is_dir():
        return candidate
    raise FileNotFoundError(
        "Could not find SensorField-M3T history directory under "
        f"{path} or {candidate}"
    )


def build_sensorfield_record(run_dir: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    run_dir = resolve_sensorfield_run_dir(run_dir)
    report_path = run_dir / "history" / "best_test_report.csv"
    confusion_path = run_dir / "history" / "best_test_confusion.csv"
    summary_path = run_dir / "summary.json"

    overall = read_overall_report(report_path)
    matrix = read_confusion_matrix(confusion_path)
    far = compute_macro_far(matrix)
    compact_score = (overall["accuracy"] + overall["f1"] + (1.0 - far)) / 3.0
    best_epoch = ""
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8-sig") as handle:
            best_epoch = str(json.load(handle).get("best_epoch", ""))

    model_key = "sensorfield_m3t"
    model_name = "SensorField-M3T"
    record = {
        "model_key": model_key,
        "model": model_name,
        "best_epoch": best_epoch,
        "test_acc": f"{overall['accuracy']:.4f}",
        "test_precision": f"{overall['precision']:.4f}",
        "test_recall": f"{overall['recall']:.4f}",
        "test_f1": f"{overall['f1']:.4f}",
        "test_far": f"{far:.4f}",
        "compact_score": f"{compact_score:.4f}",
        "support": str(int(overall["support"])),
        "source": str(run_dir),
    }
    per_class_rows = read_per_class_report(model_key, model_name, report_path)
    return record, per_class_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def legacy_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    keys = [
        "model_key",
        "model",
        "best_epoch",
        "test_acc",
        "test_precision",
        "test_recall",
        "test_f1",
        "support",
        "source",
    ]
    return [{key: row[key] for key in keys} for row in records]


def bold_best(value: str, best_value: float) -> str:
    parsed = float(value)
    text = f"{parsed:.4f}"
    return rf"\textbf{{{text}}}" if abs(parsed - best_value) < 5e-5 else text


def render_tex(records: list[dict[str, str]], output_path: Path) -> None:
    metric_keys = ["test_acc", "test_precision", "test_recall", "test_f1", "compact_score"]
    best = {key: max(float(row[key]) for row in records) for key in metric_keys}
    best["test_far"] = min(float(row["test_far"]) for row in records)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Single-task event-recognition comparison on the balanced SensorField-DAS split. The checkpoint is selected by the best validation accuracy, and test metrics are reported as macro averages.}",
        r"\label{tab:single_task_event_comparison}",
        r"\small",
        r"\setlength{\tabcolsep}{5.0pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{l|ccccc}",
        r"\hline",
        r"Model & Best epoch & ACC & Precision & Recall & F1 \\",
        r"\hline",
    ]
    for row in records:
        lines.append(
            " & ".join(
                [
                    row["model"],
                    row["best_epoch"],
                    bold_best(row["test_acc"], best["test_acc"]),
                    bold_best(row["test_precision"], best["test_precision"]),
                    bold_best(row["test_recall"], best["test_recall"]),
                    bold_best(row["test_f1"], best["test_f1"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def bold_best_direction(value: str, best_value: float, higher_is_better: bool = True) -> str:
    parsed = float(value)
    text = f"{parsed:.4f}"
    if abs(parsed - best_value) < 5e-5:
        return rf"\textbf{{{text}}}"
    return text


def render_extended_tex(
    records: list[dict[str, str]],
    output_path: Path,
    *,
    caption: str | None = None,
    label: str = "tab:single_task_event_comparison_extended",
) -> None:
    higher_keys = ["test_acc", "test_precision", "test_recall", "test_f1", "compact_score"]
    best = {key: max(float(row[key]) for row in records) for key in higher_keys}
    best["test_far"] = min(float(row["test_far"]) for row in records)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption or 'Single-task event-recognition comparison on the balanced SensorField-DAS split. The checkpoint is selected by validation macro-F1. ACC, Precision, Recall, and F1 are macro-level test metrics; FAR is macro false-alarm rate. Since this run did not save class probabilities, AUC is not reconstructed. The compact score is $(\\mathrm{ACC}+\\mathrm{F1}+1-\\mathrm{FAR})/3$.'}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{l|ccccccc}",
        r"\hline",
        r"Model & Best epoch & ACC $\uparrow$ & Precision $\uparrow$ & Recall $\uparrow$ & F1 $\uparrow$ & FAR $\downarrow$ & Score $\uparrow$ \\",
        r"\hline",
    ]
    for row in records:
        lines.append(
            " & ".join(
                [
                    row["model"],
                    row["best_epoch"],
                    bold_best_direction(row["test_acc"], best["test_acc"]),
                    bold_best_direction(row["test_precision"], best["test_precision"]),
                    bold_best_direction(row["test_recall"], best["test_recall"]),
                    bold_best_direction(row["test_f1"], best["test_f1"]),
                    bold_best_direction(row["test_far"], best["test_far"], higher_is_better=False),
                    bold_best_direction(row["compact_score"], best["compact_score"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render single-task event-recognition comparison assets."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sensorfield-run-dir",
        type=Path,
        default=None,
        help=(
            "Optional SensorField-M3T single-task run directory. The path may point "
            "to either the sensorfield_m3t model directory or its parent run root."
        ),
    )
    args = parser.parse_args()

    records, per_class_rows = build_records(args.run_root)
    write_csv(args.output_dir / "single_task_event_comparison.csv", legacy_records(records))
    write_csv(args.output_dir / "single_task_event_comparison_extended.csv", records)
    write_csv(args.output_dir / "single_task_event_per_class_metrics.csv", per_class_rows)
    render_tex(records, args.output_dir / "table_single_task_event_comparison.tex")
    render_extended_tex(records, args.output_dir / "table_single_task_event_comparison_extended.tex")
    if args.sensorfield_run_dir is not None:
        sensorfield_record, sensorfield_per_class = build_sensorfield_record(
            args.sensorfield_run_dir
        )
        combined_records = [*records, sensorfield_record]
        combined_per_class_rows = [*per_class_rows, *sensorfield_per_class]
        write_csv(
            args.output_dir / "single_task_event_comparison_with_sensorfield_m3t.csv",
            combined_records,
        )
        write_csv(
            args.output_dir / "single_task_event_per_class_metrics_with_sensorfield_m3t.csv",
            combined_per_class_rows,
        )
        render_extended_tex(
            combined_records,
            args.output_dir / "table_single_task_event_comparison_with_sensorfield_m3t.tex",
            caption=(
                "Single-task event-recognition comparison on the balanced "
                "SensorField-DAS split, supplemented with SensorField-M3T trained "
                "in event-only mode on the same seed123 split. The checkpoint is "
                "selected by validation macro-F1. ACC, Precision, Recall, and F1 "
                "are macro-level test metrics; FAR is macro false-alarm rate. "
                "Since these runs did not save class probabilities, AUC is not "
                "reconstructed. The compact score is "
                "$(\\mathrm{ACC}+\\mathrm{F1}+1-\\mathrm{FAR})/3$."
            ),
            label="tab:single_task_event_comparison_with_sensorfield_m3t",
        )
    print(f"Saved single-task comparison assets to {args.output_dir}")


if __name__ == "__main__":
    main()
