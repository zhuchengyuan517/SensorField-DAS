from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_CSV = ROOT / "_tmp_table3_balanced_rerun" / "table3_summary_all.csv"
DEFAULT_OUTPUT_DIR = ROOT / "paper_assets" / "sensorfield_m3t_experiments" / "table3_confusion_fig5"

MODEL_ORDER = [
    ("resnet", "ResNet"),
    ("vgg", "VGG"),
    ("vit", "ViT"),
    ("proposed", "所提出方法"),
]
CLASS_ORDER = ["walking", "excavator", "driving", "background"]
CLASS_LABELS = ["人工作业", "挖掘施工", "车辆行驶", "背景噪声"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Figure-5-style confusion matrices for the 4 Table-3 models.")
    parser.add_argument("--summary_csv", default=str(DEFAULT_SUMMARY_CSV), type=str)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), type=str)
    return parser.parse_args()


def setup_style() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Times New Roman",
        "Cambria",
        "Georgia",
        "STIXGeneral",
        "DejaVu Serif",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 240,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
        }
    )


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_confusion_csv(path: Path) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    rows: list[list[int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        _ = header[1:]
        for row in reader:
            labels.append(str(row[0]).strip())
            rows.append([int(float(value)) for value in row[1:]])
    return labels, np.asarray(rows, dtype=np.int64)


def reorder_matrix(labels: list[str], matrix: np.ndarray) -> np.ndarray:
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    indices = [label_to_index[label] for label in CLASS_ORDER]
    return matrix[np.ix_(indices, indices)]


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    denominator = matrix.sum(axis=1, keepdims=True).astype(np.float64)
    denominator[denominator == 0.0] = 1.0
    return matrix.astype(np.float64) / denominator


def collect_model_confusions(summary_rows: list[dict[str, str]]) -> dict[str, tuple[np.ndarray, Path]]:
    rows_by_model = {str(row["Model"]).strip().lower(): row for row in summary_rows}
    payload: dict[str, tuple[np.ndarray, Path]] = {}
    for model_key, _ in MODEL_ORDER:
        if model_key not in rows_by_model:
            raise KeyError(f"Missing model '{model_key}' in summary CSV.")
        source_dir = Path(rows_by_model[model_key]["Source"]).expanduser().resolve()
        confusion_path = source_dir / "history" / "best_test_confusion.csv"
        labels, matrix = read_confusion_csv(confusion_path)
        payload[model_key] = (reorder_matrix(labels, matrix), confusion_path)
    return payload


def annotate_matrix(ax, count_matrix: np.ndarray, norm_matrix: np.ndarray) -> None:
    for row_idx in range(count_matrix.shape[0]):
        for col_idx in range(count_matrix.shape[1]):
            count = int(count_matrix[row_idx, col_idx])
            text_color = "white" if norm_matrix[row_idx, col_idx] >= 0.55 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{count}",
                ha="center",
                va="center",
                fontsize=10.2,
                color=text_color,
            )


def render_figure(confusions: dict[str, tuple[np.ndarray, Path]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 10.6))
    axes = axes.reshape(2, 2)

    image_ref = None
    source_rows = []
    for index, (model_key, display_name) in enumerate(MODEL_ORDER):
        count_matrix, source_path = confusions[model_key]
        norm_matrix = row_normalize(count_matrix)
        ax = axes[index // 2, index % 2]
        image_ref = ax.imshow(norm_matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        annotate_matrix(ax, count_matrix, norm_matrix)
        ax.set_title(f"({chr(ord('a') + index)}) {display_name}", fontweight="semibold", pad=10)
        ax.set_xticks(np.arange(len(CLASS_LABELS)))
        ax.set_yticks(np.arange(len(CLASS_LABELS)))
        ax.set_xticklabels(CLASS_LABELS, rotation=0)
        ax.set_yticklabels(CLASS_LABELS)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        if model_key == "proposed":
            for spine in ax.spines.values():
                spine.set_edgecolor("#B3322C")
                spine.set_linewidth(1.8)

        source_rows.append(
            {
                "model": display_name,
                "source_csv": str(source_path),
                "samples": int(count_matrix.sum()),
                "diagonal": int(np.trace(count_matrix)),
                "accuracy": float(np.trace(count_matrix) / max(count_matrix.sum(), 1)),
            }
        )

    fig.suptitle("图5 不同模型在测试集上的混淆矩阵", y=0.98, fontsize=18)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.92, bottom=0.10, wspace=0.26, hspace=0.30)

    png_path = output_dir / "fig5_table3_confusion_matrices_zh_counts.png"
    pdf_path = output_dir / "fig5_table3_confusion_matrices_zh_counts.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    source_csv = output_dir / "fig5_confusion_sources.csv"
    with source_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0].keys()))
        writer.writeheader()
        writer.writerows(source_rows)
    return [png_path, pdf_path, source_csv]


def main() -> None:
    args = parse_args()
    setup_style()
    summary_csv = Path(args.summary_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary_rows = read_summary_rows(summary_csv)
    confusions = collect_model_confusions(summary_rows)
    outputs = render_figure(confusions, output_dir=output_dir)
    for output in outputs:
        print(str(output))


if __name__ == "__main__":
    main()
