"""Replot current-eight-model t-SNE figures aligned with Table I metrics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TSNE_DIR = (
    ROOT
    / "paper_assets"
    / "sensorfield_m3t_experiments"
    / "tsne_current8_models"
    / "20260626_154826"
)
DEFAULT_OUTPUT_DIR = ROOT / "paper_assets" / "sensorfield_m3t_experiments"

MODEL_ROWS = [
    {
        "folder": "ConvNeXt-Small",
        "label": "ConvNeXt",
        "event_score": 0.9539,
        "location_score": 0.9116,
        "mtl_score": 0.9328,
    },
    {
        "folder": "MultiModN",
        "label": "MultiModN",
        "event_score": 0.9721,
        "location_score": 0.8435,
        "mtl_score": 0.9078,
    },
    {
        "folder": "M4oE",
        "label": "M4oE",
        "event_score": 0.9651,
        "location_score": 0.8339,
        "mtl_score": 0.8995,
    },
    {
        "folder": "DAS-MAE_plus_downstream_fine-tuning_head",
        "label": "DAS-MAE",
        "event_score": 0.8686,
        "location_score": 0.7999,
        "mtl_score": 0.8343,
    },
    {
        "folder": "PipelineADWinT",
        "label": "PipelineADWinT",
        "event_score": 0.9844,
        "location_score": 0.7422,
        "mtl_score": 0.8633,
    },
    {
        "folder": "Aligned-MTL",
        "label": "Aligned-MTL",
        "event_score": 0.8138,
        "location_score": 0.6134,
        "mtl_score": 0.7136,
    },
    {
        "folder": "MoCo-weighting",
        "label": "MoCo-MTL",
        "event_score": 0.9380,
        "location_score": 0.5487,
        "mtl_score": 0.7433,
    },
    {
        "folder": "SensorField-M3T",
        "label": "SensorField-M3T",
        "event_score": 0.9922,
        "location_score": 0.9136,
        "mtl_score": 0.9529,
    },
]

EVENT_COLORS = {
    "walking": "#31688E",
    "excavator": "#C44536",
    "driving": "#2E7D32",
    "background": "#6D597A",
}
EVENT_MARKERS = {"walking": "o", "excavator": "s", "driving": "^", "background": "D"}
LOCATION_COLORS = {
    "Alarm area": "#E07A2D",
    "Tracking area": "#2A9D8F",
    "No-threat area": "#8D6E63",
}
LOCATION_MARKERS = {"Alarm area": "o", "Tracking area": "s", "No-threat area": "^"}


def setup_style() -> None:
    candidates = ["Times New Roman", "Cambria", "Georgia", "STIXGeneral", "DejaVu Serif"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update(
        {
            "font.size": 11.0,
            "axes.titlesize": 13.0,
            "axes.labelsize": 15.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 15.5,
            "figure.dpi": 240,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
        }
    )
    plt.rcParams["axes.unicode_minus"] = False


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    points = []
    labels = []
    names = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append([float(row["x"]), float(row["y"])])
            label_idx = int(row["label_idx"])
            labels.append(label_idx)
            names[label_idx] = row["label_name"]
    label_names = [names[idx] for idx in sorted(names)]
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int64), label_names


def scatter_panel(ax, points: np.ndarray, labels: np.ndarray, label_names: list[str], colors: dict, markers: dict):
    handles = []
    for label_idx, label_name in enumerate(label_names):
        mask = labels == label_idx
        if not np.any(mask):
            continue
        current = points[mask]
        handle = ax.scatter(
            current[:, 0],
            current[:, 1],
            s=18,
            alpha=0.80,
            c=colors.get(label_name, "#444444"),
            marker=markers.get(label_name, "o"),
            label=label_name,
            edgecolors="white",
            linewidths=0.28,
        )
        centroid = current.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=72,
            c=colors.get(label_name, "#444444"),
            marker="X",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
        handles.append(handle)
    return handles


def style_axis(ax, row: dict, panel: str, task_score: float) -> None:
    _ = task_score
    ax.set_title(f"({panel}) {row['label']}", pad=9, fontsize=13.5, fontweight="semibold")
    ax.set_xlabel("Dimension 1", fontsize=15.0)
    ax.set_ylabel("Dimension 2", fontsize=15.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")
    if row["label"] == "SensorField-M3T":
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
            spine.set_edgecolor("#B3322C")


def plot_grid(tsne_dir: Path, output_dir: Path, task: str) -> Path:
    if task == "event":
        filename = "event_tsne_points.csv"
        colors = EVENT_COLORS
        markers = EVENT_MARKERS
        score_key = "event_score"
        title = "Event-type task t-SNE distributions aligned with Table I"
        output_name = "fig_table_i_event_tsne"
    elif task == "location":
        filename = "location_tsne_points.csv"
        colors = LOCATION_COLORS
        markers = LOCATION_MARKERS
        score_key = "location_score"
        title = "Location task t-SNE distributions aligned with Table I"
        output_name = "fig_table_i_location_tsne"
    else:
        raise ValueError(f"Unsupported task: {task}")

    cols = 4
    rows = math.ceil(len(MODEL_ROWS) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.7, rows * 3.7))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    legend_handles = None

    for idx, row in enumerate(MODEL_ROWS):
        ax = axes[idx // cols, idx % cols]
        points, labels, label_names = read_points(tsne_dir / row["folder"] / filename)
        handles = scatter_panel(ax, points, labels, label_names, colors, markers)
        if legend_handles is None and handles:
            legend_handles = handles
        style_axis(ax, row, chr(ord("a") + idx), float(row[score_key]))

    for idx in range(len(MODEL_ROWS), rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    fig.suptitle(title, y=0.998, fontsize=16.0, fontweight="semibold")
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, len(legend_handles)),
            frameon=False,
            handletextpad=0.5,
            columnspacing=1.0,
            markerscale=1.15,
        )
    fig.tight_layout(rect=[0.015, 0.085, 0.985, 0.955])
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_name}.png"
    pdf_path = output_dir / f"{output_name}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replot Table I aligned t-SNE figures.")
    parser.add_argument("--tsne_dir", default=str(DEFAULT_TSNE_DIR), type=str)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), type=str)
    return parser.parse_args()


def main() -> None:
    setup_style()
    args = parse_args()
    tsne_dir = Path(args.tsne_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    event_path = plot_grid(tsne_dir, output_dir, "event")
    location_path = plot_grid(tsne_dir, output_dir, "location")
    print(f"Saved Event t-SNE: {event_path}")
    print(f"Saved Location t-SNE: {location_path}")


if __name__ == "__main__":
    main()
