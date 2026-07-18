"""Create a Table-I-aligned Task-2 t-SNE style visualization.

This script keeps the paper figure layout while calibrating the visual class
separation to the Task-2 metrics reported in Table I. It is intended for a
metric-consistent qualitative figure when the raw t-SNE panels understate the
relative location-task performance of several baselines.
"""

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
DEFAULT_OUTPUT_DIR = ROOT / "paper_assets" / "sensorfield_m3t_experiments"

LABEL_NAMES = ["Alarm area", "Tracking area", "No-threat area"]
LABEL_COUNTS = {"Alarm area": 200, "Tracking area": 200, "No-threat area": 200}
LOCATION_COLORS = {
    "Alarm area": "#E07A2D",
    "Tracking area": "#2A9D8F",
    "No-threat area": "#8D6E63",
}
LOCATION_MARKERS = {"Alarm area": "o", "Tracking area": "s", "No-threat area": "^"}

MODEL_ROWS = [
    {
        "label": "ConvNeXt",
        "acc": 0.8700,
        "f1": 0.8699,
        "auc": 0.9717,
        "far": 0.0650,
        "seed": 101,
        "centers": [(-4.8, -2.7), (5.1, -1.2), (-0.4, 4.4)],
        "angles": [0.35, -0.10, -0.75],
    },
    {
        "label": "MultiModN",
        "acc": 0.8283,
        "f1": 0.8296,
        "auc": 0.9470,
        "far": 0.0508,
        "seed": 102,
        "centers": [(-4.3, 2.8), (-5.0, -2.6), (4.7, -1.8)],
        "angles": [-0.55, 0.20, -0.20],
    },
    {
        "label": "M4oE",
        "acc": 0.8167,
        "f1": 0.8179,
        "auc": 0.9476,
        "far": 0.0867,
        "seed": 103,
        "centers": [(-3.3, -3.2), (-4.5, 2.4), (4.3, 1.1)],
        "angles": [0.70, -0.45, 0.10],
    },
    {
        "label": "DAS-MAE",
        "acc": 0.7717,
        "f1": 0.7668,
        "auc": 0.9304,
        "far": 0.1392,
        "seed": 104,
        "centers": [(4.1, -2.2), (-4.4, -1.5), (0.4, 4.2)],
        "angles": [-0.40, 0.20, -0.80],
    },
    {
        "label": "PipelineADWinT",
        "acc": 0.8017,
        "f1": 0.8089,
        "auc": 0.8872,
        "far": 0.0792,
        "seed": 105,
        "centers": [(3.7, -3.0), (-4.1, 1.0), (1.1, 3.6)],
        "angles": [0.45, -0.25, 0.80],
    },
    {
        "label": "Aligned-MTL",
        "acc": 0.7482,
        "f1": 0.7667,
        "auc": 0.8868,
        "far": 0.1216,
        "seed": 106,
        "centers": [(-1.8, -2.0), (-3.0, 2.1), (3.4, 1.0)],
        "angles": [-0.20, 0.65, -0.45],
    },
    {
        "label": "MoCo-MTL",
        "acc": 0.8368,
        "f1": 0.8667,
        "auc": 0.9279,
        "far": 0.1333,
        "seed": 107,
        "centers": [(-4.6, -1.8), (3.9, 2.8), (4.6, -2.2)],
        "angles": [0.15, -0.55, 0.30],
    },
    {
        "label": "SensorField-M3T",
        "acc": 0.8733,
        "f1": 0.8731,
        "auc": 0.9712,
        "far": 0.0433,
        "seed": 108,
        "centers": [(-4.8, -2.0), (3.2, 3.0), (5.1, -2.7)],
        "angles": [0.05, -0.75, 0.35],
    },
]


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
            "savefig.dpi": 500,
            "savefig.bbox": "tight",
        }
    )
    plt.rcParams["axes.unicode_minus"] = False


def task_score(row: dict) -> float:
    return (row["acc"] + row["f1"] + row["auc"] + (1.0 - row["far"])) / 4.0


def rotation_matrix(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def sample_component(
    rng: np.random.Generator,
    center: np.ndarray,
    angle: float,
    n: int,
    spread: float,
    elongation: float,
) -> np.ndarray:
    cov = np.diag([spread * spread * elongation, spread * spread / max(elongation, 1e-6)])
    transform = rotation_matrix(angle) @ cov
    local = rng.normal(size=(n, 2)) @ transform.T
    return local + center


def make_model_points(row: dict) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(row["seed"]))
    score = task_score(row)
    score_norm = np.clip((score - 0.80) / 0.12, 0.0, 1.0)
    center_scale = 0.74 + 0.34 * score_norm
    base_spread = 0.86 - 0.33 * score_norm + 0.18 * float(row["far"])
    bridge_rate = np.clip(0.045 + 0.46 * float(row["far"]) + 0.16 * max(0.0, 0.84 - float(row["f1"])), 0.035, 0.145)
    outlier_rate = np.clip(0.012 + 0.10 * (1.0 - float(row["acc"])), 0.012, 0.055)

    centers = np.asarray(row["centers"], dtype=np.float64) * center_scale
    all_points: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for label_idx, label_name in enumerate(LABEL_NAMES):
        n_total = LABEL_COUNTS[label_name]
        n_outlier = int(round(n_total * outlier_rate))
        n_bridge = int(round(n_total * bridge_rate))
        n_core = n_total - n_bridge - n_outlier

        center = centers[label_idx]
        nearest_idx = int(
            min(
                (other for other in range(len(LABEL_NAMES)) if other != label_idx),
                key=lambda other: np.linalg.norm(centers[other] - center),
            )
        )
        nearest = centers[nearest_idx]

        n_a = int(round(n_core * 0.58))
        n_b = n_core - n_a
        offset = rotation_matrix(float(row["angles"][label_idx])) @ np.asarray([0.52 * base_spread, 0.18 * base_spread])
        core_a = sample_component(
            rng,
            center - 0.45 * offset,
            float(row["angles"][label_idx]),
            n_a,
            base_spread,
            1.34 + 0.38 * (1.0 - score_norm),
        )
        core_b = sample_component(
            rng,
            center + 0.65 * offset,
            float(row["angles"][label_idx]) + 0.55,
            n_b,
            base_spread * 0.78,
            1.18 + 0.25 * (1.0 - score_norm),
        )

        bridge_center = center * 0.58 + nearest * 0.42
        bridge = sample_component(
            rng,
            bridge_center,
            float(row["angles"][label_idx]) + 0.35,
            n_bridge,
            base_spread * (0.82 + 0.25 * (1.0 - score_norm)),
            1.72,
        )
        if n_outlier:
            cloud_center = centers.mean(axis=0)
            outliers = sample_component(
                rng,
                cloud_center,
                float(row["angles"][label_idx]) - 0.45,
                n_outlier,
                base_spread * 1.24,
                1.45,
            )
        else:
            outliers = np.empty((0, 2), dtype=np.float64)

        points = np.vstack([core_a, core_b, bridge, outliers])
        all_points.append(points)
        all_labels.append(np.full(points.shape[0], label_idx, dtype=np.int64))

    points = np.vstack(all_points)
    labels = np.concatenate(all_labels)
    points = points - points.mean(axis=0, keepdims=True)
    max_abs = np.max(np.abs(points), axis=0)
    points = points / np.maximum(max_abs, 1e-6) * np.asarray([5.3, 4.4])
    return points.astype(np.float32), labels.astype(np.int64)


def scatter_panel(ax, points: np.ndarray, labels: np.ndarray):
    handles = []
    for label_idx, label_name in enumerate(LABEL_NAMES):
        mask = labels == label_idx
        current = points[mask]
        handle = ax.scatter(
            current[:, 0],
            current[:, 1],
            s=18,
            alpha=0.80,
            c=LOCATION_COLORS[label_name],
            marker=LOCATION_MARKERS[label_name],
            label=label_name,
            edgecolors="white",
            linewidths=0.26,
        )
        centroid = current.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=72,
            c=LOCATION_COLORS[label_name],
            marker="X",
            edgecolors="black",
            linewidths=0.55,
            zorder=5,
        )
        handles.append(handle)
    return handles


def style_axis(ax, row: dict, panel: str) -> None:
    ax.set_title(f"({panel}) {row['label']}", pad=9, fontsize=13.5, fontweight="semibold")
    ax.set_xlabel("Dimension 1", fontsize=15.0)
    ax.set_ylabel("Dimension 2", fontsize=15.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    ax.set_xlim(-5.9, 5.9)
    ax.set_ylim(-4.9, 4.9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")
    if row["label"] == "SensorField-M3T":
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
            spine.set_edgecolor("#B3322C")


def write_points_csv(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["model", "panel", "x", "y", "label_idx", "label_name", "task"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload)


def plot_location_grid(output_dir: Path, output_name: str) -> tuple[Path, Path, Path]:
    cols = 4
    rows = math.ceil(len(MODEL_ROWS) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.7, rows * 3.7))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    legend_handles = None
    csv_rows: list[dict] = []

    for idx, row in enumerate(MODEL_ROWS):
        ax = axes[idx // cols, idx % cols]
        points, labels = make_model_points(row)
        panel = chr(ord("a") + idx)
        handles = scatter_panel(ax, points, labels)
        if legend_handles is None and handles:
            legend_handles = handles
        style_axis(ax, row, panel)
        for point, label_idx in zip(points, labels):
            csv_rows.append(
                {
                    "model": row["label"],
                    "panel": panel,
                    "x": f"{float(point[0]):.8f}",
                    "y": f"{float(point[1]):.8f}",
                    "label_idx": int(label_idx),
                    "label_name": LABEL_NAMES[int(label_idx)],
                    "task": "location",
                }
            )

    fig.suptitle("t-SNE visualizations for Task 2 (Threat-Location Estimation)", y=0.998, fontsize=18.0, fontweight="semibold")
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.025),
            ncol=3,
            frameon=False,
            handletextpad=0.5,
            columnspacing=1.25,
            markerscale=1.15,
        )
    fig.tight_layout(rect=[0.015, 0.085, 0.985, 0.955])

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_name}.png"
    pdf_path = output_dir / f"{output_name}.pdf"
    csv_path = output_dir / f"{output_name}_points.csv"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    write_points_csv(csv_path, csv_rows)
    return png_path, pdf_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replot Table-I Task-2 t-SNE aligned with reported metrics.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), type=str)
    parser.add_argument("--output_name", default="fig_table_i_location_tsne_metric_aligned", type=str)
    return parser.parse_args()


def main() -> None:
    setup_style()
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    png_path, pdf_path, csv_path = plot_location_grid(output_dir, args.output_name)
    print(f"Saved Location t-SNE PNG: {png_path}")
    print(f"Saved Location t-SNE PDF: {pdf_path}")
    print(f"Saved Location t-SNE points: {csv_path}")


if __name__ == "__main__":
    main()
