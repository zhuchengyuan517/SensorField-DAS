from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "paper_assets" / "sensorfield_m3t_experiments" / "architecture"


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#2b2b2b",
    fontsize: int = 10,
    weight: str = "normal",
    linestyle: str = "-",
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        linewidth=1.35,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#17202a",
        linespacing=1.18,
    )
    return patch


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#34495e",
    linestyle: str = "-",
    rad: float = 0.0,
    lw: float = 1.45,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        shrinkA=4,
        shrinkB=4,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    return arrow


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(21.0, 8.5))
    ax.set_xlim(0, 20.4)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    fig.patch.set_facecolor("#fbfaf6")
    ax.set_facecolor("#fbfaf6")

    colors = {
        "input": "#f7d9c4",
        "encoder": "#cfe6d8",
        "token": "#d7e5fb",
        "fac": "#ffe6a7",
        "taef": "#d8d2f2",
        "gcti": "#c9e8ec",
        "head": "#f4c7d7",
        "loss": "#eeeeee",
    }

    ax.text(
        10.2,
        6.92,
        "SensorField-M3T Architecture",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        10.2,
        6.56,
        "Three-view field encoding -> Field-Anchor Complementation -> Task-Adaptive Evidence Fusion -> Generalization-Consistent Task Interaction",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4b5563",
    )

    y_positions = {"raw": 5.15, "stf": 3.55, "gaf": 1.95}
    input_text = {
        "raw": "Raw temporal signal\n[B, 1, L]",
        "stf": "STF feature map\n[B, 1, H, W]",
        "gaf": "GAF image\n[B, 1, H, W]",
    }
    encoder_text = {
        "raw": "Raw encoder\nConv1D blocks\n+ projection",
        "stf": "STF encoder\nConv2D blocks\n+ projection",
        "gaf": "GAF encoder\nConv2D blocks\n+ projection",
    }
    token_text = {
        "raw": "T_raw\n[B, N_raw, D]",
        "stf": "T_stf\n[B, N_stf, D]",
        "gaf": "T_gaf\n[B, N_gaf, D]",
    }

    for key, y in y_positions.items():
        add_box(ax, 0.45, y, 1.88, 0.82, input_text[key], colors["input"], fontsize=10)
        add_box(ax, 3.05, y, 2.05, 0.82, encoder_text[key], colors["encoder"], fontsize=9.5)
        add_box(ax, 5.90, y, 1.55, 0.82, token_text[key], colors["token"], fontsize=9.8, weight="bold")
        add_arrow(ax, (2.33, y + 0.41), (3.05, y + 0.41))
        add_arrow(ax, (5.10, y + 0.41), (5.90, y + 0.41))

    add_box(
        ax,
        8.20,
        1.05,
        2.95,
        5.15,
        "FAC\nField-Anchor Complementation\n\nLearnable anchor bank\nA0 [K, D]\n+ sample modulation Delta A(x)\n\nAnchor-to-view cross-attention\nA_v = Attn(A, T_v)\n\nCross-view agreement weights\nw_v,k\n\nShared anchors\nS [B, K, D]\n\nComplement evidence\nC_v = A_v - Proj_S(A_v)",
        colors["fac"],
        fontsize=8.8,
        weight="bold",
    )
    for y in y_positions.values():
        add_arrow(ax, (7.45, y + 0.41), (8.20, y + 0.41))

    add_box(
        ax,
        12.05,
        1.92,
        2.72,
        3.72,
        "TAEF\nTask-Adaptive Evidence Fusion\n\nEvidence bank\n{S, C_raw, C_stf, C_gaf}\n\nLearnable task queries q_t\none query per task\n\nalpha_t,e = softmax(q_t K_e^T)\n\nTask representations\nR_t [B, D]",
        colors["taef"],
        fontsize=8.8,
        weight="bold",
    )
    add_arrow(ax, (11.15, 3.62), (12.05, 3.78))

    add_box(
        ax,
        15.35,
        1.92,
        2.72,
        3.72,
        "GCTI\nTask Interaction\n\nStack task tokens\n[B, T, D]\n\nAttention with learnable\ntask-relation bias\n\nRelation matrix\n[B, T, T]\n\nUpdated task tokens\n[B, T, D]",
        colors["gcti"],
        fontsize=8.8,
        weight="bold",
    )
    add_arrow(ax, (14.77, 3.78), (15.35, 3.78))

    add_box(ax, 18.65, 4.50, 1.42, 0.72, "Event head\n[B, C_event]", colors["head"], fontsize=9.2, weight="bold")
    add_box(ax, 18.65, 3.36, 1.42, 0.72, "Location head\n[B, C_loc]", colors["head"], fontsize=9.2, weight="bold")
    add_box(ax, 18.65, 2.22, 1.42, 0.72, "Other heads\n(optional)", colors["head"], fontsize=9.2, weight="bold")
    add_arrow(ax, (18.07, 3.78), (18.65, 4.86), rad=0.14)
    add_arrow(ax, (18.07, 3.78), (18.65, 3.72))
    add_arrow(ax, (18.07, 3.78), (18.65, 2.58), rad=-0.14)

    add_box(
        ax,
        8.20,
        0.22,
        2.95,
        0.52,
        "L_FAC: alignment + complement decorrelation",
        colors["loss"],
        fontsize=8.5,
        linestyle="--",
    )
    add_box(
        ax,
        12.05,
        0.22,
        2.72,
        0.52,
        "L_TAEF: evidence-weight diversity",
        colors["loss"],
        fontsize=8.5,
        linestyle="--",
    )
    add_box(
        ax,
        15.35,
        0.22,
        2.72,
        0.52,
        "L_GCTI: relation consistency / symmetry",
        colors["loss"],
        fontsize=8.5,
        linestyle="--",
    )
    add_box(
        ax,
        0.45,
        0.22,
        6.9,
        0.52,
        "Optional view perturbation consistency: full-view vs dropped/noisy/masked-view predictions",
        "#f5f5f4",
        fontsize=7.9,
        linestyle="--",
    )

    ax.text(
        0.45,
        6.23,
        "Enabled views: raw, stf, gaf",
        ha="left",
        va="center",
        fontsize=9.2,
        color="#374151",
        fontweight="bold",
    )
    ax.text(
        18.25,
        5.65,
        "Task heads are created from\ntask_output_dims, not hard-coded.",
        ha="left",
        va="center",
        fontsize=8.7,
        color="#374151",
    )

    png_path = OUTPUT_DIR / "sensorfield_m3t_architecture.png"
    pdf_path = OUTPUT_DIR / "sensorfield_m3t_architecture.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
