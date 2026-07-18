from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "paper_assets" / "sensorfield_m3t_experiments" / "architecture"


COLORS = {
    "panel": "#fff8df",
    "raw": "#f3d9ca",
    "stf": "#dceff2",
    "gaf": "#eadcf1",
    "conv": "#dbcfe7",
    "act": "#d8ece5",
    "pool": "#fff0bf",
    "token": "#d9e8ff",
    "fac": "#ffe8b3",
    "taef": "#d9d3f1",
    "gcti": "#cfecef",
    "head": "#f5c6d6",
    "loss": "#f3f4f6",
    "line": "#263746",
}


def add_round_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str,
    ec: str = "#2d2d2d",
    lw: float = 1.15,
    fs: float = 8.4,
    weight: str = "normal",
    ls: str = "-",
    radius: float = 0.07,
    ha: str = "center",
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle=ls,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha=ha,
        va="center",
        fontsize=fs,
        fontweight=weight,
        color="#111827",
        linespacing=1.1,
        zorder=3,
    )
    return box


def add_group_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    fc: str = "#ffffff00",
    ec: str = "#d49100",
    fs: float = 8.5,
    ls: str = "--",
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.08",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
        linestyle=ls,
        zorder=1,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + 0.12,
        label,
        ha="center",
        va="bottom",
        fontsize=fs,
        family="DejaVu Sans Mono",
        color="#111827",
        zorder=3,
    )
    return box


def add_arrow(
    ax,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: str = COLORS["line"],
    lw: float = 1.1,
    ls: str = "-",
    rad: float = 0.0,
    scale: float = 9,
):
    arrow = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=scale,
        color=color,
        linewidth=lw,
        linestyle=ls,
        shrinkA=2,
        shrinkB=2,
        connectionstyle=f"arc3,rad={rad}",
        zorder=4,
    )
    ax.add_patch(arrow)
    return arrow


def add_tiny_stack(ax, x: float, y: float, vertical: bool = False) -> None:
    if vertical:
        blocks = [("Conv\n3x3", COLORS["conv"]), ("GELU", COLORS["act"]), ("Conv\n3x3", COLORS["conv"])]
        for i, (txt, col) in enumerate(blocks):
            add_round_box(ax, x, y + i * 0.44, 0.50, 0.34, txt, col, fs=6.3, lw=0.8)
    else:
        blocks = [("Conv\n3x3", COLORS["conv"]), ("GELU", COLORS["act"]), ("Conv\n3x3", COLORS["conv"])]
        for i, (txt, col) in enumerate(blocks):
            add_round_box(ax, x + i * 0.58, y, 0.50, 0.42, txt, col, fs=6.3, lw=0.8)


def draw_overview(ax) -> None:
    add_group_box(
        ax,
        0.20,
        0.55,
        12.00,
        6.95,
        "(a) SensorField-M3T overall pipeline",
        fc="#fffdf4",
        ec="#d39a1e",
        fs=9.0,
    )
    y_map = {"raw": 5.90, "stf": 4.05, "gaf": 2.20}
    names = {
        "raw": ("Raw signal", "[B, 1, L]"),
        "stf": ("STF map", "[B, 1, H, W]"),
        "gaf": ("GAF image", "[B, 1, H, W]"),
    }
    token_names = {"raw": "T_raw", "stf": "T_stf", "gaf": "T_gaf"}
    colors = {"raw": COLORS["raw"], "stf": COLORS["stf"], "gaf": COLORS["gaf"]}

    for key, y in y_map.items():
        add_round_box(ax, 0.45, y, 1.05, 0.72, f"{names[key][0]}\n{names[key][1]}", colors[key], fs=7.8)
        add_group_box(ax, 1.82, y - 0.18, 1.90, 1.05, "View Encoder", fc="#fffefa", ec="#d1a447", fs=7.2)
        if key == "raw":
            add_tiny_stack(ax, 1.98, y + 0.16, vertical=False)
            ax.text(2.74, y + 0.74, "Conv1D", ha="center", va="center", fontsize=6.8)
        else:
            add_tiny_stack(ax, 1.98, y + 0.16, vertical=False)
            ax.text(2.74, y + 0.74, "Conv2D", ha="center", va="center", fontsize=6.8)
        add_round_box(ax, 4.02, y, 0.96, 0.72, f"{token_names[key]}\n[B,N_v,D]", COLORS["token"], fs=7.8, weight="bold")
        add_arrow(ax, (1.50, y + 0.36), (1.82, y + 0.36))
        add_arrow(ax, (3.72, y + 0.36), (4.02, y + 0.36))
        add_arrow(ax, (4.98, y + 0.36), (5.38, 4.18), rad=0.06 if y > 4.1 else -0.06)

    add_group_box(ax, 5.38, 1.62, 2.35, 4.98, "FAC", fc="#fffaf0", ec="#d39a1e", fs=7.8)
    add_round_box(ax, 5.58, 5.66, 0.82, 0.42, "A0\n[K,D]", COLORS["pool"], fs=7.0)
    add_round_box(ax, 6.55, 5.66, 0.90, 0.42, "Delta A(x)\nMLP", COLORS["act"], fs=6.9)
    add_arrow(ax, (6.40, 5.87), (6.55, 5.87))
    add_round_box(ax, 5.63, 4.70, 1.78, 0.52, "Anchor-to-view\nCross-Attention", COLORS["fac"], fs=7.2, weight="bold")
    add_round_box(ax, 5.63, 3.75, 1.78, 0.52, "Agreement\nSoftmax(w_v,k)", COLORS["fac"], fs=7.2)
    add_round_box(ax, 5.63, 2.84, 1.78, 0.52, "Shared anchors\nS [B,K,D]", COLORS["fac"], fs=7.2, weight="bold")
    add_round_box(ax, 5.63, 1.94, 1.78, 0.52, "Complement\nC_v = A_v - Proj_S", COLORS["fac"], fs=7.0)

    add_group_box(ax, 8.10, 2.05, 1.72, 3.72, "TAEF", fc="#fbf9ff", ec="#8d7fd0", fs=7.8)
    add_round_box(ax, 8.28, 4.86, 1.36, 0.48, "Evidence bank\n{S,C_v}", COLORS["taef"], fs=7.0)
    add_round_box(ax, 8.28, 3.92, 1.36, 0.48, "Task queries\nq_t", COLORS["taef"], fs=7.0)
    add_round_box(ax, 8.28, 3.00, 1.36, 0.48, "alpha_t,e\nSoftmax", COLORS["taef"], fs=7.0, weight="bold")
    add_round_box(ax, 8.28, 2.28, 1.36, 0.42, "R_t [B,D]", COLORS["taef"], fs=7.2, weight="bold")

    add_group_box(ax, 10.15, 2.28, 1.58, 3.26, "GCTI", fc="#f7feff", ec="#58a9b5", fs=7.8)
    add_round_box(ax, 10.32, 4.78, 1.22, 0.44, "Stack\n[B,T,D]", COLORS["gcti"], fs=7.0)
    add_round_box(ax, 10.32, 3.86, 1.22, 0.52, "Task attention\n+ bias B_T", COLORS["gcti"], fs=7.0, weight="bold")
    add_round_box(ax, 10.32, 2.94, 1.22, 0.44, "Relation\n[B,T,T]", COLORS["gcti"], fs=7.0)
    add_round_box(ax, 10.32, 2.42, 1.22, 0.36, "Z_t [B,D]", COLORS["gcti"], fs=7.0, weight="bold")

    add_arrow(ax, (7.73, 4.18), (8.10, 4.45))
    add_arrow(ax, (9.82, 3.15), (10.15, 3.98))
    add_round_box(ax, 11.92, 4.78, 0.18, 0.50, " ", COLORS["head"], fs=6.0, weight="bold")
    add_round_box(ax, 11.92, 3.72, 0.18, 0.50, " ", COLORS["head"], fs=6.0, weight="bold")
    add_round_box(ax, 11.92, 2.66, 0.18, 0.50, " ", COLORS["head"], fs=6.0)
    ax.text(12.14, 5.03, "Event\nhead", ha="left", va="center", fontsize=7.1, fontweight="bold")
    ax.text(12.14, 3.97, "Location\nhead", ha="left", va="center", fontsize=7.1, fontweight="bold")
    ax.text(12.14, 2.91, "Other\nheads", ha="left", va="center", fontsize=6.8)
    add_arrow(ax, (11.73, 3.92), (11.92, 5.03), rad=0.18)
    add_arrow(ax, (11.73, 3.92), (11.92, 3.97))
    add_arrow(ax, (11.73, 3.92), (11.92, 2.91), rad=-0.18)

    add_round_box(ax, 5.38, 0.88, 2.35, 0.42, "L_FAC: alignment + decorrelation", COLORS["loss"], fs=6.8, ls="--")
    add_round_box(ax, 8.10, 0.88, 1.72, 0.42, "L_TAEF: diversity", COLORS["loss"], fs=6.8, ls="--")
    add_round_box(ax, 10.15, 0.88, 1.58, 0.42, "L_GCTI: relation", COLORS["loss"], fs=6.8, ls="--")


def draw_fac_panel(ax) -> None:
    add_group_box(ax, 12.55, 4.52, 7.40, 2.98, "(b) Field-Anchor Complementation (FAC)", fc="#fffdf6", ec="#aa7a7a", fs=9.0)
    add_round_box(ax, 12.80, 6.64, 0.44, 0.52, "T_v", COLORS["token"], fs=7.0, weight="bold")
    add_round_box(ax, 13.42, 6.50, 1.02, 0.80, "Mean Pool\nConcat", COLORS["act"], fs=7.0)
    add_round_box(ax, 14.68, 6.50, 0.96, 0.80, "MLP\nTanh", COLORS["conv"], fs=7.0)
    add_round_box(ax, 15.88, 6.72, 0.64, 0.36, "A0", COLORS["pool"], fs=7.0, weight="bold")
    add_round_box(ax, 15.88, 6.24, 0.64, 0.36, "Delta A", COLORS["act"], fs=6.8)
    ax.text(16.72, 6.50, "+", ha="center", va="center", fontsize=13, fontweight="bold")
    add_round_box(ax, 17.04, 6.36, 0.62, 0.50, "A", COLORS["fac"], fs=8.0, weight="bold")
    add_arrow(ax, (13.24, 6.90), (13.42, 6.90))
    add_arrow(ax, (14.44, 6.90), (14.68, 6.90))
    add_arrow(ax, (15.64, 6.90), (15.88, 6.90))
    add_arrow(ax, (16.52, 6.90), (17.04, 6.65))

    for i, (name, y) in enumerate([("raw", 5.62), ("stf", 5.18), ("gaf", 4.74)]):
        add_round_box(ax, 12.84, y, 0.70, 0.30, f"T_{name}", COLORS["token"], fs=6.4)
        add_round_box(ax, 13.84, y, 1.18, 0.30, "Cross-Attn", COLORS["fac"], fs=6.4)
        add_round_box(ax, 15.34, y, 0.78, 0.30, f"A_{name}", COLORS["pool"], fs=6.4)
        add_arrow(ax, (13.54, y + 0.15), (13.84, y + 0.15))
        add_arrow(ax, (15.02, y + 0.15), (15.34, y + 0.15))
        add_arrow(ax, (17.35, 6.36), (14.42, y + 0.30), rad=0.08 - 0.05 * i, lw=0.9)

    add_round_box(ax, 16.54, 5.40, 1.02, 0.62, "Cosine\nagreement", COLORS["act"], fs=6.8)
    add_round_box(ax, 17.78, 5.40, 0.86, 0.62, "Softmax\nw_v,k", COLORS["conv"], fs=6.8)
    add_round_box(ax, 18.88, 5.40, 0.70, 0.62, "S", COLORS["fac"], fs=9.0, weight="bold")
    add_arrow(ax, (16.12, 5.34), (16.54, 5.70))
    add_arrow(ax, (17.56, 5.70), (17.78, 5.70))
    add_arrow(ax, (18.64, 5.70), (18.88, 5.70))

    add_round_box(ax, 16.54, 4.78, 1.32, 0.36, "Orthogonal projection", COLORS["fac"], fs=6.6)
    add_round_box(ax, 18.12, 4.78, 1.46, 0.36, "C_v = A_v - Proj_S(A_v)", COLORS["fac"], fs=6.3, weight="bold")
    add_arrow(ax, (19.23, 5.40), (17.20, 5.14), rad=-0.16, lw=0.9)
    add_arrow(ax, (17.86, 4.96), (18.12, 4.96))


def draw_taef_panel(ax) -> None:
    add_group_box(ax, 12.55, 1.92, 3.80, 2.25, "(c) TAEF", fc="#fbf9ff", ec="#8d7fd0", fs=8.7)
    add_round_box(ax, 12.78, 3.42, 0.72, 0.42, "S", COLORS["fac"], fs=8.0, weight="bold")
    add_round_box(ax, 12.78, 2.85, 0.72, 0.42, "C_raw", COLORS["token"], fs=6.5)
    add_round_box(ax, 12.78, 2.28, 0.72, 0.42, "C_stf\nC_gaf", COLORS["token"], fs=6.4)
    add_round_box(ax, 13.82, 2.45, 0.78, 1.10, "Evidence\nBank E", COLORS["taef"], fs=7.0, weight="bold")
    add_round_box(ax, 14.88, 3.36, 0.62, 0.36, "q_e", COLORS["pool"], fs=6.8)
    add_round_box(ax, 14.88, 2.88, 0.62, 0.36, "q_l", COLORS["pool"], fs=6.8)
    add_round_box(ax, 14.82, 2.30, 0.90, 0.38, "QK^T\nsoftmax", COLORS["conv"], fs=6.5)
    add_round_box(ax, 15.08, 1.98, 0.84, 0.28, "alpha_t,e", COLORS["act"], fs=6.5, weight="bold")
    add_arrow(ax, (13.50, 3.63), (13.82, 3.20))
    add_arrow(ax, (13.50, 3.06), (13.82, 3.00))
    add_arrow(ax, (13.50, 2.49), (13.82, 2.82))
    add_arrow(ax, (14.60, 3.00), (14.82, 2.49))
    add_arrow(ax, (15.50, 3.06), (14.82, 2.49), rad=-0.20)
    add_arrow(ax, (15.72, 2.49), (16.10, 2.92))
    add_round_box(ax, 16.00, 2.64, 0.24, 0.70, "R_t", COLORS["taef"], fs=6.8, weight="bold")


def draw_gcti_panel(ax) -> None:
    add_group_box(ax, 16.55, 1.92, 3.40, 2.25, "(d) GCTI", fc="#f7feff", ec="#58a9b5", fs=8.7)
    add_round_box(ax, 16.78, 3.34, 0.72, 0.42, "R_e", COLORS["taef"], fs=7.0, weight="bold")
    add_round_box(ax, 16.78, 2.84, 0.72, 0.42, "R_l", COLORS["taef"], fs=7.0, weight="bold")
    add_round_box(ax, 17.82, 2.70, 0.62, 1.02, "LN\nQ K V", COLORS["gcti"], fs=7.0)
    add_round_box(ax, 18.72, 3.15, 0.52, 0.42, "B_T", COLORS["pool"], fs=7.2, weight="bold")
    add_round_box(ax, 18.60, 2.54, 0.92, 0.42, "Softmax", COLORS["conv"], fs=6.8)
    add_round_box(ax, 19.72, 2.78, 0.48, 0.58, "Z_t", COLORS["gcti"], fs=7.8, weight="bold")
    add_arrow(ax, (17.50, 3.55), (17.82, 3.25))
    add_arrow(ax, (17.50, 3.05), (17.82, 3.02))
    add_arrow(ax, (18.44, 3.21), (18.60, 2.75))
    add_arrow(ax, (19.24, 3.36), (18.86, 2.96), rad=-0.15)
    add_arrow(ax, (19.52, 2.75), (19.72, 3.08))


def draw_encoder_panel(ax) -> None:
    add_group_box(ax, 0.20, 7.72, 19.75, 1.05, "(e) Encoder and residual micro-blocks used in view-specific tokenization", fc="#fbfbfb", ec="#b7b7b7", fs=8.8)
    add_round_box(ax, 0.55, 8.08, 1.18, 0.36, "Conv1D/2D", COLORS["conv"], fs=7.0)
    add_round_box(ax, 1.95, 8.08, 0.72, 0.36, "BN", COLORS["act"], fs=7.0)
    add_round_box(ax, 2.90, 8.08, 0.82, 0.36, "GELU", COLORS["act"], fs=7.0)
    add_round_box(ax, 3.95, 8.08, 1.30, 0.36, "Dropout", COLORS["conv"], fs=7.0)
    add_round_box(ax, 5.52, 8.08, 1.52, 0.36, "Adaptive Pool", COLORS["pool"], fs=7.0)
    add_round_box(ax, 7.32, 8.08, 1.25, 0.36, "LayerNorm", COLORS["token"], fs=7.0)
    for x0, x1 in [(1.73, 1.95), (2.67, 2.90), (3.72, 3.95), (5.25, 5.52), (7.04, 7.32)]:
        add_arrow(ax, (x0, 8.26), (x1, 8.26), scale=7)
    ax.text(
        9.10,
        8.26,
        "Output tokens: raw [B,N_raw,D], stf [B,N_stf,D], gaf [B,N_gaf,D]",
        ha="left",
        va="center",
        fontsize=7.4,
        color="#374151",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(20.8, 9.1))
    ax.set_xlim(0, 20.3)
    ax.set_ylim(0.35, 9.15)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    draw_overview(ax)
    draw_fac_panel(ax)
    draw_taef_panel(ax)
    draw_gcti_panel(ax)
    draw_encoder_panel(ax)

    ax.text(
        10.15,
        9.10,
        "SensorField-M3T: Field-Anchor Multimodal Multi-Task Transformer",
        ha="center",
        va="top",
        fontsize=13.5,
        fontweight="bold",
        color="#111827",
    )

    png_path = OUTPUT_DIR / "sensorfield_m3t_architecture_detailed.png"
    pdf_path = OUTPUT_DIR / "sensorfield_m3t_architecture_detailed.pdf"
    fig.savefig(png_path, dpi=350, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
