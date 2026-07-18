from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "paper_assets"
    / "sensorfield_m3t_experiments"
    / "multiseed_test_protocol_values.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "paper_assets" / "sensorfield_m3t_experiments" / "cross_condition_radar"
)


CASES = [
    ("full_three_view", "In-distribution"),
    ("region_generalization", "Region-level"),
    ("soil_generalization", "Soil-level"),
    ("acquisition_generalization", "Acquisition-level"),
]


def _as_float(value: str | None) -> float:
    if value is None:
        return math.nan
    value = value.strip()
    if not value:
        return math.nan
    try:
        parsed = float(value)
    except ValueError:
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def _fmt(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.6f}"


def _plot_value(value: float, transform: str) -> float:
    """Return a finite radar value in [0, 1.05].

    Raw TaskScore/MTLScore values are already in [0, 1]. FAR is lower-better, so
    we plot 1-FAR while keeping the source CSV as FAR. Delta MTLScore is centered
    at the in-distribution baseline, so 1+Delta keeps the axis higher-is-better.
    Undefined values are drawn at the inner radius and kept blank in the CSV.
    """
    if math.isnan(value):
        return 0.0
    if transform == "one_minus":
        return 1.0 - value
    if transform == "one_plus":
        return 1.0 + value
    return value


def load_case_rows(input_csv: Path) -> dict[str, dict[str, str]]:
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = {
            row["case_name"]: row
            for row in reader
            if row.get("split") == "test" and row.get("case_name") in dict(CASES)
        }

    missing = [case for case, _ in CASES if case not in rows]
    if missing:
        raise RuntimeError(
            f"Missing test rows for {missing} in {input_csv}. "
            "Regenerate multiseed_test_protocol_values.csv first."
        )
    return rows


def build_values(rows: dict[str, dict[str, str]]) -> list[dict[str, float | str]]:
    baseline = _as_float(rows["full_three_view"].get("mtl_score_mean"))
    if math.isnan(baseline):
        raise RuntimeError("The in-distribution baseline MTLScore is undefined.")

    records: list[dict[str, float | str]] = []
    for case_name, setting in CASES:
        row = rows[case_name]
        mtl_score = _as_float(row.get("mtl_score_mean"))
        delta_mtl_score = math.nan if math.isnan(mtl_score) else mtl_score - baseline
        record: dict[str, float | str] = {
            "setting": setting,
            "case_name": case_name,
            "event_task_score": _as_float(row.get("event_task_score_mean")),
            "event_far": _as_float(row.get("event_far_mean")),
            "location_task_score": _as_float(row.get("location_task_score_mean")),
            "location_far": _as_float(row.get("location_far_mean")),
            "mtl_score": mtl_score,
            "delta_mtl_score": delta_mtl_score,
        }
        record["plot_event_task_score"] = _plot_value(
            float(record["event_task_score"]), "identity"
        )
        record["plot_event_far_inverted"] = _plot_value(
            float(record["event_far"]), "one_minus"
        )
        record["plot_location_task_score"] = _plot_value(
            float(record["location_task_score"]), "identity"
        )
        record["plot_location_far_inverted"] = _plot_value(
            float(record["location_far"]), "one_minus"
        )
        record["plot_mtl_score"] = _plot_value(float(record["mtl_score"]), "identity")
        record["plot_delta_mtl_score_shifted"] = _plot_value(
            float(record["delta_mtl_score"]), "one_plus"
        )
        records.append(record)
    return records


def write_values_csv(records: list[dict[str, float | str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "setting",
        "case_name",
        "event_task_score",
        "event_far",
        "location_task_score",
        "location_far",
        "mtl_score",
        "delta_mtl_score",
        "plot_event_task_score",
        "plot_event_far_inverted",
        "plot_location_task_score",
        "plot_location_far_inverted",
        "plot_mtl_score",
        "plot_delta_mtl_score_shifted",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (
                        record[key]
                        if isinstance(record[key], str)
                        else _fmt(float(record[key]))
                    )
                    for key in fields
                }
            )


def draw_radar(records: list[dict[str, float | str]], output_dir: Path) -> None:
    labels = [
        "Event\nTaskScore ↑",
        "Event\nFAR ↓",
        "Location\nTaskScore ↑",
        "Location\nFAR ↓",
        "MTLScore ↑",
        "Delta\nMTLScore ↑",
    ]
    value_keys = [
        "plot_event_task_score",
        "plot_event_far_inverted",
        "plot_location_task_score",
        "plot_location_far_inverted",
        "plot_mtl_score",
        "plot_delta_mtl_score_shifted",
    ]
    colors = ["#1F77B4", "#D95F02", "#7570B3", "#1B9E77"]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    closed_angles = angles + angles[:1]

    fig = plt.figure(figsize=(9.0, 8.2), dpi=220)
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", pad=18)

    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.75, alpha=0.48)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.75, alpha=0.32)
    ax.spines["polar"].set_color("#444444")
    ax.spines["polar"].set_linewidth(1.0)

    for record, color in zip(records, colors):
        values = [float(record[key]) for key in value_keys]
        closed_values = values + values[:1]
        setting = str(record["setting"])
        if setting == "Soil-level":
            setting = "Soil-level (N/A score axes)"
        ax.plot(
            closed_angles,
            closed_values,
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=5.2,
            label=setting,
        )
        ax.fill(closed_angles, closed_values, color=color, alpha=0.12)

    ax.set_title(
        "Cross-condition Generalization Radar",
        fontsize=18,
        fontweight="bold",
        pad=34,
    )
    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=False,
        fontsize=13,
        handlelength=2.4,
        columnspacing=1.8,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.text(
        0.5,
        0.018,
        "FAR axes are plotted as 1−FAR; Delta MTLScore is plotted as 1+Δ relative to the in-distribution setting. "
        "Undefined soil-level Event TaskScore/MTLScore are shown at the inner radius.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#333333",
    )
    fig.subplots_adjust(left=0.08, right=0.92, top=0.86, bottom=0.22)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig_cross_condition_radar.png", bbox_inches="tight")
    fig.savefig(output_dir / "fig_cross_condition_radar.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cross-condition radar chart for SensorField-M3T."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = load_case_rows(args.input)
    records = build_values(rows)
    write_values_csv(records, args.output_dir / "cross_condition_radar_values.csv")
    draw_radar(records, args.output_dir)
    print(f"Saved radar chart and values to {args.output_dir}")


if __name__ == "__main__":
    main()
