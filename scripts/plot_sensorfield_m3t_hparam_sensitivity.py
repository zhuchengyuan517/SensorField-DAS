from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_RUN_ROOT = PROJECT_ROOT / "_tmp_sensorfield_mtl43_sensitivity" / "runs"
NEW_RUN_ROOT = PROJECT_ROOT / "_tmp_sensorfield_mtl43_hparam_sensitivity" / "runs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "paper_assets" / "sensorfield_m3t_experiments" / "hparam_sensitivity"

BASELINE = {
    "num_anchors": 8,
    "fac_loss_weight": 0.05,
    "taef_loss_weight": 0.01,
    "gcti_loss_weight": 0.01,
    "view_drop_prob": 0.0,
}

EXPECTED_SWEEPS = OrderedDict(
    [
        ("num_anchors", (4, 8, 16, 32)),
        ("fac_loss_weight", (0.0, 0.01, 0.05, 0.1, 0.2)),
        ("taef_loss_weight", (0.0, 0.01, 0.05, 0.1)),
        ("gcti_loss_weight", (0.0, 0.01, 0.05, 0.1)),
        ("view_drop_prob", (0.0, 0.1, 0.2, 0.3, 0.5)),
    ]
)

PANEL_LABELS = {
    "num_anchors": ("(a) Anchor bank size", r"$K$"),
    "fac_loss_weight": ("(b) FAC loss weight", r"$\lambda_{\mathrm{FAC}}$"),
    "taef_loss_weight": ("(c) TAEF loss weight", r"$\lambda_{\mathrm{TAEF}}$"),
    "gcti_loss_weight": ("(d) GCTI loss weight", r"$\lambda_{\mathrm{GCTI}}$"),
    "view_drop_prob": ("(e) View perturbation probability", r"$p_{\mathrm{drop}}$"),
}


@dataclass(frozen=True)
class RunRecord:
    group: str
    value: float
    seed: int
    split: str
    event_score: float
    location_score: float
    mtl_score: float
    best_epoch: int
    run_dir: Path
    modified_time: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot SensorField-M3T one-factor hyperparameter sensitivity results."
    )
    parser.add_argument(
        "--run_roots",
        nargs="+",
        default=[str(LEGACY_RUN_ROOT), str(NEW_RUN_ROOT)],
        help="One or more roots containing timestamped runs with run_config.json and summary.json.",
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT), type=str)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--min_epochs", default=1, type=int)
    parser.add_argument("--strict_mtl43", action="store_true", default=True)
    parser.add_argument("--dpi", default=400, type=int)
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
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


def close_float(left: float, right: float, tol: float = 1e-9) -> bool:
    return abs(float(left) - float(right)) <= tol


def config_is_clean_full_mtl43(config: dict, strict_mtl43: bool) -> bool:
    if config.get("model") != "sensorfield_m3t":
        return False
    if str(config.get("enabled_views", "raw,stf,gaf")) != "raw,stf,gaf":
        return False
    for flag in ("disable_fac", "disable_complement", "disable_taef", "disable_gcti"):
        if bool(config.get(flag, False)):
            return False
    if strict_mtl43:
        dataset_path = str(config.get("dataset_path", "")).replace("/", "\\").rstrip("\\")
        if not dataset_path.endswith(r"converted_csv\MTL43"):
            return False
    return True


def get_numeric(config: dict, key: str) -> float:
    return float(config.get(key, BASELINE[key]))


def infer_groups(config: dict) -> list[tuple[str, float]]:
    num_anchors = int(config.get("num_anchors", BASELINE["num_anchors"]))
    fac = get_numeric(config, "fac_loss_weight")
    taef = get_numeric(config, "taef_loss_weight")
    gcti = get_numeric(config, "gcti_loss_weight")
    view_drop = get_numeric(config, "view_drop_prob")

    groups: list[tuple[str, float]] = []
    if (
        close_float(fac, BASELINE["fac_loss_weight"])
        and close_float(taef, BASELINE["taef_loss_weight"])
        and close_float(gcti, BASELINE["gcti_loss_weight"])
        and close_float(view_drop, BASELINE["view_drop_prob"])
    ):
        groups.append(("num_anchors", float(num_anchors)))
    if (
        num_anchors == BASELINE["num_anchors"]
        and close_float(taef, BASELINE["taef_loss_weight"])
        and close_float(gcti, BASELINE["gcti_loss_weight"])
        and close_float(view_drop, BASELINE["view_drop_prob"])
    ):
        groups.append(("fac_loss_weight", fac))
    if (
        num_anchors == BASELINE["num_anchors"]
        and close_float(fac, BASELINE["fac_loss_weight"])
        and close_float(gcti, BASELINE["gcti_loss_weight"])
        and close_float(view_drop, BASELINE["view_drop_prob"])
    ):
        groups.append(("taef_loss_weight", taef))
    if (
        num_anchors == BASELINE["num_anchors"]
        and close_float(fac, BASELINE["fac_loss_weight"])
        and close_float(taef, BASELINE["taef_loss_weight"])
        and close_float(view_drop, BASELINE["view_drop_prob"])
    ):
        groups.append(("gcti_loss_weight", gcti))
    if (
        num_anchors == BASELINE["num_anchors"]
        and close_float(fac, BASELINE["fac_loss_weight"])
        and close_float(taef, BASELINE["taef_loss_weight"])
        and close_float(gcti, BASELINE["gcti_loss_weight"])
    ):
        groups.append(("view_drop_prob", view_drop))
    return groups


def iter_summary_paths(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        yield from root.rglob("summary.json")


def collect_records(roots: list[Path], split: str, min_epochs: int, strict_mtl43: bool) -> list[RunRecord]:
    metric_key = "best_val_metrics" if split == "val" else "best_test_metrics"
    records_by_key: dict[tuple[str, float, int], RunRecord] = {}
    for summary_path in iter_summary_paths(roots):
        run_dir = summary_path.parent
        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            continue
        try:
            config = load_json(config_path)
            summary = load_json(summary_path)
        except Exception:
            continue
        if not config_is_clean_full_mtl43(config, strict_mtl43):
            continue
        if int(config.get("epochs", 0)) < int(min_epochs):
            continue
        metrics = summary.get(metric_key)
        if not isinstance(metrics, dict):
            continue
        seed = int(config.get("seed", -1))
        event_score = float(metrics.get("event_acc", 0.0))
        location_score = float(metrics.get("location_acc", 0.0))
        mtl_score = float(metrics.get("score", 0.5 * (event_score + location_score)))
        modified_time = summary_path.stat().st_mtime
        for group, value in infer_groups(config):
            key = (group, float(value), seed)
            record = RunRecord(
                group=group,
                value=float(value),
                seed=seed,
                split=split,
                event_score=event_score,
                location_score=location_score,
                mtl_score=mtl_score,
                best_epoch=int(summary.get("best_epoch", 0)),
                run_dir=run_dir,
                modified_time=modified_time,
            )
            previous = records_by_key.get(key)
            if previous is None or previous.modified_time < modified_time:
                records_by_key[key] = record
    return sorted(records_by_key.values(), key=lambda item: (item.group, item.value, item.seed))


def summarize(records: list[RunRecord]) -> tuple[list[dict], list[dict], list[dict]]:
    per_seed_rows = []
    for record in records:
        per_seed_rows.append(
            {
                "group": record.group,
                "value": record.value,
                "seed": record.seed,
                "split": record.split,
                "event_score": record.event_score,
                "location_score": record.location_score,
                "mtl_score": record.mtl_score,
                "best_epoch": record.best_epoch,
                "run_dir": str(record.run_dir),
            }
        )

    summary_rows = []
    for group, expected_values in EXPECTED_SWEEPS.items():
        for value in expected_values:
            subset = [record for record in records if record.group == group and close_float(record.value, float(value))]
            if not subset:
                continue
            row = {
                "group": group,
                "value": float(value),
                "num_seeds": len(subset),
            }
            for metric_name in ("event_score", "location_score", "mtl_score"):
                values = np.asarray([getattr(record, metric_name) for record in subset], dtype=np.float64)
                row[f"{metric_name}_mean"] = float(values.mean())
                row[f"{metric_name}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            summary_rows.append(row)

    status_rows = []
    for group, expected_values in EXPECTED_SWEEPS.items():
        available = sorted(
            {
                float(record.value)
                for record in records
                if record.group == group and any(close_float(record.value, float(item)) for item in expected_values)
            }
        )
        missing = [float(item) for item in expected_values if not any(close_float(item, have) for have in available)]
        status_rows.append(
            {
                "group": group,
                "expected_values": ",".join(f"{float(item):g}" for item in expected_values),
                "available_values": ",".join(f"{item:g}" for item in available),
                "missing_values": ",".join(f"{item:g}" for item in missing),
                "status": "complete" if not missing else "partial",
            }
        )
    return per_seed_rows, summary_rows, status_rows


def plot_summary(summary_rows: list[dict], status_rows: list[dict], output_root: Path, split: str, dpi: int) -> None:
    summary_by_group: dict[str, dict[float, dict]] = {group: {} for group in EXPECTED_SWEEPS}
    for row in summary_rows:
        summary_by_group[row["group"]][float(row["value"])] = row
    status_by_group = {row["group"]: row for row in status_rows}

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
        }
    )
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharey=True)
    metric_styles = [
        ("event_score", "Event TaskScore proxy", "#B64533", "o"),
        ("location_score", "Location TaskScore proxy", "#2E5EAA", "s"),
        ("mtl_score", "MTLScore proxy", "#2F855A", "^"),
    ]

    for ax, (group, expected_values) in zip(axes, EXPECTED_SWEEPS.items()):
        title, xlabel = PANEL_LABELS[group]
        x_positions = np.arange(len(expected_values))
        ax.set_title(title, pad=10, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{float(item):g}" for item in expected_values])
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_ylim(0.35, 1.03)

        for metric_name, label, color, marker in metric_styles:
            xs = []
            ys = []
            yerrs = []
            for index, value in enumerate(expected_values):
                row = summary_by_group[group].get(float(value))
                if row is None:
                    continue
                xs.append(index)
                ys.append(float(row[f"{metric_name}_mean"]))
                yerrs.append(float(row[f"{metric_name}_std"]))
            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerrs,
                    label=label,
                    color=color,
                    marker=marker,
                    linewidth=2.2,
                    markersize=6,
                    capsize=3,
                    alpha=0.96,
                )

        missing_values = status_by_group[group]["missing_values"]
        if missing_values:
            ax.text(
                0.5,
                0.08,
                f"pending: {missing_values}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#666666",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F3F3F3", "edgecolor": "#BBBBBB"},
            )

    axes[0].set_ylabel(f"{split.capitalize()} score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output_root.mkdir(parents=True, exist_ok=True)
    png_path = output_root / f"sensorfield_m3t_hparam_sensitivity_{split}.png"
    pdf_path = output_root / f"sensorfield_m3t_hparam_sensitivity_{split}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {png_path}")
    print(f"Saved plot: {pdf_path}")


def main() -> int:
    args = parse_args()
    roots = [Path(item).expanduser().resolve() for item in args.run_roots]
    output_root = Path(args.output_root).expanduser().resolve()
    records = collect_records(
        roots=roots,
        split=args.split,
        min_epochs=args.min_epochs,
        strict_mtl43=bool(args.strict_mtl43),
    )
    per_seed_rows, summary_rows, status_rows = summarize(records)
    write_csv(output_root / f"hparam_sensitivity_per_seed_{args.split}.csv", per_seed_rows)
    write_csv(output_root / f"hparam_sensitivity_summary_{args.split}.csv", summary_rows)
    write_csv(output_root / "hparam_sensitivity_status.csv", status_rows)
    plot_summary(summary_rows, status_rows, output_root, args.split, args.dpi)
    print(f"Saved CSV files to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
