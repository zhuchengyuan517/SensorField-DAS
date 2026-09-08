from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


METRIC_COLUMNS = [
    "event_acc",
    "event_macro_f1",
    "event_auc",
    "event_far",
    "event_task_score",
    "location_acc",
    "location_macro_f1",
    "location_auc",
    "location_far",
    "location_task_score",
    "mtl_score",
]

ABLATION_CASES = ["full_three_view", "wo_fac", "wo_taef", "wo_gcti", "wo_all"]
CROSS_CASES = ["full_three_view", "region_generalization", "soil_generalization", "acquisition_generalization"]
VIEW_CASES = ["raw_only", "stf_only", "gaf_only", "raw_stf", "raw_gaf", "stf_gaf", "full_three_view"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TPAMI SensorField-M3T tables and figures from run artifacts.")
    parser.add_argument("--runs_root", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_case_and_seed(run_dir: Path, runs_root: Path, config: dict[str, Any]) -> tuple[str, int | None]:
    relative_parts = run_dir.relative_to(runs_root).parts
    case_name = relative_parts[0] if relative_parts else config.get("case_name", "unknown")
    seed = config.get("seed")
    for part in relative_parts:
        if part.startswith("seed_"):
            try:
                seed = int(part.split("_", 1)[1])
            except ValueError:
                pass
    return str(case_name), int(seed) if seed is not None else None


def collect_runs(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(runs_root.rglob("summary.json")):
        run_dir = summary_path.parent
        summary = load_json(summary_path)
        config_path = run_dir / "run_config.json"
        config = load_json(config_path) if config_path.is_file() else {}
        metrics = summary.get("best_test_metrics") or {}
        case_name, seed = infer_case_and_seed(run_dir, runs_root, config)
        row = {
            "case": case_name,
            "seed": seed,
            "run_dir": str(run_dir),
            "best_epoch": summary.get("best_epoch"),
            "selection_metric": summary.get("selection_metric", config.get("selection_metric", "")),
        }
        for column in METRIC_COLUMNS:
            row[column] = metrics.get(column, np.nan)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["case"]].append(row)
    summary_rows = []
    for case_name, case_rows in sorted(grouped.items()):
        out = {"case": case_name, "runs": len(case_rows)}
        for column in METRIC_COLUMNS:
            values = np.asarray([float(row[column]) for row in case_rows if row.get(column) not in ("", None)], dtype=float)
            values = values[np.isfinite(values)]
            out[f"{column}_mean"] = float(values.mean()) if values.size else np.nan
            out[f"{column}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary_rows.append(out)
    return summary_rows


def _fmt_mean_std(row: dict[str, Any], metric: str) -> str:
    mean = row.get(f"{metric}_mean", np.nan)
    std = row.get(f"{metric}_std", np.nan)
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def write_latex_table(path: Path, rows: list[dict[str, Any]], cases: list[str], caption: str, label: str) -> None:
    row_map = {row["case"]: row for row in rows}
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Case & Event TaskScore & Location TaskScore & MTLScore & Event ACC & Location ACC & Runs \\\\",
        "\\midrule",
    ]
    for case_name in cases:
        if case_name not in row_map:
            continue
        row = row_map[case_name]
        lines.append(
            f"{case_name.replace('_', ' ')} & "
            f"{_fmt_mean_std(row, 'event_task_score')} & "
            f"{_fmt_mean_std(row, 'location_task_score')} & "
            f"{_fmt_mean_std(row, 'mtl_score')} & "
            f"{_fmt_mean_std(row, 'event_acc')} & "
            f"{_fmt_mean_std(row, 'location_acc')} & "
            f"{int(row.get('runs', 0))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def read_matrix_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        labels = [name for name in (reader.fieldnames or []) if name != "label"]
        rows = []
        for row in reader:
            rows.append([int(float(row[label])) for label in labels])
    return labels, np.asarray(rows, dtype=np.int64)


def render_confusion(matrix_path: Path, output_dir: Path, title: str) -> list[str]:
    if plt is None:
        return []
    labels, matrix = read_matrix_csv(matrix_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, str(int(matrix[row_idx, col_idx])), ha="center", va="center", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    stem = matrix_path.stem
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [str(png_path), str(pdf_path)]


def render_view_contribution(summary_rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    if plt is None:
        return []
    row_map = {row["case"]: row for row in summary_rows}
    available = [case for case in VIEW_CASES if case in row_map]
    if not available:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [case.replace("_", "+") for case in available]
    x = np.arange(len(available))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    series = [
        ("event_task_score_mean", "Event TaskScore", "#B64533"),
        ("location_task_score_mean", "Location TaskScore", "#2E5EAA"),
        ("mtl_score_mean", "MTLScore", "#2F855A"),
    ]
    for idx, (column, label, color) in enumerate(series):
        values = [float(row_map[case].get(column, np.nan)) for case in available]
        ax.bar(x + (idx - 1) * width, values, width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    png_path = output_dir / "fig_z1_view_contributions.png"
    pdf_path = output_dir / "fig_z1_view_contributions.pdf"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [str(png_path), str(pdf_path)]


def collect_per_class(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        history_dir = Path(str(row["run_dir"])) / "history"
        for task_name in ("event", "location"):
            report_path = history_dir / f"best_test_{task_name}_report.csv"
            if not report_path.is_file():
                continue
            with report_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for report_row in csv.DictReader(handle):
                    item = {"case": row["case"], "seed": row["seed"], "run_dir": row["run_dir"]}
                    item.update(report_row)
                    output.append(item)
    return output


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _as_float(row: dict[str, Any], key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def collect_checkpoint_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit_rows = []
    rules = {
        "validation_mtl_score": lambda row: _as_float(row, "mtl_score"),
        "mean_validation_task_score": lambda row: (
            _as_float(row, "event_task_score", 0.0) + _as_float(row, "location_task_score", 0.0)
        )
        / 2.0,
        "pareto_balanced": lambda row: min(
            _as_float(row, "event_task_score", 0.0),
            _as_float(row, "location_task_score", 0.0),
        ),
        "old_mean_acc": lambda row: (_as_float(row, "event_acc", 0.0) + _as_float(row, "location_acc", 0.0)) / 2.0,
        "last_checkpoint": lambda row: _as_float(row, "epoch", 0.0),
    }
    for run_row in rows:
        history = _read_history(Path(str(run_row["run_dir"])) / "history" / "val_history.csv")
        if not history:
            continue
        for rule_name, selector in rules.items():
            selected = max(history, key=selector)
            audit_rows.append(
                {
                    "case": run_row["case"],
                    "seed": run_row["seed"],
                    "selection_rule": rule_name,
                    "selected_epoch": selected.get("epoch", ""),
                    "selection_value": selector(selected),
                    "event_task_score": selected.get("event_task_score", ""),
                    "location_task_score": selected.get("location_task_score", ""),
                    "mtl_score": selected.get("mtl_score", ""),
                    "event_acc": selected.get("event_acc", ""),
                    "location_acc": selected.get("location_acc", ""),
                    "run_dir": run_row["run_dir"],
                }
            )
    return audit_rows


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_runs(runs_root)
    write_csv(output_dir / "metrics_per_seed.csv", rows)
    summary_rows = summarize_rows(rows)
    write_csv(output_dir / "protocol_results_multiseed.csv", summary_rows)
    write_csv(output_dir / "per_class_metrics.csv", collect_per_class(rows))
    write_csv(output_dir / "checkpoint_selection_metrics.csv", collect_checkpoint_audit(rows))
    write_latex_table(
        output_dir / "ablation_mean_std_table.tex",
        summary_rows,
        ABLATION_CASES,
        "SensorField-M3T ablation results under the TPAMI protocol.",
        "tab:sensorfield_m3t_ablation_tpami",
    )
    write_latex_table(
        output_dir / "cross_condition_table.tex",
        summary_rows,
        CROSS_CASES,
        "Cross-condition SensorField-M3T results under condition-disjoint protocols.",
        "tab:sensorfield_m3t_cross_condition_tpami",
    )

    rendered = []
    confusion_dir = output_dir / "confusion"
    for row in rows:
        history_dir = Path(str(row["run_dir"])) / "history"
        for task_name in ("event", "location"):
            matrix_path = history_dir / f"best_test_{task_name}_confusion.csv"
            if matrix_path.is_file():
                rendered.extend(
                    render_confusion(
                        matrix_path,
                        confusion_dir / row["case"] / f"seed_{row['seed']}",
                        f"{row['case']} seed {row['seed']} {task_name}",
                    )
                )
    rendered.extend(render_view_contribution(summary_rows, output_dir / "figures"))

    manifest = {
        "runs_root": str(runs_root),
        "output_dir": str(output_dir),
        "num_runs": len(rows),
        "rendered_artifacts": rendered,
        "missing_optional_artifacts": [
            "FAC/TAEF/GCTI heatmaps require auxiliary feature extraction artifacts and are not fabricated here."
        ],
    }
    (output_dir / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote TPAMI artifacts to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
