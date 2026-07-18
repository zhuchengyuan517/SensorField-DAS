from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_CASES = ("full", "wo_med", "wo_htt", "wo_bti", "wo_cep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SensorField-MEDHTT MTL43 ablation results.")
    parser.add_argument("--run-root", required=True, type=str)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), type=str)
    return parser.parse_args()


def latest_run_dir(case_root: Path) -> Path | None:
    if not case_root.is_dir():
        return None
    candidates = [path for path in case_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.name)


def load_case_result(case_name: str, run_root: Path) -> dict[str, Any] | None:
    case_root = run_root / case_name
    run_dir = latest_run_dir(case_root)
    if run_dir is None:
        return None
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "run_config.json"
    if not summary_path.is_file() or not config_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    test_metrics = summary["test_metrics"]
    return {
        "case_name": case_name,
        "run_dir": str(run_dir),
        "best_epoch": summary["best_epoch"],
        "best_val_score": summary["best_val_score"],
        "test_score": test_metrics["score"],
        "test_loss": test_metrics["loss"],
        "event_acc": test_metrics["event"]["acc"],
        "event_macro_f1": test_metrics["event"]["macro_f1"],
        "radial_acc": test_metrics["radial"]["acc"],
        "radial_macro_f1": test_metrics["radial"]["macro_f1"],
        "condition_acc": test_metrics["condition"]["acc"],
        "condition_macro_f1": test_metrics["condition"]["macro_f1"],
        "epochs": config["args"]["epochs"],
        "batch_size": config["args"]["batch_size"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_name",
        "best_epoch",
        "best_val_score",
        "test_score",
        "test_loss",
        "event_acc",
        "event_macro_f1",
        "radial_acc",
        "radial_macro_f1",
        "condition_acc",
        "condition_macro_f1",
        "epochs",
        "batch_size",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Case | Val Score | Test Score | Event Acc | Radial Acc | Condition Acc | Run Dir |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case_name} | {best_val_score:.4f} | {test_score:.4f} | {event_acc:.4f} | {radial_acc:.4f} | {condition_acc:.4f} | {run_dir} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    cases = [item.strip() for item in args.cases.split(",") if item.strip()]
    rows = []
    for case_name in cases:
        result = load_case_result(case_name, run_root)
        if result is not None:
            rows.append(result)
    rows.sort(key=lambda row: row["test_score"], reverse=True)

    (run_root / "comparison_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(run_root / "comparison_summary.csv", rows)
    write_markdown(run_root / "comparison_summary.md", rows)
    print(f"Saved ablation summary to: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
