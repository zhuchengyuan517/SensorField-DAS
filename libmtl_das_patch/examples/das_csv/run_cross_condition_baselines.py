from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parents[2]
DEFAULT_ENTRY = CURRENT_DIR / "pipemmtl_main.py"
DEFAULT_SPLIT_ROOT = WORKSPACE_ROOT / "converted_csv" / "sensorfield_mtl43_condition_splits_strict"
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "results"
    / f"sensorfield_m3t_cross_condition_baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

MODEL_NAMES = (
    "convnext_small",
    "multimodn",
    "m4oe",
    "das_mae",
    "pipelineadwint",
    "aligned_mtl",
    "moco_mtl",
    "sensorfield_m3t",
)
PROTOCOLS = ("region", "soil", "acquisition")
MODEL_LR = {
    "convnext_small": 1e-4,
    "pipelineadwint": 1e-4,
    "multimodn": 3e-4,
    "m4oe": 3e-4,
    "das_mae": 1e-4,
    "aligned_mtl": 3e-4,
    "moco_mtl": 3e-4,
    "sensorfield_m3t": 3e-5,
}
MODEL_BATCH_SIZE = {"convnext_small": 8, "pipelineadwint": 8}


def parse_csv_values(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run five-seed cross-condition baselines on shared MTL43 splits.")
    parser.add_argument("--python_exe", default=sys.executable, type=str)
    parser.add_argument("--entry_script", default=str(DEFAULT_ENTRY), type=str)
    parser.add_argument("--split_root", default=str(DEFAULT_SPLIT_ROOT), type=str)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT), type=str)
    parser.add_argument("--models", default=",".join(MODEL_NAMES), type=str)
    parser.add_argument("--protocols", default=",".join(PROTOCOLS), type=str)
    parser.add_argument("--seeds", default="42,43,44,45,46", type=str)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--bs", default=16, type=int)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--early_stop_patience", default=6, type=int)
    parser.add_argument("--max_train_batches", default=0, type=int)
    parser.add_argument("--max_eval_batches", default=0, type=int)
    parser.add_argument("--baseline_pretrained", action="store_true", default=False)
    parser.add_argument("--include_completed", action="store_true", default=False)
    return parser.parse_args()


def completed_summary(run_root: Path) -> Path | None:
    summaries = sorted(run_root.glob("*/summary.json"))
    return summaries[-1] if summaries else None


def task_balance_for(model_name: str) -> str:
    if model_name == "aligned_mtl":
        return "aligned_mtl"
    if model_name == "moco_mtl":
        return "moco"
    return "equal"


def build_command(args: argparse.Namespace, model_name: str, protocol: str, seed: int, save_root: Path) -> list[str]:
    dataset_path = Path(args.split_root) / f"{protocol}_level"
    batch_size = MODEL_BATCH_SIZE.get(model_name, args.bs)
    command = [
        args.python_exe,
        args.entry_script,
        "--model",
        model_name,
        "--dataset_path",
        str(dataset_path),
        "--save_path",
        str(save_root),
        "--epochs",
        str(args.epochs),
        "--bs",
        str(batch_size),
        "--num_workers",
        str(args.num_workers),
        "--gpu_id",
        str(args.gpu_id),
        "--seed",
        str(seed),
        "--lr",
        str(MODEL_LR[model_name]),
        "--weight_decay",
        "0.0005",
        "--step_size",
        "10",
        "--gamma",
        "0.7",
        "--input_height",
        "1",
        "--input_width",
        "10000",
        "--spatial_adapter",
        "center",
        "--stf_spatial_fusion",
        "group3_mean",
        "--stf_size",
        "224",
        "--gaf_size",
        "224",
        "--normalize",
        "sample",
        "--hidden_dim",
        "128",
        "--num_anchors",
        "16",
        "--num_heads",
        "4",
        "--event_loss_weight",
        "1.0",
        "--location_loss_weight",
        "1.2",
        "--train_sampler",
        "event_distance_balanced",
        "--eval_level",
        "file",
        "--vote_method",
        "mean_logits",
        "--selection_metric",
        "mtl_score",
        "--early_stop_patience",
        str(args.early_stop_patience),
        "--task_balance",
        task_balance_for(model_name),
        "--max_train_batches",
        str(args.max_train_batches),
        "--max_eval_batches",
        str(args.max_eval_batches),
    ]
    if model_name == "sensorfield_m3t":
        command.extend(
            [
                "--fac_loss_weight",
                "0.05",
                "--taef_loss_weight",
                "0.01",
                "--gcti_loss_weight",
                "0.01",
                "--view_drop_prob",
                "0.3",
                "--enable_view_consistency",
                "--view_consistency_weight",
                "0.05",
                "--view_noise_std",
                "0.01",
                "--enabled_views",
                "raw,stf,gaf",
            ]
        )
    if args.baseline_pretrained and model_name != "sensorfield_m3t":
        command.append("--baseline_pretrained")
    return command


def write_status(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "protocol", "seed", "status", "return_code", "summary_path", "log_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    models = parse_csv_values(args.models)
    protocols = parse_csv_values(args.protocols)
    seeds = parse_csv_values(args.seeds, int)
    unknown_models = sorted(set(models) - set(MODEL_NAMES))
    unknown_protocols = sorted(set(protocols) - set(PROTOCOLS))
    if unknown_models or unknown_protocols:
        raise ValueError(f"Unknown models={unknown_models}, protocols={unknown_protocols}")

    split_root = Path(args.split_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "runner_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    status_rows: list[dict] = []
    status_path = output_root / "run_status.csv"

    for protocol in protocols:
        if not (split_root / f"{protocol}_level" / "train.csv").is_file():
            raise FileNotFoundError(f"Missing split: {split_root / f'{protocol}_level'}")
        for model_name in models:
            for seed in seeds:
                run_root = output_root / "runs" / protocol / model_name / f"seed_{seed:03d}"
                existing = completed_summary(run_root)
                log_path = output_root / "logs" / protocol / model_name / f"seed_{seed:03d}.log"
                if existing is not None and not args.include_completed:
                    status_rows.append(
                        {
                            "model": model_name,
                            "protocol": protocol,
                            "seed": seed,
                            "status": "skipped_completed",
                            "return_code": 0,
                            "summary_path": str(existing),
                            "log_path": str(log_path),
                        }
                    )
                    write_status(status_path, status_rows)
                    continue

                run_root.mkdir(parents=True, exist_ok=True)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                command = build_command(args, model_name, protocol, seed, run_root)
                print(f"Running {protocol}/{model_name}/seed_{seed:03d}", flush=True)
                with log_path.open("w", encoding="utf-8") as log_handle:
                    process = subprocess.run(
                        command,
                        cwd=str(CURRENT_DIR),
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                summary = completed_summary(run_root)
                status = "completed" if process.returncode == 0 and summary is not None else "failed"
                status_rows.append(
                    {
                        "model": model_name,
                        "protocol": protocol,
                        "seed": seed,
                        "status": status,
                        "return_code": process.returncode,
                        "summary_path": str(summary or ""),
                        "log_path": str(log_path),
                    }
                )
                write_status(status_path, status_rows)
                if status == "failed":
                    print(f"Failed {protocol}/{model_name}/seed_{seed:03d}; see {log_path}", flush=True)

    failed = sum(row["status"] == "failed" for row in status_rows)
    print(f"Finished {len(status_rows)} runs with {failed} failures. Output: {output_root}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
