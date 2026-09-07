from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full 12-case SensorField-M3T ablation suite and summarize results.")
    parser.add_argument("--python_exe", default=sys.executable, type=str)
    parser.add_argument("--run_root", required=True, type=str)
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument("--mtl43_root", default=None, type=str)
    parser.add_argument("--image_root", default=None, type=str)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--bs", default=None, type=int)
    parser.add_argument("--gpu_id", default=None, type=int)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--weight_decay", default=None, type=float)
    parser.add_argument("--step_size", default=None, type=int)
    parser.add_argument("--gamma", default=None, type=float)
    return parser.parse_args()


def optional_cli_args(args: argparse.Namespace) -> list[str]:
    mapping = {
        "dataset_path": args.dataset_path,
        "mtl43_root": args.mtl43_root,
        "image_root": args.image_root,
        "epochs": args.epochs,
        "bs": args.bs,
        "gpu_id": args.gpu_id,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "step_size": args.step_size,
        "gamma": args.gamma,
    }
    cli_args: list[str] = []
    for key, value in mapping.items():
        if value is None:
            continue
        cli_args.extend([f"--{key}", str(value)])
    return cli_args


def run_command(command: list[str]) -> int:
    printable = " ".join(f'"{item}"' if " " in item else item for item in command)
    print(printable, flush=True)
    result = subprocess.run(command, cwd=str(CURRENT_DIR), check=False)
    return int(result.returncode)


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    common_args = optional_cli_args(args)

    commands = [
        [
            args.python_exe,
            "sensorfield_m3t_full_experiment.py",
            "--run",
            "--save_root",
            str(run_root / "full"),
            *common_args,
        ],
        [
            args.python_exe,
            "sensorfield_m3t_ablation_experiments.py",
            "--run",
            "--cases",
            "wo_fac,wo_complement,wo_taef,wo_gcti,wo_view_consistency",
            "--save_root",
            str(run_root / "ablations"),
            *common_args,
        ],
        [
            args.python_exe,
            "sensorfield_m3t_view_experiments.py",
            "--run",
            "--cases",
            "raw_only,stf_only,gaf_only,raw_stf,raw_gaf,stf_gaf",
            "--save_root",
            str(run_root / "view_combinations"),
            *common_args,
        ],
    ]

    return_code = 0
    for command in commands:
        current_code = run_command(command)
        if current_code != 0:
            return_code = current_code
            break

    summary_command = [
        args.python_exe,
        "summarize_sensorfield_m3t_ablation_results.py",
        "--run_root",
        str(run_root),
    ]
    summary_code = run_command(summary_command)
    if return_code == 0 and summary_code != 0:
        return_code = summary_code
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
