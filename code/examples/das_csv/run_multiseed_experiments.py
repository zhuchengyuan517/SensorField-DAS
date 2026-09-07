from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run_submission_protocol import BASE_ARGS, BOOL_FLAG_KEYS, CASES


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
WORKSPACE_ROOT = CURRENT_DIR.parents[2]
DEFAULT_ENTRY_SCRIPT = CURRENT_DIR / "train_sensorfield_m3t.py"
DEFAULT_SAVE_ROOT = WORKSPACE_ROOT / "results" / f"sensorfield_m3t_tpami_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}" / "runs"
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "converted_csv" / "MTL43"
DEFAULT_CONDITION_SPLIT_ROOT = WORKSPACE_ROOT / "converted_csv" / "sensorfield_mtl43_condition_splits_strict"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SensorField-M3T MTL43 suite over multiple seeds.")
    parser.add_argument("--python_exe", default=sys.executable, type=str)
    parser.add_argument("--entry_script", default=str(DEFAULT_ENTRY_SCRIPT), type=str)
    parser.add_argument("--save_root", default=str(DEFAULT_SAVE_ROOT), type=str)
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT), type=str)
    parser.add_argument("--condition_split_root", default=str(DEFAULT_CONDITION_SPLIT_ROOT), type=str)
    parser.add_argument("--cases", default=",".join(case["case_name"] for case in CASES), type=str)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS), type=str)
    parser.add_argument("--run", action="store_true", default=False)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--bs", default=None, type=int)
    parser.add_argument("--gpu_id", default=None, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--num_heads", default=None, type=int)
    parser.add_argument("--gaf_size", default=None, type=int)
    parser.add_argument("--stf_size", default=None, type=int)
    parser.add_argument("--spatial_adapter", default=None, type=str)
    parser.add_argument("--fac_loss_weight", default=None, type=float)
    parser.add_argument("--taef_loss_weight", default=None, type=float)
    parser.add_argument("--gcti_loss_weight", default=None, type=float)
    parser.add_argument("--location_loss_weight", default=None, type=float)
    parser.add_argument("--early_stop_patience", default=None, type=int)
    parser.add_argument(
        "--audit_epoch_cases",
        default="full_three_view",
        type=str,
        help="Comma-separated cases that should save one checkpoint per epoch for later audit.",
    )
    return parser.parse_args()


def format_cli_args(case_args: dict, save_path: str) -> list[str]:
    cli_args = []
    for key, value in case_args.items():
        flag = f"--{key}"
        if key in BOOL_FLAG_KEYS:
            if bool(value):
                cli_args.append(flag)
            continue
        cli_args.extend([flag, str(value)])
    cli_args.extend(["--save_path", str(save_path)])
    return cli_args


def run_or_print(command: list[str], run: bool) -> int:
    printable = " ".join(f'"{item}"' if " " in item else item for item in command)
    print(printable)
    if not run:
        return 0
    return int(subprocess.run(command, cwd=str(CURRENT_DIR), check=False).returncode)


def main() -> int:
    args = parse_args()
    requested_cases = {item.strip() for item in args.cases.split(",") if item.strip()}
    audit_epoch_cases = {item.strip() for item in args.audit_epoch_cases.split(",") if item.strip()}
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    cli_overrides = {
        "epochs": args.epochs,
        "bs": args.bs,
        "gpu_id": args.gpu_id,
        "lr": args.lr,
        "num_heads": args.num_heads,
        "gaf_size": args.gaf_size,
        "stf_size": args.stf_size,
        "spatial_adapter": args.spatial_adapter,
        "fac_loss_weight": args.fac_loss_weight,
        "taef_loss_weight": args.taef_loss_weight,
        "gcti_loss_weight": args.gcti_loss_weight,
        "location_loss_weight": args.location_loss_weight,
        "early_stop_patience": args.early_stop_patience,
    }

    save_root = Path(args.save_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    condition_split_root = Path(args.condition_split_root).expanduser().resolve()

    for case in CASES:
        case_name = case["case_name"]
        if case_name not in requested_cases:
            continue
        for seed in seeds:
            case_args = dict(BASE_ARGS)
            case_args.update(case["overrides"])
            for key, value in cli_overrides.items():
                if value is not None:
                    case_args[key] = value
            case_args["seed"] = seed
            case_args["save_every_epoch"] = case_name in audit_epoch_cases
            if case["dataset_subdir"] is None:
                case_args["dataset_path"] = str(dataset_root)
            else:
                case_args["dataset_path"] = str(condition_split_root / case["dataset_subdir"])
            save_path = save_root / case_name / f"seed_{seed:03d}"
            command = [args.python_exe, args.entry_script, *format_cli_args(case_args, str(save_path))]
            return_code = run_or_print(command, run=args.run)
            if return_code != 0:
                return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
