from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
WORKSPACE_ROOT = CURRENT_DIR.parents[2]
DEFAULT_ENTRY_SCRIPT = CURRENT_DIR / "train_sensorfield_m3t.py"
DEFAULT_SAVE_ROOT = WORKSPACE_ROOT / "results" / f"sensorfield_m3t_tpami_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}" / "runs"
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "converted_csv" / "MTL43"
DEFAULT_CONDITION_SPLIT_ROOT = WORKSPACE_ROOT / "converted_csv" / "sensorfield_mtl43_condition_splits_strict"


BASE_ARGS = {
    "model": "sensorfield_m3t",
    "dataset_path": str(DEFAULT_DATASET_ROOT),
    "bs": 8,
    "epochs": 80,
    "num_workers": 0,
    "gpu_id": 0,
    "seed": 42,
    "input_height": 1,
    "input_width": 10000,
    "return_multiview": True,
    "spatial_adapter": "center",
    "stf_size": 224,
    "normalize": "sample",
    "lr": 3e-5,
    "weight_decay": 5e-4,
    "step_size": 10,
    "gamma": 0.7,
    "max_grad_norm": 1.0,
    "event_loss_weight": 1.0,
    "location_loss_weight": 1.2,
    "early_stop_patience": 6,
    "early_stop_min_delta": 5e-4,
    "embed_dim": 128,
    "fusion_dim": 256,
    "time_tokens": 6,
    "freq_tokens": 48,
    "gaf_tokens": 48,
    "gaf_size": 224,
    "prior_tokens": 8,
    "num_heads": 4,
    "dropout": 0.2,
    "hidden_dim": 128,
    "num_anchors": 16,
    # Publication-facing protocol: keep the full architecture trained as a
    # genuine full model instead of disabling all auxiliary objectives.
    "fac_loss_weight": 0.05,
    "taef_loss_weight": 0.01,
    "gcti_loss_weight": 0.01,
    "view_drop_prob": 0.3,
    "enable_view_consistency": True,
    "view_consistency_weight": 0.05,
    "view_noise_std": 0.01,
    "stft_n_fft": 256,
    "stft_hop_length": 128,
    "stft_win_length": 256,
    "train_sampler": "event_distance_balanced",
    "eval_level": "file",
    "vote_method": "mean_logits",
    "selection_metric": "mtl_score",
}


BOOL_FLAG_KEYS = {
    "enable_view_consistency",
    "test_every_epoch",
    "disable_fac",
    "disable_complement",
    "disable_taef",
    "disable_gcti",
    "disable_view_consistency",
    "train_augment",
    "save_every_epoch",
    "return_multiview",
}


CASES = (
    {"case_name": "full_three_view", "dataset_subdir": None, "overrides": {"enabled_views": "raw,stf,gaf"}},
    {"case_name": "raw_only", "dataset_subdir": None, "overrides": {"enabled_views": "raw"}},
    {"case_name": "stf_only", "dataset_subdir": None, "overrides": {"enabled_views": "stf"}},
    {"case_name": "gaf_only", "dataset_subdir": None, "overrides": {"enabled_views": "gaf"}},
    {"case_name": "wo_fac", "dataset_subdir": None, "overrides": {"disable_fac": True}},
    {"case_name": "wo_taef", "dataset_subdir": None, "overrides": {"disable_taef": True}},
    {"case_name": "wo_gcti", "dataset_subdir": None, "overrides": {"disable_gcti": True}},
    {
        "case_name": "wo_all",
        "dataset_subdir": None,
        "overrides": {
            "disable_fac": True,
            "disable_complement": True,
            "disable_taef": True,
            "disable_gcti": True,
        },
    },
    {"case_name": "region_generalization", "dataset_subdir": "region_level", "overrides": {}},
    {"case_name": "soil_generalization", "dataset_subdir": "soil_level", "overrides": {}},
    {"case_name": "acquisition_generalization", "dataset_subdir": "acquisition_level", "overrides": {}},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or print the SensorField-M3T MTL43 experiment suite.")
    parser.add_argument("--python_exe", default=sys.executable, type=str)
    parser.add_argument("--entry_script", default=str(DEFAULT_ENTRY_SCRIPT), type=str)
    parser.add_argument("--save_root", default=str(DEFAULT_SAVE_ROOT), type=str)
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT), type=str)
    parser.add_argument("--condition_split_root", default=str(DEFAULT_CONDITION_SPLIT_ROOT), type=str)
    parser.add_argument("--cases", default=",".join(case["case_name"] for case in CASES), type=str)
    parser.add_argument("--run", action="store_true", default=False)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--bs", default=None, type=int)
    parser.add_argument("--gpu_id", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
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
    parser.add_argument("--save_every_epoch", action="store_true", default=False)
    return parser.parse_args()


def compose_case_args(base_args: dict, case_overrides: dict, cli_overrides: dict) -> dict:
    case_args = dict(base_args)
    case_args.update(case_overrides)
    for key, value in cli_overrides.items():
        if value is not None:
            case_args[key] = value
    return case_args


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
    cli_overrides = {
        "epochs": args.epochs,
        "bs": args.bs,
        "gpu_id": args.gpu_id,
        "seed": args.seed,
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
        "save_every_epoch": args.save_every_epoch,
    }

    save_root = Path(args.save_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    condition_split_root = Path(args.condition_split_root).expanduser().resolve()

    for case in CASES:
        if case["case_name"] not in requested_cases:
            continue
        case_args = compose_case_args(BASE_ARGS, case["overrides"], cli_overrides)
        if case["dataset_subdir"] is None:
            case_args["dataset_path"] = str(dataset_root)
        else:
            case_args["dataset_path"] = str(condition_split_root / case["dataset_subdir"])
        save_path = save_root / case["case_name"]
        command = [
            args.python_exe,
            args.entry_script,
            *format_cli_args(case_args, str(save_path)),
        ]
        return_code = run_or_print(command, run=args.run)
        if return_code != 0:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
