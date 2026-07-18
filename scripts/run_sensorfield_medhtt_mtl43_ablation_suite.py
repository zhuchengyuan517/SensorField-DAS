from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[0]

DEFAULT_RUN_ROOT = PROJECT_ROOT / "output" / "sensorfield_medhtt_mtl43_ablations"
DEFAULT_CASES = ("full", "wo_med", "wo_htt", "wo_bti", "wo_cep")
BASE_ARGS = {
    "epochs": 20,
    "batch_size": 32,
    "input_width": 2048,
    "hidden_dim": 64,
    "num_heads": 4,
    "shared_tokens": 4,
    "raw_tokens": 4,
    "stf_tokens": 4,
    "gaf_tokens": 4,
    "stf_size": 32,
    "gaf_size": 32,
    "stft_n_fft": 64,
    "stft_hop_length": 32,
    "stft_win_length": 64,
    "radial_loss_weight": 3.0,
}
CASE_OVERRIDES = {
    "full": {},
    "wo_med": {"disable_med": True},
    "wo_htt": {"disable_htt": True},
    "wo_bti": {"disable_bti": True},
    "wo_cep": {"disable_cep": True},
}
BOOL_FLAG_KEYS = {
    "train_augment",
    "disable_med",
    "disable_htt",
    "disable_bti",
    "disable_cep",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SensorField-MEDHTT MTL43 module ablations.")
    parser.add_argument("--python-exe", default=sys.executable, type=str)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT), type=str)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), type=str)
    parser.add_argument("--dataset-path", default=None, type=str)
    parser.add_argument("--event-classes", default=None, type=str)
    parser.add_argument("--distance-classes", default=None, type=str)
    parser.add_argument("--condition-classes", default=None, type=str)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--max-train-samples", default=None, type=int)
    parser.add_argument("--max-val-samples", default=None, type=int)
    parser.add_argument("--max-test-samples", default=None, type=int)
    parser.add_argument("--input-height", default=None, type=int)
    parser.add_argument("--input-width", default=None, type=int)
    parser.add_argument("--normalize", default=None, type=str)
    parser.add_argument("--train-augment", action="store_true", default=False)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--weight-decay", default=None, type=float)
    parser.add_argument("--max-grad-norm", default=None, type=float)
    parser.add_argument("--event-loss-weight", default=None, type=float)
    parser.add_argument("--radial-loss-weight", default=None, type=float)
    parser.add_argument("--condition-loss-weight", default=None, type=float)
    parser.add_argument("--hidden-dim", default=None, type=int)
    parser.add_argument("--num-heads", default=None, type=int)
    parser.add_argument("--shared-tokens", default=None, type=int)
    parser.add_argument("--raw-tokens", default=None, type=int)
    parser.add_argument("--stf-tokens", default=None, type=int)
    parser.add_argument("--gaf-tokens", default=None, type=int)
    parser.add_argument("--stf-size", default=None, type=int)
    parser.add_argument("--gaf-size", default=None, type=int)
    parser.add_argument("--stft-n-fft", default=None, type=int)
    parser.add_argument("--stft-hop-length", default=None, type=int)
    parser.add_argument("--stft-win-length", default=None, type=int)
    parser.add_argument("--propagation-steps", default=None, type=int)
    parser.add_argument("--enabled-views", default=None, type=str)
    parser.add_argument("--dropout", default=None, type=float)
    parser.add_argument("--dec-loss-weight", default=None, type=float)
    parser.add_argument("--cep-loss-weight", default=None, type=float)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--print-only", action="store_true", default=False)
    return parser.parse_args()


def cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    keys = (
        "dataset_path",
        "event_classes",
        "distance_classes",
        "condition_classes",
        "epochs",
        "batch_size",
        "num_workers",
        "seed",
        "max_train_samples",
        "max_val_samples",
        "max_test_samples",
        "input_height",
        "input_width",
        "normalize",
        "lr",
        "weight_decay",
        "max_grad_norm",
        "event_loss_weight",
        "radial_loss_weight",
        "condition_loss_weight",
        "hidden_dim",
        "num_heads",
        "shared_tokens",
        "raw_tokens",
        "stf_tokens",
        "gaf_tokens",
        "stf_size",
        "gaf_size",
        "stft_n_fft",
        "stft_hop_length",
        "stft_win_length",
        "propagation_steps",
        "enabled_views",
        "dropout",
        "dec_loss_weight",
        "cep_loss_weight",
        "device",
    )
    overrides = {key: getattr(args, key) for key in keys if getattr(args, key) is not None}
    if args.train_augment:
        overrides["train_augment"] = True
    return overrides


def build_case_args(case_name: str, overrides: dict[str, object]) -> dict[str, object]:
    args = dict(BASE_ARGS)
    args.update(CASE_OVERRIDES[case_name])
    args.update(overrides)
    return args


def build_command(python_exe: str, output_dir: Path, case_args: dict[str, object]) -> list[str]:
    command = [
        python_exe,
        str(CURRENT_DIR / "run_sensorfield_medhtt_mtl43_experiment.py"),
        "--output-dir",
        str(output_dir),
    ]
    for key, value in case_args.items():
        flag = f"--{key.replace('_', '-')}"
        if key in BOOL_FLAG_KEYS:
            if bool(value):
                command.append(flag)
            continue
        command.extend([flag, str(value)])
    return command


def run_command(command: list[str], print_only: bool) -> int:
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(printable, flush=True)
    if print_only:
        return 0
    result = subprocess.run(command, cwd=str(CURRENT_DIR), check=False)
    return int(result.returncode)


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    requested_cases = [item.strip() for item in args.cases.split(",") if item.strip()]
    overrides = cli_overrides(args)

    for case_name in requested_cases:
        if case_name not in CASE_OVERRIDES:
            raise KeyError(f"Unsupported ablation case: {case_name}")
        case_args = build_case_args(case_name, overrides)
        output_dir = run_root / case_name
        command = build_command(args.python_exe, output_dir, case_args)
        return_code = run_command(command, print_only=args.print_only)
        if return_code != 0:
            return return_code

    summary_command = [
        args.python_exe,
        str(CURRENT_DIR / "summarize_sensorfield_medhtt_mtl43_ablation.py"),
        "--run-root",
        str(run_root),
        "--cases",
        ",".join(requested_cases),
    ]
    return run_command(summary_command, print_only=args.print_only)


if __name__ == "__main__":
    raise SystemExit(main())
