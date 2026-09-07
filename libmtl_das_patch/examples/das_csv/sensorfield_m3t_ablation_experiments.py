from __future__ import annotations

import argparse
from pathlib import Path

from sensorfield_m3t_experiment_presets import (
    ABLATION_PRESETS,
    DEFAULT_ABLATION_SAVE_ROOT,
    FULL_MODEL_PRESET,
    add_common_runner_args,
    build_command,
    compose_case_args,
    extract_common_cli_overrides,
    run_or_print_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or print SensorField-M3T ablation experiment commands.")
    add_common_runner_args(parser, default_save_root=DEFAULT_ABLATION_SAVE_ROOT / "ablations")
    parser.add_argument(
        "--cases",
        default="full_sensorfield_m3t,wo_fac,wo_complement,wo_taef,wo_gcti,wo_view_consistency",
        type=str,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested = {item.strip() for item in args.cases.split(",") if item.strip()}
    available_cases = [FULL_MODEL_PRESET, *ABLATION_PRESETS]
    cli_overrides = extract_common_cli_overrides(args)
    for case in available_cases:
        if case["case_name"] not in requested:
            continue
        case_args = compose_case_args(case_overrides=case["overrides"], cli_overrides=cli_overrides)
        save_path = Path(args.save_root).expanduser().resolve() / case["case_name"]
        command = build_command(
            python_exe=args.python_exe,
            entry_script=args.entry_script,
            case_args=case_args,
            save_path=str(save_path),
        )
        return_code = run_or_print_command(command, run=args.run)
        if return_code != 0:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
