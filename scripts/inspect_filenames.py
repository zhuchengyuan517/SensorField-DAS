from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anonymizer import sha256_text
from src.label_parser import LabelParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect PipeDAS CSV filenames.")
    parser.add_argument("--input", required=True, help="Root folder that contains batch directories.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for filename inspection results.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "label_config.yaml"),
        help="Path to the YAML label configuration.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def tokenize_name(name: str) -> list[str]:
    return [token for token in __import__("re").split(r"[\s_\-]+", name) if token]


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    input_root = Path(args.input)
    output_path = Path(args.output)
    config = load_config(Path(args.config))
    parser = LabelParser(config)

    token_counter: Counter[str] = Counter()
    event_counter: Counter[str] = Counter()
    fine_counter: Counter[str] = Counter()
    soil_counter: Counter[str] = Counter()
    distance_counter: Counter[str] = Counter()
    sampling_counter: Counter[str] = Counter()
    warning_counter: Counter[str] = Counter()
    unrecognized = []

    csv_paths = sorted(input_root.rglob("*.csv"))
    logging.info("Scanning %s CSV filenames under %s", len(csv_paths), input_root)
    for index, csv_path in enumerate(csv_paths, start=1):
        for token in tokenize_name(csv_path.stem):
            token_counter[token] += 1
        parse_result = parser.parse_path(csv_path)
        event_counter[parse_result.event_type] += 1
        fine_counter[parse_result.fine_event] += 1
        soil_counter[parse_result.soil_condition] += 1
        distance_counter[parse_result.distance_label] += 1
        sampling_label = (
            "NaN"
            if parse_result.sampling_rate_hz != parse_result.sampling_rate_hz
            else f"{parse_result.sampling_rate_hz:g}"
        )
        sampling_counter[sampling_label] += 1
        for warning in parse_result.parse_warning:
            warning_counter[warning] += 1

        if parse_result.parse_warning or parse_result.event_type == "background_noise" and "unresolved" in parse_result.matched_tokens.get("event", []):
            if len(unrecognized) < int(config["inspection"]["max_unrecognized_examples"]):
                unrecognized.append(
                    {
                        "index": index,
                        "filename_hash": sha256_text(csv_path.name),
                        "path_hash": sha256_text(str(csv_path.resolve())),
                        "source_batch_id": parse_result.source_batch_id,
                        "event_type": parse_result.event_type,
                        "fine_event": parse_result.fine_event,
                        "soil_condition": parse_result.soil_condition,
                        "distance_label": parse_result.distance_label,
                        "sampling_rate_hz": sampling_label,
                        "parse_warning": parse_result.parse_warning,
                    }
                )

    payload = {
        "summary": {
            "total_csv_files": len(csv_paths),
            "inspected_root": str(input_root),
        },
        "token_stats_top_200": dict(token_counter.most_common(200)),
        "suspected_label_stats": {
            "event_type_counts": dict(event_counter),
            "fine_event_counts": dict(fine_counter),
            "soil_condition_counts": dict(soil_counter),
            "distance_label_counts": dict(distance_counter),
            "sampling_rate_hz_counts": dict(sampling_counter),
            "parse_warning_counts": dict(warning_counter),
        },
        "unrecognized_filenames": unrecognized,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Inspection JSON written to %s", output_path)
    logging.info("Top parse warnings: %s", dict(warning_counter.most_common(10)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
