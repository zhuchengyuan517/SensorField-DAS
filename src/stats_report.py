from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


def _counter_to_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def build_statistics_payload(
    sample_summaries: Sequence[Mapping[str, object]],
    warning_counter: Mapping[str, int],
) -> Dict[str, object]:
    event_counter = Counter(summary["event_type"] for summary in sample_summaries)
    soil_counter = Counter(summary["soil_condition"] for summary in sample_summaries)
    fine_counter = Counter(summary["fine_event"] for summary in sample_summaries)
    distance_counter = Counter(summary["distance_label"] for summary in sample_summaries)
    segment_counter = Counter(summary["segment_public_id"] for summary in sample_summaries)
    sampling_counter = Counter(summary["sampling_rate_label"] for summary in sample_summaries)
    invalid_count = sum(1 for summary in sample_summaries if not summary["is_valid"])

    return {
        "total_samples": len(sample_summaries),
        "event_type_counts": _counter_to_dict(event_counter),
        "soil_condition_counts": _counter_to_dict(soil_counter),
        "fine_event_counts": _counter_to_dict(fine_counter),
        "distance_label_counts": _counter_to_dict(distance_counter),
        "segment_id_counts": _counter_to_dict(segment_counter),
        "sampling_rate_hz_counts": _counter_to_dict(sampling_counter),
        "invalid_samples": invalid_count,
        "parsing_warning_counts": {key: int(value) for key, value in sorted(warning_counter.items())},
    }


def write_dataset_statistics(output_path: Path, payload: Mapping[str, object]) -> None:
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_dataset_card(output_path: Path, context: Mapping[str, object]) -> None:
    lines: List[str] = [
        "# PipeDAS-Multi Dataset Card",
        "",
        "## Dataset Overview",
        "",
        f"- Dataset name: `{context['dataset_name']}`",
        f"- Version: `{context['version']}`",
        f"- Total public samples: `{context['total_samples']}`",
        f"- Source batches: `{', '.join(context['source_batches'])}`",
        "",
        "## Data Source and Anonymization",
        "",
        "This release packages anonymized DAS signal samples gathered from multiple acquisition batches, sampling configurations, and soil conditions.",
        "Each public sample is a cropped defense-zone window rather than a full raw fence matrix.",
        "Positive event windows are selected from the strongest-response zone band of each eligible source file, and additional background samples are generated from other non-event zone bands in the same source file.",
        "The public HDF5 file excludes original paths, raw filenames, real dates, raw defense-zone identifiers, station names, geographic markers, and pipe diameter details.",
        "Batch identifiers are remapped to anonymous batch ids, and segment ids are derived from anonymous batch ids plus sampling-rate and soil-condition domains.",
        "",
        "## Label Taxonomy",
        "",
        "### Coarse Event Types",
        "",
        "- `background_noise`",
        "- `pipeline_leakage`",
        "- `mechanical_excavation`",
        "- `manual_work`",
        "- `vehicle_passing`",
        "",
        "### Fine Events",
        "",
        "- Fine-event ids are stored through `/meta/label_maps/fine_event_json`.",
        "- Public fine-event labels may include `N/A`, `unknown`, `excavator_idle`, `knocking`, `digging`, `parallel_driving`, `crossing`, `vehicle_passing`, `manual_digging`, `manual_walking`, and `vehicle_idle` when those patterns are observed in filenames.",
        "",
        "## HDF5 Structure",
        "",
        "The release HDF5 file is organized into the groups `/data`, `/labels`, `/meta`, `/quality`, and `/splits`.",
        "`/data/signals_flat` stores flattened float32 signals, while `/data/signal_index` and `/data/signal_shape` reconstruct each original `[T, C]` sample.",
        "String label maps are stored under `/meta/label_maps/*_json`.",
        "",
        "## Recommended Tasks",
        "",
        "- Coarse event classification",
        "- Fine-event recognition",
        "- Distance-aware recognition for event samples with explicit range labels",
        "- Cross-domain generalization across anonymous segments",
        "",
        "## Recommended Splits",
        "",
        "- `random`: general benchmarking with approximate 8:1:1 sample split",
        "- `segment_holdout`: generalization to unseen anonymous segments",
        "- `cross_segment/fold_*`: cross-segment cross-validation folds",
        "",
        "## Loading Example",
        "",
        "See `scripts/dataset_loader_example.py` for a minimal reader that reconstructs per-sample signals from `signals_flat` and `signal_index`.",
        "",
        "## Known Limitations",
        "",
        "- Labels are inferred from filenames and directory context; unresolved patterns are preserved through parse warnings.",
        "- Sampling rates are not resampled. Samples from different acquisition settings may therefore have different temporal lengths and channel counts.",
        "- Mixed-soil filename patterns are collapsed to `unknown` in the public release.",
        "",
        "## Citation Placeholder",
        "",
        "```text",
        "@dataset{PipeDAS_Multi_v1,",
        "  title  = {PipeDAS-Multi: Anonymized DAS Infrastructure Safety Monitoring Dataset},",
        "  author = {To Be Added},",
        "  year   = {2026},",
        "  note   = {Public HDF5 release v1.0}",
        "}",
        "```",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_build_report(output_path: Path, context: Mapping[str, object]) -> None:
    lines: List[str] = [
        "# Build Report",
        "",
        "## Build Summary",
        "",
        f"- Output HDF5: `{context['output_h5']}`",
        f"- Total input CSV files scanned: `{context['total_files_scanned']}`",
        f"- Samples written: `{context['samples_written']}`",
        f"- Positive event crops: `{context.get('positive_samples', 0)}`",
        f"- Generated background crops: `{context.get('generated_background_samples', 0)}`",
        f"- Files skipped: `{context['files_skipped']}`",
        f"- Invalid samples retained with `is_valid = False`: `{context['invalid_samples']}`",
        "",
        "## Parsing Warnings",
        "",
    ]
    if context["warning_counter"]:
        for warning_name, count in sorted(context["warning_counter"].items()):
            lines.append(f"- `{warning_name}`: `{count}`")
    else:
        lines.append("- No parse warnings were recorded.")

    lines.extend(
        [
            "",
            "## Split Notes",
            "",
        ]
    )
    if context["split_warnings"]:
        for warning in context["split_warnings"]:
            lines.append(f"- `{warning}`")
    else:
        lines.append("- No split degradation warnings were recorded.")

    lines.extend(
        [
            "",
            "## Failed or Skipped Inputs",
            "",
        ]
    )
    if context["failed_examples"]:
        for example in context["failed_examples"]:
            lines.append(
                f"- `path_hash={example['path_hash']}` `status={example['status']}` `warning={example['warning']}`"
            )
    else:
        lines.append("- No inputs were skipped.")

    lines.extend(
        [
            "",
            "## HDF5 Summary",
            "",
        ]
    )
    for key, value in context["hdf5_summary"].items():
        lines.append(f"- `{key}`: `{value}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
