from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SPLIT_FILENAMES = ("train.csv", "val.csv", "test.csv")
SOIL_TOKENS = ("land", "sand", "shizi")
BEHAVIOR_HINTS = (
    "行走",
    "行驶",
    "施工",
    "切削",
    "怠速",
    "开挖",
    "挖掘",
    "正常",
    "汽车",
)
TRAILING_ID_PATTERN = re.compile(r"-\d+(?:-\d+)?$")
TRAILING_BATCH_PATTERN = re.compile(r"-(\d+)(?:-\d+)?$")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_manifest_rows(dataset_root: Path) -> list[dict]:
    merged = []
    seen = set()
    for filename in SPLIT_FILENAMES:
        manifest_path = dataset_root / filename
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Required manifest was not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row["path"],
                    row["event_label"],
                    row.get("distance_label", ""),
                    row.get("sample_mode", "file"),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(
                    {
                        "path": row["path"],
                        "event_label": row["event_label"],
                        "distance_label": row.get("distance_label", ""),
                        "sample_mode": row.get("sample_mode", "file"),
                        "origin_split": manifest_path.stem,
                    }
                )
    if not merged:
        raise ValueError(f"No rows were loaded from {dataset_root}")
    return merged


def _strip_trailing_ids(stem: str) -> str:
    return TRAILING_ID_PATTERN.sub("", stem)


def _extract_soil_condition(path_text: str) -> str:
    tags = [token for token in SOIL_TOKENS if token in path_text]
    if not tags:
        return "unknown"
    if len(tags) == 1:
        return tags[0]
    return "mixed"


def _is_behavior_token(token: str) -> bool:
    return any(hint in token for hint in BEHAVIOR_HINTS)


def _extract_region_descriptor(path_text: str) -> str:
    stem = _strip_trailing_ids(Path(path_text).stem)
    tokens = [token for token in stem.split("-") if token]
    region_tokens = []
    for token in tokens:
        if _is_behavior_token(token):
            break
        region_tokens.append(token)
    return "-".join(region_tokens) if region_tokens else stem


def _extract_batch_id(path_text: str) -> str:
    stem = Path(path_text).stem
    match = TRAILING_BATCH_PATTERN.search(stem)
    if match:
        return match.group(1)
    return stem


def attach_metadata(rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        path_text = str(row["path"])
        enriched_row = dict(row)
        enriched_row["soil_condition"] = _extract_soil_condition(path_text)
        enriched_row["region_condition"] = _extract_region_descriptor(path_text)
        enriched_row["acquisition_condition"] = _extract_batch_id(path_text)
        enriched_row["joint_label"] = f"{row['event_label']}||{row.get('distance_label', '') or '_'}"
        enriched.append(enriched_row)
    return enriched


def group_rows(rows: list[dict], group_key: str) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    return dict(grouped)


def _build_label_counter(rows: list[dict]) -> Counter:
    counter = Counter()
    for row in rows:
        counter[row["joint_label"]] += 1
    return counter


def _score_assignment(
    split_name: str,
    group_counter: Counter,
    group_size: int,
    split_counters: dict[str, Counter],
    split_sizes: dict[str, int],
    global_counter: Counter,
    total_size: int,
    targets: dict[str, float],
) -> float:
    next_size = split_sizes[split_name] + group_size
    target_size = total_size * targets[split_name]
    size_penalty = ((next_size - target_size) / max(target_size, 1.0)) ** 2

    candidate_counter = split_counters[split_name] + group_counter
    label_penalty = 0.0
    denom = max(next_size, 1)
    for label_name, total_count in global_counter.items():
        target_ratio = total_count / max(total_size, 1)
        candidate_ratio = candidate_counter[label_name] / denom
        label_penalty += abs(candidate_ratio - target_ratio)
    return float(size_penalty + 0.2 * label_penalty)


def greedy_group_split(
    rows: list[dict],
    *,
    group_key: str,
    targets: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, list[dict]]:
    targets = targets or {"train": 0.7, "val": 0.1, "test": 0.2}
    grouped = group_rows(rows, group_key)
    global_counter = _build_label_counter(rows)
    total_size = len(rows)

    group_items = list(grouped.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    split_groups = {"train": [], "val": [], "test": []}
    split_rows = {"train": [], "val": [], "test": []}
    split_counters = {name: Counter() for name in split_rows}
    split_sizes = {name: 0 for name in split_rows}

    bootstrap_index = 0
    if len(group_items) >= 3:
        for split_name in ("train", "val", "test"):
            group_name, group_rows_list = group_items[bootstrap_index]
            group_counter = _build_label_counter(group_rows_list)
            split_groups[split_name].append(group_name)
            split_rows[split_name].extend(group_rows_list)
            split_counters[split_name].update(group_counter)
            split_sizes[split_name] += len(group_rows_list)
            bootstrap_index += 1

    for group_name, group_rows_list in group_items[bootstrap_index:]:
        group_counter = _build_label_counter(group_rows_list)
        group_size = len(group_rows_list)
        best_split = min(
            split_rows.keys(),
            key=lambda split_name: _score_assignment(
                split_name,
                group_counter,
                group_size,
                split_counters,
                split_sizes,
                global_counter,
                total_size,
                targets,
            ),
        )
        split_groups[best_split].append(group_name)
        split_rows[best_split].extend(group_rows_list)
        split_counters[best_split].update(group_counter)
        split_sizes[best_split] += group_size

    train_labels = set(split_counters["train"])
    required_labels = set(global_counter)
    missing_labels = sorted(required_labels - train_labels)
    for label_name in missing_labels:
        donor_split = None
        donor_group = None
        for split_name in ("val", "test"):
            for group_name in split_groups[split_name]:
                group_counter = _build_label_counter(grouped[group_name])
                if group_counter[label_name] > 0:
                    donor_split = split_name
                    donor_group = group_name
                    break
            if donor_group is not None:
                break
        if donor_group is None:
            continue
        split_groups[donor_split].remove(donor_group)
        split_groups["train"].append(donor_group)

    final_rows = {name: [] for name in split_rows}
    for split_name, group_names in split_groups.items():
        for group_name in group_names:
            final_rows[split_name].extend(grouped[group_name])
    return final_rows


def labelwise_group_split(rows: list[dict], *, group_key: str, seed: int = 42) -> dict[str, list[dict]]:
    per_label_rows = defaultdict(list)
    for row in rows:
        per_label_rows[row["joint_label"]].append(row)

    merged = {"train": [], "val": [], "test": []}
    for label_index, label_rows in enumerate(per_label_rows.values()):
        label_split = greedy_group_split(
            label_rows,
            group_key=group_key,
            seed=seed + label_index,
        )
        for split_name, split_rows in label_split.items():
            merged[split_name].extend(split_rows)
    return merged


def binary_group_split(
    rows: list[dict],
    *,
    group_key: str,
    targets: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, list[dict]]:
    targets = targets or {"train": 0.8, "test": 0.2}
    grouped = group_rows(rows, group_key)
    global_counter = _build_label_counter(rows)
    total_size = len(rows)

    group_items = list(grouped.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    split_groups = {"train": [], "test": []}
    split_rows = {"train": [], "test": []}
    split_counters = {name: Counter() for name in split_rows}
    split_sizes = {name: 0 for name in split_rows}

    bootstrap_index = 0
    if len(group_items) >= 2:
        for split_name in ("train", "test"):
            group_name, group_rows_list = group_items[bootstrap_index]
            group_counter = _build_label_counter(group_rows_list)
            split_groups[split_name].append(group_name)
            split_rows[split_name].extend(group_rows_list)
            split_counters[split_name].update(group_counter)
            split_sizes[split_name] += len(group_rows_list)
            bootstrap_index += 1

    for group_name, group_rows_list in group_items[bootstrap_index:]:
        group_counter = _build_label_counter(group_rows_list)
        group_size = len(group_rows_list)
        best_split = min(
            split_rows.keys(),
            key=lambda split_name: _score_assignment(
                split_name,
                group_counter,
                group_size,
                split_counters,
                split_sizes,
                global_counter,
                total_size,
                targets,
            ),
        )
        split_groups[best_split].append(group_name)
        split_rows[best_split].extend(group_rows_list)
        split_counters[best_split].update(group_counter)
        split_sizes[best_split] += group_size
    return split_rows


def binary_labelwise_group_split(rows: list[dict], *, group_key: str, seed: int = 42) -> dict[str, list[dict]]:
    per_label_rows = defaultdict(list)
    for row in rows:
        per_label_rows[row["joint_label"]].append(row)

    merged = {"train": [], "test": []}
    for label_index, label_rows in enumerate(per_label_rows.values()):
        label_split = binary_group_split(
            label_rows,
            group_key=group_key,
            seed=seed + label_index,
        )
        for split_name, split_rows in label_split.items():
            merged[split_name].extend(split_rows)
    return merged


def soil_protocol_split(rows: list[dict]) -> dict[str, list[dict]]:
    partitions = {
        "train": {"land", "mixed", "unknown", "shizi"},
        "test": {"sand"},
    }
    split_rows = {name: [] for name in partitions}
    for row in rows:
        assigned = False
        for split_name, conditions in partitions.items():
            if row["soil_condition"] in conditions:
                split_rows[split_name].append(row)
                assigned = True
                break
        if not assigned:
            split_rows["train"].append(row)
    return split_rows


def strict_soil_protocol_split(rows: list[dict]) -> dict[str, list[dict]]:
    """Use mutually exclusive soil domains while retaining all train task labels."""
    partitions = {
        "train": {"land", "mixed", "unknown"},
        "val": {"shizi"},
        "test": {"sand"},
    }
    split_rows = {name: [] for name in partitions}
    for row in rows:
        for split_name, conditions in partitions.items():
            if row["soil_condition"] in conditions:
                split_rows[split_name].append(row)
                break
    return split_rows


def _split_quality(indices: np.ndarray, labels: np.ndarray, target_fraction: float) -> float:
    expected_size = len(labels) * target_fraction
    size_penalty = abs(len(indices) - expected_size) / max(expected_size, 1.0)
    present = set(labels[indices].tolist())
    missing_penalty = len(set(labels.tolist()) - present) * 10.0
    return float(size_penalty + missing_penalty)


def strict_stratified_group_split(
    rows: list[dict], *, group_key: str, seed: int
) -> dict[str, list[dict]]:
    """Stratify joint task labels while keeping every condition in one split."""
    labels = np.asarray([row["joint_label"] for row in rows], dtype=object)
    groups = np.asarray([row[group_key] for row in rows], dtype=object)
    indices = np.arange(len(rows))

    outer_candidates = []
    for n_splits in range(3, 7):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        outer_candidates.extend(splitter.split(indices, labels, groups))
    required_labels = set(labels.tolist())
    candidates = []
    for train_val_indices, test_indices in outer_candidates:
        inner_labels = labels[train_val_indices]
        inner_groups = groups[train_val_indices]
        for n_splits in range(2, 7):
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1)
            for inner_train, inner_val in splitter.split(train_val_indices, inner_labels, inner_groups):
                train_indices = train_val_indices[inner_train]
                val_indices = train_val_indices[inner_val]
                missing_count = sum(
                    len(required_labels - set(labels[split_indices].tolist()))
                    for split_indices in (train_indices, val_indices, test_indices)
                )
                size_penalty = (
                    abs(len(train_indices) / len(rows) - 0.6)
                    + abs(len(val_indices) / len(rows) - 0.2)
                    + abs(len(test_indices) / len(rows) - 0.2)
                )
                candidates.append(
                    (missing_count * 100.0 + size_penalty, train_indices, val_indices, test_indices)
                )
    _, train_indices, val_indices, test_indices = min(candidates, key=lambda item: item[0])

    return {
        "train": [rows[index] for index in train_indices],
        "val": [rows[index] for index in val_indices],
        "test": [rows[index] for index in test_indices],
    }


def condition_overlap_audit(split_rows: dict[str, list[dict]], condition_key: str) -> dict:
    condition_sets = {
        split_name: {str(row[condition_key]) for row in rows}
        for split_name, rows in split_rows.items()
    }
    pairwise = {}
    names = ("train", "val", "test")
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sorted(condition_sets.get(left, set()) & condition_sets.get(right, set()))
            pairwise[f"{left}_{right}"] = {"count": len(overlap), "conditions": overlap}
    return {
        "passed": all(item["count"] == 0 for item in pairwise.values()),
        "pairwise": pairwise,
    }


def summarize_split(split_rows: dict[str, list[dict]], condition_key: str) -> dict:
    summary = {"condition_key": condition_key, "splits": {}}
    for split_name, rows in split_rows.items():
        event_counter = Counter(row["event_label"] for row in rows)
        distance_counter = Counter((row.get("distance_label", "") or "_") for row in rows)
        condition_counter = Counter(row[condition_key] for row in rows)
        summary["splits"][split_name] = {
            "num_rows": len(rows),
            "event_counts": dict(event_counter),
            "distance_counts": dict(distance_counter),
            "condition_counts": dict(condition_counter),
        }
    return summary


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "event_label", "distance_label", "sample_mode"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row["path"],
                    "event_label": row["event_label"],
                    "distance_label": row.get("distance_label", ""),
                    "sample_mode": row.get("sample_mode", "file"),
                }
            )


def build_protocol_splits(
    rows: list[dict], protocol_name: str, seed: int, strict_validation: bool = False
) -> dict[str, list[dict]]:
    if strict_validation:
        if protocol_name == "soil":
            return strict_soil_protocol_split(rows)
        group_key = f"{protocol_name}_condition"
        return strict_stratified_group_split(rows, group_key=group_key, seed=seed)

    original_val_rows = [row for row in rows if row.get("origin_split") == "val"]
    candidate_rows = [row for row in rows if row.get("origin_split") != "val"]
    if protocol_name == "region":
        split_rows = binary_labelwise_group_split(candidate_rows, group_key="region_condition", seed=seed)
        split_rows["val"] = list(original_val_rows)
        return split_rows
    if protocol_name == "soil":
        split_rows = soil_protocol_split(candidate_rows)
        split_rows["val"] = list(original_val_rows)
        return split_rows
    if protocol_name == "acquisition":
        split_rows = binary_labelwise_group_split(candidate_rows, group_key="acquisition_condition", seed=seed)
        split_rows["val"] = list(original_val_rows)
        return split_rows
    raise KeyError(f"Unsupported protocol: {protocol_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build condition-disjoint SensorField-DAS manifests for MTL43.")
    parser.add_argument(
        "--dataset_root",
        default=str(WORKSPACE_ROOT / "converted_csv" / "MTL43"),
        type=str,
    )
    parser.add_argument(
        "--output_root",
        default=str(WORKSPACE_ROOT / "converted_csv" / "sensorfield_mtl43_condition_splits_strict"),
        type=str,
    )
    parser.add_argument("--protocols", default="region,soil,acquisition", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--strict_validation",
        action="store_true",
        default=False,
        help="Build mutually condition-disjoint train/validation/test partitions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    protocols = [item.strip() for item in args.protocols.split(",") if item.strip()]

    rows = attach_metadata(load_manifest_rows(dataset_root))
    output_root.mkdir(parents=True, exist_ok=True)

    protocol_summaries = {}
    for protocol_name in protocols:
        split_rows = build_protocol_splits(
            rows,
            protocol_name=protocol_name,
            seed=args.seed,
            strict_validation=args.strict_validation,
        )
        protocol_dir = output_root / f"{protocol_name}_level"
        protocol_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_manifest_rows in split_rows.items():
            write_manifest(protocol_dir / f"{split_name}.csv", split_manifest_rows)
        summary = summarize_split(split_rows, f"{protocol_name}_condition")
        summary["condition_overlap_audit"] = condition_overlap_audit(
            split_rows, f"{protocol_name}_condition"
        )
        summary["strict_validation"] = bool(args.strict_validation)
        write_json(protocol_dir / "summary.json", summary)
        protocol_summaries[protocol_name] = summary

    write_json(output_root / "protocol_summaries.json", protocol_summaries)
    print(f"Saved condition-disjoint manifests to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
