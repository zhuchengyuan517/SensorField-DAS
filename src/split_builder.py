from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import numpy as np
from sklearn.model_selection import train_test_split


def _safe_split_sizes(n_samples: int) -> tuple[int, int, int]:
    if n_samples <= 0:
        return 0, 0, 0
    if n_samples < 3:
        return n_samples, 0, 0
    test_size = max(1, int(round(n_samples * 0.1)))
    val_size = max(1, int(round(n_samples * 0.1)))
    if test_size + val_size >= n_samples:
        test_size = 1
        val_size = 1 if n_samples >= 3 else 0
    train_size = n_samples - test_size - val_size
    return train_size, val_size, test_size


def build_random_split(
    event_type_ids: Sequence[int],
    seed: int,
    warnings: List[str],
) -> Dict[str, np.ndarray]:
    n_samples = len(event_type_ids)
    indices = np.arange(n_samples, dtype=np.int64)
    train_size, val_size, test_size = _safe_split_sizes(n_samples)
    if n_samples == 0:
        return {"train": indices, "val": np.array([], dtype=np.int64), "test": np.array([], dtype=np.int64)}
    if val_size == 0 and test_size == 0:
        return {"train": indices, "val": np.array([], dtype=np.int64), "test": np.array([], dtype=np.int64)}

    labels = np.asarray(event_type_ids)
    label_counts = Counter(labels.tolist())
    stratify_ready = all(count >= 2 for count in label_counts.values())
    stratify_labels = labels if stratify_ready else None
    if not stratify_ready:
        warnings.append("random_split_fallback_no_stratify")

    try:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_labels,
        )
        remaining_labels = labels[train_val_idx]
        remaining_counts = Counter(remaining_labels.tolist())
        remaining_stratify = remaining_labels if all(count >= 2 for count in remaining_counts.values()) else None
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            random_state=seed,
            stratify=remaining_stratify,
        )
    except ValueError:
        warnings.append("random_split_train_test_split_failed")
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)
        test_idx = np.sort(shuffled[:test_size])
        val_idx = np.sort(shuffled[test_size:test_size + val_size])
        train_idx = np.sort(shuffled[test_size + val_size:])
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    return {
        "train": np.sort(train_idx.astype(np.int64)),
        "val": np.sort(val_idx.astype(np.int64)),
        "test": np.sort(test_idx.astype(np.int64)),
    }


def build_segment_holdout_split(
    segment_ids: Sequence[int],
    seed: int,
    warnings: List[str],
) -> Dict[str, np.ndarray]:
    indices_by_segment = defaultdict(list)
    for index, segment_id in enumerate(segment_ids):
        indices_by_segment[int(segment_id)].append(index)
    unique_segments = sorted(indices_by_segment.keys())
    if len(unique_segments) <= 1:
        warnings.append("segment_holdout_single_segment")
        return {
            "train": np.arange(len(segment_ids), dtype=np.int64),
            "val": np.array([], dtype=np.int64),
            "test": np.array([], dtype=np.int64),
        }

    rng = np.random.default_rng(seed)
    shuffled_segments = list(rng.permutation(unique_segments))
    total_samples = len(segment_ids)
    _, val_target, test_target = _safe_split_sizes(total_samples)

    test_segments: List[int] = []
    val_segments: List[int] = []
    train_segments: List[int] = []
    running_test = 0
    running_val = 0
    for segment_id in shuffled_segments:
        segment_count = len(indices_by_segment[segment_id])
        if running_test < test_target:
            test_segments.append(segment_id)
            running_test += segment_count
        elif running_val < val_target:
            val_segments.append(segment_id)
            running_val += segment_count
        else:
            train_segments.append(segment_id)

    if not train_segments and val_segments:
        train_segments.append(val_segments.pop())
    if not train_segments and test_segments:
        train_segments.append(test_segments.pop())

    def pack(selected_segments: List[int]) -> np.ndarray:
        packed: List[int] = []
        for segment_id in selected_segments:
            packed.extend(indices_by_segment[segment_id])
        return np.array(sorted(packed), dtype=np.int64)

    return {
        "train": pack(train_segments),
        "val": pack(val_segments),
        "test": pack(test_segments),
    }


def build_cross_segment_folds(
    segment_ids: Sequence[int],
    requested_folds: int,
    seed: int,
    warnings: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    indices_by_segment = defaultdict(list)
    for index, segment_id in enumerate(segment_ids):
        indices_by_segment[int(segment_id)].append(index)
    unique_segments = sorted(indices_by_segment.keys())
    if not unique_segments:
        return {}

    n_folds = min(requested_folds, len(unique_segments))
    if n_folds < requested_folds:
        warnings.append(f"cross_segment_fold_degraded_to_{n_folds}")
    if n_folds <= 1:
        warnings.append("cross_segment_insufficient_segments")
        return {
            "fold_0": {
                "train": np.arange(len(segment_ids), dtype=np.int64),
                "val": np.array([], dtype=np.int64),
                "test": np.array([], dtype=np.int64),
            }
        }

    rng = np.random.default_rng(seed)
    segment_buckets = np.array_split(rng.permutation(unique_segments), n_folds)

    def pack(segment_group: Sequence[int]) -> np.ndarray:
        packed: List[int] = []
        for segment_id in segment_group:
            packed.extend(indices_by_segment[int(segment_id)])
        return np.array(sorted(packed), dtype=np.int64)

    folds: Dict[str, Dict[str, np.ndarray]] = {}
    for fold_index in range(n_folds):
        test_segments = segment_buckets[fold_index].tolist()
        if n_folds >= 3:
            val_segments = segment_buckets[(fold_index + 1) % n_folds].tolist()
        else:
            val_segments = []
        train_segments: List[int] = []
        for bucket_index, bucket in enumerate(segment_buckets):
            if bucket_index in {fold_index, (fold_index + 1) % n_folds if n_folds >= 3 else -1}:
                continue
            train_segments.extend(bucket.tolist())
        if not train_segments and val_segments:
            train_segments, val_segments = val_segments, []
        folds[f"fold_{fold_index}"] = {
            "train": pack(train_segments),
            "val": pack(val_segments),
            "test": pack(test_segments),
        }
    return folds
