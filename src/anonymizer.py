from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sampling_rate_key(sampling_rate_hz: float) -> str:
    if sampling_rate_hz is None or math.isnan(sampling_rate_hz):
        return "SR_UNKNOWN"
    if float(sampling_rate_hz).is_integer():
        return f"SR_{int(sampling_rate_hz)}"
    return f"SR_{sampling_rate_hz:.3f}".replace(".", "p")


def strip_sensitive_text(value: str, diameter_patterns: Iterable[str]) -> str:
    cleaned = value
    for pattern in diameter_patterns:
        cleaned = re.sub(pattern, "", cleaned)
    cleaned = re.sub(r"0715|0716", "", cleaned)
    cleaned = re.sub(r"[\\/]+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned.strip("-_ ")


@dataclass
class SegmentRecord:
    numeric_id: int
    public_id: str
    descriptor: Dict[str, str]


class SegmentIdRegistry:
    def __init__(self, prefix: str = "SEG") -> None:
        self.prefix = prefix
        self._by_key: Dict[Tuple[str, str, str], SegmentRecord] = {}
        self._by_numeric: Dict[int, SegmentRecord] = {}

    def get_or_create(
        self,
        source_batch_id: str,
        sampling_rate_hz: float,
        soil_condition: str,
    ) -> SegmentRecord:
        key = (
            source_batch_id,
            normalize_sampling_rate_key(sampling_rate_hz),
            soil_condition,
        )
        if key in self._by_key:
            return self._by_key[key]

        numeric_id = len(self._by_key)
        public_id = f"{self.prefix}_{numeric_id + 1:04d}"
        descriptor = {
            "source_batch_id": source_batch_id,
            "sampling_rate_key": key[1],
            "soil_condition": soil_condition,
            "public_segment_id": public_id,
        }
        record = SegmentRecord(
            numeric_id=numeric_id,
            public_id=public_id,
            descriptor=descriptor,
        )
        self._by_key[key] = record
        self._by_numeric[numeric_id] = record
        return record

    def to_public_json(self) -> str:
        payload = {
            str(numeric_id): record.descriptor
            for numeric_id, record in sorted(self._by_numeric.items())
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_public_file_id(prefix: str, source_batch_id: str, sample_index: int) -> str:
    return f"{prefix}_{source_batch_id}_{sample_index + 1:08d}"


def build_private_mapping_row(
    original_path: Path,
    sample_id: str,
    segment_public_id: str,
    parse_result: str,
    path_hash: str,
) -> Dict[str, str]:
    return {
        "original_path": str(original_path),
        "original_filename": original_path.name,
        "sample_id": sample_id,
        "segment_id": segment_public_id,
        "parse_result": parse_result,
        "hash": path_hash,
    }
