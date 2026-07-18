from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class ZoneWindow:
    start_row: int
    end_row: int
    window_rows: int
    score: float


def resolve_window_rows(config: Dict, event_type: str, fine_event: str) -> int:
    fine_overrides = config["signal_extraction"]["fine_event_window_rows"]
    if fine_event in fine_overrides:
        return int(fine_overrides[fine_event])
    event_defaults = config["signal_extraction"]["event_window_rows"]
    return int(event_defaults[event_type])


def _window_scores(raw_signal: np.ndarray, window_rows: int) -> np.ndarray:
    row_energy = np.sqrt(np.mean(np.square(raw_signal.astype(np.float32)), axis=1))
    if window_rows <= 1:
        return row_energy
    scores = np.empty(raw_signal.shape[0] - window_rows + 1, dtype=np.float32)
    for start in range(scores.shape[0]):
        scores[start] = float(np.mean(row_energy[start : start + window_rows]))
    return scores


def select_event_window(raw_signal: np.ndarray, window_rows: int) -> ZoneWindow:
    if raw_signal.ndim != 2:
        raise ValueError("raw_signal must be 2D")
    n_rows = raw_signal.shape[0]
    if n_rows < window_rows:
        raise ValueError(f"signal has {n_rows} rows but requires window_rows={window_rows}")

    scores = _window_scores(raw_signal, window_rows)
    start_row = int(np.argmax(scores))
    end_row = start_row + window_rows
    return ZoneWindow(
        start_row=start_row,
        end_row=end_row,
        window_rows=window_rows,
        score=float(scores[start_row]),
    )


def extract_window_time_major(raw_signal: np.ndarray, window: ZoneWindow) -> np.ndarray:
    cropped = raw_signal[window.start_row : window.end_row, :]
    return np.asarray(cropped.T, dtype=np.float32)


def generate_background_windows(
    raw_signal: np.ndarray,
    event_window: ZoneWindow,
    background_per_event: int,
    guard_rows: int,
    stride_mode: str,
) -> List[ZoneWindow]:
    n_rows = raw_signal.shape[0]
    window_rows = event_window.window_rows
    if n_rows < window_rows:
        return []

    stride = window_rows if stride_mode == "window" else 1
    guard_start = max(0, event_window.start_row - guard_rows)
    guard_end = min(n_rows, event_window.end_row + guard_rows)
    scores = _window_scores(raw_signal, window_rows)
    candidates: List[ZoneWindow] = []

    for start_row in range(0, n_rows - window_rows + 1, stride):
        end_row = start_row + window_rows
        overlaps_guard = not (end_row <= guard_start or start_row >= guard_end)
        if overlaps_guard:
            continue
        candidates.append(
            ZoneWindow(
                start_row=start_row,
                end_row=end_row,
                window_rows=window_rows,
                score=float(scores[start_row]),
            )
        )

    candidates.sort(key=lambda item: (item.score, item.start_row))
    if background_per_event > 0:
        return candidates[:background_per_event]
    return candidates
