from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = PROJECT_ROOT / "libmtl_das_patch" / "examples" / "das_csv"
for path in (PROJECT_ROOT, EXAMPLE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from single_task_signal import run_single_task  # noqa: E402


if __name__ == "__main__":
    run_single_task(
        {
            "task_name": "mtl43_event_only_balanced",
            "target_key": "event_type",
            "classes_default": "walking,excavator,driving,background",
            "save_root_default": PROJECT_ROOT / "output" / "mtl43_event_only_balanced",
            "sampler_default": "none",
            "normalize_default": "sample",
            "model_type_default": "tcn",
            "input_height_default": 6,
            "input_width_default": 1024,
            "class_weights_default": "off",
            "lr_default": 3e-4,
            "weight_decay_default": 1e-4,
            "base_channels_default": 64,
        }
    )
