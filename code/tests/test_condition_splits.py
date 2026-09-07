import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from examples.das_csv.build_condition_splits import (
    _extract_batch_id,
    _extract_region_descriptor,
    _extract_soil_condition,
    greedy_group_split,
)


class ConditionSplitHelpersTest(unittest.TestCase):
    def test_extract_soil_condition(self) -> None:
        self.assertEqual(_extract_soil_condition(r"D:\x\10km-land-5m-施工-1914-54852.csv"), "land")
        self.assertEqual(_extract_soil_condition(r"D:\x\A4-land-sand-shizi-0m-汽车-行驶-1104-7563.csv"), "mixed")
        self.assertEqual(_extract_soil_condition(r"D:\x\A4-0m-2人行走-1038-6035.csv"), "unknown")

    def test_extract_region_descriptor(self) -> None:
        self.assertEqual(
            _extract_region_descriptor(r"D:\x\10km-管道20m-land-施工-1734-48779.csv"),
            "10km-管道20m-land",
        )
        self.assertEqual(
            _extract_region_descriptor(r"D:\x\A4-land-sand-shizi-0m-汽车-行驶-1104-7563.csv"),
            "A4-land-sand-shizi-0m",
        )

    def test_extract_batch_id(self) -> None:
        self.assertEqual(_extract_batch_id(r"D:\x\10km-land-5m-施工-1914-54852.csv"), "1914")
        self.assertEqual(_extract_batch_id(r"D:\x\10km-管道40m-sand-开挖施工-165846911.csv"), "165846911")

    def test_greedy_group_split_keeps_all_rows(self) -> None:
        rows = [
            {"group": "g1", "joint_label": "a||_", "event_label": "a", "distance_label": "", "path": "p1", "sample_mode": "file"},
            {"group": "g1", "joint_label": "a||_", "event_label": "a", "distance_label": "", "path": "p2", "sample_mode": "file"},
            {"group": "g2", "joint_label": "b||x", "event_label": "b", "distance_label": "x", "path": "p3", "sample_mode": "file"},
            {"group": "g3", "joint_label": "c||_", "event_label": "c", "distance_label": "", "path": "p4", "sample_mode": "file"},
            {"group": "g4", "joint_label": "b||y", "event_label": "b", "distance_label": "y", "path": "p5", "sample_mode": "file"},
        ]
        split_rows = greedy_group_split(rows, group_key="group", seed=42)
        all_paths = {row["path"] for split in split_rows.values() for row in split}
        self.assertEqual(all_paths, {"p1", "p2", "p3", "p4", "p5"})
        train_labels = {row["joint_label"] for row in split_rows["train"]}
        self.assertTrue({"a||_", "b||x", "b||y", "c||_"} & train_labels)


if __name__ == "__main__":
    unittest.main()
