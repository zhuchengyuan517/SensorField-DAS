from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


ROOT = Path(r"D:\proj 1\converted_csv")
TARGET_ROOT = ROOT / "MTL43"
WALK_SOURCE_0716 = ROOT / "0716" / "10k-100ns"
WALK_SOURCE_EXTRA = ROOT / "excavator-mtl-dataset-v2" / "samples" / "行驶" / "5m"
EXCAVATOR_SOURCE = ROOT / "excavator-mtl-dataset-v2" / "samples" / "施工"
DRIVING_SOURCE_0716 = ROOT / "0716" / "10k-100ns"
BACKGROUND_SOURCE = ROOT / "0715" / "excavator"


def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.reader(fh))


def write_csv_rows(path: Path, rows: Iterable[Iterable[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


def extract_rows(path: Path, start_idx: int, end_idx_exclusive: int) -> list[list[str]]:
    rows = read_csv_rows(path)
    if len(rows) < end_idx_exclusive:
        raise ValueError(f"{path} only has {len(rows)} rows, cannot extract [{start_idx}:{end_idx_exclusive})")
    return rows[start_idx:end_idx_exclusive]


def reset_target_root() -> None:
    if TARGET_ROOT.exists():
        for item in sorted(TARGET_ROOT.rglob("*"), reverse=True):
            if item.is_file():
                item.unlink()
            else:
                item.rmdir()
        TARGET_ROOT.rmdir()
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)


def collect_by_keyword(folder: Path, keyword: str) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and keyword in path.name)


def main() -> None:
    reset_target_root()

    manifest_rows: list[list[str]] = [[
        "relative_path",
        "event_class",
        "distance_class",
        "source_path",
        "source_rows",
        "output_rows",
    ]]

    counts = {
        "walking": 0,
        "excavator_5m": 0,
        "excavator_20m": 0,
        "excavator_40m": 0,
        "driving": 0,
        "background": 0,
    }

    walking_dir = TARGET_ROOT / "walking"
    excavator_dir = TARGET_ROOT / "excavator"
    driving_dir = TARGET_ROOT / "driving"
    background_dir = TARGET_ROOT / "background"
    background_dir.mkdir(parents=True, exist_ok=True)

    for src in collect_by_keyword(WALK_SOURCE_0716, "行走"):
        out_name = f"0716_walk__{src.name}"
        out_path = walking_dir / out_name
        extracted = extract_rows(src, 82, 83)
        write_csv_rows(out_path, extracted)
        manifest_rows.append([
            str(out_path.relative_to(TARGET_ROOT)),
            "walking",
            "",
            str(src),
            "83",
            "1",
        ])
        counts["walking"] += 1

    for src in sorted(path for path in WALK_SOURCE_EXTRA.iterdir() if path.is_file()):
        out_name = f"excavator_drive5m__{src.name}"
        out_path = walking_dir / out_name
        extracted = extract_rows(src, 8, 9)
        write_csv_rows(out_path, extracted)
        manifest_rows.append([
            str(out_path.relative_to(TARGET_ROOT)),
            "walking",
            "",
            str(src),
            "9",
            "1",
        ])
        counts["walking"] += 1

    for distance in ("5m", "20m", "40m"):
        src_dir = EXCAVATOR_SOURCE / distance
        dst_dir = excavator_dir / distance
        for src in sorted(path for path in src_dir.iterdir() if path.is_file()):
            out_path = dst_dir / src.name
            extracted = extract_rows(src, 3, 9)
            write_csv_rows(out_path, extracted)
            manifest_rows.append([
                str(out_path.relative_to(TARGET_ROOT)),
                "excavator",
                distance,
                str(src),
                "4-9",
                "6",
            ])
            counts[f"excavator_{distance}"] += 1

    for src in collect_by_keyword(DRIVING_SOURCE_0716, "行驶"):
        out_name = f"0716_drive__{src.name}"
        out_path = driving_dir / out_name
        extracted = extract_rows(src, 78, 88)
        write_csv_rows(out_path, extracted)
        manifest_rows.append([
            str(out_path.relative_to(TARGET_ROOT)),
            "driving",
            "",
            str(src),
            "79-88",
            "10",
        ])
        counts["driving"] += 1

    for src in sorted(path for path in BACKGROUND_SOURCE.iterdir() if path.is_file()):
        out_name = f"0715_excavator__{src.name}"
        out_path = background_dir / out_name
        extracted = extract_rows(src, 4, 5)
        write_csv_rows(out_path, extracted)
        manifest_rows.append([
            str(out_path.relative_to(TARGET_ROOT)),
            "background",
            "",
            str(src),
            "5",
            "1",
        ])
        counts["background"] += 1

    write_csv_rows(TARGET_ROOT / "manifest.csv", manifest_rows)

    summary_rows = [
        ["group", "count"],
        ["walking", str(counts["walking"])],
        ["excavator/5m", str(counts["excavator_5m"])],
        ["excavator/20m", str(counts["excavator_20m"])],
        ["excavator/40m", str(counts["excavator_40m"])],
        ["driving", str(counts["driving"])],
        ["background", str(counts["background"])],
        ["total", str(
            counts["walking"]
            + counts["excavator_5m"]
            + counts["excavator_20m"]
            + counts["excavator_40m"]
            + counts["driving"]
            + counts["background"]
        )],
    ]
    write_csv_rows(TARGET_ROOT / "summary.csv", summary_rows)

    print("MTL43 dataset created.")
    for group, count in summary_rows[1:]:
        print(f"{group}: {count}")


if __name__ == "__main__":
    main()
