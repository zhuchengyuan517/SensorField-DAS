from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(r"D:\proj 1\converted_csv")
TARGET_ROOT = ROOT / "MTL43"
BACKGROUND_SOURCE = ROOT / "0715" / "excavator"
BACKGROUND_TARGET = TARGET_ROOT / "background"
MANIFEST_PATH = TARGET_ROOT / "manifest.csv"
SUMMARY_PATH = TARGET_ROOT / "summary.csv"


def read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.reader(fh))


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


def extract_single_row(path: Path, row_index: int) -> list[list[str]]:
    rows = read_csv(path)
    if len(rows) <= row_index:
        raise ValueError(f"{path} only has {len(rows)} rows, cannot extract row {row_index + 1}")
    return [rows[row_index]]


def clear_directory_files(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for file_path in path.iterdir():
        if file_path.is_file():
            file_path.unlink()


def refresh_manifest(background_rows: list[list[str]]) -> None:
    rows = read_csv(MANIFEST_PATH)
    header, body = rows[0], rows[1:]
    kept = [row for row in body if len(row) >= 2 and row[1] != "background"]
    write_csv(MANIFEST_PATH, [header, *kept, *background_rows])


def refresh_summary(background_count: int) -> None:
    summary_rows = [
        ["group", "count"],
        ["walking", str(sum(1 for _ in (TARGET_ROOT / "walking").glob("*.csv")))],
        ["excavator/5m", str(sum(1 for _ in (TARGET_ROOT / "excavator" / "5m").glob("*.csv")))],
        ["excavator/20m", str(sum(1 for _ in (TARGET_ROOT / "excavator" / "20m").glob("*.csv")))],
        ["excavator/40m", str(sum(1 for _ in (TARGET_ROOT / "excavator" / "40m").glob("*.csv")))],
        ["driving", str(sum(1 for _ in (TARGET_ROOT / "driving").glob("*.csv")))],
        ["background", str(background_count)],
    ]
    total = sum(int(row[1]) for row in summary_rows[1:])
    summary_rows.append(["total", str(total)])
    write_csv(SUMMARY_PATH, summary_rows)


def main() -> None:
    clear_directory_files(BACKGROUND_TARGET)

    manifest_rows: list[list[str]] = []
    count = 0
    for src in sorted(path for path in BACKGROUND_SOURCE.iterdir() if path.is_file()):
        out_name = f"0715_excavator__{src.name}"
        out_path = BACKGROUND_TARGET / out_name
        write_csv(out_path, extract_single_row(src, 4))
        manifest_rows.append([
            str(out_path.relative_to(TARGET_ROOT)),
            "background",
            "",
            str(src),
            "5",
            "1",
        ])
        count += 1

    refresh_manifest(manifest_rows)
    refresh_summary(count)

    print(f"background: {count}")
    print(f"total: {sum(1 for _ in TARGET_ROOT.rglob('*.csv')) - 2}")


if __name__ == "__main__":
    main()
