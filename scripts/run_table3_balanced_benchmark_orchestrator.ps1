$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\proj 1"
$PythonExe = "python"
$Runner = Join-Path $ProjectRoot "scripts\run_table3_balanced_benchmark.py"
$OutputRoot = Join-Path $ProjectRoot "_tmp_table3_balanced_rerun"
$StatusDir = Join-Path $OutputRoot "status"
$BaselineStamp = Join-Path $StatusDir "baseline_done.txt"
$ProposedStamp = Join-Path $StatusDir "proposed_done.txt"
$MergedCsv = Join-Path $OutputRoot "table3_summary_all.csv"
$MergedJson = Join-Path $OutputRoot "table3_summary_all.json"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $StatusDir | Out-Null

Write-Host "[orchestrator] Starting baseline rerun..."
& $PythonExe $Runner `
    --models "resnet,vgg,vit" `
    --epochs 10 `
    --batch_size 48 `
    --num_workers 0 `
    --device cpu `
    --output_root $OutputRoot
Set-Content -Path $BaselineStamp -Value (Get-Date -Format s)

Write-Host "[orchestrator] Starting proposed-model rerun..."
& $PythonExe $Runner `
    --models "proposed" `
    --epochs 10 `
    --batch_size 16 `
    --num_workers 0 `
    --device cpu `
    --output_root $OutputRoot
Set-Content -Path $ProposedStamp -Value (Get-Date -Format s)

Write-Host "[orchestrator] Merging summaries..."
$env:TABLE3_OUTPUT_ROOT = $OutputRoot
$env:TABLE3_MERGED_CSV = $MergedCsv
$env:TABLE3_MERGED_JSON = $MergedJson
@'
import csv
import json
import os
from pathlib import Path

output_root = Path(os.environ["TABLE3_OUTPUT_ROOT"])
merged_csv = Path(os.environ["TABLE3_MERGED_CSV"])
merged_json = Path(os.environ["TABLE3_MERGED_JSON"])

rows = []
for summary_path in sorted(output_root.glob("20*/table3_summary.csv")):
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows.extend(csv.DictReader(handle))

model_order = {"resnet": 0, "vgg": 1, "vit": 2, "proposed": 3}
rows.sort(key=lambda row: (model_order.get(str(row.get("Model", "")).lower(), 99), row.get("Model", "")))

if not rows:
    raise SystemExit("No summary rows found to merge.")

with merged_csv.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

merged_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(str(merged_csv))
'@ | python -

Write-Host "[orchestrator] Completed."
