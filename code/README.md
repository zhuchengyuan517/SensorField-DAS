# SensorField-DAS Code

This directory contains the publication-facing dataset tools and SensorField-M3T implementation.

## Layout

- `LibMTL/model/`: SensorField-M3T and comparison model definitions.
- `examples/das_csv/`: dataset loading, training, protocol, ablation, and artifact-export entry points.
- `tools/`: release construction, validation, manifest preparation, and alignment auditing.
- `tests/`: focused model and data-protocol tests.

## Main Entry Points

```powershell
python code/examples/das_csv/train_sensorfield_m3t.py --help
python code/examples/das_csv/run_submission_protocol.py --help
python code/tools/build_dataset_release.py --help
python code/tools/validate_dataset_release.py --help
python -m unittest discover -s code/tests -p "test_*.py"
```

All commands are intended to be launched from the repository root. Generated datasets, private mappings, experiment outputs, and source CSV files remain excluded from version control.
