# SensorField-DAS Code

This directory contains the publication-facing dataset tools and SensorField-M3T implementation.

## Layout

- `models/`: SensorField-M3T, comparison models, and model modules.
- `data/`: dataset loading, release construction, validation, manifest preparation, and alignment auditing.
- `training/`: training, evaluation, multi-seed, and cross-condition protocol entry points.
- `tests/`: focused model and data-protocol tests.

The principal architecture is defined in `models/sensorfield_m3t.py`. Public module-level imports are provided by `models/fac.py`, `models/taef.py`, `models/gcti.py`, and `models/encoders.py` so each paper component can be located and imported by name.

## Main Entry Points

```powershell
python code/training/train_sensorfield_m3t.py --help
python code/training/run_submission_protocol.py --help
python code/data/build_dataset_release.py --help
python code/data/validate_dataset_release.py --help
python -m unittest discover -s code/tests -p "test_*.py"
```

All commands are intended to be launched from the repository root. Generated datasets, private mappings, experiment outputs, and source CSV files remain excluded from version control.
