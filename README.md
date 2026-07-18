# SensorField-DAS Dataset

SensorField-DAS Dataset is an anonymized public dataset for distributed acoustic sensing (DAS) based infrastructure safety monitoring. It is designed for research on event classification, background discrimination, distance-aware recognition, and cross-domain generalization across multiple acquisition conditions.

This repository serves as the dataset release and build repository for the public version of the dataset. It includes dataset construction scripts, anonymization and validation utilities, release documentation, and configuration files for reproducible packaging.

## Highlights

- Public-release DAS dataset for infrastructure safety monitoring research
- Unified HDF5 packaging for scalable training and evaluation
- Cropped event-zone and background-zone signal windows instead of full raw fence matrices
- Anonymized release with sensitive field metadata removed
- Built-in dataset statistics, validation checks, and evaluation splits

## Dataset Summary

| Field | Description |
| --- | --- |
| Dataset name | SensorField-DAS Dataset |
| Modality | Distributed Acoustic Sensing (DAS) signals |
| Release type | Public anonymized research dataset |
| Main format | HDF5 |
| Sample format | `[T, C]` cropped DAS zone windows |
| Target tasks | Event recognition, background discrimination, domain generalization |

## Label Space

### Coarse-Grained Event Labels

- `background_noise`
- `pipeline_leakage`
- `mechanical_excavation`
- `manual_work`
- `vehicle_passing`

### Fine-Grained Event Labels

When supported by source naming rules, the release may additionally include:

- `excavator_idle`
- `knocking`
- `digging`
- `parallel_driving`
- `crossing`
- `manual_digging`
- `manual_walking`
- `vehicle_idle`
- `vehicle_passing`
- `N/A`
- `unknown`

## Public Signal Construction

The public release does not preserve full raw fence-level matrices.

Instead:

- each eligible event source file is cropped into one event-related zone window
- additional background samples are generated from non-event zone windows in the same source file
- background crops are constrained to the same zone-window length as the corresponding event crop
- signals are stored as `[T, C]`, where `T` is temporal length and `C` is retained zone-window width

This design supports privacy-aware publication while preserving task-relevant temporal-spatial DAS structure.

## Anonymization Policy

The public dataset removes or anonymizes sensitive infrastructure and deployment information, including:

- original file paths
- original filenames
- real acquisition dates
- project names
- station names
- GPS or location information
- defense-zone identifiers
- pipe diameter and engineering-sensitive attributes

Only hashed or remapped identifiers are retained when necessary for integrity checking or reproducible grouping.

## Repository Layout

```text
.
├── README.md
├── config/
│   └── label_config.yaml
├── docs/
│   └── dataset_card.md
├── scripts/
│   ├── inspect_filenames.py
│   ├── build_hdf5_dataset.py
│   ├── validate_hdf5.py
│   └── dataset_loader_example.py
├── src/
│   ├── anonymizer.py
│   ├── hdf5_writer.py
│   ├── label_parser.py
│   ├── split_builder.py
│   ├── stats_report.py
│   └── zone_extractor.py
└── requirements.txt
```

## Build Workflow

The repository includes a complete local release pipeline:

1. inspect source filenames and summarize parsing patterns
2. parse labels and anonymize sensitive metadata
3. crop event-zone samples and generate background-zone samples
4. package the public release into HDF5
5. build statistics, reports, and evaluation splits
6. validate structure consistency and leakage constraints

## Environment

- Python 3.10+
- Recommended dependencies are listed in `requirements.txt`

Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Commands

Inspect filenames:

```powershell
python scripts/inspect_filenames.py `
  --input "D:\proj 1\converted_csv" `
  --output "D:\proj 1\public_dataset_release\filename_inspection.json"
```

Build the public HDF5 release:

```powershell
python scripts/build_hdf5_dataset.py `
  --input "D:\proj 1\converted_csv" `
  --output "D:\proj 1\public_dataset_release\PipeDAS_Multi_v1.h5" `
  --config "D:\proj 1\config\label_config.yaml" `
  --private-map "D:\proj 1\public_dataset_release\private_mapping.csv"
```

Validate the generated HDF5 file:

```powershell
python scripts/validate_hdf5.py `
  --h5 "D:\proj 1\public_dataset_release\PipeDAS_Multi_v1.h5"
```

## Release Artifacts

The build pipeline writes the following public-facing outputs under `public_dataset_release/`:

- `PipeDAS_Multi_v1.h5`
- `dataset_statistics.json`
- `dataset_card.md`
- `build_report.md`

The pipeline also produces:

- `private_mapping.csv`

`private_mapping.csv` is for local traceability only and must not be included in the public release package.

## Recommended Research Tasks

SensorField-DAS Dataset is suitable for:

- DAS event classification
- fine-grained activity recognition
- background versus event discrimination
- distance-aware event analysis
- robustness evaluation across acquisition settings
- cross-segment and cross-domain generalization studies

## Documentation

For a more formal paper-style dataset description, see:

- [docs/dataset_card.md](</D:/proj 1/docs/dataset_card.md>)

## Notes

- The build scripts do not modify the original files under `converted_csv/`.
- Source files that do not satisfy public-release criteria can be skipped and recorded in the reports.
- Mixed or ambiguous source naming patterns may be mapped to conservative labels such as `unknown`.
- Sampling-rate metadata is preserved when recoverable from source naming rules and left unavailable otherwise.
- Full-dataset construction may require substantial disk space because all public samples are packed into one HDF5 release.

## Citation

If you use SensorField-DAS Dataset in academic work, please cite the associated paper and dataset repository.

```bibtex
@dataset{sensorfield_das_dataset,
  title  = {SensorField-DAS Dataset},
  author = {To Be Added},
  year   = {2026},
  note   = {An anonymized public DAS dataset for infrastructure safety monitoring}
}
```

## License

Please replace this section with the final dataset license statement before public release.

```text
License: To Be Determined
```
