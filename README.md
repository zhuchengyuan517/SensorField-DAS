# SensorField-DAS Dataset

SensorField-DAS is an anonymized distributed acoustic sensing (DAS) benchmark for distributed sensor-field perception and infrastructure safety monitoring. It is constructed from real-world cross-regional DAS acquisitions and is designed to support multimodal representation learning, multi-task perception, and cross-condition generalization.

Rather than treating each DAS measurement as an isolated signal, SensorField-DAS preserves multiple physically related representations of the same localized response field. In the associated manuscript, each sample is represented by a raw temporal view, a space-time-frequency (STF) view, and a Gramian Angular Field (GAF) view, which capture complementary temporal, spectral, spatial, and correlation structures.

This repository provides the dataset-release pipeline, anonymization and validation utilities, evaluation-split utilities, documentation, and configuration files used to prepare the research release.

## Highlights

- Real-world cross-regional DAS benchmark for distributed sensor-field perception
- 13,806 records covering four coarse event categories
- Two related perception tasks: event-type classification and threat-location estimation
- Three physically related field representations: Raw, STF, and GAF
- Region-, soil-, and acquisition-disjoint protocols for cross-condition generalization
- Source-group-aware partitioning to prevent samples from the same annotated recording from crossing data splits
- Anonymized release with sensitive deployment and infrastructure metadata removed
- HDF5-based packaging with validation and reproducibility utilities

## Dataset Summary

| Field | Description |
| --- | --- |
| Dataset name | SensorField-DAS |
| Sensing modality | Distributed Acoustic Sensing (DAS) |
| Sample size | 13,806 records |
| Native sampling rates | 2 kHz and 10 kHz |
| Public signal format | `[T, C]` localized DAS sensor-field windows |
| Representation views | Raw temporal signal, STF map, and GAF map |
| Task 1 | Event-type classification: 4 classes |
| Task 2 | Threat-location estimation: 3 distance-defined regions for valid mechanical-excavation samples |
| Fine-grained labels | 6 activity subclasses |
| Generalization settings | Region-disjoint, soil-disjoint, and acquisition-disjoint evaluation |
| Main release format | HDF5 |

For the experiments reported in the associated manuscript, all samples correspond to a fixed 5-s observation window. Signals acquired at 10 kHz are downsampled to 2 kHz with anti-aliasing filtering. The resulting raw input is formatted as a single-channel sequence of size `1 x 10,000`, while the STF and GAF representations are constructed as single-channel maps of size `1 x 224 x 224`.

## Label Space

### Event-Type Classification

Task 1 assigns each sample to one of four coarse event categories:

- `background_noise`
- `human_activity`
- `mechanical_excavation`
- `vehicle_driving`

The current benchmark contains:

- 5,899 background-noise records
- 1,470 human-activity records
- 1,957 mechanical-excavation records
- 4,480 vehicle-driving records

### Threat-Location Estimation

Task 2 is defined only for mechanical-excavation samples with a valid source-to-fiber distance. Three operational regions are used:

- `alarm`: `(0, 5] m`
- `tracking`: `(5, 20] m`
- `no_threat`: `(20, 40] m`

Background-noise, human-activity, and vehicle-driving samples are treated as unlabeled for Task 2 rather than being assigned to the no-threat class.

### Fine-Grained Activity Labels

The benchmark additionally provides six fine-grained activity labels:

- Human activity: `walking`, `striking`, `hoeing`
- Mechanical excavation: `construction`, `excavation`, `cutting`

## Sensor-Field Construction

Each source DAS recording initially spans a distributed sensing field containing event responses together with spatially redundant background information. During preprocessing, the event-relevant response is localized from the full sensing field and standardized into the observation window used to construct the three representation views.

During release preparation:

- event-related windows are extracted around responsive sensing zones
- background windows are sampled from non-event regions with matched spatial extents
- localized signals are stored as `[T, C]`, where `T` denotes temporal length and `C` denotes the retained sensing positions
- each extracted sample retains an anonymized source-group identifier linking it to its originating continuous recording, acquisition session, and physical event
- samples from the same source group are never distributed across different partitions in the paper-reported evaluation protocol
- grouping identifiers are used only for partition construction and are not provided to the learning models

The localized sensor-field windows are subsequently transformed into the Raw, STF, and GAF representations used by the learning models.

## Evaluation Protocols

The associated manuscript evaluates SensorField-DAS under both standard and condition-disjoint settings.

### Source-Group-Disjoint Standard Split

The standard benchmark uses a `7:2:1` training/validation/test ratio based on source identifiers. Samples originating from the same annotated recording do not appear across different subsets.

### Condition-Disjoint Evaluation

Three distribution-shift settings are considered:

- **Region-disjoint:** target sensing regions are excluded from training and validation
- **Soil-disjoint:** target soil conditions are excluded from training and validation
- **Acquisition-disjoint:** target acquisition conditions are excluded from training and validation

All compared methods use identical partitions, and model selection is performed only on source-condition validation data.

## Anonymization Policy

The research release removes or anonymizes sensitive infrastructure and deployment information, including:

- original file paths and filenames
- real acquisition dates
- project and station identifiers
- GPS and geographic information
- defense-zone identifiers
- pipe diameter and other engineering-sensitive attributes

Only hashed or remapped identifiers are retained where necessary for integrity checking, grouping, and reproducible evaluation.

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

The repository includes a local release pipeline for:

1. source-file inspection and label parsing
2. sensitive-metadata anonymization
3. sensor-field window extraction and background generation
4. HDF5 packaging
5. dataset statistics and evaluation-split construction
6. structural, leakage, and release validation

## Environment

- Python 3.10+
- Recommended dependencies are listed in `requirements.txt`

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Example Commands

Inspect source files:

```bash
python scripts/inspect_filenames.py \
  --input "path/to/source_csv" \
  --output "public_dataset_release/filename_inspection.json"
```

Build an HDF5 release:

```bash
python scripts/build_hdf5_dataset.py \
  --input "path/to/source_csv" \
  --output "public_dataset_release/SensorField_DAS.h5" \
  --config "config/label_config.yaml" \
  --private-map "public_dataset_release/private_mapping.csv"
```

Validate the generated HDF5 file:

```bash
python scripts/validate_hdf5.py \
  --h5 "public_dataset_release/SensorField_DAS.h5"
```

## Release Artifacts

A complete local build can generate:

- the HDF5 dataset specified by `--output`
- `dataset_statistics.json`
- `dataset_card.md`
- `build_report.md`

The build process may also generate `private_mapping.csv` for local traceability. This file contains private mapping information and **must not be included in the public release package**.

## Recommended Research Tasks

SensorField-DAS is intended to support research on:

- event-type classification
- threat-location estimation
- fine-grained activity recognition
- multimodal and multi-view sensor-field representation learning
- multi-task learning for distributed sensing
- region-, soil-, and acquisition-disjoint generalization
- robustness under sensing-condition variation and incomplete field views

## Documentation

For a more detailed dataset description, see [docs/dataset_card.md](docs/dataset_card.md).

## Notes

- The build scripts do not modify the original source files.
- Samples that do not satisfy release criteria can be skipped and recorded in the build report.
- Sampling-rate and condition metadata are retained only when they can be recovered reliably and released without exposing sensitive deployment information.
- Public samples are task-oriented localized sensor-field windows rather than untouched full deployment-level recordings.
- Exact paper-reported partitions should be used when reproducing results from the associated manuscript.

## Associated Manuscript

**SensorField-M3T: Generalizable Multimodal Multi-Task Learning for Distributed Sensor-Field Perception**

Authors: Chengyuan Zhu, Peiliang Gong, Sean Xu, and Xiaoli Li.

Formal publication metadata will be added when available.

## License

No open-source or open-data license has been granted yet. Until a license file is added, the repository remains subject to default copyright restrictions. A formal license will be added before unrestricted public reuse of the released code or data.
