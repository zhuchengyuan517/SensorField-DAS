# SensorField-DAS

SensorField-DAS is a real-world distributed acoustic sensing benchmark developed for the paper *SensorField-M3T: Generalizable Multimodal Multi-Task Learning for Distributed Sensor-Field Perception*. Rather than treating DAS recordings as isolated one-dimensional traces, the dataset is organized as a field-oriented benchmark that preserves localized temporal-spatial responses, hierarchical event semantics, and acquisition-condition metadata needed for cross-condition evaluation.

This repository organizes the paper-facing dataset documentation, the SensorField-M3T research implementation, and the reproducibility utilities used for the current submission.

## Overview

SensorField-DAS is constructed from cross-regional DAS acquisitions collected across multiple batches, sensing configurations, and environmental conditions. During preprocessing, each source recording is localized to the most responsive sensing neighborhood, and a task-matched background counterpart is sampled from non-event regions with the same spatial extent. The resulting dataset emphasizes event-relevant field structure instead of full raw fence recordings with large redundant backgrounds.

The current benchmark is designed for multimodal and multi-task learning under both standard in-distribution evaluation and condition-disjoint generalization settings.

## Key Characteristics

| Attribute | Value |
| --- | --- |
| Dataset name | SensorField-DAS |
| Total records | 13,806 |
| Event classes | 4 coarse-grained classes |
| Fine-grained labels | 6 subclasses |
| Soil conditions | 3 |
| Native sampling rates | 2 kHz and 10 kHz |
| Raw sensing coverage | 150 spatial monitoring zones per source recording |
| Retained spatial width | `C in {1, 6, 10}` |
| Observation window | 5 s per released record |
| Standardized raw input | `1 x 10000` |
| Modalities | Raw waveform, STF, and GAF |
| Main tasks | Event-type classification and threat-location estimation |

## Dataset Composition

### Coarse-Grained Event Types

- `background_noise`: 5,899
- `human_activity`: 1,470
- `mechanical_excavation`: 1,957
- `vehicle_driving`: 4,480

The 448 localized vehicle source files each contain 10 sensing channels. Each channel is released as one independent five-second record, yielding 4,480 vehicle records and 13,806 records in total.

### Fine-Grained Labels

Human activity:

- `walking`
- `striking`
- `hoeing`

Mechanical excavation:

- `construction`
- `excavation`
- `cutting`

### Threat-Location Labels

For mechanical excavation records, the benchmark further provides a three-level distance-aware threat label:

- `alarm`: `(0, 5] m`
- `tracking`: `(5, 20] m`
- `no_threat`: `(20, 40] m`

Task 2 is only defined for mechanical-excavation samples with a valid source-to-fiber distance. All other event types are masked from the Task-2 loss and evaluation; they are not assigned to the no-threat class.

## Released Record Form

Each released record is a localized field sample represented as `X in R^(T x C)`, where `T` is the temporal length and `C` is the retained spatial width.

- Human activity records retain 1 channel.
- Mechanical excavation records retain 6 adjacent channels.
- Vehicle source fields contain 10 adjacent channels; each channel is packaged as an independent `1 x 10000` record.
- Background records are sampled with the same spatial width as their matched event configuration.

For model benchmarking, all records correspond to a fixed 5-second observation window. Raw signals are standardized to 10,000 temporal points. STF and GAF views are constructed as single-channel maps and resized to `224 x 224`.

## Modalities

SensorField-DAS provides three aligned field representations for each sample:

- `Raw waveform`: preserves temporal response and inter-channel variation.
- `STF`: characterizes space-time-frequency energy evolution.
- `GAF`: encodes temporal correlation structure in an image-like form.

These representations are aligned at the sample, event, and physical-context levels and are intended for multimodal fusion, representation analysis, and multi-task learning.

## Benchmark Tasks

### Task 1: Event-Type Classification

Four-way classification over:

- background noise
- human activity
- mechanical excavation
- vehicle driving

### Task 2: Threat-Location Estimation

Three-way classification for mechanical excavation samples:

- alarm zone
- tracking zone
- no-threat zone

## Evaluation Protocols

The current paper version uses both in-distribution and cross-condition protocols:

- `IID`: standard random train/validation/test partition at a `7:2:1` ratio
- `Region-level`: condition-disjoint split across sensing regions
- `Soil-level`: condition-disjoint split across soil conditions
- `Acquisition-level`: condition-disjoint split across acquisition settings

These protocols are designed to evaluate not only recognition performance but also robustness under deployment shifts.

## SensorField-M3T

The accompanying model uses separate encoders for Raw, STF, and GAF inputs, followed by three coordinated components:

- `FAC` separates cross-view shared field factors from view-specific complementary evidence.
- `TAEF` constructs task-dependent representations by selecting evidence for each prediction objective.
- `GCTI` constructs a task-relation matrix from query/key projections, applies residual cross-task feature interaction, and enforces prediction plus stop-gradient representation consistency between complete and perturbed observations, following the paper formulation.

The paper-facing training protocol uses AdamW for 80 epochs, batch size 8, learning rate `3e-5`, weight decay `5e-4`, 16 field anchors, and five independent seeds. The full machine-readable specification is in `config/sensorfield_m3t_submission.yaml`.

## Repository Contents

```text
.
|-- README.md
|-- config/
|   |-- sensorfield_m3t_submission.yaml
|   `-- label_config.yaml                 # legacy PipeDAS release rules
|-- docs/
|   |-- dataset_card.md
|   `-- repository_organization.md
|-- code/
|   |-- LibMTL/model/sensorfield_m3t.py
|   |-- examples/das_csv/sensorfield_dataset.py
|   |-- examples/das_csv/train_sensorfield_m3t.py
|   |-- examples/das_csv/run_submission_protocol.py
|   |-- tests/
|   `-- tools/
|-- scripts/                              # legacy analysis utilities
|   |-- inspect_filenames.py
|   `-- build_hdf5_dataset.py
|-- src/                                  # legacy release modules
|   |-- anonymizer.py
|   |-- hdf5_writer.py
|   |-- label_parser.py
|   |-- split_builder.py
|   `-- stats_report.py
|-- requirements.txt
`-- requirements-model.txt
```

## Reproducibility Checks

Install dataset and model dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-model.txt
```

Audit the local assets against the current submission:

```powershell
python code/tools/audit_dataset_alignment.py
```

Run focused model and protocol tests:

```powershell
python -m unittest discover -s code/tests -p "test_*.py"
```

## HDF5 Release

Build the paper-aligned SensorField-DAS package from the curated manifests:

```powershell
python code/tools/build_dataset_release.py `
  --dataset-root "<PRIVATE_CSV_ROOT>\MTL43" `
  --condition-root "<PRIVATE_CSV_ROOT>\sensorfield_mtl43_condition_splits_strict" `
  --output "<LOCAL_OUTPUT_ROOT>\SensorField_DAS_v1.h5" `
  --private-map "<LOCAL_OUTPUT_ROOT>\private_mapping.csv"
```

Validate its structure, paper-reported counts, vehicle record shape, source-group isolation, label maps, and metadata safety:

```powershell
python code/tools/validate_dataset_release.py `
  --h5 "<LOCAL_OUTPUT_ROOT>\SensorField_DAS_v1.h5"
```

The private mapping is for local traceability only and must not be included in a public archive.

## Documentation

The long-form dataset description is available in [the Dataset Card](docs/dataset_card.md). The code and data map, including known submission-alignment gaps, is maintained in [the Repository Guide](docs/repository_organization.md).

## Citation

If you use SensorField-DAS, please cite the associated SensorField-M3T paper and this repository. A formal BibTeX entry can be added here once the publication metadata is finalized.

## License

License information will be added together with the final public release package.
