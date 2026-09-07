# SensorField-DAS Dataset Card

## 1. Overview

SensorField-DAS is a real-world distributed acoustic sensing benchmark created for research on multimodal sensor-field perception and generalizable multi-task learning. The dataset accompanies the paper *SensorField-M3T: Generalizable Multimodal Multi-Task Learning for Distributed Sensor-Field Perception* and is designed as a field-oriented benchmark rather than a collection of isolated vibration snippets.

Each record preserves a localized temporal-spatial response of the sensing field together with event semantics and acquisition-condition context. The benchmark is built from cross-regional DAS acquisitions collected across multiple batches, sensing configurations, and environmental conditions.

## 2. Why This Dataset Exists

Many DAS studies operate on task-specific signal crops or single-condition datasets, which makes it difficult to evaluate how well a model generalizes across sensing regions, soil conditions, or acquisition settings. SensorField-DAS is intended to support a more realistic evaluation setting in which:

- the sensing field is observed as a structured spatial response rather than a single trace
- multiple physical representations of the same event are aligned at the sample level
- coarse and fine semantic labels coexist with distance-aware threat annotations
- condition-disjoint protocols can be used to test robustness under deployment shifts

## 3. Dataset Scale and Composition

The current dataset version contains 13,806 released records.

### 3.1 Coarse-Grained Event Distribution

- `background_noise`: 5,899
- `human_activity`: 1,470
- `mechanical_excavation`: 1,957
- `vehicle_driving`: 4,480

### 3.2 Fine-Grained Labels

Human activity labels:

- `walking`
- `striking`
- `hoeing`

Mechanical excavation labels:

- `construction`
- `excavation`
- `cutting`

### 3.3 Threat-Location Labels

For mechanical excavation records, a three-level threat-location label is provided:

- `alarm`: `(0, 5] m`
- `tracking`: `(5, 20] m`
- `no_threat`: `(20, 40] m`

These labels are only available for mechanical-excavation samples with a valid source-to-fiber distance. Background, human-activity, and vehicle-driving records are unlabeled for this task and are excluded from its loss and evaluation.

### 3.4 Acquisition Characteristics

- Soil conditions: 3
- Native sampling rates: 2 kHz and 10 kHz
- Raw source coverage: 150 spatial monitoring zones per source recording
- Retained released spatial widths: `1`, `6`, and `10` channels
- Observation duration: 5 seconds per released record

## 4. Source Recordings and Preprocessing

Each original DAS source recording spans a long sensing field and therefore contains both event-relevant responses and a large amount of unrelated spatial background. SensorField-DAS is constructed by first localizing the most event-relevant spatial neighborhood and then retaining an event-dependent number of adjacent sensing channels.

- Human activity retains 1 channel.
- Mechanical excavation retains 6 adjacent channels.
- Vehicle source fields retain 10 adjacent channels, which are released as 10 independent single-channel records.
- Background samples are taken from non-event regions using the same spatial width as the matched event configuration.

The released record is represented as `X in R^(T x C)`, where `T` is the temporal length and `C` is the retained spatial width. This organization reduces redundant field observations while preserving localized spatial structure.

## 5. Modalities

Each sample is represented by three aligned modalities:

- `Raw waveform`: preserves native temporal dynamics and inter-channel variation.
- `STF`: a space-time-frequency representation describing spectral energy evolution across time and sensing positions.
- `GAF`: a Gramian angular field representation highlighting temporal correlation patterns in an image-like form.

These three modalities describe complementary aspects of the same physical field response and are intended for multimodal fusion and representation analysis.

## 6. Standardized Benchmark Inputs

For benchmark experiments, all records are mapped to a fixed 5-second observation window.

- Raw records acquired at 2 kHz contain 10,000 temporal samples.
- Raw records originally acquired at 10 kHz are downsampled to 2 kHz with anti-aliasing filtering.
- Each raw input is therefore standardized to `1 x 10000`.
- STF and GAF inputs are constructed as single-channel maps and resized to `1 x 224 x 224`.

This preprocessing keeps a consistent physical duration across conditions while preserving modality-specific structure.

## 7. Supported Learning Tasks

### 7.1 Task 1: Event-Type Classification

The first task is a four-way event classification problem over:

- background noise
- human activity
- mechanical excavation
- vehicle driving

### 7.2 Task 2: Threat-Location Estimation

The second task is a three-way threat-location classification problem defined for mechanical excavation samples:

- alarm zone
- tracking zone
- no-threat zone

Together, these tasks enable joint learning of semantic recognition and distance-aware spatial inference.

## 8. Evaluation Protocols

The current benchmark version supports both standard and condition-shift evaluation.

### 8.1 IID Protocol

The in-distribution protocol uses a train/validation/test split at a ratio of `7:2:1`. Source-group identifiers associate samples with their originating continuous recording, acquisition session, and physical event; a source group must not appear in more than one split.

### 8.2 Condition-Disjoint Protocols

To evaluate generalization under deployment shifts, the benchmark further defines:

- `Region-level`: train and test partitions are disjoint in sensing region
- `Soil-level`: train and test partitions are disjoint in soil condition
- `Acquisition-level`: train and test partitions are disjoint in acquisition setting

These protocols are intended to reveal whether a model remains reliable when the sensing environment changes.

## 9. Data Packaging

The executable paper benchmark is organized through CSV manifests and on-the-fly Raw/STF/GAF construction. The paper-aligned distribution package is `SensorField_DAS_v1.h5`; it contains 13,806 records and preserves the same source-group assignments used by the IID and condition-disjoint protocols. The repository also includes:

- filename inspection utilities
- label parsing logic
- preprocessing and sample-construction scripts
- statistics and reporting helpers
- validation scripts
- example loading code

The build utilities are designed for local reconstruction from the private source corpus and do not alter the original raw CSV files.

## 10. Intended Uses

SensorField-DAS is intended for research on:

- event recognition in DAS-based field monitoring
- fine-grained activity discrimination
- distance-aware threat assessment
- multimodal representation learning
- multi-task learning
- cross-condition generalization
- robustness analysis under sensing-domain shifts

## 11. Known Limitations

Users should keep several limitations in mind:

- The released benchmark is a structured and localized field dataset, not a dump of untouched full-length raw recordings.
- Threat-location labels are defined for mechanical excavation and are not uniformly applicable to every event category.
- Some condition-disjoint evaluations may produce class-imbalance effects or missing-class cases for certain metrics.
- Benchmark-ready inputs are standardized for comparability and may differ from the native raw acquisition format used in deployment.
- Vehicle samples are channel-level records: 448 localized ten-channel source fields produce 4,480 independent records.
- Historical experiment outputs may predate the finalized submission defaults; new runs should use `config/sensorfield_m3t_submission.yaml`.

## 12. Repository Documentation

This repository provides three complementary entry points:

- [README.md](../README.md) for the concise repository overview
- [dataset_card.md](dataset_card.md) for the formal long-form dataset description
- [repository_organization.md](repository_organization.md) for the code map and submission-alignment audit notes

## 13. Citation

If you use SensorField-DAS in academic work, please cite the associated SensorField-M3T paper and the dataset repository. A final BibTeX entry can be added here when the publication metadata is finalized.

## 14. License

The final license statement should be added together with the public dataset release package.
