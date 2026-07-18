# SensorField-DAS Dataset Card

## 1. Overview

SensorField-DAS Dataset is an anonymized public dataset for distributed acoustic sensing (DAS) based infrastructure safety monitoring. The dataset is intended for academic research on event recognition, background discrimination, distance-aware analysis, and domain generalization across heterogeneous acquisition settings.

This release is structured as a publication-ready research dataset rather than an internal engineering archive. The goal is to provide a reusable benchmark format while removing sensitive operational metadata and limiting the disclosure of deployment-specific infrastructure details.

## 2. Motivation

Distributed acoustic sensing has become an important sensing modality for infrastructure monitoring, especially in scenarios where weak vibrations, activity patterns, and environmental disturbances need to be analyzed over extended spatial coverage. However, public DAS datasets for infrastructure safety monitoring remain limited, particularly those that support:

- multiple event categories
- matched background examples
- anonymized public release
- cross-domain evaluation across acquisition conditions

SensorField-DAS Dataset is designed to help fill that gap.

## 3. Intended Research Use

The dataset is intended for:

- supervised event classification
- fine-grained activity recognition
- background versus event discrimination
- distance-aware recognition
- cross-segment generalization
- multi-domain representation learning
- robustness evaluation under different sampling configurations and field conditions

## 4. Out-of-Scope Use

The dataset is not intended to:

- expose real operational sensing layouts
- reveal field deployment identities
- provide access to confidential engineering metadata
- support reverse reconstruction of protected infrastructure attributes

## 5. Data Source and Release Policy

The public release is derived from multi-batch DAS signal archives that include multiple acquisition conditions and environment types. During release preparation, the source corpus is processed through:

- filename inspection
- label parsing
- signal cropping
- anonymization
- background sample generation
- HDF5 packaging
- consistency validation

Only the processed public release outputs are intended for publication.

## 6. Sample Construction

The public dataset does not preserve complete raw fence-level matrices.

Instead, each public sample is constructed as a cropped DAS zone window:

- for positive event samples, the release keeps only the event-related zone segment
- for background samples, the release crops non-event zone segments from the same source file
- background crops are matched in zone-window length to the positive crop
- the final public sample is stored in `[T, C]` format

Where:

- `T` is temporal length
- `C` is the retained zone-window width

This release strategy improves privacy protection and better aligns the dataset with model-training use cases.

## 7. Label Taxonomy

### 7.1 Coarse-Grained Event Labels

The primary event labels are:

- `background_noise`
- `pipeline_leakage`
- `mechanical_excavation`
- `manual_work`
- `vehicle_passing`

### 7.2 Fine-Grained Event Labels

When supported by source naming rules, the dataset may include:

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

### 7.3 Additional Metadata Labels

The release may also provide:

- `distance_label`
- `distance_value_m`
- `soil_condition`
- `segment_id`
- `sampling_rate_hz`
- `is_background`
- `has_distance_label`

## 8. Anonymization

The public release removes or anonymizes sensitive metadata, including:

- original source paths
- original filenames
- real dates
- project identifiers
- station names
- geographic markers
- GPS references
- defense-zone identifiers
- pipe diameter and engineering-sensitive field attributes

Only hashed or remapped identifiers are retained where necessary for integrity, grouping, or traceability in the local build process.

## 9. Data Format

The public dataset is packaged into an HDF5 file with structured groups for:

- signal storage
- labels
- metadata
- quality control
- split definitions
- label maps

This organization is intended to support efficient training, reproducible evaluation, and straightforward downstream loading in Python-based research pipelines.

## 10. Quality Control

During release construction, the pipeline performs multiple checks, including:

- numeric-column extraction from CSV inputs
- NaN and Inf handling
- per-sample signal quality statistics
- structural consistency checks across HDF5 groups
- split-index validation
- leakage checks for sensitive strings or path metadata

Samples that fail public-release requirements may be skipped, while recoverable issues may be retained together with warning annotations.

## 11. Splits and Evaluation

The release is designed to support multiple evaluation settings:

- random split
  - standard supervised benchmarking with approximate train/validation/test partitioning
- segment holdout
  - evaluation on unseen anonymous acquisition segments
- cross-segment folds
  - repeated cross-domain validation across anonymous segments

These splits are intended to support fairer comparison under both standard and domain-shift conditions.

## 12. Limitations

Several limitations should be considered when using the dataset:

- labels are derived from curated naming rules and release-time logic
- some source files may be excluded because they do not satisfy release criteria
- some metadata fields may be unavailable when they cannot be inferred reliably
- public samples are cropped task-oriented signal windows rather than untouched raw full-field recordings
- some ambiguous field patterns may be mapped conservatively to `unknown`

## 13. Reproducibility and Repository Contents

The repository includes:

- release build scripts
- filename inspection utilities
- anonymization logic
- label parsing rules
- split generation code
- validation scripts
- example data loading code
- release configuration files

This allows the public build process to be audited and reproduced locally, subject to access to the original private source corpus.

## 14. Maintenance

Before public release, the following items should be finalized:

- public HDF5 filename and download location
- dataset license
- associated paper citation
- maintainer contact information
- any institutional or funding acknowledgements

## 15. Citation

If you use SensorField-DAS Dataset in academic work, please cite the associated paper and the dataset repository.

```bibtex
@dataset{sensorfield_das_dataset,
  title  = {SensorField-DAS Dataset},
  author = {To Be Added},
  year   = {2026},
  note   = {An anonymized public DAS dataset for infrastructure safety monitoring}
}
```

## 16. License

Please replace this placeholder with the final public dataset license before release.

```text
License: To Be Determined
```
