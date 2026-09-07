# SensorField-DAS and SensorField-M3T Repository Guide

This document maps the local workspace to the definitions used in the current submission of *SensorField-M3T: Generalizable Multimodal Multi-Task Learning for Distributed Sensor-Field Perception*.

## Canonical Submission Scope

The paper-facing benchmark is `converted_csv/MTL43`, with four event classes (`background`, `walking`, `excavator`, and `driving`) and a conditional three-class threat-location task for mechanical-excavation samples. The model uses aligned Raw, STF, and GAF views and the FAC, TAEF, and GCTI modules.

The canonical machine-readable description is `config/sensorfield_m3t_submission.yaml`. When a historical script default conflicts with that file, the YAML file records the current submission claim and the discrepancy must be resolved before a reproducibility release.

## Data Code

| Purpose | Canonical local path |
| --- | --- |
| Paper benchmark | `converted_csv/MTL43/` |
| IID manifests | `converted_csv/MTL43/{train,val,test}.csv` |
| Strict condition splits | `converted_csv/sensorfield_mtl43_condition_splits_strict/` |
| CSV dataset and Raw/STF/GAF construction | `libmtl_das_patch/examples/das_csv/create_dataset.py` |
| Condition-disjoint split builder | `libmtl_das_patch/examples/das_csv/build_sensorfield_condition_splits.py` |
| Submission alignment audit | `scripts/audit_submission_alignment.py` |
| Paper-aligned HDF5 builder | `scripts/build_sensorfield_hdf5.py` |
| Generated public dataset | `public_dataset_release/SensorField_DAS_v1.h5` |

The root `scripts/build_hdf5_dataset.py`, `src/`, and `config/label_config.yaml` belong to the earlier five-class PipeDAS workflow and are retained only for provenance. The legacy `PipeDAS_Multi_v1.h5` artifact has been removed; `SensorField_DAS_v1.h5` is the paper-aligned local package.

## Model Code

| Purpose | Canonical local path |
| --- | --- |
| SensorField-M3T model | `libmtl_das_patch/LibMTL/model/sensorfield_m3t.py` |
| Training and evaluation entry | `libmtl_das_patch/examples/das_csv/pipemmtl_main.py` |
| Metrics and MTLScore | `libmtl_das_patch/examples/das_csv/sensorfield_metrics.py` |
| Multi-seed protocol runner | `libmtl_das_patch/examples/das_csv/sensorfield_m3t_mtl43_multiseed_runner.py` |
| Cross-condition runner | `libmtl_das_patch/examples/das_csv/run_cross_condition_baselines.py` |
| Core model tests | `libmtl_das_patch/tests/test_sensorfield_m3t.py` |
| Data/protocol tests | `libmtl_das_patch/tests/test_sensorfield_tpami_protocol.py` |

## Paper-Facing Input and Task Contract

- Raw input: `1 x 10000`, representing a 5-second signal standardized to 2 kHz.
- STF input: `1 x 224 x 224`.
- GAF input: `1 x 224 x 224`.
- Task 1: four-class event-type classification for every sample.
- Task 2: alarm `(0, 5] m`, tracking `(5, 20] m`, or no-threat `(20, 40] m`, only for excavation samples with valid distance labels.
- IID partition: `7:2:1`, grouped by source identifiers.
- Generalization protocols: condition-disjoint region, soil, and acquisition splits.

## Current Alignment Findings

The dataset descriptions and paper-facing configuration now follow the submission. The local research code contains the three encoders and FAC/TAEF/GCTI modules, conditional Task-2 masking, MTLScore metrics, source-manifest canonicalization, and strict condition-split tooling.

The current executable path is aligned as follows:

- Vehicle channels are independent records: 448 ten-channel source fields yield 4,480 vehicle records and 13,806 total records.
- Paper-facing runners use 80 epochs, batch size 8, 16 anchors, perturbation probability 0.3, and representation-consistency weight 0.05.
- GCTI implements the paper's query/key task-relation interaction and combines prediction consistency with stop-gradient representation consistency for the perturbed-view path; Task-2 terms are masked when distance labels are unavailable.
- The HDF5 package and all IID, region-, soil-, and acquisition-level splits are source-group disjoint.

## Recommended Local Checks

```powershell
python scripts/audit_submission_alignment.py
python -m unittest `
  libmtl_das_patch.tests.test_sensorfield_m3t `
  libmtl_das_patch.tests.test_sensorfield_tpami_protocol `
  libmtl_das_patch.tests.test_build_sensorfield_condition_splits
```

The audit writes a local JSON report under `output/`; generated data, checkpoints, results, and private paths remain excluded from version control.
