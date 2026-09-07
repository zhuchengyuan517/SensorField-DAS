# SensorField-M3T Implementation Spec

## Data And Views

The executable TPAMI protocol uses the CSV-based MTL43 benchmark under
`converted_csv/MTL43`. Each run writes canonicalized manifests into
its own output directory before training, so stale paths are fixed without
mutating the source dataset.

Expected model inputs are:

- Raw: `[B, 1, 10000]`
- STF: `[B, 1, 224, 224]`
- GAF: `[B, 1, 224, 224]`

The retained CSV channel matrix is adapted to the Raw view using
`spatial_adapter=center` for the main paper protocol. STF is generated from the
retained channel matrix before spatial reduction using
`stf_spatial_fusion=group3_mean`: log-magnitude spectra from every adjacent
group of up to three sensing channels are averaged before the group-level maps
are composed and resized. GAF is generated from the
Raw center-adapted trace. Views are normalized independently when
`normalize=sample`.

## Tasks

Task 1 is event-type classification over `walking`, `excavator`, `driving`, and
`background`. Task 2 is conditional threat-location estimation over `Alarm
area`, `Tracking area`, and `No-threat area`, evaluated only where
`distance_cls != -1`. The dataset returns `task_mask = [1, valid_distance]`.

## Metrics

All validation and test metrics are computed through `sensorfield_metrics.py`.
For each task the exported metrics are accuracy, macro-F1, one-vs-rest
macro-AUC, macro-FAR, per-class precision/recall/F1/FAR, confusion matrix,
TaskScore, and MTLScore:

```text
TaskScore = (ACC + Macro-F1 + Macro-AUC + (1 - FAR)) / 4
MTLScore = mean(TaskScore_event, TaskScore_location)
```

Checkpoint selection defaults to validation `mtl_score`.

## Model

`SensorFieldM3T` keeps separate Raw, STF, and GAF encoders. FAC, TAEF, and GCTI
remain separate modules and expose auxiliary outputs when `return_auxiliary` is
enabled. `modality_mask` is supported with columns `[raw, stf, gaf]` and is used
to suppress missing-view evidence weights in TAEF.
