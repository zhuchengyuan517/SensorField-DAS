# SensorField-M3T Assumptions And Gaps

## Confirmed In This Workspace

- The executable four-class MTL43 benchmark and the paper-aligned
  `SensorField_DAS_v1.h5` package use the same 13,806-record accounting.
- Current MTL43 manifests contain stale `MTL43\walking` paths. The actual files
  are available under `MTL43\human activities` and top-level `converted_csv\walking`.
- CSV files use retained channel matrices with observed shapes `1x10000`,
  `6x10000`, and `10x10000`.

## Assumptions

- The main Raw adapter is `center`, as confirmed for the current implementation
  pass.
- `mean` and `learned` adapters are sensitivity settings only. The current
  `learned` adapter is a deterministic high-energy row weighting proxy in the
  data pipeline; it should not be described as the main paper result.
- Event/source IDs are derived from canonicalized filenames because the current
  CSV manifests do not contain explicit `source_recording_id` or
  `event_instance_id` columns.

## Open Gaps

- The exact private preprocessing rule for `150 source zones -> retained
  channels -> single Raw trace` is not present in the public CSV manifest.
- Vehicle source fields contain 10 channels, and each channel is consistently
  treated as an independent record, giving 4,480 vehicle-driving records.
