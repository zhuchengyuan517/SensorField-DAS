# SensorField-DAS Data Availability

The generated `SensorField_DAS_v1.h5` file is not currently stored in this GitHub repository. Its local size is approximately 726 MiB, which exceeds GitHub's 100 MiB limit for regular repository files.

## Current Status

- Public download: not yet available
- Repository source data: not included
- Private traceability map: not included
- Planned distribution: a versioned GitHub Release asset or an external archival repository after the associated paper reaches the intended release stage

The repository does not provide a working download command until a permanent public asset URL has been created. This avoids publishing a placeholder or misleading link.

## Expected Release File

| Field | Value |
| --- | --- |
| Filename | `SensorField_DAS_v1.h5` |
| Records | 13,806 |
| Approximate size | 726 MiB |
| SHA256 | `811cb5a7ce5e539089bb0d84d067fd293e700e821fb6358e14dba28f62c266e8` |

The checksum above corresponds to the latest locally generated artifact at the time this repository metadata was prepared. It must be refreshed if the dataset is rebuilt before publication.

## Future Download

After a GitHub Release asset is published, users will be able to download it with a command of the following form:

```powershell
gh release download <VERSION> --pattern "SensorField_DAS_v1.h5" --dir data
```

Do not include `private_mapping.csv` in a public release.
