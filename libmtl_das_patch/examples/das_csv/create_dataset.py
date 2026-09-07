import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


DISTANCE_IGNORE_INDEX = -1
VIEW_NAMES = ("raw", "stf", "gaf")
STF_SPATIAL_FUSIONS = ("group3_mean", "channel_stack")
DISTANCE_LABEL_ALIASES = {
    "5m": "Alarm area",
    "20m": "Tracking area",
    "40m": "No-threat area",
    "Alarm area": "Alarm area",
    "Tracking area": "Tracking area",
    "No-threat area": "No-threat area",
}
TRAILING_ID_PATTERN = re.compile(r"-\d+(?:-\d+)?$")


def parse_label_list(label_text):
    labels = [item.strip() for item in label_text.split(",") if item.strip()]
    if not labels:
        raise ValueError("Label list cannot be empty.")
    return labels


def canonicalize_distance_label(label_text):
    label = str(label_text).strip()
    if not label:
        return ""
    return DISTANCE_LABEL_ALIASES.get(label, label)


def _strip_trailing_ids(stem):
    return TRAILING_ID_PATTERN.sub("", stem)


def resolve_manifest_sample_path(path, manifest_path):
    """Resolve stale MTL43 manifest paths without modifying the source CSV files."""
    manifest_path = Path(manifest_path)
    path = Path(path)
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    if path.is_file():
        return path, False

    candidates = []
    path_text = str(path)
    if "\\MTL43\\walking\\" in path_text:
        candidates.append(Path(path_text.replace("\\MTL43\\walking\\", "\\MTL43\\human activities\\")))
        candidates.append(Path(path_text.replace("\\MTL43\\walking\\", "\\walking\\")))
    if "/MTL43/walking/" in path_text:
        candidates.append(Path(path_text.replace("/MTL43/walking/", "/MTL43/human activities/")))
        candidates.append(Path(path_text.replace("/MTL43/walking/", "/walking/")))

    for parent in [manifest_path.parent, *manifest_path.parents]:
        candidates.extend(
            [
                parent / "human activities" / path.name,
                parent / "walking" / path.name,
                parent.parent / "walking" / path.name,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(), True
    raise FileNotFoundError(f"CSV sample does not exist: {path}")


def canonicalize_dataset_manifests(dataset_path, output_root):
    dataset_root = Path(dataset_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_dataset_path": str(dataset_root),
        "output_root": str(output_root),
        "splits": {},
        "canonicalized_paths": [],
    }
    for split in ("train", "val", "test"):
        source_manifest = dataset_root / f"{split}.csv"
        if not source_manifest.is_file():
            raise FileNotFoundError(f"Required manifest was not found: {source_manifest}")
        output_manifest = output_root / f"{split}.csv"
        with source_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            if "path" not in fieldnames:
                raise ValueError(f"{source_manifest} must contain a path column.")
            for optional_field in ("distance_label", "sample_mode"):
                if optional_field not in fieldnames:
                    fieldnames.append(optional_field)
            rows = []
            event_counter = Counter()
            distance_counter = Counter()
            mode_counter = Counter()
            for row in reader:
                original_path = row["path"]
                resolved_path, changed = resolve_manifest_sample_path(original_path, source_manifest)
                row["path"] = str(resolved_path)
                row["distance_label"] = canonicalize_distance_label(row.get("distance_label", ""))
                rows.append(row)
                event_counter[row.get("event_label", "")] += 1
                distance_counter[row.get("distance_label", "") or "_"] += 1
                mode_counter[row.get("sample_mode", "") or "file"] += 1
                if changed:
                    summary["canonicalized_paths"].append(
                        {"split": split, "from": original_path, "to": str(resolved_path)}
                    )
        with output_manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        summary["splits"][split] = {
            "rows": len(rows),
            "event_counts": dict(event_counter),
            "distance_counts": dict(distance_counter),
            "sample_mode_counts": dict(mode_counter),
        }
    return summary


def audit_dataset_manifests(dataset_path, output_path=None):
    dataset_root = Path(dataset_path).expanduser().resolve()
    split_ids = {}
    split_families = {}
    split_rows = {}
    for split in ("train", "val", "test"):
        manifest_path = dataset_root / f"{split}.csv"
        rows = []
        ids = set()
        families = set()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Required manifest was not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                resolved_path, _ = resolve_manifest_sample_path(row["path"], manifest_path)
                source_recording_id = resolved_path.stem
                event_instance_id = resolved_path.stem
                source_family_id = _strip_trailing_ids(resolved_path.stem)
                row = dict(row)
                row["resolved_path"] = str(resolved_path)
                row["source_recording_id"] = source_recording_id
                row["event_instance_id"] = event_instance_id
                row["source_family_id"] = source_family_id
                rows.append(row)
                ids.add(event_instance_id)
                families.add(source_family_id)
        split_rows[split] = rows
        split_ids[split] = ids
        split_families[split] = families

    overlaps = {}
    family_overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(split_ids[left] & split_ids[right])
        overlaps[f"{left}_vs_{right}"] = {"count": len(overlap), "examples": overlap[:20]}
        family_overlap = sorted(split_families[left] & split_families[right])
        family_overlaps[f"{left}_vs_{right}"] = {"count": len(family_overlap), "examples": family_overlap[:20]}
    summary = {
        "dataset_path": str(dataset_root),
        "splits": {name: {"rows": len(rows)} for name, rows in split_rows.items()},
        "event_instance_overlaps": overlaps,
        "coarse_source_family_overlaps": family_overlaps,
        "leakage_passed": all(item["count"] == 0 for item in overlaps.values()),
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


class MultiTaskCSVDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        event_to_idx,
        distance_to_idx,
        input_height,
        input_width,
        sample_level="manifest",
        normalize="sample",
        augment=False,
        augment_noise_std=0.0,
        augment_gain_std=0.0,
        augment_shift=0,
        augment_mask_width=0,
        augment_drop_rows=0,
        location_aug_repeats=0,
        location_aug_noise_std=0.0,
        location_aug_gain_std=0.0,
        location_aug_shift=0,
        location_aug_mask_width=0,
        location_aug_drop_rows=0,
        return_multiview=False,
        spatial_adapter="center",
        stf_spatial_fusion="group3_mean",
        stf_size=224,
        gaf_size=224,
        use_stft_aux=False,
        stft_n_fft=256,
        stft_hop_length=128,
        stft_win_length=256,
    ):
        self.manifest_path = Path(manifest_path)
        self.event_to_idx = event_to_idx
        self.distance_to_idx = distance_to_idx
        self.input_height = input_height
        self.input_width = input_width
        self.sample_level = sample_level
        self.normalize = normalize
        self.augment = augment
        self.augment_noise_std = max(float(augment_noise_std), 0.0)
        self.augment_gain_std = max(float(augment_gain_std), 0.0)
        self.augment_shift = max(int(augment_shift), 0)
        self.augment_mask_width = max(int(augment_mask_width), 0)
        self.augment_drop_rows = max(int(augment_drop_rows), 0)
        self.location_aug_repeats = max(int(location_aug_repeats), 0)
        self.location_aug_noise_std = max(float(location_aug_noise_std), 0.0)
        self.location_aug_gain_std = max(float(location_aug_gain_std), 0.0)
        self.location_aug_shift = max(int(location_aug_shift), 0)
        self.location_aug_mask_width = max(int(location_aug_mask_width), 0)
        self.location_aug_drop_rows = max(int(location_aug_drop_rows), 0)
        self.return_multiview = bool(return_multiview)
        self.spatial_adapter = str(spatial_adapter).strip().lower()
        if self.spatial_adapter not in {"center", "mean", "learned"}:
            raise ValueError("spatial_adapter must be one of: center, mean, learned")
        self.stf_spatial_fusion = str(stf_spatial_fusion).strip().lower()
        if self.stf_spatial_fusion not in STF_SPATIAL_FUSIONS:
            raise ValueError(
                f"stf_spatial_fusion must be one of: {', '.join(STF_SPATIAL_FUSIONS)}"
            )
        self.stf_size = int(stf_size)
        self.gaf_size = int(gaf_size)
        self.use_stft_aux = bool(use_stft_aux)
        self.stft_n_fft = int(stft_n_fft)
        self.stft_hop_length = int(stft_hop_length)
        self.stft_win_length = int(stft_win_length)
        self._stft_window = torch.hann_window(self.stft_win_length)
        self._cached_path = None
        self._cached_signal = None
        self.samples = self._read_manifest()
        if self.augment and self.location_aug_repeats > 0:
            self.samples = self._expand_location_samples(self.samples)

    def _count_rows(self, path):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return sum(1 for line in handle if line.strip())

    def _read_manifest(self):
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        samples = []
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {"path", "event_label"}
            if reader.fieldnames is None or not required_columns.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"{self.manifest_path} must contain columns: path,event_label"
                )

            for row in reader:
                path, _ = resolve_manifest_sample_path(row["path"], self.manifest_path)

                event_label = row["event_label"].strip()
                if event_label not in self.event_to_idx:
                    raise KeyError(
                        f"Unknown event label '{event_label}' in {self.manifest_path}"
                    )

                distance_label = canonicalize_distance_label(row.get("distance_label", ""))
                if distance_label:
                    if distance_label not in self.distance_to_idx:
                        raise KeyError(
                            f"Unknown distance label '{distance_label}' in {self.manifest_path}"
                        )
                    distance_idx = self.distance_to_idx[distance_label]
                else:
                    distance_idx = DISTANCE_IGNORE_INDEX

                if self.sample_level == "manifest":
                    sample_mode = row.get("sample_mode", "").strip() or "file"
                else:
                    sample_mode = self.sample_level
                if sample_mode not in {"row", "file", "group3"}:
                    raise ValueError(f"Unsupported sample mode '{sample_mode}' in {self.manifest_path}")

                augment_profile = str(row.get("augment_profile", "")).strip() or "base"
                supervision_profile = str(row.get("supervision_profile", "")).strip() or "full"
                source_recording_id = path.stem
                event_instance_id = path.stem
                source_family_id = _strip_trailing_ids(path.stem)

                row_count = self._count_rows(path)
                if row_count <= 0:
                    raise ValueError(f"CSV sample is empty: {path}")

                if sample_mode == "file":
                    samples.append(
                        {
                            "path": path,
                            "row_count": row_count,
                            "sample_mode": sample_mode,
                            "event_label": self.event_to_idx[event_label],
                            "distance_label": distance_idx,
                            "augment_profile": augment_profile,
                            "supervision_profile": supervision_profile,
                            "source_recording_id": source_recording_id,
                            "event_instance_id": event_instance_id,
                            "source_family_id": source_family_id,
                        }
                    )
                    continue

                if sample_mode == "group3":
                    group_size = 3
                    group_count = row_count // group_size
                    if group_count <= 0:
                        raise ValueError(
                            f"CSV sample requires at least {group_size} rows for group3 mode: {path}"
                        )
                    for group_index in range(group_count):
                        row_start = group_index * group_size
                        row_end = row_start + group_size
                        samples.append(
                            {
                                "path": path,
                                "row_start": row_start,
                                "row_end": row_end,
                                "row_count": row_count,
                                "sample_mode": sample_mode,
                                "event_label": self.event_to_idx[event_label],
                                "distance_label": distance_idx,
                                "augment_profile": augment_profile,
                                "supervision_profile": supervision_profile,
                                "source_recording_id": source_recording_id,
                                "event_instance_id": event_instance_id,
                                "source_family_id": source_family_id,
                            }
                        )
                    continue

                for row_index in range(row_count):
                    samples.append(
                        {
                            "path": path,
                            "row_index": row_index,
                            "row_count": row_count,
                            "sample_mode": sample_mode,
                            "event_label": self.event_to_idx[event_label],
                            "distance_label": distance_idx,
                            "augment_profile": augment_profile,
                            "supervision_profile": supervision_profile,
                            "source_recording_id": source_recording_id,
                            "event_instance_id": event_instance_id,
                            "source_family_id": source_family_id,
                        }
                    )

        if not samples:
            raise ValueError(f"No samples found in manifest: {self.manifest_path}")
        return samples

    def _expand_location_samples(self, samples):
        expanded = []
        for sample in samples:
            base_sample = dict(sample)
            base_sample.setdefault("augment_profile", "base")
            expanded.append(base_sample)
            if int(base_sample["distance_label"]) == DISTANCE_IGNORE_INDEX:
                continue
            for _ in range(self.location_aug_repeats):
                aug_sample = dict(base_sample)
                aug_sample["augment_profile"] = "location"
                expanded.append(aug_sample)
        return expanded

    def _load_signal(self, path):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return np.loadtxt(handle, delimiter=",", dtype=np.float32)

    def _resize_signal(self, signal):
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]
        if signal.ndim != 2:
            raise ValueError(
                f"Expected a 1D or 2D CSV signal in {self.manifest_path}, got shape {signal.shape}"
            )

        signal_tensor = torch.from_numpy(signal.astype(np.float32, copy=False))
        signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)
        if signal_tensor.shape[-2:] != (self.input_height, self.input_width):
            signal_tensor = F.interpolate(
                signal_tensor,
                size=(self.input_height, self.input_width),
                mode="bilinear",
                align_corners=False,
            )
        return signal_tensor.squeeze(0)

    def _resize_raw_signal(self, signal_1d):
        signal_np = np.asarray(signal_1d, dtype=np.float32).reshape(1, -1)
        signal_tensor = torch.from_numpy(signal_np).unsqueeze(0)
        if signal_tensor.shape[-1] != self.input_width:
            signal_tensor = F.interpolate(
                signal_tensor,
                size=self.input_width,
                mode="linear",
                align_corners=False,
            )
        return signal_tensor.squeeze(0)

    def _resize_map(self, map_2d, size):
        map_tensor = torch.as_tensor(np.asarray(map_2d, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        if map_tensor.shape[-2:] != (size, size):
            map_tensor = F.interpolate(map_tensor, size=(size, size), mode="bilinear", align_corners=False)
        return map_tensor.squeeze(0)

    def _adapt_spatial_signal(self, signal):
        signal_2d = self._ensure_2d_signal(signal).astype(np.float32, copy=False)
        if signal_2d.shape[0] == 1:
            return signal_2d[0]
        if self.spatial_adapter == "mean":
            return signal_2d.mean(axis=0)
        if self.spatial_adapter == "learned":
            # Sensitivity-only deterministic proxy: emphasize high-energy rows
            # without inventing extra spatial dimensions before the Raw encoder.
            row_energy = np.sqrt(np.mean(np.square(signal_2d), axis=1))
            weights = row_energy / max(float(row_energy.sum()), 1e-6)
            return np.sum(signal_2d * weights[:, np.newaxis], axis=0)
        center_index = signal_2d.shape[0] // 2
        return signal_2d[center_index]

    def _compute_stft_feature(self, signal):
        signal_2d = self._ensure_2d_signal(signal)
        group_features = []
        for start in range(0, signal_2d.shape[0], 3):
            group = signal_2d[start : start + 3]
            row_features = []
            for row in group:
                row_tensor = torch.from_numpy(np.asarray(row, dtype=np.float32))
                spec = torch.stft(
                    row_tensor,
                    n_fft=self.stft_n_fft,
                    hop_length=self.stft_hop_length,
                    win_length=self.stft_win_length,
                    window=self._stft_window,
                    return_complex=True,
                    center=True,
                )
                row_features.append(torch.log1p(torch.abs(spec)))
            if self.stf_spatial_fusion == "group3_mean":
                # One STF spatial unit represents the joint response of up to
                # three adjacent sensing channels: [C_g, F, T] -> [F, T].
                group_feature = torch.stack(row_features, dim=0).mean(dim=0)
            else:
                # Retained only for auditing the invalid channel-stacking runs.
                group_feature = torch.cat(row_features, dim=0)
            group_features.append(group_feature)
        fused = torch.cat(group_features, dim=0)
        return fused.unsqueeze(0)

    def _compute_gaf_feature(self, raw_signal):
        pooled = self._resize_raw_signal(raw_signal).squeeze(0)
        if pooled.numel() != self.gaf_size:
            pooled = F.adaptive_avg_pool1d(pooled.view(1, 1, -1), self.gaf_size).view(-1)
        min_val = pooled.amin()
        max_val = pooled.amax()
        scaled = 2.0 * (pooled - min_val) / (max_val - min_val + 1e-6) - 1.0
        scaled = scaled.clamp(-0.999999, 0.999999)
        phase = torch.acos(scaled)
        gaf = torch.cos(phase.unsqueeze(1) + phase.unsqueeze(0))
        return gaf.unsqueeze(0)

    def _normalize_view(self, tensor):
        if self.normalize == "none":
            return tensor
        if self.normalize == "sample":
            mean = tensor.mean()
            std = tensor.std(unbiased=False)
            if torch.isnan(std) or std.item() < 1e-6:
                std = torch.tensor(1.0, dtype=tensor.dtype, device=tensor.device)
            return (tensor - mean) / std
        raise ValueError(f"Unsupported normalize mode: {self.normalize}")

    def _build_multiview_payload(self, signal_sample):
        # raw: [1, 10000], stf/gaf: [1, 224, 224] by default.
        raw_1d = self._adapt_spatial_signal(signal_sample)
        raw_tensor = self._normalize_view(self._resize_raw_signal(raw_1d))
        stf_tensor = self._compute_stft_feature(signal_sample).squeeze(0).numpy()
        stf_tensor = self._normalize_view(self._resize_map(stf_tensor, self.stf_size))
        gaf_tensor = self._normalize_view(self._compute_gaf_feature(raw_1d))
        return {
            "raw": raw_tensor,
            "stf": stf_tensor,
            "gaf": gaf_tensor,
            "modality_mask": torch.ones(len(VIEW_NAMES), dtype=torch.float32),
        }

    def _normalize_signal(self, signal_tensor):
        if self.normalize == "none":
            return signal_tensor
        if self.normalize == "sample":
            mean = signal_tensor.mean()
            std = signal_tensor.std(unbiased=False)
            if torch.isnan(std) or std.item() < 1e-6:
                std = torch.tensor(1.0, dtype=signal_tensor.dtype)
            return (signal_tensor - mean) / std
        raise ValueError(f"Unsupported normalize mode: {self.normalize}")

    def _ensure_2d_signal(self, signal):
        if signal.ndim == 1:
            return signal[np.newaxis, :]
        return signal

    def _shift_signal(self, signal, shift_width):
        if shift_width <= 0:
            return signal

        width = signal.shape[-1]
        if width <= 1:
            return signal

        shift = np.random.randint(-shift_width, shift_width + 1)
        if shift == 0:
            return signal

        shifted = np.zeros_like(signal)
        if shift > 0:
            shifted[..., shift:] = signal[..., : width - shift]
        else:
            shifted[..., : width + shift] = signal[..., -shift:]
        return shifted

    def _mask_signal(self, signal, mask_width_limit):
        if mask_width_limit <= 0:
            return signal

        width = signal.shape[-1]
        if width <= 1:
            return signal

        mask_width = min(mask_width_limit, width)
        actual_width = np.random.randint(1, mask_width + 1)
        start = np.random.randint(0, width - actual_width + 1)
        signal[..., start : start + actual_width] = 0.0
        return signal

    def _drop_rows(self, signal, drop_rows):
        if signal.ndim != 2 or drop_rows <= 0 or signal.shape[0] <= 1:
            return signal

        drop_count = min(drop_rows, signal.shape[0] - 1)
        if drop_count <= 0:
            return signal

        rows = np.random.choice(signal.shape[0], size=drop_count, replace=False)
        signal[rows, :] = 0.0
        return signal

    def _resolve_augment_config(self, sample):
        profile = sample.get("augment_profile", "base")
        if profile == "location":
            return {
                "noise_std": self.location_aug_noise_std,
                "gain_std": self.location_aug_gain_std,
                "shift": self.location_aug_shift,
                "mask_width": self.location_aug_mask_width,
                "drop_rows": self.location_aug_drop_rows,
            }
        return {
            "noise_std": self.augment_noise_std,
            "gain_std": self.augment_gain_std,
            "shift": self.augment_shift,
            "mask_width": self.augment_mask_width,
            "drop_rows": self.augment_drop_rows,
        }

    def _should_augment_sample(self, sample):
        profile = str(sample.get("augment_profile", "base")).strip() or "base"
        return self.augment or profile != "base"

    def _augment_signal(self, signal, augment_config, enabled):
        if not enabled:
            return signal

        augmented = signal.astype(np.float32, copy=True)

        gain_std = augment_config["gain_std"]
        shift_width = augment_config["shift"]
        mask_width = augment_config["mask_width"]
        drop_rows = augment_config["drop_rows"]
        noise_std = augment_config["noise_std"]

        if gain_std > 0:
            gain = np.random.uniform(1.0 - gain_std, 1.0 + gain_std)
            augmented *= gain

        augmented = self._shift_signal(augmented, shift_width)
        augmented = self._mask_signal(augmented, mask_width)
        augmented = self._drop_rows(augmented, drop_rows)

        if noise_std > 0:
            signal_std = float(np.std(augmented))
            noise_scale = max(signal_std, 1e-6) * noise_std
            noise = np.random.normal(0.0, noise_scale, size=augmented.shape).astype(np.float32)
            augmented += noise

        return augmented

    def __getitem__(self, index):
        sample = self.samples[index]
        if self._cached_path != sample["path"]:
            self._cached_signal = self._load_signal(sample["path"])
            self._cached_path = sample["path"]

        signal = self._ensure_2d_signal(self._cached_signal)
        if sample["sample_mode"] == "file":
            signal_sample = signal
        elif sample["sample_mode"] == "group3":
            signal_sample = signal[sample["row_start"] : sample["row_end"]]
        else:
            row_signal = signal[sample["row_index"]]
            signal_sample = row_signal

        augment_config = self._resolve_augment_config(sample)
        signal_sample = self._augment_signal(signal_sample, augment_config, self._should_augment_sample(sample))
        if self.return_multiview:
            signal_tensor = self._build_multiview_payload(signal_sample)
        else:
            raw_tensor = self._resize_signal(signal_sample)
            feature_tensors = [raw_tensor]
            if self.use_stft_aux:
                stft_tensor = self._resize_signal(self._compute_stft_feature(signal_sample).squeeze(0).numpy())
                feature_tensors.append(stft_tensor)
            signal_tensor = torch.cat(feature_tensors, dim=0)
            signal_tensor = self._normalize_signal(signal_tensor)

        distance_valid = int(sample["distance_label"]) != DISTANCE_IGNORE_INDEX
        labels = {
            "event_type": torch.tensor(sample["event_label"], dtype=torch.long),
            "distance_cls": torch.tensor(sample["distance_label"], dtype=torch.long),
            "task_mask": torch.tensor([1.0, 1.0 if distance_valid else 0.0], dtype=torch.float32),
        }
        return signal_tensor, labels

    def __len__(self):
        return len(self.samples)


def csv_dataloader(
    dataset_path,
    batch_size,
    event_classes,
    distance_classes,
    input_height,
    input_width,
    sample_level="manifest",
    normalize="sample",
    num_workers=0,
    augment=False,
    augment_noise_std=0.0,
    augment_gain_std=0.0,
    augment_shift=0,
    augment_mask_width=0,
    augment_drop_rows=0,
    location_aug_repeats=0,
    location_aug_noise_std=0.0,
    location_aug_gain_std=0.0,
    location_aug_shift=0,
    location_aug_mask_width=0,
    location_aug_drop_rows=0,
    return_multiview=False,
    spatial_adapter="center",
    stf_spatial_fusion="group3_mean",
    stf_size=224,
    gaf_size=224,
    train_sampler="none",
    use_stft_aux=False,
    stft_n_fft=256,
    stft_hop_length=128,
    stft_win_length=256,
):
    dataset_root = Path(dataset_path)
    event_to_idx = {label: idx for idx, label in enumerate(event_classes)}
    distance_to_idx = {label: idx for idx, label in enumerate(distance_classes)}

    data_loader = {}
    iter_data_loader = {}
    for split in ["train", "val", "test"]:
        manifest_path = dataset_root / f"{split}.csv"
        shuffle = split == "train"
        dataset = MultiTaskCSVDataset(
            manifest_path=manifest_path,
            event_to_idx=event_to_idx,
            distance_to_idx=distance_to_idx,
            input_height=input_height,
            input_width=input_width,
            sample_level=sample_level,
            normalize=normalize,
            augment=augment and split == "train",
            augment_noise_std=augment_noise_std,
            augment_gain_std=augment_gain_std,
            augment_shift=augment_shift,
            augment_mask_width=augment_mask_width,
            augment_drop_rows=augment_drop_rows,
            location_aug_repeats=location_aug_repeats,
            location_aug_noise_std=location_aug_noise_std,
            location_aug_gain_std=location_aug_gain_std,
            location_aug_shift=location_aug_shift,
            location_aug_mask_width=location_aug_mask_width,
            location_aug_drop_rows=location_aug_drop_rows,
            return_multiview=return_multiview,
            spatial_adapter=spatial_adapter,
            stf_spatial_fusion=stf_spatial_fusion,
            stf_size=stf_size,
            gaf_size=gaf_size,
            use_stft_aux=use_stft_aux,
            stft_n_fft=stft_n_fft,
            stft_hop_length=stft_hop_length,
            stft_win_length=stft_win_length,
        )
        sampler = None
        if split == "train" and train_sampler != "none":
            if train_sampler == "event_balanced":
                keys = [int(sample["event_label"]) for sample in dataset.samples]
            elif train_sampler == "distance_balanced":
                keys = [
                    int(sample["distance_label"]) if int(sample["distance_label"]) >= 0 else "unlabeled"
                    for sample in dataset.samples
                ]
            elif train_sampler == "joint_balanced":
                keys = [
                    (
                        int(sample["event_label"]),
                        int(sample["distance_label"]) if int(sample["distance_label"]) >= 0 else "unlabeled",
                    )
                    for sample in dataset.samples
                ]
            elif train_sampler == "event_distance_balanced":
                keys = []
                for sample in dataset.samples:
                    event_key = int(sample["event_label"])
                    distance_key = int(sample["distance_label"])
                    if distance_key >= 0:
                        keys.append(("excavator", distance_key))
                    else:
                        keys.append((event_key, "event"))
            else:
                raise ValueError(f"Unsupported train_sampler: {train_sampler}")

            counts = Counter(keys)
            weights = torch.as_tensor([1.0 / counts[key] for key in keys], dtype=torch.double)
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            shuffle = False
        data_loader[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=shuffle and len(dataset) > 1,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        iter_data_loader[split] = iter(data_loader[split])
    return data_loader, iter_data_loader


def _write_manifest(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "event_label", "distance_label", "sample_mode"])
        writer.writeheader()
        writer.writerows(rows)


def _split_group(items, train_ratio, val_ratio):
    total = len(items)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    if total >= 3:
        if train_count == 0:
            train_count = 1
        if val_count == 0:
            val_count = 1
        if train_count + val_count >= total:
            val_count = max(1, total - train_count - 1)
    test_count = total - train_count - val_count
    return {
        "train": items[:train_count],
        "val": items[train_count: train_count + val_count],
        "test": items[train_count + val_count: train_count + val_count + test_count],
    }


def ensure_mtl43_manifests(dataset_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=0, overwrite=False):
    dataset_root = Path(dataset_path)
    manifest_paths = [dataset_root / f"{split}.csv" for split in ("train", "val", "test")]
    if not overwrite and all(path.is_file() for path in manifest_paths):
        return False

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    walking_dir = dataset_root / "walking"
    if not walking_dir.is_dir():
        walking_dir = dataset_root.parent / "walking"
    driving_dir = dataset_root / "driving"
    background_dir = dataset_root / "background"
    excavator_dir = dataset_root / "excavator"
    expected_dirs = [walking_dir, driving_dir, background_dir, excavator_dir]
    if not all(path.exists() for path in expected_dirs):
        raise FileNotFoundError(
            "MTL43 dataset is expected to contain walking, driving, background, and excavator folders."
        )

    items = []
    for path in sorted(walking_dir.glob("*.csv")):
        items.append(
            {
                "path": str(path),
                "event_label": "walking",
                "distance_label": "",
                "sample_mode": "row",
            }
        )
    for path in sorted(driving_dir.glob("*.csv")):
        items.append(
            {
                "path": str(path),
                "event_label": "driving",
                "distance_label": "",
                "sample_mode": "row",
            }
        )
    for path in sorted(background_dir.glob("*.csv")):
        items.append(
            {
                "path": str(path),
                "event_label": "background",
                "distance_label": "",
                "sample_mode": "file",
            }
        )
    for distance_dir in sorted(path for path in excavator_dir.iterdir() if path.is_dir()):
        distance_label = canonicalize_distance_label(distance_dir.name)
        for path in sorted(distance_dir.glob("*.csv")):
            items.append(
                {
                    "path": str(path),
                    "event_label": "excavator",
                    "distance_label": distance_label,
                    "sample_mode": "file",
                }
            )

    if not items:
        raise ValueError(f"No CSV files found under {dataset_root}")

    rng = random.Random(seed)
    grouped = defaultdict(list)
    for item in items:
        group_key = (item["event_label"], item["distance_label"] or "_")
        grouped[group_key].append(item)

    split_rows = {"train": [], "val": [], "test": []}
    for group_items in grouped.values():
        group_copy = list(group_items)
        rng.shuffle(group_copy)
        group_split = _split_group(group_copy, train_ratio=train_ratio, val_ratio=val_ratio)
        for split_name, rows in group_split.items():
            split_rows[split_name].extend(rows)

    for split_name in split_rows:
        rng.shuffle(split_rows[split_name])
        _write_manifest(dataset_root / f"{split_name}.csv", split_rows[split_name])

    return True
