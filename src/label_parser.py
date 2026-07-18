from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ZONE_PATTERN = re.compile(r"^[A-Z]\d{1,3}$")
DISTANCE_PATTERN = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)m(?![a-z])", re.IGNORECASE)
SAMPLING_K_PATTERN = re.compile(r"(?<![a-z])(\d+(?:\.\d+)?)k(?:hz)?(?![a-z])", re.IGNORECASE)
SAMPLING_HZ_PATTERN = re.compile(r"(?<![a-z])(\d+(?:\.\d+)?)hz(?![a-z])", re.IGNORECASE)


@dataclass
class ParseResult:
    event_type: str
    fine_event: str
    distance_label: str
    distance_value_m: float
    soil_condition: str
    sampling_rate_hz: float
    is_background: bool
    has_distance_label: bool
    source_batch_id: str
    parse_status: str
    parse_warning: List[str] = field(default_factory=list)
    current_zone: Optional[str] = None
    event_zone: Optional[str] = None
    matched_tokens: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def parse_warning_text(self) -> str:
        return ";".join(self.parse_warning)


class LabelParser:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.batch_map = config["batch_map"]
        self.aliases = config["aliases"]
        self.distance_default = config["distance"]["default_label"]

    def parse_source_batch_id(self, path: Path) -> Tuple[str, List[str]]:
        warnings: List[str] = []
        path_tokens = [part for part in path.parts if part]
        for token in path_tokens:
            if token in self.batch_map:
                return self.batch_map[token], warnings
        warnings.append("source_batch_unmapped")
        return "BUNK", warnings

    def parse_path(self, path: Path) -> ParseResult:
        source_batch_id, warnings = self.parse_source_batch_id(path)
        stem = path.stem
        context_parts = [stem]
        context_parts.extend(part for part in path.parts if part and part != path.drive)
        tokens = self._tokenize(" ".join(context_parts))
        normalized = " ".join(tokens)
        zone_tokens = [token.upper() for token in tokens if ZONE_PATTERN.match(token.upper())]
        current_zone = zone_tokens[0] if zone_tokens else None
        event_zone = zone_tokens[1] if len(zone_tokens) > 1 else None

        sampling_rate_hz = self._parse_sampling_rate(context_parts)
        if math.isnan(sampling_rate_hz):
            warnings.append("sampling_rate_unknown")

        soil_condition, soil_matches, soil_warnings = self._parse_soil_condition(tokens)
        warnings.extend(soil_warnings)

        distance_label, distance_value_m, distance_warnings = self._parse_distance_label(stem)
        warnings.extend(distance_warnings)
        has_distance_label = distance_label != self.distance_default

        is_background = self._has_alias(normalized, self.aliases["background"])
        if current_zone and event_zone and current_zone != event_zone:
            is_background = True
            warnings.append("zone_mismatch_background")

        event_type, event_matches = self._parse_event_type(normalized, path, is_background)
        if "unresolved" in event_matches:
            warnings.append("event_type_unresolved")
        if event_type == "background_noise" and has_distance_label:
            # Keep explicit distance while treating the sample as background.
            pass
        elif event_type in {"background_noise", "pipeline_leakage"} and not has_distance_label:
            distance_label = self.distance_default
            distance_value_m = math.nan
            has_distance_label = False

        fine_event = self._parse_fine_event(normalized, path, event_type, is_background)
        parse_status = "ok"
        if warnings:
            parse_status = "warning"
        if event_type == "background_noise" and "event_type_unresolved" in warnings:
            parse_status = "warning"

        matched_tokens = {
            "soil": soil_matches,
            "event": event_matches,
        }
        return ParseResult(
            event_type=event_type,
            fine_event=fine_event,
            distance_label=distance_label,
            distance_value_m=distance_value_m,
            soil_condition=soil_condition,
            sampling_rate_hz=sampling_rate_hz,
            is_background=is_background,
            has_distance_label=has_distance_label,
            source_batch_id=source_batch_id,
            parse_status=parse_status,
            parse_warning=self._deduplicate_warnings(warnings),
            current_zone=current_zone,
            event_zone=event_zone,
            matched_tokens=matched_tokens,
        )

    def _tokenize(self, value: str) -> List[str]:
        rough_tokens = re.split(r"[\s_\-]+", value)
        tokens: List[str] = []
        for token in rough_tokens:
            cleaned = token.strip()
            if not cleaned:
                continue
            tokens.append(cleaned.lower())
        return tokens

    def _parse_sampling_rate(self, values: Sequence[str]) -> float:
        for value in values:
            lowered = value.lower()
            for token in re.split(r"[\s_\-]+", lowered):
                if not token:
                    continue
                match_k = re.fullmatch(r"(\d+(?:\.\d+)?)k(?:hz)?", token)
                if match_k:
                    return float(match_k.group(1)) * 1000.0
                match_hz = re.fullmatch(r"(\d+(?:\.\d+)?)hz", token)
                if match_hz:
                    return float(match_hz.group(1))
            match_k = SAMPLING_K_PATTERN.search(lowered)
            if match_k and not match_k.group(0).endswith("km"):
                return float(match_k.group(1)) * 1000.0
            match_hz = SAMPLING_HZ_PATTERN.search(lowered)
            if match_hz:
                return float(match_hz.group(1))
        return math.nan

    def _parse_soil_condition(self, tokens: Sequence[str]) -> Tuple[str, List[str], List[str]]:
        soil_alias_map = {
            "land": "land",
            "sand": "sand",
            "shizi": "stone",
            "stone": "stone",
        }
        matches: List[str] = []
        distinct: List[str] = []
        warnings: List[str] = []
        for token in tokens:
            if token in soil_alias_map:
                resolved = soil_alias_map[token]
                matches.append(token)
                if resolved not in distinct:
                    distinct.append(resolved)
        if not distinct:
            warnings.append("soil_condition_unknown")
            return "unknown", matches, warnings
        if len(distinct) > 1:
            warnings.append("multiple_soil_tokens")
            return "unknown", matches, warnings
        return distinct[0], matches, warnings

    def _parse_distance_label(self, value: str) -> Tuple[str, float, List[str]]:
        matches = [float(match.group(1)) for match in DISTANCE_PATTERN.finditer(value.lower())]
        if not matches:
            return self.distance_default, math.nan, []
        if len(matches) > 1:
            warning = ["multiple_distance_tokens"]
        else:
            warning = []
        distance_value = matches[-1]
        if float(distance_value).is_integer():
            label = f"{int(distance_value)}m"
        else:
            label = f"{distance_value:g}m"
        return label, float(distance_value), warning

    def _parse_event_type(self, normalized: str, path: Path, is_background: bool) -> Tuple[str, List[str]]:
        matches: List[str] = []
        if is_background:
            matches.append("background")
            return "background_noise", matches
        if self._has_alias(normalized, self.aliases["leakage"]):
            matches.append("leak")
            return "pipeline_leakage", matches
        if self._has_alias(normalized, self.aliases["vehicle"]):
            matches.append("vehicle")
            return "vehicle_passing", matches
        if self._has_alias(normalized, self.aliases["manual"]):
            matches.append("manual")
            return "manual_work", matches
        if self._has_alias(normalized, self.aliases["mechanical"]):
            matches.append("mechanical")
            return "mechanical_excavation", matches
        if "human activities" in str(path).lower():
            matches.append("manual_dir")
            return "manual_work", matches
        if "excavator" in str(path).lower():
            matches.append("excavator_dir")
            return "mechanical_excavation", matches
        matches.append("unresolved")
        return "background_noise", matches

    def _parse_fine_event(
        self,
        normalized: str,
        path: Path,
        event_type: str,
        is_background: bool,
    ) -> str:
        if is_background or event_type == "pipeline_leakage":
            return "N/A"

        fine_aliases = self.aliases["fine_event"]
        if self._has_alias(normalized, fine_aliases["idle"]):
            if event_type == "vehicle_passing":
                return "vehicle_idle"
            return "excavator_idle"
        if self._has_alias(normalized, fine_aliases["parallel"]):
            return "parallel_driving"
        if self._has_alias(normalized, fine_aliases["crossing"]):
            return "crossing"
        if self._has_alias(normalized, fine_aliases["walking"]):
            return "manual_walking"
        if self._has_alias(normalized, fine_aliases["knocking"]):
            return "knocking"
        if self._has_alias(normalized, fine_aliases["digging"]):
            if event_type == "manual_work" or "human activities" in str(path).lower():
                return "manual_digging"
            return "digging"
        if self._has_alias(normalized, fine_aliases["vehicle_passing"]):
            return "vehicle_passing"
        return "N/A"

    def _has_alias(self, normalized: str, aliases: Iterable[str]) -> bool:
        lowered = normalized.lower()
        return any(alias.lower() in lowered for alias in aliases)

    def _deduplicate_warnings(self, warnings: Sequence[str]) -> List[str]:
        ordered: List[str] = []
        for warning in warnings:
            if warning == "source_batch_unmapped":
                continue
            if warning not in ordered:
                ordered.append(warning)
        return ordered
