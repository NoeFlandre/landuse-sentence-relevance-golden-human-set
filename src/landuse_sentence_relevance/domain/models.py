from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Source(StrEnum):
    WIKIPEDIA = "wikipedia"
    WEBSITE = "website"


class Label(StrEnum):
    YES = "yes"
    NO = "no"


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    sentence: str
    source: Source
    source_record_id: str
    source_field: str
    h3_cell: str
    h3_resolution: int
    latitude: float
    longitude: float
    place_name: str | None = None
    region: str | None = None
    source_url: str | None = None
    language: str = "en"

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.sentence, "sentence")
        _require_text(self.source_record_id, "source_record_id")
        _require_text(self.source_field, "source_field")
        _require_text(self.h3_cell, "h3_cell")
        if self.h3_resolution != 3:
            raise ValueError("h3_resolution must be 3 for the project sampling grid")
        if not -90 <= self.latitude <= 90:
            raise ValueError("latitude must be between -90 and 90")
        if not -180 <= self.longitude <= 180:
            raise ValueError("longitude must be between -180 and 180")
        if self.language != "en":
            raise ValueError("candidate language must be English")

    @property
    def stratum(self) -> tuple[Source, str]:
        return self.source, self.h3_cell

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "sentence": self.sentence,
            "source": self.source.value,
            "source_record_id": self.source_record_id,
            "source_field": self.source_field,
            "h3_cell": self.h3_cell,
            "h3_resolution": self.h3_resolution,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "place_name": self.place_name,
            "region": self.region,
            "source_url": self.source_url,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Candidate:
        return cls(
            candidate_id=str(row["candidate_id"]),
            sentence=str(row["sentence"]),
            source=Source(row["source"]),
            source_record_id=str(row["source_record_id"]),
            source_field=str(row["source_field"]),
            h3_cell=str(row["h3_cell"]),
            h3_resolution=int(row["h3_resolution"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            place_name=row.get("place_name"),
            region=row.get("region"),
            source_url=row.get("source_url"),
            language=str(row.get("language", "en")),
        )


@dataclass(frozen=True, slots=True)
class Annotation:
    candidate: Candidate
    label: Label

    def to_dict(self) -> dict[str, Any]:
        return {**self.candidate.to_dict(), "label": self.label.value}

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Annotation:
        return cls(candidate=Candidate.from_dict(row), label=Label(row["label"]))
