from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ZoneAlternativeItem(BaseModel):
    zone_type: str
    confidence: float = 0.0


class ClassifyZoneDebug(BaseModel):
    qwen_call_count: int = 1
    elapsed_seconds: float = 0.0


class ClassifyZoneResponse(BaseModel):
    run_id: str
    mode: Literal["zone_classification"] = "zone_classification"
    zone_type: str
    orientation: str = Field(
        ...,
        description="landscape | portrait | square | wide | unknown (from aspect ratio rules)",
    )
    confidence: float = 0.0
    reason: str = ""
    alternatives: list[ZoneAlternativeItem] = Field(default_factory=list)
    debug: ClassifyZoneDebug

    @field_validator("orientation")
    @classmethod
    def _orientation_allowed(cls, v: str) -> str:
        allowed = {"landscape", "portrait", "square", "wide", "unknown"}
        s = (v or "").strip()
        return s if s in allowed else "unknown"
