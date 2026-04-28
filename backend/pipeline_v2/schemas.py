from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ClassifyZonePluginRequest(BaseModel):
    """Figma plugin: same transport as ``/api/convert`` (strict PNG base64, no multipart)."""

    banner_png_base64: str


class ClassifyZoneDebug(BaseModel):
    qwen_call_count: int = 1
    elapsed_seconds: float = 0.0


class ClassifyZoneResponse(BaseModel):
    run_id: str
    mode: Literal["zone_classification"] = "zone_classification"
    zone_type: str
    orientation: str = Field(
        ...,
        description="landscape | wide | portrait",
    )
    confidence: float = 0.0
    reason: str = ""
    debug: ClassifyZoneDebug

    @field_validator("orientation")
    @classmethod
    def _orientation_allowed(cls, v: str) -> str:
        allowed = {"landscape", "portrait", "wide"}
        s = (v or "").strip()
        return s if s in allowed else "landscape"
