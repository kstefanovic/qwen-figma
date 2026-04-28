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


class NormalizedBbox(BaseModel):
    """BBox relative to full banner: 0..1 in x, y, width, height."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class TextZoneGroupItem(BaseModel):
    role: Literal["brand_group", "headline_group", "legal_text"]
    bbox: NormalizedBbox
    confidence: float = 0.0
    reason: str = ""


class TextZoneVisual(BaseModel):
    groups: list[TextZoneGroupItem] = Field(default_factory=list)


class AnalyzeTextZoneVisualDebug(BaseModel):
    qwen_call_count: int = 1
    elapsed_seconds: float = 0.0
    validation_warnings: list[str] = Field(default_factory=list)


class AnalyzeTextZoneVisualResponse(BaseModel):
    run_id: str
    mode: Literal["text_zone_visual"] = "text_zone_visual"
    orientation: str
    zone_type: str
    confidence: float = 0.0
    reason: str = ""
    text_zone: TextZoneVisual
    debug: AnalyzeTextZoneVisualDebug

    @field_validator("orientation")
    @classmethod
    def _orientation_allowed_v2(cls, v: str) -> str:
        allowed = {"landscape", "portrait", "wide"}
        s = (v or "").strip()
        return s if s in allowed else "landscape"
