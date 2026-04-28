from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

TextZoneChildRole = Literal[
    "logo",
    "logo_back",
    "logo_fore",
    "brand_name",
    "brand_name_first",
    "brand_name_second",
    "headline",
    "subheadline",
    "subheadline_delivery_time",
    "subheadline_weight",
    "product_name",
    "subheadline_discount",
    "legal_text",
    "age_badge",
]


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


class TextZoneChildItem(BaseModel):
    """Visual sub-part inside a text-zone group (brand, headline, legal, or age_badge)."""

    role: TextZoneChildRole
    text: str = ""
    bbox: NormalizedBbox
    confidence: float = 0.0
    reason: str = ""


class TextZoneGroupItem(BaseModel):
    role: Literal["brand_group", "headline_group", "age_badge_group", "legal_text_group"]
    bbox: NormalizedBbox
    confidence: float = 0.0
    reason: str = ""
    children: list[TextZoneChildItem] = Field(default_factory=list)


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
