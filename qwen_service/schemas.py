from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class BannerAnnotateRequest(BaseModel):
    banner_image_path: str
    candidate_bundle: dict[str, Any]
    heuristic_bundle: Optional[dict[str, Any]] = None


class CandidateAnnotateRequest(BaseModel):
    banner_image_path: str
    candidate: dict[str, Any]
    heuristic_annotation: Optional[dict[str, Any]] = None
    context_padding_ratio: float = Field(default=0.08, ge=0.0, le=1.0)


class GroupAnnotateRequest(BaseModel):
    banner_image_path: str
    candidate: dict[str, Any]
    heuristic_annotation: Optional[dict[str, Any]] = None
    context_padding_ratio: float = Field(default=0.08, ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_path: str
