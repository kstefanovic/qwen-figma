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


class BrandContextAnnotateRequest(BaseModel):
    banner_image_path: str
    candidate_bundle: Optional[dict[str, Any]] = None
    heuristic_bundle: Optional[dict[str, Any]] = None


class SemanticStructureAnnotateRequest(BaseModel):
    """Single-pass VLM: full banner + compact Figma summary (nodes + candidates + heuristic hints)."""

    banner_image_path: str
    figma_summary: dict[str, Any]


class SceneAnnotateRequest(BaseModel):
    """
    Single-pass scene annotation payload:
    compact metadata + Figma-derived elements/groups/heuristics + optional banner image path.
    """

    banner_metadata: dict[str, Any]
    elements: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    heuristic_roles: dict[str, Any] = Field(default_factory=dict)
    figma_summary: Optional[dict[str, Any]] = None
    banner_image_path: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_path: str
