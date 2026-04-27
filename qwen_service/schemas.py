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
    element_image_paths: list[str] = Field(default_factory=list)
    element_atlas_image_path: Optional[str] = Field(
        default=None,
        description="Optional packed leaf atlas PNG; model sees it as image 2 after the banner.",
    )


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
    element_image_paths: list[str] = Field(
        default_factory=list,
        description="Absolute paths to per-leaf PNG crops (same design); after banner and optional atlas.",
    )
    element_atlas_image_path: Optional[str] = Field(
        default=None,
        description="Packed element atlas PNG path; when set, loaded as image 2 (after banner, before crops).",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_path: str
