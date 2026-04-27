from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RunCreateResponse(BaseModel):
    run_id: str
    status: str
    message: str


class RunSummaryResponse(BaseModel):
    run_id: str
    status: str
    created_at: str
    updated_at: str
    input_files: dict[str, Optional[str]]
    output_files: dict[str, Optional[str]]
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    backend: str
    qwen_service_ok: bool
    qwen_service_base_url: str


class RunListItem(BaseModel):
    run_id: str
    status: str
    created_at: str
    updated_at: str
    error: Optional[str] = None


class RunListResponse(BaseModel):
    runs: list[RunListItem]


class RunResultEnvelope(BaseModel):
    run_id: str
    status: str
    data: dict[str, Any]


class ErrorResponse(BaseModel):
    detail: str


class ConvertElementImageRef(BaseModel):
    """IMAGE fill/stroke ref; image_hash is a Figma correlation id only (no bitmap in request)."""

    model_config = ConfigDict(extra="allow")

    path: Optional[str] = None
    node_id: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = None
    image_hash: Optional[str] = None
    fill_role: Optional[str] = None


MAX_ATLAS_REGIONS = 512


class ConvertElementAtlasRegion(BaseModel):
    """One packed leaf cell in element_atlas_png_base64 (pixel coords in atlas space)."""

    model_config = ConfigDict(extra="allow")

    path: Optional[str] = None
    node_id: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = None
    atlas_x: int = Field(ge=0)
    atlas_y: int = Field(ge=0)
    atlas_width: int = Field(gt=0)
    atlas_height: int = Field(gt=0)


class ConvertRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    banner_png_base64: str = Field(min_length=1)
    raw_json: dict[str, Any]
    target_width: int = Field(gt=0)
    target_height: int = Field(gt=0)
    mode: Literal["apply_to_clone_fast", "apply_to_clone_vlm", "full_layout_debug"] = "apply_to_clone_fast"
    use_qwen: bool = False
    qwen_mode: Optional[str] = Field(
        default=None,
        description="scene_only, single_pass, per_candidate legacy, or off. Ignored when use_qwen is false.",
    )
    brand_family: Optional[str] = None
    language: Optional[str] = None
    category: Optional[str] = None
    element_image_refs: Optional[list[ConvertElementImageRef]] = Field(
        default=None,
        description="IMAGE paint metadata (path, node_id, image_hash, fill_role, …).",
    )
    element_atlas_png_base64: str = Field(
        default="",
        description=(
            "Packed leaf atlas (PNG). Empty when there are no qualifying leaves. "
            "When non-empty, crops use element_atlas_regions if provided, else raw_json atlas_region."
        ),
    )
    element_atlas_regions: list[ConvertElementAtlasRegion] = Field(
        default_factory=list,
        description=(
            "Atlas rectangles (atlas_x/y/width/height). If empty while atlas PNG is set, "
            "the server walks raw_json for atlas_region on nodes."
        ),
    )
    element_atlas_regions_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional; must match len(element_atlas_regions) when set.",
    )
    element_image_refs_count: Optional[int] = Field(default=None, ge=0)

    @field_validator("element_atlas_png_base64", mode="before")
    @classmethod
    def element_atlas_png_base64_coerce(cls, v: Any) -> str:
        if v is None:
            return ""
        return v if isinstance(v, str) else str(v)

    @field_validator("element_atlas_regions", mode="before")
    @classmethod
    def element_atlas_regions_coerce(cls, v: Any) -> list[Any]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise TypeError("element_atlas_regions must be a list or null.")
        return v

    @field_validator("raw_json")
    @classmethod
    def raw_json_nonempty(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("raw_json must be a non-empty JSON object.")
        return v

    @field_validator("element_atlas_regions")
    @classmethod
    def limit_element_atlas_regions(
        cls, v: list[ConvertElementAtlasRegion]
    ) -> list[ConvertElementAtlasRegion]:
        if len(v) > MAX_ATLAS_REGIONS:
            raise ValueError(
                f"element_atlas_regions supports at most {MAX_ATLAS_REGIONS} entries (got {len(v)})."
            )
        return v

    @model_validator(mode="after")
    def element_atlas_consistency(self) -> ConvertRequest:
        atlas_b64 = (self.element_atlas_png_base64 or "").strip()
        regions = self.element_atlas_regions
        has_atlas = bool(atlas_b64)
        if regions and not has_atlas:
            raise ValueError(
                "element_atlas_png_base64 must be non-empty when element_atlas_regions is non-empty."
            )
        if self.element_atlas_regions_count is not None and self.element_atlas_regions_count != len(
            regions
        ):
            raise ValueError(
                f"element_atlas_regions_count ({self.element_atlas_regions_count}) must equal "
                f"len(element_atlas_regions) ({len(regions)})."
            )
        return self


class LayoutBounds(BaseModel):
    x: float
    y: float
    width: float
    height: float


class LayoutUpdateItem(BaseModel):
    source_figma_id: str
    path: Optional[str] = None
    name: Optional[str] = None
    role: str
    semantic_name: str
    parent_semantic_name: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""
    bounds: LayoutBounds


class ConvertFrameSpec(BaseModel):
    width: int
    height: int


class ConvertDebug(BaseModel):
    semantic_graph_path: str
    validation_report_path: Optional[str] = None
    qwen_call_count: int = 0
    qwen_mode: Optional[str] = None
    stage_timings: dict[str, float] = Field(default_factory=dict)
    nodes_annotated: int = 0
    nodes_left_unnamed: int = 0
    validation_warnings: list[str] = Field(default_factory=list)
    low_confidence_examples: list[dict[str, Any]] = Field(default_factory=list)


class ConvertSemanticElement(BaseModel):
    id: str
    figma_node_id: str
    path: Optional[str] = None
    role: str
    semantic_name: str
    parent_semantic_name: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""


class ConvertSemanticGroup(BaseModel):
    id: str
    role: str
    semantic_name: str
    children: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""


class ConvertSemanticSummary(BaseModel):
    elements: list[ConvertSemanticElement] = Field(default_factory=list)
    groups: list[ConvertSemanticGroup] = Field(default_factory=list)


class ConvertResponse(BaseModel):
    run_id: str
    mode: Literal["apply_to_clone_fast", "apply_to_clone_vlm", "full_layout_debug"]
    frame: ConvertFrameSpec
    updates: list[LayoutUpdateItem]
    semantic: ConvertSemanticSummary
    debug: ConvertDebug