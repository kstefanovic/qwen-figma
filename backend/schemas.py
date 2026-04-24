from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


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


class ConvertRequest(BaseModel):
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

    @field_validator("raw_json")
    @classmethod
    def raw_json_nonempty(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("raw_json must be a non-empty JSON object.")
        return v


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