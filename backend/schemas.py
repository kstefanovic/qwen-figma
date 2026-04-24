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
    use_qwen: bool = True
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
    bounds: LayoutBounds


class ConvertFrameSpec(BaseModel):
    width: int
    height: int


class ConvertDebug(BaseModel):
    semantic_graph_path: str
    validation_report_path: Optional[str] = None


class ConvertResponse(BaseModel):
    run_id: str
    mode: Literal["apply_to_clone"] = "apply_to_clone"
    frame: ConvertFrameSpec
    updates: list[LayoutUpdateItem]
    debug: ConvertDebug