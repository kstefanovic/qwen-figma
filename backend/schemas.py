from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


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