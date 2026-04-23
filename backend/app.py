from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.runner import PipelineRunner
from backend.schemas import (
    HealthResponse,
    RunCreateResponse,
    RunListItem,
    RunListResponse,
    RunResultEnvelope,
    RunSummaryResponse,
)
from backend.storage import RunStorage


RUNS_DIR = Path("runs")
QWEN_BASE_URL = "http://127.0.0.1:10196"

app = FastAPI(title="Banner Pipeline Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = RunStorage(RUNS_DIR)
runner = PipelineRunner(storage=storage, qwen_base_url=QWEN_BASE_URL)


def _assert_run_exists(run_id: str) -> None:
    if not storage.exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")


def _safe_filename(name: str) -> str:
    return Path(name).name.replace(" ", "_")


def _read_json_or_404(path_str: Optional[str], label: str) -> dict[str, Any]:
    if not path_str:
        raise HTTPException(status_code=404, detail=f"{label} path not found in run metadata.")
    path = Path(path_str)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{label} file does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse {label}: {e}")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    qwen_ok = False
    try:
        resp = requests.get(f"{QWEN_BASE_URL}/health", timeout=10)
        qwen_ok = resp.ok
    except Exception:
        qwen_ok = False

    return HealthResponse(
        status="ok",
        backend="banner-pipeline-backend",
        qwen_service_ok=qwen_ok,
        qwen_service_base_url=QWEN_BASE_URL,
    )


@app.post("/api/run", response_model=RunCreateResponse)
async def create_run(
    banner_image: UploadFile = File(...),
    raw_json: UploadFile = File(...),
    brand_family: str = Form("unknown_brand"),
    language: str = Form("unknown"),
    category: str = Form("unknown"),
    use_qwen: bool = Form(True),
) -> RunCreateResponse:
    if not banner_image.filename:
        raise HTTPException(status_code=400, detail="banner_image filename is missing.")
    if not raw_json.filename:
        raise HTTPException(status_code=400, detail="raw_json filename is missing.")

    if not banner_image.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        raise HTTPException(status_code=400, detail="banner_image must be png/jpg/jpeg/webp.")

    if not raw_json.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="raw_json must be a .json file.")

    run_id = storage.create_run()

    banner_bytes = await banner_image.read()
    raw_json_bytes = await raw_json.read()

    banner_path = storage.save_upload_bytes(
        run_id,
        _safe_filename(banner_image.filename),
        banner_bytes,
    )
    raw_json_path = storage.save_upload_bytes(
        run_id,
        _safe_filename(raw_json.filename),
        raw_json_bytes,
    )

    storage.update_meta(
        run_id,
        input_files={
            "banner_image": banner_path,
            "raw_json": raw_json_path,
        },
        metadata={
            "brand_family": brand_family,
            "language": language,
            "category": category,
            "use_qwen": use_qwen,
        },
    )

    try:
        runner.run(
            run_id=run_id,
            raw_json_path=raw_json_path,
            banner_image_path=banner_path,
            brand_family=brand_family,
            language=language,
            category=category,
            use_qwen=use_qwen,
        )
    except Exception as e:
        meta = storage.read_meta(run_id)
        return RunCreateResponse(
            run_id=run_id,
            status=meta["status"],
            message=f"Run failed: {e}",
        )

    meta = storage.read_meta(run_id)
    return RunCreateResponse(
        run_id=run_id,
        status=meta["status"],
        message="Run completed.",
    )


@app.get("/api/runs", response_model=RunListResponse)
def list_runs(limit: int = 100) -> RunListResponse:
    runs = storage.list_runs(limit=limit)
    items = [
        RunListItem(
            run_id=meta["run_id"],
            status=meta["status"],
            created_at=meta["created_at"],
            updated_at=meta["updated_at"],
            error=meta.get("error"),
        )
        for meta in runs
    ]
    return RunListResponse(runs=items)


@app.get("/api/run/{run_id}", response_model=RunSummaryResponse)
def get_run_summary(run_id: str) -> RunSummaryResponse:
    _assert_run_exists(run_id)
    meta = storage.read_meta(run_id)
    return RunSummaryResponse(**meta)


@app.get("/api/run/{run_id}/semantic-graph", response_model=RunResultEnvelope)
def get_semantic_graph(run_id: str) -> RunResultEnvelope:
    _assert_run_exists(run_id)
    meta = storage.read_meta(run_id)
    data = _read_json_or_404(meta["output_files"].get("semantic_graph"), "semantic_graph")
    return RunResultEnvelope(run_id=run_id, status=meta["status"], data=data)


@app.get("/api/run/{run_id}/validation-report", response_model=RunResultEnvelope)
def get_validation_report(run_id: str) -> RunResultEnvelope:
    _assert_run_exists(run_id)
    meta = storage.read_meta(run_id)
    data = _read_json_or_404(meta["output_files"].get("validation_report"), "validation_report")
    return RunResultEnvelope(run_id=run_id, status=meta["status"], data=data)


@app.get("/api/run/{run_id}/candidates", response_model=RunResultEnvelope)
def get_candidates(run_id: str) -> RunResultEnvelope:
    _assert_run_exists(run_id)
    meta = storage.read_meta(run_id)
    path = storage.get_intermediate_dir(run_id) / "04_candidates.json"
    data = _read_json_or_404(str(path), "candidates")
    return RunResultEnvelope(run_id=run_id, status=meta["status"], data=data)


@app.get("/api/run/{run_id}/heuristics", response_model=RunResultEnvelope)
def get_heuristics(run_id: str) -> RunResultEnvelope:
    _assert_run_exists(run_id)
    meta = storage.read_meta(run_id)
    path = storage.get_intermediate_dir(run_id) / "05_heuristics.json"
    data = _read_json_or_404(str(path), "heuristics")
    return RunResultEnvelope(run_id=run_id, status=meta["status"], data=data)


@app.get("/api/run/{run_id}/banner-annotation", response_model=RunResultEnvelope)
def get_banner_annotation(run_id: str) -> RunResultEnvelope:
    _assert_run_exists(run_id)
    meta = storage.read_meta(run_id)
    path = storage.get_intermediate_dir(run_id) / "06_banner_annotation.json"
    data = _read_json_or_404(str(path), "banner_annotation")
    return RunResultEnvelope(run_id=run_id, status=meta["status"], data=data)


@app.get("/api/run/{run_id}/candidate-annotations", response_model=RunResultEnvelope)
def get_candidate_annotations(run_id: str) -> RunResultEnvelope:
    _assert_run_exists(run_id)
    meta = storage.read_meta(run_id)
    path = storage.get_intermediate_dir(run_id) / "07_candidate_annotations.json"
    data = _read_json_or_404(str(path), "candidate_annotations")
    return RunResultEnvelope(run_id=run_id, status=meta["status"], data=data)


@app.get("/api/run/{run_id}/group-annotations", response_model=RunResultEnvelope)
def get_group_annotations(run_id: str) -> RunResultEnvelope:
    _assert_run_exists(run_id)
    meta = storage.read_meta(run_id)
    path = storage.get_intermediate_dir(run_id) / "08_group_annotations.json"
    data = _read_json_or_404(str(path), "group_annotations")
    return RunResultEnvelope(run_id=run_id, status=meta["status"], data=data)


@app.get("/api/run/{run_id}/log")
def get_run_log(run_id: str) -> JSONResponse:
    _assert_run_exists(run_id)
    log_path = storage.get_log_path(run_id)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Run log not found.")
    return JSONResponse(
        content={
            "run_id": run_id,
            "log": log_path.read_text(encoding="utf-8"),
        }
    )