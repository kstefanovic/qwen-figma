from __future__ import annotations

from env_load import default_qwen_base_url, load_project_env

load_project_env()

import base64
import binascii
import json
import re
from pathlib import Path
from typing import Any, Optional

import requests
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.convert_scene import build_convert_semantic_payload
from backend.runner import PipelineRunner
from backend.schemas import (
    ConvertDebug,
    ConvertFrameSpec,
    ConvertRequest,
    ConvertResponse,
    ConvertSemanticElement,
    ConvertSemanticGroup,
    ConvertSemanticSummary,
    LayoutUpdateItem,
    HealthResponse,
    RunCreateResponse,
    RunListItem,
    RunListResponse,
    RunResultEnvelope,
    RunSummaryResponse,
)
from backend.storage import RunStorage


RUNS_DIR = Path("runs")
QWEN_BASE_URL = default_qwen_base_url()


def _optional_form_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _decode_png_base64(data: str) -> bytes:
    s = (data or "").strip()
    if not s:
        raise ValueError("banner_png_base64 is empty.")
    if "base64," in s:
        s = s.split("base64,", 1)[1]
    s = re.sub(r"\s+", "", s)
    try:
        raw = base64.b64decode(s, validate=False)
    except binascii.Error as e:
        raise ValueError(f"Invalid base64 data: {e}") from e
    if len(raw) < 8 or raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Decoded banner is not a PNG (expected PNG file signature).")
    return raw


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


def _build_confidence_by_element_id(run_id: str) -> dict[str, float]:
    confidence_by_element_id: dict[str, float] = {}
    candidate_ann_path = storage.get_intermediate_dir(run_id) / "07_candidate_annotations.json"
    if candidate_ann_path.exists():
        try:
            payload = json.loads(candidate_ann_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for candidate_id, ann in payload.items():
                    if not isinstance(ann, dict):
                        continue
                    try:
                        confidence = float(ann.get("confidence", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        confidence = 0.0
                    confidence_by_element_id[f"el_{candidate_id}"] = confidence
        except Exception:
            confidence_by_element_id = {}
    return confidence_by_element_id


def _read_annotation_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _build_convert_annotation_context(run_id: str, pipeline_result: dict[str, Any] | None = None) -> dict[str, Any]:
    pipeline_result = pipeline_result or {}
    intermediate_dir = storage.get_intermediate_dir(run_id)
    candidate_payload = _read_annotation_payload(intermediate_dir / "07_candidate_annotations.json")
    group_payload = _read_annotation_payload(intermediate_dir / "08_group_annotations.json")
    scene_payload = _read_annotation_payload(intermediate_dir / "06b_scene_semantics.json")

    result_candidate_annotations = {
        str(k): v.to_dict() if hasattr(v, "to_dict") else v
        for k, v in (pipeline_result.get("candidate_annotations") or {}).items()
    }
    result_group_annotations = {
        str(k): v.to_dict() if hasattr(v, "to_dict") else v
        for k, v in (pipeline_result.get("group_annotations") or {}).items()
    }
    if result_candidate_annotations:
        candidate_payload = result_candidate_annotations
    if result_group_annotations:
        group_payload = result_group_annotations

    scene_updates = list(pipeline_result.get("scene_semantic_updates") or [])
    scene_groups = list(pipeline_result.get("scene_semantic_groups") or [])
    if not scene_updates and scene_payload:
        scene_updates = list(scene_payload.get("updates", []) or [])
    if not scene_groups and scene_payload:
        scene_groups = list(scene_payload.get("groups", []) or [])

    candidate_reason_by_element_id: dict[str, str] = {}
    candidate_meta_by_id: dict[str, dict[str, Any]] = {}
    confidence_by_element_id = _build_confidence_by_element_id(run_id)
    if candidate_payload:
        for candidate_id, ann in candidate_payload.items():
            if not isinstance(ann, dict):
                continue
            element_id = f"el_{candidate_id}"
            candidate_reason_by_element_id[element_id] = str(ann.get("reason_short", "") or "")
            candidate_meta_by_id[str(candidate_id)] = ann
            try:
                confidence_by_element_id[element_id] = float(ann.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence_by_element_id[element_id] = 0.0

    return {
        "confidence_by_element_id": confidence_by_element_id,
        "candidate_reason_by_element_id": candidate_reason_by_element_id,
        "candidate_annotations_by_id": candidate_meta_by_id,
        "group_annotations_by_id": group_payload,
        "scene_semantic_updates": scene_updates,
        "scene_semantic_groups": scene_groups,
    }


def _resolve_convert_execution(body: ConvertRequest) -> tuple[str, bool, str]:
    mode = body.mode or "apply_to_clone_fast"
    if mode == "apply_to_clone_fast":
        return mode, False, "off"
    if mode == "apply_to_clone_vlm":
        return mode, True, "scene_only"

    qwen_enabled = body.use_qwen if "use_qwen" in body.model_fields_set else True
    qwen_mode = body.qwen_mode if qwen_enabled else "off"
    return mode, qwen_enabled, qwen_mode or "per_candidate"


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


@app.post("/api/convert", response_model=ConvertResponse)
async def convert_from_plugin(body: ConvertRequest = Body(...)) -> ConvertResponse:
    """
    Figma plugin entrypoint: run pipeline, return layout updates for clone-and-apply (no placeholder scene).
    """
    try:
        png_bytes = _decode_png_base64(body.banner_png_base64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    brand_family_override = _optional_form_str(body.brand_family)
    language_override = _optional_form_str(body.language)
    category_override = _optional_form_str(body.category)
    pipeline_mode, use_qwen, qwen_mode = _resolve_convert_execution(body)

    run_id = storage.create_run()

    banner_path = storage.save_upload_bytes(run_id, "banner.png", png_bytes)
    raw_json_path = storage.save_upload_bytes(
        run_id,
        "raw_figma.json",
        json.dumps(body.raw_json, ensure_ascii=False).encode("utf-8"),
    )

    storage.update_meta(
        run_id,
        input_files={
            "banner_image": banner_path,
            "raw_json": raw_json_path,
        },
        metadata={
            "source": "figma_plugin_convert",
            "mode": pipeline_mode,
            "target_width": body.target_width,
            "target_height": body.target_height,
            "brand_family_override": brand_family_override,
            "language_override": language_override,
            "category_override": category_override,
            "use_qwen": use_qwen,
            "qwen_mode": qwen_mode,
        },
    )

    try:
        pipeline_result = runner.run(
            run_id=run_id,
            raw_json_path=raw_json_path,
            banner_image_path=banner_path,
            brand_family=brand_family_override,
            language=language_override,
            category=category_override,
            use_qwen=use_qwen,
            qwen_mode=qwen_mode,
            pipeline_mode=pipeline_mode,
        )
    except Exception as e:
        meta = storage.read_meta(run_id)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed (run_id={run_id}, status={meta.get('status')}): {e}",
        ) from e

    meta = storage.read_meta(run_id)
    sg_path = (meta.get("output_files") or {}).get("semantic_graph")
    if not sg_path:
        raise HTTPException(status_code=500, detail="semantic_graph path missing in run metadata after success.")
    graph_path = Path(sg_path)
    if not graph_path.exists():
        raise HTTPException(status_code=500, detail=f"semantic_graph file not found: {graph_path}")

    try:
        semantic_graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read semantic_graph.json: {e}") from e

    annotation_context = _build_convert_annotation_context(run_id, pipeline_result)
    convert_payload = build_convert_semantic_payload(
        semantic_graph,
        body.raw_json,
        body.target_width,
        body.target_height,
        confidence_by_element_id=annotation_context["confidence_by_element_id"],
        reason_by_element_id=annotation_context["candidate_reason_by_element_id"],
        candidate_annotations_by_id=annotation_context["candidate_annotations_by_id"],
        group_annotations_by_id=annotation_context["group_annotations_by_id"],
        scene_semantic_updates=annotation_context["scene_semantic_updates"],
        scene_semantic_groups=annotation_context["scene_semantic_groups"],
    )
    vr_path = (meta.get("output_files") or {}).get("validation_report")
    vr_str = str(vr_path) if vr_path else None
    validation_payload = _read_annotation_payload(Path(vr_str)) if vr_str else {}
    validation_warnings = []
    if validation_payload:
        for item in validation_payload.get("warnings", []) or []:
            if isinstance(item, dict):
                validation_warnings.append(str(item.get("message", "") or ""))

    return ConvertResponse(
        run_id=run_id,
        mode=pipeline_mode,
        frame=ConvertFrameSpec(**convert_payload["frame"]),
        updates=[LayoutUpdateItem.model_validate(u) for u in convert_payload["updates"]],
        semantic=ConvertSemanticSummary(
            elements=[ConvertSemanticElement.model_validate(x) for x in convert_payload["semantic_elements"]],
            groups=[ConvertSemanticGroup.model_validate(x) for x in convert_payload["semantic_groups"]],
        ),
        debug=ConvertDebug(
            semantic_graph_path=str(graph_path),
            validation_report_path=vr_str,
            qwen_call_count=int((meta.get("metadata") or {}).get("qwen_call_count") or 0),
            qwen_mode=(meta.get("metadata") or {}).get("qwen_mode"),
            stage_timings=dict((meta.get("metadata") or {}).get("stage_timings") or {}),
            nodes_annotated=int(convert_payload.get("nodes_annotated", 0) or pipeline_result.get("nodes_annotated", 0) or 0),
            nodes_left_unnamed=int(convert_payload.get("nodes_left_unnamed", 0) or 0),
            validation_warnings=validation_warnings[:12],
            low_confidence_examples=list(convert_payload.get("low_confidence_examples", []) or [])[:8],
        ),
    )


@app.post("/api/run", response_model=RunCreateResponse)
async def create_run(
    banner_image: UploadFile = File(...),
    raw_json: UploadFile = File(...),
    brand_family: Optional[str] = Form(default=None),
    language: Optional[str] = Form(default=None),
    category: Optional[str] = Form(default=None),
    use_qwen: bool = Form(True),
    qwen_mode: str = Form("single_pass"),
) -> RunCreateResponse:
    if not banner_image.filename:
        raise HTTPException(status_code=400, detail="banner_image filename is missing.")
    if not raw_json.filename:
        raise HTTPException(status_code=400, detail="raw_json filename is missing.")

    if not banner_image.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        raise HTTPException(status_code=400, detail="banner_image must be png/jpg/jpeg/webp.")

    if not raw_json.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="raw_json must be a .json file.")

    brand_family_override = _optional_form_str(brand_family)
    language_override = _optional_form_str(language)
    category_override = _optional_form_str(category)

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
            "brand_family_override": brand_family_override,
            "language_override": language_override,
            "category_override": category_override,
            "use_qwen": use_qwen,
            "qwen_mode": (qwen_mode or "single_pass") if use_qwen else "off",
        },
    )

    try:
        runner.run(
            run_id=run_id,
            raw_json_path=raw_json_path,
            banner_image_path=banner_path,
            brand_family=brand_family_override,
            language=language_override,
            category=category_override,
            use_qwen=use_qwen,
            qwen_mode=qwen_mode if use_qwen else "off",
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