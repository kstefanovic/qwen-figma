from __future__ import annotations

from env_load import default_qwen_base_url, load_project_env

load_project_env()

import base64
import binascii
import io
import json
import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image

MAX_ELEMENT_REFERENCE_PNGS = 32
_MIN_DECODED_RASTER_BYTES = 24

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
    MAX_ATLAS_REGIONS,
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


def _decode_strict_png_base64(data: str, *, field_label: str) -> bytes:
    """Strict PNG bytes (PNG signature); same encoding rules as banner / atlas."""
    s = (data or "").strip()
    if not s:
        raise ValueError(f"{field_label} is empty.")
    if "base64," in s:
        s = s.split("base64,", 1)[1]
    s = re.sub(r"\s+", "", s)
    try:
        raw = base64.b64decode(s, validate=False)
    except binascii.Error as e:
        raise ValueError(f"{field_label}: invalid base64 ({e}).") from e
    if len(raw) < 8 or raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(
            f"{field_label}: decoded data is not a PNG (expected PNG file signature)."
        )
    return raw


def _decode_png_base64(data: str) -> bytes:
    """Strict PNG for full-frame banner (plugin contract: PNG export)."""
    return _decode_strict_png_base64(data, field_label="banner_png_base64")


def _decode_base64_to_raster_png_bytes(data: str, *, field_label: str) -> bytes:
    """
    Base64 → PNG bytes for disk/Qwen. Accepts PNG, JPEG, WebP, and other formats Pillow can read.
    """
    s = (data or "").strip()
    if not s:
        raise ValueError(f"{field_label}: empty.")
    if "base64," in s:
        s = s.split("base64,", 1)[1]
    s = re.sub(r"\s+", "", s)
    try:
        raw = base64.b64decode(s, validate=False)
    except binascii.Error as e:
        raise ValueError(f"{field_label}: invalid base64 ({e}).") from e
    if len(raw) < _MIN_DECODED_RASTER_BYTES:
        raise ValueError(f"{field_label}: decoded payload is too small to be an image.")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            rgb = im.convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="PNG", optimize=True)
            out = buf.getvalue()
    except Exception as e:
        raise ValueError(
            f"{field_label}: not a supported raster image (PNG/JPEG/WebP, etc.): {e}"
        ) from e
    if len(out) < 32:
        raise ValueError(f"{field_label}: normalized PNG output is unexpectedly small.")
    return out


def _crop_atlas_region_to_png_bytes(
    atlas_im: Image.Image, *, x: int, y: int, w: int, h: int, idx: int
) -> bytes:
    aw, ah = atlas_im.size
    if w <= 0 or h <= 0:
        raise ValueError(f"element_atlas_regions[{idx}]: width and height must be positive.")
    if x < 0 or y < 0 or x + w > aw or y + h > ah:
        raise ValueError(
            f"element_atlas_regions[{idx}]: crop box ({x}, {y}, {w}, {h}) "
            f"is outside atlas bounds ({aw}, {ah})."
        )
    crop = atlas_im.crop((x, y, x + w, y + h))
    rgb = crop.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="PNG", optimize=True)
    out = buf.getvalue()
    if len(out) < 32:
        raise ValueError(f"element_atlas_regions[{idx}]: cropped PNG output is unexpectedly small.")
    return out


def _atlas_regions_from_raw_json(raw: Any) -> list[dict[str, Any]]:
    """Collect atlas_region {x,y,width,height} from any depth; order is tree walk order."""
    acc: list[dict[str, Any]] = []

    def walk(o: Any) -> None:
        if len(acc) >= MAX_ATLAS_REGIONS:
            return
        if isinstance(o, dict):
            ar = o.get("atlas_region")
            if isinstance(ar, dict):
                try:
                    x = int(ar["x"])
                    y = int(ar["y"])
                    w = int(ar["width"])
                    h = int(ar["height"])
                except (KeyError, TypeError, ValueError):
                    pass
                else:
                    if w > 0 and h > 0 and x >= 0 and y >= 0:
                        acc.append(
                            {
                                "path": o.get("path"),
                                "node_id": o.get("node_id") or o.get("id"),
                                "name": o.get("name"),
                                "type": o.get("type"),
                                "atlas_x": x,
                                "atlas_y": y,
                                "atlas_width": w,
                                "atlas_height": h,
                                "from_raw_json_atlas_region": True,
                            }
                        )
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for item in o:
                walk(item)

    walk(raw)
    return acc


def _effective_atlas_crop_rows(body: ConvertRequest) -> list[dict[str, Any]]:
    """Prefer element_atlas_regions; if atlas is set but the list is empty, use raw_json atlas_region."""
    from_regions = [r.model_dump() for r in (body.element_atlas_regions or [])]
    atlas_b64 = (body.element_atlas_png_base64 or "").strip()
    if atlas_b64 and not from_regions:
        return _atlas_regions_from_raw_json(body.raw_json)
    return from_regions


def _save_convert_plugin_element_assets(run_id: str, body: ConvertRequest) -> dict[str, Any]:
    """
    Decode element atlas PNG to elements/atlas.png, crop leaves from element_atlas_regions
    or raw_json atlas_region, persist element_image_refs JSON; Qwen paths are cropped PNGs only.
    """
    leaf_saved: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    leaf_source: str = "none"
    atlas_path: str | None = None

    atlas_b64 = (body.element_atlas_png_base64 or "").strip()
    crop_rows = _effective_atlas_crop_rows(body)
    if atlas_b64:
        leaf_source = "element_atlas"
        if not crop_rows:
            raise HTTPException(
                status_code=400,
                detail=(
                    "element_atlas_png_base64 is non-empty but no crop rectangles were found "
                    "in element_atlas_regions or raw_json atlas_region."
                ),
            )
        try:
            atlas_png_bytes = _decode_strict_png_base64(
                atlas_b64, field_label="element_atlas_png_base64"
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        atlas_path = storage.save_upload_bytes(run_id, "elements/atlas.png", atlas_png_bytes)
        try:
            with Image.open(io.BytesIO(atlas_png_bytes)) as atlas_im:
                atlas_rgb = atlas_im.convert("RGB")
                for i, row in enumerate(crop_rows[:MAX_ATLAS_REGIONS]):
                    x = int(row["atlas_x"])
                    y = int(row["atlas_y"])
                    w = int(row["atlas_width"])
                    h = int(row["atlas_height"])
                    try:
                        raw = _crop_atlas_region_to_png_bytes(
                            atlas_rgb, x=x, y=y, w=w, h=h, idx=i
                        )
                    except ValueError as e:
                        raise HTTPException(status_code=400, detail=str(e)) from None
                    rel = f"elements/leaf_{i:03d}.png"
                    leaf_saved.append(storage.save_upload_bytes(run_id, rel, raw))
                    mrow = dict(row)
                    mrow["saved_as"] = rel
                    mrow["from_element_atlas"] = True
                    manifest_rows.append(mrow)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"element_atlas_png_base64: could not open or crop atlas ({e}).",
            ) from None

    manifest_path: str | None = None
    if manifest_rows:
        manifest_path = storage.save_upload_bytes(
            run_id,
            "elements/element_atlas_crops_manifest.json",
            json.dumps(manifest_rows, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    refs_path: str | None = None
    if body.element_image_refs:
        refs_data = [r.model_dump() for r in body.element_image_refs]
        refs_path = storage.save_upload_bytes(
            run_id,
            "elements/element_image_refs.json",
            json.dumps(refs_data, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    qwen_paths: list[str] = list(leaf_saved[:MAX_ELEMENT_REFERENCE_PNGS])

    return {
        "qwen_paths": qwen_paths,
        "manifest_path": manifest_path,
        "refs_path": refs_path,
        "atlas_path": atlas_path,
        "leaf_saved_count": len(leaf_saved),
        "library_saved_count": 0,
        "leaf_source": leaf_source,
    }


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

    assets = _save_convert_plugin_element_assets(run_id, body)
    element_paths: list[str] = list(assets["qwen_paths"])

    input_files: dict[str, Any] = {
        "banner_image": banner_path,
        "raw_json": raw_json_path,
    }
    if element_paths:
        input_files["element_images"] = element_paths
    if assets.get("atlas_path"):
        input_files["element_atlas_png"] = assets["atlas_path"]
    if assets.get("manifest_path"):
        input_files["element_atlas_crops_manifest"] = assets["manifest_path"]
    if assets.get("refs_path"):
        input_files["element_image_refs"] = assets["refs_path"]

    storage.update_meta(
        run_id,
        input_files=input_files,
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
            "plugin_element_atlas_regions_in_json": len(body.element_atlas_regions or []),
            "plugin_leaf_png_source": assets.get("leaf_source", "none"),
            "plugin_element_image_refs_received": len(body.element_image_refs or []),
            "plugin_atlas_png_saved": bool(assets.get("atlas_path")),
            "plugin_leaf_pngs_saved": assets["leaf_saved_count"],
            "plugin_qwen_extra_image_count": len(element_paths),
            "element_atlas_regions_count_declared": body.element_atlas_regions_count,
            "element_image_refs_count_declared": body.element_image_refs_count,
        },
    )

    try:
        pipeline_result = runner.run(
            run_id=run_id,
            raw_json_path=raw_json_path,
            banner_image_path=banner_path,
            element_image_paths=element_paths if element_paths else None,
            atlas_image_path=assets.get("atlas_path"),
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
    element_images: UploadFile | list[UploadFile] | None = File(default=None),
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

    element_paths: list[str] = []
    raw_uploads = element_images
    if raw_uploads is None:
        upload_list: list[UploadFile] = []
    elif isinstance(raw_uploads, list):
        upload_list = raw_uploads
    else:
        upload_list = [raw_uploads]
    for i, uf in enumerate(upload_list[:MAX_ELEMENT_REFERENCE_PNGS]):
        if not uf.filename:
            continue
        suf = Path(uf.filename).suffix.lower()
        if suf not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise HTTPException(
                status_code=400,
                detail=f"element_images[{i}] must be .png, .jpg, .jpeg, or .webp (got {uf.filename!r}).",
            )
        eb = await uf.read()
        if not eb:
            continue
        elem_path = storage.save_upload_bytes(run_id, f"elements/elem_{i:03d}{suf}", eb)
        element_paths.append(elem_path)

    storage.update_meta(
        run_id,
        input_files={
            "banner_image": banner_path,
            "raw_json": raw_json_path,
            "element_images": element_paths if element_paths else None,
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
            element_image_paths=element_paths if element_paths else None,
            atlas_image_path=None,
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