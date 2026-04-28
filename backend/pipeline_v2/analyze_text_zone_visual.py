from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from env_load import default_qwen_base_url

from backend.pipeline_v2.image_utils import decode_banner_raster, resize_and_encode_for_zone_qwen
from backend.pipeline_v2.schemas import (
    AnalyzeTextZoneVisualDebug,
    AnalyzeTextZoneVisualResponse,
    NormalizedBbox,
    TextZoneGroupItem,
    TextZoneVisual,
)
from backend.pipeline_v2.zone_types import (
    deterministic_orientation,
    is_allowed_orientation,
    is_allowed_text_zone_role,
    is_allowed_zone_type,
)
from banner_pipeline.qwen_annotator import QwenAnnotator

logger = logging.getLogger(__name__)


def _coerce_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _sanitize_bbox(b: Any, warnings: list[str], idx: int, role: str) -> NormalizedBbox | None:  # noqa: PLR0911
    if not isinstance(b, dict):
        warnings.append(f"groups[{idx}] role={role!r}: bbox not an object")
        return None
    x = _coerce_float(b.get("x"))
    y = _coerce_float(b.get("y"))
    w = _coerce_float(b.get("width"))
    h = _coerce_float(b.get("height"))
    if x is None or y is None or w is None or h is None:
        warnings.append(f"groups[{idx}] role={role!r}: bbox non-numeric")
        return None
    x, y = _clamp01(x), _clamp01(y)
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    if w <= 1e-6 or h <= 1e-6:
        warnings.append(f"groups[{idx}] role={role!r}: bbox width/height not positive")
        return None
    if x + w > 1.0:
        w = max(1e-6, 1.0 - x)
    if y + h > 1.0:
        h = max(1e-6, 1.0 - y)
    return NormalizedBbox(x=x, y=y, width=w, height=h)


def _coerce_confidence(v: Any) -> float:
    try:
        return float(v if v is not None else 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_text_zone_group_role(role: str, warnings: list[str], idx: int) -> str | None:
    s = (role or "").strip()
    if s == "logo_group":
        warnings.append(f"groups[{idx}]: role logo_group normalized to brand_group")
        s = "brand_group"
    if is_allowed_text_zone_role(s):
        return s
    return None


def _dedupe_text_zone_group_items(
    items: list[TextZoneGroupItem],
    warnings: list[str],
) -> list[TextZoneGroupItem]:
    """One group per role; highest confidence wins (ties keep first kept)."""
    by_role: dict[str, TextZoneGroupItem] = {}
    for g in items:
        prev = by_role.get(g.role)
        if prev is None:
            by_role[g.role] = g
            continue
        if g.confidence > prev.confidence:
            warnings.append(
                f"duplicate role={g.role!r}: kept higher confidence ({g.confidence:.4f} vs dropped {prev.confidence:.4f})",
            )
            by_role[g.role] = g
        else:
            warnings.append(
                f"duplicate role={g.role!r}: dropped lower confidence ({g.confidence:.4f} vs kept {prev.confidence:.4f})",
            )
    order = ("brand_group", "headline_group", "legal_text")
    return [by_role[r] for r in order if r in by_role]


def analyze_text_zone_visual_from_banner_bytes(
    raw_banner_bytes: bytes,
    *,
    qwen_base_url: str | None = None,
) -> AnalyzeTextZoneVisualResponse:
    """
    One Qwen ``POST /analyze-text-zone-visual`` call: orientation, zone_type, text_zone.groups.
    Banner PNG only — no raw JSON, atlas, or Figma IDs.
    """
    t_total0 = time.perf_counter()
    timings: dict[str, float] = {}
    run_id = str(uuid.uuid4())
    warnings: list[str] = []

    t0 = time.perf_counter()
    logger.info("pipeline_v2 analyze_text_zone_visual receive_request run_id=%s", run_id)
    timings["receive_request"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    image = decode_banner_raster(raw_banner_bytes)
    orig_w, orig_h = image.size
    timings["load_image"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    prepared_bytes, resize_meta = resize_and_encode_for_zone_qwen(image)
    timings["resize_image"] = time.perf_counter() - t0
    rw = int(resize_meta.get("prepared_width", 0) or 0)
    rh = int(resize_meta.get("prepared_height", 0) or 0)

    det_orientation = deterministic_orientation(orig_w, orig_h)

    base = ((qwen_base_url or "").strip() or default_qwen_base_url()).rstrip("/")
    annotator = QwenAnnotator(base_url=base)

    t0 = time.perf_counter()
    data = annotator.analyze_text_zone_visual_from_banner(prepared_bytes)
    qwen_elapsed = time.perf_counter() - t0
    timings["qwen_elapsed_seconds"] = qwen_elapsed

    t0 = time.perf_counter()
    zone_type = str(data.get("zone_type", "") or "").strip()
    model_orientation = str(data.get("orientation", "") or "").strip()
    confidence = _coerce_confidence(data.get("confidence"))
    reason = str(data.get("reason", "") or "")

    orientation = det_orientation
    if model_orientation and model_orientation != det_orientation and is_allowed_orientation(model_orientation):
        reason = (
            (reason + f" [orientation from pixels: {det_orientation}; model: {model_orientation}]")
            if reason
            else f"[orientation from pixels: {det_orientation}; model: {model_orientation}]"
        ).strip()

    if not is_allowed_zone_type(zone_type):
        zone_type = "whole_text_no_image"
        confidence = min(confidence, 0.2)
        wmsg = (
            "Model zone_type not in allowed enum; normalized to whole_text_no_image "
            "(allowed: left_text_right_image, upper_image_lower_text, "
            "whole_text_no_image, upper_text_mid_image_lower_text)."
        )
        warnings.append(wmsg)
        reason = f"{reason} ({wmsg})".strip() if reason else wmsg

    groups_out: list[TextZoneGroupItem] = []
    tz = data.get("text_zone")
    raw_groups: list[Any] = []
    if isinstance(tz, dict) and isinstance(tz.get("groups"), list):
        raw_groups = tz["groups"]
    for i, item in enumerate(raw_groups):
        if not isinstance(item, dict):
            warnings.append(f"groups[{i}]: skipped non-object")
            continue
        raw_role = str(item.get("role", "") or "").strip()
        role = _normalize_text_zone_group_role(raw_role, warnings, i)
        if role is None:
            warnings.append(f"groups[{i}]: removed invalid role={raw_role!r}")
            continue
        bbox_m = _sanitize_bbox(item.get("bbox"), warnings, i, role)
        if bbox_m is None:
            continue
        gc = _coerce_confidence(item.get("confidence"))
        gr = str(item.get("reason", "") or "")
        groups_out.append(
            TextZoneGroupItem(role=role, bbox=bbox_m, confidence=gc, reason=gr),
        )

    groups_out = _dedupe_text_zone_group_items(groups_out, warnings)

    timings["parse_qwen_json"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_total0

    group_roles = [g.role for g in groups_out]
    logger.info(
        "pipeline_v2 analyze_text_zone_visual endpoint=/api/v2/analyze-text-zone-visual "
        "run_id=%s original_image=%dx%d resized_image=%dx%d qwen_elapsed_seconds=%.4f "
        "orientation=%r zone_type=%r text_zone_group_count=%d groups=%r "
        "validation_warning_count=%d total_seconds=%.4f",
        run_id,
        orig_w,
        orig_h,
        rw,
        rh,
        qwen_elapsed,
        orientation,
        zone_type,
        len(groups_out),
        group_roles,
        len(warnings),
        timings["total"],
    )
    for g in groups_out:
        logger.info(
            "pipeline_v2 analyze_text_zone_visual group role=%r confidence=%.4f bbox=%s",
            g.role,
            g.confidence,
            g.bbox.model_dump(),
        )
    if warnings:
        logger.info("pipeline_v2 analyze_text_zone_visual validation_warnings=%s", warnings)

    return AnalyzeTextZoneVisualResponse(
        run_id=run_id,
        mode="text_zone_visual",
        orientation=orientation,
        zone_type=zone_type,
        confidence=confidence,
        reason=reason,
        text_zone=TextZoneVisual(groups=groups_out),
        debug=AnalyzeTextZoneVisualDebug(
            qwen_call_count=1,
            elapsed_seconds=round(timings["total"], 4),
            validation_warnings=warnings,
        ),
    )
