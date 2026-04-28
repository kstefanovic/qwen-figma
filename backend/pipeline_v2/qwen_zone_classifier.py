from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from env_load import default_qwen_base_url

from backend.pipeline_v2.image_utils import decode_banner_raster, resize_and_encode_for_zone_qwen
from backend.pipeline_v2.schemas import ClassifyZoneDebug, ClassifyZoneResponse
from backend.pipeline_v2.zone_types import (
    deterministic_orientation,
    is_allowed_orientation,
    is_allowed_zone_type,
)
from banner_pipeline.qwen_annotator import QwenAnnotator

logger = logging.getLogger(__name__)


def _coerce_confidence(raw: Any) -> float:
    try:
        return float(raw if raw is not None else 0.0)
    except (TypeError, ValueError):
        return 0.0


def classify_zone_from_banner_bytes(
    raw_banner_bytes: bytes,
    *,
    qwen_base_url: str | None = None,
) -> ClassifyZoneResponse:
    """
    One Qwen ``POST /classify-zone`` call (orientation + zone_type only).
    Does not use candidates, heuristics, grouping, or convert pipeline.
    """
    t_total0 = time.perf_counter()
    timings: dict[str, float] = {}
    run_id = str(uuid.uuid4())

    t0 = time.perf_counter()
    logger.info("pipeline_v2 timing receive_request: run_id=%s", run_id)
    timings["receive_request"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    image = decode_banner_raster(raw_banner_bytes)
    orig_w, orig_h = image.size
    timings["load_image"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    prepared_bytes, _ = resize_and_encode_for_zone_qwen(image)
    timings["resize_image"] = time.perf_counter() - t0

    det_orientation = deterministic_orientation(orig_w, orig_h)

    base = ((qwen_base_url or "").strip() or default_qwen_base_url()).rstrip("/")
    annotator = QwenAnnotator(base_url=base)

    t0 = time.perf_counter()
    data = annotator.classify_zone_from_banner(prepared_bytes)
    timings["qwen_classify_zone"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    zone_type = str(data.get("zone_type", "") or "").strip()
    model_orientation = str(data.get("orientation", "") or "").strip()
    confidence = _coerce_confidence(data.get("confidence"))
    reason = str(data.get("reason", "") or "")

    # Match Qwen service: orientation from decoded image pixels only (VLM often mislabels portrait).
    orientation = det_orientation
    if model_orientation and model_orientation != det_orientation and is_allowed_orientation(model_orientation):
        note = (
            f" [orientation from image size: {det_orientation}; model said: {model_orientation}]"
        )
        reason = (reason + note).strip() if reason else note.strip()

    if not is_allowed_zone_type(zone_type):
        zone_type = "whole_text_no_image"
        confidence = min(confidence, 0.2)
        warn = (
            "Model zone_type not in allowed enum; normalized to whole_text_no_image "
            "(allowed: left_text_right_image, upper_image_lower_text, "
            "whole_text_no_image, upper_text_mid_image_lower_text)."
        )
        reason = f"{reason} ({warn})".strip() if reason else warn

    timings["parse_qwen_json"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_total0

    logger.info(
        "pipeline_v2 timings run_id=%s receive_request=%.3fs load_image=%.3fs resize_image=%.3fs "
        "qwen_classify_zone=%.3fs parse_qwen_json=%.3fs total=%.3fs",
        run_id,
        timings.get("receive_request", 0.0),
        timings.get("load_image", 0.0),
        timings.get("resize_image", 0.0),
        timings.get("qwen_classify_zone", 0.0),
        timings.get("parse_qwen_json", 0.0),
        timings.get("total", 0.0),
    )

    return ClassifyZoneResponse(
        run_id=run_id,
        mode="zone_classification",
        zone_type=zone_type,
        orientation=orientation,
        confidence=confidence,
        reason=reason,
        debug=ClassifyZoneDebug(
            qwen_call_count=1,
            elapsed_seconds=round(timings["total"], 4),
        ),
    )
