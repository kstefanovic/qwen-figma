from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from env_load import default_qwen_base_url

from backend.pipeline_v2.image_utils import decode_banner_raster, resize_and_encode_for_zone_qwen
from backend.pipeline_v2.schemas import ClassifyZoneDebug, ClassifyZoneResponse, ZoneAlternativeItem
from backend.pipeline_v2.zone_types import is_allowed_zone_type
from banner_pipeline.qwen_annotator import QwenAnnotator

logger = logging.getLogger(__name__)


def compute_orientation(width: int, height: int) -> str:
    if width < 1 or height < 1:
        return "unknown"
    wh = width / float(height)
    hw = height / float(width)
    if wh >= 3.0:
        return "wide"
    if wh > 1.2:
        return "landscape"
    if hw > 1.2:
        return "portrait"
    return "square"


class ZoneClassificationParseError(Exception):
    def __init__(self, message: str, raw_model_output: str) -> None:
        super().__init__(message)
        self.raw_model_output = raw_model_output


def _normalize_alternatives(raw: Any) -> list[ZoneAlternativeItem]:
    out: list[ZoneAlternativeItem] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        zt = str(item.get("zone_type", "") or "").strip()
        try:
            conf = float(item.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if not is_allowed_zone_type(zt):
            zt = "unknown"
        out.append(ZoneAlternativeItem(zone_type=zt, confidence=conf))
    return out


def _normalize_primary_zone_type(raw: str) -> tuple[str, bool]:
    s = (raw or "").strip()
    if is_allowed_zone_type(s):
        return s, False
    return "unknown", True


def classify_zone_from_banner_bytes(
    raw_banner_bytes: bytes,
    *,
    qwen_base_url: str | None = None,
) -> ClassifyZoneResponse:
    """
    End-to-end zone classification: preprocess banner once, one Qwen HTTP call, validate output.
    Does not touch legacy runner, candidates, heuristics, or convert pipeline.
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

    orientation = compute_orientation(orig_w, orig_h)

    base = ((qwen_base_url or "").strip() or default_qwen_base_url()).rstrip("/")
    annotator = QwenAnnotator(base_url=base)

    t0 = time.perf_counter()
    try:
        data = annotator.classify_zone_from_banner(prepared_bytes)
    finally:
        timings["qwen_classify_zone"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if not data.get("json_parse_ok", True):
        timings["parse_qwen_json"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - t_total0
        logger.info(
            "pipeline_v2 timing parse_qwen_json: run_id=%s ok=false total=%.3fs timings=%s",
            run_id,
            timings["total"],
            timings,
        )
        raise ZoneClassificationParseError(
            "Qwen returned output that could not be parsed as JSON.",
            raw_model_output=str(data.get("raw_model_output", "") or ""),
        )

    raw_zt = str(data.get("zone_type", "") or "")
    zone_type, invalid_primary = _normalize_primary_zone_type(raw_zt)
    reason = str(data.get("reason", "") or "")
    if invalid_primary:
        warn = f"Model zone_type {raw_zt!r} is not in the allowed enum; normalized to unknown."
        reason = f"{reason} ({warn})" if reason else warn

    try:
        confidence = float(data.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    alternatives = _normalize_alternatives(data.get("alternatives"))

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
        alternatives=alternatives,
        debug=ClassifyZoneDebug(
            qwen_call_count=1,
            elapsed_seconds=round(timings["total"], 4),
        ),
    )
