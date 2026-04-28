from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

from env_load import default_qwen_base_url

from backend.pipeline_v2.image_utils import decode_banner_raster, resize_and_encode_for_zone_qwen
from backend.pipeline_v2.schemas import (
    AnalyzeTextZoneVisualDebug,
    AnalyzeTextZoneVisualResponse,
    NormalizedBbox,
    TextZoneChildItem,
    TextZoneGroupItem,
    TextZoneVisual,
)
from backend.pipeline_v2.zone_types import (
    TEXT_ZONE_CHILD_TEXT_MAX_CHARS,
    TEXT_ZONE_LOGO_CHILD_ROLES,
    canonical_text_zone_child_role,
    deterministic_orientation,
    is_allowed_orientation,
    is_allowed_text_zone_child_for_parent,
    is_allowed_text_zone_role,
    is_allowed_zone_type,
    text_zone_child_may_omit_transcribed_text,
    text_zone_child_sort_key,
)
from banner_pipeline.qwen_annotator import QwenAnnotator

logger = logging.getLogger(__name__)


def _infer_child_text_from_reason(reason: str) -> str:
    """Backfill empty child \"text\" from quoted snippets in \"reason\" (matches qwen_service)."""
    s = (reason or "").strip()
    if len(s) < 3:
        return ""
    cap = str(TEXT_ZONE_CHILD_TEXT_MAX_CHARS)
    patterns = (
        rf"«([^»]{{2,{cap}}})»",
        rf"[\u201c\u201e]([^\u201d]{{2,{cap}}})[\u201d]",
        rf'"([^"]{{2,{cap}}})"',
        rf"'([^']{{2,{cap}}})'",
    )
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            t = (m.group(1) or "").strip()
            if len(t) >= 2:
                return t[:TEXT_ZONE_CHILD_TEXT_MAX_CHARS]
    return ""


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


def _scale_bbox_component_if_pixel(val: float, denom: int) -> float:
    """Match qwen_service: only divide values > 1.0 (mixed pixel x,y with 0–1 w,h)."""
    if val > 1.0:
        return val / float(max(1, denom))
    return val


def _clamp_norm_bbox_model(x: float, y: float, w: float, h: float) -> NormalizedBbox | None:
    x, y = _clamp01(x), _clamp01(y)
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    if w <= 1e-6 or h <= 1e-6:
        return None
    if x + w > 1.0:
        w = max(1e-6, 1.0 - x)
    if y + h > 1.0:
        h = max(1e-6, 1.0 - y)
    return NormalizedBbox(x=x, y=y, width=w, height=h)


def _sanitize_bbox(
    b: Any,
    warnings: list[str],
    idx: int,
    role: str,
    *,
    ctx: str = "groups",
    pixel_w: int | None = None,
    pixel_h: int | None = None,
) -> NormalizedBbox | None:  # noqa: PLR0911
    pw = max(1, int(pixel_w)) if pixel_w and pixel_w > 0 else None
    ph = max(1, int(pixel_h)) if pixel_h and pixel_h > 0 else None

    if isinstance(b, (list, tuple)) and len(b) == 4 and pw and ph:
        fv = [_coerce_float(x) for x in b]
        if any(v is None for v in fv):
            warnings.append(f"{ctx}[{idx}] role={role!r}: bbox list non-numeric")
            return None
        a, c, d, e = float(fv[0]), float(fv[1]), float(fv[2]), float(fv[3])  # type: ignore[arg-type]
        if max(a, c, d, e) > 1.0:
            if d > a and e > c and (d - a) <= float(pw) * 1.02 and (e - c) <= float(ph) * 1.02:
                x1, y1, x2, y2 = a, c, d, e
                x1 = _scale_bbox_component_if_pixel(x1, pw)
                x2 = _scale_bbox_component_if_pixel(x2, pw)
                y1 = _scale_bbox_component_if_pixel(y1, ph)
                y2 = _scale_bbox_component_if_pixel(y2, ph)
                x_px, y_px, w_px, h_px = x1, y1, x2 - x1, y2 - y1
            else:
                x_px = _scale_bbox_component_if_pixel(a, pw)
                y_px = _scale_bbox_component_if_pixel(c, ph)
                w_px = _scale_bbox_component_if_pixel(d, pw)
                h_px = _scale_bbox_component_if_pixel(e, ph)
            if w_px <= 0 or h_px <= 0:
                return None
            x, y, w, h = x_px, y_px, w_px, h_px
        else:
            x, y, w, h = a, c, d, e
        return _clamp_norm_bbox_model(x, y, w, h)

    if not isinstance(b, dict):
        warnings.append(f"{ctx}[{idx}] role={role!r}: bbox not an object")
        return None
    x = _coerce_float(b.get("x"))
    y = _coerce_float(b.get("y"))
    w = _coerce_float(b.get("width"))
    h = _coerce_float(b.get("height"))
    if x is None or y is None or w is None or h is None:
        warnings.append(f"{ctx}[{idx}] role={role!r}: bbox non-numeric")
        return None
    if pw and ph:
        x = _scale_bbox_component_if_pixel(float(x), pw)
        y = _scale_bbox_component_if_pixel(float(y), ph)
        w = _scale_bbox_component_if_pixel(float(w), pw)
        h = _scale_bbox_component_if_pixel(float(h), ph)
    return _clamp_norm_bbox_model(x, y, w, h)


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


def _dedupe_child_items(
    parent_role: str,
    items: list[TextZoneChildItem],
    warnings: list[str],
) -> list[TextZoneChildItem]:
    by_role: dict[str, TextZoneChildItem] = {}
    for c in items:
        prev = by_role.get(c.role)
        if prev is None:
            by_role[c.role] = c
            continue
        if c.confidence > prev.confidence:
            warnings.append(
                f"{parent_role} duplicate child role={c.role!r}: kept higher confidence "
                f"({c.confidence:.4f} vs dropped {prev.confidence:.4f})",
            )
            by_role[c.role] = c
        else:
            warnings.append(
                f"{parent_role} duplicate child role={c.role!r}: dropped lower confidence "
                f"({c.confidence:.4f} vs kept {prev.confidence:.4f})",
            )
    return sorted(by_role.values(), key=lambda x: text_zone_child_sort_key(parent_role, x.role))


def _parse_group_children(
    parent_role: str,
    raw: Any,
    warnings: list[str],
    *,
    px_w: int,
    px_h: int,
    group_idx: int,
) -> list[TextZoneChildItem]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        warnings.append(f"groups[{group_idx}] role={parent_role!r}: children is not a list")
        return []
    out: list[TextZoneChildItem] = []
    for j, ch in enumerate(raw):
        if not isinstance(ch, dict):
            continue
        rraw = str(ch.get("role", "") or "").strip()
        if not is_allowed_text_zone_child_for_parent(parent_role, rraw):
            warnings.append(
                f"groups[{group_idx}] children[{j}]: removed invalid role={rraw!r} for parent={parent_role!r}",
            )
            continue
        r = canonical_text_zone_child_role(rraw)
        if rraw and r != rraw:
            warnings.append(f"groups[{group_idx}] children[{j}]: role {rraw!r} normalized to {r!r}")
        bbox_m = _sanitize_bbox(
            ch.get("bbox"),
            warnings,
            j,
            r,
            ctx=f"groups[{group_idx}].children",
            pixel_w=px_w,
            pixel_h=px_h,
        )
        if bbox_m is None:
            continue
        txt_raw = ch.get("text")
        if txt_raw is None:
            tstr = ""
        else:
            tstr = str(txt_raw).strip()
            if len(tstr) > TEXT_ZONE_CHILD_TEXT_MAX_CHARS:
                tstr = tstr[:TEXT_ZONE_CHILD_TEXT_MAX_CHARS]
                warnings.append(
                    f"groups[{group_idx}] children[{j}] role={r!r}: text truncated to "
                    f"{TEXT_ZONE_CHILD_TEXT_MAX_CHARS} chars",
                )
        cr = str(ch.get("reason", "") or "")
        if not tstr and r not in TEXT_ZONE_LOGO_CHILD_ROLES:
            inferred = _infer_child_text_from_reason(cr)
            if inferred:
                tstr = inferred
                warnings.append(
                    f"groups[{group_idx}] children[{j}] role={r!r}: filled \"text\" from quoted snippet in "
                    f"\"reason\" (model should still emit text directly in JSON).",
                )
        if r in TEXT_ZONE_LOGO_CHILD_ROLES and tstr:
            snippet = tstr[:80] + ("…" if len(tstr) > 80 else "")
            warnings.append(
                f"groups[{group_idx}] children[{j}] role={r!r}: cleared non-empty \"text\" ({snippet!r}) — "
                f"logo/logo_back/logo_fore must use empty \"text\"; use brand_name_first / brand_name_second "
                f"for words.",
            )
            tstr = ""
        if not text_zone_child_may_omit_transcribed_text(parent_role, r) and not tstr:
            warnings.append(
                f"groups[{group_idx}] children[{j}] role={r!r}: empty \"text\" — transcribe visible copy "
                f"(required except logo, logo_back, logo_fore).",
            )
        cc = _coerce_confidence(ch.get("confidence"))
        out.append(TextZoneChildItem(role=r, text=tstr, bbox=bbox_m, confidence=cc, reason=cr))
    return _dedupe_child_items(parent_role, out, warnings)


def _warn_brand_group_missing_word_children(
    groups: list[TextZoneGroupItem],
    warnings: list[str],
) -> None:
    word_roles = frozenset({"brand_name", "brand_name_first", "brand_name_second"})
    for g in groups:
        if g.role != "brand_group" or not g.children:
            continue
        roles = {c.role for c in g.children}
        if roles & TEXT_ZONE_LOGO_CHILD_ROLES and not (roles & word_roles):
            warnings.append(
                "brand_group has logo/logo_back/logo_fore but no brand_name / brand_name_first / "
                "brand_name_second; two-word+mark layouts should use brand_name_first + logo + brand_name_second.",
            )


def _warn_headline_group_missing_headline(
    groups: list[TextZoneGroupItem],
    warnings: list[str],
) -> None:
    for g in groups:
        if g.role != "headline_group":
            continue
        roles = {c.role for c in g.children}
        if "headline" not in roles:
            warnings.append(
                "headline_group present but no child with role headline; model should include headline when group exists.",
            )


def _warn_missing_legal_text_top_level(
    groups: list[TextZoneGroupItem],
    warnings: list[str],
) -> None:
    roles = {g.role for g in groups}
    if "legal_text" not in roles:
        warnings.append(
            "No legal_text top-level group. Commercial banners normally include disclaimer/footer micro-copy "
            "(e.g. Реклама, ООО, ИНН) — re-check the bottom ~35% of the canvas and lower corners.",
        )


def _warn_headline_group_missing_subheadline_support(
    groups: list[TextZoneGroupItem],
    warnings: list[str],
) -> None:
    support_roles = frozenset(
        {
            "subheadline",
            "subheadline_delivery_time",
            "subheadline_weight",
            "product_name",
            "subheadline_discount",
        },
    )
    for g in groups:
        if g.role != "headline_group":
            continue
        roles = {c.role for c in g.children}
        if roles & support_roles:
            continue
        if "headline" not in roles:
            continue
        headline_child = next((c for c in g.children if c.role == "headline"), None)
        if headline_child is None:
            continue
        gy, gh = g.bbox.y, g.bbox.height
        hy, hh = headline_child.bbox.y, headline_child.bbox.height
        bottom_g = gy + gh
        bottom_h = hy + hh
        if bottom_g - bottom_h > 0.06:
            warnings.append(
                "headline_group extends noticeably below the headline child bbox; likely missing a "
                "subheadline (e.g. subheadline_delivery_time) with non-empty \"text\" under the main offer.",
            )


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
        bbox_m = _sanitize_bbox(
            item.get("bbox"),
            warnings,
            i,
            role,
            pixel_w=rw if rw > 0 else None,
            pixel_h=rh if rh > 0 else None,
        )
        if bbox_m is None:
            continue
        gc = _coerce_confidence(item.get("confidence"))
        gr = str(item.get("reason", "") or "")
        children = _parse_group_children(
            role,
            item.get("children"),
            warnings,
            px_w=rw,
            px_h=rh,
            group_idx=i,
        )
        groups_out.append(
            TextZoneGroupItem(role=role, bbox=bbox_m, confidence=gc, reason=gr, children=children),
        )

    groups_out = _dedupe_text_zone_group_items(groups_out, warnings)
    _warn_headline_group_missing_headline(groups_out, warnings)
    _warn_missing_legal_text_top_level(groups_out, warnings)
    _warn_headline_group_missing_subheadline_support(groups_out, warnings)
    _warn_brand_group_missing_word_children(groups_out, warnings)

    timings["parse_qwen_json"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_total0

    group_roles = [g.role for g in groups_out]
    child_counts = [len(g.children) for g in groups_out]
    logger.info(
        "pipeline_v2 analyze_text_zone_visual endpoint=/api/v2/analyze-text-zone-visual "
        "run_id=%s original_image=%dx%d resized_image=%dx%d qwen_elapsed_seconds=%.4f "
        "orientation=%r zone_type=%r text_zone_group_count=%d groups=%r child_counts_per_group=%s "
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
        child_counts,
        len(warnings),
        timings["total"],
    )
    for g in groups_out:
        logger.info(
            "pipeline_v2 analyze_text_zone_visual group role=%r confidence=%.4f bbox=%s child_count=%d",
            g.role,
            g.confidence,
            g.bbox.model_dump(),
            len(g.children),
        )
        for c in g.children:
            logger.info(
                "pipeline_v2 analyze_text_zone_visual   child role=%r confidence=%.4f bbox=%s",
                c.role,
                c.confidence,
                c.bbox.model_dump(),
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
