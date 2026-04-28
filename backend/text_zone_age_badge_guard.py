"""
Heuristics to drop VLM-hallucinated ``age_badge`` boxes that sit on the brand row
(e.g. model labels the top-left brand corner as ``0+`` when no rating mark exists).

Lives under ``backend/`` (not ``pipeline_v2/``) so imports avoid ``pipeline_v2`` package init.
"""

from __future__ import annotations

from typing import Any


def _as_xywh(b: Any) -> tuple[float, float, float, float] | None:
    if b is None:
        return None
    if hasattr(b, "x") and hasattr(b, "width"):
        try:
            return (float(b.x), float(b.y), float(b.width), float(b.height))
        except (TypeError, ValueError):
            return None
    if isinstance(b, dict):
        try:
            x = float(b.get("x", 0) or 0)
            y = float(b.get("y", 0) or 0)
            w = float(b.get("width", 0) or 0)
            h = float(b.get("height", 0) or 0)
        except (TypeError, ValueError):
            return None
        if w <= 1e-9 or h <= 1e-9:
            return None
        return (x, y, w, h)
    return None


def _intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    iw = max(0.0, x1 - x0)
    ih = max(0.0, y1 - y0)
    return iw * ih


def age_badge_bbox_likely_hallucinated_on_brand_row(
    age_xywh: tuple[float, float, float, float],
    brand_xywh: tuple[float, float, float, float],
) -> bool:
    """
    True when the age badge box is implausibly merged with the brand strip
    (same band height as the brand row and almost entirely inside it).
    """
    ax, ay, aw, ah = age_xywh
    bx, by, bw, bh = brand_xywh
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return False
    age_area = aw * ah
    inter = _intersection_area(age_xywh, brand_xywh)
    frac_age_covered_by_overlap = inter / age_area if age_area > 1e-12 else 0.0
    cx = ax + 0.5 * aw
    cy = ay + 0.5 * ah
    center_in_brand = (bx <= cx <= bx + bw) and (by <= cy <= by + bh)
    if not center_in_brand:
        return False
    tall_vs_brand_band = ah >= 0.72 * bh
    if not tall_vs_brand_band:
        return False
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    y_overlap_frac = iy / min(ah, bh) if min(ah, bh) > 1e-9 else 0.0
    if y_overlap_frac < 0.78:
        return False
    if frac_age_covered_by_overlap >= 0.82:
        return True
    iou = inter / (age_area + bw * bh - inter) if (age_area + bw * bh - inter) > 1e-12 else 0.0
    if iou >= 0.18 and ah >= 0.85 * bh:
        return True
    return False


def should_strip_age_badge_group_vs_brand(
    brand_bbox: Any,
    age_child_bbox: Any,
) -> bool:
    """Convenience: dict, NormalizedBbox, or None."""
    b_brand = _as_xywh(brand_bbox)
    b_age = _as_xywh(age_child_bbox)
    if b_brand is None or b_age is None:
        return False
    return age_badge_bbox_likely_hallucinated_on_brand_row(b_age, b_brand)
