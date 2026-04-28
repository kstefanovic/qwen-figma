from __future__ import annotations

ZONE_TYPES: frozenset[str] = frozenset(
    {
        "left_text_right_image",
        "upper_image_lower_text",
        "whole_text_no_image",
        "upper_text_mid_image_lower_text",
    }
)

ORIENTATIONS: frozenset[str] = frozenset({"landscape", "wide", "portrait"})


def deterministic_orientation(width: int, height: int) -> str:
    """Same rules as Qwen service v2 (pixel dimensions)."""
    if width < 1 or height < 1:
        return "landscape"
    r = width / float(height)
    if r > 3.0:
        return "wide"
    if width > height:
        return "landscape"
    if width < height:
        return "portrait"
    return "landscape"


def is_allowed_zone_type(value: str) -> bool:
    s = (value or "").strip()
    return s in ZONE_TYPES


def is_allowed_orientation(value: str) -> bool:
    s = (value or "").strip()
    return s in ORIENTATIONS
