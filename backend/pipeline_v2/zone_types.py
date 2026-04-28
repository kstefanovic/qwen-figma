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

TEXT_ZONE_GROUP_ROLES: frozenset[str] = frozenset(
    {"brand_group", "headline_group", "legal_text"},
)


def deterministic_orientation(width: int, height: int) -> str:
    """Same rules as Qwen service v2 (pixel dimensions)."""
    if width < 1 or height < 1:
        return "landscape"
    r = width / float(height)
    if r >= 3.0:
        return "wide"
    if width > height and r < 3.0:
        return "landscape"
    if height > width:
        return "portrait"
    return "landscape"


def is_allowed_zone_type(value: str) -> bool:
    s = (value or "").strip()
    return s in ZONE_TYPES


def is_allowed_orientation(value: str) -> bool:
    s = (value or "").strip()
    return s in ORIENTATIONS


def is_allowed_text_zone_role(value: str) -> bool:
    s = (value or "").strip()
    return s in TEXT_ZONE_GROUP_ROLES
