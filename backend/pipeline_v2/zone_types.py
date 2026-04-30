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
    {
        "brand_group",
        "headline_group",
        "legal_text_group",
        "age_badge_group",
        "hero_image_group",
        "star_group",
        "glow_group",
        "bg_shape_group",
    },
)

# Top-level group JSON may use legacy names; normalize before validation.
TEXT_ZONE_LEGACY_GROUP_ALIASES: dict[str, str] = {
    "logo_group": "brand_group",
    "age_badge": "age_badge_group",
    "legal_text": "legal_text_group",
}

TEXT_ZONE_BRAND_CHILD_ROLES: frozenset[str] = frozenset(
    {
        "logo",
        "logo_back",
        "logo_fore",
        "brand_name",
        "brand_name_first",
        "brand_name_second",
    },
)

TEXT_ZONE_HEADLINE_CHILD_ROLES: frozenset[str] = frozenset(
    {
        "headline",
        "subheadline",
        "subheadline_delivery_time",
        "subheadline_weight",
        "product_name",
        "subheadline_discount",
    },
)

TEXT_ZONE_LEGAL_CHILD_ROLES: frozenset[str] = frozenset({"legal_text"})

TEXT_ZONE_AGE_BADGE_CHILD_ROLES: frozenset[str] = frozenset({"age_badge"})
TEXT_ZONE_HERO_IMAGE_CHILD_ROLES: frozenset[str] = frozenset({"hero_image", "here_image"})
TEXT_ZONE_STAR_CHILD_ROLES: frozenset[str] = frozenset({"star_1", "star_2"})
TEXT_ZONE_GLOW_CHILD_ROLES: frozenset[str] = frozenset({"glow_1", "glow_2"})
TEXT_ZONE_BG_SHAPE_CHILD_ROLES: frozenset[str] = frozenset({"bg_shape"})

TEXT_ZONE_LOGO_CHILD_ROLES: frozenset[str] = frozenset({"logo", "logo_back", "logo_fore"})

# Long Cyrillic disclaimers (seller, address, ОГРН) need a higher cap than hero lines.
TEXT_ZONE_CHILD_TEXT_MAX_CHARS: int = 8000


def text_zone_child_may_omit_transcribed_text(parent_role: str, child_role: str) -> bool:
    """Visual/non-text roles omit transcribed text. Legal/text roles must carry OCR."""
    role = (child_role or "").strip()
    return role in (
        TEXT_ZONE_LOGO_CHILD_ROLES
        | TEXT_ZONE_HERO_IMAGE_CHILD_ROLES
        | TEXT_ZONE_STAR_CHILD_ROLES
        | TEXT_ZONE_GLOW_CHILD_ROLES
        | TEXT_ZONE_BG_SHAPE_CHILD_ROLES
    )

TEXT_ZONE_CHILD_ROLE_ALIASES: dict[str, str] = {
    "subheadlne_product_name": "product_name",
    "subheadine_product_name": "product_name",
    "subheadline_delivery": "subheadline_delivery_time",
}


def canonical_text_zone_child_role(role_raw: str) -> str:
    s = (role_raw or "").strip()
    return TEXT_ZONE_CHILD_ROLE_ALIASES.get(s, s)


def is_allowed_text_zone_child_for_parent(parent_role: str, child_role_raw: str) -> bool:
    c = canonical_text_zone_child_role(child_role_raw)
    if parent_role == "brand_group":
        return c in TEXT_ZONE_BRAND_CHILD_ROLES
    if parent_role == "headline_group":
        return c in TEXT_ZONE_HEADLINE_CHILD_ROLES
    if parent_role == "legal_text_group":
        return c in TEXT_ZONE_LEGAL_CHILD_ROLES
    if parent_role == "age_badge_group":
        return c in TEXT_ZONE_AGE_BADGE_CHILD_ROLES
    if parent_role == "hero_image_group":
        return c in TEXT_ZONE_HERO_IMAGE_CHILD_ROLES
    if parent_role == "star_group":
        return c in TEXT_ZONE_STAR_CHILD_ROLES
    if parent_role == "glow_group":
        return c in TEXT_ZONE_GLOW_CHILD_ROLES
    if parent_role == "bg_shape_group":
        return c in TEXT_ZONE_BG_SHAPE_CHILD_ROLES
    return False


def text_zone_child_sort_key(parent_role: str, child_role: str) -> int:
    if parent_role == "brand_group":
        order = (
            "logo",
            "logo_back",
            "logo_fore",
            "brand_name",
            "brand_name_first",
            "brand_name_second",
        )
    elif parent_role == "headline_group":
        order = (
            "headline",
            "subheadline",
            "subheadline_delivery_time",
            "subheadline_weight",
            "product_name",
            "subheadline_discount",
        )
    elif parent_role == "legal_text_group":
        order = ("legal_text",)
    elif parent_role == "age_badge_group":
        order = ("age_badge",)
    elif parent_role == "hero_image_group":
        order = ("hero_image", "here_image")
    elif parent_role == "star_group":
        order = ("star_1", "star_2")
    elif parent_role == "glow_group":
        order = ("glow_1", "glow_2")
    elif parent_role == "bg_shape_group":
        order = ("bg_shape",)
    else:
        order = ()
    try:
        return order.index(child_role)
    except ValueError:
        return 999


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


def canonical_text_zone_group_role(role_raw: str) -> str:
    s = (role_raw or "").strip()
    return TEXT_ZONE_LEGACY_GROUP_ALIASES.get(s, s)


def is_allowed_text_zone_role(value: str) -> bool:
    return canonical_text_zone_group_role((value or "").strip()) in TEXT_ZONE_GROUP_ROLES
