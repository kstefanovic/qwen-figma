from __future__ import annotations

ZONE_TYPES: frozenset[str] = frozenset(
    {
        "left_txt_right_img",
        "right_txt_left_img",
        "upper_txt_lower_img",
        "upper_img_lower_txt",
        "full_bg_txt_left_img_right",
        "full_bg_txt_center_img_side",
        "full_bg_txt_overlay_img",
        "wide_left_brand_center_txt_right_img",
        "wide_left_txt_center_info_right_img",
        "price_left_product_right",
        "price_top_product_bottom",
        "offer_left_product_right",
        "offer_center_product_side",
        "brand_top_txt_left_img_right",
        "brand_top_txt_center_img_right",
        "brand_top_img_upper_txt_lower",
        "img_dominant_txt_overlay",
        "img_dominant_txt_bottom",
        "img_dominant_txt_top",
        "two_panel_vertical_split",
        "two_panel_horizontal_split",
        "repeated_card_grid",
        "txt_only_center",
        "txt_only_left",
        "txt_on_background",
        "background_only",
        "mixed_complex",
        "unknown",
    }
)


def is_allowed_zone_type(value: str) -> bool:
    s = (value or "").strip()
    return s in ZONE_TYPES
