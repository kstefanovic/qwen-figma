from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

from env_load import default_qwen_base_url

from backend.text_zone_age_badge_guard import should_strip_age_badge_group_vs_brand

from backend.pipeline_v2.image_utils import decode_banner_raster, resize_and_encode_for_zone_qwen
from backend.pipeline_v2.paddle_ocr_bbox_refine import refine_text_zone_bboxes_with_paddle_ocr
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
    canonical_text_zone_group_role,
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
    raw = (role or "").strip()
    s = canonical_text_zone_group_role(raw)
    if raw != s:
        warnings.append(f"groups[{idx}]: role {raw!r} normalized to {s!r}")
    if not is_allowed_text_zone_role(s):
        return None
    return s


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
    if "legal_text_group" not in roles:
        warnings.append(
            "No legal_text_group top-level group. Commercial banners normally include disclaimer/footer micro-copy "
            "(e.g. Реклама, ООО, ИНН) — re-check the bottom ~35% of the canvas and lower corners.",
        )


def _warn_age_badge_group_incomplete(
    groups: list[TextZoneGroupItem],
    warnings: list[str],
) -> None:
    for g in groups:
        if g.role != "age_badge_group":
            continue
        if not g.children:
            warnings.append(
                "age_badge_group present but children missing or empty; add one child role age_badge with "
                "text like \"0+\" and a tight bbox.",
            )
            continue
        if not any(c.role == "age_badge" and (c.text or "").strip() for c in g.children):
            warnings.append(
                "age_badge_group present but no child with role age_badge and non-empty text; transcribe the "
                "badge (e.g. 0+).",
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


def _maybe_strip_hallucinated_age_badge_on_brand_row(
    groups: list[TextZoneGroupItem],
    warnings: list[str],
) -> None:
    brand_bbox = next((g.bbox for g in groups if g.role == "brand_group"), None)
    if brand_bbox is None:
        return
    drop = False
    for g in groups:
        if g.role != "age_badge_group":
            continue
        for c in g.children:
            if c.role == "age_badge" and should_strip_age_badge_group_vs_brand(brand_bbox, c.bbox):
                drop = True
                break
        if drop:
            break
    if not drop:
        return
    warnings.append(
        "Removed age_badge_group: age_badge bbox matches the brand row size/placement "
        "(likely hallucinated N+; real marks are usually a much smaller plate elsewhere).",
    )
    groups[:] = [g for g in groups if g.role != "age_badge_group"]


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
    order = ("brand_group", "headline_group", "age_badge_group", "legal_text_group")
    return [by_role[r] for r in order if r in by_role]


_NON_OVERLAP_GROUP_ROLES = frozenset({"brand_group", "headline_group", "legal_text_group"})
_BRAND_NAME_CHILD_ROLES = frozenset({"brand_name", "brand_name_first", "brand_name_second"})
_STRUCTURAL_GROUP_ORDER = (
    "brand_group",
    "headline_group",
    "legal_text_group",
    "age_badge_group",
    "hero_image_group",
    "star_group",
    "glow_group",
    "bg_shape_group",
)


def _bbox_union_for_children(children: list[TextZoneChildItem]) -> NormalizedBbox | None:
    if not children:
        return None
    x0 = min(c.bbox.x for c in children)
    y0 = min(c.bbox.y for c in children)
    x1 = max(c.bbox.x + c.bbox.width for c in children)
    y1 = max(c.bbox.y + c.bbox.height for c in children)
    return _clamp_norm_bbox_model(x0, y0, x1 - x0, y1 - y0)


def _placeholder_bbox() -> NormalizedBbox:
    # Positive dimensions are required by the schema; this is visually negligible.
    return NormalizedBbox(x=0.0, y=0.0, width=1e-6, height=1e-6)


def _preferred_group_bbox(g: TextZoneGroupItem) -> NormalizedBbox:
    if g.role == "brand_group":
        name_children = [c for c in g.children if c.role in _BRAND_NAME_CHILD_ROLES]
        return _bbox_union_for_children(name_children) or g.bbox
    if g.role in ("headline_group", "legal_text_group"):
        return _bbox_union_for_children(g.children) or g.bbox
    return g.bbox


def _bbox_overlaps(a: NormalizedBbox, b: NormalizedBbox) -> bool:
    ax1 = a.x + a.width
    ay1 = a.y + a.height
    bx1 = b.x + b.width
    by1 = b.y + b.height
    return min(ax1, bx1) > max(a.x, b.x) and min(ay1, by1) > max(a.y, b.y)


def _bbox_center_y(b: NormalizedBbox) -> float:
    return b.y + b.height / 2.0


def _bbox_bottom(b: NormalizedBbox) -> float:
    return b.y + b.height


def _replace_group_bbox(
    g: TextZoneGroupItem,
    bbox: NormalizedBbox,
    *,
    reason_suffix: str | None = None,
    sync_single_legal_child: bool = False,
) -> TextZoneGroupItem:
    reason = g.reason
    if reason_suffix and reason_suffix not in reason:
        reason = (reason + f" [{reason_suffix}]").strip()
    children = g.children
    if sync_single_legal_child and g.role == "legal_text_group" and len(children) == 1:
        child = children[0]
        if child.role == "legal_text":
            child_reason = child.reason
            if reason_suffix and reason_suffix not in child_reason:
                child_reason = (child_reason + f" [{reason_suffix}]").strip()
            children = [
                TextZoneChildItem(
                    role=child.role,
                    text=child.text,
                    bbox=bbox,
                    confidence=child.confidence,
                    reason=child_reason,
                ),
            ]
    return TextZoneGroupItem(
        role=g.role,
        bbox=bbox,
        confidence=g.confidence,
        reason=reason,
        children=children,
    )


def _normalize_text_zone_group_bboxes(
    groups: list[TextZoneGroupItem],
    warnings: list[str],
) -> list[TextZoneGroupItem]:
    """
    Top-level groups must not overlap each other, except age_badge_group is ignored.
    Child bboxes may overlap within their own group.
    """
    by_role = {g.role: g for g in groups}
    changed = 0

    for role in _NON_OVERLAP_GROUP_ROLES:
        g = by_role.get(role)
        if g is None:
            continue
        bbox = _preferred_group_bbox(g)
        if bbox.model_dump() != g.bbox.model_dump():
            by_role[role] = _replace_group_bbox(g, bbox, reason_suffix="group bbox normalized")
            changed += 1

    cared = [g for g in by_role.values() if g.role in _NON_OVERLAP_GROUP_ROLES]
    for _ in range(4):
        fixed_this_round = 0
        cared = [g for g in by_role.values() if g.role in _NON_OVERLAP_GROUP_ROLES]
        for i in range(len(cared)):
            for j in range(i + 1, len(cared)):
                a = by_role.get(cared[i].role)
                b = by_role.get(cared[j].role)
                if a is None or b is None or not _bbox_overlaps(a.bbox, b.bbox):
                    continue

                # Legal copy is identified by smallest text rows. If it conflicts, preserve legal
                # and trim the other top-level group away from it.
                if a.role == "legal_text_group" or b.role == "legal_text_group":
                    legal = a if a.role == "legal_text_group" else b
                    other = b if legal is a else a
                    if _bbox_center_y(other.bbox) <= _bbox_center_y(legal.bbox):
                        new_other = _clamp_norm_bbox_model(
                            other.bbox.x,
                            other.bbox.y,
                            other.bbox.width,
                            legal.bbox.y - other.bbox.y,
                        )
                    else:
                        legal_bottom = _bbox_bottom(legal.bbox)
                        new_other = _clamp_norm_bbox_model(
                            other.bbox.x,
                            legal_bottom,
                            other.bbox.width,
                            _bbox_bottom(other.bbox) - legal_bottom,
                        )
                    if new_other is not None:
                        by_role[other.role] = _replace_group_bbox(
                            other,
                            new_other,
                            reason_suffix="group bbox de-overlapped",
                            sync_single_legal_child=False,
                        )
                        changed += 1
                        fixed_this_round += 1
                    continue

                upper, lower = (a, b) if _bbox_center_y(a.bbox) <= _bbox_center_y(b.bbox) else (b, a)
                upper_bottom = _bbox_bottom(upper.bbox)
                lower_bottom = _bbox_bottom(lower.bbox)
                if lower_bottom - upper_bottom > 1e-4:
                    new_lower = _clamp_norm_bbox_model(
                        lower.bbox.x,
                        upper_bottom,
                        lower.bbox.width,
                        lower_bottom - upper_bottom,
                    )
                    if new_lower is not None:
                        by_role[lower.role] = _replace_group_bbox(
                            lower,
                            new_lower,
                            reason_suffix="group bbox de-overlapped",
                            sync_single_legal_child=True,
                        )
                        changed += 1
                        fixed_this_round += 1
                elif lower.bbox.y - upper.bbox.y > 1e-4:
                    new_upper = _clamp_norm_bbox_model(
                        upper.bbox.x,
                        upper.bbox.y,
                        upper.bbox.width,
                        lower.bbox.y - upper.bbox.y,
                    )
                    if new_upper is not None:
                        by_role[upper.role] = _replace_group_bbox(
                            upper,
                            new_upper,
                            reason_suffix="group bbox de-overlapped",
                            sync_single_legal_child=True,
                        )
                        changed += 1
                        fixed_this_round += 1
        if fixed_this_round == 0:
            break

    if changed:
        warnings.append(
            f"Group bbox normalization: updated {changed} top-level bbox(es); "
            "brand/headline/legal groups do not overlap (age_badge ignored).",
        )
    return [by_role[g.role] for g in groups if g.role in by_role]


def _structural_child(role: str) -> TextZoneChildItem:
    return TextZoneChildItem(role=role, text="", bbox=_placeholder_bbox(), confidence=0.0, reason="Structural placeholder")


def _logo_structural_child(children: list[TextZoneChildItem]) -> TextZoneChildItem:
    existing_logo = next((c for c in children if c.role == "logo"), None)
    logo_fore = next((c for c in children if c.role == "logo_fore"), None)
    logo_back = next((c for c in children if c.role == "logo_back"), None)
    nested = [c for c in (logo_fore, logo_back) if c is not None]
    bbox = _bbox_union_for_children(nested) or (existing_logo.bbox if existing_logo is not None else _placeholder_bbox())
    return TextZoneChildItem(
        role="logo",
        text="",
        bbox=bbox,
        confidence=max([c.confidence for c in nested] + ([existing_logo.confidence] if existing_logo else [0.0])),
        reason=(existing_logo.reason if existing_logo is not None else "Logo structural parent"),
        children=nested,
    )


def _infer_hero_bbox_from_zone(groups: list[TextZoneGroupItem], zone_type: str) -> NormalizedBbox | None:
    text_groups = [g for g in groups if g.role in _NON_OVERLAP_GROUP_ROLES]
    if not text_groups:
        return None
    zt = (zone_type or "").strip()
    if zt == "upper_image_lower_text":
        top_text_y = min(g.bbox.y for g in text_groups)
        if top_text_y > 0.08:
            return _clamp_norm_bbox_model(0.0, 0.0, 1.0, top_text_y)
        return None
    if zt == "left_text_right_image":
        min_x = min(g.bbox.x for g in text_groups)
        max_x = max(g.bbox.x + g.bbox.width for g in text_groups)
        text_center_x = (min_x + max_x) / 2.0
        if text_center_x < 0.5 and max_x < 0.92:
            return _clamp_norm_bbox_model(max_x, 0.0, 1.0 - max_x, 1.0)
        if text_center_x >= 0.5 and min_x > 0.08:
            return _clamp_norm_bbox_model(0.0, 0.0, min_x, 1.0)
        return None
    if zt == "upper_text_mid_image_lower_text":
        ordered = sorted(text_groups, key=lambda g: g.bbox.y)
        if len(ordered) < 2:
            return None
        upper_bottom = min(0.98, max(g.bbox.y + g.bbox.height for g in ordered[:-1]))
        lower_top = ordered[-1].bbox.y
        if lower_top - upper_bottom > 0.08:
            return _clamp_norm_bbox_model(0.0, upper_bottom, 1.0, lower_top - upper_bottom)
    return None


def _enrich_text_zone_structure(
    groups: list[TextZoneGroupItem],
    warnings: list[str],
    *,
    zone_type: str,
) -> list[TextZoneGroupItem]:
    by_role = {g.role: g for g in groups}

    brand = by_role.get("brand_group")
    if brand is not None:
        first = next((c for c in brand.children if c.role == "brand_name_first"), None)
        single = next((c for c in brand.children if c.role == "brand_name"), None)
        second = next((c for c in brand.children if c.role == "brand_name_second"), None)
        logo = _logo_structural_child(brand.children)
        brand_children = [c for c in (first or single, logo, second) if c is not None]
        by_role["brand_group"] = TextZoneGroupItem(
            role=brand.role,
            bbox=brand.bbox,
            confidence=brand.confidence,
            reason=brand.reason,
            children=brand_children,
        )

    headline = by_role.get("headline_group")
    if headline is not None:
        head = next((c for c in headline.children if c.role == "headline"), None)
        sub = next((c for c in headline.children if c.role == "subheadline"), None)
        if sub is None:
            sub = next(
                (
                    c
                    for c in headline.children
                    if c.role
                    in (
                        "subheadline_delivery_time",
                        "subheadline_weight",
                        "product_name",
                        "subheadline_discount",
                    )
                ),
                None,
            )
            if sub is not None:
                sub = TextZoneChildItem(
                    role="subheadline",
                    text=sub.text,
                    bbox=sub.bbox,
                    confidence=sub.confidence,
                    reason=(sub.reason + " [structural role: subheadline]").strip(),
                    children=sub.children,
                )
        headline_children = [c for c in (head, sub) if c is not None]
        by_role["headline_group"] = TextZoneGroupItem(
            role=headline.role,
            bbox=headline.bbox,
            confidence=headline.confidence,
            reason=headline.reason,
            children=headline_children,
        )

    hero = by_role.get("hero_image_group")
    if hero is None or not hero.children:
        inferred_hero_bbox = _infer_hero_bbox_from_zone(list(by_role.values()), zone_type)
        if inferred_hero_bbox is not None:
            hero_child = TextZoneChildItem(
                role="hero_image",
                text="",
                bbox=inferred_hero_bbox,
                confidence=0.55,
                reason=f"Inferred from zone_type={zone_type!r} image zone",
            )
            by_role["hero_image_group"] = TextZoneGroupItem(
                role="hero_image_group",
                bbox=inferred_hero_bbox,
                confidence=0.55,
                reason=f"Inferred from zone_type={zone_type!r} image zone",
                children=[hero_child],
            )
            warnings.append("Hero image fallback: added hero_image from zone_type image region.")

    for role in ("hero_image_group", "star_group", "glow_group", "bg_shape_group"):
        if role not in by_role:
            by_role[role] = TextZoneGroupItem(
                role=role,
                bbox=_placeholder_bbox(),
                confidence=0.0,
                reason="Structural placeholder; not detected by current text-zone logic",
                children=[],
            )
        elif not by_role[role].children:
            g = by_role[role]
            by_role[role] = TextZoneGroupItem(
                role=g.role,
                bbox=_placeholder_bbox(),
                confidence=g.confidence,
                reason=g.reason,
                children=[],
            )

    warnings.append("Structural enrichment: normalized text_zone groups to requested schema.")
    return [by_role[r] for r in _STRUCTURAL_GROUP_ORDER if r in by_role]


def analyze_text_zone_visual_from_banner_bytes(
    raw_banner_bytes: bytes,
    *,
    qwen_base_url: str | None = None,
    run_id: str | None = None,
) -> AnalyzeTextZoneVisualResponse:
    """
    One Qwen ``POST /analyze-text-zone-visual`` call: orientation, zone_type, text_zone.groups.
    Banner PNG only — no raw JSON, atlas, or Figma IDs.
    """
    t_total0 = time.perf_counter()
    timings: dict[str, float] = {}
    rid = (run_id or "").strip() or str(uuid.uuid4())
    warnings: list[str] = []

    t0 = time.perf_counter()
    logger.info("pipeline_v2 analyze_text_zone_visual receive_request run_id=%s", rid)
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
    qwen_call_count = 1
    dbg = data.get("debug")
    if isinstance(dbg, dict):
        try:
            qwen_call_count = max(1, int(dbg.get("qwen_call_count") or 1))
        except (TypeError, ValueError):
            qwen_call_count = 1

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
    _maybe_strip_hallucinated_age_badge_on_brand_row(groups_out, warnings)
    _warn_headline_group_missing_headline(groups_out, warnings)
    _warn_missing_legal_text_top_level(groups_out, warnings)
    _warn_headline_group_missing_subheadline_support(groups_out, warnings)
    _warn_brand_group_missing_word_children(groups_out, warnings)
    _warn_age_badge_group_incomplete(groups_out, warnings)
    groups_out = refine_text_zone_bboxes_with_paddle_ocr(image, groups_out, warnings)
    groups_out = _normalize_text_zone_group_bboxes(groups_out, warnings)
    groups_out = _enrich_text_zone_structure(groups_out, warnings, zone_type=zone_type)

    timings["parse_qwen_json"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_total0

    group_roles = [g.role for g in groups_out]
    child_counts = [len(g.children) for g in groups_out]
    logger.info(
        "pipeline_v2 analyze_text_zone_visual endpoint=/api/v2/analyze-text-zone-visual "
        "run_id=%s original_image=%dx%d resized_image=%dx%d qwen_elapsed_seconds=%.4f "
        "orientation=%r zone_type=%r text_zone_group_count=%d groups=%r child_counts_per_group=%s "
        "validation_warning_count=%d total_seconds=%.4f",
        rid,
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
        run_id=rid,
        mode="text_zone_visual",
        orientation=orientation,
        zone_type=zone_type,
        confidence=confidence,
        reason=reason,
        text_zone=TextZoneVisual(groups=groups_out),
        debug=AnalyzeTextZoneVisualDebug(
            qwen_call_count=qwen_call_count,
            elapsed_seconds=round(timings["total"], 4),
            validation_warnings=warnings,
        ),
    )
