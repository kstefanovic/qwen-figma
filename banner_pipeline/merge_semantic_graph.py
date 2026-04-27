from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

ZONE_DEFAULT_ID = "zone_default"

from banner_pipeline.build_candidates import CandidateBundle, SemanticCandidate
from banner_pipeline.collapse_groups import CollapsedNode
from banner_pipeline.heuristics import HeuristicAnnotatedCandidate, HeuristicBundle
from banner_pipeline.qwen_annotator import (
    BannerAnnotation,
    CandidateAnnotation,
    GroupAnnotation,
)
from schemas.enums import (
    AnchorType,
    ConstraintType,
    ElementRole,
    FunctionalType,
    GroupRole,
    ImportanceLevel,
    InternalLayout,
    LayoutPattern,
    RelationType,
    ZoneRole,
)
from schemas.semantic_graph import (
    AdaptationPolicy,
    BBox,
    Constraint,
    Element,
    Group,
    Metadata,
    Relation,
    SemanticGraph,
    SourceInfo,
    TextFeatures,
    Zone,
)


@dataclass
class MergeConfig:
    default_brand_family: str = "unknown_brand"
    default_language: str = "unknown"
    default_category: str = "unknown"
    qwen_low_confidence_threshold: float = 0.55
    keep_unknown_candidates: bool = True
    min_candidate_w_norm: float = 0.01
    min_candidate_h_norm: float = 0.01
    dedupe_iou_threshold: float = 0.95


def _safe_layout_pattern(value: str | None) -> LayoutPattern:
    if value is None:
        return LayoutPattern.UNKNOWN
    try:
        return LayoutPattern(value)
    except Exception:
        return LayoutPattern.UNKNOWN


def _safe_zone_role(value: str | None) -> ZoneRole:
    if value is None:
        return ZoneRole.OVERLAY_ZONE
    try:
        return ZoneRole(value)
    except Exception:
        return ZoneRole.OVERLAY_ZONE


def _safe_group_role(value: str | None) -> GroupRole:
    if value is None:
        return GroupRole.UNKNOWN
    try:
        return GroupRole(value)
    except Exception:
        return GroupRole.UNKNOWN


def _safe_element_role(value: str | None) -> ElementRole:
    if value is None:
        return ElementRole.UNKNOWN
    try:
        return ElementRole(value)
    except Exception:
        return ElementRole.UNKNOWN


def _safe_importance(value: str | None) -> ImportanceLevel:
    if value is None:
        return ImportanceLevel.MEDIUM
    try:
        return ImportanceLevel(value)
    except Exception:
        return ImportanceLevel.MEDIUM


def _safe_anchor(value: str | None) -> AnchorType:
    if value is None:
        return AnchorType.FREE
    try:
        return AnchorType(value)
    except Exception:
        return AnchorType.FREE


def _safe_internal_layout(value: str | None) -> InternalLayout:
    if value is None:
        return InternalLayout.FREEFORM
    try:
        return InternalLayout(value)
    except Exception:
        return InternalLayout.FREEFORM


def _safe_functional_type(value: str | None) -> FunctionalType:
    if value is None:
        return FunctionalType.FUNCTIONAL
    try:
        return FunctionalType(value)
    except Exception:
        return FunctionalType.FUNCTIONAL


_MIN_BBOX_WH = 1e-6


def _clamp_unit_bbox(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    x = max(0.0, min(1.0, float(x)))
    y = max(0.0, min(1.0, float(y)))
    w = max(_MIN_BBOX_WH, float(w))
    h = max(_MIN_BBOX_WH, float(h))
    w = min(w, 1.0 - x)
    h = min(h, 1.0 - y)
    w = max(_MIN_BBOX_WH, w)
    h = max(_MIN_BBOX_WH, h)
    return x, y, w, h


def _bbox_from_candidate(candidate: SemanticCandidate) -> BBox:
    x, y, w, h = _clamp_unit_bbox(
        float(candidate.x_norm),
        float(candidate.y_norm),
        float(candidate.w_norm),
        float(candidate.h_norm),
    )
    return BBox(x=x, y=y, w=w, h=h)


def _bbox_full() -> BBox:
    return BBox(x=0.0, y=0.0, w=1.0, h=1.0)


def _zone_registry_ids(zones: list[Zone]) -> set[str]:
    """Central registry of valid zone ids for the current merge."""
    return {z.id for z in zones}


def _ensure_zone_default_registry(zones: list[Zone]) -> None:
    """
    Guarantee ZONE_DEFAULT_ID exists: full-canvas overlay used when layout/Qwen
    references a zone id that is not in the banner-derived zone list.
    """
    if any(z.id == ZONE_DEFAULT_ID for z in zones):
        return
    zones.append(
        Zone(
            id=ZONE_DEFAULT_ID,
            role=ZoneRole.OVERLAY_ZONE,
            bbox_canvas=_bbox_full(),
            importance=ImportanceLevel.MEDIUM,
        )
    )


def _coerce_zone_id(zone_id: str, registry: set[str], *, log_context: str = "") -> str:
    """
    Map any zone_id to a member of registry. Never trust guessed/Qwen-aligned ids blindly.
    """
    if zone_id in registry:
        return zone_id
    logger.warning(
        "Unknown zone_id from Qwen/layout: %r (%s), fallback applied -> %r",
        zone_id,
        log_context or "merge",
        ZONE_DEFAULT_ID,
    )
    if ZONE_DEFAULT_ID in registry:
        return ZONE_DEFAULT_ID
    if registry:
        fallback = sorted(registry)[0]
        logger.warning("zone_default missing from registry; using first zone %r", fallback)
        return fallback
    raise RuntimeError("merge_semantic_graph: zone registry is empty after _ensure_zone_default_registry")


def _count_words(text: Optional[str]) -> int:
    if not text:
        return 0
    return len([t for t in text.replace("\n", " ").split(" ") if t.strip()])


def _count_lines(text: Optional[str]) -> int:
    if not text:
        return 0
    lines = [line for line in text.splitlines() if line.strip()]
    return len(lines) if lines else 1


def _center_from_candidate(candidate: SemanticCandidate) -> list[float]:
    return [float(candidate.center_x_norm), float(candidate.center_y_norm)]


def _pick_candidate_annotation(candidate_id: str, qwen_candidate_annotations: dict[str, CandidateAnnotation] | None) -> Optional[CandidateAnnotation]:
    if not qwen_candidate_annotations:
        return None
    return qwen_candidate_annotations.get(candidate_id)


def _pick_group_annotation(candidate_id: str, qwen_group_annotations: dict[str, GroupAnnotation] | None) -> Optional[GroupAnnotation]:
    if not qwen_group_annotations:
        return None
    return qwen_group_annotations.get(candidate_id)


def _pick_heuristic_annotation(candidate_id: str, heuristic_bundle: Optional[HeuristicBundle]) -> Optional[HeuristicAnnotatedCandidate]:
    if heuristic_bundle is None:
        return None
    return heuristic_bundle.by_candidate_id.get(candidate_id)


def _resolve_role_hint(heuristic_ann: Optional[HeuristicAnnotatedCandidate], qwen_ann: Optional[CandidateAnnotation], config: MergeConfig) -> str | None:
    heuristic_role = heuristic_ann.final_role_hint if heuristic_ann else None
    qwen_role = qwen_ann.element_role if qwen_ann else None
    qwen_conf = qwen_ann.confidence if qwen_ann else 0.0
    qwen_role_low_info = qwen_role in {"unknown", "text", "image_like", "background"}

    if qwen_role and not qwen_role_low_info and qwen_conf >= config.qwen_low_confidence_threshold:
        return qwen_role
    if heuristic_role:
        return heuristic_role
    return qwen_role


def _resolve_group_hint(heuristic_ann: Optional[HeuristicAnnotatedCandidate], qwen_group_ann: Optional[GroupAnnotation], config: MergeConfig) -> str | None:
    heuristic_group = heuristic_ann.final_group_hint if heuristic_ann else None
    qwen_group = qwen_group_ann.group_role if qwen_group_ann else None
    qwen_conf = qwen_group_ann.confidence if qwen_group_ann else 0.0
    qwen_group_low_info = qwen_group in {"unknown", "text_group"}

    if qwen_group and not qwen_group_low_info and qwen_conf >= config.qwen_low_confidence_threshold:
        return qwen_group
    if heuristic_group:
        return heuristic_group
    return qwen_group


def _resolve_importance(
    heuristic_ann: Optional[HeuristicAnnotatedCandidate],
    qwen_candidate_ann: Optional[CandidateAnnotation],
    qwen_group_ann: Optional[GroupAnnotation],
    config: MergeConfig,
) -> ImportanceLevel:
    heuristic_value = heuristic_ann.final_importance_hint if heuristic_ann else None

    if qwen_candidate_ann and qwen_candidate_ann.confidence >= config.qwen_low_confidence_threshold:
        return _safe_importance(qwen_candidate_ann.importance_level)
    if qwen_group_ann and qwen_group_ann.confidence >= config.qwen_low_confidence_threshold:
        return _safe_importance(qwen_group_ann.importance_level)
    if heuristic_value:
        return _safe_importance(heuristic_value)
    if qwen_candidate_ann:
        return _safe_importance(qwen_candidate_ann.importance_level)
    if qwen_group_ann:
        return _safe_importance(qwen_group_ann.importance_level)
    return ImportanceLevel.MEDIUM


def _adaptation_policy_from_candidate_annotation(candidate_ann: Optional[CandidateAnnotation]) -> AdaptationPolicy:
    if candidate_ann is None:
        return AdaptationPolicy()

    policy = candidate_ann.adaptation_policy or {}
    return AdaptationPolicy(
        preserve_as_unit=bool(policy.get("preserve_as_unit", True)),
        allow_reflow=bool(policy.get("allow_reflow", False)),
        allow_scale=bool(policy.get("allow_scale", True)),
        allow_crop=bool(policy.get("allow_crop", False)),
        allow_shift=bool(policy.get("allow_shift", True)),
        drop_priority=int(policy.get("drop_priority", 0)),
        anchor_type=_safe_anchor(policy.get("anchor_type")),
    )


def _adaptation_policy_from_group_annotation(group_ann: Optional[GroupAnnotation]) -> AdaptationPolicy:
    if group_ann is None:
        return AdaptationPolicy()

    policy = group_ann.adaptation_policy or {}
    return AdaptationPolicy(
        preserve_as_unit=bool(group_ann.preserve_as_unit),
        allow_reflow=bool(policy.get("allow_reflow", False)),
        allow_scale=bool(policy.get("allow_scale", True)),
        allow_crop=bool(policy.get("allow_crop", False)),
        allow_shift=bool(policy.get("allow_shift", True)),
        drop_priority=int(policy.get("drop_priority", 0)),
        anchor_type=_safe_anchor(policy.get("anchor_type")),
    )


def _guess_zone_id_from_position(candidate: SemanticCandidate, layout_pattern: LayoutPattern) -> str:
    x = candidate.center_x_norm
    y = candidate.center_y_norm

    if layout_pattern in {
        LayoutPattern.LEFT_TEXT_RIGHT_IMAGE,
        LayoutPattern.LEFT_TEXT_RIGHT_PRODUCT,
        LayoutPattern.PRICE_LEFT_PRODUCT_RIGHT,
    }:
        if x < 0.55:
            return "zone_text_left"
        return "zone_image_right"

    if layout_pattern in {
        LayoutPattern.TOP_IMAGE_BOTTOM_TEXT_MOBILE,
        LayoutPattern.PRODUCT_DOMINANT_MOBILE,
    }:
        if y < 0.5:
            return "zone_image_top"
        return "zone_text_bottom"

    if layout_pattern == LayoutPattern.TOP_TEXT_BOTTOM_PRODUCT_MOBILE:
        if y < 0.5:
            return "zone_text_top"
        return "zone_product_bottom"

    if layout_pattern in {
        LayoutPattern.CENTERED_TEXT_DECORATIVE_BACKGROUND,
        LayoutPattern.FULL_BACKGROUND_TEXT_OVERLAY,
        LayoutPattern.PROMO_TEXT_ONLY,
        LayoutPattern.CATALOG_PRICE_CARD,
    }:
        return "zone_overlay_global"

    return "zone_overlay_global"


def _default_zone_set(layout_pattern: LayoutPattern) -> list[Zone]:
    if layout_pattern in {
        LayoutPattern.LEFT_TEXT_RIGHT_IMAGE,
        LayoutPattern.LEFT_TEXT_RIGHT_PRODUCT,
        LayoutPattern.PRICE_LEFT_PRODUCT_RIGHT,
    }:
        return [
            Zone(id="zone_text_left", role=ZoneRole.TEXT_ZONE, bbox_canvas=BBox(x=0.0, y=0.0, w=0.55, h=1.0)),
            Zone(id="zone_image_right", role=ZoneRole.IMAGE_ZONE, bbox_canvas=BBox(x=0.55, y=0.0, w=0.45, h=1.0)),
            Zone(id="zone_overlay_global", role=ZoneRole.OVERLAY_ZONE, bbox_canvas=_bbox_full()),
        ]

    return [Zone(id="zone_overlay_global", role=ZoneRole.OVERLAY_ZONE, bbox_canvas=_bbox_full())]


def _zones_from_banner_annotation(banner_ann: Optional[BannerAnnotation], layout_pattern: LayoutPattern) -> list[Zone]:
    if banner_ann is None or not banner_ann.zones:
        return _default_zone_set(layout_pattern)

    zones: list[Zone] = []
    used_ids: set[str] = set()

    for idx, zone_info in enumerate(banner_ann.zones, start=1):
        role = _safe_zone_role(zone_info.get("zone_role"))
        approx_position = str(zone_info.get("approx_position", "full"))

        zone_id = _zone_id_from_role_and_position(role, approx_position, idx)
        if zone_id in used_ids:
            zone_id = f"{zone_id}_{idx}"
        used_ids.add(zone_id)

        bbox = _bbox_from_position_hint(approx_position)
        zones.append(
            Zone(
                id=zone_id,
                role=role,
                bbox_canvas=bbox,
                importance=_safe_importance(zone_info.get("importance_level")),
            )
        )

    if not any(z.role == ZoneRole.OVERLAY_ZONE for z in zones):
        zones.append(Zone(id="zone_overlay_global", role=ZoneRole.OVERLAY_ZONE, bbox_canvas=_bbox_full()))

    return zones


def _zone_id_from_role_and_position(role: ZoneRole, approx_position: str, idx: int) -> str:
    mapping = {
        (ZoneRole.TEXT_ZONE, "left"): "zone_text_left",
        (ZoneRole.TEXT_ZONE, "right"): "zone_text_right",
        (ZoneRole.TEXT_ZONE, "top"): "zone_text_top",
        (ZoneRole.TEXT_ZONE, "bottom"): "zone_text_bottom",
        (ZoneRole.IMAGE_ZONE, "left"): "zone_image_left",
        (ZoneRole.IMAGE_ZONE, "right"): "zone_image_right",
        (ZoneRole.IMAGE_ZONE, "top"): "zone_image_top",
        (ZoneRole.IMAGE_ZONE, "bottom"): "zone_image_bottom",
        (ZoneRole.PRODUCT_ZONE, "left"): "zone_product_left",
        (ZoneRole.PRODUCT_ZONE, "right"): "zone_product_right",
        (ZoneRole.PRODUCT_ZONE, "top"): "zone_product_top",
        (ZoneRole.PRODUCT_ZONE, "bottom"): "zone_product_bottom",
        (ZoneRole.LEGAL_ZONE, "bottom"): "zone_legal_bottom",
        (ZoneRole.BRAND_ZONE, "top"): "zone_brand_top",
        (ZoneRole.OVERLAY_ZONE, "full"): "zone_overlay_global",
        (ZoneRole.BACKGROUND_ZONE, "full"): "zone_background_full",
    }
    return mapping.get((role, approx_position), f"zone_{role.value}_{idx}")


def _bbox_from_position_hint(position: str) -> BBox:
    if position == "left":
        return BBox(x=0.0, y=0.0, w=0.55, h=1.0)
    if position == "right":
        return BBox(x=0.45, y=0.0, w=0.55, h=1.0)
    if position == "top":
        return BBox(x=0.0, y=0.0, w=1.0, h=0.50)
    if position == "bottom":
        return BBox(x=0.0, y=0.50, w=1.0, h=0.50)
    if position == "center":
        return BBox(x=0.15, y=0.15, w=0.70, h=0.70)
    return _bbox_full()


def _candidate_to_group_role(
    candidate: SemanticCandidate,
    heuristic_ann: Optional[HeuristicAnnotatedCandidate],
    qwen_group_ann: Optional[GroupAnnotation],
    qwen_candidate_ann: Optional[CandidateAnnotation],
    config: MergeConfig,
) -> GroupRole:
    resolved = _resolve_group_hint(heuristic_ann, qwen_group_ann, config)

    if resolved is None and qwen_candidate_ann is not None:
        role_to_group = {
            "headline": "headline_group",
            "subheadline": "headline_group",
            "legal": "legal_group",
            "brand": "brand_group",
            "age_badge": "badge_group",
            "badge": "badge_group",
            "decoration": "decoration_group",
            "background": "background_group",
            "price_main": "price_group",
            "price_old": "price_group",
            "price_fraction": "price_group",
            "product_image": "hero_group",
            "hero_photo": "hero_group",
        }
        resolved = role_to_group.get(qwen_candidate_ann.element_role)

    if resolved is None and heuristic_ann is not None:
        resolved = heuristic_ann.final_group_hint

    if resolved is None:
        by_type = {
            "brand": "brand_group",
            "badge": "badge_group",
            "decoration": "decoration_group",
            "background": "background_group",
            "text_group": "headline_group",
            "text": "text_group",
            "image_like": "hero_group",
        }
        resolved = by_type.get(candidate.candidate_type, "unknown")

    return _safe_group_role(resolved)


def _candidate_to_element_role(
    candidate: SemanticCandidate,
    heuristic_ann: Optional[HeuristicAnnotatedCandidate],
    qwen_candidate_ann: Optional[CandidateAnnotation],
    config: MergeConfig,
) -> ElementRole:
    resolved = _resolve_role_hint(heuristic_ann, qwen_candidate_ann, config)

    if resolved in {None, "unknown", "text"}:
        group_hint = heuristic_ann.final_group_hint if heuristic_ann is not None else None
        group_to_element = {
            "headline_group": "headline",
            "legal_group": "legal",
            "brand_group": "brand_mark",
            "badge_group": "age_badge",
            "hero_group": "hero_photo",
            "background_group": "background_shape",
            "text_group": "text_container",
        }
        mapped = group_to_element.get(group_hint)
        if mapped is not None:
            resolved = mapped

    if resolved is None:
        by_type = {
            "text": "unknown",
            "text_group": "text_container",
            "brand": "brand_mark",
            "badge": "badge_text",
            "decoration": "decoration",
            "background": "background_shape",
            "image_like": "hero_photo",
        }
        resolved = by_type.get(candidate.candidate_type, "unknown")

    return _safe_element_role(resolved)


def _should_make_element(candidate: SemanticCandidate) -> bool:
    return candidate.candidate_type in {"text", "brand", "badge", "decoration", "background", "image_like"}


def _text_features_from_candidate(candidate: SemanticCandidate) -> Optional[TextFeatures]:
    if not candidate.text_content:
        return None
    text = candidate.text_content
    return TextFeatures(char_count=len(text), word_count=_count_words(text), line_count=_count_lines(text))


def _element_flags_from_role(role: ElementRole) -> tuple[bool, bool]:
    brand_related_roles = {ElementRole.LOGO_TEXT, ElementRole.LOGO_ICON, ElementRole.BRAND_MARK}
    compliance_roles = {ElementRole.LEGAL, ElementRole.AGE_BADGE}
    return role in brand_related_roles, role in compliance_roles


def _group_anchor_from_role(role: GroupRole) -> AnchorType:
    mapping = {
        GroupRole.BRAND_GROUP: AnchorType.TOP_LEFT,
        GroupRole.BADGE_GROUP: AnchorType.TOP_RIGHT,
        GroupRole.LEGAL_GROUP: AnchorType.BOTTOM_LEFT,
        GroupRole.HEADLINE_GROUP: AnchorType.LEFT_CENTER,
        GroupRole.PRICE_GROUP: AnchorType.LEFT_CENTER,
        GroupRole.DECORATION_GROUP: AnchorType.FREE,
        GroupRole.BACKGROUND_GROUP: AnchorType.FREE,
        GroupRole.PRODUCT_GROUP: AnchorType.RIGHT_CENTER,
        GroupRole.HERO_GROUP: AnchorType.RIGHT_CENTER,
        GroupRole.TEXT_GROUP: AnchorType.LEFT_CENTER,
    }
    return mapping.get(role, AnchorType.FREE)


def _element_anchor_from_role(role: ElementRole) -> AnchorType:
    mapping = {
        ElementRole.HEADLINE: AnchorType.LEFT_CENTER,
        ElementRole.SUBHEADLINE: AnchorType.LEFT_CENTER,
        ElementRole.LEGAL: AnchorType.BOTTOM_LEFT,
        ElementRole.AGE_BADGE: AnchorType.TOP_RIGHT,
        ElementRole.LOGO_TEXT: AnchorType.TOP_LEFT,
        ElementRole.LOGO_ICON: AnchorType.TOP_LEFT,
        ElementRole.BRAND_MARK: AnchorType.TOP_LEFT,
        ElementRole.DECORATION: AnchorType.FREE,
        ElementRole.BACKGROUND_PANEL: AnchorType.FREE,
        ElementRole.BACKGROUND_SHAPE: AnchorType.FREE,
        ElementRole.HERO_PHOTO: AnchorType.RIGHT_CENTER,
        ElementRole.PRODUCT_IMAGE: AnchorType.RIGHT_CENTER,
        ElementRole.PACKSHOT: AnchorType.RIGHT_CENTER,
        ElementRole.PRICE_MAIN: AnchorType.LEFT_CENTER,
        ElementRole.PRICE_OLD: AnchorType.LEFT_CENTER,
        ElementRole.PRICE_FRACTION: AnchorType.LEFT_CENTER,
    }
    return mapping.get(role, AnchorType.FREE)


def _is_single_text_candidate(candidate: SemanticCandidate) -> bool:
    return candidate.candidate_type == "text"


def _is_multi_text_group(candidate: SemanticCandidate) -> bool:
    return candidate.candidate_type == "text_group" and len(candidate.source_node_ids) >= 2


def _collect_multi_text_nodes(bundle: CandidateBundle) -> set[str]:
    ids = set()
    for c in bundle.all_candidates:
        if _is_multi_text_group(c):
            ids.update(c.source_node_ids)
    return ids


def _find_covering_text_group_id(candidate: SemanticCandidate, candidate_bundle: CandidateBundle, candidate_group_map: dict[str, str]) -> str | None:
    if candidate.candidate_type != "text":
        return None
    node_ids = set(candidate.source_node_ids)
    for group_candidate in candidate_bundle.all_candidates:
        if not _is_multi_text_group(group_candidate):
            continue
        if node_ids.issubset(set(group_candidate.source_node_ids)):
            return candidate_group_map.get(group_candidate.candidate_id)
    return None


def _candidate_priority(candidate_type: str) -> int:
    priority = {
        "brand": 100,
        "badge": 90,
        "text_group": 80,
        "text": 70,
        "image_like": 60,
        "background": 50,
        "decoration": 40,
    }
    return priority.get(candidate_type, 0)


def _candidate_iou(a: SemanticCandidate, b: SemanticCandidate) -> float:
    ax1, ay1 = a.x_norm, a.y_norm
    ax2, ay2 = a.x_norm + a.w_norm, a.y_norm + a.h_norm
    bx1, by1 = b.x_norm, b.y_norm
    bx2, by2 = b.x_norm + b.w_norm, b.y_norm + b.h_norm

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h

    area_a = max(0.0, a.w_norm) * max(0.0, a.h_norm)
    area_b = max(0.0, b.w_norm) * max(0.0, b.h_norm)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def _candidate_source_key(candidate: SemanticCandidate) -> tuple[str, ...]:
    return tuple(sorted(candidate.source_node_ids))


def _is_tiny_candidate(candidate: SemanticCandidate, config: MergeConfig) -> bool:
    return candidate.w_norm < config.min_candidate_w_norm or candidate.h_norm < config.min_candidate_h_norm


def _should_drop_left_side_image_like(candidate: SemanticCandidate, layout_pattern: LayoutPattern) -> bool:
    if candidate.candidate_type != "image_like":
        return False
    if layout_pattern in {LayoutPattern.LEFT_TEXT_RIGHT_IMAGE, LayoutPattern.LEFT_TEXT_RIGHT_PRODUCT, LayoutPattern.PRICE_LEFT_PRODUCT_RIGHT}:
        return candidate.center_x_norm < 0.5
    return False


def _dedupe_visual_candidates(candidates: list[SemanticCandidate], config: MergeConfig, layout_pattern: LayoutPattern) -> list[SemanticCandidate]:
    filtered: list[SemanticCandidate] = []
    for c in candidates:
        if _is_tiny_candidate(c, config):
            continue
        if _should_drop_left_side_image_like(c, layout_pattern):
            continue
        filtered.append(c)

    best_by_source: dict[tuple[str, ...], SemanticCandidate] = {}
    for c in filtered:
        key = _candidate_source_key(c)
        prev = best_by_source.get(key)
        if prev is None or _candidate_priority(c.candidate_type) > _candidate_priority(prev.candidate_type):
            best_by_source[key] = c

    exact_deduped = list(best_by_source.values())

    kept: list[SemanticCandidate] = []
    for c in exact_deduped:
        drop = False
        for prev in list(kept):
            iou = _candidate_iou(c, prev)
            if iou < config.dedupe_iou_threshold:
                continue

            if c.candidate_type == "background" and prev.candidate_type == "image_like":
                drop = True
                break
            if c.candidate_type == "image_like" and prev.candidate_type == "background":
                kept.remove(prev)
                break

            if _candidate_priority(c.candidate_type) <= _candidate_priority(prev.candidate_type):
                drop = True
                break
            else:
                kept.remove(prev)
                break

        if not drop:
            kept.append(c)

    return kept


def _scene_semantic_allowed_for_element_role(role: ElementRole, semantic_name: str) -> bool:
    """Reject decoration_* VLM labels on headline/brand/legal/etc. elements."""
    s = (semantic_name or "").strip()
    if not s:
        return False
    sl = s.lower()
    if sl.startswith("decoration_") and sl != "decoration_group" and role != ElementRole.DECORATION:
        return False
    return True


def _index_scene_updates_by_source(updates: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not updates:
        return out
    for item in updates:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("source_figma_id", "") or "").strip()
        if sid:
            out[sid] = item
    return out


def _safe_element_id_suffix(source_figma_id: str) -> str:
    return source_figma_id.replace(":", "_").replace("/", "_")


def _fallback_decoration_member_semantic_name(
    node: CollapsedNode,
    *,
    ordinal: int,
    total: int,
) -> str:
    """When the VLM omits per-member names inside a decoration cluster, infer from geometry/text."""
    txt = (node.text or "").strip() if node.text else ""
    if txt:
        tl = txt.lower()
        if any(x in tl for x in ("+", "18+", "16+", "12+", "0+", "6+", "3+")):
            return "age_badge"
        if any(x in tl for x in ("₽", "$", "€", "%", "руб", "скид", "от ")):
            return "price_group"
        if len(txt) > 72:
            return "legal_text"
        return "headline" if ordinal <= 1 else "subheadline"
    tl = (node.type or "").lower()
    if tl == "star":
        if total <= 1:
            return "decoration_star"
        return f"decoration_star_{ordinal}"
    if tl == "text":
        return "headline" if ordinal <= 1 else "subheadline"
    if tl in {"rectangle", "ellipse", "vector"} and (node.w_norm or 0) > 0.08 and (node.h_norm or 0) > 0.04:
        return "logo_mark" if ordinal <= 2 else "visual_asset"
    if total <= 1:
        return "decoration_sparkle"
    return f"decoration_sparkle_{ordinal}"


def _dedupe_constraints(constraints: list[Constraint]) -> list[Constraint]:
    deduped: list[Constraint] = []
    seen: set[tuple[str, str, str]] = set()
    for c in constraints:
        key = (c.target, c.type.value, str(c.value))
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def merge_semantic_graph(
    *,
    banner_id: str,
    template_id: Optional[str],
    canvas_width: int,
    canvas_height: int,
    raw_figma_frame_id: Optional[str],
    collapsed_nodes: list[CollapsedNode],
    candidate_bundle: CandidateBundle,
    heuristic_bundle: Optional[HeuristicBundle] = None,
    banner_annotation: Optional[BannerAnnotation] = None,
    qwen_candidate_annotations: Optional[dict[str, CandidateAnnotation]] = None,
    qwen_group_annotations: Optional[dict[str, GroupAnnotation]] = None,
    scene_semantic_updates: Optional[list[dict[str, Any]]] = None,
    config: Optional[MergeConfig] = None,
) -> SemanticGraph:
    config = config or MergeConfig()
    collapsed_by_id: dict[str, CollapsedNode] = {n.id: n for n in collapsed_nodes}
    update_by_source = _index_scene_updates_by_source(scene_semantic_updates)

    layout_pattern = _safe_layout_pattern(banner_annotation.layout_pattern if banner_annotation else None)
    zones = _zones_from_banner_annotation(banner_annotation, layout_pattern)
    _ensure_zone_default_registry(zones)
    zone_registry = _zone_registry_ids(zones)

    source = SourceInfo(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        aspect_ratio=canvas_width / canvas_height,
        source_type="figma",
        raw_figma_frame_id=raw_figma_frame_id,
    )

    metadata = Metadata(
        brand_family=config.default_brand_family,
        language=config.default_language,
        category=config.default_category,
        layout_pattern=layout_pattern,
        pattern_confidence=(banner_annotation.pattern_confidence if banner_annotation else 0.0),
    )

    groups: list[Group] = []
    elements: list[Element] = []
    relations: list[Relation] = []
    constraints: list[Constraint] = []

    candidate_group_map: dict[str, str] = {}
    multi_text_node_ids = _collect_multi_text_nodes(candidate_bundle)

    merged_candidates = _dedupe_visual_candidates(list(candidate_bundle.all_candidates), config=config, layout_pattern=layout_pattern)

    for candidate in merged_candidates:
        if _is_single_text_candidate(candidate) and any(node_id in multi_text_node_ids for node_id in candidate.source_node_ids):
            continue

        heuristic_ann = _pick_heuristic_annotation(candidate.candidate_id, heuristic_bundle)
        qwen_group_ann = _pick_group_annotation(candidate.candidate_id, qwen_group_annotations)
        qwen_candidate_ann = _pick_candidate_annotation(candidate.candidate_id, qwen_candidate_annotations)

        group_role = _candidate_to_group_role(candidate, heuristic_ann, qwen_group_ann, qwen_candidate_ann, config)
        if group_role == GroupRole.UNKNOWN and not config.keep_unknown_candidates:
            continue

        zone_id = _guess_zone_id_from_position(candidate, layout_pattern)
        zone_id = _coerce_zone_id(
            zone_id,
            zone_registry,
            log_context=f"group:{candidate.candidate_id}:layout_guess",
        )

        if group_role == GroupRole.BADGE_GROUP:
            zone_id = _coerce_zone_id(
                "zone_overlay_global",
                zone_registry,
                log_context=f"group:{candidate.candidate_id}:badge",
            )
        if candidate.candidate_type == "image_like":
            prefer = "zone_image_right" if "zone_image_right" in zone_registry else "zone_overlay_global"
            zone_id = _coerce_zone_id(prefer, zone_registry, log_context=f"group:{candidate.candidate_id}:image_like")
        if candidate.candidate_type == "background":
            if candidate.center_x_norm < 0.5 and "zone_text_left" in zone_registry:
                zone_id = _coerce_zone_id("zone_text_left", zone_registry, log_context=f"group:{candidate.candidate_id}:background_left")
            elif candidate.center_x_norm >= 0.5 and "zone_image_right" in zone_registry:
                zone_id = _coerce_zone_id("zone_image_right", zone_registry, log_context=f"group:{candidate.candidate_id}:background_right")
            else:
                zone_id = _coerce_zone_id(
                    "zone_overlay_global",
                    zone_registry,
                    log_context=f"group:{candidate.candidate_id}:background_fallback",
                )

        group_id = f"group_{candidate.candidate_id}"
        candidate_group_map[candidate.candidate_id] = group_id

        adaptation_policy = _adaptation_policy_from_group_annotation(qwen_group_ann)
        if adaptation_policy.anchor_type == AnchorType.FREE:
            adaptation_policy.anchor_type = _group_anchor_from_role(group_role)

        groups.append(
            Group(
                id=group_id,
                role=group_role,
                zone_id=zone_id,
                bbox_canvas=_bbox_from_candidate(candidate),
                source_figma_ids=list(candidate.source_node_ids),
                internal_layout=_safe_internal_layout(qwen_group_ann.internal_layout if qwen_group_ann else None),
                anchor_type=adaptation_policy.anchor_type,
                importance_level=_resolve_importance(heuristic_ann, qwen_candidate_ann, qwen_group_ann, config),
                adaptation_policy=adaptation_policy,
            )
        )

    for candidate in merged_candidates:
        if not _should_make_element(candidate):
            continue

        group_id = candidate_group_map.get(candidate.candidate_id)
        if _is_single_text_candidate(candidate):
            covering_group_id = _find_covering_text_group_id(candidate, candidate_bundle, candidate_group_map)
            if covering_group_id is not None:
                group_id = covering_group_id
        if group_id is None:
            continue

        heuristic_ann = _pick_heuristic_annotation(candidate.candidate_id, heuristic_bundle)
        qwen_candidate_ann = _pick_candidate_annotation(candidate.candidate_id, qwen_candidate_annotations)
        qwen_group_ann = _pick_group_annotation(candidate.candidate_id, qwen_group_annotations)

        element_role = _candidate_to_element_role(candidate, heuristic_ann, qwen_candidate_ann, config)
        if candidate.candidate_type == "brand":
            element_role = ElementRole.BRAND_MARK
        if candidate.candidate_type == "image_like" and element_role == ElementRole.HERO_PHOTO:
            element_role = ElementRole.PRODUCT_IMAGE
        if candidate.candidate_type == "background" and element_role == ElementRole.UNKNOWN:
            element_role = ElementRole.BACKGROUND_PANEL
        if element_role == ElementRole.UNKNOWN and not config.keep_unknown_candidates:
            continue

        zone_id = _guess_zone_id_from_position(candidate, layout_pattern)
        zone_id = _coerce_zone_id(
            zone_id,
            zone_registry,
            log_context=f"element:{candidate.candidate_id}:layout_guess",
        )
        if element_role == ElementRole.AGE_BADGE:
            zone_id = _coerce_zone_id(
                "zone_overlay_global",
                zone_registry,
                log_context=f"element:{candidate.candidate_id}:age_badge",
            )
        elif candidate.candidate_type == "image_like":
            prefer = "zone_image_right" if "zone_image_right" in zone_registry else "zone_overlay_global"
            zone_id = _coerce_zone_id(prefer, zone_registry, log_context=f"element:{candidate.candidate_id}:image_like")
        elif candidate.candidate_type == "background":
            if candidate.center_x_norm < 0.5 and "zone_text_left" in zone_registry:
                zone_id = _coerce_zone_id("zone_text_left", zone_registry, log_context=f"element:{candidate.candidate_id}:background_left")
            elif candidate.center_x_norm >= 0.5 and "zone_image_right" in zone_registry:
                zone_id = _coerce_zone_id("zone_image_right", zone_registry, log_context=f"element:{candidate.candidate_id}:background_right")
            else:
                zone_id = _coerce_zone_id(
                    "zone_overlay_global",
                    zone_registry,
                    log_context=f"element:{candidate.candidate_id}:background_fallback",
                )

        adaptation_policy = _adaptation_policy_from_candidate_annotation(qwen_candidate_ann)
        if adaptation_policy.anchor_type == AnchorType.FREE:
            adaptation_policy.anchor_type = _element_anchor_from_role(element_role)

        is_brand_related, is_required_for_compliance = _element_flags_from_role(element_role)

        role_to_functional = {
            ElementRole.HEADLINE: FunctionalType.FUNCTIONAL,
            ElementRole.SUBHEADLINE: FunctionalType.FUNCTIONAL,
            ElementRole.LEGAL: FunctionalType.FUNCTIONAL,
            ElementRole.LOGO_TEXT: FunctionalType.FUNCTIONAL,
            ElementRole.LOGO_ICON: FunctionalType.FUNCTIONAL,
            ElementRole.BRAND_MARK: FunctionalType.FUNCTIONAL,
            ElementRole.AGE_BADGE: FunctionalType.FUNCTIONAL,
            ElementRole.PRODUCT_IMAGE: FunctionalType.FUNCTIONAL,
            ElementRole.HERO_PHOTO: FunctionalType.FUNCTIONAL,
            ElementRole.DECORATION: FunctionalType.DECORATIVE,
            ElementRole.BACKGROUND_PANEL: FunctionalType.BACKGROUND,
            ElementRole.BACKGROUND_SHAPE: FunctionalType.BACKGROUND,
        }
        qwen_ft = _safe_functional_type(qwen_candidate_ann.functional_type) if qwen_candidate_ann is not None else None
        if element_role in role_to_functional:
            functional_type = role_to_functional[element_role]
        elif qwen_ft is not None:
            functional_type = qwen_ft
        else:
            functional_type = FunctionalType.FUNCTIONAL

        text_content = candidate.text_content if candidate.text_content else None
        if candidate.candidate_type == "brand":
            is_text = False
        elif qwen_candidate_ann is not None:
            is_text = qwen_candidate_ann.is_text
        else:
            is_text = candidate.text_count > 0

        if candidate.candidate_type == "decoration" and len(candidate.source_node_ids) > 1:
            ordered = sorted(
                list(candidate.source_node_ids),
                key=lambda nid: (
                    collapsed_by_id[nid].y_norm if nid in collapsed_by_id else 0.0,
                    collapsed_by_id[nid].x_norm if nid in collapsed_by_id else 0.0,
                ),
            )
            total_m = len(ordered)
            for ord_i, sid in enumerate(ordered, start=1):
                sub = collapsed_by_id.get(sid)
                if sub is None:
                    continue
                up = update_by_source.get(sid) or {}
                raw_name = str(up.get("semantic_name", "") or "").strip()
                if raw_name.startswith("decoration_") and ((sub.text or "").strip() or sub.type.lower() == "text"):
                    raw_name = ""
                sem_name = raw_name or _fallback_decoration_member_semantic_name(sub, ordinal=ord_i, total=total_m)
                bw = max(float(sub.w_norm), 1e-4)
                bh = max(float(sub.h_norm), 1e-4)
                bbox = BBox(x=float(sub.x_norm), y=float(sub.y_norm), w=bw, h=bh)
                center = [float(sub.x_norm) + bw / 2.0, float(sub.y_norm) + bh / 2.0]
                elements.append(
                    Element(
                        id=f"el_{candidate.candidate_id}_{_safe_element_id_suffix(sid)}",
                        source_figma_id=sid,
                        type=candidate.candidate_type,
                        role=element_role,
                        group_id=group_id,
                        zone_id=zone_id,
                        semantic_name=sem_name,
                        bbox_canvas=bbox,
                        center_canvas=center,
                        visible=True,
                        functional_type=functional_type,
                        importance_level=_resolve_importance(heuristic_ann, qwen_candidate_ann, qwen_group_ann, config),
                        is_text=is_text,
                        is_brand_related=is_brand_related or (qwen_candidate_ann.is_brand_related if qwen_candidate_ann else False),
                        is_required_for_compliance=is_required_for_compliance or (qwen_candidate_ann.is_required_for_compliance if qwen_candidate_ann else False),
                        text_content=text_content,
                        text_features=_text_features_from_candidate(candidate),
                        adaptation_policy=adaptation_policy,
                    )
                )
            continue

        source_figma_id = candidate.source_node_ids[0]
        up0 = update_by_source.get(source_figma_id) or {}
        sem0 = str(up0.get("semantic_name", "") or "").strip()
        if not _scene_semantic_allowed_for_element_role(element_role, sem0):
            sem0 = ""
        if not sem0 and qwen_candidate_ann is not None:
            q0 = str(qwen_candidate_ann.semantic_name or "").strip()
            if _scene_semantic_allowed_for_element_role(element_role, q0):
                sem0 = q0
        semantic_name_single = sem0 or None

        elements.append(
            Element(
                id=f"el_{candidate.candidate_id}",
                source_figma_id=source_figma_id,
                type=candidate.candidate_type,
                role=element_role,
                group_id=group_id,
                zone_id=zone_id,
                semantic_name=semantic_name_single,
                bbox_canvas=_bbox_from_candidate(candidate),
                center_canvas=_center_from_candidate(candidate),
                visible=True,
                functional_type=functional_type,
                importance_level=_resolve_importance(heuristic_ann, qwen_candidate_ann, qwen_group_ann, config),
                is_text=is_text,
                is_brand_related=is_brand_related or (qwen_candidate_ann.is_brand_related if qwen_candidate_ann else False),
                is_required_for_compliance=is_required_for_compliance or (qwen_candidate_ann.is_required_for_compliance if qwen_candidate_ann else False),
                text_content=text_content,
                text_features=_text_features_from_candidate(candidate),
                adaptation_policy=adaptation_policy,
            )
        )

    group_map = {g.id: g for g in groups}

    for element in elements:
        if element.group_id in group_map:
            group_map[element.group_id].children_elements.append(element.id)

    groups = [g for g in groups if g.children_elements or g.role in {GroupRole.HEADLINE_GROUP, GroupRole.LEGAL_GROUP, GroupRole.BRAND_GROUP}]
    group_map = {g.id: g for g in groups}

    referenced_zone_ids = {g.zone_id for g in groups} | {e.zone_id for e in elements}
    if referenced_zone_ids:
        zones = [z for z in zones if z.id in referenced_zone_ids]
    zone_registry = _zone_registry_ids(zones)

    for gi, g in enumerate(groups):
        if g.zone_id not in zone_registry:
            fixed = _coerce_zone_id(g.zone_id, zone_registry, log_context=f"post_trim_group:{g.id}")
            groups[gi] = g.model_copy(update={"zone_id": fixed})

    for ei, el in enumerate(elements):
        if el.zone_id not in zone_registry:
            fixed = _coerce_zone_id(el.zone_id, zone_registry, log_context=f"post_trim_element:{el.id}")
            elements[ei] = el.model_copy(update={"zone_id": fixed})

    referenced_zone_ids = {g.zone_id for g in groups} | {e.zone_id for e in elements}
    if referenced_zone_ids - _zone_registry_ids(zones):
        _ensure_zone_default_registry(zones)
        zones = [z for z in zones if z.id in referenced_zone_ids or z.id == ZONE_DEFAULT_ID]
    zone_registry = _zone_registry_ids(zones)

    zone_children: dict[str, list[str]] = {}
    for group in groups:
        if group.zone_id in zone_registry:
            zone_children.setdefault(group.zone_id, []).append(group.id)

    for zone in zones:
        zone.children_groups = zone_children.get(zone.id, [])

    zone_map = {z.id: z for z in zones}

    for group in groups:
        if group.zone_id in zone_map:
            relations.append(Relation(src=group.id, dst=group.zone_id, type=RelationType.BELONGS_TO_ZONE, strength=1.0))

    for element in elements:
        if element.group_id in group_map:
            relations.append(Relation(src=element.id, dst=element.group_id, type=RelationType.BELONGS_TO_GROUP, strength=1.0))
        if element.zone_id in zone_map:
            relations.append(Relation(src=element.id, dst=element.zone_id, type=RelationType.BELONGS_TO_ZONE, strength=1.0))

    group_to_elements: dict[str, list[Element]] = {}
    for element in elements:
        group_to_elements.setdefault(element.group_id, []).append(element)

    for _, els in group_to_elements.items():
        text_els = [e for e in els if e.is_text]
        text_els.sort(key=lambda e: (e.bbox_canvas.y, e.bbox_canvas.x))
        for i in range(len(text_els) - 1):
            upper = text_els[i]
            lower = text_els[i + 1]
            if lower.bbox_canvas.y >= upper.bbox_canvas.y:
                relations.append(Relation(src=upper.id, dst=lower.id, type=RelationType.ABOVE, strength=0.95))
                relations.append(Relation(src=upper.id, dst=lower.id, type=RelationType.VERTICAL_STACK, strength=0.90))

    for element in elements:
        if element.role in {ElementRole.HEADLINE, ElementRole.SUBHEADLINE, ElementRole.LEGAL, ElementRole.AGE_BADGE, ElementRole.BRAND_MARK}:
            constraints.append(Constraint(target=element.id, type=ConstraintType.MUST_REMAIN_VISIBLE, value=True))
        if element.role == ElementRole.LEGAL:
            constraints.append(Constraint(target=element.id, type=ConstraintType.MIN_FONT_SIZE, value=10))
        if element.role == ElementRole.AGE_BADGE:
            constraints.append(Constraint(target=element.id, type=ConstraintType.MUST_ANCHOR_CORNER, value="top_right"))

    for group in groups:
        if group.role in {GroupRole.BRAND_GROUP, GroupRole.HEADLINE_GROUP, GroupRole.LEGAL_GROUP}:
            constraints.append(Constraint(target=group.id, type=ConstraintType.MUST_REMAIN_VISIBLE, value=True))

    if banner_annotation and banner_annotation.preservation_priorities:
        role_to_targets: dict[str, list[str]] = {}
        for element in elements:
            role_to_targets.setdefault(element.role.value, []).append(element.id)
        for group in groups:
            role_to_targets.setdefault(group.role.value, []).append(group.id)

        for item in banner_annotation.preservation_priorities:
            role_name = str(item.get("role", "")).strip().lower()
            priority = int(item.get("priority", 999))
            if priority != 1:
                continue
            alias_map = {
                "logo": ["brand_group", "logo_text", "logo_icon", "brand_mark"],
                "headline": ["headline", "headline_group", "subheadline"],
                "legal": ["legal", "legal_group"],
                "badge": ["age_badge", "badge_group"],
            }
            target_roles = alias_map.get(role_name, [role_name])
            for target_role in target_roles:
                for target_id in role_to_targets.get(target_role, []):
                    constraints.append(Constraint(target=target_id, type=ConstraintType.MUST_REMAIN_VISIBLE, value=True))

    constraints = _dedupe_constraints(constraints)

    return SemanticGraph(
        banner_id=banner_id,
        template_id=template_id,
        source=source,
        metadata=metadata,
        zones=zones,
        groups=groups,
        elements=elements,
        relations=relations,
        constraints=constraints,
    )
