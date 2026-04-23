from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from schemas.enums import ElementRole, GroupRole
from schemas.semantic_graph import Element, Group, SemanticGraph


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    target_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "target_id": self.target_id,
            "extra": self.extra,
        }


@dataclass
class ValidationReport:
    is_valid: bool
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "stats": self.stats,
        }


def _add_error(errors: list[ValidationIssue], code: str, message: str, target_id: str | None = None, extra: dict[str, Any] | None = None) -> None:
    errors.append(ValidationIssue(severity="error", code=code, message=message, target_id=target_id, extra=extra or {}))


def _add_warning(warnings: list[ValidationIssue], code: str, message: str, target_id: str | None = None, extra: dict[str, Any] | None = None) -> None:
    warnings.append(ValidationIssue(severity="warning", code=code, message=message, target_id=target_id, extra=extra or {}))


def _bbox_area(x: float, y: float, w: float, h: float) -> float:
    return max(0.0, w) * max(0.0, h)


def _overlap_area(a, b) -> float:
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x + a.w, b.x + b.w)
    bottom = min(a.y + a.h, b.y + b.h)
    return max(0.0, right - left) * max(0.0, bottom - top)


def _overlap_ratio(a, b) -> float:
    inter = _overlap_area(a, b)
    min_area = min(_bbox_area(a.x, a.y, a.w, a.h), _bbox_area(b.x, b.y, b.w, b.h))
    if min_area <= 0:
        return 0.0
    return inter / min_area


def _contains_bbox(outer, inner, tol: float = 1e-6) -> bool:
    return (
        inner.x >= outer.x - tol
        and inner.y >= outer.y - tol
        and (inner.x + inner.w) <= (outer.x + outer.w) + tol
        and (inner.y + inner.h) <= (outer.y + outer.h) + tol
    )


def _group_elements(group_id: str, elements: list[Element]) -> list[Element]:
    return [e for e in elements if e.group_id == group_id]


def _zone_groups(zone_id: str, groups: list[Group]) -> list[Group]:
    return [g for g in groups if g.zone_id == zone_id]


def _validate_non_empty_graph(graph: SemanticGraph, errors: list[ValidationIssue], warnings: list[ValidationIssue]) -> None:
    if not graph.zones:
        _add_error(errors, "empty_zones", "Graph has no zones.")
    if not graph.groups:
        _add_error(errors, "empty_groups", "Graph has no groups.")
    if not graph.elements:
        _add_warning(warnings, "empty_elements", "Graph has no elements.")


def _validate_ids_unique(graph: SemanticGraph, errors: list[ValidationIssue]) -> None:
    all_ids = [z.id for z in graph.zones] + [g.id for g in graph.groups] + [e.id for e in graph.elements]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item_id in all_ids:
        if item_id in seen:
            duplicates.add(item_id)
        seen.add(item_id)
    for dup in sorted(duplicates):
        _add_error(errors, "duplicate_id", f"Duplicate graph id detected: '{dup}'.", target_id=dup)


def _validate_zone_children_consistency(graph: SemanticGraph, errors: list[ValidationIssue], warnings: list[ValidationIssue]) -> None:
    group_ids = {g.id for g in graph.groups}
    for zone in graph.zones:
        for child_group_id in zone.children_groups:
            if child_group_id not in group_ids:
                _add_error(errors, "zone_unknown_child_group", f"Zone '{zone.id}' references unknown child group '{child_group_id}'.", target_id=zone.id)

        actual_groups = _zone_groups(zone.id, graph.groups)
        if not zone.children_groups and actual_groups:
            _add_warning(warnings, "zone_children_missing", f"Zone '{zone.id}' has groups assigned to it, but children_groups is empty.", target_id=zone.id, extra={"actual_group_count": len(actual_groups)})


def _validate_group_children_consistency(graph: SemanticGraph, errors: list[ValidationIssue], warnings: list[ValidationIssue]) -> None:
    element_ids = {e.id for e in graph.elements}
    group_ids = {g.id for g in graph.groups}

    for group in graph.groups:
        for child_element_id in group.children_elements:
            if child_element_id not in element_ids:
                _add_error(errors, "group_unknown_child_element", f"Group '{group.id}' references unknown child element '{child_element_id}'.", target_id=group.id)
        for child_group_id in group.children_groups:
            if child_group_id not in group_ids:
                _add_error(errors, "group_unknown_child_group", f"Group '{group.id}' references unknown child group '{child_group_id}'.", target_id=group.id)

        actual_elements = _group_elements(group.id, graph.elements)
        if not group.children_elements and actual_elements:
            _add_warning(warnings, "group_children_missing", f"Group '{group.id}' has elements assigned to it, but children_elements is empty.", target_id=group.id, extra={"actual_element_count": len(actual_elements)})


def _validate_bbox_ranges(graph: SemanticGraph, errors: list[ValidationIssue]) -> None:
    def check_bbox(obj_id: str, bbox, obj_type: str) -> None:
        if bbox.x < 0 or bbox.y < 0 or bbox.w <= 0 or bbox.h <= 0:
            _add_error(errors, "invalid_bbox_values", f"{obj_type} '{obj_id}' has invalid bbox values.", target_id=obj_id, extra={"bbox": [bbox.x, bbox.y, bbox.w, bbox.h]})
        if bbox.x + bbox.w > 1.000001 or bbox.y + bbox.h > 1.000001:
            _add_error(errors, "bbox_out_of_unit_space", f"{obj_type} '{obj_id}' bbox exceeds unit canvas bounds.", target_id=obj_id, extra={"bbox": [bbox.x, bbox.y, bbox.w, bbox.h]})

    for zone in graph.zones:
        check_bbox(zone.id, zone.bbox_canvas, "Zone")
    for group in graph.groups:
        if group.bbox_canvas is not None:
            check_bbox(group.id, group.bbox_canvas, "Group")
    for element in graph.elements:
        check_bbox(element.id, element.bbox_canvas, "Element")


def _validate_group_bbox_contains_elements(graph: SemanticGraph, warnings: list[ValidationIssue]) -> None:
    group_map = {g.id: g for g in graph.groups}
    for element in graph.elements:
        group = group_map.get(element.group_id)
        if group is None or group.bbox_canvas is None:
            continue
        if not _contains_bbox(group.bbox_canvas, element.bbox_canvas, tol=0.03):
            _add_warning(warnings, "element_outside_group_bbox", f"Element '{element.id}' is not well contained in group '{group.id}'.", target_id=element.id, extra={"group_id": group.id})


def _validate_zone_bbox_contains_groups(graph: SemanticGraph, warnings: list[ValidationIssue]) -> None:
    zone_map = {z.id: z for z in graph.zones}
    for group in graph.groups:
        zone = zone_map.get(group.zone_id)
        if zone is None or group.bbox_canvas is None:
            continue
        if not _contains_bbox(zone.bbox_canvas, group.bbox_canvas, tol=0.08):
            _add_warning(warnings, "group_outside_zone_bbox", f"Group '{group.id}' is not well contained in zone '{zone.id}'.", target_id=group.id, extra={"zone_id": zone.id, "group_bbox": [group.bbox_canvas.x, group.bbox_canvas.y, group.bbox_canvas.w, group.bbox_canvas.h], "zone_bbox": [zone.bbox_canvas.x, zone.bbox_canvas.y, zone.bbox_canvas.w, zone.bbox_canvas.h]})


def _validate_critical_presence(graph: SemanticGraph, warnings: list[ValidationIssue]) -> None:
    roles = {e.role for e in graph.elements}
    group_roles = {g.role for g in graph.groups}
    if ElementRole.HEADLINE not in roles and GroupRole.HEADLINE_GROUP not in group_roles and GroupRole.TEXT_GROUP not in group_roles:
        _add_warning(warnings, "missing_headline", "Graph has no obvious headline or headline-like group.")
    if GroupRole.BRAND_GROUP not in group_roles:
        _add_warning(warnings, "missing_brand_group", "Graph has no brand_group.")
    if ElementRole.LEGAL not in roles and GroupRole.LEGAL_GROUP not in group_roles:
        _add_warning(warnings, "missing_legal", "Graph has no legal text/group.")


def _validate_brand_zone_alignment(graph: SemanticGraph, warnings: list[ValidationIssue]) -> None:
    for group in graph.groups:
        if group.role != GroupRole.BRAND_GROUP:
            continue
        bbox = group.bbox_canvas
        if bbox is None:
            _add_warning(warnings, "brand_group_missing_bbox", "Brand group has no bbox.", target_id=group.id)
            continue
        if bbox.x > 0.35 or bbox.y > 0.30:
            _add_warning(warnings, "brand_group_unusual_position", "Brand group is far from typical top-left placement.", target_id=group.id)


def _validate_age_badge_position(graph: SemanticGraph, warnings: list[ValidationIssue]) -> None:
    for element in graph.elements:
        if element.role != ElementRole.AGE_BADGE:
            continue
        bbox = element.bbox_canvas
        top_rightish = bbox.x >= 0.70 and bbox.y <= 0.25
        if not top_rightish:
            _add_warning(warnings, "age_badge_unusual_position", "Age badge is far from typical top-right placement.", target_id=element.id)


def _validate_legal_text_position(graph: SemanticGraph, warnings: list[ValidationIssue]) -> None:
    for element in graph.elements:
        if element.role != ElementRole.LEGAL:
            continue
        bbox = element.bbox_canvas
        if bbox.y < 0.45:
            _add_warning(warnings, "legal_unusual_position", "Legal text is not near the lower portion of the canvas.", target_id=element.id)


def _validate_suspicious_overlaps(graph: SemanticGraph, warnings: list[ValidationIssue]) -> None:
    important_text_roles = {
        ElementRole.HEADLINE,
        ElementRole.SUBHEADLINE,
        ElementRole.LEGAL,
        ElementRole.PRICE_MAIN,
        ElementRole.PRICE_OLD,
        ElementRole.PRICE_FRACTION,
    }
    candidates = [e for e in graph.elements if e.role in important_text_roles]

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            ratio = _overlap_ratio(a.bbox_canvas, b.bbox_canvas)
            if ratio >= 0.30:
                _add_warning(warnings, "suspicious_text_overlap", f"Important text elements '{a.id}' and '{b.id}' strongly overlap.", target_id=a.id, extra={"other_id": b.id, "overlap_ratio": round(ratio, 4)})


def _validate_relations_reference_existing_nodes(graph: SemanticGraph, errors: list[ValidationIssue]) -> None:
    valid_ids = {z.id for z in graph.zones} | {g.id for g in graph.groups} | {e.id for e in graph.elements}
    for rel in graph.relations:
        if rel.src not in valid_ids:
            _add_error(errors, "relation_missing_src", f"Relation source '{rel.src}' does not exist.", target_id=rel.src)
        if rel.dst not in valid_ids:
            _add_error(errors, "relation_missing_dst", f"Relation destination '{rel.dst}' does not exist.", target_id=rel.dst)


def _validate_constraints_reference_existing_nodes(graph: SemanticGraph, errors: list[ValidationIssue]) -> None:
    valid_ids = {z.id for z in graph.zones} | {g.id for g in graph.groups} | {e.id for e in graph.elements}
    for constraint in graph.constraints:
        if constraint.target not in valid_ids:
            _add_error(errors, "constraint_missing_target", f"Constraint target '{constraint.target}' does not exist.", target_id=constraint.target)


def _validate_relations_duplicates(graph: SemanticGraph, warnings: list[ValidationIssue]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for rel in graph.relations:
        key = (rel.src, rel.dst, rel.type.value)
        if key in seen:
            _add_warning(warnings, "duplicate_relation", f"Duplicate relation found: {rel.src} -> {rel.dst} ({rel.type.value})", target_id=rel.src, extra={"dst": rel.dst, "type": rel.type.value})
        seen.add(key)


def _collect_stats(graph: SemanticGraph) -> dict[str, Any]:
    role_counts: dict[str, int] = {}
    group_role_counts: dict[str, int] = {}
    zone_role_counts: dict[str, int] = {}

    for e in graph.elements:
        role_counts[e.role.value] = role_counts.get(e.role.value, 0) + 1
    for g in graph.groups:
        group_role_counts[g.role.value] = group_role_counts.get(g.role.value, 0) + 1
    for z in graph.zones:
        zone_role_counts[z.role.value] = zone_role_counts.get(z.role.value, 0) + 1

    return {
        "zone_count": len(graph.zones),
        "group_count": len(graph.groups),
        "element_count": len(graph.elements),
        "relation_count": len(graph.relations),
        "constraint_count": len(graph.constraints),
        "element_role_counts": role_counts,
        "group_role_counts": group_role_counts,
        "zone_role_counts": zone_role_counts,
    }


def validate_graph(graph: SemanticGraph) -> ValidationReport:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    _validate_non_empty_graph(graph, errors, warnings)
    _validate_ids_unique(graph, errors)
    _validate_zone_children_consistency(graph, errors, warnings)
    _validate_group_children_consistency(graph, errors, warnings)
    _validate_bbox_ranges(graph, errors)
    _validate_group_bbox_contains_elements(graph, warnings)
    _validate_zone_bbox_contains_groups(graph, warnings)
    _validate_critical_presence(graph, warnings)
    _validate_brand_zone_alignment(graph, warnings)
    _validate_age_badge_position(graph, warnings)
    _validate_legal_text_position(graph, warnings)
    _validate_suspicious_overlaps(graph, warnings)
    _validate_relations_reference_existing_nodes(graph, errors)
    _validate_constraints_reference_existing_nodes(graph, errors)
    _validate_relations_duplicates(graph, warnings)

    stats = _collect_stats(graph)
    return ValidationReport(is_valid=(len(errors) == 0), errors=errors, warnings=warnings, stats=stats)
