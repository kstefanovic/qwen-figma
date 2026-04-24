from __future__ import annotations

from typing import Any, Optional

from banner_pipeline.build_candidates import CandidateBundle
from banner_pipeline.collapse_groups import CollapsedNode
from banner_pipeline.heuristics import HeuristicBundle


def _child_path(parent_path: str, index: int) -> str:
    seg = str(index)
    return seg if parent_path == "" else f"{parent_path}/{seg}"


def _font_family_from_extra(extra: dict[str, Any]) -> str | None:
    fn = extra.get("fontName")
    if isinstance(fn, dict):
        fam = fn.get("family")
        if isinstance(fam, str) and fam.strip():
            return fam.strip()
    if isinstance(fn, str) and fn.strip():
        return fn.strip()
    return None


def _font_size_from_extra(extra: dict[str, Any]) -> float | None:
    for key in ("fontSize", "font_size"):
        v = extra.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _visual_hints_from_extra(extra: dict[str, Any]) -> dict[str, Any]:
    fills = extra.get("fills")
    fill_types: list[str] = []
    has_image_fill = False
    if isinstance(fills, list):
        for fill in fills[:4]:
            if not isinstance(fill, dict):
                continue
            ftype = str(fill.get("type", "") or "").strip().lower()
            if ftype:
                fill_types.append(ftype)
            if fill.get("imageRef") or ftype == "image":
                has_image_fill = True

    out: dict[str, Any] = {
        "fill_types": sorted(set(fill_types)),
        "has_image_fill": has_image_fill,
        "boolean_operation": str(extra.get("booleanOperation", "") or ""),
        "blend_mode": str(extra.get("blendMode", "") or ""),
    }
    return out


def build_figma_summary(
    collapsed_nodes: list[CollapsedNode],
    candidate_bundle: CandidateBundle,
    heuristic_bundle: Optional[HeuristicBundle],
    canvas_width: int,
    canvas_height: int,
    *,
    max_nodes: int = 280,
    max_text_len: int = 400,
) -> dict[str, Any]:
    """
    Compact JSON for single-pass VLM: hierarchy, geometry, text hints, and candidate/heuristic summaries.
    Paths follow collapsed-tree child_ids order (root "", children "0", "1", nested "1/0", ...).
    """
    node_by_id: dict[str, CollapsedNode] = {n.id: n for n in collapsed_nodes}

    roots = [n for n in collapsed_nodes if n.parent_id is None or n.parent_id not in node_by_id]
    if not roots:
        roots = collapsed_nodes[:1]

    nodes_out: list[dict[str, Any]] = []
    truncated = False

    def visit(node_id: str, path: str) -> None:
        nonlocal truncated
        if len(nodes_out) >= max_nodes:
            truncated = True
            return
        n = node_by_id.get(node_id)
        if n is None:
            return
        ex = dict(n.extra_data or {})
        text = (n.text or "").strip()
        if len(text) > max_text_len:
            text = text[: max_text_len - 1] + "…"
        nodes_out.append(
            {
                "id": n.id,
                "path": path,
                "name": n.name,
                "name_is_generic": not bool((n.name or "").strip()) or str(n.name).strip().lower().startswith(("group ", "frame ", "rectangle ", "vector ")),
                "type": n.type,
                "text": text,
                "bbox_canvas": [n.x_norm, n.y_norm, n.w_norm, n.h_norm],
                "font_size": _font_size_from_extra(ex),
                "font_family": _font_family_from_extra(ex),
                "parent_id": n.parent_id,
                "children_ids": list(n.child_ids),
                "visual_hints": _visual_hints_from_extra(ex),
            }
        )
        for i, cid in enumerate(n.child_ids):
            if cid in node_by_id:
                visit(cid, _child_path(path, i))

    if len(roots) == 1:
        visit(roots[0].id, "")
    else:
        for ri, r in enumerate(roots):
            if len(nodes_out) >= max_nodes:
                truncated = True
                break
            visit(r.id, str(ri))

    heuristic_map = heuristic_bundle.by_candidate_id if heuristic_bundle else {}

    candidates_out: list[dict[str, Any]] = []
    for c in candidate_bundle.all_candidates:
        ann = heuristic_map.get(c.candidate_id)
        tc = (c.text_content or "").strip()
        if len(tc) > max_text_len:
            tc = tc[: max_text_len - 1] + "…"
        candidates_out.append(
            {
                "candidate_id": c.candidate_id,
                "candidate_type": c.candidate_type,
                "source_node_ids": list(c.source_node_ids),
                "bbox_canvas": list(c.bbox_canvas),
                "text_content": tc,
                "heuristic_role": ann.final_role_hint if ann else None,
                "heuristic_group": ann.final_group_hint if ann else None,
                "importance": ann.final_importance_hint if ann else None,
                "grouping_reason": (c.extra_data or {}).get("grouping_reason"),
                "member_count": int((c.extra_data or {}).get("member_count", len(c.source_node_ids)) or len(c.source_node_ids)),
            }
        )

    group_like_ids: list[str] = []
    seen_g: set[str] = set()
    for c in (
        *candidate_bundle.text_group_candidates,
        *candidate_bundle.brand_candidates,
        *candidate_bundle.decoration_candidates,
        *candidate_bundle.background_candidates,
        *candidate_bundle.image_like_candidates,
    ):
        if c.candidate_id not in seen_g:
            seen_g.add(c.candidate_id)
            group_like_ids.append(c.candidate_id)

    element_ids = [c.candidate_id for c in candidate_bundle.all_candidates]

    summary: dict[str, Any] = {
        "canvas": {"width": int(canvas_width), "height": int(canvas_height)},
        "nodes": nodes_out,
        "candidates": candidates_out,
        "element_annotation_candidate_ids": element_ids,
        "group_annotation_candidate_ids": group_like_ids,
    }
    if truncated:
        summary["_truncated_nodes"] = True
    return summary


def build_qwen_scene_payload(
    *,
    figma_summary: dict[str, Any],
    collapsed_nodes: list[CollapsedNode],
    candidate_bundle: CandidateBundle,
    heuristic_bundle: Optional[HeuristicBundle],
) -> dict[str, Any]:
    """
    Build richer single-pass payload for /annotate/scene.
    Uses Figma-derived structure as source of truth and keeps VLM in a correction/enrichment role.
    """
    heur_by_id = heuristic_bundle.by_candidate_id if heuristic_bundle else {}

    elements: list[dict[str, Any]] = []
    for n in collapsed_nodes:
        elements.append(
            {
                "id": n.id,
                "name": n.name,
                "type": n.type,
                "text": n.text,
                "bbox_canvas": [n.x_norm, n.y_norm, n.w_norm, n.h_norm],
                "depth": n.depth,
                "z_order_hint": n.depth,
                "parent_id": n.parent_id,
                "children_ids": list(n.child_ids),
                "source_figma_ids": list(n.source_figma_ids),
            }
        )

    grouped_ids: set[str] = set()
    groups: list[dict[str, Any]] = []
    for c in (
        *candidate_bundle.text_group_candidates,
        *candidate_bundle.brand_candidates,
        *candidate_bundle.decoration_candidates,
        *candidate_bundle.background_candidates,
        *candidate_bundle.image_like_candidates,
    ):
        if c.candidate_id in grouped_ids:
            continue
        grouped_ids.add(c.candidate_id)
        h = heur_by_id.get(c.candidate_id)
        groups.append(
            {
                "candidate_id": c.candidate_id,
                "candidate_type": c.candidate_type,
                "source_node_ids": list(c.source_node_ids),
                "bbox_canvas": list(c.bbox_canvas),
                "heuristic_group": h.final_group_hint if h else None,
                "heuristic_importance": h.final_importance_hint if h else None,
                "grouping_reason": (c.extra_data or {}).get("grouping_reason"),
                "member_count": int((c.extra_data or {}).get("member_count", len(c.source_node_ids)) or len(c.source_node_ids)),
            }
        )

    heuristic_roles: dict[str, dict[str, Any]] = {}
    for c in candidate_bundle.all_candidates:
        h = heur_by_id.get(c.candidate_id)
        if h is None:
            continue
        heuristic_roles[c.candidate_id] = {
            "role": h.final_role_hint,
            "group": h.final_group_hint,
            "importance": h.final_importance_hint,
            "confidence": h.confidence,
        }

    canvas = figma_summary.get("canvas") or {}
    banner_metadata = {
        "canvas_width": int(canvas.get("width", 0) or 0),
        "canvas_height": int(canvas.get("height", 0) or 0),
        "node_count": len(elements),
        "candidate_count": len(candidate_bundle.all_candidates),
        "group_candidate_count": len(groups),
    }

    return {
        "banner_metadata": banner_metadata,
        "elements": elements,
        "groups": groups,
        "heuristic_roles": heuristic_roles,
        "figma_summary": figma_summary,
    }
