from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


def _norm_text(value: str) -> str:
    return re.sub(r"[^0-9a-zа-я]+", "", (value or "").lower().replace("ё", "е"))


def _bbox_from_node(node: dict[str, Any], frame_w: float, frame_h: float) -> dict[str, float] | None:
    bounds = node.get("bounds")
    if not isinstance(bounds, dict):
        return None
    try:
        x = float(bounds.get("x", 0) or 0) / frame_w
        y = float(bounds.get("y", 0) or 0) / frame_h
        w = float(bounds.get("width", 0) or 0) / frame_w
        h = float(bounds.get("height", 0) or 0) / frame_h
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if w <= 0 or h <= 0:
        return None
    return {
        "x": max(0.0, min(1.0, x)),
        "y": max(0.0, min(1.0, y)),
        "width": max(0.0, min(1.0, w)),
        "height": max(0.0, min(1.0, h)),
    }


def _area(b: dict[str, float]) -> float:
    return max(0.0, b["width"]) * max(0.0, b["height"])


def _node_area_ratio(node: dict[str, Any]) -> float:
    bbox = node.get("_bbox")
    return _area(bbox) if isinstance(bbox, dict) else 0.0


def _right(b: dict[str, float]) -> float:
    return b["x"] + b["width"]


def _bottom(b: dict[str, float]) -> float:
    return b["y"] + b["height"]


def _iou(a: dict[str, float], b: dict[str, float]) -> float:
    ix = max(0.0, min(_right(a), _right(b)) - max(a["x"], b["x"]))
    iy = max(0.0, min(_bottom(a), _bottom(b)) - max(a["y"], b["y"]))
    inter = ix * iy
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _center_distance_score(a: dict[str, float], b: dict[str, float]) -> float:
    ax = a["x"] + a["width"] / 2.0
    ay = a["y"] + a["height"] / 2.0
    bx = b["x"] + b["width"] / 2.0
    by = b["y"] + b["height"] / 2.0
    d = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    return max(0.0, 1.0 - d / 0.55)


def _union_bboxes(boxes: list[dict[str, float] | None]) -> dict[str, float] | None:
    boxes = [b for b in boxes if isinstance(b, dict)]
    if not boxes:
        return None
    x0 = min(b["x"] for b in boxes)
    y0 = min(b["y"] for b in boxes)
    x1 = max(_right(b) for b in boxes)
    y1 = max(_bottom(b) for b in boxes)
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def _q_child(group: dict[str, Any] | None, role: str) -> dict[str, Any] | None:
    if not isinstance(group, dict):
        return None
    for child in group.get("children") or []:
        if isinstance(child, dict) and child.get("role") == role:
            return child
    return None


def _q_group(qwen_json: dict[str, Any], role: str) -> dict[str, Any] | None:
    groups = ((qwen_json.get("text_zone") or {}).get("groups") or [])
    for group in groups:
        if isinstance(group, dict) and group.get("role") == role:
            return group
    return None


def _nested_q_child(child: dict[str, Any] | None, role: str) -> dict[str, Any] | None:
    if not isinstance(child, dict):
        return None
    for nested in child.get("children") or []:
        if isinstance(nested, dict) and nested.get("role") == role:
            return nested
    return None


def _text_of_node(node: dict[str, Any]) -> str:
    raw = node.get("characters")
    if isinstance(raw, str):
        return raw
    raw = node.get("text")
    return raw if isinstance(raw, str) else ""


def _node_type(node: dict[str, Any]) -> str:
    return str(node.get("type", "") or "").strip().lower()


def _abs_bounds_from_bbox(bbox: dict[str, float] | None, frame_w: float, frame_h: float) -> dict[str, float] | None:
    if not isinstance(bbox, dict):
        return None
    return {
        "x": round(float(bbox.get("x", 0) or 0) * frame_w, 4),
        "y": round(float(bbox.get("y", 0) or 0) * frame_h, 4),
        "width": round(float(bbox.get("width", 0) or 0) * frame_w, 4),
        "height": round(float(bbox.get("height", 0) or 0) * frame_h, 4),
    }


def _clean_output_node(node: dict[str, Any], name: str) -> dict[str, Any]:
    out = deepcopy(node)
    out.pop("_bbox", None)
    out["name"] = name
    return out


def _synthetic_node(
    name: str,
    *,
    bbox: dict[str, float] | None,
    frame_w: float,
    frame_h: float,
    children: list[dict[str, Any]] | None = None,
    node_type: str = "group",
) -> dict[str, Any]:
    return {
        "id": None,
        "path": None,
        "name": name,
        "type": node_type,
        "bounds": _abs_bounds_from_bbox(bbox, frame_w, frame_h)
        or {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0},
        "visible": True,
        "opacity": 1,
        "children": children or [],
    }


def _element(
    role: str,
    nodes: list[dict[str, Any]],
    *,
    frame_w: float,
    frame_h: float,
    bbox: dict[str, float] | None = None,
    text: str = "",
) -> dict[str, Any]:
    boxes = [n["_bbox"] for n in nodes if isinstance(n.get("_bbox"), dict)]
    resolved_bbox = bbox or _union_bboxes(boxes)
    if len(nodes) == 1:
        return _clean_output_node(nodes[0], role)
    return _synthetic_node(role, bbox=resolved_bbox, frame_w=frame_w, frame_h=frame_h)


def _bbox_for_output_node(node: dict[str, Any], frame_w: float, frame_h: float) -> dict[str, float] | None:
    if isinstance(node.get("_bbox"), dict):
        return node["_bbox"]
    return _bbox_from_node(node, frame_w, frame_h)


class _Mapper:
    def __init__(self, mid_json: dict[str, Any], qwen_json: dict[str, Any]) -> None:
        self.mid_json = mid_json
        self.qwen_json = qwen_json
        bounds = mid_json.get("bounds") if isinstance(mid_json.get("bounds"), dict) else {}
        self.frame_w = max(1.0, float(bounds.get("width", 1.0) or 1.0))
        self.frame_h = max(1.0, float(bounds.get("height", 1.0) or 1.0))
        self.nodes: list[dict[str, Any]] = []
        for node in mid_json.get("children") or []:
            if not isinstance(node, dict):
                continue
            bbox = _bbox_from_node(node, self.frame_w, self.frame_h)
            if bbox is None:
                continue
            copied = deepcopy(node)
            copied["_bbox"] = bbox
            self.nodes.append(copied)
        self.used_ids: set[str] = set()

    def _available(self, *, types: set[str] | None = None) -> list[dict[str, Any]]:
        out = []
        for node in self.nodes:
            nid = str(node.get("id", "") or node.get("path", ""))
            if nid in self.used_ids:
                continue
            if types is not None and _node_type(node) not in types:
                continue
            out.append(node)
        return out

    def _mark_used(self, nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            self.used_ids.add(str(node.get("id", "") or node.get("path", "")))

    def _score(self, node: dict[str, Any], guide: dict[str, Any] | None, target_text: str = "") -> float:
        score = 0.0
        if guide and isinstance(guide.get("bbox"), dict):
            score += 3.0 * _iou(node["_bbox"], guide["bbox"])
            score += 1.2 * _center_distance_score(node["_bbox"], guide["bbox"])
        if target_text:
            nt = _norm_text(target_text)
            nn = _norm_text(_text_of_node(node))
            if nt and nn:
                if nt == nn:
                    score += 2.0
                elif nt in nn or nn in nt:
                    score += min(len(nt), len(nn)) / max(len(nt), len(nn))
        return score

    def pick_one(
        self,
        role: str,
        guide: dict[str, Any] | None,
        *,
        types: set[str] | None = None,
        target_text: str = "",
        min_score: float = 0.05,
    ) -> dict[str, Any] | None:
        best_node = None
        best_score = -1.0
        for node in self._available(types=types):
            score = self._score(node, guide, target_text)
            if score > best_score:
                best_node = node
                best_score = score
        if best_node is None or best_score < min_score:
            return None
        self._mark_used([best_node])
        return _element(
            role,
            [best_node],
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            text=_text_of_node(best_node) if _node_type(best_node) == "text" else "",
        )

    def pick_text(self, role: str, guide: dict[str, Any] | None, target_text: str) -> dict[str, Any] | None:
        picked = self.pick_one(role, guide, types={"text"}, target_text=target_text, min_score=0.15)
        if picked is not None:
            return picked
        if not target_text:
            return None
        target_norm = _norm_text(target_text)
        guide_bbox = guide.get("bbox") if isinstance(guide, dict) and isinstance(guide.get("bbox"), dict) else None
        candidates = sorted(
            self._available(types={"text"}),
            key=lambda n: (n["_bbox"]["y"], n["_bbox"]["x"]),
        )
        if guide_bbox is not None:
            candidates = [n for n in candidates if _iou(n["_bbox"], guide_bbox) > 0 or _center_distance_score(n["_bbox"], guide_bbox) > 0.55]
        combo: list[dict[str, Any]] = []
        text_acc = ""
        for node in candidates:
            combo.append(node)
            text_acc += _text_of_node(node)
            if _norm_text(text_acc) == target_norm or target_norm in _norm_text(text_acc):
                self._mark_used(combo)
                children = [
                    _element(f"{role}_{i + 1}", [n], frame_w=self.frame_w, frame_h=self.frame_h, text=_text_of_node(n))
                    for i, n in enumerate(combo)
                ]
                parent = _element(
                    role,
                    combo,
                    frame_w=self.frame_w,
                    frame_h=self.frame_h,
                    text="".join(_text_of_node(n) for n in combo),
                )
                parent["children"] = children
                return parent
            if len(_norm_text(text_acc)) > len(target_norm) + 6:
                combo = []
                text_acc = ""
        return None

    def pick_brand_visual_text(
        self,
        role: str,
        guide: dict[str, Any] | None,
        target_text: str,
    ) -> dict[str, Any] | None:
        """
        Brand names often arrive from Figma as vector outlines, not TEXT nodes.
        Use Qwen's bbox as the primary guide and allow multiple vector fragments.
        """
        guide_bbox = guide.get("bbox") if isinstance(guide, dict) and isinstance(guide.get("bbox"), dict) else None
        if guide_bbox is None:
            return self.pick_text(role, guide, target_text)

        visual_types = {"vector", "boolean_operation", "frame", "instance", "group"}
        candidates: list[tuple[float, dict[str, Any]]] = []
        for node in self._available(types=visual_types):
            if _node_type(node) in {"vector", "boolean_operation"} and _node_area_ratio(node) >= 0.30:
                continue
            overlap = _iou(node["_bbox"], guide_bbox)
            center = _center_distance_score(node["_bbox"], guide_bbox)
            # Small vector glyphs may sit inside the OCR bbox but have low IoU.
            inside_x = node["_bbox"]["x"] >= guide_bbox["x"] - 0.015 and _right(node["_bbox"]) <= _right(guide_bbox) + 0.015
            inside_y = node["_bbox"]["y"] >= guide_bbox["y"] - 0.08 and _bottom(node["_bbox"]) <= _bottom(guide_bbox) + 0.08
            if overlap <= 0 and center < 0.62 and not (inside_x and inside_y):
                continue
            score = 3.0 * overlap + 1.2 * center + (0.8 if inside_x and inside_y else 0.0)
            candidates.append((score, node))

        if not candidates:
            return self.pick_text(role, guide, target_text)

        candidates.sort(key=lambda item: item[0], reverse=True)
        # Keep all strong fragments inside/near the guide bbox. This covers split text such as "Я" + "ндекс".
        best_score = candidates[0][0]
        selected = [
            node
            for score, node in candidates
            if score >= max(0.7, best_score * 0.45)
        ]
        if not selected:
            return None
        selected.sort(key=lambda node: (node["_bbox"]["x"], node["_bbox"]["y"]))
        self._mark_used(selected)
        if len(selected) == 1:
            return _element(role, selected, frame_w=self.frame_w, frame_h=self.frame_h, text=target_text)
        parent = _element(role, selected, frame_w=self.frame_w, frame_h=self.frame_h, text=target_text)
        parent["children"] = [
            _element(f"{role}_{idx + 1}", [node], frame_w=self.frame_w, frame_h=self.frame_h, text="")
            for idx, node in enumerate(selected)
        ]
        return parent

    def _try_map_brand_vector_cluster(
        self,
        brand_group: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        """Map outlined brand row when Figma exported words/logo as vectors."""
        if not isinstance(brand_group, dict) or not isinstance(brand_group.get("bbox"), dict):
            return None, None, None
        guide_bbox = brand_group["bbox"]
        guide_area = max(_area(guide_bbox), 1e-6)
        visual_types = {"vector", "boolean_operation", "ellipse", "rectangle"}
        candidates: list[dict[str, Any]] = []
        for node in self._available(types=visual_types):
            b = node["_bbox"]
            if _node_type(node) in {"vector", "boolean_operation"} and _node_area_ratio(node) >= 0.30:
                continue
            if _area(b) > guide_area * 1.8:
                continue
            overlaps = _iou(b, guide_bbox) > 0
            near = _center_distance_score(b, guide_bbox) > 0.50
            if overlaps or near:
                candidates.append(node)
        if len(candidates) < 4:
            return None, None, None

        parent_counts: dict[str, int] = {}
        for node in candidates:
            pp = str(node.get("parent_path", "") or "")
            parent_counts[pp] = parent_counts.get(pp, 0) + 1
        common_parent = max(parent_counts.items(), key=lambda item: item[1])[0]
        if not common_parent:
            return None, None, None

        sibling_nodes = [n for n in candidates if str(n.get("parent_path", "") or "") == common_parent]
        nested_nodes = [
            n
            for n in candidates
            if str(n.get("parent_path", "") or "").startswith(common_parent + "/")
        ]
        if len(sibling_nodes) < 2 or len(nested_nodes) < 2:
            return None, None, None

        nested_nodes.sort(key=lambda n: _area(n["_bbox"]), reverse=True)
        logo_back_node = nested_nodes[0]
        logo_fore_node = nested_nodes[-1]
        logo_nodes = [logo_fore_node, logo_back_node]
        logo_box = _union_bboxes([n["_bbox"] for n in logo_nodes])
        if logo_box is None:
            return None, None, None
        logo_center_x = logo_box["x"] + logo_box["width"] / 2.0

        left_nodes = sorted(
            [n for n in sibling_nodes if n["_bbox"]["x"] + n["_bbox"]["width"] / 2.0 < logo_center_x],
            key=lambda n: n["_bbox"]["x"],
        )
        right_nodes = sorted(
            [n for n in sibling_nodes if n["_bbox"]["x"] + n["_bbox"]["width"] / 2.0 >= logo_center_x],
            key=lambda n: n["_bbox"]["x"],
        )
        if not left_nodes or not right_nodes:
            return None, None, None

        self._mark_used(left_nodes + right_nodes + logo_nodes)

        logo_fore = _element("logo_fore", [logo_fore_node], frame_w=self.frame_w, frame_h=self.frame_h)
        logo_back = _element("logo_back", [logo_back_node], frame_w=self.frame_w, frame_h=self.frame_h)
        logo = _synthetic_node(
            "logo",
            bbox=logo_box,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            children=[logo_fore, logo_back],
        )

        first_q = _q_child(brand_group, "brand_name_first") or _q_child(brand_group, "brand_name")
        second_q = _q_child(brand_group, "brand_name_second")
        brand_first = _element(
            "brand_name_first",
            left_nodes,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            text=str((first_q or {}).get("text", "") or ""),
        )
        if len(left_nodes) > 1:
            brand_first["children"] = [
                _element(f"brand_name_first_{idx + 1}", [node], frame_w=self.frame_w, frame_h=self.frame_h)
                for idx, node in enumerate(left_nodes)
            ]
        brand_second = _element(
            "brand_name_second",
            right_nodes,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            text=str((second_q or {}).get("text", "") or ""),
        )
        if len(right_nodes) > 1:
            brand_second["children"] = [
                _element(f"brand_name_second_{idx + 1}", [node], frame_w=self.frame_w, frame_h=self.frame_h)
                for idx, node in enumerate(right_nodes)
            ]
        return brand_first, logo, brand_second

    def pick_remaining_first(self, role: str, types: set[str]) -> dict[str, Any] | None:
        nodes = self._available(types=types)
        if not nodes:
            return None
        nodes = sorted(nodes, key=lambda n: (_area(n["_bbox"]), n["_bbox"]["y"], n["_bbox"]["x"]), reverse=True)
        node = nodes[0]
        self._mark_used([node])
        return _element(
            role,
            [node],
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            text=_text_of_node(node) if _node_type(node) == "text" else "",
        )

    def pick_bg_shape(self) -> dict[str, Any] | None:
        candidates = [
            node
            for node in self._available(types={"vector", "boolean_operation"})
            if _node_area_ratio(node) > 0.50
        ]
        if not candidates:
            return self.pick_remaining_first("bg_shape", {"vector", "boolean_operation"})
        candidates.sort(key=lambda node: _node_area_ratio(node), reverse=True)
        node = candidates[0]
        self._mark_used([node])
        return _element("bg_shape", [node], frame_w=self.frame_w, frame_h=self.frame_h)

    def pick_hero_image(self, guide: dict[str, Any] | None) -> dict[str, Any] | None:
        if guide is not None:
            candidates = [
                node
                for node in self._available(types=None)
                if not (_node_type(node) in {"vector", "boolean_operation"} and _node_area_ratio(node) > 0.50)
            ]
            best_node = None
            best_score = -1.0
            for node in candidates:
                score = self._score(node, guide)
                if score > best_score:
                    best_node = node
                    best_score = score
            if best_node is not None and best_score >= 0.01:
                self._mark_used([best_node])
                return _element("hero_image", [best_node], frame_w=self.frame_w, frame_h=self.frame_h)
        hero_types = {"image", "rectangle", "ellipse", "frame", "instance"}
        return self.pick_remaining_first("hero_image", hero_types)

    def build(self) -> dict[str, Any]:
        brand_g = _q_group(self.qwen_json, "brand_group")
        logo_q = _q_child(brand_g, "logo")
        logo_fore_q = _nested_q_child(logo_q, "logo_fore") or _q_child(brand_g, "logo_fore")
        logo_back_q = _nested_q_child(logo_q, "logo_back") or _q_child(brand_g, "logo_back")
        headline_g = _q_group(self.qwen_json, "headline_group")
        legal_g = _q_group(self.qwen_json, "legal_text_group")
        hero_g = _q_group(self.qwen_json, "hero_image_group")

        brand_first, logo, brand_second = self._try_map_brand_vector_cluster(brand_g)
        if brand_first is None and logo is None and brand_second is None:
            logo_fore = self.pick_one("logo_fore", logo_fore_q, types={"vector", "boolean_operation", "ellipse", "rectangle"})
            logo_back = self.pick_one("logo_back", logo_back_q, types={"ellipse", "rectangle", "vector"})
            logo_children = [x for x in (logo_fore, logo_back) if x is not None]
            logo = _synthetic_node(
                "logo",
                bbox=_union_bboxes([_bbox_for_output_node(c, self.frame_w, self.frame_h) for c in logo_children if c is not None]),
                frame_w=self.frame_w,
                frame_h=self.frame_h,
                children=logo_children,
            )

            brand_second_q = _q_child(brand_g, "brand_name_second")
            brand_first_q = _q_child(brand_g, "brand_name_first") or _q_child(brand_g, "brand_name")
            brand_second = self.pick_brand_visual_text("brand_name_second", brand_second_q, str((brand_second_q or {}).get("text", "") or ""))
            brand_first = self.pick_brand_visual_text("brand_name_first", brand_first_q, str((brand_first_q or {}).get("text", "") or ""))
        brand_children = [x for x in (brand_first, logo, brand_second) if x is not None]
        brand_group = _synthetic_node(
            "brand_group",
            bbox=_union_bboxes([_bbox_for_output_node(c, self.frame_w, self.frame_h) for c in brand_children]),
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            children=brand_children,
        )

        legal_q = _q_child(legal_g, "legal_text")
        legal = self.pick_text("legal_text", legal_q, str((legal_q or {}).get("text", "") or ""))
        legal_group = _synthetic_node(
            "legal_text_group",
            bbox=_bbox_for_output_node(legal, self.frame_w, self.frame_h) if legal else None,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            children=[legal] if legal else [],
        )

        headline_q = _q_child(headline_g, "headline")
        subheadline_q = _q_child(headline_g, "subheadline") or _q_child(headline_g, "subheadline_delivery_time")
        headline = self.pick_text("headline", headline_q, str((headline_q or {}).get("text", "") or ""))
        subheadline = self.pick_text("subheadline", subheadline_q, str((subheadline_q or {}).get("text", "") or "")) if subheadline_q else None
        headline_children = [x for x in (headline, subheadline) if x is not None]
        headline_group = _synthetic_node(
            "headline_group",
            bbox=_union_bboxes([_bbox_for_output_node(c, self.frame_w, self.frame_h) for c in headline_children]),
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            children=headline_children,
        )

        age = self.pick_remaining_first("age_badge", {"text"})
        age_group = _synthetic_node(
            "age_badge_group",
            bbox=_bbox_for_output_node(age, self.frame_w, self.frame_h) if age else None,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            children=[age] if age else [],
        )

        hero_q = _q_child(hero_g, "hero_image")
        hero = self.pick_hero_image(hero_q)
        hero_group = _synthetic_node(
            "hero_image_group",
            bbox=_bbox_for_output_node(hero, self.frame_w, self.frame_h) if hero else None,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            children=[hero] if hero else [],
        )

        star_nodes = []
        for idx in range(2):
            star = self.pick_remaining_first(f"star_{idx + 1}", {"star"})
            if star:
                star_nodes.append(star)
        star_group = _synthetic_node(
            "star_group",
            bbox=_union_bboxes([_bbox_for_output_node(s, self.frame_w, self.frame_h) for s in star_nodes]),
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            children=star_nodes,
        )

        bg_shape = self.pick_bg_shape()
        bg_shape_group = _synthetic_node(
            "bg_shape_group",
            bbox=_bbox_for_output_node(bg_shape, self.frame_w, self.frame_h) if bg_shape else None,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            children=[bg_shape] if bg_shape else [],
        )

        glow_nodes = []
        for idx in range(2):
            glow = self.pick_remaining_first(f"glow_{idx + 1}", {"rectangle", "ellipse"})
            if glow:
                glow_nodes.append(glow)
        glow_group = _synthetic_node(
            "glow_group",
            bbox=_union_bboxes([_bbox_for_output_node(g, self.frame_w, self.frame_h) for g in glow_nodes]),
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            children=glow_nodes,
        )

        groups = [brand_group, headline_group, legal_group, age_group, hero_group, star_group, bg_shape_group, glow_group]
        final_root = deepcopy(self.mid_json)
        final_root["children"] = groups
        final_root["final_json"] = {
            "version": 1,
            "description": "Qwen-guided hierarchy; mapped nodes preserve mid_json properties except semantic name.",
        }
        return final_root


def build_final_json(mid_json: dict[str, Any], qwen_json: dict[str, Any]) -> dict[str, Any]:
    return _Mapper(mid_json, qwen_json).build()


def build_final_json_from_paths(mid_json_path: str | Path, qwen_json_path: str | Path) -> dict[str, Any]:
    mid_json = json.loads(Path(mid_json_path).read_text(encoding="utf-8"))
    qwen_json = json.loads(Path(qwen_json_path).read_text(encoding="utf-8"))
    return build_final_json(mid_json, qwen_json)
