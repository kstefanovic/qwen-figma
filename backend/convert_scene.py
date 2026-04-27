from __future__ import annotations

import re
from typing import Any


def _child_path(parent_path: str, index: int) -> str:
    seg = str(index)
    return seg if parent_path == "" else f"{parent_path}/{seg}"


def build_id_to_path_map(raw_json: dict[str, Any]) -> dict[str, str]:
    """
    Map each node's Figma id to a stable child-index path from the export root.
    Root path is \"\"; children are \"0\", \"1\", nested \"1/2\", etc.
    If a node includes a string \"path\" field, that value is stored for its id (plugin-provided).
    """
    out: dict[str, str] = {}

    def walk(node: dict[str, Any], computed_path: str) -> None:
        explicit = node.get("path")
        explicit_s = explicit if isinstance(explicit, str) else None
        effective = explicit_s if explicit_s is not None else computed_path
        nid = str(node.get("id", ""))
        if nid:
            out[nid] = effective
        children = node.get("children")
        if not isinstance(children, list):
            return
        for i, ch in enumerate(children):
            if isinstance(ch, dict):
                walk(ch, _child_path(effective, i))

    walk(raw_json, "")
    return out


def _build_id_to_name_map(raw_json: dict[str, Any]) -> dict[str, str | None]:
    names: dict[str, str | None] = {}

    def walk(node: dict[str, Any]) -> None:
        nid = str(node.get("id", ""))
        if nid:
            n = node.get("name")
            names[nid] = n if isinstance(n, str) else (str(n) if n is not None else None)
        children = node.get("children")
        if not isinstance(children, list):
            return
        for ch in children:
            if isinstance(ch, dict):
                walk(ch)

    walk(raw_json)
    return names


def _build_raw_indexes(raw_json: dict[str, Any]) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    by_path: dict[str, dict[str, Any]] = {}

    def walk(node: dict[str, Any], computed_path: str, parent_path: str | None) -> dict[str, Any]:
        explicit = node.get("path")
        explicit_s = explicit if isinstance(explicit, str) else None
        path = explicit_s if explicit_s is not None else computed_path
        node_id = str(node.get("id", "") or "")
        name = node.get("name")
        text = node.get("characters") if isinstance(node.get("characters"), str) else None
        bounds = node.get("bounds") if isinstance(node.get("bounds"), dict) else {}
        info = {
            "id": node_id,
            "path": path,
            "parent_path": parent_path,
            "name": name if isinstance(name, str) else (str(name) if name is not None else None),
            "type": str(node.get("type", "") or "").lower(),
            "text": text,
            "bounds": bounds,
            "visible": bool(node.get("visible", True)),
            "children_paths": [],
        }
        if node_id:
            by_id[node_id] = info
        by_path[path] = info

        children = node.get("children")
        if isinstance(children, list):
            for i, child in enumerate(children):
                if not isinstance(child, dict):
                    continue
                child_info = walk(child, _child_path(path, i), path)
                info["children_paths"].append(child_info["path"])
        return info

    root = walk(raw_json, "", None)
    return {
        "by_id": by_id,
        "by_path": by_path,
        "root": root,
        "root_width": max(1.0, float((raw_json.get("bounds") or {}).get("width", 1.0) or 1.0)),
        "root_height": max(1.0, float((raw_json.get("bounds") or {}).get("height", 1.0) or 1.0)),
    }


def _path_depth(path: str | None) -> int:
    if not path:
        return 0
    return len([seg for seg in path.split("/") if seg != ""])


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_") or "node"


def _node_area(node_info: dict[str, Any]) -> float:
    bounds = node_info.get("bounds") or {}
    try:
        return max(0.0, float(bounds.get("width", 0.0))) * max(0.0, float(bounds.get("height", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _node_aspect(node_info: dict[str, Any]) -> float:
    bounds = node_info.get("bounds") or {}
    try:
        w = max(1e-6, float(bounds.get("width", 0.0)))
        h = max(1e-6, float(bounds.get("height", 0.0)))
        return w / h
    except (TypeError, ValueError):
        return 1.0


def _node_center_x(node_info: dict[str, Any]) -> float:
    bounds = node_info.get("bounds") or {}
    try:
        return float(bounds.get("x", 0.0)) + float(bounds.get("width", 0.0)) * 0.5
    except (TypeError, ValueError):
        return 0.0


def _is_container_node(node_info: dict[str, Any] | None) -> bool:
    if not node_info:
        return False
    if node_info.get("children_paths"):
        return True
    return str(node_info.get("type", "")).lower() in {"group", "frame", "component", "instance"}


def _glow_export_label(
    node_info: dict[str, Any] | None, *, root_width: float, root_height: float
) -> tuple[str, str] | None:
    """
    Named glow / soft-edge blobs (e.g. glow_left) are background accents, not logos.
    Returns (semantic_name, role) or None.
    """
    if not node_info or node_info.get("children_paths"):
        return None
    nm = str(node_info.get("name") or "").lower()
    if "glow" not in nm and "glow" not in str(node_info.get("path") or "").lower():
        return None
    tl = str(node_info.get("type", "")).lower()
    if tl not in {"ellipse", "vector", "rectangle"}:
        return None
    bounds = node_info.get("bounds") or {}
    try:
        x = float(bounds.get("x", 0.0))
        w = float(bounds.get("width", 0.0))
        cx = x + w * 0.5
    except (TypeError, ValueError):
        return None
    if cx < 0.38 * root_width:
        side = "left"
    elif cx > 0.62 * root_width:
        side = "right"
    else:
        side = "part"
    return (f"background_glow_{side}", "background_shape")


def _infer_container_export_from_children(
    node_info: dict[str, Any],
    by_path: dict[str, dict[str, Any]],
) -> tuple[str, str] | None:
    """
    Map a mis-decorated frame/group to headline_group, brand_group, or decoration_group
    from raw Figma children (text vs shapes only).
    """
    paths = node_info.get("children_paths") or []
    children: list[dict[str, Any]] = []
    for p in paths:
        c = by_path.get(p)
        if isinstance(c, dict):
            children.append(c)
    if not children:
        return None

    texts = [
        c
        for c in children
        if (str(c.get("text") or "").strip()) or str(c.get("type", "")).lower() == "text"
    ]
    blob = " ".join(str(c.get("text") or "").strip() for c in texts).lower()

    brand_name_hit = any(
        k in blob
        for k in (
            "яндекс",
            "yandex",
            "лавка",
            "lavka",
            "маркет",
            "market",
        )
    )
    brand_child_hit = any(
        any(
            t in str(c.get("name") or "").lower()
            for t in ("logo", "brand", "яндекс", "yandex", "лавка", "lavka", "маркет", "market")
        )
        for c in children
    )

    if texts:
        if brand_name_hit or brand_child_hit:
            return ("brand_group", "brand_group")
        return ("headline_group", "headline_group")

    shapeish = {"vector", "star", "ellipse", "rectangle", "boolean_operation", "line"}
    if all(str(c.get("type", "")).lower() in shapeish for c in children):
        return ("decoration_group", "decoration_group")

    return None


def _looks_like_background_glow(node_info: dict[str, Any] | None, *, root_width: float, root_height: float) -> bool:
    if not node_info or node_info.get("children_paths") or not node_info.get("visible", True):
        return False
    if str(node_info.get("type", "")).lower() not in {"ellipse", "vector"}:
        return False
    if not _is_generic_figma_name(node_info.get("name")):
        return False
    bounds = node_info.get("bounds") or {}
    try:
        x = float(bounds.get("x", 0.0))
        y = float(bounds.get("y", 0.0))
        width = float(bounds.get("width", 0.0))
        height = float(bounds.get("height", 0.0))
    except (TypeError, ValueError):
        return False
    if width <= 0.0 or height <= 0.0:
        return False
    aspect = width / max(height, 1e-6)
    very_tall = height >= 1.3 * root_height
    edge_or_overflow = y >= 0.75 * root_height or x <= -0.05 * root_width or (x + width) >= 1.05 * root_width
    return very_tall and 0.18 <= aspect <= 0.75 and edge_or_overflow


def _looks_like_overflow_hero_image(node_info: dict[str, Any] | None, *, root_width: float, root_height: float) -> bool:
    if not node_info:
        return False
    bounds = node_info.get("bounds") or {}
    try:
        x = float(bounds.get("x", 0.0))
        y = float(bounds.get("y", 0.0))
        width = float(bounds.get("width", 0.0))
        height = float(bounds.get("height", 0.0))
    except (TypeError, ValueError):
        return False
    if width <= 0.0 or height <= 0.0:
        return False
    overflow = x < -0.02 * root_width or y < -0.02 * root_height or (x + width) > 1.02 * root_width or (y + height) > 1.02 * root_height
    large_enough = width >= 0.7 * root_width and height >= 0.75 * root_height
    right_weighted = _node_center_x(node_info) >= 0.58 * root_width
    return overflow and large_enough and right_weighted


def _scale_raw_node_bounds(
    node_info: dict[str, Any],
    *,
    target_width: int,
    target_height: int,
    root_width: float,
    root_height: float,
) -> dict[str, float]:
    bounds = node_info.get("bounds") or {}
    try:
        x = float(bounds.get("x", 0.0)) / root_width * target_width
        y = float(bounds.get("y", 0.0)) / root_height * target_height
        width = float(bounds.get("width", 0.0)) / root_width * target_width
        height = float(bounds.get("height", 0.0)) / root_height * target_height
    except (TypeError, ValueError):
        x = y = width = height = 0.0
    return {
        "x": round(x, 2),
        "y": round(y, 2),
        "width": round(width, 2),
        "height": round(height, 2),
    }


def _common_ancestor_path(paths: list[str]) -> str | None:
    if not paths:
        return None
    split_paths = [p.split("/") if p else [] for p in paths]
    prefix: list[str] = []
    for items in zip(*split_paths):
        if len(set(items)) != 1:
            break
        prefix.append(items[0])
    return "/".join(prefix)


def _normalize_text(text: str | None) -> str:
    return (text or "").strip()


def _compact_text(text: str | None) -> str:
    return re.sub(r"\s+", "", _normalize_text(text).lower())


def _matches_age_badge(text: str | None) -> bool:
    return bool(re.fullmatch(r"(0|3|6|12|16|18)\+", _compact_text(text)))


def _looks_like_price_text(text: str | None) -> bool:
    raw = _normalize_text(text).lower()
    if not raw or _matches_age_badge(raw):
        return False
    if re.search(r"\d[\d\s]*([.,]\d{1,2})?\s*(₽|\$|€|руб|руб\.|₸|грн)", raw):
        return True
    pricing_tokens = ["₽", "$", "€", "руб", "руб.", "₸", "грн", "от ", "за ", "скид", "%", "sale", "off"]
    return any(token in raw for token in pricing_tokens) and any(ch.isdigit() for ch in raw)


def _semantic_name_from_text(text: str | None) -> str | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None
    lower = normalized.lower()
    compact = _compact_text(normalized)
    if "яндекс" in lower or "yandex" in lower or compact == "яндекс":
        return "brand_name_yandex"
    if "маркет" in lower or "market" in lower:
        return "brand_name_market"
    if "лавка" in lower or "lavka" in lower:
        return "brand_name_lavka"
    if _matches_age_badge(normalized):
        return "age_badge"
    if _looks_like_price_text(normalized):
        return "price_group"
    return None


def _is_generic_figma_name(name: str | None) -> bool:
    raw = (name or "").strip().lower()
    if not raw:
        return True
    if raw.startswith(("group ", "frame ", "rectangle ", "vector ", "ellipse ", "star ")):
        return True
    return bool(re.fullmatch(r"\d+", raw))


def _is_generic_semantic_name(name: str | None) -> bool:
    raw = (name or "").strip().lower()
    return raw in {"", "unknown", "node", "text", "group", "visual_asset", "brand_text"}


def _is_decoration_semantic_label(name: str | None) -> bool:
    """Qwen often over-uses decoration_sparkle; treat these as decoration-only labels."""
    n = (name or "").strip().lower()
    if not n.startswith("decoration_"):
        return False
    if n in {"decoration_group"}:
        return False
    return True


def _coerce_scene_element_semantic_name(
    role: str,
    scene_sn: str | None,
    baseline: str,
    node_info: dict[str, Any] | None,
) -> str:
    """
    If the VLM assigned a decoration_* name to a headline/brand/legal/badge node, drop the
    override and keep the merge-graph / heuristic baseline.
    """
    s = (scene_sn or "").strip()
    if not s:
        return baseline
    r = (role or "").lower()
    if r in {"decoration", "decoration_group"} or r.startswith("decoration"):
        return s
    if _is_decoration_semantic_label(s):
        return baseline
    ni = node_info or {}
    if (ni.get("text") or "").strip() and s.startswith("decoration_"):
        return baseline
    if str(ni.get("type", "")).lower() == "text" and s.startswith("decoration_"):
        return baseline
    return s


def _coerce_scene_group_semantic_name(group_role: str, ann_sn: str | None) -> str:
    s = (ann_sn or "").strip()
    gr = (group_role or "").lower()
    if gr != "decoration_group" and _is_decoration_semantic_label(s):
        return ""
    return s


def _is_member_style_decoration_group_name(name: str | None) -> bool:
    """VLM often puts per-element names (sparkle/star) on whole groups."""
    sl = (name or "").strip().lower()
    if not sl:
        return False
    if sl in {"decoration_sparkle", "decoration_star", "decoration_glow"}:
        return True
    return sl.startswith("decoration_sparkle_") or sl.startswith("decoration_star_")


def _finalize_group_export_semantic_name(group_role: str, semantic_name: str | None) -> str:
    """Never export a frame/group layer as decoration_sparkle — use *_group defaults."""
    s = (semantic_name or "").strip()
    if _is_member_style_decoration_group_name(s):
        return _default_group_semantic_name(group_role)
    gr = (group_role or "").lower()
    if _is_decoration_semantic_label(s) and gr != "decoration_group":
        return _default_group_semantic_name(group_role)
    return s or _default_group_semantic_name(group_role)


def _finalize_layer_export_labels(
    semantic_name: str | None,
    role: str | None,
    node_info: dict[str, Any] | None,
    *,
    by_path: dict[str, Any] | None = None,
    root_width: float = 1920.0,
    root_height: float = 1080.0,
) -> tuple[str, str]:
    """
    Last-pass repair for Figma renames: merge graph + VLM often label text/frames as
    decoration_sparkle or visual_asset. Use raw_json text/type/children to recover
    headline_group, brand_group, decoration_group, background glows, and stars.
    """
    ni = node_info or {}
    name = (semantic_name or "").strip()
    r = (role or "unknown").lower()
    tl = str(ni.get("type", "")).lower()
    txt = (ni.get("text") or "").strip()
    n_low = name.lower()
    mis_dec = _is_decoration_semantic_label(name)
    loose_visual = n_low in {"visual_asset", "visual_group", ""}
    childless = not ni.get("children_paths")
    nm = str(ni.get("name") or "").lower()
    leaf_shapes = {"rectangle", "ellipse", "vector", "boolean_operation", "line", "star"}
    wrong_shape_name = mis_dec or loose_visual or n_low == "logo_mark"

    def _glow_side_from_bounds() -> str:
        try:
            bb = ni.get("bounds") or {}
            cx = float(bb.get("x", 0.0)) + float(bb.get("width", 0.0)) * 0.5
            if cx < 0.38 * root_width:
                return "left"
            if cx > 0.62 * root_width:
                return "right"
        except (TypeError, ValueError):
            pass
        return "part"

    # Leaf glow blobs (Figma name e.g. glow_left) mislabeled logo_mark / visual_asset / decoration_*
    if tl in leaf_shapes and childless:
        glow = _glow_export_label(ni, root_width=root_width, root_height=root_height)
        if glow:
            return glow
        if _looks_like_background_glow(ni, root_width=root_width, root_height=root_height):
            return (f"background_glow_{_glow_side_from_bounds()}", "background_shape")
        if tl == "star" or ("star" in nm and wrong_shape_name):
            return ("decoration_star", "decoration")
        if mis_dec:
            bb = ni.get("bounds") or {}
            try:
                mw = max(float(bb.get("width", 0) or 0), float(bb.get("height", 0) or 0))
            except (TypeError, ValueError):
                mw = 0.0
            if mw < 100.0 and tl == "vector" and ("star" in nm or "spark" in nm):
                return ("decoration_star", "decoration")
            if mw >= 160.0:
                return ("logo_mark", "brand_mark")
            return ("decoration_sparkle", "decoration")

    # Text mislabeled as decoration_* or visual_asset
    if (mis_dec or loose_visual) and (txt or tl == "text"):
        st = _semantic_name_from_text(ni.get("text"))
        if st:
            if st == "age_badge":
                return (st, "age_badge")
            if st == "price_group":
                return (st, "price_main")
            if st.startswith("brand_name"):
                return (st, "brand_text")
            return (st, "headline")
        if _matches_age_badge(ni.get("text")):
            return ("age_badge", "age_badge")
        if _looks_like_price_text(ni.get("text")):
            return ("price_group", "price_main")
        if len(txt) > 96:
            return ("legal_text", "legal")
        return ("headline", "headline")

    # Frames/groups: infer headline_group / brand_group / decoration_group from children
    if tl in {"group", "frame", "component", "instance"}:
        if by_path and (mis_dec or loose_visual):
            inferred = _infer_container_export_from_children(ni, by_path)
            if inferred:
                return inferred
        if mis_dec:
            if r.endswith("_group"):
                return (_default_group_semantic_name(r), r)
            return ("decoration_group", "decoration_group")
        if loose_visual and r.endswith("_group"):
            return (_default_group_semantic_name(r), r)

    if not mis_dec:
        return (name or "visual_asset", r)

    return ("visual_asset", r if r != "unknown" else "decoration")


def _default_semantic_name(role: str, node_info: dict[str, Any] | None = None) -> str:
    text_name = _semantic_name_from_text(node_info.get("text") if node_info else None)
    if text_name is not None and text_name != "price_group":
        return text_name
    has_children = bool(node_info and node_info.get("children_paths"))
    mapping = {
        "headline": "headline",
        "subheadline": "subheadline",
        "legal": "legal_text",
        "legal_group": "legal_group",
        "headline_group": "headline_group",
        "text_group": "headline_group",
        "brand_group": "brand_group",
        "brand_mark": "brand_group" if has_children else "logo",
        "brand_text": "brand_text",
        "age_badge": "age_badge",
        "badge_group": "badge_group",
        "hero_group": "hero_group",
        "product_group": "hero_group",
        "hero_photo": "hero_image",
        "product_image": "product_item",
        "background_shape": "background_shape",
        "background_panel": "background_shape",
        "background_group": "background_group",
        "logo": "logo",
        "logo_mark": "logo_mark",
        "logo_shape": "logo_ellipse",
        "decoration_group": "decoration_group",
        "decoration": "decoration",
        "price_group": "price_group",
    }
    if role in mapping:
        return mapping[role]
    if node_info and not _is_generic_figma_name(node_info.get("name")):
        return _safe_slug(str(node_info.get("name")))
    return "visual_asset"


def _default_group_semantic_name(group_role: str) -> str:
    mapping = {
        "headline_group": "headline_group",
        "legal_group": "legal_group",
        "brand_group": "brand_group",
        "badge_group": "badge_group",
        "hero_group": "hero_group",
        "product_group": "hero_group",
        "background_group": "background_group",
        "decoration_group": "decoration_group",
        "text_group": "headline_group",
    }
    return mapping.get(group_role, group_role or "visual_group")


def _root_semantic_name(raw_json: dict[str, Any]) -> str:
    name = raw_json.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return "canvas"


def _semantic_reason(role: str, semantic_name: str, reason: str | None = None) -> str:
    if reason:
        return reason
    return f"Assigned from current design semantics as {semantic_name or role}."


def _make_update_from_node(
    node_info: dict[str, Any],
    *,
    role: str,
    semantic_name: str,
    parent_semantic_name: str | None = None,
    confidence: float = 0.0,
    reason: str = "",
    target_width: int,
    target_height: int,
    root_width: float,
    root_height: float,
) -> dict[str, Any]:
    return {
        "source_figma_id": node_info["id"],
        "path": node_info["path"],
        "name": node_info.get("name"),
        "role": role,
        "semantic_name": semantic_name,
        "parent_semantic_name": parent_semantic_name,
        "confidence": float(confidence),
        "reason": _semantic_reason(role, semantic_name, reason),
        "bounds": _scale_raw_node_bounds(
            node_info,
            target_width=target_width,
            target_height=target_height,
            root_width=root_width,
            root_height=root_height,
        ),
    }


def _infer_brand_children(
    group_node: dict[str, Any],
    by_path: dict[str, dict[str, Any]],
    *,
    target_width: int,
    target_height: int,
    root_width: float,
    root_height: float,
    parent_semantic_name: str = "brand_group",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    children = [by_path[p] for p in group_node.get("children_paths", []) if p in by_path]
    if not children:
        return out

    combined_text = " ".join(_normalize_text(child.get("text")) for child in children if _normalize_text(child.get("text")))
    compact_group_text = _compact_text(combined_text)
    has_yandex = any(token in compact_group_text for token in {"яндекс", "yandex"})
    has_market = any(token in compact_group_text for token in {"маркет", "market"})
    has_lavka = any(token in compact_group_text for token in {"лавка", "lavka"})
    logo_center_x: float | None = None
    mark_name = "logo_mark"

    logo_candidates = [c for c in children if c.get("children_paths")]
    if not logo_candidates:
        logo_candidates = sorted(children, key=lambda c: (abs(_node_aspect(c) - 1.0), c["path"]))
    logo_node = logo_candidates[0] if logo_candidates else None
    if logo_node is not None:
        logo_center_x = _node_center_x(logo_node)
        out.append(
            {
                **_make_update_from_node(
                    logo_node,
                    role="logo",
                    semantic_name="logo",
                    parent_semantic_name=parent_semantic_name,
                    confidence=0.95,
                    reason="Top-left grouped node with nested structure selected as logo container.",
                    target_width=target_width,
                    target_height=target_height,
                    root_width=root_width,
                    root_height=root_height,
                ),
            }
        )
        logo_parts = [by_path[p] for p in logo_node.get("children_paths", []) if p in by_path]
        if len(logo_parts) >= 1:
            ellipse_node = max(logo_parts, key=_node_area)
            out.append(
                {
                    **_make_update_from_node(
                        ellipse_node,
                        role="logo_shape",
                        semantic_name="logo_ellipse",
                        parent_semantic_name="logo",
                        confidence=0.88,
                        reason="Largest logo child behaves like the outer ellipse/background shape.",
                        target_width=target_width,
                        target_height=target_height,
                        root_width=root_width,
                        root_height=root_height,
                    ),
                }
            )
            remaining = [p for p in logo_parts if p["path"] != ellipse_node["path"]]
            if remaining:
                mark_node = min(remaining, key=_node_area)
                if has_market:
                    mark_name = "logo_m"
                elif has_lavka:
                    mark_name = "logo_heart"
                out.append(
                    {
                        **_make_update_from_node(
                            mark_node,
                            role="logo_mark",
                            semantic_name=mark_name,
                            parent_semantic_name="logo",
                            confidence=0.84,
                            reason="Logo mark named from current brand context instead of fixed example reuse.",
                            target_width=target_width,
                            target_height=target_height,
                            root_width=root_width,
                            root_height=root_height,
                        ),
                    }
                )

    textish = [c for c in children if logo_node is None or c["path"] != logo_node["path"]]
    textish.sort(key=lambda c: ((c.get("bounds") or {}).get("y", 0.0), (c.get("bounds") or {}).get("x", 0.0)))
    for idx, child in enumerate(textish):
        semantic_name = _semantic_name_from_text(child.get("text"))
        if semantic_name is None:
            compact_text = _compact_text(child.get("text"))
            child_center_x = _node_center_x(child)
            left_of_logo = logo_center_x is not None and child_center_x < logo_center_x
            right_of_logo = logo_center_x is not None and child_center_x > logo_center_x
            if idx == 0 and has_yandex:
                semantic_name = "brand_name_yandex" if compact_text in {"яндекс", "yandex"} else "brand_name_yandex_part"
            elif has_market:
                semantic_name = "brand_name_market" if compact_text in {"маркет", "market"} else "brand_name_market_part"
            elif has_lavka:
                semantic_name = "brand_name_lavka" if compact_text in {"лавка", "lavka"} else "brand_name_lavka_part"
            elif has_yandex and right_of_logo:
                semantic_name = "brand_name_lavka_part"
            elif left_of_logo:
                semantic_name = "brand_name_yandex_part"
            elif right_of_logo and mark_name == "logo_heart":
                semantic_name = "brand_name_lavka_part"
            elif right_of_logo and mark_name == "logo_m":
                semantic_name = "brand_name_market"
            elif right_of_logo:
                semantic_name = "brand_name_lavka_part"
            else:
                semantic_name = "brand_text"
        out.append(
            {
                **_make_update_from_node(
                    child,
                    role="brand_text",
                    semantic_name=semantic_name,
                    parent_semantic_name=parent_semantic_name,
                    confidence=0.78,
                    reason="Brand text naming derived from current text fragments inside the brand group.",
                    target_width=target_width,
                    target_height=target_height,
                    root_width=root_width,
                    root_height=root_height,
                ),
            }
        )
    return out


def _infer_hero_children(
    group_node: dict[str, Any],
    by_path: dict[str, dict[str, Any]],
    *,
    target_width: int,
    target_height: int,
    root_width: float,
    root_height: float,
    parent_semantic_name: str = "hero_group",
) -> list[dict[str, Any]]:
    children = [by_path[p] for p in group_node.get("children_paths", []) if p in by_path]
    if not children:
        return []
    children_sorted = sorted(children, key=_node_area, reverse=True)
    out: list[dict[str, Any]] = []
    if children_sorted:
        first_name = "hero_item" if children_sorted[0].get("type") in {"image", "bitmap"} else "product_item"
        out.append(
            {
                **_make_update_from_node(
                    children_sorted[0],
                    role="product_image",
                    semantic_name=first_name,
                    parent_semantic_name=parent_semantic_name,
                    confidence=0.86,
                    reason="Largest child in hero/product region selected as primary hero asset.",
                    target_width=target_width,
                    target_height=target_height,
                    root_width=root_width,
                    root_height=root_height,
                ),
            }
        )
    if len(children_sorted) >= 2:
        out.append(
            {
                **_make_update_from_node(
                    children_sorted[1],
                    role="product_image",
                    semantic_name="product_item_secondary",
                    parent_semantic_name=parent_semantic_name,
                    confidence=0.82,
                    reason="Second-largest child in hero/product region kept as secondary meaningful asset.",
                    target_width=target_width,
                    target_height=target_height,
                    root_width=root_width,
                    root_height=root_height,
                ),
            }
        )
    return out


def _infer_text_children(
    group_node: dict[str, Any],
    by_path: dict[str, dict[str, Any]],
    *,
    group_role: str,
    target_width: int,
    target_height: int,
    root_width: float,
    root_height: float,
    parent_semantic_name: str | None = None,
) -> list[dict[str, Any]]:
    children = [by_path[p] for p in group_node.get("children_paths", []) if p in by_path]
    if not children:
        return []
    children.sort(key=lambda c: (_node_area(c) * -1.0, (c.get("bounds") or {}).get("y", 0.0)))
    out: list[dict[str, Any]] = []
    for idx, child in enumerate(children):
        if group_role in {"headline_group", "text_group"}:
            if idx == 0:
                role = "headline"
                semantic_name = "headline"
            elif idx == 1:
                role = "subheadline"
                semantic_name = "subheadline"
            else:
                role = "headline"
                semantic_name = f"headline_line_{idx + 1}"
        elif group_role == "legal_group":
            role = "legal"
            semantic_name = "legal_text"
        elif group_role == "badge_group":
            if _matches_age_badge(child.get("text")):
                role = "age_badge"
                semantic_name = "age_badge"
            elif _looks_like_price_text(child.get("text")):
                role = "price_main"
                semantic_name = "price_group"
            else:
                role = "badge"
                semantic_name = "badge"
        else:
            role = "text"
            semantic_name = _default_semantic_name(role, child)
        out.append(
            {
                **_make_update_from_node(
                    child,
                    role=role,
                    semantic_name=semantic_name,
                    parent_semantic_name=parent_semantic_name or group_role,
                    confidence=0.8 if idx == 0 else 0.72,
                    reason="Text child named from current group semantics, text content, and prominence.",
                    target_width=target_width,
                    target_height=target_height,
                    root_width=root_width,
                    root_height=root_height,
                ),
            }
        )
    return out


def _infer_background_children(
    group_node: dict[str, Any],
    by_path: dict[str, dict[str, Any]],
    *,
    target_width: int,
    target_height: int,
    root_width: float,
    root_height: float,
    parent_semantic_name: str = "background_group",
) -> list[dict[str, Any]]:
    children = [by_path[p] for p in group_node.get("children_paths", []) if p in by_path]
    if not children:
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(children, key=_node_area, reverse=True)[:3]:
        role = "background_shape"
        semantic_name = "background_color" if child.get("type") == "rectangle" else "background_shape"
        if _looks_like_overflow_hero_image(child, root_width=root_width, root_height=root_height):
            role = "hero_photo"
            semantic_name = "hero_image"
        out.append(
            _make_update_from_node(
                child,
                role=role,
                semantic_name=semantic_name,
                parent_semantic_name=parent_semantic_name,
                confidence=0.72,
                reason="Large background child retained as part of the background system.",
                target_width=target_width,
                target_height=target_height,
                root_width=root_width,
                root_height=root_height,
            )
        )
    return out


def _infer_decoration_children(
    group_node: dict[str, Any],
    by_path: dict[str, dict[str, Any]],
    *,
    target_width: int,
    target_height: int,
    root_width: float,
    root_height: float,
    parent_semantic_name: str = "decoration_group",
) -> list[dict[str, Any]]:
    children = [by_path[p] for p in group_node.get("children_paths", []) if p in by_path]
    out: list[dict[str, Any]] = []
    for child in children:
        ct = str(child.get("type", "")).lower()
        txt = (child.get("text") or "").strip()
        if txt or ct == "text":
            st = _semantic_name_from_text(child.get("text"))
            if st:
                role, semantic_name = (
                    ("age_badge", "age_badge")
                    if st == "age_badge"
                    else ("price_main", "price_group")
                    if st == "price_group"
                    else ("brand_text", st)
                    if st.startswith("brand_name")
                    else ("headline", st)
                )
            elif _matches_age_badge(child.get("text")):
                role, semantic_name = "age_badge", "age_badge"
            elif _looks_like_price_text(child.get("text")):
                role, semantic_name = "price_main", "price_group"
            elif len(txt) > 96:
                role, semantic_name = "legal", "legal_text"
            else:
                role, semantic_name = "headline", "headline"
        elif ct == "star":
            role, semantic_name = "decoration", "decoration_star"
        else:
            role, semantic_name = "decoration", "decoration_sparkle"
        out.append(
            _make_update_from_node(
                child,
                role=role,
                semantic_name=semantic_name,
                parent_semantic_name=parent_semantic_name,
                confidence=0.7,
                reason="Child under decoration_group: split text vs vector vs star from raw node.",
                target_width=target_width,
                target_height=target_height,
                root_width=root_width,
                root_height=root_height,
            )
        )
    return out


def _infer_leaf_group_update(
    group_role: str,
    node_info: dict[str, Any],
    *,
    target_width: int,
    target_height: int,
    root_width: float,
    root_height: float,
    parent_semantic_name: str,
) -> dict[str, Any] | None:
    role = group_role
    semantic_name = _default_semantic_name(group_role, node_info)
    confidence = 0.78
    reason = f"Leaf node kept as {parent_semantic_name} content without collapsing it onto the canvas root."
    if group_role == "badge_group":
        if _matches_age_badge(node_info.get("text")):
            role = "age_badge"
            semantic_name = "age_badge"
        elif _looks_like_price_text(node_info.get("text")):
            role = "price_main"
            semantic_name = "price_group"
        else:
            role = "badge"
            semantic_name = "badge"
    elif group_role == "decoration_group":
        tl = str(node_info.get("type", "")).lower()
        txt = (node_info.get("text") or "").strip()
        if txt or tl == "text":
            st = _semantic_name_from_text(node_info.get("text"))
            if st:
                role = (
                    "age_badge"
                    if st == "age_badge"
                    else "price_main"
                    if st == "price_group"
                    else "brand_text"
                    if st.startswith("brand_name")
                    else "headline"
                )
                semantic_name = st
            elif _matches_age_badge(node_info.get("text")):
                role, semantic_name = "age_badge", "age_badge"
            elif _looks_like_price_text(node_info.get("text")):
                role, semantic_name = "price_main", "price_group"
            elif len(txt) > 96:
                role, semantic_name = "legal", "legal_text"
            else:
                role, semantic_name = "headline", "headline"
        elif tl == "star":
            role, semantic_name = "decoration", "decoration_star"
        else:
            role, semantic_name = "decoration", "decoration_sparkle"
    elif group_role in {"hero_group", "product_group"}:
        role = "hero_photo"
        semantic_name = "hero_image"
    elif group_role == "background_group" and _looks_like_overflow_hero_image(node_info, root_width=root_width, root_height=root_height):
        role = "hero_photo"
        semantic_name = "hero_image"
    elif group_role == "background_group":
        role = "background_shape"
        semantic_name = "background_color" if node_info.get("type") == "rectangle" else "background_shape"
    elif group_role in {"headline_group", "legal_group", "text_group"} and node_info.get("text"):
        if group_role == "headline_group":
            role = "headline"
            semantic_name = "headline"
        elif group_role == "legal_group":
            role = "legal"
            semantic_name = "legal_text"
        else:
            role = "text"
            semantic_name = _default_semantic_name(role, node_info)
    return _make_update_from_node(
        node_info,
        role=role,
        semantic_name=semantic_name,
        parent_semantic_name=parent_semantic_name,
        confidence=confidence,
        reason=reason,
        target_width=target_width,
        target_height=target_height,
        root_width=root_width,
        root_height=root_height,
    )


def _infer_uncovered_background_glows(
    by_path: dict[str, dict[str, Any]],
    *,
    target_width: int,
    target_height: int,
    root_width: float,
    root_height: float,
    covered_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    candidates = [
        node
        for node in by_path.values()
        if node.get("parent_path") == ""
        and node.get("id")
        and node["id"] not in covered_ids
        and _looks_like_background_glow(node, root_width=root_width, root_height=root_height)
    ]
    if not candidates:
        return [], None
    candidates = sorted(candidates, key=_node_center_x)
    updates: list[dict[str, Any]] = []
    multiple = len(candidates) >= 2
    group_name = "background_glow_group"
    for idx, node in enumerate(candidates):
        side = "left" if idx == 0 else "right" if idx == len(candidates) - 1 else f"part_{idx+1}"
        semantic_name = f"background_glow_{side}" if multiple else "background_glow"
        updates.append(
            _make_update_from_node(
                node,
                role="background_shape",
                semantic_name=semantic_name,
                parent_semantic_name=group_name,
                confidence=0.74,
                reason="Large blurred edge glow preserved as a named background glow instead of being left unlabeled.",
                target_width=target_width,
                target_height=target_height,
                root_width=root_width,
                root_height=root_height,
            )
        )
    group = {
        "id": "group_background_glow_1",
        "role": "background_group",
        "semantic_name": group_name,
        "children": [node["id"] for node in candidates],
        "confidence": 0.74,
        "reason": "Large blurred numeric nodes were merged into one glow/background-effects group from the raw Figma canvas.",
    }
    return updates, group


def _infer_group_path(group_role: str, source_ids: list[str], by_id: dict[str, dict[str, Any]], by_path: dict[str, dict[str, Any]]) -> str | None:
    paths = [by_id[sid]["path"] for sid in source_ids if sid in by_id]
    if not paths:
        return None
    if len(paths) == 1:
        path = paths[0]
        parent_path = by_path.get(path, {}).get("parent_path")
        if parent_path not in {None, ""} and parent_path in by_path:
            parent_type = by_path[parent_path].get("type", "")
            if parent_type in {"group", "frame", "component", "instance"}:
                return parent_path
        return path
    return _common_ancestor_path(paths)


def build_convert_semantic_payload(
    semantic_graph: dict[str, Any],
    raw_json: dict[str, Any],
    target_width: int,
    target_height: int,
    *,
    confidence_by_element_id: dict[str, float] | None = None,
    reason_by_element_id: dict[str, str] | None = None,
    candidate_annotations_by_id: dict[str, dict[str, Any]] | None = None,
    group_annotations_by_id: dict[str, dict[str, Any]] | None = None,
    scene_semantic_updates: list[dict[str, Any]] | None = None,
    scene_semantic_groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tw = int(target_width)
    th = int(target_height)
    confidence_by_element_id = confidence_by_element_id or {}
    reason_by_element_id = reason_by_element_id or {}
    candidate_annotations_by_id = candidate_annotations_by_id or {}
    group_annotations_by_id = group_annotations_by_id or {}
    scene_semantic_updates = scene_semantic_updates or []
    scene_semantic_groups = scene_semantic_groups or []
    index = _build_raw_indexes(raw_json)
    by_id = index["by_id"]
    by_path = index["by_path"]
    root_width = index["root_width"]
    root_height = index["root_height"]
    scene_updates_by_key: dict[tuple[str, str | None], dict[str, Any]] = {}
    for item in scene_semantic_updates:
        if not isinstance(item, dict):
            continue
        figma_id = str(item.get("source_figma_id", "") or "")
        if not figma_id:
            continue
        scene_updates_by_key[(figma_id, item.get("path"))] = item

    merged_updates: dict[tuple[str, str | None], dict[str, Any]] = {}
    semantic_elements: dict[str, dict[str, Any]] = {}
    semantic_groups: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    group_semantic_name_by_id: dict[str, str] = {}
    root_node = index["root"]
    root_name = _root_semantic_name(raw_json)
    if root_node.get("id"):
        merged_updates[(root_node["id"], root_node["path"])] = _make_update_from_node(
            root_node,
            role=str(raw_json.get("type", "frame") or "frame").lower(),
            semantic_name=root_name,
            parent_semantic_name=None,
            confidence=1.0,
            reason="Root canvas preserved from the raw Figma frame name.",
            target_width=tw,
            target_height=th,
            root_width=root_width,
            root_height=root_height,
        )

    elements = semantic_graph.get("elements") or []
    if isinstance(elements, list):
        for el in elements:
            if not isinstance(el, dict):
                continue
            figma_id = str(el.get("source_figma_id", "") or "")
            if not figma_id:
                continue
            node_info = by_id.get(figma_id)
            element_id = str(el.get("id", "") or f"figma_{_safe_slug(figma_id)}")
            candidate_id = element_id.removeprefix("el_")
            candidate_ann = candidate_annotations_by_id.get(candidate_id, {})
            role = str(candidate_ann.get("element_role", "") or el.get("role", "unknown") or "unknown").lower()
            semantic_name = str(candidate_ann.get("semantic_name", "") or "").strip() or _default_semantic_name(role, node_info)
            if role in {"background_shape", "background_panel"} and _looks_like_overflow_hero_image(node_info, root_width=root_width, root_height=root_height):
                role = "hero_photo"
                semantic_name = "hero_image"
            path_val = node_info["path"] if node_info is not None else None
            name_val = node_info.get("name") if node_info is not None else None
            scene_override = scene_updates_by_key.get((figma_id, path_val)) or scene_updates_by_key.get((figma_id, None)) or {}
            scene_sn_raw = str(scene_override.get("semantic_name", "") or "").strip()
            semantic_resolved = _coerce_scene_element_semantic_name(
                role, scene_sn_raw, semantic_name, node_info
            )
            bx, by, bw, bh = _bbox_canvas_scale(el)
            update = {
                "source_figma_id": figma_id,
                "path": path_val,
                "name": name_val,
                "role": str(scene_override.get("role", "") or role).lower(),
                "semantic_name": semantic_resolved,
                "parent_semantic_name": str(scene_override.get("parent_semantic_name", "") or candidate_ann.get("parent_semantic_name", "") or ""),
                "confidence": float(scene_override.get("confidence", candidate_ann.get("confidence", confidence_by_element_id.get(element_id, 0.0))) or 0.0),
                "reason": str(scene_override.get("reason", "") or candidate_ann.get("reason_short", "") or reason_by_element_id.get(element_id, "") or _semantic_reason(role, semantic_name)),
                "bounds": {
                    "x": round(bx * tw, 2),
                    "y": round(by * th, 2),
                    "width": round(bw * tw, 2),
                    "height": round(bh * th, 2),
                },
            }
            fin_sn, fin_r = _finalize_layer_export_labels(
                update["semantic_name"],
                update["role"],
                node_info,
                by_path=by_path,
                root_width=root_width,
                root_height=root_height,
            )
            update["semantic_name"] = fin_sn
            update["role"] = fin_r
            merged_updates[(figma_id, path_val)] = update
            semantic_elements[figma_id] = {
                "id": element_id,
                "figma_node_id": figma_id,
                "path": path_val,
                "role": update["role"],
                "semantic_name": update["semantic_name"],
                "parent_semantic_name": update["parent_semantic_name"] or None,
                "confidence": update["confidence"],
                "reason": update["reason"],
            }

    groups = semantic_graph.get("groups") or []
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict):
                continue
            group_id = str(group.get("id", "") or "")
            group_role = str(group.get("role", "unknown") or "unknown").lower()
            source_ids = [str(x) for x in (group.get("source_figma_ids") or []) if str(x)]
            group_path = _infer_group_path(group_role, source_ids, by_id, by_path)
            group_node = by_path.get(group_path) if group_path is not None else None
            group_candidate_id = group_id.removeprefix("group_")
            group_ann = group_annotations_by_id.get(group_candidate_id, {})
            parent_group_id = str(group.get("parent_group_id", "") or "")
            parent_semantic_name = group_semantic_name_by_id.get(parent_group_id)
            if group_role == "background_group" and _looks_like_overflow_hero_image(group_node, root_width=root_width, root_height=root_height):
                group_role = "hero_group"
            ann_group_sn = _coerce_scene_group_semantic_name(
                group_role, str(group_ann.get("semantic_name", "") or "").strip()
            )
            group_semantic_name = ann_group_sn or _default_group_semantic_name(group_role)
            group_semantic_name = _finalize_group_export_semantic_name(group_role, group_semantic_name)
            group_confidence = float(group_ann.get("confidence", 0.72) or 0.72)
            group_reason = str(group_ann.get("reason_short", "") or f"Grouped as {group_semantic_name} from graph role and raw Figma hierarchy.")
            group_semantic_name_by_id[group_id] = group_semantic_name
            group_key = (group_semantic_name, tuple(sorted(source_ids)))
            if group_role == "decoration_group":
                group_key = (group_semantic_name, ("__all__",))
            existing_group = semantic_groups.get(group_key)
            semantic_groups[group_key] = {
                "id": group_id,
                "role": group_role,
                "semantic_name": group_semantic_name,
                "children": sorted(set((existing_group or {}).get("children", [])) | set(source_ids)),
                "confidence": max(group_confidence, float((existing_group or {}).get("confidence", 0.0) or 0.0)),
                "reason": group_reason,
            }

            if group_node is not None and group_node.get("id") and _is_container_node(group_node):
                group_update = _make_update_from_node(
                    group_node,
                    role=group_role,
                    semantic_name=group_semantic_name,
                    parent_semantic_name=parent_semantic_name,
                    confidence=group_confidence,
                    reason=group_reason,
                    target_width=tw,
                    target_height=th,
                    root_width=root_width,
                    root_height=root_height,
                )
                merged_updates[(group_update["source_figma_id"], group_update["path"])] = group_update
                semantic_elements[group_update["source_figma_id"]] = {
                    "id": group_id,
                    "figma_node_id": group_update["source_figma_id"],
                    "path": group_update.get("path"),
                    "role": group_update["role"],
                    "semantic_name": group_update["semantic_name"],
                    "parent_semantic_name": group_update["parent_semantic_name"] or None,
                    "confidence": group_update["confidence"],
                    "reason": group_update["reason"],
                }

                inferred_children: list[dict[str, Any]] = []
                if group_role == "brand_group":
                    inferred_children = _infer_brand_children(
                        group_node,
                        by_path,
                        target_width=tw,
                        target_height=th,
                        root_width=root_width,
                        root_height=root_height,
                        parent_semantic_name=group_semantic_name,
                    )
                elif group_role in {"hero_group", "product_group"}:
                    inferred_children = _infer_hero_children(
                        group_node,
                        by_path,
                        target_width=tw,
                        target_height=th,
                        root_width=root_width,
                        root_height=root_height,
                        parent_semantic_name=group_semantic_name,
                    )
                elif group_role in {"headline_group", "legal_group", "badge_group", "text_group"}:
                    inferred_children = _infer_text_children(
                        group_node,
                        by_path,
                        group_role=group_role,
                        target_width=tw,
                        target_height=th,
                        root_width=root_width,
                        root_height=root_height,
                        parent_semantic_name=group_semantic_name,
                    )
                elif group_role == "background_group":
                    inferred_children = _infer_background_children(
                        group_node,
                        by_path,
                        target_width=tw,
                        target_height=th,
                        root_width=root_width,
                        root_height=root_height,
                        parent_semantic_name=group_semantic_name,
                    )
                elif group_role == "decoration_group":
                    inferred_children = _infer_decoration_children(
                        group_node,
                        by_path,
                        target_width=tw,
                        target_height=th,
                        root_width=root_width,
                        root_height=root_height,
                        parent_semantic_name=group_semantic_name,
                    )

                for child in inferred_children:
                    scene_override = scene_updates_by_key.get((child["source_figma_id"], child["path"])) or {}
                    ch_node = by_id.get(child["source_figma_id"])
                    ch_role = str(scene_override.get("role", "") or child["role"]).lower()
                    ch_sn = _coerce_scene_element_semantic_name(
                        ch_role,
                        str(scene_override.get("semantic_name", "") or ""),
                        str(child["semantic_name"]),
                        ch_node,
                    )
                    child_update = {
                        **child,
                        "role": ch_role,
                        "semantic_name": ch_sn,
                        "parent_semantic_name": str(scene_override.get("parent_semantic_name", "") or child.get("parent_semantic_name", "") or ""),
                        "confidence": float(scene_override.get("confidence", child.get("confidence", 0.0)) or 0.0),
                        "reason": str(scene_override.get("reason", "") or child.get("reason", "")),
                    }
                    cfs, cfr = _finalize_layer_export_labels(
                        child_update["semantic_name"],
                        child_update["role"],
                        ch_node,
                        by_path=by_path,
                        root_width=root_width,
                        root_height=root_height,
                    )
                    child_update["semantic_name"] = cfs
                    child_update["role"] = cfr
                    merged_updates[(child["source_figma_id"], child["path"])] = child_update
                    figma_id = child["source_figma_id"]
                    semantic_elements[figma_id] = {
                        "id": f"figma_{_safe_slug(figma_id)}",
                        "figma_node_id": figma_id,
                        "path": child.get("path"),
                        "role": child_update["role"],
                        "semantic_name": child_update["semantic_name"],
                        "parent_semantic_name": child_update["parent_semantic_name"] or None,
                        "confidence": child_update["confidence"],
                        "reason": child_update["reason"],
                    }
            elif group_node is not None and group_node.get("id"):
                leaf_update = _infer_leaf_group_update(
                    group_role,
                    group_node,
                    target_width=tw,
                    target_height=th,
                    root_width=root_width,
                    root_height=root_height,
                    parent_semantic_name=group_semantic_name,
                )
                if leaf_update is not None:
                    scene_ov = scene_updates_by_key.get(
                        (leaf_update["source_figma_id"], leaf_update["path"])
                    ) or {}
                    leaf_node = by_id.get(leaf_update["source_figma_id"])
                    lr = str(scene_ov.get("role", "") or leaf_update["role"]).lower()
                    lsn = _coerce_scene_element_semantic_name(
                        lr,
                        str(scene_ov.get("semantic_name", "") or ""),
                        str(leaf_update["semantic_name"]),
                        leaf_node,
                    )
                    lpar = str(scene_ov.get("parent_semantic_name", "") or "").strip() or leaf_update.get(
                        "parent_semantic_name"
                    )
                    leaf_update = {
                        **leaf_update,
                        "role": lr,
                        "semantic_name": lsn,
                        "parent_semantic_name": lpar,
                        "confidence": float(
                            scene_ov.get("confidence", leaf_update.get("confidence", 0.0)) or 0.0
                        ),
                        "reason": str(scene_ov.get("reason", "") or leaf_update.get("reason", "")),
                    }
                    lfn, lfr = _finalize_layer_export_labels(
                        leaf_update["semantic_name"],
                        leaf_update["role"],
                        leaf_node,
                        by_path=by_path,
                        root_width=root_width,
                        root_height=root_height,
                    )
                    leaf_update["semantic_name"] = lfn
                    leaf_update["role"] = lfr
                    merged_updates[(leaf_update["source_figma_id"], leaf_update["path"])] = leaf_update
                    semantic_elements[leaf_update["source_figma_id"]] = {
                        "id": f"figma_{_safe_slug(leaf_update['source_figma_id'])}",
                        "figma_node_id": leaf_update["source_figma_id"],
                        "path": leaf_update.get("path"),
                        "role": leaf_update["role"],
                        "semantic_name": leaf_update["semantic_name"],
                        "parent_semantic_name": leaf_update["parent_semantic_name"] or None,
                        "confidence": leaf_update["confidence"],
                        "reason": leaf_update["reason"],
                    }

    covered_ids = set(semantic_elements)
    glow_updates, glow_group = _infer_uncovered_background_glows(
        by_path,
        target_width=tw,
        target_height=th,
        root_width=root_width,
        root_height=root_height,
        covered_ids=covered_ids,
    )
    for glow_update in glow_updates:
        merged_updates[(glow_update["source_figma_id"], glow_update["path"])] = glow_update
        semantic_elements[glow_update["source_figma_id"]] = {
            "id": f"figma_{_safe_slug(glow_update['source_figma_id'])}",
            "figma_node_id": glow_update["source_figma_id"],
            "path": glow_update.get("path"),
            "role": glow_update["role"],
            "semantic_name": glow_update["semantic_name"],
            "parent_semantic_name": glow_update["parent_semantic_name"] or None,
            "confidence": glow_update["confidence"],
            "reason": glow_update["reason"],
        }
    if glow_group is not None:
        semantic_groups[(glow_group["semantic_name"], tuple(sorted(glow_group["children"])))] = glow_group

    for scene_group in scene_semantic_groups:
        if not isinstance(scene_group, dict):
            continue
        children = [str(x) for x in (scene_group.get("children") or []) if str(x)]
        key = (str(scene_group.get("semantic_name", "") or "visual_group"), tuple(sorted(children)))
        if key not in semantic_groups:
            semantic_groups[key] = {
                "id": str(scene_group.get("id", "") or f"scene_group_{len(semantic_groups)+1}"),
                "role": str(scene_group.get("role", "unknown") or "unknown"),
                "semantic_name": str(scene_group.get("semantic_name", "") or "visual_group"),
                "children": children,
                "confidence": float(scene_group.get("confidence", 0.0) or 0.0),
                "reason": str(scene_group.get("reason", "") or ""),
            }

    updates = sorted(
        merged_updates.values(),
        key=lambda item: (_path_depth(item.get("path")), item.get("path") or "", item.get("source_figma_id") or ""),
    )
    semantic_elements_list = sorted(
        semantic_elements.values(),
        key=lambda item: (_path_depth(item.get("path")), item.get("path") or "", item.get("figma_node_id") or ""),
    )
    semantic_groups_list = list(semantic_groups.values())
    unnamed = [item for item in semantic_elements_list if _is_generic_semantic_name(item.get("semantic_name"))]
    low_confidence_examples = [
        {
            "figma_node_id": item.get("figma_node_id"),
            "path": item.get("path"),
            "semantic_name": item.get("semantic_name"),
            "confidence": item.get("confidence", 0.0),
            "reason": item.get("reason", ""),
        }
        for item in sorted(semantic_elements_list, key=lambda row: float(row.get("confidence", 0.0)))[:8]
    ]
    return {
        "mode": "apply_to_clone",
        "frame": {"width": tw, "height": th},
        "updates": updates,
        "semantic_elements": semantic_elements_list,
        "semantic_groups": semantic_groups_list,
        "nodes_annotated": len(semantic_elements_list),
        "nodes_left_unnamed": len(unnamed),
        "low_confidence_examples": low_confidence_examples,
    }


def _bbox_canvas_scale(el: dict[str, Any]) -> tuple[float, float, float, float]:
    """Normalized bbox_canvas x,y,w,h without clamping (layout math per API contract)."""
    b = el.get("bbox_canvas")
    if not isinstance(b, dict):
        return 0.0, 0.0, 0.0, 0.0
    try:
        return (
            float(b.get("x", 0.0)),
            float(b.get("y", 0.0)),
            float(b.get("w", 0.0)),
            float(b.get("h", 0.0)),
        )
    except (TypeError, ValueError):
        return 0.0, 0.0, 0.0, 0.0


def semantic_graph_to_layout_updates(
    semantic_graph: dict[str, Any],
    raw_json: dict[str, Any],
    target_width: int,
    target_height: int,
) -> dict[str, Any]:
    payload = build_convert_semantic_payload(
        semantic_graph,
        raw_json,
        target_width,
        target_height,
    )
    return {
        "mode": payload["mode"],
        "frame": payload["frame"],
        "updates": payload["updates"],
    }


def _solid_fill(r: float, g: float, b: float, a: float = 1.0) -> list[dict[str, Any]]:
    return [{"type": "SOLID", "color": {"r": r, "g": g, "b": b, "a": a}}]


def _get_bbox(el: dict[str, Any]) -> dict[str, float]:
    b = el.get("bbox_canvas")
    if not isinstance(b, dict):
        return {"x": 0.0, "y": 0.0, "w": 0.05, "h": 0.05}
    try:
        x = float(b.get("x", 0.0))
        y = float(b.get("y", 0.0))
        w = max(1e-6, float(b.get("w", 0.05)))
        h = max(1e-6, float(b.get("h", 0.05)))
        return {"x": max(0.0, min(1.0, x)), "y": max(0.0, min(1.0, y)), "w": w, "h": h}
    except (TypeError, ValueError):
        return {"x": 0.0, "y": 0.0, "w": 0.05, "h": 0.05}


def _bbox_to_pixels(b: dict[str, float], tw: int, th: int) -> tuple[float, float, float, float]:
    x = b["x"] * tw
    y = b["y"] * th
    w = max(1.0, b["w"] * tw)
    h = max(1.0, b["h"] * th)
    return x, y, w, h


def _union_bbox_canvas(elements: list[dict[str, Any]]) -> dict[str, float]:
    if not elements:
        return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    x0, y0 = 1.0, 1.0
    x1, y1 = 0.0, 0.0
    for el in elements:
        b = _get_bbox(el)
        x0 = min(x0, b["x"])
        y0 = min(y0, b["y"])
        x1 = max(x1, b["x"] + b["w"])
        y1 = max(y1, b["y"] + b["h"])
    w = max(1e-4, x1 - x0)
    h = max(1e-4, y1 - y0)
    return {"x": x0, "y": y0, "w": w, "h": h}


def _font_for_role(role: str) -> tuple[int, dict[str, str]]:
    r = role.lower()
    if r == "headline":
        return 48, {"family": "Inter", "style": "Bold"}
    if r == "subheadline":
        return 32, {"family": "Inter", "style": "Semi Bold"}
    if r == "legal":
        return 12, {"family": "Inter", "style": "Regular"}
    if r in {"price_main", "price_old", "price_fraction", "discount_text"}:
        return 28, {"family": "Inter", "style": "Bold"}
    if r == "cta":
        return 22, {"family": "Inter", "style": "Semi Bold"}
    return 18, {"family": "Inter", "style": "Regular"}


def _text_node(
    el: dict[str, Any],
    tw: int,
    th: int,
    *,
    name_suffix: str = "",
) -> dict[str, Any]:
    role = str(el.get("role", "text")).lower()
    bbox = _get_bbox(el)
    x, y, w, h = _bbox_to_pixels(bbox, tw, th)
    fs, font_name = _font_for_role(role)
    text = el.get("text_content")
    if not isinstance(text, str):
        text = text or ""
    fills = _solid_fill(0.12, 0.12, 0.14, 1.0)
    if role == "legal":
        fills = _solid_fill(0.35, 0.35, 0.38, 1.0)
    return {
        "kind": "TEXT",
        "name": f"{role}{name_suffix}",
        "sourceElementId": str(el.get("id", "")) or None,
        "x": round(x, 2),
        "y": round(y, 2),
        "width": round(w, 2),
        "height": round(h, 2),
        "fills": fills,
        "characters": text[:8000],
        "fontSize": fs,
        "fontName": font_name,
    }


def _rect_placeholder(el: dict[str, Any], tw: int, th: int, *, name: str, rgb: tuple[float, float, float]) -> dict[str, Any]:
    bbox = _get_bbox(el)
    x, y, w, h = _bbox_to_pixels(bbox, tw, th)
    return {
        "kind": "RECTANGLE",
        "name": name,
        "sourceElementId": str(el.get("id", "")) or None,
        "x": round(x, 2),
        "y": round(y, 2),
        "width": round(w, 2),
        "height": round(h, 2),
        "fills": _solid_fill(*rgb),
    }


def _ellipse_node(el: dict[str, Any], tw: int, th: int, *, name: str, rgb: tuple[float, float, float]) -> dict[str, Any]:
    bbox = _get_bbox(el)
    x, y, w, h = _bbox_to_pixels(bbox, tw, th)
    return {
        "kind": "ELLIPSE",
        "name": name,
        "sourceElementId": str(el.get("id", "")) or None,
        "x": round(x, 2),
        "y": round(y, 2),
        "width": round(w, 2),
        "height": round(h, 2),
        "fills": _solid_fill(*rgb),
    }


def _age_badge_group(el: dict[str, Any], tw: int, th: int) -> dict[str, Any]:
    bbox = _get_bbox(el)
    x, y, w, h = _bbox_to_pixels(bbox, tw, th)
    side = min(w, h)
    label = el.get("text_content") if isinstance(el.get("text_content"), str) else ""
    if not label.strip():
        label = "18+"
    return {
        "kind": "GROUP",
        "name": "age_badge",
        "sourceElementId": str(el.get("id", "")) or None,
        "x": round(x, 2),
        "y": round(y, 2),
        "width": round(w, 2),
        "height": round(h, 2),
        "fills": [],
        "children": [
            {
                "kind": "ELLIPSE",
                "name": "age_badge_disc",
                "x": 0,
                "y": 0,
                "width": round(side, 2),
                "height": round(side, 2),
                "fills": _solid_fill(0.85, 0.1, 0.15, 1.0),
            },
            {
                "kind": "TEXT",
                "name": "age_badge_label",
                "x": round(side * 0.15, 2),
                "y": round(side * 0.2, 2),
                "width": round(max(1, w - side * 0.2), 2),
                "height": round(max(1, h * 0.6), 2),
                "fills": _solid_fill(1, 1, 1, 1),
                "characters": label[:32],
                "fontSize": max(10, int(side * 0.35)),
                "fontName": {"family": "Inter", "style": "Bold"},
            },
        ],
    }


def _element_to_scene_node(el: dict[str, Any], tw: int, th: int) -> dict[str, Any]:
    role = str(el.get("role", "unknown")).lower()

    text_roles = {
        "headline",
        "subheadline",
        "body_text",
        "legal",
        "price_main",
        "price_old",
        "price_fraction",
        "discount_text",
        "cta",
        "label",
        "badge_text",
        "logo_text",
        "text_container",
    }
    if role in text_roles:
        return _text_node(el, tw, th)

    if role == "age_badge":
        return _age_badge_group(el, tw, th)

    if role in {"background_panel", "background_shape"}:
        return _rect_placeholder(el, tw, th, name=role, rgb=(0.94, 0.94, 0.95))

    if role in {"product_image", "hero_photo", "brand_mark", "logo_icon", "packshot", "image_container"}:
        return _rect_placeholder(el, tw, th, name=f"{role}_placeholder", rgb=(0.78, 0.8, 0.84))

    if role == "decoration":
        bbox = _get_bbox(el)
        ar = bbox["w"] / max(bbox["h"], 1e-6)
        if 0.75 <= ar <= 1.25:
            return _ellipse_node(el, tw, th, name="decoration", rgb=(0.55, 0.7, 0.9))
        return _rect_placeholder(el, tw, th, name="decoration", rgb=(0.55, 0.7, 0.9))

    if role == "discount_badge":
        return _ellipse_node(el, tw, th, name="discount_badge", rgb=(0.95, 0.35, 0.2))

    if role == "promo_container":
        return _rect_placeholder(el, tw, th, name="promo_container", rgb=(0.9, 0.9, 0.92))

    # unknown / fallback
    return _rect_placeholder(el, tw, th, name=f"placeholder_{role}", rgb=(0.88, 0.88, 0.9))


def semantic_graph_to_scene(semantic_graph: dict[str, Any], target_width: int, target_height: int) -> dict[str, Any]:
    """
    Convert pipeline semantic_graph.json (dict) into a simple Figma-friendly scene spec.
    Coordinates are absolute pixels in target_width x target_height (normalized bbox scaled).
    """
    tw = int(target_width)
    th = int(target_height)
    elements = semantic_graph.get("elements") or []
    groups = semantic_graph.get("groups") or []

    if not isinstance(elements, list):
        elements = []
    if not isinstance(groups, list):
        groups = []

    group_by_id: dict[str, dict[str, Any]] = {}
    for g in groups:
        if isinstance(g, dict) and g.get("id"):
            group_by_id[str(g["id"])] = g

    members_by_group: dict[str, list[dict[str, Any]]] = {}
    for el in elements:
        if not isinstance(el, dict):
            continue
        gid = el.get("group_id")
        if not gid:
            continue
        gid = str(gid)
        members_by_group.setdefault(gid, []).append(el)

    nodes: list[dict[str, Any]] = []
    placed_element_ids: set[str] = set()

    for gid, members in members_by_group.items():
        if not members:
            continue
        members_sorted = sorted(
            members,
            key=lambda e: (_get_bbox(e)["y"], _get_bbox(e)["x"]),
        )
        child_nodes = []
        for el in members_sorted:
            eid = str(el.get("id", ""))
            child_nodes.append(_element_to_scene_node(el, tw, th))
            if eid:
                placed_element_ids.add(eid)

        gmeta = group_by_id.get(gid, {})
        grole = str(gmeta.get("role", "group")).lower()
        ub = _union_bbox_canvas(members_sorted)
        gx, gy, gw, gh = _bbox_to_pixels(ub, tw, th)

        nodes.append(
            {
                "kind": "GROUP",
                "name": f"group_{grole}_{gid}",
                "sourceGroupId": gid,
                "x": round(gx, 2),
                "y": round(gy, 2),
                "width": round(gw, 2),
                "height": round(gh, 2),
                "fills": [],
                "children": child_nodes,
            }
        )

    # Orphan elements (no group members bucket — still render flat)
    for el in elements:
        if not isinstance(el, dict):
            continue
        eid = str(el.get("id", ""))
        if eid and eid in placed_element_ids:
            continue
        gid = el.get("group_id")
        if gid and str(gid) in members_by_group:
            continue
        nodes.append(_element_to_scene_node(el, tw, th))

    return {
        "version": 1,
        "name": "Converted Layout",
        "frame": {
            "width": tw,
            "height": th,
            "background": _solid_fill(0.98, 0.98, 0.99, 1.0),
        },
        "nodes": nodes,
    }
