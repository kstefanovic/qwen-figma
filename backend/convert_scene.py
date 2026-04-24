from __future__ import annotations

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
    """
    Produce layout payload for clone-and-apply (no placeholder scene nodes).
    Returns mode, frame size, and updates with pixel bounds from bbox_canvas.
    """
    tw = int(target_width)
    th = int(target_height)
    id_to_path = build_id_to_path_map(raw_json)
    id_to_name = _build_id_to_name_map(raw_json)

    elements = semantic_graph.get("elements") or []
    if not isinstance(elements, list):
        elements = []

    updates: list[dict[str, Any]] = []
    for el in elements:
        if not isinstance(el, dict):
            continue
        sfid = el.get("source_figma_id")
        if sfid is None:
            continue
        sfid_s = str(sfid)
        bx, by, bw, bh = _bbox_canvas_scale(el)
        x = bx * tw
        y = by * th
        width = bw * tw
        height = bh * th
        role_raw = el.get("role", "")
        role_s = str(role_raw).lower() if role_raw is not None else ""
        path_val = id_to_path.get(sfid_s)
        name_val = id_to_name.get(sfid_s)

        updates.append(
            {
                "source_figma_id": sfid_s,
                "path": path_val,
                "name": name_val,
                "role": role_s,
                "bounds": {
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "width": round(width, 2),
                    "height": round(height, 2),
                },
            }
        )

    return {
        "mode": "apply_to_clone",
        "frame": {"width": tw, "height": th},
        "updates": updates,
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
