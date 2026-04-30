from __future__ import annotations

from copy import deepcopy
from typing import Any


_WRAPPER_TYPES = {"frame", "group", "component", "component set", "section"}


def _child_path(parent_path: str, index: int) -> str:
    seg = str(index)
    return seg if parent_path == "" else f"{parent_path}/{seg}"


def _is_wrapper_node(node: dict[str, Any]) -> bool:
    node_type = str(node.get("type", "") or "").strip().lower()
    return node_type in _WRAPPER_TYPES


def _node_area(node: dict[str, Any]) -> float:
    bounds = node.get("bounds")
    if not isinstance(bounds, dict):
        return 0.0
    try:
        return max(0.0, float(bounds.get("width", 0) or 0)) * max(
            0.0,
            float(bounds.get("height", 0) or 0),
        )
    except (TypeError, ValueError):
        return 0.0


def _is_visible_node(node: dict[str, Any]) -> bool:
    if node.get("visible", True) is False:
        return False
    try:
        if float(node.get("opacity", 1.0) or 1.0) <= 0:
            return False
    except (TypeError, ValueError):
        pass
    return _node_area(node) > 0


def _copy_leaf_node(node: dict[str, Any], path: str, parent_path: str | None) -> dict[str, Any]:
    leaf = deepcopy(node)
    leaf.pop("children", None)
    leaf["path"] = str(node.get("path") if isinstance(node.get("path"), str) else path)
    leaf["parent_path"] = parent_path
    leaf["source_depth"] = 0 if not leaf["path"] else len([p for p in leaf["path"].split("/") if p])
    return leaf


def build_mid_json(raw_json: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten plugin raw_json into a shallow, pipeline-friendly JSON.

    The root/frame is preserved, but wrapper/group hierarchy below it is removed.
    The resulting root.children contains only deepest visible nodes in parallel.
    """
    if not isinstance(raw_json, dict) or not raw_json:
        raise ValueError("raw_json must be a non-empty object.")

    root = deepcopy(raw_json)
    original_children = raw_json.get("children")
    root["children"] = []
    root["mid_json"] = {
        "version": 1,
        "description": "Root with deepest visible raw_json nodes flattened in parallel.",
    }

    leaves: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], path: str, parent_path: str | None) -> None:
        if not _is_visible_node(node):
            return
        children = node.get("children")
        valid_children = [ch for ch in children if isinstance(ch, dict)] if isinstance(children, list) else []
        if valid_children:
            for i, child in enumerate(valid_children):
                explicit_path = child.get("path")
                child_path = explicit_path if isinstance(explicit_path, str) else _child_path(path, i)
                walk(child, child_path, path if path != "" else None)
            return
        leaves.append(_copy_leaf_node(node, path, parent_path))

    if isinstance(original_children, list):
        for i, child in enumerate(original_children):
            if not isinstance(child, dict):
                continue
            explicit_path = child.get("path")
            child_path = explicit_path if isinstance(explicit_path, str) else _child_path("", i)
            walk(child, child_path, "")

    # If the selected frame has no children, keep it parseable as a single root with no flat leaves.
    root["children"] = leaves
    root["mid_json"]["leaf_count"] = len(leaves)
    root["mid_json"]["removed_wrapper_types"] = sorted(_WRAPPER_TYPES)
    root["mid_json"]["removed_wrapper_count"] = count_removed_wrappers(raw_json)
    return root


def count_removed_wrappers(raw_json: dict[str, Any]) -> int:
    count = 0

    def walk(node: dict[str, Any], *, is_root: bool = False) -> None:
        nonlocal count
        children = node.get("children")
        valid_children = [ch for ch in children if isinstance(ch, dict)] if isinstance(children, list) else []
        if not is_root and valid_children and _is_wrapper_node(node):
            count += 1
        for child in valid_children:
            walk(child, is_root=False)

    if isinstance(raw_json, dict):
        walk(raw_json, is_root=True)
    return count
