from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from banner_pipeline.normalize import NormalizedNode


@dataclass
class CollapsedNode:
    """
    Working-tree node after collapsing meaningless wrapper groups.
    Still not semantic, but much cleaner than raw export structure.
    """

    id: str
    name: str | None
    type: str
    parent_id: str | None
    depth: int

    x: float
    y: float
    width: float
    height: float

    x_norm: float
    y_norm: float
    w_norm: float
    h_norm: float
    center_x_norm: float
    center_y_norm: float

    area_px: float
    area_ratio: float
    aspect_ratio: float

    left_norm: float
    top_norm: float
    right_norm: float
    bottom_norm: float
    dist_left: float
    dist_top: float
    dist_right: float
    dist_bottom: float

    visible: bool
    opacity: float

    text: str | None
    template_id: str | None
    child_ids: list[str]

    layout_mode: str | None
    item_spacing: float | None
    padding: dict[str, float] | None
    extra_data: dict[str, Any]

    source_figma_ids: list[str]
    collapsed_from_ids: list[str]

    @property
    def bbox_canvas(self) -> list[float]:
        return [self.x_norm, self.y_norm, self.w_norm, self.h_norm]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "bbox_canvas": self.bbox_canvas,
            "visible": self.visible,
            "opacity": self.opacity,
            "text": self.text,
            "template_id": self.template_id,
            "child_ids": self.child_ids,
            "layout_mode": self.layout_mode,
            "item_spacing": self.item_spacing,
            "padding": self.padding,
            "extra_data": self.extra_data,
            "source_figma_ids": self.source_figma_ids,
            "collapsed_from_ids": self.collapsed_from_ids,
        }


def _is_group_like(node: NormalizedNode) -> bool:
    return node.type.lower() in {"group", "frame", "component", "instance"}


def _bbox_almost_equal(a: NormalizedNode, b: NormalizedNode, tol: float = 0.01) -> bool:
    return (
        abs(a.x_norm - b.x_norm) <= tol
        and abs(a.y_norm - b.y_norm) <= tol
        and abs(a.w_norm - b.w_norm) <= tol
        and abs(a.h_norm - b.h_norm) <= tol
    )


def _build_node_map(nodes: list[NormalizedNode]) -> dict[str, NormalizedNode]:
    return {node.id: node for node in nodes}


def _build_children_map(nodes: list[NormalizedNode]) -> dict[str | None, list[str]]:
    children_map: dict[str | None, list[str]] = {}
    for node in nodes:
        children_map.setdefault(node.parent_id, []).append(node.id)
    return children_map


def _should_collapse_wrapper(
    node: NormalizedNode,
    only_child: NormalizedNode,
) -> bool:
    if not _is_group_like(node):
        return False
    if node.text is not None:
        return False
    if not _bbox_almost_equal(node, only_child):
        return False
    if node.layout_mode not in (None, "", "NONE"):
        return False
    if node.item_spacing not in (None, 0, 0.0):
        return False
    if node.padding is not None:
        if any(abs(v) > 1e-9 for v in node.padding.values()):
            return False
    return True


def collapse_wrapper_groups(nodes: list[NormalizedNode]) -> list[CollapsedNode]:
    node_map = _build_node_map(nodes)
    children_map = _build_children_map(nodes)

    collapsible_ids: set[str] = set()

    for node in nodes:
        child_ids = children_map.get(node.id, [])
        if len(child_ids) != 1:
            continue

        only_child = node_map[child_ids[0]]
        if _should_collapse_wrapper(node, only_child):
            collapsible_ids.add(node.id)

    def resolve_surviving_parent(parent_id: str | None) -> str | None:
        current = parent_id
        while current is not None and current in collapsible_ids:
            current = node_map[current].parent_id
        return current

    surviving_nodes: list[CollapsedNode] = []

    for node in nodes:
        if node.id in collapsible_ids:
            continue

        collapsed_from_ids = [node.id]
        current_parent = node.parent_id
        while current_parent is not None and current_parent in collapsible_ids:
            collapsed_from_ids.append(current_parent)
            current_parent = node_map[current_parent].parent_id

        surviving_parent_id = resolve_surviving_parent(node.parent_id)

        collapsed_node = CollapsedNode(
            id=node.id,
            name=node.name,
            type=node.type,
            parent_id=surviving_parent_id,
            depth=node.depth,
            x=node.x,
            y=node.y,
            width=node.width,
            height=node.height,
            x_norm=node.x_norm,
            y_norm=node.y_norm,
            w_norm=node.w_norm,
            h_norm=node.h_norm,
            center_x_norm=node.center_x_norm,
            center_y_norm=node.center_y_norm,
            area_px=node.area_px,
            area_ratio=node.area_ratio,
            aspect_ratio=node.aspect_ratio,
            left_norm=node.left_norm,
            top_norm=node.top_norm,
            right_norm=node.right_norm,
            bottom_norm=node.bottom_norm,
            dist_left=node.dist_left,
            dist_top=node.dist_top,
            dist_right=node.dist_right,
            dist_bottom=node.dist_bottom,
            visible=node.visible,
            opacity=node.opacity,
            text=node.text,
            template_id=node.template_id,
            child_ids=[],
            layout_mode=node.layout_mode,
            item_spacing=node.item_spacing,
            padding=dict(node.padding) if node.padding is not None else None,
            extra_data=dict(node.extra_data),
            source_figma_ids=[node.id],
            collapsed_from_ids=collapsed_from_ids,
        )
        surviving_nodes.append(collapsed_node)

    surviving_map = {node.id: node for node in surviving_nodes}

    for node in surviving_nodes:
        node.child_ids = []

    for node in surviving_nodes:
        if node.parent_id is not None and node.parent_id in surviving_map:
            surviving_map[node.parent_id].child_ids.append(node.id)

    def compute_depth(node_id: str) -> int:
        depth = 0
        current_parent = surviving_map[node_id].parent_id
        while current_parent is not None:
            depth += 1
            current_parent = surviving_map[current_parent].parent_id
        return depth

    for node in surviving_nodes:
        node.depth = compute_depth(node.id)

    return surviving_nodes
