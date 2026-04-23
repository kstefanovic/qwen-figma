from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from banner_pipeline.parse_figma import ParsedNode


@dataclass
class NormalizedNode:
    """
    ParsedNode + normalized geometry relative to the root canvas.
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

    @property
    def bbox_px(self) -> list[float]:
        return [self.x, self.y, self.width, self.height]

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
            "bbox_px": self.bbox_px,
            "x_norm": self.x_norm,
            "y_norm": self.y_norm,
            "w_norm": self.w_norm,
            "h_norm": self.h_norm,
            "bbox_canvas": self.bbox_canvas,
            "center_x_norm": self.center_x_norm,
            "center_y_norm": self.center_y_norm,
            "area_px": self.area_px,
            "area_ratio": self.area_ratio,
            "aspect_ratio": self.aspect_ratio,
            "left_norm": self.left_norm,
            "top_norm": self.top_norm,
            "right_norm": self.right_norm,
            "bottom_norm": self.bottom_norm,
            "dist_left": self.dist_left,
            "dist_top": self.dist_top,
            "dist_right": self.dist_right,
            "dist_bottom": self.dist_bottom,
            "visible": self.visible,
            "opacity": self.opacity,
            "text": self.text,
            "template_id": self.template_id,
            "child_ids": self.child_ids,
            "layout_mode": self.layout_mode,
            "item_spacing": self.item_spacing,
            "padding": self.padding,
            "extra_data": self.extra_data,
        }


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero while normalizing geometry.")
    return a / b


def normalize_node(node: ParsedNode, canvas_width: int, canvas_height: int) -> NormalizedNode:
    """
    Normalize one ParsedNode against the root canvas.

    Important:
    - raw x/y/width/height are preserved
    - normalized bbox is CLIPPED to the canvas [0, 1]
    """
    raw_left = _safe_div(node.x, canvas_width)
    raw_top = _safe_div(node.y, canvas_height)
    raw_right = _safe_div(node.x + node.width, canvas_width)
    raw_bottom = _safe_div(node.y + node.height, canvas_height)

    left_norm = max(0.0, min(1.0, raw_left))
    top_norm = max(0.0, min(1.0, raw_top))
    right_norm = max(0.0, min(1.0, raw_right))
    bottom_norm = max(0.0, min(1.0, raw_bottom))

    w_norm = max(0.0, right_norm - left_norm)
    h_norm = max(0.0, bottom_norm - top_norm)

    center_x_norm = left_norm + (w_norm / 2.0)
    center_y_norm = top_norm + (h_norm / 2.0)

    area_px = node.width * node.height
    canvas_area = canvas_width * canvas_height
    area_ratio = _safe_div(area_px, canvas_area)

    aspect_ratio = node.width / node.height if node.height > 0 else 0.0

    dist_left = left_norm
    dist_top = top_norm
    dist_right = 1.0 - right_norm
    dist_bottom = 1.0 - bottom_norm

    return NormalizedNode(
        id=node.id,
        name=node.name,
        type=node.type,
        parent_id=node.parent_id,
        depth=node.depth,
        x=node.x,
        y=node.y,
        width=node.width,
        height=node.height,
        x_norm=left_norm,
        y_norm=top_norm,
        w_norm=w_norm,
        h_norm=h_norm,
        center_x_norm=center_x_norm,
        center_y_norm=center_y_norm,
        area_px=area_px,
        area_ratio=area_ratio,
        aspect_ratio=aspect_ratio,
        left_norm=left_norm,
        top_norm=top_norm,
        right_norm=right_norm,
        bottom_norm=bottom_norm,
        dist_left=dist_left,
        dist_top=dist_top,
        dist_right=dist_right,
        dist_bottom=dist_bottom,
        visible=node.visible,
        opacity=node.opacity,
        text=node.text,
        template_id=node.template_id,
        child_ids=list(node.child_ids),
        layout_mode=node.layout_mode,
        item_spacing=node.item_spacing,
        padding=dict(node.padding) if node.padding is not None else None,
        extra_data=dict(node.extra_data),
    )


def normalize_nodes(nodes: list[ParsedNode], canvas_width: int, canvas_height: int) -> list[NormalizedNode]:
    return [normalize_node(node, canvas_width, canvas_height) for node in nodes]


def get_normalized_node_map(nodes: list[NormalizedNode]) -> dict[str, NormalizedNode]:
    return {node.id: node for node in nodes}
