from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from schemas.raw_figma import RawFigmaDocument, RawFigmaNode


@dataclass
class ParsedNode:
    """
    Flat, traversal-friendly representation of one raw Figma node.
    This is still raw/editor-oriented, not semantic yet.
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

    visible: bool
    opacity: float

    text: str | None = None
    template_id: str | None = None

    child_ids: list[str] = field(default_factory=list)
    layout_mode: str | None = None
    item_spacing: float | None = None
    padding: dict[str, float] | None = None

    extra_data: dict[str, Any] = field(default_factory=dict)

    @property
    def bbox_px(self) -> list[float]:
        return [self.x, self.y, self.width, self.height]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_text(self) -> bool:
        return self.type.lower() == "text"

    @property
    def is_group_like(self) -> bool:
        return self.type.lower() in {"group", "frame", "component", "instance"}

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


def load_raw_figma_json(json_path: str | Path) -> RawFigmaDocument:
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return RawFigmaDocument.from_json_data(data)


def _padding_to_dict(node: RawFigmaNode) -> dict[str, float] | None:
    if node.padding is None:
        return None
    return {
        "top": float(node.padding.top),
        "right": float(node.padding.right),
        "bottom": float(node.padding.bottom),
        "left": float(node.padding.left),
    }


def flatten_figma_tree(root: RawFigmaNode) -> list[ParsedNode]:
    flat_nodes: list[ParsedNode] = []

    def _walk(node: RawFigmaNode, parent_id: str | None, depth: int) -> None:
        parsed = ParsedNode(
            id=node.id,
            name=node.name,
            type=node.type,
            parent_id=parent_id,
            depth=depth,
            x=float(node.bounds.x),
            y=float(node.bounds.y),
            width=float(node.bounds.width),
            height=float(node.bounds.height),
            visible=bool(node.visible),
            opacity=float(node.opacity),
            text=node.text_content,
            template_id=node.templateId,
            child_ids=[child.id for child in node.children],
            layout_mode=node.layoutMode,
            item_spacing=float(node.itemSpacing) if node.itemSpacing is not None else None,
            padding=_padding_to_dict(node),
            extra_data=dict(node.extra_data or {}),
        )
        flat_nodes.append(parsed)

        for child in node.children:
            _walk(child, parent_id=node.id, depth=depth + 1)

    _walk(root, parent_id=None, depth=0)
    return flat_nodes


def parse_figma_file(json_path: str | Path) -> tuple[RawFigmaDocument, RawFigmaNode, list[ParsedNode]]:
    doc = load_raw_figma_json(json_path)
    root = doc.first_root
    flat_nodes = flatten_figma_tree(root)
    return doc, root, flat_nodes


def get_node_map(nodes: Iterable[ParsedNode]) -> dict[str, ParsedNode]:
    return {node.id: node for node in nodes}


def get_root_canvas_size(root: RawFigmaNode) -> tuple[int, int]:
    return int(root.bounds.width), int(root.bounds.height)


def save_flat_nodes(nodes: Iterable[ParsedNode], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialized = [node.to_dict() for node in nodes]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)
