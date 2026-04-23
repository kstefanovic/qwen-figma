from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class RawBounds(BaseModel):
    x: float
    y: float
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class RawPadding(BaseModel):
    top: float = 0
    right: float = 0
    bottom: float = 0
    left: float = 0


class RawFigmaNode(BaseModel):
    """
    Generic recursive node for raw Figma export.
    Works for frame, group, text, vector, rectangle, ellipse, star, image-like nodes.
    """

    id: str
    name: Optional[str] = None
    type: str

    bounds: RawBounds

    visible: bool = True
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)

    layoutMode: Optional[str] = None
    itemSpacing: Optional[float] = None
    padding: Optional[RawPadding] = None

    characters: Optional[str] = None
    templateId: Optional[str] = None

    children: List["RawFigmaNode"] = Field(default_factory=list)

    extra_data: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @field_validator("type")
    @classmethod
    def normalize_type(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="before")
    @classmethod
    def capture_unknown_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        known = {
            "id",
            "name",
            "type",
            "bounds",
            "visible",
            "opacity",
            "layoutMode",
            "itemSpacing",
            "padding",
            "characters",
            "templateId",
            "children",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        if "extra_data" not in data:
            data["extra_data"] = extra
        return data

    @property
    def is_text(self) -> bool:
        return self.type.lower() == "text"

    @property
    def is_group_like(self) -> bool:
        return self.type.lower() in {"group", "frame", "component", "instance"}

    @property
    def has_children(self) -> bool:
        return len(self.children) > 0

    @property
    def text_content(self) -> Optional[str]:
        return self.characters


RawFigmaNode.model_rebuild()


class RawFigmaDocument(BaseModel):
    """
    Top-level wrapper for uploaded JSON files.
    Supports either a single root object or a list of root objects.
    """

    roots: List[RawFigmaNode]

    model_config = {"extra": "allow"}

    @classmethod
    def from_json_data(cls, data: Any) -> "RawFigmaDocument":
        if isinstance(data, list):
            return cls(roots=[RawFigmaNode.model_validate(item) for item in data])
        if isinstance(data, dict):
            return cls(roots=[RawFigmaNode.model_validate(data)])
        raise TypeError("Raw Figma JSON must be a dict or a list of dicts.")

    @property
    def first_root(self) -> RawFigmaNode:
        if not self.roots:
            raise ValueError("RawFigmaDocument contains no roots.")
        return self.roots[0]
