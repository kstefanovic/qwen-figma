from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from schemas.enums import (
    AnchorType,
    ConstraintType,
    ElementRole,
    FunctionalType,
    GroupRole,
    ImportanceLevel,
    InternalLayout,
    LayoutPattern,
    RelationType,
    ZoneRole,
)


class BBox(BaseModel):
    """
    Normalized bounding box relative to canvas or parent/group.
    All values are expected to be in [0, 1].
    """

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)

    @field_validator("w", "h")
    @classmethod
    def positive_size(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Width and height must be positive.")
        return value

    @model_validator(mode="after")
    def validate_box_inside_unit_space(self) -> "BBox":
        if self.x + self.w > 1.000001:
            raise ValueError("BBox x + w must be <= 1.")
        if self.y + self.h > 1.000001:
            raise ValueError("BBox y + h must be <= 1.")
        return self

    def as_list(self) -> List[float]:
        return [self.x, self.y, self.w, self.h]


class SourceInfo(BaseModel):
    canvas_width: int = Field(gt=0)
    canvas_height: int = Field(gt=0)
    aspect_ratio: float = Field(gt=0)
    source_type: str
    raw_figma_frame_id: Optional[str] = None


class Metadata(BaseModel):
    brand_family: Optional[str] = None
    language: Optional[str] = None
    category: Optional[str] = None
    layout_pattern: LayoutPattern = LayoutPattern.UNKNOWN
    pattern_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AdaptationPolicy(BaseModel):
    preserve_as_unit: bool = True
    allow_reflow: bool = False
    allow_scale: bool = True
    allow_crop: bool = False
    allow_shift: bool = True
    drop_priority: int = Field(default=0, ge=0, le=3)
    anchor_type: AnchorType = AnchorType.FREE


class TextFeatures(BaseModel):
    char_count: Optional[int] = Field(default=None, ge=0)
    word_count: Optional[int] = Field(default=None, ge=0)
    line_count: Optional[int] = Field(default=None, ge=0)
    language: Optional[str] = None
    uppercase_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    digit_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class StyleFeatures(BaseModel):
    font_family: Optional[str] = None
    font_weight: Optional[int] = None
    font_size: Optional[float] = None
    line_height: Optional[float] = None
    text_align: Optional[str] = None
    text_color: Optional[List[int]] = None

    @field_validator("text_color")
    @classmethod
    def validate_text_color(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is None:
            return value
        if len(value) != 3:
            raise ValueError("text_color must be RGB with exactly 3 integers.")
        if not all(isinstance(c, int) and 0 <= c <= 255 for c in value):
            raise ValueError("text_color must contain integers in [0, 255].")
        return value


class VisualFeatures(BaseModel):
    dominant_color: Optional[List[int]] = None
    contrast_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("dominant_color")
    @classmethod
    def validate_dominant_color(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is None:
            return value
        if len(value) != 3:
            raise ValueError("dominant_color must be RGB with exactly 3 integers.")
        if not all(isinstance(c, int) and 0 <= c <= 255 for c in value):
            raise ValueError("dominant_color must contain integers in [0, 255].")
        return value


class Zone(BaseModel):
    id: str
    role: ZoneRole
    bbox_canvas: BBox
    children_groups: List[str] = Field(default_factory=list)
    layout_flow: Optional[Literal["vertical_stack", "horizontal_flow", "freeform", "overlay"]] = None
    background_type: Optional[Literal["solid_color", "photo", "gradient", "transparent"]] = None
    background_color: Optional[List[int]] = None
    importance: Optional[ImportanceLevel] = None

    @field_validator("background_color")
    @classmethod
    def validate_background_color(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is None:
            return value
        if len(value) != 3:
            raise ValueError("background_color must be RGB with exactly 3 integers.")
        if not all(isinstance(c, int) and 0 <= c <= 255 for c in value):
            raise ValueError("background_color must contain integers in [0, 255].")
        return value


class Group(BaseModel):
    id: str
    role: GroupRole
    zone_id: str
    parent_group_id: Optional[str] = None
    bbox_canvas: Optional[BBox] = None
    bbox_zone: Optional[BBox] = None
    children_elements: List[str] = Field(default_factory=list)
    children_groups: List[str] = Field(default_factory=list)
    source_figma_ids: List[str] = Field(default_factory=list)
    internal_layout: InternalLayout = InternalLayout.FREEFORM
    anchor_type: AnchorType = AnchorType.FREE
    importance_level: ImportanceLevel = ImportanceLevel.MEDIUM
    adaptation_policy: AdaptationPolicy = Field(default_factory=AdaptationPolicy)


class Element(BaseModel):
    id: str
    source_figma_id: str
    type: str
    role: ElementRole
    group_id: str
    zone_id: str
    semantic_name: Optional[str] = None
    bbox_canvas: BBox
    bbox_group: Optional[BBox] = None
    center_canvas: Optional[List[float]] = None
    z_index: Optional[int] = None
    visible: bool = True

    functional_type: FunctionalType = FunctionalType.FUNCTIONAL
    importance_level: ImportanceLevel = ImportanceLevel.MEDIUM

    is_text: Optional[bool] = None
    is_brand_related: bool = False
    is_required_for_compliance: bool = False

    text_content: Optional[str] = None
    text_features: Optional[TextFeatures] = None
    style_features: Optional[StyleFeatures] = None
    visual_features: Optional[VisualFeatures] = None

    adaptation_policy: AdaptationPolicy = Field(default_factory=AdaptationPolicy)

    @field_validator("center_canvas")
    @classmethod
    def validate_center_canvas(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is None:
            return value
        if len(value) != 2:
            raise ValueError("center_canvas must have exactly 2 floats.")
        if not all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0 for v in value):
            raise ValueError("center_canvas values must be in [0, 1].")
        return [float(v) for v in value]


class Relation(BaseModel):
    src: str
    dst: str
    type: RelationType
    strength: float = Field(default=1.0, ge=0.0, le=1.0)


class Constraint(BaseModel):
    target: str
    type: ConstraintType
    value: Any


class SemanticGraph(BaseModel):
    banner_id: str
    template_id: Optional[str] = None
    source: SourceInfo
    metadata: Metadata
    zones: List[Zone] = Field(default_factory=list)
    groups: List[Group] = Field(default_factory=list)
    elements: List[Element] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    constraints: List[Constraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "SemanticGraph":
        zone_ids = {z.id for z in self.zones}
        group_ids = {g.id for g in self.groups}
        element_ids = {e.id for e in self.elements}
        valid_targets = zone_ids | group_ids | element_ids

        for group in self.groups:
            if group.zone_id not in zone_ids:
                raise ValueError(f"Group '{group.id}' references unknown zone_id '{group.zone_id}'.")
            if group.parent_group_id is not None and group.parent_group_id not in group_ids:
                raise ValueError(
                    f"Group '{group.id}' references unknown parent_group_id '{group.parent_group_id}'."
                )
            for child_group_id in group.children_groups:
                if child_group_id not in group_ids:
                    raise ValueError(
                        f"Group '{group.id}' references unknown child group '{child_group_id}'."
                    )
            for child_element_id in group.children_elements:
                if child_element_id not in element_ids:
                    raise ValueError(
                        f"Group '{group.id}' references unknown child element '{child_element_id}'."
                    )

        for element in self.elements:
            if element.group_id not in group_ids:
                raise ValueError(
                    f"Element '{element.id}' references unknown group_id '{element.group_id}'."
                )
            if element.zone_id not in zone_ids:
                raise ValueError(
                    f"Element '{element.id}' references unknown zone_id '{element.zone_id}'."
                )

        for zone in self.zones:
            for child_group_id in zone.children_groups:
                if child_group_id not in group_ids:
                    raise ValueError(
                        f"Zone '{zone.id}' references unknown child group '{child_group_id}'."
                    )

        for relation in self.relations:
            if relation.src not in valid_targets:
                raise ValueError(f"Relation src '{relation.src}' not found in graph.")
            if relation.dst not in valid_targets:
                raise ValueError(f"Relation dst '{relation.dst}' not found in graph.")

        for constraint in self.constraints:
            if constraint.target not in valid_targets:
                raise ValueError(f"Constraint target '{constraint.target}' not found in graph.")

        return self
