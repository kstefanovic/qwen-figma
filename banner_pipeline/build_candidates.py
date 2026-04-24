from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from banner_pipeline.collapse_groups import CollapsedNode


@dataclass
class CandidateNodeRef:
    node_id: str
    role_hint: str | None = None


@dataclass
class SemanticCandidate:
    candidate_id: str
    candidate_type: str
    source_node_ids: list[str]

    x_norm: float
    y_norm: float
    w_norm: float
    h_norm: float

    center_x_norm: float
    center_y_norm: float
    area_ratio: float
    aspect_ratio: float

    member_refs: list[CandidateNodeRef] = field(default_factory=list)

    role_hint: str | None = None
    group_hint: str | None = None
    importance_hint: str | None = None

    text_content: str | None = None
    text_count: int = 0
    non_text_count: int = 0

    extra_data: dict[str, Any] = field(default_factory=dict)

    @property
    def bbox_canvas(self) -> list[float]:
        return [self.x_norm, self.y_norm, self.w_norm, self.h_norm]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type,
            "source_node_ids": self.source_node_ids,
            "bbox_canvas": self.bbox_canvas,
            "center_x_norm": self.center_x_norm,
            "center_y_norm": self.center_y_norm,
            "area_ratio": self.area_ratio,
            "aspect_ratio": self.aspect_ratio,
            "member_refs": [
                {"node_id": ref.node_id, "role_hint": ref.role_hint}
                for ref in self.member_refs
            ],
            "role_hint": self.role_hint,
            "group_hint": self.group_hint,
            "importance_hint": self.importance_hint,
            "text_content": self.text_content,
            "text_count": self.text_count,
            "non_text_count": self.non_text_count,
            "extra_data": self.extra_data,
        }


@dataclass
class CandidateBundle:
    all_candidates: list[SemanticCandidate] = field(default_factory=list)

    text_candidates: list[SemanticCandidate] = field(default_factory=list)
    text_group_candidates: list[SemanticCandidate] = field(default_factory=list)
    brand_candidates: list[SemanticCandidate] = field(default_factory=list)
    badge_candidates: list[SemanticCandidate] = field(default_factory=list)
    decoration_candidates: list[SemanticCandidate] = field(default_factory=list)
    background_candidates: list[SemanticCandidate] = field(default_factory=list)
    image_like_candidates: list[SemanticCandidate] = field(default_factory=list)

    node_to_candidate_ids: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_candidates": [c.to_dict() for c in self.all_candidates],
            "text_candidates": [c.to_dict() for c in self.text_candidates],
            "text_group_candidates": [c.to_dict() for c in self.text_group_candidates],
            "brand_candidates": [c.to_dict() for c in self.brand_candidates],
            "badge_candidates": [c.to_dict() for c in self.badge_candidates],
            "decoration_candidates": [c.to_dict() for c in self.decoration_candidates],
            "background_candidates": [c.to_dict() for c in self.background_candidates],
            "image_like_candidates": [c.to_dict() for c in self.image_like_candidates],
            "node_to_candidate_ids": self.node_to_candidate_ids,
        }


def _node_map(nodes: Iterable[CollapsedNode]) -> dict[str, CollapsedNode]:
    return {node.id: node for node in nodes}


def _children_map(nodes: Iterable[CollapsedNode]) -> dict[str | None, list[str]]:
    out: dict[str | None, list[str]] = {}
    for node in nodes:
        out.setdefault(node.parent_id, []).append(node.id)
    return out


def _is_text(node: CollapsedNode) -> bool:
    return node.type.lower() == "text"


def _is_group_like(node: CollapsedNode) -> bool:
    return node.type.lower() in {"group", "frame", "component", "instance"}


def _is_shape_like(node: CollapsedNode) -> bool:
    return node.type.lower() in {"vector", "rectangle", "ellipse", "star", "polygon", "line", "shape"}


def _is_image_like(node: CollapsedNode) -> bool:
    t = node.type.lower()
    return t in {"image", "bitmap", "rectangle", "vector"}


def _merge_bboxes(nodes: list[CollapsedNode]) -> tuple[float, float, float, float]:
    min_x = min(n.x_norm for n in nodes)
    min_y = min(n.y_norm for n in nodes)
    max_right = max(n.x_norm + n.w_norm for n in nodes)
    max_bottom = max(n.y_norm + n.h_norm for n in nodes)
    return min_x, min_y, max_right - min_x, max_bottom - min_y


def _bbox_center(x: float, y: float, w: float, h: float) -> tuple[float, float]:
    return x + w / 2.0, y + h / 2.0


def _aspect_ratio(w: float, h: float) -> float:
    return w / h if h > 0 else 0.0


def _text_summary(nodes: list[CollapsedNode]) -> str | None:
    texts = [n.text.strip() for n in nodes if n.text and n.text.strip()]
    if not texts:
        return None
    return "\n".join(texts)


def _candidate_from_nodes(
    candidate_id: str,
    candidate_type: str,
    nodes: list[CollapsedNode],
    role_hint: str | None = None,
    group_hint: str | None = None,
    importance_hint: str | None = None,
    extra_data: dict[str, Any] | None = None,
) -> SemanticCandidate:
    x, y, w, h = _merge_bboxes(nodes)
    cx, cy = _bbox_center(x, y, w, h)

    text_nodes = [n for n in nodes if _is_text(n)]
    non_text_nodes = [n for n in nodes if not _is_text(n)]

    return SemanticCandidate(
        candidate_id=candidate_id,
        candidate_type=candidate_type,
        source_node_ids=[n.id for n in nodes],
        x_norm=x,
        y_norm=y,
        w_norm=w,
        h_norm=h,
        center_x_norm=cx,
        center_y_norm=cy,
        area_ratio=w * h,
        aspect_ratio=_aspect_ratio(w, h),
        member_refs=[CandidateNodeRef(node_id=n.id) for n in nodes],
        role_hint=role_hint,
        group_hint=group_hint,
        importance_hint=importance_hint,
        text_content=_text_summary(nodes),
        text_count=len(text_nodes),
        non_text_count=len(non_text_nodes),
        extra_data=extra_data or {},
    )


def _add_candidate(bundle: CandidateBundle, candidate: SemanticCandidate) -> None:
    bundle.all_candidates.append(candidate)
    for node_id in candidate.source_node_ids:
        bundle.node_to_candidate_ids.setdefault(node_id, []).append(candidate.candidate_id)

    if candidate.candidate_type == "text":
        bundle.text_candidates.append(candidate)
    elif candidate.candidate_type == "text_group":
        bundle.text_group_candidates.append(candidate)
    elif candidate.candidate_type == "brand":
        bundle.brand_candidates.append(candidate)
    elif candidate.candidate_type == "badge":
        bundle.badge_candidates.append(candidate)
    elif candidate.candidate_type == "decoration":
        bundle.decoration_candidates.append(candidate)
    elif candidate.candidate_type == "background":
        bundle.background_candidates.append(candidate)
    elif candidate.candidate_type == "image_like":
        bundle.image_like_candidates.append(candidate)


def _vertical_overlap_ratio(a: CollapsedNode, b: CollapsedNode) -> float:
    top = max(a.y_norm, b.y_norm)
    bottom = min(a.y_norm + a.h_norm, b.y_norm + b.h_norm)
    overlap = max(0.0, bottom - top)
    denom = min(a.h_norm, b.h_norm)
    if denom <= 0:
        return 0.0
    return overlap / denom


def _horizontal_overlap_ratio(a: CollapsedNode, b: CollapsedNode) -> float:
    left = max(a.x_norm, b.x_norm)
    right = min(a.x_norm + a.w_norm, b.x_norm + b.w_norm)
    overlap = max(0.0, right - left)
    denom = min(a.w_norm, b.w_norm)
    if denom <= 0:
        return 0.0
    return overlap / denom


def _vertical_gap(a: CollapsedNode, b: CollapsedNode) -> float:
    return b.y_norm - (a.y_norm + a.h_norm)


def _sort_by_reading_order(nodes: list[CollapsedNode]) -> list[CollapsedNode]:
    return sorted(nodes, key=lambda n: (round(n.y_norm, 4), round(n.x_norm, 4)))


def _node_text(node: CollapsedNode) -> str:
    return (node.text or "").strip()


def _looks_like_legal_text_node(node: CollapsedNode) -> bool:
    if not _is_text(node):
        return False

    text = _node_text(node).lower()
    if not text:
        return False

    legal_markers = [
        "ооо",
        "огрн",
        "продавец",
        "доставку осуществляют",
        "партнёры сервиса",
        "адрес",
        "юрид",
        "услов",
        "disclaimer",
        "terms",
        "conditions",
    ]

    if any(marker in text for marker in legal_markers):
        return True

    if len(text) >= 60 and node.y_norm >= 0.72:
        return True

    return False


def _looks_like_badge_text_node(node: CollapsedNode) -> bool:
    if not _is_text(node):
        return False
    text = _node_text(node).replace(" ", "")
    return text in {"0+", "6+", "12+", "16+", "18+"}


def _likely_text_group(a: CollapsedNode, b: CollapsedNode) -> bool:
    if not (_is_text(a) and _is_text(b)):
        return False

    if _looks_like_legal_text_node(a) or _looks_like_legal_text_node(b):
        return False

    if _looks_like_badge_text_node(a) or _looks_like_badge_text_node(b):
        return False

    left_aligned = abs(a.x_norm - b.x_norm) <= 0.03
    horiz_overlap = _horizontal_overlap_ratio(a, b) >= 0.5

    gap = _vertical_gap(a, b)
    close_vertically = -0.02 <= gap <= 0.10

    width_ratio = min(a.w_norm, b.w_norm) / max(a.w_norm, b.w_norm) if max(a.w_norm, b.w_norm) > 0 else 0
    similar_region = width_ratio >= 0.25

    if b.y_norm >= 0.72:
        return False

    return (left_aligned or horiz_overlap) and close_vertically and similar_region


def _is_top_left_region(node: CollapsedNode) -> bool:
    return node.x_norm <= 0.4 and node.y_norm <= 0.25


def _is_top_right_region(node: CollapsedNode) -> bool:
    return node.x_norm >= 0.7 and node.y_norm <= 0.25


def _is_bottom_region(node: CollapsedNode) -> bool:
    return node.y_norm >= 0.72


def _is_small_short_text(node: CollapsedNode) -> bool:
    if not _is_text(node):
        return False
    text = (node.text or "").strip()
    if not text:
        return False
    return len(text) <= 6 and node.area_ratio <= 0.03


def _is_long_text(node: CollapsedNode) -> bool:
    if not _is_text(node):
        return False
    text = (node.text or "").strip()
    return len(text) >= 40


def _is_star_like(node: CollapsedNode) -> bool:
    return node.type.lower() == "star"


def _is_large_background_like(node: CollapsedNode) -> bool:
    t = node.type.lower()
    if t not in {"rectangle", "vector", "ellipse"}:
        return False
    return node.area_ratio >= 0.18 or node.h_norm >= 0.75 or node.w_norm >= 0.65


def _build_text_candidates(nodes: list[CollapsedNode], bundle: CandidateBundle) -> None:
    text_nodes = [n for n in nodes if _is_text(n)]
    text_idx = 1

    for node in _sort_by_reading_order(text_nodes):
        if _looks_like_badge_text_node(node) and _is_top_right_region(node):
            continue

        candidate = _candidate_from_nodes(
            candidate_id=f"text_{text_idx}",
            candidate_type="text",
            nodes=[node],
            role_hint="text",
        )
        _add_candidate(bundle, candidate)
        text_idx += 1


def _build_text_group_candidates(nodes: list[CollapsedNode], bundle: CandidateBundle) -> None:
    text_nodes = _sort_by_reading_order([n for n in nodes if _is_text(n)])
    used: set[str] = set()
    group_idx = 1

    for i, node in enumerate(text_nodes):
        if node.id in used:
            continue

        current_group = [node]
        used.add(node.id)

        for j in range(i + 1, len(text_nodes)):
            other = text_nodes[j]
            if other.id in used:
                continue

            last = current_group[-1]
            if _likely_text_group(last, other):
                current_group.append(other)
                used.add(other.id)
            else:
                break

        if len(current_group) >= 2:
            candidate = _candidate_from_nodes(
                candidate_id=f"text_group_{group_idx}",
                candidate_type="text_group",
                nodes=current_group,
                role_hint="text_group",
                group_hint="stacked_text",
                extra_data={
                    "grouping_reason": "vertical_text_proximity",
                    "member_count": len(current_group),
                },
            )
            _add_candidate(bundle, candidate)
            group_idx += 1


def _build_brand_candidates(nodes: list[CollapsedNode], bundle: CandidateBundle) -> None:
    brand_like_nodes: list[CollapsedNode] = []

    for node in nodes:
        if not _is_top_left_region(node):
            continue
        if _is_text(node):
            continue

        if _is_group_like(node) or _is_shape_like(node):
            if node.area_ratio <= 0.20:
                brand_like_nodes.append(node)

    if not brand_like_nodes:
        return

    brand_like_nodes = sorted(brand_like_nodes, key=lambda n: (n.y_norm, n.x_norm))

    seed = brand_like_nodes[0]
    cluster = [seed]
    for node in brand_like_nodes[1:]:
        close_x = abs(node.center_x_norm - seed.center_x_norm) <= 0.25
        close_y = abs(node.center_y_norm - seed.center_y_norm) <= 0.18
        if close_x and close_y:
            cluster.append(node)

    candidate = _candidate_from_nodes(
        candidate_id="brand_1",
        candidate_type="brand",
        nodes=cluster,
        role_hint="brand",
        group_hint="brand_group",
        importance_hint="critical",
        extra_data={"grouping_reason": "top_left_brand_cluster"},
    )
    _add_candidate(bundle, candidate)


def _build_badge_candidates(nodes: list[CollapsedNode], bundle: CandidateBundle) -> None:
    badge_idx = 1
    for node in nodes:
        if _is_top_right_region(node) and _is_small_short_text(node):
            candidate = _candidate_from_nodes(
                candidate_id=f"badge_{badge_idx}",
                candidate_type="badge",
                nodes=[node],
                role_hint="badge",
                group_hint="badge_group",
                importance_hint="high",
                extra_data={"grouping_reason": "top_right_small_short_text"},
            )
            _add_candidate(bundle, candidate)
            badge_idx += 1


def _bundle_source_node_ids(candidates: list[SemanticCandidate]) -> set[str]:
    out: set[str] = set()
    for candidate in candidates:
        out.update(candidate.source_node_ids)
    return out


def _build_decoration_candidates(nodes: list[CollapsedNode], bundle: CandidateBundle) -> None:
    brand_node_ids = _bundle_source_node_ids(bundle.brand_candidates)
    decoration_nodes: list[CollapsedNode] = []
    for node in nodes:
        if node.id in brand_node_ids:
            continue
        if _is_star_like(node):
            decoration_nodes.append(node)
            continue
        if _is_shape_like(node) and not _is_large_background_like(node) and node.area_ratio <= 0.06:
            decoration_nodes.append(node)

    if not decoration_nodes:
        return

    clusters: list[list[CollapsedNode]] = []
    for node in sorted(decoration_nodes, key=lambda n: (n.y_norm, n.x_norm)):
        matched_cluster: list[CollapsedNode] | None = None
        node_is_star = _is_star_like(node)
        for cluster in clusters:
            seed = cluster[0]
            same_family = _is_star_like(seed) == node_is_star
            close_x = abs(node.center_x_norm - seed.center_x_norm) <= 0.18
            close_y = abs(node.center_y_norm - seed.center_y_norm) <= 0.18
            similar_area = 0.35 <= (min(node.area_ratio, seed.area_ratio) / max(node.area_ratio, seed.area_ratio, 1e-6)) <= 1.0
            if same_family and close_x and close_y and similar_area:
                matched_cluster = cluster
                break
        if matched_cluster is None:
            clusters.append([node])
        else:
            matched_cluster.append(node)

    for decoration_idx, cluster in enumerate(clusters, start=1):
        grouping_reason = "star_family_cluster" if any(_is_star_like(n) for n in cluster) else "small_shape_cluster"
        candidate = _candidate_from_nodes(
            candidate_id=f"decoration_{decoration_idx}",
            candidate_type="decoration",
            nodes=cluster,
            role_hint="decoration",
            group_hint="decoration_group",
            importance_hint="low",
            extra_data={
                "grouping_reason": grouping_reason,
                "member_count": len(cluster),
            },
        )
        _add_candidate(bundle, candidate)


def _build_background_candidates(nodes: list[CollapsedNode], bundle: CandidateBundle) -> None:
    bg_idx = 1
    for node in nodes:
        if _is_large_background_like(node):
            candidate = _candidate_from_nodes(
                candidate_id=f"background_{bg_idx}",
                candidate_type="background",
                nodes=[node],
                role_hint="background",
                group_hint="background_group",
                importance_hint="medium",
                extra_data={"grouping_reason": "large_background_like_shape"},
            )
            _add_candidate(bundle, candidate)
            bg_idx += 1


def _build_image_like_candidates(nodes: list[CollapsedNode], bundle: CandidateBundle) -> None:
    img_idx = 1
    for node in nodes:
        if _is_image_like(node) and node.area_ratio >= 0.08:
            candidate = _candidate_from_nodes(
                candidate_id=f"image_like_{img_idx}",
                candidate_type="image_like",
                nodes=[node],
                role_hint="image_like",
                importance_hint="medium",
                extra_data={"grouping_reason": "large_image_like_node"},
            )
            _add_candidate(bundle, candidate)
            img_idx += 1


def build_candidates(nodes: list[CollapsedNode]) -> CandidateBundle:
    bundle = CandidateBundle()

    _build_text_candidates(nodes, bundle)
    _build_text_group_candidates(nodes, bundle)
    _build_brand_candidates(nodes, bundle)
    _build_badge_candidates(nodes, bundle)
    _build_decoration_candidates(nodes, bundle)
    _build_background_candidates(nodes, bundle)
    _build_image_like_candidates(nodes, bundle)

    return bundle
