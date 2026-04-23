from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from banner_pipeline.build_candidates import CandidateBundle, SemanticCandidate
from banner_pipeline.collapse_groups import CollapsedNode


@dataclass
class HeuristicDecision:
    rule_name: str
    assigned_role_hint: str | None = None
    assigned_group_hint: str | None = None
    assigned_importance_hint: str | None = None
    confidence: float = 0.0
    reason: str = ""


@dataclass
class HeuristicAnnotatedCandidate:
    candidate: SemanticCandidate
    final_role_hint: str | None = None
    final_group_hint: str | None = None
    final_importance_hint: str | None = None
    confidence: float = 0.0
    decisions: list[HeuristicDecision] = field(default_factory=list)
    extra_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "final_role_hint": self.final_role_hint,
            "final_group_hint": self.final_group_hint,
            "final_importance_hint": self.final_importance_hint,
            "confidence": self.confidence,
            "decisions": [
                {
                    "rule_name": d.rule_name,
                    "assigned_role_hint": d.assigned_role_hint,
                    "assigned_group_hint": d.assigned_group_hint,
                    "assigned_importance_hint": d.assigned_importance_hint,
                    "confidence": d.confidence,
                    "reason": d.reason,
                }
                for d in self.decisions
            ],
            "extra_data": self.extra_data,
        }


@dataclass
class HeuristicBundle:
    annotated_candidates: list[HeuristicAnnotatedCandidate] = field(default_factory=list)
    by_candidate_id: dict[str, HeuristicAnnotatedCandidate] = field(default_factory=dict)

    headline_candidates: list[str] = field(default_factory=list)
    subheadline_candidates: list[str] = field(default_factory=list)
    legal_candidates: list[str] = field(default_factory=list)
    badge_candidates: list[str] = field(default_factory=list)
    decoration_candidates: list[str] = field(default_factory=list)
    brand_candidates: list[str] = field(default_factory=list)
    background_candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotated_candidates": [c.to_dict() for c in self.annotated_candidates],
            "headline_candidates": self.headline_candidates,
            "subheadline_candidates": self.subheadline_candidates,
            "legal_candidates": self.legal_candidates,
            "badge_candidates": self.badge_candidates,
            "decoration_candidates": self.decoration_candidates,
            "brand_candidates": self.brand_candidates,
            "background_candidates": self.background_candidates,
        }


def _node_map(nodes: list[CollapsedNode]) -> dict[str, CollapsedNode]:
    return {n.id: n for n in nodes}


def _candidate_nodes(candidate: SemanticCandidate, nodes_map: dict[str, CollapsedNode]) -> list[CollapsedNode]:
    return [nodes_map[nid] for nid in candidate.source_node_ids if nid in nodes_map]


def _safe_text(candidate: SemanticCandidate) -> str:
    return (candidate.text_content or "").strip()


def _char_count(candidate: SemanticCandidate) -> int:
    return len(_safe_text(candidate))


def _word_count(candidate: SemanticCandidate) -> int:
    text = _safe_text(candidate)
    if not text:
        return 0
    return len([t for t in text.replace("\n", " ").split(" ") if t.strip()])


def _line_count(candidate: SemanticCandidate) -> int:
    text = _safe_text(candidate)
    if not text:
        return 0
    return len([line for line in text.splitlines() if line.strip()]) or 1


def _is_text_candidate(candidate: SemanticCandidate) -> bool:
    return candidate.candidate_type == "text"


def _is_text_group_candidate(candidate: SemanticCandidate) -> bool:
    return candidate.candidate_type == "text_group"


def _top_region(candidate: SemanticCandidate) -> bool:
    return candidate.y_norm <= 0.22


def _bottom_region(candidate: SemanticCandidate) -> bool:
    return candidate.y_norm >= 0.72


def _left_region(candidate: SemanticCandidate) -> bool:
    return candidate.x_norm <= 0.45


def _right_region(candidate: SemanticCandidate) -> bool:
    return candidate.x_norm >= 0.70


def _small_area(candidate: SemanticCandidate) -> bool:
    return candidate.area_ratio <= 0.03


def _large_area(candidate: SemanticCandidate) -> bool:
    return candidate.area_ratio >= 0.12


def _short_text(candidate: SemanticCandidate) -> bool:
    return _char_count(candidate) <= 12


def _long_text(candidate: SemanticCandidate) -> bool:
    return _char_count(candidate) >= 40


def _contains_digits(candidate: SemanticCandidate) -> bool:
    return any(ch.isdigit() for ch in _safe_text(candidate))


def _looks_like_age_badge_text(candidate: SemanticCandidate) -> bool:
    text = _safe_text(candidate)
    normalized = text.replace(" ", "")
    return normalized in {"0+", "6+", "12+", "16+", "18+"}


def _looks_like_price(candidate: SemanticCandidate) -> bool:
    text = _safe_text(candidate).lower()
    if not text:
        return False

    currency_tokens = ["₽", "$", "€", "руб", "руб.", "сом", "тенге", "₸", "uah", "грн"]
    has_currency = any(tok in text for tok in currency_tokens)
    has_digits = _contains_digits(candidate)
    return has_digits and (has_currency or len(text) <= 10)


def _looks_like_discount(candidate: SemanticCandidate) -> bool:
    text = _safe_text(candidate).lower()
    return "%" in text or "скид" in text or "sale" in text or "off" in text


def _looks_like_delivery_or_time(candidate: SemanticCandidate) -> bool:
    text = _safe_text(candidate).lower()
    time_markers = [
        "мин", "minute", "minutes", "mins", "час", "hours",
        "доставка", "delivery", "от ", "from "
    ]
    return any(marker in text for marker in time_markers)


def _looks_like_legal(candidate: SemanticCandidate) -> bool:
    text = _safe_text(candidate).lower()
    if not text:
        return False

    legal_markers = [
        "ооо", "огрн", "продавец", "доставку осуществляют",
        "партнёры сервиса", "юрид", "адрес", "inn", "llc",
        "terms", "conditions", "disclaimer"
    ]
    marker_hit = any(marker in text for marker in legal_markers)

    return _long_text(candidate) or marker_hit


def _looks_like_headline(candidate: SemanticCandidate) -> bool:
    if not _is_text_candidate(candidate):
        return False
    if _looks_like_legal(candidate):
        return False
    if _looks_like_price(candidate):
        return False

    return candidate.area_ratio >= 0.05 and _left_region(candidate) and _word_count(candidate) >= 2


def _looks_like_subheadline(candidate: SemanticCandidate) -> bool:
    if not _is_text_candidate(candidate):
        return False
    if _looks_like_legal(candidate):
        return False
    if _looks_like_price(candidate):
        return False

    text = _safe_text(candidate)
    return 2 <= _word_count(candidate) <= 6 or _looks_like_delivery_or_time(candidate) or len(text) <= 30


def _overlap_1d(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _horizontal_overlap_ratio(a: SemanticCandidate, b: SemanticCandidate) -> float:
    overlap = _overlap_1d(a.x_norm, a.x_norm + a.w_norm, b.x_norm, b.x_norm + b.w_norm)
    denom = min(a.w_norm, b.w_norm)
    if denom <= 0:
        return 0.0
    return overlap / denom


def _vertical_gap(a: SemanticCandidate, b: SemanticCandidate) -> float:
    return b.y_norm - (a.y_norm + a.h_norm)


def _sort_text_candidates_by_prominence(candidates: list[SemanticCandidate]) -> list[SemanticCandidate]:
    return sorted(candidates, key=lambda c: (-c.area_ratio, c.y_norm, c.x_norm))


def _find_best_headline_candidate(candidates: list[SemanticCandidate]) -> SemanticCandidate | None:
    text_candidates = [c for c in candidates if _is_text_candidate(c)]
    plausible = [c for c in text_candidates if _looks_like_headline(c)]
    if not plausible:
        return None
    ranked = _sort_text_candidates_by_prominence(plausible)
    return ranked[0]


def _find_subheadline_for_headline(
    headline: SemanticCandidate | None,
    candidates: list[SemanticCandidate],
) -> SemanticCandidate | None:
    if headline is None:
        return None

    text_candidates = [c for c in candidates if _is_text_candidate(c) and c.candidate_id != headline.candidate_id]

    plausible: list[SemanticCandidate] = []
    for c in text_candidates:
        if not _looks_like_subheadline(c):
            continue

        gap = _vertical_gap(headline, c)
        horiz_overlap = _horizontal_overlap_ratio(headline, c)

        if -0.02 <= gap <= 0.20 and horiz_overlap >= 0.35:
            plausible.append(c)

    if not plausible:
        return None

    plausible.sort(key=lambda c: (abs(_vertical_gap(headline, c)), c.y_norm, c.x_norm))
    return plausible[0]


def _assign(
    annotated: HeuristicAnnotatedCandidate,
    decision: HeuristicDecision,
) -> None:
    annotated.decisions.append(decision)

    if decision.assigned_role_hint is not None:
        annotated.final_role_hint = decision.assigned_role_hint
    if decision.assigned_group_hint is not None:
        annotated.final_group_hint = decision.assigned_group_hint
    if decision.assigned_importance_hint is not None:
        annotated.final_importance_hint = decision.assigned_importance_hint

    annotated.confidence = max(annotated.confidence, decision.confidence)


def _make_default_annotation(candidate: SemanticCandidate) -> HeuristicAnnotatedCandidate:
    return HeuristicAnnotatedCandidate(
        candidate=candidate,
        final_role_hint=candidate.role_hint,
        final_group_hint=candidate.group_hint,
        final_importance_hint=candidate.importance_hint,
        confidence=0.0,
        decisions=[],
        extra_data={},
    )


def _annotate_candidate_type_priors(
    annotated: HeuristicAnnotatedCandidate,
) -> None:
    candidate = annotated.candidate

    if candidate.candidate_type == "brand":
        _assign(
            annotated,
            HeuristicDecision(
                rule_name="candidate_type_brand",
                assigned_role_hint="brand",
                assigned_group_hint="brand_group",
                assigned_importance_hint="critical",
                confidence=0.92,
                reason="Brand candidate from top-left compact non-text cluster.",
            ),
        )

    elif candidate.candidate_type == "badge":
        role = "age_badge" if _looks_like_age_badge_text(candidate) else "badge"
        _assign(
            annotated,
            HeuristicDecision(
                rule_name="candidate_type_badge",
                assigned_role_hint=role,
                assigned_group_hint="badge_group",
                assigned_importance_hint="high",
                confidence=0.88,
                reason="Small short text in top-right behaves like badge.",
            ),
        )

    elif candidate.candidate_type == "decoration":
        _assign(
            annotated,
            HeuristicDecision(
                rule_name="candidate_type_decoration",
                assigned_role_hint="decoration",
                assigned_group_hint="decoration_group",
                assigned_importance_hint="low",
                confidence=0.90,
                reason="Decorative candidate from star/small shape logic.",
            ),
        )

    elif candidate.candidate_type == "background":
        _assign(
            annotated,
            HeuristicDecision(
                rule_name="candidate_type_background",
                assigned_role_hint="background",
                assigned_group_hint="background_group",
                assigned_importance_hint="medium",
                confidence=0.82,
                reason="Large shape-like node likely serves as background/panel.",
            ),
        )

    elif candidate.candidate_type == "image_like":
        _assign(
            annotated,
            HeuristicDecision(
                rule_name="candidate_type_image_like",
                assigned_role_hint="image_like",
                assigned_importance_hint="medium",
                confidence=0.70,
                reason="Large rectangle/vector-like node may serve as image/product region.",
            ),
        )

    elif candidate.candidate_type == "text_group":
        _assign(
            annotated,
            HeuristicDecision(
                rule_name="candidate_type_text_group",
                assigned_role_hint="text_group",
                assigned_group_hint="text_group",
                assigned_importance_hint="high",
                confidence=0.74,
                reason="Grouped stacked text likely forms a semantic text block.",
            ),
        )

    elif candidate.candidate_type == "text":
        _assign(
            annotated,
            HeuristicDecision(
                rule_name="candidate_type_text",
                assigned_role_hint="text",
                assigned_importance_hint="medium",
                confidence=0.35,
                reason="Single text candidate prior.",
            ),
        )


def _annotate_text_specific_rules(
    annotated: HeuristicAnnotatedCandidate,
) -> None:
    candidate = annotated.candidate
    if not _is_text_candidate(candidate):
        return

    text = _safe_text(candidate)

    if not text:
        return

    if _looks_like_age_badge_text(candidate) and _top_region(candidate) and _right_region(candidate):
        _assign(
            annotated,
            HeuristicDecision(
                rule_name="text_age_badge",
                assigned_role_hint="age_badge",
                assigned_group_hint="badge_group",
                assigned_importance_hint="high",
                confidence=0.96,
                reason="Short age-mark text in top-right.",
            ),
        )
        return

    if _looks_like_legal(candidate) and _bottom_region(candidate):
        _assign(
            annotated,
            HeuristicDecision(
                rule_name="text_legal_bottom",
                assigned_role_hint="legal",
                assigned_group_hint="legal_group",
                assigned_importance_hint="critical",
                confidence=0.95,
                reason="Long legal-like text near bottom region.",
            ),
        )
        return

    if _looks_like_discount(candidate):
        _assign(
            annotated,
            HeuristicDecision(
                rule_name="text_discount",
                assigned_role_hint="discount_text",
                assigned_group_hint="badge_group",
                assigned_importance_hint="high",
                confidence=0.82,
                reason="Text contains discount marker like % or sale token.",
            ),
        )

    if _looks_like_price(candidate):
        price_role = "price_main"
        if len(text) <= 4 and _contains_digits(candidate):
            price_role = "price_fraction"

        _assign(
            annotated,
            HeuristicDecision(
                rule_name="text_price",
                assigned_role_hint=price_role,
                assigned_group_hint="price_group",
                assigned_importance_hint="high",
                confidence=0.83,
                reason="Text resembles numeric price/currency expression.",
            ),
        )

    if _looks_like_delivery_or_time(candidate) and not _looks_like_legal(candidate):
        _assign(
            annotated,
            HeuristicDecision(
                rule_name="text_delivery_time",
                assigned_role_hint="subheadline",
                assigned_group_hint="headline_group",
                assigned_importance_hint="high",
                confidence=0.78,
                reason="Short delivery/time text under main promo copy.",
            ),
        )

    if _looks_like_headline(candidate):
        _assign(
            annotated,
            HeuristicDecision(
                rule_name="text_headline",
                assigned_role_hint="headline",
                assigned_group_hint="headline_group",
                assigned_importance_hint="critical",
                confidence=0.80,
                reason="Large prominent left-side multi-word text.",
            ),
        )


def _annotate_text_group_rules(
    annotated: HeuristicAnnotatedCandidate,
) -> None:
    candidate = annotated.candidate
    if not _is_text_group_candidate(candidate):
        return

    line_count = _line_count(candidate)
    word_count = _word_count(candidate)

    if candidate.text_count >= 2 and _left_region(candidate):
        _assign(
            annotated,
            HeuristicDecision(
                rule_name="text_group_left_stack",
                assigned_role_hint="headline_group",
                assigned_group_hint="headline_group",
                assigned_importance_hint="critical",
                confidence=0.84,
                reason="Multiple left-side stacked texts likely form main text hierarchy.",
            ),
        )

    if _bottom_region(candidate) and _looks_like_legal(candidate):
        _assign(
            annotated,
            HeuristicDecision(
                rule_name="text_group_legal",
                assigned_role_hint="legal_group",
                assigned_group_hint="legal_group",
                assigned_importance_hint="critical",
                confidence=0.86,
                reason="Bottom grouped text with legal-like content.",
            ),
        )

    if line_count >= 2 and word_count >= 4 and candidate.area_ratio >= 0.05:
        _assign(
            annotated,
            HeuristicDecision(
                rule_name="text_group_promo_copy",
                assigned_role_hint="text_group",
                assigned_group_hint="headline_group",
                assigned_importance_hint="high",
                confidence=0.70,
                reason="Large multi-line text block likely serves promo/headline area.",
            ),
        )


def _annotate_global_headline_subheadline(
    annotations: list[HeuristicAnnotatedCandidate],
) -> None:
    candidates = [a.candidate for a in annotations]
    headline = _find_best_headline_candidate(candidates)
    subheadline = _find_subheadline_for_headline(headline, candidates)

    if headline is not None:
        ann = next(a for a in annotations if a.candidate.candidate_id == headline.candidate_id)
        _assign(
            ann,
            HeuristicDecision(
                rule_name="global_best_headline",
                assigned_role_hint="headline",
                assigned_group_hint="headline_group",
                assigned_importance_hint="critical",
                confidence=0.93,
                reason="Selected as strongest global headline candidate by prominence and placement.",
            ),
        )

    if subheadline is not None:
        ann = next(a for a in annotations if a.candidate.candidate_id == subheadline.candidate_id)
        _assign(
            ann,
            HeuristicDecision(
                rule_name="global_best_subheadline",
                assigned_role_hint="subheadline",
                assigned_group_hint="headline_group",
                assigned_importance_hint="high",
                confidence=0.90,
                reason="Selected as best supporting text below headline.",
            ),
        )


def _annotate_brand_confidence_boost(
    annotated: HeuristicAnnotatedCandidate,
    nodes_map: dict[str, CollapsedNode],
) -> None:
    candidate = annotated.candidate
    if candidate.candidate_type != "brand":
        return

    nodes = _candidate_nodes(candidate, nodes_map)
    has_group_like = any(n.type.lower() in {"group", "frame", "component", "instance"} for n in nodes)
    has_shape_like = any(n.type.lower() in {"vector", "rectangle", "ellipse", "star"} for n in nodes)

    if has_group_like and has_shape_like:
        _assign(
            annotated,
            HeuristicDecision(
                rule_name="brand_structure_boost",
                assigned_role_hint="brand",
                assigned_group_hint="brand_group",
                assigned_importance_hint="critical",
                confidence=0.96,
                reason="Mixed group/vector structure strongly resembles logo/brand assembly.",
            ),
        )


def _fallback_defaults(annotated: HeuristicAnnotatedCandidate) -> None:
    if annotated.final_role_hint is None:
        annotated.final_role_hint = annotated.candidate.role_hint or annotated.candidate.candidate_type

    if annotated.final_group_hint is None:
        if annotated.candidate.candidate_type == "text_group":
            annotated.final_group_hint = "text_group"
        elif annotated.candidate.candidate_type == "text":
            annotated.final_group_hint = "text_group"
        elif annotated.candidate.candidate_type == "brand":
            annotated.final_group_hint = "brand_group"
        elif annotated.candidate.candidate_type == "badge":
            annotated.final_group_hint = "badge_group"
        elif annotated.candidate.candidate_type == "decoration":
            annotated.final_group_hint = "decoration_group"
        elif annotated.candidate.candidate_type == "background":
            annotated.final_group_hint = "background_group"
        else:
            annotated.final_group_hint = None

    if annotated.final_importance_hint is None:
        annotated.final_importance_hint = annotated.candidate.importance_hint or "medium"

    if annotated.confidence <= 0.0:
        annotated.confidence = 0.25


def _bucketize(bundle: HeuristicBundle) -> None:
    for ann in bundle.annotated_candidates:
        cid = ann.candidate.candidate_id
        role = ann.final_role_hint

        if role == "headline":
            bundle.headline_candidates.append(cid)
        elif role == "subheadline":
            bundle.subheadline_candidates.append(cid)
        elif role == "legal":
            bundle.legal_candidates.append(cid)
        elif role in {"age_badge", "badge"}:
            bundle.badge_candidates.append(cid)
        elif role == "decoration":
            bundle.decoration_candidates.append(cid)
        elif role == "brand":
            bundle.brand_candidates.append(cid)
        elif role == "background":
            bundle.background_candidates.append(cid)


def apply_heuristics(
    candidate_bundle: CandidateBundle,
    collapsed_nodes: list[CollapsedNode],
) -> HeuristicBundle:
    nodes_map = _node_map(collapsed_nodes)
    out = HeuristicBundle()

    annotations: list[HeuristicAnnotatedCandidate] = []

    for candidate in candidate_bundle.all_candidates:
        ann = _make_default_annotation(candidate)

        _annotate_candidate_type_priors(ann)
        _annotate_text_specific_rules(ann)
        _annotate_text_group_rules(ann)
        _annotate_brand_confidence_boost(ann, nodes_map)

        annotations.append(ann)

    _annotate_global_headline_subheadline(annotations)

    for ann in annotations:
        _fallback_defaults(ann)

    out.annotated_candidates = annotations
    out.by_candidate_id = {ann.candidate.candidate_id: ann for ann in annotations}

    _bucketize(out)
    return out
