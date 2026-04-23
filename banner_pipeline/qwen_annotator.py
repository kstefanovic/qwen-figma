from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from banner_pipeline.build_candidates import CandidateBundle, SemanticCandidate
from banner_pipeline.heuristics import HeuristicAnnotatedCandidate, HeuristicBundle


@dataclass
class BannerAnnotation:
    layout_pattern: str = "unknown"
    pattern_confidence: float = 0.0
    zones: list[dict[str, Any]] = field(default_factory=list)
    preservation_priorities: list[dict[str, Any]] = field(default_factory=list)
    reason_short: str = ""
    raw_model_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_pattern": self.layout_pattern,
            "pattern_confidence": self.pattern_confidence,
            "zones": self.zones,
            "preservation_priorities": self.preservation_priorities,
            "reason_short": self.reason_short,
            "raw_model_output": self.raw_model_output,
        }


@dataclass
class CandidateAnnotation:
    candidate_id: str
    element_role: str = "unknown"
    functional_type: str = "functional"
    importance_level: str = "medium"
    is_text: Optional[bool] = None
    is_brand_related: bool = False
    is_required_for_compliance: bool = False
    adaptation_policy: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason_short: str = ""
    raw_model_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "element_role": self.element_role,
            "functional_type": self.functional_type,
            "importance_level": self.importance_level,
            "is_text": self.is_text,
            "is_brand_related": self.is_brand_related,
            "is_required_for_compliance": self.is_required_for_compliance,
            "adaptation_policy": self.adaptation_policy,
            "confidence": self.confidence,
            "reason_short": self.reason_short,
            "raw_model_output": self.raw_model_output,
        }


@dataclass
class GroupAnnotation:
    candidate_id: str
    is_meaningful_group: bool = True
    group_role: str = "unknown"
    internal_layout: str = "freeform"
    preserve_as_unit: bool = True
    importance_level: str = "medium"
    adaptation_policy: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason_short: str = ""
    raw_model_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "is_meaningful_group": self.is_meaningful_group,
            "group_role": self.group_role,
            "internal_layout": self.internal_layout,
            "preserve_as_unit": self.preserve_as_unit,
            "importance_level": self.importance_level,
            "adaptation_policy": self.adaptation_policy,
            "confidence": self.confidence,
            "reason_short": self.reason_short,
            "raw_model_output": self.raw_model_output,
        }


class QwenAnnotator:
    """
    Client for persistent Qwen HTTP service.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        timeout_seconds: int = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def load_model(self) -> None:
        health = self.health_check()
        if health.get("status") != "ok":
            raise RuntimeError(f"Qwen service unhealthy: {health}")

    def health_check(self) -> dict[str, Any]:
        resp = requests.get(
            f"{self.base_url}/health",
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        return resp.json()

    def annotate_banner(
        self,
        banner_image_path: str,
        candidate_bundle: CandidateBundle,
        heuristic_bundle: Optional[HeuristicBundle] = None,
    ) -> BannerAnnotation:
        payload = {
            "banner_image_path": banner_image_path,
            "candidate_bundle": candidate_bundle.to_dict(),
            "heuristic_bundle": heuristic_bundle.to_dict() if heuristic_bundle else None,
        }
        data = self._post("/annotate/banner", payload)
        return BannerAnnotation(
            layout_pattern=str(data.get("layout_pattern", "unknown")),
            pattern_confidence=float(data.get("pattern_confidence", 0.0) or 0.0),
            zones=list(data.get("zones", []) or []),
            preservation_priorities=list(data.get("preservation_priorities", []) or []),
            reason_short=str(data.get("reason_short", "")),
            raw_model_output=str(data.get("raw_model_output", "")),
        )

    def annotate_candidate(
        self,
        banner_image_path: str,
        candidate: SemanticCandidate,
        heuristic_annotation: Optional[HeuristicAnnotatedCandidate] = None,
        context_padding_ratio: float = 0.08,
    ) -> CandidateAnnotation:
        payload = {
            "banner_image_path": banner_image_path,
            "candidate": candidate.to_dict(),
            "heuristic_annotation": heuristic_annotation.to_dict() if heuristic_annotation else None,
            "context_padding_ratio": context_padding_ratio,
        }
        data = self._post("/annotate/candidate", payload)
        return CandidateAnnotation(
            candidate_id=str(data.get("candidate_id", candidate.candidate_id)),
            element_role=str(data.get("element_role", "unknown")),
            functional_type=str(data.get("functional_type", "functional")),
            importance_level=str(data.get("importance_level", "medium")),
            is_text=data.get("is_text", None),
            is_brand_related=bool(data.get("is_brand_related", False)),
            is_required_for_compliance=bool(data.get("is_required_for_compliance", False)),
            adaptation_policy=dict(data.get("adaptation_policy", {}) or {}),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            reason_short=str(data.get("reason_short", "")),
            raw_model_output=str(data.get("raw_model_output", "")),
        )

    def annotate_group_candidate(
        self,
        banner_image_path: str,
        candidate: SemanticCandidate,
        heuristic_annotation: Optional[HeuristicAnnotatedCandidate] = None,
        context_padding_ratio: float = 0.08,
    ) -> GroupAnnotation:
        payload = {
            "banner_image_path": banner_image_path,
            "candidate": candidate.to_dict(),
            "heuristic_annotation": heuristic_annotation.to_dict() if heuristic_annotation else None,
            "context_padding_ratio": context_padding_ratio,
        }
        data = self._post("/annotate/group", payload)
        return GroupAnnotation(
            candidate_id=str(data.get("candidate_id", candidate.candidate_id)),
            is_meaningful_group=bool(data.get("is_meaningful_group", True)),
            group_role=str(data.get("group_role", "unknown")),
            internal_layout=str(data.get("internal_layout", "freeform")),
            preserve_as_unit=bool(data.get("preserve_as_unit", True)),
            importance_level=str(data.get("importance_level", "medium")),
            adaptation_policy=dict(data.get("adaptation_policy", {}) or {}),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            reason_short=str(data.get("reason_short", "")),
            raw_model_output=str(data.get("raw_model_output", "")),
        )

    def annotate_candidates(
        self,
        banner_image_path: str,
        candidate_bundle: CandidateBundle,
        heuristic_bundle: Optional[HeuristicBundle] = None,
    ) -> dict[str, CandidateAnnotation]:
        out: dict[str, CandidateAnnotation] = {}
        heuristic_map = heuristic_bundle.by_candidate_id if heuristic_bundle else {}

        for candidate in candidate_bundle.all_candidates:
            heuristic_ann = heuristic_map.get(candidate.candidate_id)
            out[candidate.candidate_id] = self.annotate_candidate(
                banner_image_path=banner_image_path,
                candidate=candidate,
                heuristic_annotation=heuristic_ann,
            )
        return out

    def annotate_group_candidates(
        self,
        banner_image_path: str,
        candidate_bundle: CandidateBundle,
        heuristic_bundle: Optional[HeuristicBundle] = None,
    ) -> dict[str, GroupAnnotation]:
        out: dict[str, GroupAnnotation] = {}
        heuristic_map = heuristic_bundle.by_candidate_id if heuristic_bundle else {}

        group_candidates = [
            *candidate_bundle.text_group_candidates,
            *candidate_bundle.brand_candidates,
            *candidate_bundle.decoration_candidates,
            *candidate_bundle.background_candidates,
            *candidate_bundle.image_like_candidates,
        ]

        seen: set[str] = set()
        for candidate in group_candidates:
            if candidate.candidate_id in seen:
                continue
            seen.add(candidate.candidate_id)

            heuristic_ann = heuristic_map.get(candidate.candidate_id)
            out[candidate.candidate_id] = self.annotate_group_candidate(
                banner_image_path=banner_image_path,
                candidate=candidate,
                heuristic_annotation=heuristic_ann,
            )
        return out

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        return resp.json()
