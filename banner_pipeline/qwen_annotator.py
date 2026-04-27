from __future__ import annotations

import json
import logging

from env_load import default_qwen_base_url
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
from PIL import Image

from banner_pipeline.build_candidates import CandidateBundle, SemanticCandidate
from banner_pipeline.heuristics import HeuristicAnnotatedCandidate, HeuristicBundle


logger = logging.getLogger(__name__)


def _extract_http_error_detail(resp: requests.Response) -> str:
    try:
        payload = resp.json()
    except Exception:
        text = (resp.text or "").strip()
        return text[:8000] if text else "(empty response body)"

    detail = payload.get("detail")
    if detail is None:
        return str(payload)[:8000]
    if isinstance(detail, list):
        parts: list[str] = []
        for item in detail:
            if isinstance(item, dict):
                loc = item.get("loc")
                msg = item.get("msg")
                parts.append(f"{loc}: {msg}" if loc or msg else str(item))
            else:
                parts.append(str(item))
        joined = "; ".join(parts)
        return joined[:8000]
    return str(detail)[:8000]


def _format_qwen_service_http_error(endpoint: str, resp: requests.Response) -> str:
    detail = _extract_http_error_detail(resp)
    return f"Qwen service request failed: {endpoint} status={resp.status_code} detail={detail}"


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
    semantic_name: str = ""
    parent_semantic_name: str = ""
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
            "semantic_name": self.semantic_name,
            "parent_semantic_name": self.parent_semantic_name,
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
class BrandContextAnnotation:
    brand_family: str = "generic"
    brand_confidence: float = 0.0
    language: str = "unknown"
    category: str = "unknown"
    reason_short: str = ""
    raw_model_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "brand_family": self.brand_family,
            "brand_confidence": self.brand_confidence,
            "language": self.language,
            "category": self.category,
            "reason_short": self.reason_short,
            "raw_model_output": self.raw_model_output,
        }


@dataclass
class GroupAnnotation:
    candidate_id: str
    is_meaningful_group: bool = True
    group_role: str = "unknown"
    semantic_name: str = ""
    parent_semantic_name: str = ""
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
            "semantic_name": self.semantic_name,
            "parent_semantic_name": self.parent_semantic_name,
            "internal_layout": self.internal_layout,
            "preserve_as_unit": self.preserve_as_unit,
            "importance_level": self.importance_level,
            "adaptation_policy": self.adaptation_policy,
            "confidence": self.confidence,
            "reason_short": self.reason_short,
            "raw_model_output": self.raw_model_output,
        }


@dataclass
class SemanticStructureResult:
    """Single-pass /annotate/semantic-structure unpacked for merge_semantic_graph."""

    brand_context: BrandContextAnnotation
    banner_annotation: BannerAnnotation
    candidate_annotations: dict[str, CandidateAnnotation]
    group_annotations: dict[str, GroupAnnotation]
    semantic_updates: list[dict[str, Any]]
    semantic_groups: list[dict[str, Any]]
    qwen_calls_made: int
    candidates_annotated: int
    groups_annotated: int
    heuristic_fallback: bool = False


@dataclass
class QwenRequestMetric:
    endpoint: str
    elapsed_seconds: float
    payload_size_bytes: int
    image_size: Optional[dict[str, Any]] = None
    status_code: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "elapsed_seconds": self.elapsed_seconds,
            "payload_size_bytes": self.payload_size_bytes,
            "image_size": self.image_size,
            "status_code": self.status_code,
        }


def _infer_qwen_calls_from_semantic_payload(data: dict[str, Any]) -> int:
    """0 VLM calls when heuristic-only (e.g. extreme aspect); 1 when the model ran (success or parse fallback)."""
    hf = bool(data.get("heuristic_fallback"))
    raw = str(data.get("raw_model_output", "") or "").strip()
    if hf and not raw:
        return 0
    return 1


def unpack_semantic_structure_response(data: dict[str, Any]) -> SemanticStructureResult:
    """Convert /annotate/semantic-structure JSON into merge-compatible annotation objects."""
    raw_all = str(data.get("raw_model_output", "") or "")
    hf = bool(data.get("heuristic_fallback"))

    bc = BrandContextAnnotation(
        brand_family=str(data.get("brand_family", "generic") or "generic"),
        brand_confidence=float(data.get("brand_confidence", 0.0) or 0.0),
        language=str(data.get("language", "unknown") or "unknown"),
        category=str(data.get("category", "unknown") or "unknown"),
        reason_short=str(data.get("reason_short", "")),
        raw_model_output=raw_all,
    )

    banner = BannerAnnotation(
        layout_pattern=str(data.get("layout_pattern", "unknown")),
        pattern_confidence=float(data.get("pattern_confidence", 0.0) or 0.0),
        zones=list(data.get("zones", []) or []),
        preservation_priorities=list(data.get("preservation_priorities", []) or []),
        reason_short=str(data.get("reason_short", "")),
        raw_model_output=raw_all,
    )

    candidate_annotations: dict[str, CandidateAnnotation] = {}
    for cid, v in (data.get("element_annotations") or {}).items():
        if not isinstance(v, dict):
            continue
        cid_s = str(cid)
        candidate_annotations[cid_s] = CandidateAnnotation(
            candidate_id=cid_s,
            element_role=str(v.get("element_role", "unknown")),
            semantic_name=str(v.get("semantic_name", "")),
            parent_semantic_name=str(v.get("parent_semantic_name", "")),
            functional_type=str(v.get("functional_type", "functional")),
            importance_level=str(v.get("importance_level", "medium")),
            is_text=v.get("is_text", None),
            is_brand_related=bool(v.get("is_brand_related", False)),
            is_required_for_compliance=bool(v.get("is_required_for_compliance", False)),
            adaptation_policy=dict(v.get("adaptation_policy") or {}),
            confidence=float(v.get("confidence", 0.0) or 0.0),
            reason_short=str(v.get("reason_short", "")),
            raw_model_output="",
        )

    group_annotations: dict[str, GroupAnnotation] = {}
    for cid, v in (data.get("group_annotations") or {}).items():
        if not isinstance(v, dict):
            continue
        cid_s = str(cid)
        group_annotations[cid_s] = GroupAnnotation(
            candidate_id=cid_s,
            is_meaningful_group=True,
            group_role=str(v.get("group_role", "unknown")),
            semantic_name=str(v.get("semantic_name", "")),
            parent_semantic_name=str(v.get("parent_semantic_name", "")),
            internal_layout=str(v.get("internal_layout", "freeform")),
            preserve_as_unit=bool(v.get("preserve_as_unit", True)),
            importance_level=str(v.get("importance_level", "medium")),
            adaptation_policy=dict(v.get("adaptation_policy") or {}),
            confidence=float(v.get("confidence", 0.0) or 0.0),
            reason_short=str(v.get("reason_short", "")),
            raw_model_output="",
        )

    calls = _infer_qwen_calls_from_semantic_payload(data)
    return SemanticStructureResult(
        brand_context=bc,
        banner_annotation=banner,
        candidate_annotations=candidate_annotations,
        group_annotations=group_annotations,
        semantic_updates=list(data.get("updates", []) or []),
        semantic_groups=list(data.get("groups", []) or []),
        qwen_calls_made=calls,
        candidates_annotated=len(candidate_annotations),
        groups_annotated=len(group_annotations),
        heuristic_fallback=hf,
    )


class QwenAnnotator:
    """
    Client for persistent Qwen HTTP service.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: int = 300,
    ) -> None:
        self.base_url = ((base_url or "").strip() or default_qwen_base_url()).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.request_metrics: list[QwenRequestMetric] = []

    @property
    def qwen_call_count(self) -> int:
        return len(self.request_metrics)

    def request_metrics_dicts(self) -> list[dict[str, Any]]:
        return [metric.to_dict() for metric in self.request_metrics]

    def _estimate_payload_size_bytes(self, payload: dict[str, Any]) -> int:
        try:
            encoded = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        except Exception:
            encoded = repr(payload).encode("utf-8", errors="ignore")
        return len(encoded)

    def _read_image_size(self, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        image_path = payload.get("banner_image_path")
        if not image_path:
            return None

        path = Path(str(image_path))
        if not path.exists():
            return {"path": str(path), "exists": False}

        info: dict[str, Any] = {
            "path": str(path),
            "exists": True,
            "bytes": path.stat().st_size,
        }
        try:
            with Image.open(path) as image:
                info["width"] = image.width
                info["height"] = image.height
                info["format"] = image.format
        except Exception:
            pass
        return info

    def load_model(self) -> None:
        health = self.health_check()
        if health.get("status") != "ok":
            raise RuntimeError(f"Qwen service unhealthy: {health}")

    def health_check(self) -> dict[str, Any]:
        resp = requests.get(
            f"{self.base_url}/health",
            timeout=self.timeout_seconds,
        )
        if not resp.ok:
            raise RuntimeError(_format_qwen_service_http_error("/health", resp))
        return resp.json()

    def annotate_brand_context(
        self,
        banner_image_path: str,
        candidate_bundle: Optional[CandidateBundle] = None,
        heuristic_bundle: Optional[HeuristicBundle] = None,
    ) -> BrandContextAnnotation:
        payload: dict[str, Any] = {
            "banner_image_path": banner_image_path,
            "candidate_bundle": candidate_bundle.to_dict() if candidate_bundle else None,
            "heuristic_bundle": heuristic_bundle.to_dict() if heuristic_bundle else None,
        }
        data = self._post("/annotate/brand-context", payload)
        return BrandContextAnnotation(
            brand_family=str(data.get("brand_family", "generic") or "generic"),
            brand_confidence=float(data.get("brand_confidence", 0.0) or 0.0),
            language=str(data.get("language", "unknown") or "unknown"),
            category=str(data.get("category", "unknown") or "unknown"),
            reason_short=str(data.get("reason_short", "")),
            raw_model_output=str(data.get("raw_model_output", "")),
        )

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
            try:
                out[candidate.candidate_id] = self.annotate_candidate(
                    banner_image_path=banner_image_path,
                    candidate=candidate,
                    heuristic_annotation=heuristic_ann,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Candidate annotation failed for candidate_id={candidate.candidate_id!r}: {e}"
                ) from e
        return out

    def annotate_semantic_structure(
        self,
        banner_image_path: str,
        figma_summary: dict[str, Any],
    ) -> SemanticStructureResult:
        """Backward-compatible alias for single-pass annotation."""
        scene_payload: dict[str, Any] = {
            "banner_metadata": figma_summary.get("canvas") or {},
            "elements": figma_summary.get("nodes") or [],
            "groups": figma_summary.get("candidates") or [],
            "heuristic_roles": {},
            "figma_summary": figma_summary,
        }
        return self.annotate_scene(banner_image_path=banner_image_path, scene_payload=scene_payload)

    def annotate_scene(
        self,
        *,
        banner_image_path: str,
        scene_payload: dict[str, Any],
    ) -> SemanticStructureResult:
        """Single VLM call against /annotate/scene with compact scene payload."""
        payload: dict[str, Any] = {
            "banner_image_path": banner_image_path,
            "banner_metadata": scene_payload.get("banner_metadata") or {},
            "elements": scene_payload.get("elements") or [],
            "groups": scene_payload.get("groups") or [],
            "heuristic_roles": scene_payload.get("heuristic_roles") or {},
            "figma_summary": scene_payload.get("figma_summary") or {},
        }
        extra_paths = scene_payload.get("element_image_paths") or []
        if extra_paths:
            payload["element_image_paths"] = [str(p) for p in extra_paths if str(p).strip()]
        atlas_p = (scene_payload.get("element_atlas_image_path") or "").strip()
        if atlas_p:
            payload["element_atlas_image_path"] = atlas_p
        data = self._post("/annotate/scene", payload)
        return unpack_semantic_structure_response(data)

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
            try:
                out[candidate.candidate_id] = self.annotate_group_candidate(
                    banner_image_path=banner_image_path,
                    candidate=candidate,
                    heuristic_annotation=heuristic_ann,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Group candidate annotation failed for candidate_id={candidate.candidate_id!r}: {e}"
                ) from e
        return out

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        payload_size_bytes = self._estimate_payload_size_bytes(payload)
        image_size = self._read_image_size(payload)
        started = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            timeout=self.timeout_seconds,
        )
        elapsed = time.perf_counter() - started
        metric = QwenRequestMetric(
            endpoint=endpoint,
            elapsed_seconds=elapsed,
            payload_size_bytes=payload_size_bytes,
            image_size=image_size,
            status_code=resp.status_code,
        )
        self.request_metrics.append(metric)
        logger.info(
            "qwen_request endpoint=%s elapsed_seconds=%.3f payload_size_bytes=%d image_size=%s status_code=%s",
            endpoint,
            elapsed,
            payload_size_bytes,
            image_size,
            resp.status_code,
        )
        if not resp.ok:
            raise RuntimeError(_format_qwen_service_http_error(endpoint, resp))
        return resp.json()
