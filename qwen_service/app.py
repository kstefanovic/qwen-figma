from __future__ import annotations

import ast
import io
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from PIL import Image

logger = logging.getLogger(__name__)

_BBOX_MIN_WH = 1e-5
_QWEN_MIN_PIXEL_SIDE = 28
# Qwen2-VL processor rejects absolute aspect ratio >= ~200; stay well below.
_QWEN_SAFE_MAX_ASPECT_RATIO = 150

try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]

from qwen_service.schemas import (
    BannerAnnotateRequest,
    BrandContextAnnotateRequest,
    CandidateAnnotateRequest,
    GroupAnnotateRequest,
    HealthResponse,
    SceneAnnotateRequest,
    SemanticStructureAnnotateRequest,
)

try:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
except Exception:
    torch = None
    AutoProcessor = None
    Qwen2_5_VLForConditionalGeneration = None


@dataclass
class BannerAnnotation:
    layout_pattern: str = "unknown"
    pattern_confidence: float = 0.0
    zones: list[dict[str, Any]] = None
    preservation_priorities: list[dict[str, Any]] = None
    reason_short: str = ""
    raw_model_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_pattern": self.layout_pattern,
            "pattern_confidence": self.pattern_confidence,
            "zones": self.zones or [],
            "preservation_priorities": self.preservation_priorities or [],
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
    adaptation_policy: dict[str, Any] = None
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
            "adaptation_policy": self.adaptation_policy or {},
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
    adaptation_policy: dict[str, Any] = None
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
            "adaptation_policy": self.adaptation_policy or {},
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


def _slugify_machine_brand(value: str) -> str:
    s = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s or "generic"


def _normalize_language_code(value: str) -> str:
    s = (value or "").strip().lower()
    allowed = {"ru", "en", "kk", "mixed", "unknown"}
    if s in allowed:
        return s
    if not s:
        return "unknown"
    if len(s) <= 3 and s.isalpha():
        return s
    return "unknown"


def _normalize_category_code(value: str) -> str:
    s = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    if not s:
        return "unknown"
    return s


def _prompt_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _compact_candidate_bundle_for_brand_context(candidate_bundle: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not candidate_bundle:
        return {}
    counts: dict[str, int] = {}
    all_c = candidate_bundle.get("all_candidates") or []
    for c in all_c:
        if not isinstance(c, dict):
            continue
        t = str(c.get("candidate_type") or "unknown")
        counts[t] = counts.get(t, 0) + 1
    slim: list[dict[str, Any]] = []
    for c in all_c[:20]:
        if not isinstance(c, dict):
            continue
        tc = (c.get("text_content") or "") if isinstance(c.get("text_content"), str) else ""
        slim.append(
            {
                "candidate_id": c.get("candidate_id"),
                "candidate_type": c.get("candidate_type"),
                "role_hint": c.get("role_hint"),
                "text_preview": tc[:120],
            }
        )
    return {"counts_by_candidate_type": counts, "sample_candidates": slim, "total_candidates": len(all_c)}


def _compact_heuristic_bundle_for_brand_context(heuristic_bundle: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not heuristic_bundle:
        return {}
    return {
        "headline_candidates": heuristic_bundle.get("headline_candidates", [])[:8],
        "subheadline_candidates": heuristic_bundle.get("subheadline_candidates", [])[:8],
        "legal_candidates": heuristic_bundle.get("legal_candidates", [])[:8],
        "badge_candidates": heuristic_bundle.get("badge_candidates", [])[:8],
        "brand_candidates": heuristic_bundle.get("brand_candidates", [])[:8],
        "background_candidates": heuristic_bundle.get("background_candidates", [])[:8],
        "decoration_candidates": heuristic_bundle.get("decoration_candidates", [])[:8],
    }


def _finalize_brand_context_output(data: Optional[dict[str, Any]], raw_output: str) -> BrandContextAnnotation:
    """Apply conservative gates so missing or low-confidence inference never crashes downstream."""
    if not isinstance(data, dict):
        return BrandContextAnnotation(
            brand_family="generic",
            brand_confidence=0.0,
            language="unknown",
            category="unknown",
            reason_short="Failed to parse brand-context JSON from model.",
            raw_model_output=raw_output,
        )

    try:
        conf = float(data.get("brand_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    brand = _slugify_machine_brand(str(data.get("brand_family", "") or ""))
    lang = _normalize_language_code(str(data.get("language", "") or ""))
    cat = _normalize_category_code(str(data.get("category", "") or ""))

    reason = str(data.get("reason_short", "") or "").strip()

    low_conf = conf < 0.45
    if low_conf:
        brand = "generic"
        if not reason:
            reason = "Low brand_confidence; defaulted brand_family to generic."
        elif "generic" not in reason.lower():
            reason = f"{reason} (brand_family forced to generic due to low confidence.)"

    if brand in {"", "unknown", "unk", "unknown_brand"}:
        brand = "generic"

    if lang == "unknown" and not low_conf and not reason:
        reason = "Language unclear; using unknown."

    return BrandContextAnnotation(
        brand_family=brand,
        brand_confidence=conf,
        language=lang if lang else "unknown",
        category=cat if cat else "unknown",
        reason_short=reason or "Model-provided brand context.",
        raw_model_output=raw_output,
    )


def _upscale_to_min_side(image: Image.Image, min_side: int) -> Image.Image:
    """Upscale so min(width, height) >= min_side; preserves aspect ratio. Uses LANCZOS."""
    w, h = image.size
    if min(w, h) >= min_side:
        return image
    scale = min_side / float(min(w, h))
    nw = max(min_side, int(round(w * scale)))
    nh = max(min_side, int(round(h * scale)))
    nw = max(min_side, nw)
    nh = max(min_side, nh)
    return image.resize((nw, nh), _LANCZOS)


def _ensure_valid_image_for_qwen(image: Optional[Image.Image], label: str) -> Image.Image:
    """
    Ensure image is safe before resize/model: not None, RGB, positive dimensions, minimum side for Qwen processor.
    Raises ValueError with label on invalid input.
    """
    if image is None:
        raise ValueError(f"{label}: image is None")
    if not isinstance(image, Image.Image):
        raise ValueError(f"{label}: expected PIL.Image, got {type(image).__name__}")
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError(f"{label}: invalid size width={w} height={h}")
    if min(w, h) < _QWEN_MIN_PIXEL_SIDE:
        image = _upscale_to_min_side(image, _QWEN_MIN_PIXEL_SIDE)
    return image


def _resize_image_for_qwen(image: Image.Image, max_long_side: int) -> Image.Image:
    """
    Downscale if max(w,h) > max_long_side (aspect preserved, LANCZOS).
    Does not upscale except when min(w,h) < _QWEN_MIN_PIXEL_SIDE after downscale (thin strips).
    Guarantees min dimension >= _QWEN_MIN_PIXEL_SIDE and both dimensions >= 1 before return.
    """
    if max_long_side < _QWEN_MIN_PIXEL_SIDE:
        raise ValueError(f"max_long_side must be >= {_QWEN_MIN_PIXEL_SIDE}, got {max_long_side}")
    w, h = image.size
    if w < 1 or h < 1:
        raise ValueError(f"_resize_image_for_qwen: invalid size ({w}, {h})")
    long_side = max(w, h)
    if long_side > max_long_side:
        scale = max_long_side / float(long_side)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        if nw < 1 or nh < 1:
            raise ValueError(f"_resize_image_for_qwen: degenerate resize from ({w},{h}) scale={scale}")
        image = image.resize((nw, nh), _LANCZOS)
    w, h = image.size
    if min(w, h) < _QWEN_MIN_PIXEL_SIDE:
        image = _upscale_to_min_side(image, _QWEN_MIN_PIXEL_SIDE)
    w, h = image.size
    if w < 1 or h < 1 or min(w, h) < _QWEN_MIN_PIXEL_SIDE:
        raise ValueError(f"_resize_image_for_qwen: post-condition failed size=({w}, {h})")
    return image


def _aspect_ratio(image: Image.Image) -> float:
    """max(w,h) / min(w,h); >= 1.0 for valid positive dimensions."""
    w, h = image.size
    if w < 1 or h < 1:
        return float("inf")
    return max(w, h) / float(min(w, h))


def _is_extreme_aspect_ratio(image: Image.Image, threshold: float = _QWEN_SAFE_MAX_ASPECT_RATIO) -> bool:
    return _aspect_ratio(image) > threshold


def _pad_image_to_safe_aspect_ratio(image: Image.Image, max_ratio: float = 20.0) -> Image.Image:
    """
    Center image on a padded canvas to cap aspect ratio (max side / min side <= max_ratio).
    Preserves original pixels; unused by default — primary mitigation is heuristic fallback.
    """
    if max_ratio < 1.0:
        raise ValueError(f"max_ratio must be >= 1, got {max_ratio}")
    w, h = image.size
    if w < 1 or h < 1:
        raise ValueError(f"_pad_image_to_safe_aspect_ratio: invalid size ({w}, {h})")
    ratio = max(w, h) / float(min(w, h))
    if ratio <= max_ratio:
        return image.copy()
    if w >= h:
        target_h = max(h, int(math.ceil(w / max_ratio)))
        canvas = Image.new("RGB", (w, target_h), (0, 0, 0))
        y0 = max(0, (target_h - h) // 2)
        canvas.paste(image, (0, y0))
    else:
        target_w = max(w, int(math.ceil(h / max_ratio)))
        canvas = Image.new("RGB", (target_w, h), (0, 0, 0))
        x0 = max(0, (target_w - w) // 2)
        canvas.paste(image, (x0, 0))
    return canvas


def _normalize_importance_level(value: Optional[str]) -> str:
    if not value:
        return "medium"
    s = str(value).lower().strip()
    if s in {"critical", "high", "medium", "low"}:
        return s
    return "medium"


def _functional_type_from_element_role(role: str) -> str:
    r = (role or "").lower().strip()
    if r in {"headline", "subheadline", "legal", "brand_mark", "age_badge", "product_image"}:
        return "functional"
    if r in {"background", "background_panel", "background_shape"}:
        return "background"
    if r in {"decoration", "discount_badge"}:
        return "decorative"
    return "functional"


def _default_candidate_adaptation_policy() -> dict[str, Any]:
    return {
        "preserve_as_unit": True,
        "allow_reflow": False,
        "allow_scale": True,
        "allow_crop": False,
        "allow_shift": True,
        "drop_priority": 0,
        "anchor_type": "free",
    }


def _default_group_adaptation_policy() -> dict[str, Any]:
    return {
        "allow_reflow": True,
        "allow_scale": True,
        "allow_crop": False,
        "allow_shift": True,
        "drop_priority": 0,
        "anchor_type": "free",
    }


def _fallback_candidate_annotation(
    candidate_id: str,
    candidate: dict[str, Any],
    heuristic_annotation: Optional[dict[str, Any]],
    reason: str,
    aspect_detail: str = "",
) -> CandidateAnnotation:
    """Valid CandidateAnnotation without calling the model (extreme-aspect or similar guard)."""
    role: Optional[str] = None
    importance: Optional[str] = None
    if heuristic_annotation:
        role = heuristic_annotation.get("final_role_hint")  # type: ignore[assignment]
        importance = heuristic_annotation.get("final_importance_hint")  # type: ignore[assignment]

    if not role:
        role = candidate.get("role_hint") or candidate.get("candidate_type") or "unknown"
    role = str(role).strip() or "unknown"

    element_role = str(role).lower().strip()
    functional_type = _functional_type_from_element_role(element_role)
    importance_level = _normalize_importance_level(importance if isinstance(importance, str) else None)

    ctype = str(candidate.get("candidate_type") or "").lower()
    is_text: Optional[bool] = None
    if "text" in ctype:
        is_text = True
    elif ctype in {"image", "image_like", "brand", "decoration", "background"}:
        is_text = False

    role_l = element_role.lower()
    is_brand_related = "brand" in role_l or "brand" in ctype
    is_required_for_compliance = "legal" in role_l or "legal" in ctype

    reason_short = reason
    if aspect_detail:
        reason_short = f"{reason_short} ({aspect_detail})"

    display_role = element_role if element_role != "unknown" else str(candidate.get("candidate_type") or "unknown")
    return CandidateAnnotation(
        candidate_id=candidate_id,
        element_role=display_role,
        functional_type=functional_type,
        importance_level=importance_level,
        is_text=is_text,
        is_brand_related=is_brand_related,
        is_required_for_compliance=is_required_for_compliance,
        adaptation_policy=_default_candidate_adaptation_policy(),
        confidence=0.45,
        reason_short=reason_short,
        raw_model_output="",
    )


def _infer_group_role_from_candidate(candidate: dict[str, Any]) -> str:
    ctype = str(candidate.get("candidate_type") or "").lower()
    rh = str(candidate.get("role_hint") or "").lower()
    if "background" in ctype or "background" in rh:
        return "background_group"
    if "brand" in ctype or "brand" in rh:
        return "brand_group"
    if "decoration" in ctype:
        return "decoration_group"
    if "text" in ctype:
        return "text_group"
    if "image" in ctype:
        return "hero_group"
    return "unknown"


def _fallback_group_annotation(
    candidate_id: str,
    candidate: dict[str, Any],
    heuristic_annotation: Optional[dict[str, Any]],
    reason: str,
    aspect_detail: str = "",
) -> GroupAnnotation:
    group_role = _infer_group_role_from_candidate(candidate)
    importance: Optional[str] = None
    if heuristic_annotation:
        gh = heuristic_annotation.get("final_group_hint")
        if isinstance(gh, str) and gh.strip():
            group_role = gh.strip()
        importance = heuristic_annotation.get("final_importance_hint")  # type: ignore[assignment]

    reason_short = reason
    if aspect_detail:
        reason_short = f"{reason_short} ({aspect_detail})"

    return GroupAnnotation(
        candidate_id=candidate_id,
        is_meaningful_group=True,
        group_role=group_role,
        internal_layout="single",
        preserve_as_unit=True,
        importance_level=_normalize_importance_level(importance if isinstance(importance, str) else None),
        adaptation_policy=_default_group_adaptation_policy(),
        confidence=0.45,
        reason_short=reason_short,
        raw_model_output="",
    )


def _parse_bbox_canvas_for_crop(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    """
    Validate candidate['bbox_canvas'], clamp to the unit canvas, and return (x, y, w, h) in normalized coords.
    Raises ValueError with candidate_id context on invalid input (no silent fixes without surfacing issues).
    """
    cid = candidate.get("candidate_id", "<missing candidate_id>")
    cid_repr = repr(cid)

    if "bbox_canvas" not in candidate:
        raise ValueError(
            f"candidate_id={cid_repr}: bbox_canvas is missing; candidate keys={sorted(candidate.keys())!r}"
        )

    raw = candidate["bbox_canvas"]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"candidate_id={cid_repr}: bbox_canvas must be a list or tuple of length 4, got {type(raw).__name__}: {raw!r}"
        )
    if len(raw) != 4:
        raise ValueError(
            f"candidate_id={cid_repr}: bbox_canvas must have length 4, got len={len(raw)} value={raw!r}"
        )

    nums: list[float] = []
    for i, v in enumerate(raw):
        if isinstance(v, bool):
            raise ValueError(
                f"candidate_id={cid_repr}: bbox_canvas[{i}] must be numeric, got bool {v!r}"
            )
        if not isinstance(v, (int, float)):
            raise ValueError(
                f"candidate_id={cid_repr}: bbox_canvas[{i}] must be int or float, got {type(v).__name__}: {v!r}"
            )
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError(
                f"candidate_id={cid_repr}: bbox_canvas[{i}]={v!r} is not finite after float conversion"
            )
        nums.append(fv)

    x, y, w, h = nums
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    w = max(w, _BBOX_MIN_WH)
    h = max(h, _BBOX_MIN_WH)

    x = min(max(0.0, x), 1.0 - w)
    y = min(max(0.0, y), 1.0 - h)

    if w < _BBOX_MIN_WH or h < _BBOX_MIN_WH:
        raise ValueError(
            f"candidate_id={cid_repr}: bbox width/height too small after normalization "
            f"(raw={raw!r}, x={x:.6g} y={y:.6g} w={w:.6g} h={h:.6g}, min_side={_BBOX_MIN_WH})"
        )
    if x + w > 1.0 + 1e-9 or y + h > 1.0 + 1e-9:
        raise ValueError(
            f"candidate_id={cid_repr}: bbox extends outside unit canvas after fit "
            f"(raw={raw!r}, x={x:.6g} y={y:.6g} w={w:.6g} h={h:.6g})"
        )

    return (x, y, w, h)


class QwenRuntime:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:1",
        max_new_tokens: int = 768,
        temperature: float = 0.0,
        use_fp16: bool = True,
        max_image_long_side: int = 1024,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_fp16 = use_fp16
        self.max_image_long_side = max_image_long_side
        self.model = None
        self.processor = None

    def load(self) -> None:
        if AutoProcessor is None or Qwen2_5_VLForConditionalGeneration is None or torch is None:
            raise RuntimeError("transformers/torch are not installed correctly.")

        if self.device.startswith("cuda"):
            device_index = int(self.device.split(":")[1])
            torch.cuda.set_device(device_index)

        dtype = torch.float16 if (self.use_fp16 and self.device.startswith("cuda")) else torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        if self.temperature <= 0 and hasattr(self.model, "generation_config"):
            self.model.generation_config.do_sample = False
            for attr in ("temperature", "top_p", "top_k"):
                if hasattr(self.model.generation_config, attr):
                    setattr(self.model.generation_config, attr, None)

    def ensure_loaded(self) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen runtime not loaded.")

    def _load_image(self, image_path: str) -> Image.Image:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        return Image.open(path).convert("RGB")

    def _prepare_image_for_model(self, image: Image.Image, label: str) -> Image.Image:
        """Validate, RGB, cap long side to 1024, then JPEG-recompress for smaller VLM payloads."""
        img = _ensure_valid_image_for_qwen(image, label)
        max_long_side = min(int(self.max_image_long_side), 1024)
        img = _resize_image_for_qwen(img, max_long_side)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82, optimize=True)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def _crop_candidate(self, image: Image.Image, candidate: dict[str, Any], expand_ratio: float = 0.0) -> Image.Image:
        width, height = image.size
        cid_repr = repr(candidate.get("candidate_id", "<missing candidate_id>"))
        if width < 1 or height < 1:
            raise ValueError(
                f"candidate_id={cid_repr}: source banner has invalid size width={width} height={height}"
            )

        x_norm, y_norm, w_norm, h_norm = _parse_bbox_canvas_for_crop(candidate)
        x = x_norm * width
        y = y_norm * height
        w = w_norm * width
        h = h_norm * height
        pad_x = w * expand_ratio
        pad_y = h * expand_ratio

        left = max(0, min(int(round(x - pad_x)), width - 1))
        top = max(0, min(int(round(y - pad_y)), height - 1))
        right = max(left + 1, min(int(round(x + w + pad_x)), width))
        bottom = max(top + 1, min(int(round(y + h + pad_y)), height))

        cropped = image.crop((left, top, right, bottom))
        cw, ch = cropped.size
        if cw < 1 or ch < 1:
            raise ValueError(
                f"candidate_id={cid_repr}: crop has invalid pixel size ({cw}, {ch}) "
                f"box_px=({left},{top},{right},{bottom}) banner=({width},{height}) "
                f"bbox_norm=({x_norm:.8g},{y_norm:.8g},{w_norm:.8g},{h_norm:.8g}) expand_ratio={expand_ratio}"
            )
        return cropped

    def _run_model(self, images: list[Image.Image], prompt: str, *, max_new_tokens: Optional[int] = None) -> str:
        self.ensure_loaded()
        for i, img in enumerate(images):
            if img is None:
                raise ValueError(f"_run_model: images[{i}] is None")
            w, h = img.size
            if w < 1 or h < 1:
                raise ValueError(f"_run_model: images[{i}] invalid size ({w}, {h})")
            if min(w, h) < _QWEN_MIN_PIXEL_SIDE:
                raise ValueError(
                    f"_run_model: images[{i}] min side {min(w, h)} < {_QWEN_MIN_PIXEL_SIDE} (should have been prepared)"
                )
        mtokens = int(max_new_tokens) if max_new_tokens is not None else int(self.max_new_tokens)
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": mtokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
        output_text = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output_text.strip()

    def _extract_json(self, text: str) -> Any:
        text = text.strip()
        parsed = self._json_load_relaxed(text)
        if parsed is not None:
            return parsed
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        for block in fenced:
            parsed = self._json_load_relaxed(block.strip())
            if parsed is not None:
                return parsed
        obj_match = self._find_balanced_json_substring(text, "{", "}")
        if obj_match is not None:
            parsed = self._json_load_relaxed(obj_match)
            if parsed is not None:
                return parsed
        arr_match = self._find_balanced_json_substring(text, "[", "]")
        if arr_match is not None:
            parsed = self._json_load_relaxed(arr_match)
            if parsed is not None:
                return parsed
        return None

    def _json_load_relaxed(self, text: str) -> Any:
        candidate = (text or "").strip()
        if not candidate:
            return None
        try:
            return json.loads(candidate)
        except Exception:
            pass

        repaired = candidate.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        repaired = repaired.replace("\x00", "")
        try:
            return json.loads(repaired)
        except Exception:
            pass

        try:
            value = ast.literal_eval(repaired)
            if isinstance(value, (dict, list)):
                return value
        except Exception:
            pass
        return None

    def _find_balanced_json_substring(self, text: str, open_char: str, close_char: str) -> Optional[str]:
        start = text.find(open_char)
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    def _build_banner_prompt(self, candidate_bundle: dict[str, Any], heuristic_bundle: Optional[dict[str, Any]]) -> str:
        heuristic_summary = {}
        if heuristic_bundle is not None:
            heuristic_summary = {
                "headline_candidates": heuristic_bundle.get("headline_candidates", [])[:5],
                "subheadline_candidates": heuristic_bundle.get("subheadline_candidates", [])[:5],
                "legal_candidates": heuristic_bundle.get("legal_candidates", [])[:5],
                "badge_candidates": heuristic_bundle.get("badge_candidates", [])[:5],
                "brand_candidates": heuristic_bundle.get("brand_candidates", [])[:5],
                "background_candidates": heuristic_bundle.get("background_candidates", [])[:5],
            }
        return f"""
You are a visual layout annotation assistant.

Task:
Classify the overall layout pattern and major semantic zones of this ad banner.

Allowed layout patterns:
left_text_right_image,
left_text_right_product,
price_left_product_right,
centered_text_decorative_background,
full_background_text_overlay,
top_image_bottom_text_mobile,
top_text_bottom_product_mobile,
product_dominant_mobile,
promo_text_only,
catalog_price_card,
unknown

Allowed zone roles:
text_zone, image_zone, product_zone, legal_zone, promo_zone,
background_zone, overlay_zone, brand_zone

Candidate summary JSON:
{json.dumps(candidate_bundle, ensure_ascii=False, indent=2)}

Heuristic summary JSON:
{json.dumps(heuristic_summary, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "layout_pattern": "...",
  "pattern_confidence": 0.0,
  "zones": [
    {{
      "zone_role": "...",
      "description": "...",
      "approx_position": "top|bottom|left|right|center|full",
      "importance_level": "critical|high|medium|low"
    }}
  ],
  "preservation_priorities": [
    {{"role": "logo", "priority": 1}},
    {{"role": "headline", "priority": 1}},
    {{"role": "legal", "priority": 1}}
  ],
  "reason_short": "..."
}}
""".strip()

    def _build_element_prompt(self, candidate: dict[str, Any], heuristic_annotation: Optional[dict[str, Any]]) -> str:
        return f"""
You are a visual layout annotation assistant.

Task:
Classify the semantic role and adaptation behavior of ONE candidate element from an ad banner.

Use the full banner for context, but classify only the candidate element.

Allowed element roles:
headline, subheadline, body_text, legal, price_main, price_old, price_fraction,
discount_text, cta, label, badge_text, logo_text,
product_image, hero_photo, logo_icon, brand_mark, background_shape,
background_panel, decoration, discount_badge, age_badge, packshot,
text_container, image_container, promo_container, unknown

Allowed functional_type:
functional, decorative, background

Allowed importance_level:
critical, high, medium, low

Allowed anchor_type:
top_left, top_center, top_right, left_center, center, right_center,
bottom_left, bottom_center, bottom_right, free

Candidate summary JSON:
{json.dumps(candidate, ensure_ascii=False, indent=2)}

Heuristic summary JSON:
{json.dumps(heuristic_annotation, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "element_role": "...",
  "functional_type": "...",
  "importance_level": "...",
  "is_text": true,
  "is_brand_related": false,
  "is_required_for_compliance": false,
  "adaptation_policy": {{
    "preserve_as_unit": true,
    "allow_reflow": false,
    "allow_scale": true,
    "allow_crop": false,
    "allow_shift": true,
    "drop_priority": 0,
    "anchor_type": "..."
  }},
  "confidence": 0.0,
  "reason_short": "..."
}}
""".strip()

    def _build_group_prompt(self, candidate: dict[str, Any], heuristic_annotation: Optional[dict[str, Any]]) -> str:
        return f"""
You are a visual layout annotation assistant.

Task:
Determine whether a set of elements forms a meaningful semantic group in an ad banner,
and classify the role of that group.

Allowed group roles:
brand_group, headline_group, price_group, cta_group, product_group,
legal_group, badge_group, decoration_group, text_group, hero_group,
background_group, unknown

Allowed internal_layout:
vertical_stack, horizontal_row, overlay, freeform, single

Candidate summary JSON:
{json.dumps(candidate, ensure_ascii=False, indent=2)}

Heuristic summary JSON:
{json.dumps(heuristic_annotation, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "is_meaningful_group": true,
  "group_role": "...",
  "internal_layout": "...",
  "preserve_as_unit": true,
  "importance_level": "critical",
  "adaptation_policy": {{
    "allow_reflow": true,
    "allow_scale": true,
    "allow_crop": false,
    "allow_shift": true,
    "drop_priority": 0,
    "anchor_type": "top_left"
  }},
  "confidence": 0.0,
  "reason_short": "..."
}}
""".strip()

    def _build_brand_context_prompt(
        self,
        candidate_bundle: Optional[dict[str, Any]],
        heuristic_bundle: Optional[dict[str, Any]],
    ) -> str:
        compact_candidates = _compact_candidate_bundle_for_brand_context(candidate_bundle)
        compact_heuristics = _compact_heuristic_bundle_for_brand_context(heuristic_bundle)
        return f"""
You are a brand and market-context analyst for digital advertising banners exported from Figma.

Task:
From the banner IMAGE plus the structured summaries below, infer likely brand context.
Be conservative: if you are not reasonably confident about a specific retail brand name, use brand_family "generic".
Do not invent famous brand names from weak evidence.

Output rules:
- brand_family: machine-friendly snake_case string (a-z, 0-9, underscore). Examples: yandex_lavka, generic, unknown_brand, nike, coca_cola
- brand_confidence: number between 0 and 1 (how confident you are in brand_family specifically)
- language: one of ru, en, kk, mixed, unknown (or a short ISO-like 2-3 letter code if obvious)
- category: one of grocery, beverage, fashion, electronics, delivery, unknown, or another short snake_case category
- reason_short: one or two sentences explaining the inference

If the brand is unclear or could be many retailers: brand_family MUST be "generic" and brand_confidence <= 0.4.
If language is unclear: language MUST be "unknown".
If category is unclear: category MUST be "unknown".

Compact candidate summary JSON:
{json.dumps(compact_candidates, ensure_ascii=False, indent=2)}

Compact heuristic summary JSON:
{json.dumps(compact_heuristics, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "brand_family": "...",
  "brand_confidence": 0.0,
  "language": "...",
  "category": "...",
  "reason_short": "..."
}}
""".strip()

    def annotate_brand_context(self, request: BrandContextAnnotateRequest) -> BrandContextAnnotation:
        image = self._load_image(request.banner_image_path)
        image = self._prepare_image_for_model(image, "annotate_brand_context/banner")

        ar = _aspect_ratio(image)
        logger.info(
            "annotate_brand_context preprocessing: banner=%dx%d aspect=%.2f threshold=%.0f",
            image.width,
            image.height,
            ar,
            _QWEN_SAFE_MAX_ASPECT_RATIO,
        )
        if _is_extreme_aspect_ratio(image):
            logger.warning(
                "annotate_brand_context using fallback (extreme banner aspect ratio) aspect=%.2f",
                ar,
            )
            return BrandContextAnnotation(
                brand_family="generic",
                brand_confidence=0.25,
                language="unknown",
                category="unknown",
                reason_short="Fallback: banner aspect ratio exceeded safe Qwen threshold for vision encoding.",
                raw_model_output="",
            )

        prompt = self._build_brand_context_prompt(request.candidate_bundle, request.heuristic_bundle)
        logger.info("annotate_brand_context calling model (single banner image)")
        raw_output = self._run_model([image], prompt)
        parsed = self._extract_json(raw_output)
        out = _finalize_brand_context_output(parsed if isinstance(parsed, dict) else None, raw_output)
        logger.info(
            "annotate_brand_context result: brand_family=%r confidence=%.2f language=%r category=%r",
            out.brand_family,
            out.brand_confidence,
            out.language,
            out.category,
        )
        return out

    def annotate_banner(self, request: BannerAnnotateRequest) -> BannerAnnotation:
        image = self._load_image(request.banner_image_path)
        image = self._prepare_image_for_model(image, "annotate_banner/banner")
        logger.info(
            "annotate_banner model input: banner=%dx%d",
            image.width,
            image.height,
        )
        prompt = self._build_banner_prompt(request.candidate_bundle, request.heuristic_bundle)
        raw_output = self._run_model([image], prompt)
        data = self._extract_json(raw_output)
        if not isinstance(data, dict):
            return BannerAnnotation(raw_model_output=raw_output, reason_short="Failed to parse banner-level JSON output.")
        return BannerAnnotation(
            layout_pattern=str(data.get("layout_pattern", "unknown")),
            pattern_confidence=float(data.get("pattern_confidence", 0.0) or 0.0),
            zones=list(data.get("zones", []) or []),
            preservation_priorities=list(data.get("preservation_priorities", []) or []),
            reason_short=str(data.get("reason_short", "")),
            raw_model_output=raw_output,
        )

    def annotate_candidate(self, request: CandidateAnnotateRequest) -> CandidateAnnotation:
        candidate_id = str(request.candidate.get("candidate_id", "unknown_candidate"))
        banner_image = self._load_image(request.banner_image_path)
        tight_crop = self._crop_candidate(banner_image, request.candidate, expand_ratio=0.0)
        context_crop = self._crop_candidate(banner_image, request.candidate, expand_ratio=request.context_padding_ratio)

        bw, bh = banner_image.size
        rtw, rth = tight_crop.size
        rcw, rch = context_crop.size

        banner_for_model = self._prepare_image_for_model(
            banner_image,
            f"annotate_candidate/{candidate_id}/banner",
        )
        tight_for_model = self._prepare_image_for_model(
            tight_crop,
            f"annotate_candidate/{candidate_id}/tight_crop",
        )
        context_for_model = self._prepare_image_for_model(
            context_crop,
            f"annotate_candidate/{candidate_id}/context_crop",
        )

        ar_banner = _aspect_ratio(banner_for_model)
        ar_tight = _aspect_ratio(tight_for_model)
        ar_context = _aspect_ratio(context_for_model)

        logger.info(
            "annotate_candidate preprocessing candidate_id=%r: raw_banner=%dx%d raw_tight=%dx%d raw_context=%dx%d "
            "prepared_aspect banner=%.2f tight=%.2f context=%.2f threshold=%.0f",
            candidate_id,
            bw,
            bh,
            rtw,
            rth,
            rcw,
            rch,
            ar_banner,
            ar_tight,
            ar_context,
            _QWEN_SAFE_MAX_ASPECT_RATIO,
        )

        fb_reason = (
            "Fallback annotation used because candidate crop aspect ratio exceeded safe Qwen threshold."
        )
        if (
            _is_extreme_aspect_ratio(banner_for_model)
            or _is_extreme_aspect_ratio(tight_for_model)
            or _is_extreme_aspect_ratio(context_for_model)
        ):
            aspect_detail = (
                f"prepared_banner={banner_for_model.width}x{banner_for_model.height} ar={ar_banner:.2f}; "
                f"prepared_tight={tight_for_model.width}x{tight_for_model.height} ar={ar_tight:.2f}; "
                f"prepared_context={context_for_model.width}x{context_for_model.height} ar={ar_context:.2f}"
            )
            logger.warning(
                "annotate_candidate using fallback (extreme aspect ratio) candidate_id=%r "
                "ar_banner=%.2f ar_tight=%.2f ar_context=%.2f threshold=%.0f",
                candidate_id,
                ar_banner,
                ar_tight,
                ar_context,
                _QWEN_SAFE_MAX_ASPECT_RATIO,
            )
            return _fallback_candidate_annotation(
                candidate_id,
                request.candidate,
                request.heuristic_annotation,
                fb_reason,
                aspect_detail=aspect_detail,
            )

        logger.info(
            "annotate_candidate model input candidate_id=%r: banner=%dx%d tight=%dx%d context=%dx%d",
            candidate_id,
            banner_for_model.width,
            banner_for_model.height,
            tight_for_model.width,
            tight_for_model.height,
            context_for_model.width,
            context_for_model.height,
        )

        prompt = self._build_element_prompt(request.candidate, request.heuristic_annotation)
        raw_output = self._run_model([banner_for_model, tight_for_model, context_for_model], prompt)
        data = self._extract_json(raw_output)
        if not isinstance(data, dict):
            return CandidateAnnotation(candidate_id=candidate_id, raw_model_output=raw_output, reason_short="Failed to parse candidate-level JSON output.")
        return CandidateAnnotation(
            candidate_id=candidate_id,
            element_role=str(data.get("element_role", "unknown")),
            functional_type=str(data.get("functional_type", "functional")),
            importance_level=str(data.get("importance_level", "medium")),
            is_text=data.get("is_text", None),
            is_brand_related=bool(data.get("is_brand_related", False)),
            is_required_for_compliance=bool(data.get("is_required_for_compliance", False)),
            adaptation_policy=dict(data.get("adaptation_policy", {}) or {}),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            reason_short=str(data.get("reason_short", "")),
            raw_model_output=raw_output,
        )

    def annotate_group(self, request: GroupAnnotateRequest) -> GroupAnnotation:
        candidate_id = str(request.candidate.get("candidate_id", "unknown_candidate"))
        banner_image = self._load_image(request.banner_image_path)
        group_crop = self._crop_candidate(banner_image, request.candidate, expand_ratio=request.context_padding_ratio)

        bw, bh = banner_image.size
        gw, gh = group_crop.size

        banner_for_model = self._prepare_image_for_model(
            banner_image,
            f"annotate_group/{candidate_id}/banner",
        )
        group_for_model = self._prepare_image_for_model(
            group_crop,
            f"annotate_group/{candidate_id}/group_crop",
        )

        ar_banner = _aspect_ratio(banner_for_model)
        ar_group = _aspect_ratio(group_for_model)

        logger.info(
            "annotate_group preprocessing candidate_id=%r: raw_banner=%dx%d raw_group_crop=%dx%d "
            "prepared_aspect banner=%.2f group_crop=%.2f threshold=%.0f",
            candidate_id,
            bw,
            bh,
            gw,
            gh,
            ar_banner,
            ar_group,
            _QWEN_SAFE_MAX_ASPECT_RATIO,
        )

        fb_reason = (
            "Fallback group annotation used because crop aspect ratio exceeded safe Qwen threshold."
        )
        if _is_extreme_aspect_ratio(banner_for_model) or _is_extreme_aspect_ratio(group_for_model):
            aspect_detail = (
                f"prepared_banner={banner_for_model.width}x{banner_for_model.height} ar={ar_banner:.2f}; "
                f"prepared_group={group_for_model.width}x{group_for_model.height} ar={ar_group:.2f}"
            )
            logger.warning(
                "annotate_group using fallback (extreme aspect ratio) candidate_id=%r ar_banner=%.2f ar_group=%.2f threshold=%.0f",
                candidate_id,
                ar_banner,
                ar_group,
                _QWEN_SAFE_MAX_ASPECT_RATIO,
            )
            return _fallback_group_annotation(
                candidate_id,
                request.candidate,
                request.heuristic_annotation,
                fb_reason,
                aspect_detail=aspect_detail,
            )

        logger.info(
            "annotate_group model input candidate_id=%r: banner=%dx%d group_crop=%dx%d",
            candidate_id,
            banner_for_model.width,
            banner_for_model.height,
            group_for_model.width,
            group_for_model.height,
        )

        prompt = self._build_group_prompt(request.candidate, request.heuristic_annotation)
        raw_output = self._run_model([banner_for_model, group_for_model], prompt)
        data = self._extract_json(raw_output)
        if not isinstance(data, dict):
            return GroupAnnotation(candidate_id=candidate_id, raw_model_output=raw_output, reason_short="Failed to parse group-level JSON output.")
        return GroupAnnotation(
            candidate_id=candidate_id,
            is_meaningful_group=bool(data.get("is_meaningful_group", True)),
            group_role=str(data.get("group_role", "unknown")),
            internal_layout=str(data.get("internal_layout", "freeform")),
            preserve_as_unit=bool(data.get("preserve_as_unit", True)),
            importance_level=str(data.get("importance_level", "medium")),
            adaptation_policy=dict(data.get("adaptation_policy", {}) or {}),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            reason_short=str(data.get("reason_short", "")),
            raw_model_output=raw_output,
        )

    def _build_semantic_structure_prompt(self, figma_summary: dict[str, Any]) -> str:
        elem_ids = figma_summary.get("element_annotation_candidate_ids") or []
        grp_ids = figma_summary.get("group_annotation_candidate_ids") or []
        summary_json = json.dumps(figma_summary, ensure_ascii=False, indent=2)
        return f"""
You are a senior layout + semantics analyst for retail / marketplace ad banners.

You receive:
1) The full banner image.
2) A compact JSON summary built from structured Figma export data (hierarchy, ids, names, types, text,
   geometry, parent/child ids, plus semantic CANDIDATES and heuristic hints).

Rules (critical):
- Treat the Figma summary as the source of truth for structure. Do NOT invent node ids or candidate_ids.
- Only emit keys in "element_annotations" for candidate_ids listed in element_annotation_candidate_ids.
- Only emit keys in "group_annotations" for candidate_ids listed in group_annotation_candidate_ids.
- If uncertain about a candidate, keep the heuristic_role as element_role or use "unknown".
- Prefer fewer, high-confidence annotations; do not label every decoration as functional.
- Do not over-label large background shapes as headline/CTA.

Allowed element_role values:
headline, subheadline, body_text, legal, price_main, price_old, price_fraction,
discount_text, cta, label, badge_text, logo_text,
product_image, hero_photo, logo_icon, brand_mark, background_shape,
background_panel, decoration, discount_badge, age_badge, packshot,
text_container, image_container, promo_container, unknown

Allowed functional_type: functional, decorative, background
Allowed importance_level: critical, high, medium, low

Allowed group_role values:
brand_group, headline_group, price_group, cta_group, product_group,
legal_group, badge_group, decoration_group, text_group, hero_group,
background_group, unknown

Allowed internal_layout: vertical_stack, horizontal_row, overlay, freeform, single

Allowed layout_pattern values:
left_text_right_image,
left_text_right_product,
price_left_product_right,
centered_text_decorative_background,
full_background_text_overlay,
top_image_bottom_text_mobile,
top_text_bottom_product_mobile,
product_dominant_mobile,
promo_text_only,
catalog_price_card,
unknown

Figma summary JSON:
{summary_json}

element_annotation_candidate_ids (must only use these keys in element_annotations):
{json.dumps(elem_ids, ensure_ascii=False)}

group_annotation_candidate_ids (must only use these keys in group_annotations):
{json.dumps(grp_ids, ensure_ascii=False)}

Return JSON only (no markdown fences):
{{
  "brand_family": "snake_case_or_generic",
  "brand_confidence": 0.0,
  "language": "ru|en|kk|mixed|unknown",
  "category": "snake_case_category_or_unknown",
  "layout_pattern": "...",
  "pattern_confidence": 0.0,
  "zones": [
    {{
      "zone_role": "text_zone|image_zone|product_zone|legal_zone|promo_zone|background_zone|overlay_zone|brand_zone",
      "description": "short",
      "approx_position": "top|bottom|left|right|center|full",
      "importance_level": "critical|high|medium|low"
    }}
  ],
  "element_annotations": {{
    "CANDIDATE_ID": {{
      "element_role": "...",
      "semantic_name": "",
      "parent_semantic_name": "",
      "functional_type": "functional|decorative|background",
      "importance_level": "critical|high|medium|low",
      "is_text": true,
      "is_brand_related": false,
      "is_required_for_compliance": false,
      "adaptation_policy": {{
        "preserve_as_unit": true,
        "allow_reflow": false,
        "allow_scale": true,
        "allow_crop": false,
        "allow_shift": true,
        "drop_priority": 0,
        "anchor_type": "free"
      }},
      "confidence": 0.0,
      "reason_short": "one short sentence"
    }}
  }},
  "group_annotations": {{
    "CANDIDATE_ID": {{
      "group_role": "...",
      "semantic_name": "",
      "parent_semantic_name": "",
      "internal_layout": "vertical_stack|horizontal_row|overlay|freeform|single",
      "preserve_as_unit": true,
      "importance_level": "medium",
      "adaptation_policy": {{
        "allow_reflow": true,
        "allow_scale": true,
        "allow_crop": false,
        "allow_shift": true,
        "drop_priority": 0,
        "anchor_type": "free"
      }},
      "confidence": 0.0,
      "reason_short": "one short sentence"
    }}
  }},
  "updates": [
    {{
      "source_figma_id": "...",
      "path": "...",
      "role": "...",
      "semantic_name": "...",
      "parent_semantic_name": "",
      "confidence": 0.0,
      "reason": ""
    }}
  ],
  "groups": [
    {{
      "id": "...",
      "role": "...",
      "semantic_name": "...",
      "children": ["..."],
      "confidence": 0.0,
      "reason": ""
    }}
  ],
  "preservation_priorities": [
    {{"role": "logo", "priority": 1}}
  ],
  "reason_short": "overall note"
}}
""".strip()

    def _semantic_structure_heuristic_fallback(self, figma_summary: dict[str, Any]) -> dict[str, Any]:
        """No VLM / parse recovery: reuse pipeline heuristics via existing fallback builders."""
        elem_ann: dict[str, Any] = {}
        grp_ann: dict[str, Any] = {}
        cands = figma_summary.get("candidates") or []
        cid_to_row: dict[str, dict[str, Any]] = {}
        for row in cands:
            if isinstance(row, dict) and row.get("candidate_id"):
                cid_to_row[str(row["candidate_id"])] = row

        for cid in figma_summary.get("element_annotation_candidate_ids") or []:
            cid_s = str(cid)
            row = cid_to_row.get(cid_s, {})
            cand_dict: dict[str, Any] = {
                "candidate_id": cid_s,
                "candidate_type": str(row.get("candidate_type") or "unknown"),
                "role_hint": row.get("heuristic_role"),
                "bbox_canvas": row.get("bbox_canvas") or [0.0, 0.0, 0.1, 0.1],
            }
            ha: dict[str, Any] | None = None
            if row.get("heuristic_role") is not None or row.get("importance") is not None:
                ha = {
                    "final_role_hint": row.get("heuristic_role"),
                    "final_importance_hint": row.get("importance"),
                }
            ca = _fallback_candidate_annotation(
                cid_s,
                cand_dict,
                ha,
                "Heuristic fallback (single-pass semantic_structure).",
            )
            elem_ann[cid_s] = {
                "element_role": ca.element_role,
                "functional_type": ca.functional_type,
                "importance_level": ca.importance_level,
                "is_text": ca.is_text,
                "is_brand_related": ca.is_brand_related,
                "is_required_for_compliance": ca.is_required_for_compliance,
                "adaptation_policy": ca.adaptation_policy,
                "confidence": ca.confidence,
                "reason_short": ca.reason_short,
            }

        for cid in figma_summary.get("group_annotation_candidate_ids") or []:
            cid_s = str(cid)
            row = cid_to_row.get(cid_s, {})
            cand_dict = {
                "candidate_id": cid_s,
                "candidate_type": str(row.get("candidate_type") or "unknown"),
                "role_hint": row.get("heuristic_role"),
                "bbox_canvas": row.get("bbox_canvas") or [0.0, 0.0, 0.1, 0.1],
            }
            ha = None
            if row.get("heuristic_group") is not None or row.get("importance") is not None:
                ha = {
                    "final_group_hint": row.get("heuristic_group"),
                    "final_importance_hint": row.get("importance"),
                }
            ga = _fallback_group_annotation(
                cid_s,
                cand_dict,
                ha,
                "Heuristic fallback (single-pass semantic_structure).",
            )
            grp_ann[cid_s] = {
                "group_role": ga.group_role,
                "internal_layout": ga.internal_layout,
                "preserve_as_unit": ga.preserve_as_unit,
                "importance_level": ga.importance_level,
                "adaptation_policy": ga.adaptation_policy,
                "confidence": ga.confidence,
                "reason_short": ga.reason_short,
            }

        return {
            "brand_family": "generic",
            "brand_confidence": 0.25,
            "language": "unknown",
            "category": "unknown",
            "layout_pattern": "unknown",
            "pattern_confidence": 0.0,
            "zones": [],
            "element_annotations": elem_ann,
            "group_annotations": grp_ann,
            "updates": [],
            "groups": [],
            "preservation_priorities": [],
            "reason_short": "Heuristic fallback without VLM output.",
            "raw_model_output": "",
        }

    def _finalize_semantic_structure_response(
        self,
        data: dict[str, Any],
        figma_summary: dict[str, Any],
        raw_output: str,
    ) -> dict[str, Any]:
        elem_allowed = set(str(x) for x in (figma_summary.get("element_annotation_candidate_ids") or []))
        grp_allowed = set(str(x) for x in (figma_summary.get("group_annotation_candidate_ids") or []))

        elem_in = data.get("element_annotations") if isinstance(data.get("element_annotations"), dict) else {}
        elem_out: dict[str, Any] = {}
        for k, v in elem_in.items():
            ks = str(k)
            if ks not in elem_allowed or not isinstance(v, dict):
                continue
            elem_out[ks] = {
                "element_role": str(v.get("element_role", "unknown")).lower().strip() or "unknown",
                "semantic_name": str(v.get("semantic_name", "") or "").strip(),
                "parent_semantic_name": str(v.get("parent_semantic_name", "") or "").strip(),
                "functional_type": str(v.get("functional_type", "functional")).lower().strip() or "functional",
                "importance_level": _normalize_importance_level(str(v.get("importance_level", ""))),
                "is_text": v.get("is_text", None),
                "is_brand_related": bool(v.get("is_brand_related", False)),
                "is_required_for_compliance": bool(v.get("is_required_for_compliance", False)),
                "adaptation_policy": dict(v.get("adaptation_policy") or {}) or _default_candidate_adaptation_policy(),
                "confidence": float(v.get("confidence", 0.0) or 0.0),
                "reason_short": str(v.get("reason_short", "")),
            }

        grp_in = data.get("group_annotations") if isinstance(data.get("group_annotations"), dict) else {}
        grp_out: dict[str, Any] = {}
        for k, v in grp_in.items():
            ks = str(k)
            if ks not in grp_allowed or not isinstance(v, dict):
                continue
            grp_out[ks] = {
                "group_role": str(v.get("group_role", "unknown")).lower().strip() or "unknown",
                "semantic_name": str(v.get("semantic_name", "") or "").strip(),
                "parent_semantic_name": str(v.get("parent_semantic_name", "") or "").strip(),
                "internal_layout": str(v.get("internal_layout", "freeform")).lower().strip() or "freeform",
                "preserve_as_unit": bool(v.get("preserve_as_unit", True)),
                "importance_level": _normalize_importance_level(str(v.get("importance_level", ""))),
                "adaptation_policy": dict(v.get("adaptation_policy") or {}) or _default_group_adaptation_policy(),
                "confidence": float(v.get("confidence", 0.0) or 0.0),
                "reason_short": str(v.get("reason_short", "")),
            }

        brand_slice = {
            "brand_family": data.get("brand_family", "generic"),
            "brand_confidence": float(data.get("brand_confidence", data.get("pattern_confidence", 0.0)) or 0.0),
            "language": data.get("language", "unknown"),
            "category": data.get("category", "unknown"),
            "reason_short": str(data.get("reason_short", "") or "single-pass semantic_structure"),
        }
        bc = _finalize_brand_context_output(brand_slice, raw_output)

        return {
            "brand_family": bc.brand_family,
            "brand_confidence": bc.brand_confidence,
            "language": bc.language,
            "category": bc.category,
            "layout_pattern": str(data.get("layout_pattern", "unknown")),
            "pattern_confidence": float(data.get("pattern_confidence", 0.0) or 0.0),
            "zones": list(data.get("zones", []) or []),
            "element_annotations": elem_out,
            "group_annotations": grp_out,
            "updates": list(data.get("updates", []) or []),
            "groups": list(data.get("groups", []) or []),
            "preservation_priorities": list(data.get("preservation_priorities", []) or []),
            "reason_short": str(data.get("reason_short", "") or bc.reason_short),
            "raw_model_output": raw_output,
        }

    def _build_scene_prompt(self, request: SceneAnnotateRequest) -> str:
        figma_summary = request.figma_summary or {}
        node_refs = []
        for node in (figma_summary.get("nodes") or [])[:120]:
            if not isinstance(node, dict):
                continue
            ref: dict[str, Any] = {
                "id": node.get("id"),
                "path": node.get("path"),
                "name": node.get("name"),
                "name_is_generic": node.get("name_is_generic"),
                "type": node.get("type"),
                "text": node.get("text"),
                "bbox_canvas": node.get("bbox_canvas"),
                "parent_id": node.get("parent_id"),
                "children_ids": node.get("children_ids"),
                "visual_hints": node.get("visual_hints"),
            }
            ar = node.get("atlas_region")
            if isinstance(ar, dict):
                ref["atlas_region"] = ar
            node_refs.append(ref)
        candidate_refs = [
            {
                "candidate_id": cand.get("candidate_id"),
                "candidate_type": cand.get("candidate_type"),
                "source_node_ids": cand.get("source_node_ids"),
                "bbox_canvas": cand.get("bbox_canvas"),
                "text_content": cand.get("text_content"),
                "heuristic_role": cand.get("heuristic_role"),
                "heuristic_group": cand.get("heuristic_group"),
                "importance": cand.get("importance"),
                "grouping_reason": cand.get("grouping_reason"),
                "member_count": cand.get("member_count"),
            }
            for cand in (figma_summary.get("candidates") or [])[:80]
            if isinstance(cand, dict)
        ]
        atlas_ps = (request.element_atlas_image_path or "").strip()
        n_crops = len(request.element_image_paths or [])
        n_after_banner = (1 if atlas_ps else 0) + n_crops
        extra_image_rules = ""
        if n_after_banner:
            if atlas_ps and n_crops:
                last_idx = 2 + n_crops
                extra_image_rules = f"""
Image inputs (order matters; same design):
- Image 1: full banner (canonical layout on the frame canvas; normalized bbox in JSON).
- Image 2: **element atlas** — one PNG packing rasterized leaf layers in rows with gaps between cells. Use it to see each leaf’s pixels at export resolution. In the compact node refs JSON, `atlas_region` on a node (when present) is {{x, y, width, height}} in **atlas pixel space** for that node’s `path`; align regions with image 2.
- Images 3 through {last_idx}: optional **per-leaf crops** (same leaves, higher per-tile detail). Prefer correlating by `path` / `id` with the JSON and atlas.
- Treat Figma `id`, `path`, and `bbox_canvas` as source of truth for geometry; do not invent ids from pixels alone.
"""
            elif atlas_ps:
                extra_image_rules = """
Image inputs (order matters; same design):
- Image 1: full banner (canonical layout on the frame canvas; normalized bbox in JSON).
- Image 2: **element atlas** — one PNG packing rasterized leaf layers in rows with gaps. In the compact node refs JSON, `atlas_region` (when present) is {x, y, width, height} in **atlas pixel space** for that node’s `path`; use it to read each leaf’s pixels in image 2.
- Treat Figma `id`, `path`, and `bbox_canvas` as source of truth; use the atlas for visual detail and disambiguation.
"""
            else:
                extra_image_rules = f"""
Image inputs (order matters):
- Image 1 is the full banner (canonical layout and composition).
- Images 2 through {1 + n_crops} are PNG crops of individual layers or regions from the same design. Use them with image 1 for fine detail. Still treat the Figma JSON ids and geometry as source of truth; do not invent node ids from crops alone.
"""

        return f"""
You are a scene-level semantic correction engine for a Figma banner adaptation backend.

Critical rules:
- Analyze ONLY the current design. Do not copy names from previous examples.
- Use the Figma-derived node tree and ids as source of truth.
- **Never** assign `decoration_sparkle`, `decoration_star`, `decoration_glow`, or other `decoration_*` **element** names to: TEXT nodes, nodes with visible copy (`text` in JSON), headline/subheadline/legal clusters, brand or logo parts, age badges, or price lines. Those need roles like `headline`, `subheadline`, `legal_text`, `brand_name_*`, `logo`, `age_badge`, `price_group`, `star_straight` / `star_rotated` for actual star shapes. Reserve `decoration_sparkle` for small non-text sparkles only.
- **Group** `semantic_name` for `headline_group` / `brand_group` / `legal_group` / `badge_group` must stay `headline_group`, `brand_group`, `legal_group`, `badge_group` (or similar)—not `decoration_sparkle`.
- Do NOT invent candidate ids, figma ids, paths, or child ids.
- Return ONE valid JSON object only. No markdown, no prose, no comments.
- The response must start with `{{` and end with `}}`.
- If unsure, keep heuristic semantics or return empty strings/empty arrays.
- Classify 0+, 3+, 6+, 12+, 16+, 18+ as age_badge, never as price_group.
- Only use price_group when text clearly contains real pricing markers like ₽, $, €, руб, %, скид, от, numeric price formats.
- If multiple nearby stars/sparkles belong to one decorative family, use one decoration_group.
- For a decoration group candidate whose `grouping_reason` is `star_family_global` (or all `source_node_ids` are star shapes), every Figma `source_node_id` must appear exactly once in `updates` with a distinct `semantic_name` from the banner image: use `star_straight` or `star_rotated` for a single star. If two or more stars share the same orientation (all upright vs all tilted), use `star_straight_1`, `star_straight_2`, … or `star_rotated_1`, `star_rotated_2`, … Use mixed forms (`star_straight`, `star_rotated`) when orientations differ. Do not emit separate decoration groups per star; keep one `group_annotations` entry for that decoration candidate with `semantic_name` `decoration_group`.
- Brand/logo naming must adapt to the current design: if the mark is not M-shaped, do not call it logo_m. Use logo_heart, logo_leaf, or logo_mark when appropriate.
- For fragmented brand text like Я + ндекс or Ya + ndex, name parts adaptively using brand_name_yandex_part, brand_name_market_part, brand_name_lavka_part when needed.
- Prefer short semantic names suitable for Figma layer names.
{extra_image_rules}
Allowed element semantic_name examples:
headline, subheadline, legal_text, age_badge, price_group,
brand_group, logo, logo_mark, logo_m, logo_heart, logo_leaf, logo_ellipse,
brand_name_yandex, brand_name_market, brand_name_lavka,
brand_name_yandex_part, brand_name_market_part, brand_name_lavka_part,
decoration_group, decoration_star, decoration_sparkle, decoration_glow,
star_straight, star_rotated, star_straight_1, star_straight_2, star_rotated_1, star_rotated_2,
hero_group, hero_item, hero_person, product_item, product_food, product_drink,
background_group, background_color, background_shape, background_gradient, visual_asset

Allowed group semantic_name examples:
brand_group, headline_group, legal_group, badge_group, decoration_group, hero_group, background_group

Compact banner metadata JSON:
{_prompt_json(request.banner_metadata)}

Compact node refs JSON:
{_prompt_json(node_refs)}

Compact candidate refs JSON:
{_prompt_json(candidate_refs)}

Compact heuristic roles JSON:
{_prompt_json(request.heuristic_roles)}

Allowed element candidate ids:
{_prompt_json(figma_summary.get("element_annotation_candidate_ids") or [])}

Allowed group candidate ids:
{_prompt_json(figma_summary.get("group_annotation_candidate_ids") or [])}

Return this JSON shape only:
{{
  "brand_family": "generic",
  "brand_confidence": 0.0,
  "language": "unknown",
  "category": "unknown",
  "layout_pattern": "unknown",
  "pattern_confidence": 0.0,
  "zones": [],
  "element_annotations": {{
    "candidate_id": {{
      "element_role": "unknown",
      "semantic_name": "",
      "parent_semantic_name": "",
      "functional_type": "functional",
      "importance_level": "medium",
      "is_text": null,
      "is_brand_related": false,
      "is_required_for_compliance": false,
      "adaptation_policy": {{
        "preserve_as_unit": true,
        "allow_reflow": false,
        "allow_scale": true,
        "allow_crop": false,
        "allow_shift": true,
        "drop_priority": 0,
        "anchor_type": "free"
      }},
      "confidence": 0.0,
      "reason_short": ""
    }}
  }},
  "group_annotations": {{
    "candidate_id": {{
      "group_role": "unknown",
      "semantic_name": "",
      "parent_semantic_name": "",
      "internal_layout": "freeform",
      "preserve_as_unit": true,
      "importance_level": "medium",
      "adaptation_policy": {{
        "allow_reflow": true,
        "allow_scale": true,
        "allow_crop": false,
        "allow_shift": true,
        "drop_priority": 0,
        "anchor_type": "free"
      }},
      "confidence": 0.0,
      "reason_short": ""
    }}
  }},
  "updates": [
    {{
      "source_figma_id": "...",
      "path": "...",
      "role": "...",
      "semantic_name": "...",
      "parent_semantic_name": "",
      "confidence": 0.0,
      "reason": ""
    }}
  ],
  "groups": [
    {{
      "id": "...",
      "role": "...",
      "semantic_name": "...",
      "parent_semantic_name": "",
      "children": ["..."],
      "confidence": 0.0,
      "reason": ""
    }}
  ],
  "semantic_relations": [],
  "layout_constraints": [],
  "preservation_priorities": [],
  "reason_short": ""
}}
""".strip()

    def _scene_heuristic_fallback(self, request: SceneAnnotateRequest) -> dict[str, Any]:
        figma_summary = request.figma_summary or {}
        out = self._semantic_structure_heuristic_fallback(figma_summary)
        out["semantic_relations"] = []
        out["layout_constraints"] = []
        return out

    def _finalize_scene_response(
        self,
        parsed: dict[str, Any],
        request: SceneAnnotateRequest,
        raw_output: str,
    ) -> dict[str, Any]:
        out = self._finalize_semantic_structure_response(parsed, request.figma_summary or {}, raw_output)
        figma_summary = request.figma_summary or {}
        node_by_id = {}
        node_by_path = {}
        for node in figma_summary.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", "") or "")
            path = str(node.get("path", "") or "")
            if node_id:
                node_by_id[node_id] = node
            node_by_path[path] = node

        allowed_children = set(node_by_id.keys())
        normalized_updates: list[dict[str, Any]] = []
        for item in parsed.get("updates") or []:
            if not isinstance(item, dict):
                continue
            source_figma_id = str(item.get("source_figma_id", "") or "")
            path = str(item.get("path", "") or "")
            if not source_figma_id or source_figma_id not in allowed_children:
                continue
            if path and path not in node_by_path:
                continue
            normalized_updates.append(
                {
                    "source_figma_id": source_figma_id,
                    "path": path,
                    "role": str(item.get("role", "unknown") or "unknown").lower().strip(),
                    "semantic_name": str(item.get("semantic_name", "") or "").strip(),
                    "parent_semantic_name": str(item.get("parent_semantic_name", "") or "").strip(),
                    "confidence": float(item.get("confidence", 0.0) or 0.0),
                    "reason": str(item.get("reason", "") or "").strip(),
                }
            )

        normalized_groups: list[dict[str, Any]] = []
        for item in parsed.get("groups") or []:
            if not isinstance(item, dict):
                continue
            children = [str(x) for x in (item.get("children") or []) if str(x) in allowed_children]
            if not children:
                continue
            normalized_groups.append(
                {
                    "id": str(item.get("id", "") or ""),
                    "role": str(item.get("role", "unknown") or "unknown").lower().strip(),
                    "semantic_name": str(item.get("semantic_name", "") or "").strip(),
                    "parent_semantic_name": str(item.get("parent_semantic_name", "") or "").strip(),
                    "children": children,
                    "confidence": float(item.get("confidence", 0.0) or 0.0),
                    "reason": str(item.get("reason", "") or "").strip(),
                }
            )

        out["updates"] = normalized_updates
        out["groups"] = normalized_groups
        out["semantic_relations"] = list(parsed.get("semantic_relations", []) or [])
        out["layout_constraints"] = list(parsed.get("layout_constraints", []) or [])
        return out

    def annotate_scene(self, request: SceneAnnotateRequest) -> dict[str, Any]:
        image: Optional[Image.Image] = None
        if request.banner_image_path:
            image = self._load_image(request.banner_image_path)
            image = self._prepare_image_for_model(image, "annotate_scene/banner")
            ar = _aspect_ratio(image)
            logger.info(
                "annotate_scene preprocessing: banner=%dx%d aspect=%.2f threshold=%.0f",
                image.width,
                image.height,
                ar,
                _QWEN_SAFE_MAX_ASPECT_RATIO,
            )
            if _is_extreme_aspect_ratio(image):
                logger.warning("annotate_scene: extreme aspect ratio, heuristic fallback ar=%.2f", ar)
                out = self._scene_heuristic_fallback(request)
                out["heuristic_fallback"] = True
                return out

        prompt = self._build_scene_prompt(request)
        mtokens = max(512, min(768, int(self.max_new_tokens)))
        images: list[Image.Image] = [image] if image is not None else []
        max_after_banner = 32
        atlas_ps = (request.element_atlas_image_path or "").strip()
        if atlas_ps:
            try:
                ai = self._load_image(atlas_ps)
                ai = self._prepare_image_for_model(ai, "annotate_scene/element_atlas")
                images.append(ai)
                max_after_banner -= 1
            except Exception as e:
                logger.warning("annotate_scene: skip element_atlas_image_path=%r: %s", atlas_ps, e)
        for i, p in enumerate((request.element_image_paths or [])[:max_after_banner]):
            ps = str(p).strip()
            if not ps:
                continue
            try:
                ei = self._load_image(ps)
                ei = self._prepare_image_for_model(ei, f"annotate_scene/element_crop_{i}")
                images.append(ei)
            except Exception as e:
                logger.warning("annotate_scene: skip element_image_paths[%d]=%r: %s", i, ps, e)
        logger.info(
            "annotate_scene calling model (images=%d incl. banner, max_new_tokens=%s temperature=%.2f do_sample=%s)",
            len(images),
            mtokens,
            self.temperature,
            self.temperature > 0,
        )
        raw_output = self._run_model(images, prompt, max_new_tokens=mtokens)
        parsed = self._extract_json(raw_output)
        if not isinstance(parsed, dict):
            logger.warning("annotate_scene: JSON parse failed, heuristic fallback")
            out = self._scene_heuristic_fallback(request)
            out["raw_model_output"] = raw_output
            out["heuristic_fallback"] = True
            return out

        out = self._finalize_scene_response(parsed, request, raw_output)
        out["heuristic_fallback"] = False
        return out

    def annotate_semantic_structure(self, request: SemanticStructureAnnotateRequest) -> dict[str, Any]:
        # Backward-compatible alias for older clients.
        scene_req = SceneAnnotateRequest(
            banner_metadata=request.figma_summary.get("canvas") or {},
            elements=request.figma_summary.get("nodes") or [],
            groups=request.figma_summary.get("candidates") or [],
            heuristic_roles={},
            figma_summary=request.figma_summary,
            banner_image_path=request.banner_image_path,
            element_image_paths=list(request.element_image_paths or []),
            element_atlas_image_path=request.element_atlas_image_path,
        )
        return self.annotate_scene(scene_req)


def create_app(
    model_path: str,
    device: str = "cuda:1",
    max_new_tokens: int = 768,
    temperature: float = 0.0,
    max_image_long_side: int = 1024,
) -> FastAPI:
    app = FastAPI(title="Qwen Layout Annotation Service")
    runtime = QwenRuntime(
        model_path=model_path,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        max_image_long_side=max_image_long_side,
    )
    runtime.load()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=(runtime.model is not None and runtime.processor is not None), device=runtime.device, model_path=runtime.model_path)

    @app.post("/annotate/brand-context")
    def annotate_brand_context(request: BrandContextAnnotateRequest) -> dict[str, Any]:
        try:
            return runtime.annotate_brand_context(request).to_dict()
        except Exception as e:
            logger.exception("annotate_brand_context failed")
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    @app.post("/annotate/banner")
    def annotate_banner(request: BannerAnnotateRequest) -> dict[str, Any]:
        try:
            return runtime.annotate_banner(request).to_dict()
        except Exception as e:
            logger.exception("annotate_banner failed")
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    @app.post("/annotate/candidate")
    def annotate_candidate(request: CandidateAnnotateRequest) -> dict[str, Any]:
        cid = request.candidate.get("candidate_id", "unknown_candidate")
        try:
            return runtime.annotate_candidate(request).to_dict()
        except Exception as e:
            logger.exception("annotate_candidate failed candidate_id=%r", cid)
            raise HTTPException(
                status_code=500,
                detail=f"candidate_id={cid!r}: {type(e).__name__}: {e}",
            ) from e

    @app.post("/annotate/group")
    def annotate_group(request: GroupAnnotateRequest) -> dict[str, Any]:
        cid = request.candidate.get("candidate_id", "unknown_candidate")
        try:
            return runtime.annotate_group(request).to_dict()
        except Exception as e:
            logger.exception("annotate_group failed candidate_id=%r", cid)
            raise HTTPException(
                status_code=500,
                detail=f"candidate_id={cid!r}: {type(e).__name__}: {e}",
            ) from e

    @app.post("/annotate/semantic-structure")
    def annotate_semantic_structure(request: SemanticStructureAnnotateRequest) -> dict[str, Any]:
        try:
            return runtime.annotate_semantic_structure(request)
        except Exception as e:
            logger.exception("annotate_semantic_structure failed")
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    @app.post("/annotate/scene")
    def annotate_scene(request: SceneAnnotateRequest) -> dict[str, Any]:
        try:
            return runtime.annotate_scene(request)
        except Exception as e:
            logger.exception("annotate_scene failed")
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    return app
