from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from PIL import Image

from qwen_service.schemas import BannerAnnotateRequest, CandidateAnnotateRequest, GroupAnnotateRequest, HealthResponse

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


class QwenRuntime:
    def __init__(self, model_path: str, device: str = "cuda:1", max_new_tokens: int = 512, temperature: float = 0.1, use_fp16: bool = True) -> None:
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_fp16 = use_fp16
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

    def ensure_loaded(self) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen runtime not loaded.")

    def _load_image(self, image_path: str) -> Image.Image:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        return Image.open(path).convert("RGB")

    def _crop_candidate(self, image: Image.Image, candidate: dict[str, Any], expand_ratio: float = 0.0) -> Image.Image:
        width, height = image.size
        x_norm, y_norm, w_norm, h_norm = candidate["bbox_canvas"]
        x = x_norm * width
        y = y_norm * height
        w = w_norm * width
        h = h_norm * height
        pad_x = w * expand_ratio
        pad_y = h * expand_ratio
        left = max(0, int(round(x - pad_x)))
        top = max(0, int(round(y - pad_y)))
        right = min(width, int(round(x + w + pad_x)))
        bottom = min(height, int(round(y + h + pad_y)))
        if right <= left:
            right = min(width, left + 1)
        if bottom <= top:
            bottom = min(height, top + 1)
        return image.crop((left, top, right, bottom))

    def _run_model(self, images: list[Image.Image], prompt: str) -> str:
        self.ensure_loaded()
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=self.temperature > 0, temperature=self.temperature)
        output_text = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output_text.strip()

    def _extract_json(self, text: str) -> Any:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        for block in fenced:
            try:
                return json.loads(block.strip())
            except Exception:
                continue
        obj_match = self._find_balanced_json_substring(text, "{", "}")
        if obj_match is not None:
            try:
                return json.loads(obj_match)
            except Exception:
                pass
        arr_match = self._find_balanced_json_substring(text, "[", "]")
        if arr_match is not None:
            try:
                return json.loads(arr_match)
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

    def annotate_banner(self, request: BannerAnnotateRequest) -> BannerAnnotation:
        image = self._load_image(request.banner_image_path)
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
        banner_image = self._load_image(request.banner_image_path)
        tight_crop = self._crop_candidate(banner_image, request.candidate, expand_ratio=0.0)
        context_crop = self._crop_candidate(banner_image, request.candidate, expand_ratio=request.context_padding_ratio)
        prompt = self._build_element_prompt(request.candidate, request.heuristic_annotation)
        raw_output = self._run_model([banner_image, tight_crop, context_crop], prompt)
        data = self._extract_json(raw_output)
        candidate_id = str(request.candidate.get("candidate_id", "unknown_candidate"))
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
        banner_image = self._load_image(request.banner_image_path)
        group_crop = self._crop_candidate(banner_image, request.candidate, expand_ratio=request.context_padding_ratio)
        prompt = self._build_group_prompt(request.candidate, request.heuristic_annotation)
        raw_output = self._run_model([banner_image, group_crop], prompt)
        data = self._extract_json(raw_output)
        candidate_id = str(request.candidate.get("candidate_id", "unknown_candidate"))
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


def create_app(model_path: str, device: str = "cuda:1", max_new_tokens: int = 512, temperature: float = 0.1) -> FastAPI:
    app = FastAPI(title="Qwen Layout Annotation Service")
    runtime = QwenRuntime(model_path=model_path, device=device, max_new_tokens=max_new_tokens, temperature=temperature)
    runtime.load()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=(runtime.model is not None and runtime.processor is not None), device=runtime.device, model_path=runtime.model_path)

    @app.post("/annotate/banner")
    def annotate_banner(request: BannerAnnotateRequest) -> dict[str, Any]:
        try:
            return runtime.annotate_banner(request).to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/annotate/candidate")
    def annotate_candidate(request: CandidateAnnotateRequest) -> dict[str, Any]:
        try:
            return runtime.annotate_candidate(request).to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/annotate/group")
    def annotate_group(request: GroupAnnotateRequest) -> dict[str, Any]:
        try:
            return runtime.annotate_group(request).to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
