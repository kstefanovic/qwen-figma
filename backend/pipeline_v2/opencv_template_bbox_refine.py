from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from backend.pipeline_v2.schemas import NormalizedBbox, TextZoneChildItem, TextZoneGroupItem

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATE_PATHS = {
    "logo": [
        "/root/.cursor/projects/root-figma-design-qwen-figma/assets/"
        "c__Users_Administrator_AppData_Roaming_Cursor_User_workspaceStorage_b949b4ea37566c8770ac24d261e5abb0_images_image-22435870-1aea-4dda-bf05-7c35881a2b97.png",
    ],
    "age_badge": [
        "/root/.cursor/projects/root-figma-design-qwen-figma/assets/"
        "c__Users_Administrator_AppData_Roaming_Cursor_User_workspaceStorage_b949b4ea37566c8770ac24d261e5abb0_images_image-9f4df642-b56d-4a6d-8b1d-18b63951cb6d.png",
    ],
}


@dataclass(frozen=True)
class TemplateMatch:
    role: str
    bbox: NormalizedBbox
    score: float
    template_path: str


def opencv_template_refinement_enabled() -> bool:
    return os.environ.get("USE_OPENCV_TEMPLATE_BBOX_REFINEMENT", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _mk_bbox(x: float, y: float, w: float, h: float) -> NormalizedBbox | None:
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    if x + w > 1.0:
        w = max(0.0, 1.0 - x)
    if y + h > 1.0:
        h = max(0.0, 1.0 - y)
    if w <= 1e-6 or h <= 1e-6:
        return None
    return NormalizedBbox(x=x, y=y, width=w, height=h)


def _bbox_union(items: list[NormalizedBbox]) -> NormalizedBbox | None:
    if not items:
        return None
    x0 = min(b.x for b in items)
    y0 = min(b.y for b in items)
    x1 = max(b.x + b.width for b in items)
    y1 = max(b.y + b.height for b in items)
    return _mk_bbox(x0, y0, x1 - x0, y1 - y0)


def _template_paths_for_role(role: str) -> list[str]:
    env_key = f"OPENCV_TEMPLATE_{role.upper()}_PATHS"
    raw = (os.environ.get(env_key) or "").strip()
    if raw:
        return [p for p in raw.split(os.pathsep) if p.strip()]
    return _DEFAULT_TEMPLATE_PATHS.get(role, [])


def _candidate_scales() -> list[float]:
    raw = (os.environ.get("OPENCV_TEMPLATE_SCALES") or "").strip()
    if raw:
        vals: list[float] = []
        for part in raw.split(","):
            try:
                v = float(part.strip())
            except ValueError:
                continue
            if 0.08 <= v <= 8.0:
                vals.append(v)
        if vals:
            return vals
    return [0.35, 0.45, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.4]


def _min_score(role: str) -> float:
    default = "0.60" if role == "age_badge" else "0.68"
    try:
        return float(os.environ.get(f"OPENCV_TEMPLATE_{role.upper()}_MIN_SCORE", default) or default)
    except ValueError:
        return float(default)


def _image_to_bgr_array(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return rgb[:, :, ::-1]


def _load_template(path: str) -> np.ndarray | None:
    try:
        import cv2  # type: ignore

        arr = cv2.imread(path, cv2.IMREAD_COLOR)
        return arr if arr is not None and arr.size > 0 else None
    except Exception as exc:
        logger.warning("OpenCV template load failed path=%r: %s", path, exc)
        return None


def _match_one_template(
    image_bgr: np.ndarray,
    template_bgr: np.ndarray,
    *,
    role: str,
    template_path: str,
) -> TemplateMatch | None:
    import cv2  # type: ignore

    ih, iw = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    best: TemplateMatch | None = None
    for scale in _candidate_scales():
        tpl = template_bgr
        if scale != 1.0:
            tpl = cv2.resize(
                template_bgr,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
            )
        th, tw = tpl.shape[:2]
        if th < 6 or tw < 6 or th >= ih or tw >= iw:
            continue
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        bbox = _mk_bbox(max_loc[0] / iw, max_loc[1] / ih, tw / iw, th / ih)
        if bbox is None:
            continue
        match = TemplateMatch(role=role, bbox=bbox, score=float(max_val), template_path=template_path)
        if best is None or match.score > best.score:
            best = match
    if best and best.score >= _min_score(role):
        return best
    return None


def _find_template_match(image: Image.Image, role: str) -> TemplateMatch | None:
    image_bgr = _image_to_bgr_array(image)
    best: TemplateMatch | None = None
    for path in _template_paths_for_role(role):
        if not Path(path).exists():
            continue
        template = _load_template(path)
        if template is None:
            continue
        match = _match_one_template(image_bgr, template, role=role, template_path=path)
        if match is not None and (best is None or match.score > best.score):
            best = match
    return best


def _refined_group_bbox(group: TextZoneGroupItem) -> NormalizedBbox:
    return _bbox_union([child.bbox for child in group.children]) or group.bbox


def refine_text_zone_bboxes_with_opencv_templates(
    image: Image.Image,
    groups: list[TextZoneGroupItem],
    warnings: list[str],
) -> list[TextZoneGroupItem]:
    if not opencv_template_refinement_enabled() or not groups:
        return groups
    try:
        import cv2  # noqa: F401  # type: ignore
    except Exception as exc:
        warnings.append(f"OpenCV template bbox refinement skipped: cv2 unavailable ({exc}).")
        return groups

    matches: dict[str, TemplateMatch] = {}
    for role in ("logo", "age_badge"):
        match = _find_template_match(image, role)
        if match is not None:
            matches[role] = match
    if not matches:
        warnings.append("OpenCV template bbox refinement: no template match above threshold.")
        return groups

    refined_count = 0
    refined_groups: list[TextZoneGroupItem] = []
    for group in groups:
        refined_children: list[TextZoneChildItem] = []
        group_refined_count = 0
        for child in group.children:
            match = matches.get(str(child.role))
            if match is None:
                refined_children.append(child)
                continue
            refined_count += 1
            group_refined_count += 1
            refined_children.append(
                TextZoneChildItem(
                    role=child.role,
                    text=child.text,
                    bbox=match.bbox,
                    confidence=max(child.confidence, min(1.0, match.score)),
                    reason=(
                        child.reason
                        + f" [bbox refined by OpenCV template score={match.score:.3f}]"
                    ).strip(),
                ),
            )
        refined_group = TextZoneGroupItem(
            role=group.role,
            bbox=group.bbox,
            confidence=group.confidence,
            reason=group.reason,
            children=refined_children,
        )
        if refined_children:
            refined_group = TextZoneGroupItem(
                role=refined_group.role,
                bbox=_refined_group_bbox(refined_group),
                confidence=refined_group.confidence,
                reason=(
                    refined_group.reason + " [bbox refined by OpenCV template]"
                    if group_refined_count
                    else refined_group.reason
                ).strip(),
                children=refined_children,
            )
        refined_groups.append(refined_group)

    details = ", ".join(f"{role}={match.score:.3f}" for role, match in sorted(matches.items()))
    warnings.append(
        f"OpenCV template bbox refinement: refined {refined_count} child bbox(es); {details}.",
    )
    return refined_groups
