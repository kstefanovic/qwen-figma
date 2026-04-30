from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from PIL import Image

from backend.pipeline_v2.schemas import NormalizedBbox, TextZoneChildItem, TextZoneGroupItem

logger = logging.getLogger(__name__)

_OCR: Any | None = None
_OCR_INIT_FAILED = False
_TEXT_CHILD_ROLES = frozenset(
    {
        "brand_name",
        "brand_name_first",
        "brand_name_second",
        "headline",
        "subheadline",
        "subheadline_delivery_time",
        "subheadline_weight",
        "product_name",
        "subheadline_discount",
        "legal_text",
        "age_badge",
    },
)
_HEADLINE_SUPPORT_ROLES = frozenset(
    {
        "subheadline",
        "subheadline_delivery_time",
        "subheadline_weight",
        "product_name",
        "subheadline_discount",
    },
)


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    bbox: NormalizedBbox


def paddle_ocr_refinement_enabled() -> bool:
    return os.environ.get("USE_PADDLE_OCR_BBOX_REFINEMENT", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _get_ocr() -> Any | None:
    global _OCR, _OCR_INIT_FAILED
    if _OCR is not None:
        return _OCR
    if _OCR_INIT_FAILED:
        return None
    try:
        from paddleocr import PaddleOCR  # type: ignore

        raw_lang = (os.environ.get("PADDLE_OCR_LANG") or "ru").strip() or "ru"
        langs = [raw_lang]
        for fallback in ("ru", "en"):
            if fallback not in langs:
                langs.append(fallback)
        last_exc: Exception | None = None
        for lang in langs:
            try:
                _OCR = PaddleOCR(
                    lang=lang,
                    enable_mkldnn=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
                logger.info("PaddleOCR bbox refinement initialized lang=%r", lang)
                return _OCR
            except Exception as exc:
                last_exc = exc
                logger.warning("PaddleOCR init failed for lang=%r: %s", lang, exc)
        if last_exc is not None:
            raise last_exc
        return None
    except Exception as exc:
        _OCR_INIT_FAILED = True
        logger.warning("PaddleOCR bbox refinement disabled: %s", exc)
        return None


def _norm_text(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", "", s)


def _bbox_union(items: list[NormalizedBbox]) -> NormalizedBbox | None:
    if not items:
        return None
    x0 = min(b.x for b in items)
    y0 = min(b.y for b in items)
    x1 = max(b.x + b.width for b in items)
    y1 = max(b.y + b.height for b in items)
    return _mk_bbox(x0, y0, x1 - x0, y1 - y0)


def _bbox_bottom(b: NormalizedBbox) -> float:
    return b.y + b.height


def _bbox_right(b: NormalizedBbox) -> float:
    return b.x + b.width


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


def _poly_to_bbox(poly: Any, width: int, height: int) -> NormalizedBbox | None:
    if hasattr(poly, "tolist"):
        try:
            poly = poly.tolist()
        except Exception:
            pass
    points: list[tuple[float, float]] = []
    if isinstance(poly, (list, tuple)):
        if len(poly) == 4 and all(isinstance(v, (int, float)) for v in poly):
            x0, y0, x1, y1 = (float(poly[0]), float(poly[1]), float(poly[2]), float(poly[3]))
            return _mk_bbox(x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height)
        for p in poly:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    points.append((float(p[0]), float(p[1])))
                except (TypeError, ValueError):
                    continue
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return _mk_bbox(min(xs) / width, min(ys) / height, (max(xs) - min(xs)) / width, (max(ys) - min(ys)) / height)


def _iter_v2_ocr_result(raw: Any, width: int, height: int) -> list[OcrLine]:
    if not isinstance(raw, list):
        return []
    pages = raw
    if len(raw) == 1 and isinstance(raw[0], list):
        pages = raw[0]
    out: list[OcrLine] = []
    for item in pages:
        if not (isinstance(item, (list, tuple)) and len(item) >= 2):
            continue
        bbox = _poly_to_bbox(item[0], width, height)
        rec = item[1]
        if bbox is None or not isinstance(rec, (list, tuple)) or len(rec) < 2:
            continue
        text = str(rec[0] or "").strip()
        try:
            confidence = float(rec[1] or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if text:
            out.append(OcrLine(text=text, confidence=confidence, bbox=bbox))
    return out


def _iter_v3_ocr_result(raw: Any, width: int, height: int) -> list[OcrLine]:
    out: list[OcrLine] = []
    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        if hasattr(item, "json"):
            try:
                item = item.json() if callable(item.json) else item.json
            except Exception:
                pass
        if hasattr(item, "res"):
            item = getattr(item, "res")
        if isinstance(item, dict) and isinstance(item.get("res"), dict):
            item = item["res"]
        if not isinstance(item, dict):
            continue
        texts = item.get("rec_texts") or item.get("texts") or []
        scores = item.get("rec_scores") or item.get("scores") or []
        boxes = item.get("rec_polys") or item.get("rec_boxes") or item.get("dt_polys") or []
        if not (isinstance(texts, list) and isinstance(boxes, list)):
            continue
        for i, text_raw in enumerate(texts):
            text = str(text_raw or "").strip()
            if not text:
                continue
            score_raw = scores[i] if i < len(scores) else 0.0
            try:
                confidence = float(score_raw or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            bbox = _poly_to_bbox(boxes[i] if i < len(boxes) else None, width, height)
            if bbox is not None:
                out.append(OcrLine(text=text, confidence=confidence, bbox=bbox))
    return out


def _run_paddle_ocr(image: Image.Image) -> list[OcrLine]:
    ocr = _get_ocr()
    if ocr is None:
        return []
    width, height = image.size
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        image.convert("RGB").save(tmp, format="PNG")
    try:
        raw: Any
        if hasattr(ocr, "predict"):
            raw = ocr.predict(str(tmp_path))
        elif hasattr(ocr, "ocr"):
            try:
                raw = ocr.ocr(str(tmp_path), cls=True)
            except TypeError:
                raw = ocr.ocr(str(tmp_path))
        else:
            raw = []
        lines = _iter_v2_ocr_result(raw, width, height)
        if not lines:
            lines = _iter_v3_ocr_result(raw, width, height)
        return [line for line in lines if line.confidence >= _min_ocr_confidence()]
    except Exception as exc:
        logger.warning("PaddleOCR bbox refinement failed: %s", exc)
        return []
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _min_ocr_confidence() -> float:
    try:
        return float(os.environ.get("PADDLE_OCR_MIN_CONFIDENCE", "0.35") or 0.35)
    except (TypeError, ValueError):
        return 0.35


def _bbox_center(b: NormalizedBbox) -> tuple[float, float]:
    return b.x + b.width / 2.0, b.y + b.height / 2.0


def _bbox_intersection_ratio(a: NormalizedBbox, b: NormalizedBbox) -> float:
    ax1, ay1 = a.x + a.width, a.y + a.height
    bx1, by1 = b.x + b.width, b.y + b.height
    ix = max(0.0, min(ax1, bx1) - max(a.x, b.x))
    iy = max(0.0, min(ay1, by1) - max(a.y, b.y))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / max(1e-9, min(a.width * a.height, b.width * b.height))


def _line_match_score(target_norm: str, qwen_bbox: NormalizedBbox, line: OcrLine) -> float:
    ocr_norm = _norm_text(line.text)
    if not target_norm or not ocr_norm:
        return 0.0
    if target_norm == ocr_norm:
        text_score = 1.0
    elif target_norm in ocr_norm or ocr_norm in target_norm:
        text_score = min(len(target_norm), len(ocr_norm)) / max(len(target_norm), len(ocr_norm))
    else:
        text_score = SequenceMatcher(None, target_norm, ocr_norm).ratio()
    qx, qy = _bbox_center(qwen_bbox)
    ox, oy = _bbox_center(line.bbox)
    dist = ((qx - ox) ** 2 + (qy - oy) ** 2) ** 0.5
    proximity = max(0.0, 1.0 - dist / 0.45)
    overlap = _bbox_intersection_ratio(qwen_bbox, line.bbox)
    return text_score * 0.72 + proximity * 0.16 + overlap * 0.12


def _legal_bbox_from_smallest_text(lines: list[OcrLine]) -> NormalizedBbox | None:
    """
    Legal/disclaimer copy is usually the smallest readable text. Use this as a fallback
    when OCR text differs from Qwen's legal transcription but the small footer rows are clear.
    """
    legal_markers = (
        "раздел",
        "аптека",
        "предоставляется",
        "партнер",
        "партнеры",
        "продавец",
        "огрн",
        "ооо",
        "инн",
        "реклама",
        "доставку",
        "сервис",
        "контакт",
        "адрес",
    )
    def legal_like(line: OcrLine) -> bool:
        text_norm = _norm_text(line.text)
        if not text_norm:
            return False
        has_marker = any(marker in text_norm for marker in legal_markers)
        has_legal_shape = len(text_norm) >= 18 and line.bbox.y >= 0.68
        return has_marker or has_legal_shape

    candidates: list[OcrLine] = []
    for line in lines:
        text_norm = _norm_text(line.text)
        if not text_norm or re.fullmatch(r"\d{1,2}", text_norm):
            continue
        # Avoid grabbing the age badge or isolated price/number fragments.
        if re.fullmatch(r"\d{1,2}\+?", re.sub(r"\s+", "", line.text or "")):
            continue
        if not legal_like(line):
            continue
        candidates.append(line)
    if not candidates:
        return None

    min_h = min(line.bbox.height for line in candidates)
    threshold_h = min(0.10, max(min_h * 2.05, min_h + 0.012))
    small_rows = [
        line
        for line in candidates
        if line.bbox.height <= threshold_h
        and legal_like(line)
    ]
    if not small_rows:
        return None

    # Keep the main dense block: legal rows are close together vertically and horizontally.
    small_rows.sort(key=lambda line: (line.bbox.y, line.bbox.x))
    vertical_clusters: list[list[OcrLine]] = []
    for line in small_rows:
        if not vertical_clusters:
            vertical_clusters.append([line])
            continue
        prev = vertical_clusters[-1][-1]
        if line.bbox.y - (prev.bbox.y + prev.bbox.height) <= max(0.035, threshold_h):
            vertical_clusters[-1].append(line)
        else:
            vertical_clusters.append([line])
    clusters: list[list[OcrLine]] = []
    for cluster in vertical_clusters:
        by_x = sorted(cluster, key=lambda line: line.bbox.x)
        current: list[OcrLine] = []
        current_right = 0.0
        for line in by_x:
            if not current:
                current = [line]
                current_right = line.bbox.x + line.bbox.width
                continue
            gap = line.bbox.x - current_right
            if gap <= 0.18:
                current.append(line)
                current_right = max(current_right, line.bbox.x + line.bbox.width)
            else:
                clusters.append(current)
                current = [line]
                current_right = line.bbox.x + line.bbox.width
        if current:
            clusters.append(current)
    best = max(
        clusters,
        key=lambda cluster: (
            sum(len(_norm_text(line.text)) for line in cluster),
            sum(line.bbox.width * line.bbox.height for line in cluster),
        ),
    )
    if sum(len(_norm_text(line.text)) for line in best) < 12:
        return None
    return _bbox_union([line.bbox for line in best])


def _match_ocr_bbox_for_child(child: TextZoneChildItem, lines: list[OcrLine]) -> NormalizedBbox | None:
    target = _norm_text(child.text)
    if not target:
        return None
    if child.role == "age_badge":
        target_age = re.sub(r"\s+", "", child.text or "").lower()
        candidates = [
            line
            for line in lines
            if re.sub(r"\s+", "", line.text or "").lower() == target_age
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda line: _bbox_intersection_ratio(child.bbox, line.bbox)).bbox
    if child.role == "legal_text" or "\n" in child.text:
        legal_markers = (
            "раздел",
            "аптека",
            "предоставляется",
            "партнер",
            "партнеры",
            "продавец",
            "огрн",
            "ооо",
            "инн",
            "реклама",
            "доставку",
            "сервис",
        )
        picked: list[NormalizedBbox] = []
        seen = ""
        for line in sorted(lines, key=lambda item: (item.bbox.y, item.bbox.x)):
            ocr_norm = _norm_text(line.text)
            if not ocr_norm:
                continue
            is_text_match = ocr_norm in target or SequenceMatcher(None, target, ocr_norm).ratio() >= 0.52
            # Legal blocks are dense footer/base-copy. Do not let a shared brand word
            # like "Яндекс" in the brand row become the legal bbox.
            is_footer_like = line.bbox.y >= 0.72 or any(marker in ocr_norm for marker in legal_markers)
            if not (is_text_match and is_footer_like):
                continue
            picked.append(line.bbox)
            seen += ocr_norm
            if len(seen) >= max(4, int(len(target) * 0.55)):
                break
        if len(picked) < 2 or len(seen) < max(12, int(len(target) * 0.12)):
            return _legal_bbox_from_smallest_text(lines)
        return _bbox_union(picked)

    # Short marks like 0+ need exact/substring matching to avoid random OCR noise.
    if len(target) <= 3:
        candidates = [line for line in lines if target == _norm_text(line.text) or target in _norm_text(line.text)]
        if not candidates:
            return None
        return max(candidates, key=lambda line: _line_match_score(target, child.bbox, line)).bbox

    scored = sorted(
        ((_line_match_score(target, child.bbox, line), line) for line in lines),
        key=lambda item: item[0],
        reverse=True,
    )
    if scored and scored[0][0] >= 0.72:
        return scored[0][1].bbox

    picked: list[NormalizedBbox] = []
    seen = ""
    for _score, line in scored:
        ocr_norm = _norm_text(line.text)
        if not ocr_norm or ocr_norm not in target:
            continue
        picked.append(line.bbox)
        seen += ocr_norm
        if len(seen) >= max(4, int(len(target) * 0.55)):
            break
    return _bbox_union(picked)


def _refined_group_bbox(group: TextZoneGroupItem) -> NormalizedBbox:
    if group.role == "brand_group":
        children_with_boxes = [
            child.bbox
            for child in group.children
            if child.role in ("brand_name", "brand_name_first", "brand_name_second")
        ]
        if children_with_boxes:
            return _bbox_union(children_with_boxes) or group.bbox
    children_with_boxes = [child.bbox for child in group.children]
    return _bbox_union(children_with_boxes) or group.bbox


def _looks_like_delivery_subline(text: str) -> bool:
    t = _norm_text(text)
    if not t:
        return False
    legal_markers = ("огрн", "ооо", "продавец", "партнер", "партнеры", "сервис", "москва", "красногвард")
    if any(marker in t for marker in legal_markers):
        return False
    markers = ("достав", "курьер", "минут", "мин", "привез", "privez", "delivery", "courier")
    if any(marker in t for marker in markers):
        return True
    # PaddleOCR often reads "от" as Latin "OT" / "ot".
    raw = re.sub(r"\s+", "", text or "").lower()
    return raw in {"от", "ot"} or bool(re.fullmatch(r"\d{1,2}", raw))


def _clean_delivery_text(lines: list[OcrLine]) -> str:
    parts: list[str] = []
    for line in sorted(lines, key=lambda item: (item.bbox.y, item.bbox.x)):
        s = re.sub(r"\s+", " ", (line.text or "").strip())
        if not s:
            continue
        if s.lower() == "ot":
            s = "от"
        parts.append(s)
    return " ".join(parts).strip()


def _maybe_add_delivery_subheadline_child(
    group: TextZoneGroupItem,
    children: list[TextZoneChildItem],
    lines: list[OcrLine],
    *,
    legal_top: float | None,
) -> tuple[list[TextZoneChildItem], bool]:
    if group.role != "headline_group":
        return children, False
    if any(child.role in _HEADLINE_SUPPORT_ROLES for child in children):
        return children, False
    headline = next((child for child in children if child.role == "headline"), None)
    if headline is None:
        return children, False

    hx0 = max(0.0, headline.bbox.x - 0.05)
    hx1 = min(1.0, _bbox_right(headline.bbox) + 0.20)
    hy0 = _bbox_bottom(headline.bbox) - 0.01
    if legal_top is not None and legal_top > hy0 + 0.05:
        hy1 = min(legal_top, 0.86)
    else:
        hy1 = 0.86
    legalish_markers = ("огрн", "ооо", "продавец", "партнер", "партнеры", "сервис", "москва", "красногвард")
    legalish_tops = [
        line.bbox.y
        for line in lines
        if line.bbox.y > hy0 and any(marker in _norm_text(line.text) for marker in legalish_markers)
    ]
    if legalish_tops:
        hy1 = min(hy1, max(hy0 + 0.02, min(legalish_tops) - 0.01), 0.84)

    candidates: list[OcrLine] = []
    for line in lines:
        if line.bbox.y < hy0 or line.bbox.y >= hy1:
            continue
        cx = line.bbox.x + line.bbox.width / 2.0
        if cx < hx0 or cx > hx1:
            continue
        if line.bbox.height > headline.bbox.height * 0.85:
            continue
        if _looks_like_delivery_subline(line.text):
            candidates.append(line)
    if not candidates:
        return children, False

    candidates.sort(key=lambda item: (item.bbox.y, item.bbox.x))
    cluster: list[OcrLine] = []
    for line in candidates:
        if not cluster:
            cluster = [line]
            continue
        prev_bottom = max(_bbox_bottom(item.bbox) for item in cluster)
        if line.bbox.y - prev_bottom <= 0.045:
            cluster.append(line)
        else:
            break
    candidates = cluster
    # Include neighboring OCR fragments on the same short delivery block, e.g. "от" + "15" + "минут".
    first_y = candidates[0].bbox.y
    last_bottom = max(_bbox_bottom(line.bbox) for line in candidates)
    expanded = [
        line
        for line in lines
        if line.bbox.y >= first_y - 0.015
        and line.bbox.y <= last_bottom + 0.035
        and hx0 <= line.bbox.x + line.bbox.width / 2.0 <= hx1
        and line.bbox.height <= headline.bbox.height * 0.9
        and not any(marker in _norm_text(line.text) for marker in legalish_markers)
        and not re.fullmatch(r"\d{1,2}\+", re.sub(r"\s+", "", line.text or ""))
    ]
    if expanded:
        candidates = sorted(expanded, key=lambda item: (item.bbox.y, item.bbox.x))

    bbox = _bbox_union([line.bbox for line in candidates])
    text = _clean_delivery_text(candidates)
    if bbox is None or len(_norm_text(text)) < 4:
        return children, False
    new_child = TextZoneChildItem(
        role="subheadline_delivery_time",
        text=text,
        bbox=bbox,
        confidence=0.82,
        reason="Recovered from PaddleOCR delivery subline under headline",
    )
    return [*children, new_child], True


def refine_text_zone_bboxes_with_paddle_ocr(
    image: Image.Image,
    groups: list[TextZoneGroupItem],
    warnings: list[str],
) -> list[TextZoneGroupItem]:
    if not paddle_ocr_refinement_enabled() or not groups:
        return groups
    lines = _run_paddle_ocr(image)
    if not lines:
        detail = "PaddleOCR unavailable" if _OCR_INIT_FAILED else "no OCR lines detected"
        warnings.append(f"PaddleOCR bbox refinement skipped: {detail}.")
        return groups

    legal_top = next((g.bbox.y for g in groups if g.role == "legal_text_group"), None)
    refined_groups: list[TextZoneGroupItem] = []
    refined_count = 0
    recovered_subheadline_count = 0
    for group in groups:
        refined_children: list[TextZoneChildItem] = []
        group_refined_count = 0
        for child in group.children:
            if child.role not in _TEXT_CHILD_ROLES:
                refined_children.append(child)
                continue
            refined_bbox = _match_ocr_bbox_for_child(child, lines)
            if refined_bbox is None:
                refined_children.append(child)
                continue
            refined_count += 1
            group_refined_count += 1
            refined_children.append(
                TextZoneChildItem(
                    role=child.role,
                    text=child.text,
                    bbox=refined_bbox,
                    confidence=child.confidence,
                    reason=(child.reason + " [bbox refined by PaddleOCR]").strip(),
                ),
            )
        refined_children, added_subheadline = _maybe_add_delivery_subheadline_child(
            group,
            refined_children,
            lines,
            legal_top=legal_top,
        )
        if added_subheadline:
            recovered_subheadline_count += 1
            group_refined_count += 1

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
                reason=(refined_group.reason + " [bbox refined by PaddleOCR]").strip()
                if group_refined_count
                else refined_group.reason,
                children=refined_children,
            )
        refined_groups.append(refined_group)

    warnings.append(
        f"PaddleOCR bbox refinement: refined {refined_count} child bbox(es) from {len(lines)} OCR line(s).",
    )
    if recovered_subheadline_count:
        warnings.append(
            f"PaddleOCR subheadline recovery: added {recovered_subheadline_count} missing delivery subheadline child.",
        )
    return refined_groups
