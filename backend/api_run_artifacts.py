from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from backend.final_json_mapper import build_final_json
from backend.pipeline_v2.schemas import AnalyzeTextZoneVisualResponse, ClassifyZoneResponse
from backend.storage import RunStorage

logger = logging.getLogger(__name__)

_GROUP_COLORS: dict[str, tuple[int, int, int]] = {
    "brand_group": (0, 140, 255),
    "headline_group": (255, 120, 0),
    "age_badge_group": (255, 0, 180),
    "legal_text_group": (80, 200, 80),
}
_CHILD_COLOR = (255, 220, 0)


def repo_runs_dir() -> Path:
    """Default ``…/qwen-figma/runs`` next to ``backend/``; override with BACKEND_RUNS_DIR."""
    import os

    env = (os.environ.get("BACKEND_RUNS_DIR") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    root = Path(__file__).resolve().parent.parent
    return (root / "runs").resolve()


def ensure_output_dir(storage: RunStorage, run_id: str) -> Path:
    out = storage.get_run_dir(run_id) / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _decode_banner_rgb(raw: bytes) -> Image.Image:
    with Image.open(io.BytesIO(raw)) as im:
        return im.convert("RGB")


def _norm_rect_to_pixels(
    bbox: dict[str, Any] | Any,
    pw: int,
    ph: int,
) -> tuple[int, int, int, int] | None:
    if hasattr(bbox, "model_dump"):
        d = bbox.model_dump()
    elif isinstance(bbox, dict):
        d = bbox
    else:
        return None
    if "bounds" in d and isinstance(d.get("bounds"), dict):
        d = d["bounds"]
        try:
            x0 = max(0, min(pw - 1, int(round(float(d.get("x", 0) or 0)))))
            y0 = max(0, min(ph - 1, int(round(float(d.get("y", 0) or 0)))))
            x1 = max(0, min(pw, int(round(float(d.get("x", 0) or 0) + float(d.get("width", 0) or 0)))))
            y1 = max(0, min(ph, int(round(float(d.get("y", 0) or 0) + float(d.get("height", 0) or 0)))))
        except (TypeError, ValueError):
            return None
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1
    try:
        x = float(d.get("x", 0) or 0)
        y = float(d.get("y", 0) or 0)
        w = float(d.get("width", 0) or 0)
        h = float(d.get("height", 0) or 0)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    x0 = max(0, min(pw - 1, int(round(x * pw))))
    y0 = max(0, min(ph - 1, int(round(y * ph))))
    x1 = max(0, min(pw, int(round((x + w) * pw))))
    y1 = max(0, min(ph, int(round((y + h) * ph))))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def draw_text_zone_on_image(
    rgb: Image.Image,
    text_zone: dict[str, Any] | Any,
) -> Image.Image:
    """Draw group (solid) and child (dashed) boxes from ``text_zone.groups``."""
    out = rgb.copy()
    dr = ImageDraw.Draw(out)
    pw, ph = out.size
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    groups: list[Any]
    if hasattr(text_zone, "model_dump"):
        tz = text_zone.model_dump()
        groups = list(tz.get("groups") or [])
    elif isinstance(text_zone, dict):
        groups = list(text_zone.get("groups") or [])
    else:
        groups = []

    for g in groups:
        if not isinstance(g, dict):
            continue
        role = str(g.get("role", "") or "")
        color = _GROUP_COLORS.get(role, (200, 200, 200))
        r = _norm_rect_to_pixels(g.get("bbox"), pw, ph)
        if r:
            dr.rectangle(r, outline=color, width=3)
            label = role
            if font:
                dr.text((r[0] + 2, max(0, r[1] - 11)), label, fill=color, font=font)
        for ch in g.get("children") or []:
            _draw_text_zone_child(dr, ch, pw, ph, font=font, depth=0)
    return out


def _draw_text_zone_child(
    dr: ImageDraw.ImageDraw,
    child: Any,
    pw: int,
    ph: int,
    *,
    font: Any,
    depth: int,
) -> None:
    if not isinstance(child, dict):
        return
    cr = _norm_rect_to_pixels(child.get("bbox"), pw, ph)
    if cr:
        x0, y0, x1, y1 = cr
        crole = str(child.get("role", "") or "")
        color = _CHILD_COLOR if depth == 0 else (80, 255, 255)
        _dashed_rect(dr, x0, y0, x1, y1, outline=color, width=2, dash=6)
        if font:
            t = (child.get("text") or "")[:24]
            lab = f"{crole}" + (f": {t}" if t else "")
            dr.text((x0 + 2, min(ph - 12, y1 + 1 + depth * 10)), lab, fill=color, font=font)
    for nested in child.get("children") or []:
        _draw_text_zone_child(dr, nested, pw, ph, font=font, depth=depth + 1)


def _dashed_rect(
    dr: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    outline: tuple[int, int, int],
    width: int,
    dash: int,
) -> None:
    """Simple axis-aligned dashed rectangle (PIL has no native dashed rects)."""

    def hline(y: int, xa: int, xb: int) -> None:
        if xa >= xb:
            return
        x = xa
        draw = True
        while x < xb:
            xn = min(xb, x + dash if draw else x + dash)
            if draw:
                dr.line([(x, y), (xn, y)], fill=outline, width=width)
            x = xn
            draw = not draw

    def vline(x: int, ya: int, yb: int) -> None:
        if ya >= yb:
            return
        y = ya
        draw = True
        while y < yb:
            yn = min(yb, y + dash if draw else y + dash)
            if draw:
                dr.line([(x, y), (x, yn)], fill=outline, width=width)
            y = yn
            draw = not draw

    for woff in range(width):
        hline(y0 + woff, x0, x1)
        hline(y1 - 1 - woff, x0, x1)
        vline(x0 + woff, y0, y1)
        vline(x1 - 1 - woff, y0, y1)


def draw_convert_updates_on_image(rgb: Image.Image, updates: list[dict[str, Any]]) -> Image.Image:
    """Draw ``LayoutUpdateItem.bounds`` (pixel coords in frame space)."""
    out = rgb.copy()
    dr = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, u in enumerate(updates):
        if not isinstance(u, dict):
            continue
        b = u.get("bounds")
        if not isinstance(b, dict):
            continue
        try:
            x0 = int(float(b.get("x", 0)))
            y0 = int(float(b.get("y", 0)))
            x1 = int(float(b.get("x", 0)) + float(b.get("width", 0)))
            y1 = int(float(b.get("y", 0)) + float(b.get("height", 0)))
        except (TypeError, ValueError):
            continue
        hue = (60 + (i * 47) % 200, (120 + i * 31) % 255, (200 - i * 13) % 255)
        dr.rectangle([x0, y0, x1, y1], outline=hue, width=2)
        if font:
            dr.text((x0 + 2, max(0, y0 - 10)), str(u.get("role", "") or ""), fill=hue, font=font)
    return out


def save_png(path: Path, im: Image.Image) -> None:
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    path.write_bytes(buf.getvalue())


def _draw_final_json_group(
    dr: ImageDraw.ImageDraw,
    item: Any,
    pw: int,
    ph: int,
    *,
    font: Any,
    depth: int = 0,
) -> None:
    if not isinstance(item, dict):
        return
    r = _norm_rect_to_pixels(item.get("bbox") if item.get("bbox") is not None else item, pw, ph)
    if r:
        color = (255, 220, 0) if depth else (0, 255, 180)
        dr.rectangle(r, outline=color, width=2 if depth else 3)
        if font:
            dr.text((r[0] + 2, max(0, r[1] - 11 + depth * 9)), str(item.get("role", "") or ""), fill=color, font=font)
    for child in item.get("children") or []:
        _draw_final_json_group(dr, child, pw, ph, font=font, depth=depth + 1)


def draw_final_json_on_image(rgb: Image.Image, final_json: dict[str, Any]) -> Image.Image:
    out = rgb.copy()
    dr = ImageDraw.Draw(out)
    pw, ph = out.size
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for group in final_json.get("groups") or final_json.get("children") or []:
        _draw_final_json_group(dr, group, pw, ph, font=font)
    return out


def persist_v2_call(
    storage: RunStorage,
    run_id: str,
    *,
    endpoint: str,
    raw_banner_bytes: bytes,
    response: ClassifyZoneResponse | AnalyzeTextZoneVisualResponse,
) -> None:
    """Write ``output/response.json`` and ``output/banner_annotate.png`` (bbox when present)."""
    out_dir = ensure_output_dir(storage, run_id)
    payload = response.model_dump(mode="json")
    _write_json(out_dir / "response.json", payload)
    _write_json(out_dir / "qwen_json.json", payload)
    final_json: dict[str, Any] | None = None
    mid_path = storage.get_input_dir(run_id) / "mid.json"
    if mid_path.exists() and isinstance(response, AnalyzeTextZoneVisualResponse):
        try:
            mid_json = json.loads(mid_path.read_text(encoding="utf-8"))
            final_json = build_final_json(mid_json, payload)
            _write_json(out_dir / "final_json.json", final_json)
        except Exception as exc:
            logger.warning("persist_v2_call: final_json failed run_id=%s: %s", run_id, exc)
    try:
        base = _decode_banner_rgb(raw_banner_bytes)
        if final_json is not None:
            ann = draw_final_json_on_image(base, final_json)
        elif isinstance(response, AnalyzeTextZoneVisualResponse):
            ann = draw_text_zone_on_image(base, response.text_zone)
        else:
            ann = base.copy()
        save_png(out_dir / "banner_annotate.png", ann)
    except Exception as exc:
        logger.warning("persist_v2_call: banner_annotate.png failed run_id=%s: %s", run_id, exc)


def persist_convert_call(
    storage: RunStorage,
    run_id: str,
    *,
    endpoint: str,
    response: Any,
    banner_png_bytes: bytes,
) -> None:
    out_dir = ensure_output_dir(storage, run_id)
    _write_json(out_dir / "response.json", response.model_dump(mode="json"))
    try:
        base = _decode_banner_rgb(banner_png_bytes)
        updates = [u.model_dump(mode="json") for u in response.updates]
        ann = draw_convert_updates_on_image(base, updates)
        save_png(out_dir / "banner_annotate.png", ann)
    except Exception as exc:
        logger.warning("persist_convert_call: banner_annotate.png failed run_id=%s: %s", run_id, exc)


def draw_candidates_json_on_image(rgb: Image.Image, candidates_payload: dict[str, Any]) -> Image.Image:
    """Draw ``bbox_canvas`` [x,y,w,h] normalized for each entry in ``all_candidates``."""
    out = rgb.copy()
    dr = ImageDraw.Draw(out)
    pw, ph = out.size
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    items = candidates_payload.get("all_candidates")
    if not isinstance(items, list):
        return out
    for i, c in enumerate(items):
        if not isinstance(c, dict):
            continue
        bc = c.get("bbox_canvas")
        if not isinstance(bc, (list, tuple)) or len(bc) < 4:
            continue
        try:
            x, y, w, h = (float(bc[0]), float(bc[1]), float(bc[2]), float(bc[3]))
        except (TypeError, ValueError):
            continue
        r = _norm_rect_to_pixels({"x": x, "y": y, "width": w, "height": h}, pw, ph)
        if not r:
            continue
        hue = (40 + (i * 53) % 215, 180, 255 - (i * 17) % 200)
        dr.rectangle(r, outline=hue, width=2)
        if font:
            lab = str(c.get("candidate_id", "") or "")[:12]
            dr.text((r[0] + 2, max(0, r[1] - 10)), lab, fill=hue, font=font)
    return out


def persist_multipart_pipeline_run(
    storage: RunStorage,
    run_id: str,
    *,
    endpoint: str,
    response_payload: dict[str, Any],
    banner_png_bytes: bytes,
) -> None:
    """After ``POST /api/run``: mirror summary to ``output/`` and annotate from ``04_candidates.json``."""
    out_dir = ensure_output_dir(storage, run_id)
    _write_json(out_dir / "response.json", response_payload)
    cand_path = storage.get_intermediate_dir(run_id) / "04_candidates.json"
    try:
        base = _decode_banner_rgb(banner_png_bytes)
        if cand_path.exists():
            data = json.loads(cand_path.read_text(encoding="utf-8"))
            ann = draw_candidates_json_on_image(base, data if isinstance(data, dict) else {})
        else:
            ann = base.copy()
        save_png(out_dir / "banner_annotate.png", ann)
    except Exception as exc:
        logger.warning("persist_multipart_pipeline_run: annotate failed run_id=%s: %s", run_id, exc)


def persist_api_error(
    storage: RunStorage,
    run_id: str,
    *,
    endpoint: str,
    error_type: str,
    detail: Any,
    raw_banner_bytes: bytes | None = None,
) -> None:
    out_dir = ensure_output_dir(storage, run_id)
    _write_json(
        out_dir / "error.json",
        {"endpoint": endpoint, "error_type": error_type, "detail": detail},
    )
    if raw_banner_bytes:
        try:
            save_png(out_dir / "banner_annotate.png", _decode_banner_rgb(raw_banner_bytes))
        except Exception as exc:
            logger.warning("persist_api_error: banner copy failed run_id=%s: %s", run_id, exc)
