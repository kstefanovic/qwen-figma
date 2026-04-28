from __future__ import annotations

import io
import logging
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]

_DEFAULT_MAX_LONG_SIDE = 1280
_JPEG_QUALITY = 85


def decode_banner_raster(raw_bytes: bytes) -> Image.Image:
    """Decode upload bytes to a Pillow image (caller may close underlying buffer)."""
    im = Image.open(io.BytesIO(raw_bytes))
    im.load()
    return im


def _needs_png_raster(image: Image.Image) -> bool:
    if image.mode in ("RGBA", "LA"):
        return True
    if image.mode == "P":
        tr = image.info.get("transparency")
        return tr is not None
    return False


def resize_and_encode_for_zone_qwen(
    image: Image.Image,
    *,
    max_long_side: int = _DEFAULT_MAX_LONG_SIDE,
    jpeg_quality: int = _JPEG_QUALITY,
) -> tuple[bytes, dict[str, Any]]:
    """
    Resize so max(w,h) <= max_long_side, encode as JPEG (quality) or PNG when alpha requires it.

    Returns (encoded_bytes, meta) including original and prepared pixel sizes.
    """
    orig_w, orig_h = image.size
    work = image
    if work.mode not in ("RGB", "RGBA", "L", "P", "LA"):
        work = work.convert("RGBA" if "A" in work.mode else "RGB")

    m = max(work.width, work.height)
    if m > max_long_side and m > 0:
        scale = max_long_side / float(m)
        nw = max(1, int(round(work.width * scale)))
        nh = max(1, int(round(work.height * scale)))
        work = work.resize((nw, nh), _LANCZOS)

    use_png = _needs_png_raster(work)
    buf = io.BytesIO()
    if use_png:
        if work.mode != "RGBA":
            work = work.convert("RGBA")
        work.save(buf, format="PNG", optimize=True)
        fmt = "png"
    else:
        rgb = work.convert("RGB")
        rgb.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        fmt = "jpeg"

    out = buf.getvalue()
    meta: dict[str, Any] = {
        "original_width": orig_w,
        "original_height": orig_h,
        "prepared_width": work.width,
        "prepared_height": work.height,
        "output_format": fmt,
        "output_bytes": len(out),
    }
    logger.info(
        "pipeline_v2 resize_image: original=%dx%d prepared=%dx%d format=%s bytes=%d",
        orig_w,
        orig_h,
        work.width,
        work.height,
        fmt,
        len(out),
    )
    return out, meta
