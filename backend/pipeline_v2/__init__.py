"""Independent v2 pipelines (no legacy candidate/heuristic paths)."""

from backend.pipeline_v2.qwen_zone_classifier import classify_zone_from_banner_bytes

__all__ = ["classify_zone_from_banner_bytes"]
