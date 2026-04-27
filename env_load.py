"""Load repository-root `.env` (non-destructive: existing OS env wins)."""
from __future__ import annotations

from pathlib import Path

_loaded = False

DEFAULT_QWEN_BASE_URL = "http://127.0.0.1:30078"


def load_project_env() -> None:
    global _loaded
    if _loaded:
        return
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env", override=False)
    _loaded = True


def default_qwen_base_url() -> str:
    """Base URL for Qwen HTTP API (loads `.env` on first use if not already loaded)."""
    import os

    load_project_env()
    return (os.environ.get("QWEN_BASE_URL") or DEFAULT_QWEN_BASE_URL).rstrip("/")
