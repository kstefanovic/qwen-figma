"""Run the gateway FastAPI app; host/port from BACKEND_HOST / BACKEND_PORT (.env)."""
from __future__ import annotations

import os

import uvicorn

from env_load import load_project_env

load_project_env()

if __name__ == "__main__":
    host = os.environ.get("BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("BACKEND_PORT", "30079"))
    uvicorn.run("backend.app:app", host=host, port=port)
