import logging
import os
import sys

import uvicorn

from env_load import load_project_env

# For the model service, repo `.env` is the source of truth. PM2 can keep stale
# environment values across restarts, so force `.env` to override inherited keys.
load_project_env(override=True)

from qwen_service.app import create_app, resolve_qwen_vl_family

logger = logging.getLogger(__name__)


MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "Qwen2.5-VL-7B-Instruct")
# qwen2_5 | qwen3 — empty = infer from QWEN_MODEL_PATH (e.g. Qwen3-VL-* → qwen3)
VL_FAMILY = os.environ.get("QWEN_VL_FAMILY", "").strip()
DEVICE = os.environ.get("QWEN_DEVICE", "cuda:1")
HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
PORT = int(os.environ.get("QWEN_PORT", "30078"))
MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "768"))
TEMPERATURE = float(os.environ.get("QWEN_TEMPERATURE", "0.0"))
MAX_IMAGE_LONG_SIDE = int(os.environ.get("QWEN_MAX_IMAGE_LONG_SIDE", "1024"))

_resolved_vl = resolve_qwen_vl_family(MODEL_PATH, VL_FAMILY or None)
_vl_env = VL_FAMILY if VL_FAMILY else "(auto from QWEN_MODEL_PATH)"
_startup_line = (
    f"qwen-figma-service starting: vl_family={_resolved_vl} "
    f"QWEN_VL_FAMILY={_vl_env!r} QWEN_MODEL_PATH={MODEL_PATH!r} "
    f"QWEN_DEVICE={DEVICE!r} listen={HOST}:{PORT}"
)
# stdout → PM2 *-out.log; stderr → *-error.log (many users only tail error.log)
print(_startup_line, flush=True)
print(_startup_line, file=sys.stderr, flush=True)
logger.info("%s", _startup_line)


app = create_app(
    model_path=MODEL_PATH,
    device=DEVICE,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    max_image_long_side=MAX_IMAGE_LONG_SIDE,
    vl_family=VL_FAMILY or None,
)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
