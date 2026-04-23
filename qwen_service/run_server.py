import os

import uvicorn

from qwen_service.app import create_app


MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "Qwen2.5-VL-7B-Instruct")
DEVICE = os.environ.get("QWEN_DEVICE", "cuda:1")
HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
PORT = int(os.environ.get("QWEN_PORT", "10196"))
MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("QWEN_TEMPERATURE", "0.1"))


app = create_app(
    model_path=MODEL_PATH,
    device=DEVICE,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
