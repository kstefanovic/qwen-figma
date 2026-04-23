module.exports = {
  apps: [
    {
      name: "qwen-figma-backend",
      cwd: "/root/figma/qwen-figma",
      script: "/root/figma/qwen-figma/.venv/bin/uvicorn",
      args: "backend.app:app --host 0.0.0.0 --port 10197",
      interpreter: "none",
      env: {
        QWEN_BASE_URL: "http://127.0.0.1:10196",
        PYTHONUNBUFFERED: "1"
      },
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 3000,
      time: true
    },
    {
      name: "qwen-figma-service",
      cwd: "/root/figma/qwen-figma",
      script: "/root/figma/qwen-figma/.venv/bin/python",
      args: "-m qwen_service.run_server",
      interpreter: "none",
      env: {
        QWEN_MODEL_PATH: "Qwen2.5-VL-7B-Instruct",
        QWEN_DEVICE: "cuda:1",
        QWEN_HOST: "127.0.0.1",
        QWEN_PORT: "10196",
        PYTHONUNBUFFERED: "1"
      },
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 5000,
      time: true
    }
  ]
}