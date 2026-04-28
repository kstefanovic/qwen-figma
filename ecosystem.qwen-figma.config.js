/**
 * PM2 apps load repo-root `.env` inside Python via `env_load.load_project_env()`
 * (see `qwen_service/run_server.py`). Set QWEN_MODEL_PATH / QWEN_VL_FAMILY there, then:
 *   pm2 restart qwen-figma-service
 */
const path = require("path");

const root = __dirname;
const py = path.join(root, ".venv", "bin", "python");

module.exports = {
  apps: [
    {
      name: "qwen-figma-backend",
      cwd: root,
      script: py,
      args: "run_backend.py",
      interpreter: "none",
      env: {
        PYTHONUNBUFFERED: "1",
      },
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 3000,
      time: true,
    },
    {
      name: "qwen-figma-service",
      cwd: root,
      script: py,
      args: "-m qwen_service.run_server",
      interpreter: "none",
      env: {
        PYTHONUNBUFFERED: "1",
      },
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 5000,
      time: true,
    },
  ],
};
