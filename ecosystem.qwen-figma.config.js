/**
 * PM2 **v1** stack — process names ``qwen-figma-backend`` / ``qwen-figma-service``.
 * Ports come from repo-root ``.env`` (typically Qwen **3035**, backend **3037**).
 * Set QWEN_MODEL_PATH / QWEN_VL_FAMILY there, then: ``pm2 restart qwen-figma-service``
 *
 * **v2** stack uses the same ports but ``*-v2`` names: ``ecosystem.qwen-figma.v2.config.js``.
 * Run only one ecosystem at a time unless you change ports in one file.
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
