/**
 * PM2 ecosystem for the **v2** stack (zone classification + shared codebase).
 *
 * Binds to the same ports as repo-root ``.env`` by default:
 *   - Qwen HTTP: **3035** (``QWEN_PORT``)
 *   - Backend:   **3037** (``BACKEND_PORT``, ``QWEN_BASE_URL`` → Qwen)
 *
 * Process names are ``*-v2`` so you can tell them apart in ``pm2 ls``. Do **not** run this
 * alongside ``ecosystem.qwen-figma.config.js`` on the same machine with the same ports —
 * stop the v1 apps first, or change ports in one of the ecosystem files.
 *
 *   pm2 start ecosystem.qwen-figma.v2.config.js
 *
 * Plugin Backend URL: ``http://127.0.0.1:3037`` or ``http://localhost:3037``
 */
const path = require("path");

const root = __dirname;
const py = path.join(root, ".venv", "bin", "python");

const V2_QWEN_PORT = 3035;
const V2_BACKEND_PORT = 3037;

module.exports = {
  apps: [
    {
      name: "qwen-figma-backend-v2",
      cwd: root,
      script: py,
      args: "run_backend.py",
      interpreter: "none",
      env: {
        PYTHONUNBUFFERED: "1",
        BACKEND_HOST: "0.0.0.0",
        BACKEND_PORT: String(V2_BACKEND_PORT),
        QWEN_BASE_URL: `http://127.0.0.1:${V2_QWEN_PORT}`,
      },
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 3000,
      time: true,
    },
    {
      name: "qwen-figma-service-v2",
      cwd: root,
      script: py,
      args: "-m qwen_service.run_server",
      interpreter: "none",
      env: {
        PYTHONUNBUFFERED: "1",
        QWEN_HOST: "127.0.0.1",
        QWEN_PORT: String(V2_QWEN_PORT),
      },
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 5000,
      time: true,
    },
  ],
};
