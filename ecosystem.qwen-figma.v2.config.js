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
const fs = require("fs");
const path = require("path");

const root = __dirname;
const py = path.join(root, ".venv", "bin", "python");

const V2_QWEN_PORT = 40038;
const V2_BACKEND_PORT = 40039;

function readDotEnv(filePath) {
  const env = {};
  if (!fs.existsSync(filePath)) return env;
  for (const rawLine of fs.readFileSync(filePath, "utf8").split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq <= 0) continue;
    const key = line.slice(0, eq).trim();
    let value = line.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    env[key] = value;
  }
  return env;
}

const dotEnv = readDotEnv(path.join(root, ".env"));
const qwenPort = dotEnv.QWEN_PORT || String(V2_QWEN_PORT);
const backendPort = dotEnv.BACKEND_PORT || String(V2_BACKEND_PORT);
const qwenEnv = {
  PYTHONUNBUFFERED: "1",
  QWEN_HOST: dotEnv.QWEN_HOST || "127.0.0.1",
  QWEN_PORT: qwenPort,
  QWEN_MODEL_PATH: dotEnv.QWEN_MODEL_PATH || "Qwen2.5-VL-7B-Instruct",
  QWEN_DEVICE: dotEnv.QWEN_DEVICE || "cuda:0",
  QWEN_MAX_NEW_TOKENS: dotEnv.QWEN_MAX_NEW_TOKENS || "768",
  QWEN_TEMPERATURE: dotEnv.QWEN_TEMPERATURE || "0.0",
  QWEN_MAX_IMAGE_LONG_SIDE: dotEnv.QWEN_MAX_IMAGE_LONG_SIDE || "1024",
  USE_QWEN_BBOX_REFINEMENT: dotEnv.USE_QWEN_BBOX_REFINEMENT || "1",
};

if (dotEnv.QWEN_VL_FAMILY) {
  qwenEnv.QWEN_VL_FAMILY = dotEnv.QWEN_VL_FAMILY;
}

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
        BACKEND_HOST: dotEnv.BACKEND_HOST || "0.0.0.0",
        BACKEND_PORT: backendPort,
        QWEN_BASE_URL: dotEnv.QWEN_BASE_URL || `http://127.0.0.1:${qwenPort}`,
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
      env: qwenEnv,
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 5000,
      time: true,
    },
  ],
};
