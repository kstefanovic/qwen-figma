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
