from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class RunStorage:
    def __init__(self, base_dir: str | Path = "runs") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run(self) -> str:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = self.get_run_dir(run_id)
        self._ensure_layout(run_dir)

        now = self._now_iso()
        meta = {
            "run_id": run_id,
            "status": "created",
            "created_at": now,
            "updated_at": now,
            "input_files": {
                "banner_image": None,
                "raw_json": None,
            },
            "output_files": {
                "semantic_graph": None,
                "validation_report": None,
                "intermediate_dir": None,
                "final_dir": None,
            },
            "metadata": {},
            "error": None,
        }
        self.write_meta(run_id, meta)
        return run_id

    def get_run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id

    def get_input_dir(self, run_id: str) -> Path:
        return self.get_run_dir(run_id) / "input"

    def get_intermediate_dir(self, run_id: str) -> Path:
        return self.get_run_dir(run_id) / "intermediate"

    def get_final_dir(self, run_id: str) -> Path:
        return self.get_run_dir(run_id) / "final"

    def get_meta_path(self, run_id: str) -> Path:
        return self.get_run_dir(run_id) / "run_meta.json"

    def get_log_path(self, run_id: str) -> Path:
        return self.get_run_dir(run_id) / "run.log"

    def exists(self, run_id: str) -> bool:
        return self.get_run_dir(run_id).exists()

    def save_input_file(
        self,
        run_id: str,
        source_path: str | Path,
        target_name: str,
    ) -> str:
        source_path = Path(source_path)
        target_path = self.get_input_dir(run_id) / target_name
        shutil.copy2(source_path, target_path)
        return str(target_path)

    def save_upload_bytes(
        self,
        run_id: str,
        filename: str,
        content: bytes,
    ) -> str:
        target_path = self.get_input_dir(run_id) / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(content)
        return str(target_path.resolve())

    def read_meta(self, run_id: str) -> dict[str, Any]:
        meta_path = self.get_meta_path(run_id)
        if not meta_path.exists():
            raise FileNotFoundError(f"Run metadata not found for run_id={run_id}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def write_meta(self, run_id: str, meta: dict[str, Any]) -> None:
        meta_path = self.get_meta_path(run_id)
        meta["updated_at"] = self._now_iso()
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def update_meta(
        self,
        run_id: str,
        *,
        status: Optional[str] = None,
        input_files: Optional[dict[str, Optional[str]]] = None,
        output_files: Optional[dict[str, Optional[str]]] = None,
        metadata: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> dict[str, Any]:
        meta = self.read_meta(run_id)

        if status is not None:
            meta["status"] = status
        if input_files is not None:
            meta["input_files"].update(input_files)
        if output_files is not None:
            meta["output_files"].update(output_files)
        if metadata is not None:
            meta["metadata"].update(metadata)
        if error is not None:
            meta["error"] = error

        self.write_meta(run_id, meta)
        return meta

    def append_log(self, run_id: str, text: str) -> None:
        log_path = self.get_log_path(run_id)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")

    def read_json_file(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for run_dir in sorted(self.base_dir.glob("run_*"), reverse=True):
            meta_path = run_dir / "run_meta.json"
            if meta_path.exists():
                try:
                    items.append(json.loads(meta_path.read_text(encoding="utf-8")))
                except Exception:
                    continue
            if len(items) >= limit:
                break
        return items

    def _ensure_layout(self, run_dir: Path) -> None:
        (run_dir / "input").mkdir(parents=True, exist_ok=True)
        (run_dir / "intermediate").mkdir(parents=True, exist_ok=True)
        (run_dir / "final").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()