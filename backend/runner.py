from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from backend.storage import RunStorage
from main import run_pipeline


class PipelineRunner:
    def __init__(self, storage: RunStorage, qwen_base_url: str = "http://127.0.0.1:8001") -> None:
        self.storage = storage
        self.qwen_base_url = qwen_base_url

    def run(
        self,
        *,
        run_id: str,
        raw_json_path: str | Path,
        banner_image_path: str | Path,
        brand_family: str = "unknown_brand",
        language: str = "unknown",
        category: str = "unknown",
        use_qwen: bool = True,
    ) -> dict[str, Any]:
        try:
            self.storage.update_meta(
                run_id,
                status="running",
                metadata={
                    "brand_family": brand_family,
                    "language": language,
                    "category": category,
                    "use_qwen": use_qwen,
                    "qwen_base_url": self.qwen_base_url if use_qwen else None,
                },
            )

            self.storage.append_log(run_id, "Pipeline started.")

            output_dir = self.storage.get_run_dir(run_id)

            result = run_pipeline(
                raw_json_path=str(raw_json_path),
                banner_image_path=str(banner_image_path),
                output_dir=str(output_dir),
                use_qwen=use_qwen,
                qwen_base_url=self.qwen_base_url,
                default_brand_family=brand_family,
                default_language=language,
                default_category=category,
            )

            semantic_graph_path = self.storage.get_final_dir(run_id) / "semantic_graph.json"
            validation_report_path = self.storage.get_final_dir(run_id) / "validation_report.json"

            self.storage.update_meta(
                run_id,
                status="completed",
                output_files={
                    "semantic_graph": str(semantic_graph_path),
                    "validation_report": str(validation_report_path),
                    "intermediate_dir": str(self.storage.get_intermediate_dir(run_id)),
                    "final_dir": str(self.storage.get_final_dir(run_id)),
                },
                error=None,
            )

            self.storage.append_log(run_id, "Pipeline completed successfully.")
            return result

        except Exception as e:
            error_text = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()

            self.storage.append_log(run_id, error_text)
            self.storage.append_log(run_id, tb)
            self.storage.update_meta(
                run_id,
                status="failed",
                error=error_text,
            )
            raise