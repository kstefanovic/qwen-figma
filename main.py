from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from env_load import default_qwen_base_url, load_project_env

load_project_env()

from banner_pipeline.build_candidates import CandidateBundle, build_candidates
from banner_pipeline.collapse_groups import collapse_wrapper_groups
from banner_pipeline.figma_summary import build_figma_summary, build_qwen_scene_payload
from banner_pipeline.heuristics import apply_heuristics
from banner_pipeline.merge_semantic_graph import MergeConfig, merge_semantic_graph
from banner_pipeline.normalize import normalize_nodes
from banner_pipeline.parse_figma import parse_figma_file, save_flat_nodes
from banner_pipeline.qwen_annotator import QwenAnnotator
from banner_pipeline.validate_graph import validate_graph


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_model_json(model_obj: Any, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model_obj, "model_dump"):
        payload = model_obj.model_dump(mode="json", exclude_none=True)
    elif hasattr(model_obj, "to_dict"):
        payload = model_obj.to_dict()
    else:
        payload = model_obj

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _normalize_qwen_mode(use_qwen: bool, qwen_mode: str | None) -> str:
    if not use_qwen:
        return "off"
    m = (qwen_mode or "single_pass").strip().lower()
    if m not in {"single_pass", "scene_only", "per_candidate", "off"}:
        return "single_pass"
    return m


def _normalize_pipeline_mode(pipeline_mode: str | None) -> str:
    m = (pipeline_mode or "full_layout_debug").strip().lower()
    if m not in {"apply_to_clone_fast", "apply_to_clone_vlm", "full_layout_debug"}:
        return "full_layout_debug"
    return m


def _flag_use_qwen_scene_only() -> bool:
    raw = os.getenv("USE_QWEN_SCENE_ONLY", "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _flag_debug_save_artifacts() -> bool:
    raw = os.getenv("DEBUG_SAVE_ARTIFACTS", "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _should_save_intermediate_artifacts(pipeline_mode: str) -> bool:
    if pipeline_mode in {"apply_to_clone_fast", "apply_to_clone_vlm"}:
        return False
    return _flag_debug_save_artifacts()


def _count_per_candidate_group_calls(candidate_bundle: CandidateBundle) -> int:
    seen: set[str] = set()
    n = 0
    for candidate in (
        *candidate_bundle.text_group_candidates,
        *candidate_bundle.brand_candidates,
        *candidate_bundle.decoration_candidates,
        *candidate_bundle.background_candidates,
        *candidate_bundle.image_like_candidates,
    ):
        if candidate.candidate_id not in seen:
            seen.add(candidate.candidate_id)
            n += 1
    return n


def _print_stage_timings(stage_timings: dict[str, float]) -> None:
    ordered = [
        "parse_figma",
        "normalize",
        "compact_tree",
        "heuristics",
        "qwen_scene",
        "merge",
        "validate",
        "save_debug",
        "total",
    ]
    print("Stage timings (seconds):")
    for key in ordered:
        print(f"  {key:>11}: {stage_timings.get(key, 0.0):.3f}")


def run_pipeline(
    *,
    raw_json_path: str | Path,
    banner_image_path: str | Path,
    output_dir: str | Path,
    use_qwen: bool = False,
    qwen_base_url: str | None = None,
    brand_family: str | None = None,
    language: str | None = None,
    category: str | None = None,
    qwen_mode: str | None = "single_pass",
    pipeline_mode: str = "full_layout_debug",
) -> dict[str, Any]:
    resolved_qwen_base = ((qwen_base_url or "").strip() or default_qwen_base_url()).rstrip("/")
    total_started = time.perf_counter()
    output_dir = ensure_dir(output_dir)
    intermediate_dir = ensure_dir(output_dir / "intermediate")
    final_dir = ensure_dir(output_dir / "final")
    effective_pipeline_mode = _normalize_pipeline_mode(pipeline_mode)
    save_intermediate_artifacts = _should_save_intermediate_artifacts(effective_pipeline_mode)
    stage_timings = {
        "parse_figma": 0.0,
        "normalize": 0.0,
        "compact_tree": 0.0,
        "heuristics": 0.0,
        "qwen_scene": 0.0,
        "merge": 0.0,
        "validate": 0.0,
        "save_debug": 0.0,
    }

    t0 = time.perf_counter()
    doc, root, parsed_nodes = parse_figma_file(raw_json_path)
    stage_timings["parse_figma"] = time.perf_counter() - t0
    canvas_w = int(root.bounds.width)
    canvas_h = int(root.bounds.height)
    if save_intermediate_artifacts:
        td = time.perf_counter()
        save_flat_nodes(parsed_nodes, intermediate_dir / "01_flat_nodes.json")
        stage_timings["save_debug"] += time.perf_counter() - td

    t0 = time.perf_counter()
    normalized_nodes = normalize_nodes(parsed_nodes, canvas_w, canvas_h)
    stage_timings["normalize"] = time.perf_counter() - t0
    if save_intermediate_artifacts:
        td = time.perf_counter()
        save_json([node.to_dict() for node in normalized_nodes], intermediate_dir / "02_normalized_nodes.json")
        stage_timings["save_debug"] += time.perf_counter() - td

    t0 = time.perf_counter()
    collapsed_nodes = collapse_wrapper_groups(normalized_nodes)
    candidate_bundle = build_candidates(collapsed_nodes)
    stage_timings["compact_tree"] = time.perf_counter() - t0
    if save_intermediate_artifacts:
        td = time.perf_counter()
        save_json([node.to_dict() for node in collapsed_nodes], intermediate_dir / "03_collapsed_nodes.json")
        save_json(candidate_bundle.to_dict(), intermediate_dir / "04_candidates.json")
        stage_timings["save_debug"] += time.perf_counter() - td

    t0 = time.perf_counter()
    heuristic_bundle = apply_heuristics(candidate_bundle, collapsed_nodes)
    stage_timings["heuristics"] = time.perf_counter() - t0
    if save_intermediate_artifacts:
        td = time.perf_counter()
        save_json(heuristic_bundle.to_dict(), intermediate_dir / "05_heuristics.json")
        stage_timings["save_debug"] += time.perf_counter() - td

    banner_annotation = None
    candidate_annotations = {}
    group_annotations = {}
    brand_context_annotation = None

    requested_qwen_mode = _normalize_qwen_mode(use_qwen, qwen_mode)
    effective_qwen_mode = requested_qwen_mode
    if effective_pipeline_mode == "apply_to_clone_fast":
        effective_qwen_mode = "off"
    elif effective_pipeline_mode == "apply_to_clone_vlm":
        effective_qwen_mode = "scene_only"

    use_qwen_scene_only = effective_qwen_mode == "scene_only" or (
        effective_qwen_mode == "single_pass" and _flag_use_qwen_scene_only()
    )
    qwen_calls_made = 0
    candidates_annotated = 0
    groups_annotated = 0
    qwen_elapsed_seconds = 0.0
    qwen_request_metrics: list[dict[str, Any]] = []

    if effective_qwen_mode == "off":
        resolved_brand_family = brand_family if brand_family is not None else "generic"
        resolved_language = language if language is not None else "unknown"
        resolved_category = category if category is not None else "unknown"
    elif effective_qwen_mode in {"single_pass", "scene_only"} and use_qwen_scene_only:
        t0 = time.perf_counter()
        annotator = QwenAnnotator(base_url=resolved_qwen_base)
        annotator.load_model()

        figma_summary = build_figma_summary(
            collapsed_nodes,
            candidate_bundle,
            heuristic_bundle,
            canvas_w,
            canvas_h,
        )
        if save_intermediate_artifacts:
            td = time.perf_counter()
            save_json(figma_summary, intermediate_dir / "05a_figma_summary.json")
            stage_timings["save_debug"] += time.perf_counter() - td

        scene_payload = build_qwen_scene_payload(
            figma_summary=figma_summary,
            collapsed_nodes=collapsed_nodes,
            candidate_bundle=candidate_bundle,
            heuristic_bundle=heuristic_bundle,
        )
        if save_intermediate_artifacts:
            td = time.perf_counter()
            save_json(scene_payload, intermediate_dir / "05a_scene_payload.json")
            stage_timings["save_debug"] += time.perf_counter() - td

        struct = annotator.annotate_scene(
            banner_image_path=str(banner_image_path),
            scene_payload=scene_payload,
        )
        brand_context_annotation = struct.brand_context
        banner_annotation = struct.banner_annotation
        candidate_annotations = struct.candidate_annotations
        group_annotations = struct.group_annotations
        qwen_calls_made = annotator.qwen_call_count
        candidates_annotated = struct.candidates_annotated
        groups_annotated = struct.groups_annotated
        qwen_request_metrics = annotator.request_metrics_dicts()

        td = time.perf_counter()
        save_json(
            {
                "updates": struct.semantic_updates,
                "groups": struct.semantic_groups,
            },
            intermediate_dir / "06b_scene_semantics.json",
        )
        stage_timings["save_debug"] += time.perf_counter() - td

        if save_intermediate_artifacts:
            td = time.perf_counter()
            save_json(brand_context_annotation.to_dict(), intermediate_dir / "05b_brand_context.json")
            save_json(banner_annotation.to_dict(), intermediate_dir / "06_banner_annotation.json")
            save_json({k: v.to_dict() for k, v in candidate_annotations.items()}, intermediate_dir / "07_candidate_annotations.json")
            save_json({k: v.to_dict() for k, v in group_annotations.items()}, intermediate_dir / "08_group_annotations.json")
            stage_timings["save_debug"] += time.perf_counter() - td

        inferred_brand = brand_context_annotation.brand_family
        inferred_language = brand_context_annotation.language
        inferred_category = brand_context_annotation.category

        resolved_brand_family = brand_family if brand_family is not None else inferred_brand
        resolved_language = language if language is not None else inferred_language
        resolved_category = category if category is not None else inferred_category
        qwen_elapsed_seconds = time.perf_counter() - t0
        stage_timings["qwen_scene"] = qwen_elapsed_seconds
    elif effective_qwen_mode == "single_pass":
        # fallback when USE_QWEN_SCENE_ONLY is disabled
        t0 = time.perf_counter()
        annotator = QwenAnnotator(base_url=resolved_qwen_base)
        annotator.load_model()

        brand_context_annotation = annotator.annotate_brand_context(
            banner_image_path=str(banner_image_path),
            candidate_bundle=candidate_bundle,
            heuristic_bundle=heuristic_bundle,
        )
        if save_intermediate_artifacts:
            td = time.perf_counter()
            save_json(brand_context_annotation.to_dict(), intermediate_dir / "05b_brand_context.json")
            stage_timings["save_debug"] += time.perf_counter() - td

        inferred_brand = brand_context_annotation.brand_family
        inferred_language = brand_context_annotation.language
        inferred_category = brand_context_annotation.category

        resolved_brand_family = brand_family if brand_family is not None else inferred_brand
        resolved_language = language if language is not None else inferred_language
        resolved_category = category if category is not None else inferred_category

        banner_annotation = annotator.annotate_banner(
            banner_image_path=str(banner_image_path),
            candidate_bundle=candidate_bundle,
            heuristic_bundle=heuristic_bundle,
        )
        if save_intermediate_artifacts:
            td = time.perf_counter()
            save_json(banner_annotation.to_dict(), intermediate_dir / "06_banner_annotation.json")
            stage_timings["save_debug"] += time.perf_counter() - td

        candidate_annotations = annotator.annotate_candidates(
            banner_image_path=str(banner_image_path),
            candidate_bundle=candidate_bundle,
            heuristic_bundle=heuristic_bundle,
        )
        if save_intermediate_artifacts:
            td = time.perf_counter()
            save_json({k: v.to_dict() for k, v in candidate_annotations.items()}, intermediate_dir / "07_candidate_annotations.json")
            stage_timings["save_debug"] += time.perf_counter() - td

        group_annotations = annotator.annotate_group_candidates(
            banner_image_path=str(banner_image_path),
            candidate_bundle=candidate_bundle,
            heuristic_bundle=heuristic_bundle,
        )
        if save_intermediate_artifacts:
            td = time.perf_counter()
            save_json({k: v.to_dict() for k, v in group_annotations.items()}, intermediate_dir / "08_group_annotations.json")
            stage_timings["save_debug"] += time.perf_counter() - td

        qwen_calls_made = annotator.qwen_call_count
        candidates_annotated = len(candidate_annotations)
        groups_annotated = len(group_annotations)
        qwen_elapsed_seconds = time.perf_counter() - t0
        qwen_request_metrics = annotator.request_metrics_dicts()
        stage_timings["qwen_scene"] = qwen_elapsed_seconds
    else:
        # per_candidate legacy: brand + banner + each candidate + group-like candidates
        t0 = time.perf_counter()
        annotator = QwenAnnotator(base_url=resolved_qwen_base)
        annotator.load_model()

        brand_context_annotation = annotator.annotate_brand_context(
            banner_image_path=str(banner_image_path),
            candidate_bundle=candidate_bundle,
            heuristic_bundle=heuristic_bundle,
        )
        if save_intermediate_artifacts:
            td = time.perf_counter()
            save_json(brand_context_annotation.to_dict(), intermediate_dir / "05b_brand_context.json")
            stage_timings["save_debug"] += time.perf_counter() - td

        inferred_brand = brand_context_annotation.brand_family
        inferred_language = brand_context_annotation.language
        inferred_category = brand_context_annotation.category

        resolved_brand_family = brand_family if brand_family is not None else inferred_brand
        resolved_language = language if language is not None else inferred_language
        resolved_category = category if category is not None else inferred_category

        banner_annotation = annotator.annotate_banner(
            banner_image_path=str(banner_image_path),
            candidate_bundle=candidate_bundle,
            heuristic_bundle=heuristic_bundle,
        )
        if save_intermediate_artifacts:
            td = time.perf_counter()
            save_json(banner_annotation.to_dict(), intermediate_dir / "06_banner_annotation.json")
            stage_timings["save_debug"] += time.perf_counter() - td

        candidate_annotations = annotator.annotate_candidates(
            banner_image_path=str(banner_image_path),
            candidate_bundle=candidate_bundle,
            heuristic_bundle=heuristic_bundle,
        )
        if save_intermediate_artifacts:
            td = time.perf_counter()
            save_json({k: v.to_dict() for k, v in candidate_annotations.items()}, intermediate_dir / "07_candidate_annotations.json")
            stage_timings["save_debug"] += time.perf_counter() - td

        group_annotations = annotator.annotate_group_candidates(
            banner_image_path=str(banner_image_path),
            candidate_bundle=candidate_bundle,
            heuristic_bundle=heuristic_bundle,
        )
        if save_intermediate_artifacts:
            td = time.perf_counter()
            save_json({k: v.to_dict() for k, v in group_annotations.items()}, intermediate_dir / "08_group_annotations.json")
            stage_timings["save_debug"] += time.perf_counter() - td

        qwen_calls_made = annotator.qwen_call_count
        candidates_annotated = len(candidate_annotations)
        groups_annotated = len(group_annotations)
        qwen_elapsed_seconds = time.perf_counter() - t0
        qwen_request_metrics = annotator.request_metrics_dicts()
        stage_timings["qwen_scene"] = qwen_elapsed_seconds

    if effective_pipeline_mode == "apply_to_clone_fast" and qwen_calls_made != 0:
        raise RuntimeError(
            f"apply_to_clone_fast exceeded Qwen guard: expected 0 calls, got {qwen_calls_made}"
        )
    if effective_pipeline_mode == "apply_to_clone_vlm" and qwen_calls_made > 1:
        raise RuntimeError(
            f"apply_to_clone_vlm exceeded Qwen guard: expected <=1 call, got {qwen_calls_made}"
        )

    config = MergeConfig(
        default_brand_family=resolved_brand_family,
        default_language=resolved_language,
        default_category=resolved_category,
    )

    t0 = time.perf_counter()
    semantic_graph = merge_semantic_graph(
        banner_id=root.id,
        template_id=root.templateId,
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        raw_figma_frame_id=root.id,
        collapsed_nodes=collapsed_nodes,
        candidate_bundle=candidate_bundle,
        heuristic_bundle=heuristic_bundle,
        banner_annotation=banner_annotation,
        qwen_candidate_annotations=candidate_annotations,
        qwen_group_annotations=group_annotations,
        config=config,
    )
    stage_timings["merge"] = time.perf_counter() - t0

    save_model_json(semantic_graph, final_dir / "semantic_graph.json")

    t0 = time.perf_counter()
    validation_report = validate_graph(semantic_graph)
    stage_timings["validate"] = time.perf_counter() - t0
    save_json(validation_report.to_dict(), final_dir / "validation_report.json")
    stage_timings["total"] = time.perf_counter() - total_started

    print("=" * 70)
    print("Pipeline finished")
    print(f"Raw JSON:      {raw_json_path}")
    print(f"Banner image:  {banner_image_path}")
    print(f"Output dir:    {output_dir}")
    print("-" * 70)
    print(f"Pipeline mode: {effective_pipeline_mode}")
    print(f"Qwen mode:     {effective_qwen_mode}")
    print(f"Qwen scene-only flag: {use_qwen_scene_only}")
    print(f"qwen_call_count: {qwen_calls_made}")
    print(f"qwen_elapsed_seconds: {qwen_elapsed_seconds:.3f}")
    print(f"Qwen cand ann: {candidates_annotated}")
    print(f"Qwen grp ann:  {groups_annotated}")
    print(f"Canvas size:   {canvas_w} x {canvas_h}")
    print(f"Brand context: brand_family={resolved_brand_family} language={resolved_language} category={resolved_category}")
    print(f"Parsed nodes:  {len(parsed_nodes)}")
    print(f"Collapsed:     {len(collapsed_nodes)}")
    print(f"Candidates:    {len(candidate_bundle.all_candidates)}")
    print(f"Groups:        {len(semantic_graph.groups)}")
    print(f"Elements:      {len(semantic_graph.elements)}")
    print(f"Relations:     {len(semantic_graph.relations)}")
    print(f"Constraints:   {len(semantic_graph.constraints)}")
    print("-" * 70)
    print(f"Graph valid:   {validation_report.is_valid}")
    print(f"Errors:        {len(validation_report.errors)}")
    print(f"Warnings:      {len(validation_report.warnings)}")
    _print_stage_timings(stage_timings)
    if qwen_request_metrics:
        print("Qwen request metrics:")
        for metric in qwen_request_metrics:
            print(
                "  - endpoint={endpoint} elapsed_seconds={elapsed_seconds:.3f} "
                "payload_size_bytes={payload_size_bytes} image_size={image_size}".format(**metric)
            )
    print("=" * 70)

    return {
        "semantic_graph": semantic_graph,
        "validation_report": validation_report,
        "candidate_bundle": candidate_bundle,
        "heuristic_bundle": heuristic_bundle,
        "banner_annotation": banner_annotation,
        "candidate_annotations": candidate_annotations,
        "group_annotations": group_annotations,
        "brand_context_annotation": brand_context_annotation,
        "resolved_brand_family": resolved_brand_family,
        "resolved_language": resolved_language,
        "resolved_category": resolved_category,
        "pipeline_mode": effective_pipeline_mode,
        "qwen_mode": effective_qwen_mode,
        "qwen_calls_made": qwen_calls_made,
        "qwen_elapsed_seconds": qwen_elapsed_seconds,
        "qwen_candidates_annotated": candidates_annotated,
        "qwen_groups_annotated": groups_annotated,
        "qwen_request_metrics": qwen_request_metrics,
        "stage_timings": stage_timings,
        "nodes_annotated": len(semantic_graph.elements),
        "scene_semantic_updates": struct.semantic_updates if 'struct' in locals() else [],
        "scene_semantic_groups": struct.semantic_groups if 'struct' in locals() else [],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert raw Figma JSON + banner image into semantic_graph.json")
    parser.add_argument("--raw-json", type=str, required=True, help="Path to raw Figma JSON file")
    parser.add_argument("--banner-image", type=str, required=True, help="Path to banner image file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for outputs")
    parser.add_argument("--use-qwen", action="store_true", help="Enable Qwen API annotation stage")
    parser.add_argument(
        "--qwen-mode",
        type=str,
        default="single_pass",
        choices=["single_pass", "scene_only", "per_candidate", "off"],
        help="Qwen VLM strategy (default single_pass). Ignored unless --use-qwen.",
    )
    parser.add_argument(
        "--qwen-base-url",
        type=str,
        default=default_qwen_base_url(),
        help="Base URL of the running Qwen service (default: QWEN_BASE_URL from env or http://127.0.0.1:30078)",
    )
    parser.add_argument(
        "--brand-family",
        type=str,
        default=None,
        help="Optional override for inferred brand_family (snake_case). Omit to use Qwen inference when --use-qwen.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional override for inferred language. Omit for automatic inference when --use-qwen.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional override for inferred category. Omit for automatic inference when --use-qwen.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(
        raw_json_path=args.raw_json,
        banner_image_path=args.banner_image,
        output_dir=args.output_dir,
        use_qwen=args.use_qwen,
        qwen_base_url=args.qwen_base_url,
        brand_family=args.brand_family,
        language=args.language,
        category=args.category,
        qwen_mode=args.qwen_mode if args.use_qwen else "off",
    )


if __name__ == "__main__":
    main()
