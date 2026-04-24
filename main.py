from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from banner_pipeline.build_candidates import build_candidates
from banner_pipeline.collapse_groups import collapse_wrapper_groups
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


def run_pipeline(
    *,
    raw_json_path: str | Path,
    banner_image_path: str | Path,
    output_dir: str | Path,
    use_qwen: bool = False,
    qwen_base_url: str = "http://127.0.0.1:8001",
    brand_family: str | None = None,
    language: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    intermediate_dir = ensure_dir(output_dir / "intermediate")
    final_dir = ensure_dir(output_dir / "final")

    doc, root, parsed_nodes = parse_figma_file(raw_json_path)
    canvas_w = int(root.bounds.width)
    canvas_h = int(root.bounds.height)

    save_flat_nodes(parsed_nodes, intermediate_dir / "01_flat_nodes.json")

    normalized_nodes = normalize_nodes(parsed_nodes, canvas_w, canvas_h)
    save_json([node.to_dict() for node in normalized_nodes], intermediate_dir / "02_normalized_nodes.json")

    collapsed_nodes = collapse_wrapper_groups(normalized_nodes)
    save_json([node.to_dict() for node in collapsed_nodes], intermediate_dir / "03_collapsed_nodes.json")

    candidate_bundle = build_candidates(collapsed_nodes)
    save_json(candidate_bundle.to_dict(), intermediate_dir / "04_candidates.json")

    heuristic_bundle = apply_heuristics(candidate_bundle, collapsed_nodes)
    save_json(heuristic_bundle.to_dict(), intermediate_dir / "05_heuristics.json")

    banner_annotation = None
    candidate_annotations = {}
    group_annotations = {}
    brand_context_annotation = None

    if use_qwen:
        annotator = QwenAnnotator(base_url=qwen_base_url)
        annotator.load_model()

        brand_context_annotation = annotator.annotate_brand_context(
            banner_image_path=str(banner_image_path),
            candidate_bundle=candidate_bundle,
            heuristic_bundle=heuristic_bundle,
        )
        save_json(brand_context_annotation.to_dict(), intermediate_dir / "05b_brand_context.json")

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
        save_json(banner_annotation.to_dict(), intermediate_dir / "06_banner_annotation.json")

        candidate_annotations = annotator.annotate_candidates(
            banner_image_path=str(banner_image_path),
            candidate_bundle=candidate_bundle,
            heuristic_bundle=heuristic_bundle,
        )
        save_json({k: v.to_dict() for k, v in candidate_annotations.items()}, intermediate_dir / "07_candidate_annotations.json")

        group_annotations = annotator.annotate_group_candidates(
            banner_image_path=str(banner_image_path),
            candidate_bundle=candidate_bundle,
            heuristic_bundle=heuristic_bundle,
        )
        save_json({k: v.to_dict() for k, v in group_annotations.items()}, intermediate_dir / "08_group_annotations.json")
    else:
        resolved_brand_family = brand_family if brand_family is not None else "generic"
        resolved_language = language if language is not None else "unknown"
        resolved_category = category if category is not None else "unknown"

    config = MergeConfig(
        default_brand_family=resolved_brand_family,
        default_language=resolved_language,
        default_category=resolved_category,
    )

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

    save_model_json(semantic_graph, final_dir / "semantic_graph.json")

    validation_report = validate_graph(semantic_graph)
    save_json(validation_report.to_dict(), final_dir / "validation_report.json")

    print("=" * 70)
    print("Pipeline finished")
    print(f"Raw JSON:      {raw_json_path}")
    print(f"Banner image:  {banner_image_path}")
    print(f"Output dir:    {output_dir}")
    print("-" * 70)
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
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert raw Figma JSON + banner image into semantic_graph.json")
    parser.add_argument("--raw-json", type=str, required=True, help="Path to raw Figma JSON file")
    parser.add_argument("--banner-image", type=str, required=True, help="Path to banner image file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for outputs")
    parser.add_argument("--use-qwen", action="store_true", help="Enable Qwen API annotation stage")
    parser.add_argument("--qwen-base-url", type=str, default="http://127.0.0.1:8001", help="Base URL of the running Qwen service")
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
    )


if __name__ == "__main__":
    main()
