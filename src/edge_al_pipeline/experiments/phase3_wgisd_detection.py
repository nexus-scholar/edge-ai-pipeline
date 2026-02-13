from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from edge_al_pipeline.backbones import resolve_backbone_name
from edge_al_pipeline.artifacts import ArtifactStore
from edge_al_pipeline.config import BootstrapConfig, ExperimentConfig
from edge_al_pipeline.data_pool import DataPoolManager
from edge_al_pipeline.experiments.bootstrap import build_bootstrap_splits, build_run_dir
from edge_al_pipeline.models.wgisd_detection_runner import (
    WgisdDetectionRunner,
    WgisdDetectionRunnerConfig,
)
from edge_al_pipeline.pipeline import ActiveLearningPipeline
from edge_al_pipeline.profiling.edge_profiler import EdgeProfiler
from edge_al_pipeline.strategies import build_strategy
from edge_al_pipeline.teacher import HeuristicTeacherVerifier


@dataclass(frozen=True)
class Phase3SeedResult:
    seed: int
    run_dir: Path
    final_labeled_count: int
    final_unlabeled_count: int
    uncertainty_summary_path: Path


@dataclass(frozen=True)
class Phase3RunSummary:
    results: list[Phase3SeedResult]


def run_phase3_wgisd_detection(
    config: ExperimentConfig, config_source: Path | None = None
) -> Phase3RunSummary:
    if config.dataset.task not in {"detection", "segmentation"}:
        raise ValueError("Phase 3 runner requires detection/segmentation task.")

    params = config.model_params
    images_root = str(params.get("images_root", config.dataset.root))
    annotations_path = str(
        params.get("annotations_path", Path(config.dataset.root) / "annotations.json")
    )
    max_samples = params.get("max_samples")
    max_samples_int = int(max_samples) if max_samples is not None else None
    dataset_size, class_count = WgisdDetectionRunner.inspect_dataset(
        images_root=images_root,
        annotations_path=annotations_path,
        max_samples=max_samples_int,
    )
    effective_bootstrap = _effective_bootstrap(config.bootstrap, dataset_size)

    # Collect all IDs from the annotation file
    # We need to know the filenames to use as sample IDs
    annotation_payload = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    # Sort by ID to ensure deterministic order before shuffling in bootstrap
    sorted_images = sorted(annotation_payload.get("images", []), key=lambda x: x["id"])
    all_ids = [img["file_name"] for img in sorted_images]
    
    if max_samples_int is not None:
        all_ids = all_ids[:max_samples_int]

    results: list[Phase3SeedResult] = []
    for seed in config.seeds:
        run_dir = build_run_dir(config.output_root, config.experiment_name, seed)
        artifacts = ArtifactStore(run_dir)
        artifacts.initialize(config, config_source=config_source)

        splits = build_bootstrap_splits(effective_bootstrap, all_ids, seed=seed)
        artifacts.write_splits(
            splits,
            dataset_hash=f"{config.dataset.name}:{config.dataset.version}",
        )

        pool = DataPoolManager.from_splits(splits)
        strategy = build_strategy(config.strategy_name, config.strategy_params)
        profiler = EdgeProfiler(
            device=config.edge_device,
            quantization_mode=config.quantization_mode,
        )
        model_runner = WgisdDetectionRunner(
            config=_runner_config_from_experiment(
                config=config,
                images_root=images_root,
                annotations_path=annotations_path,
            ),
            val_ids=splits.val,
            test_ids=splits.test,
        )

        teacher = None
        if config.teacher_enabled:
            teacher = HeuristicTeacherVerifier(
                max_detection_count=int(params.get("teacher_max_detection_count", 120)),
                rerank_alpha=float(params.get("teacher_rerank_alpha", 0.5)),
            )

        pipeline = ActiveLearningPipeline(
            pool=pool,
            model_runner=model_runner,
            strategy=strategy,
            artifacts=artifacts,
            profiler=profiler,
            query_size=config.query_size,
            teacher=teacher,
        )
        pipeline.run_seed(seed=seed, rounds=config.rounds)

        final_splits = pool.to_splits()
        artifacts.write_splits(
            final_splits,
            dataset_hash=f"{config.dataset.name}:{config.dataset.version}",
        )
        uncertainty_summary_path = write_uncertainty_summary(run_dir=run_dir)

        final_counts = final_splits.counts()
        _write_phase3_manifest(
            run_dir=run_dir,
            class_count=class_count,
            images_root=images_root,
            annotations_path=annotations_path,
            teacher_enabled=config.teacher_enabled,
        )
        results.append(
            Phase3SeedResult(
                seed=seed,
                run_dir=run_dir,
                final_labeled_count=final_counts["labeled"],
                final_unlabeled_count=final_counts["unlabeled"],
                uncertainty_summary_path=uncertainty_summary_path,
            )
        )

    return Phase3RunSummary(results=results)


def write_uncertainty_summary(run_dir: Path) -> Path:
    rows: list[dict[str, float | int]] = []
    for path in sorted(run_dir.glob("round_*_selected.csv")):
        round_token = path.stem.replace("round_", "").replace("_selected", "")
        round_index = int(round_token)
        cls_values: list[float] = []
        loc_values: list[float] = []
        combined_values: list[float] = []
        teacher_values: list[float] = []
        det_counts: list[int] = []

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                metadata = json.loads(row.get("metadata", "{}"))
                cls_values.append(float(metadata.get("uncertainty_classification", 0.0)))
                loc_values.append(float(metadata.get("uncertainty_localization", 0.0)))
                combined_values.append(float(metadata.get("uncertainty_combined", row["score"])))
                if "teacher_score" in metadata:
                    teacher_values.append(float(metadata["teacher_score"]))
                det_counts.append(int(metadata.get("det_count", 0)))

        if not combined_values:
            continue
        rows.append(
            {
                "round_index": round_index,
                "selected_count": len(combined_values),
                "mean_uncertainty_classification": _mean(cls_values),
                "mean_uncertainty_localization": _mean(loc_values),
                "mean_uncertainty_combined": _mean(combined_values),
                "mean_teacher_score": _mean(teacher_values) if teacher_values else 0.0,
                "mean_detection_count": _mean(det_counts),
            }
        )

    target_path = run_dir / "uncertainty_summary.csv"
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "round_index",
                "selected_count",
                "mean_uncertainty_classification",
                "mean_uncertainty_localization",
                "mean_uncertainty_combined",
                "mean_teacher_score",
                "mean_detection_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return target_path


def _runner_config_from_experiment(
    config: ExperimentConfig, images_root: str, annotations_path: str
) -> WgisdDetectionRunnerConfig:
    params = config.model_params
    backbone_name = resolve_backbone_name(config.model_name, params)
    if backbone_name is None:
        backbone_name = "mobilenet_v3_large_320_fpn"
    max_samples = params.get("max_samples")
    max_samples_int = int(max_samples) if max_samples is not None else None
    return WgisdDetectionRunnerConfig(
        images_root=images_root,
        annotations_path=annotations_path,
        batch_size=int(params.get("batch_size", 2)),
        score_batch_size=int(params.get("score_batch_size", 1)),
        epochs_per_round=int(params.get("epochs_per_round", 1)),
        learning_rate=float(params.get("learning_rate", 5e-4)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
        num_workers=int(params.get("num_workers", 0)),
        device=str(params.get("device", "cpu")),
        pretrained_backbone=bool(params.get("pretrained_backbone", True)),
        score_threshold=float(params.get("score_threshold", 0.2)),
        score_top_n=int(params.get("score_top_n", 10)),
        uncertainty_alpha=float(params.get("uncertainty_alpha", 0.5)),
        localization_tta=bool(params.get("localization_tta", True)),
        max_samples=max_samples_int,
        quantization_mode=config.quantization_mode,
        backbone_name=backbone_name,
    )


def _effective_bootstrap(bootstrap: BootstrapConfig, dataset_size: int) -> BootstrapConfig:
    pool_size = min(bootstrap.pool_size, dataset_size)
    known_total = bootstrap.initial_labeled_size + bootstrap.val_size + bootstrap.test_size
    if known_total >= pool_size:
        raise ValueError(
            "Phase 3 bootstrap sizes are too large for available samples "
            f"({dataset_size})."
        )
    return BootstrapConfig(
        pool_size=pool_size,
        initial_labeled_size=bootstrap.initial_labeled_size,
        val_size=bootstrap.val_size,
        test_size=bootstrap.test_size,
    )


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _write_phase3_manifest(
    run_dir: Path,
    class_count: int,
    images_root: str,
    annotations_path: str,
    teacher_enabled: bool,
) -> Path:
    path = run_dir / "phase3_manifest.json"
    path.write_text(
        json.dumps(
            {
                "class_count": class_count,
                "images_root": images_root,
                "annotations_path": annotations_path,
                "teacher_enabled": teacher_enabled,
                "notes": "Phase 3 detection AL run with uncertainty summary.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path

