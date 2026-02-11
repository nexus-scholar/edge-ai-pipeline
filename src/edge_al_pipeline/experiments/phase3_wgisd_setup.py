from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from edge_al_pipeline.artifacts import ArtifactStore
from edge_al_pipeline.config import ExperimentConfig
from edge_al_pipeline.experiments.bootstrap import build_bootstrap_splits, build_run_dir


@dataclass(frozen=True)
class Phase3SetupSeedResult:
    seed: int
    run_dir: Path
    setup_manifest_path: Path
    uncertainty_plan_path: Path
    teacher_policy_path: Path


@dataclass(frozen=True)
class Phase3SetupSummary:
    results: list[Phase3SetupSeedResult]


def setup_phase3_wgisd(
    config: ExperimentConfig, config_source: Path | None = None
) -> Phase3SetupSummary:
    if config.dataset.task not in {"detection", "segmentation"}:
        raise ValueError("Phase 3 setup expects dataset.task to be detection/segmentation.")

    results: list[Phase3SetupSeedResult] = []
    for seed in config.seeds:
        run_dir = build_run_dir(config.output_root, config.experiment_name, seed)
        artifacts = ArtifactStore(run_dir)
        artifacts.initialize(config, config_source=config_source)
        splits = build_bootstrap_splits(config.bootstrap, seed=seed)
        artifacts.write_splits(
            splits,
            dataset_hash=f"{config.dataset.name}:{config.dataset.version}",
        )

        setup_manifest_path = run_dir / "phase3_setup.json"
        uncertainty_plan_path = run_dir / "uncertainty_plan.json"
        teacher_policy_path = run_dir / "teacher_policy.json"
        readme_path = run_dir / "README_PHASE3.md"

        setup_manifest_path.write_text(
            json.dumps(
                {
                    "dataset": config.dataset.to_dict(),
                    "model_name": config.model_name,
                    "strategy_name": config.strategy_name,
                    "query_size": config.query_size,
                    "rounds": config.rounds,
                    "seed": seed,
                    "expected_artifacts": [
                        "splits.json",
                        "round_{r}_selected.csv",
                        "metrics.csv",
                        "profile.csv",
                        "checkpoints/",
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        uncertainty_plan_path.write_text(
            json.dumps(
                {
                    "primary": "classification_entropy",
                    "secondary": "localization_jitter",
                    "blend": {"mode": "weighted_sum", "alpha": 0.5},
                    "notes": "Compare independent and blended uncertainty metrics.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        teacher_policy_path.write_text(
            json.dumps(
                {
                    "enabled": True,
                    "edge_pre_filter_top_k_factor": 4,
                    "teacher_model": "resnet50_or_efficientnet",
                    "selection_rule": "teacher_rerank_then_take_top_k",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        readme_path.write_text(_phase3_readme_text(run_dir), encoding="utf-8")

        results.append(
            Phase3SetupSeedResult(
                seed=seed,
                run_dir=run_dir,
                setup_manifest_path=setup_manifest_path,
                uncertainty_plan_path=uncertainty_plan_path,
                teacher_policy_path=teacher_policy_path,
            )
        )
    return Phase3SetupSummary(results=results)


def _phase3_readme_text(run_dir: Path) -> str:
    return "\n".join(
        [
            "# Phase 3 Setup (WGISD)",
            "",
            "This directory is initialized for Phase 3 detection/segmentation AL runs.",
            "",
            "Files:",
            "- `phase3_setup.json`: run metadata and expected artifacts",
            "- `uncertainty_plan.json`: classification/localization uncertainty strategy",
            "- `teacher_policy.json`: edge-student + fog-teacher rerank policy",
            "",
            "Next implementation step:",
            "1. Connect a detector trainer (YOLO nano-class) and write per-round metrics.",
            "2. Write selected IDs to `round_{r}_selected.csv` and profiling to `profile.csv`.",
            "3. Compare FP32 vs INT8 uncertainty behavior in this same run structure.",
            "",
            f"Run root: {run_dir}",
        ]
    )
