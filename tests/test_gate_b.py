from __future__ import annotations

from pathlib import Path

from edge_al_pipeline.evaluation.gate_b import _build_seed_rows, read_metric_series
from edge_al_pipeline.experiments.phase2_agri_classification import Phase2SeedResult


def test_read_metric_series_extracts_round_values(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "\n".join(
            [
                "round_index,seed,split,metric,value",
                "0,1,train,test_accuracy,0.30",
                "1,1,train,test_accuracy,0.45",
                "1,1,train,loss,0.8",
            ]
        ),
        encoding="utf-8",
    )
    series = read_metric_series(metrics, "test_accuracy")
    assert series == [(0, 0.30), (1, 0.45)]


def test_build_seed_rows_computes_deltas(tmp_path):
    pretrain = _seed_result(tmp_path, "pretrain", seed=7)
    imagenet = _seed_result(tmp_path, "imagenet", seed=7, acc0=0.2, acc1=0.3)
    agri = _seed_result(tmp_path, "agri", seed=7, acc0=0.25, acc1=0.4)
    rows = _build_seed_rows(
        shared_seeds=[7],
        pretrain_by_seed={7: pretrain},
        imagenet_by_seed={7: imagenet},
        agri_by_seed={7: agri},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.seed == 7
    assert row.delta_final_test_accuracy > 0
    assert row.delta_auc_test_accuracy > 0


def _seed_result(
    tmp_path: Path, name: str, seed: int, acc0: float = 0.1, acc1: float = 0.2
) -> Phase2SeedResult:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "checkpoints" / f"agri_backbone_seed{seed}.pt").write_bytes(b"x")
    (run_dir / "metrics.csv").write_text(
        "\n".join(
            [
                "round_index,seed,split,metric,value",
                f"0,{seed},train,test_accuracy,{acc0}",
                f"1,{seed},train,test_accuracy,{acc1}",
            ]
        ),
        encoding="utf-8",
    )
    return Phase2SeedResult(
        seed=seed,
        run_dir=run_dir,
        backbone_path=run_dir / "checkpoints" / f"agri_backbone_seed{seed}.pt",
        final_labeled_count=0,
        final_unlabeled_count=0,
        dataset_size_used=0,
    )
