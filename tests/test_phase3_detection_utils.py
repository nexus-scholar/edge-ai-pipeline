from __future__ import annotations

import csv
import json

import torch

from edge_al_pipeline.contracts import SelectionCandidate
from edge_al_pipeline.experiments.phase3_wgisd_detection import write_uncertainty_summary
from edge_al_pipeline.models.wgisd_detection_runner import (
    classification_uncertainty_from_scores,
    compute_detection_metrics,
    greedy_match_counts,
)
from edge_al_pipeline.teacher.heuristic_verifier import HeuristicTeacherVerifier


def test_classification_uncertainty_from_scores():
    scores = torch.tensor([0.9, 0.7, 0.6], dtype=torch.float32)
    uncertainty = classification_uncertainty_from_scores(scores, top_n=2)
    assert 0.0 <= uncertainty <= 1.0
    assert round(uncertainty, 4) == round(1.0 - ((0.9 + 0.7) / 2.0), 4)


def test_greedy_match_counts_and_metrics():
    pred_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    pred_labels = torch.tensor([1, 2], dtype=torch.int64)
    pred_scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [40.0, 40.0, 50.0, 50.0]])
    gt_labels = torch.tensor([1, 2], dtype=torch.int64)

    tp, fp, fn = greedy_match_counts(
        pred_boxes=pred_boxes,
        pred_labels=pred_labels,
        pred_scores=pred_scores,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        iou_threshold=0.5,
    )
    assert (tp, fp, fn) == (1, 1, 1)

    precision, recall, map50_proxy = compute_detection_metrics(tp=tp, fp=fp, fn=fn)
    assert round(precision, 3) == 0.5
    assert round(recall, 3) == 0.5
    assert round(map50_proxy, 3) == 0.5


def test_teacher_verifier_reranks_and_filters():
    verifier = HeuristicTeacherVerifier(max_detection_count=3, rerank_alpha=0.25)
    candidates = [
        SelectionCandidate(
            sample_id="sample_1",
            score=0.2,
            metadata={
                "uncertainty_classification": 0.8,
                "uncertainty_localization": 0.4,
                "det_count": 2,
            },
        ),
        SelectionCandidate(
            sample_id="sample_2",
            score=0.9,
            metadata={
                "uncertainty_classification": 0.2,
                "uncertainty_localization": 0.9,
                "det_count": 10,
            },
        ),
    ]
    selected = verifier.verify(round_index=0, candidates=candidates, k=1)
    assert len(selected) == 1
    assert selected[0].sample_id == "sample_1"
    assert "teacher_score" in selected[0].metadata


def test_write_uncertainty_summary(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    round_path = run_dir / "round_0_selected.csv"

    with round_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["round_index", "seed", "strategy", "sample_id", "score", "metadata"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "round_index": 0,
                "seed": 1,
                "strategy": "entropy",
                "sample_id": "sample_0",
                "score": 0.7,
                "metadata": json.dumps(
                    {
                        "uncertainty_classification": 0.6,
                        "uncertainty_localization": 0.8,
                        "uncertainty_combined": 0.7,
                        "teacher_score": 0.65,
                        "det_count": 5,
                    }
                ),
            }
        )

    summary_path = write_uncertainty_summary(run_dir)
    assert summary_path.exists()
    content = summary_path.read_text(encoding="utf-8")
    assert "mean_uncertainty_combined" in content
