from __future__ import annotations

import math

import torch

from edge_al_pipeline.models.cifar10_runner import (
    _compute_domain_confusion_score,
    _compute_domain_reference,
)


def test_compute_domain_reference_returns_center_and_scale():
    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    center, mean_distance = _compute_domain_reference(embeddings)
    assert center.shape == (2,)
    assert torch.allclose(center, torch.tensor([0.5, 0.5], dtype=torch.float32))
    assert mean_distance > 0.0


def test_domain_confusion_falls_back_to_entropy_without_reference():
    entropy_max = math.log(10.0)
    score = _compute_domain_confusion_score(
        embedding=torch.tensor([0.0, 0.0], dtype=torch.float32),
        reference_center=None,
        reference_mean_distance=None,
        entropy_score=1.2,
        entropy_max=entropy_max,
    )
    assert score == 1.2


def test_domain_confusion_increases_with_distance_from_reference():
    entropy_max = math.log(10.0)
    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.1, 0.1],
        ],
        dtype=torch.float32,
    )
    center, mean_distance = _compute_domain_reference(embeddings)
    near_score = _compute_domain_confusion_score(
        embedding=torch.tensor([0.05, 0.05], dtype=torch.float32),
        reference_center=center,
        reference_mean_distance=mean_distance,
        entropy_score=0.3,
        entropy_max=entropy_max,
    )
    far_score = _compute_domain_confusion_score(
        embedding=torch.tensor([4.0, 4.0], dtype=torch.float32),
        reference_center=center,
        reference_mean_distance=mean_distance,
        entropy_score=0.3,
        entropy_max=entropy_max,
    )
    assert far_score > near_score
    assert far_score <= entropy_max

