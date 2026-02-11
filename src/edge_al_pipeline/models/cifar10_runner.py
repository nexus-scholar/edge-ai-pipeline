from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as tnf
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from edge_al_pipeline.contracts import SelectionCandidate


@dataclass(frozen=True)
class Cifar10RunnerConfig:
    data_root: str
    batch_size: int = 128
    score_batch_size: int = 256
    epochs_per_round: int = 1
    learning_rate: float = 1e-3
    num_workers: int = 0
    device: str = "cpu"
    download: bool = True
    embedding_dim: int = 128
    low_contrast_factor: float = 0.5


class Cifar10CnnRunner:
    """Simple RGB CNN runner for Phase 1b (CIFAR-10)."""

    name = "simple_cnn_cifar10"

    def __init__(
        self,
        config: Cifar10RunnerConfig,
        val_ids: Sequence[str],
        test_ids: Sequence[str] | None = None,
    ) -> None:
        self._config = config
        self._device = torch.device(config.device)

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        )
        self._train_dataset = datasets.CIFAR10(
            root=config.data_root,
            train=True,
            download=config.download,
            transform=train_transform,
        )
        self._train_eval_dataset = datasets.CIFAR10(
            root=config.data_root,
            train=True,
            download=config.download,
            transform=eval_transform,
        )
        self._test_dataset = datasets.CIFAR10(
            root=config.data_root,
            train=False,
            download=config.download,
            transform=eval_transform,
        )

        self._val_indices = _ids_to_indices(val_ids)
        self._test_indices = (
            _ids_to_indices(test_ids) if test_ids else list(range(len(self._test_dataset)))
        )

        self._model = _SimpleCifarCnn(embedding_dim=config.embedding_dim).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=config.learning_rate)
        self._criterion = nn.CrossEntropyLoss()

    def train_round(
        self, round_index: int, seed: int, labeled_ids: Sequence[str]
    ) -> Mapping[str, float]:
        _set_seed(seed + round_index)
        labeled_indices = _ids_to_indices(labeled_ids)
        if not labeled_indices:
            return {
                "loss": math.nan,
                "train_accuracy": math.nan,
                "val_accuracy": math.nan,
                "test_accuracy": math.nan,
                "test_accuracy_blur": math.nan,
                "test_accuracy_low_contrast": math.nan,
            }

        train_loader = DataLoader(
            Subset(self._train_dataset, labeled_indices),
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=self._config.num_workers,
        )

        self._model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for _ in range(self._config.epochs_per_round):
            for images, labels in train_loader:
                images = images.to(self._device)
                labels = labels.to(self._device)
                self._optimizer.zero_grad(set_to_none=True)
                logits, _ = self._model(images)
                loss = self._criterion(logits, labels)
                loss.backward()
                self._optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total += batch_size
                correct += (logits.argmax(dim=1) == labels).sum().item()

        train_accuracy = float(correct) / float(total) if total else math.nan
        mean_loss = total_loss / float(total) if total else math.nan
        val_accuracy = self._evaluate_subset(
            dataset=self._train_eval_dataset, indices=self._val_indices
        )
        test_accuracy = self._evaluate_subset(
            dataset=self._test_dataset, indices=self._test_indices
        )
        test_accuracy_blur = self._evaluate_subset(
            dataset=self._test_dataset,
            indices=self._test_indices,
            perturbation=_apply_blur,
        )
        test_accuracy_low_contrast = self._evaluate_subset(
            dataset=self._test_dataset,
            indices=self._test_indices,
            perturbation=lambda x: _apply_low_contrast(x, self._config.low_contrast_factor),
        )
        return {
            "loss": mean_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "test_accuracy_blur": test_accuracy_blur,
            "test_accuracy_low_contrast": test_accuracy_low_contrast,
        }

    def score_unlabeled(self, unlabeled_ids: Sequence[str]) -> list[SelectionCandidate]:
        indices = _ids_to_indices(unlabeled_ids)
        if not indices:
            return []

        loader = DataLoader(
            Subset(self._train_eval_dataset, indices),
            batch_size=self._config.score_batch_size,
            shuffle=False,
            num_workers=self._config.num_workers,
        )

        self._model.eval()
        candidates: list[SelectionCandidate] = []
        pointer = 0
        with torch.no_grad():
            for images, _labels in loader:
                images = images.to(self._device)
                logits, embeddings = self._model(images)
                probs = torch.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=1)

                batch_size = entropy.size(0)
                batch_ids = unlabeled_ids[pointer : pointer + batch_size]
                pointer += batch_size
                embeddings_cpu = embeddings.cpu().tolist()
                entropy_cpu = entropy.cpu().tolist()
                for sample_id, score, embedding in zip(
                    batch_ids, entropy_cpu, embeddings_cpu
                ):
                    candidates.append(
                        SelectionCandidate(
                            sample_id=str(sample_id),
                            score=float(score),
                            embedding=tuple(float(value) for value in embedding),
                            metadata={"uncertainty": "entropy"},
                        )
                    )
        return candidates

    def _evaluate_subset(
        self,
        dataset: datasets.CIFAR10,
        indices: Sequence[int],
        perturbation: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> float:
        if not indices:
            return math.nan
        loader = DataLoader(
            Subset(dataset, list(indices)),
            batch_size=self._config.score_batch_size,
            shuffle=False,
            num_workers=self._config.num_workers,
        )
        self._model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self._device)
                if perturbation is not None:
                    images = perturbation(images)
                labels = labels.to(self._device)
                logits, _ = self._model(images)
                predictions = logits.argmax(dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        return float(correct) / float(total) if total else math.nan


class _SimpleCifarCnn(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, 10)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(inputs)
        embedding = self.embedding(x)
        logits = self.classifier(embedding)
        return logits, embedding


def _apply_blur(images: torch.Tensor) -> torch.Tensor:
    return tnf.avg_pool2d(images, kernel_size=3, stride=1, padding=1)


def _apply_low_contrast(images: torch.Tensor, factor: float) -> torch.Tensor:
    mean = images.mean(dim=(2, 3), keepdim=True)
    return mean + (images - mean) * factor


def _ids_to_indices(ids: Sequence[str]) -> list[int]:
    return [_sample_id_to_index(sample_id) for sample_id in ids]


def _sample_id_to_index(sample_id: str) -> int:
    token = str(sample_id)
    if token.startswith("sample_"):
        return int(token.replace("sample_", ""))
    return int(token)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
