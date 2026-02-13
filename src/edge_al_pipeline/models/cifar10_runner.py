from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as tnf
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
)

from edge_al_pipeline.backbones import CLASSIFICATION_BACKBONES
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
    backbone_name: str = "simple_cnn"
    pretrained_backbone: bool = True
    image_size: int = 224


class Cifar10CnnRunner:
    """CIFAR10 runner with plug-and-play classifier backbones."""

    name = "cifar10_classifier"

    def __init__(
        self,
        config: Cifar10RunnerConfig,
        val_ids: Sequence[str],
        test_ids: Sequence[str] | None = None,
    ) -> None:
        self._config = config
        self._device = torch.device(config.device)
        self._backbone_name = config.backbone_name.strip().lower()

        train_transform, eval_transform = _build_transforms(
            backbone_name=self._backbone_name,
            image_size=config.image_size,
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

        self._model = _build_classifier(
            backbone_name=self._backbone_name,
            num_classes=10,
            embedding_dim=config.embedding_dim,
            pretrained_backbone=config.pretrained_backbone,
        ).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=config.learning_rate)
        self._criterion = nn.CrossEntropyLoss()
        self._domain_center: torch.Tensor | None = None
        self._domain_mean_distance: float | None = None
        self._entropy_max = math.log(10.0)

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
        self._update_domain_reference(labeled_indices)
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
                embeddings_cpu_tensor = embeddings.detach().cpu()
                for offset, (sample_id, score, embedding) in enumerate(
                    zip(batch_ids, entropy_cpu, embeddings_cpu)
                ):
                    domain_confusion = _compute_domain_confusion_score(
                        embedding=embeddings_cpu_tensor[offset],
                        reference_center=self._domain_center,
                        reference_mean_distance=self._domain_mean_distance,
                        entropy_score=float(score),
                        entropy_max=self._entropy_max,
                    )
                    candidates.append(
                        SelectionCandidate(
                            sample_id=str(sample_id),
                            score=float(score),
                            embedding=tuple(float(value) for value in embedding),
                            metadata={
                                "uncertainty": "entropy",
                                "uncertainty_entropy": float(score),
                                "uncertainty_combined": float(score),
                                "domain_confusion": float(domain_confusion),
                            },
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

    def _update_domain_reference(self, labeled_indices: Sequence[int]) -> None:
        if not labeled_indices:
            self._domain_center = None
            self._domain_mean_distance = None
            return
        loader = DataLoader(
            Subset(self._train_eval_dataset, list(labeled_indices)),
            batch_size=self._config.score_batch_size,
            shuffle=False,
            num_workers=self._config.num_workers,
        )
        self._model.eval()
        embeddings_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for images, _labels in loader:
                images = images.to(self._device)
                _, embeddings = self._model(images)
                embeddings_chunks.append(embeddings.detach().cpu())
        if not embeddings_chunks:
            self._domain_center = None
            self._domain_mean_distance = None
            return
        reference_embeddings = torch.cat(embeddings_chunks, dim=0)
        center, mean_distance = _compute_domain_reference(reference_embeddings)
        self._domain_center = center
        self._domain_mean_distance = mean_distance


class _SimpleCifarCnn(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_classes: int = 10) -> None:
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
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(inputs)
        embedding = self.embedding(x)
        logits = self.classifier(embedding)
        return logits, embedding


class _MobileNetClassifier(nn.Module):
    def __init__(
        self, num_classes: int, backbone_name: str, pretrained_backbone: bool
    ) -> None:
        super().__init__()
        if backbone_name == "mobilenet_v3_small":
            weights = (
                MobileNet_V3_Small_Weights.IMAGENET1K_V1
                if pretrained_backbone
                else None
            )
            backbone = models.mobilenet_v3_small(weights=weights)
        elif backbone_name == "mobilenet_v3_large":
            weights = (
                MobileNet_V3_Large_Weights.IMAGENET1K_V1
                if pretrained_backbone
                else None
            )
            backbone = models.mobilenet_v3_large(weights=weights)
        else:
            raise ValueError(
                "Unsupported MobileNet backbone_name. "
                "Expected 'mobilenet_v3_small' or 'mobilenet_v3_large'."
            )
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(inputs)
        pooled = self.avgpool(features)
        embedding = torch.flatten(pooled, 1)
        logits = self.classifier(embedding)
        return logits, embedding


class _ResNetClassifier(nn.Module):
    def __init__(
        self, num_classes: int, backbone_name: str, pretrained_backbone: bool
    ) -> None:
        super().__init__()
        if backbone_name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            backbone = models.resnet18(weights=weights)
        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
            backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(
                "Unsupported ResNet backbone_name. Expected 'resnet18' or 'resnet50'."
            )
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.model.conv1(inputs)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        embedding = torch.flatten(x, 1)
        logits = self.model.fc(embedding)
        return logits, embedding


def _build_classifier(
    backbone_name: str,
    num_classes: int,
    embedding_dim: int,
    pretrained_backbone: bool,
) -> nn.Module:
    if backbone_name == "simple_cnn":
        return _SimpleCifarCnn(embedding_dim=embedding_dim, num_classes=num_classes)
    if backbone_name.startswith("mobilenet_v4"):
        raise ValueError(
            "backbone_name uses MobileNetV4, but this environment does not expose "
            "MobileNetV4 in torchvision. Use v3/resnet now, or upgrade torchvision "
            "and add a MobileNetV4 adapter before running."
        )
    if backbone_name not in CLASSIFICATION_BACKBONES:
        raise ValueError(
            "Unsupported CIFAR10 backbone_name. Expected one of: "
            f"{sorted(CLASSIFICATION_BACKBONES | {'simple_cnn'})}."
        )
    if backbone_name in {"mobilenet_v3_small", "mobilenet_v3_large"}:
        return _MobileNetClassifier(
            num_classes=num_classes,
            backbone_name=backbone_name,
            pretrained_backbone=pretrained_backbone,
        )
    return _ResNetClassifier(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained_backbone=pretrained_backbone,
    )


def _build_transforms(backbone_name: str, image_size: int):
    if backbone_name == "simple_cnn":
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
        return train_transform, eval_transform

    resize_to = max(32, int(image_size))
    train_transform = transforms.Compose(
        [
            transforms.Resize((resize_to, resize_to)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((resize_to, resize_to)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    return train_transform, eval_transform


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


def _compute_domain_reference(embeddings: torch.Tensor) -> tuple[torch.Tensor, float]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D tensor.")
    if embeddings.shape[0] == 0:
        raise ValueError("embeddings must contain at least one sample.")
    center = embeddings.mean(dim=0)
    distances = torch.norm(embeddings - center.unsqueeze(0), p=2, dim=1)
    mean_distance = float(distances.mean().item())
    return center, max(mean_distance, 1e-8)


def _compute_domain_confusion_score(
    embedding: torch.Tensor,
    reference_center: torch.Tensor | None,
    reference_mean_distance: float | None,
    entropy_score: float,
    entropy_max: float,
) -> float:
    if (
        reference_center is None
        or reference_mean_distance is None
        or reference_mean_distance <= 0.0
    ):
        return _clamp(entropy_score, low=0.0, high=entropy_max)
    distance = float(torch.norm(embedding - reference_center, p=2).item())
    normalized = distance / max(reference_mean_distance, 1e-8)
    compressed = normalized / (1.0 + normalized)
    return _clamp(compressed * entropy_max, low=0.0, high=entropy_max)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
