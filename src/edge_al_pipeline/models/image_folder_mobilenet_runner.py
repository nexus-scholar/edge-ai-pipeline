from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
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
class ImageFolderMobileNetRunnerConfig:
    data_root: str
    batch_size: int = 32
    score_batch_size: int = 64
    epochs_per_round: int = 1
    learning_rate: float = 1e-3
    num_workers: int = 0
    device: str = "cpu"
    image_size: int = 224
    pretrained_backbone: bool = True
    freeze_backbone: bool = False
    backbone_name: str = "mobilenet_v3_small"
    backbone_checkpoint: str | None = None
    num_classes: int | None = None
    max_samples: int | None = None


class ImageFolderMobileNetRunner:
    """ImageFolder runner with interchangeable classification backbones."""

    name = "imagefolder_classifier"

    def __init__(
        self,
        config: ImageFolderMobileNetRunnerConfig,
        val_ids: Sequence[str],
        test_ids: Sequence[str],
    ) -> None:
        self._config = config
        self._device = torch.device(config.device)

        train_transform, eval_transform = _build_transforms(config.image_size)
        self._train_dataset = datasets.ImageFolder(
            root=config.data_root,
            transform=train_transform,
        )
        self._eval_dataset = datasets.ImageFolder(
            root=config.data_root,
            transform=eval_transform,
        )

        if config.max_samples is not None:
            limit = min(config.max_samples, len(self._train_dataset))
            self._allowed_indices = set(range(limit))
        else:
            self._allowed_indices = None

        inferred_num_classes = len(self._train_dataset.classes)
        if config.num_classes is not None and config.num_classes != inferred_num_classes:
            raise ValueError(
                f"num_classes mismatch: config={config.num_classes}, "
                f"dataset={inferred_num_classes}."
            )
        self._num_classes = inferred_num_classes
        self._val_indices = self._filter_indices(_ids_to_indices(val_ids))
        self._test_indices = self._filter_indices(_ids_to_indices(test_ids))

        self._model = _build_classifier(
            num_classes=self._num_classes,
            backbone_name=config.backbone_name,
            pretrained_backbone=config.pretrained_backbone,
        ).to(self._device)
        if config.backbone_checkpoint:
            _load_backbone_weights(
                model=self._model,
                checkpoint_path=Path(config.backbone_checkpoint),
            )
        if config.freeze_backbone:
            self._model.freeze_backbone()

        self._optimizer = torch.optim.Adam(
            [param for param in self._model.parameters() if param.requires_grad],
            lr=config.learning_rate,
        )
        self._criterion = nn.CrossEntropyLoss()

    @classmethod
    def inspect_dataset(
        cls, root: str, max_samples: int | None = None
    ) -> tuple[int, int]:
        dataset = datasets.ImageFolder(root=root)
        size = len(dataset)
        if max_samples is not None:
            size = min(size, max_samples)
        return size, len(dataset.classes)

    def train_round(
        self, round_index: int, seed: int, labeled_ids: Sequence[str]
    ) -> Mapping[str, float]:
        _set_seed(seed + round_index)
        labeled_indices = self._filter_indices(_ids_to_indices(labeled_ids))
        if not labeled_indices:
            return {
                "loss": math.nan,
                "train_accuracy": math.nan,
                "val_accuracy": math.nan,
                "test_accuracy": math.nan,
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
        val_accuracy = self._evaluate_subset(self._val_indices)
        test_accuracy = self._evaluate_subset(self._test_indices)
        return {
            "loss": mean_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
        }

    def score_unlabeled(self, unlabeled_ids: Sequence[str]) -> list[SelectionCandidate]:
        indices = self._filter_indices(_ids_to_indices(unlabeled_ids))
        if not indices:
            return []

        loader = DataLoader(
            Subset(self._eval_dataset, indices),
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

    def export_backbone(self, target_path: Path) -> Path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        backbone_state = self._model.backbone_state_dict()
        payload = {
            "model": self._config.backbone_name,
            "num_classes": self._num_classes,
            "backbone_state_dict": backbone_state,
        }
        if self._config.backbone_name.startswith("mobilenet_v3_"):
            payload["features_state_dict"] = backbone_state
        torch.save(payload, target_path)
        return target_path

    def _evaluate_subset(self, indices: Sequence[int]) -> float:
        if not indices:
            return math.nan
        loader = DataLoader(
            Subset(self._eval_dataset, list(indices)),
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
                labels = labels.to(self._device)
                logits, _ = self._model(images)
                predictions = logits.argmax(dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        return float(correct) / float(total) if total else math.nan

    def _filter_indices(self, indices: Sequence[int]) -> list[int]:
        if self._allowed_indices is None:
            return list(indices)
        return [index for index in indices if index in self._allowed_indices]


class _MobileNetV3Classifier(nn.Module):
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
        self.model = backbone
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier

    def freeze_backbone(self) -> None:
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(inputs)
        pooled = self.avgpool(features)
        embedding = torch.flatten(pooled, 1)
        logits = self.classifier(embedding)
        return logits, embedding

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        return self.features.state_dict()

    def load_backbone_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.features.load_state_dict(state_dict, strict=True)


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

    def freeze_backbone(self) -> None:
        for name, parameter in self.model.named_parameters():
            if name.startswith("fc."):
                continue
            parameter.requires_grad = False

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

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.model.state_dict().items()
            if not key.startswith("fc.")
        }

    def load_backbone_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        current = self.model.state_dict()
        merged = {
            key: value
            for key, value in state_dict.items()
            if key in current and not key.startswith("fc.")
        }
        current.update(merged)
        missing, unexpected = self.model.load_state_dict(current, strict=False)
        invalid_missing = [key for key in missing if not key.startswith("fc.")]
        if invalid_missing or unexpected:
            raise ValueError(
                "Backbone checkpoint is incompatible with this resnet architecture."
            )


def _build_classifier(
    num_classes: int, backbone_name: str, pretrained_backbone: bool
) -> nn.Module:
    normalized = backbone_name.strip().lower()
    if normalized.startswith("mobilenet_v4"):
        raise ValueError(
            "backbone_name uses MobileNetV4, but this environment does not expose "
            "MobileNetV4 in torchvision. Use v3/resnet now, or upgrade torchvision "
            "and add a MobileNetV4 adapter before running."
        )
    if normalized not in CLASSIFICATION_BACKBONES:
        raise ValueError(
            "Unsupported classification backbone_name. Expected one of: "
            f"{sorted(CLASSIFICATION_BACKBONES)}."
        )
    if normalized in {"mobilenet_v3_small", "mobilenet_v3_large"}:
        return _MobileNetV3Classifier(
            num_classes=num_classes,
            backbone_name=normalized,
            pretrained_backbone=pretrained_backbone,
        )
    return _ResNetClassifier(
        num_classes=num_classes,
        backbone_name=normalized,
        pretrained_backbone=pretrained_backbone,
    )


def _build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_transform, eval_transform


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


def _load_backbone_weights(model: nn.Module, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("backbone_state_dict")
    if not isinstance(state_dict, dict):
        state_dict = payload.get("features_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(
            "Backbone checkpoint payload missing 'backbone_state_dict'. "
            f"Path: {checkpoint_path}"
        )
    if not hasattr(model, "load_backbone_state_dict"):
        raise ValueError("Model does not support backbone checkpoint loading.")
    model.load_backbone_state_dict(state_dict)
