from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as tvf

from edge_al_pipeline.contracts import SelectionCandidate


@dataclass(frozen=True)
class WgisdDetectionRunnerConfig:
    images_root: str
    annotations_path: str
    batch_size: int = 2
    score_batch_size: int = 1
    epochs_per_round: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "cpu"
    pretrained_backbone: bool = True
    score_threshold: float = 0.2
    score_top_n: int = 10
    uncertainty_alpha: float = 0.5
    localization_tta: bool = True
    max_samples: int | None = None
    quantization_mode: str = "fp32"


class WgisdDetectionRunner:
    """Detection runner with AL uncertainty signals for Phase 3."""

    name = "wgisd_detection_fasterrcnn_mobilenetv3"

    def __init__(
        self,
        config: WgisdDetectionRunnerConfig,
        val_ids: Sequence[str],
        test_ids: Sequence[str],
    ) -> None:
        self._config = config
        self._train_device = torch.device(config.device)
        self._score_device = torch.device("cpu")
        self._quantization_applied = False

        dataset = _CocoDetectionDataset(
            images_root=Path(config.images_root),
            annotations_path=Path(config.annotations_path),
            max_samples=config.max_samples,
        )
        self._train_dataset = dataset
        self._eval_dataset = dataset
        self._num_classes = dataset.num_classes

        self._val_indices = _ids_to_indices(val_ids)
        self._test_indices = _ids_to_indices(test_ids)

        self._model = _build_detector(
            num_classes=self._num_classes,
            pretrained_backbone=config.pretrained_backbone,
        ).to(self._train_device)
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self._inference_model: nn.Module = self._model.eval()
        self._refresh_inference_model()

    @classmethod
    def inspect_dataset(
        cls, images_root: str, annotations_path: str, max_samples: int | None = None
    ) -> tuple[int, int]:
        dataset = _CocoDetectionDataset(
            images_root=Path(images_root),
            annotations_path=Path(annotations_path),
            max_samples=max_samples,
        )
        return len(dataset), dataset.num_classes

    def train_round(
        self, round_index: int, seed: int, labeled_ids: Sequence[str]
    ) -> Mapping[str, float]:
        _set_seed(seed + round_index)
        labeled_indices = _ids_to_indices(labeled_ids)
        if not labeled_indices:
            return {
                "loss": math.nan,
                "precision50_val": math.nan,
                "recall50_val": math.nan,
                "map50_proxy_val": math.nan,
                "precision50_test": math.nan,
                "recall50_test": math.nan,
                "map50_proxy_test": math.nan,
                "quantization_applied": float(self._quantization_applied),
            }

        loader = DataLoader(
            Subset(self._train_dataset, labeled_indices),
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=self._config.num_workers,
            collate_fn=_collate_detection,
        )

        self._model.train()
        total_loss = 0.0
        steps = 0
        for _ in range(self._config.epochs_per_round):
            for images, targets in loader:
                images = [image.to(self._train_device) for image in images]
                targets = [
                    {key: value.to(self._train_device) for key, value in target.items()}
                    for target in targets
                ]
                self._optimizer.zero_grad(set_to_none=True)
                loss_dict = self._model(images, targets)
                loss = sum(loss_dict.values())
                loss.backward()
                self._optimizer.step()
                total_loss += float(loss.item())
                steps += 1

        mean_loss = total_loss / float(steps) if steps else math.nan
        self._refresh_inference_model()
        val_metrics = self._evaluate_subset(self._val_indices)
        test_metrics = self._evaluate_subset(self._test_indices)
        return {
            "loss": mean_loss,
            "precision50_val": val_metrics["precision50"],
            "recall50_val": val_metrics["recall50"],
            "map50_proxy_val": val_metrics["map50_proxy"],
            "precision50_test": test_metrics["precision50"],
            "recall50_test": test_metrics["recall50"],
            "map50_proxy_test": test_metrics["map50_proxy"],
            "quantization_applied": float(self._quantization_applied),
        }

    def score_unlabeled(self, unlabeled_ids: Sequence[str]) -> list[SelectionCandidate]:
        indices = _ids_to_indices(unlabeled_ids)
        if not indices:
            return []

        loader = DataLoader(
            Subset(self._eval_dataset, indices),
            batch_size=self._config.score_batch_size,
            shuffle=False,
            num_workers=self._config.num_workers,
            collate_fn=_collate_detection,
        )

        self._inference_model.eval()
        candidates: list[SelectionCandidate] = []
        pointer = 0
        with torch.no_grad():
            for images, _targets in loader:
                for image in images:
                    output = self._predict_single(image)
                    cls_unc = classification_uncertainty_from_scores(
                        output["scores"], top_n=self._config.score_top_n
                    )
                    loc_unc = localization_uncertainty_from_detection(
                        image=image,
                        output=output,
                        model=self._inference_model,
                        score_threshold=self._config.score_threshold,
                        top_n=self._config.score_top_n,
                        use_tta=self._config.localization_tta,
                    )
                    combined = (self._config.uncertainty_alpha * cls_unc) + (
                        (1.0 - self._config.uncertainty_alpha) * loc_unc
                    )
                    sample_id = unlabeled_ids[pointer]
                    pointer += 1
                    candidates.append(
                        SelectionCandidate(
                            sample_id=str(sample_id),
                            score=float(combined),
                            embedding=(float(cls_unc), float(loc_unc)),
                            metadata={
                                "uncertainty_combined": float(combined),
                                "uncertainty_classification": float(cls_unc),
                                "uncertainty_localization": float(loc_unc),
                                "det_count": int(output["boxes"].shape[0]),
                            },
                        )
                    )
        return candidates

    def _evaluate_subset(self, indices: Sequence[int]) -> dict[str, float]:
        if not indices:
            return {"precision50": math.nan, "recall50": math.nan, "map50_proxy": math.nan}

        loader = DataLoader(
            Subset(self._eval_dataset, list(indices)),
            batch_size=1,
            shuffle=False,
            num_workers=self._config.num_workers,
            collate_fn=_collate_detection,
        )

        self._inference_model.eval()
        total_tp = 0
        total_fp = 0
        total_fn = 0
        with torch.no_grad():
            for images, targets in loader:
                image = images[0]
                target = targets[0]
                output = self._predict_single(image)
                pred_mask = output["scores"] >= self._config.score_threshold
                pred_boxes = output["boxes"][pred_mask]
                pred_labels = output["labels"][pred_mask]
                pred_scores = output["scores"][pred_mask]
                gt_boxes = target["boxes"]
                gt_labels = target["labels"]
                tp, fp, fn = greedy_match_counts(
                    pred_boxes=pred_boxes,
                    pred_labels=pred_labels,
                    pred_scores=pred_scores,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    iou_threshold=0.5,
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn
        precision, recall, map50_proxy = compute_detection_metrics(
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
        )
        return {
            "precision50": precision,
            "recall50": recall,
            "map50_proxy": map50_proxy,
        }

    def _predict_single(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image_for_model = image.to(self._score_device)
        outputs = self._inference_model([image_for_model])
        output = outputs[0]
        return {
            "boxes": output["boxes"].detach().cpu(),
            "labels": output["labels"].detach().cpu(),
            "scores": output["scores"].detach().cpu(),
        }

    def _refresh_inference_model(self) -> None:
        if self._config.quantization_mode != "int8":
            self._inference_model = copy.deepcopy(self._model).to(self._score_device).eval()
            self._quantization_applied = False
            return

        try:
            fp32_copy = copy.deepcopy(self._model).to(torch.device("cpu")).eval()
            quantized = torch.ao.quantization.quantize_dynamic(
                fp32_copy,
                {nn.Linear},
                dtype=torch.qint8,
            )
            self._inference_model = quantized
            self._quantization_applied = True
        except Exception:
            self._inference_model = copy.deepcopy(self._model).to(self._score_device).eval()
            self._quantization_applied = False


class _CocoDetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(
        self, images_root: Path, annotations_path: Path, max_samples: int | None = None
    ) -> None:
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
        images = sorted(payload.get("images", []), key=lambda item: item["id"])
        if max_samples is not None:
            images = images[: max_samples]
        self._images = images
        self._images_root = images_root

        categories = payload.get("categories", [])
        self._cat_id_to_index = {
            int(cat["id"]): idx + 1 for idx, cat in enumerate(categories)
        }
        self.num_classes = len(categories)
        if self.num_classes <= 0:
            raise ValueError("COCO annotation must include at least one category.")

        self._image_id_to_annotations: dict[int, list[dict[str, Any]]] = {}
        image_ids = {int(image["id"]) for image in self._images}
        for ann in payload.get("annotations", []):
            image_id = int(ann["image_id"])
            if image_id not in image_ids:
                continue
            self._image_id_to_annotations.setdefault(image_id, []).append(ann)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        item = self._images[index]
        image_path = self._images_root / str(item["file_name"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = tvf.to_tensor(image)

        image_id = int(item["id"])
        anns = self._image_id_to_annotations.get(image_id, [])
        boxes: list[list[float]] = []
        labels: list[int] = []
        for ann in anns:
            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self._cat_id_to_index[int(ann["category_id"])])

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((labels_tensor.shape[0],), dtype=torch.int64),
        }
        return image_tensor, target


def _build_detector(num_classes: int, pretrained_backbone: bool) -> nn.Module:
    weights_backbone = (
        MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    )
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None,
        weights_backbone=weights_backbone,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def classification_uncertainty_from_scores(scores: torch.Tensor, top_n: int) -> float:
    if scores.numel() == 0:
        return 1.0
    n = min(top_n, scores.shape[0])
    top_scores = scores[:n]
    confidence = float(top_scores.mean().item())
    uncertainty = 1.0 - confidence
    return float(max(0.0, min(1.0, uncertainty)))


def localization_uncertainty_from_detection(
    image: torch.Tensor,
    output: dict[str, torch.Tensor],
    model: nn.Module,
    score_threshold: float,
    top_n: int,
    use_tta: bool,
) -> float:
    if not use_tta:
        return 0.0

    orig_mask = output["scores"] >= score_threshold
    orig_boxes = output["boxes"][orig_mask][:top_n]
    if orig_boxes.numel() == 0:
        return 1.0

    flipped = torch.flip(image, dims=[2])
    with torch.no_grad():
        flipped_output = model([flipped])[0]
    flip_boxes = flipped_output["boxes"].detach().cpu()
    flip_scores = flipped_output["scores"].detach().cpu()
    flip_mask = flip_scores >= score_threshold
    flip_boxes = flip_boxes[flip_mask][:top_n]
    if flip_boxes.numel() == 0:
        return 1.0

    width = image.shape[2]
    flip_boxes = _unflip_boxes(flip_boxes, width=width)
    iou_matrix = _box_iou_matrix(orig_boxes, flip_boxes)
    best_iou, _ = iou_matrix.max(dim=1)
    jitter = 1.0 - float(best_iou.mean().item())
    return float(max(0.0, min(1.0, jitter)))


def greedy_match_counts(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    if pred_boxes.numel() == 0:
        return 0, 0, int(gt_boxes.shape[0])
    if gt_boxes.numel() == 0:
        return 0, int(pred_boxes.shape[0]), 0

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]
    pred_labels = pred_labels[order]

    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    for idx in range(pred_boxes.shape[0]):
        p_box = pred_boxes[idx : idx + 1]
        p_label = int(pred_labels[idx].item())

        ious = _box_iou_matrix(p_box, gt_boxes).squeeze(0)
        best_iou, best_idx_tensor = torch.max(ious, dim=0)
        best_idx = int(best_idx_tensor.item())
        label_match = p_label == int(gt_labels[best_idx].item())
        if (
            float(best_iou.item()) >= iou_threshold
            and label_match
            and best_idx not in matched_gt
        ):
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1
    fn = int(gt_boxes.shape[0]) - len(matched_gt)
    return tp, fp, fn


def compute_detection_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def _box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    union = torch.where(union > 0, union, torch.ones_like(union))
    return inter / union


def _unflip_boxes(boxes: torch.Tensor, width: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    unflipped = boxes.clone()
    x1 = unflipped[:, 0].clone()
    x2 = unflipped[:, 2].clone()
    unflipped[:, 0] = width - x2
    unflipped[:, 2] = width - x1
    return unflipped


def _collate_detection(batch: Sequence[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


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
