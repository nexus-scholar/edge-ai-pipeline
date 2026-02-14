from __future__ import annotations

from typing import Any, Mapping


CLASSIFICATION_BACKBONES = frozenset(
    {
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "mobilenetv3_small_100",
        "mobilenetv3_large_100",
        "mobilenet_v4",
        "mobilenet_v4_small",
        "mobilenet_v4_medium",
        "mobilenet_v4_large",
        "mobilenetv4_conv_small",
        "mobilenetv4_conv_medium",
        "mobilenetv4_conv_large",
        "mobilenetv4_hybrid_medium",
        "mobilenetv4_hybrid_large",
        "resnet18",
        "resnet50",
    }
)
DETECTION_BACKBONES = frozenset({
    "mobilenet_v3_large_320_fpn",
    "mobilenet_v3_large",
    "mobilenet_v4_medium",
    "mobilenet_v4_small",
    "mobilenet_v4_large"
})
SUPPORTED_BACKBONES = CLASSIFICATION_BACKBONES | DETECTION_BACKBONES

_MODEL_NAME_TO_BACKBONE = {
    "mobilenet_v3_small": "mobilenet_v3_small",
    "mobilenet_v3_large": "mobilenet_v3_large",
    "mobilenet_v4": "mobilenet_v4",
    "mobilenet_v4_small": "mobilenet_v4_small",
    "mobilenet_v4_medium": "mobilenet_v4_medium",
    "mobilenet_v4_large": "mobilenet_v4_large",
    "mobilenet_v4_hybrid_medium": "mobilenet_v4_hybrid_medium",
    "mobilenet_v4_hybrid_large": "mobilenet_v4_hybrid_large",
    "mobilenetv4_conv_small": "mobilenetv4_conv_small",
    "mobilenetv4_conv_medium": "mobilenetv4_conv_medium",
    "mobilenetv4_conv_large": "mobilenetv4_conv_large",
    "mobilenetv4_hybrid_medium": "mobilenetv4_hybrid_medium",
    "mobilenetv4_hybrid_large": "mobilenetv4_hybrid_large",
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "fasterrcnn_mobilenet_v3_large_320_fpn": "mobilenet_v3_large_320_fpn",
}


def normalize_backbone_name(name: str) -> str:
    return name.strip().lower()


def resolve_backbone_name(
    model_name: str, model_params: Mapping[str, Any]
) -> str | None:
    raw_name = model_params.get("backbone_name")
    if raw_name is None:
        return _MODEL_NAME_TO_BACKBONE.get(normalize_backbone_name(model_name))
    if not isinstance(raw_name, str):
        raise ValueError("model_params.backbone_name must be a string when provided.")
    normalized = normalize_backbone_name(raw_name)
    if not normalized:
        raise ValueError("model_params.backbone_name must not be empty when provided.")
    return normalized
