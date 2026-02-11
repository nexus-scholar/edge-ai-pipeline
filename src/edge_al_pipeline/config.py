from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

TaskType = Literal["classification", "detection", "segmentation"]
StrategyName = Literal["random", "entropy", "k_center_greedy"]
QuantizationMode = Literal["fp32", "int8"]

_SUPPORTED_TASKS = {"classification", "detection", "segmentation"}
_SUPPORTED_STRATEGIES = {"random", "entropy", "k_center_greedy"}
_SUPPORTED_QUANTIZATION = {"fp32", "int8"}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: str
    version: str = "unknown"
    task: TaskType = "classification"
    num_classes: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetConfig":
        return cls(
            name=str(data["name"]),
            root=str(data["root"]),
            version=str(data.get("version", "unknown")),
            task=str(data.get("task", "classification")),
            num_classes=data.get("num_classes"),
        )

    def validate(self) -> None:
        if not self.name:
            raise ValueError("dataset.name must not be empty.")
        if not self.root:
            raise ValueError("dataset.root must not be empty.")
        if self.task not in _SUPPORTED_TASKS:
            raise ValueError(
                f"dataset.task must be one of {sorted(_SUPPORTED_TASKS)}; got {self.task!r}."
            )
        if self.num_classes is not None:
            if self.task == "classification" and self.num_classes <= 1:
                raise ValueError(
                    "dataset.num_classes must be greater than 1 for classification."
                )
            if self.task in {"detection", "segmentation"} and self.num_classes <= 0:
                raise ValueError(
                    "dataset.num_classes must be greater than 0 for detection/segmentation."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": self.root,
            "version": self.version,
            "task": self.task,
            "num_classes": self.num_classes,
        }


@dataclass(frozen=True)
class BootstrapConfig:
    pool_size: int
    initial_labeled_size: int
    val_size: int
    test_size: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BootstrapConfig":
        return cls(
            pool_size=int(data["pool_size"]),
            initial_labeled_size=int(data["initial_labeled_size"]),
            val_size=int(data["val_size"]),
            test_size=int(data["test_size"]),
        )

    def validate(self) -> None:
        if self.pool_size <= 0:
            raise ValueError("bootstrap.pool_size must be greater than 0.")
        if self.initial_labeled_size < 0:
            raise ValueError("bootstrap.initial_labeled_size must be >= 0.")
        if self.val_size < 0:
            raise ValueError("bootstrap.val_size must be >= 0.")
        if self.test_size < 0:
            raise ValueError("bootstrap.test_size must be >= 0.")
        known_total = self.initial_labeled_size + self.val_size + self.test_size
        if known_total >= self.pool_size:
            raise ValueError(
                "bootstrap sizes consume the full pool. Reserve room for unlabeled data."
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "pool_size": self.pool_size,
            "initial_labeled_size": self.initial_labeled_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    output_root: str
    dataset: DatasetConfig
    model_name: str
    strategy_name: StrategyName
    model_params: dict[str, Any] = field(default_factory=dict)
    strategy_params: dict[str, Any] = field(default_factory=dict)
    rounds: int = 1
    query_size: int = 1
    seeds: list[int] = field(default_factory=lambda: [42])
    quantization_mode: QuantizationMode = "fp32"
    teacher_enabled: bool = False
    edge_device: str = "unknown"
    bootstrap: BootstrapConfig = field(
        default_factory=lambda: BootstrapConfig(
            pool_size=1000,
            initial_labeled_size=50,
            val_size=100,
            test_size=100,
        )
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            experiment_name=str(data["experiment_name"]),
            output_root=str(data.get("output_root", "runs")),
            dataset=DatasetConfig.from_dict(dict(data["dataset"])),
            model_name=str(data["model_name"]),
            model_params=dict(data.get("model_params", {})),
            strategy_name=str(data["strategy_name"]),
            strategy_params=dict(data.get("strategy_params", {})),
            rounds=int(data.get("rounds", 1)),
            query_size=int(data.get("query_size", 1)),
            seeds=[int(seed) for seed in data.get("seeds", [42])],
            quantization_mode=str(data.get("quantization_mode", "fp32")),
            teacher_enabled=bool(data.get("teacher_enabled", False)),
            edge_device=str(data.get("edge_device", "unknown")),
            bootstrap=BootstrapConfig.from_dict(
                dict(
                    data.get(
                        "bootstrap",
                        {
                            "pool_size": 1000,
                            "initial_labeled_size": 50,
                            "val_size": 100,
                            "test_size": 100,
                        },
                    )
                )
            ),
        )

    def validate(self) -> None:
        if not self.experiment_name:
            raise ValueError("experiment_name must not be empty.")
        if not self.output_root:
            raise ValueError("output_root must not be empty.")
        if not self.model_name:
            raise ValueError("model_name must not be empty.")
        if self.strategy_name not in _SUPPORTED_STRATEGIES:
            raise ValueError(
                "strategy_name must be one of "
                f"{sorted(_SUPPORTED_STRATEGIES)}; got {self.strategy_name!r}."
            )
        if self.rounds <= 0:
            raise ValueError("rounds must be greater than 0.")
        if self.query_size <= 0:
            raise ValueError("query_size must be greater than 0.")
        if not self.seeds:
            raise ValueError("seeds must include at least one integer.")
        if self.quantization_mode not in _SUPPORTED_QUANTIZATION:
            raise ValueError(
                "quantization_mode must be one of "
                f"{sorted(_SUPPORTED_QUANTIZATION)}; got {self.quantization_mode!r}."
            )

        self.dataset.validate()
        self.bootstrap.validate()
        if self.query_size > self.bootstrap.pool_size:
            raise ValueError("query_size cannot exceed bootstrap.pool_size.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "output_root": self.output_root,
            "dataset": self.dataset.to_dict(),
            "model_name": self.model_name,
            "model_params": self.model_params,
            "strategy_name": self.strategy_name,
            "strategy_params": self.strategy_params,
            "rounds": self.rounds,
            "query_size": self.query_size,
            "seeds": self.seeds,
            "quantization_mode": self.quantization_mode,
            "teacher_enabled": self.teacher_enabled,
            "edge_device": self.edge_device,
            "bootstrap": self.bootstrap.to_dict(),
        }


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.suffix.lower() != ".json":
        raise ValueError("This scaffold currently supports JSON config files only.")

    raw = config_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    config = ExperimentConfig.from_dict(data)
    config.validate()
    return config


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return target
