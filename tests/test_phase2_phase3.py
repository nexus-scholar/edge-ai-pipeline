from __future__ import annotations

from pathlib import Path

from PIL import Image

from edge_al_pipeline.config import BootstrapConfig, DatasetConfig, ExperimentConfig
from edge_al_pipeline.experiments.phase2_agri_classification import _effective_bootstrap
from edge_al_pipeline.experiments.phase3_wgisd_setup import setup_phase3_wgisd
from edge_al_pipeline.models.image_folder_mobilenet_runner import (
    ImageFolderMobileNetRunner,
)


def test_effective_bootstrap_caps_pool_size():
    bootstrap = BootstrapConfig(
        pool_size=1000,
        initial_labeled_size=100,
        val_size=100,
        test_size=100,
    )
    effective = _effective_bootstrap(bootstrap, dataset_size=450)
    assert effective.pool_size == 450
    assert effective.initial_labeled_size == 100


def test_imagefolder_inspection_reads_size_and_classes(tmp_path):
    _create_image_class(tmp_path, "class_a", 3)
    _create_image_class(tmp_path, "class_b", 2)
    size, classes = ImageFolderMobileNetRunner.inspect_dataset(str(tmp_path))
    assert size == 5
    assert classes == 2


def test_phase3_setup_writes_manifests(tmp_path):
    config = ExperimentConfig(
        experiment_name="phase3_setup_test",
        output_root=str(tmp_path / "runs"),
        dataset=DatasetConfig(
            name="wgisd",
            root="data/wgisd",
            version="1.0",
            task="detection",
            num_classes=5,
        ),
        model_name="yolo_nano",
        strategy_name="entropy",
        rounds=2,
        query_size=8,
        seeds=[5],
        quantization_mode="fp32",
        teacher_enabled=True,
        edge_device="cpu",
        bootstrap=BootstrapConfig(
            pool_size=100,
            initial_labeled_size=10,
            val_size=10,
            test_size=10,
        ),
    )
    summary = setup_phase3_wgisd(config, config_source=None)
    assert len(summary.results) == 1
    item = summary.results[0]
    assert item.setup_manifest_path.exists()
    assert item.uncertainty_plan_path.exists()
    assert item.teacher_policy_path.exists()


def _create_image_class(root: Path, class_name: str, count: int) -> None:
    class_dir = root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = Image.new("RGB", (16, 16), color=(index * 30 % 255, 80, 120))
        image.save(class_dir / f"{index}.png")
