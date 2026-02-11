from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from edge_al_pipeline.config import ExperimentConfig
from edge_al_pipeline.contracts import (
    DatasetSplits,
    MetricRecord,
    ProfileRecord,
    SelectionRecord,
)


class ArtifactStore:
    """Writes experiment artifacts using the contract from IMPLEMENTATION_SPEC.md."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.splits_path = self.run_dir / "splits.json"
        self.metrics_path = self.run_dir / "metrics.csv"
        self.profile_path = self.run_dir / "profile.csv"
        self.config_snapshot_path = self.run_dir / "config_snapshot.json"

    def initialize(
        self, config: ExperimentConfig, config_source: Path | None = None
    ) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._init_csv(
            self.metrics_path,
            ["round_index", "seed", "split", "metric", "value"],
        )
        self._init_csv(
            self.profile_path,
            [
                "round_index",
                "stage",
                "latency_ms",
                "memory_mb",
                "quantization_mode",
                "device",
                "notes",
            ],
        )
        payload = config.to_dict()
        payload["created_at_utc"] = _now_utc_iso()
        if config_source is not None:
            payload["config_source"] = str(config_source)
        self.config_snapshot_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def write_splits(self, splits: DatasetSplits, dataset_hash: str) -> Path:
        splits.validate()
        payload = {
            "created_at_utc": _now_utc_iso(),
            "dataset_hash": dataset_hash,
            "counts": splits.counts(),
            "splits": {
                "L": splits.labeled,
                "U": splits.unlabeled,
                "V": splits.val,
                "T": splits.test,
            },
        }
        self.splits_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.splits_path

    def write_round_selection(
        self, round_index: int, records: Sequence[SelectionRecord]
    ) -> Path:
        path = self.run_dir / f"round_{round_index}_selected.csv"
        with path.open("w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=[
                    "round_index",
                    "seed",
                    "strategy",
                    "sample_id",
                    "score",
                    "metadata",
                ],
            )
            writer.writeheader()
            for record in records:
                row = asdict(record)
                row["metadata"] = json.dumps(record.metadata, sort_keys=True)
                writer.writerow(row)
        return path

    def append_metrics(self, records: Iterable[MetricRecord]) -> None:
        self._append_rows(
            self.metrics_path,
            [
                {
                    "round_index": record.round_index,
                    "seed": record.seed,
                    "split": record.split,
                    "metric": record.metric,
                    "value": record.value,
                }
                for record in records
            ],
        )

    def append_profile(self, records: Iterable[ProfileRecord]) -> None:
        self._append_rows(
            self.profile_path,
            [
                {
                    "round_index": record.round_index,
                    "stage": record.stage,
                    "latency_ms": round(record.latency_ms, 3),
                    "memory_mb": record.memory_mb,
                    "quantization_mode": record.quantization_mode,
                    "device": record.device,
                    "notes": record.notes,
                }
                for record in records
            ],
        )

    @staticmethod
    def _init_csv(path: Path, headers: list[str]) -> None:
        if path.exists():
            return
        with path.open("w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=headers)
            writer.writeheader()

    @staticmethod
    def _append_rows(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        with path.open("a", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=list(rows[0].keys()))
            writer.writerows(rows)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
