# Kaggle Notebook Runbook (Week 1 CDGP)

Use this as a copy/paste template in a Kaggle Notebook.

Important notebook settings before running:
- Accelerator: `GPU` (T4 or better)
- Internet: `ON` (required for CIFAR10 download in these configs)

## Cell 1 (Markdown)

```markdown
# CDGP Week 1 on Kaggle
This notebook runs Week 1 CIFAR10 active learning experiments (random, entropy, and calibrated domain-guided).
```

## Cell 2 (Code) - GPU/Runtime Check

```python
import os
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
```

## Cell 3 (Code) - Clone Repo

```bash
%cd /kaggle/working
!rm -rf edge-al-pipeline
!git clone https://github.com/nexus-scholar/edge-ai-pipeline.git edge-al-pipeline
%cd /kaggle/working/edge-al-pipeline
```

## Cell 4 (Code) - Optional: install test deps

```bash
!python -m pip install -q --upgrade pip
!python -m pip install -q pytest
```

## Cell 5 (Code) - Quick sanity check (one config, one seed)

```bash
%cd /kaggle/working/edge-al-pipeline
!python - <<'PY'
import json
from pathlib import Path

src = Path("configs/cloud/cdgp_week1_batch_cifar10_domain_guided_calibrated_w02_kaggle.json")
dst = Path("configs/cloud/cdgp_week1_batch_cifar10_domain_guided_calibrated_w02_kaggle_quick.json")
cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["experiment_name"] = "cdgp_week1_batch_cifar10_domain_guided_calibrated_w02_quick_kaggle"
cfg["seeds"] = [7]
cfg["rounds"] = 2
dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("wrote", dst)
PY

!PYTHONPATH=src python src/main.py --mode phase1b --config configs/cloud/cdgp_week1_batch_cifar10_domain_guided_calibrated_w02_kaggle_quick.json
```

## Cell 6 (Code) - Run Week 1 batch on Kaggle

Core run (recommended first):

```bash
%cd /kaggle/working/edge-al-pipeline
!bash scripts/run_week1_batch_kaggle.sh core
```

Full sweep (includes calibrated w05/w08):

```bash
%cd /kaggle/working/edge-al-pipeline
!bash scripts/run_week1_batch_kaggle.sh full
```

## Cell 7 (Code) - Generate comparison reports

```bash
%cd /kaggle/working/edge-al-pipeline
!python scripts/report_week1_batch.py --profile calibrated
!python scripts/diagnose_week1_domain_signal.py --profile calibrated
```

## Cell 8 (Code) - Inspect outputs

```bash
%cd /kaggle/working/edge-al-pipeline
!ls -lah /kaggle/working/runs/reports | tail -n 20
```

## Cell 9 (Code) - Bundle artifacts for download

```bash
!cd /kaggle/working && zip -r week1_kaggle_runs_bundle.zip runs
!ls -lh /kaggle/working/week1_kaggle_runs_bundle.zip
```

## Cell 10 (Markdown)

```markdown
## After Run
- Download `/kaggle/working/week1_kaggle_runs_bundle.zip`
- On local machine, extract into your repo root so reports appear under `runs/reports/`
- Compare against previous local runs using the generated markdown report files
```
