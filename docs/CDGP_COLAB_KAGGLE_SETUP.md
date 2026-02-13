# CDGP Cloud Run Setup (Colab + Kaggle)

This guide is a runbook to execute CDGP experiments from your GitHub repo with minimal setup friction.

## 1) One-Time Prep (Laptop)

1. Confirm repo is pushed and clean:
- `git status`
- `git push origin <branch>`
2. Keep datasets outside Git and document expected paths.
3. Create cloud override configs in `configs/` (for example `*_colab.json`, `*_kaggle.json`) instead of editing base configs.
4. Freeze your initial seed list and experiment naming convention before large runs.

## 2) Standard Naming Convention (Recommended)

Use consistent names so reruns are idempotent and traceable.

- Experiment name pattern:
`<phase>_<dataset>_<backbone>_<strategy>_<env>_<version>`

Examples:
- `cdgp_toy_mnistm_resnet18_domain_guided_colab_v1`
- `cdgp_clean_wgisd_mobilenetv3_entropy_kaggle_v1`

## 3) Colab Setup

### 3.1 Runtime

1. Open Colab notebook.
2. Set runtime to GPU if needed: `Runtime -> Change runtime type -> GPU`.

### 3.2 Clone + install

```bash
!git clone <YOUR_GITHUB_REPO_URL>
%cd edge-al-pipeline
!python -m pip install --upgrade pip
!python -m pip install -e .[dev,phase1,phase2,phase3]
```

### 3.3 Data and persistence (recommended)

```python
from google.colab import drive
drive.mount('/content/drive')
```

Suggested layout on Drive:
- `/content/drive/MyDrive/edge-al-data/` for datasets
- `/content/drive/MyDrive/edge-al-runs/` for run artifacts

### 3.4 Config override pattern

Use cloud-specific config files that only change:
- dataset root paths
- output root
- device (`cuda` or `cpu`)

Example expected values:
- dataset root: `/content/drive/MyDrive/edge-al-data/<dataset>`
- output root: `/content/drive/MyDrive/edge-al-runs`

### 3.5 Run commands

```bash
!python src/main.py --mode phase1 --config configs/phase1_fashion_mnist_colab.json
!python src/main.py --mode gate_a --config configs/phase1_fashion_mnist_colab.json --budget-ratio 0.10 --min-improvement 0.05
```

For CDGP phases, use your `cdgp_*_colab.json` files with the same pattern.

### 3.6 Save and export results

1. Keep outputs in Drive-backed `output_root`.
2. Optionally zip a run folder for sharing:

```bash
!zip -r cdgp_run_bundle.zip /content/drive/MyDrive/edge-al-runs/<experiment_dir>
```

## 4) Kaggle Setup

### 4.1 Notebook config

1. Create Kaggle Notebook.
2. Enable Internet if you need `git clone`.
3. Enable GPU if required.

### 4.2 Clone + install

```bash
!git clone <YOUR_GITHUB_REPO_URL>
%cd edge-al-pipeline
!python -m pip install --upgrade pip
!python -m pip install -e .[dev,phase1,phase2,phase3]
```

If internet is disabled, upload repo as a Kaggle Dataset and copy it to `/kaggle/working/`.

### 4.3 Data layout

Kaggle input datasets are read-only under:
- `/kaggle/input/<dataset-name>/`

Writable output path:
- `/kaggle/working/`

Recommended:
- dataset root: `/kaggle/input/<dataset-name>/<subdir>`
- output root: `/kaggle/working/runs`

### 4.4 Run commands

```bash
!python src/main.py --mode phase3 --config configs/phase3_wgisd_kaggle.json
!python src/main.py --mode gate_c --config configs/phase3_wgisd_kaggle.json --gate-c-min-improvement 0.0
```

### 4.5 Persist results

1. Keep generated runs under `/kaggle/working/runs`.
2. Download artifacts manually or publish `/kaggle/working/runs` as a Kaggle Dataset version.

## 5) Plug-and-Play Backbone/Strategy in Cloud Runs

For each cloud config, explicitly set:
- `model_name`
- `model_params.backbone_name`
- `strategy_name`
- `strategy_params`
- `edge_device`
- `quantization_mode`

Do not change code between backbone/strategy comparisons; only change config files.

## 6) Idempotent Run Rules (Cloud)

1. Before launching, check if target report already exists for the same config+seed.
2. If exists and config is unchanged, skip rerun.
3. If config changed, bump experiment suffix (`_v2`, `_v3`, ...).
4. Never overwrite or delete previous run directories during analysis.
5. Keep a changelog markdown file listing config deltas per run batch.

## 7) Minimal Troubleshooting

1. OOM on GPU:
- reduce batch size
- reduce score batch size
- reduce image size

2. Slow CPU-only runs:
- use smoke configs first
- reduce rounds/query size for debug

3. Path errors:
- print and verify `dataset.root` and `output_root` at notebook start
- use absolute paths in cloud configs

4. Reproducibility drift:
- verify seed lists
- verify config snapshot files
- verify same dataset version/path for compared runs

## 8) Quick Start Checklist

- [ ] Repo cloned from GitHub.
- [ ] Dependencies installed.
- [ ] Cloud-specific config selected.
- [ ] Dataset paths validated.
- [ ] Output path writable and persistent.
- [ ] Smoke run completed.
- [ ] Full run launched.
- [ ] Gate/report artifacts generated and archived.
