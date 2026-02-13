# What To Run Now

This is the shortest path for your current Week 1 objective.

## 1) Activate environment

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2) Confirm GPU is available

```powershell
python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
```

Expected: `torch.cuda.is_available()` is `True`.

## 3) Run the 3 Week 1 toy experiments

```powershell
python.exe src\main.py --mode phase1b --config .\configs\cdgp_week1_toy_cifar10_random.json
python.exe src\main.py --mode phase1b --config .\configs\cdgp_week1_toy_cifar10_entropy.json
python.exe src\main.py --mode phase1b --config .\configs\cdgp_week1_toy_cifar10_domain_guided.json
```

## 4) Where outputs are written

- `runs/cdgp_week1_toy_cifar10_random/...`
- `runs/cdgp_week1_toy_cifar10_entropy/...`
- `runs/cdgp_week1_toy_cifar10_domain_guided/...`

Each run folder contains `metrics.csv`, `config_snapshot.json`, selections, and profiles.

## 5) Submission-style batch (base profile)

Run:

```powershell
.\scripts\run_week1_batch.ps1
```

Generate comparison report:

```powershell
python.exe .\scripts\report_week1_batch.py --profile base
python.exe .\scripts\diagnose_week1_domain_signal.py --profile base
```

## 6) Calibrated domain-guided follow-up (new)

Run calibrated variants only:

```powershell
.\scripts\run_week1_batch_calibrated.ps1
```

If you also want to refresh random+entropy in the same batch:

```powershell
.\scripts\run_week1_batch_calibrated.ps1 -IncludeBaselines
```

Generate calibrated comparison + diagnostics:

```powershell
python.exe .\scripts\report_week1_batch.py --profile calibrated
python.exe .\scripts\diagnose_week1_domain_signal.py --profile calibrated
```

## Config folder layout

- `configs/` : only the 3 active Week 1 configs
- `configs/week1_batch/` : submission-style Week 1 batch configs
- `configs/smoke/` : smoke/debug configs
- `configs/cloud/` : Colab/Kaggle configs
- `configs/legacy/` : older phase/gate configs

Kaggle notebook template:
- `docs/KAGGLE_WEEK1_NOTEBOOK.md`
