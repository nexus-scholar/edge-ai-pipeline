# Gemini CLI Context: Edge-AL Pipeline

This repository implements a reliable, repeatable **Active Learning (AL) pipeline** specifically designed for agricultural computer vision and optimized for edge constraints.

## Project Overview

- **Core Goal:** Demonstrate that uncertainty-based AL strategies can outperform random sampling on agricultural tasks while remaining feasible for edge deployment (e.g., Raspberry Pi, Jetson, Coral).
- **Architecture:**
  - **Data Pool Manager (`data_pool.py`):** Manages labeled (`L`) and unlabeled (`U`) sets, tracking splits and hashes.
  - **Pipeline Orchestrator (`pipeline.py`):** Drives the iterative AL loop: Train -> Score -> Select -> Label -> Update.
  - **Query Strategy Engine (`strategies/`):** Pluggable strategies including Random, Entropy, Domain-Guided, and K-Center Greedy.
  - **Evaluation Gates:** Automated validation gates (Gate A: logic, Gate B: transfer, Gate C: field proxy).
  - **Edge Profiler (`profiling/`):** Measures inference latency, scan time, and resource usage per AL round.
- **Technologies:** Python 3.13+, PyTorch, Torchvision, `uv` (package management), `pytest`.

## Curriculum Phases

1.  **Phase 1 (Fashion-MNIST):** "Clean room" verification of AL logic.
2.  **Phase 1b (CIFAR-10):** Validation of RGB handling and robustness.
3.  **Phase 2 (PlantVillage/Fruits-360):** Controlled agricultural feature learning and imbalance stress tests.
4.  **Phase 3 (WGISD):** High-quality field proxy with detection and localization uncertainty.
5.  **Phase 4 (Field Data):** Real-world complexity with "Teacher/Fog" filtering for noise mitigation.

## Building and Running

### Prerequisites
- Python >= 3.13
- `uv` (recommended) or `pip`

### Setup
```bash
# Using uv (preferred)
uv sync

# Using pip
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -e .
```

### Key Commands
- **Run Experiment:** `python src/main.py --mode <phase> --config <config_path>`
  - Modes: `bootstrap`, `phase1`, `phase1b`, `phase2`, `phase3`, `gate_a`, `gate_b`, `gate_c`.
- **Run Tests:** `python -m pytest`
- **Bootstrap Run Directory:** `python src/main.py --mode bootstrap --config configs/cdgp_week1_toy_cifar10_random.json`

## Development Conventions

- **Module Organization:** Domain-based packages under `src/edge_al_pipeline/`.
- **Coding Style:** PEP 8, strict type hints for public APIs, 4-space indentation.
- **Naming:** `snake_case` (modules/functions), `PascalCase` (classes), `UPPER_CASE` (constants).
- **Reproducibility:** Always pass explicit seeds via configuration; avoid hardcoding magic numbers.
- **Testing:** Mirror `src/` structure in `tests/`. Use `pytest`. Fast unit tests preferred; smoke tests for pipeline flows.
- **Commit Messages:** Follow `<type>: <short description>` (e.g., `feat: add k-center greedy strategy`).

## Key Files & Directories
- `src/main.py`: Primary entry point for all experiment modes and gates.
- `configs/`: JSON configuration files for various experiment phases.
- `docs/`: In-depth documentation, including `GOALS.md` and `IMPLEMENTATION_SPEC.md`.
- `scripts/`: Utility scripts for batch runs and reporting (e.g., `run_week1_batch.ps1`).
- `runs/`: (Generated) Artifacts from experiment runs, including metrics, splits, and checkpoints.

## Key Lessons & Project History

- **Validation Strategy:** The project uses a standalone `agri_external_validation` dataset for model validation, rather than a full merge. The foundation dataset serves only as a reference for class mapping.
- **Class Mapping (Wheat):** To avoid misclassification between whole plants/heads and leaves, the 'Healthy Wheat' dataset (whole plants) is mapped to `wheat_healthy_whole`, distinct from the foundation's `wheat_healthyleaf`.
