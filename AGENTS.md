# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains Python source code. `src/main.py` is the current runnable entry point.
- `docs/` stores project documentation and research inputs (for example, `docs/GOALS.md` and `docs/IMPLEMENTATION_SPEC.md`).
- `pyproject.toml` defines package metadata and runtime requirements (`python>=3.13`).
- Keep new implementation code in `src/` and keep planning/report artifacts in `docs/`.
- When adding modules, prefer domain-based packages (for example: `src/data_pool.py`, `src/query_strategies.py`, `src/profiler.py`).

## Build, Test, and Development Commands
- `python -m venv .venv` creates a local virtual environment.
- `.\.venv\Scripts\Activate.ps1` activates the environment in PowerShell.
- `python -m pip install -e .` installs the project in editable mode.
- `python src/main.py` runs the current local entry point.
- `python -m pytest -q` runs tests (after tests are added).

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and explicit, readable function names.
- Use `snake_case` for modules/functions/variables, `PascalCase` for classes, and `UPPER_CASE` for constants.
- Add type hints for public functions and data structures used across modules.
- Keep experimental logic reproducible: pass explicit seeds/config values instead of hardcoding magic numbers.
- No formatter/linter is configured in-repo yet; keep style consistent and easy to diff.

## Testing Guidelines
- Use `pytest` and place tests under `tests/`, mirroring `src/` modules.
- Name files `test_<module>.py` and test cases `test_<behavior>()`.
- Prefer fast unit tests for selection logic, pool updates, and metric calculations.
- For pipeline behavior, add small smoke tests with tiny datasets to validate one AL round end-to-end.

## Commit & Pull Request Guidelines
- The repository currently has no commit history; use clear, imperative commit subjects.
- Recommended format: `<type>: <short description>` (for example, `feat: add entropy query strategy`).
- Keep commits focused on one logical change and include related docs/config updates.
- PRs should include: purpose, key changes, validation steps/commands, and relevant metrics or plots for ML changes.
- Link the associated issue/task when available.

## Security & Configuration Tips
- Do not commit secrets, API keys, raw datasets, or large model artifacts.
- Keep environment-specific settings local (for example, `.env` files excluded by `.gitignore`).
- Document external data/model locations in `docs/` so results remain reproducible.
