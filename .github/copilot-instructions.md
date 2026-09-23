# Copilot Instructions for ML-Marathon-2026

## Project shape
- This is a notebook-first ML repo for the Snapshot Serengeti WI (Oh Deer) competition.
- The main working artifact is `boundingAndClassification.ipynb`; `test.py` is a tiny import smoke test.
- Local competition data lives under `snapshot-wi-oh-deer/`, which is ignored by git via `.gitignore`.
- Treat the data bundle as local-only; do not move, download, or commit raw/processed datasets.

## Environment and dependencies
- The project targets Python 3.12 (`.python-version`) and is managed with `uv` (`pyproject.toml`, `uv.lock`).
- Current runtime dependencies are `librosa`, `soundfile`, `setuptools`, and `pytorchwildlife[all]`.
- Notebook and script imports currently rely on `numpy` and `PytorchWildlife.models.detection`.
- If imports fail in analysis, assume the environment is not synced yet rather than the code being wrong.

## Data and notebook conventions
- The notebook currently hardcodes a local absolute photo path like `/Users/.../snapshot-wi-oh-deer/Snapshot_WI-Oh_Deer_Photos`; prefer repo-relative or configurable paths when editing.
- Keep exploratory work in the notebook unless it is becoming reusable logic.
- If code starts to stabilize, move it into a small module or script rather than growing the notebook indefinitely.

## Working style
- Preserve the repo owner’s branch-scoped work and keep changes narrowly focused.
- Do not introduce new dependencies without asking; the repo explicitly treats dependency changes as review-worthy.
- Avoid committing generated artifacts, model outputs, checkpoints, or dataset dumps.
- Keep PR-sized changes small and reviewable; one notebook-level idea or one pipeline step at a time.

## Useful files
- `README.md` is currently minimal, so the notebook and contribution guide carry most of the practical guidance.
- `CONTRIBUTING.md` defines the collaboration rules, repo layout, and what counts as shared versus personal code.
- `pyproject.toml` and `uv.lock` define the execution environment.
- `snapshot-wi-oh-deer/` contains the competition data bundle and should stay local.

## When editing
- Prefer minimal, notebook-friendly changes that preserve exploratory flow.
- Match existing import style and keep quick experiments lightweight.
- If you add reusable logic, place it in a shared location only when it is actually ready for reuse.
