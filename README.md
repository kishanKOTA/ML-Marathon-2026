# ML-Marathon-2026

Team Rocket's entry in the [Wisconsin ML Marathon 2026](https://ml-marathon.wisc.edu/schedule/), tackling the Kaggle [Snapshot WI - Oh Deer](https://www.kaggle.com/competitions/snapshot-wi-oh-deer) challenge: detecting deer age and antler status from Snapshot Wisconsin trail-camera images.

- `docs/PLAN.md`: challenge, scoring, AI Use Agreement
- `docs/CONTRIBUTING.md`: repo structure, PR workflow, experiment logging
- `docs/DATA_SETUP.md`: data download and preprocessing
- `docs/KICKOFF.md`: decisions to agree on as a team before building

## Approach

No off-the-shelf detectors (YOLO, MegaDetector). Each member hand-builds a PyTorch CNN that classifies trail-cam images for age and antler status, then tinkers with one part (e.g. preprocessing, bounding-box cropping, augmentation, architecture). Slower, but the point is to learn. **Model code is written by us, not AI.**

Shared code: `scripts/` (data pipeline) and `metrics/`. Personal models: `workspaces/<name>/` (see `workspaces/README.md`). Runs: `logs/experiments.md`.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for package management.

1. Install uv (if not already installed):

   ```bash
   # macOS / Linux / WSL
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Sync the environment (creates a `.venv` and installs dependencies from `uv.lock`):

   ```bash
   uv sync
   ```

3. Set up Kaggle credentials (once):
   - Accept the competition rules on the [competition page](https://www.kaggle.com/competitions/snapshot-wi-oh-deer) (downloads fail otherwise).
   - Kaggle -> Settings -> API -> "Create New Token", and save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`). Never commit this file.

4. Download the data into `data/raw/` (safe to rerun; skips if already set up). All commands in this repo work as written in bash, zsh, and PowerShell:

   ```bash
   uv run scripts/fetch_data.py
   ```

5. Create the shared train/val/test split (`data/processed/split.csv`). The printed fingerprint should match your teammates':

   ```bash
   uv run scripts/make_split.py
   ```

6. Run scripts within the environment using `uv run`, from the repo root:

   ```bash
   uv run python -m workspaces.<name>.train
   ```

To add a new dependency (shared config, needs a teammate's review):

```bash
uv add <package-name>
```

## AI use disclaimer

Claude Code was used to plan and generate repo scaffolding, data-download/preprocessing scripts, and documentation under human direction and review. Model architectures and training code are written by team members without AI generation.
