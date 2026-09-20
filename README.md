# ML-Marathon-2026

Team Rocket's entry in the [Wisconsin ML Marathon 2026](https://ml-marathon.wisc.edu/schedule/), tackling the Kaggle [Snapshot WI - Oh Deer](https://www.kaggle.com/competitions/snapshot-wi-oh-deer) challenge: detecting deer age and antler status from Snapshot Wisconsin trail-camera images.

- **What we're building, evaluation criteria, data description**: see `docs/MLM26 Team Rocket Plan.md`.
- **Week-by-week schedule**: see `docs/ROADMAP.md`.
- **Git/PR workflow, repo structure, data and experiment conventions**: see `CONTRIBUTING.md`.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for package management.

1. Install uv (if not already installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Sync the environment (creates a `.venv` and installs dependencies from `uv.lock`):

   ```bash
   uv sync
   ```

3. Run scripts within the environment using `uv run`:

   ```bash
   uv run python your_script.py
   ```

To add a new dependency:

```bash
uv add <package-name>
```