# Contributing

4 people, each with a coding agent, on one repo. These rules keep `main` clean and every PR reviewable in one sitting.

## Repo structure

```
configs/split.toml   # official train/val/test split settings (don't edit)
data/                # gitignored
  raw/               # from scripts/fetch_data.py
  processed/         # regeneratable outputs (crops, splits)
docs/                # plan, data setup, this file
logs/
  experiments.md     # committed, one row per run
  runs/<name>/       # gitignored: checkpoints, curves, predictions
metrics/             # shared evaluation code (F1)
scripts/             # shared data pipeline (fetch, split, crop)
tests/               # tests for metrics/ and scripts/
workspaces/<name>/   # personal: your CNN code, configs/, notebooks/
```

Reusable by others -> `metrics/` or `scripts/`. Yours only -> `workspaces/<name>/`. Promote personal code to a shared folder in its own PR once it's stable.

## Branches and PRs

- Branch name: `<name>/<short-topic>`, e.g. `lucas/crop-preprocessing`. One branch per topic; delete after merge.
- One PR = one idea you can summarize in 2-3 sentences, **under ~400 changed lines** (excluding lockfiles).
- Review and trim your agent's diff before opening a PR. Squash `wip`/`fix typo` commits.
- Rebase (don't merge) onto `main` before opening/updating a PR. Whoever pushes second resolves conflicts.
- Before starting, check open branches (`git fetch && git branch -r`) for someone already editing the same shared file.

## Review

- Every PR needs **1 approval from someone other than the author**. "Approve" means a human understood it.
- `main` has branch protection (no direct pushes). If you *can* push to `main`, flag it.
- **Self-merge exception**: a PR touching only your `workspaces/<name>/` with no review after ~3-4 days may be self-merged; say so in the PR description. Shared code (`metrics/`, `scripts/`, `pyproject.toml`, `uv.lock`, docs) always waits for a real review.

## Agents

Agents may edit freely inside the branch owner's `workspaces/<name>/` and the PR's stated scope, **except model code**. They must ask a human before:
- Editing shared code or config outside the PR's scope.
- Deleting or renaming files, or adding a dependency.
- Committing/pushing to `main`, force-pushing, or rewriting shared history.
- Downloading, moving, or committing competition data.

**No AI-generated model code.** Each member writes their own architecture, training loop, and loss/optimizer setup. Agents may explain, review, and debug code in `workspaces/*`, and may help with shared plumbing (data pipeline, metrics).

## Data

Never commit data. Run `uv run scripts/fetch_data.py` then `uv run scripts/make_split.py` (credential setup in `README.md`, layout in `docs/DATA_SETUP.md`). Write derived data to `data/processed/` and document how to regenerate it in the script's docstring.

## Experiments

Log every run as a row in `logs/experiments.md`. Save its outputs to `logs/runs/<name>/`. A run that's only in your branch or notebook gets lost.

## Commit messages

Imperative mood (`Add night-image contrast normalization`), one logical change per commit, body explains *why*.

## AI use disclaimer

Code, docs, and config in this repo may be generated or assisted by coding agents (e.g. Claude Code) under human direction and review, except model code, which members write themselves. Each contributor is responsible for understanding their agent's output before opening a PR.
