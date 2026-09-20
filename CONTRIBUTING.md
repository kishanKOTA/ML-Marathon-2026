# Contributing

We're 4 people, each running a coding agent, on one repo for [Snapshot WI (Oh Deer)](https://www.kaggle.com/competitions/snapshot-wi-oh-deer). Agents generate branches, commits, and diffs faster than we can review them — these rules exist to keep `main` clean and every PR reviewable in one sitting.

## Branch naming

`<initials>/<short-topic>`, e.g. `lp/megadetector-preprocessing`, `jk/eda-species-balance`.

- One branch = one topic. Don't reuse a branch for unrelated follow-up work.
- Delete your branch after merge.

## PR size

- One PR = one reviewable idea (a preprocessing step, a model variant, one notebook's worth of EDA). If you can't summarize the diff in 2-3 sentences, split it.
- Target **under ~400 changed lines**, excluding generated/data files. Large model checkpoints, dataset dumps, or notebook output blobs don't count against this but shouldn't be committed at all (see repo structure).
- If an agent produced a huge diff, review and trim before opening the PR — squash exploratory commits, drop dead code, don't ship what you wouldn't have written yourself.

## Review

- Every PR needs **1 approval from someone other than the author**, before merge.
- Author is responsible for having actually read their own agent's diff first — "approve" means "a human understood this," not "CI passed."
- Anything touching shared config (`pyproject.toml`, `uv.lock`, CI, this file) needs a second look from whoever last touched it, if that's not you.
- **Low-availability exception**: we're all at 3-5 hrs/week, so a reviewer may not show up in time. If a PR scoped to your own `src/<initials>/` or `notebooks/<initials>/` gets no review after ~3-4 days, you may self-merge — just say so in the PR description ("self-merged, no reviewer in 4 days"). PRs touching shared modules (`src/data/`, `src/models/`, `src/eval/`, configs) always wait for a real review, no exception.

## What an agent may touch without asking

Agents may freely create/edit files **inside the branch owner's own working area** (their scratch notebooks, their `src/<initials>/` folder, their PR's stated scope) without stopping to ask.

An agent must pause and check with a human before it:
- Edits a file outside the current PR's stated scope, especially shared modules (`src/common/`, `src/data/`, configs, `pyproject.toml`).
- Deletes or renames existing files.
- Adds a new dependency.
- Commits or pushes to `main`, force-pushes, or rewrites shared branch history.
- Downloads, moves, or commits competition data.

## Avoiding two agents editing the same file

- Before starting agent work, check open PRs/branches (`git fetch && git branch -r`) touching the same path. If someone else has a live PR on `src/data/loader.py`, don't also start editing it — coordinate first.
- Keep shared modules (`src/data/`, `src/models/`, `src/eval/`) small and single-purpose so two people rarely need the same file at once.
- Rebase (don't merge) onto `main` before opening/updating a PR, so conflicts surface early and small.
- If a conflict does happen, the person who pushed second resolves it.

## Branch protection on `main`

`main` has GitHub branch protection enabled: no direct pushes, PR + 1 approval required to merge. This isn't just a convention in this file — it's enforced by GitHub itself (Settings -> Branches on the repo). If you find you *can* push directly to `main`, that's a misconfiguration — flag it, don't use it.

## Getting the data

Competition data is never committed (see `.gitignore` / Repo structure below). Instead:

- Data is fetched with the Kaggle API into the gitignored `data/` folder via `scripts/fetch_data.py` (uses your own `~/.kaggle/kaggle.json` credentials — never commit that file either).
- If `scripts/fetch_data.py` doesn't exist yet, writing it is a good first small PR — it should just wrap `kaggle competitions download -c snapshot-wi-oh-deer` and unzip into `data/raw/`.
- If you preprocess data into a new form (crops, resized images, split manifests), write it to `data/processed/` (also gitignored) and document how to regenerate it in the relevant script's docstring/README, rather than committing the output.

## Tracking experiments

Model runs, metrics, and checkpoints shouldn't live only on someone's branch or in their notebook — they get lost. Log every run in `results/log.md` (one row per run): date, who ran it, model/config summary, metric(s), and a one-line takeaway. Large artifacts (checkpoints, prediction dumps) stay gitignored in `results/artifacts/`; only the log is committed.

## Commit messages

- Imperative mood, present tense: `Add MegaDetector filtering step`, not `Added` / `Adds`.
- One logical change per commit; no `wip`, `fix typo`, `asdf` commits in the final PR — squash those before merge.
- Body (optional) explains *why*, not what the diff already shows.
- If a commit was substantially agent-generated, that's covered once by this repo's blanket AI-use disclaimer (see below) — no need to flag it per commit.

## Repo structure

```
configs/             # model/training config files (yaml/json), committed
data/                # gitignored — never commit raw or processed data
  raw/               # fetched via scripts/fetch_data.py
  processed/         # regeneratable preprocessing outputs
notebooks/<initials>/  # personal exploratory notebooks
results/
  log.md             # committed — one row per experiment run
  artifacts/         # gitignored — checkpoints, prediction dumps
src/
  data/              # loading, preprocessing, dataset classes (shared)
  models/            # model definitions / wrappers (shared)
  eval/              # metrics, evaluation scripts (shared)
  <initials>/         # personal, not-yet-shared scratch code
scripts/             # one-off CLI entry points (fetch_data, training, inference)
tests/               # mirrors src/ layout
```

Rule of thumb: if it's reusable by others, it goes in `src/<domain>/`; if it's exploratory or yours-only, it goes in `notebooks/<initials>/` or `src/<initials>/`. Promote code out of your personal folder into a shared one in its own PR once it's stable.

## AI use disclaimer

This repository's code, documentation, and configuration may be generated or assisted by coding agents (e.g. Claude Code) under human direction and review. Each contributor is responsible for reviewing and understanding code from their own agent before opening a PR.
