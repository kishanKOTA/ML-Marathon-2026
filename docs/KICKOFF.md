# Kickoff checklist

Things to agree on as a team before everyone starts building, so we don't find out in November that our numbers can't be compared. Check off each item once it's decided, and write the decision next to it.

## Scoring and evaluation (settle these first)

- [ ] **Which F1 exactly?** Check the Kaggle page: F1 on the positive class, macro F1, or something else? Is age and antler scored separately or combined? `metrics/` should match it exactly.
- [ ] **Who writes `metrics/`**, and by when? Until then everyone uses `sklearn.metrics.f1_score` with the same settings.
- [ ] **Label rule**: ADULT = 0 / YOUNG = 1, ANTLERLESS = 0 / ANTLERED = 1 (in `workspaces/README.md`). Everyone OK with it?
- [ ] **Baseline to beat.** Always predicting the majority class gives F1 = 0 on the rarer class, so agree on what "trivial baseline" means (e.g. always predict class 1, or random guessing at the class ratio).
- [ ] **When do we touch test?** Suggest: nobody runs `evaluate.py` until a shared freeze date (e.g. before the 11/10 draft).
- [ ] **How is the final model picked?** Suggest: best val F1 on the official split, speed as tie-breaker. One member's model or a combination?
- [ ] **Does the Kaggle test data include bounding boxes?** If not, a crop-based model (Amanda's experiment) needs a way to find the deer at prediction time.

## Data

- [ ] Everyone has run `fetch_data.py` and `make_split.py` and sees fingerprint `673e0097`.
- [ ] Nobody edits `configs/split.toml`. Alternate splits are only for checking that an improvement is real.
- [ ] Competition data stays local: don't upload images or the CSVs to AI tools or other services, and check Kaggle's competition rules on data use.

## Experiments

- [ ] Do the `logs/experiments.md` columns work for everyone? Consider adding the config filename so each row can be reproduced.
- [ ] One model for age + one for antlers, or one model for both? Personal choice is fine, but always report F1 per attribute.
- [ ] Everyone has filled in "What I'm exploring" in their workspace README (Kishan, Pranshul still open).

## Compute

- [ ] Who has what: NVIDIA GPU, Apple Silicon (M1+), or CPU only? On Windows, `uv sync` installs CPU-only torch; NVIDIA users there need a config change.
- [ ] Is the AWS workshop (10/7) worth following up on, given our scale?
- [ ] Inference speed and compute efficiency are scored: agree to measure them the same way (same machine, images per second, CPU or GPU).

## Process

- [ ] Everyone has read `docs/CONTRIBUTING.md`: branch names, PR size, 1 review, self-merge exception.
- [ ] Branch protection on `main` is actually on (repo owner checks Settings -> Branches).
- [ ] Who reviews shared-code PRs (`scripts/`, `metrics/`, `pyproject.toml`)? Rotate, or a fixed person?
- [ ] AI agreement: everyone clear on "explain yes, write workspace code no"? Any gray areas (e.g. autocomplete in the IDE)?
- [ ] Check-in rhythm: a weekly meeting, or a short async update (what I tried, what's next, blockers)?

## Repo housekeeping

- [ ] Reconcile the `crop-images` branch (paths, script names; steps in `docs/DATA_SETUP.md`). Who owns it?
- [ ] Add dev tools (`pytest`, `ruff`) and agree on formatting settings (quote style, line length) so diffs don't fill up with reformatting.

## Writeup and extra points

- [ ] Who writes which writeup sections? Draft due 11/10, final 12/2.
- [ ] Ease of use (10 pts): agree the final model runs with one documented command.
- [ ] Additional predictions (15 pts): is anyone taking this on (e.g. day vs night, season from `month`)?
