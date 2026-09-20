# Project: ML Marathon 2026 - Snapshot WI, Oh Deer

Team Rocket's entry in the [Wisconsin ML Marathon](https://ml-marathon.wisc.edu/schedule/) (9/9-12/9), competing in the Kaggle [Snapshot WI - Oh Deer](https://www.kaggle.com/competitions/snapshot-wi-oh-deer) challenge. 4-person team, each with only ~3-5 hrs/week, all beginners in ML/CV.

**Task**: detect deer **age** (young = bright fur spots; adult = no spots, unless discoloration) and **antler status** (antlered if >=3 inches, longer than ear) from Snapshot Wisconsin trail-cam images. 4880 images with bounding-box ground truth (`Snapshot_WI-Oh_Deer_data-v2.csv`) and county-level locations (`Snapshot_WI-Oh_Deer_locs-v2.csv`). Images are color by day, black-and-white at night; motion blur and variable lighting are common. Scoring: F1 (100 pts), additional non-ground-truth predictions like time-of-day/animal state/land cover (15 pts), inference speed (15 pts), compute efficiency (10 pts), ease of use (10 pts).

For full challenge details and pre-modeling notes, read `docs/MLM26 Team Rocket Plan.md`. For the current week-by-week plan, read `docs/ROADMAP.md`. For git/PR workflow and repo conventions, read `CONTRIBUTING.md` — **follow it exactly**, especially:
- Repo structure (`data/`, `src/<domain>/` vs `src/<initials>/`, `notebooks/<initials>/`, `results/`).
- Never commit competition data or large artifacts (`data/`, `results/artifacts/` are gitignored).
- Never push directly to `main` (branch protection is on); one branch per topic, PR under ~400 changed lines.
- Log experiment runs in `results/log.md`, not just in a branch/notebook.
- Get confirmation before touching shared modules (`src/data/`, `src/models/`, `src/eval/`, configs, `pyproject.toml`), deleting/renaming files, adding dependencies, or downloading/committing data — per CONTRIBUTING.md's "what an agent may touch without asking" section.

## Modeling approach (current plan)

Don't jump ahead of the priority order in `docs/ROADMAP.md` (transfer learning baseline -> fine-tuning -> stretch goals) without checking with the team first — the point is a working, honestly-reported baseline by 12/2, not the fanciest model.

## Team AI-use agreement

Source of truth: the "AI Use Agreement" section in `docs/MLM26 Team Rocket Plan.md` — fine for planning/learning/explanations/troubleshooting, not fine for generating the entire training pipeline or committing/merging PRs on the team's behalf (the team wants to learn by building the modeling logic themselves).

In practice: prefer explaining approaches and writing small, reviewable pieces the team can understand and extend, over generating a complete pipeline in one shot. When asked to implement something non-trivial (a training loop, a new model architecture, a metric), check whether the person wants you to build it directly or would rather work through it with your guidance.

## Orienting a future session

If you're picking this up fresh: read `docs/MLM26 Team Rocket Plan.md` (challenge + status), `docs/ROADMAP.md` (what week we're in, what's next), and `CONTRIBUTING.md` (how to work in this repo) before doing anything else. Check `results/log.md` for what's already been tried.
