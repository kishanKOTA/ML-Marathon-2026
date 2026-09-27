# Project: ML Marathon 2026 - Snapshot WI, Oh Deer

Team Rocket's entry in the [Wisconsin ML Marathon](https://ml-marathon.wisc.edu/schedule/) (9/9-12/9), competing in the Kaggle [Snapshot WI - Oh Deer](https://www.kaggle.com/competitions/snapshot-wi-oh-deer) challenge. 4-person team, each with only ~3-5 hrs/week, all beginners in ML/CV.

**Task**: detect deer **age** (young = bright fur spots; adult = no spots, unless discoloration) and **antler status** (antlered if >=3 inches, longer than ear) from Snapshot Wisconsin trail-cam images. 4880 images with bounding-box ground truth (`Snapshot_WI-Oh_Deer_data-v2.csv`) and county-level locations (`Snapshot_WI-Oh_Deer_locs-v2.csv`). Images are color by day, black-and-white at night; motion blur and variable lighting are common. Scoring: F1 (100 pts), additional non-ground-truth predictions like time-of-day/animal state/land cover (15 pts), inference speed (15 pts), compute efficiency (10 pts), ease of use (10 pts).

Read before doing anything: `docs/PLAN.md` (challenge + AI Use Agreement), `docs/CONTRIBUTING.md` (**follow exactly**: repo structure, PR size, what agents may touch without asking), and `logs/experiments.md` (what's been tried).

## Modeling approach

Team decision (late Sept 2026): **no off-the-shelf detectors** (YOLO, MegaDetector). Each member hand-builds a PyTorch CNN in `workspaces/<name>/` that classifies trail-cam images for age and antler status (bbox cropping is one member's experiment, not the default input), then tinkers on one axis (e.g. Lucas: image preprocessing). The goal is a working, honestly-reported model by 12/2, not the fanciest one; check with the team before changing direction.

**Do not write model architectures, training loops, or loss/optimizer code** in `workspaces/*`, even if asked casually. Explain, review, and debug instead. Shared plumbing (`scripts/`, `metrics/`) is fine to implement, in small reviewable pieces; for anything non-trivial, ask whether the person wants it built or wants to work through it with guidance.
