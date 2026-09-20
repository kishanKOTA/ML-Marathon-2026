# Roadmap

Guiding principle: at 3-5 hrs/week/person, the goal is a working, honestly-reported baseline by 12/2 — not a state-of-the-art model. It's fine to under-deliver on stretch goals. Dates below are soft internal checkpoints, not hard commitments, except where marked (external marathon deadlines).

## Week of 9/22 (-> 9/28 slide, 9/30 lightning talk)
Write down what's already been done informally: challenge summary, EDA notes, links to papers/repos found so far. This becomes the basis of the **First Steps Slide (9/28, external deadline)**. Goal: have something concrete to present 9/30 — not a polished pipeline.

## Week of 9/29 - 10/5
Finalize EDA (image/label formats, class balance, day/night split, motion blur prevalence). Stand up the repo skeleton (`data/`, `src/`, `notebooks/<initials>/`, `scripts/`, `tests/`) and the Kaggle data-fetch convention (see `CONTRIBUTING.md`). Pick the baseline metric (F1, per challenge rules) and a trivial baseline (e.g. off-the-shelf detector with no fine-tuning) to have a number to beat.

## Week of 10/6-10/12 (AWS SageMaker/Bedrock workshop 10/7)
Evaluate whether AWS compute is worth using given the team's scale. Get a pretrained detector (e.g. MegaDetector) running on a handful of images as a smoke test.

## Weeks of 10/13-11/9 (sprints 10/21, 10/28, 11/3)
Core modeling stretch. Priority order — take as far as time allows:

1. Pretrained detector + simple classifier heads for age/antler, evaluated against the baseline metric.
2. If that's solid with time left: fine-tune the detector end-to-end.
3. Stretch, only if ahead of schedule: augmentation/ablation experiments; auxiliary unsupervised predictions (time of day/season, animal pose) for the "additional predictions" scoring category.

Log every run per the experiment-tracking convention in `CONTRIBUTING.md` so results aren't lost across branches.

## Week of 11/10 (draft writeup, external deadline) -> 11/17 (peer review)
Freeze whatever's working. Write the draft writeup covering approach + current results honestly — it doesn't need to reflect the final model.

## 11/18 sprint -> 12/2 (final deadline)
Polish based on peer review feedback. Finalize inference speed and compute-efficiency notes (both are scored). Submit the final writeup.

## 12/9
Closing celebration — no work items.
