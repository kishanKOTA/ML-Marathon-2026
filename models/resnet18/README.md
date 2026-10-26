# ResNet-18

**Owner**: Amanda

Shared step-by-step guide, label rule, and resources: `models/README.md`.

**Architecture**: ResNet-18 ([He et al., 2016](https://arxiv.org/abs/1512.03385)), hand-built and trained from scratch (no pretrained weights).

**What I'm exploring**: Bounding-box cropping (training on each deer's crop instead of the full image; see steps 3-4 in `docs/DATA_SETUP.md`).

**How to run**: from the repo root, `uv run python -m models.resnet18.train` (then `.evaluate` once done tuning).

## What I've tried / what I learned

_(Short notes as you go: what you changed, what happened, what surprised you. Handy for the writeup.)_
