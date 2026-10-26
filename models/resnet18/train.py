"""Teaches your model and checks its progress.

Input:  your config, the 'train' and 'val' datasets, and your model
Output: the best saved model + training history in your runs folder, and its val F1

Purpose: repeatedly show the model the training images, measure its mistakes,
and update it. After each pass, score it on the validation images (which it
never learns from) to see whether it's improving or just memorizing.

What to include: loading your config, datasets, and model; the training loop;
the validation check (F1 via metrics/); saving the best model. After each run,
add a row to logs/experiments.md.

Worth thinking about:
- How imbalanced are the classes, and does it matter?
- One model for age and one for antlers, or one model for both?
"""
