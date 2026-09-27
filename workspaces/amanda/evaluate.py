"""Gives your finished model its final grade.

Input:  your config, a saved model from train.py, and the 'test' dataset
Output: test F1 and average prediction time per image

Purpose: score the model once on the test images, which none of your decisions
(architecture, settings, when to stop) were based on. This is the honest number
to report. Inference speed is also scored, so time the predictions here.

What to include: loading the saved model and test data, predicting, and F1 via metrics/.

Worth thinking about:
- Why wait until you're done tuning to run this?
"""
