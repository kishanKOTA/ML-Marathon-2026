# Lucas's workspace

Shared step-by-step guide, label rule, and resources: `workspaces/README.md`.

**What I'm exploring**: Image preprocessing (e.g. contrast normalization for night IR images, deblurring).

**How to run**: from the repo root, `uv run python -m workspaces.lucas.train` (then `.evaluate` once done tuning).

## What I've tried / what I learned

_(Short notes as you go: what you changed, what happened, what surprised you. Handy for the writeup.)_


### October 5, 2026

Over the weekend got done setting up `dataset.py` and `train.py`. The initial model is a very simple CNN with the following design:

- Three convolution layers (16-32-64), followed by a ReLU activation function, and then MaxPool
- Adaptive 2D pooling to convert the last 64 channels into a single number, so the dimensions get flattened
- A fully connected layer at the end,

 Let the model running overnight on Sunday and ran a first pass at training  


### October 12, 2026

One week post-presentation. The original model performed poorly with a best F1 of 0.48 at epoch 82. I had the chance to talk with Ryan about potential next steps for the rest of the marathon. I liked his suggestions on taking a proven model architecture and using it for our purposes. I don't want to take their weights though (which would most almost be a guaranteed improvement), since I want a clean baseline to experiment image enhancement and data augmentation techniques. Again, not going for the best metrics, but maximizing learning as much as possible.

To Do's
- Verify how the weights are being initialized
- Choose and mimick an existing proven model architecture (I want to write them layer by layer to see whats going on). For comparability, might be worth choosing a model that has ImageNet or Coco benchmarks. Consider removing/freezing layers
- Find a way to separate day/night images and deal with headless deer. Images at night are very hard to detect spots in deer. However, we can still see antlers there, which would indicate not young. One idea for the headless deers could be to look at bbox extremes, and weigh images higher when the bounding box is not at any of the edges where cutoffs might happen. We can't detect antler/antlerless with headless deer, but we can detect age.

### October 18, 2026

Updated Plan:

1. Add day/night and deer visibility tag to images by looking at RGB distributions and calculating bbox extent (helpful for diagnosis since those two introduce a lot of variability). Create some long-due EDA histograms showing data distribution by day/night, month of images, and visibility.
2. Adapt a proven model architecture (AlexNet, UNet) to train a 3-class baseline (antlered adult / antlerless adult / young) with class weights for imbalance. Log val F1 overall and split by day/night, visibility, and month.
3. (Setup pre-processing step to adjust brightness channel on images. Test and see if that improves model before making it permanent.) - do this at the end, a model can prob pick up 
4. Test different data augmentation techniques one at a time (one row in `logs/experiments.md`): horizontal flip, brightness/contrast jitter, mild random crop/zoom (ensuring antlers aren't cut), and small blur/noise (most likely to hurt spots). We only appply augmentation to train split, and we can give less to the train split since we can augment them and get more samples. 
5. Combine the best preprocessing + augmentations into the final run, note inference time

Extra things to try (if I get to it):
- Downweight bboxes touching the image edge
- Grayscale-everything test (night is NIR, so not identical to grayscale day, but tests whether color matters)
- Month as an input (deer mating season → antlers form + young deer a few months later)

### October 25, 2026

- Amanda and I met on Sunday, and decided ResNet18 will be the model we will adapt for our deer dataset.

Additional potential issue:
- Downsampling to 224 might make deer spots unresolvable. Cropping to the bbox first, then downsampling is probably the best approach to saving details. Though that would mean that at inference, we wouLd need megadetectors or yolo to find the deer for us, then crop, then classify. 

We could try a ResNet run first, since it doesn't take long in theory, then add this as another step.


