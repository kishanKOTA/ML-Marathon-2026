# Workspaces

One folder per member for your hand-built CNN. This file is the shared guide; your own `README.md` tracks your personal path.

## A note on AI

This folder is where the learning happens, so we recommend **not** using AI to write code here. Reading docs, getting stuck, and fixing your own bugs is the point. If you're stuck, it's fine to ask AI (or a teammate) to *explain* a concept or an error message, but write the code yourself.

## Layout

```
workspaces/<name>/
  README.md     # what you're exploring + what you've tried
  configs/      # settings your code reads (paths, image size, epochs, ...)
  notebooks/
  dataset.py    # images + labels -> model input
  model.py      # the CNN
  train.py      # learn on train, check on val
  evaluate.py   # final score on test
```

## Pipeline

```
data/raw/images/ + data/processed/split.csv     (shared, same for everyone)
  -> dataset.py   one split's images + labels, with your preprocessing
  -> model.py     your CNN
  -> train.py     learn on 'train', check on 'val', save best model
  -> evaluate.py  final F1 + speed on 'test'
```

Both shared inputs are set in `configs/default.toml` (`images`, `split`). Each file's docstring lists its input and output.

Useful for others (e.g. a dataset class)? Promote it to a shared folder in its own PR.

## Running

From the repo root, so `metrics` and `workspaces.*` imports resolve:

```bash
uv run python -m workspaces.<name>.train
```

PyCharm: set the run config's working directory to the repo root and the interpreter to `.venv`. Config paths are relative to the repo root.

Save run outputs to `logs/runs/<name>/` and add a row to `logs/experiments.md`.

## Step by step

Do these in order. After each step, run something small to check it works before moving on. Most bugs are easier to find one step at a time.

### 0. Get the data and look at it
- If you haven't yet: `uv run scripts/fetch_data.py`, then `uv run scripts/make_split.py`.
- In a notebook in `notebooks/`, open `data/processed/split.csv` with pandas (`pd.read_csv`). Look at the columns, and count how many rows each split has (`value_counts()`).
- Open a few images (a day one and a night one) with PIL (`Image.open`). Check their size and `.mode` (e.g. `RGB`). Knowing your data saves hours later.

### 1. Fill in your config (`configs/default.toml`)
- Add settings you'll want to change later, e.g. `attribute = "age"`, `image_size = 128`, `batch_size = 32`, `epochs = 10`, `learning_rate = 0.001`. Group them under `[data]` / `[train]` sections if you like.
- In Python, `tomllib.load` gives you back a regular dictionary, e.g. `config['paths']['images']`. Open the file with `'rb'` (read bytes), which tomllib requires.

### 2. `dataset.py`: turn rows into (image, label) pairs
- Write a class that inherits from `torch.utils.data.Dataset`. It needs three methods:
  - `__init__`: read `split.csv`, keep only the rows for your split (e.g. `df[df['split'] == 'train']`), and store them.
  - `__len__`: how many images are in this split.
  - `__getitem__(i)`: open image *i*, apply transforms, and return it with its label as a number.
- **Team rule for labels**: ADULT = 0, YOUNG = 1 and ANTLERLESS = 0, ANTLERED = 1. F1 is measured on class 1 (the rarer one), so everyone must use the same mapping or scores aren't comparable.
- Transforms (from `torchvision.transforms.v2`): every image must become the same size (`Resize`) and a float tensor (`ToImage` + `ToDtype`). `Normalize` is optional to start. Your preprocessing ideas go here later.
- **Check**: create the train dataset and confirm `len()` is 3482. Grab one item and print the image's `.shape` (should be `[3, H, W]`) and the label. Compare each split's length and label counts with a teammate's; if they match, you loaded the same data.
- Wrap it in a `torch.utils.data.DataLoader` (`batch_size`, and `shuffle=True` for train only). One batch should have shape `[batch_size, 3, H, W]`.

### 3. `model.py`: your CNN
- Write a class that inherits from `torch.nn.Module`. Define layers in `__init__` and describe how an image flows through them in `forward`.
- A good first model is small: 2-3 blocks of `nn.Conv2d` -> `nn.ReLU` -> `nn.MaxPool2d`, then `nn.AdaptiveAvgPool2d(1)` + `nn.Flatten()`, then `nn.Linear(..., 2)` for the two classes. Adaptive pooling saves you from calculating the flattened size by hand.
- Don't add a softmax at the end. The loss in step 4 expects raw scores.
- **Check**: pass in a fake batch, `torch.randn(4, 3, H, W)`. The output shape should be `[4, 2]`.

### 4. `train.py`: teach the model
- Pick a device: `'cuda'` (NVIDIA GPU), `'mps'` (Apple Silicon), or `'cpu'`. Move the model and each batch to it with `.to(device)`.
- Loss: `nn.CrossEntropyLoss()`. Optimizer: `torch.optim.Adam(model.parameters(), lr=...)` is a forgiving default.
- Each epoch has two halves:
  - **Train**: `model.train()`. For each batch, five steps in this order: clear old gradients (`optimizer.zero_grad()`), predict, compute the loss, backpropagate (`loss.backward()`), update the weights (`optimizer.step()`).
  - **Validate**: `model.eval()` and wrap it in `with torch.no_grad():`. Predict on every val batch, take the higher score as the prediction (`argmax`), and compute F1 against the true labels. Until `metrics/` has the team's F1, use `sklearn.metrics.f1_score`.
- Print the train loss and val F1 every epoch. When val F1 beats your best so far, save the model with `torch.save(model.state_dict(), ...)` into your runs folder.
- **Sanity check first**: train on just ~20 images for many epochs. If the loss doesn't drop close to zero, something is broken. Fix that before training on everything.
- `if __name__ == '__main__':` at the bottom, so `uv run python -m workspaces.<name>.train` runs it.
- After each real run, add a row to `logs/experiments.md`.

### 5. `evaluate.py`: the final grade (run rarely)
- Rebuild your model, load the saved weights (`model.load_state_dict(torch.load(...))`), and run it on the **test** split, the same way as validation.
- Report test F1 and speed: time the predictions with `time.perf_counter()` and divide by the number of images.
- Only do this once you're done tuning. If you keep checking test and changing things, it quietly becomes a second validation set and the number stops being honest.

### 6. Experiment
- Change **one thing at a time** (a preprocessing step, the image size, one more conv block). Copy your config for each experiment so the settings are recorded, and log every run.

## Resources

- [PyTorch: Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html): the closest tutorial to what you're building (a small CNN on images).
- [PyTorch basics](https://pytorch.org/tutorials/beginner/basics/intro.html): short pages on [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), [building the model](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html), [the training loop](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html), and [saving/loading](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html).
- [torchvision transforms](https://pytorch.org/vision/stable/transforms.html): resizing, normalizing, augmentation.
- [torch.nn reference](https://pytorch.org/docs/stable/nn.html): every layer type.
- [CS231n: Convolutional Networks](https://cs231n.github.io/convolutional-networks/): intuition for what conv and pooling layers actually do.
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/): practical debugging habits (the "overfit a tiny batch first" idea).
- [pandas in 10 minutes](https://pandas.pydata.org/docs/user_guide/10min.html): reading and filtering `split.csv`.
- [tomllib](https://docs.python.org/3/library/tomllib.html) and the [TOML format](https://toml.io/en/): reading your config.
- [scikit-learn `f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html): what F1 measures and its options.
