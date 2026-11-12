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

import random
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

import time
from tqdm import tqdm

from dataset import DeerDataset, make_transform, make_train_transform, config, ROOT
from model import ResNet18, get_pretrained_resnet18

cfg = config["train"]
random.seed(cfg["seed"])
np.random.seed(cfg["seed"])
torch.manual_seed(cfg["seed"])

if torch.backends.mps.is_available():     # Apple Silicon GPU
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("using device:", device)

# -- DATA ---

# augmentation only on training images; val is always the plain resize
# transform turns the raw photo into model input
if cfg["augment"]:
    train_tf = make_train_transform(cfg["image_size"])
else:
    train_tf = make_transform(cfg["image_size"])
val_tf = make_transform(cfg["image_size"])

train_ds = DeerDataset(cfg["attribute"], "train", transform=train_tf, crop=cfg["crop"])
val_ds = DeerDataset(cfg["attribute"], "val", transform=val_tf, crop=cfg["crop"])

print("train class balance:\n", train_ds.df[train_ds.label_col].value_counts(normalize=True))
print("val class balance:\n", val_ds.df[val_ds.label_col].value_counts(normalize=True))


train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

# -- MODEL ---
frozen = cfg["pretrained"] and cfg["freeze_backbone"]
if cfg["pretrained"]:
    model = get_pretrained_resnet18(freeze_backbone=cfg["freeze_backbone"])
else:
    model = ResNet18()
model = model.to(device)

# sanity check: 1,026 if frozen, ~11.2M if fine-tuning or from scratch
print("trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# -- LOSS ---
if cfg["class_weights"]:
    # rarer class gets bigger weight, so mistakes on YOUNG costs more
    counts = train_ds.df[train_ds.label_col].value_counts().sort_index().values
    w = torch.tensor(len(train_ds) / (2 * counts), dtype=torch.float32).to(device)
    print("class weights:", w.tolist())
    loss_fn = nn.CrossEntropyLoss(weight=w)
else:
    loss_fn = nn.CrossEntropyLoss()     # measures how wrong the predictions are

# -- OPTIMIZER (only the nums allowed to change) ---
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"]) 
# model.parameters() tells Adam which numbers it’s allowed to change
# slowly lowers the learning rate to ~0 by the last epoch
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

def train_one_epoch():
    model.train()
    if frozen:
        # keep frozen BatchNorm layers from updating their running stats
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    total_loss = 0.0
    for images, labels in tqdm(train_loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()           # clear old gradients
        outputs = model(images)         # forward pass
        loss = loss_fn(outputs, labels) # check how wrong it was
        loss.backward()                 # compute gradients
        optimizer.step()                # update weights

        total_loss += loss.item() * images.size(0)
    return total_loss / len(train_ds)

def evaluate():
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():       # no gradients needed, saves memory
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += loss_fn(outputs, labels).item() * images.size(0)
            preds = outputs.argmax(dim=1)   # pick the higher scoring class
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    f1 = f1_score(all_labels, all_preds, zero_division=0)  # 0 if never predicts young
    precision = precision_score(all_labels, all_preds, zero_division=0) # of predicted young, how many were actually young
    recall = recall_score(all_labels, all_preds, zero_division=0) # of real YOUNG, how many were found
    return total_loss / len(val_ds), f1, precision, recall

runs_dir = ROOT / config["paths"]["runs"]
runs_dir.mkdir(parents=True, exist_ok=True) # create the folder if it's missing

# mode = "pretrained" if cfg["pretrained"] else "scratch"
tag = f"{cfg['attribute']}_{cfg['run_name']}"
print("saving to", runs_dir / f"{tag}_log.csv")

best_f1 = -1.0
best_epoch = 0
start = time.time()
with open(runs_dir / f"{tag}_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_f1", "val_precision", "val_recall", "epoch_seconds"])

    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch()
        val_loss, val_f1, val_p, val_r = evaluate()
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()  # update learning rate

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start
        pct = 100 * epoch / cfg["epochs"]

        print(f"epoch {epoch}/{cfg['epochs']} ({pct:.0f}%) | "
              f"train {train_loss:.4f}  val {val_loss:.4f}  F1 {val_f1:.3f} | "
              f"F1 {val_f1:.3f}  P {val_p:.3f}  R {val_r:.3f} | "
              f"{epoch_time:.0f}s this epoch, {total_time/60:.1f} min total")
        writer.writerow([epoch, train_loss, val_loss, val_f1, val_p, val_r, round(epoch_time, 1)])

        # print(f"epoch {epoch}: train {train_loss:.4f}  val {val_loss:.4f}  F1 {val_f1:.3f}")
        # writer.writerow([epoch, train_loss, val_loss, val_f1])
        f.flush()

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), runs_dir / f"{tag}_best.pt")

print(f"best val F1: {best_f1:.3f} (epoch {best_epoch})")