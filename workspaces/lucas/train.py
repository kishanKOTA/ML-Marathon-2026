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

from pathlib import Path
import tomllib

from workspaces.lucas import dataset, model

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score
import pandas as pd
from datetime import datetime

# Run this uv run python -m workspaces.lucas.train 2>&1 | tee logs/runs/lucas/console.log

def main():
    # Defines paths and load config
    root = Path(__file__).resolve().parents[2]
    config_path = root / Path("workspaces/lucas/configs/default.toml")
    config_name = config_path.stem

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    img_dir = data['paths']['images']
    split_path = data['paths']['split']

    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    run_log = data['paths']['runs']

    run_log_folder = Path(run_log) / Path(f"{config_name}_{timestamp}")
    run_log_folder.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')   # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device('mps')    # Apple Sillicon GPU (M1/M2/M3/M4)
    else:
        device = torch.device('cpu')    # Fallback to CPU

    print(f"Using device: {device}")

    cnn = model.DeerCNN()
    cnn.to(device) # for a model no need to reassign after to(device), weights change in place
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=data['train']['learning_rate'])

    train_set = dataset.DeerDataset(img_dir=img_dir, attribute=data['train']['attribute'], split_path=split_path, split_name='train')
    train_loader = DataLoader(dataset=train_set, batch_size=data['train']['batch_size'],shuffle=True)

    validate_set = dataset.DeerDataset(img_dir=img_dir, attribute=data['train']['attribute'], split_path=split_path, split_name='val')
    validate_loader = DataLoader(dataset=validate_set, batch_size=data['train']['batch_size'],shuffle=False)


    n_epochs = data['train']['epochs']
    metrics_list = [] 
    best_f1 = -1   # -1 instead of 0 In case never predicts young? REVIEW 

    for i in range(n_epochs):
        train_avg_loss = train_epoch(cnn, train_loader, loss_function, optimizer, device)
        val_avg_loss, f1 = validate_epoch(cnn, validate_loader, loss_function, device)

        print(f"\nEPOCH {i} - Average train loss: {train_avg_loss} | Average val loss: {val_avg_loss} | F1 score: {f1}\n")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(cnn.state_dict(), run_log_folder / Path(f"best_model_{config_name}.pt"))   # stores a dictionary from parameter names to the learned numbers.

        metrics = (i, train_avg_loss, val_avg_loss, f1)
        metrics_list.append(metrics)

        metrics_df = pd.DataFrame(metrics_list, columns=["epoch", "avg_train_loss", "avg_val_loss", "f1"])  # Kept inside the loop so a crash preserves outputs
        metrics_df.to_csv(run_log_folder / Path(f"metrics_{config_name}.csv"), index=False)



def train_epoch(model, dataloader, loss_function, optimizer, device) -> float:
    '''Trains single epoch, doing a forward pass and backpropagation'''

    model.train()

    total_loss = 0
    total_batches = 0

    for image, label in dataloader:

        # Get the input data (batch of 32*)
        image = image.to(device)
        label = label.to(device)

        # Clear parameter gradients
        optimizer.zero_grad()

        outputs = model(image)                 # forward pass
        loss = loss_function(outputs, label)   # compute the loss
        loss.backward()                         # backpropagate
        optimizer.step()                        # update the weights

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss/total_batches

    return avg_loss


def validate_epoch(model, dataloader, loss_function, device) -> tuple[float, float]:
    '''Returns average batch los and validation F1 score'''

    model.eval()
    total_loss = 0
    total_batches = 0

    with torch.no_grad():

        prediction_list = []
        labels_list = []

        for image, label in dataloader:

            labels_list.append(label) # append before moving to GPU

            image = image.to(device)
            label = label.to(device)

            outputs = model(image)
            loss = loss_function(outputs, label)

            total_loss += loss.item()
            total_batches += 1

            predictions = outputs.argmax(dim=1) # Turns outputs into predictions (higher scoring class)
            predictions = predictions.to('cpu')

            prediction_list.append(predictions)


        avg_loss = total_loss/total_batches

        labels_tensor = torch.cat(labels_list)  # joins the batches into one sequence.  puts the batches end to end into a flattened tensor
        prediction_tensor = torch.cat(prediction_list) # standard flatten fails because its a list of tensors

        f1 = f1_score(y_true=labels_tensor, y_pred=prediction_tensor, zero_division=0) #  Compares two equal-length sequences and counts matches itself.

        print(torch.bincount(prediction_tensor, minlength=2)) # Minlength to catch if there are not  2 classes

        return avg_loss, float(f1)  # f1 is returned as numpy float, conversion is cleaner to match type hints

            
if __name__ == '__main__':
    main()