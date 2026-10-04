"""Your CNN: the network that looks at an image and predicts a class.

Input:  a batch of images from dataset.py
Output: one score per class for each image (e.g. ADULT vs YOUNG)

What to include: the layers of your network and how an image passes through them.

Worth thinking about:
- How deep does it need to be for ~5k images?
"""

# Shallow 5, or 10-20 layers recommended
# ReLu layers
# sotmax at the end
# Dont streess about loss function or optimizer

# After the first run, we can try importing another model and tweaking the imprted 

from pathlib import Path
import tomllib
import torch.nn as nn # Module in nn is the fundamental base class for all neural network building blocks in PyTorch
import torch.nn.functional as F  # Functional module that contains helper operations*

# Defines paths and load config
root = Path(__file__).resolve().parents[2]
config_path = root / Path("workspaces/lucas/configs/default.toml")
with open(config_path, "rb") as f:
    data = tomllib.load(f)

class DeerCNN(nn.Module):   # Inherits nn.Module class
    def __init__(self):
        '''Initialization function for model. Include here anything that has weights an learns (convolution/linear layers) 
        on't include non-learnabel layers that just apply operations (eg. ReLU, pooling) '''

        super().__init__()  # Calls the initialization fuction for nn.Module which sets up internal settings

        '''Layers breakdown:
            nn.Conv2d(in_channels, out_channels, kernel_size)

            in_channels - Input channels; 3 for RGB, 1 for grayscale. Subsequent layers need to match output from previous convolution
            out_channels - Number of output channels. For each one, a filter of k x k is created initialized with small random numbers. With training, these begin to identify patterns (eg. edges, curves, corners, etc)
            kernel_size - Size of sliding window. A bigger kernel sees a larger area at once, while smaller sizes capture more fine detail at the cost of compute
            padding - optional, but preserves size between layers. Calculate as (kernel_size - 1) / 2


            nn.Linear(in_features, out_features) - Fully connected layer. Where convs only look at patches, a linear layer looks at everything at once (after it has been summarized by convolutions and pooling layers) 

            in_features - Number of input features. Needs to match previous layer
            out_features - Number of output classes / features
        '''

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) # If converting to grayscale, need to adjust the first in channel value
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.linear = nn.Linear(64, 2)

    def forward(self, x):
        '''Forward convolution. Defines the order of execution. '''
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # maxpool of size 2; halves the input size
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # ReLU maps the output of the convolution sum into a 0-x range
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.adaptive_avg_pool2d(x, 1) # Averages each of the 64 feature maps down to one value, no matter the image size
        x = x.flatten(1) # built in tensor opeartion to reduce dimension by 1
        return self.linear(x)