"""Your CNN: the network that looks at an image and predicts a class.

Input:  a batch of images from dataset.py
Output: one score per class for each image (e.g. ADULT vs YOUNG)

What to include: the layers of your network and how an image passes through them.

Worth thinking about:
- How deep does it need to be for ~5k images?
"""
# Basic Block / ResNet18

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1): 
        # in_ch = # channels going in, out_ch = # channels going out 
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False) # creates convolution layer, small 3x3 filter that slides across the img looking for patterns
        self.bn1 = nn.BatchNorm2d(out_ch) # batch normalization layer
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True) # replace negative nums w/ 0

        # adds its input to its output (same shape for both)
        # if the block changes channel count / shrinks img -> input needs 1x1 conv to match
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x))) # conv -> batchnorm -> relu
        out = self.bn2(self.conv2(out))          # conv -> batchnorm (no relu yet)
        out = out + self.shortcut(x)             # skip connection
        return self.relu(out)                    # relu after the addition

class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_stage(64,64, num_blocks=2, stride=1)
        self.layer2 = self._make_stage(64,128, num_blocks=2, stride=2)
        self.layer3 = self._make_stage(128,256, num_blocks=2, stride=2)
        self.layer4 = self._make_stage(256,512, num_blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes) # final decision layer (512 nums in, 2 scores out)

    def _make_stage(self, in_ch, out_ch, num_blocks, stride):
        blocks = [BasicBlock(in_ch, out_ch, stride)] # first block may change size
        for _ in range(num_blocks - 1):
            blocks.append(BasicBlock(out_ch, out_ch, 1)) # the rest keep the same size
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)               # shape (batch, 512, 1, 1)
        x = torch.flatten(x, 1)        # shape (batch, 512)
        return self.fc(x)              # shape (batch, 2)

def get_pretrained_resnet18(num_classes=2):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


if __name__ == "__main__":
    model = ResNet18(num_classes=2)
    for size in [224, 384]:
        out = model(torch.randn(2, 3, size, size))
        print(size, "→", out.shape)          # expect torch.Size([2, 2]) both times

    n_params = sum(p.numel() for p in model.parameters())
    print("parameters:", n_params)
    ref = resnet18(num_classes=2)
    print("torchvision parameters:", sum(p.numel() for p in ref.parameters()))