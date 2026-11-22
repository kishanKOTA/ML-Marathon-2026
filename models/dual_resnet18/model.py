"""
DualResNet18: one shared model, two inputs (image, month), two outputs (age, antler).

THIS FILE IS FOR YOU TO IMPLEMENT. Everything else in models/dual_resnet18/
codes against the interface below, so keep the names, arguments and return types.
Run tests/test_dual_resnet18.py when you are done; it should pass.

Before starting, look at the existing ResNet18 code in this repo. You can reuse
how it loads the backbone.
"""
import torch
import torch.nn as nn
import torchvision


class DualResNet18(nn.Module):
    def __init__(self, n_age=2, n_antler=2, month_dim=2, month_hidden=16, pretrained=True):
        super().__init__()

        # ---------------------------------------------------------------
        # TODO 1: IMAGE BRANCH (the backbone)
        # ---------------------------------------------------------------
        # Goal: turn a (B, 3, 224, 224) image into a (B, 512) feature vector.
        #
        # - Load torchvision.models.resnet18, with ImageNet weights if
        #   `pretrained` is True (weights="IMAGENET1K_V1"), or reuse the
        #   repo's existing loader.
        # - A standard ResNet18 ends in `fc`, a Linear(512, 1000) layer that
        #   classifies ImageNet. You do NOT want that layer. You want the
        #   512 features that go INTO it.
        # - Two common ways to remove it:
        #     a) nn.Sequential(*list(resnet.children())[:-1])
        #        (keeps everything up to and including the avgpool;
        #         output is (B, 512, 1, 1), so you flatten it in forward)
        #     b) replace it: resnet.fc = nn.Identity()
        #        (output is already (B, 512))
        # - Store the result as self.backbone. Note which option you chose,
        #   because it changes whether forward() needs a flatten.
        # - Hint: resnet.fc.in_features gives you 512 without hard-coding it.
        #   Save it as self.img_dim.

        # ---------------------------------------------------------------
        # TODO 2: MONTH BRANCH (small MLP)
        # ---------------------------------------------------------------
        # Goal: turn the (B, 2) sin/cos month into a (B, month_hidden) vector.
        #
        # - nn.Sequential with a Linear(month_dim, month_hidden) followed by
        #   a ReLU is enough. Store it as self.month_mlp.
        # - Why a learned layer instead of concatenating the raw 2 numbers?
        #   It lets the model build its own month features (e.g. "early
        #   summer") before they meet the image features.

        # ---------------------------------------------------------------
        # TODO 3: THE TWO HEADS
        # ---------------------------------------------------------------
        # Goal: two separate classifiers that read the SAME combined vector.
        #
        # - Combined size = self.img_dim + month_hidden (512 + 16 = 528).
        # - self.age_head    = Linear(combined, n_age)     -> adult / young
        # - self.antler_head = Linear(combined, n_antler)  -> antlerless /
        #                                                     antlered
        #   Class order is fixed in constants.py (team rule: class 1 is the
        #   rarer class that F1 is scored on).
        # - Each head has its own weights. Positions 513-528 of the input
        #   always hold month features, so the weights at those positions
        #   learn how much the month should matter. Nothing labels them as
        #   "month"; consistent position is what makes it work.
        # - Optional: add nn.Dropout(p=0.2) before each head to reduce
        #   overfitting.

        raise NotImplementedError("TODO: build the layers above")

    def forward(self, image, month):
        # ---------------------------------------------------------------
        # TODO 4: FORWARD PASS
        # ---------------------------------------------------------------
        # image: (B, 3, 224, 224)   month: (B, 2)
        #
        # Step A: img_feat = image through self.backbone.
        #         If you used option (a) in TODO 1, flatten it:
        #         img_feat = img_feat.flatten(1)   -> (B, 512)
        # Step B: month_feat = month through self.month_mlp -> (B, 16)
        #         (the two branches are independent; order does not matter)
        # Step C: combine with torch.cat([img_feat, month_feat], dim=1)
        #         -> (B, 528). dim=1 joins along features, not the batch.
        #         The image features MUST come first, every time.
        # Step D: age_logits    = self.age_head(combined)     -> (B, 2)
        #         antler_logits = self.antler_head(combined)  -> (B, n_antler)
        # Step E: return age_logits, antler_logits   (a tuple, raw logits,
        #         NO softmax; cross_entropy applies it internally)
        #
        # How gradients flow back (you don't write this; autograd does):
        # the loss gradient for the 528-vector is split by torch.cat,
        # the first 512 values go to the backbone, the last 16 to month_mlp.
        raise NotImplementedError("TODO: implement forward")


# ---------------------------------------------------------------------
# OPTIONAL TODO 5: freezing helpers for staged training
# ---------------------------------------------------------------------
# def freeze_backbone(self): set requires_grad=False on self.backbone params
# def unfreeze_backbone(self): set it back to True
# Your train.py can call these if they exist (warm up the heads first,
# then fine-tune everything with a lower learning rate).
