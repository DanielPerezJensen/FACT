#!/usr/bin/env python

"""Classes that define our self-created classifier models to aid in
the reproducibility study of
'Generative causal explanations of black-box classifiers'"""

import os
import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    """InceptionBlock, described in: https://arxiv.org/abs/1409.4842"""

    def __init__(self, c_in, c_out, c_red, act_fn):
        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
                nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
                nn.BatchNorm2d(c_out["1x1"]),
                act_fn()
            )

        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
                nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
                nn.BatchNorm2d(c_red["3x3"]),
                act_fn(),
                nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out["3x3"]),
                act_fn()
            )

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
                nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
                nn.BatchNorm2d(c_red["5x5"]),
                act_fn(),
                nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out["5x5"]),
                act_fn()
            )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            act_fn()
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out


class InceptionNetDerivative(nn.Module):
    """Derivative of Google net with only one inception block"""

    def __init__(self, num_classes=10, act_fn=nn.ReLU):
        super().__init__()

        self.act_fn = act_fn

        self.input_net = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                self.act_fn()
            )

        self.blocks = nn.Sequential(
                InceptionBlock(64, c_red={"3x3": 32, "5x5": 16},
                               c_out={"1x1": 16, "3x3": 48, "5x5": 8, "max": 8},
                               act_fn=self.act_fn),
                nn.MaxPool2d(3, stride=2, padding=1)
            )

        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(80, num_classes)
            )

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        y = self.output_net(x)

        return y


class ResNetBlock(nn.Module):
    """Original ResNetBlock, described in https://arxiv.org/abs/1603.05027"""

    def __init__(self, c_int, act_fn, c_out):
        """
        Inputs:
            c_in - Number of input dims
            act_fn - Activation function (e.g. nn.ReLU)
            c_out - Number of classes
        """
        super().__init__()

        self.net = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(c_out),
                act_fn(),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2)

        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        x = self.downsample(x)

        out = z + x
        out = self.act_fn(out)

        return out
