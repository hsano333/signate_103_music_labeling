import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import IntEnum


D_in = 120
H = 4
D_out = 1
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2


class Diff(IntEnum):
    NEG = 0
    ZERO = 1
    POS = 2


class Model(nn.Module):
    # class LeNet(nn.Module):
    def __init__(self, output_size=10, activation="relu"):
        # def __init__(self, output_size=10, activation="tanh"):
        super(Model, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

        # 活性化関数の辞書
        self.activations = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
        }

    def forward(self, x):
        act_func = self.activations.get(self.activation, F.relu)
        x = self.pool(act_func(self.conv1(x)))
        x = self.pool(act_func(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = act_func(self.fc1(x))
        x = act_func(self.fc2(x))
        x = self.fc3(x)
        return x
