import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import IntEnum


METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2


class Model(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.5) -> None:
        super().__init__()
        # super(Model, self).__init__()
        self.D_in = input_size
        self.D_out = output_size
        self.dropout1 = nn.Dropout(dropout_rate)

        print(f"{int(self.D_in / 4)=}")
        print(f"{int(self.D_in / 16)=}")
        print(f"{4 * self.D_out=}")

        self.linear1 = torch.nn.Linear(self.D_in, int(self.D_in / 4))
        self.linear2 = torch.nn.Linear(int(self.D_in / 4), int(self.D_in / 16))
        self.linear3 = torch.nn.Linear(int(self.D_in / 16), 4 * self.D_out)
        self.linear4 = torch.nn.Linear(4 * self.D_out, self.D_out)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout1(x)
        x = self.relu(self.linear2(x))
        x = self.dropout1(x)
        x = self.relu(self.linear3(x))
        x = self.dropout1(x)
        x = self.linear4(x)
        return x
