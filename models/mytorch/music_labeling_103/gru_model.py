import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import IntEnum


D_in = 128
H = 4
D_out = 10
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2

INPUT_DIM = 128  # MelSpectrogramのn_mels (特徴量次元)
HIDDEN_DIM = 256  # RNNの隠れ状態の次元
NUM_LAYERS = 2  # RNN層の深さ
NUM_CLASSES = 10  # 分類する音楽ラベルの数
SEQUENCE_LENGTH = 1320  # 時間ステップの長さ


class Diff(IntEnum):
    NEG = 0
    ZERO = 1
    POS = 2


class Model(nn.Module):
    # class LeNet(nn.Module):
    def __init__(self, output_size=10, activation="relu"):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=INPUT_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.5 if NUM_LAYERS > 1 else 0,  # 複数層の場合のみdropout適用
        )

        in_channels = 1
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        # self.fc2 = nn.Linear(256, D_out)

        # super(Model, self).__init__()
        # self.activation = activation
        # self.conv1 = nn.Conv2d(128, 1320, kernel_size=5)
        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(32, , kernel_size=5)
        # self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, output_size)

        # 活性化関数の辞書
        self.activations = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
        }

    def forward(self, x):
        # x = x
        # act_func = self.activations.get(self.activation, F.relu)
        # print(f"{x.shape=}")
        # print(f"No.1:{x.shape=}")
        x = x.transpose(1, 2)

        # print(f"No.2:{x.shape=}")

        out, h_n = self.rnn(x)
        final_hidden_state = h_n[-1, :, :]
        out = self.dropout(final_hidden_state)
        # print(f"No.3:{out.shape=}")
        out = self.fc1(out)

        # x = self.pool(F.relu(self.conv1(x)))  # → (32, n_mels/2, time/2)
        # # x = self.pool(F.relu(self.conv2(x)))  # → (64, n_mels/4, time/4)
        # x = self.pool(F.relu(self.conv3(x)))  # → (128, n_mels/8, time/8)
        # x = x.flatten(1)
        # # print(f"{x.shape=}")
        # x = self.dropout(F.relu(self.fc1(x)))
        # x = self.fc2(x)
        return out
