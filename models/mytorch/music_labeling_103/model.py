import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import IntEnum
from torchvision import models


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
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.features = nn.Sequential(*list(base.children())[:-1])

        self.classifier = nn.Linear(base.fc.in_features, 10)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        # RNN部分（時間方向の依存関係を捉える）
        self.lstm = nn.LSTM(
            input_size=64 * 32,  # 周波数方向が半分×半分になった
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # 全結合層
        self.fc = nn.Linear(128 * 2, 10)

    def forward(self, x):
        # x = x
        # print(f"{x.shape=}")
        # print(f"{x.shape=}")
        # print(f"{x.shape=}")
        x = x.unsqueeze(1)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

        x = self.cnn(x)  # (B, 64, 32, 330)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T, C, F)
        x = x.view(b, t, c * f)  # (B, T, C*F)
        x, _ = self.lstm(x)  # (B, T, H*2)
        x = x[:, -1, :]  # 最後の時刻を使用
        x = self.fc(x)

        return x
