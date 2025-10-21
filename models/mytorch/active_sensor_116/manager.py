import os
from enum import Enum
import importlib
import torch
import torchvision
import torch.nn as nn
from torch import optim
from enum import IntEnum

from .model import Model


METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3

SAVED_LEARNING_DATA = "learning_dataset.npz"
SAVED_MODEL_NAME = "learned_model.pth"
SAVED_TMP_MODEL_NAME = "tmp_learned_model.pth"




class BaseManager:
    def __init__(self, model, save_path):
        self.save_path = save_path
        if os.path.isdir(save_path) is False:
            os.makedirs(save_path)

        # self.model = Model()
        self.model = model

        # 損失関数
        self.criterion = nn.CrossEntropyLoss()

        self.save_path = save_path

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return optim.Adam(
            self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8
        )

    def get_tmp_model_name(self):
        return SAVED_TMP_MODEL_NAME

    def get_model_name(self):
        return SAVED_MODEL_NAME

    def get_path(self):
        return self.save_path

    def get_mode(self):
        return self.mode

    def compute_batch_loss(
        self,
        model,
        batch_ndx,
        data_label,
        device,
        metrics,
        batch_max_size,
    ):
        (data, label) = data_label
        data = data.to(device)
        label = label.to(device)
        prediction = model(data)
        loss = self.criterion(prediction, label)

        start_ndx = batch_ndx * batch_max_size
        end_ndx = start_ndx + label.size(0)

        with torch.no_grad():
            tmp_label = torch.max(label.detach(), 1)[1]
            tmp_prediction = prediction.detach()
            result_prediction = torch.max(tmp_prediction, 1)[1]

            metrics[METRICS_LABEL_NDX, start_ndx:end_ndx] = tmp_label
            metrics[METRICS_PRED_NDX, start_ndx:end_ndx] = result_prediction
            metrics[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss.detach()
        return loss

    def evaluate(self, metrics):
        val_loss = (
            metrics[METRICS_LABEL_NDX, :] == metrics[METRICS_PRED_NDX, :]
        ).sum() / len(metrics[METRICS_LABEL_NDX, :])
        return val_loss

    def get_metrics_size(self):
        return METRICS_SIZE

    def log_metrics(self, epoch_ndx, mode_str, metrics, writer):
        results = self.evaluate(metrics)
        writer.add_scalar(mode_str + ":result", results, epoch_ndx)
        return results
