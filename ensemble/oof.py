"""
Out of Folder (OOF) module.
"""

from common.my_enum import MLTask
from dataset.src.combined_dataset import CombinedDataset

from dataset.src.simple_dataset import SimpleDataset
from dataset.src.image_dataset import ImageDataset
from pathlib import Path
from common.utility import Utility
import numpy as np
import torch
import os


class OOF:
    def __init__(self, name, config, dir_path, task=MLTask.Classification):
        self.name = name
        self.config = config
        self.dataset_models = {}
        self.dir_path = dir_path
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.KFoldSplit = self.config["stacking"]["k_fold"]
        self.RANDAOM_STATE = self.config["stacking"]["random_state"]
        self.task = task
        self.scores = {}

    def add_model(self, name, dataset, model, evaluation):
        # 最後はwegit
        self.dataset_models[name] = (dataset, model, evaluation, 1.0)

    def clear_models(self):
        self.dataset_models.clear()

    def make_features(self, forece_make=False):
        preds = []
        tests = []
        # preds = np.zeros((), np.float32)
        # tests = np.zeros((), np.float32)
        # self.scores = np.zeros(len(self.dataset_models))

        i = -1
        for name, (dataset, model, evaluation, _) in self.dataset_models.items():
            i = i + 1
            model_name = name
            dataset_name = dataset.get_name()
            (directory, file_cv_trained_path, file_cv_new_test_path) = (
                self.make_directory(dataset_name, model_name)
            )

            if not os.path.isfile(file_cv_trained_path) or forece_make:
                directory.mkdir(parents=True, exist_ok=True)
                (pred, test) = self.make_new_feature(
                    model, model_name, dataset, directory, evaluation
                )
                np.savez(file_cv_trained_path, pred=pred, test=test)
                print("make:" + file_cv_trained_path)
            else:
                data = np.load(file_cv_trained_path)
                pred = data["pred"]
                test = data["test"]
                print("load:" + file_cv_trained_path)

            preds.append(pred)
            tests.append(test)
            label = dataset.get_numpy_data()[1]
            # print(f"{label=}")
            # print(f"{pred=}")
            # print(f"{label.shape=}")
            # print(f"{pred.shape=}")

            self.scores[name] = evaluation(label, pred)

        result_pred = np.hstack(preds)
        result_test = np.hstack(tests)
        # print(f"{result_pred.shape=}, {result_test.shape=}")

        # ラベルはすべてのデータセットで同じなので、最初のデータセットから取得
        (_, label, _) = next(iter(self.dataset_models.values()))[0].get_numpy_data()
        np.savez(file_cv_new_test_path, pred=result_pred, test=result_test)

        combined_dataset = CombinedDataset(result_pred, dataset, result_test, self.name)
        return combined_dataset

    def make_new_feature(self, model, model_name, dataset, directory, evaluation):
        kf = Utility().get_kf(self.task, self.KFoldSplit, self.RANDAOM_STATE)
        (data, label, test) = dataset.get_numpy_data()
        # print(f"{dataset.get_name()=}")
        # print(f"{label=}")
        label_number = dataset.get_label_number()
        label_unique_num = np.unique(label).shape[0]

        tmp_name = dataset.get_name()
        preds = []
        preds_test = []
        va_idxes = []
        label_num = dataset.get_label_number()

        if label.ndim == 1:
            split_label = label
        else:
            split_label = label[:, 0]
        # print(f"{split_label=}")
        #print(f"{data.shape=}, {label.shape=}, {test.shape=}")
        y_prob_all = np.zeros((len(data), label_unique_num), np.float32)
        y_test_all = np.zeros((len(test), label_unique_num), np.float32)
        # y_true_all = np.zeros((len(data), 10), np.float32)
        y_true_all = np.zeros(len(data), np.float32)
        # print(f"{split_label=}")
        for i, (tr_idx, val_idx) in enumerate(kf.split(data, split_label)):
            k_directory = directory / f"fold_{i + 1}"
            va_idxes.append(val_idx)
            # print(f"{tr_idx=}, {val_idx=}, {len(data)=}")
            train_data, val_data = data[tr_idx], data[val_idx]
            train_label, val_label = label[tr_idx], label[val_idx]

            new_train = SimpleDataset(
                train_data,
                train_label,
                label_number,
                tmp_name + "_train" + str(i + 1),
            )
            new_val = SimpleDataset(
                val_data,
                val_label,
                label_number,
                tmp_name + "_val" + str(i + 1),
            )
            new_test = SimpleDataset(
                test,
                None,
                label_number,
                tmp_name + "_test" + str(i + 1),
            )

            model_config = self.config["model"][model_name]
            model_config["label_number"] = label_num
            new_model = model(
                model_config,
                k_directory.as_posix(),
                new_train,
                new_val,
                new_test,
                evaluation,
            )
            # params = model_config.get("learned_params", {})
            params = new_model.get_model_optimized_params(model_config)
            # print(f"{params=}")
            new_model.learn(i, params)
            new_model.reload()
            # print(f"{val_data.shape=}")
            pred = new_model.forecast(torch.tensor(val_data.astype(np.float32)))
            # print(f"{pred.shape=}, {val_data.shape=}, {train_data.shape=}")
            preds.append(pred.numpy())
            # print(f"{preds[-1].shape=}, {len(preds)=}, {preds[0:10]=}")
            # print(f"{test.shape=}")
            pred_test = new_model.forecast(torch.tensor(test.astype(np.float32)))

            preds_test.append(pred_test.numpy())
            # print(f"{pred_test.shape=}, {pred.shape=}, {y_prob_all.shape=}")
            # print(f"{pred_test[0:5]=}, {pred[0:5]=},{len(val_idx)=} ")
            # print(f"{pred_test.shape=}, {len(preds_test)=}, {preds_test[0:10]=}")
            # print(f"{pred.shape=}, {val_label.shape=}")
            y_prob_all[val_idx] = pred.numpy()
            y_true_all[val_idx] = val_label
            y_test_all += pred_test.numpy()

        # final_score = evaluation(y_true_all, np.argmax(y_prob_all, axis=1))
        final_score = evaluation(y_true_all, y_prob_all)
        print(f"Out of Fold Final Score: {final_score:.4f}")
        return (y_prob_all, y_test_all / self.KFoldSplit)

    def make_directory(self, dataset_name, model_name):
        name = dataset_name + "_" + model_name + "_k" + str(self.KFoldSplit)
        directory = self.dir_path / name
        file_cv_trained_path = (directory / "cv_trained.npz").as_posix()
        file_cv_new_test_path = (directory / "cv_new_test.npz").as_posix()
        Utility().make_kfold_directory(directory, self.KFoldSplit)
        return (directory, file_cv_trained_path, file_cv_new_test_path)
