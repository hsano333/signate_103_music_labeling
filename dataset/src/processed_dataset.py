# from enum import auto
from torch.utils.data import Dataset
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from dataset.src.simple_dataset import SimpleDataset
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    Shift,
    PitchShift,
    Gain,
    ApplyImpulseResponse,
)
import glob
from natsort import natsorted
import librosa
from sklearn.decomposition import PCA
import numpy as np
import torchaudio
from torchaudio import transforms as T
import torch
import random


# from sklearn.preprocessing import Normalizer


from common.my_enum import MLTask

# from models.linear_regression.my_linear_regression import MyLinearRegression
from dataset.src.time_dataset import TimeDataset

import joblib

import pandas as pd
import os
import cv2
from torchvision import transforms


class ProcessedDataset(Dataset):
    def __init__(self, config, name="processed_dataset"):
        current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
        self.TRAIN_PATH = os.path.join(
            current_dir, "data", "original", "train_master.tsv"
        )
        self.LABEL_PATH = os.path.join(
            current_dir, "data", "original", "label_master.tsv"
        )
        self.SAMPLE_PATH = os.path.join(
            current_dir, "data", "original", "sample_submit.tsv"
        )
        self.TRAIN_DATA_PATH = os.path.join(
            current_dir, "data", "original", "train_sound_*", "train_*.au"
        )
        self.TEST_DATA_PATH = os.path.join(
            current_dir, "data", "original", "test_sound_*", "test_*.au"
        )

        self.TRAIN_IMAGE_PATH = os.path.join(current_dir, "data", "original", "train")
        self.TEST_IMAGE_PATH = os.path.join(current_dir, "data", "original", "test")

        self.PREPOCESSED_PATH = os.path.join(
            current_dir, "data", "original", "dataset_preprocessed.pkl"
        )

        self.dateset_name = name
        self.config = config
        self.target_feature = self.config["target_feature"]

    def get_numpy_data(self):
        return (self.data, self.label, self.test_data)
        # return (self.data.numpy(), self.label.numpy(), self.test_data.numpy())

    def get_test_id(self):
        return self.test_id

    def get_label_index(self):
        return self.raw_label.index

    def set_raw_label(self, raw_label):
        # pandas
        self.raw_label = raw_label

    def get_raw_label(self):
        # pandas
        return self.raw_label

    def convert_np_to_torch(self, data):
        return torch.tensor(data.values.astype(np.float32))

    def get_label_scaler(self):
        return self.label_scaler

    def get_mean_df(self):
        return (self.train_mean, self.test_mean)

    def convert_onehot_encoder(self, encoder, data, name):
        encoderd_data = encoder.transform(data[[name]])
        feature_names = encoder.get_feature_names_out([name])
        tmp_df = pd.DataFrame(encoderd_data, columns=feature_names)
        return pd.concat([data, tmp_df], axis=1)

    def convert_oneshot(self, data, name):
        encoded_data = pd.get_dummies(
            data,
            columns=[
                name,
            ],
        )
        return encoded_data

    def transform_label(self, task):
        if task == MLTask.Regression:
            self.label_number = self.label.shape[0]
            # print(f"{type(self.label)=}")
            # self.label = torch.from_numpy(self.label)
        else:
            unique = self.raw_label.unique()
            self.le = LabelEncoder()
            raw_np_label = self.le.fit_transform(self.label)
            self.label = torch.from_numpy(raw_np_label)
            self.label_number = unique.shape[0]

    def make_date_id(self):
        start_date = "2013-11-18"
        end_date = "2014-12-01"

        all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
        df = pd.DataFrame(all_dates, columns=["date"])
        df["datetime"] = pd.to_datetime(df["date"])
        df["weekday"] = df["datetime"].dt.day_name()
        # df["weekday_i"] = df["date"].dt.dayofweek

        df = df.reset_index()
        df = df.rename(columns={"index": "id"})
        df["week_id"] = df["id"] // 7
        # print(f"{df["week_id"]=}")
        df = df.drop(columns=["id", "date"])

        return df

    def get_mean_y(self, data, onehot_encoder):
        def trimmed_mean(x):
            if len(x) >= 3:
                return x.sort_values().iloc[0:-2].mean()
            else:
                return x.mean()  # データ数が2以下なら普通に平均

        mean_y = data.groupby("week_id")["y"].agg(trimmed_mean).reset_index()
        tmp_label = mean_y["y"]

        tmp_data = mean_y["week_id"]

        liner = LinearRegression()
        # print(f"{tmp_data.shape=}, {tmp_label.shape=}")
        self.model = liner.fit(np.array(tmp_data).reshape(-1, 1), tmp_label)

        return self.model

    def test_augmentation(self, data, test_dir):
        test_images = []
        return_images = []

        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

        file_names = data[0].tolist()
        kernel_2x2 = np.ones((2, 2), np.uint8)

        # num_images = len(file_names) * 7
        # Earr = np.empty((len(file_names), 7, 1, 28, 28), dtype=np.float32)
        arr = np.empty((len(file_names), 1, 28, 28), dtype=np.float32)
        i = -1

        for name in file_names:
            filepath = os.path.join(test_dir, name)
            load_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            image = transform_test(load_image.astype(dtype=np.uint8))
            i = i + 1
            j = 0
            arr[i] = image.numpy()

        return arr

    def train_test_augmentation(self, data, data_dir):
        train_images = []

        transform_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        file_names = data["file_name"].tolist()
        file_names[0]
        kernel_2x2 = np.ones((2, 2), np.uint8)

        # num_images = len(file_names) * 10
        arr = np.empty((len(file_names), 1, 28, 28), dtype=np.float32)
        i = -1
        for name in file_names:
            filepath = os.path.join(data_dir, name)
            load_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            image = transform_train(load_image.astype(dtype=np.uint8))
            i = i + 1
            arr[i] = image.numpy()

        return arr

    def train_test_augmentation2(self, data, data_dir):
        train_images = []

        transform_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        transform_augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        file_names = data["file_name"].tolist()
        file_names[0]
        kernel_2x2 = np.ones((2, 2), np.uint8)

        # num_images = len(file_names) * 10
        arr = np.empty((len(file_names), 1, 28, 28), dtype=np.float32)
        i = -1
        num_images = len(file_names) * 4
        arr = np.empty((num_images, 1, 28, 28), dtype=np.float32)
        for name in file_names:
            filepath = os.path.join(data_dir, name)
            load_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            image = transform_train(load_image.astype(dtype=np.uint8))
            i = i + 1
            arr[i] = image.numpy()

            for k in range(3):
                image = transform_augmentation(load_image.astype(dtype=np.uint8))
                i = i + 1
                arr[i] = image.numpy()

            # erode_image = cv2.erode(load_image, kernel_2x2, iterations=1)
            # for i in range(3):
            #     image = transform_augmentation(erode_image.astype(dtype=np.uint8))
            #     # train_images.append(image)
            #     i = i + 1
            #     arr[i] = image.numpy()

        return arr

    def augmentation(self, data, data_dir):
        train_images = []

        transform_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        transform_augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        file_names = data["file_name"].tolist()
        file_names[0]
        kernel_2x2 = np.ones((2, 2), np.uint8)

        num_images = len(file_names) * 10
        arr = np.empty((num_images, 1, 28, 28), dtype=np.float32)
        i = -1
        for name in file_names:
            filepath = os.path.join(data_dir, name)
            load_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            image = transform_train(load_image.astype(dtype=np.uint8))
            i = i + 1
            arr[i] = image.numpy()

            for m in range(3):
                image = transform_augmentation(load_image.astype(dtype=np.uint8))
                # train_images.append(image)
                i = i + 1
                arr[i] = image.numpy()

            erode_image = cv2.erode(load_image, kernel_2x2, iterations=1)
            for m in range(3):
                image = transform_augmentation(erode_image.astype(dtype=np.uint8))
                # train_images.append(image)
                i = i + 1
                arr[i] = image.numpy()

            dilate_image = cv2.dilate(load_image, kernel_2x2, iterations=1)
            for m in range(3):
                image = transform_augmentation(dilate_image.astype(dtype=np.uint8))
                # train_images.append(image)
                i = i + 1
                arr[i] = image.numpy()

        return arr

    def convert_mal2db(self, data, max_data, sr):
        if max_data < data.shape[1]:
            data = data[:, :max_data]
        elif max_data > data.shape[1]:
            data = torch.nn.functional.pad(data, (0, max_data - data.shape[1]))
        return librosa.power_to_db(data, ref=np.max)
        # return data

    def get_mean_std(self, data_all):
        global_mean = data_all.mean(axis=(0, 2))
        global_std = data_all.std(axis=(0, 2))
        epsilon = 1e-7
        global_std_safe = global_std + epsilon

        # ブロードキャストのために形状を (1, 96, 1) に拡張
        mean_broadcast = global_mean[:, np.newaxis]
        std_broadcast = global_std_safe[:, np.newaxis]

        # 標準化の実行: (データ - 平均) / 標準偏差
        # NumPyが自動的に (96, 1) を (500, 96, 1305) にブロードキャストして計算します
        # standardized_data = (data_all - mean_broadcast) / std_broadcast

        return mean_broadcast, std_broadcast

    def my_preprocessing(self, files, master, precision=np.float32, is_train=False):
        arr = 0
        max_data = 960
        n_mels = 96
        if is_train:
            arr = np.empty((1000, n_mels, max_data), dtype=np.float32)
        else:
            arr = np.empty((500, n_mels, max_data), dtype=np.float32)
        i = 0
        for file in files:
            y, sr = librosa.load(file)

            transforms = Compose(
                [
                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.02, p=0.75),
                    PitchShift(min_semitones=-4, max_semitones=4, p=0.75),
                ]
            )
            augmented_data = transforms(samples=y, sample_rate=sr)
            to_mel = T.MelSpectrogram(
                sample_rate=sr,
                n_fft=2048,
                win_length=2048,
                hop_length=512,
                n_mels=n_mels,
            )
            waveform = torch.tensor(augmented_data, dtype=torch.float32)
            waveform_ori = torch.tensor(y, dtype=torch.float32)
            spec = to_mel(waveform)
            spec_ori = to_mel(waveform_ori)
            if is_train:
                spec_aug = torch.nn.Sequential(
                    T.FrequencyMasking(freq_mask_param=random.randint(6, 20)),
                    T.TimeMasking(time_mask_param=random.randint(50, 120)),
                )(spec)

            arr[i] = self.convert_mal2db(spec_ori, max_data, sr)
            i = i + 1
            if is_train:
                arr[i] = self.convert_mal2db(spec_aug, max_data, sr)
                i = i + 1
        return arr

    def load(self):
        print(f"{os.path.exists(self.PREPOCESSED_PATH)=}")
        print(f"{self.config["overwrite"]=}")
        if os.path.exists(self.PREPOCESSED_PATH) and self.config["overwrite"] is False:
            print(f"Loading preprocessed dataset from {self.PREPOCESSED_PATH}")
            loaded_data = joblib.load(self.PREPOCESSED_PATH)
            self.data = loaded_data["data"]
            self.label = loaded_data["label"]
            self.label_number = 1
            self.test_data = loaded_data["test_data"]
            self.test_id = loaded_data["id"]
            self.raw_label = loaded_data["raw_label"]
            self.target_feature = loaded_data["target_feature"]
            print(f"Data Load:{self.target_feature=}")
            return

        print("Preprocessing dataset...")
        # train_master = pd.read_csv(self.TRAIN_PATH, sep="\t")
        # test_master = pd.read_csv(self.SAMPLE_PATH, sep="\t", header=None)

        train_master = pd.read_csv(self.TRAIN_PATH, sep="\t", index_col=0)
        label_master = pd.read_csv(self.LABEL_PATH, sep="\t")
        sample_submit = pd.read_csv(self.SAMPLE_PATH, sep="\t", header=None)
        sample_submit.columns = ["file_name", "label_id"]

        train_files = natsorted(glob.glob(self.TRAIN_DATA_PATH))
        test_files = natsorted(glob.glob(self.TEST_DATA_PATH))

        train_np = self.my_preprocessing(train_files, train_master, np.float32)
        test_np = self.my_preprocessing(test_files, sample_submit, np.float16, False)

        all_np = np.concatenate((train_np, test_np), axis=0)
        (mean, std) = self.get_mean_std(all_np)

        standardized_train_np = (train_np - mean) / std
        standardized_test_np = (test_np - mean) / std

        n_components = 166
        # pca = PCA(n_components=n_components, random_state=82)
        # col = [f"pc{i}" for i in range(n_components)]
        # train_pca = pd.DataFrame(pca.fit_transform(train_df), columns=col)
        # test_pca = pd.DataFrame(pca.transform(test_pd), columns=col)

        # test_label = sample_submit[0].astype(str).to_numpy()

        self.data = standardized_train_np
        self.label = train_master["label_id"].astype(int).to_numpy()
        self.raw_label = train_master["label_id"]

        self.label_number = 1
        self.test_data = standardized_test_np
        self.test_id = sample_submit["file_name"].to_numpy()

        data_to_save = {
            "data": self.data,
            "label": self.label,
            "test_data": self.test_data,
            "id": self.test_id,
            "raw_label": self.raw_label,
            "target_feature": self.target_feature,
        }
        joblib.dump(data_to_save, self.PREPOCESSED_PATH)
        print(f"Preprocessed dataset saved to {self.PREPOCESSED_PATH}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ndx):
        return (self.data[ndx], self.label[ndx])

    def get_name(self):
        return self.dateset_name

    def get_column_number(self):
        return self.data.shape[1]

    def get_label_number(self):
        return self.label_number

    def get_train_dataset(self):
        (train_data, train_label, test_data) = self.get_numpy_data()
        return SimpleDataset(
            train_data,
            train_label,
            0,
            self.dateset_name + "_train" + str(0 + 1),
        )

    def get_test_dataset(self):
        (train_data, train_label, test_data) = self.get_numpy_data()
        return SimpleDataset(
            test_data,
            None,
            0,
            self.dateset_name + "_train" + str(0 + 1),
        )
