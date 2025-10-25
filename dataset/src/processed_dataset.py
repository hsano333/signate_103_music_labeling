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

import torch
import numpy as np
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

    def stretch_data(self, data, max_data, sr):
        if max_data < data.shape[1]:
            data = data[:, :max_data]
        elif max_data > data.shape[1]:
            data = torch.nn.functional.pad(data, (0, max_data - data.shape[1]))
        return data

    def my_preprocessing(self, files, master, precision=np.float32, is_train=False):
        arr = 0
        max_data = 1320
        if is_train:
            arr = np.empty((1000, 64, max_data), dtype=np.float32)
        else:
            arr = np.empty((500, 64, max_data), dtype=np.float32)
        data = []
        max_f = 0
        i = 0
        for file in files:
            y, sr = librosa.load(file)
            _ = librosa.feature.melspectrogram(y=y, sr=sr)
            y_mel = librosa.amplitude_to_db(_).flatten()
            data.append(y_mel.astype(precision))

            transforms = Compose(
                [
                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.02, p=0.75),
                    PitchShift(min_semitones=-4, max_semitones=4, p=0.75),
                ]
            )
            augmented_data = transforms(samples=y, sample_rate=sr)
            to_mel = T.MelSpectrogram(
                sample_rate=sr, n_fft=1024, hop_length=512, n_mels=64
            )
            waveform = torch.tensor(augmented_data, dtype=torch.float32)
            spec = to_mel(waveform)
            # i = i + 1
            if is_train:
                spec_aug = torch.nn.Sequential(
                    T.FrequencyMasking(freq_mask_param=random.randint(6, 20)),
                    T.TimeMasking(time_mask_param=random.randint(50, 120)),
                )(spec)

            # if max_data < spec_aug.shape[1]:
            #     spec_aug = spec_aug[:, :max_data]
            # elif max_data > spec_aug.shape[1]:
            #     spec_aug = torch.nn.functional.pad(
            #         spec_aug, (0, max_data - spec_aug.shape[1])
            #     )
            #print(f"{spec_aug.shape=}")
            arr[i] = self.stretch_data(spec, max_data, sr).numpy()
            i = i + 1
            if is_train:
                arr[i] = self.stretch_data(spec_aug, max_data, sr).numpy()
                i = i + 1
        print(f"{max_f=}")

        # データフレームを作成
        data_df_ = pd.DataFrame(data[0])
        for i in range(1, 500):
            data_df_ = pd.concat([data_df_, pd.DataFrame(data[i])], axis=1)
        data_df = data_df_.T
        data_df.index = master.index
        # data_df.dropna(axis=1, inplace=True)
        data_df = data_df[data_df.columns[:165120]]
        return data_df

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

        train_files = natsorted(glob.glob(self.TRAIN_DATA_PATH))
        test_files = natsorted(glob.glob(self.TEST_DATA_PATH))

        train_df = self.my_preprocessing(train_files, train_master, np.float32)
        test_pd = self.my_preprocessing(test_files, sample_submit, np.float16, False)

        n_components = 166
        pca = PCA(n_components=n_components, random_state=82)
        col = [f"pc{i}" for i in range(n_components)]
        train_pca = pd.DataFrame(pca.fit_transform(train_df), columns=col)
        test_pca = pd.DataFrame(pca.transform(test_pd), columns=col)

        test_label = sample_submit[0].astype(str).to_numpy()

        self.data = train_pca.to_numpy()
        self.label = train_master["label_id"].astype(int).to_numpy()
        self.raw_label = train_master["label_id"]

        self.label_number = 1
        self.test_data = test_pca.to_numpy()
        self.test_id = sample_submit.index.astype(str).to_numpy()

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
