import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import wfdb
from sklearn.preprocessing import OneHotEncoder


def onehot_label(labels):
    labels = labels.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_vector = onehot_encoder.fit_transform(labels)
    tmp_list = []
    for j in range(onehot_vector.shape[0]):
        tmp_list.append(onehot_vector[j, :])
    return tmp_list


class ECGDataset_unseen(Dataset):
    def __init__(self, phase, data_dir, label_csv, leads, length):
        super(ECGDataset_unseen, self).__init__()
        self.phase = phase
        df = pd.read_csv(label_csv)
        labels_int = df['label'].values
        label_types = list(set(labels_int))
        onehot_vector = onehot_label(labels_int)
        df["label_vec"] = onehot_vector
        df = df.loc[df['split'] == phase]
        self.data_dir = data_dir
        self.labels = df
        self.nleads = leads
        data_df = pd.DataFrame()
        data_df["Recording"], data_df["label_vec"] = df["Recording"], df["label_vec"]
        self.data = data_df.values
        self.length = length

    def __getitem__(self, index: int):
        file_name, onehot_label = self.data[index]
        file_path = os.path.join(self.data_dir, file_name)
        df = pd.read_csv(file_path, sep=",")
        ecg_data = df.values[:]
        nsteps, _ = ecg_data.shape
        ecg_data = ecg_data[-self.length:, :]
        result = np.zeros((self.length, self.nleads))
        result[-nsteps:, :] = ecg_data

        return torch.from_numpy(result.transpose()).float(), torch.from_numpy(onehot_label).float()

    def __len__(self):
        return len(self.labels)


class ECGDataset_unseen_MHL_stage2(Dataset):
    def __init__(self, phase, data_dir, label_csv, leads, length):
        super(ECGDataset_unseen_MHL_stage2, self).__init__()
        self.phase = phase
        df = pd.read_csv(label_csv)
        labels_int = df['label'].values
        label_types = list(set(labels_int))
        onehot_vector = onehot_label(labels_int)
        df["label_vec"] = onehot_vector
        assert phase == 'train_valid'
        trn_indx_stage2 = df[df['split'] == 'train'].index.tolist()
        val_indx_stage2 = df[df['split'] == 'valid'].index.tolist()
        trn_val_indx = []
        trn_val_indx.extend(trn_indx_stage2)
        trn_val_indx.extend(val_indx_stage2)
        df = df.loc[trn_val_indx]
        self.data_dir = data_dir
        self.labels = df
        self.nleads = leads
        data_df = pd.DataFrame()
        data_df["Recording"], data_df["label_vec"] = df["Recording"], df["label_vec"]
        self.data = data_df.values
        self.length = length

    def __getitem__(self, index: int):
        file_name, onehot_label = self.data[index]
        file_path = os.path.join(self.data_dir, file_name)
        df = pd.read_csv(file_path, sep=",")
        ecg_data = df.values[:]
        nsteps, _ = ecg_data.shape
        ecg_data = ecg_data[-self.length:, :]
        result = np.zeros((self.length, self.nleads))
        result[-nsteps:, :] = ecg_data

        return torch.from_numpy(result.transpose()).float(), torch.from_numpy(onehot_label).float()

    def __len__(self):
        return len(self.labels)