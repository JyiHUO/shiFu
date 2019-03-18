import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
from config import Config

random.seed(5)


class my_dataset(Dataset):
    def __init__(self, t="train"):
        if Config["normal_config"]["large_file"]:
            if t == "train":
                self.h5f = h5py.File(Config["normal_config"]["hd5_train_path"], 'r')
            elif t == "val":
                self.h5f = h5py.File(Config["normal_config"]['hd5_val_path'], 'r')
            elif t == "test":
                self.data = pd.read_csv(Config["normal_config"]["test_path"]).values
        else:
            if t == "train":
                self.data = pd.read_csv(Config["normal_config"]["train_path"]).values
            elif t == "val":
                self.data = pd.read_csv(Config["normal_config"]["val_path"]).values
            elif t == "test":
                self.data = pd.read_csv(Config["normal_config"]["test_path"]).values

            # ["user_id"0, "user_city"1, "item_id"2, "author_id"3, "item_city"4,
            # "channel"5, "finish"6, "like"7, "music_id"8, "device"9, "time"10, "duration_time"11]

    def __getitem__(self, index):
        if Config["normal_config"]["large_file"]:
            pass
        else:
            '''
            columns = [
            "uid", "user_city", "item_id", "author_id", "item_city",
            "channel", "finish", "like", "music_id", "device",
            "create_time", "duration_time"
            ]
            '''
            return self.data[index][[0, 1, 2, 3, 4, 5, 8, 9, 11, 6, 7]]  # finish and like is in the end

    def __len__(self):
        if Config["normal_config"]["large_file"]:
            return self.h5f['target'].shape[0]
        else:
            return self.data.shape[0]


class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self):
        pass

    def instance_a_loader(self, t="train"):
        """instance train loader for one training epoch"""
        dataset = my_dataset(t=t)
        if t == "train":
            shuffle = True
        else:
            shuffle = False
        return DataLoader(dataset, batch_size=Config["training_config"]["batch_size"],
                          shuffle=shuffle, num_workers=Config["normal_config"]["num_workers"])

