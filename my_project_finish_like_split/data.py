import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
from config import Config
import json

random.seed(5)


class my_dataset(Dataset):
    def __init__(self, title, face, video, audio,  t="train"):
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

            self.title = title
            # item_id, title_length, word_0, word_1, word_2, word_3, word_4, word_5, word_6, word_7, word_8, word_9
            self.face = face
            #item_id, male_perc, female_perc, faces	maleBeauty, femaleBeauty, faceSquare, male_num, femalie_num
            self.video = video
            # item_id, video_feature_dim_128
            self.audio = audio
            # item_id, audio_feature_128_dim

    def __getitem__(self, index):
        if Config["normal_config"]["large_file"]:
            pass
        else:
            '''
            columns = [
            "uid" 0, "user_city" 1, "item_id" 2, "author_id" 3, "item_city" 4,
            "channel" 5, "finish" 6, "like" 7, "music_id" 8, "device" 9,
            "create_time 10", "duration_time 11", "fl_00 12", "fl_01 13", "fl_11 14", "fl_10 15", "target 16"
            ]
            '''
            item_id = str(self.data[index][2])
            face = self.face.get(item_id, np.zeros(8))
            title_data = self.title.get(item_id, np.zeros(11))
            title_length = title_data[0]
            title_word = title_data[1:]
            video = self.video.get(item_id, np.zeros(128))
            audio = self.audio.get(item_id, np.zeros(128))
            behaviour = None

            if Config["normal_config"]["task"] == "finish":
                behaviour = self.data[index][[0, 1, 2, 3, 4, 5, 8, 9, 11, 6]]  # finish or like is in the end
            else:
                behaviour = self.data[index][[0, 1, 2, 3, 4, 5, 8, 9, 11, 7]]
            return np.concatenate([behaviour[:-1], [title_length], face, title_word, video, audio, [behaviour[-1]]])

    def __len__(self):
        if Config["normal_config"]["large_file"]:
            return self.h5f['target'].shape[0]
        else:
            return self.data.shape[0]


class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self):
        with open(Config["normal_config"]["title_data_path"], "r") as f:
            self.title_json = json.load(f)
        with open(Config["normal_config"]["face_data_path"], "r") as f:
            self.face_json = json.load(f)
        with open(Config["normal_config"]["video_data_path"], "r") as f:
            self.video_json = json.load(f)
        with open(Config["normal_config"]["audio_data_path"], "r") as f:
            self.audio_json = json.load(f)

    def instance_a_loader(self, t="train"):
        """instance train loader for one training epoch"""
        dataset = my_dataset(self.title_json, self.face_json, self.video_json, self.audio_json, t=t)
        if t == "train":
            shuffle = True
        else:
            shuffle = False
        return DataLoader(dataset, batch_size=Config["training_config"]["batch_size"],
                          shuffle=shuffle, num_workers=Config["normal_config"]["num_workers"])

