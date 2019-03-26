import pandas as pd
import lightgbm as lgb
from config import Config
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold
import numpy as np
import ast
import pickle
import json

test_data = pd.read_csv(Config["save_test_path"])
finish_model = []
like_model = []


def lgb_predict_media():
    for i in range(1, 8):
        with open(Config["model_path_media"] + "_" + "finish" + "_" + str(i), "wb") as f:
            clf = pickle.load(f)
            finish_model.append(clf)
        with open(Config["model_path_media"] + "_" + "like" + "_" + str(i), "wb") as f:
            clf = pickle.load(f)
            like_model.append(clf)

        for fm in finish_model:
            pred = fm.predict_proba(test_data)
            print(pred.shape)