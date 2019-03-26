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
test_data.drop(["finish", "like"], axis=1, inplace=True)
finish_model = []
like_model = []

uid = test_data["uid"].values
print(uid.shape)
item_id = test_data["item_id"].values
print(item_id.shape)


def lgb_predict_media():
    final_pred_list = []
    like_pred_list = []
    for i in range(1, 8):
        with open(Config["model_path_media"] + "_" + "finish" + "_" + str(i), "rb") as f:
            clf = pickle.load(f)
            finish_model.append(clf)
        with open(Config["model_path_media"] + "_" + "like" + "_" + str(i), "rb") as f:
            clf = pickle.load(f)
            like_model.append(clf)

    for fm in finish_model:
        pred = fm.predict_proba(test_data)
        final_pred_list.append(pred[:, 1][:, None])
        print(pred.shape)

    final_pred_list = np.concatenate(final_pred_list, 1)
    for lm in like_model:
        pred = lm.predict_proba(test_data)
        like_pred_list.append(pred[:, 1][:, None])
        print(pred.shape)
    like_pred_list = np.concatenate(like_pred_list, 1)

    big_result = np.concatenate([uid[:, None], item_id[:, None],
                                 final_pred_list, like_pred_list], 1)
    result = np.concatenate([uid[:, None], item_id[:, None],
                             np.mean(final_pred_list, 1, keepdims=True),
                             np.mean(like_pred_list, 1, keepdims=True)], 1)
    big_result_columns = ["uid", "item_id"]
    big_result_columns.extend(["finish_probability_"+str(1) for i in range(1, 8)])
    big_result_columns.extend(["like_probability_"+str(1) for i in range(1, 8)])
    result_columns = ["uid", "item_id","finish_probability", "like_probability"]
    big_result = pd.DataFrame(big_result, columns=big_result_columns)
    result = pd.DataFrame(result, columns=result_columns)
    big_result.to_csv(Config["normal_config"]["predict_file"]+"_big_result",
                      index=None, float_format="%.6f")
    result.to_csv(Config["normal_config"]["predict_file"] + "_result",
                  index=None, float_format="%.6f")

lgb_predict_media()